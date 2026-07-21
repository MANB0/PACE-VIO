#!/usr/bin/env python3
"""Convert an ORB-SLAM3/EuRoC-style field recording for GeneralStereoIMU.

The source is never modified. Only exact left/right timestamp intersections are
materialized, while unmatched images and calibration provenance are recorded.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import re
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


KALIBR_IMU = {
    "AccelSigma": 1.4010923726394892e-02,
    "AccelSigmaXYZ": [
        1.4442181107898914e-02,
        1.2710374866485316e-02,
        1.4880215204800450e-02,
    ],
    "AngVelSigma": 2.2213273606702497e-03,
    "AngVelSigmaXYZ": [
        3.0610134263713202e-03,
        2.2978040261454912e-03,
        1.3051646294939369e-03,
    ],
    "AccelBiasSigma": 3.8863058420233695e-04,
    "AccelBiasSigmaXYZ": [
        3.5304434274723283e-04,
        3.9471680732362725e-04,
        4.1813060253615087e-04,
    ],
    "AngVelBiasSigma": 5.9156563030152897e-05,
    "AngVelBiasSigmaXYZ": [
        6.4500684655530900e-05,
        6.5409387638584233e-05,
        4.7559616796343543e-05,
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_pngs(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.glob("*.png"):
        try:
            timestamp = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"Non-numeric image filename: {path}") from exc
        if timestamp in result:
            raise ValueError(f"Duplicate image timestamp {timestamp} in {directory}")
        result[timestamp] = path
    if not result:
        raise ValueError(f"No PNG images found in {directory}")
    return result


def read_camera_intrinsics(camera_info: Path) -> tuple[float, float, float, float]:
    text = camera_info.read_text(encoding="utf-8", errors="replace")
    rows: list[list[float]] = []
    for group in re.findall(r"\[([^\]]+)\]", text):
        values = [float(item.strip()) for item in group.split(",")]
        if len(values) == 3:
            rows.append(values)
        if len(rows) == 3:
            break
    if len(rows) != 3 or rows[2] != [0.0, 0.0, 1.0]:
        raise ValueError(f"Cannot parse camera K from {camera_info}")
    return rows[0][0], rows[1][1], rows[0][2], rows[1][2]


def image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image {path}")
    return int(image.shape[1]), int(image.shape[0])


def imu_timestamps(path: Path) -> list[int]:
    timestamps: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != 7:
            raise ValueError(f"Expected seven IMU columns, found {len(header)}: {header}")
        for row_index, row in enumerate(reader, start=2):
            if len(row) != 7:
                raise ValueError(f"Malformed IMU row {row_index}: {row}")
            timestamps.append(int(row[0]))
    if not timestamps or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("IMU timestamps are empty, duplicated, or non-monotonic")
    return timestamps


def median_rate_hz(timestamps: list[int]) -> float:
    intervals_s = [(b - a) * 1e-9 for a, b in zip(timestamps, timestamps[1:])]
    return 1.0 / statistics.median(intervals_s)


def copy_if_needed(source: Path, destination: Path) -> str:
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return "copied"


def copy_stereo(
    timestamps: list[int],
    left: dict[int, Path],
    right: dict[int, Path],
    destination: Path,
    workers: int,
) -> dict[str, int]:
    jobs = []
    for timestamp in timestamps:
        jobs.append((left[timestamp], destination / "left" / f"{timestamp}.png"))
        jobs.append((right[timestamp], destination / "right" / f"{timestamp}.png"))

    copied = 0
    skipped = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, status in enumerate(
            executor.map(lambda pair: copy_if_needed(*pair), jobs), start=1
        ):
            copied += status == "copied"
            skipped += status == "skipped"
            if index % 1000 == 0 or index == len(jobs):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"images {index}/{len(jobs)} "
                    f"({index / len(jobs):.1%}), {index / elapsed:.1f} files/s",
                    flush=True,
                )
    return {"copied": int(copied), "skipped": int(skipped)}


def convert_imu(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    row_count = 0
    with source.open("r", encoding="utf-8", newline="") as input_stream, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as output_stream:
        reader = csv.reader(input_stream)
        writer = csv.writer(output_stream, lineterminator="\n")
        source_header = next(reader)
        if len(source_header) != 7:
            raise ValueError(f"Expected seven IMU columns, found: {source_header}")
        writer.writerow(
            [
                "timestamp",
                "ang_vel_x",
                "ang_vel_y",
                "ang_vel_z",
                "lin_acc_x",
                "lin_acc_y",
                "lin_acc_z",
            ]
        )
        for row_index, row in enumerate(reader, start=2):
            if len(row) != 7:
                raise ValueError(f"Malformed IMU row {row_index}: {row}")
            int(row[0])
            for value in row[1:]:
                float(value)
            writer.writerow(row)
            row_count += 1
    temporary.replace(destination)
    return row_count


def copy_auxiliary(source: Path, destination: Path) -> list[str]:
    relative_files = [
        Path("05111-Tracker6.csv"),
        Path("rostime.txt"),
        Path("mav0/extraction_report.txt"),
        Path("mav0/topic_statistics_report.txt"),
        Path("mav0/cam0/camera_info.txt"),
        Path("mav0/cam0/rostime.txt"),
        Path("mav0/cam0/sectime.txt"),
        Path("mav0/cam0/timestamps.txt"),
        Path("mav0/cam1/rostime.txt"),
        Path("mav0/cam1/sectime.txt"),
        Path("mav0/cam1/timestamps.txt"),
        Path("mav0/imu0/extraction_info.txt"),
        Path("mav0/ping1d/extraction_info.txt"),
        Path("mav0/ping1d/ping1d_data_20260512_053051.csv"),
    ]
    copied: list[str] = []
    for relative in relative_files:
        source_file = source / relative
        if source_file.exists():
            copy_if_needed(source_file, destination / "source_auxiliary" / relative)
            copied.append(relative.as_posix())
    return copied


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_sequence_config(
    path: Path,
    root_wsl: str,
    camera: dict[str, float | int],
    baseline_m: float,
) -> None:
    text = f"""# This recording has no initial static interval.
# Keep odometry imu_static_initialization_enable=false when running it.
type: GeneralStereoIMU
name: field_0512_4
args:
  root: {root_wsl}
  camera:
    fx: {camera['fx']:.12g}
    fy: {camera['fy']:.12g}
    cx: {camera['cx']:.12g}
    cy: {camera['cy']:.12g}
  bl: {baseline_m:.12g}
  format: png
  gravity: 9.81
  pose_output_frame: NED
  imu_window_ns: 100000000
  imu_fallback_max_dt_ns: 50000000
  auto_estimate_time_offset: false
  imu_time_offset_ns: 0
"""
    path.write_text(text, encoding="utf-8")


def write_gap_manifests(
    destination: Path,
    camera_times: list[int],
    imu_times: list[int],
    camera_gap_threshold_s: float = 0.1,
    imu_gap_threshold_s: float = 0.05,
) -> dict[str, object]:
    camera_threshold_ns = int(camera_gap_threshold_s * 1e9)
    imu_threshold_ns = int(imu_gap_threshold_s * 1e9)
    camera_gaps = [
        (index - 1, index, camera_times[index - 1], camera_times[index])
        for index in range(1, len(camera_times))
        if camera_times[index] - camera_times[index - 1] > camera_threshold_ns
    ]
    imu_gaps = [
        (index - 1, index, imu_times[index - 1], imu_times[index])
        for index in range(1, len(imu_times))
        if imu_times[index] - imu_times[index - 1] > imu_threshold_ns
    ]

    with (destination / "sensor_gap_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["stream", "index_before", "index_after", "time_before_ns", "time_after_ns", "gap_s"]
        )
        for stream_name, gaps in (("camera", camera_gaps), ("imu", imu_gaps)):
            for index_before, index_after, time_before, time_after in gaps:
                writer.writerow(
                    [
                        stream_name,
                        index_before,
                        index_after,
                        time_before,
                        time_after,
                        f"{(time_after - time_before) * 1e-9:.9f}",
                    ]
                )

    break_edges = {gap[1] for gap in camera_gaps}
    for _, _, imu_before, imu_after in imu_gaps:
        first_edge = max(1, bisect.bisect_right(camera_times, imu_before))
        last_edge = min(len(camera_times) - 1, bisect.bisect_left(camera_times, imu_after))
        for edge_index in range(first_edge, last_edge + 1):
            if camera_times[edge_index - 1] < imu_after and camera_times[edge_index] > imu_before:
                break_edges.add(edge_index)

    segment_bounds: list[tuple[int, int]] = []
    segment_start = 0
    for edge_index in sorted(break_edges):
        if edge_index > segment_start:
            segment_bounds.append((segment_start, edge_index - 1))
        segment_start = edge_index
    if segment_start < len(camera_times):
        segment_bounds.append((segment_start, len(camera_times) - 1))

    with (destination / "continuous_segments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "segment_id",
                "start_frame_index",
                "end_frame_index",
                "frame_count",
                "start_timestamp_ns",
                "end_timestamp_ns",
                "duration_s",
            ]
        )
        for segment_id, (start, end) in enumerate(segment_bounds):
            writer.writerow(
                [
                    segment_id,
                    start,
                    end,
                    end - start + 1,
                    camera_times[start],
                    camera_times[end],
                    f"{(camera_times[end] - camera_times[start]) * 1e-9:.9f}",
                ]
            )

    return {
        "camera_gap_threshold_s": camera_gap_threshold_s,
        "imu_gap_threshold_s": imu_gap_threshold_s,
        "camera_gap_count": len(camera_gaps),
        "imu_gap_count": len(imu_gaps),
        "continuous_segment_count": len(segment_bounds),
        "continuous_segments": [
            {
                "segment_id": segment_id,
                "start_frame_index": start,
                "end_frame_index": end,
                "frame_count": end - start + 1,
            }
            for segment_id, (start, end) in enumerate(segment_bounds)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--baseline-m", type=float, default=0.12)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == destination or source in destination.parents:
        raise ValueError("Destination must not be the source or a child of the source")

    left = numeric_pngs(source / "mav0/cam0/data")
    right = numeric_pngs(source / "mav0/cam1/data")
    common = sorted(set(left) & set(right))
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    if not common:
        raise ValueError("No exact stereo timestamp intersection")

    camera_info = source / "mav0/cam0/camera_info.txt"
    fx, fy, cx, cy = read_camera_intrinsics(camera_info)
    width, height = image_shape(left[common[0]])
    if image_shape(right[common[0]]) != (width, height):
        raise ValueError("Left/right image dimensions differ")

    imu_source = source / "mav0/imu0/data.csv"
    imu_times = imu_timestamps(imu_source)
    camera_rate = median_rate_hz(common)
    imu_rate = median_rate_hz(imu_times)

    destination.mkdir(parents=True, exist_ok=True)
    print(f"source={source}")
    print(f"destination={destination}")
    print(
        f"stereo common={len(common)}, left_only={len(left_only)}, "
        f"right_only={len(right_only)}"
    )
    copy_stats = copy_stereo(common, left, right, destination, args.workers)
    imu_rows = convert_imu(imu_source, destination / "imu_data.csv")
    auxiliary = copy_auxiliary(source, destination)
    gap_summary = write_gap_manifests(destination, common, imu_times)

    with (destination / "stereo_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["timestamp", "left_source_bytes", "right_source_bytes"])
        for timestamp in common:
            writer.writerow(
                [timestamp, left[timestamp].stat().st_size, right[timestamp].stat().st_size]
            )

    with (destination / "unmatched_stereo_frames.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["side", "timestamp", "source_path"])
        for timestamp in left_only:
            writer.writerow(["left", timestamp, str(left[timestamp])])
        for timestamp in right_only:
            writer.writerow(["right", timestamp, str(right[timestamp])])

    camera = {
        "image_width": width,
        "image_height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline_m": args.baseline_m,
        "camera_model": "pinhole",
        "distortion_model": "none (rectified source images)",
        "camera_rate_hz": camera_rate,
        "left_image_folder": "left",
        "right_image_folder": "right",
        "image_format": "PNG",
    }
    metadata = {
        "dataset": {
            "scene_name": "field_0512_4",
            "source_format": "ORB-SLAM3/EuRoC-style ROS2 extraction",
            "source_path": str(source),
            "num_camera_frames": len(common),
            "num_imu_samples": imu_rows,
            "camera_rate_hz_nominal": 30,
            "camera_rate_hz_median": camera_rate,
            "imu_rate_hz_nominal": 100,
            "imu_rate_hz_median": imu_rate,
            "timestamp_unit": "ns",
            "timestamp_source": "ROS2 message header",
            "has_large_sensor_gaps": True,
        },
        "camera": camera,
        "imu": {
            "rate_hz": 100,
            "frame": "FLU",
            "axis_definition": {"x": "forward", "y": "left", "z": "up"},
            "acc_unit": "m/s^2",
            "gyro_unit": "rad/s",
            "acc_includes_gravity": True,
            "gravity_m_s2": 9.81,
            **KALIBR_IMU,
            "sigma_unit": "continuous noise density",
            "bias_sigma_unit": "continuous random-walk density",
            "bias_random_walk_update_hz": 100,
            "calibration_source": "user-provided Kalibr zed_imu axis calibration",
            "calibration_scope_warning": (
                "The source directory has no sensor serial or calibration file; "
                "verify that this Kalibr profile belongs to the recorded ZED unit."
            ),
            "imu_csv_format": (
                "timestamp,ang_vel_x,ang_vel_y,ang_vel_z,"
                "lin_acc_x,lin_acc_y,lin_acc_z"
            ),
            "imu_data_file": "imu_data.csv",
        },
        "extrinsics": {
            "stereo_baseline_m": args.baseline_m,
            "stereo_baseline_source": (
                "existing MACVO ZED 2i real-data sequence configuration; "
                "not present in F:/0512_4 extraction"
            ),
            "camera_imu_rotation": "FLU to internal NED via Rx(180 deg) in loader",
            "camera_imu_translation_m": [0.0, 0.0, 0.0],
            "camera_imu_translation_status": (
                "UNVERIFIED placeholder because source extraction contains no TF/extrinsic"
            ),
        },
        "coordinate_convention": {
            "export_world_frame": "NED (optimizer internal; no ground truth world frame)",
            "body_frame": "NED internal",
            "imu_measurement_frame": "FLU",
            "camera_frame": "rectified stereo camera / MACVO body reference",
        },
        "time_synchronization": {
            "timestamp_unit": "ns",
            "timestamp_source": "ROS2 message header for camera and IMU",
            "camera_imu_time_offset_ns": 0,
            "offset_status": "shared timestamp domain; no external offset calibration supplied",
        },
        "initialization": {
            "initial_static_segment_available": False,
            "imu_static_initialization_enable": False,
            "note": (
                "Do not estimate startup bias from the first three seconds; the recording "
                "is already moving. Kalibr noise parameters do not provide per-run bias."
            ),
        },
        "ground_truth": {
            "available": False,
            "external_tracker_file": "source_auxiliary/05111-Tracker6.csv",
            "status": (
                "not converted: tracker/world axes, tracked reference point, and exact "
                "timestamp-to-frame mapping are not established"
            ),
        },
        "quality": {
            "left_source_count": len(left),
            "right_source_count": len(right),
            "stereo_intersection_count": len(common),
            "left_only_count": len(left_only),
            "right_only_count": len(right_only),
            "camera_max_gap_s": max(
                (b - a) * 1e-9 for a, b in zip(common, common[1:])
            ),
            "imu_max_gap_s": max(
                (b - a) * 1e-9 for a, b in zip(imu_times, imu_times[1:])
            ),
            "unmatched_manifest": "unmatched_stereo_frames.csv",
            "gap_summary": gap_summary,
            "sensor_gap_manifest": "sensor_gap_manifest.csv",
            "continuous_segments": "continuous_segments.csv",
        },
        "files": {
            "left_images": "left/",
            "right_images": "right/",
            "imu_csv": "imu_data.csv",
            "stereo_manifest": "stereo_manifest.csv",
            "sensor_gap_manifest": "sensor_gap_manifest.csv",
            "continuous_segments": "continuous_segments.csv",
            "source_auxiliary": "source_auxiliary/",
        },
    }
    write_json(destination / "metadata.json", metadata)

    destination_posix = destination.as_posix()
    if destination_posix.startswith("/mnt/"):
        destination_wsl = destination_posix
    elif destination.drive:
        destination_wsl = (
            "/mnt/"
            + destination.drive[0].lower()
            + "/"
            + destination_posix.split(":/", 1)[1]
        )
    else:
        destination_wsl = destination_posix
    write_sequence_config(
        destination / "sequence_config.yaml", destination_wsl, camera, args.baseline_m
    )

    sampled_indices = sorted({0, len(common) // 2, len(common) - 1})
    sampled_hashes = {}
    for index in sampled_indices:
        timestamp = common[index]
        sampled_hashes[f"left/{timestamp}.png"] = sha256(left[timestamp])
        sampled_hashes[f"right/{timestamp}.png"] = sha256(right[timestamp])
    source_hashes = {
        "mav0/imu0/data.csv": sha256(imu_source),
        "mav0/cam0/camera_info.txt": sha256(camera_info),
        "mav0/extraction_report.txt": sha256(source / "mav0/extraction_report.txt"),
        "05111-Tracker6.csv": sha256(source / "05111-Tracker6.csv"),
        **sampled_hashes,
    }
    report = {
        "source": str(source),
        "destination": str(destination),
        "copy_stats": copy_stats,
        "auxiliary_files": auxiliary,
        "source_sha256": source_hashes,
        "output_imu_sha256": sha256(destination / "imu_data.csv"),
        "gap_summary": gap_summary,
        "warnings": [
            "Camera-to-IMU translation is not present in the source and remains unverified.",
            "The external tracker CSV was preserved but not exported as ref_pose.csv.",
            "Large camera and IMU gaps exist; downstream code must not treat the sequence as gap-free.",
        ],
    }
    write_json(destination / "conversion_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export estimate and HoloOcean reference trajectories at the IMU origin."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Utility.PoseFrame import convert_pose_frame, write_timed_se3_csv  # noqa: E402
from Utility.TrajectoryReference import (  # noqa: E402
    compose_camera_to_imu_poses,
    constant_camera_T_imu,
    rebase_pose_positions,
    rigid_body_velocity_at_target,
    translate_pose_reference_point,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_camera_T_imu_internal(metadata: dict) -> np.ndarray:
    extrinsics = metadata.get("extrinsics", {})
    body_imu = extrinsics.get("T_body_imu", {}).get("translation_body_nwu_m")
    body_camera = extrinsics.get("T_body_camera", {}).get("translation_body_nwu_m")
    if not (
        isinstance(body_imu, list)
        and len(body_imu) == 3
        and isinstance(body_camera, list)
        and len(body_camera) == 3
    ):
        raise ValueError("metadata must provide T_body_imu and T_body_camera translations")
    camera_to_imu_nwu = np.asarray(body_imu, dtype=np.float64) - np.asarray(
        body_camera, dtype=np.float64
    )
    camera_to_imu_ned = camera_to_imu_nwu.copy()
    camera_to_imu_ned[1:3] *= -1.0
    return np.concatenate([camera_to_imu_ned, np.asarray([0.0, 0.0, 0.0, 1.0])])


def _metadata_reference_to_imu_nwu(metadata: dict, reference: str) -> np.ndarray:
    extrinsics = metadata.get("extrinsics", {})
    body_imu = np.asarray(
        extrinsics.get("T_body_imu", {}).get("translation_body_nwu_m"),
        dtype=np.float64,
    )
    body_camera = np.asarray(
        extrinsics.get("T_body_camera", {}).get("translation_body_nwu_m"),
        dtype=np.float64,
    )
    if body_imu.shape != (3,) or body_camera.shape != (3,):
        raise ValueError("metadata must provide 3D T_body_imu and T_body_camera")
    if reference == "body":
        return body_imu
    if reference == "camera":
        return body_imu - body_camera
    if reference == "imu":
        return np.zeros(3, dtype=np.float64)
    raise ValueError(f"unsupported reference point: {reference}")


def _load_reference(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if not required.issubset(fields):
        raise ValueError(f"ref_pose.csv is missing fields: {sorted(required - set(fields))}")
    return fields, rows


def _write_imu_reference(
    source: Path,
    destination: Path,
    metadata: dict,
    *,
    position_reference: str,
    velocity_reference: str,
) -> int:
    fields, rows = _load_reference(source)
    camera_poses = np.asarray(
        [
            [
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
                float(row["qw"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    position_lever_nwu = _metadata_reference_to_imu_nwu(
        metadata, position_reference
    )
    imu_poses = translate_pose_reference_point(camera_poses, position_lever_nwu)
    has_velocity = all(field in fields for field in ("vx", "vy", "vz"))
    has_angular_velocity = all(field in fields for field in ("wx", "wy", "wz"))
    imu_velocity = None
    if has_velocity and has_angular_velocity:
        velocity = np.asarray(
            [[float(row["vx"]), float(row["vy"]), float(row["vz"])] for row in rows]
        )
        angular_velocity = np.asarray(
            [[float(row["wx"]), float(row["wy"]), float(row["wz"])] for row in rows]
        )
        velocity_lever_nwu = _metadata_reference_to_imu_nwu(
            metadata, velocity_reference
        )
        imu_velocity = rigid_body_velocity_at_target(
            velocity,
            angular_velocity,
            camera_poses[:, 3:7],
            velocity_lever_nwu,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            converted = dict(row)
            for field, value in zip(
                ("x", "y", "z", "qx", "qy", "qz", "qw"), imu_poses[index]
            ):
                converted[field] = f"{float(value):.17g}"
            if imu_velocity is not None:
                for field, value in zip(("vx", "vy", "vz"), imu_velocity[index]):
                    converted[field] = f"{float(value):.17g}"
            writer.writerow(converted)
    return len(rows)


def export(
    result_dir: Path,
    dataset_dir: Path,
    output_dir: Path | None = None,
    *,
    gt_position_reference: str | None = None,
    gt_velocity_reference: str = "camera",
) -> dict:
    result_dir = result_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = (output_dir or result_dir).expanduser().resolve()
    tensor_path = result_dir / "tensor_map.npz"
    metadata_path = dataset_dir / "metadata.json"
    reference_path = dataset_dir / "ref_pose.csv"
    for path in (tensor_path, metadata_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(tensor_path, allow_pickle=False) as data:
        camera_poses_internal = np.asarray(data["frames//pose"], dtype=np.float64)
        time_ns = np.asarray(data["frames//time_ns"], dtype=np.int64).reshape(-1)
        runtime_extrinsics = np.asarray(
            data["frames//imu_vio_sensor_T_imu"], dtype=np.float64
        )
    runtime_extrinsic = constant_camera_T_imu(runtime_extrinsics)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_extrinsic = _metadata_camera_T_imu_internal(metadata)
    if not np.allclose(runtime_extrinsic, metadata_extrinsic, atol=1e-7, rtol=0.0):
        raise ValueError(
            "runtime T_CI does not match metadata-derived T_CI: "
            f"runtime={runtime_extrinsic.tolist()}, metadata={metadata_extrinsic.tolist()}"
        )

    imu_poses_internal = compose_camera_to_imu_poses(
        camera_poses_internal, runtime_extrinsics
    )
    imu_poses_nwu = convert_pose_frame(imu_poses_internal, "NED", "NWU")
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate_output = output_dir / "poses_imu.csv"
    estimate_rebased_output = output_dir / "poses_imu_rebased.csv"
    write_timed_se3_csv(estimate_output, time_ns, imu_poses_nwu)
    write_timed_se3_csv(
        estimate_rebased_output,
        time_ns,
        rebase_pose_positions(imu_poses_nwu),
    )
    reference_output = None
    reference_rebased_output = None
    reference_count = 0
    if gt_position_reference is not None:
        reference_output = output_dir / "ref_pose_imu.csv"
        reference_count = _write_imu_reference(
            reference_path,
            reference_output,
            metadata,
            position_reference=gt_position_reference,
            velocity_reference=gt_velocity_reference,
        )
        reference_fields, reference_rows = _load_reference(reference_output)
        reference_poses = np.asarray(
            [
                [float(row[field]) for field in ("x", "y", "z", "qx", "qy", "qz", "qw")]
                for row in reference_rows
            ],
            dtype=np.float64,
        )
        reference_rebased_output = output_dir / "ref_pose_imu_rebased.csv"
        rebased_reference = rebase_pose_positions(reference_poses)
        with reference_rebased_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=reference_fields)
            writer.writeheader()
            for index, row in enumerate(reference_rows):
                converted = dict(row)
                for field, value in zip(
                    ("x", "y", "z", "qx", "qy", "qz", "qw"),
                    rebased_reference[index],
                ):
                    converted[field] = f"{float(value):.17g}"
                writer.writerow(converted)

    contract = {
        "schema_version": 2,
        "result_dir": str(result_dir),
        "dataset_dir": str(dataset_dir),
        "metadata_sha256": _sha256(metadata_path),
        "tensor_map_sha256": _sha256(tensor_path),
        "input_estimate_reference": "CameraLeftSocket / T_WC in internal NED",
        "input_truth_position_reference": gt_position_reference,
        "input_truth_velocity_reference": (
            gt_velocity_reference if gt_position_reference is not None else None
        ),
        "truth_reference_warning": (
            "metadata names orientation/velocity sources but does not explicitly name "
            "the ref_pose position sensor; GT conversion is emitted only when the caller "
            "provides --gt-position-reference"
        ),
        "output_reference": "IMU origin",
        "composition": "T_WI = T_WC * T_CI",
        "velocity_composition": "v_WI = v_WC + R_WC * (omega_C x r_CI)",
        "runtime_T_CI_internal_ned_xyzw": runtime_extrinsic.tolist(),
        "metadata_T_CI_internal_ned_xyzw": metadata_extrinsic.tolist(),
        "estimate_rows": int(camera_poses_internal.shape[0]),
        "reference_rows": int(reference_count),
        "estimate_output": str(estimate_output),
        "estimate_rebased_output": str(estimate_rebased_output),
        "reference_output": str(reference_output) if reference_output else None,
        "reference_rebased_output": (
            str(reference_rebased_output) if reference_rebased_output else None
        ),
    }
    contract_path = output_dir / "imu_center_reference_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--gt-position-reference",
        choices=("body", "camera", "imu"),
        default=None,
        help="Physical point represented by ref_pose x/y/z; omitted means do not convert GT.",
    )
    parser.add_argument(
        "--gt-velocity-reference",
        choices=("body", "camera", "imu"),
        default="camera",
        help="Physical point represented by ref_pose velocity fields.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.result_dir,
                args.dataset_dir,
                args.output_dir,
                gt_position_reference=args.gt_position_reference,
                gt_velocity_reference=args.gt_velocity_reference,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

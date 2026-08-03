import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from Utility.OnlinePoseRecorder import OnlinePoseRecorder
from Utility.PaperEvaluation import (
    PoseSeries,
    _segment_ids,
    _trajectory_metrics,
    export_paper_evaluation,
)


def _write_pose_csv(path: Path, rows: list[tuple[int, list[float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"))
        for timestamp, pose in rows:
            writer.writerow((timestamp, *pose))


def _write_gt_csv(path: Path, rows: list[tuple[int, list[float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"))
        for timestamp, pose in rows:
            writer.writerow((timestamp, *pose))


def test_paper_evaluation_uses_complete_active_sequence(tmp_path: Path):
    dataset = tmp_path / "dataset"
    result_root = tmp_path / "result"
    bundle = result_root / "nested_run"
    (dataset / "left").mkdir(parents=True)
    (dataset / "right").mkdir()
    bundle.mkdir(parents=True)
    for folder in (dataset / "left", dataset / "right"):
        for index in range(4):
            (folder / f"{index:06d}.png").write_bytes(f"{folder.name}-{index}".encode())
    (dataset / "metadata.json").write_text(
        json.dumps({"ground_truth": {"reference_point": "IMUSocket", "world_frame": "NWU"}}),
        encoding="utf-8",
    )
    (dataset / "imu_data.csv").write_text("timestamp,ax\n0,0\n", encoding="utf-8")

    timestamps = [3_000_000_000 + index * 100_000_000 for index in range(4)]
    identity = [0.0, 0.0, 0.0, 1.0]
    gt_rows = [
        (timestamp, [float(index), 0.0, 0.0, *identity])
        for index, timestamp in enumerate(timestamps)
    ]
    estimate_rows = [
        (timestamp, [float(index), 0.1 * index, 0.0, *identity])
        for index, timestamp in enumerate(timestamps)
    ]
    _write_gt_csv(dataset / "ref_pose.csv", gt_rows)
    _write_pose_csv(bundle / "poses.csv", estimate_rows)
    _write_pose_csv(bundle / "poses_online.csv", estimate_rows)
    _write_pose_csv(bundle / "macvo_raw_poses_imu.csv", estimate_rows)
    (bundle / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    diagnostic_fields = (
        "frame_i", "frame_j", "timestamp_i", "timestamp_j", "local_ba_graph_build_s",
        "local_ba_lm_s", "local_ba_optimize_total_s", "two_state_solver_converged",
        "two_state_solver_iterations", "two_state_solver_convergence_reason", "vio_backend",
        "isam2_update_ms", "isam2_state_count", "isam2_history_revision",
        "near_zero_velocity_candidate", "near_zero_velocity_active",
    )
    with (bundle / "frame_pair_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic_fields)
        writer.writeheader()
        for index in range(3):
            writer.writerow({
                "frame_i": 90 + index,
                "frame_j": 91 + index,
                "timestamp_i": timestamps[index],
                "timestamp_j": timestamps[index + 1],
                "local_ba_graph_build_s": 0.001,
                "local_ba_lm_s": 0.002,
                "local_ba_optimize_total_s": 0.004,
                "two_state_solver_converged": True,
                "two_state_solver_iterations": 2,
                "two_state_solver_convergence_reason": "ok",
                "vio_backend": "isam2",
                "isam2_update_ms": 2.5,
                "isam2_state_count": index + 2,
                "isam2_history_revision": True,
                "near_zero_velocity_candidate": False,
                "near_zero_velocity_active": False,
            })
    pipeline_fields = (
        "frame_i", "frame_j", "timestamp_i_ns", "timestamp_j_ns", "frontend_ms",
        "backend_solver_ms", "backend_wait_ms", "commit_ms", "backend_submitted",
        "static_initialization_active",
    )
    with (result_root / "pipeline_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=pipeline_fields)
        writer.writeheader()
        for index in range(3):
            writer.writerow({
                "frame_i": 90 + index,
                "frame_j": 91 + index,
                "timestamp_i_ns": timestamps[index],
                "timestamp_j_ns": timestamps[index + 1],
                "frontend_ms": 10.0,
                "backend_solver_ms": 4.0,
                "backend_wait_ms": 12.0,
                "commit_ms": 0.5,
                "backend_submitted": 1,
                "static_initialization_active": 0,
            })
    (bundle / "elapsed_time.json").write_text(
        json.dumps({
            "CPU_ElapsedTime": {"Odom_Runtime": [float(index) for index in range(100)]},
            "GPU_ElapsedTime": {},
        }),
        encoding="utf-8",
    )

    output = export_paper_evaluation(
        project_root=tmp_path,
        dataset_root=dataset,
        result_root=result_root,
    )
    summary = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    evaluation_contract = json.loads((output / "evaluation_alignment.json").read_text(encoding="utf-8"))
    final = summary["trajectories"]["final"]

    assert final["state_count"] == 4
    assert final["edge_count"] == 3
    assert final["ape"]["translation_m"]["rmse"] > 0.0
    assert final["ape"]["rotation_deg"]["rmse"] == 0.0
    assert final["rpe"]["translation_m"]["rmse"] > 0.0
    assert summary["timing"]["edge_count"] == 3
    assert summary["timing"]["factor_build_ms"]["median"] == 1.0
    assert summary["timing"]["backend_update_ms"]["median"] == 2.5
    assert summary["timing"]["end_to_end_frame_ms"]["median"] == 92.0
    assert summary["motion_detection"]["available"] is False
    assert len(list(csv.DictReader((output / "metrics_per_edge.csv").open()))) == 3
    assert dataset_manifest["left_image_count"] == 4
    assert dataset_manifest["right_image_count"] == 4
    assert dataset_manifest["stereo_pair_count"] == 4
    assert dataset_manifest["imu_sample_count"] == 1
    assert dataset_manifest["camera_median_frequency_hz"] == 10.0
    assert evaluation_contract["active_range"]["evaluated_state_count"] == 4
    assert evaluation_contract["active_range"]["evaluated_edge_count"] == 3
    run_summary = next(csv.DictReader((output / "run_summary.csv").open()))
    assert run_summary["backend"] == "isam2"
    assert float(run_summary["factor_build_median_ms"]) == 1.0
    assert float(run_summary["backend_update_median_ms"]) == 2.5
    assert float(run_summary["end_to_end_frame_median_ms"]) == 92.0
    assert float(run_summary["convergence_rate"]) == 1.0
    evaluated = list(csv.DictReader((output / "trajectory_evaluated.csv").open()))
    evaluated_gt = list(csv.DictReader((output / "ground_truth_evaluated.csv").open()))
    assert float(evaluated[0]["tx"]) == 0.0
    assert float(evaluated_gt[0]["tx"]) == 0.0


def test_fixed_or_none_alignment_is_not_overwritten_by_first_pose_anchoring(tmp_path: Path):
    timestamps = np.asarray([0, 1], dtype=np.int64)
    frames = np.asarray([0, 1], dtype=np.int64)
    identity = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    estimate = PoseSeries(
        timestamps,
        np.asarray([[5.0, 0.0, 0.0, *identity], [6.0, 0.0, 0.0, *identity]]),
        frames,
    )
    ground_truth = PoseSeries(
        timestamps,
        np.asarray([[2.0, 0.0, 0.0, *identity], [3.0, 0.0, 0.0, *identity]]),
        frames,
    )

    fixed = _trajectory_metrics(
        estimate,
        ground_truth,
        np.zeros(2, dtype=np.int64),
        output_dir=tmp_path,
        prefix="fixed",
        anchor_first=False,
    )
    anchored = _trajectory_metrics(
        estimate,
        ground_truth,
        np.zeros(2, dtype=np.int64),
        output_dir=tmp_path,
        prefix="anchored",
        anchor_first=True,
    )

    assert fixed["ape"]["translation_m"]["rmse"] == 3.0
    assert anchored["ape"]["translation_m"]["rmse"] == 0.0


def test_online_pose_recorder_exports_imu_center_in_output_frame(tmp_path: Path):
    camera = pp.identity_SE3(2, dtype=torch.float64).tensor()
    camera[1, 0] = 2.0
    extrinsic = pp.identity_SE3(2, dtype=torch.float64).tensor()
    extrinsic[:, 1] = 0.25

    class Value:
        def __init__(self, tensor: torch.Tensor):
            self.tensor = tensor

    class Frames:
        def __init__(self):
            self.data = {
                "pose": Value(camera),
                "time_ns": Value(torch.tensor([100, 200], dtype=torch.long)),
                "imu_vio_sensor_T_imu": Value(extrinsic),
            }

        def __len__(self):
            return 2

    system = SimpleNamespace(
        graph=SimpleNamespace(frames=Frames()),
        Optimizer=SimpleNamespace(last_pair_diagnostics={
            "frame_idx": 1,
            "vio_backend": "isam2",
            "isam2_history_revision": True,
            "isam2_state_count": 2,
        }),
    )
    path = tmp_path / "poses_online.csv"
    recorder = OnlinePoseRecorder(path, output_world_frame="NWU")
    recorder(system)
    recorder(system)
    recorder.close()
    rows = list(csv.DictReader(path.open()))

    assert len(rows) == 2
    assert [int(row["frame_idx"]) for row in rows] == [0, 1]
    assert float(rows[0]["ty"]) == -0.25
    assert float(rows[1]["tx"]) == 2.0
    assert all(row["backend"] == "isam2" for row in rows)


def test_ground_truth_gaps_start_new_rpe_segments():
    timestamps = np.asarray([0, 10, 20, 1000, 1010], dtype=np.int64)
    assert _segment_ids(timestamps).tolist() == [0, 0, 0, 1, 1]

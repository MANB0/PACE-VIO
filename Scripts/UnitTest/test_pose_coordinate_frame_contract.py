import csv
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch

from Scripts.eval_qa_vif import evaluate_trajectory_direct, load_macvo_timed_poses
from Utility.Sandbox import Sandbox
from Utility.PoseFrame import convert_pose_frame, write_timed_se3_csv


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
        ],
        dtype=np.float64,
    )


def _write_ref_pose_csv(path: Path, time_ns: np.ndarray, poses_nwu: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for timestamp, pose in zip(time_ns, poses_nwu):
            writer.writerow([str(int(timestamp)), *[f"{float(value):.17g}" for value in pose]])


class _TensorStore:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class _MinimalMap:
    def __init__(self, poses_ned: np.ndarray, time_ns: np.ndarray) -> None:
        n = int(poses_ned.shape[0])
        self.frames = SimpleNamespace(data={
            "pose": _TensorStore(torch.tensor(poses_ned, dtype=torch.float32)),
            "T_BS": _TensorStore(pp.identity_SE3(n).tensor().float()),
            "time_ns": _TensorStore(torch.tensor(time_ns, dtype=torch.long)),
        })

    def serialize(self) -> dict[str, np.ndarray]:
        return {
            "frames/pose": self.frames.data["pose"].tensor.numpy(),
            "frames/T_BS": self.frames.data["T_BS"].tensor.numpy(),
            "frames/time_ns": self.frames.data["time_ns"].tensor.numpy(),
        }


def _load_iodometry_class():
    module_stub = types.ModuleType("Module")
    map_stub = types.ModuleType("Module.Map")
    map_stub.VisualMap = object
    old_module = sys.modules.get("Module")
    old_map = sys.modules.get("Module.Map")
    sys.modules["Module"] = module_stub
    sys.modules["Module.Map"] = map_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "odometry_interface_under_test",
            Path("Odometry/Interface.py"),
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module.IOdometry
    finally:
        if old_module is None:
            sys.modules.pop("Module", None)
        else:
            sys.modules["Module"] = old_module
        if old_map is None:
            sys.modules.pop("Module.Map", None)
        else:
            sys.modules["Module.Map"] = old_map


class _StaticMapOdometry(_load_iodometry_class()):
    def __init__(self, graph: _MinimalMap) -> None:
        super().__init__()
        self._graph = graph

    def run(self, frame) -> None:
        return None

    def get_map(self) -> _MinimalMap:
        return self._graph


class _TinySequence:
    pose_output_frame = "NWU"

    def __init__(self, time_ns: np.ndarray) -> None:
        self._frames = [SimpleNamespace(gt_pose=None, time_ns=[int(t)]) for t in time_ns]

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)


def test_convert_pose_frame_changes_orientation_by_ned_nwu_basis_transform():
    pose_ned = np.array(
        [[1.0, 2.0, -3.0, 0.10, -0.20, 0.30, 0.9273618495495703]],
        dtype=np.float64,
    )

    pose_nwu = convert_pose_frame(pose_ned, "NED", "NWU")

    basis = np.diag([1.0, -1.0, -1.0])
    expected_translation = basis @ pose_ned[0, :3]
    expected_rotation = basis @ _quat_xyzw_to_matrix(pose_ned[0, 3:7]) @ basis

    assert np.allclose(pose_nwu[0, :3], expected_translation)
    assert np.allclose(_quat_xyzw_to_matrix(pose_nwu[0, 3:7]), expected_rotation)


def test_convert_pose_frame_rejects_non_se3_pose_shape():
    with pytest.raises(ValueError, match="Nx7"):
        convert_pose_frame(np.zeros((2, 6), dtype=np.float64), "NED", "NWU")


def test_eval_reads_ned_pose_csv_sidecar_in_holocean_nwu_frame(tmp_path: Path):
    time_ns = np.array([100, 200, 300], dtype=np.int64)
    poses_ned = np.array(
        [
            [1.0, 2.0, -3.0, 0.10, 0.20, -0.30, 0.9273618495495703],
            [1.5, 2.5, -3.5, 0.15, -0.25, 0.05, 0.95524865872714],
            [2.0, 3.0, -4.0, -0.05, 0.10, 0.20, 0.9733961166965892],
        ],
        dtype=np.float64,
    )
    poses_nwu = convert_pose_frame(poses_ned, "NED", "NWU")

    result_dir = tmp_path / "run"
    result_dir.mkdir()
    write_timed_se3_csv(result_dir / "poses.csv", time_ns, poses_ned)
    (result_dir / "pose_coordinate_frame.txt").write_text("NED\n", encoding="utf-8")

    ref_pose = tmp_path / "ref_pose.csv"
    _write_ref_pose_csv(ref_pose, time_ns, poses_nwu)

    loaded_time, loaded_poses = load_macvo_timed_poses(result_dir / "poses.csv", target_frame="NWU")
    assert np.array_equal(loaded_time, time_ns)
    assert np.allclose(loaded_poses, poses_nwu)

    result = evaluate_trajectory_direct(result_dir / "poses.csv", ref_pose)
    assert result["vo_source_frame"] == "NED"
    assert result["coordinate_frame"] == "NWU"
    assert result["matching"]["mode"] == "exact_timestamp"
    assert result["ate"]["ate_rmse"] == pytest.approx(0.0)


def test_odometry_export_converts_internal_ned_map_to_declared_nwu_without_mutating_map(tmp_path: Path):
    time_ns = np.array([100, 200], dtype=np.int64)
    poses_ned = np.array(
        [
            [1.0, 2.0, -3.0, 0.10, -0.20, 0.30, 0.9273618495495703],
            [2.0, -4.0, 5.0, -0.05, 0.10, 0.20, 0.9733961166965892],
        ],
        dtype=np.float64,
    )
    graph = _MinimalMap(poses_ned, time_ns)
    internal_before = graph.frames.data["pose"].tensor.clone()

    _StaticMapOdometry(graph).receive_frames(_TinySequence(time_ns), Sandbox(tmp_path / "export"))

    exported_time, exported_poses = load_macvo_timed_poses(tmp_path / "export" / "poses.csv", target_frame="NWU")

    assert np.array_equal(exported_time, time_ns)
    assert np.allclose(exported_poses, convert_pose_frame(poses_ned, "NED", "NWU"))
    assert (tmp_path / "export" / "pose_coordinate_frame.txt").read_text(encoding="utf-8").strip() == "NWU"
    assert torch.allclose(graph.frames.data["pose"].tensor, internal_before)

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch
import yaml

import Module
from DataLoader import IMUData, StereoData, StereoInertialFrame
from Module.Frontend.Frontend import FrontendReplayViolation, IFrontend, ReplayFrontend
from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Module.KeyframeSelector import AllKeyframe
from Module.MotionModel import StaticMotionModel
from Module.Optimization.TwoFramePGO.Graphs import LocalWindowGraphInput
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Odometry.Interface import IOdometry
from Odometry.MACVO import MACVO
from Scripts.export_visual_factor_cache import export_result_to_visual_cache
from Utility.FramePairDiagnostics import FramePairDiagnosticsWriter
from Utility.PrettyPrint import Logger
from Utility.Sandbox import Sandbox
from Utility.VisualFactorCache import (
    VisualFactorCacheError,
    VisualFactorCacheReader,
    VisualFactorPacket,
    write_visual_factor_cache,
)
from Utility.VisualInputFingerprint import visual_input_sha256


SCENE = "cache-replay-scene"
TIMESTAMPS_NS = (1_000_000_000, 1_100_000_000, 1_200_000_000)
K = torch.tensor(
    [[100.0, 0.0, 8.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
    dtype=torch.float64,
)
BASELINE_M = 0.125
POINT_ROWS = 12


def _se3(translation: tuple[float, float, float]) -> pp.LieTensor:
    return pp.SE3(torch.tensor([*translation, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32))


def _match_fields(offset: float) -> dict[str, torch.Tensor]:
    rows = torch.arange(POINT_ROWS, dtype=torch.float32)
    return {
        "pixel1_uv": torch.stack((rows + offset, rows + 1.0), dim=-1),
        "pixel2_uv": torch.stack((rows + offset + 0.5, rows + 1.5), dim=-1),
        "pixel1_d": (rows + 2.0).unsqueeze(-1),
        "pixel2_d": (rows + 2.5).unsqueeze(-1),
        "pixel1_disp": (rows + 3.0).unsqueeze(-1),
        "pixel2_disp": (rows + 3.5).unsqueeze(-1),
        "pixel1_disp_cov": torch.full((POINT_ROWS, 1), 0.1),
        "pixel2_disp_cov": torch.full((POINT_ROWS, 1), 0.2),
        "pixel1_d_cov": torch.full((POINT_ROWS, 1), 0.3),
        "pixel2_d_cov": torch.full((POINT_ROWS, 1), 0.4),
        "pixel1_uv_cov": torch.tensor([0.1, 0.2, 0.0]).repeat(POINT_ROWS, 1),
        "pixel2_uv_cov": torch.tensor([0.2, 0.3, 0.0]).repeat(POINT_ROWS, 1),
        "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(POINT_ROWS, 1, 1),
        "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(POINT_ROWS, 1, 1) * 2.0,
    }


def _packet(frame_i: int) -> VisualFactorPacket:
    fields = _match_fields(float(frame_i) * 100.0)
    points_local = torch.stack(
        (
            torch.arange(POINT_ROWS, dtype=torch.float64) + 1.0,
            torch.full((POINT_ROWS,), 2.0, dtype=torch.float64),
            torch.full((POINT_ROWS,), 3.0, dtype=torch.float64),
        ),
        dim=-1,
    )
    return VisualFactorPacket(
        frame_i=frame_i,
        frame_j=frame_i + 1,
        timestamp_i_ns=TIMESTAMPS_NS[frame_i],
        timestamp_j_ns=TIMESTAMPS_NS[frame_i + 1],
        K=K,
        baseline_m=BASELINE_M,
        relative_pose_init=torch.eye(4, dtype=torch.float64),
        points_local=points_local,
        points_cov_local=torch.diag(
            torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        ).repeat(POINT_ROWS, 1, 1),
        point_colors=torch.arange(POINT_ROWS * 3, dtype=torch.uint8).reshape(POINT_ROWS, 3),
        match_fields=fields,
        covariance_diagnostics={
            "median_flow_u_cov": 10.0 + frame_i,
            "num_selected_keypoints": POINT_ROWS + 5 + frame_i,
        },
        visual_sha256=visual_input_sha256(fields),
    )


def _write_cache(tmp_path: Path) -> tuple[Path, tuple[VisualFactorPacket, VisualFactorPacket]]:
    packets = (
        _packet(0),
        _packet(1),
    )
    cache_path = tmp_path / "cache"
    write_visual_factor_cache(cache_path, SCENE, packets, source={"frame_count": 3})
    return cache_path, packets


def _write_lost_track_cache(tmp_path: Path) -> tuple[Path, VisualFactorPacket]:
    packet = _packet(0)
    rows = 3
    match_fields = {name: value[:rows] for name, value in packet.match_fields.items()}
    packet = replace(
        packet,
        points_local=packet.points_local[:rows],
        points_cov_local=packet.points_cov_local[:rows],
        point_colors=packet.point_colors[:rows],
        match_fields=match_fields,
        covariance_diagnostics={
            **packet.covariance_diagnostics,
            "num_selected_keypoints": 17,
        },
        visual_sha256=visual_input_sha256(match_fields),
    )
    cache_path = tmp_path / "lost-track-cache"
    write_visual_factor_cache(cache_path, SCENE, [packet], source={"frame_count": 2})
    return cache_path, packet


def _frame(index: int, K_value: torch.Tensor = K, timestamp_ns: int | None = None) -> StereoInertialFrame:
    timestamp = TIMESTAMPS_NS[index] if timestamp_ns is None else timestamp_ns
    previous_timestamp = timestamp - 100_000_000
    return StereoInertialFrame(
        idx=[index],
        time_ns=[timestamp],
        stereo=StereoData(
            T_BS=pp.identity_SE3(1, dtype=torch.float32),
            K=K_value.float().unsqueeze(0),
            baseline=torch.tensor([BASELINE_M], dtype=torch.float32),
            time_ns=[timestamp],
            height=16,
            width=16,
            imageL=torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            imageR=torch.zeros((1, 3, 16, 16), dtype=torch.float32),
        ),
        imu=IMUData(
            T_BS=pp.identity_SE3(1, dtype=torch.float32),
            time_ns=torch.tensor([[[previous_timestamp], [timestamp]]], dtype=torch.long),
            gravity=[9.81],
            acc=torch.tensor([[[1.0, 0.0, 9.81], [1.0, 0.0, 9.81]]]),
            gyro=torch.zeros((1, 2, 3), dtype=torch.float32),
        ),
    )


class _CountingReplayFrontend(ReplayFrontend):
    def __init__(self) -> None:
        super().__init__(SimpleNamespace())
        self.inference_call_count = 0

    def estimate_depth(self, frame):
        self.inference_call_count += 1
        return super().estimate_depth(frame)

    def estimate_pair(self, frame_t1, frame_t2):
        self.inference_call_count += 1
        return super().estimate_pair(frame_t1, frame_t2)


class _TrackingStaticMotionModel(StaticMotionModel):
    def __init__(self) -> None:
        super().__init__(SimpleNamespace())
        self.predict_inputs: list[tuple[object, object]] = []

    def predict(self, frame, flow, depth):
        self.predict_inputs.append((flow, depth))
        return super().predict(frame, flow, depth)


class _ForbiddenReplayModule:
    def __getattr__(self, name):
        raise AssertionError(f"replay called forbidden module method {name}")


class _HistoryOptimizer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            imu_factor_mode="preintegrated_vio",
            imu_trans_prior_scale=1.0,
            post_imu_fusion_enable=False,
        )
        self.write_calls = 0
        self.graph_history: list[tuple[int, int, int, int]] = []
        self.started: list[int] = []
        self.terminate_calls = 0

    def write_map(self, graph) -> None:
        self.write_calls += 1
        if self.write_calls == 2 and len(graph.frames) > 1:
            graph.frames.data["pose"][1] = torch.tensor(
                [5.0, -1.0, 0.5, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5],
                dtype=torch.float32,
            )

    def get_graph_data(self, graph, frame_idx):
        current_index = int(frame_idx.item())
        self.graph_history.append(
            (current_index, len(graph.frames), len(graph.match), len(graph.points))
        )
        return current_index

    def start_optimize(self, graph_data) -> None:
        self.started.append(int(graph_data))

    def terminate(self) -> None:
        self.terminate_calls += 1


class _LocalBAHistoryOptimizer(_HistoryOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            imu_factor_mode="local_inertial_ba",
            graph_type="disp",
            imu_rot_prior=True,
            imu_rot_prior_scale=1.0,
            imu_trans_prior_scale=1.0,
            imu_vio_cov_scale=1.0,
            local_ba_window_size=3,
            local_ba_fix_first_frame=True,
            local_ba_writeback="current",
            device="cpu",
            post_imu_fusion_enable=False,
        )
        self.local_graph_history: list[LocalWindowGraphInput] = []
        self._extractor = object.__new__(TwoFrame_PGO)
        self._extractor.config = self.config

    def get_graph_data(self, graph, frame_idx):
        graph_data = TwoFrame_PGO.get_graph_data(self._extractor, graph, frame_idx)
        assert isinstance(graph_data, LocalWindowGraphInput)
        self.local_graph_history.append(graph_data)
        return graph_data

    def start_optimize(self, graph_data) -> None:
        self.started.append(graph_data)


class _LiveDiagnosticsFrontend(IFrontend):
    def __init__(self) -> None:
        super().__init__(SimpleNamespace())
        grid = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
        self._depth0 = IStereoDepth.Output(
            depth=torch.full((1, 1, 16, 16), 2.0, dtype=torch.float32),
            disparity=torch.full((1, 1, 16, 16), 6.0, dtype=torch.float32),
            cov=grid / 10.0 + 0.25,
            disparity_uncertainty=torch.full((1, 1, 16, 16), 0.1, dtype=torch.float32),
        )
        self._depth1 = IStereoDepth.Output(
            depth=torch.full((1, 1, 16, 16), 2.5, dtype=torch.float32),
            disparity=torch.full((1, 1, 16, 16), 5.0, dtype=torch.float32),
            cov=grid / 5.0 + 0.5,
            disparity_uncertainty=torch.full((1, 1, 16, 16), 0.2, dtype=torch.float32),
        )
        self._match = IMatcher.Output(
            flow=torch.zeros((1, 2, 16, 16), dtype=torch.float32),
            cov=torch.cat((grid + 1.0, grid * 2.0 + 3.0, torch.zeros_like(grid)), dim=1),
        )

    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return True, True

    def estimate_depth(self, _frame):
        return self._depth0

    def estimate_pair(self, _frame_t1, _frame_t2):
        return self._depth1, self._match


class _FixedCandidateSelector:
    def __init__(self) -> None:
        self.points = torch.tensor(
            [[2 + column, 2 + row] for row in range(3) for column in range(4)],
            dtype=torch.long,
        )

    def select_point(self, *_args):
        return self.points.clone()


class _KeepFirstThreeFilter:
    def set_meta(self, _frame) -> None:
        return None

    def verify_shape(self, _match_obs) -> bool:
        return True

    def filter(self, match_obs, _device):
        mask = torch.zeros(len(match_obs), dtype=torch.bool)
        mask[:3] = True
        return mask


class _FixedObservationCovariance:
    def estimate(self, _frame, keypoints, *_args):
        return torch.diag(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)).repeat(
            len(keypoints), 1, 1
        )


class _CountingMapRefiner:
    def __init__(self) -> None:
        self.calls = 0

    def elaborate_map(self, _frames) -> None:
        self.calls += 1


class _CountingDiagnosticsWriter:
    def __init__(self, path: Path) -> None:
        self._filepath = path
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _DeclaredSequence:
    def __init__(self, frames, declared_length: int) -> None:
        self._frames = list(frames)
        self.indices = torch.arange(declared_length, dtype=torch.long)
        self.pose_output_frame = "NED"

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __iter__(self):
        return iter(self._frames)


def _build_system(
    cache_path: Path,
    *,
    mapping: bool = False,
    frontend: ReplayFrontend | None = None,
    motion_model: StaticMotionModel | None = None,
    keyframe_selector: AllKeyframe | None = None,
    optimizer: _HistoryOptimizer | None = None,
    visual_cache_mode: str = "replay",
) -> tuple[MACVO, _HistoryOptimizer]:
    optimizer = optimizer or _HistoryOptimizer()
    system = MACVO(
        device="cpu",
        num_point=POINT_ROWS,
        edgewidth=1,
        match_cov_default=1.0,
        profile=False,
        mapping=mapping,
        frontend=frontend or _CountingReplayFrontend(),
        motion_model=motion_model or _TrackingStaticMotionModel(),
        kp_selector=_ForbiddenReplayModule(),
        map_selector=_ForbiddenReplayModule(),
        obs_filter=_ForbiddenReplayModule(),
        obs_covmodel=_ForbiddenReplayModule(),
        post_process=_ForbiddenReplayModule(),
        kf_selector=keyframe_selector or AllKeyframe(SimpleNamespace()),
        optimizer=optimizer,
        visual_cache_mode=visual_cache_mode,
        visual_cache_path=str(cache_path) if visual_cache_mode == "replay" else None,
    )
    system.set_diagnostics_writer(None, scene=SCENE)
    return system, optimizer


def _build_live_diagnostics_source(result_dir: Path) -> tuple[MACVO, _HistoryOptimizer, _CountingMapRefiner]:
    optimizer = _HistoryOptimizer()
    map_refiner = _CountingMapRefiner()
    system = MACVO(
        device="cpu",
        num_point=POINT_ROWS,
        edgewidth=1,
        match_cov_default=1.0,
        profile=False,
        mapping=False,
        frontend=_LiveDiagnosticsFrontend(),
        motion_model=_TrackingStaticMotionModel(),
        kp_selector=_FixedCandidateSelector(),
        map_selector=_ForbiddenReplayModule(),
        obs_filter=_KeepFirstThreeFilter(),
        obs_covmodel=_FixedObservationCovariance(),
        post_process=map_refiner,
        kf_selector=AllKeyframe(SimpleNamespace()),
        optimizer=optimizer,
        visual_cache_mode="off",
    )
    writer = FramePairDiagnosticsWriter(
        result_dir / "frame_pair_diagnostics.csv",
        scene=SCENE,
        method="live-source",
    )
    system.set_diagnostics_writer(writer, scene=SCENE, method="live-source")
    return system, optimizer, map_refiner


def _graph_match_fields(system: MACVO) -> dict[str, torch.Tensor]:
    return {name: values.tensor for name, values in system.graph.match.data.items()}


def _disable_termination_side_effects(system: MACVO) -> None:
    system.Optimizer.terminate = lambda: None
    system.MapRefiner = SimpleNamespace(elaborate_map=lambda _frames: None)


def test_live_source_visual_diagnostics_records_complete_factor_hash(tmp_path: Path):
    result_dir = tmp_path / "source"
    result_dir.mkdir()
    system, _, _ = _build_live_diagnostics_source(result_dir)
    frame0, frame1 = _frame(0), _frame(1)

    system.initialize(frame0)
    system.run_pair(frame0, frame1)

    with (result_dir / "visual_factor_diagnostics.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["visual_input_sha256"] == visual_input_sha256(_graph_match_fields(system))


def test_live_source_stores_exact_filtered_local_tracking_points(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "source"
    result_dir.mkdir()
    system, _, _ = _build_live_diagnostics_source(result_dir)
    selected_rows = torch.tensor([0, 4, 11], dtype=torch.long)

    class _KeepNonContiguousFilter(_KeepFirstThreeFilter):
        def filter(self, match_obs, _device):
            mask = torch.zeros(len(match_obs), dtype=torch.bool)
            mask[selected_rows] = True
            return mask

    sentinel = torch.arange(POINT_ROWS * 3, dtype=torch.float32).reshape(POINT_ROWS, 3)
    call_count = 0

    def exact_runtime_projection(*_args):
        nonlocal call_count
        call_count += 1
        return sentinel.clone() if call_count == 1 else sentinel.clone() + 10_000.0

    monkeypatch.setattr("Odometry.MACVO.pixel2point_NED", exact_runtime_projection)
    system.OutlierFilter = _KeepNonContiguousFilter()
    frame0, frame1 = _frame(0), _frame(1)

    system.initialize(frame0)
    system.run_pair(frame0, frame1)

    assert call_count == 1
    assert "pos_Tc" in system.graph.points.data
    assert torch.equal(system.graph.points.data["pos_Tc"].tensor, sentinel[selected_rows])


def test_tracking_and_mapping_point_stores_share_point_feature_contract(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)

    assert set(system.graph.points.data) == set(system.graph.map_points.data)
    assert "pos_Tc" in system.graph.map_points.data


def test_replay_rematerializes_cached_local_points_from_current_backend_pose(tmp_path: Path):
    cache_path, packets = _write_cache(tmp_path)
    system, optimizer = _build_system(cache_path)
    frame0, frame1, frame2 = (_frame(index) for index in range(3))

    system.initialize(frame0)
    system.run_pair(frame0, frame1)

    assert system.Frontend.inference_call_count == 0
    assert system.MotionEstimator.predict_inputs == [(None, None)]
    assert visual_input_sha256(_graph_match_fields(system)) == packets[0].visual_sha256
    assert system._pending_cov_diag == packets[0].covariance_diagnostics

    system.run_pair(frame1, frame2)

    current_previous_pose = pp.SE3(system.graph.frames.data["pose"][1].double())
    expected_world_points = current_previous_pose.Act(packets[1].points_local)
    source_style_rotation = pp.SE3(system.graph.frames.data["pose"][1]).rotation().matrix().double()
    expected_world_covariances = (
        source_style_rotation
        @ packets[1].points_cov_local
        @ source_style_rotation.transpose(-1, -2)
    )
    source_world_points = pp.SE3(_se3((0.25, 0.0, 0.0)).tensor().double()).Act(packets[1].points_local)
    second_pair = slice(POINT_ROWS, POINT_ROWS * 2)
    expected_written_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    expected_rotated_covariance = torch.diag(torch.tensor([4.0, 1.0, 9.0], dtype=torch.float64))

    assert torch.allclose(
        current_previous_pose.rotation().matrix().double(),
        expected_written_rotation,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(system.graph.points.data["pos_Tw"][second_pair].double(), expected_world_points)
    assert not torch.allclose(system.graph.points.data["pos_Tw"][second_pair].double(), source_world_points)
    assert torch.allclose(system.graph.points.data["cov_Tw"][second_pair], expected_world_covariances)
    assert torch.allclose(
        system.graph.points.data["cov_Tw"][POINT_ROWS].double(),
        expected_rotated_covariance,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.equal(system.graph.points.data["color"][second_pair], packets[1].point_colors)
    assert torch.allclose(
        pp.SE3(system.graph.frames.data["pose"][2]).tensor().double(),
        current_previous_pose.tensor(),
    )
    assert system._pending_cov_diag == packets[1].covariance_diagnostics
    assert system.graph.frames.data["imu_vio_delta_p"][2].abs().sum() > 0
    assert optimizer.graph_history == [(1, 2, POINT_ROWS, POINT_ROWS), (2, 3, POINT_ROWS * 2, POINT_ROWS * 2)]
    assert optimizer.started == [1, 2]


def test_live_lost_track_visual_diagnostics_survive_export_and_replay(tmp_path: Path):
    result_dir = tmp_path / "source-result"
    dataset_root = tmp_path / "dataset"
    result_dir.mkdir()
    dataset_root.mkdir()
    source, _, _ = _build_live_diagnostics_source(result_dir)
    frame0, frame1 = _frame(0), _frame(1)

    source.initialize(frame0)
    source.run_pair(frame0, frame1)
    source_covariance = dict(source._pending_cov_diag or {})
    source_candidate_count = source._pending_tracking_state["num_keypoints_candidate"]

    assert source_candidate_count == POINT_ROWS
    assert source_covariance["num_selected_keypoints"] == POINT_ROWS
    assert len(source.graph.match) == 3

    source.terminate()
    (result_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "Odometry": {
                    "args": {"mapping": False},
                    "motion": {"type": "StaticMotionModel"},
                    "keyframe": {"type": "AllKeyframe"},
                }
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(result_dir / "tensor_map.npz", **source.graph.serialize())

    cache_dir = export_result_to_visual_cache(
        result_dir,
        tmp_path / "cache",
        SCENE,
        dataset_root,
    )
    packet = VisualFactorCacheReader(cache_dir).load_pair(
        0,
        1,
        TIMESTAMPS_NS[0],
        TIMESTAMPS_NS[1],
    )

    assert len(packet.match_fields["pixel1_uv"]) == 3
    assert packet.covariance_diagnostics == source_covariance

    replay, _ = _build_system(cache_dir)
    replay.initialize(frame0)
    replay.run_pair(frame0, frame1)

    assert replay._pending_cov_diag == source_covariance
    assert replay._pending_tracking_state["num_keypoints_candidate"] == source_candidate_count


def test_replay_runs_adaptive_post_writeback_lifecycle_before_assigning_current_packet(tmp_path: Path):
    cache_path, packets = _write_cache(tmp_path)
    system, optimizer = _build_system(cache_path)
    frame0, frame1, frame2 = (_frame(index) for index in range(3))
    events: list[tuple[str, float | None]] = []

    class _Gate:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, signals, *, pair_id):
            self.calls += 1
            events.append(("gate", signals.get("median_flow_cov")))
            triggered = self.calls == 2
            return SimpleNamespace(
                mode="full_imu" if triggered else "rotation_only",
                use_imu_rotation=True,
                use_imu_translation=True,
                velocity_reset_triggered=triggered,
                velocity_reset_strategy="zero",
                d2_rerun_triggered=triggered,
                d2_rerun_pair_id=pair_id,
            )

    system.enable_adaptive(_Gate(), version="v3b")
    system._write_adaptive_decision_csv = lambda: events.append(
        ("adaptive_csv", (system._pending_cov_diag or {}).get("median_flow_u_cov"))
    )
    system._write_frame_pair_diagnostics = lambda _frame0, _frame1: events.append(
        ("diagnostics", (system._pending_cov_diag or {}).get("median_flow_u_cov"))
    )
    original_get_graph_data = optimizer.get_graph_data

    def get_graph_data(graph, frame_idx):
        if int(frame_idx.item()) == 1 and len(graph.frames) == 2:
            events.append(("d2_graph", None))
        return original_get_graph_data(graph, frame_idx)

    def sequential_optimize(graph_data):
        events.append(("d2_optimize", None))
        optimizer.last_pair_diagnostics = {
            "est_delta_t_norm": 0.2,
            "r_p_whitened_norm": 0.3,
            "num_visual_residuals": POINT_ROWS,
        }
        return graph_data

    def write_graph_data(_result, _graph):
        events.append(("d2_writeback", None))

    optimizer.get_graph_data = get_graph_data
    optimizer.sequential_optimize = sequential_optimize
    optimizer.write_graph_data = write_graph_data
    optimizer.last_pair_diagnostics = {
        "from_idx": 0,
        "frame_idx": 1,
        "r_p_whitened_norm": 1.5,
        "num_visual_residuals": POINT_ROWS,
    }

    system.initialize(frame0)
    system.run_pair(frame0, frame1)
    events.clear()
    system.imu_vio_velocity_feedback_enable = False
    system._imu_vel_w = torch.tensor([3.0, 4.0, 0.0])
    system._pending_imu_diag = {
        "trans_prior_full": torch.zeros((1, 3), dtype=torch.float32),
        "trans_cov_full": torch.eye(3, dtype=torch.float32).unsqueeze(0),
    }

    system.run_pair(frame1, frame2)

    assert events[:6] == [
        ("gate", packets[0].covariance_diagnostics["median_flow_u_cov"]),
        ("d2_graph", None),
        ("d2_optimize", None),
        ("d2_writeback", None),
        ("adaptive_csv", packets[0].covariance_diagnostics["median_flow_u_cov"]),
        ("diagnostics", packets[0].covariance_diagnostics["median_flow_u_cov"]),
    ]
    assert torch.equal(system._imu_vel_w, torch.zeros(3))
    assert system._pending_cov_diag == packets[1].covariance_diagnostics


def test_replay_builds_real_local_inertial_ba_window_history(tmp_path: Path):
    cache_path, packets = _write_cache(tmp_path)
    optimizer = _LocalBAHistoryOptimizer()
    system, _ = _build_system(cache_path, optimizer=optimizer)
    frame0, frame1, frame2 = (_frame(index) for index in range(3))

    system.initialize(frame0)
    system.run_pair(frame0, frame1)
    system.run_pair(frame1, frame2)

    assert len(optimizer.local_graph_history) == 2
    first_window, second_window = optimizer.local_graph_history
    assert first_window.frame_indices.tolist() == [0, 1]
    assert second_window.frame_indices.tolist() == [0, 1, 2]
    assert len(first_window.edges) == 1
    assert len(second_window.edges) == 2
    assert [int(edge.from_idx.item()) for edge in second_window.edges] == [0, 1]
    assert [int(edge.frame_idx.item()) for edge in second_window.edges] == [1, 2]
    assert all(edge.imu_vio_factor_enable for edge in second_window.edges)
    assert all(float(edge.imu_vio_dt.item()) > 0.0 for edge in second_window.edges)
    assert torch.equal(
        system.graph.match2frame1.mapping.tensor,
        torch.tensor([0] * POINT_ROWS + [1] * POINT_ROWS, dtype=torch.long),
    )
    assert torch.equal(
        system.graph.match2frame2.mapping.tensor,
        torch.tensor([1] * POINT_ROWS + [2] * POINT_ROWS, dtype=torch.long),
    )
    for packet, edge in (
        (packets[0], first_window.edges[0]),
        (packets[1], second_window.edges[-1]),
    ):
        assert len(edge.observations) == POINT_ROWS
        for name, expected in packet.match_fields.items():
            assert torch.equal(edge.observations.data[name], expected)


def test_replay_lost_track_uses_cached_candidate_count_and_adaptive_imu_fallback(tmp_path: Path):
    cache_path, packet = _write_lost_track_cache(tmp_path)
    system, optimizer = _build_system(cache_path)
    frame0, frame1 = _frame(0), _frame(1)
    imu_rel_pose = pp.SE3(
        torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5]], dtype=torch.float32)
    )
    system._adaptive_enabled = True
    system._adaptive_decision = SimpleNamespace(
        use_imu_rotation=True,
        use_imu_translation=True,
    )
    system._estimate_imu_priors = lambda _frame: (
        torch.zeros((1, 3), dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.eye(3, dtype=torch.float32).unsqueeze(0),
        imu_rel_pose,
    )

    system.initialize(frame0)
    system.run_pair(frame0, frame1)

    tracking = system._pending_tracking_state
    assert tracking == {
        "optimizer_skipped": True,
        "match_idx_size_after_filter": 3,
        "num_keypoints_candidate": packet.covariance_diagnostics["num_selected_keypoints"],
        "min_num_point": system.min_num_point,
        "visual_input_sha256": packet.visual_sha256,
        "lost_track_fallback_applied": True,
        "lost_track_fallback_reason": "imu_rotation_translation",
    }
    assert bool(system.graph.frames.data["need_interp"][1].item())
    assert torch.allclose(
        pp.SE3(system.graph.frames.data["pose"][1]).tensor(),
        imu_rel_pose.tensor(),
        atol=1e-6,
        rtol=0.0,
    )
    assert optimizer.started == []


@pytest.mark.parametrize(
    ("setup", "expected_message"),
    [
        (lambda system, frames: system.set_diagnostics_writer(None, scene="wrong-scene"), "scene"),
        (lambda system, frames: None, "K"),
    ],
)
def test_replay_rejects_incompatible_run_metadata(tmp_path: Path, setup, expected_message: str):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)
    frames = (_frame(0, K * 2.0 if expected_message == "K" else K), _frame(1))
    setup(system, frames)

    with pytest.raises(VisualFactorCacheError, match=expected_message):
        system.initialize(frames[0])


def test_replay_rejects_timestamp_mismatch_and_never_falls_back(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)
    frame0 = _frame(0)
    frame1 = _frame(1, timestamp_ns=TIMESTAMPS_NS[1] + 1)

    system.initialize(frame0)

    with pytest.raises(VisualFactorCacheError, match="timestamps"):
        system.run_pair(frame0, frame1)

    assert system.Frontend.inference_call_count == 0


def test_replay_rejects_unavailable_source_pair_and_never_falls_back(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)
    frame0 = _frame(0)
    unavailable = _frame(1)
    unavailable.idx = [3]

    system.initialize(frame0)

    with pytest.raises(VisualFactorCacheError, match="not present"):
        system.run_pair(frame0, unavailable)

    assert system.Frontend.inference_call_count == 0


def test_replay_rejects_starting_from_manifest_suffix(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)

    with pytest.raises(VisualFactorCacheError, match="first.*source frame|start"):
        system.initialize(_frame(1))


def test_replay_completion_validation_rejects_incomplete_consumed_pair_range(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)
    _disable_termination_side_effects(system)
    frame0, frame1 = _frame(0), _frame(1)
    system.initialize(frame0)
    system.run_pair(frame0, frame1)

    with pytest.raises(VisualFactorCacheError, match="complete.*pair range|consumed"):
        system.validate_completion()


def test_replay_preflights_actual_sequence_coverage_metadata(tmp_path: Path, monkeypatch):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)
    delegated: list[bool] = []

    class _ClippedSequence:
        indices = torch.tensor([0, 1], dtype=torch.long)

        def __len__(self):
            return 2

    monkeypatch.setattr(
        IOdometry,
        "receive_frames",
        lambda _self, _sequence, _saveto, on_frame_finished=None: delegated.append(True),
    )

    with pytest.raises(VisualFactorCacheError, match="sequence.*frame_count|complete.*range"):
        system.receive_frames(_ClippedSequence(), object())

    assert delegated == []


def test_receive_frames_preserves_cache_timestamp_error_and_finalizes_once(tmp_path: Path, monkeypatch):
    cache_path, _ = _write_cache(tmp_path)
    system, optimizer = _build_system(cache_path)
    refiner = _CountingMapRefiner()
    diagnostics = _CountingDiagnosticsWriter(tmp_path / "timestamp-error" / "frame_pair_diagnostics.csv")
    system.MapRefiner = refiner
    system.set_diagnostics_writer(diagnostics, scene=SCENE)
    reported: list[BaseException | None] = []
    escaped: list[BaseException] = []
    monkeypatch.setattr(Logger, "show_exception", lambda: reported.append(sys.exc_info()[1]))
    sequence = _DeclaredSequence(
        [_frame(0), _frame(1, timestamp_ns=TIMESTAMPS_NS[1] + 1), _frame(2)],
        declared_length=3,
    )

    try:
        system.receive_frames(sequence, Sandbox(tmp_path / "timestamp-error"))
    except BaseException as error:
        escaped.append(error)
    try:
        system.terminate()
    except BaseException as error:
        escaped.append(error)

    assert escaped == []
    assert len(reported) == 1
    assert isinstance(reported[0], VisualFactorCacheError)
    assert "timestamps" in str(reported[0])
    assert optimizer.write_calls == 2
    assert optimizer.terminate_calls == 1
    assert refiner.calls == 1
    assert diagnostics.close_calls == 1


def test_receive_frames_reports_incomplete_replay_once_and_finalizes_idempotently(tmp_path: Path, monkeypatch):
    cache_path, _ = _write_cache(tmp_path)
    system, optimizer = _build_system(cache_path)
    refiner = _CountingMapRefiner()
    diagnostics = _CountingDiagnosticsWriter(tmp_path / "incomplete" / "frame_pair_diagnostics.csv")
    system.MapRefiner = refiner
    system.set_diagnostics_writer(diagnostics, scene=SCENE)
    reported: list[BaseException | None] = []
    escaped: list[BaseException] = []
    monkeypatch.setattr(Logger, "show_exception", lambda: reported.append(sys.exc_info()[1]))
    sequence = _DeclaredSequence([_frame(0), _frame(1)], declared_length=3)

    try:
        system.receive_frames(sequence, Sandbox(tmp_path / "incomplete"))
    except BaseException as error:
        escaped.append(error)
    try:
        system.terminate()
    except BaseException as error:
        escaped.append(error)

    assert escaped == []
    assert len(reported) == 1
    assert isinstance(reported[0], VisualFactorCacheError)
    assert "complete manifest pair range" in str(reported[0])
    assert optimizer.write_calls == 2
    assert optimizer.terminate_calls == 1
    assert refiner.calls == 1
    assert diagnostics.close_calls == 1


def test_nonreplay_sequence_slicing_still_delegates_without_cache_preflight(tmp_path: Path, monkeypatch):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path, visual_cache_mode="off")
    delegated: list[object] = []

    class _ClippedSequence:
        indices = torch.tensor([4, 6], dtype=torch.long)

        def __len__(self):
            return 2

    sequence = _ClippedSequence()
    monkeypatch.setattr(
        IOdometry,
        "receive_frames",
        lambda _self, received, _saveto, on_frame_finished=None: delegated.append(received),
    )

    system.receive_frames(sequence, object())

    assert delegated == [sequence]


def test_replay_termination_accepts_complete_consumed_pair_range(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)
    _disable_termination_side_effects(system)
    frame0, frame1, frame2 = (_frame(index) for index in range(3))
    system.initialize(frame0)
    system.run_pair(frame0, frame1)
    system.run_pair(frame1, frame2)

    system.terminate()


def test_termination_writes_diagnostics_for_final_backend_result(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)
    system.MapRefiner = SimpleNamespace(elaborate_map=lambda _frames: None)
    frame0, frame1, frame2 = (_frame(index) for index in range(3))
    system.initialize(frame0)
    system.run_pair(frame0, frame1)
    system.run_pair(frame1, frame2)
    written: list[tuple[int, int]] = []
    system._write_frame_pair_diagnostics = lambda left, right: written.append(
        (int(left.frame_idx), int(right.frame_idx))
    )

    system.terminate()

    assert written == [(2, 2)]


def test_replay_rejects_schema_v1_incompatible_modules_and_mapping(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)

    with pytest.raises(ValueError, match="mapping"):
        _build_system(cache_path, mapping=True)
    with pytest.raises(ValueError, match="ReplayFrontend"):
        _build_system(cache_path, frontend=object())
    with pytest.raises(ValueError, match="StaticMotionModel"):
        _build_system(cache_path, motion_model=object())
    with pytest.raises(ValueError, match="AllKeyframe"):
        _build_system(cache_path, keyframe_selector=object())


def test_replay_frontend_would_raise_if_replay_attempted_live_inference(tmp_path: Path):
    cache_path, _ = _write_cache(tmp_path)
    system, _ = _build_system(cache_path)

    with pytest.raises(FrontendReplayViolation):
        system.Frontend.estimate_depth(_frame(0).stereo)


def test_from_config_selects_replay_frontend_before_configured_neural_factory(tmp_path: Path, monkeypatch):
    cache_path, _ = _write_cache(tmp_path)
    neural_allocations: list[str] = []

    def frontend_factory(_cls, frontend_type, _args):
        neural_allocations.append(frontend_type)
        raise AssertionError("configured neural frontend factory executed")

    monkeypatch.setattr(Module.IFrontend, "instantiate", classmethod(frontend_factory))
    monkeypatch.setattr(
        Module.IMotionModel,
        "instantiate",
        classmethod(lambda _cls, _type, _args: _TrackingStaticMotionModel()),
    )
    monkeypatch.setattr(
        Module.IKeypointSelector,
        "instantiate",
        classmethod(lambda _cls, _type, _args: _ForbiddenReplayModule()),
    )
    monkeypatch.setattr(
        Module.IObservationFilter,
        "instantiate",
        classmethod(lambda _cls, _type, _args: _ForbiddenReplayModule()),
    )
    monkeypatch.setattr(
        Module.ICovariance2to3,
        "instantiate",
        classmethod(lambda _cls, _type, _args: _ForbiddenReplayModule()),
    )
    monkeypatch.setattr(
        Module.IMapProcessor,
        "instantiate",
        classmethod(lambda _cls, _type, _args: _ForbiddenReplayModule()),
    )
    monkeypatch.setattr(
        Module.IKeyframeSelector,
        "instantiate",
        classmethod(lambda _cls, _type, _args: AllKeyframe(SimpleNamespace())),
    )
    monkeypatch.setattr(
        Module.IOptimizer,
        "instantiate",
        classmethod(lambda _cls, _type, _args: _HistoryOptimizer()),
    )
    module_config = lambda module_type="unused": SimpleNamespace(
        type=module_type,
        args=SimpleNamespace(),
    )
    cfg = SimpleNamespace(
        Odometry=SimpleNamespace(
            args=SimpleNamespace(
                device="cpu",
                num_point=POINT_ROWS,
                edgewidth=1,
                match_cov_default=1.0,
                profile=False,
                mapping=False,
                visual_cache_mode="replay",
                visual_cache_path=str(cache_path),
            ),
            frontend=module_config("FailOnAllocationNeuralFrontend"),
            motion=module_config(),
            keypoint=module_config(),
            mappoint=module_config(),
            outlier=module_config(),
            cov=SimpleNamespace(obs=module_config()),
            postprocess=module_config(),
            keyframe=module_config(),
            optimizer=module_config(),
        )
    )

    system = MACVO.from_config(cfg)

    assert isinstance(system.Frontend, ReplayFrontend)
    assert neural_allocations == []

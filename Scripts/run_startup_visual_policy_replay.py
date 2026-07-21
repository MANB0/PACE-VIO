#!/usr/bin/env python3
"""Run Gate-8 startup visual policies in the unchanged SA-v2 N=2 backend.

The policy factors live only in this replay script. The production optimizer,
IMU factor, cross-edge sampling model, marginal prior, and LM settings are not
modified.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pypose as pp
import torch
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Module.Optimization.TwoFramePGO.Optimizer as optimizer_module  # noqa: E402
import Scripts.audit_macvo_translation_point_level as d3  # noqa: E402
import Scripts.diagnose_startup_visual_upward_drift as startup  # noqa: E402
import Scripts.run_circle_translation_oracles as base  # noqa: E402
import Scripts.run_direct_uvd_short_experiments as direct  # noqa: E402
import Utility.TwoStateSamplingAwareVIO as sampling_vio  # noqa: E402
import Utility.TwoStateVIO as two_state_vio  # noqa: E402
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO  # noqa: E402
from Utility.Point import pixel2point_NED  # noqa: E402
from Utility.RelativePoseFactorCache import (  # noqa: E402
    RelativePoseFactorCacheReader,
    RelativePoseFactorPacket,
    camera_factor_to_body_factor,
    relative_pose_information_from_packet,
    write_relative_pose_factor_cache,
)
from Utility.TwoStateSamplingAwareVIO import CrossEdgeTwoStateSolver  # noqa: E402
from Utility.TwoStateVIO import (  # noqa: E402
    NavigationState,
    RelativePoseFactor,
    TwoStateVIOSolver,
    _whiten,
    _whiten_rows,
)
from Utility.VisualFactorCache import (  # noqa: E402
    MATCH_FIELDS,
    VisualFactorCacheReader,
    VisualFactorPacket,
    write_visual_factor_cache,
)


OUTPUT = ROOT / "analysis_startup_visual_upward_drift_20260718"
GATE8 = OUTPUT / "gate8_vio_policy_replay"
CONTRACT = OUTPUT / "active_start_boundary_contract.json"
CONTINUOUS_CACHE = ROOT / "VisualCache/static63_unique_visual_20260713/clear_circle_truth_normal_noise"
COLD_CACHE = (
    OUTPUT
    / "same_build_cold_start_ab/caches/C1_cold/clear_circle_truth_normal_noise"
)
HYBRID_CACHE = GATE8 / "cold_start_hybrid_cache"
BASE_CONFIG = (
    ROOT
    / "Results/circle_direct_uvd_sampling_aware_v2_full_20260717/"
    "sampling_aware_v2/configs/odometry.yaml"
)
DATA_CONFIG = base.DATA_CONFIG_SOURCE
DATASET = startup.DATASET
EDGE_COUNT = 120
MODES = ("V0", "V1", "V2", "V3", "V4", "V5")


@dataclass(frozen=True)
class RotationOnlyFactor:
    measurement_BiBj: torch.Tensor
    rotation_covariance: torch.Tensor
    huber_delta: float = 3.0

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "RotationOnlyFactor":
        return RotationOnlyFactor(
            measurement_BiBj=self.measurement_BiBj.reshape(1, 7).to(device=device, dtype=dtype),
            rotation_covariance=self.rotation_covariance.reshape(3, 3).to(
                device=device, dtype=dtype
            ),
            huber_delta=float(self.huber_delta),
        )


@dataclass(frozen=True)
class ThreeDThreeDFactor:
    points_Ci: torch.Tensor
    points_Cj: torch.Tensor
    covariance_Ci: torch.Tensor
    covariance_Cj: torch.Tensor
    extrinsic_CI: torch.Tensor
    huber_delta: float = 3.0

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "ThreeDThreeDFactor":
        count = int(self.points_Ci.reshape(-1, 3).shape[0])
        return ThreeDThreeDFactor(
            points_Ci=self.points_Ci.reshape(count, 3).to(device=device, dtype=dtype),
            points_Cj=self.points_Cj.reshape(count, 3).to(device=device, dtype=dtype),
            covariance_Ci=self.covariance_Ci.reshape(count, 3, 3).to(
                device=device, dtype=dtype
            ),
            covariance_Cj=self.covariance_Cj.reshape(count, 3, 3).to(
                device=device, dtype=dtype
            ),
            extrinsic_CI=self.extrinsic_CI.reshape(1, 7).to(device=device, dtype=dtype),
            huber_delta=float(self.huber_delta),
        )


@dataclass
class PolicyRuntime:
    mode: str
    active_start: int
    observable_streak: int = 0
    full_translation_enable_frame: int | None = None
    ramp_index: int = 0
    decisions: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.decisions is None:
            self.decisions = []


RUNTIME: PolicyRuntime | None = None
ORIGINAL_FACTORY = optimizer_module._make_two_state_uvd_factor
ORIGINAL_VISUAL_RESIDUAL = two_state_vio.visual_whitened_residuals


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(startup.jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def active_start() -> int:
    return int(json.loads(CONTRACT.read_text(encoding="utf-8"))["active_start_frame"])


def _copy_packet(packet: VisualFactorPacket, frame_i: int, frame_j: int) -> VisualFactorPacket:
    return replace(packet, frame_i=frame_i, frame_j=frame_j)


def prepare_hybrid_cache(*, force: bool = False) -> Path:
    """Use continuous packets before s and true-cold packets from s onward."""

    s = active_start()
    frame_limit = s + EDGE_COUNT + 1
    if (HYBRID_CACHE / "manifest.json").exists() and (
        HYBRID_CACHE / "relative_pose_factors.npz"
    ).exists() and not force:
        reader = VisualFactorCacheReader(HYBRID_CACHE)
        if reader.manifest.frame_count == frame_limit:
            return HYBRID_CACHE
    if HYBRID_CACHE.exists():
        shutil.rmtree(HYBRID_CACHE)

    continuous = VisualFactorCacheReader(CONTINUOUS_CACHE)
    cold = VisualFactorCacheReader(COLD_CACHE)
    if cold.manifest.frame_count < EDGE_COUNT + 1:
        raise RuntimeError("same-build C1 cache does not contain 120 active edges")
    packets: list[VisualFactorPacket] = []
    for frame_i in range(s):
        packets.append(
            continuous.load_pair(
                frame_i,
                frame_i + 1,
                int(continuous.manifest.timestamps_ns[frame_i]),
                int(continuous.manifest.timestamps_ns[frame_i + 1]),
            )
        )
    for local in range(EDGE_COUNT):
        packet = cold.load_pair(
            local,
            local + 1,
            int(cold.manifest.timestamps_ns[local]),
            int(cold.manifest.timestamps_ns[local + 1]),
        )
        packets.append(_copy_packet(packet, s + local, s + local + 1))
    write_visual_factor_cache(
        HYBRID_CACHE,
        startup.SCENE,
        packets,
        source={
            "frame_count": frame_limit,
            "result": "diagnostic hybrid: continuous pre-s packets plus same-build C1 cold packets",
            "config": str(BASE_CONFIG),
            "git": "working-tree Gate-8 replay",
            "exporter": Path(__file__).name,
        },
    )

    continuous_pose = RelativePoseFactorCacheReader(CONTINUOUS_CACHE)
    cold_pose = RelativePoseFactorCacheReader(COLD_CACHE)
    pose_packets: list[RelativePoseFactorPacket] = []
    for frame_i in range(s):
        visual = packets[frame_i]
        pose_packets.append(continuous_pose.load_pair(frame_i, frame_i + 1, visual.visual_sha256))
    for local in range(EDGE_COUNT):
        visual = packets[s + local]
        item = cold_pose.load_pair(local, local + 1, visual.visual_sha256)
        pose_packets.append(replace(item, frame_i=s + local, frame_j=s + local + 1))
    write_relative_pose_factor_cache(HYBRID_CACHE, pose_packets)
    write_json(
        GATE8 / "hybrid_cache_contract.json",
        {
            "active_start_frame": s,
            "frame_limit": frame_limit,
            "formal_active_edges": [s, frame_limit - 1],
            "pre_s_packets": "continuous cache; VIO factor inactive",
            "post_s_packets": "same-build true-cold C1 cache local 0..120",
            "first_active_visual_edge": [s, s + 1],
            "pre_s_visual_measurement_used_by_formal_vio": False,
        },
    )
    return HYBRID_CACHE


def graph_packet(graph_data) -> VisualFactorPacket:
    count = int(graph_data.points.data["pos_Tc"].reshape(-1, 3).shape[0])
    match_fields = {}
    for name in MATCH_FIELDS:
        value = graph_data.observations.data.get(name)
        if value is None:
            raise ValueError(f"Gate-8 visual packet is missing {name}")
        match_fields[name] = value
    return VisualFactorPacket(
        frame_i=int(graph_data.from_idx.reshape(-1)[0].item()),
        frame_j=int(graph_data.frame_idx.reshape(-1)[0].item()),
        timestamp_i_ns=0,
        timestamp_j_ns=0,
        K=graph_data.images_intrinsic,
        baseline_m=float(graph_data.baseline.reshape(-1)[0].item()),
        relative_pose_init=graph_data.visual_relative_pose_CiCj,
        points_local=graph_data.points.data["pos_Tc"],
        points_cov_local=match_fields["obs1_covTc"],
        point_colors=torch.zeros((count, 3), dtype=torch.uint8),
        match_fields=match_fields,
        covariance_diagnostics={},
        visual_sha256="diagnostic-graph-packet",
    )


def numpy_pose(value: torch.Tensor) -> np.ndarray:
    return startup.se3_from_xyzw(value.detach().cpu().numpy().reshape(7))


def stabilize_covariance(covariance: torch.Tensor, floor: float = 1.0e-12) -> torch.Tensor:
    covariance = 0.5 * (covariance + covariance.mT)
    values, vectors = torch.linalg.eigh(covariance)
    return vectors @ torch.diag(values.clamp_min(floor)) @ vectors.mT


def information_scaled_covariance(
    covariance: torch.Tensor, translation_scale: torch.Tensor
) -> torch.Tensor:
    covariance = stabilize_covariance(covariance)
    information = torch.linalg.inv(covariance)
    scale = torch.eye(6, dtype=covariance.dtype, device=covariance.device)
    scale[:3, :3] = translation_scale
    modified = scale @ information @ scale.mT
    return stabilize_covariance(torch.linalg.inv(stabilize_covariance(modified)))


def body_pose_factor(graph_data, covariance_camera: torch.Tensor | None = None) -> RelativePoseFactor:
    dtype = torch.float64
    device = torch.device("cpu")
    covariance = (
        graph_data.visual_relative_pose_cov.to(device=device, dtype=dtype)
        if covariance_camera is None
        else covariance_camera.to(device=device, dtype=dtype)
    )
    measurement, covariance_body = camera_factor_to_body_factor(
        graph_data.visual_relative_pose_CiCj.to(device=device, dtype=dtype),
        covariance,
        graph_data.imu_vio_sensor_T_imu.to(device=device, dtype=dtype),
    )
    return RelativePoseFactor(measurement, covariance_body, huber_delta=3.0)


def rotation_only_factor(graph_data) -> RotationOnlyFactor:
    factor = body_pose_factor(graph_data)
    return RotationOnlyFactor(
        measurement_BiBj=factor.measurement_BiBj,
        rotation_covariance=stabilize_covariance(factor.covariance[3:6, 3:6]),
        huber_delta=3.0,
    )


def observability(packet: VisualFactorPacket, measurement: np.ndarray) -> dict[str, Any]:
    terms, hessian, _ = startup.uvd_linearization(packet, measurement)
    h_tt = hessian[:3, :3]
    values, vectors = np.linalg.eigh(h_tt)
    derotated = startup.projection_rotation_only_flow(packet, measurement)
    derotated_median = float(np.median(np.linalg.norm(derotated, axis=1)))
    uv_cov = packet.match_fields["pixel2_uv_cov"].numpy()
    pixel_sigma = float(np.median(np.sqrt(np.maximum(uv_cov[:, 0] + uv_cov[:, 1], 0.0))))
    d1 = packet.match_fields["pixel1_d"].numpy().reshape(-1)
    d2 = packet.match_fields["pixel2_d"].numpy().reshape(-1)
    c1 = packet.match_fields["pixel1_d_cov"].numpy().reshape(-1)
    c2 = packet.match_fields["pixel2_d_cov"].numpy().reshape(-1)
    rel1 = np.sqrt(np.maximum(c1, 0.0)) / np.maximum(np.abs(d1), 1.0e-6)
    rel2 = np.sqrt(np.maximum(c2, 0.0)) / np.maximum(np.abs(d2), 1.0e-6)
    low_depth_count = int(np.sum(np.maximum(rel1, rel2) <= 0.25))
    flow_snr = derotated_median / max(pixel_sigma, 1.0e-12)
    condition = float(values[-1] / max(values[0], 1.0e-12))
    observable = bool(
        len(terms["weight"]) >= 100
        and values[0] >= 1.0e4
        and condition <= 10.0
        and low_depth_count >= 50
        and flow_snr >= 3.0
    )
    return {
        "point_count": int(len(terms["weight"])),
        "H_tt_eig_min": float(values[0]),
        "H_tt_eig_mid": float(values[1]),
        "H_tt_eig_max": float(values[2]),
        "H_tt_condition": condition,
        "H_tt_eigenvectors": vectors,
        "median_derotated_flow_px": derotated_median,
        "median_pixel_sigma_px": pixel_sigma,
        "derotated_flow_snr": flow_snr,
        "low_relative_depth_cov_count": low_depth_count,
        "observable": observable,
    }


def policy_factory(graph_data, extrinsic_CI, *, device, dtype, huber_delta):
    if RUNTIME is None:
        raise RuntimeError("Gate-8 policy runtime is not initialized")
    standard = ORIGINAL_FACTORY(
        graph_data,
        extrinsic_CI,
        device=device,
        dtype=dtype,
        huber_delta=huber_delta,
    )
    frame_i = int(graph_data.from_idx.reshape(-1)[0].item())
    packet = graph_packet(graph_data)
    z_mac = numpy_pose(graph_data.visual_relative_pose_CiCj)
    metrics = observability(packet, z_mac)
    action = "full_uvd"
    translation_information_scale = 1.0

    if RUNTIME.mode == "V0":
        visual = standard
    elif RUNTIME.mode == "V1":
        visual = rotation_only_factor(graph_data)
        action = "rotation_only"
        translation_information_scale = 0.0
    elif RUNTIME.mode == "V2":
        eigenvalues = np.asarray(
            [metrics["H_tt_eig_min"], metrics["H_tt_eig_mid"], metrics["H_tt_eig_max"]]
        )
        eigenvectors = np.asarray(metrics["H_tt_eigenvectors"])
        ratios = np.maximum(eigenvalues / max(eigenvalues[-1], 1.0e-12), 1.0e-3)
        scale_t = eigenvectors @ np.diag(np.sqrt(ratios)) @ eigenvectors.T
        camera_cov = information_scaled_covariance(
            graph_data.visual_relative_pose_cov.double(), torch.from_numpy(scale_t).double()
        )
        visual = body_pose_factor(graph_data, camera_cov)
        action = "directional_translation_information"
        translation_information_scale = float(ratios[0])
    elif RUNTIME.mode == "V3":
        RUNTIME.observable_streak = (
            RUNTIME.observable_streak + 1 if metrics["observable"] else 0
        )
        if RUNTIME.full_translation_enable_frame is None and RUNTIME.observable_streak >= 3:
            RUNTIME.full_translation_enable_frame = frame_i
            RUNTIME.ramp_index = 1
        if RUNTIME.full_translation_enable_frame is None:
            visual = rotation_only_factor(graph_data)
            action = "rotation_only_waiting_for_three_observable_edges"
            translation_information_scale = 0.0
        else:
            alpha = min(RUNTIME.ramp_index / 3.0, 1.0)
            RUNTIME.ramp_index += 1
            camera_cov = information_scaled_covariance(
                graph_data.visual_relative_pose_cov.double(),
                torch.eye(3, dtype=torch.float64) * math.sqrt(alpha),
            )
            visual = body_pose_factor(graph_data, camera_cov)
            action = "translation_ramp"
            translation_information_scale = alpha
    elif RUNTIME.mode == "V4":
        estimate, _ = startup.optimize_uvd(
            packet, z_mac, fixed_rotation=z_mac[:3, :3], max_iterations=50
        )
        measurement = pp.mat2SE3(torch.from_numpy(estimate).reshape(1, 4, 4)).tensor()
        covariance, _ = relative_pose_information_from_packet(
            packet, measurement, huber_delta=3.0
        )
        measurement_body, covariance_body = camera_factor_to_body_factor(
            measurement,
            covariance,
            graph_data.imu_vio_sensor_T_imu.double(),
        )
        visual = RelativePoseFactor(measurement_body, covariance_body, huber_delta=3.0)
        action = "U1_fixed_MACVO_rotation_resolved_translation"
    elif RUNTIME.mode == "V5":
        points_j = pixel2point_NED(
            packet.match_fields["pixel2_uv"].double().unsqueeze(0),
            packet.match_fields["pixel2_d"].double().reshape(1, -1),
            packet.K.double(),
        ).squeeze(0)
        visual = ThreeDThreeDFactor(
            points_Ci=packet.points_local,
            points_Cj=points_j,
            covariance_Ci=packet.match_fields["obs1_covTc"],
            covariance_Cj=packet.match_fields["obs2_covTc"],
            extrinsic_CI=extrinsic_CI.tensor(),
            huber_delta=d3.HUBER_DELTA,
        ).to(device=device, dtype=dtype)
        action = "covariance_aware_3D3D_joint"
    else:
        raise ValueError(f"unsupported Gate-8 mode {RUNTIME.mode}")

    record = {
        "mode": RUNTIME.mode,
        "frame_i": frame_i,
        "frame_j": frame_i + 1,
        "action": action,
        "translation_information_scale": translation_information_scale,
        "observable_streak": RUNTIME.observable_streak,
        "full_translation_enable_frame": RUNTIME.full_translation_enable_frame,
        **{key: value for key, value in metrics.items() if key != "H_tt_eigenvectors"},
    }
    RUNTIME.decisions.append(record)
    return visual


def policy_visual_residuals(
    state_i: NavigationState,
    state_j: NavigationState,
    visual,
    covariance_eigenvalue_floor: float,
) -> torch.Tensor:
    if isinstance(visual, RotationOnlyFactor):
        predicted = pp.SE3(state_i.pose_WB).Inv() @ pp.SE3(state_j.pose_WB)
        raw = (
            pp.SO3(pp.SE3(visual.measurement_BiBj).rotation()).Inv()
            @ pp.SO3(predicted.rotation())
        ).Log().tensor().reshape(3)
        return _whiten(
            raw, visual.rotation_covariance, covariance_eigenvalue_floor
        ).reshape(1, 3)
    if isinstance(visual, ThreeDThreeDFactor):
        extrinsic_CI = pp.SE3(visual.extrinsic_CI)
        pose_WCi = pp.SE3(state_i.pose_WB) @ extrinsic_CI.Inv()
        pose_WCj = pp.SE3(state_j.pose_WB) @ extrinsic_CI.Inv()
        relative_CiCj = pose_WCi.Inv() @ pose_WCj
        predicted_i = relative_CiCj.Act(visual.points_Cj)
        raw = visual.points_Ci - predicted_i
        rotation = relative_CiCj.rotation().matrix().reshape(3, 3)
        covariance = (
            visual.covariance_Ci
            + rotation.detach().unsqueeze(0)
            @ visual.covariance_Cj
            @ rotation.detach().mT.unsqueeze(0)
        )
        return _whiten_rows(raw, covariance, covariance_eigenvalue_floor)
    return ORIGINAL_VISUAL_RESIDUAL(
        state_i, state_j, visual, covariance_eigenvalue_floor
    )


def prepare_config(mode: str) -> Path:
    root = GATE8 / mode
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    odometry = config["Odometry"]["args"]
    odometry["mapping"] = False
    odometry["two_state_covariance_mode"] = "sampling_aware_cross_edge"
    optimizer = config["Odometry"]["optimizer"]["args"]
    optimizer["two_state_visual_factor_mode"] = "direct_uvd"
    optimizer["two_state_warm_start"] = "macvo_pose"
    path = root / "effective_odometry.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def install_policy_hooks() -> None:
    optimizer_module._make_two_state_uvd_factor = policy_factory
    optimizer_module.visual_whitened_residuals = policy_visual_residuals
    two_state_vio.visual_whitened_residuals = policy_visual_residuals
    sampling_vio.visual_whitened_residuals = policy_visual_residuals


def restore_policy_hooks() -> None:
    optimizer_module._make_two_state_uvd_factor = ORIGINAL_FACTORY
    optimizer_module.visual_whitened_residuals = ORIGINAL_VISUAL_RESIDUAL
    two_state_vio.visual_whitened_residuals = ORIGINAL_VISUAL_RESIDUAL
    sampling_vio.visual_whitened_residuals = ORIGINAL_VISUAL_RESIDUAL


def run_mode(mode: str, *, force: bool) -> None:
    global RUNTIME
    s = active_start()
    frame_limit = s + EDGE_COUNT + 1
    cache = prepare_hybrid_cache()
    root = GATE8 / mode
    run_root = root / "run_result"
    if run_root.exists():
        if not force:
            raise FileExistsError(f"{run_root} exists; pass --force")
        shutil.rmtree(run_root)
    config = prepare_config(mode)
    base.TRACE_ROWS.clear()
    RUNTIME = PolicyRuntime(mode=mode, active_start=s)
    install_policy_hooks()
    TwoStateVIOSolver.solve = base.traced_solve
    CrossEdgeTwoStateSolver.solve = direct.traced_cross_edge_solve
    TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(base.traced_optimize)
    started = time.perf_counter()
    try:
        sys.argv = [
            str(ROOT / "MACVO.py"),
            "--odom", str(config),
            "--data", str(DATA_CONFIG),
            "--resultRoot", str(run_root),
            "--visual-cache-mode", "replay",
            "--visual-cache-path", str(cache),
            "--seq_to", str(frame_limit),
        ]
        runpy.run_path(str(ROOT / "MACVO.py"), run_name="__main__")
    finally:
        elapsed = time.perf_counter() - started
        TwoStateVIOSolver.solve = base.ORIGINAL_SOLVE
        CrossEdgeTwoStateSolver.solve = direct.ORIGINAL_CROSS_EDGE_SOLVE
        TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(base.ORIGINAL_OPTIMIZE)
        restore_policy_hooks()
    if not base.TRACE_ROWS:
        raise RuntimeError(f"{mode} produced no two-state trace rows")
    pd.DataFrame(base.TRACE_ROWS).to_csv(root / "factor_trace.csv", index=False)
    pd.DataFrame(RUNTIME.decisions).to_csv(root / "policy_decisions.csv", index=False)
    (root / "runtime_seconds.txt").write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    summarize_mode(mode)


def localize(poses: np.ndarray, start: int) -> np.ndarray:
    anchor = base.invert_transform(poses[start])
    return np.stack([anchor @ pose for pose in poses[start:]])


def segment_summary(estimate: np.ndarray, truth: np.ndarray, edge_count: int) -> dict[str, Any]:
    count = min(edge_count + 1, len(estimate), len(truth))
    metrics = base.trajectory_metrics(estimate[:count], truth[:count])
    difference = estimate[:count, :3, 3] - truth[:count, :3, 3]
    metrics["final_plot_up_error_nwu_y_m"] = float(difference[-1, 1])
    metrics["final_position_error_m"] = float(np.linalg.norm(difference[-1]))
    return metrics


def summarize_mode(mode: str) -> dict[str, Any]:
    s = active_start()
    root = GATE8 / mode
    run_root = root / "run_result"
    pose_path = base.latest(run_root, "poses.csv")
    tensor_path = base.latest(run_root, "tensor_map.npz")
    diagnostics_path = base.latest(run_root, "frame_pair_diagnostics.csv")
    timestamps, pose_est = base.read_pose_csv(pose_path)
    ref = pd.read_csv(DATASET / "ref_pose.csv").iloc[: len(timestamps)]
    if not np.array_equal(timestamps, ref["timestamp"].to_numpy(np.int64)):
        raise AssertionError("Gate-8 replay timestamps differ from reference")
    pose_gt = np.stack(
        [
            base.make_transform(Rotation.from_quat(row[3:7]).as_matrix(), row[:3])
            for row in ref[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(np.float64)
        ]
    )
    estimate_local = localize(pose_est, s)
    truth_local = localize(pose_gt, s)
    trace = pd.read_csv(root / "factor_trace.csv")
    trace = (
        trace.sort_values(["frame_i", "solver_call"])
        .groupby(["frame_i", "frame_j"], as_index=False)
        .tail(1)
    )
    trace = trace[(trace.frame_i >= s) & (trace.frame_j < len(timestamps))].reset_index(drop=True)
    decisions = pd.read_csv(root / "policy_decisions.csv")

    with np.load(tensor_path, allow_pickle=False) as tensor:
        velocity_est = tensor["frames//imu_vio_velocity_world"][: len(timestamps)].astype(np.float64)
        ba_est = tensor["frames//imu_vio_acc_bias"][: len(timestamps)].astype(np.float64)
        bg_est = tensor["frames//imu_vio_gyro_bias"][: len(timestamps)].astype(np.float64)
    diagnostics = pd.read_csv(diagnostics_path)
    valid = diagnostics[(diagnostics.frame_i >= s) & (diagnostics.frame_j < len(timestamps))]
    velocity_by_frame: dict[int, np.ndarray] = {}
    for row in valid.itertuples(index=False):
        velocity_by_frame[int(row.frame_i)] = np.array(
            [row.gt_velocity_i_x, row.gt_velocity_i_y, row.gt_velocity_i_z]
        )
        velocity_by_frame[int(row.frame_j)] = np.array(
            [row.gt_velocity_j_x, row.gt_velocity_j_y, row.gt_velocity_j_z]
        )
    fallback_velocity = ref[["vx", "vy", "vz"]].to_numpy(np.float64) @ base.FLU_TO_NED.T
    velocity_gt = np.stack(
        [velocity_by_frame.get(index, fallback_velocity[index]) for index in range(len(timestamps))]
    )
    decomposition = pd.read_csv(DATASET / "imu_truth_decomposition.csv")
    imu_time = decomposition.timestamp.to_numpy(np.int64)
    ba_gt = base.interpolate_rows(
        imu_time,
        decomposition[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(np.float64),
        timestamps,
    ) @ base.FLU_TO_NED.T
    bg_gt = base.interpolate_rows(
        imu_time,
        decomposition[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(np.float64),
        timestamps,
    ) @ base.FLU_TO_NED.T
    active = slice(s, len(timestamps))
    factor_costs = {
        name: {
            "sum_before": float(trace[f"{name}_cost_before"].sum()),
            "sum_after": float(trace[f"{name}_cost_after"].sum()),
            "median_after": float(trace[f"{name}_cost_after"].median()),
        }
        for name in ("prior", "imu", "bias", "pose")
    }
    summary = {
        "mode": mode,
        "visual_policy": str(decisions.action.iloc[0]) if decisions.action.nunique() == 1 else "dynamic",
        "active_start_frame": s,
        "active_edge_count": int(len(trace)),
        "segments": {
            str(count): segment_summary(estimate_local, truth_local, count)
            for count in (30, 60, 120)
        },
        "truth_rmse": {
            "velocity_mps": base.rmse_norm(velocity_est[active] - velocity_gt[active]),
            "acc_bias_mps2": base.rmse_norm(ba_est[active] - ba_gt[active]),
            "gyro_bias_radps": base.rmse_norm(bg_est[active] - bg_gt[active]),
        },
        "factor_costs": factor_costs,
        "convergence": {
            "rate": float(trace.converged.astype(bool).mean()),
            "iterations_median": float(trace.iterations.median()),
            "iterations_p95": float(trace.iterations.quantile(0.95)),
            "iteration_limit_count": int((trace.iterations >= 20).sum()),
        },
        "bias_update": {
            "ba_state_j_update_norm_median": float(
                np.median(
                    np.linalg.norm(
                        trace[[f"state_j_update_{i}" for i in range(9, 12)]].to_numpy(np.float64), axis=1
                    )
                )
            ),
            "bg_state_j_update_norm_median": float(
                np.median(
                    np.linalg.norm(
                        trace[[f"state_j_update_{i}" for i in range(12, 15)]].to_numpy(np.float64), axis=1
                    )
                )
            ),
        },
        "full_translation_enable_frame": (
            None
            if decisions.full_translation_enable_frame.dropna().empty
            else int(decisions.full_translation_enable_frame.dropna().iloc[0])
        ),
        "runtime_seconds": float((root / "runtime_seconds.txt").read_text().strip()),
        "all_finite": bool(np.isfinite(trace.select_dtypes(include=[np.number])).all().all()),
        "artifacts": {
            "poses": str(pose_path),
            "tensor_map": str(tensor_path),
            "diagnostics": str(diagnostics_path),
            "factor_trace": str(root / "factor_trace.csv"),
            "policy_decisions": str(root / "policy_decisions.csv"),
        },
    }
    write_json(root / "summary.json", summary)
    return summary


def build_final_summary() -> dict[str, Any]:
    summaries = {
        mode: json.loads((GATE8 / mode / "summary.json").read_text(encoding="utf-8"))
        for mode in MODES
    }
    ranking = sorted(
        MODES,
        key=lambda mode: summaries[mode]["segments"]["120"]["ate_xy_rmse_m_no_alignment"],
    )
    payload = {
        "gate8_executed": True,
        "backend_contract": {
            "state_window": "N=2, both navigation states optimized",
            "imu": "same standard local-frame preintegration",
            "sampling_covariance": "same SA-v2 cross-edge covariance",
            "bias_and_prior": "same bias RW and Schur prior",
            "solver": "same LM settings from the locked SA-v2 full config",
            "visual_start": "same-build true-cold MACVO packets from frame s",
            "pre_s_visual_factor_count": 0,
            "production_code_modified": False,
        },
        "observability_rule_V3": {
            "uses_ground_truth": False,
            "required_consecutive_edges": 3,
            "point_count_min": 100,
            "H_tt_min_eigenvalue": 1.0e4,
            "H_tt_condition_max": 10.0,
            "relative_depth_std_max": 0.25,
            "low_depth_point_count_min": 50,
            "derotated_flow_snr_min": 3.0,
            "translation_information_ramp": [1 / 3, 2 / 3, 1.0],
        },
        "mode_definitions": {
            "V0": "current full nonlinear UVD R+t point factor",
            "V1": "rotation-only relative-pose factor; translation constrained only by IMU/prior",
            "V2": (
                "relative-pose compression with translation information scaled in the production-UVD "
                "H_tt eigenbasis; this is a directional policy test, not the original nonlinear UVD"
            ),
            "V3": (
                "V1 until three consecutive non-GT observable edges, then a 1/3, 2/3, 1 translation-"
                "information ramp; the rule never opened within these 120 low-motion edges"
            ),
            "V4": "Gate-6 U1: keep MACVO rotation, re-solve UVD translation, then compress to a pose factor",
            "V5": (
                "joint covariance-aware 3D-3D factor with Sigma_i + R Sigma_j R^T; included as the "
                "required 3D-3D candidate even though Gate 7 did not recommend it"
            ),
        },
        "modes": summaries,
        "ranking_by_first120_xy_ate": ranking,
        "best_mode": ranking[0],
        "production_approval": "diagnostic evidence only; no default policy changed",
    }
    write_json(OUTPUT / "startup_visual_policy_summary.json", payload)
    build_comparison_artifacts(summaries)
    build_chinese_report(payload)
    return payload


def build_comparison_artifacts(summaries: dict[str, Any]) -> None:
    rows = []
    trajectory_rows = []
    s = active_start()
    ref = pd.read_csv(DATASET / "ref_pose.csv")
    truth = np.stack(
        [
            base.make_transform(Rotation.from_quat(row[3:7]).as_matrix(), row[:3])
            for row in ref[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(np.float64)
        ]
    )
    truth_local = localize(truth, s)[: EDGE_COUNT + 1]
    for mode, summary in summaries.items():
        row: dict[str, Any] = {
            "mode": mode,
            "visual_policy": summary["visual_policy"],
            "full_translation_enable_frame": summary["full_translation_enable_frame"],
            "velocity_truth_rmse_mps": summary["truth_rmse"]["velocity_mps"],
            "acc_bias_truth_rmse_mps2": summary["truth_rmse"]["acc_bias_mps2"],
            "gyro_bias_truth_rmse_radps": summary["truth_rmse"]["gyro_bias_radps"],
            "converged_rate": summary["convergence"]["rate"],
            "median_iterations": summary["convergence"]["iterations_median"],
            "runtime_seconds": summary["runtime_seconds"],
        }
        for count in (30, 60, 120):
            metrics = summary["segments"][str(count)]
            for key in (
                "ate_xy_rmse_m_no_alignment",
                "translation_rpe_rmse_m",
                "rotation_rpe_rmse_rad",
                "final_plot_up_error_nwu_y_m",
                "final_position_error_m",
            ):
                row[f"{key}_{count}"] = metrics[key]
        rows.append(row)

        _, estimate = base.read_pose_csv(Path(summary["artifacts"]["poses"]))
        estimate_local = localize(estimate, s)[: EDGE_COUNT + 1]
        for local, (est, gt) in enumerate(zip(estimate_local, truth_local)):
            trajectory_rows.append(
                {
                    "mode": mode,
                    "local_frame": local,
                    "source_frame": s + local,
                    "est_x_nwu": est[0, 3],
                    "est_y_nwu": est[1, 3],
                    "est_z_nwu": est[2, 3],
                    "gt_x_nwu": gt[0, 3],
                    "gt_y_nwu": gt[1, 3],
                    "gt_z_nwu": gt[2, 3],
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "startup_visual_policy_comparison.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(
        OUTPUT / "startup_visual_policy_trajectories.csv", index=False
    )


def build_chinese_report(policy: dict[str, Any]) -> None:
    boundary = json.loads(CONTRACT.read_text(encoding="utf-8"))
    cold = json.loads((OUTPUT / "cold_start_reset_audit.json").read_text(encoding="utf-8"))
    oracle = json.loads((OUTPUT / "startup_edge_oracle_summary.json").read_text(encoding="utf-8"))
    observability_summary = json.loads(
        (OUTPUT / "startup_uvd_observability_summary.json").read_text(encoding="utf-8")
    )
    rt = json.loads((OUTPUT / "startup_rt_decoupling_summary.json").read_text(encoding="utf-8"))
    comparison_3d = json.loads(
        (OUTPUT / "startup_uvd_vs_3d3d_summary.json").read_text(encoding="utf-8")
    )
    modes = policy["modes"]
    table = [
        "| 模式 | 30边 XY ATE (m) | 60边 | 120边 | 120边 plot-up (m) | v RMSE (m/s) | 收敛率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = modes[mode]
        table.append(
            f"| {mode} | {item['segments']['30']['ate_xy_rmse_m_no_alignment']:.6f} | "
            f"{item['segments']['60']['ate_xy_rmse_m_no_alignment']:.6f} | "
            f"{item['segments']['120']['ate_xy_rmse_m_no_alignment']:.6f} | "
            f"{item['segments']['120']['final_plot_up_error_nwu_y_m']:.6f} | "
            f"{item['truth_rmse']['velocity_mps']:.6f} | {item['convergence']['rate']:.1%} |"
        )
    report = f"""# 三秒 IMU 静止初始化后的 MACVO 启动漂移诊断

## 首页结论

1. **VIO 是否在三秒处把 MACVO 重新置零：是。** 活动起点由时间戳推导为 frame `{boundary['active_start_frame']}`，时间 `{boundary['active_start_timestamp_ns']}` ns。生产结果中该帧位姿到单位阵的李代数误差为 `{boundary['production_first_state_identity_log_norm']:.3e}`。第一视觉边和第一 IMU 边均为 `{boundary['first_visual_edge'][0]}->{boundary['first_visual_edge'][1]}`；首条 IMU 区间严格为三秒之后，未使用 pre-s support。
2. **三秒前的 MACVO 历史是否是主要根因：现有证据不支持。** 生产 sidecar 的确来自 frame 0 连续运行，因此来源契约并非严格冷启动；但同版本 C0/C1 差异没有超过独立重复运行因随机选点产生的差异，`history_is_primary_cause={cold['history_is_primary_cause']}`。Gate 2–8 正式诊断均使用 frame s 冷启动数据。
3. **漂移不是只由前几条边污染造成。** Gate 4 分类为 `{oracle['classification']}`；只替换前 1/3/5/10/20 条平移均只能部分改善，说明活动段存在持续视觉平移偏差。
4. **plot-up 不是 H_tt 的弱可观方向。** 最弱平移特征向量与 plot-up 的夹角中位数为 `{observability_summary['weak_direction_angle_to_plot_up_deg']['median']:.2f}` 度，H_tt 条件数中位数仅 `{observability_summary['H_tt_condition']['median']:.3f}`。因此不能把“向上漂移”解释成对应方向几何秩亏。
5. **R/t 耦合很强，但当前可部署解耦候选没有修复。** 归一化耦合最大奇异值中位数为 `{observability_summary['coupling_sigma_max']['median']:.6f}`；然而 U1 约等于 U0，非 GT U2 明显恶化，只有 GT 旋转 oracle U3 部分改善。这说明耦合存在，但尚未找到可靠的非 GT 解耦测量。
6. **把 UVD 换成 3D–3D 没有独立收益。** Gate 7 的 F1 首 120 边位置 RMSE 为 `{comparison_3d['modes']['F1']['first_120_trajectory']['position_rmse_m']:.6f}` m，高于 UVD F0 的 `{comparison_3d['modes']['F0']['first_120_trajectory']['position_rmse_m']:.6f}` m；Gate 8 的 V5 也明显劣于 V0。
7. **Gate 8 最好模式仍是 V0 完整 UVD。** Rotation-only 在前 30 边短暂较好，但持续使用会失去平移约束并发散。V3 的非 GT 流量 SNR 门控在 120 条低速边内从未满足，因此等同 V1。方向性降权、U1 解耦 pose factor、3D–3D 均未超过 V0。
8. **下一步应落在观测均值与启动期策略设计，而不是改 IMU 或输出滤波。** 优先审计 flow/disparity/depth 的系统均值，以及设计“有最大持续时间的短 rotation-only 过渡”；不得把本轮失败的 V2/V4/V5 直接设为生产默认值。

## Gate 0：三秒边界

- 初始化开始：`{boundary['initialization_start_timestamp_ns']}` ns。
- 初始化结束：`{boundary['initialization_end_timestamp_ns']}` ns。
- 活动首帧：`{boundary['active_start_frame']}`。
- 首条 IMU knot：`{boundary['first_imu_interval']['first_knot_ns']}` 到 `{boundary['first_imu_interval']['last_knot_ns']}` ns。
- `pre_start_imu_support_used = {boundary['pre_start_imu_support_used']}`。

需要区分两件事：**状态坐标原点已经正确重置**；旧生产 sidecar 的**生成进程**曾从 frame 0 开始。后者经严格冷启动重复对照后，没有表现出超出随机选点基线的历史依赖。

## Gate 1–7 摘要

- Gate 1：同版本连续/冷启动差异不高于 C0/C1 各自重复运行差异，不能据此声称历史污染。
- Gate 4：分类 `{oracle['classification']}`。
- Gate 5：H_tt 并不病态；R/t coupling 高，但 plot-up 与弱方向近乎正交。
- Gate 6：U0 plot-up 均值 `{rt['modes']['U0']['translation_plot_up_error_m']['mean']:.6e}` m/edge；U1 `{rt['modes']['U1']['translation_plot_up_error_m']['mean']:.6e}`；U2 `{rt['modes']['U2']['translation_plot_up_error_m']['mean']:.6e}`；GT-R U3 `{rt['modes']['U3']['translation_plot_up_error_m']['mean']:.6e}`。
- Gate 7：完整 covariance 的 3D–3D 没有稳定优于 UVD。

## Gate 8：同一 N=2 SA-v2 后端

所有模式使用同一标准局部预积分、SA-v2 cross-edge covariance、bias RW、Schur prior、两状态变量和 LM 参数。只有视觉因子策略不同。正式视觉包均来自同版本 C1 冷启动缓存，frame s 之前没有正式视觉 factor。

{chr(10).join(table)}

### 方法学限制

- V2/V3/V4 使用相对位姿压缩来表达方向性、门控或解耦策略；它们不是原始 nonlinear UVD 的逐点等价实现。
- V5 使用每轮当前旋转更新双端 covariance，但 covariance 对状态的导数冻结，匹配 Gate 7 的迭代加权约定。
- V3 阈值不使用 GT；由于真实启动运动极小，去旋转流量始终低于三倍像素标准差，完整 translation 未开启。
- 本轮只覆盖 120 条活动边，结论用于启动阶段，不替代完整序列泛化验证。

## 最终决策

- **排除：** MACVO/VIO 绝对起点未重置、首条 IMU 跨三秒边界、plot-up 方向秩亏。
- **证据不支持直接采用：** H_tt 方向降权、当前 U1 解耦、直接 3D–3D 替换。
- **仍需研究：** flow/disparity/depth 观测均值；短时 rotation-only 后如何用非 GT 且带最大持续时间的规则恢复平移。
- **生产默认保持：** V0 完整 UVD；本轮没有修改生产优化器。

## 产物

- `startup_visual_policy_summary.json`
- `startup_visual_policy_comparison.csv`
- `startup_visual_policy_trajectories.csv`
- `gate8_vio_policy_replay/<V0..V5>/factor_trace.csv`
- `gate8_vio_policy_replay/<V0..V5>/policy_decisions.csv`
- `Scripts/diagnose_startup_visual_upward_drift.py`
- `Scripts/run_startup_macvo_cold_start_ab.py`
- `Scripts/run_startup_visual_policy_replay.py`
"""
    (OUTPUT / "startup_visual_upward_drift_report_cn.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    GATE8.mkdir(parents=True, exist_ok=True)
    prepare_hybrid_cache(force=bool(args.force and args.prepare))
    if args.prepare:
        return 0
    selected = MODES if args.mode == "all" else (args.mode,)
    if args.summarize_only:
        for mode in selected:
            summarize_mode(mode)
    else:
        for mode in selected:
            print(f"[Gate8 {mode}] starting", flush=True)
            run_mode(mode, force=bool(args.force))
            print(f"[Gate8 {mode}] complete", flush=True)
    if all((GATE8 / mode / "summary.json").exists() for mode in MODES):
        print(json.dumps(build_final_summary(), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

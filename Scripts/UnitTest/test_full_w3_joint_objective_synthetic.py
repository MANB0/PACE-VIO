import csv
import inspect
import json
import math
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypose as pp
import pytest
import torch

from Module.Optimization.TwoFramePGO.Graphs import (
    GraphInput,
    LocalWindowGraphInput,
    LocalWindowInertialGraph,
)
import Scripts.synthetic_w3_validation_runner as synthetic_runner
import Scripts.synthetic_w3_validation_data as synthetic_data
from Scripts.synthetic_w3_validation_data import (
    SyntheticSequenceConfig,
    generate_imu_input,
    generate_truth,
    write_sensor_artifacts,
)
from Scripts.synthetic_w3_validation_runner import (
    CaseRunResult,
    ValidationCase,
    _fake_map,
    build_cases,
    query_imu_interval,
    run_validation_case,
)
from Utility.Point import pixel2point_NED


EXPECTED_ACC_BIAS = torch.tensor([0.004, -0.003, 0.002], dtype=torch.float64)
EXPECTED_GYRO_BIAS = torch.tensor([0.0004, -0.0003, 0.002], dtype=torch.float64)

RUNNER_FRAME_DATA_KEYS = {
    "pose",
    "imu_vio_prev_velocity_world",
    "imu_vio_curr_velocity_init_world",
    "imu_vio_velocity_world",
    "imu_vio_prev_acc_bias",
    "imu_vio_prev_gyro_bias",
    "imu_vio_acc_bias",
    "imu_vio_gyro_bias",
}

RUNNER_BIAS_EDGE_FIELDS = {
    "imu_vio_prev_acc_bias",
    "imu_vio_prev_gyro_bias",
    "imu_vio_curr_acc_bias_init",
    "imu_vio_curr_gyro_bias_init",
    "imu_vio_linearized_acc_bias",
    "imu_vio_linearized_gyro_bias",
    "imu_vio_bias_jacobian",
    "imu_vio_bias_rw_cov",
}


@pytest.fixture(scope="module")
def short_runner_inputs():
    config = SyntheticSequenceConfig(duration_s=0.2)
    truth = generate_truth(
        config,
        bias_mode="constant_bias",
        noise_mode="fixed_seed_normal",
    )
    imu_input = generate_imu_input(
        config,
        truth,
        bias_mode="constant_bias",
        noise_mode="fixed_seed_normal",
    )
    visual_input = synthetic_data.generate_visual_input(config, truth, "drifted_visual")
    return config, imu_input, visual_input


def _validation_case(name: str) -> ValidationCase:
    return next(case for case in build_cases(duration_s=0.2) if case.name == name)


@pytest.fixture(scope="module")
def short_all_run(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    calls = []
    events = []
    production_preintegrate = synthetic_runner.preintegrate_imu

    def recording_preintegrate(**kwargs):
        calls.append(
            {
                "time_ns": kwargs["time_ns"].detach().clone(),
                "R0_world": kwargs["R0_world"].tensor().detach().clone(),
                "acc_bias": kwargs["acc_bias"].detach().clone(),
                "gyro_bias": kwargs["gyro_bias"].detach().clone(),
            }
        )
        return production_preintegrate(**kwargs)

    synthetic_runner.preintegrate_imu = recording_preintegrate
    try:
        result = run_validation_case(
            config,
            imu_input,
            visual_input,
            _validation_case("w3_full_all"),
            events.append,
        )
    finally:
        synthetic_runner.preintegrate_imu = production_preintegrate
    return result, calls, events


@pytest.fixture(scope="module")
def short_current_result(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    return run_validation_case(
        config,
        imu_input,
        visual_input,
        _validation_case("w3_full_current"),
    )


@pytest.fixture(scope="module")
def short_locked_result(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    return run_validation_case(
        config,
        imu_input,
        visual_input,
        _validation_case("w3_bias_locked_zero"),
    )


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def _production_visual_covariance_mean(observations) -> float:
    point_variance = observations.data["obs2_covTc"].diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    weights = 1.0 / (point_variance + 1e-6)
    weights = weights / weights.sum()
    return float((weights * point_variance).sum().item())


@pytest.fixture(scope="module")
def drifted_full_trajectory():
    config = SyntheticSequenceConfig(duration_s=10.0)
    truth = generate_truth(config)
    visual = synthetic_data.generate_visual_input(config, truth, "drifted_visual")
    return truth, visual


def test_analytic_truth_includes_exact_endpoints_and_closes():
    short = generate_truth(SyntheticSequenceConfig(duration_s=1.0))
    main = generate_truth(SyntheticSequenceConfig(duration_s=10.0))

    assert short.camera_time_ns.numel() == 31
    assert main.camera_time_ns.numel() == 301
    assert main.imu_time_ns.numel() == 1001
    torch.testing.assert_close(main.position_world[0], main.position_world[-1], atol=1e-10, rtol=0)
    torch.testing.assert_close(main.velocity_world[0], main.velocity_world[-1], atol=1e-10, rtol=0)
    assert float((main.pose_body_to_world[0].Inv() @ main.pose_body_to_world[-1]).Log().norm()) < 1e-9
    torch.testing.assert_close(short.position_world, main.position_world[:31], atol=1e-12, rtol=0)
    assert not torch.allclose(short.position_world[0], short.position_world[-1])


def test_mean_measurement_and_bias_realization_follow_contract():
    config = SyntheticSequenceConfig(duration_s=1.0)
    truth = generate_truth(config, bias_mode="constant_bias", noise_mode="mean_measurement")
    imu = generate_imu_input(config, truth, "constant_bias", "mean_measurement")

    omega = 2.0 * torch.pi / config.trajectory_period_s
    omega_sq = float(omega) ** 2
    expected_acc0 = torch.tensor(
        [
            0.0,
            config.radius_m * omega_sq,
            4.0 * config.depth_amplitude_m * omega_sq - config.gravity_m_s2,
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        truth.true_acc_bias,
        EXPECTED_ACC_BIAS.repeat(truth.imu_time_ns.numel(), 1),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        truth.true_gyro_bias,
        EXPECTED_GYRO_BIAS.repeat(truth.imu_time_ns.numel(), 1),
        atol=0.0,
        rtol=0.0,
    )
    assert imu.measured_acc_body.shape == (truth.imu_time_ns.numel(), 3)
    assert imu.measured_gyro_body.shape == (truth.imu_time_ns.numel(), 3)
    torch.testing.assert_close(imu.measured_acc_body[0], expected_acc0 + EXPECTED_ACC_BIAS, atol=1e-12, rtol=0.0)
    assert truth.bias_mode == "constant_bias"
    assert truth.noise_mode == "mean_measurement"


def test_zero_mean_measurements_are_near_fixed_point_of_production_preintegration():
    from Utility.IMUKinematics import vio_preintegrated_imu_residual

    config = SyntheticSequenceConfig(duration_s=10.0)
    truth = generate_truth(config, bias_mode="zero_bias", noise_mode="mean_measurement")
    imu = generate_imu_input(config, truth, "zero_bias", "mean_measurement")
    zero = torch.zeros(3, dtype=torch.float32)
    residuals = []

    for frame_j in range(1, truth.camera_time_ns.numel()):
        frame_i = frame_j - 1
        time_ns, acc, gyro = query_imu_interval(
            imu,
            int(truth.camera_time_ns[frame_i].item()),
            int(truth.camera_time_ns[frame_j].item()),
        )
        preintegrated = synthetic_runner.preintegrate_imu(
            time_ns=time_ns,
            acc=acc.float(),
            gyro=gyro.float(),
            R0_world=truth.pose_body_to_world[frame_i].rotation(),
            gravity=config.gravity_m_s2,
            sigma_acc=synthetic_data.SIGMA_ACC,
            sigma_gyro=synthetic_data.SIGMA_GYRO,
            sigma_acc_w=synthetic_data.SIGMA_ACC_W,
            sigma_gyro_w=synthetic_data.SIGMA_GYRO_W,
            acc_bias=zero,
            gyro_bias=zero,
        )
        residuals.append(
            vio_preintegrated_imu_residual(
                from_pose=truth.pose_body_to_world[frame_i],
                to_pose=truth.pose_body_to_world[frame_j],
                prev_velocity_world=truth.velocity_world[frame_i],
                curr_velocity_world=truth.velocity_world[frame_j],
                delta_R=preintegrated.delta_R,
                delta_v=preintegrated.delta_v,
                delta_p=preintegrated.delta_p,
                dt_total=preintegrated.dt_total,
                prev_acc_bias=zero,
                prev_gyro_bias=zero,
                linearized_acc_bias=preintegrated.linearized_acc_bias,
                linearized_gyro_bias=preintegrated.linearized_gyro_bias,
                bias_jacobian=preintegrated.bias_jacobian,
            ).double()
        )

    residual = torch.stack(residuals)
    max_position, max_velocity, max_rotation = residual.norm(dim=-1).max(dim=0).values

    maxima = tuple(float(value) for value in (max_position, max_velocity, max_rotation))
    assert float(max_position) < 5e-7, maxima
    assert float(max_velocity) < 2e-5, maxima
    assert float(max_rotation) < 5e-6, maxima


def test_continuous_density_noise_and_walk_are_discretized_at_imu_rate():
    config = SyntheticSequenceConfig(duration_s=10.0)
    noisy_truth = generate_truth(config, bias_mode="zero_bias", noise_mode="fixed_seed_normal")
    noisy_input = generate_imu_input(
        config,
        noisy_truth,
        bias_mode="zero_bias",
        noise_mode="fixed_seed_normal",
    )
    mean_truth = generate_truth(config, bias_mode="zero_bias", noise_mode="mean_measurement")
    mean_input = generate_imu_input(
        config,
        mean_truth,
        bias_mode="zero_bias",
        noise_mode="mean_measurement",
    )

    acc_noise = noisy_input.measured_acc_body - mean_input.measured_acc_body
    gyro_noise = noisy_input.measured_gyro_body - mean_input.measured_gyro_body
    expected_acc_std = synthetic_data.SIGMA_ACC * math.sqrt(config.imu_rate_hz)
    expected_gyro_std = synthetic_data.SIGMA_GYRO * math.sqrt(config.imu_rate_hz)
    assert math.isclose(float(acc_noise.std()), expected_acc_std, rel_tol=0.05)
    assert math.isclose(float(gyro_noise.std()), expected_gyro_std, rel_tol=0.05)

    drifting_truth = generate_truth(
        config,
        bias_mode="drifting_bias",
        noise_mode="mean_measurement",
    )
    acc_steps = drifting_truth.true_acc_bias[1:] - drifting_truth.true_acc_bias[:-1]
    gyro_steps = drifting_truth.true_gyro_bias[1:] - drifting_truth.true_gyro_bias[:-1]
    expected_acc_walk_std = synthetic_data.SIGMA_ACC_W / math.sqrt(config.imu_rate_hz)
    expected_gyro_walk_std = synthetic_data.SIGMA_GYRO_W / math.sqrt(config.imu_rate_hz)
    assert math.isclose(float(acc_steps.std()), expected_acc_walk_std, rel_tol=0.05)
    assert math.isclose(float(gyro_steps.std()), expected_gyro_walk_std, rel_tol=0.05)


def test_generate_imu_input_keeps_truth_bitwise_unchanged_for_matching_mode():
    config = SyntheticSequenceConfig(duration_s=1.0)
    truth = generate_truth(config, bias_mode="drifting_bias", noise_mode="fixed_seed_normal")
    original_acc_bias = truth.true_acc_bias.clone()
    original_gyro_bias = truth.true_gyro_bias.clone()
    original_position = truth.position_world.clone()
    original_angular_velocity = truth.angular_velocity_body.clone()

    imu_a = generate_imu_input(config, truth, "drifting_bias", "fixed_seed_normal")
    imu_b = generate_imu_input(config, truth, "drifting_bias", "fixed_seed_normal")

    assert not hasattr(imu_a, "true_acc_bias")
    assert not hasattr(imu_a, "true_gyro_bias")
    assert torch.equal(imu_a.measured_acc_body, imu_b.measured_acc_body)
    assert torch.equal(imu_a.measured_gyro_body, imu_b.measured_gyro_body)
    assert torch.equal(truth.true_acc_bias, original_acc_bias)
    assert torch.equal(truth.true_gyro_bias, original_gyro_bias)
    assert torch.equal(truth.position_world, original_position)
    assert torch.equal(truth.angular_velocity_body, original_angular_velocity)
    torch.testing.assert_close(truth.true_acc_bias[0], EXPECTED_ACC_BIAS, atol=0.0, rtol=0.0)
    torch.testing.assert_close(truth.true_gyro_bias[0], EXPECTED_GYRO_BIAS, atol=0.0, rtol=0.0)
    assert not torch.equal(truth.true_acc_bias[0], truth.true_acc_bias[-1])
    assert not torch.equal(truth.true_gyro_bias[0], truth.true_gyro_bias[-1])


def test_generate_imu_input_rejects_truth_mode_mismatch():
    config = SyntheticSequenceConfig(duration_s=1.0)
    truth = generate_truth(config, bias_mode="zero_bias", noise_mode="mean_measurement")

    with pytest.raises(ValueError, match="mode"):
        generate_imu_input(config, truth, "constant_bias", "mean_measurement")


def test_trajectory_period_s_cannot_be_overridden_by_callers():
    with pytest.raises(TypeError):
        SyntheticSequenceConfig(duration_s=1.0, trajectory_period_s=5.0)


def test_write_sensor_artifacts_emits_exact_ground_truth_schema_and_manifest(tmp_path: Path):
    config = SyntheticSequenceConfig(duration_s=1.0)
    truth = generate_truth(config, bias_mode="constant_bias", noise_mode="mean_measurement")
    imu = generate_imu_input(config, truth, "constant_bias", "mean_measurement")

    artifacts = write_sensor_artifacts(tmp_path, config, truth, imu)

    ground_truth_path = artifacts["ground_truth_csv"]
    imu_path = artifacts["imu_csv"]
    manifest_path = artifacts["manifest_json"]

    assert ground_truth_path.name == "synthetic_ground_truth.csv"
    assert imu_path.name == "synthetic_imu.csv"
    assert manifest_path.name == "generation_manifest.json"
    assert b"\r\n" not in ground_truth_path.read_bytes()
    assert b"\r\n" not in imu_path.read_bytes()
    assert _read_header(ground_truth_path) == [
        "timestamp_ns",
        "position_world_x",
        "position_world_y",
        "position_world_z",
        "orientation_body_to_world_qx",
        "orientation_body_to_world_qy",
        "orientation_body_to_world_qz",
        "orientation_body_to_world_qw",
        "velocity_world_x",
        "velocity_world_y",
        "velocity_world_z",
        "acceleration_world_x",
        "acceleration_world_y",
        "acceleration_world_z",
        "angular_velocity_body_x",
        "angular_velocity_body_y",
        "angular_velocity_body_z",
        "true_acc_bias_x",
        "true_acc_bias_y",
        "true_acc_bias_z",
        "true_gyro_bias_x",
        "true_gyro_bias_y",
        "true_gyro_bias_z",
    ]
    assert _read_header(imu_path) == [
        "timestamp_ns",
        "measured_acc_body_x",
        "measured_acc_body_y",
        "measured_acc_body_z",
        "measured_gyro_body_x",
        "measured_gyro_body_y",
        "measured_gyro_body_z",
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bias_mode"] == "constant_bias"
    assert manifest["noise_mode"] == "mean_measurement"
    assert manifest["trajectory_period_s"] == 10.0
    assert manifest["camera_rate_hz"] == 30.0
    assert manifest["imu_rate_hz"] == 100.0
    assert manifest["gravity_m_s2"] == 9.8
    assert manifest["frame_convention"] == "world/body NED"
    assert manifest["orientation_profile"] == "roll=0, pitch=0, yaw=2*pi*t/trajectory_period_s"
    assert (
        manifest["motion_sampling_contract"]
        == "analytic_yaw_rate_at_imu_timestamp_v1"
    )
    assert manifest["noise_parameter_semantics"] == "continuous-time density"
    assert "specific_force_body = R_body_to_world^T * (acceleration_world - gravity_world)" in manifest["formulas"]
    assert "white_noise_sample_std = sigma_density*sqrt(imu_rate_hz)" in manifest["formulas"]
    assert "bias_random_walk_step_std = sigma_walk_density/sqrt(imu_rate_hz)" in manifest["formulas"]


def test_visual_edges_are_complete_production_graph_inputs_for_all_31_frames():
    config = SyntheticSequenceConfig(duration_s=1.0)
    truth = generate_truth(config)

    visual = synthetic_data.generate_visual_input(config, truth, "drifted_visual")

    assert visual.pose_initial.shape[0] == truth.camera_time_ns.numel()
    assert len(visual.edges) == truth.camera_time_ns.numel() - 1
    assert not hasattr(visual, "landmark_truth_world")
    assert len(visual.edges) == 30

    for frame_i, edge in enumerate(visual.edges):
        assert type(edge) is GraphInput
        assert edge.from_idx.dtype == torch.long
        assert edge.frame_idx.dtype == torch.long
        assert edge.from_idx.tolist() == [frame_i]
        assert edge.frame_idx.tolist() == [frame_i + 1]
        assert edge.from_pose.shape == (1, 7)
        assert edge.init_motion.shape == (1, 7)
        assert edge.baseline.shape == (1,)
        assert edge.images_intrinsic.shape == (3, 3)
        assert edge.device == "cpu"

        count = edge.observations.index.numel()
        assert count >= 12
        assert edge.num_observations == count
        assert edge.edges_index.shape == (count,)
        assert edge.edges_index.dtype == torch.long
        assert torch.count_nonzero(edge.edges_index).item() == 0
        assert edge.visual_obs_cov_mean == pytest.approx(
            _production_visual_covariance_mean(edge.observations), rel=0.0, abs=1e-15
        )

        for graph_field in fields(GraphInput):
            if graph_field.name.startswith("imu_"):
                assert getattr(edge, graph_field.name) == graph_field.default

        observations = edge.observations.data
        points = edge.points.data
        assert observations["pixel1_uv"].shape == (count, 2)
        assert observations["pixel2_uv"].shape == (count, 2)
        assert observations["pixel1_d"].shape == (count, 1)
        assert observations["pixel2_d"].shape == (count, 1)
        assert observations["pixel1_disp"].shape == (count, 1)
        assert observations["pixel2_disp"].shape == (count, 1)
        assert points["pos_Tw"].shape == (count, 3)
        assert points["cov_Tw"].shape == (count, 3, 3)

        for covariance_key in ("obs1_covTc", "obs2_covTc"):
            covariance = observations[covariance_key]
            assert covariance.shape == (count, 3, 3)
            assert torch.isfinite(covariance).all()
            torch.testing.assert_close(covariance, covariance.transpose(-1, -2), atol=1e-12, rtol=0.0)

        point_source = pixel2point_NED(
            observations["pixel1_uv"].double(),
            observations["pixel1_d"].squeeze(-1).double(),
            edge.images_intrinsic.double(),
        )
        reconstructed = (edge.from_pose[0] * point_source).reshape(-1, 3)
        torch.testing.assert_close(reconstructed.float(), points["pos_Tw"], atol=1e-4, rtol=0.0)

        expected_from_pose = visual.pose_initial[frame_i : frame_i + 1]
        expected_motion = visual.pose_initial[frame_i + 1 : frame_i + 2]
        torch.testing.assert_close(edge.from_pose.tensor(), expected_from_pose.tensor(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(edge.init_motion.tensor(), expected_motion.tensor(), atol=0.0, rtol=0.0)
        if frame_i + 1 < len(visual.edges):
            torch.testing.assert_close(
                edge.init_motion.tensor(),
                visual.edges[frame_i + 1].from_pose.tensor(),
                atol=0.0,
                rtol=0.0,
            )


def test_visual_generation_is_deterministic_and_uses_a_common_frame_zero_origin():
    config = SyntheticSequenceConfig(duration_s=1.0)
    truth = generate_truth(config)

    clean_a = synthetic_data.generate_visual_input(config, truth, "clean_visual")
    clean_b = synthetic_data.generate_visual_input(config, truth, "clean_visual")
    drifted_a = synthetic_data.generate_visual_input(config, truth, "drifted_visual")
    drifted_b = synthetic_data.generate_visual_input(config, truth, "drifted_visual")

    torch.testing.assert_close(clean_a.pose_initial.tensor(), clean_b.pose_initial.tensor(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(drifted_a.pose_initial.tensor(), drifted_b.pose_initial.tensor(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        clean_a.relative_motion_initial.tensor(), clean_b.relative_motion_initial.tensor(), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        drifted_a.relative_motion_initial.tensor(), drifted_b.relative_motion_initial.tensor(), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(clean_a.pose_initial[0].tensor(), drifted_a.pose_initial[0].tensor(), atol=0.0, rtol=0.0)
    assert (
        drifted_a.pose_initial[1].translation() - truth.pose_body_to_world[1].translation()
    ).norm().item() > 0.0

    for edge_a, edge_b in zip(drifted_a.edges, drifted_b.edges, strict=True):
        torch.testing.assert_close(
            edge_a.observations.data["pixel1_uv"],
            edge_b.observations.data["pixel1_uv"],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(edge_a.points.data["pos_Tw"], edge_b.points.data["pos_Tw"], atol=0.0, rtol=0.0)


def test_full_trajectory_visual_generation_covers_last_adjacent_edge(drifted_full_trajectory):
    truth, visual = drifted_full_trajectory

    assert truth.camera_time_ns.numel() == 301
    assert len(visual.edges) == 300
    last = visual.edges[-1]
    assert type(last) is GraphInput
    assert last.from_idx.tolist() == [299]
    assert last.frame_idx.tolist() == [300]
    assert last.num_observations >= 12
    assert last.observations.index.numel() == last.num_observations


def test_full_trajectory_drift_accumulates_from_shared_frame_zero(drifted_full_trajectory):
    truth, visual = drifted_full_trajectory

    assert visual.relative_motion_initial.shape == (300, 7)
    for edge_idx, relative_motion in enumerate(visual.relative_motion_initial):
        recomposed = visual.pose_initial[edge_idx] @ relative_motion
        torch.testing.assert_close(
            recomposed.tensor(), visual.pose_initial[edge_idx + 1].tensor(), atol=1e-12, rtol=0.0
        )

        truth_relative = truth.pose_body_to_world[edge_idx].Inv() @ truth.pose_body_to_world[edge_idx + 1]
        perturbation_norm = (truth_relative.Inv() @ relative_motion).Log().norm().item()
        assert 0.0 < perturbation_norm < 0.02

    frame_zero_error = (truth.pose_body_to_world[0].Inv() @ visual.pose_initial[0]).Log().norm().item()
    early_error = (truth.pose_body_to_world[1].Inv() @ visual.pose_initial[1]).Log().norm().item()
    final_error = (truth.pose_body_to_world[-1].Inv() @ visual.pose_initial[-1]).Log().norm().item()

    assert frame_zero_error < 1e-12
    assert early_error > 0.0
    assert final_error > 0.02
    assert final_error > early_error * 5.0


def test_write_visual_artifact_serializes_complete_graph_inputs_without_pickle_payloads(tmp_path: Path):
    config = SyntheticSequenceConfig(duration_s=1.0)
    truth = generate_truth(config)
    visual = synthetic_data.generate_visual_input(config, truth, "drifted_visual")

    artifact_path = synthetic_data.write_visual_artifact(tmp_path, visual)

    assert artifact_path.name == "synthetic_visual_observations.npz"
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required_keys = {
            "edge_from_idx",
            "edge_frame_idx",
            "edge_from_pose",
            "edge_init_motion",
            "edge_baseline",
            "edge_images_intrinsic",
            "edge_edges_index",
            "edge_device_code",
            "edge_num_observations",
            "edge_visual_obs_cov_mean",
            "relative_motion_initial",
            "edge_point_offset",
            "edge_point_count",
            "edge_pixel1_uv",
            "edge_pixel2_uv",
            "edge_obs1_covTc",
            "edge_obs2_covTc",
            "edge_pos_Tw",
            "edge_cov_Tw",
        }
        assert required_keys.issubset(artifact.files)
        assert artifact["pose_initial"].shape[0] == truth.camera_time_ns.numel()
        assert artifact["edge_from_idx"].tolist() == list(range(30))
        assert artifact["edge_frame_idx"].tolist() == list(range(1, 31))
        assert artifact["edge_from_pose"].shape == (30, 7)
        assert artifact["edge_init_motion"].shape == (30, 7)
        assert artifact["relative_motion_initial"].shape == (30, 7)
        np.testing.assert_array_equal(artifact["edge_init_motion"], artifact["pose_initial"][1:])
        np.testing.assert_array_equal(
            artifact["relative_motion_initial"], visual.relative_motion_initial.tensor().cpu().numpy()
        )
        assert artifact["edge_baseline"].shape == (30, 1)
        assert artifact["edge_images_intrinsic"].shape == (30, 3, 3)
        assert artifact["edge_point_count"].min() >= 12
        assert artifact["edge_num_observations"].tolist() == artifact["edge_point_count"].tolist()
        assert artifact["edge_edges_index"].shape[0] == artifact["edge_point_count"].sum()
        assert np.count_nonzero(artifact["edge_edges_index"]) == 0
        assert "edge_device" not in artifact.files
        assert artifact["edge_device_code"].dtype == np.dtype(np.int8)
        assert artifact["edge_device_code"].tolist() == [0] * 30
        assert not any("truth" in key or "landmark" in key for key in artifact.files)
        assert all(artifact[key].dtype != np.dtype("O") for key in artifact.files)
        assert all(
            np.issubdtype(artifact[key].dtype, np.number)
            or np.issubdtype(artifact[key].dtype, np.bool_)
            for key in artifact.files
        )


def test_runner_public_contract_has_no_truth_and_exact_cases():
    signature = inspect.signature(run_validation_case)
    assert list(signature.parameters) == [
        "config",
        "imu_input",
        "visual_input",
        "case",
        "progress",
    ]
    assert "truth" not in str(signature).lower()
    assert all("truth" not in field.name.lower() for field in fields(ValidationCase))
    assert all("truth" not in field.name.lower() for field in fields(CaseRunResult))

    short_cases = build_cases(duration_s=0.2)
    assert [case.name for case in short_cases] == [
        "visual_initial",
        "w3_bias_locked_zero",
        "w3_full_current",
        "w3_full_all",
        "w3_full_all_zero_mean",
        "w3_full_all_zero_normal",
        "w3_full_all_drifting_bias",
    ]
    assert [case.name for case in build_cases(duration_s=10.0)] == [
        "visual_initial",
        "w3_bias_locked_zero",
        "w3_full_current",
        "w3_full_all",
        "w3_full_all_zero_normal",
        "w3_full_all_drifting_bias",
    ]
    assert short_cases[0] == ValidationCase(
        name="visual_initial",
        visual_condition="drifted_visual",
        imu_noise_mode="fixed_seed_normal",
        bias_mode="constant_bias",
        bias_enabled=False,
        writeback="none",
    )


def test_runner_camera_time_and_imu_queries_use_exact_interpolated_endpoints(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    result = run_validation_case(
        config,
        imu_input,
        visual_input,
        build_cases(duration_s=config.duration_s)[0],
    )
    expected_time_ns = torch.round(
        torch.arange(visual_input.pose_initial.shape[0], dtype=torch.float64)
        * (1e9 / 30.0)
    ).to(torch.int64)

    assert expected_time_ns.numel() == 7
    assert imu_input.time_ns.numel() == 21
    torch.testing.assert_close(result.frame_time_ns, expected_time_ns, atol=0, rtol=0)

    for frame_j in range(1, expected_time_ns.numel()):
        start_ns = int(expected_time_ns[frame_j - 1].item())
        end_ns = int(expected_time_ns[frame_j].item())
        time_ns, acc, gyro = query_imu_interval(imu_input, start_ns, end_ns)

        assert int(time_ns[0].item()) == start_ns
        assert int(time_ns[-1].item()) == end_ns
        assert torch.all(time_ns[1:] > time_ns[:-1])
        assert time_ns.shape[0] == acc.shape[0] == gyro.shape[0]
        assert float((time_ns[-1] - time_ns[0]).item()) == float(end_ns - start_ns)


def test_fake_map_uses_visual_pose_finite_difference_velocity_and_eight_keys(short_runner_inputs):
    config, _, visual_input = short_runner_inputs
    fake_map = _fake_map(config, visual_input)
    frame_data = fake_map.frames.data

    assert set(frame_data) == RUNNER_FRAME_DATA_KEYS
    torch.testing.assert_close(
        frame_data["pose"],
        visual_input.pose_initial.tensor().float(),
        atol=0.0,
        rtol=0.0,
    )

    frame_time_ns = torch.round(
        torch.arange(visual_input.pose_initial.shape[0], dtype=torch.float64)
        * (1e9 / 30.0)
    ).to(torch.int64)
    position = visual_input.pose_initial.translation().reshape(-1, 3).float()
    dt = (frame_time_ns[1:] - frame_time_ns[:-1]).float().unsqueeze(-1) * 1e-9
    interval_velocity = (position[1:] - position[:-1]) / dt
    expected_velocity = torch.cat([interval_velocity[:1], interval_velocity], dim=0)
    expected_previous = expected_velocity.clone()
    expected_previous[1:] = expected_velocity[:-1]

    torch.testing.assert_close(
        frame_data["imu_vio_velocity_world"], expected_velocity, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        frame_data["imu_vio_curr_velocity_init_world"], expected_velocity, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        frame_data["imu_vio_prev_velocity_world"], expected_previous, atol=1e-6, rtol=1e-6
    )
    for key in (
        "imu_vio_prev_acc_bias",
        "imu_vio_prev_gyro_bias",
        "imu_vio_acc_bias",
        "imu_vio_gyro_bias",
    ):
        assert torch.count_nonzero(frame_data[key]) == 0


def test_visual_initial_returns_direct_state_without_windows_or_progress(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    events = []
    visual_case = build_cases(duration_s=config.duration_s)[0]

    result = run_validation_case(config, imu_input, visual_input, visual_case, events.append)

    assert result.case == visual_case
    assert result.num_windows == 0
    assert result.local_ba_window_sizes == []
    assert result.frame_diagnostics.empty
    assert result.max_shift_source_bias_error == 0.0
    assert not result.bias_state_active
    assert events == []
    torch.testing.assert_close(
        result.pose_est.tensor(), visual_input.pose_initial.tensor(), atol=0.0, rtol=0.0
    )
    assert torch.isfinite(result.velocity_est).all()
    assert torch.count_nonzero(result.acc_bias_est) == 0
    assert torch.count_nonzero(result.gyro_bias_est) == 0


def test_runner_rejects_synchronously_truncated_six_frame_visual_input(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    truncated_visual = replace(
        visual_input,
        pose_initial=pp.SE3(visual_input.pose_initial.tensor()[:6].clone()),
        relative_motion_initial=pp.SE3(
            visual_input.relative_motion_initial.tensor()[:5].clone()
        ),
        edges=visual_input.edges[:5],
    )

    with pytest.raises(ValueError, match="expected_frame_count=7.*pose_initial=6"):
        run_validation_case(
            config,
            imu_input,
            truncated_visual,
            _validation_case("visual_initial"),
        )


def test_runner_rejects_misaligned_duration_and_w3_with_fewer_than_three_frames(
    short_runner_inputs,
):
    config, imu_input, visual_input = short_runner_inputs
    misaligned_config = replace(config, duration_s=0.21)
    with pytest.raises(ValueError, match="duration_s.*not aligned"):
        run_validation_case(
            misaligned_config,
            imu_input,
            visual_input,
            _validation_case("visual_initial"),
        )

    two_frame_config = replace(config, duration_s=1.0 / 30.0)
    two_frame_visual = replace(
        visual_input,
        pose_initial=pp.SE3(visual_input.pose_initial.tensor()[:2].clone()),
        relative_motion_initial=pp.SE3(
            visual_input.relative_motion_initial.tensor()[:1].clone()
        ),
        edges=visual_input.edges[:1],
    )
    with pytest.raises(ValueError, match="W3.*at least 3 frames"):
        run_validation_case(
            two_frame_config,
            imu_input,
            two_frame_visual,
            _validation_case("w3_full_all"),
        )


def test_runner_rejects_wrong_visual_edge_count_and_nonadjacent_indices(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    with pytest.raises(ValueError, match=r"len\(edges\)=5.*expected 6"):
        run_validation_case(
            config,
            imu_input,
            replace(visual_input, edges=visual_input.edges[:-1]),
            _validation_case("visual_initial"),
        )


def test_runner_rejects_fractional_visual_edge_indices_before_integer_conversion(
    short_runner_inputs,
):
    config, imu_input, visual_input = short_runner_inputs
    malformed_edges = list(visual_input.edges)
    malformed_edges[0] = replace(
        malformed_edges[0],
        from_idx=torch.tensor([0.5], dtype=torch.float32),
        frame_idx=torch.tensor([1.5], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="from_idx.*integer dtype"):
        run_validation_case(
            config,
            imu_input,
            replace(visual_input, edges=tuple(malformed_edges)),
            _validation_case("visual_initial"),
        )


def test_runner_rejects_invalid_visual_edges_index_dtype_length_and_values(
    short_runner_inputs,
):
    config, imu_input, visual_input = short_runner_inputs
    base_edge = visual_input.edges[0]
    nonzero_edges_index = base_edge.edges_index.clone()
    nonzero_edges_index[0] = 1
    invalid_edges_index = [
        (base_edge.edges_index.float(), "edges_index must use integer dtype"),
        (base_edge.edges_index[:-1], "edges_index length"),
        (nonzero_edges_index, "single-pose edge edges_index must be all zero"),
    ]

    for edges_index, message in invalid_edges_index:
        malformed_edges = list(visual_input.edges)
        malformed_edges[0] = replace(base_edge, edges_index=edges_index)
        with pytest.raises(ValueError, match=message):
            run_validation_case(
                config,
                imu_input,
                replace(visual_input, edges=tuple(malformed_edges)),
                _validation_case("visual_initial"),
            )


def test_query_and_runner_reject_duplicate_imu_timestamps(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    duplicate_time_ns = imu_input.time_ns.clone()
    duplicate_time_ns[5] = duplicate_time_ns[4]
    duplicate_imu = replace(imu_input, time_ns=duplicate_time_ns)

    with pytest.raises(ValueError, match="time_ns must be strictly increasing"):
        query_imu_interval(duplicate_imu, 0, 33_333_333)
    with pytest.raises(ValueError, match="time_ns must be strictly increasing"):
        run_validation_case(
            config,
            duplicate_imu,
            visual_input,
            _validation_case("visual_initial"),
        )


def test_query_rejects_invalid_imu_shape_dtype_length_and_finite_values(short_runner_inputs):
    _, imu_input, _ = short_runner_inputs
    nonfinite_acc = imu_input.measured_acc_body.clone()
    nonfinite_acc[3, 1] = float("nan")
    invalid_inputs = [
        (replace(imu_input, time_ns=imu_input.time_ns.reshape(-1, 1)), "time_ns must be 1D int64"),
        (replace(imu_input, time_ns=imu_input.time_ns.float()), "time_ns must be 1D int64"),
        (
            replace(
                imu_input,
                time_ns=imu_input.time_ns[:1],
                measured_acc_body=imu_input.measured_acc_body[:1],
                measured_gyro_body=imu_input.measured_gyro_body[:1],
            ),
            "at least 2 samples",
        ),
        (
            replace(imu_input, measured_acc_body=imu_input.measured_acc_body[:, :2]),
            "measured_acc_body must have shape",
        ),
        (
            replace(imu_input, measured_gyro_body=imu_input.measured_gyro_body[:-1]),
            "measured_gyro_body must have shape",
        ),
        (
            replace(imu_input, measured_acc_body=imu_input.measured_acc_body.to(torch.int64)),
            "measured_acc_body must be floating point",
        ),
        (replace(imu_input, measured_acc_body=nonfinite_acc), "measured_acc_body must be finite"),
    ]

    for invalid_imu, message in invalid_inputs:
        with pytest.raises(ValueError, match=message):
            query_imu_interval(invalid_imu, 0, 33_333_333)


def test_runner_entry_rejects_imu_shape_mismatch_even_for_visual_initial(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    malformed_imu = replace(
        imu_input,
        measured_acc_body=imu_input.measured_acc_body[:-1],
    )

    with pytest.raises(ValueError, match="measured_acc_body must have shape"):
        run_validation_case(
            config,
            malformed_imu,
            visual_input,
            _validation_case("visual_initial"),
        )

    malformed_edges = list(visual_input.edges)
    malformed_edges[2] = replace(
        malformed_edges[2],
        from_idx=torch.tensor([0], dtype=torch.long),
        frame_idx=torch.tensor([3], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="edge 2.*from_idx=2.*frame_idx=3"):
        run_validation_case(
            config,
            imu_input,
            replace(visual_input, edges=tuple(malformed_edges)),
            _validation_case("visual_initial"),
        )


def test_new_edge_preintegrates_once_from_runtime_rotation_bias_and_exact_interval(
    short_runner_inputs,
    monkeypatch,
):
    config, imu_input, visual_input = short_runner_inputs
    frame_time_ns = torch.round(
        torch.arange(visual_input.pose_initial.shape[0], dtype=torch.float64)
        * (1e9 / 30.0)
    ).to(torch.int64)
    frame_data = _fake_map(config, visual_input).frames.data
    source_index = 1
    target_index = 2
    creation_acc_bias = torch.tensor([0.01, -0.02, 0.03], dtype=torch.float32)
    creation_gyro_bias = torch.tensor([-0.004, 0.005, -0.006], dtype=torch.float32)
    frame_data["imu_vio_acc_bias"][source_index] = creation_acc_bias
    frame_data["imu_vio_gyro_bias"][source_index] = creation_gyro_bias

    production_preintegrate = synthetic_runner.preintegrate_imu
    calls = []

    def recording_preintegrate(**kwargs):
        calls.append(
            {
                key: value.tensor().detach().clone()
                if isinstance(value, pp.LieTensor)
                else value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in kwargs.items()
            }
        )
        return production_preintegrate(**kwargs)

    monkeypatch.setattr(synthetic_runner, "preintegrate_imu", recording_preintegrate)
    stored = synthetic_runner._build_stored_edge(
        config,
        imu_input,
        visual_input.edges[target_index - 1],
        frame_time_ns,
        frame_data,
    )

    assert len(calls) == 1
    call = calls[0]
    assert int(call["time_ns"][0].item()) == int(frame_time_ns[source_index].item())
    assert int(call["time_ns"][-1].item()) == int(frame_time_ns[target_index].item())
    assert torch.all(call["time_ns"][1:] > call["time_ns"][:-1])
    torch.testing.assert_close(
        call["R0_world"],
        pp.SE3(frame_data["pose"][source_index : source_index + 1]).rotation().tensor(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(call["acc_bias"], creation_acc_bias, atol=0.0, rtol=0.0)
    torch.testing.assert_close(call["gyro_bias"], creation_gyro_bias, atol=0.0, rtol=0.0)
    assert call["gravity"] == 9.8
    assert call["sigma_acc"] == synthetic_data.SIGMA_ACC
    assert call["sigma_gyro"] == synthetic_data.SIGMA_GYRO
    assert call["sigma_acc_w"] == synthetic_data.SIGMA_ACC_W
    assert call["sigma_gyro_w"] == synthetic_data.SIGMA_GYRO_W

    edge = stored.graph_input
    torch.testing.assert_close(
        edge.imu_vio_linearized_acc_bias, creation_acc_bias, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        edge.imu_vio_linearized_gyro_bias, creation_gyro_bias, atol=0.0, rtol=0.0
    )
    assert edge.imu_vio_bias_jacobian is not None
    assert edge.imu_vio_bias_rw_cov is not None
    assert math.isclose(
        float(edge.imu_vio_dt.item()),
        float(frame_time_ns[target_index] - frame_time_ns[source_index]) * 1e-9,
        rel_tol=0.0,
        abs_tol=1e-9,
    )

    immutable_snapshot = {
        name: getattr(edge, name).clone()
        for name in (
            "imu_vio_delta_rotvec",
            "imu_vio_delta_v",
            "imu_vio_delta_p",
            "imu_vio_cov",
            "imu_vio_dt",
            "imu_vio_linearized_acc_bias",
            "imu_vio_linearized_gyro_bias",
            "imu_vio_bias_jacobian",
            "imu_vio_bias_rw_cov",
        )
    }

    latest_acc_bias = torch.tensor([0.04, -0.05, 0.06], dtype=torch.float32)
    latest_gyro_bias = torch.tensor([-0.007, 0.008, -0.009], dtype=torch.float32)
    frame_data["pose"][source_index, 0] += 0.125
    frame_data["pose"][target_index, 1] -= 0.25
    frame_data["imu_vio_velocity_world"][source_index] += 0.3
    frame_data["imu_vio_prev_velocity_world"][target_index] = frame_data[
        "imu_vio_velocity_world"
    ][source_index]
    frame_data["imu_vio_curr_velocity_init_world"][target_index] -= 0.4
    frame_data["imu_vio_acc_bias"][source_index] = latest_acc_bias
    frame_data["imu_vio_gyro_bias"][source_index] = latest_gyro_bias
    frame_data["imu_vio_prev_acc_bias"][target_index] = latest_acc_bias
    frame_data["imu_vio_prev_gyro_bias"][target_index] = latest_gyro_bias
    frame_data["imu_vio_acc_bias"][target_index] = latest_acc_bias * 1.25
    frame_data["imu_vio_gyro_bias"][target_index] = latest_gyro_bias * 1.25

    rebased = synthetic_runner._rebase_window_edge(stored, frame_data, bias_enabled=True)
    _ = synthetic_runner._rebase_window_edge(stored, frame_data, bias_enabled=True)
    assert len(calls) == 1
    torch.testing.assert_close(
        rebased.from_pose.tensor(),
        frame_data["pose"][source_index : source_index + 1],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rebased.init_motion.tensor(),
        frame_data["pose"][target_index : target_index + 1],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rebased.imu_vio_prev_velocity_world,
        frame_data["imu_vio_prev_velocity_world"][target_index],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rebased.imu_vio_curr_velocity_init_world,
        frame_data["imu_vio_curr_velocity_init_world"][target_index],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(rebased.imu_vio_prev_acc_bias, latest_acc_bias, atol=0.0, rtol=0.0)
    torch.testing.assert_close(rebased.imu_vio_prev_gyro_bias, latest_gyro_bias, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        rebased.imu_vio_curr_acc_bias_init,
        frame_data["imu_vio_acc_bias"][target_index],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rebased.imu_vio_curr_gyro_bias_init,
        frame_data["imu_vio_gyro_bias"][target_index],
        atol=0.0,
        rtol=0.0,
    )
    for name, expected in immutable_snapshot.items():
        torch.testing.assert_close(getattr(rebased, name), expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(getattr(stored.graph_input, name), expected, atol=0.0, rtol=0.0)
    assert not torch.equal(rebased.imu_vio_linearized_acc_bias, latest_acc_bias)
    assert not torch.equal(rebased.imu_vio_linearized_gyro_bias, latest_gyro_bias)


def test_bias_locked_rebase_clears_exact_bias_fields_and_keeps_main_vio(short_runner_inputs):
    config, imu_input, visual_input = short_runner_inputs
    frame_time_ns = torch.round(
        torch.arange(visual_input.pose_initial.shape[0], dtype=torch.float64)
        * (1e9 / 30.0)
    ).to(torch.int64)
    frame_data = _fake_map(config, visual_input).frames.data
    stored = synthetic_runner._build_stored_edge(
        config,
        imu_input,
        visual_input.edges[0],
        frame_time_ns,
        frame_data,
    )

    enabled = synthetic_runner._rebase_window_edge(stored, frame_data, bias_enabled=True)
    locked = synthetic_runner._rebase_window_edge(stored, frame_data, bias_enabled=False)

    assert {name for name in RUNNER_BIAS_EDGE_FIELDS if getattr(locked, name) is None} == RUNNER_BIAS_EDGE_FIELDS
    assert all(getattr(enabled, name) is not None for name in RUNNER_BIAS_EDGE_FIELDS)
    assert LocalWindowInertialGraph._edge_has_vio(locked)
    assert not LocalWindowInertialGraph._edge_has_bias(locked)
    assert locked.observations is enabled.observations
    assert locked.points is enabled.points
    for name in (
        "imu_vio_delta_rotvec",
        "imu_vio_delta_v",
        "imu_vio_delta_p",
        "imu_vio_cov",
        "imu_vio_dt",
        "imu_vio_sensor_T_imu",
    ):
        torch.testing.assert_close(getattr(locked, name), getattr(enabled, name), atol=0.0, rtol=0.0)


def test_bias_rw_energy_uses_full_offdiagonal_covariance_and_global_edge_endpoints(
    short_runner_inputs,
):
    _, _, visual_input = short_runner_inputs
    mixing = torch.eye(6, dtype=torch.float64)
    mixing[1, 0] = 0.35
    mixing[3, 0] = -0.20
    mixing[4, 2] = 0.25
    mixing[5, 1] = -0.15
    covariance = mixing @ mixing.mT + torch.eye(6, dtype=torch.float64) * 0.4
    assert torch.count_nonzero(covariance - torch.diag(covariance.diagonal())) > 0

    edge = replace(
        visual_input.edges[1],
        imu_vio_prev_acc_bias=torch.zeros(3),
        imu_vio_prev_gyro_bias=torch.zeros(3),
        imu_vio_curr_acc_bias_init=torch.zeros(3),
        imu_vio_curr_gyro_bias_init=torch.zeros(3),
        imu_vio_linearized_acc_bias=torch.zeros(3),
        imu_vio_linearized_gyro_bias=torch.zeros(3),
        imu_vio_bias_jacobian=torch.zeros((9, 6)),
        imu_vio_bias_rw_cov=covariance,
    )
    frame_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    acc_bias = torch.tensor(
        [[100.0, -100.0, 50.0], [0.10, -0.20, 0.30], [0.40, 0.05, -0.50]],
        dtype=torch.float64,
    )
    gyro_bias = torch.tensor(
        [[-80.0, 90.0, -70.0], [0.01, -0.02, 0.03], [-0.04, 0.06, 0.08]],
        dtype=torch.float64,
    )
    residual = torch.cat(
        [
            acc_bias[2] - acc_bias[1],
            gyro_bias[2] - gyro_bias[1],
        ]
    )
    expected = float(residual @ torch.linalg.pinv(covariance) @ residual)
    diagonal_only = float(
        residual
        @ torch.linalg.pinv(torch.diag(covariance.diagonal()))
        @ residual
    )

    actual = synthetic_runner._bias_rw_energy(
        frame_indices,
        [edge],
        acc_bias,
        gyro_bias,
    )

    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert not math.isclose(actual, diagonal_only, rel_tol=1e-6, abs_tol=1e-6)


def test_window_bias_rw_energies_use_distinct_initial_graph_and_final_output_states(
    short_runner_inputs,
):
    _, _, visual_input = short_runner_inputs
    mixing = torch.eye(6, dtype=torch.float64)
    mixing[1, 0] = -0.30
    mixing[3, 2] = 0.22
    mixing[4, 0] = 0.18
    mixing[5, 1] = -0.27
    covariance = mixing @ mixing.mT + torch.eye(6, dtype=torch.float64) * 0.25
    edge = replace(
        visual_input.edges[1],
        imu_vio_prev_acc_bias=torch.zeros(3),
        imu_vio_prev_gyro_bias=torch.zeros(3),
        imu_vio_curr_acc_bias_init=torch.zeros(3),
        imu_vio_curr_gyro_bias_init=torch.zeros(3),
        imu_vio_linearized_acc_bias=torch.zeros(3),
        imu_vio_linearized_gyro_bias=torch.zeros(3),
        imu_vio_bias_jacobian=torch.zeros((9, 6)),
        imu_vio_bias_rw_cov=covariance,
    )
    frame_indices = torch.tensor([1, 2], dtype=torch.long)
    window_input = LocalWindowGraphInput(
        frame_indices=frame_indices,
        frame_poses=visual_input.pose_initial.tensor()[1:3].float(),
        edges=[edge],
        fixed_first_frame=True,
        writeback="all_optimized",
        device="cpu",
    )
    initial_acc = torch.tensor(
        [[0.10, -0.05, 0.20], [0.35, 0.15, -0.10]], dtype=torch.float64
    )
    initial_gyro = torch.tensor(
        [[0.01, -0.02, 0.03], [0.08, 0.04, -0.06]], dtype=torch.float64
    )
    final_acc = torch.tensor(
        [[-0.20, 0.30, 0.05], [0.45, -0.25, 0.15]], dtype=torch.float64
    )
    final_gyro = torch.tensor(
        [[-0.04, 0.07, 0.02], [0.11, -0.08, 0.09]], dtype=torch.float64
    )
    diagnostic_graph = SimpleNamespace(
        _all_acc_bias=lambda: initial_acc,
        _all_gyro_bias=lambda: initial_gyro,
    )
    output = SimpleNamespace(
        window_acc_bias=final_acc,
        window_gyro_bias=final_gyro,
    )
    initial_residual = torch.cat(
        [initial_acc[1] - initial_acc[0], initial_gyro[1] - initial_gyro[0]]
    )
    final_residual = torch.cat(
        [final_acc[1] - final_acc[0], final_gyro[1] - final_gyro[0]]
    )
    information = torch.linalg.pinv(covariance)
    expected_initial = float(initial_residual @ information @ initial_residual)
    expected_final = float(final_residual @ information @ final_residual)
    assert not math.isclose(expected_initial, expected_final, rel_tol=1e-6, abs_tol=1e-6)

    actual_initial, actual_final = synthetic_runner._window_bias_rw_energies(
        window_input,
        diagnostic_graph,
        output,
    )

    assert math.isclose(actual_initial, expected_initial, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(actual_final, expected_final, rel_tol=1e-12, abs_tol=1e-12)
    diagnostic_source = inspect.getsource(synthetic_runner._window_diagnostic_row)
    assert diagnostic_source.count("_window_bias_rw_energies(") == 1
    assert "_bias_rw_energy(" not in diagnostic_source


def test_runner_executes_five_production_w3_windows_once_per_edge_and_reports_progress(short_all_run):
    result, preintegrate_calls, events = short_all_run

    assert result.num_windows == 5
    assert result.local_ba_window_sizes == [3] * 5
    assert len(preintegrate_calls) == 6
    assert torch.isfinite(result.pose_est.tensor()).all()
    assert torch.isfinite(result.velocity_est).all()
    assert torch.isfinite(result.acc_bias_est).all()
    assert torch.isfinite(result.gyro_bias_est).all()

    assert len(events) == 5
    assert [event["completed_windows"] for event in events] == [1, 2, 3, 4, 5]
    assert [event["total_windows"] for event in events] == [5] * 5
    assert [event["frame_index"] for event in events] == [2, 3, 4, 5, 6]
    assert [event["case"] for event in events] == ["w3_full_all"] * 5
    assert all(set(event) >= {"case", "completed_windows", "total_windows", "frame_index", "elapsed_s"} for event in events)
    assert all(isinstance(event["elapsed_s"], float) and event["elapsed_s"] >= 0.0 for event in events)
    assert all(events[index]["elapsed_s"] <= events[index + 1]["elapsed_s"] for index in range(4))


def test_runner_records_complete_finite_production_diagnostics(short_all_run):
    result, _, _ = short_all_run
    trace = result.frame_diagnostics
    required_columns = {
        "case",
        "window_start",
        "window_target",
        "graph_class",
        "context_device",
        "bias_state_active",
        "vio_factor_active",
        "imu_residual_rows",
        "initial_visual_energy",
        "final_visual_energy",
        "initial_p_energy",
        "final_p_energy",
        "initial_v_energy",
        "final_v_energy",
        "initial_R_energy",
        "final_R_energy",
        "initial_main_imu_energy",
        "final_main_imu_energy",
        "initial_bias_rw_energy",
        "final_bias_rw_energy",
        "pose_translation_update_norm",
        "pose_rotation_update_norm",
        "velocity_update_norm",
        "acc_bias_update_norm",
        "gyro_bias_update_norm",
        "source_main_imu_bias_jacobian_rank",
        "terminal_main_imu_bias_jacobian_rank",
        "finite",
        "optimizer_completed",
        "optimizer_final_loss_finite",
        "optimizer_convergence_status",
        "leading_interval_start_ns",
        "leading_interval_end_ns",
        "leading_imu_vio_dt_s",
        "trailing_interval_start_ns",
        "trailing_interval_end_ns",
        "trailing_imu_vio_dt_s",
        "all_edges_rebase_fields_match_runtime",
        "local_ba_graph_build_s",
        "local_ba_lm_s",
        "local_ba_refine_s",
        "local_ba_optimize_total_s",
        "influence_sampled",
    }
    assert required_columns.issubset(trace.columns)
    assert len(trace) == 5
    assert trace["graph_class"].eq("LocalWindowInertialGraph").all()
    assert trace["context_device"].eq("cpu").all()
    assert trace["bias_state_active"].all()
    assert trace["vio_factor_active"].all()
    assert trace["imu_residual_rows"].eq(10).all()
    assert trace["finite"].all()
    assert trace["optimizer_completed"].all()
    assert trace["optimizer_final_loss_finite"].all()
    assert trace["optimizer_convergence_status"].eq(
        "unknown_not_exposed_by_production"
    ).all()
    assert "optimizer_converged" not in trace.columns
    assert trace["all_edges_rebase_fields_match_runtime"].all()
    assert trace["source_main_imu_bias_jacobian_rank"].gt(0).all()
    assert trace["terminal_main_imu_bias_jacobian_rank"].eq(0).all()
    assert trace["influence_sampled"].eq(0).all()

    numeric_columns = [
        "initial_visual_energy",
        "final_visual_energy",
        "initial_p_energy",
        "final_p_energy",
        "initial_v_energy",
        "final_v_energy",
        "initial_R_energy",
        "final_R_energy",
        "initial_main_imu_energy",
        "final_main_imu_energy",
        "initial_bias_rw_energy",
        "final_bias_rw_energy",
        "pose_translation_update_norm",
        "pose_rotation_update_norm",
        "velocity_update_norm",
        "acc_bias_update_norm",
        "gyro_bias_update_norm",
        "local_ba_graph_build_s",
        "local_ba_lm_s",
        "local_ba_refine_s",
        "local_ba_optimize_total_s",
    ]
    assert np.isfinite(trace[numeric_columns].to_numpy(dtype=np.float64)).all()
    assert trace[[column for column in numeric_columns if column.startswith("local_ba_")]].ge(0.0).all().all()

    for row in trace.itertuples(index=False):
        assert row.leading_interval_end_ns > row.leading_interval_start_ns
        assert row.trailing_interval_end_ns > row.trailing_interval_start_ns
        assert math.isclose(
            row.leading_imu_vio_dt_s,
            (row.leading_interval_end_ns - row.leading_interval_start_ns) * 1e-9,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            row.trailing_imu_vio_dt_s,
            (row.trailing_interval_end_ns - row.trailing_interval_start_ns) * 1e-9,
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_shifted_window_rebases_all_runtime_fields_without_reintegrating(short_all_run):
    result, _, _ = short_all_run
    shifted = result.frame_diagnostics.loc[
        result.frame_diagnostics["window_target"] == 3
    ].iloc[0]

    assert shifted["edge_12_source_state_matches_runtime"]
    for prefix in ("leading", "trailing"):
        for field in (
            "from_pose",
            "init_motion",
            "prev_velocity",
            "curr_velocity",
            "prev_acc_bias",
            "prev_gyro_bias",
            "curr_acc_bias",
            "curr_gyro_bias",
        ):
            assert shifted[f"{prefix}_{field}_matches_runtime"]
        assert shifted[f"{prefix}_preintegration_fields_immutable"]
    assert shifted["trailing_linearization_bias_matches_creation_source"]
    assert shifted["trailing_preintegration_rotation_matches_creation_source"]

    linearized = torch.tensor(shifted["leading_linearized_bias"], dtype=torch.float32)
    runtime = torch.tensor(shifted["leading_runtime_source_bias"], dtype=torch.float32)
    expected_difference = float((linearized - runtime).norm().item())
    assert math.isclose(
        shifted["leading_linearization_runtime_bias_difference_norm"],
        expected_difference,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_all_writeback_preserves_complete_middle_state_but_current_bias_does_not(
    short_all_run,
    short_current_result,
):
    all_result, _, _ = short_all_run

    assert short_current_result.max_shift_source_bias_error > 1e-8
    assert all_result.max_shift_source_bias_error < 1e-8
    assert all_result.max_shift_source_pose_translation_error_m < 1e-8
    assert all_result.max_shift_source_pose_rotation_error_rad < 1e-8
    assert all_result.max_shift_source_velocity_error_m_s < 1e-8


def test_bias_locked_case_reports_runner_bias_inactive_with_main_vio_active(short_locked_result):
    trace = short_locked_result.frame_diagnostics

    assert not short_locked_result.bias_state_active
    assert not trace["bias_state_active"].any()
    assert trace["vio_factor_active"].all()
    assert trace["imu_residual_rows"].eq(6).all()
    assert trace["bias_fields_cleared_per_edge"].eq(8).all()
    assert trace["source_main_imu_bias_jacobian_rank"].eq(0).all()
    assert trace["terminal_main_imu_bias_jacobian_rank"].eq(0).all()
    assert trace["acc_bias_update_norm"].eq(0.0).all()
    assert trace["gyro_bias_update_norm"].eq(0.0).all()
    assert trace["bias_parameter_structurally_present"].all()
    assert trace["bias_parameter_limitation"].eq(
        "parameters_exist_even_when_bias_factor_is_disabled"
    ).all()
    assert torch.count_nonzero(short_locked_result.acc_bias_est) == 0
    assert torch.count_nonzero(short_locked_result.gyro_bias_est) == 0


def _task4_metric_fixture() -> tuple[CaseRunResult, synthetic_data.EvaluationTruth, SimpleNamespace]:
    config = SyntheticSequenceConfig(duration_s=0.2)
    original_truth = generate_truth(
        config,
        bias_mode="constant_bias",
        noise_mode="mean_measurement",
    )
    imu_seconds = original_truth.imu_time_ns.double().unsqueeze(-1) * 1e-9
    true_acc_bias = imu_seconds * torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    true_gyro_bias = imu_seconds * torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)

    frame_time_ns = original_truth.camera_time_ns.clone()
    frame_time_ns[-1] = 195_000_000
    truth = replace(
        original_truth,
        camera_time_ns=frame_time_ns,
        true_acc_bias=true_acc_bias,
        true_gyro_bias=true_gyro_bias,
    )
    pose_tensor = truth.pose_body_to_world.tensor().clone()
    pose_tensor[:, 0] += 1.0
    acc_bias_est = torch.zeros((7, 3), dtype=torch.float64)
    gyro_bias_est = torch.zeros((7, 3), dtype=torch.float64)
    acc_bias_est[-1] = torch.tensor([0.195, 0.390, 0.585], dtype=torch.float64)
    gyro_bias_est[-1] = torch.tensor([0.0195, 0.0390, 0.0585], dtype=torch.float64)
    case = ValidationCase(
        name="w3_full_all",
        visual_condition="clean_visual",
        imu_noise_mode="mean_measurement",
        bias_mode="constant_bias",
        bias_enabled=True,
        writeback="all_optimized",
    )
    result = CaseRunResult(
        case=case,
        frame_time_ns=frame_time_ns,
        pose_est=pp.SE3(pose_tensor),
        velocity_est=truth.velocity_world.clone(),
        acc_bias_est=acc_bias_est,
        gyro_bias_est=gyro_bias_est,
        frame_diagnostics=pd.DataFrame(),
        num_windows=5,
        local_ba_window_sizes=[3] * 5,
        max_shift_source_bias_error=0.0,
        max_shift_source_pose_translation_error_m=0.0,
        max_shift_source_pose_rotation_error_rad=0.0,
        max_shift_source_velocity_error_m_s=0.0,
        bias_state_active=True,
    )
    visual = SimpleNamespace(pose_initial=pp.SE3(truth.pose_body_to_world.tensor().clone()))
    return result, truth, visual


class TestTask4Metrics:
    def test_nonfinite_window_energy_is_not_hidden_by_aggregate(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import _diagnostic_aggregates

        result, _, _ = _task4_metric_fixture()
        diagnostics = pd.DataFrame(
            {
                "finite": [True, True],
                "optimizer_completed": [True, True],
                "optimizer_final_loss_finite": [True, True],
                "all_edges_rebase_fields_match_runtime": [True, True],
                "initial_visual_energy": [1.0, np.nan],
            }
        )

        aggregates = _diagnostic_aggregates(replace(result, frame_diagnostics=diagnostics))

        assert not aggregates["diagnostics_finite"]
        assert math.isnan(aggregates["initial_visual_energy_total"])
        assert math.isnan(aggregates["initial_visual_energy_mean"])

    def test_visual_control_does_not_claim_optimizer_execution(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import evaluate_case

        result, truth, visual = _task4_metric_fixture()
        visual_result = replace(
            result,
            case=ValidationCase(
                name="visual_initial",
                visual_condition="drifted_visual",
                imu_noise_mode="fixed_seed_normal",
                bias_mode="constant_bias",
                bias_enabled=False,
                writeback="none",
            ),
            frame_diagnostics=pd.DataFrame(),
            num_windows=0,
            local_ba_window_sizes=[],
            bias_state_active=False,
        )

        metrics, _ = evaluate_case(visual_result, truth, visual)

        assert not metrics["diagnostics_available"]
        assert not metrics["optimizer_completed"]
        assert not metrics["optimizer_final_loss_finite"]
        assert metrics["optimizer_convergence_status"] == "not_applicable_visual_initial"
        assert metrics["finite"]

    def test_metrics_use_strict_common_origin_without_alignment(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import evaluate_case

        result, truth, visual = _task4_metric_fixture()
        metrics, frames = evaluate_case(result, truth, visual)

        assert math.isclose(metrics["translation_rmse_m"], 1.0, abs_tol=1e-12)
        assert metrics["alignment"] == "none_common_origin"
        assert metrics["evaluation_start_frame"] == 6
        assert frames["in_evaluation"].sum() == 1
        assert math.isclose(frames.loc[6, "translation_error_m"], 1.0, abs_tol=1e-12)
        assert math.isclose(metrics["visual_translation_rmse_m"], 0.0, abs_tol=1e-12)

    def test_bias_truth_is_interpolated_and_persistence_uses_previous_estimate(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import evaluate_case

        result, truth, visual = _task4_metric_fixture()
        metrics, frames = evaluate_case(result, truth, visual)
        evaluated = frames.loc[6]
        expected_acc = torch.tensor([0.195, 0.390, 0.585], dtype=torch.float64)
        expected_gyro = torch.tensor([0.0195, 0.0390, 0.0585], dtype=torch.float64)
        expected_six = float(torch.cat((expected_acc, expected_gyro)).norm())

        assert math.isclose(evaluated["true_acc_bias_x"], expected_acc[0].item(), abs_tol=1e-12)
        assert math.isclose(evaluated["true_gyro_bias_z"], expected_gyro[2].item(), abs_tol=1e-12)
        assert math.isclose(metrics["six_axis_bias_rmse"], 0.0, abs_tol=1e-12)
        assert math.isclose(
            evaluated["previous_estimate_persistence_six_axis_bias_error"],
            expected_six,
            abs_tol=1e-12,
        )
        assert math.isclose(
            metrics["previous_estimate_persistence_six_axis_bias_rmse"],
            expected_six,
            abs_tol=1e-12,
        )

    def test_metrics_include_geodesic_velocity_baselines_and_gate_quantities(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import evaluate_case

        result, truth, visual = _task4_metric_fixture()
        angle_rad = math.radians(10.0)
        rotation_delta = pp.so3(
            torch.tensor([[0.0, 0.0, angle_rad]], dtype=torch.float64)
        ).Exp()
        base_rotation = pp.SO3(result.pose_est.rotation().tensor().double())
        estimated_rotation = base_rotation @ pp.SO3(
            rotation_delta.tensor().repeat(result.frame_time_ns.numel(), 1)
        )
        pose_tensor = result.pose_est.tensor().double().clone()
        pose_tensor[:, 3:] = estimated_rotation.tensor()
        velocity_est = result.velocity_est.double() + torch.tensor(
            [0.0, 2.0, 0.0], dtype=torch.float64
        )
        diagnostics = pd.DataFrame(
            {
                "finite": [True],
                "optimizer_completed": [True],
                "optimizer_final_loss_finite": [True],
                "optimizer_convergence_status": ["unknown_not_exposed_by_production"],
                "all_edges_rebase_fields_match_runtime": [True],
                "initial_visual_energy": [4.0],
                "final_visual_energy": [2.0],
                "initial_p_energy": [3.0],
                "final_p_energy": [1.5],
                "initial_v_energy": [2.0],
                "final_v_energy": [1.0],
                "initial_R_energy": [1.0],
                "final_R_energy": [0.5],
                "initial_main_imu_energy": [8.0],
                "final_main_imu_energy": [4.0],
                "initial_bias_rw_energy": [0.2],
                "final_bias_rw_energy": [0.1],
                "pose_translation_update_norm": [0.4],
                "pose_rotation_update_norm": [0.3],
                "velocity_update_norm": [0.2],
                "acc_bias_update_norm": [0.1],
                "gyro_bias_update_norm": [0.05],
                "source_main_imu_bias_jacobian_rank": [6],
                "terminal_main_imu_bias_jacobian_rank": [0],
            }
        )
        result = replace(
            result,
            pose_est=pp.SE3(pose_tensor),
            velocity_est=velocity_est,
            frame_diagnostics=diagnostics,
        )

        metrics, frames = evaluate_case(result, truth, visual)
        evaluated = frames.loc[6]

        assert math.isclose(metrics["rotation_rmse_deg"], 10.0, abs_tol=1e-9)
        assert math.isclose(metrics["velocity_rmse_m_s"], 2.0, abs_tol=1e-12)
        assert math.isclose(evaluated["rotation_error_deg"], 10.0, abs_tol=1e-9)
        assert math.isclose(evaluated["velocity_error_m_s"], 2.0, abs_tol=1e-12)
        assert math.isclose(metrics["acc_bias_rmse_m_s2"], 0.0, abs_tol=1e-12)
        assert math.isclose(metrics["gyro_bias_rmse_rad_s"], 0.0, abs_tol=1e-12)
        assert metrics["zero_bias_baseline_six_axis_bias_rmse"] > 0.0
        assert metrics["visual_six_axis_bias_rmse"] == metrics[
            "zero_bias_baseline_six_axis_bias_rmse"
        ]
        expected_final_six = float(
            torch.tensor(
                [0.195, 0.390, 0.585, 0.0195, 0.0390, 0.0585],
                dtype=torch.float64,
            ).norm()
        )
        assert math.isclose(
            metrics["final_six_axis_bias_norm"], expected_final_six, rel_tol=1e-12
        )
        assert metrics["max_acc_bias_norm_m_s2"] >= metrics["final_acc_bias_norm_m_s2"]
        assert math.isclose(
            metrics["final10_six_axis_bias_norm_slope_per_s"], 0.0, abs_tol=1e-12
        )
        assert metrics["initial_visual_energy_total"] == 4.0
        assert metrics["final_main_imu_energy_total"] == 4.0
        assert metrics["pose_translation_update_norm_max"] == 0.4
        assert metrics["source_main_imu_bias_jacobian_rank_min"] == 6
        assert metrics["finite"]
        assert metrics["optimizer_completed"]
        assert metrics["optimizer_final_loss_finite"]
        assert metrics["optimizer_convergence_status"] == "unknown_not_exposed_by_production"
        assert "optimizer_converged" not in metrics
        assert frames["frame_state_finite"].all()


def _task4_ready_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    common = {
        "finite": True,
        "state_propagation_ok": True,
        "diagnostics_available": True,
        "optimizer_completed": True,
        "optimizer_final_loss_finite": True,
        "all_edges_rebase_fields_match_runtime": True,
        "window_contract_ok": True,
        "max_shift_source_bias_error": 0.0,
        "max_shift_source_pose_translation_error_m": 0.0,
        "max_shift_source_pose_rotation_error_rad": 0.0,
        "max_shift_source_velocity_error_m_s": 0.0,
        "translation_rmse_m": 1.0,
        "rotation_rmse_deg": 1.0,
        "velocity_rmse_m_s": 1.0,
        "visual_translation_rmse_m": 1.0,
        "visual_rotation_rmse_deg": 1.0,
        "visual_velocity_rmse_m_s": 1.0,
        "acc_bias_rmse_m_s2": 1.0,
        "gyro_bias_rmse_rad_s": 1.0,
        "six_axis_bias_rmse": 1.0,
        "zero_bias_baseline_acc_bias_rmse_m_s2": 1.0,
        "zero_bias_baseline_gyro_bias_rmse_rad_s": 1.0,
        "zero_bias_baseline_six_axis_bias_rmse": 1.0,
        "previous_estimate_persistence_acc_bias_rmse_m_s2": 1.0,
        "previous_estimate_persistence_gyro_bias_rmse_rad_s": 1.0,
        "previous_estimate_persistence_six_axis_bias_rmse": 1.0,
        "max_acc_bias_norm_m_s2": 0.1,
        "max_gyro_bias_norm_rad_s": 0.1,
        "final_acc_bias_norm_m_s2": 0.1,
        "final_gyro_bias_norm_rad_s": 0.1,
        "final10_acc_bias_norm_slope_m_s3": 0.0,
        "final10_gyro_bias_norm_slope_rad_s2": 0.0,
        "final_six_axis_bias_norm": 0.1,
        "final10_six_axis_bias_norm_slope_per_s": 0.0,
    }
    rows = []
    specifications = (
        ("visual_initial", "drifted_visual", "fixed_seed_normal", "constant_bias", False, "none", 301, 30, 271),
        ("w3_bias_locked_zero", "drifted_visual", "fixed_seed_normal", "constant_bias", False, "all_optimized", 301, 30, 271),
        ("w3_full_current", "drifted_visual", "fixed_seed_normal", "constant_bias", True, "current", 301, 30, 271),
        ("w3_full_all", "drifted_visual", "fixed_seed_normal", "constant_bias", True, "all_optimized", 301, 30, 271),
        ("w3_full_all_zero_mean", "clean_visual", "mean_measurement", "zero_bias", True, "all_optimized", 31, 6, 25),
        ("w3_full_all_zero_normal", "drifted_visual", "fixed_seed_normal", "zero_bias", True, "all_optimized", 301, 30, 271),
        ("w3_full_all_drifting_bias", "drifted_visual", "fixed_seed_normal", "drifting_bias", True, "all_optimized", 301, 30, 271),
    )
    for (
        case_name,
        visual_condition,
        imu_noise_mode,
        bias_mode,
        bias_enabled,
        writeback,
        frame_count,
        evaluation_start,
        evaluation_count,
    ) in specifications:
        row = dict(
            common,
            case=case_name,
            visual_condition=visual_condition,
            imu_noise_mode=imu_noise_mode,
            bias_mode=bias_mode,
            bias_enabled=bias_enabled,
            writeback=writeback,
            frame_count=frame_count,
            evaluation_start_frame=evaluation_start,
            evaluation_frame_count=evaluation_count,
        )
        rows.append(row)
    cases = pd.DataFrame(rows).set_index("case", drop=False)
    cases.loc[
        cases["case"] == "visual_initial",
        ["diagnostics_available", "optimizer_completed", "optimizer_final_loss_finite"],
    ] = False
    cases.loc[
        "w3_full_all",
        ["acc_bias_rmse_m_s2", "gyro_bias_rmse_rad_s", "six_axis_bias_rmse"],
    ] = [0.4, 0.4, 0.4]
    cases.loc[
        "w3_full_all",
        [
            "previous_estimate_persistence_acc_bias_rmse_m_s2",
            "previous_estimate_persistence_gyro_bias_rmse_rad_s",
            "previous_estimate_persistence_six_axis_bias_rmse",
        ],
    ] = [0.8, 0.8, 0.8]
    cases.loc["w3_full_all", "translation_rmse_m"] = 0.94
    cases.loc[
        "w3_full_all_drifting_bias",
        ["acc_bias_rmse_m_s2", "gyro_bias_rmse_rad_s", "six_axis_bias_rmse"],
    ] = [0.8, 0.8, 0.8]
    cases.loc[
        "w3_full_all_zero_mean",
        ["translation_rmse_m", "rotation_rmse_deg", "velocity_rmse_m_s"],
    ] = [0.0005, 0.005, 0.0005]
    cases.loc[
        "w3_full_all_zero_mean",
        ["visual_translation_rmse_m", "visual_rotation_rmse_deg", "visual_velocity_rmse_m_s"],
    ] = [0.0005, 0.005, 0.0005]
    cases.loc[
        "w3_full_all_zero_mean",
        ["max_acc_bias_norm_m_s2", "max_gyro_bias_norm_rad_s"],
    ] = [9e-7, 9e-7]
    cases.loc[
        "w3_full_all_zero_normal",
        [
            "final_acc_bias_norm_m_s2",
            "final_gyro_bias_norm_rad_s",
            "final10_acc_bias_norm_slope_m_s3",
            "final10_gyro_bias_norm_slope_rad_s2",
            "final_six_axis_bias_norm",
            "final10_six_axis_bias_norm_slope_per_s",
        ],
    ] = [0.249, 0.249, 0.1, 0.1, 0.249, 0.1]
    cases = cases.reset_index(drop=True)
    frame_rows = []
    for case in cases.itertuples(index=False):
        for frame_index in range(case.frame_count):
            is_zero_mean = case.case == "w3_full_all_zero_mean"
            frame_rows.append(
                {
                    "case": case.case,
                    "frame_index": frame_index,
                    "timestamp_ns": frame_index * 1_000_000_000 // 30,
                    "in_evaluation": frame_index >= case.evaluation_start_frame,
                    "frame_state_finite": True,
                    "translation_error_m": case.translation_rmse_m,
                    "rotation_error_deg": case.rotation_rmse_deg,
                    "velocity_error_m_s": case.velocity_rmse_m_s,
                    "acc_bias_error_m_s2": 0.0,
                    "gyro_bias_error_rad_s": 0.0,
                    "estimated_acc_bias_norm_m_s2": 9e-7 if is_zero_mean else 0.1,
                    "estimated_gyro_bias_norm_rad_s": 9e-7 if is_zero_mean else 0.1,
                }
            )
    frames = pd.DataFrame(frame_rows)
    return cases, frames


def _task4_ready_windows(cases: pd.DataFrame) -> pd.DataFrame:
    runtime_match_fields = (
        "from_pose",
        "init_motion",
        "prev_velocity",
        "curr_velocity",
        "prev_acc_bias",
        "prev_gyro_bias",
        "curr_acc_bias",
        "curr_gyro_bias",
    )
    rows = []
    for case in cases.itertuples(index=False):
        if case.case == "visual_initial":
            continue
        for target in range(2, int(case.frame_count)):
            row = {
                "case": case.case,
                "visual_condition": case.visual_condition,
                "imu_noise_mode": case.imu_noise_mode,
                "bias_mode": case.bias_mode,
                "writeback": case.writeback,
                "window_target": target,
                "graph_class": "LocalWindowInertialGraph",
                "context_device": "cpu",
                "bias_state_active": bool(case.bias_enabled),
                "bias_parameter_structurally_present": True,
                "vio_factor_active": True,
                "imu_residual_rows": 10 if case.bias_enabled else 6,
                "finite": True,
                "optimizer_completed": True,
                "optimizer_final_loss_finite": True,
                "all_edges_rebase_fields_match_runtime": True,
                "leading_preintegration_fields_immutable": True,
                "trailing_preintegration_fields_immutable": True,
                "trailing_linearization_bias_matches_creation_source": True,
                "trailing_preintegration_rotation_matches_creation_source": True,
                "edge_12_source_state_matches_runtime": True,
                "bias_fields_cleared_per_edge": 0 if case.bias_enabled else 8,
                "source_main_imu_bias_jacobian_rank": 6 if case.bias_enabled else 0,
                "terminal_main_imu_bias_jacobian_rank": 0,
                "shift_source_bias_error": 0.0,
                "shift_source_pose_translation_error_m": 0.0,
                "shift_source_pose_rotation_error_rad": 0.0,
                "shift_source_velocity_error_m_s": 0.0,
            }
            for prefix in ("leading", "trailing"):
                for field in runtime_match_fields:
                    row[f"{prefix}_{field}_matches_runtime"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def _write_task4_source_manifest(
    bundle_dir: Path,
    sensor_key: str,
    *,
    motion_sampling_contract: str = "analytic_yaw_rate_at_imu_timestamp_v1",
) -> None:
    sensor_dir = bundle_dir / "inputs" / "sensors" / sensor_key
    sensor_dir.mkdir(parents=True, exist_ok=True)
    generation_manifest = {
        "motion_sampling_contract": motion_sampling_contract,
        "orientation_profile": "roll=0, pitch=0, yaw=2*pi*t/trajectory_period_s",
        "noise_parameter_semantics": "continuous-time density",
        "camera_rate_hz": 30.0,
        "imu_rate_hz": 100.0,
        "trajectory_period_s": 10.0,
        "formulas": [
            "gyro_body = [0, 0, 2*pi/trajectory_period_s] at each IMU timestamp",
            "white_noise_sample_std = sigma_density*sqrt(imu_rate_hz)",
            "bias_random_walk_step_std = sigma_walk_density/sqrt(imu_rate_hz)",
        ],
    }
    generation_path = sensor_dir / "generation_manifest.json"
    generation_path.write_text(json.dumps(generation_manifest), encoding="utf-8")
    input_manifest = {
        "schema_version": 1,
        "sensor_artifacts": {
            sensor_key: {
                "manifest_json": str(generation_path.relative_to(bundle_dir)),
            }
        },
    }
    (bundle_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest),
        encoding="utf-8",
    )


class TestTask4Readiness:
    def test_window_contract_rejects_missing_production_structure(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            ReadinessInputContractError,
            _validate_window_diagnostics_contract,
        )

        cases, _ = _task4_ready_tables()
        windows = _task4_ready_windows(cases)
        _validate_window_diagnostics_contract(cases, windows)
        assert cases["window_contract_ok"].all()

        malformed = windows.drop(columns=["graph_class"])
        with pytest.raises(ReadinessInputContractError, match="missing required columns"):
            _validate_window_diagnostics_contract(cases.copy(), malformed)

    @pytest.mark.parametrize(
        "failure_flag",
        ["optimizer_final_loss_finite", "vio_factor_active"],
    )
    def test_complete_window_failure_is_classified_as_numerical_not_incomplete(
        self,
        failure_flag,
    ):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            _validate_window_diagnostics_contract,
            classify_readiness,
        )

        cases, frames = _task4_ready_tables()
        windows = _task4_ready_windows(cases)
        failed = (windows["case"] == "w3_full_all") & (windows["window_target"] == 10)
        windows.loc[failed, failure_flag] = False

        _validate_window_diagnostics_contract(cases, windows)

        assert not bool(cases.loc[cases["case"] == "w3_full_all", "window_contract_ok"].iloc[0])
        assert classify_readiness(cases, frames) == "numerical_or_state_propagation_failure"

    def test_raw_window_shift_recomputes_gate7_case_summary(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            _validate_window_diagnostics_contract,
            classify_readiness,
        )

        cases, frames = _task4_ready_tables()
        windows = _task4_ready_windows(cases)
        shifted = (windows["case"] == "w3_full_all") & (windows["window_target"] == 10)
        windows.loc[shifted, "shift_source_pose_translation_error_m"] = 1.0

        _validate_window_diagnostics_contract(cases, windows)

        full = cases.loc[cases["case"] == "w3_full_all"].iloc[0]
        assert full["max_shift_source_pose_translation_error_m"] == 1.0
        assert classify_readiness(cases, frames) == "numerical_or_state_propagation_failure"

    def test_safe_ratio_treats_zero_over_zero_as_unchanged_not_improved(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import _safe_ratio

        assert _safe_ratio(0.0, 0.0) == 1.0
        assert math.isinf(_safe_ratio(1.0, 0.0))

    def test_ready_decision_preserves_all_eight_gate_details(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        cases, frames = _task4_ready_tables()
        decision = classify_readiness(cases, frames)

        assert decision == "ready_for_manual_scene_validation"
        details = cases.attrs["readiness_gates"]
        assert len(details) == 8
        assert [gate["gate"] for gate in details] == list(range(1, 9))
        assert all(gate["pass"] for gate in details)
        assert all(set(gate) >= {"gate", "name", "pass", "values", "threshold"} for gate in details)
        assert cases["readiness_decision"].eq(decision).all()
        for gate_number in range(1, 9):
            assert cases[f"readiness_gate_{gate_number}_pass"].all()

    def test_constant_bias_gate_accepts_converged_estimate_equal_to_persistence(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        cases, frames = _task4_ready_tables()
        full = cases["case"] == "w3_full_all"
        cases.loc[
            full,
            ["acc_bias_rmse_m_s2", "gyro_bias_rmse_rad_s", "six_axis_bias_rmse"],
        ] = [0.1, 0.1, 0.1]
        cases.loc[
            full,
            [
                "previous_estimate_persistence_acc_bias_rmse_m_s2",
                "previous_estimate_persistence_gyro_bias_rmse_rad_s",
                "previous_estimate_persistence_six_axis_bias_rmse",
            ],
        ] = [0.1, 0.1, 0.1]

        decision = classify_readiness(cases, frames)
        gate2 = cases.attrs["readiness_gates"][1]

        assert gate2["pass"]
        assert decision == "ready_for_manual_scene_validation"

    def test_constant_bias_gate_requires_acc_and_gyro_to_pass_independently(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        cases, frames = _task4_ready_tables()
        full = cases["case"] == "w3_full_all"
        cases.loc[full, "acc_bias_rmse_m_s2"] = 0.1
        cases.loc[full, "gyro_bias_rmse_rad_s"] = 0.9
        cases.loc[full, "six_axis_bias_rmse"] = 0.1

        assert classify_readiness(cases, frames) == "bias_initialization_or_tracking_fails"
        gate2 = cases.attrs["readiness_gates"][1]
        assert gate2["values"]["acc_ratio_to_zero"] == 0.1
        assert gate2["values"]["gyro_ratio_to_zero"] == 0.9
        assert not gate2["pass"]

    @pytest.mark.parametrize("failure_kind", ["nonfinite", "state"])
    def test_numerical_or_state_failure_has_highest_precedence(self, failure_kind):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        cases, frames = _task4_ready_tables()
        cases.loc[cases["case"] == "w3_full_all", "six_axis_bias_rmse"] = 2.0
        if failure_kind == "nonfinite":
            cases.loc[cases["case"] == "w3_full_current", "finite"] = False
        else:
            cases.loc[cases["case"] == "w3_full_current", "state_propagation_ok"] = False

        assert classify_readiness(cases, frames) == "numerical_or_state_propagation_failure"
        assert len(cases.attrs["readiness_gates"]) == 8

    def test_writeback_pose_or_velocity_discontinuity_is_state_propagation_failure(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        for field in (
            "max_shift_source_pose_translation_error_m",
            "max_shift_source_pose_rotation_error_rad",
            "max_shift_source_velocity_error_m_s",
        ):
            cases, frames = _task4_ready_tables()
            cases.loc[cases["case"] == "w3_full_all", field] = 1e-7

            assert (
                classify_readiness(cases, frames)
                == "numerical_or_state_propagation_failure"
            )

    def test_input_contract_rejects_incomplete_duplicate_wrong_horizon_and_nan_evidence(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            ReadinessInputContractError,
            classify_readiness,
        )

        cases, frames = _task4_ready_tables()
        with pytest.raises(ReadinessInputContractError, match="empty"):
            classify_readiness(pd.DataFrame(), pd.DataFrame())

        short_only_cases = cases[cases["case"] == "w3_full_all_zero_mean"].copy()
        short_only_frames = frames[frames["case"] == "w3_full_all_zero_mean"].copy()
        with pytest.raises(ReadinessInputContractError, match="missing required cases"):
            classify_readiness(short_only_cases, short_only_frames)

        main_only_cases = cases[cases["case"] != "w3_full_all_zero_mean"].copy()
        main_only_frames = frames[frames["case"] != "w3_full_all_zero_mean"].copy()
        with pytest.raises(ReadinessInputContractError, match="missing required cases"):
            classify_readiness(main_only_cases, main_only_frames)

        duplicate_cases = pd.concat(
            [cases, cases[cases["case"] == "w3_full_all"].assign(frame_count=31)],
            ignore_index=True,
        )
        with pytest.raises(ReadinessInputContractError, match="duplicate case"):
            classify_readiness(duplicate_cases, frames)

        wrong_horizon = cases.copy()
        wrong_horizon.loc[wrong_horizon["case"] == "w3_full_all", "frame_count"] = 300
        with pytest.raises(ReadinessInputContractError, match="301 frames"):
            classify_readiness(wrong_horizon, frames)

        wrong_eval_count = cases.copy()
        wrong_eval_count.loc[
            wrong_eval_count["case"] == "w3_full_all", "evaluation_frame_count"
        ] = 270
        with pytest.raises(ReadinessInputContractError, match="271 evaluation frames"):
            classify_readiness(wrong_eval_count, frames)

        missing_frame = frames.drop(
            frames.index[(frames["case"] == "w3_full_all") & (frames["frame_index"] == 300)][0]
        )
        with pytest.raises(ReadinessInputContractError, match="frame rows"):
            classify_readiness(cases, missing_frame)

        malformed_cases = cases.copy()
        malformed_cases["six_axis_bias_rmse"] = malformed_cases[
            "six_axis_bias_rmse"
        ].astype(object)
        malformed_cases.loc[
            malformed_cases["case"] == "w3_full_all", "six_axis_bias_rmse"
        ] = "not-a-number"
        with pytest.raises(ReadinessInputContractError, match="numeric"):
            classify_readiness(malformed_cases, frames)

        nan_cases = cases.copy()
        nan_cases.loc[nan_cases["case"] == "w3_full_all", "six_axis_bias_rmse"] = np.nan
        assert (
            classify_readiness(nan_cases, frames)
            == "numerical_or_state_propagation_failure"
        )

    def test_bias_gate_failure_precedes_weighting_failure(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        cases, frames = _task4_ready_tables()
        full = cases["case"] == "w3_full_all"
        cases.loc[full, "acc_bias_rmse_m_s2"] = 0.5000001
        cases.loc[full, "translation_rmse_m"] = 2.0

        assert classify_readiness(cases, frames) == "bias_initialization_or_tracking_fails"
        details = cases.attrs["readiness_gates"]
        assert not details[1]["pass"]
        assert not details[5]["pass"]

    def test_weighting_failure_is_reported_after_bias_path_passes(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        cases, frames = _task4_ready_tables()
        full = cases["case"] == "w3_full_all"
        cases.loc[full, "translation_rmse_m"] = 1.0500001

        assert classify_readiness(cases, frames) == "bias_path_works_but_joint_weighting_fails"
        details = cases.attrs["readiness_gates"]
        assert all(details[index]["pass"] for index in (1, 2, 3, 4, 6))
        assert not details[5]["pass"]

    def test_gate_boundaries_follow_strict_and_inclusive_inequalities(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        cases, frames = _task4_ready_tables()
        by_case = cases.set_index("case")
        by_case.loc[
            "w3_full_all", ["acc_bias_rmse_m_s2", "gyro_bias_rmse_rad_s"]
        ] = [0.5, 0.5]
        by_case.loc[
            "w3_full_all",
            [
                "previous_estimate_persistence_acc_bias_rmse_m_s2",
                "previous_estimate_persistence_gyro_bias_rmse_rad_s",
            ],
        ] = [1.0, 1.0]
        by_case.loc["w3_full_all", "translation_rmse_m"] = 0.95
        by_case.loc[
            "w3_full_all_drifting_bias",
            ["acc_bias_rmse_m_s2", "gyro_bias_rmse_rad_s"],
        ] = [0.8, 0.8]
        below_floor = np.nextafter(1e-6, 0.0)
        by_case.loc[
            "w3_full_all_zero_mean",
            ["max_acc_bias_norm_m_s2", "max_gyro_bias_norm_rad_s"],
        ] = below_floor
        by_case.loc[
            "w3_full_all_zero_mean",
            ["translation_rmse_m", "rotation_rmse_deg", "velocity_rmse_m_s"],
        ] = [0.002, 0.02, 0.002]
        by_case.loc[
            "w3_full_all_zero_mean",
            ["visual_translation_rmse_m", "visual_rotation_rmse_deg", "visual_velocity_rmse_m_s"],
        ] = [0.002, 0.02, 0.002]
        by_case.loc[
            "w3_full_all_zero_normal",
            [
                "final_acc_bias_norm_m_s2",
                "final_gyro_bias_norm_rad_s",
                "final10_acc_bias_norm_slope_m_s3",
                "final10_gyro_bias_norm_slope_rad_s2",
            ],
        ] = [np.nextafter(0.25, 0.0), np.nextafter(0.25, 0.0), 0.1, 0.1]
        optimized = by_case.index != "visual_initial"
        by_case.loc[optimized, "max_shift_source_bias_error"] = 1e-8
        cases = by_case.reset_index()
        zero_mean_eval = (frames["case"] == "w3_full_all_zero_mean") & frames[
            "in_evaluation"
        ]
        frames.loc[
            zero_mean_eval,
            ["estimated_acc_bias_norm_m_s2", "estimated_gyro_bias_norm_rad_s"],
        ] = below_floor

        assert classify_readiness(cases, frames) == "ready_for_manual_scene_validation"

        strict_floor_frames = frames.copy()
        strict_floor_frames.loc[
            (strict_floor_frames["case"] == "w3_full_all_zero_mean")
            & (strict_floor_frames["frame_index"] == 10),
            "estimated_acc_bias_norm_m_s2",
        ] = 1e-6
        assert (
            classify_readiness(cases.copy(), strict_floor_frames)
            == "bias_initialization_or_tracking_fails"
        )

        strict_probe_cases = cases.copy()
        strict_probe_cases.loc[
            strict_probe_cases["case"] == "w3_full_all_zero_normal",
            "final_acc_bias_norm_m_s2",
        ] = 0.25
        assert (
            classify_readiness(strict_probe_cases, frames)
            == "bias_initialization_or_tracking_fails"
        )

    @pytest.mark.parametrize(
        "bias_norm_column",
        ["estimated_acc_bias_norm_m_s2", "estimated_gyro_bias_norm_rad_s"],
    )
    def test_gate4_checks_every_short_run_evaluation_frame(self, bias_norm_column):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import classify_readiness

        cases, frames = _task4_ready_tables()
        target = (frames["case"] == "w3_full_all_zero_mean") & (
            frames["frame_index"] == 17
        )
        frames.loc[target, bias_norm_column] = 1e-6

        decision = classify_readiness(cases, frames)
        assert decision == "bias_initialization_or_tracking_fails"
        gate4 = cases.attrs["readiness_gates"][3]
        assert not gate4["pass"]
        assert gate4["values"]["evaluated_frame_indices"] == list(range(6, 31))

    def test_contract_rejects_string_boolean_evidence(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            ReadinessInputContractError,
            classify_readiness,
        )

        cases, frames = _task4_ready_tables()
        cases["finite"] = "False"

        with pytest.raises(ReadinessInputContractError, match="boolean"):
            classify_readiness(cases, frames)

    @pytest.mark.parametrize("target", ["case_count", "frame_index"])
    def test_contract_rejects_fractional_counts_and_indices(self, target):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            ReadinessInputContractError,
            classify_readiness,
        )

        cases, frames = _task4_ready_tables()
        if target == "case_count":
            cases["frame_count"] = cases["frame_count"].astype(float)
            cases.loc[cases["case"] == "w3_full_all", "frame_count"] += 0.25
        else:
            frames["frame_index"] = frames["frame_index"].astype(float)
            frames.loc[
                (frames["case"] == "w3_full_all") & (frames["frame_index"] == 100),
                "frame_index",
            ] += 0.25

        with pytest.raises(ReadinessInputContractError, match="integer"):
            classify_readiness(cases, frames)

    def test_contract_rejects_canonical_case_definition_mismatch(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            ReadinessInputContractError,
            classify_readiness,
        )

        cases, frames = _task4_ready_tables()
        cases.loc[cases["case"] == "w3_full_all", "writeback"] = "current"

        with pytest.raises(ReadinessInputContractError, match="canonical definition"):
            classify_readiness(cases, frames)


class TestTask4BundleAndCli:
    def test_interactive_model_keeps_visual_conditions_and_case_controls_separate(self):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            _build_interactive_model,
            _render_interactive_html,
            evaluate_case,
        )

        result, truth, visual = _task4_metric_fixture()
        drifted_result = replace(
            result,
            case=replace(result.case, visual_condition="drifted_visual"),
        )
        zero_truth = replace(
            truth,
            bias_mode="zero_bias",
            true_acc_bias=torch.zeros_like(truth.true_acc_bias),
            true_gyro_bias=torch.zeros_like(truth.true_gyro_bias),
        )
        clean_result = replace(
            result,
            case=ValidationCase(
                name="w3_full_all_zero_mean",
                visual_condition="clean_visual",
                imu_noise_mode="mean_measurement",
                bias_mode="zero_bias",
                bias_enabled=True,
                writeback="all_optimized",
            ),
        )
        drifted_metrics, drifted_frames = evaluate_case(drifted_result, truth, visual)
        clean_metrics, clean_frames = evaluate_case(clean_result, zero_truth, visual)
        case_table = pd.DataFrame([drifted_metrics, clean_metrics])
        frame_table = pd.concat([drifted_frames, clean_frames], ignore_index=True)

        model = _build_interactive_model(
            case_table,
            frame_table,
            decision=None,
            readiness_status="incomplete_evidence_contract",
        )
        trace_ids = {trace["id"] for trace in model["trace_controls"]}

        assert "visual:drifted_visual:7" in trace_ids
        assert "visual:clean_visual:7" in trace_ids
        assert "w3_full_all" in trace_ids
        assert "w3_full_all_zero_mean" in trace_ids
        assert len(model["trace_styles"]) == len(trace_ids)
        page = _render_interactive_html(model)
        assert 'data-trace="w3_full_all"' in page
        assert 'data-trace="w3_full_all_zero_mean"' in page
        assert "visual: clean_visual (7 frames)" in page
        assert "w3_full_all (drifted_visual, 7 frames)" in page
        assert "w3_full_all_zero_mean (clean_visual, 7 frames)" in page

    def test_combined_bundle_selects_short_zero_mean_and_six_main_cases(
        self,
        tmp_path: Path,
    ):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            combine_evidence_bundles,
        )

        cases, frames = _task4_ready_tables()
        short_dir = tmp_path / "short"
        main_dir = tmp_path / "main"
        short_dir.mkdir()
        main_dir.mkdir()
        short_case = cases["case"] == "w3_full_all_zero_mean"
        short_frames = frames["case"] == "w3_full_all_zero_mean"
        legacy_case_columns = [
            "final10_acc_bias_norm_slope_m_s3",
            "final10_gyro_bias_norm_slope_rad_s2",
        ]
        source_cases = cases.drop(columns=legacy_case_columns)
        source_cases.loc[short_case].to_csv(
            short_dir / "joint_objective_cases.csv", index=False
        )
        frames.loc[short_frames].to_csv(short_dir / "frame_metrics.csv", index=False)
        ready_windows = _task4_ready_windows(cases)
        ready_windows.loc[ready_windows["case"] == "w3_full_all_zero_mean"].to_csv(
            short_dir / "window_diagnostics.csv", index=False
        )

        source_cases.loc[~short_case].to_csv(
            main_dir / "joint_objective_cases.csv", index=False
        )
        frames.loc[~short_frames].to_csv(main_dir / "frame_metrics.csv", index=False)
        ready_windows.loc[ready_windows["case"] != "w3_full_all_zero_mean"].to_csv(
            main_dir / "window_diagnostics.csv", index=False
        )
        _write_task4_source_manifest(short_dir, "zero_bias__mean_measurement")
        _write_task4_source_manifest(main_dir, "constant_bias__fixed_seed_normal")

        paths = combine_evidence_bundles(short_dir, main_dir, tmp_path / "combined")

        combined_cases = pd.read_csv(paths["cases_csv"])
        combined_frames = pd.read_csv(paths["frames_csv"])
        assert set(combined_cases["case"]) == set(cases["case"])
        assert len(combined_cases) == 7
        assert len(combined_frames) == 31 + 6 * 301
        assert set(legacy_case_columns).issubset(combined_cases.columns)
        assert set(combined_cases["readiness_decision"]) == {
            "ready_for_manual_scene_validation"
        }
        manifest = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
        assert manifest["readiness"]["decision"] == "ready_for_manual_scene_validation"
        assert manifest["evidence_sources"]["short"] == str(short_dir)
        assert manifest["evidence_sources"]["main"] == str(main_dir)

        import Scripts.diagnose_full_w3_joint_objective_synthetic as diagnose

        cli_output = tmp_path / "combined_cli"
        assert (
            diagnose.main(
                [
                    "--output-dir",
                    str(cli_output),
                    "--combine-short-dir",
                    str(short_dir),
                    "--combine-main-dir",
                    str(main_dir),
                ]
            )
            == 0
        )
        assert (cli_output / "diagnostics_interactive.html").is_file()

    def test_combined_bundle_rejects_legacy_motion_sampling_contract(
        self,
        tmp_path: Path,
    ):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import (
            ReadinessInputContractError,
            combine_evidence_bundles,
        )

        cases, frames = _task4_ready_tables()
        windows = _task4_ready_windows(cases)
        short_dir = tmp_path / "short"
        main_dir = tmp_path / "main"
        short_dir.mkdir()
        main_dir.mkdir()
        short_case = cases["case"] == "w3_full_all_zero_mean"
        short_frames = frames["case"] == "w3_full_all_zero_mean"
        cases.loc[short_case].to_csv(short_dir / "joint_objective_cases.csv", index=False)
        frames.loc[short_frames].to_csv(short_dir / "frame_metrics.csv", index=False)
        windows.loc[windows["case"] == "w3_full_all_zero_mean"].to_csv(
            short_dir / "window_diagnostics.csv", index=False
        )
        cases.loc[~short_case].to_csv(main_dir / "joint_objective_cases.csv", index=False)
        frames.loc[~short_frames].to_csv(main_dir / "frame_metrics.csv", index=False)
        windows.loc[windows["case"] != "w3_full_all_zero_mean"].to_csv(
            main_dir / "window_diagnostics.csv", index=False
        )
        _write_task4_source_manifest(
            short_dir,
            "zero_bias__mean_measurement",
            motion_sampling_contract="interval_average_angular_velocity_v0",
        )
        _write_task4_source_manifest(main_dir, "constant_bias__fixed_seed_normal")

        with pytest.raises(ReadinessInputContractError, match="motion sampling contract"):
            combine_evidence_bundles(short_dir, main_dir, tmp_path / "combined")

    def test_bundle_writes_raw_window_diagnostics(self, tmp_path: Path):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import write_validation_bundle

        result, truth, visual = _task4_metric_fixture()
        diagnostics = pd.DataFrame(
            {
                "target_frame": [6],
                "finite": [True],
                "optimizer_completed": [True],
                "optimizer_final_loss_finite": [True],
                "all_edges_rebase_fields_match_runtime": [True],
                "initial_visual_energy": [1.0],
            }
        )
        result = replace(result, frame_diagnostics=diagnostics)

        paths = write_validation_bundle(
            tmp_path,
            [result],
            {"constant_bias__mean_measurement": truth},
            {"clean_visual": visual},
        )

        assert paths["window_diagnostics"].is_file()
        rows = pd.read_csv(paths["window_diagnostics"])
        assert rows.loc[0, "case"] == "w3_full_all"
        assert rows.loc[0, "target_frame"] == 6
        manifest = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
        assert manifest["outputs"]["window_diagnostics"] == "window_diagnostics.csv"

    def test_bundle_rejects_truth_metadata_that_disagrees_with_case_key(self, tmp_path: Path):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import write_validation_bundle

        result, truth, visual = _task4_metric_fixture()
        mismatched_truth = replace(truth, bias_mode="zero_bias")

        with pytest.raises(ValueError, match="truth metadata"):
            write_validation_bundle(
                tmp_path,
                [result],
                {"constant_bias__mean_measurement": mismatched_truth},
                {"clean_visual": visual},
            )

    def test_direct_script_entrypoint_resolves_project_imports(self):
        script = Path(__file__).resolve().parents[1] / "diagnose_full_w3_joint_objective_synthetic.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert "--duration-s" in completed.stdout

    def test_small_bundle_writes_csv_chinese_report_html_and_manifest_without_fake_sensors(
        self,
        tmp_path: Path,
    ):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import write_validation_bundle

        result, truth, visual = _task4_metric_fixture()
        paths = write_validation_bundle(
            tmp_path,
            [result],
            {"constant_bias__mean_measurement": truth},
            {"clean_visual": visual},
        )

        expected_names = {
            "cases_csv": "joint_objective_cases.csv",
            "frames_csv": "frame_metrics.csv",
            "report": "joint_objective_summary_cn.md",
            "html": "diagnostics_interactive.html",
            "input_manifest": "input_manifest.json",
        }
        for key, filename in expected_names.items():
            assert paths[key] == tmp_path / filename
            assert paths[key].is_file()
        assert len(pd.read_csv(paths["cases_csv"])) == 1
        assert len(pd.read_csv(paths["frames_csv"])) == 7
        assert not (tmp_path / "inputs" / "sensors").exists()

        report = paths["report"].read_text(encoding="utf-8")
        assert "未形成最终四选一结论" in report
        assert "证据契约不完整" in report
        assert "synthetic 不是 scene" in report
        assert "12 个未观测 Bias 参数" in report
        assert "unknown_not_exposed_by_production" in report
        assert "本阶段无需新 HoloOcean 数据" in report
        assert "export realized Bias" in report

        manifest = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
        assert manifest["truth_keys"] == ["constant_bias__mean_measurement"]
        assert manifest["visual_conditions"] == ["clean_visual"]
        assert manifest["sensor_artifacts"] == {}
        assert manifest["readiness"]["status"] == "incomplete_evidence_contract"
        assert manifest["readiness"]["decision"] is None

    def test_html_is_self_contained_has_all_views_traces_and_fixed_extent_data(self, tmp_path: Path):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import write_validation_bundle

        result, truth, visual = _task4_metric_fixture()
        paths = write_validation_bundle(
            tmp_path,
            [result],
            {"constant_bias__mean_measurement": truth},
            {"clean_visual": visual},
        )
        page = paths["html"].read_text(encoding="utf-8")
        views = (
            "XY",
            "XZ",
            "Trajectory",
            "Position Error",
            "Rotation Error",
            "Velocity Error",
            "Acc Bias Error",
            "Gyro Bias Error",
        )
        for view in views:
            assert view in page
        assert "Bias Error" in page
        assert "<canvas" in page
        assert "type=\"checkbox\"" in page
        assert "addEventListener('wheel'" in page
        assert "pointermove" in page
        assert "Reset view" in page
        assert "https://" not in page
        assert "http://" not in page
        assert "<script src=" not in page
        assert "<link" not in page
        assert "MODEL.view_extents[activeView]" in page
        assert "@media" in page

        marker = "const MODEL = "
        model_start = page.index(marker) + len(marker)
        model_end = page.index(";\n", model_start)
        model = json.loads(page[model_start:model_end])
        assert set(model["view_extents"]) == set(views)
        assert set(model["trace_groups"]) == {"gt", "visual", "locked", "current", "all"}
        for extent in model["view_extents"].values():
            assert set(extent) == {"x_min", "x_max", "y_min", "y_max"}
            assert all(math.isfinite(value) for value in extent.values())
            assert extent["x_min"] < extent["x_max"]
            assert extent["y_min"] < extent["y_max"]

    def test_html_toolbar_stacks_view_and_trace_rows_without_auto_column_squeeze(self, tmp_path: Path):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import write_validation_bundle

        result, truth, visual = _task4_metric_fixture()
        paths = write_validation_bundle(
            tmp_path,
            [result],
            {"constant_bias__mean_measurement": truth},
            {"clean_visual": visual},
        )
        page = paths["html"].read_text(encoding="utf-8")

        assert ".toolbar { display: flex; flex-direction: column;" in page
        assert ".trace-controls { display: flex; align-items: center; justify-content: flex-start;" in page
        assert "grid-template-columns: minmax(0, 1fr) auto" not in page

    def test_complete_tiny_inputs_write_mode_scoped_sensor_and_visual_artifacts(
        self,
        tmp_path: Path,
        short_runner_inputs,
    ):
        from Scripts.diagnose_full_w3_joint_objective_synthetic import write_validation_bundle

        config, imu_input, visual_input = short_runner_inputs
        truth = generate_truth(
            config,
            bias_mode="constant_bias",
            noise_mode="fixed_seed_normal",
        )
        result = run_validation_case(
            config,
            imu_input,
            visual_input,
            _validation_case("visual_initial"),
        )
        mode_key = "constant_bias__fixed_seed_normal"
        paths = write_validation_bundle(
            tmp_path,
            [result],
            {mode_key: truth},
            {"drifted_visual": visual_input},
            config=config,
            imu_inputs_by_mode={mode_key: imu_input},
        )

        sensor_dir = tmp_path / "inputs" / "sensors" / mode_key
        visual_dir = tmp_path / "inputs" / "visual" / "drifted_visual"
        assert (sensor_dir / "synthetic_ground_truth.csv").is_file()
        assert (sensor_dir / "synthetic_imu.csv").is_file()
        assert (sensor_dir / "generation_manifest.json").is_file()
        assert (visual_dir / "synthetic_visual_observations.npz").is_file()
        manifest = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
        assert set(manifest["sensor_artifacts"]) == {mode_key}
        assert set(manifest["visual_artifacts"]) == {"drifted_visual"}

    def test_cli_case_filter_and_unknown_case_error_without_long_optimization(
        self,
        tmp_path: Path,
        capsys,
    ):
        import Scripts.diagnose_full_w3_joint_objective_synthetic as diagnose

        selected = diagnose.select_cases(0.2, ["w3_full_all", "visual_initial"])
        assert [case.name for case in selected] == ["visual_initial", "w3_full_all"]
        with pytest.raises(ValueError, match="Unknown case"):
            diagnose.select_cases(0.2, ["does_not_exist"])
        with pytest.raises(SystemExit):
            diagnose.main(
                [
                    "--duration-s",
                    "0.2",
                    "--output-dir",
                    str(tmp_path / "unknown"),
                    "--case",
                    "does_not_exist",
                ]
            )

        exit_code = diagnose.main(
            [
                "--duration-s",
                "0.2",
                "--output-dir",
                str(tmp_path / "cli"),
                "--case",
                "visual_initial",
            ]
        )
        assert exit_code == 0
        events = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
        assert [event["event"] for event in events] == [
            "case_start",
            "case_complete",
            "bundle_complete",
        ]

    def test_cli_emits_window_progress_for_optimized_case(self, tmp_path: Path, capsys):
        import Scripts.diagnose_full_w3_joint_objective_synthetic as diagnose

        exit_code = diagnose.main(
            [
                "--duration-s",
                "0.2",
                "--output-dir",
                str(tmp_path / "progress"),
                "--case",
                "w3_full_all",
            ]
        )

        assert exit_code == 0
        events = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
        progress_events = [event for event in events if event["event"] == "window_progress"]
        assert progress_events
        assert [event["completed_windows"] for event in progress_events] == list(
            range(1, progress_events[-1]["total_windows"] + 1)
        )

    def test_cli_persists_case_failure_and_continues_bundle(
        self,
        tmp_path: Path,
        capsys,
        monkeypatch,
    ):
        import Scripts.diagnose_full_w3_joint_objective_synthetic as diagnose

        original_runner = diagnose.run_validation_case

        def fail_one_case(config, imu_input, visual_input, case, progress=None):
            if case.name == "w3_full_all":
                raise RuntimeError("synthetic optimizer failure")
            return original_runner(config, imu_input, visual_input, case, progress=progress)

        monkeypatch.setattr(diagnose, "run_validation_case", fail_one_case)
        output_dir = tmp_path / "failure"
        exit_code = diagnose.main(
            [
                "--duration-s",
                "0.2",
                "--output-dir",
                str(output_dir),
                "--case",
                "visual_initial",
                "--case",
                "w3_full_all",
            ]
        )

        assert exit_code == 1
        events = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
        assert [event["event"] for event in events] == [
            "case_start",
            "case_complete",
            "case_start",
            "case_failed",
            "bundle_complete",
        ]
        cases = pd.read_csv(output_dir / "joint_objective_cases.csv")
        failed = cases.loc[cases["case"] == "w3_full_all"].iloc[0]
        assert failed["run_status"] == "failed"
        assert failed["failure_type"] == "RuntimeError"
        assert failed["failure_message"] == "synthetic optimizer failure"
        manifest = json.loads((output_dir / "input_manifest.json").read_text(encoding="utf-8"))
        assert manifest["failures"][0]["case"] == "w3_full_all"
        assert (output_dir / "joint_objective_summary_cn.md").is_file()

        only_failure_dir = tmp_path / "only_failure"
        assert (
            diagnose.main(
                [
                    "--duration-s",
                    "0.2",
                    "--output-dir",
                    str(only_failure_dir),
                    "--case",
                    "w3_full_all",
                ]
            )
            == 1
        )
        only_failure_cases = pd.read_csv(
            only_failure_dir / "joint_objective_cases.csv"
        )
        assert only_failure_cases.loc[0, "run_status"] == "failed"

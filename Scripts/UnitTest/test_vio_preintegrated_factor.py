import math
import importlib.util
import subprocess
import sys
from pathlib import Path

import pypose as pp
import pytest
import torch

from Utility.IMUKinematics import (
    gravity_roll_pitch_aligned_rotation,
    build_weight_matrix_from_covariances,
    integrate_gyro_attitude_world,
    should_enable_preintegrated_vio_factor,
    vio_bias_random_walk_residual,
    vio_preintegrated_covariance_blocks,
    vio_preintegrated_covariance_matrix,
    vio_preintegrated_imu_residual,
)


def _load_preintegrate_imu():
    module_path = Path("Module/IMUPreintegration.py").resolve()
    spec = importlib.util.spec_from_file_location("imu_preintegration_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.preintegrate_imu


def _se3(translation, yaw_rad=0.0):
    half = yaw_rad * 0.5
    quat_xyzw = torch.tensor(
        [[0.0, 0.0, math.sin(half), math.cos(half)]],
        dtype=torch.float64,
    )
    trans = torch.tensor([translation], dtype=torch.float64)
    return pp.SE3(torch.cat([trans, quat_xyzw], dim=-1))


def test_vio_residual_is_zero_for_constant_world_velocity():
    dt = 0.1
    pose_i = _se3([0.0, 0.0, 0.0])
    pose_j = _se3([0.1, 0.0, 0.0])

    residual = vio_preintegrated_imu_residual(
        from_pose=pose_i,
        to_pose=pose_j,
        prev_velocity_world=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        curr_velocity_world=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        delta_R=pp.identity_SO3(dtype=torch.float64),
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.zeros(3, dtype=torch.float64),
        dt_total=dt,
    )

    assert residual.shape == (3, 3)
    assert torch.allclose(residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-9)


def test_vio_residual_projects_world_velocity_into_previous_body_frame():
    dt = 0.1
    yaw = math.pi / 2.0
    pose_i = _se3([0.0, 0.0, 0.0], yaw_rad=yaw)
    pose_j = _se3([0.0, 0.1, 0.0], yaw_rad=yaw)

    residual = vio_preintegrated_imu_residual(
        from_pose=pose_i,
        to_pose=pose_j,
        prev_velocity_world=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        curr_velocity_world=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        delta_R=pp.identity_SO3(dtype=torch.float64),
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.zeros(3, dtype=torch.float64),
        dt_total=dt,
    )

    assert torch.allclose(residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-9)


def test_vio_residual_matches_preintegrated_rotation():
    yaw = 0.2
    pose_i = _se3([0.0, 0.0, 0.0])
    pose_j = _se3([0.0, 0.0, 0.0], yaw_rad=yaw)
    delta_R = pp.so3(torch.tensor([[0.0, 0.0, yaw]], dtype=torch.float64)).Exp()

    residual = vio_preintegrated_imu_residual(
        from_pose=pose_i,
        to_pose=pose_j,
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=torch.zeros(3, dtype=torch.float64),
        delta_R=delta_R,
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.zeros(3, dtype=torch.float64),
        dt_total=0.1,
    )

    assert torch.allclose(residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-9)


def test_vio_residual_uses_sensor_to_imu_extrinsic_for_lever_arm_motion():
    yaw = math.pi / 2.0
    pose_i_camera = _se3([0.0, 0.0, 0.0])
    pose_j_camera = _se3([0.0, 0.0, 0.0], yaw_rad=yaw)
    sensor_T_imu = pp.SE3(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64))
    delta_R = pp.so3(torch.tensor([[0.0, 0.0, yaw]], dtype=torch.float64)).Exp()

    residual = vio_preintegrated_imu_residual(
        from_pose=pose_i_camera,
        to_pose=pose_j_camera,
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=torch.zeros(3, dtype=torch.float64),
        delta_R=delta_R,
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float64),
        dt_total=0.0,
        sensor_T_imu=sensor_T_imu,
    )

    assert torch.allclose(residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-9)


def test_preintegrator_exposes_bias_jacobian_and_random_walk_covariance():
    preintegrate_imu = _load_preintegrate_imu()
    time_ns = torch.tensor([0, 10_000_000, 20_000_000], dtype=torch.long)
    acc = torch.tensor([[0.0, 0.0, -9.81]] * 3, dtype=torch.float32)
    gyro = torch.zeros((3, 3), dtype=torch.float32)

    acc_bias = torch.tensor([0.01, -0.02, 0.03], dtype=torch.float32)
    gyro_bias = torch.tensor([0.001, -0.002, 0.003], dtype=torch.float32)

    result = preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=pp.identity_SO3(dtype=torch.float64),
        gravity=9.81,
        sigma_acc=0.01,
        sigma_gyro=0.001,
        sigma_acc_w=0.02,
        sigma_gyro_w=0.003,
        acc_bias=acc_bias,
        gyro_bias=gyro_bias,
    )

    assert result.bias_jacobian is not None
    assert result.bias_jacobian.shape == (9, 6)
    assert result.bias_rw_cov is not None
    assert result.bias_rw_cov.shape == (6, 6)
    assert torch.all(result.bias_rw_cov.diagonal() > 0.0)
    assert result.linearized_acc_bias is not None
    assert torch.allclose(result.linearized_acc_bias, acc_bias)
    assert result.linearized_gyro_bias is not None
    assert torch.allclose(result.linearized_gyro_bias, gyro_bias)


def test_vio_residual_uses_bias_jacobian_to_correct_preintegrated_rotation():
    yaw = 0.1
    pose_i = _se3([0.0, 0.0, 0.0])
    pose_j = _se3([0.0, 0.0, 0.0], yaw_rad=yaw)
    bias_jacobian = torch.zeros((9, 6), dtype=torch.float64)
    bias_jacobian[8, 5] = 1.0

    residual = vio_preintegrated_imu_residual(
        from_pose=pose_i,
        to_pose=pose_j,
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=torch.zeros(3, dtype=torch.float64),
        delta_R=pp.identity_SO3(dtype=torch.float64),
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.zeros(3, dtype=torch.float64),
        dt_total=0.1,
        prev_acc_bias=torch.zeros(3, dtype=torch.float64),
        prev_gyro_bias=torch.tensor([0.0, 0.0, yaw], dtype=torch.float64),
        curr_acc_bias=torch.zeros(3, dtype=torch.float64),
        curr_gyro_bias=torch.tensor([0.0, 0.0, yaw], dtype=torch.float64),
        linearized_acc_bias=torch.zeros(3, dtype=torch.float64),
        linearized_gyro_bias=torch.zeros(3, dtype=torch.float64),
        bias_jacobian=bias_jacobian,
    )

    assert torch.allclose(residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-6)


def test_vio_bias_jacobian_uses_edge_linearization_not_bias_random_walk_step():
    yaw = 0.1
    pose_i = _se3([0.0, 0.0, 0.0])
    pose_j = _se3([0.0, 0.0, 0.0])
    bias_jacobian = torch.zeros((9, 6), dtype=torch.float64)
    bias_jacobian[8, 5] = 1.0

    residual = vio_preintegrated_imu_residual(
        from_pose=pose_i,
        to_pose=pose_j,
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=torch.zeros(3, dtype=torch.float64),
        delta_R=pp.identity_SO3(dtype=torch.float64),
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.zeros(3, dtype=torch.float64),
        dt_total=0.1,
        prev_acc_bias=torch.zeros(3, dtype=torch.float64),
        prev_gyro_bias=torch.tensor([0.0, 0.0, 0.02], dtype=torch.float64),
        curr_acc_bias=torch.zeros(3, dtype=torch.float64),
        curr_gyro_bias=torch.tensor([0.0, 0.0, 0.02 + yaw], dtype=torch.float64),
        linearized_acc_bias=torch.zeros(3, dtype=torch.float64),
        linearized_gyro_bias=torch.tensor([0.0, 0.0, 0.02], dtype=torch.float64),
        bias_jacobian=bias_jacobian,
    )

    assert torch.allclose(residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-6)


def test_vio_bias_jacobian_corrects_from_start_bias_to_edge_linearization():
    yaw = 0.1
    pose_i = _se3([0.0, 0.0, 0.0])
    pose_j = _se3([0.0, 0.0, 0.0], yaw_rad=yaw)
    bias_jacobian = torch.zeros((9, 6), dtype=torch.float64)
    bias_jacobian[8, 5] = 1.0

    residual = vio_preintegrated_imu_residual(
        from_pose=pose_i,
        to_pose=pose_j,
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=torch.zeros(3, dtype=torch.float64),
        delta_R=pp.identity_SO3(dtype=torch.float64),
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.zeros(3, dtype=torch.float64),
        dt_total=0.1,
        prev_acc_bias=torch.zeros(3, dtype=torch.float64),
        prev_gyro_bias=torch.tensor([0.0, 0.0, yaw], dtype=torch.float64),
        curr_acc_bias=torch.zeros(3, dtype=torch.float64),
        curr_gyro_bias=torch.tensor([0.0, 0.0, yaw], dtype=torch.float64),
        linearized_acc_bias=torch.zeros(3, dtype=torch.float64),
        linearized_gyro_bias=torch.zeros(3, dtype=torch.float64),
        bias_jacobian=bias_jacobian,
    )

    assert torch.allclose(residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-6)


def test_vio_residual_uses_bias_jacobian_to_correct_position_velocity_and_rotation():
    bias_delta_acc = torch.tensor([0.12, -0.04, 0.03], dtype=torch.float64)
    bias_delta_gyro = torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64)
    pose_i = _se3([0.0, 0.0, 0.0])
    pose_j = _se3(bias_delta_acc.tolist(), yaw_rad=float(bias_delta_gyro[2]))
    bias_jacobian = torch.zeros((9, 6), dtype=torch.float64)
    bias_jacobian[0:3, 0:3] = torch.eye(3, dtype=torch.float64)
    bias_jacobian[3:6, 0:3] = torch.eye(3, dtype=torch.float64)
    bias_jacobian[6:9, 3:6] = torch.eye(3, dtype=torch.float64)

    uncorrected = vio_preintegrated_imu_residual(
        from_pose=pose_i,
        to_pose=pose_j,
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=bias_delta_acc,
        delta_R=pp.identity_SO3(dtype=torch.float64),
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.zeros(3, dtype=torch.float64),
        dt_total=0.0,
    )
    corrected = vio_preintegrated_imu_residual(
        from_pose=pose_i,
        to_pose=pose_j,
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=bias_delta_acc,
        delta_R=pp.identity_SO3(dtype=torch.float64),
        delta_v=torch.zeros(3, dtype=torch.float64),
        delta_p=torch.zeros(3, dtype=torch.float64),
        dt_total=0.0,
        prev_acc_bias=bias_delta_acc,
        prev_gyro_bias=bias_delta_gyro,
        curr_acc_bias=bias_delta_acc,
        curr_gyro_bias=bias_delta_gyro,
        linearized_acc_bias=torch.zeros(3, dtype=torch.float64),
        linearized_gyro_bias=torch.zeros(3, dtype=torch.float64),
        bias_jacobian=bias_jacobian,
    )

    assert uncorrected.norm().item() > 0.1
    assert corrected.norm().item() < uncorrected.norm().item() * 1e-5


def test_bias_random_walk_residual_reports_acc_and_gyro_bias_steps():
    residual = vio_bias_random_walk_residual(
        prev_acc_bias=torch.tensor([0.1, 0.2, 0.3]),
        prev_gyro_bias=torch.tensor([0.01, 0.02, 0.03]),
        curr_acc_bias=torch.tensor([0.2, 0.1, 0.5]),
        curr_gyro_bias=torch.tensor([0.02, 0.01, 0.05]),
    )

    assert residual.shape == (2, 3)
    assert torch.allclose(residual[0], torch.tensor([0.1, -0.1, 0.2]))
    assert torch.allclose(residual[1], torch.tensor([0.01, -0.01, 0.02]))


def test_vio_covariance_blocks_keep_preintegration_order():
    cov9 = torch.diag(torch.arange(1, 10, dtype=torch.float64))

    blocks = vio_preintegrated_covariance_blocks(cov9)

    assert blocks.shape == (3, 3, 3)
    assert torch.allclose(blocks[0].diagonal(), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert torch.allclose(blocks[1].diagonal(), torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64))
    assert torch.allclose(blocks[2].diagonal(), torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))


def test_vio_covariance_matrix_preserves_cross_block_terms():
    cov9 = torch.eye(9, dtype=torch.float64)
    cov9[0, 3] = 0.25
    cov9[3, 0] = 0.25
    cov9[2, 8] = -0.10
    cov9[8, 2] = -0.10

    regularized = vio_preintegrated_covariance_matrix(cov9, diagonal_floor=1e-6)

    assert regularized.shape == (9, 9)
    assert regularized[0, 3].item() == pytest.approx(0.25)
    assert regularized[3, 0].item() == pytest.approx(0.25)
    assert regularized[2, 8].item() == pytest.approx(-0.10)
    assert regularized[8, 2].item() == pytest.approx(-0.10)
    assert torch.all(regularized.diagonal() >= 1.0 + 1e-6)


def test_weight_matrix_can_mix_visual_blocks_with_full_vio_covariance():
    visual_blocks = torch.stack(
        [
            torch.eye(3, dtype=torch.float64) * 2.0,
            torch.eye(3, dtype=torch.float64) * 3.0,
        ],
        dim=0,
    )
    vio_cov = torch.eye(9, dtype=torch.float64)
    vio_cov[0, 3] = 0.25
    vio_cov[3, 0] = 0.25
    bias_cov = torch.eye(6, dtype=torch.float64) * 4.0

    weight = build_weight_matrix_from_covariances(
        visual_blocks,
        full_covariances=[vio_cov, bias_cov],
    )

    assert weight.shape == (21, 21)
    assert torch.allclose(weight[0:3, 0:3], torch.eye(3, dtype=torch.float64) * 0.5)
    assert torch.allclose(weight[3:6, 3:6], torch.eye(3, dtype=torch.float64) / 3.0)
    assert weight[6, 9].abs().item() > 0.0
    assert weight[9, 6].abs().item() > 0.0
    assert torch.allclose(weight[-6:, -6:], torch.eye(6, dtype=torch.float64) * 0.25)


def test_gravity_rp_alignment_keeps_static_identity_ned_rotation():
    estimated = pp.identity_SO3(dtype=torch.float64)
    acc_body = torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float64)

    aligned = gravity_roll_pitch_aligned_rotation(
        estimated_body_to_world=estimated,
        acc_body=acc_body,
        gravity=9.81,
        correction_gain=1.0,
        acc_norm_tol=0.15,
    )

    assert aligned.active
    assert aligned.correction_angle_rad == 0.0
    assert torch.allclose(aligned.rotation.tensor(), estimated.tensor(), atol=1e-9)


def test_gravity_rp_alignment_corrects_tilt_without_using_gt_pose():
    estimated = pp.so3(torch.tensor([[0.0, 0.20, 0.0]], dtype=torch.float64)).Exp()
    acc_body = torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float64)

    aligned = gravity_roll_pitch_aligned_rotation(
        estimated_body_to_world=estimated,
        acc_body=acc_body,
        gravity=9.81,
        correction_gain=1.0,
        acc_norm_tol=0.15,
    )

    target_up_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
    measured_up_world = aligned.rotation.double().Act(acc_body.reshape(3) / acc_body.norm())

    assert aligned.active
    assert aligned.correction_angle_rad > 0.0
    assert torch.allclose(measured_up_world, target_up_world, atol=1e-6)


def test_gravity_rp_alignment_rejects_dynamic_acceleration_norm():
    estimated = pp.so3(torch.tensor([[0.0, 0.20, 0.0]], dtype=torch.float64)).Exp()
    acc_body = torch.tensor([[3.0, 0.0, -9.81]], dtype=torch.float64)

    aligned = gravity_roll_pitch_aligned_rotation(
        estimated_body_to_world=estimated,
        acc_body=acc_body,
        gravity=9.81,
        correction_gain=1.0,
        acc_norm_tol=0.01,
    )

    assert not aligned.active
    assert aligned.correction_angle_rad == 0.0
    assert torch.allclose(aligned.rotation.tensor(), estimated.tensor(), atol=1e-9)


def test_integrate_gyro_attitude_world_keeps_identity_without_rotation():
    R0 = pp.identity_SO3(dtype=torch.float64)
    time_ns = torch.tensor([0, 10_000_000], dtype=torch.long)
    gyro = torch.zeros((2, 3), dtype=torch.float64)

    R1 = integrate_gyro_attitude_world(R0, time_ns, gyro)

    assert torch.allclose(R1.tensor(), R0.tensor(), atol=1e-9)


def test_integrate_gyro_attitude_world_accumulates_yaw_rate():
    R0 = pp.identity_SO3(dtype=torch.float64)
    time_ns = torch.tensor([0, 1_000_000_000], dtype=torch.long)
    gyro = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.2]], dtype=torch.float64)

    R1 = integrate_gyro_attitude_world(R0, time_ns, gyro)

    expected = pp.so3(torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float64)).Exp()
    assert torch.allclose(R1.tensor(), expected.tensor(), atol=1e-6)


def test_macvo_accepts_imu_integrated_estinit_gravity_pose_source():
    source = Path("Odometry/MACVO.py").read_text(encoding="utf-8")

    assert '"imu_integrated_estinit"' in source
    assert "integrate_gyro_attitude_world" in source
    assert "imu_attitude_source_active" in source


def test_macvo_standard_mode_uses_pose_free_local_preintegrator():
    source = Path("Odometry/MACVO.py").read_text(encoding="utf-8")

    assert "preintegrate_imu_local_frame" in source
    assert "if gravity_in_residual:" in source
    standard_call = source.index("preint = preintegrate_imu_local_frame(")
    legacy_call = source.index("preint = preintegrate_imu(", standard_call)
    standard_block = source[standard_call:legacy_call]
    assert "R0_world" not in standard_block
    assert "gravity=" not in standard_block


def test_preintegrated_vio_factor_requires_full_inertial_mode():
    assert should_enable_preintegrated_vio_factor(
        use_imu_rotation=True,
        use_imu_translation=True,
        dt_total=0.1,
    )
    assert not should_enable_preintegrated_vio_factor(
        use_imu_rotation=True,
        use_imu_translation=False,
        dt_total=0.1,
    )
    assert not should_enable_preintegrated_vio_factor(
        use_imu_rotation=False,
        use_imu_translation=True,
        dt_total=0.1,
    )
    assert not should_enable_preintegrated_vio_factor(
        use_imu_rotation=True,
        use_imu_translation=True,
        dt_total=0.0,
    )


def test_macvo_stores_vio_factor_only_after_full_mode_gate():
    source = Path("Odometry/MACVO.py").read_text(encoding="utf-8")

    assert "should_enable_preintegrated_vio_factor" in source
    assert "vio_factor_allowed" in source
    assert '"delta_rotvec": preint_rotvec' in source
    assert '"delta_rotvec": rot_prior_vec' not in source


def test_reproj_disp_graph_appends_vio_factor_without_replacing_visual_residuals():
    code = r'''
import torch, pypose as pp
from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput, ReprojDisp_TwoFramePGO

n_points = 2
intrinsics = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
baseline = torch.tensor([0.225])
from_pose = pp.identity_SE3(1)
init_motion = pp.SE3(torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

points_tw = torch.tensor([[2.0, 0.1, 0.1], [2.2, -0.1, 0.05]])
points = PointNode.init({
    "pos_Tw": points_tw,
    "cov_Tw": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.01,
    "color": torch.zeros(n_points, 3, dtype=torch.uint8),
})

pixel2_uv = torch.tensor([[336.0, 256.0], [305.0, 247.0]])
observations = MatchObs.init({
    "pixel1_uv": pixel2_uv.clone(),
    "pixel1_d": torch.ones(n_points, 1) * 2.0,
    "pixel2_uv": pixel2_uv.clone(),
    "pixel2_d": torch.ones(n_points, 1) * 2.0,
    "pixel1_disp": torch.ones(n_points, 1) * 36.0,
    "pixel2_disp": torch.ones(n_points, 1) * 36.0,
    "pixel1_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel2_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel1_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel1_disp_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_disp_cov": torch.ones(n_points, 1) * 0.1,
    "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
    "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
})

graph_data = GraphInput(
    torch.tensor([1]), torch.tensor([0]), init_motion, from_pose, baseline,
    observations, points, intrinsics, torch.zeros(n_points, dtype=torch.long), "cpu",
    imu_vio_factor_enable=True,
    imu_vio_prev_velocity_world=torch.tensor([1.0, 0.0, 0.0]),
    imu_vio_curr_velocity_init_world=torch.tensor([1.0, 0.0, 0.0]),
    imu_vio_delta_rotvec=torch.zeros(3),
    imu_vio_delta_v=torch.zeros(3),
    imu_vio_delta_p=torch.zeros(3),
    imu_vio_cov=torch.diag(torch.arange(1, 10, dtype=torch.float32)),
    imu_vio_dt=torch.tensor([0.1]),
)

graph = ReprojDisp_TwoFramePGO(graph_data).to(dtype=torch.double)
residual = graph()
cov_blocks = graph.covariance_array()
output = graph.write_back()

assert residual.shape == (n_points + 3, 3)
assert cov_blocks.shape == (n_points + 3, 3, 3)
assert output.velocity_world is not None
assert torch.allclose(output.velocity_world.reshape(3), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
assert output.vio_factor_active is True
assert output.imu_factor_mode == "preintegrated_vio"
assert output.imu_residual_rows == 3
assert output.use_imu_rotation is True
assert output.use_imu_translation is True
assert torch.allclose(cov_blocks[-3].diagonal(), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
assert torch.allclose(cov_blocks[-2].diagonal(), torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64))
assert torch.allclose(cov_blocks[-1].diagonal(), torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))
'''
    result = subprocess.run([sys.executable, "-c", code], cwd="/home/admin1/macvo-dev", text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reproj_disp_graph_vio_alpha_scales_actual_imu_objective():
    code = r'''
import torch, pypose as pp
from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput, ReprojDisp_TwoFramePGO

n_points = 1
intrinsics = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
baseline = torch.tensor([0.225])
from_pose = pp.identity_SE3(1)
init_motion = pp.SE3(torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

points = PointNode.init({
    "pos_Tw": torch.tensor([[2.0, 0.1, 0.1]]),
    "cov_Tw": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.01,
    "color": torch.zeros(n_points, 3, dtype=torch.uint8),
})
observations = MatchObs.init({
    "pixel1_uv": torch.tensor([[336.0, 256.0]]),
    "pixel1_d": torch.ones(n_points, 1) * 2.0,
    "pixel2_uv": torch.tensor([[336.0, 256.0]]),
    "pixel2_d": torch.ones(n_points, 1) * 2.0,
    "pixel1_disp": torch.ones(n_points, 1) * 36.0,
    "pixel2_disp": torch.ones(n_points, 1) * 36.0,
    "pixel1_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel2_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel1_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel1_disp_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_disp_cov": torch.ones(n_points, 1) * 0.1,
    "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
    "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
})

def make_graph(alpha_p):
    graph_data = GraphInput(
        torch.tensor([1]), torch.tensor([0]), init_motion, from_pose, baseline,
        observations, points, intrinsics, torch.zeros(n_points, dtype=torch.long), "cpu",
        imu_vio_factor_enable=True,
        imu_vio_prev_velocity_world=torch.zeros(3),
        imu_vio_curr_velocity_init_world=torch.zeros(3),
        imu_vio_delta_rotvec=torch.zeros(3),
        imu_vio_delta_v=torch.zeros(3),
        imu_vio_delta_p=torch.zeros(3),
        imu_vio_cov=torch.eye(9),
        imu_vio_dt=torch.tensor([0.1]),
        imu_vio_alpha_p=alpha_p,
        imu_vio_alpha_v=1.0,
        imu_vio_alpha_R=1.0,
    )
    return ReprojDisp_TwoFramePGO(graph_data).to(dtype=torch.double)

def imu_cost(graph):
    residual_matrix = graph().detach().clone()
    residual = residual_matrix.reshape(-1).double()
    weight = graph.weight_matrix().detach().clone().double()
    r_imu = residual[-9:]
    w_imu = weight[-9:, -9:]
    return float((r_imu @ (w_imu @ r_imu)).item()), residual_matrix

cost_full, residual_full = imu_cost(make_graph(1.0))
cost_zero, residual_zero = imu_cost(make_graph(0.0))

assert cost_full > 1e-3
assert cost_zero < 1e-12
assert residual_full[1, 0].abs().item() > 0.1
assert residual_zero[1].abs().max().item() < 1e-12
'''
    result = subprocess.run([sys.executable, "-c", code], cwd="/home/admin1/macvo-dev", text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reproj_disp_legacy_translation_prior_uses_sensor_to_imu_extrinsic():
    code = r'''
import math
import torch, pypose as pp
from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput, ReprojDisp_TwoFramePGO

n_points = 1
intrinsics = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
baseline = torch.tensor([0.225])
from_pose = pp.identity_SE3(1)
half = math.pi / 4.0
init_motion = pp.SE3(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.sin(half), math.cos(half)]]))
sensor_T_imu = pp.SE3(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

points = PointNode.init({
    "pos_Tw": torch.tensor([[2.0, 0.1, 0.1]]),
    "cov_Tw": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.01,
    "color": torch.zeros(n_points, 3, dtype=torch.uint8),
})
observations = MatchObs.init({
    "pixel1_uv": torch.tensor([[336.0, 256.0]]),
    "pixel1_d": torch.ones(n_points, 1) * 2.0,
    "pixel2_uv": torch.tensor([[336.0, 256.0]]),
    "pixel2_d": torch.ones(n_points, 1) * 2.0,
    "pixel1_disp": torch.ones(n_points, 1) * 36.0,
    "pixel2_disp": torch.ones(n_points, 1) * 36.0,
    "pixel1_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel2_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel1_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel1_disp_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_disp_cov": torch.ones(n_points, 1) * 0.1,
    "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
    "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
})

graph_data = GraphInput(
    torch.tensor([1]), torch.tensor([0]), init_motion, from_pose, baseline,
    observations, points, intrinsics, torch.zeros(n_points, dtype=torch.long), "cpu",
    imu_trans_prior=torch.tensor([-1.0, 1.0, 0.0]),
    imu_trans_cov=torch.eye(3),
    imu_vio_sensor_T_imu=sensor_T_imu.tensor(),
)

graph = ReprojDisp_TwoFramePGO(graph_data).to(dtype=torch.double)
assert torch.allclose(graph._imu_trans_residual(), torch.zeros((1, 3), dtype=torch.float64), atol=1e-6)
'''
    result = subprocess.run([sys.executable, "-c", code], cwd="/home/admin1/macvo-dev", text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reproj_disp_graph_appends_vio_bias_state_rows_and_outputs_biases():
    code = r'''
import torch, pypose as pp
from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput, ReprojDisp_TwoFramePGO

n_points = 2
intrinsics = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
baseline = torch.tensor([0.225])
from_pose = pp.identity_SE3(1)
init_motion = pp.SE3(torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

points_tw = torch.tensor([[2.0, 0.1, 0.1], [2.2, -0.1, 0.05]])
points = PointNode.init({
    "pos_Tw": points_tw,
    "cov_Tw": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.01,
    "color": torch.zeros(n_points, 3, dtype=torch.uint8),
})

pixel2_uv = torch.tensor([[336.0, 256.0], [305.0, 247.0]])
observations = MatchObs.init({
    "pixel1_uv": pixel2_uv.clone(),
    "pixel1_d": torch.ones(n_points, 1) * 2.0,
    "pixel2_uv": pixel2_uv.clone(),
    "pixel2_d": torch.ones(n_points, 1) * 2.0,
    "pixel1_disp": torch.ones(n_points, 1) * 36.0,
    "pixel2_disp": torch.ones(n_points, 1) * 36.0,
    "pixel1_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel2_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel1_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel1_disp_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_disp_cov": torch.ones(n_points, 1) * 0.1,
    "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
    "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
})

graph_data = GraphInput(
    torch.tensor([1]), torch.tensor([0]), init_motion, from_pose, baseline,
    observations, points, intrinsics, torch.zeros(n_points, dtype=torch.long), "cpu",
    imu_vio_factor_enable=True,
    imu_vio_prev_velocity_world=torch.tensor([1.0, 0.0, 0.0]),
    imu_vio_curr_velocity_init_world=torch.tensor([1.0, 0.0, 0.0]),
    imu_vio_prev_acc_bias=torch.zeros(3),
    imu_vio_prev_gyro_bias=torch.zeros(3),
    imu_vio_curr_acc_bias_init=torch.zeros(3),
    imu_vio_curr_gyro_bias_init=torch.zeros(3),
    imu_vio_linearized_acc_bias=torch.zeros(3),
    imu_vio_linearized_gyro_bias=torch.zeros(3),
    imu_vio_bias_jacobian=torch.zeros(9, 6),
    imu_vio_bias_rw_cov=torch.eye(6, dtype=torch.float32) * 0.05,
    imu_vio_delta_rotvec=torch.zeros(3),
    imu_vio_delta_v=torch.zeros(3),
    imu_vio_delta_p=torch.zeros(3),
    imu_vio_cov=torch.eye(9, dtype=torch.float32) * 0.01,
    imu_vio_dt=torch.tensor([0.1]),
)

graph = ReprojDisp_TwoFramePGO(graph_data).to(dtype=torch.double)
residual = graph()
cov_blocks = graph.covariance_array()
output = graph.write_back()

assert residual.shape == (n_points + 5, 3)
assert cov_blocks.shape == (n_points + 5, 3, 3)
assert output.velocity_world is not None
assert output.acc_bias is not None
assert output.gyro_bias is not None
assert torch.allclose(output.acc_bias.reshape(3), torch.zeros(3, dtype=torch.float64))
assert torch.allclose(output.gyro_bias.reshape(3), torch.zeros(3, dtype=torch.float64))
assert output.vio_factor_active is True
assert output.vio_bias_state_active is True
assert output.imu_residual_rows == 5
assert torch.allclose(cov_blocks[-2].diagonal(), torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64))
assert torch.allclose(cov_blocks[-1].diagonal(), torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64))
'''
    result = subprocess.run([sys.executable, "-c", code], cwd="/home/admin1/macvo-dev", text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_twoframe_terminal_bias_does_not_correct_start_edge_preintegration():
    code = r'''
from types import SimpleNamespace
import torch, pypose as pp
from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput, ReprojDisp_TwoFramePGO
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Utility.Point import point2pixel_NED

n_points = 6
intrinsics = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
baseline = torch.tensor([0.225])
target_yaw = 0.08
half = target_yaw * 0.5
from_pose = pp.identity_SE3(1)
init_motion = pp.SE3(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, torch.sin(torch.tensor(half)), torch.cos(torch.tensor(half))]]))

points_tc = torch.tensor([
    [2.0,  0.12,  0.08],
    [2.2, -0.15,  0.04],
    [1.8,  0.20, -0.05],
    [2.5, -0.25, -0.02],
    [2.1,  0.05,  0.14],
    [2.4,  0.18, -0.11],
], dtype=torch.float64)
points_tw = (init_motion.double() * points_tc).float()
pixel2_uv = point2pixel_NED(points_tc.float(), intrinsics)
pixel2_disp = (intrinsics[0, 0] * baseline / points_tc[:, 0]).reshape(-1, 1).float()
points = PointNode.init({
    "pos_Tw": points_tw,
    "cov_Tw": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 1e-5,
    "color": torch.zeros(n_points, 3, dtype=torch.uint8),
})
observations = MatchObs.init({
    "pixel1_uv": pixel2_uv.clone(),
    "pixel1_d": torch.ones(n_points, 1) * 2.0,
    "pixel2_uv": pixel2_uv.clone(),
    "pixel2_d": torch.ones(n_points, 1) * 2.0,
    "pixel1_disp": pixel2_disp.clone(),
    "pixel2_disp": pixel2_disp.clone(),
    "pixel1_uv_cov": torch.ones(n_points, 3) * 1e-5,
    "pixel2_uv_cov": torch.ones(n_points, 3) * 1e-5,
    "pixel1_d_cov": torch.ones(n_points, 1) * 1e-5,
    "pixel2_d_cov": torch.ones(n_points, 1) * 1e-5,
    "pixel1_disp_cov": torch.ones(n_points, 1) * 1e-5,
    "pixel2_disp_cov": torch.ones(n_points, 1) * 1e-5,
    "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 1e-5,
    "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 1e-5,
})
bias_jacobian = torch.zeros(9, 6)
bias_jacobian[8, 5] = 1.0
graph_data = GraphInput(
    torch.tensor([1]), torch.tensor([0]), init_motion, from_pose, baseline,
    observations, points, intrinsics, torch.zeros(n_points, dtype=torch.long), "cpu",
    imu_vio_factor_enable=True,
    imu_vio_prev_velocity_world=torch.zeros(3),
    imu_vio_curr_velocity_init_world=torch.zeros(3),
    imu_vio_prev_acc_bias=torch.zeros(3),
    imu_vio_prev_gyro_bias=torch.zeros(3),
    imu_vio_curr_acc_bias_init=torch.zeros(3),
    imu_vio_curr_gyro_bias_init=torch.zeros(3),
    imu_vio_linearized_acc_bias=torch.zeros(3),
    imu_vio_linearized_gyro_bias=torch.zeros(3),
    imu_vio_bias_jacobian=bias_jacobian,
    imu_vio_bias_rw_cov=torch.eye(6, dtype=torch.float32) * 10.0,
    imu_vio_delta_rotvec=torch.zeros(3),
    imu_vio_delta_v=torch.zeros(3),
    imu_vio_delta_p=torch.zeros(3),
    imu_vio_cov=torch.eye(9, dtype=torch.float32) * 0.01,
    imu_vio_dt=torch.tensor([0.1]),
)
graph_before = ReprojDisp_TwoFramePGO(graph_data).to(dtype=torch.double)
initial_rot_residual = graph_before._imu_vio_residual()[2].norm().item()
param_names = [name for name, _ in graph_before.named_parameters()]
loss_probe = (graph_before._imu_vio_residual() ** 2).sum()
loss_probe.backward()
gyro_grad_tensor = graph_before.gyro_bias2opt.grad
gyro_grad = None if gyro_grad_tensor is None else gyro_grad_tensor.detach().clone()
graph_before.zero_grad(set_to_none=True)

config = SimpleNamespace(
    autodiff=True,
    graph_type="disp",
    vectorize=True,
    device="cpu",
    imu_factor_mode="preintegrated_vio",
)
context = TwoFrame_PGO.init_context(config)
_, output = TwoFrame_PGO._optimize(context, graph_data)
optimized_yaw_bias = float(output.gyro_bias.reshape(3)[2])
optimized_motion = output.motion.tensor().reshape(-1).tolist()

graph_data_after = graph_data
graph_data_after.imu_vio_curr_gyro_bias_init = output.gyro_bias.float()
graph_after = ReprojDisp_TwoFramePGO(graph_data_after).to(dtype=torch.double)
final_rot_residual = graph_after._imu_vio_residual()[2].norm().item()

debug = (
    f"initial_rot_residual={initial_rot_residual} "
    f"final_rot_residual={final_rot_residual} "
    f"optimized_yaw_bias={optimized_yaw_bias} target_yaw={target_yaw} "
    f"final_loss={output.final_loss} optimized_motion={optimized_motion} "
    f"param_names={param_names} gyro_grad={None if gyro_grad is None else gyro_grad.tolist()}"
)
assert "gyro_bias2opt" in param_names, debug
assert gyro_grad is None or abs(float(gyro_grad[2])) < 1e-9, debug
assert output.vio_bias_state_active is True, debug
assert abs(optimized_yaw_bias) < target_yaw * 0.25, debug
assert final_rot_residual >= initial_rot_residual * 0.99, debug
'''
    result = subprocess.run([sys.executable, "-c", code], cwd="/home/admin1/macvo-dev", text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_vio_vector_refinement_updates_velocity_acc_bias_and_gyro_bias_states():
    code = r'''
import torch
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO

class SyntheticVIOGraph(torch.nn.Module):
    def __init__(self, case: str) -> None:
        super().__init__()
        self.case = case
        self.use_vio_imu_factor = True
        self.use_vio_bias_state = True
        self.velocity2opt = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.acc_bias2opt = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.gyro_bias2opt = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def _imu_vio_residual(self) -> torch.Tensor:
        residual = torch.zeros(3, 3, dtype=torch.float64)
        if self.case == "velocity":
            residual[1, 0] = self.velocity2opt[0] - 0.3
        elif self.case == "acc_bias":
            residual[1, 0] = 0.2 + self.acc_bias2opt[0]
        elif self.case == "gyro_bias":
            residual[2, 2] = 0.08 - self.gyro_bias2opt[2]
        else:
            raise AssertionError(self.case)
        return residual

    def _imu_vio_bias_residual(self) -> torch.Tensor:
        return torch.stack([self.acc_bias2opt, self.gyro_bias2opt], dim=0)

weight = torch.eye(15, dtype=torch.float64)
weight[:9, :9] *= 100.0
weight[9:, 9:] *= 0.1

velocity_graph = SyntheticVIOGraph("velocity")
TwoFrame_PGO._refine_vio_vector_states(velocity_graph, weight)
assert abs(float(velocity_graph.velocity2opt[0]) - 0.3) < 1e-6
assert velocity_graph._imu_vio_residual().norm().item() < 1e-6

acc_graph = SyntheticVIOGraph("acc_bias")
initial_acc_residual = acc_graph._imu_vio_residual().norm().item()
TwoFrame_PGO._refine_vio_vector_states(acc_graph, weight)
assert acc_graph.acc_bias2opt[0].item() < -0.19
assert acc_graph._imu_vio_residual().norm().item() < initial_acc_residual * 0.01

gyro_graph = SyntheticVIOGraph("gyro_bias")
initial_gyro_residual = gyro_graph._imu_vio_residual().norm().item()
TwoFrame_PGO._refine_vio_vector_states(gyro_graph, weight)
assert gyro_graph.gyro_bias2opt[2].item() > 0.079
assert gyro_graph._imu_vio_residual().norm().item() < initial_gyro_residual * 0.01
'''
    result = subprocess.run([sys.executable, "-c", code], cwd="/home/admin1/macvo-dev", text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_twoframe_optimizer_runs_with_vio_pose_and_velocity_parameters():
    code = r'''
from types import SimpleNamespace
import torch, pypose as pp
from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO

n_points = 4
intrinsics = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
baseline = torch.tensor([0.225])
from_pose = pp.identity_SE3(1)
init_motion = pp.SE3(torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

points_tw = torch.tensor([[2.0, 0.1, 0.1], [2.2, -0.1, 0.05], [1.8, 0.2, -0.05], [2.5, -0.2, 0.0]])
points = PointNode.init({
    "pos_Tw": points_tw,
    "cov_Tw": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.01,
    "color": torch.zeros(n_points, 3, dtype=torch.uint8),
})

pixel2_uv = torch.tensor([[336.0, 256.0], [305.0, 247.0], [356.0, 231.0], [294.0, 240.0]])
observations = MatchObs.init({
    "pixel1_uv": pixel2_uv.clone(),
    "pixel1_d": torch.ones(n_points, 1) * 2.0,
    "pixel2_uv": pixel2_uv.clone(),
    "pixel2_d": torch.ones(n_points, 1) * 2.0,
    "pixel1_disp": torch.ones(n_points, 1) * 36.0,
    "pixel2_disp": torch.ones(n_points, 1) * 36.0,
    "pixel1_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel2_uv_cov": torch.ones(n_points, 3) * 0.1,
    "pixel1_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_d_cov": torch.ones(n_points, 1) * 0.1,
    "pixel1_disp_cov": torch.ones(n_points, 1) * 0.1,
    "pixel2_disp_cov": torch.ones(n_points, 1) * 0.1,
    "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
    "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.1,
})

graph_data = GraphInput(
    torch.tensor([1]), torch.tensor([0]), init_motion, from_pose, baseline,
    observations, points, intrinsics, torch.zeros(n_points, dtype=torch.long), "cpu",
    imu_vio_factor_enable=True,
    imu_vio_prev_velocity_world=torch.tensor([1.0, 0.0, 0.0]),
    imu_vio_curr_velocity_init_world=torch.tensor([1.0, 0.0, 0.0]),
    imu_vio_delta_rotvec=torch.zeros(3),
    imu_vio_delta_v=torch.zeros(3),
    imu_vio_delta_p=torch.zeros(3),
    imu_vio_cov=torch.eye(9, dtype=torch.float32) * 0.01,
    imu_vio_dt=torch.tensor([0.1]),
)

config = SimpleNamespace(
    autodiff=True,
    graph_type="disp",
    vectorize=True,
    device="cpu",
    imu_factor_mode="preintegrated_vio",
)
context = TwoFrame_PGO.init_context(config)
_, output = TwoFrame_PGO._optimize(context, graph_data)

assert output.motion.shape[-1] == 7
assert output.velocity_world is not None
assert output.velocity_world.numel() == 3
assert output.vio_factor_active is True
assert output.imu_factor_mode == "preintegrated_vio"
assert output.imu_residual_rows == 3
assert output.use_imu_rotation is True
assert output.use_imu_translation is True
assert output.num_visual_residuals == n_points
assert output.r_p_whitened_norm is not None
assert output.r_v_whitened_norm is not None
assert output.r_R_whitened_norm is not None
assert output.imu_vio_whitened_norm is not None
assert output.imu_vio_raw_norm is not None
assert output.imu_vel_loss is not None
assert output.imu_vio_cov_trace is not None
assert output.imu_vio_weight_trace is not None
assert output.imu_vio_weight_diag_min is not None
assert output.imu_vio_weight_diag_max is not None
'''
    result = subprocess.run([sys.executable, "-c", code], cwd="/home/admin1/macvo-dev", text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr

import importlib.util
import inspect
import sys
from pathlib import Path

import pypose as pp
import pytest
import torch

from Module.Map import MatchObs, PointNode
from Module.Map.VisualMap import VisualMap
from Module.Optimization.TwoFramePGO.Graphs import (
    GraphInput,
    LocalWindowGraphInput,
    LocalWindowInertialGraph,
    ReprojDisp_TwoFramePGO,
)
from Utility.IMUKinematics import (
    LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION,
    STANDARD_LOCAL_FRAME_PREINTEGRATION,
    normalize_gravity_handling,
    propagate_imu_velocity_world,
    vio_preintegrated_imu_residual,
)


def _load_preintegration_module():
    module_path = Path("Module/IMUPreintegration.py").resolve()
    spec = importlib.util.spec_from_file_location("imu_preintegration_gravity_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_preintegrate_imu():
    return _load_preintegration_module().preintegrate_imu


def _stationary_interval(dt_s: float = 0.1):
    time_ns = torch.tensor([0, int(dt_s * 1e9)], dtype=torch.long)
    acc = torch.tensor([[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]], dtype=torch.float32)
    gyro = torch.zeros((2, 3), dtype=torch.float32)
    return time_ns, acc, gyro


def _identity_pose(dtype=torch.float64):
    return pp.identity_SE3(1, dtype=dtype)


def _stationary_graph_input() -> GraphInput:
    observations = MatchObs.init(
        {
            "pixel1_uv": torch.tensor([[50.0, 50.0]]),
            "pixel2_uv": torch.tensor([[50.0, 50.0]]),
            "pixel1_d": torch.tensor([[2.0]]),
            "pixel2_d": torch.tensor([[2.0]]),
            "pixel1_disp": torch.tensor([[5.0]]),
            "pixel2_disp": torch.tensor([[5.0]]),
            "pixel1_disp_cov": torch.ones((1, 1)),
            "pixel2_disp_cov": torch.ones((1, 1)),
            "obs1_covTc": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "obs2_covTc": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "pixel1_uv_cov": torch.tensor([[1.0, 1.0, 0.0]]),
            "pixel2_uv_cov": torch.tensor([[1.0, 1.0, 0.0]]),
            "pixel1_d_cov": torch.ones((1, 1)),
            "pixel2_d_cov": torch.ones((1, 1)),
        }
    )
    points = PointNode.init(
        {
            "pos_Tw": torch.tensor([[2.0, 0.0, 0.0]]),
            "cov_Tw": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "color": torch.zeros((1, 3), dtype=torch.uint8),
        }
    )
    pose = pp.identity_SE3(1, dtype=torch.float32)
    return GraphInput(
        frame_idx=torch.tensor([1]),
        from_idx=torch.tensor([0]),
        init_motion=pose,
        from_pose=pose,
        baseline=torch.tensor([0.1]),
        observations=observations,
        points=points,
        images_intrinsic=torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        edges_index=torch.zeros(1, dtype=torch.long),
        device="cpu",
        imu_vio_factor_enable=True,
        imu_vio_prev_velocity_world=torch.zeros(3),
        imu_vio_curr_velocity_init_world=torch.zeros(3),
        imu_vio_prev_acc_bias=torch.zeros(3),
        imu_vio_prev_gyro_bias=torch.zeros(3),
        imu_vio_curr_acc_bias_init=torch.zeros(3),
        imu_vio_curr_gyro_bias_init=torch.zeros(3),
        imu_vio_linearized_acc_bias=torch.zeros(3),
        imu_vio_linearized_gyro_bias=torch.zeros(3),
        imu_vio_bias_jacobian=torch.zeros((9, 6)),
        imu_vio_bias_rw_cov=torch.eye(6),
        imu_vio_delta_rotvec=torch.zeros(3),
        imu_vio_delta_v=torch.tensor([0.0, 0.0, -0.981]),
        imu_vio_delta_p=torch.tensor([0.0, 0.0, -0.04905]),
        imu_vio_cov=torch.eye(9),
        imu_vio_dt=torch.tensor([0.1]),
        imu_vio_sensor_T_imu=pp.identity_SE3(1).tensor(),
        imu_vio_gravity_world=torch.tensor([0.0, 0.0, 9.81]),
        imu_vio_gravity_in_residual=True,
    )


def test_preintegrator_rejects_unknown_gravity_handling():
    preintegrate_imu = _load_preintegrate_imu()
    time_ns, acc, gyro = _stationary_interval()

    with pytest.raises(ValueError, match="gravity_handling"):
        preintegrate_imu(
            time_ns=time_ns,
            acc=acc,
            gyro=gyro,
            R0_world=pp.identity_SO3(dtype=torch.float64),
            gravity=9.81,
            gravity_handling="unknown",
        )


def test_explicit_gravity_mode_names_keep_legacy_aliases_compatible():
    assert normalize_gravity_handling("preintegration") == (
        LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION
    )
    assert normalize_gravity_handling("residual") == STANDARD_LOCAL_FRAME_PREINTEGRATION
    assert normalize_gravity_handling(LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION) == (
        LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION
    )
    assert normalize_gravity_handling(STANDARD_LOCAL_FRAME_PREINTEGRATION) == (
        STANDARD_LOCAL_FRAME_PREINTEGRATION
    )


def test_standard_local_frame_api_has_no_world_pose_or_gravity_input():
    module = _load_preintegration_module()
    signature = inspect.signature(module.preintegrate_imu_local_frame)

    assert "R0_world" not in signature.parameters
    assert "gravity" not in signature.parameters
    assert set(signature.parameters) == {
        "time_ns",
        "acc",
        "gyro",
        "sigma_acc",
        "sigma_gyro",
        "sigma_acc_w",
        "sigma_gyro_w",
        "gyro_bias",
        "acc_bias",
    }


def test_standard_local_frame_delta_is_invariant_to_external_attitude():
    preintegrate_imu = _load_preintegrate_imu()
    time_ns = torch.tensor([0, 10_000_000, 20_000_000], dtype=torch.long)
    acc = torch.tensor(
        [[0.2, -0.1, -9.7], [0.3, -0.2, -9.6], [0.4, -0.1, -9.5]],
        dtype=torch.float32,
    )
    gyro = torch.tensor(
        [[0.1, -0.2, 0.3], [0.12, -0.18, 0.28], [0.11, -0.17, 0.31]],
        dtype=torch.float32,
    )
    tilted = pp.so3(torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float64)).Exp()

    identity = preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=pp.identity_SO3(dtype=torch.float64),
        gravity=9.81,
        gravity_handling=STANDARD_LOCAL_FRAME_PREINTEGRATION,
    )
    nonzero_pose = preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=tilted,
        gravity=-3.7,
        gravity_handling=STANDARD_LOCAL_FRAME_PREINTEGRATION,
    )

    assert torch.equal(identity.delta_R.tensor(), nonzero_pose.delta_R.tensor())
    assert torch.equal(identity.delta_v, nonzero_pose.delta_v)
    assert torch.equal(identity.delta_p, nonzero_pose.delta_p)
    assert torch.equal(identity.cov, nonzero_pose.cov)
    assert torch.equal(identity.bias_jacobian, nonzero_pose.bias_jacobian)


def test_local_frame_api_matches_standard_generic_path_exactly():
    module = _load_preintegration_module()
    time_ns, acc, gyro = _stationary_interval()
    kwargs = dict(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        sigma_acc=[0.01, 0.02, 0.03],
        sigma_gyro=[0.001, 0.002, 0.003],
        sigma_acc_w=[0.0001, 0.0002, 0.0003],
        sigma_gyro_w=[0.00001, 0.00002, 0.00003],
        acc_bias=torch.tensor([0.1, -0.2, 0.3]),
        gyro_bias=torch.tensor([0.01, -0.02, 0.03]),
    )
    local = module.preintegrate_imu_local_frame(**kwargs)
    generic = module.preintegrate_imu(
        **kwargs,
        R0_world=pp.so3(torch.tensor([[0.2, 0.1, -0.3]], dtype=torch.float64)).Exp(),
        gravity=42.0,
        gravity_handling="residual",
    )

    assert torch.equal(local.delta_R.tensor(), generic.delta_R.tensor())
    assert torch.equal(local.delta_v, generic.delta_v)
    assert torch.equal(local.delta_p, generic.delta_p)
    assert torch.equal(local.cov, generic.cov)
    assert torch.equal(local.bias_jacobian, generic.bias_jacobian)
    assert torch.equal(local.bias_rw_cov, generic.bias_rw_cov)


def test_stationary_interval_closes_in_both_gravity_conventions():
    preintegrate_imu = _load_preintegrate_imu()
    time_ns, acc, gyro = _stationary_interval()
    gravity_world = torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64)

    legacy = preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=pp.identity_SO3(dtype=torch.float64),
        gravity=9.81,
        gravity_handling="preintegration",
    )
    standard = preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=None,
        gravity=9.81,
        gravity_handling="residual",
    )

    assert torch.allclose(legacy.delta_v, torch.zeros(3), atol=1e-7)
    assert torch.allclose(legacy.delta_p, torch.zeros(3), atol=1e-7)
    assert torch.allclose(
        standard.delta_v.double(),
        torch.tensor([0.0, 0.0, -0.981], dtype=torch.float64),
        atol=1e-6,
    )
    assert torch.allclose(
        standard.delta_p.double(),
        torch.tensor([0.0, 0.0, -0.04905], dtype=torch.float64),
        atol=1e-6,
    )

    common = dict(
        from_pose=_identity_pose(),
        to_pose=_identity_pose(),
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=torch.zeros(3, dtype=torch.float64),
        delta_R=pp.identity_SO3(dtype=torch.float64),
        dt_total=0.1,
    )
    legacy_residual = vio_preintegrated_imu_residual(
        **common,
        delta_v=legacy.delta_v,
        delta_p=legacy.delta_p,
        gravity_world=gravity_world,
        gravity_handling="preintegration",
    )
    standard_residual = vio_preintegrated_imu_residual(
        **common,
        delta_v=standard.delta_v,
        delta_p=standard.delta_p,
        gravity_world=gravity_world,
        gravity_handling="residual",
    )

    assert torch.allclose(legacy_residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-7)
    assert torch.allclose(standard_residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-7)


def test_residual_gravity_velocity_propagation_keeps_stationary_state():
    velocity = propagate_imu_velocity_world(
        velocity_world=torch.zeros(3),
        delta_v_body=torch.tensor([0.0, 0.0, -0.981]),
        R_body_to_world=torch.eye(3),
        gravity_world=torch.tensor([0.0, 0.0, 9.81]),
        dt_total=0.1,
        gravity_handling="residual",
    )

    assert torch.allclose(velocity, torch.zeros(3), atol=1e-7)


def test_residual_gravity_matches_nontrivial_constant_acceleration_motion():
    dt = 0.2
    residual = vio_preintegrated_imu_residual(
        from_pose=_identity_pose(),
        to_pose=pp.SE3(
            torch.tensor([[0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        ),
        prev_velocity_world=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        curr_velocity_world=torch.tensor([1.1, 0.0, 0.0], dtype=torch.float64),
        delta_R=pp.identity_SO3(dtype=torch.float64),
        delta_v=torch.tensor([0.1, 0.0, -1.962], dtype=torch.float64),
        delta_p=torch.tensor([0.01, 0.0, -0.1962], dtype=torch.float64),
        dt_total=dt,
        gravity_world=torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64),
        gravity_handling="residual",
    )

    assert torch.allclose(residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-9)


def test_default_gravity_handling_preserves_legacy_velocity_propagation():
    velocity = propagate_imu_velocity_world(
        velocity_world=torch.tensor([1.0, 2.0, 3.0]),
        delta_v_body=torch.tensor([0.1, -0.2, 0.3]),
        R_body_to_world=torch.eye(3),
    )

    assert torch.allclose(velocity, torch.tensor([1.1, 1.8, 3.3]), atol=1e-7)


def test_twoframe_graph_consumes_residual_gravity_metadata():
    graph = ReprojDisp_TwoFramePGO(_stationary_graph_input()).double()

    imu_residual = graph._imu_vio_residual()

    assert torch.allclose(imu_residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-7)


def test_local_window_graph_consumes_same_residual_gravity_metadata():
    edge = _stationary_graph_input()
    graph = LocalWindowInertialGraph(
        LocalWindowGraphInput(
            frame_indices=torch.tensor([0, 1]),
            frame_poses=torch.cat([edge.from_pose.tensor(), edge.init_motion.tensor()], dim=0),
            edges=[edge],
            fixed_first_frame=True,
            writeback="current",
            device="cpu",
        )
    ).double()

    imu_residual = graph._imu_edge_residual(
        edge,
        graph._all_poses()[0],
        graph._all_poses()[1],
        0,
        1,
    )

    assert torch.allclose(imu_residual, torch.zeros((3, 3), dtype=torch.float64), atol=1e-7)


def test_visual_map_stores_per_edge_gravity_contract():
    visual_map = VisualMap()

    assert "imu_vio_gravity_world" in visual_map.frames.data
    assert "imu_vio_gravity_in_residual" in visual_map.frames.data

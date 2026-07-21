from types import SimpleNamespace

import pypose as pp
import torch

from Module.Optimization.TwoFramePGO.Graphs import GraphInput
from Module.Optimization.TwoFramePGO.Optimizer import (
    TwoFrame_PGO,
    _gate_two_state_visual_factor,
)
from Module.Map import VisualMap
from Utility.TwoStateVIO import NavigationState, SquareRootPrior


def _config():
    return SimpleNamespace(
        graph_type="disp",
        device="cpu",
        vectorize=True,
        parallel=False,
        autodiff=True,
        imu_rot_prior=True,
        imu_factor_mode="two_state_fixed_lag",
        two_state_max_iterations=10,
    )


def _pose(x: float):
    return pp.SE3(
        torch.tensor([[x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    )


def _input(frame_i: int, frame_j: int, x_i: float, x_j: float) -> GraphInput:
    identity = pp.identity_SE3(1, dtype=torch.float32).tensor()
    return GraphInput(
        frame_idx=torch.tensor([frame_j], dtype=torch.long),
        from_idx=torch.tensor([frame_i], dtype=torch.long),
        init_motion=_pose(x_j),
        from_pose=_pose(x_i),
        baseline=torch.tensor([0.2], dtype=torch.float32),
        observations=None,  # type: ignore[arg-type]
        points=None,  # type: ignore[arg-type]
        images_intrinsic=torch.eye(3, dtype=torch.float32),
        edges_index=torch.empty(0, dtype=torch.long),
        device="cpu",
        visual_relative_pose_CiCj=_pose(1.0).tensor(),
        visual_relative_pose_cov=torch.eye(6, dtype=torch.float32) * 1.0e-4,
        imu_vio_factor_enable=True,
        imu_vio_prev_velocity_world=torch.tensor([1.0, 0.0, 0.0]),
        imu_vio_curr_velocity_init_world=torch.tensor([1.0, 0.0, 0.0]),
        imu_vio_prev_acc_bias=torch.zeros(3),
        imu_vio_prev_gyro_bias=torch.zeros(3),
        imu_vio_curr_acc_bias_init=torch.zeros(3),
        imu_vio_curr_gyro_bias_init=torch.zeros(3),
        imu_vio_linearized_acc_bias=torch.zeros(3),
        imu_vio_linearized_gyro_bias=torch.zeros(3),
        imu_vio_bias_jacobian=torch.zeros((9, 6)),
        imu_vio_bias_rw_cov=torch.eye(6) * 1.0e-3,
        imu_vio_delta_rotvec=torch.zeros(3),
        imu_vio_delta_v=torch.zeros(3),
        imu_vio_delta_p=torch.zeros(3),
        imu_vio_cov=torch.eye(9) * 1.0e-3,
        imu_vio_dt=torch.tensor([1.0]),
        imu_vio_sensor_T_imu=identity,
    )


def test_optimizer_keeps_prior_in_context_and_writes_both_states():
    context = TwoFrame_PGO.init_context(_config())
    context, first = TwoFrame_PGO._optimize(context, _input(0, 1, 0.0, 1.1))

    assert isinstance(context["two_state_prior"], SquareRootPrior)
    assert context["two_state_last_frame_idx"] == 1
    assert first.imu_factor_mode == "two_state_fixed_lag"
    assert first.local_ba_writeback == "all_two_state"
    assert first.window_motions is not None and first.window_motions.shape == (2, 7)
    assert abs(float(first.window_motions[1, 0]) - 1.0) < 2.0e-3

    context, second = TwoFrame_PGO._optimize(context, _input(1, 2, 1.0, 2.2))
    assert context["two_state_last_frame_idx"] == 2
    assert abs(float(second.window_motions[1, 0]) - 2.0) < 2.0e-3


def test_config_rejects_parallel_two_state_mode():
    config = _config()
    config.parallel = True
    try:
        TwoFrame_PGO.init_context(config)
    except ValueError:
        return
    raise AssertionError("parallel two-state mode must be rejected until prior sequencing is versioned")


def test_visual_map_allocates_cached_relative_pose_factor_fields():
    visual_map = VisualMap()

    assert "visual_relative_pose_CiCj" in visual_map.frames.data
    assert "visual_relative_pose_cov" in visual_map.frames.data
    assert visual_map.frames.data["visual_relative_pose_CiCj"].shape == (0, 7)
    assert visual_map.frames.data["visual_relative_pose_cov"].shape == (0, 6, 6)
    assert visual_map.frames.data["visual_relative_pose_num_points"].shape == (0,)
    assert visual_map.frames.data["visual_relative_pose_num_inliers"].shape == (0,)
    assert visual_map.frames.data["visual_relative_pose_mean_mahalanobis_sq"].shape == (0,)


def test_visual_pose_gate_accepts_good_edge_and_rejects_large_whitened_residual():
    state_i = NavigationState(_pose(0.0).tensor(), torch.zeros(3), torch.zeros(3), torch.zeros(3))
    state_j = NavigationState(_pose(1.0).tensor(), torch.zeros(3), torch.zeros(3), torch.zeros(3))
    config = {
        "soft_inlier_ratio": 0.5,
        "reject_inlier_ratio": 0.2,
        "soft_mean_mahalanobis_sq": 9.0,
        "reject_mean_mahalanobis_sq": 100.0,
        "soft_whitened_pose_norm": 6.0,
        "reject_whitened_pose_norm": 20.0,
        "max_covariance_inflation": 1.0e6,
    }
    covariance = torch.eye(6, dtype=torch.float64) * 1.0e-2

    accepted_covariance, accepted = _gate_two_state_visual_factor(
        state_i.to(device=torch.device("cpu"), dtype=torch.float64),
        state_j.to(device=torch.device("cpu"), dtype=torch.float64),
        _pose(1.0).tensor().double(),
        covariance,
        num_points=200,
        num_inliers=190,
        mean_mahalanobis_sq=1.0,
        config=config,
        eigenvalue_floor=1.0e-12,
    )
    assert accepted["action"] == "accept"
    assert accepted["covariance_inflation"] == 1.0
    assert torch.allclose(accepted_covariance, covariance)

    rejected_covariance, rejected = _gate_two_state_visual_factor(
        state_i.to(device=torch.device("cpu"), dtype=torch.float64),
        state_j.to(device=torch.device("cpu"), dtype=torch.float64),
        _pose(-2.0).tensor().double(),
        covariance,
        num_points=200,
        num_inliers=190,
        mean_mahalanobis_sq=1.0,
        config=config,
        eigenvalue_floor=1.0e-12,
    )
    assert str(rejected["action"]).startswith("reject:")
    assert rejected["covariance_inflation"] == 1.0e6
    assert torch.allclose(rejected_covariance, covariance * 1.0e6)

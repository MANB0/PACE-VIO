from types import SimpleNamespace

import pypose as pp
import pytest
import torch

from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import (
    GraphInput,
    GraphOutput,
    LocalWindowGraphInput,
    LocalWindowInertialGraph,
    ReprojDisp_TwoFramePGO,
)
from Module.Optimization.Interface import move_dataclass_to_local
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO


def _optimizer_cfg(**overrides):
    cfg = {
        "graph_type": "disp",
        "device": "cpu",
        "vectorize": True,
        "parallel": False,
        "autodiff": True,
        "imu_rot_prior": True,
        "imu_factor_mode": "local_inertial_ba",
        "local_ba_window_size": 3,
        "local_ba_fix_first_frame": True,
        "local_ba_writeback": "current",
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _se3(x=0.0, y=0.0, z=0.0):
    return pp.SE3(torch.tensor([[x, y, z, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))


def _match_obs(pixel1_uv, pixel2_uv, depth=2.0, disp=5.0) -> MatchObs:
    pixel1_uv_t = torch.tensor([pixel1_uv], dtype=torch.float32)
    pixel2_uv_t = torch.tensor([pixel2_uv], dtype=torch.float32)
    return MatchObs.init(
        {
            "pixel1_uv": pixel1_uv_t,
            "pixel2_uv": pixel2_uv_t,
            "pixel1_d": torch.tensor([[depth]], dtype=torch.float32),
            "pixel2_d": torch.tensor([[depth]], dtype=torch.float32),
            "pixel1_disp": torch.tensor([[disp]], dtype=torch.float32),
            "pixel2_disp": torch.tensor([[disp]], dtype=torch.float32),
            "pixel1_disp_cov": torch.tensor([[1.0]], dtype=torch.float32),
            "pixel2_disp_cov": torch.tensor([[1.0]], dtype=torch.float32),
            "obs1_covTc": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "obs2_covTc": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "pixel1_uv_cov": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
            "pixel2_uv_cov": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
            "pixel1_d_cov": torch.ones((1, 1), dtype=torch.float32),
            "pixel2_d_cov": torch.ones((1, 1), dtype=torch.float32),
        }
    )


def _dummy_points(pos_Tw=None) -> PointNode:
    if pos_Tw is None:
        pos_Tw = torch.zeros((1, 3), dtype=torch.float32)
    else:
        pos_Tw = torch.as_tensor(pos_Tw, dtype=torch.float32).reshape(1, 3)
    return PointNode.init(
        {
            "pos_Tw": pos_Tw,
            "cov_Tw": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "color": torch.zeros((1, 3), dtype=torch.uint8),
        }
    )


def _pair_graph_input(
    from_idx: int,
    frame_idx: int,
    from_pose,
    to_pose,
    obs: MatchObs,
    *,
    delta_p=None,
    dt=0.1,
    point_w=None,
) -> GraphInput:
    if delta_p is None:
        delta_p = torch.zeros(3, dtype=torch.float32)
    return GraphInput(
        frame_idx=torch.tensor([frame_idx], dtype=torch.long),
        from_idx=torch.tensor([from_idx], dtype=torch.long),
        init_motion=to_pose,
        from_pose=from_pose,
        baseline=torch.tensor([0.1], dtype=torch.float32),
        observations=obs,
        points=_dummy_points(point_w),
        images_intrinsic=torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        edges_index=torch.zeros((len(obs),), dtype=torch.long),
        device="cpu",
        imu_vio_factor_enable=True,
        imu_vio_prev_velocity_world=torch.zeros(3, dtype=torch.float32),
        imu_vio_curr_velocity_init_world=torch.zeros(3, dtype=torch.float32),
        imu_vio_prev_acc_bias=torch.zeros(3, dtype=torch.float32),
        imu_vio_prev_gyro_bias=torch.zeros(3, dtype=torch.float32),
        imu_vio_curr_acc_bias_init=torch.zeros(3, dtype=torch.float32),
        imu_vio_curr_gyro_bias_init=torch.zeros(3, dtype=torch.float32),
        imu_vio_linearized_acc_bias=torch.zeros(3, dtype=torch.float32),
        imu_vio_linearized_gyro_bias=torch.zeros(3, dtype=torch.float32),
        imu_vio_bias_jacobian=torch.zeros((9, 6), dtype=torch.float32),
        imu_vio_bias_rw_cov=torch.eye(6, dtype=torch.float32),
        imu_vio_delta_rotvec=torch.zeros(3, dtype=torch.float32),
        imu_vio_delta_v=torch.zeros(3, dtype=torch.float32),
        imu_vio_delta_p=torch.as_tensor(delta_p, dtype=torch.float32).reshape(3),
        imu_vio_cov=torch.eye(9, dtype=torch.float32),
        imu_vio_dt=torch.tensor([dt], dtype=torch.float32),
        imu_vio_sensor_T_imu=pp.identity_SE3(1, dtype=torch.float32).tensor(),
        imu_vio_alpha_p=1.0,
        imu_vio_alpha_v=1.0,
        imu_vio_alpha_R=1.0,
    )


def test_w2_local_window_residual_matches_two_frame_graph_for_same_input():
    obs = _match_obs([52.0, 49.0], [51.5, 48.75], depth=2.0, disp=10.0)
    pose0 = _se3(0.0, 0.0, 0.0)
    pose1 = _se3(0.15, 0.02, -0.01)
    edge = _pair_graph_input(
        0,
        1,
        pose0,
        pose1,
        obs,
        delta_p=[0.15, 0.02, -0.01],
        dt=0.1,
        point_w=[2.0, 0.04, -0.02],
    )
    two_frame = ReprojDisp_TwoFramePGO(edge).double()
    local_w2 = LocalWindowInertialGraph(
        LocalWindowGraphInput(
            frame_indices=torch.tensor([0, 1], dtype=torch.long),
            frame_poses=torch.cat([pose0.tensor(), pose1.tensor()], dim=0),
            edges=[edge],
            fixed_first_frame=True,
            writeback="current",
            device="cpu",
        )
    ).double()

    assert torch.allclose(local_w2.forward(), two_frame.forward(), atol=1e-7, rtol=1e-7)
    assert torch.allclose(local_w2.weight_matrix(), two_frame.weight_matrix(), atol=1e-9, rtol=1e-9)


def test_w2_local_window_uses_same_world_point_anchor_as_two_frame_graph():
    obs = _match_obs([52.0, 49.0], [51.5, 48.75], depth=2.0, disp=10.0)
    pose0 = _se3(0.0, 0.0, 0.0)
    pose1 = _se3(0.15, 0.02, -0.01)
    edge = _pair_graph_input(
        0,
        1,
        pose0,
        pose1,
        obs,
        delta_p=[0.15, 0.02, -0.01],
        dt=0.1,
        point_w=[2.3, -0.20, 0.10],
    )
    two_frame = ReprojDisp_TwoFramePGO(edge).double()
    local_w2 = LocalWindowInertialGraph(
        LocalWindowGraphInput(
            frame_indices=torch.tensor([0, 1], dtype=torch.long),
            frame_poses=torch.cat([pose0.tensor(), pose1.tensor()], dim=0),
            edges=[edge],
            fixed_first_frame=True,
            writeback="current",
            device="cpu",
        )
    ).double()

    assert torch.allclose(local_w2.forward(), two_frame.forward(), atol=1e-7, rtol=1e-7)


def test_w2_local_window_optimizer_matches_two_frame_vio_optimizer():
    obs = _match_obs([52.0, 49.0], [51.5, 48.75], depth=2.0, disp=10.0)
    pose0 = _se3(0.0, 0.0, 0.0)
    pose1 = _se3(0.15, 0.02, -0.01)
    edge = _pair_graph_input(
        0,
        1,
        pose0,
        pose1,
        obs,
        delta_p=[0.15, 0.02, -0.01],
        dt=0.1,
        point_w=[2.3, -0.20, 0.10],
    )
    local_data = LocalWindowGraphInput(
        frame_indices=torch.tensor([0, 1], dtype=torch.long),
        frame_poses=torch.cat([pose0.tensor(), pose1.tensor()], dim=0),
        edges=[edge],
        fixed_first_frame=True,
        writeback="current",
        device="cpu",
    )

    _, two_frame_output = TwoFrame_PGO._optimize(
        TwoFrame_PGO.init_context(_optimizer_cfg(imu_factor_mode="preintegrated_vio")),
        edge,
    )
    _, local_output = TwoFrame_PGO._optimize(
        TwoFrame_PGO.init_context(_optimizer_cfg(local_ba_window_size=2)),
        local_data,
    )

    assert torch.allclose(
        local_output.motion.double(),
        two_frame_output.motion.tensor().double(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(local_output.velocity_world.double(), two_frame_output.velocity_world.double(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(local_output.acc_bias.double(), two_frame_output.acc_bias.double(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(local_output.gyro_bias.double(), two_frame_output.gyro_bias.double(), atol=1e-6, rtol=1e-6)


def test_w2_current_writeback_uses_local_window_graph():
    cfg = _optimizer_cfg(local_ba_window_size=2, local_ba_writeback="current", local_ba_fix_first_frame=True)

    assert TwoFrame_PGO._should_use_local_window_graph(cfg)


def test_w3_current_writeback_uses_local_window_graph():
    cfg = _optimizer_cfg(local_ba_window_size=3, local_ba_writeback="current", local_ba_fix_first_frame=True)

    assert TwoFrame_PGO._should_use_local_window_graph(cfg)


def test_local_inertial_ba_config_accepts_adjustable_window_size():
    TwoFrame_PGO.is_valid_config(_optimizer_cfg(local_ba_window_size=2))
    TwoFrame_PGO.is_valid_config(_optimizer_cfg(local_ba_window_size=5))


@pytest.mark.parametrize(
    "overrides",
    [
        {"local_ba_window_size": 1},
        {"local_ba_writeback": "invalid"},
        {"graph_type": "reproj"},
        {"autodiff": False},
    ],
)
def test_local_inertial_ba_config_rejects_invalid_values(overrides):
    with pytest.raises(AssertionError):
        TwoFrame_PGO.is_valid_config(_optimizer_cfg(**overrides))


def test_window_visual_residual_uses_fixed_world_point_anchor():
    obs = _match_obs([50.0, 50.0], [50.0, 50.0])
    edge = _pair_graph_input(0, 1, _se3(), _se3(), obs)
    data = LocalWindowGraphInput(
        frame_indices=torch.tensor([0, 1], dtype=torch.long),
        frame_poses=torch.cat([_se3().tensor(), _se3().tensor()], dim=0),
        edges=[edge],
        fixed_first_frame=False,
        writeback="all_optimized",
        device="cpu",
    )
    graph = LocalWindowInertialGraph(data).double()
    baseline = graph.forward().detach().clone()

    with torch.no_grad():
        graph.pose_window.tensor()[0, 1] += 0.25

    shifted = graph.forward().detach()
    assert torch.allclose(baseline[:1], shifted[:1], atol=1e-8)

    with torch.no_grad():
        graph.pose_window.tensor()[1, 1] += 0.25

    shifted_target = graph.forward().detach()
    assert not torch.allclose(shifted[:1], shifted_target[:1], atol=1e-8)


def test_window_graph_accepts_parallel_worker_dict_edges():
    obs = _match_obs([50.0, 50.0], [50.0, 50.0], depth=2.0, disp=10.0)
    pose0 = _se3(0.0, 0.0, 0.0)
    pose1 = _se3(1.0, 0.0, 0.0)
    edge = _pair_graph_input(0, 1, pose0, pose1, obs, delta_p=[1.0, 0.0, 0.0], dt=0.0)
    data = LocalWindowGraphInput(
        frame_indices=torch.tensor([0, 1], dtype=torch.long),
        frame_poses=torch.cat([pose0.tensor(), pose1.tensor()], dim=0),
        edges=[edge],
        fixed_first_frame=True,
        writeback="current",
        device="cpu",
    )

    worker_local_data = move_dataclass_to_local(data)

    assert isinstance(worker_local_data.edges[0], dict)
    graph = LocalWindowInertialGraph(worker_local_data)
    assert graph.forward().numel() > 0


def test_window_optimizer_reduces_drifted_local_poses():
    obs = _match_obs([50.0, 50.0], [50.0, 50.0], depth=2.0, disp=10.0)
    pose0 = _se3(0.0, 0.0, 0.0)
    pose1_init = _se3(1.0, 0.35, 0.0)
    pose2_init = _se3(2.0, 0.35, 0.0)
    edge01 = _pair_graph_input(0, 1, pose0, pose1_init, obs, delta_p=[1.0, 0.0, 0.0], dt=0.0)
    edge12 = _pair_graph_input(1, 2, pose1_init, pose2_init, obs, delta_p=[1.0, 0.0, 0.0], dt=0.0)
    data = LocalWindowGraphInput(
        frame_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        frame_poses=torch.cat([pose0.tensor(), pose1_init.tensor(), pose2_init.tensor()], dim=0),
        edges=[edge01, edge12],
        fixed_first_frame=True,
        writeback="current",
        device="cpu",
    )
    context = TwoFrame_PGO.init_context(_optimizer_cfg())

    _, output = TwoFrame_PGO._optimize(context, data)

    initial_y_norm = torch.tensor([0.35, 0.35]).norm().item()
    final_y_norm = output.window_motions[1:, 1].norm().item()
    assert final_y_norm < initial_y_norm * 0.5


def test_window_optimizer_records_profile_for_two_frame_window():
    obs = _match_obs([50.0, 50.0], [50.0, 50.0], depth=2.0, disp=10.0)
    pose0 = _se3(0.0, 0.0, 0.0)
    pose1_init = _se3(1.0, 0.25, 0.0)
    edge01 = _pair_graph_input(0, 1, pose0, pose1_init, obs, delta_p=[1.0, 0.0, 0.0], dt=0.0)
    data = LocalWindowGraphInput(
        frame_indices=torch.tensor([0, 1], dtype=torch.long),
        frame_poses=torch.cat([pose0.tensor(), pose1_init.tensor()], dim=0),
        edges=[edge01],
        fixed_first_frame=True,
        writeback="current",
        device="cpu",
    )
    context = TwoFrame_PGO.init_context(_optimizer_cfg(local_ba_window_size=2))

    _, output = TwoFrame_PGO._optimize(context, data)

    assert output.local_ba_window_size == 2
    assert output.local_ba_num_edges == 1
    assert output.local_ba_num_frames == 2
    assert output.local_ba_optimize_total_s is not None
    assert output.local_ba_optimize_total_s >= 0.0
    assert output.local_ba_lm_s is not None
    assert output.local_ba_lm_s >= 0.0


def test_local_ba_writeback_keeps_adjacent_imu_edge_states_consistent():
    """Optimized per-frame VIO states must be reused by later local-window edges."""
    opt = object.__new__(TwoFrame_PGO)
    opt.last_pair_diagnostics = {}
    opt.last_breakpoint_trace = None
    opt.last_breakpoint_frame_indices = []

    n = 4
    frame_data = {
        "pose": pp.identity_SE3(n, dtype=torch.float32).tensor().clone(),
        "imu_vio_prev_velocity_world": torch.zeros((n, 3), dtype=torch.float32),
        "imu_vio_curr_velocity_init_world": torch.zeros((n, 3), dtype=torch.float32),
        "imu_vio_velocity_world": torch.zeros((n, 3), dtype=torch.float32),
        "imu_vio_prev_acc_bias": torch.zeros((n, 3), dtype=torch.float32),
        "imu_vio_prev_gyro_bias": torch.zeros((n, 3), dtype=torch.float32),
        "imu_vio_acc_bias": torch.zeros((n, 3), dtype=torch.float32),
        "imu_vio_gyro_bias": torch.zeros((n, 3), dtype=torch.float32),
    }
    stale = torch.tensor([99.0, 99.0, 99.0], dtype=torch.float32)
    frame_data["imu_vio_curr_velocity_init_world"][2] = stale
    frame_data["imu_vio_prev_velocity_world"][3] = stale
    frame_data["imu_vio_prev_acc_bias"][3] = stale
    frame_data["imu_vio_prev_gyro_bias"][3] = stale
    fake_map = SimpleNamespace(frames=SimpleNamespace(data=frame_data))

    window_velocity = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [2.0, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    window_acc_bias = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
        ],
        dtype=torch.float32,
    )
    window_gyro_bias = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.01, -0.02, -0.03],
            [-0.04, -0.05, -0.06],
        ],
        dtype=torch.float32,
    )
    result = GraphOutput(
        motion=pp.identity_SE3(1, dtype=torch.float32),
        from_idx=torch.tensor([1], dtype=torch.long),
        frame_idx=torch.tensor([2], dtype=torch.long),
        window_frame_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        window_motions=pp.identity_SE3(3, dtype=torch.float32).tensor(),
        window_velocity_world=window_velocity,
        window_acc_bias=window_acc_bias,
        window_gyro_bias=window_gyro_bias,
        local_ba_writeback="all_optimized",
    )

    opt._write_local_ba_graph_data(result, fake_map)

    torch.testing.assert_close(frame_data["imu_vio_velocity_world"][2], window_velocity[2])
    torch.testing.assert_close(frame_data["imu_vio_curr_velocity_init_world"][2], window_velocity[2])
    torch.testing.assert_close(frame_data["imu_vio_prev_velocity_world"][3], window_velocity[2])
    torch.testing.assert_close(frame_data["imu_vio_prev_acc_bias"][3], window_acc_bias[2])
    torch.testing.assert_close(frame_data["imu_vio_prev_gyro_bias"][3], window_gyro_bias[2])

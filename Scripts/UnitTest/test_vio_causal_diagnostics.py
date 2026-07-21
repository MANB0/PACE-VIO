import subprocess
import sys


BASE_CODE = r'''
from types import SimpleNamespace

import pypose as pp
import torch

from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO


def make_graph_data():
    n_points = 4
    intrinsics = torch.tensor(
        [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]
    )
    baseline = torch.tensor([0.225])
    from_pose = pp.identity_SE3(1)
    init_motion = pp.SE3(torch.tensor([[0.12, 0.01, 0.0, 0.0, 0.0, 0.02, 0.9998]]))
    points_tw = torch.tensor(
        [[2.0, 0.1, 0.1], [2.2, -0.1, 0.05], [1.8, 0.2, -0.05], [2.5, -0.2, 0.0]]
    )
    points = PointNode.init({
        "pos_Tw": points_tw,
        "cov_Tw": torch.eye(3, dtype=torch.float64).repeat(n_points, 1, 1) * 0.01,
        "color": torch.zeros(n_points, 3, dtype=torch.uint8),
    })
    pixel2_uv = torch.tensor(
        [[336.0, 256.0], [305.0, 247.0], [356.0, 231.0], [294.0, 240.0]]
    )
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
    return GraphInput(
        torch.tensor([1]), torch.tensor([0]), init_motion, from_pose, baseline,
        observations, points, intrinsics, torch.zeros(n_points, dtype=torch.long), "cpu",
        imu_vio_factor_enable=True,
        imu_vio_prev_velocity_world=torch.tensor([0.2, 0.0, 0.0]),
        imu_vio_curr_velocity_init_world=torch.tensor([0.3, 0.02, 0.0]),
        imu_vio_delta_rotvec=torch.tensor([0.0, 0.0, 0.01]),
        imu_vio_delta_v=torch.tensor([0.01, 0.0, 0.0]),
        imu_vio_delta_p=torch.tensor([0.005, 0.0, 0.0]),
        imu_vio_cov=torch.eye(9, dtype=torch.float32) * 0.01,
        imu_vio_dt=torch.tensor([0.1]),
    )


def make_context(enabled):
    config = SimpleNamespace(
        autodiff=True,
        graph_type="disp",
        vectorize=True,
        device="cpu",
        imu_factor_mode="preintegrated_vio",
        vio_causal_diagnostics_enable=enabled,
        vio_causal_diagnostics_interval=1,
    )
    return TwoFrame_PGO.init_context(config)
'''


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BASE_CODE + code],
        cwd="/home/admin1/macvo-dev",
        text=True,
        capture_output=True,
    )


def test_causal_diagnostics_are_read_only_and_report_factor_victory_metrics():
    result = _run(
        r'''
_, disabled = TwoFrame_PGO._optimize(make_context(False), make_graph_data())
_, enabled = TwoFrame_PGO._optimize(make_context(True), make_graph_data())

assert torch.allclose(enabled.motion.tensor(), disabled.motion.tensor(), atol=1e-10, rtol=1e-10)
assert enabled.velocity_world is not None and disabled.velocity_world is not None
assert torch.allclose(enabled.velocity_world, disabled.velocity_world, atol=1e-10, rtol=1e-10)
assert disabled.initial_energy_visual_weighted is None
assert disabled.influence_visual_grad_norm is None
assert disabled.counterfactual_full_step_norm is None
assert enabled.initial_motion is not None
assert torch.allclose(enabled.initial_motion, make_graph_data().init_motion.tensor().double())
assert abs(enabled.init_delta_x - 0.12) < 1e-8
assert abs(enabled.init_delta_y - 0.01) < 1e-8
assert abs(enabled.init_velocity_j_x - 0.3) < 1e-7
assert abs(enabled.init_velocity_j_y - 0.02) < 1e-7

required = [
    enabled.initial_energy_visual_weighted,
    enabled.initial_energy_imu_weighted,
    enabled.energy_visual_change,
    enabled.energy_imu_change,
    enabled.update_pose_translation_norm,
    enabled.update_pose_rotation_norm,
    enabled.update_velocity_norm,
    enabled.influence_visual_grad_norm,
    enabled.influence_imu_grad_norm,
    enabled.influence_visual_hessian_trace,
    enabled.influence_imu_hessian_trace,
    enabled.counterfactual_visual_step_norm,
    enabled.counterfactual_imu_step_norm,
    enabled.counterfactual_full_step_norm,
    enabled.actual_to_visual_step_cosine,
    enabled.actual_to_imu_step_cosine,
    enabled.actual_to_full_step_cosine,
    enabled.predicted_visual_change_on_actual_step,
    enabled.predicted_imu_change_on_actual_step,
]
assert all(value is not None and torch.isfinite(torch.tensor(value)) for value in required)
assert enabled.influence_sampled == 1
'''
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_causal_influence_sampling_interval_leaves_expensive_fields_empty():
    result = _run(
        r'''
context = make_context(True)
context["vio_causal_diagnostics_interval"] = 5
graph_data = make_graph_data()
graph_data.frame_idx = torch.tensor([2])
_, output = TwoFrame_PGO._optimize(context, graph_data)

assert output.initial_energy_visual_weighted is not None
assert output.energy_visual_change is not None
assert output.influence_sampled == 0
assert output.influence_visual_grad_norm is None
assert output.counterfactual_full_step_norm is None
'''
    )
    assert result.returncode == 0, result.stdout + result.stderr

import subprocess
import sys


def test_analytic_imu_rotation_jacobian_matches_autograd():
    code = r'''
import torch, pypose as pp
from pypose.optim.functional import modjac
from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput, Analytic_ReprojDisp_TwoFramePGO

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
    imu_rotvec_prior=torch.tensor([0.01, 0.02, -0.01]),
    imu_rot_prior_std=torch.tensor([0.1]),
    imu_trans_prior=torch.tensor([0.1, 0.0, 0.0]),
    imu_trans_cov=torch.eye(3) * 0.1,
)
graph = Analytic_ReprojDisp_TwoFramePGO(graph_data).to(dtype=torch.double)
residual = graph()
analytic = graph.build_jacobian()

autograd = modjac(graph, input=graph._call_args, flatten=False, vectorize=True)
params = tuple(dict(graph.named_parameters()).values())
if isinstance(autograd, (tuple, list)):
    autograd = torch.cat([j.reshape(-1, p.numel()) for j, p in zip(autograd, params)], 1)
if isinstance(autograd, (tuple, list)):
    autograd = torch.cat(autograd)

n_visual_rows = residual.shape[0] * 3 - 6
imu_rot_slice = slice(n_visual_rows, n_visual_rows + 3)
if not torch.allclose(analytic[imu_rot_slice], autograd[imu_rot_slice], atol=1e-8, rtol=1e-8):
    print((analytic[imu_rot_slice] - autograd[imu_rot_slice]).abs().max().item())
    raise SystemExit(1)
'''
    result = subprocess.run([sys.executable, "-c", code], cwd="/home/admin1/macvo-dev", text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr

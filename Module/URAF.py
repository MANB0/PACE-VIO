"""
Uncertainty-Ratio Adaptive Fusion (URAF)

Minimal, principled visual-inertial post-fusion.

Core idea: MACVO provides BOTH visual uncertainty (obs2_covTc) AND IMU uncertainty
(imu_rot_prior_std). The optimal fusion weight in information space is the
precision ratio. URAF makes the IMU trust floor adaptive to visual quality:

    sigma_imu_eff = max(sigma_imu_raw, sqrt(var_vis) * lambda)

When vision is GOOD (small var_vis) → floor is low → IMU can contribute.
When vision is BAD  (large var_vis) → floor is high → don't blindly trust IMU.

This replaces fixed-configuration IMU std floors with a single interpretable
parameter lambda that controls maximum relative IMU-to-visual trust.
"""

import math
import torch
import pypose as pp
from typing import Optional
from Utility.Math import NormalizeQuat


def uraf_post_fusion(
    to_pose: pp.LieTensor,
    prev_pose: pp.LieTensor,
    rot_prior: torch.Tensor,       # (3,) IMU rotation prior (rotvec in body frame)
    rot_std_raw: torch.Tensor,     # (1,) raw IMU rotation std from preintegration
    trans_prior: torch.Tensor,     # (3,) IMU translation prior
    trans_cov_raw: torch.Tensor,   # (3,3) IMU translation covariance
    visual_cov_mean: Optional[float],  # mean visual obs covariance from MACVO frontend
    num_observations: int = 0,
    uraf_lambda: float = 0.5,      # maximum IMU-to-visual trust ratio
    uraf_min_floor: float = 0.005, # absolute minimum IMU std floor
    uraf_max_floor: float = 1.0,   # absolute maximum IMU std floor
    imu_trans_enable: bool = False,
    imu_trans_z_only: bool = False,
) -> tuple[pp.LieTensor, dict]:
    """
    URAF: Uncertainty-Ratio Adaptive post-Fusion.

    Returns:
        fused_pose: SE3 LieTensor
        log: diagnostic dict
    """
    log = {"mode": "URAF", "skipped": False}

    # ── Convert visual cov to std ─────────────────────────────────────
    if visual_cov_mean is not None and visual_cov_mean > 0:
        vis_std = max(math.sqrt(visual_cov_mean), 0.001)
    else:
        vis_std = 0.05  # fallback

    # ── URAF: adaptive IMU rotation floor ─────────────────────────────
    # sigma_imu_floor = max(raw_imu_std, vis_std * lambda)
    # When vision is good (small vis_std): floor ≈ raw_imu_std → trust IMU
    # When vision is bad (large vis_std): floor ≈ vis_std * lambda → don't over-trust
    adaptive_rot_floor = max(
        float(rot_std_raw[0].item()),
        vis_std * uraf_lambda
    )
    adaptive_rot_floor = max(uraf_min_floor, min(uraf_max_floor, adaptive_rot_floor))

    # ── Observation-count modulation ──────────────────────────────────
    if num_observations > 0:
        obs_factor = min(1.0, num_observations / 80.0)
    else:
        obs_factor = 0.3
    adaptive_rot_floor = adaptive_rot_floor / max(obs_factor, 0.15)

    # ── Decompose visual and IMU poses ────────────────────────────────
    rel_visual = prev_pose.Inv() @ pp.SE3(to_pose).double()
    xi_visual = torch.cat([
        rel_visual.translation().reshape(3).double(),
        rel_visual.rotation().Log().tensor().reshape(3).double(),
    ], dim=0)
    xi_imu = torch.cat([trans_prior, rot_prior], dim=0).reshape(6).double()

    # ── Build visual covariance ───────────────────────────────────────
    # Visual uncertainty per axis from obs cov + heuristics
    vis_std_xy = max(vis_std, 0.03)
    vis_std_z  = max(vis_std * 1.3, 0.04)
    vis_std_rot = max(vis_std * 0.5, 0.01)

    Sigma_v = torch.diag(torch.tensor([
        vis_std_xy ** 2,
        vis_std_xy ** 2,
        vis_std_z ** 2,
        vis_std_rot ** 2,
        vis_std_rot ** 2,
        vis_std_rot ** 2,
    ], dtype=torch.double))

    # ── Build IMU covariance ──────────────────────────────────────────
    imu_trans_std = max(float(trans_cov_raw.diagonal().mean().sqrt().item()), 0.15)

    Sigma_i = torch.zeros((6, 6), dtype=torch.double)
    if imu_trans_enable:
        if imu_trans_z_only:
            Sigma_i[0, 0] = 1e12; Sigma_i[1, 1] = 1e12
            Sigma_i[2, 2] = imu_trans_std ** 2
        else:
            Sigma_i[0, 0] = imu_trans_std ** 2
            Sigma_i[1, 1] = imu_trans_std ** 2
            Sigma_i[2, 2] = imu_trans_std ** 2
    else:
        Sigma_i[0, 0] = 1e12; Sigma_i[1, 1] = 1e12; Sigma_i[2, 2] = 1e12

    Sigma_i[3, 3] = adaptive_rot_floor ** 2
    Sigma_i[4, 4] = adaptive_rot_floor ** 2
    Sigma_i[5, 5] = adaptive_rot_floor ** 2

    # ── Information fusion ────────────────────────────────────────────
    epsI = torch.eye(6, dtype=torch.double) * 1e-9
    Wv = torch.linalg.pinv(Sigma_v + epsI)
    Wi = torch.linalg.pinv(Sigma_i + epsI)

    A = Wv + Wi + epsI
    b = Wv @ xi_visual + Wi @ xi_imu
    xi_fused = torch.linalg.solve(A, b).reshape(1, 6)

    rel_fused = pp.se3(xi_fused).Exp()
    fused_pose = (prev_pose @ rel_fused).float()

    # ── Diagnostics ───────────────────────────────────────────────────
    wi_trace = float(Wi.trace().item())
    wv_trace = float(Wv.trace().item())

    log.update({
        "vis_std": round(vis_std, 6),
        "adapt_rot_floor": round(adaptive_rot_floor, 6),
        "imu_weight_ratio": round(wi_trace / max(wi_trace + wv_trace, 1e-9), 4),
        "num_obs": num_observations,
    })
    return fused_pose, log

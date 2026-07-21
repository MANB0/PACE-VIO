import pypose as pp
import torch

from Utility.Point import point2pixel_NED
from Utility.RelativePoseFactorCache import (
    RelativePoseFactorCacheReader,
    RelativePoseFactorPacket,
    camera_factor_to_body_factor,
    relative_pose_information_from_packet,
    write_relative_pose_factor_cache,
)
from Utility.VisualFactorCache import VisualFactorPacket


def _packet(points_i: torch.Tensor, points_j: torch.Tensor) -> VisualFactorPacket:
    dtype = torch.float64
    K = torch.tensor(
        [[120.0, 0.0, 64.0], [0.0, 120.0, 48.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    count = points_i.shape[0]
    covariance = torch.eye(3, dtype=dtype).repeat(count, 1, 1) * 1.0e-4
    return VisualFactorPacket(
        frame_i=0,
        frame_j=1,
        timestamp_i_ns=0,
        timestamp_j_ns=1,
        K=K,
        baseline_m=0.2,
        relative_pose_init=torch.eye(4, dtype=dtype),
        points_local=points_i,
        points_cov_local=covariance,
        point_colors=torch.zeros((count, 3), dtype=torch.uint8),
        match_fields={
            "pixel2_uv": point2pixel_NED(points_j, K),
            "pixel2_d": points_j[:, 0:1],
            "obs1_covTc": covariance,
            "obs2_covTc": covariance,
        },
        covariance_diagnostics={},
        visual_sha256="synthetic",
    )


def test_point_hessian_produces_finite_positive_pose_covariance():
    dtype = torch.float64
    points_j = torch.tensor(
        [
            [2.0, -0.5, -0.2],
            [2.5, 0.4, 0.1],
            [3.0, -0.2, 0.5],
            [3.5, 0.7, -0.4],
            [4.0, -0.8, 0.3],
            [4.5, 0.1, -0.6],
        ],
        dtype=dtype,
    )
    measurement = (
        pp.SE3(torch.tensor([[0.1, -0.03, 0.02, 0.0, 0.0, 0.0, 1.0]], dtype=dtype))
        @ pp.se3(torch.tensor([[0.0, 0.0, 0.0, 0.01, -0.02, 0.03]], dtype=dtype)).Exp()
    )
    points_i = measurement.Act(points_j)
    covariance, diagnostics = relative_pose_information_from_packet(
        _packet(points_i, points_j),
        measurement.tensor(),
    )

    assert covariance.shape == (6, 6)
    assert torch.linalg.eigvalsh(covariance).min() > 0.0
    assert diagnostics["num_inliers"] == len(points_j)
    assert diagnostics["mean_mahalanobis_sq"] < 1.0e-18


def test_camera_to_body_conjugates_mean_and_covariance():
    dtype = torch.float64
    measurement_camera = pp.SE3(
        torch.tensor([[0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 1.0]], dtype=dtype)
    )
    extrinsic_CI = pp.SE3(
        torch.tensor([[0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]], dtype=dtype)
    )
    covariance_camera = torch.diag(
        torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=dtype)
    )
    measurement_body, covariance_body = camera_factor_to_body_factor(
        measurement_camera.tensor(),
        covariance_camera,
        extrinsic_CI.tensor(),
    )

    expected = extrinsic_CI.Inv() @ measurement_camera @ extrinsic_CI
    assert torch.allclose(measurement_body, expected.tensor(), atol=1.0e-12)
    assert torch.allclose(covariance_body, covariance_body.mT, atol=1.0e-12)
    assert torch.linalg.eigvalsh(covariance_body).min() > 0.0


def test_sidecar_round_trip_binds_factor_to_visual_hash(tmp_path):
    packet = RelativePoseFactorPacket(
        frame_i=0,
        frame_j=1,
        measurement_CiCj=pp.identity_SE3(1, dtype=torch.float64).tensor(),
        covariance=torch.eye(6, dtype=torch.float64) * 1.0e-3,
        visual_sha256="visual-hash",
        num_points=20,
        num_inliers=18,
        mean_mahalanobis_sq=2.5,
    )
    write_relative_pose_factor_cache(tmp_path, [packet])
    reader = RelativePoseFactorCacheReader(tmp_path)
    restored = reader.load_pair(0, 1, "visual-hash")

    assert torch.equal(restored.measurement_CiCj, packet.measurement_CiCj)
    assert torch.equal(restored.covariance, packet.covariance)
    assert restored.num_inliers == 18

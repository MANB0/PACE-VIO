from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import pypose as pp
import torch

from Utility.CompressedUVDFactorCache import (
    CompressedUVDFactorCacheReader,
    CompressedUVDFactorPacket,
    write_compressed_uvd_factor_cache,
)
from Utility.VisualFactorCache import VisualFactorCacheError


def _packet(frame_i: int = 0) -> CompressedUVDFactorPacket:
    diagonal = torch.tensor([20.0, 25.0, 30.0, 40.0, 45.0, 50.0], dtype=torch.float64)
    return CompressedUVDFactorPacket(
        frame_i=frame_i,
        frame_j=frame_i + 1,
        reference_CjCi=pp.se3(
            torch.tensor([[0.1, -0.02, 0.03, 0.01, -0.015, 0.02]], dtype=torch.float64)
        ).Exp().tensor(),
        hessian=torch.diag(diagonal),
        gradient=torch.tensor([0.1, -0.2, 0.3, 0.01, -0.02, 0.03], dtype=torch.float64),
        robust_cost=12.5,
        visual_sha256=f"visual-{frame_i}",
        num_points=100,
        num_inliers=92,
        mean_mahalanobis_sq=1.7,
        huber_delta=0.1,
    )


def test_compressed_uvd_cache_round_trip(tmp_path: Path) -> None:
    packets = [_packet(0), _packet(1)]
    write_compressed_uvd_factor_cache(tmp_path, packets)
    reader = CompressedUVDFactorCacheReader(tmp_path)
    loaded = reader.load_pair(1, 2, "visual-1")
    expected = packets[1]
    assert torch.equal(loaded.reference_CjCi, expected.reference_CjCi)
    assert torch.equal(loaded.hessian, expected.hessian)
    assert torch.equal(loaded.gradient, expected.gradient)
    assert loaded.robust_cost == expected.robust_cost
    assert loaded.num_points == expected.num_points
    assert loaded.num_inliers == expected.num_inliers
    assert loaded.huber_delta == expected.huber_delta


def test_compressed_uvd_cache_rejects_visual_hash_mismatch(tmp_path: Path) -> None:
    write_compressed_uvd_factor_cache(tmp_path, [_packet()])
    reader = CompressedUVDFactorCacheReader(tmp_path)
    with pytest.raises(VisualFactorCacheError, match="visual hash differs"):
        reader.load_pair(0, 1, "wrong")


def test_compressed_uvd_cache_rejects_indefinite_hessian(tmp_path: Path) -> None:
    invalid = replace(_packet(), hessian=torch.diag(torch.tensor(
        [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64
    )))
    with pytest.raises(VisualFactorCacheError, match="positive semidefinite"):
        write_compressed_uvd_factor_cache(tmp_path, [invalid])

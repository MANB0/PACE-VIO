from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Utility.T2HistorySmoother import (
    compressed_factor_equivalence,
    factor_cost_breakdown,
    load_t2_history_archive,
    smooth_t2_history,
    state_arrays,
)


def _write_synthetic_archive(path: Path) -> None:
    count = 4
    identity = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64
    )
    poses = np.tile(identity, (count, 1))
    poses[2:, 0] = 0.1
    zeros3 = np.zeros((count, 3), dtype=np.float64)
    arrays = {
        "frames//pose": poses,
        "frames//time_ns": np.arange(count, dtype=np.int64) * 100_000_000,
        "frames//imu_vio_velocity_world": zeros3.copy(),
        "frames//imu_vio_acc_bias": zeros3.copy(),
        "frames//imu_vio_gyro_bias": zeros3.copy(),
        "frames//imu_vio_sensor_T_imu": np.tile(identity, (count, 1)),
        "frames//imu_vio_delta_rotvec": zeros3.copy(),
        "frames//imu_vio_delta_v": zeros3.copy(),
        "frames//imu_vio_delta_p": zeros3.copy(),
        "frames//imu_vio_cov": np.tile(np.eye(9), (count, 1, 1)),
        "frames//imu_vio_dt": np.asarray([0.0, 0.1, 0.1, 0.1]),
        "frames//imu_vio_bias_jacobian": np.zeros((count, 9, 6)),
        "frames//imu_vio_linearized_acc_bias": zeros3.copy(),
        "frames//imu_vio_linearized_gyro_bias": zeros3.copy(),
        "frames//imu_vio_bias_rw_cov": np.tile(np.eye(6), (count, 1, 1)),
        "frames//imu_vio_gravity_world": zeros3.copy(),
        "frames//imu_vio_gravity_in_residual": np.zeros(count, dtype=bool),
        "frames//visual_compressed_uvd_reference_CjCi": np.tile(identity, (count, 1)),
        "frames//visual_compressed_uvd_hessian": np.tile(np.eye(6) * 10.0, (count, 1, 1)),
        "frames//visual_compressed_uvd_gradient": np.zeros((count, 6)),
    }
    np.savez_compressed(path, **arrays)


class T2HistorySmootherTest(unittest.TestCase):
    def test_archive_contract_and_normal_equation_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor_map.npz"
            _write_synthetic_archive(path)
            archive = load_t2_history_archive(path, start_frame=0, end_frame=3)

        self.assertEqual(archive.frame_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(len(archive.edges), 3)
        report = compressed_factor_equivalence(archive)
        self.assertLess(report["max_hessian_relative_error"], 1.0e-12)
        self.assertLess(report["max_gradient_absolute_error"], 1.0e-12)

    def test_smoothing_reduces_cost_without_mutating_online_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor_map.npz"
            _write_synthetic_archive(path)
            archive = load_t2_history_archive(path, start_frame=0, end_frame=3)
            online_before = state_arrays(archive.online_states)
            result = smooth_t2_history(archive, max_iterations=8)

        online_after = state_arrays(archive.online_states)
        for name in online_before:
            np.testing.assert_array_equal(online_before[name], online_after[name])
        self.assertTrue(np.isfinite(result.final_cost))
        self.assertLess(result.final_cost, result.initial_cost)
        self.assertLess(
            abs(float(result.states[-1].pose_WB.reshape(7)[0].item())), 0.1
        )
        self.assertFalse(any(torch.isnan(state.pose_WB).any() for state in result.states))
        breakdown = factor_cost_breakdown(result.states, archive)
        reconstructed_cost = sum(float(values.sum()) for values in breakdown.values())
        self.assertAlmostEqual(reconstructed_cost, result.final_cost, places=9)


if __name__ == "__main__":
    unittest.main()

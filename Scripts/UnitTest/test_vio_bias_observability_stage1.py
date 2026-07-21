from __future__ import annotations

from pathlib import Path

import numpy as np

from Scripts.diagnose_vio_bias_observability_stage1 import (
    run_stage1_cases,
    write_stage1_report,
)


EXPECTED_CASES = {
    "current_two_frame",
    "relaxed_random_walk",
    "fixed_pose_terminal_states",
    "fixed_pose_velocity_start_bias",
}


def _rows_by_case():
    rows = run_stage1_cases()
    assert set(rows["case"]) == EXPECTED_CASES
    return rows.set_index("case")


def test_stage1_uses_real_sequence_timing_and_finite_covariances():
    rows = _rows_by_case()

    assert np.allclose(rows["dt_s"].to_numpy(dtype=float), 1.0 / 30.0, atol=1e-9)
    assert (rows["num_imu_samples"].to_numpy(dtype=int) == 5).all()
    assert (rows["preintegration_cov_min_diag"].to_numpy(dtype=float) > 0.0).all()
    assert (rows["bias_rw_cov_min_diag"].to_numpy(dtype=float) > 0.0).all()


def test_terminal_bias_has_no_imu_gradient_even_when_random_walk_is_relaxed():
    rows = _rows_by_case()

    for case in ("current_two_frame", "relaxed_random_walk", "fixed_pose_terminal_states"):
        assert float(rows.loc[case, "initial_curr_bias_imu_grad_norm"]) < 1e-12
        assert float(rows.loc[case, "estimated_curr_acc_bias_norm"]) < 1e-10
        assert float(rows.loc[case, "estimated_curr_gyro_bias_norm"]) < 1e-10

    assert float(rows.loc["relaxed_random_walk", "random_walk_cov_scale"]) > 1e6


def test_fixed_pose_case_updates_velocity_instead_of_terminal_bias():
    rows = _rows_by_case()
    row = rows.loc["fixed_pose_terminal_states"]

    assert float(row["velocity_update_norm"]) > 1e-5
    # Full 9x9 whitening couples p/v/R, so the weighted optimum need not drive
    # the raw velocity block exactly to zero while pose is fixed.
    assert float(row["final_velocity_residual_norm"]) < float(row["initial_velocity_residual_norm"]) * 0.20
    assert float(row["final_position_residual_norm"]) > 0.0
    assert float(row["final_rotation_residual_norm"]) > 0.0


def test_start_bias_is_observable_when_correct_pose_and_velocity_are_fixed():
    rows = _rows_by_case()
    row = rows.loc["fixed_pose_velocity_start_bias"]

    assert float(row["initial_start_bias_imu_grad_norm"]) > 1e-6
    assert float(row["estimated_start_acc_bias_cosine"]) > 0.99
    assert float(row["estimated_start_gyro_bias_cosine"]) > 0.99
    assert float(row["start_acc_bias_relative_error"]) < 0.10
    assert float(row["start_gyro_bias_relative_error"]) < 0.10
    assert float(row["final_imu_energy"]) < float(row["initial_imu_energy"]) * 0.05


def test_stage1_report_writes_csv_and_chinese_decision(tmp_path: Path):
    csv_path, report_path = write_stage1_report(tmp_path)

    assert csv_path.exists()
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    for case in EXPECTED_CASES:
        assert case in report
    assert "终点 bias" in report
    assert "起点 bias" in report
    assert "random-walk" in report
    assert "terminal_j" in report
    assert "start_i" in report

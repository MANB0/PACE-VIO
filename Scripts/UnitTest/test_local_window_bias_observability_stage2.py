from __future__ import annotations

from pathlib import Path

import numpy as np

from Scripts.diagnose_local_window_bias_observability_stage2 import (
    build_four_frame_problem,
    run_stage2_cases,
    write_stage2_report,
)


EXPECTED_CASES = {
    "w2_terminal_constant",
    "w3_middle_constant",
    "w3_middle_drift",
    "w3_zero_bias",
    "w3_shift_current",
    "w3_shift_all_optimized",
}


def _rows_by_case():
    rows = run_stage2_cases()
    assert set(rows["case"]) == EXPECTED_CASES
    return rows.set_index("case")


def test_stage2_uses_four_frames_three_edges_and_finite_covariances():
    problem = build_four_frame_problem("constant")
    rows = _rows_by_case()

    assert problem.frame_poses.shape[0] == 4
    assert len(problem.edges) == 3
    assert problem.frame_time_ns.tolist() == [0, 33_333_333, 66_666_667, 100_000_000]
    for edge_idx, sample_times in enumerate(problem.imu_time_ns):
        assert int(sample_times[0]) == int(problem.frame_time_ns[edge_idx])
        assert int(sample_times[-1]) == int(problem.frame_time_ns[edge_idx + 1])
        assert np.isclose(float(np.median(np.diff(sample_times.numpy()))), 10_000_000.0)

    assert (rows["num_camera_frames"].to_numpy(dtype=int) == 4).all()
    assert (rows["num_imu_edges"].to_numpy(dtype=int) == 3).all()
    assert (rows["camera_rate_hz"].to_numpy(dtype=float) == 30.0).all()
    assert (rows["imu_rate_hz"].to_numpy(dtype=float) == 100.0).all()
    assert np.isfinite(rows["preintegration_cov_min_diag"].to_numpy(dtype=float)).all()
    assert (rows["preintegration_cov_min_diag"].to_numpy(dtype=float) > 0.0).all()


def test_w2_terminal_bias_has_no_residual_jacobian_but_w3_middle_source_is_full_rank():
    rows = _rows_by_case()
    w2 = rows.loc["w2_terminal_constant"]
    w3 = rows.loc["w3_middle_constant"]

    assert float(w2["terminal_bias_residual_jacobian_norm"]) < 1e-12
    assert int(w2["terminal_bias_residual_jacobian_rank"]) == 0
    assert float(w3["source_bias_residual_jacobian_norm"]) > 1e-6
    assert int(w3["source_bias_residual_jacobian_rank"]) == 6
    assert float(w3["source_bias_residual_jacobian_min_singular"]) > 1e-6
    assert float(w3["terminal_bias_residual_jacobian_norm"]) < 1e-12
    assert int(w3["terminal_bias_residual_jacobian_rank"]) == 0


def test_w3_middle_bias_recovers_known_constant_bias_with_fixed_truth_states():
    row = _rows_by_case().loc["w3_middle_constant"]

    assert row["bias_recovery_objective"] == "imu_edge_only"
    assert float(row["acc_bias_cosine"]) > 0.99
    assert float(row["gyro_bias_cosine"]) > 0.99
    assert float(row["acc_bias_relative_error"]) < 0.10
    assert float(row["gyro_bias_relative_error"]) < 0.10
    assert float(row["final_imu_energy"]) < float(row["initial_imu_energy"]) * 0.05


def test_w3_zero_bias_control_remains_zero():
    row = _rows_by_case().loc["w3_zero_bias"]

    # Production preintegration and graph payloads are float32. A 1e-6 floor
    # keeps this control below the numerical resolution of the physical units
    # without mistaking sub-micro bias compensation for an estimated bias.
    assert float(row["estimated_acc_bias_norm"]) < 1e-6
    assert float(row["estimated_gyro_bias_norm"]) < 1e-6


def test_w3_middle_bias_tracks_drift_better_than_previous_bias_persistence():
    row = _rows_by_case().loc["w3_middle_drift"]

    assert row["bias_recovery_objective"] == "imu_edge_only"
    assert float(row["estimated_bias_error_norm"]) < float(row["persistence_baseline_bias_error_norm"]) * 0.10


def test_all_optimized_preserves_middle_bias_across_window_shift():
    rows = _rows_by_case()
    current = rows.loc["w3_shift_current"]
    all_optimized = rows.loc["w3_shift_all_optimized"]

    assert int(current["shifted_graph_constructed"]) == 1
    assert int(current["intermediate_writeback_applied"]) == 0
    assert float(current["shift_source_bias_error_norm"]) > 1e-6
    assert int(all_optimized["shifted_graph_constructed"]) == 1
    assert int(all_optimized["intermediate_writeback_applied"]) == 1
    assert float(all_optimized["shift_source_bias_error_norm"]) < 1e-8


def test_stage2_report_writes_csv_and_chinese_decision(tmp_path: Path):
    csv_path, report_path = write_stage2_report(tmp_path)

    assert csv_path.exists()
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    for case in EXPECTED_CASES:
        assert case in report
    assert "W=2" in report
    assert "W=3" in report
    assert "all_optimized" in report
    assert "9x6" in report
    assert "imu_edge_only" in report
    assert "上一帧保持" in report
    assert "three_frame_bias_path_and_all_writeback_confirmed" in report
    assert "不代表场景轨迹精度" in report

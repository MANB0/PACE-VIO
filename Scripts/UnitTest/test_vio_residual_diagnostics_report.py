import math

import pandas as pd

from Scripts.analyze_vio_residual_diagnostics_closed_paths import (
    DIAGNOSTIC_METRICS,
    augment_energy_columns,
    summarize_diagnostics,
    write_scene_html,
)


def test_summarize_diagnostics_reports_vio_residual_blocks_and_biases():
    rows = pd.DataFrame(
        [
            {
                "scene": "s",
                "method": "m",
                "vio_factor_active": 1,
                "r_p_whitened_norm": 1.0,
                "r_v_whitened_norm": 2.0,
                "r_R_whitened_norm": 3.0,
                "imu_vio_whitened_norm": 4.0,
                "visual_loss_per_residual": 0.1,
                "imu_vio_weight_diag_max": 100.0,
                "imu_vio_acc_bias_norm": 0.01,
                "imu_vio_gyro_bias_norm": 0.001,
            },
            {
                "scene": "s",
                "method": "m",
                "vio_factor_active": 1,
                "r_p_whitened_norm": 10.0,
                "r_v_whitened_norm": 20.0,
                "r_R_whitened_norm": 30.0,
                "imu_vio_whitened_norm": 40.0,
                "visual_loss_per_residual": 0.2,
                "imu_vio_weight_diag_max": 200.0,
                "imu_vio_acc_bias_norm": 0.02,
                "imu_vio_gyro_bias_norm": 0.002,
            },
        ]
    )

    summary = summarize_diagnostics(rows)

    assert len(summary) == 1
    row = summary.iloc[0].to_dict()
    assert row["scene"] == "s"
    assert row["method"] == "m"
    assert row["pairs"] == 2
    assert row["vio_active_pairs"] == 2
    assert math.isclose(row["r_p_whitened_norm_median"], 5.5)
    assert math.isclose(row["r_v_whitened_norm_max"], 20.0)
    assert math.isclose(row["r_R_whitened_norm_p95"], 28.65)
    assert math.isclose(row["imu_vio_whitened_norm_max"], 40.0)
    assert math.isclose(row["imu_vio_weight_diag_max_max"], 200.0)
    assert math.isclose(row["imu_vio_acc_bias_norm_final"], 0.02)
    assert math.isclose(row["imu_vio_gyro_bias_norm_final"], 0.002)


def test_augment_energy_columns_derives_legacy_energy_metrics():
    rows = pd.DataFrame(
        [
            {
                "scene": "s",
                "method": "m",
                "r_p_whitened_norm": 2.0,
                "r_v_whitened_norm": 3.0,
                "r_R_whitened_norm": 4.0,
                "imu_vio_whitened_norm": 6.0,
            }
        ]
    )

    out = augment_energy_columns(rows)
    row = out.iloc[0].to_dict()

    assert math.isclose(row["energy_p_weighted"], 4.0)
    assert math.isclose(row["energy_v_weighted"], 9.0)
    assert math.isclose(row["energy_R_weighted"], 16.0)
    assert math.isclose(row["energy_pv_weighted"], 13.0)
    assert math.isclose(row["energy_imu_diag_weighted"], 29.0)
    assert math.isclose(row["energy_imu_weighted"], 36.0)


def test_causal_summary_classifies_factor_victory_and_ignores_nonfinite_values(tmp_path):
    rows = pd.DataFrame(
        [
            {"scene": "s", "method": "m", "pair_id": 1, "frame_j": 2,
             "energy_visual_change": 2.0, "energy_imu_change": -3.0,
             "influence_sampled": 1, "actual_to_imu_step_cosine": 0.9},
            {"scene": "s", "method": "m", "pair_id": 2, "frame_j": 3,
             "energy_visual_change": -2.0, "energy_imu_change": 3.0,
             "influence_sampled": 1, "actual_to_imu_step_cosine": float("inf")},
            {"scene": "s", "method": "m", "pair_id": 3, "frame_j": 4,
             "energy_visual_change": -1.0, "energy_imu_change": -1.0,
             "influence_sampled": 0, "actual_to_imu_step_cosine": 0.2},
            {"scene": "s", "method": "m", "pair_id": 4, "frame_j": 5,
             "energy_visual_change": 1.0, "energy_imu_change": 1.0,
             "influence_sampled": 0, "actual_to_imu_step_cosine": float("nan")},
            {"scene": "s", "method": "m", "pair_id": 5, "frame_j": 6,
             "energy_visual_change": 1e-13, "energy_imu_change": -1e-13,
             "influence_sampled": 0, "actual_to_imu_step_cosine": -0.2},
        ]
    )

    summary = summarize_diagnostics(rows)
    row = summary.iloc[0]

    assert row["imu_wins_pairs"] == 1
    assert row["visual_wins_pairs"] == 1
    assert row["compatible_pairs"] == 1
    assert row["both_worse_pairs"] == 1
    assert row["undetermined_pairs"] == 1
    assert row["influence_sampled_pairs"] == 2
    assert math.isclose(row["actual_to_imu_step_cosine_median"], 0.2)
    assert row["factor_victory_majority"] == "tie"

    required_metrics = {
        "energy_visual_change",
        "energy_imu_change",
        "influence_imu_to_visual_grad_ratio",
        "influence_imu_to_visual_hessian_ratio",
        "actual_to_visual_step_cosine",
        "actual_to_imu_step_cosine",
        "init_rotation_error_angle",
        "imu_rotation_error_angle",
        "rotation_error_angle",
        "init_velocity_error_norm",
        "est_velocity_error_norm",
    }
    assert required_metrics.issubset(DIAGNOSTIC_METRICS)

    page = write_scene_html("s", rows, tmp_path)
    page_text = page.read_text(encoding="utf-8")
    assert "Factor-victory audit" in page_text
    assert "energy_visual_change" in page_text
    assert "energy_imu_change" in page_text
    assert "signed log10(1 + abs(value))" in page_text

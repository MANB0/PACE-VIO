from pathlib import Path

import Scripts.check_clear_circle_imu_gt_consistency as clear_diag


def test_build_report_text_summarizes_raw_and_mixed_coordinate_checks():
    summary_rows = [
        {
            "scene": "clear_circle_zero_noise",
            "variant": "raw_nwu",
            "protocol": "standard_vio",
            "n_pairs": 899,
            "rot_err_deg_median": 0.01,
            "rot_err_deg_p95": 0.02,
            "vel_err_median": 0.03,
            "vel_err_p95": 0.04,
            "pos_err_median": 0.001,
            "pos_err_p95": 0.002,
            "imu_over_gt_delta_p_median": 0.98,
            "imu_over_gt_delta_v_median": 1.01,
        },
        {
            "scene": "clear_circle_zero_noise",
            "variant": "imu_rx180_only_gt_nwu",
            "protocol": "standard_vio",
            "n_pairs": 899,
            "rot_err_deg_median": 5.0,
            "rot_err_deg_p95": 8.0,
            "vel_err_median": 1.2,
            "vel_err_p95": 2.4,
            "pos_err_median": 0.3,
            "pos_err_p95": 0.6,
            "imu_over_gt_delta_p_median": 3.0,
            "imu_over_gt_delta_v_median": 4.0,
        },
    ]
    roots = {"clear_circle_zero_noise": Path("/tmp/zero_noise/clear_circle_path")}

    report = clear_diag.build_report_text(summary_rows, roots)

    assert "clear_circle_zero_noise" in report
    assert "/tmp/zero_noise/clear_circle_path" in report
    assert "`raw_nwu + standard_vio`" in report
    assert "`imu_rx180_only_gt_nwu + standard_vio`" in report
    assert "坐标系错误检测" in report
    assert "自动结论" in report

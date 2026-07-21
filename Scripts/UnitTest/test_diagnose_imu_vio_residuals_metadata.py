import json
from pathlib import Path

import numpy as np

import Scripts.diagnose_imu_vio_residuals as diag
from Scripts.diagnose_imu_vio_residuals import load_gt


def _write_ref_pose(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "timestamp,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz",
                "0,0,0,0,0,0,1,0,-1,0,0,0,0,0",
                "1000000000,1,0,0,0,0,1,0,-1,0,0,0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_gt_uses_metadata_declared_world_velocity_frame(tmp_path):
    scene_root = tmp_path / "scene"
    scene_root.mkdir()
    _write_ref_pose(scene_root / "ref_pose.csv")
    (scene_root / "metadata.json").write_text(
        json.dumps(
            {
                "ground_truth": {"velocity_frame": "world NWU"},
                "coordinate_convention": {"ref_pose_velocity_frame": "world NWU"},
            }
        ),
        encoding="utf-8",
    )

    traj = load_gt(scene_root)

    assert traj.velocity_frame_used == "world"
    assert np.allclose(traj.v_w[0], [-1.0, 0.0, 0.0])


def test_write_report_distinguishes_metadata_velocity_frame_from_fd_check(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "OUTPUT_ROOT", tmp_path)
    summary_rows = [
        {
            "scene": "clear_shallow",
            "paper_role": "main",
            "variant": "raw_nwu",
            "protocol": "standard_vio",
            "rot_err_deg_median": 0.1,
            "vel_err_median": 0.01,
            "pos_err_median": 0.001,
            "imu_over_gt_delta_p_median": 1.0,
        },
        {
            "scene": "clear_shallow",
            "paper_role": "main",
            "variant": "imu_rx180_only_gt_nwu",
            "protocol": "standard_vio",
            "rot_err_deg_median": 10.0,
            "vel_err_median": 1.0,
            "pos_err_median": 0.1,
            "imu_over_gt_delta_p_median": 2.0,
        },
    ]
    velocity_checks = [
        {
            "scene": "clear_shallow",
            "metadata_velocity_frame": "world",
            "finite_difference_best_frame": "world",
            "velocity_frame_used": "world",
            "metadata_matches_finite_difference": True,
        }
    ]

    diag.write_report(summary_rows, records=[], velocity_checks=velocity_checks)

    report = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "metadata declares" in report
    assert "finite-difference check independently prefers" in report
    assert "better explained by treating" not in report

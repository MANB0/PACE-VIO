from __future__ import annotations

from copy import deepcopy
import csv

import yaml

import Scripts.run_direct_uvd_short_experiments as runner
import Scripts.run_circle_direct_uvd_sampling_aware_v2_full_parallel as parallel_runner
import Scripts.run_static63_cached_imu_fusion as full_runner


def _nested_differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_nested_differences(left[key], right[key], path))
        return differences
    return [] if left == right else [prefix]


def test_u1_sampling_aware_changes_only_covariance_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)

    current_path = runner.prepare_config("U1")
    current = yaml.safe_load(current_path.read_text(encoding="utf-8"))
    sampling_path = runner.prepare_config("U1_sampling_aware")
    sampling = yaml.safe_load(sampling_path.read_text(encoding="utf-8"))

    assert current["Odometry"]["args"]["two_state_covariance_mode"] == (
        "current_independent_step"
    )
    assert sampling["Odometry"]["args"]["two_state_covariance_mode"] == (
        "sampling_aware"
    )
    assert current["Odometry"]["optimizer"]["args"]["two_state_visual_factor_mode"] == (
        "direct_uvd"
    )
    assert sampling["Odometry"]["optimizer"]["args"]["two_state_warm_start"] == (
        "macvo_pose"
    )
    assert _nested_differences(current, sampling) == [
        "Odometry.args.two_state_covariance_mode"
    ]


def test_u1_sampling_aware_v2_changes_only_covariance_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)

    current = yaml.safe_load(
        runner.prepare_config("U1").read_text(encoding="utf-8")
    )
    sampling_v2 = yaml.safe_load(
        runner.prepare_config("U1_sampling_aware_v2").read_text(encoding="utf-8")
    )

    assert sampling_v2["Odometry"]["args"]["two_state_covariance_mode"] == (
        "sampling_aware_cross_edge"
    )
    assert _nested_differences(current, sampling_v2) == [
        "Odometry.args.two_state_covariance_mode"
    ]


def test_rank_aware_v2_changes_only_psd_whitening_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)

    sampling_v2 = yaml.safe_load(
        runner.prepare_config("U1_sampling_aware_v2").read_text(encoding="utf-8")
    )
    rank_aware = yaml.safe_load(
        runner.prepare_config("U1_sampling_aware_v2_rank_aware").read_text(
            encoding="utf-8"
        )
    )

    assert rank_aware["Odometry"]["optimizer"]["args"][
        "two_state_cross_edge_rank_aware_imu_whitening"
    ] is True
    assert _nested_differences(sampling_v2, rank_aware) == [
        "Odometry.optimizer.args.two_state_cross_edge_rank_aware_imu_whitening"
    ]


def test_current_u1_remains_the_approved_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    generated_path = runner.prepare_config("U1")
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    frozen = yaml.safe_load(runner.BASELINE_CONFIG.read_text(encoding="utf-8"))

    expected = deepcopy(frozen)
    optimizer = expected["Odometry"]["optimizer"]["args"]
    optimizer["two_state_visual_factor_mode"] = "direct_uvd"
    optimizer["two_state_warm_start"] = "macvo_pose"
    optimizer["two_state_uvd_huber_delta"] = 0.1
    expected["Odometry"]["args"]["mapping"] = False
    expected["Odometry"]["args"]["two_state_covariance_mode"] = (
        "current_independent_step"
    )

    assert generated == expected


def test_full_runner_exposes_sampling_aware_v2_as_only_covariance_change(tmp_path):
    def write_config(path):
        path.write_text(
            yaml.safe_dump(
                {
                    "Odometry": {
                        "args": {},
                        "optimizer": {"args": {}},
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    current_path = tmp_path / "current.yaml"
    sampling_v2_path = tmp_path / "sampling_v2.yaml"
    write_config(current_path)
    write_config(sampling_v2_path)
    common = {
        "static_initialization": False,
        "imu_vio_cov_diagonal_floor": None,
        "two_state_fixed_lag": True,
        "imu_vio_gravity_handling": "standard_local_frame_preintegration",
        "two_state_visual_factor_mode": "direct_uvd",
        "two_state_warm_start": "macvo_pose",
        "two_state_uvd_huber_delta": 0.1,
    }
    full_runner.configure_odometry(
        current_path,
        **common,
        two_state_covariance_mode="current_independent_step",
    )
    full_runner.configure_odometry(
        sampling_v2_path,
        **common,
        two_state_covariance_mode="sampling_aware_cross_edge",
    )

    current = yaml.safe_load(current_path.read_text(encoding="utf-8"))
    sampling_v2 = yaml.safe_load(sampling_v2_path.read_text(encoding="utf-8"))
    assert _nested_differences(current, sampling_v2) == [
        "Odometry.args.two_state_covariance_mode"
    ]


def test_parallel_full_runner_uses_sa_v1_and_sa_v2_and_parses_progress(tmp_path):
    assert set(parallel_runner.CASES) == {
        "sampling_aware_v1",
        "sampling_aware_v2",
    }
    assert parallel_runner.CASES["sampling_aware_v1"]["covariance_mode"] == (
        "sampling_aware"
    )
    assert parallel_runner.CASES["sampling_aware_v2"]["covariance_mode"] == (
        "sampling_aware_cross_edge"
    )

    log_path = tmp_path / "task.log"
    log_path.write_bytes(
        b"VisualMap(#frame=7, #point=10) 7/1890\n"
        b"VisualMap(#frame=123, #point=20) 123/1890\n"
    )
    assert parallel_runner.latest_logged_frame(log_path) == 123


def test_parallel_full_runner_writes_legacy_dashboard_manifest(tmp_path):
    manifest = tmp_path / "run_manifest.csv"
    parallel_runner.write_dashboard_manifest(manifest, tmp_path)

    with manifest.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["variant"] for row in rows] == [
        "vio_two_state_direct_uvd_sampling_aware_v1_full",
        "vio_two_state_direct_uvd_sampling_aware_v2_full",
    ]
    assert all(row["scene"] == parallel_runner.SCENE for row in rows)
    assert rows[0]["result_dir"] == str(
        parallel_runner.case_result_dir(tmp_path, "sampling_aware_v1")
    )
    assert rows[1]["result_dir"] == str(
        parallel_runner.case_result_dir(tmp_path, "sampling_aware_v2")
    )

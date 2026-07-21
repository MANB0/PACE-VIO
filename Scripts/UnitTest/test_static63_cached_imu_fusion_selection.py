from __future__ import annotations

import csv
from pathlib import Path

import yaml

from Scripts import run_static63_cached_imu_fusion as runner


ISOLATED_SCENES = [
    "clear_stop_turn_rectangle_truth_bias_no_noise",
    "clear_stop_turn_rectangle_truth_noise_no_bias",
]


def test_select_tasks_is_exact_and_ordered() -> None:
    selected = runner.select_tasks(ISOLATED_SCENES)

    assert [task.dataset_scene for task in selected] == ISOLATED_SCENES
    assert {task.trajectory for task in selected} == {"stop_turn_rectangle"}
    assert {task.cache_scene for task in selected} == {
        "clear_stop_turn_rectangle_truth_normal_noise"
    }


def test_select_tasks_rejects_unknown_scene() -> None:
    try:
        runner.select_tasks(["not_a_scene"])
    except ValueError as error:
        assert "unknown Static63 datasets" in str(error)
    else:
        raise AssertionError("unknown scene must be rejected")


def test_manifest_contains_only_selected_tasks_and_floor(tmp_path: Path) -> None:
    selected = runner.select_tasks(ISOLATED_SCENES)
    runner.write_manifest(
        tmp_path,
        None,
        tasks=selected,
        method_name="isolated_biasfix_floor_1e-8",
        static_initialization=True,
        imu_vio_cov_diagonal_floor=1e-8,
    )

    with (tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 2
    assert [row["scene"] for row in rows] == ISOLATED_SCENES
    assert all("corrected_bias_persistence=true" in row["args"] for row in rows)
    assert all("imu_vio_cov_diagonal_floor=1e-08" in row["args"] for row in rows)


def test_two_state_manifest_records_factor_mode(tmp_path: Path) -> None:
    selected = runner.select_tasks(ISOLATED_SCENES[:1])
    runner.write_manifest(
        tmp_path,
        20,
        tasks=selected,
        method_name=runner.TWO_STATE_FIXED_LAG_METHOD,
        static_initialization=True,
        two_state_fixed_lag=True,
        imu_vio_gravity_handling="standard_local_frame_preintegration",
    )

    with (tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert "imu_factor_mode=two_state_fixed_lag" in row["args"]
    assert (
        "imu_vio_gravity_handling=standard_local_frame_preintegration" in row["args"]
    )


def test_configure_odometry_applies_static_init_and_floor(tmp_path: Path) -> None:
    config_path = tmp_path / "odometry.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "Odometry": {
                    "args": {},
                    "optimizer": {"args": {"imu_vio_cov_diagonal_floor": 0.0}},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    runner.configure_odometry(
        config_path,
        static_initialization=True,
        imu_vio_cov_diagonal_floor=1e-8,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    odom_args = config["Odometry"]["args"]
    optimizer_args = config["Odometry"]["optimizer"]["args"]
    assert odom_args["imu_static_initialization_enable"] is True
    assert odom_args["imu_static_initialization_duration_s"] == 3.0
    assert optimizer_args["imu_vio_cov_diagonal_floor"] == 1e-8


def test_configure_odometry_enables_two_state_fixed_lag(tmp_path: Path) -> None:
    config_path = tmp_path / "odometry.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "Odometry": {
                    "args": {},
                    "optimizer": {
                        "args": {
                            "graph_type": "disp",
                            "autodiff": True,
                            "parallel": True,
                        }
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    runner.configure_odometry(
        config_path,
        static_initialization=True,
        imu_vio_cov_diagonal_floor=None,
        two_state_fixed_lag=True,
        imu_vio_gravity_handling="standard_local_frame_preintegration",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    optimizer_args = config["Odometry"]["optimizer"]["args"]
    assert optimizer_args["imu_factor_mode"] == "two_state_fixed_lag"
    assert optimizer_args["parallel"] is False
    assert optimizer_args["autodiff"] is True
    assert optimizer_args["two_state_initial_acc_bias_std"] == 0.2
    assert (
        config["Odometry"]["args"]["imu_vio_gravity_handling"]
        == "standard_local_frame_preintegration"
    )


def test_two_state_method_is_an_explicit_runner_choice() -> None:
    assert runner.METHOD_CHOICES["two-state-fixed-lag"] == (
        runner.TWO_STATE_FIXED_LAG_METHOD
    )


def test_reset_task_output_only_removes_a_child_of_result_root(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    output = result_root / "trial_1" / "method" / "scene"
    output.mkdir(parents=True)
    (output / "stale.txt").write_text("partial", encoding="utf-8")

    runner.reset_task_output(result_root, output)

    assert not output.exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        runner.reset_task_output(result_root, outside)
    except ValueError:
        pass
    else:
        raise AssertionError("task cleanup must reject paths outside the result root")

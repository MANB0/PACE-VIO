from argparse import Namespace
from pathlib import Path

import pytest

from Scripts.run_paper_experiments import (
    METHODS,
    _completed_result,
    _publish_pure_macvo,
    build_cache_record_command,
    build_command,
)
from Scripts.run_realtime_t2 import validate_visual_cache_options
from Scripts.run_realtime_t2 import configure_odom


def _cache_args(tmp_path: Path, mode: str, path: Path | None) -> Namespace:
    return Namespace(
        visual_cache_mode=mode,
        visual_cache_path=path,
        visual_factor="pace",
        seq_from=0,
        seq_to=-1,
    )


def test_live_mode_rejects_an_unused_cache_path(tmp_path: Path):
    with pytest.raises(ValueError, match="valid only"):
        validate_visual_cache_options(
            _cache_args(tmp_path, "live", tmp_path / "unused")
        )


def test_record_mode_requires_a_new_full_sequence_target(tmp_path: Path):
    cache = tmp_path / "cache"
    args = _cache_args(tmp_path, "record", cache)
    assert validate_visual_cache_options(args) == cache.resolve()

    cache.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validate_visual_cache_options(args)

    partial = _cache_args(tmp_path, "record", tmp_path / "partial")
    partial.seq_to = 10
    with pytest.raises(ValueError, match="complete sequence"):
        validate_visual_cache_options(partial)


def test_paper_commands_record_once_then_replay_the_same_cache(tmp_path: Path):
    dataset_path = tmp_path / "circle"
    dataset_path.mkdir()
    dataset = {
        "scenario": "Circle",
        "path": str(dataset_path),
        "static_initialization": {"mode": "fixed", "duration_s": 3.0},
    }
    runtime = {"mode": "pipeline", "cpu_threads": 2, "timeout_s": 60}
    cache = tmp_path / "visual_cache" / "circle"
    model = tmp_path / "model.pth"

    record = build_cache_record_command(
        dataset=dataset,
        output_root=tmp_path / "results",
        cache_path=cache,
        model=model,
        runtime=runtime,
        dry_run=True,
    )
    assert record[record.index("--visual-cache-mode") + 1] == "record"
    assert record[record.index("--visual-cache-path") + 1] == str(cache.resolve())
    assert "--no-paper-evaluation" in record

    commands = [
        build_command(
            dataset=dataset,
            method=method,
            output_root=tmp_path / "results",
            model=model,
            runtime=runtime,
            dry_run=True,
            visual_cache_mode="replay",
            visual_cache_path=cache,
        )
        for method in METHODS
    ]
    assert all(command[command.index("--visual-cache-mode") + 1] == "replay" for command in commands)
    assert all(command[command.index("--visual-cache-path") + 1] == str(cache.resolve()) for command in commands)


def test_field_replay_disables_inline_paper_evaluation(tmp_path: Path):
    dataset_path = tmp_path / "field"
    dataset_path.mkdir()
    dataset = {
        "scenario": "Field",
        "path": str(dataset_path),
        "paper_evaluation": False,
        "static_initialization": {"mode": "off"},
    }
    command = build_command(
        dataset=dataset,
        method=METHODS[-1],
        output_root=tmp_path / "results",
        model=tmp_path / "model.pth",
        runtime={},
        dry_run=True,
        visual_cache_mode="replay",
        visual_cache_path=tmp_path / "cache",
    )
    assert "--no-paper-evaluation" in command
    assert "--paper-evaluation" not in command


def test_no_evaluation_run_can_resume_from_execution_marker(tmp_path: Path):
    result = tmp_path / "result"
    result.mkdir()
    (result / "run_execution.json").write_text(
        '{"return_code": 0, "status": "ok"}',
        encoding="utf-8",
    )
    assert _completed_result(result, paper_evaluation=False)
    assert not _completed_result(result, paper_evaluation=True)


def test_pure_macvo_trajectories_are_published_as_scene_artifacts(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pure_macvo_poses_camera.csv").write_text("timestamp_ns,tx\n1,0\n")
    (cache / "pure_macvo_poses_imu.csv").write_text("timestamp_ns,tx\n1,0\n")
    dataset_path = tmp_path / "circle"
    dataset_path.mkdir()
    destination = _publish_pure_macvo(
        cache,
        tmp_path / "results",
        {"scenario": "Circle", "path": str(dataset_path)},
    )
    assert (destination / "poses_camera.csv").is_file()
    assert (destination / "poses_imu.csv").is_file()
    assert (destination / "manifest.json").is_file()


def test_record_configuration_is_pure_visual(tmp_path: Path):
    target = tmp_path / "record.yaml"
    configure_odom(
        target,
        model=tmp_path / "model.pth",
        trace=tmp_path / "trace.csv",
        parallel=False,
        static_mode="fixed",
        static_state_policy="estimated",
        static_duration_s=3.0,
        static_min_duration_s=1.0,
        static_max_duration_s=8.0,
        static_window_s=0.25,
        static_stable_hold_s=0.75,
        cpu_threads=4,
        vio_backend="isam2",
        visual_factor="pace",
        near_zero_velocity_detector="off",
        near_zero_velocity_prior_std_m_s=0.01,
        visual_cache_mode="record",
        visual_cache_path=tmp_path / "cache",
    )

    import yaml

    config = yaml.safe_load(target.read_text(encoding="utf-8"))["Odometry"]
    assert config["name"] == "MACVO-Visual-Cache-Recorder"
    assert config["args"]["imu_static_initialization_mode"] == "off"
    assert config["args"]["imu_rot_prior_enable"] is False
    assert config["args"]["imu_trans_prior_enable"] is False
    assert config["args"]["imu_pose_fusion_enable"] is False
    assert config["optimizer"] == {
        "type": "TwoFrame_PGO",
        "args": {
            "device": "cpu",
            "parallel": False,
            "autodiff": False,
            "graph_type": "disp",
            "vectorize": True,
            "imu_rot_prior": False,
        },
    }

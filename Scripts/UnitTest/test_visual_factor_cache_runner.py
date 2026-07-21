from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import MACVO as macvo_entrypoint
from Scripts import run_visual_factor_cache_batch as cache_batch
from Utility.VisualFactorCache import VisualFactorPacket, write_visual_factor_cache
from Utility.VisualInputFingerprint import visual_input_sha256


def _packet() -> VisualFactorPacket:
    rows = 1
    match_fields = {
        "pixel1_uv": torch.tensor([[10.0, 20.0]]),
        "pixel2_uv": torch.tensor([[11.0, 21.0]]),
        "pixel1_d": torch.full((rows, 1), 2.0),
        "pixel2_d": torch.full((rows, 1), 3.0),
        "pixel1_disp": torch.full((rows, 1), 5.0),
        "pixel2_disp": torch.full((rows, 1), 6.0),
        "pixel1_disp_cov": torch.full((rows, 1), 0.1),
        "pixel2_disp_cov": torch.full((rows, 1), 0.2),
        "pixel1_d_cov": torch.full((rows, 1), 0.3),
        "pixel2_d_cov": torch.full((rows, 1), 0.4),
        "pixel1_uv_cov": torch.tensor([[0.1, 0.2, 0.0]]),
        "pixel2_uv_cov": torch.tensor([[0.2, 0.3, 0.0]]),
        "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(rows, 1, 1),
        "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(rows, 1, 1) * 2.0,
    }
    return VisualFactorPacket(
        frame_i=0,
        frame_j=1,
        timestamp_i_ns=10,
        timestamp_j_ns=20,
        K=torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
        baseline_m=0.12,
        relative_pose_init=torch.eye(4, dtype=torch.float64),
        points_local=torch.tensor([[1.0, 2.0, 3.0]]),
        points_cov_local=torch.eye(3, dtype=torch.float64).repeat(rows, 1, 1),
        point_colors=torch.tensor([[1, 2, 3]], dtype=torch.uint8),
        match_fields=match_fields,
        covariance_diagnostics={"rows": rows},
        visual_sha256=visual_input_sha256(match_fields),
    )


def _runtime_diagnostics_csv(*, variant: str, visual_sha256: str) -> str:
    imuatt = variant == "vio_preintegrated_full_imuatt_estinit"
    header = (
        "frame_i,frame_j,timestamp_i,timestamp_j,visual_input_sha256,"
        "adaptive_mode,adaptive_use_rotation,adaptive_use_translation,"
        "use_imu_rotation,use_imu_translation,autodiff_enabled,"
        "imu_factor_mode,vio_factor_active\n"
    )
    row = (
        f"0,1,10,20,{visual_sha256},,,,{imuatt},{imuatt},{imuatt},"
        f"{'preintegrated_vio' if imuatt else 'legacy_pose_prior'},"
        f"{1 if imuatt else 0}\n"
    )
    return header + row


def _rewrite_runtime_diagnostics(path: Path, **updates: object) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        row.update({name: str(value) for name, value in updates.items()})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_complete_source_result(
    result_dir: Path,
    *,
    dataset_root: Path | None = None,
    scene: str = "scene",
) -> None:
    result_dir.mkdir(parents=True)
    (result_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"git_version": cache_batch.current_git_revision()}),
        encoding="utf-8",
    )
    config = {
        "Data": {
            "args": {
                "args": {
                    "root": str(dataset_root or result_dir / "dataset"),
                    "scene": scene,
                }
            }
        },
        "Odometry": {
            "args": {
                "mapping": False,
                "imu_pose_fusion_enable": False,
                "imu_rot_prior_enable": False,
                "imu_trans_prior_enable": False,
                "imu_trans_prior_mode": "off",
                "imu_vio_gravity_handling": "preintegration",
            },
            "frontend": {"type": "CUDAGraph_FlowFormerCovFrontend", "args": {}},
            "motion": {"type": "StaticMotionModel", "args": None},
            "keyframe": {"type": "AllKeyframe", "args": None},
            "optimizer": {
                "type": "TwoFrame_PGO",
                "args": {
                    "autodiff": False,
                    "graph_type": "disp",
                    "imu_factor_mode": "legacy_pose_prior",
                    "imu_rot_prior": False,
                    "imu_trans_prior_scale": 1.0,
                    "parallel": True,
                    "post_imu_fusion_enable": False,
                    "post_imu_fusion_mode": "none",
                },
            },
        }
    }
    (result_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    (result_dir / "poses.csv").write_text(
        "timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n10,0,0,0,0,0,0,1\n20,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    packet = _packet()
    (result_dir / "visual_factor_diagnostics.csv").write_text(
        "frame_i,frame_j,timestamp_i,timestamp_j,visual_input_sha256\n"
        f"0,1,10,20,{packet.visual_sha256}\n",
        encoding="utf-8",
    )
    (result_dir / "frame_pair_diagnostics.csv").write_text(
        _runtime_diagnostics_csv(
            variant="pure_macvo",
            visual_sha256=packet.visual_sha256,
        ),
        encoding="utf-8",
    )
    tensor_map = {
        "frames//time_ns": np.array([10, 20], dtype=np.int64),
        "edge/match2frame1/mapping": np.array([0], dtype=np.int64),
        "edge/match2frame2/mapping": np.array([1], dtype=np.int64),
        "edge/match2point/mapping": np.array([0], dtype=np.int64),
        "points//pos_Tc": packet.points_local.detach().cpu().numpy(),
        "points//pos_Tw": packet.points_local.detach().cpu().numpy(),
        "points//cov_Tw": packet.points_cov_local.detach().cpu().numpy(),
        "points//color": packet.point_colors.detach().cpu().numpy(),
        **{
            f"match//{name}": value.detach().cpu().numpy()
            for name, value in packet.match_fields.items()
        },
    }
    np.savez_compressed(result_dir / "tensor_map.npz", **tensor_map)


def _write_complete_replay_result(
    result_dir: Path,
    cache_dir: Path,
    *,
    variant: str = "pure_macvo",
) -> None:
    result_dir.mkdir(parents=True)
    (result_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"git_version": cache_batch.current_git_revision()}),
        encoding="utf-8",
    )
    imuatt = variant == "vio_preintegrated_full_imuatt_estinit"
    from Utility.VisualFactorCache import VisualFactorCacheReader

    manifest = VisualFactorCacheReader(cache_dir).manifest
    configured_dataset = manifest.source.get("dataset", str(result_dir / "dataset"))
    config = {
        "Data": {
            "args": {
                "args": {
                    "root": str(configured_dataset),
                    "scene": manifest.scene,
                }
            }
        },
        "Odometry": {
            "args": {
                "mapping": False,
                "visual_cache_mode": "replay",
                "visual_cache_path": str(cache_dir),
                "imu_pose_fusion_enable": False,
                "imu_rot_prior_enable": imuatt,
                "imu_trans_prior_enable": imuatt,
                "imu_trans_prior_mode": "imu_velocity_composed" if imuatt else "off",
                "imu_vio_gravity_pose_source": "imu_integrated_estinit" if imuatt else "estimated",
                "imu_vio_gravity_handling": "preintegration",
            },
            "frontend": {"type": "ReplayFrontend", "args": {}},
            "motion": {"type": "StaticMotionModel", "args": None},
            "keyframe": {"type": "AllKeyframe", "args": None},
            "optimizer": {
                "type": "TwoFrame_PGO",
                "args": {
                    "autodiff": imuatt,
                    "graph_type": "disp",
                    "imu_factor_mode": "preintegrated_vio" if imuatt else "legacy_pose_prior",
                    "imu_rot_prior": imuatt,
                    "imu_trans_prior_scale": 1.0,
                    "parallel": True,
                    "post_imu_fusion_enable": False,
                    "post_imu_fusion_mode": "none",
                },
            },
        }
    }
    (result_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    (result_dir / "poses.csv").write_text(
        "timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n10,0,0,0,0,0,0,1\n20,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    visual_hash = _packet().visual_sha256
    (result_dir / "frame_pair_diagnostics.csv").write_text(
        _runtime_diagnostics_csv(variant=variant, visual_sha256=visual_hash),
        encoding="utf-8",
    )


def test_replay_cli_override_replaces_only_frontend_and_injects_cache_args(tmp_path: Path):
    config = {
        "args": {"mapping": False, "device": "cuda"},
        "frontend": {"type": "CUDAGraph_FlowFormerCovFrontend", "args": {"device": "cuda"}},
        "optimizer": {"type": "TwoFrame_PGO", "args": {"imu_factor_mode": "preintegrated_vio"}},
    }
    optimizer_before = deepcopy(config["optimizer"])
    cache_path = tmp_path / "cache"
    args = SimpleNamespace(visual_cache_mode="replay", visual_cache_path=cache_path)

    macvo_entrypoint._apply_visual_cache_cli_overrides(args, config)

    assert config["args"]["visual_cache_mode"] == "replay"
    assert config["args"]["visual_cache_path"] == str(cache_path)
    assert config["frontend"] == {"type": "ReplayFrontend", "args": {}}
    assert config["optimizer"] == optimizer_before


@pytest.mark.parametrize(
    ("mode", "cache_path", "message"),
    [
        ("off", Path("unused"), "requires --visual-cache-mode replay"),
        ("replay", None, "is required"),
    ],
)
def test_visual_cache_cli_rejects_incomplete_argument_pairs(mode, cache_path, message):
    config = {
        "args": {},
        "frontend": {"type": "CUDAGraph_FlowFormerCovFrontend", "args": {}},
        "optimizer": {"type": "TwoFrame_PGO", "args": {}},
    }

    with pytest.raises(ValueError, match=message):
        macvo_entrypoint._apply_visual_cache_cli_overrides(
            SimpleNamespace(visual_cache_mode=mode, visual_cache_path=cache_path),
            config,
        )


def test_scene_identity_prefers_explicit_sequence_scene_over_directory_name():
    data_config = SimpleNamespace(
        args=SimpleNamespace(
            scene="clear_circle_zero_noise",
            root="/dataset/zero_noise/clear_circle_path",
        )
    )

    assert macvo_entrypoint._resolve_scene_name(data_config) == "clear_circle_zero_noise"


def test_four_scene_schedule_contains_every_phase_once_per_scene():
    tasks = cache_batch.build_schedule()

    assert cache_batch.dashboard_url(8765) == "http://127.0.0.1:8765/"
    assert len(tasks) == 20
    for phase in ("source", "export", "replay-pure", "replay-imuatt", "verify"):
        phase_tasks = [task for task in tasks if task.phase == phase]
        assert len(phase_tasks) == 4
        assert {task.scene for task in phase_tasks} == set(cache_batch.SCENE_ROOTS)
    assert {task.variant for task in tasks if task.phase == "source"} == {"pure_macvo"}
    assert {task.variant for task in tasks if task.phase == "replay-pure"} == {"pure_macvo"}
    assert {task.variant for task in tasks if task.phase == "replay-imuatt"} == {
        "vio_preintegrated_full_imuatt_estinit"
    }
    assert {task.variant for task in tasks if task.phase in {"export", "verify"}} == {""}
    assert {task.variant for task in tasks if task.variant} == cache_batch.RETAINED_METHODS


def test_batch_retains_only_pure_macvo_and_current_bugfixed_imuatt():
    expected = {
        "pure_macvo",
        "vio_preintegrated_full_imuatt_estinit",
    }
    assert getattr(cache_batch, "RETAINED_METHODS", set()) == expected
    assert set(getattr(cache_batch, "RETAINED_VARIANTS", {})) == expected


def test_method_contract_rejects_archived_diagnostic_variant(tmp_path: Path):
    result_dir = tmp_path / "source"
    _write_complete_source_result(result_dir)
    config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))

    valid, reason = cache_batch._validate_method_contract(
        config["Odometry"],
        expected_variant="rotation_only",
    )

    assert not valid
    assert "retained" in reason or "unsupported" in reason


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("optimizer", "imu_vio_alpha_R", 0.0),
        ("optimizer", "imu_vio_alpha_p", 0.1),
        ("optimizer", "imu_vio_alpha_v", 0.1),
        ("optimizer", "imu_vio_cov_scale", 1000.0),
        ("odometry", "imu_vio_velocity_feedback_enable", False),
        ("odometry", "imu_vio_bias_feedback_enable", False),
    ],
)
def test_latest_imuatt_contract_rejects_archived_diagnostic_override(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
):
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [_packet()], source={"frame_count": 2})
    result_dir = tmp_path / "replay"
    _write_complete_replay_result(
        result_dir,
        cache_dir,
        variant="vio_preintegrated_full_imuatt_estinit",
    )
    config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))
    target = (
        config["Odometry"]["optimizer"]["args"]
        if section == "optimizer"
        else config["Odometry"]["args"]
    )
    target[field] = value

    valid, reason = cache_batch._validate_method_contract(
        config["Odometry"],
        expected_variant="vio_preintegrated_full_imuatt_estinit",
    )

    assert not valid
    assert field in reason


def test_latest_imuatt_runtime_rejects_adaptive_controller_artifact(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [_packet()], source={"frame_count": 2})
    result_dir = tmp_path / "replay"
    _write_complete_replay_result(
        result_dir,
        cache_dir,
        variant="vio_preintegrated_full_imuatt_estinit",
    )
    _rewrite_runtime_diagnostics(
        result_dir / "frame_pair_diagnostics.csv",
        adaptive_mode="pure_macvo",
        adaptive_use_rotation=False,
        adaptive_use_translation=False,
    )

    valid, reason = cache_batch.validate_replay_artifact(
        result_dir,
        cache_dir=cache_dir,
        scene="scene",
        expected_frame_count=2,
        expected_variant="vio_preintegrated_full_imuatt_estinit",
    )

    assert not valid
    assert "adaptive" in reason


def test_source_resume_rejects_artifact_from_different_code_revision(tmp_path: Path):
    result_dir = tmp_path / "source"
    _write_complete_source_result(result_dir)
    (result_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"git_version": "stale-revision"}),
        encoding="utf-8",
    )

    valid, reason = cache_batch.validate_source_artifact(
        result_dir,
        expected_frame_count=2,
        expected_variant="pure_macvo",
    )

    assert not valid
    assert "revision" in reason or "git" in reason


def test_formal_batch_rejects_dirty_tracked_runtime_code(monkeypatch):
    def fake_run(command, **_kwargs):
        assert command[:4] == ["git", "diff", "--name-only", "HEAD"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Odometry/MACVO.py\n",
            stderr="",
        )

    monkeypatch.setattr(cache_batch.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="uncommitted tracked runtime code"):
        cache_batch.validate_formal_code_state()


def test_formal_batch_rejects_revision_change_between_tasks(monkeypatch):
    def fake_run(command, **_kwargs):
        if command[:4] == ["git", "diff", "--name-only", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "rev-parse", "--short", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="newrev\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(cache_batch.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="revision changed during batch"):
        cache_batch.validate_formal_code_state(expected_revision="oldrev")


def test_source_and_replay_commands_have_opposite_frontend_contracts(tmp_path: Path):
    tasks = cache_batch.build_schedule(
        source_result_root=tmp_path / "source",
        cache_root=tmp_path / "cache",
        replay_result_root=tmp_path / "replay",
        control_root=tmp_path / "control",
    )
    source = next(task for task in tasks if task.phase == "source")
    replay = next(task for task in tasks if task.phase == "replay-pure")
    replay_imuatt = next(task for task in tasks if task.phase == "replay-imuatt")
    odom_cfg = tmp_path / "odom.yaml"
    seq_cfg = tmp_path / "seq.yaml"

    source_cmd = cache_batch.build_task_command(source, odom_cfg=odom_cfg, seq_cfg=seq_cfg, seq_to=120)
    replay_cmd = cache_batch.build_task_command(replay, odom_cfg=odom_cfg, seq_cfg=seq_cfg, seq_to=120)
    replay_imuatt_cmd = cache_batch.build_task_command(
        replay_imuatt,
        odom_cfg=odom_cfg,
        seq_cfg=seq_cfg,
        seq_to=120,
    )

    assert "--visual-cache-mode" not in source_cmd
    assert "--visual-cache-path" not in source_cmd
    assert replay_cmd[-4:] == [
        "--visual-cache-mode",
        "replay",
        "--visual-cache-path",
        str(replay.cache_dir),
    ]
    assert source_cmd[source_cmd.index("--seq_to") + 1] == "120"
    assert replay_cmd[replay_cmd.index("--seq_to") + 1] == "120"
    for command in (source_cmd, replay_cmd, replay_imuatt_cmd):
        assert "--adaptive-v3b" not in command
        assert "--v3b-force" not in command
        assert "--v3b-vc-mode" not in command


def test_manifest_has_one_unique_dashboard_key_per_scheduled_task(tmp_path: Path):
    tasks = cache_batch.build_schedule(
        source_result_root=tmp_path / "source",
        cache_root=tmp_path / "cache",
        replay_result_root=tmp_path / "replay",
        control_root=tmp_path / "control",
    )

    manifest = cache_batch.write_manifest(tmp_path / "control", tasks, seq_to=None)

    with manifest.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(row["trial"], row["scene"], row["variant"]) for row in rows}
    assert len(rows) == 20
    assert len(keys) == 20
    assert {row["phase"] for row in rows} == set(cache_batch.PHASES)


def test_custom_analysis_root_owns_verification_artifacts(tmp_path: Path):
    analysis_root = tmp_path / "analysis"
    tasks = cache_batch.build_schedule(
        source_result_root=tmp_path / "source",
        cache_root=tmp_path / "cache",
        replay_result_root=tmp_path / "replay",
        control_root=tmp_path / "control",
        analysis_root=analysis_root,
    )

    assert {
        task.result_dir.parent
        for task in tasks
        if task.phase == "verify"
    } == {analysis_root / "verification"}


def test_relative_runtime_roots_are_resolved_against_project_root():
    args = SimpleNamespace(
        source_result_root=Path("Results/source-relative"),
        cache_root=Path("VisualCache/cache-relative"),
        replay_result_root=Path("Results/replay-relative"),
        control_root=Path("Results/control-relative"),
        analysis_root=Path("analysis-relative"),
        log_path=Path("logs/relative.log"),
    )

    cache_batch.resolve_runtime_paths(args)

    for name in (
        "source_result_root",
        "cache_root",
        "replay_result_root",
        "control_root",
        "analysis_root",
        "log_path",
    ):
        value = getattr(args, name)
        assert value.is_absolute()
        assert str(value).startswith(str(cache_batch.WORKDIR))


def test_runtime_layout_rejects_overlapping_output_roots(tmp_path: Path):
    args = SimpleNamespace(
        source_result_root=tmp_path / "outputs",
        cache_root=tmp_path / "outputs" / "cache",
        replay_result_root=tmp_path / "replay",
        control_root=tmp_path / "control",
        analysis_root=tmp_path / "analysis",
    )

    with pytest.raises(ValueError, match="overlap"):
        cache_batch.validate_runtime_layout(args)


def test_runtime_layout_rejects_cross_device_staging(monkeypatch, tmp_path: Path):
    args = SimpleNamespace(
        source_result_root=tmp_path / "source",
        cache_root=tmp_path / "cache",
        replay_result_root=tmp_path / "replay",
        control_root=tmp_path / "control",
        analysis_root=tmp_path / "analysis",
    )
    monkeypatch.setattr(
        cache_batch,
        "_path_device",
        lambda path: 1 if Path(path) == args.control_root else 2,
    )

    with pytest.raises(ValueError, match="filesystem"):
        cache_batch.validate_runtime_layout(args)


def test_source_resume_requires_complete_adjacent_visual_diagnostics(tmp_path: Path):
    result_dir = tmp_path / "source"
    _write_complete_source_result(result_dir)

    valid, reason = cache_batch.validate_source_artifact(result_dir, expected_frame_count=2)
    assert valid, reason

    (result_dir / "visual_factor_diagnostics.csv").write_text(
        "frame_i,frame_j,timestamp_i,timestamp_j,visual_input_sha256\n",
        encoding="utf-8",
    )
    valid, reason = cache_batch.validate_source_artifact(result_dir, expected_frame_count=2)
    assert not valid
    assert "diagnostic" in reason


def test_source_resume_rejects_nonfinite_pose_values(tmp_path: Path):
    result_dir = tmp_path / "source"
    _write_complete_source_result(result_dir)
    poses_path = result_dir / "poses.csv"
    poses_path.write_text(
        poses_path.read_text(encoding="utf-8").replace(
            "20,0,0,0,0,0,0,1",
            "20,nan,0,0,0,0,0,1",
        ),
        encoding="utf-8",
    )

    valid, reason = cache_batch.validate_source_artifact(
        result_dir,
        expected_frame_count=2,
        expected_variant="pure_macvo",
    )

    assert not valid
    assert "pose" in reason


def test_source_resume_rejects_runtime_mode_mismatch(tmp_path: Path):
    result_dir = tmp_path / "source"
    _write_complete_source_result(result_dir)
    _rewrite_runtime_diagnostics(
        result_dir / "frame_pair_diagnostics.csv",
        adaptive_mode="full_imu",
        adaptive_use_rotation=True,
        adaptive_use_translation=True,
        use_imu_rotation=True,
        use_imu_translation=True,
        autodiff_enabled=True,
        imu_factor_mode="preintegrated_vio",
        vio_factor_active=1,
    )

    valid, reason = cache_batch.validate_source_artifact(
        result_dir,
        expected_frame_count=2,
        expected_variant="pure_macvo",
    )

    assert not valid
    assert "runtime" in reason or "adaptive_mode" in reason


def test_source_resume_requires_authoritative_macvo_frontend(tmp_path: Path):
    result_dir = tmp_path / "source"
    _write_complete_source_result(result_dir)
    config_path = result_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["Odometry"]["frontend"]["type"] = "GTFrontend"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    valid, reason = cache_batch.validate_source_artifact(
        result_dir,
        expected_frame_count=2,
        expected_variant="pure_macvo",
    )

    assert not valid
    assert "frontend" in reason


def test_source_resume_recomputes_visual_hash_from_tensor_map(tmp_path: Path):
    result_dir = tmp_path / "source"
    _write_complete_source_result(result_dir)
    rows = list(csv.DictReader((result_dir / "visual_factor_diagnostics.csv").open(encoding="utf-8")))
    rows[0]["visual_input_sha256"] = "0" * 64
    with (result_dir / "visual_factor_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid, reason = cache_batch.validate_source_artifact(result_dir, expected_frame_count=2)

    assert not valid
    assert "visual" in reason and "hash" in reason


def test_source_resume_is_bound_to_scheduled_scene_and_dataset(tmp_path: Path):
    result_dir = tmp_path / "source"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _write_complete_source_result(result_dir, dataset_root=dataset_root, scene="scene")

    valid, reason = cache_batch.validate_source_artifact(
        result_dir,
        expected_frame_count=2,
        expected_scene="scene",
        expected_dataset_root=dataset_root,
    )
    assert valid, reason

    valid, reason = cache_batch.validate_source_artifact(
        result_dir,
        expected_frame_count=2,
        expected_scene="other-scene",
        expected_dataset_root=dataset_root,
    )
    assert not valid
    assert "scene" in reason

    valid, reason = cache_batch.validate_source_artifact(
        result_dir,
        expected_frame_count=2,
        expected_scene="scene",
        expected_dataset_root=tmp_path / "other-dataset",
    )
    assert not valid
    assert "dataset" in reason


def test_cache_resume_revalidates_every_packet_checksum(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [_packet()], source={"frame_count": 2})

    valid, reason = cache_batch.validate_cache_artifact(cache_dir, scene="scene", expected_frame_count=2)
    assert valid, reason

    pair_path = cache_dir / "pairs" / "000000_000001.npz"
    pair_path.write_bytes(pair_path.read_bytes() + b"corrupt")
    valid, reason = cache_batch.validate_cache_artifact(cache_dir, scene="scene", expected_frame_count=2)
    assert not valid
    assert "checksum" in reason


def test_cache_resume_recomputes_packet_visual_hash(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    bad_packet = replace(_packet(), visual_sha256="0" * 64)
    write_visual_factor_cache(cache_dir, "scene", [bad_packet], source={"frame_count": 2})

    valid, reason = cache_batch.validate_cache_artifact(
        cache_dir,
        scene="scene",
        expected_frame_count=2,
    )

    assert not valid
    assert "visual" in reason and "hash" in reason


def test_cache_resume_is_bound_to_exact_source_result_and_files(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source_dir = tmp_path / "source"
    _write_complete_source_result(source_dir, dataset_root=dataset_root, scene="scene")
    config_hash = hashlib.sha256((source_dir / "config.yaml").read_bytes()).hexdigest()
    tensor_hash = hashlib.sha256((source_dir / "tensor_map.npz").read_bytes()).hexdigest()
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(
        cache_dir,
        "scene",
        [_packet()],
        source={
            "frame_count": 2,
            "dataset": str(dataset_root.resolve()),
            "result": str(source_dir.resolve()),
            "config": config_hash,
            "checksums": tensor_hash,
            "git": cache_batch.current_git_revision(),
            "motion": "StaticMotionModel",
            "keyframes": "AllKeyframe",
        },
    )

    valid, reason = cache_batch.validate_cache_artifact(
        cache_dir,
        scene="scene",
        expected_frame_count=2,
        source_result_dir=source_dir,
        dataset_root=dataset_root,
    )
    assert valid, reason

    (source_dir / "tensor_map.npz").write_bytes((source_dir / "tensor_map.npz").read_bytes() + b"changed")
    valid, reason = cache_batch.validate_cache_artifact(
        cache_dir,
        scene="scene",
        expected_frame_count=2,
        source_result_dir=source_dir,
        dataset_root=dataset_root,
    )
    assert not valid
    assert "source tensor-map" in reason


def test_source_resume_requires_exact_runtime_local_points(tmp_path: Path):
    source_dir = tmp_path / "source"
    _write_complete_source_result(source_dir)
    with np.load(source_dir / "tensor_map.npz", allow_pickle=False) as data:
        tensor_map = {name: np.array(data[name], copy=True) for name in data.files}
    tensor_map.pop("points//pos_Tc")
    np.savez_compressed(source_dir / "tensor_map.npz", **tensor_map)

    valid, reason = cache_batch.validate_source_artifact(
        source_dir,
        expected_frame_count=2,
    )

    assert not valid
    assert "points//pos_Tc" in reason


def test_source_resume_rejects_exact_local_point_row_count_mismatch(tmp_path: Path):
    source_dir = tmp_path / "source"
    _write_complete_source_result(source_dir)
    with np.load(source_dir / "tensor_map.npz", allow_pickle=False) as data:
        tensor_map = {name: np.array(data[name], copy=True) for name in data.files}
    tensor_map["points//pos_Tc"] = np.concatenate(
        [tensor_map["points//pos_Tc"], tensor_map["points//pos_Tc"]],
        axis=0,
    )
    np.savez_compressed(source_dir / "tensor_map.npz", **tensor_map)

    valid, reason = cache_batch.validate_source_artifact(
        source_dir,
        expected_frame_count=2,
    )

    assert not valid
    assert "point tensor row counts" in reason


def test_source_resume_rejects_non_float32_exact_local_points(tmp_path: Path):
    source_dir = tmp_path / "source"
    _write_complete_source_result(source_dir)
    with np.load(source_dir / "tensor_map.npz", allow_pickle=False) as data:
        tensor_map = {name: np.array(data[name], copy=True) for name in data.files}
    tensor_map["points//pos_Tc"] = tensor_map["points//pos_Tc"].astype(np.float64)
    np.savez_compressed(source_dir / "tensor_map.npz", **tensor_map)

    valid, reason = cache_batch.validate_source_artifact(
        source_dir,
        expected_frame_count=2,
    )

    assert not valid
    assert "float32" in reason


def test_cache_resume_rejects_local_points_that_differ_from_source(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source_dir = tmp_path / "source"
    _write_complete_source_result(source_dir, dataset_root=dataset_root, scene="scene")
    mismatched_packet = replace(
        _packet(),
        points_local=_packet().points_local + torch.tensor([[0.25, 0.0, 0.0]]),
    )
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(
        cache_dir,
        "scene",
        [mismatched_packet],
        source={
            "frame_count": 2,
            "dataset": str(dataset_root.resolve()),
            "result": str(source_dir.resolve()),
            "config": hashlib.sha256((source_dir / "config.yaml").read_bytes()).hexdigest(),
            "checksums": hashlib.sha256((source_dir / "tensor_map.npz").read_bytes()).hexdigest(),
            "git": cache_batch.current_git_revision(),
            "motion": "StaticMotionModel",
            "keyframes": "AllKeyframe",
        },
    )

    valid, reason = cache_batch.validate_cache_artifact(
        cache_dir,
        scene="scene",
        expected_frame_count=2,
        source_result_dir=source_dir,
        dataset_root=dataset_root,
    )

    assert not valid
    assert "local points" in reason


def test_cache_resume_binds_packet_hashes_to_source_visual_diagnostics(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source_dir = tmp_path / "source"
    _write_complete_source_result(source_dir, dataset_root=dataset_root, scene="scene")
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(
        cache_dir,
        "scene",
        [_packet()],
        source={
            "frame_count": 2,
            "dataset": str(dataset_root.resolve()),
            "result": str(source_dir.resolve()),
            "config": hashlib.sha256((source_dir / "config.yaml").read_bytes()).hexdigest(),
            "checksums": hashlib.sha256((source_dir / "tensor_map.npz").read_bytes()).hexdigest(),
            "git": cache_batch.current_git_revision(),
            "motion": "StaticMotionModel",
            "keyframes": "AllKeyframe",
        },
    )
    diagnostics_path = source_dir / "visual_factor_diagnostics.csv"
    diagnostics = list(csv.DictReader(diagnostics_path.open(encoding="utf-8")))
    diagnostics[0]["visual_input_sha256"] = "f" * 64
    with diagnostics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    valid, reason = cache_batch.validate_cache_artifact(
        cache_dir,
        scene="scene",
        expected_frame_count=2,
        source_result_dir=source_dir,
        dataset_root=dataset_root,
    )

    assert not valid
    assert "source visual" in reason


def test_replay_resume_requires_replay_frontend_and_matching_visual_hashes(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [_packet()], source={"frame_count": 2})
    result_dir = tmp_path / "replay"
    _write_complete_replay_result(result_dir, cache_dir)

    valid, reason = cache_batch.validate_replay_artifact(
        result_dir,
        cache_dir=cache_dir,
        scene="scene",
        expected_frame_count=2,
        expected_variant="pure_macvo",
    )
    assert valid, reason

    config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))
    config["Odometry"]["frontend"]["type"] = "CUDAGraph_FlowFormerCovFrontend"
    (result_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    valid, reason = cache_batch.validate_replay_artifact(
        result_dir,
        cache_dir=cache_dir,
        scene="scene",
        expected_frame_count=2,
        expected_variant="pure_macvo",
    )
    assert not valid
    assert "ReplayFrontend" in reason


def test_replay_resume_rejects_pure_result_for_imuatt_task(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [_packet()], source={"frame_count": 2})
    result_dir = tmp_path / "replay"
    _write_complete_replay_result(result_dir, cache_dir, variant="pure_macvo")

    valid, reason = cache_batch.validate_replay_artifact(
        result_dir,
        cache_dir=cache_dir,
        scene="scene",
        expected_frame_count=2,
        expected_variant="vio_preintegrated_full_imuatt_estinit",
    )

    assert not valid
    assert "imu_factor_mode" in reason or "imu_" in reason or "IMU" in reason


def test_replay_resume_rejects_material_optimizer_setting_change(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [_packet()], source={"frame_count": 2})
    result_dir = tmp_path / "replay"
    _write_complete_replay_result(
        result_dir,
        cache_dir,
        variant="vio_preintegrated_full_imuatt_estinit",
    )
    config_path = result_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["Odometry"]["optimizer"]["args"]["autodiff"] = False
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    valid, reason = cache_batch.validate_replay_artifact(
        result_dir,
        cache_dir=cache_dir,
        scene="scene",
        expected_frame_count=2,
        expected_variant="vio_preintegrated_full_imuatt_estinit",
    )

    assert not valid
    assert "autodiff" in reason


def test_replay_resume_rejects_nonfinite_pose_values(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [_packet()], source={"frame_count": 2})
    result_dir = tmp_path / "replay"
    _write_complete_replay_result(
        result_dir,
        cache_dir,
        variant="vio_preintegrated_full_imuatt_estinit",
    )
    poses_path = result_dir / "poses.csv"
    poses_path.write_text(
        poses_path.read_text(encoding="utf-8").replace(
            "20,0,0,0,0,0,0,1",
            "20,inf,0,0,0,0,0,1",
        ),
        encoding="utf-8",
    )

    valid, reason = cache_batch.validate_replay_artifact(
        result_dir,
        cache_dir=cache_dir,
        scene="scene",
        expected_frame_count=2,
        expected_variant="vio_preintegrated_full_imuatt_estinit",
    )

    assert not valid
    assert "pose" in reason


def test_replay_resume_rejects_runtime_mode_mismatch(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [_packet()], source={"frame_count": 2})
    result_dir = tmp_path / "replay"
    _write_complete_replay_result(
        result_dir,
        cache_dir,
        variant="vio_preintegrated_full_imuatt_estinit",
    )
    _rewrite_runtime_diagnostics(
        result_dir / "frame_pair_diagnostics.csv",
        adaptive_mode="pure_macvo",
        adaptive_use_rotation=False,
        adaptive_use_translation=False,
        use_imu_rotation=False,
        use_imu_translation=False,
        autodiff_enabled=False,
        imu_factor_mode="legacy_pose_prior",
        vio_factor_active=0,
    )

    valid, reason = cache_batch.validate_replay_artifact(
        result_dir,
        cache_dir=cache_dir,
        scene="scene",
        expected_frame_count=2,
        expected_variant="vio_preintegrated_full_imuatt_estinit",
    )

    assert not valid
    assert "runtime" in reason or "adaptive_mode" in reason


def test_verification_resume_is_invalidated_when_any_compared_artifact_changes(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source_dir = tmp_path / "source"
    _write_complete_source_result(source_dir, dataset_root=dataset_root, scene="scene")
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(
        cache_dir,
        "scene",
        [_packet()],
        source={
            "frame_count": 2,
            "dataset": str(dataset_root.resolve()),
            "result": str(source_dir.resolve()),
            "config": hashlib.sha256((source_dir / "config.yaml").read_bytes()).hexdigest(),
            "checksums": hashlib.sha256((source_dir / "tensor_map.npz").read_bytes()).hexdigest(),
            "git": cache_batch.current_git_revision(),
            "motion": "StaticMotionModel",
            "keyframes": "AllKeyframe",
        },
    )
    replay_pure = tmp_path / "replay-pure"
    replay_imuatt = tmp_path / "replay-imuatt"
    _write_complete_replay_result(replay_pure, cache_dir, variant="pure_macvo")
    _write_complete_replay_result(
        replay_imuatt,
        cache_dir,
        variant="vio_preintegrated_full_imuatt_estinit",
    )
    verification_dir = tmp_path / "verification"
    task = cache_batch.BatchTask(
        phase="verify",
        scene="scene",
        variant="",
        scene_root=dataset_root,
        result_dir=verification_dir,
        cache_dir=cache_dir,
        manifest_variant="verify_cache_replay",
        source_result_dir=source_dir,
        replay_pure_result_dir=replay_pure,
        replay_imuatt_result_dir=replay_imuatt,
    )
    passed, reason = cache_batch.verify_scene_task(task, expected_frame_count=2)
    assert passed, reason
    valid, reason = cache_batch.validate_verification_artifact(
        task,
        expected_frame_count=2,
    )
    assert valid, reason

    source_runtime_path = source_dir / "frame_pair_diagnostics.csv"
    source_runtime_original = source_runtime_path.read_bytes()
    source_runtime_path.write_bytes(source_runtime_original + b"\n")
    valid, reason = cache_batch.validate_verification_artifact(
        task,
        expected_frame_count=2,
    )
    assert not valid
    assert "stale" in reason or "fingerprint" in reason
    source_runtime_path.write_bytes(source_runtime_original)
    valid, reason = cache_batch.validate_verification_artifact(
        task,
        expected_frame_count=2,
    )
    assert valid, reason

    with (replay_pure / "poses.csv").open("a", encoding="utf-8") as stream:
        stream.write("30,0,0,0,0,0,0,1\n")
    valid, reason = cache_batch.validate_verification_artifact(
        task,
        expected_frame_count=2,
    )
    assert not valid
    assert "stale" in reason or "fingerprint" in reason


def test_promote_attempt_archives_old_bundle_and_replaces_it_atomically(tmp_path: Path):
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    archive_root = tmp_path / "archive"
    staging.mkdir()
    final.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    (final / "old.txt").write_text("old", encoding="utf-8")

    archived = cache_batch.promote_attempt_directory(
        staging,
        final,
        archive_root=archive_root,
    )

    assert (final / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (final / "old.txt").exists()
    assert archived is not None
    assert (archived / "old.txt").read_text(encoding="utf-8") == "old"


def test_guarded_promotion_rolls_back_when_code_changes_after_promotion(
    tmp_path: Path,
    monkeypatch,
):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "old.txt").write_text("old", encoding="utf-8")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    (staging_dir / "new.txt").write_text("new", encoding="utf-8")
    calls = 0

    def changing_code_state(*, expected_revision=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("runtime code changed")
        return expected_revision or "rev"

    monkeypatch.setattr(cache_batch, "validate_formal_code_state", changing_code_state)

    with pytest.raises(ValueError, match="runtime code changed"):
        cache_batch.promote_attempt_directory_guarded(
            staging_dir,
            final_dir,
            archive_root=tmp_path / "archive",
            rejected_root=tmp_path / "rejected",
            expected_git_revision="rev",
        )

    assert calls == 2
    assert (final_dir / "old.txt").read_text(encoding="utf-8") == "old"
    rejected = list((tmp_path / "rejected").iterdir())
    assert len(rejected) == 1
    assert (rejected[0] / "new.txt").read_text(encoding="utf-8") == "new"


def test_dashboard_pid_filter_only_selects_requested_port(monkeypatch):
    fake_ps = """\
  PID CMD
  111 python Scripts/run_progress_dashboard.py --result-root /a --port 8765
  222 python Scripts/run_progress_dashboard.py --result-root /b --port 9999
  333 python Scripts/run_progress_dashboard.py --result-root /c
  444 python unrelated.py --port 8765
"""

    monkeypatch.setattr(
        cache_batch.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=fake_ps, returncode=0),
    )

    assert cache_batch.existing_dashboard_pids(port=8765) == [111, 333]
    assert cache_batch.existing_dashboard_pids(port=9999) == [222]


def test_progress_schema_records_active_staging_artifact(tmp_path: Path):
    task = cache_batch.build_schedule(
        source_result_root=tmp_path / "source",
        cache_root=tmp_path / "cache",
        replay_result_root=tmp_path / "replay",
        control_root=tmp_path / "control",
        analysis_root=tmp_path / "analysis",
    )[0]
    active_dir = tmp_path / "control" / "staging" / "source" / task.scene / "attempt"

    cache_batch.append_progress(
        tmp_path / "control",
        task,
        status="running",
        active_result_dir=active_dir,
    )

    with (tmp_path / "control" / "progress.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["result_dir"] == str(task.result_dir)
    assert row["active_result_dir"] == str(active_dir)


def test_changed_batch_signature_archives_stale_manifest_and_progress(tmp_path: Path):
    control_root = tmp_path / "control"
    tasks_a = cache_batch.build_schedule(
        source_result_root=tmp_path / "source-a",
        cache_root=tmp_path / "cache-a",
        replay_result_root=tmp_path / "replay-a",
        control_root=control_root,
        analysis_root=tmp_path / "analysis-a",
    )
    cache_batch.prepare_control_state(control_root, tasks_a, seq_to=120)
    (control_root / "progress.csv").write_text("trial,scene,variant\n1,old,old\n", encoding="utf-8")
    (control_root / "logs").mkdir()
    (control_root / "logs" / "old.log").write_text("old context\n", encoding="utf-8")

    tasks_b = cache_batch.build_schedule(
        source_result_root=tmp_path / "source-b",
        cache_root=tmp_path / "cache-b",
        replay_result_root=tmp_path / "replay-b",
        control_root=control_root,
        analysis_root=tmp_path / "analysis-b",
    )
    cache_batch.prepare_control_state(control_root, tasks_b, seq_to=900)

    assert not (control_root / "progress.csv").exists()
    archives = list((control_root / "context_archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "progress.csv").exists()
    assert (archives[0] / "logs" / "old.log").exists()
    context = json.loads((control_root / "batch_context.json").read_text(encoding="utf-8"))
    assert context["seq_to"] == 900


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("all", set(cache_batch.PHASES)),
        ("source-export", {"source", "export"}),
        ("replay-verify", {"replay-pure", "replay-imuatt", "verify"}),
        ("verify", {"verify"}),
    ],
)
def test_phase_selection_is_explicit(selection, expected):
    assert set(cache_batch.selected_phases(selection)) == expected


def test_dry_run_writes_full_manifest_without_starting_processes(tmp_path: Path, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not start a dashboard or execute a task")

    monkeypatch.setattr(cache_batch, "switch_dashboard", forbidden)
    monkeypatch.setattr(cache_batch, "execute_task", forbidden)
    args = SimpleNamespace(
        phase="all",
        source_result_root=tmp_path / "source",
        cache_root=tmp_path / "cache",
        replay_result_root=tmp_path / "replay",
        control_root=tmp_path / "control",
        analysis_root=tmp_path / "analysis",
        log_path=tmp_path / "batch.log",
        timeout=10,
        seq_to=None,
        dashboard_port=8765,
        no_dashboard=False,
        dry_run=True,
        force=False,
        jobs=1,
    )

    assert cache_batch.run_batch(args) == 0

    output = capsys.readouterr().out
    assert "20 scheduled tasks" in output
    assert "http://127.0.0.1:8765/" in output
    assert "method=-" in output
    assert "method=verify" not in output
    with (args.control_root / "run_manifest.csv").open("r", newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 20


def test_runner_is_directly_executable_from_project_root(tmp_path: Path):
    script = cache_batch.WORKDIR / "Scripts" / "run_visual_factor_cache_batch.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--source-result-root",
            str(tmp_path / "source"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--replay-result-root",
            str(tmp_path / "replay"),
            "--control-root",
            str(tmp_path / "control"),
        ],
        cwd=str(cache_batch.WORKDIR),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert process.returncode == 0, process.stderr
    assert "20 scheduled tasks" in process.stdout

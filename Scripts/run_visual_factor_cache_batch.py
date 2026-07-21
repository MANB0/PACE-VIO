#!/usr/bin/env python3
"""Run and verify the fixed visual-factor cache/replay experiment batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from Scripts.run_vio_imu_prior_mode_grid import (
    RunSpec,
    Variant,
    flatten_nested,
    make_odom_cfg,
    make_seq_cfg,
)
from Utility.VisualFactorCache import MATCH_FIELDS, VisualFactorCacheError, VisualFactorCacheReader
from Utility.VisualInputFingerprint import visual_input_sha256


WORKDIR = PROJECT_ROOT
PYTHON = Path("/home/admin1/miniconda3/envs/macvo/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)
BATCH_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_zed100_closed_paths_smooth_20260705"
)

SCENE_ROOTS = {
    "clear_circle_zero_noise": BATCH_ROOT / "zero_noise" / "clear_circle_path",
    "clear_rectangle_zero_noise": BATCH_ROOT / "zero_noise" / "clear_rectangle_path",
    "clear_circle_normal_noise": BATCH_ROOT / "normal_noise" / "clear_circle_path",
    "clear_rectangle_normal_noise": BATCH_ROOT / "normal_noise" / "clear_rectangle_path",
}

SOURCE_RESULT_ROOT = WORKDIR / "Results" / "visual_factor_cache_source_20260712"
CACHE_ROOT = WORKDIR / "VisualCache" / "closed_paths_20260712"
REPLAY_RESULT_ROOT = WORKDIR / "Results" / "visual_factor_cache_replay_20260712"
CONTROL_ROOT = WORKDIR / "Results" / "visual_factor_cache_batch_20260712"
ANALYSIS_ROOT = WORKDIR / "analysis_visual_factor_cache_replay_20260712"
DEFAULT_LOG_PATH = WORKDIR / "logs" / "visual_factor_cache_batch_20260712.log"
DEFAULT_DASHBOARD_LOG = WORKDIR / "logs" / "progress_dashboard_8765.log"

PHASES = ("source", "export", "replay-pure", "replay-imuatt", "verify")
PHASE_SELECTIONS = {
    "all": PHASES,
    "source-export": ("source", "export"),
    "replay-verify": ("replay-pure", "replay-imuatt", "verify"),
    **{phase: (phase,) for phase in PHASES},
}

PURE_REPLAY_TRANSLATION_RMSE_MAX_M = 1.0e-5
PURE_REPLAY_TRANSLATION_MAX_M = 2.0e-5
PURE_REPLAY_ROTATION_RMSE_MAX_DEG = 1.0e-5
PURE_REPLAY_ROTATION_MAX_DEG = 1.0e-4

PURE_MACVO_METHOD = "pure_macvo"
LATEST_IMUATT_METHOD = "vio_preintegrated_full_imuatt_estinit"
RETAINED_METHODS = frozenset({PURE_MACVO_METHOD, LATEST_IMUATT_METHOD})
SOURCE_FRONTEND_TYPE = "CUDAGraph_FlowFormerCovFrontend"
REPLAY_FRONTEND_TYPE = "ReplayFrontend"
FORMAL_CODE_PATHS = (
    "MACVO.py",
    "Config",
    "DataLoader",
    "Module",
    "Odometry",
    "Utility",
    "Scripts/export_visual_factor_cache.py",
    "Scripts/run_visual_factor_cache_batch.py",
    "Scripts/run_vio_imu_prior_mode_grid.py",
)

RETAINED_VARIANTS = {
    PURE_MACVO_METHOD: Variant(
        PURE_MACVO_METHOD,
        "off",
        1.0,
        False,
        False,
        PURE_MACVO_METHOD,
    ),
    LATEST_IMUATT_METHOD: Variant(
        LATEST_IMUATT_METHOD,
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "imu_integrated_estinit",
    ),
}

METHOD_CONTRACTS: dict[str, dict[str, dict[str, object]]] = {
    PURE_MACVO_METHOD: {
        "odometry": {
            "mapping": False,
            "imu_pose_fusion_enable": False,
            "imu_rot_prior_enable": False,
            "imu_trans_prior_enable": False,
            "imu_trans_prior_mode": "off",
            "imu_vio_gravity_handling": "preintegration",
        },
        "optimizer": {
            "autodiff": False,
            "graph_type": "disp",
            "imu_factor_mode": "legacy_pose_prior",
            "imu_rot_prior": False,
            "imu_trans_prior_scale": 1.0,
            "parallel": True,
            "post_imu_fusion_enable": False,
            "post_imu_fusion_mode": "none",
        },
        "runtime": {
            "adaptive_mode": None,
            "adaptive_use_rotation": None,
            "adaptive_use_translation": None,
            "use_imu_rotation": False,
            "use_imu_translation": False,
            "autodiff_enabled": False,
            "imu_factor_mode": "legacy_pose_prior",
            "vio_factor_active": False,
        },
        "odometry_defaults": {
            "imu_vio_velocity_feedback_enable": True,
            "imu_vio_bias_feedback_enable": True,
        },
        "optimizer_defaults": {
            "imu_vio_alpha_p": 1.0,
            "imu_vio_alpha_v": 1.0,
            "imu_vio_alpha_R": 1.0,
            "imu_vio_cov_scale": 1.0,
        },
    },
    LATEST_IMUATT_METHOD: {
        "odometry": {
            "mapping": False,
            "imu_pose_fusion_enable": False,
            "imu_rot_prior_enable": True,
            "imu_trans_prior_enable": True,
            "imu_trans_prior_mode": "imu_velocity_composed",
            "imu_vio_gravity_pose_source": "imu_integrated_estinit",
            "imu_vio_gravity_handling": "preintegration",
        },
        "optimizer": {
            "autodiff": True,
            "graph_type": "disp",
            "imu_factor_mode": "preintegrated_vio",
            "imu_rot_prior": True,
            "imu_trans_prior_scale": 1.0,
            "parallel": True,
            "post_imu_fusion_enable": False,
            "post_imu_fusion_mode": "none",
        },
        "runtime": {
            "adaptive_mode": None,
            "adaptive_use_rotation": None,
            "adaptive_use_translation": None,
            "use_imu_rotation": True,
            "use_imu_translation": True,
            "autodiff_enabled": True,
            "imu_factor_mode": "preintegrated_vio",
            "vio_factor_active": True,
        },
        "odometry_defaults": {
            "imu_vio_velocity_feedback_enable": True,
            "imu_vio_bias_feedback_enable": True,
        },
        "optimizer_defaults": {
            "imu_vio_alpha_p": 1.0,
            "imu_vio_alpha_v": 1.0,
            "imu_vio_alpha_R": 1.0,
            "imu_vio_cov_scale": 1.0,
        },
    },
}


@dataclass(frozen=True)
class BatchTask:
    phase: str
    scene: str
    variant: str
    scene_root: Path
    result_dir: Path
    cache_dir: Path
    manifest_variant: str
    source_result_dir: Path
    replay_pure_result_dir: Path
    replay_imuatt_result_dir: Path


def dashboard_url(port: int) -> str:
    return f"http://127.0.0.1:{int(port)}/"


def build_schedule(
    *,
    source_result_root: Path = SOURCE_RESULT_ROOT,
    cache_root: Path = CACHE_ROOT,
    replay_result_root: Path = REPLAY_RESULT_ROOT,
    control_root: Path = CONTROL_ROOT,
    analysis_root: Path = ANALYSIS_ROOT,
) -> list[BatchTask]:
    tasks: list[BatchTask] = []
    for phase in PHASES:
        for scene, scene_root in SCENE_ROOTS.items():
            cache_dir = cache_root / scene
            source_result_dir = source_result_root / "trial_1" / PURE_MACVO_METHOD / scene
            replay_pure_result_dir = replay_result_root / "trial_1" / PURE_MACVO_METHOD / scene
            replay_imuatt_result_dir = (
                replay_result_root
                / "trial_1"
                / LATEST_IMUATT_METHOD
                / scene
            )
            if phase == "source":
                variant = PURE_MACVO_METHOD
                result_dir = source_result_dir
                manifest_variant = "source_pure_macvo"
            elif phase == "export":
                variant = ""
                result_dir = cache_dir
                manifest_variant = "export_visual_cache"
            elif phase == "replay-pure":
                variant = PURE_MACVO_METHOD
                result_dir = replay_pure_result_dir
                manifest_variant = "replay_pure_macvo"
            elif phase == "replay-imuatt":
                variant = LATEST_IMUATT_METHOD
                result_dir = replay_imuatt_result_dir
                manifest_variant = "replay_imuatt"
            elif phase == "verify":
                variant = ""
                result_dir = analysis_root / "verification" / scene
                manifest_variant = "verify_cache_replay"
            else:  # pragma: no cover - PHASES is a closed constant above.
                raise AssertionError(f"unsupported batch phase: {phase}")
            tasks.append(
                BatchTask(
                    phase=phase,
                    scene=scene,
                    variant=variant,
                    scene_root=scene_root,
                    result_dir=result_dir,
                    cache_dir=cache_dir,
                    manifest_variant=manifest_variant,
                    source_result_dir=source_result_dir,
                    replay_pure_result_dir=replay_pure_result_dir,
                    replay_imuatt_result_dir=replay_imuatt_result_dir,
                )
            )
    return tasks


def build_task_command(
    task: BatchTask,
    *,
    odom_cfg: Path | None = None,
    seq_cfg: Path | None = None,
    seq_to: int | None = None,
) -> list[str]:
    if task.phase in {"source", "replay-pure", "replay-imuatt"}:
        if odom_cfg is None or seq_cfg is None:
            raise ValueError(f"{task.phase} requires odom_cfg and seq_cfg")
        command = [
            str(PYTHON),
            str(WORKDIR / "MACVO.py"),
            "--odom",
            str(odom_cfg),
            "--data",
            str(seq_cfg),
            "--resultRoot",
            str(task.result_dir),
        ]
        if seq_to is not None:
            command.extend(["--seq_to", str(int(seq_to))])
        if task.phase.startswith("replay-"):
            command.extend(
                [
                    "--visual-cache-mode",
                    "replay",
                    "--visual-cache-path",
                    str(task.cache_dir),
                ]
            )
        return command
    if task.phase == "export":
        return [
            str(PYTHON),
            str(WORKDIR / "Scripts" / "export_visual_factor_cache.py"),
            str(task.source_result_dir),
            str(task.cache_dir),
            task.scene,
            str(task.scene_root),
        ]
    if task.phase == "verify":
        return []
    raise ValueError(f"unsupported task phase: {task.phase!r}")


def selected_phases(selection: str) -> tuple[str, ...]:
    try:
        return tuple(PHASE_SELECTIONS[str(selection)])
    except KeyError as error:
        raise ValueError(f"unsupported phase selection: {selection!r}") from error


def resolve_runtime_paths(args: argparse.Namespace) -> None:
    for name in (
        "source_result_root",
        "cache_root",
        "replay_result_root",
        "control_root",
        "analysis_root",
        "log_path",
    ):
        value = Path(getattr(args, name)).expanduser()
        if not value.is_absolute():
            value = WORKDIR / value
        setattr(args, name, value.resolve())


def _path_device(path: Path) -> int:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return int(candidate.stat().st_dev)


def validate_runtime_layout(args: argparse.Namespace) -> None:
    roots = {
        name: Path(getattr(args, name)).expanduser().resolve()
        for name in (
            "source_result_root",
            "cache_root",
            "replay_result_root",
            "control_root",
            "analysis_root",
        )
    }
    root_items = list(roots.items())
    for index, (left_name, left_path) in enumerate(root_items):
        for right_name, right_path in root_items[index + 1:]:
            if left_path == right_path or left_path in right_path.parents or right_path in left_path.parents:
                raise ValueError(
                    f"output roots overlap: {left_name}={left_path} and {right_name}={right_path}"
                )
    staging_device = _path_device(roots["control_root"])
    for name, path in roots.items():
        if _path_device(path) != staging_device:
            raise ValueError(
                f"{name}={path} is on a different filesystem from control_root; "
                "atomic staging promotion requires one filesystem"
            )


def validate_formal_code_state(*, expected_revision: str | None = None) -> str:
    process = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *FORMAL_CODE_PATHS],
        cwd=WORKDIR,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    dirty_paths = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if dirty_paths:
        listed = ", ".join(dirty_paths[:8])
        suffix = "" if len(dirty_paths) <= 8 else f", ... ({len(dirty_paths)} files)"
        raise ValueError(
            "formal batch refuses uncommitted tracked runtime code: "
            f"{listed}{suffix}"
        )
    revision = current_git_revision()
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            "formal code revision changed during batch: "
            f"started at {expected_revision!r}, now {revision!r}"
        )
    return revision


MANIFEST_FIELDS = [
    "trial",
    "scene",
    "variant",
    "phase",
    "method_variant",
    "scene_root",
    "result_dir",
    "cache_dir",
    "seq_to",
    "created_at",
]


def write_manifest(control_root: Path, tasks: list[BatchTask], seq_to: int | None) -> Path:
    control_root.mkdir(parents=True, exist_ok=True)
    manifest_path = control_root / "run_manifest.csv"
    created_at = datetime.now().isoformat(timespec="seconds")
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "trial": 1,
                    "scene": task.scene,
                    "variant": task.manifest_variant,
                    "phase": task.phase,
                    "method_variant": task.variant,
                    "scene_root": str(task.scene_root),
                    "result_dir": str(task.result_dir),
                    "cache_dir": str(task.cache_dir),
                    "seq_to": "" if seq_to is None else int(seq_to),
                    "created_at": created_at,
                }
            )
    return manifest_path


def _batch_context(tasks: list[BatchTask], seq_to: int | None) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": 1,
        "seq_to": seq_to,
        "tasks": [
            {
                "phase": task.phase,
                "scene": task.scene,
                "variant": task.variant,
                "scene_root": str(task.scene_root.resolve()),
                "result_dir": str(task.result_dir.resolve()),
                "cache_dir": str(task.cache_dir.resolve()),
                "source_result_dir": str(task.source_result_dir.resolve()),
                "replay_pure_result_dir": str(task.replay_pure_result_dir.resolve()),
                "replay_imuatt_result_dir": str(task.replay_imuatt_result_dir.resolve()),
            }
            for task in tasks
        ],
    }
    serialized = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**core, "run_id": hashlib.sha256(serialized.encode("utf-8")).hexdigest()}


def prepare_control_state(
    control_root: Path,
    tasks: list[BatchTask],
    *,
    seq_to: int | None,
) -> Path:
    control_root.mkdir(parents=True, exist_ok=True)
    context_path = control_root / "batch_context.json"
    desired = _batch_context(tasks, seq_to)
    current: dict[str, object] | None = None
    try:
        current = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    state_paths = [
        control_root / "run_manifest.csv",
        control_root / "progress.csv",
        control_root / "logs",
        context_path,
    ]
    has_existing_state = any(path.exists() for path in state_paths)
    if has_existing_state and (current is None or current.get("run_id") != desired["run_id"]):
        archive = control_root / "context_archive" / str(time.time_ns())
        archive.mkdir(parents=True, exist_ok=False)
        for path in state_paths:
            if path.exists():
                os.replace(path, archive / path.name)
    temporary_context = context_path.with_suffix(".json.tmp")
    temporary_context.write_text(
        json.dumps(desired, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_context, context_path)
    return write_manifest(control_root, tasks, seq_to)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def current_git_revision() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=WORKDIR,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    revision = process.stdout.strip()
    if not revision:
        raise ValueError("current git revision is empty")
    return revision


def _artifact_git_revision(result_dir: Path) -> str:
    metadata = yaml.safe_load((result_dir / "metadata.yaml").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("artifact metadata is not a mapping")
    revision = str(metadata.get("git_version", "")).strip()
    if not revision:
        raise ValueError("artifact metadata is missing git_version")
    return revision


def _source_visual_hashes(
    tensor_map: np.lib.npyio.NpzFile,
    *,
    frame_count: int,
) -> dict[tuple[int, int], str]:
    match_fields = {
        name: np.asarray(tensor_map[f"match//{name}"])
        for name in MATCH_FIELDS
    }
    match_rows = len(match_fields["pixel1_uv"])
    if any(len(values) != match_rows for values in match_fields.values()):
        raise ValueError("source match tensor row counts differ")
    frame1 = np.asarray(tensor_map["edge/match2frame1/mapping"])
    frame2 = np.asarray(tensor_map["edge/match2frame2/mapping"])
    if frame1.shape != (match_rows,) or frame2.shape != (match_rows,):
        raise ValueError("source match-to-frame mappings have invalid shape")
    hashes: dict[tuple[int, int], str] = {}
    for frame_i in range(frame_count - 1):
        rows = np.flatnonzero((frame1 == frame_i) & (frame2 == frame_i + 1))
        if rows.size == 0:
            raise ValueError(f"source visual pair ({frame_i}, {frame_i + 1}) has no matches")
        packet_fields = {
            name: torch.from_numpy(np.array(values[rows], copy=True))
            for name, values in match_fields.items()
        }
        hashes[(frame_i, frame_i + 1)] = visual_input_sha256(packet_fields)
    return hashes


def _source_exact_local_points(
    tensor_map: np.lib.npyio.NpzFile | dict[str, np.ndarray],
    *,
    frame_count: int,
) -> dict[tuple[int, int], np.ndarray]:
    try:
        frame1 = np.asarray(tensor_map["edge/match2frame1/mapping"])
        frame2 = np.asarray(tensor_map["edge/match2frame2/mapping"])
        point_mapping = np.asarray(tensor_map["edge/match2point/mapping"])
        points_local = np.asarray(tensor_map["points//pos_Tc"])
        points_world = np.asarray(tensor_map["points//pos_Tw"])
        points_cov_world = np.asarray(tensor_map["points//cov_Tw"])
        point_colors = np.asarray(tensor_map["points//color"])
    except KeyError as error:
        raise ValueError(f"source tensor-map is missing {error.args[0]}") from error
    match_rows = len(frame1)
    if frame2.shape != (match_rows,) or point_mapping.shape != (match_rows,):
        raise ValueError("source exact local-point mappings have invalid shape")
    if not all(np.issubdtype(values.dtype, np.integer) for values in (frame1, frame2, point_mapping)):
        raise ValueError("source exact local-point mappings must contain integers")
    if points_local.ndim != 2 or points_local.shape[1:] != (3,):
        raise ValueError("source points//pos_Tc has invalid shape")
    point_count = len(points_world)
    if any(len(values) != point_count for values in (points_local, points_cov_world, point_colors)):
        raise ValueError("source point tensor row counts differ")
    if points_local.dtype != np.float32:
        raise ValueError("source points//pos_Tc must use float32")
    if not np.isfinite(points_local).all():
        raise ValueError("source points//pos_Tc contains non-finite values")
    if np.any(point_mapping < 0) or np.any(point_mapping >= len(points_local)):
        raise ValueError("source exact local-point mapping is out of bounds")
    by_pair: dict[tuple[int, int], np.ndarray] = {}
    for frame_i in range(frame_count - 1):
        pair = (frame_i, frame_i + 1)
        rows = np.flatnonzero((frame1 == pair[0]) & (frame2 == pair[1]))
        if len(rows) == 0:
            raise ValueError(f"source exact local points are missing pair {pair}")
        by_pair[pair] = np.array(points_local[point_mapping[rows]], copy=True)
    if np.any(frame2 != frame1 + 1):
        raise ValueError("source exact local-point mappings must connect adjacent frames")
    return by_pair


def _validate_method_contract(
    odometry: dict[str, object],
    *,
    expected_variant: str,
) -> tuple[bool, str]:
    contract = METHOD_CONTRACTS.get(expected_variant)
    if contract is None:
        return False, f"unsupported retained method: {expected_variant}"
    try:
        odometry_args = odometry["args"]
        optimizer = odometry["optimizer"]
        optimizer_args = optimizer["args"]
    except (KeyError, TypeError):
        return False, "method configuration is missing odometry or optimizer arguments"
    if optimizer.get("type") != "TwoFrame_PGO":
        return False, "method optimizer type differs from TwoFrame_PGO"
    for name, expected in contract["odometry"].items():
        if odometry_args.get(name) != expected:
            return False, f"method odometry setting {name} differs from expected {expected!r}"
    for name, expected in contract["optimizer"].items():
        if optimizer_args.get(name) != expected:
            return False, f"method optimizer setting {name} differs from expected {expected!r}"
    for name, expected in contract.get("odometry_defaults", {}).items():
        if name in odometry_args and odometry_args[name] != expected:
            return False, f"method odometry setting {name} differs from default {expected!r}"
    for name, expected in contract.get("optimizer_defaults", {}).items():
        if name in optimizer_args and optimizer_args[name] != expected:
            return False, f"method optimizer setting {name} differs from default {expected!r}"
    return True, "method configuration matches expected variant"


def _runtime_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"runtime field {field} is not boolean: {value!r}")


def _validate_runtime_method_contract(
    rows: list[dict[str, str]],
    *,
    expected_variant: str,
) -> tuple[bool, str]:
    contract = METHOD_CONTRACTS.get(expected_variant)
    if contract is None:
        return False, f"unsupported retained runtime method: {expected_variant}"
    if not rows:
        return False, "runtime method diagnostics are empty"
    for row in rows:
        pair = f"{row.get('frame_i', '?')}->{row.get('frame_j', '?')}"
        for field, expected in contract["runtime"].items():
            if field not in row:
                return False, f"runtime method field {field} is missing for pair {pair}"
            raw_value = str(row[field]).strip()
            if expected is None:
                if raw_value:
                    return False, (
                        f"runtime method field {field}={raw_value!r} must be empty for "
                        f"static retained method {expected_variant} at pair {pair}"
                    )
                continue
            if not raw_value:
                return False, f"runtime method field {field} is missing for pair {pair}"
            try:
                actual = (
                    _runtime_bool(row[field], field=field)
                    if isinstance(expected, bool)
                    else str(row[field]).strip()
                )
            except ValueError as error:
                return False, str(error)
            if actual != expected:
                return False, (
                    f"runtime method field {field}={actual!r} differs from "
                    f"expected {expected!r} for {expected_variant} at pair {pair}"
                )
    return True, "runtime diagnostics match expected retained method"


def validate_source_artifact(
    result_dir: Path,
    *,
    expected_frame_count: int,
    expected_scene: str | None = None,
    expected_dataset_root: Path | None = None,
    expected_variant: str | None = None,
) -> tuple[bool, str]:
    required = (
        "metadata.yaml",
        "config.yaml",
        "tensor_map.npz",
        "poses.csv",
        "pose_coordinate_frame.txt",
        "visual_factor_diagnostics.csv",
        "frame_pair_diagnostics.csv",
    )
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        return False, f"missing source artifacts: {', '.join(missing)}"
    try:
        artifact_revision = _artifact_git_revision(result_dir)
        expected_revision = current_git_revision()
        if artifact_revision != expected_revision:
            return False, (
                f"source git revision {artifact_revision!r} differs from current "
                f"revision {expected_revision!r}"
            )
        config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))
        odometry = config["Odometry"]
        if expected_variant is not None:
            valid_method, method_reason = _validate_method_contract(
                odometry,
                expected_variant=expected_variant,
            )
            if not valid_method:
                return False, method_reason
        if (expected_scene is None) != (expected_dataset_root is None):
            return False, "source dataset binding requires both scene and dataset root"
        if expected_scene is not None and expected_dataset_root is not None:
            data_args = config["Data"]["args"]["args"]
            if str(data_args.get("scene", "")) != expected_scene:
                return False, "source configured scene differs from scheduled scene"
            configured_root = Path(str(data_args.get("root", ""))).expanduser().resolve()
            if configured_root != expected_dataset_root.expanduser().resolve():
                return False, "source configured dataset differs from scheduled dataset"
        if odometry["args"].get("mapping") is not False:
            return False, "source config must use mapping=false"
        if odometry["motion"].get("type") != "StaticMotionModel":
            return False, "source config must use StaticMotionModel"
        if odometry["keyframe"].get("type") != "AllKeyframe":
            return False, "source config must use AllKeyframe"
        if odometry["frontend"].get("type") != SOURCE_FRONTEND_TYPE:
            return False, f"source frontend must use {SOURCE_FRONTEND_TYPE}"

        with np.load(result_dir / "tensor_map.npz", allow_pickle=False) as tensor_map:
            timestamps = np.asarray(tensor_map["frames//time_ns"])
            visual_hashes = _source_visual_hashes(
                tensor_map,
                frame_count=expected_frame_count,
            )
            _source_exact_local_points(
                tensor_map,
                frame_count=expected_frame_count,
            )
        if timestamps.shape != (expected_frame_count,):
            return False, "source tensor-map frame count differs from expected frame count"
        if not np.issubdtype(timestamps.dtype, np.integer) or np.any(np.diff(timestamps) <= 0):
            return False, "source tensor-map timestamps are not strictly increasing integers"

        pose_timestamps, _, _ = _pose_table(result_dir / "poses.csv")
        if len(pose_timestamps) != expected_frame_count:
            return False, "source pose count differs from expected frame count"
        if tuple(int(value) for value in pose_timestamps) != tuple(int(value) for value in timestamps):
            return False, "source pose timestamps differ from tensor-map timestamps"

        diagnostics = _read_csv_rows(result_dir / "visual_factor_diagnostics.csv")
        if len(diagnostics) != expected_frame_count - 1:
            return False, "source visual diagnostic pair count is incomplete"
        for frame_i, row in enumerate(diagnostics):
            expected_pair = (frame_i, frame_i + 1)
            actual_pair = (int(row["frame_i"]), int(row["frame_j"]))
            if actual_pair != expected_pair:
                return False, "source visual diagnostics are not contiguous adjacent pairs"
            timestamp_i = int(row.get("timestamp_i_ns", row.get("timestamp_i", "")))
            timestamp_j = int(row.get("timestamp_j_ns", row.get("timestamp_j", "")))
            if (
                timestamp_i != int(timestamps[frame_i])
                or timestamp_j != int(timestamps[frame_i + 1])
            ):
                return False, "source visual diagnostic timestamps differ from tensor map"
            recorded_hash = str(row.get("visual_input_sha256", "")).strip()
            if recorded_hash != visual_hashes[expected_pair]:
                return False, "source visual diagnostic hash differs from tensor-map factors"

        runtime_diagnostics = _read_csv_rows(result_dir / "frame_pair_diagnostics.csv")
        if len(runtime_diagnostics) != expected_frame_count - 1:
            return False, "source runtime diagnostic pair count is incomplete"
        for frame_i, row in enumerate(runtime_diagnostics):
            if (int(row["frame_i"]), int(row["frame_j"])) != (frame_i, frame_i + 1):
                return False, "source runtime diagnostics are not contiguous adjacent pairs"
            timestamp_i = int(row.get("timestamp_i_ns", row.get("timestamp_i", "")))
            timestamp_j = int(row.get("timestamp_j_ns", row.get("timestamp_j", "")))
            if timestamp_i != int(timestamps[frame_i]) or timestamp_j != int(timestamps[frame_i + 1]):
                return False, "source runtime diagnostic timestamps differ from tensor map"
        runtime_valid, runtime_reason = _validate_runtime_method_contract(
            runtime_diagnostics,
            expected_variant=expected_variant or PURE_MACVO_METHOD,
        )
        if not runtime_valid:
            return False, runtime_reason
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        return False, f"invalid source artifact: {error}"
    return True, "source artifact is complete"


def validate_cache_artifact(
    cache_dir: Path,
    *,
    scene: str,
    expected_frame_count: int,
    source_result_dir: Path | None = None,
    dataset_root: Path | None = None,
) -> tuple[bool, str]:
    try:
        reader = VisualFactorCacheReader(cache_dir)
        manifest = reader.manifest
        if manifest.scene != scene:
            return False, "cache scene differs from scheduled scene"
        if manifest.frame_count != expected_frame_count:
            return False, "cache frame count differs from expected frame count"
        if len(manifest.pairs) != expected_frame_count - 1:
            return False, "cache pair count is incomplete"
        if len(manifest.timestamps_ns) != expected_frame_count:
            return False, "cache timestamp count is incomplete"
        if (source_result_dir is None) != (dataset_root is None):
            return False, "cache source binding requires both source result and dataset root"
        if source_result_dir is not None and dataset_root is not None:
            source_valid, source_reason = validate_source_artifact(
                source_result_dir,
                expected_frame_count=expected_frame_count,
                expected_scene=scene,
                expected_dataset_root=dataset_root,
                expected_variant=PURE_MACVO_METHOD,
            )
            if not source_valid:
                return False, f"cache source artifact is invalid: {source_reason}"
            source = manifest.source
            required_provenance = {
                "dataset",
                "result",
                "config",
                "checksums",
                "git",
                "motion",
                "keyframes",
            }
            missing_provenance = sorted(required_provenance - set(source))
            if missing_provenance:
                return False, f"cache source provenance is missing: {', '.join(missing_provenance)}"
            if Path(str(source["dataset"])).expanduser().resolve() != dataset_root.expanduser().resolve():
                return False, "cache dataset provenance differs from scheduled dataset"
            if Path(str(source["result"])).expanduser().resolve() != source_result_dir.expanduser().resolve():
                return False, "cache source result provenance differs from scheduled source"
            config_path = source_result_dir / "config.yaml"
            tensor_map_path = source_result_dir / "tensor_map.npz"
            if hashlib.sha256(config_path.read_bytes()).hexdigest() != str(source["config"]):
                return False, "cache source config checksum differs from current source"
            if hashlib.sha256(tensor_map_path.read_bytes()).hexdigest() != str(source["checksums"]):
                return False, "cache source tensor-map checksum differs from current source"
            if str(source["git"]).strip() != _artifact_git_revision(source_result_dir):
                return False, "cache source git provenance differs from source artifact"
            if source["motion"] != "StaticMotionModel" or source["keyframes"] != "AllKeyframe":
                return False, "cache source motion/keyframe provenance is incompatible"
        source_hashes: dict[tuple[int, int], str] = {}
        source_local_points: dict[tuple[int, int], np.ndarray] = {}
        if source_result_dir is not None:
            for row in _read_csv_rows(source_result_dir / "visual_factor_diagnostics.csv"):
                source_hashes[(int(row["frame_i"]), int(row["frame_j"]))] = str(
                    row["visual_input_sha256"]
                ).strip()
            with np.load(source_result_dir / "tensor_map.npz", allow_pickle=False) as tensor_map:
                source_local_points = _source_exact_local_points(
                    tensor_map,
                    frame_count=expected_frame_count,
                )
        for frame_i in range(expected_frame_count - 1):
            packet = reader.load_pair(
                frame_i,
                frame_i + 1,
                manifest.timestamps_ns[frame_i],
                manifest.timestamps_ns[frame_i + 1],
            )
            computed_hash = visual_input_sha256(packet.match_fields)
            if packet.visual_sha256 != computed_hash:
                return False, "cache packet visual hash differs from packet factors"
            if source_hashes and source_hashes.get((frame_i, frame_i + 1)) != packet.visual_sha256:
                return False, "cache packet hash differs from source visual diagnostics"
            if source_local_points:
                expected_points = torch.from_numpy(source_local_points[(frame_i, frame_i + 1)])
                if not torch.equal(packet.points_local.detach().cpu(), expected_points):
                    return False, "cache packet local points differ from source exact local points"
    except (OSError, ValueError, VisualFactorCacheError) as error:
        return False, f"invalid cache artifact: {error}"
    return True, "cache artifact is complete and checksummed"


def validate_replay_artifact(
    result_dir: Path,
    *,
    cache_dir: Path,
    scene: str,
    expected_frame_count: int,
    expected_variant: str,
    source_result_dir: Path | None = None,
    dataset_root: Path | None = None,
) -> tuple[bool, str]:
    valid_cache, cache_reason = validate_cache_artifact(
        cache_dir,
        scene=scene,
        expected_frame_count=expected_frame_count,
        source_result_dir=source_result_dir,
        dataset_root=dataset_root,
    )
    if not valid_cache:
        return False, cache_reason
    required = (
        "metadata.yaml",
        "config.yaml",
        "poses.csv",
        "pose_coordinate_frame.txt",
        "frame_pair_diagnostics.csv",
    )
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        return False, f"missing replay artifacts: {', '.join(missing)}"
    try:
        artifact_revision = _artifact_git_revision(result_dir)
        expected_revision = current_git_revision()
        if artifact_revision != expected_revision:
            return False, (
                f"replay git revision {artifact_revision!r} differs from current "
                f"revision {expected_revision!r}"
            )
        reader = VisualFactorCacheReader(cache_dir)
        config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))
        odometry = config["Odometry"]
        valid_method, method_reason = _validate_method_contract(
            odometry,
            expected_variant=expected_variant,
        )
        if not valid_method:
            return False, method_reason
        if dataset_root is not None:
            data_args = config["Data"]["args"]["args"]
            if str(data_args.get("scene", "")) != scene:
                return False, "replay configured scene differs from scheduled scene"
            configured_root = Path(str(data_args.get("root", ""))).expanduser().resolve()
            if configured_root != dataset_root.expanduser().resolve():
                return False, "replay configured dataset differs from scheduled dataset"
        odometry_args = odometry["args"]
        if odometry["frontend"].get("type") != REPLAY_FRONTEND_TYPE:
            return False, f"replay frontend must use {REPLAY_FRONTEND_TYPE}"
        if str(odometry_args.get("visual_cache_mode", "")).strip().lower() != "replay":
            return False, "replay config is missing visual_cache_mode=replay"
        configured_cache = Path(str(odometry_args.get("visual_cache_path", ""))).expanduser().resolve()
        if configured_cache != cache_dir.expanduser().resolve():
            return False, "replay config visual cache path differs from scheduled cache"
        if odometry_args.get("mapping") is not False:
            return False, "replay config must use mapping=false"
        optimizer = odometry.get("optimizer", {})
        if optimizer.get("type") != "TwoFrame_PGO":
            return False, "replay config must use the TwoFrame_PGO optimizer"
        optimizer_args = optimizer.get("args", {})
        if expected_variant == PURE_MACVO_METHOD:
            if optimizer_args.get("imu_factor_mode") != "legacy_pose_prior":
                return False, "pure replay imu_factor_mode must be legacy_pose_prior"
            if odometry_args.get("imu_rot_prior_enable") is not False:
                return False, "pure replay must disable the IMU rotation prior"
            if odometry_args.get("imu_trans_prior_enable") is not False:
                return False, "pure replay must disable the IMU translation prior"
        elif expected_variant == LATEST_IMUATT_METHOD:
            if optimizer_args.get("imu_factor_mode") != "preintegrated_vio":
                return False, "imuatt replay imu_factor_mode must be preintegrated_vio"
            if odometry_args.get("imu_rot_prior_enable") is not True:
                return False, "imuatt replay must enable the IMU rotation prior"
            if odometry_args.get("imu_trans_prior_enable") is not True:
                return False, "imuatt replay must enable the IMU translation prior"
            if odometry_args.get("imu_vio_gravity_pose_source") != "imu_integrated_estinit":
                return False, "imuatt replay must use imu_integrated_estinit"
        else:
            return False, f"unsupported expected replay variant: {expected_variant}"

        pose_timestamps, _, _ = _pose_table(result_dir / "poses.csv")
        if len(pose_timestamps) != expected_frame_count:
            return False, "replay pose count differs from expected frame count"
        if tuple(int(value) for value in pose_timestamps) != reader.manifest.timestamps_ns:
            return False, "replay pose timestamps differ from cache manifest"

        diagnostics = _read_csv_rows(result_dir / "frame_pair_diagnostics.csv")
        if len(diagnostics) != expected_frame_count - 1:
            return False, "replay frame-pair diagnostic count is incomplete"
        for frame_i, row in enumerate(diagnostics):
            actual_pair = (int(row["frame_i"]), int(row["frame_j"]))
            if actual_pair != (frame_i, frame_i + 1):
                return False, "replay frame-pair diagnostics are not contiguous"
            timestamp_i = int(row.get("timestamp_i_ns", row.get("timestamp_i", "")))
            timestamp_j = int(row.get("timestamp_j_ns", row.get("timestamp_j", "")))
            packet = reader.load_pair(frame_i, frame_i + 1, timestamp_i, timestamp_j)
            if str(row.get("visual_input_sha256", "")).strip() != packet.visual_sha256:
                return False, "replay visual hash differs from cached packet"
        runtime_valid, runtime_reason = _validate_runtime_method_contract(
            diagnostics,
            expected_variant=expected_variant,
        )
        if not runtime_valid:
            return False, runtime_reason
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError, VisualFactorCacheError) as error:
        return False, f"invalid replay artifact: {error}"
    return True, "replay artifact is complete, frontend-free, and visually identical"


def _dataset_frame_count(scene_root: Path, seq_to: int | None) -> int:
    ref_pose = scene_root / "ref_pose.csv"
    try:
        total = len(_read_csv_rows(ref_pose))
    except OSError as error:
        raise FileNotFoundError(f"unable to read scene reference poses: {ref_pose}") from error
    if total < 2:
        raise ValueError(f"scene requires at least two frames: {scene_root}")
    if seq_to is None:
        return total
    requested = int(seq_to)
    if requested < 2:
        raise ValueError("--seq-to must select at least two frames")
    return min(total, requested)


def _pose_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv_rows(path)
    timestamps = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    positions = np.asarray(
        [[float(row[name]) for name in ("tx", "ty", "tz")] for row in rows],
        dtype=np.float64,
    )
    quaternions = np.asarray(
        [[float(row[name]) for name in ("qx", "qy", "qz", "qw")] for row in rows],
        dtype=np.float64,
    )
    if (
        len(rows) == 0
        or np.any(np.diff(timestamps) <= 0)
        or not np.isfinite(positions).all()
        or not np.isfinite(quaternions).all()
        or np.any(np.linalg.norm(quaternions, axis=1) <= 0.0)
    ):
        raise ValueError(f"invalid pose table: {path}")
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return timestamps, positions, quaternions


def _quaternion_angle_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(left * right, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verification_fingerprints(task: BatchTask) -> dict[str, str]:
    paths = {
        "source_config": task.source_result_dir / "config.yaml",
        "source_tensor_map": task.source_result_dir / "tensor_map.npz",
        "source_poses": task.source_result_dir / "poses.csv",
        "source_visual_diagnostics": task.source_result_dir / "visual_factor_diagnostics.csv",
        "source_runtime_diagnostics": task.source_result_dir / "frame_pair_diagnostics.csv",
        "cache_manifest": task.cache_dir / "manifest.json",
        "replay_pure_config": task.replay_pure_result_dir / "config.yaml",
        "replay_pure_poses": task.replay_pure_result_dir / "poses.csv",
        "replay_pure_diagnostics": task.replay_pure_result_dir / "frame_pair_diagnostics.csv",
        "replay_imuatt_config": task.replay_imuatt_result_dir / "config.yaml",
        "replay_imuatt_poses": task.replay_imuatt_result_dir / "poses.csv",
        "replay_imuatt_diagnostics": task.replay_imuatt_result_dir / "frame_pair_diagnostics.csv",
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def verify_scene_task(task: BatchTask, *, expected_frame_count: int) -> tuple[bool, str]:
    checks = [
        validate_source_artifact(
            task.source_result_dir,
            expected_frame_count=expected_frame_count,
            expected_scene=task.scene,
            expected_dataset_root=task.scene_root,
            expected_variant=PURE_MACVO_METHOD,
        ),
        validate_cache_artifact(
            task.cache_dir,
            scene=task.scene,
            expected_frame_count=expected_frame_count,
            source_result_dir=task.source_result_dir,
            dataset_root=task.scene_root,
        ),
        validate_replay_artifact(
            task.replay_pure_result_dir,
            cache_dir=task.cache_dir,
            scene=task.scene,
            expected_frame_count=expected_frame_count,
            expected_variant=PURE_MACVO_METHOD,
            source_result_dir=task.source_result_dir,
            dataset_root=task.scene_root,
        ),
        validate_replay_artifact(
            task.replay_imuatt_result_dir,
            cache_dir=task.cache_dir,
            scene=task.scene,
            expected_frame_count=expected_frame_count,
            expected_variant=LATEST_IMUATT_METHOD,
            source_result_dir=task.source_result_dir,
            dataset_root=task.scene_root,
        ),
    ]
    failed = [reason for valid, reason in checks if not valid]
    if failed:
        return False, "; ".join(failed)

    try:
        source_ts, source_pos, source_quat = _pose_table(task.source_result_dir / "poses.csv")
        replay_ts, replay_pos, replay_quat = _pose_table(task.replay_pure_result_dir / "poses.csv")
        if not np.array_equal(source_ts, replay_ts):
            return False, "source and pure replay pose timestamps differ"
        translation_error = np.linalg.norm(replay_pos - source_pos, axis=1)
        rotation_error_deg = _quaternion_angle_degrees(source_quat, replay_quat)
        metrics = {
            "scene": task.scene,
            "frame_count": int(expected_frame_count),
            "cache_pair_count": int(expected_frame_count - 1),
            "translation_rmse_m": float(np.sqrt(np.mean(translation_error**2))),
            "translation_max_m": float(np.max(translation_error)),
            "translation_final_m": float(translation_error[-1]),
            "rotation_rmse_deg": float(np.sqrt(np.mean(rotation_error_deg**2))),
            "rotation_max_deg": float(np.max(rotation_error_deg)),
            "rotation_final_deg": float(rotation_error_deg[-1]),
            "source_result_dir": str(task.source_result_dir),
            "replay_pure_result_dir": str(task.replay_pure_result_dir),
            "replay_imuatt_result_dir": str(task.replay_imuatt_result_dir),
            "cache_dir": str(task.cache_dir),
            "artifact_sha256": _verification_fingerprints(task),
        }
        passed = (
            metrics["translation_rmse_m"] <= PURE_REPLAY_TRANSLATION_RMSE_MAX_M
            and metrics["translation_max_m"] <= PURE_REPLAY_TRANSLATION_MAX_M
            and metrics["rotation_rmse_deg"] <= PURE_REPLAY_ROTATION_RMSE_MAX_DEG
            and metrics["rotation_max_deg"] <= PURE_REPLAY_ROTATION_MAX_DEG
        )
        metrics["passed"] = bool(passed)
        task.result_dir.mkdir(parents=True, exist_ok=True)
        report_path = task.result_dir / "verification.json"
        report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        with (task.result_dir / "verification.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
    except (OSError, KeyError, TypeError, ValueError) as error:
        return False, f"unable to compare source and pure replay trajectories: {error}"
    if not passed:
        return False, "pure replay trajectory differs beyond the strict numerical tolerance"
    return True, "source, cache, pure replay, and imuatt replay all passed verification"


def validate_verification_artifact(
    task: BatchTask,
    *,
    expected_frame_count: int,
) -> tuple[bool, str]:
    report = task.result_dir / "verification.json"
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"unable to read verification report: {error}"
    if payload.get("passed") is not True:
        return False, "verification report does not record a passing result"
    if payload.get("scene") != task.scene or payload.get("frame_count") != expected_frame_count:
        return False, "verification report scene or frame count is stale"
    prerequisite_checks = [
        validate_source_artifact(
            task.source_result_dir,
            expected_frame_count=expected_frame_count,
            expected_scene=task.scene,
            expected_dataset_root=task.scene_root,
            expected_variant=PURE_MACVO_METHOD,
        ),
        validate_cache_artifact(
            task.cache_dir,
            scene=task.scene,
            expected_frame_count=expected_frame_count,
            source_result_dir=task.source_result_dir,
            dataset_root=task.scene_root,
        ),
        validate_replay_artifact(
            task.replay_pure_result_dir,
            cache_dir=task.cache_dir,
            scene=task.scene,
            expected_frame_count=expected_frame_count,
            expected_variant=PURE_MACVO_METHOD,
            source_result_dir=task.source_result_dir,
            dataset_root=task.scene_root,
        ),
        validate_replay_artifact(
            task.replay_imuatt_result_dir,
            cache_dir=task.cache_dir,
            scene=task.scene,
            expected_frame_count=expected_frame_count,
            expected_variant=LATEST_IMUATT_METHOD,
            source_result_dir=task.source_result_dir,
            dataset_root=task.scene_root,
        ),
    ]
    failed = [reason for valid, reason in prerequisite_checks if not valid]
    if failed:
        return False, f"verification report prerequisites are stale: {'; '.join(failed)}"
    try:
        current_fingerprints = _verification_fingerprints(task)
    except OSError as error:
        return False, f"unable to fingerprint verification artifacts: {error}"
    if payload.get("artifact_sha256") != current_fingerprints:
        return False, "verification report artifact fingerprints are stale"
    return True, "verification report records a passing result"


def validate_task_artifact(
    task: BatchTask,
    *,
    expected_frame_count: int,
) -> tuple[bool, str]:
    if task.phase == "source":
        return validate_source_artifact(
            task.result_dir,
            expected_frame_count=expected_frame_count,
            expected_scene=task.scene,
            expected_dataset_root=task.scene_root,
            expected_variant=task.variant,
        )
    if task.phase == "export":
        return validate_cache_artifact(
            task.cache_dir,
            scene=task.scene,
            expected_frame_count=expected_frame_count,
            source_result_dir=task.source_result_dir,
            dataset_root=task.scene_root,
        )
    if task.phase in {"replay-pure", "replay-imuatt"}:
        return validate_replay_artifact(
            task.result_dir,
            cache_dir=task.cache_dir,
            scene=task.scene,
            expected_frame_count=expected_frame_count,
            expected_variant=task.variant,
            source_result_dir=task.source_result_dir,
            dataset_root=task.scene_root,
        )
    if task.phase == "verify":
        return validate_verification_artifact(task, expected_frame_count=expected_frame_count)
    return False, f"unsupported task phase: {task.phase}"


PROGRESS_FIELDS = [
    "trial",
    "scene",
    "variant",
    "phase",
    "method_variant",
    "status",
    "return_code",
    "runtime_s",
    "result_dir",
    "active_result_dir",
    "cache_dir",
    "detail",
]


def append_progress(control_root: Path, task: BatchTask, **values: object) -> None:
    control_root.mkdir(parents=True, exist_ok=True)
    progress_path = control_root / "progress.csv"
    exists = progress_path.exists()
    row = {
        "trial": 1,
        "scene": task.scene,
        "variant": task.manifest_variant,
        "phase": task.phase,
        "method_variant": task.variant,
        "result_dir": str(task.result_dir),
        "cache_dir": str(task.cache_dir),
        **values,
    }
    with progress_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in PROGRESS_FIELDS})


def existing_dashboard_pids(*, port: int) -> list[int]:
    process = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        cwd=str(WORKDIR),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    pids: list[int] = []
    for line in process.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or "Scripts/run_progress_dashboard.py" not in parts[1]:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            command = shlex.split(parts[1])
            port_index = command.index("--port")
            process_port = int(command[port_index + 1])
        except ValueError:
            process_port = 8765
        except (IndexError, TypeError):
            continue
        if pid != os.getpid() and process_port == int(port):
            pids.append(pid)
    return pids


def dashboard_health_matches(control_root: Path, *, port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{int(port)}/api/status",
            timeout=0.5,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
        return False
    try:
        reported_root = Path(str(payload["result_root"])).expanduser().resolve()
    except (KeyError, TypeError, ValueError):
        return False
    return reported_root == control_root.expanduser().resolve()


def switch_dashboard(
    control_root: Path,
    log_path: Path,
    *,
    port: int,
    launch_script: Path | None = None,
    launch_log: Path | None = None,
    host: str | None = None,
) -> None:
    for pid in existing_dashboard_pids(port=port):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    DEFAULT_DASHBOARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(WORKDIR / "Scripts" / "run_progress_dashboard.py"),
        "--result-root",
        str(control_root),
        "--log",
        str(log_path),
        "--port",
        str(int(port)),
    ]
    if host is not None:
        command.extend(("--host", host))
    if launch_script is not None:
        command.extend(("--launch-script", str(launch_script)))
    if launch_log is not None:
        command.extend(("--launch-log", str(launch_log)))
    with DEFAULT_DASHBOARD_LOG.open("ab") as stream:
        process = subprocess.Popen(
            command,
            cwd=str(WORKDIR),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if dashboard_health_matches(control_root, port=port):
            return
        time.sleep(0.25)
    if process.poll() is None:
        process.terminate()
    raise RuntimeError(
        f"progress dashboard failed health check on {dashboard_url(port)} for {control_root}"
    )


def _task_configs(task: BatchTask, temporary_root: Path) -> tuple[Path, Path]:
    if task.variant not in RETAINED_METHODS:
        raise ValueError(f"batch task uses unsupported retained method: {task.variant}")
    variant = RETAINED_VARIANTS[task.variant]
    task_tmp = temporary_root / task.phase / task.scene
    task_tmp.mkdir(parents=True, exist_ok=True)
    spec = RunSpec(
        trial=1,
        scene=task.scene,
        scene_root=task.scene_root,
        variant=variant,
        result_dir=task.result_dir,
    )
    return make_odom_cfg(variant, task_tmp), make_seq_cfg(spec, task_tmp)


def _log_line(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.now().isoformat(timespec='seconds')}] {text}\n")


def promote_attempt_directory(
    staging_dir: Path,
    final_dir: Path,
    *,
    archive_root: Path,
) -> Path | None:
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"validated staging directory is missing: {staging_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    archived: Path | None = None
    if final_dir.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        archived = archive_root / f"{final_dir.name}_{time.time_ns()}"
        os.replace(final_dir, archived)
    try:
        os.replace(staging_dir, final_dir)
    except Exception:
        if archived is not None and not final_dir.exists():
            os.replace(archived, final_dir)
        raise
    return archived


def rollback_promoted_directory(
    final_dir: Path,
    archived_dir: Path | None,
    *,
    rejected_root: Path,
) -> Path:
    rejected_root.mkdir(parents=True, exist_ok=True)
    rejected = rejected_root / f"{final_dir.name}_{time.time_ns()}"
    if not final_dir.exists():
        raise FileNotFoundError(f"promoted artifact is missing during rollback: {final_dir}")
    os.replace(final_dir, rejected)
    if archived_dir is not None and archived_dir.exists():
        os.replace(archived_dir, final_dir)
    return rejected


def promote_attempt_directory_guarded(
    staging_dir: Path,
    final_dir: Path,
    *,
    archive_root: Path,
    rejected_root: Path,
    expected_git_revision: str,
) -> Path | None:
    validate_formal_code_state(expected_revision=expected_git_revision)
    archived = promote_attempt_directory(
        staging_dir,
        final_dir,
        archive_root=archive_root,
    )
    try:
        validate_formal_code_state(expected_revision=expected_git_revision)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        rejected = rollback_promoted_directory(
            final_dir,
            archived,
            rejected_root=rejected_root,
        )
        raise ValueError(f"{error}; promoted artifact rolled back to {rejected}") from error
    return archived


def execute_task(
    task: BatchTask,
    args: argparse.Namespace,
    temporary_root: Path,
    *,
    expected_git_revision: str | None = None,
) -> int:
    expected_frame_count = _dataset_frame_count(task.scene_root, args.seq_to)
    if not args.force:
        complete, reason = validate_task_artifact(task, expected_frame_count=expected_frame_count)
        if complete:
            append_progress(
                args.control_root,
                task,
                status="ok",
                return_code=0,
                runtime_s="0.0",
                detail=f"resume: {reason}",
            )
            print(f"SKIP valid {task.phase:13s} {task.scene}: {reason}")
            return 0

    attempt_root = (
        args.control_root
        / "staging"
        / task.phase
        / task.scene
        / str(time.time_ns())
    )
    attempt_artifact = attempt_root / "artifact"
    if task.phase == "export":
        attempt_task = replace(task, result_dir=attempt_artifact, cache_dir=attempt_artifact)
    else:
        attempt_task = replace(task, result_dir=attempt_artifact)
    append_progress(
        args.control_root,
        task,
        status="running",
        active_result_dir=str(attempt_artifact),
        detail=f"task started in staging: {attempt_artifact}",
    )
    started = time.monotonic()
    if task.phase == "verify":
        complete, reason = verify_scene_task(attempt_task, expected_frame_count=expected_frame_count)
        return_code = 0 if complete else 2
    else:
        odom_cfg = seq_cfg = None
        if task.phase in {"source", "replay-pure", "replay-imuatt"}:
            odom_cfg, seq_cfg = _task_configs(attempt_task, temporary_root)
        command = build_task_command(
            attempt_task,
            odom_cfg=odom_cfg,
            seq_cfg=seq_cfg,
            seq_to=args.seq_to,
        )
        task_log = args.control_root / "logs" / task.phase / f"{task.scene}.log"
        task_log.parent.mkdir(parents=True, exist_ok=True)
        _log_line(args.log_path, f"START {task.phase} {task.scene}: {subprocess.list2cmdline(command)}")
        try:
            with task_log.open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    command,
                    cwd=str(WORKDIR),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=int(args.timeout),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            return_code = int(process.returncode)
            if task.phase in {"source", "replay-pure", "replay-imuatt"}:
                flatten_nested(attempt_task.result_dir)
        except subprocess.TimeoutExpired:
            return_code = 124
        if return_code == 0:
            complete, reason = validate_task_artifact(
                attempt_task,
                expected_frame_count=expected_frame_count,
            )
            if not complete:
                return_code = 2
        else:
            reason = f"process exited with return code {return_code}"
        _log_line(args.log_path, f"END {task.phase} {task.scene}: rc={return_code}; {reason}")

    if return_code == 0 and expected_git_revision is not None:
        try:
            validate_formal_code_state(expected_revision=expected_git_revision)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            return_code = 2
            reason = f"formal code state changed while task was running: {error}"

    if return_code == 0:
        final_artifact = task.cache_dir if task.phase == "export" else task.result_dir
        try:
            archive_root = args.control_root / "archive" / task.phase / task.scene
            rejected_root = args.control_root / "rejected" / task.phase / task.scene
            if expected_git_revision is None:
                archived = promote_attempt_directory(
                    attempt_artifact,
                    final_artifact,
                    archive_root=archive_root,
                )
            else:
                archived = promote_attempt_directory_guarded(
                    attempt_artifact,
                    final_artifact,
                    archive_root=archive_root,
                    rejected_root=rejected_root,
                    expected_git_revision=expected_git_revision,
                )
            complete, reason = validate_task_artifact(
                task,
                expected_frame_count=expected_frame_count,
            )
            if not complete:
                return_code = 2
                rejected = rollback_promoted_directory(
                    final_artifact,
                    archived,
                    rejected_root=rejected_root,
                )
                reason = (
                    f"promoted artifact failed validation and was rolled back to {rejected}: "
                    f"{reason}"
                )
            elif archived is not None:
                reason = f"{reason}; previous artifact archived at {archived}"
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            return_code = 2
            reason = f"unable to promote validated staging artifact: {error}"

    runtime_s = time.monotonic() - started
    status = "ok" if return_code == 0 else ("timeout" if return_code == 124 else "failed")
    append_progress(
        args.control_root,
        task,
        status=status,
        return_code=return_code,
        runtime_s=f"{runtime_s:.3f}",
        detail=reason,
    )
    print(f"{status.upper():7s} {task.phase:13s} {task.scene} ({runtime_s:.1f}s): {reason}")
    return return_code


def run_batch(args: argparse.Namespace) -> int:
    resolve_runtime_paths(args)
    validate_runtime_layout(args)
    formal_git_revision = None if args.dry_run else validate_formal_code_state()
    if int(args.jobs) != 1:
        raise ValueError("this GPU-backed batch currently requires --jobs 1")
    tasks = build_schedule(
        source_result_root=args.source_result_root,
        cache_root=args.cache_root,
        replay_result_root=args.replay_result_root,
        control_root=args.control_root,
        analysis_root=args.analysis_root,
    )
    prepare_control_state(args.control_root, tasks, seq_to=args.seq_to)
    phases = set(selected_phases(args.phase))
    selected = [task for task in tasks if task.phase in phases]
    print(f"Visual-factor cache batch: {len(tasks)} scheduled tasks; {len(selected)} selected")
    print(f"Progress dashboard: {dashboard_url(args.dashboard_port)}")
    print(f"Control root: {args.control_root}")
    for task in selected:
        method_label = task.variant or "-"
        print(
            f"  {task.phase:13s} {task.scene:32s} "
            f"method={method_label} result={task.result_dir} cache={task.cache_dir}"
        )
    if args.dry_run:
        return 0

    for task in selected:
        _dataset_frame_count(task.scene_root, args.seq_to)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_dashboard:
        switch_dashboard(args.control_root, args.log_path, port=int(args.dashboard_port))

    with tempfile.TemporaryDirectory(prefix="visual_factor_cache_batch_") as temporary:
        temporary_root = Path(temporary)
        for task in selected:
            if formal_git_revision is not None:
                validate_formal_code_state(expected_revision=formal_git_revision)
            return_code = execute_task(
                task,
                args,
                temporary_root,
                expected_git_revision=formal_git_revision,
            )
            if return_code != 0:
                print(f"Batch stopped after failure in {task.phase}/{task.scene}.")
                return return_code
    print("Selected visual-factor cache phases completed and passed artifact validation.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASE_SELECTIONS), default="all")
    parser.add_argument("--source-result-root", type=Path, default=SOURCE_RESULT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--replay-result-root", type=Path, default=REPLAY_RESULT_ROOT)
    parser.add_argument("--control-root", type=Path, default=CONTROL_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_batch(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

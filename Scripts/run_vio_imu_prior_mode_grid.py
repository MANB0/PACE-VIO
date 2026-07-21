#!/usr/bin/env python3
"""
Run a small VIO/IMU translation-prior diagnostic grid.

This is not a paper-result batch. It is a code-side diagnostic batch for
checking whether AIM-VO's weak translation prior semantics or weight explains
the poor runs.

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/run_vio_imu_prior_mode_grid.py --dry-run
    python Scripts/run_vio_imu_prior_mode_grid.py --seq-to 600
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

import yaml

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader
from Utility.RunOutputBundle import find_output_bundle
from Utility.VIOConventionDiagnostics import is_active_full_vio_row

RESULT_ROOT = WORKDIR / "Results" / "vio_imu_prior_mode_grid"
BASE_ODOM_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"
RUN_TIMEOUT_S = 7200
PROGRESS_LOCK = threading.Lock()
INCOMPLETE_OUTPUT_RC = 2

SCENE_METADATA_EXPECTED_EXACT = {
    "dataset.timestamp_unit": "ns",
    "imu.frame": "FLU",
    "imu.acc_unit": "m/s^2",
    "imu.gyro_unit": "rad/s",
    "coordinate_convention.imu_measurement_frame": "FLU",
    "time_synchronization.timestamp_unit": "ns",
    "time_synchronization.camera_imu_time_offset_ns": "0",
}
SCENE_METADATA_EXPECTED_CONTAINS = {
    "dataset.simulator": "HoloOcean",
    "coordinate_convention.holocean_world_frame": "NWU",
    "coordinate_convention.export_world_frame": "NWU",
    "coordinate_convention.ref_pose_position_frame": "world NWU",
    "coordinate_convention.ref_pose_velocity_frame": "world NWU",
    "coordinate_convention.ref_pose_angular_velocity_frame": "body NWU",
    "ground_truth.velocity_frame": "world NWU",
    "ground_truth.angular_velocity_frame": "body NWU",
}

SCENE_ROOTS = {
    "turbid_harbor": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/turbid_harbor"),
    "clear_shallow": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/clear_shallow"),
    "deep_dark": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/deep_dark"),
    "caustic_shallow": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/caustic_shallow"),
    "dam_inspection": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/dam_inspection"),
    "murky_coast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/murky_coast"),
    "open_water_overcast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/open_water_overcast"),
    "validation_transient_dropout": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_transient_dropout"),
    "validation_twilight_structure": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_twilight_structure"),
    "locked_clear_imu_harm": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_clear_imu_harm"),
    "locked_murky_entry_help": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_murky_entry_help"),
    "locked_quality_degrade_no_dropout": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_quality_degrade_no_dropout"),
}

SMOKE3_SCENES = ["clear_shallow", "murky_coast", "locked_murky_entry_help"]
KEPT12_SCENES = [
    "turbid_harbor",
    "clear_shallow",
    "deep_dark",
    "caustic_shallow",
    "dam_inspection",
    "murky_coast",
    "open_water_overcast",
    "validation_transient_dropout",
    "validation_twilight_structure",
    "locked_clear_imu_harm",
    "locked_murky_entry_help",
    "locked_quality_degrade_no_dropout",
]
SCENE_PRESETS = {
    "smoke3": SMOKE3_SCENES,
    "kept12": KEPT12_SCENES,
}
DEFAULT_SCENES = SMOKE3_SCENES
DEFAULT_VARIANTS = [
    "pure_macvo",
    "rotation_only",
    "aimvo_damping_s005",
    "aimvo_damping_s1",
    "aimvo_visualvel_s1",
    "aimvo_imuvel_s1",
]
MAIN3_VARIANTS = [
    "pure_macvo",
    "rotation_only",
    "aimvo_damping_s005",
]
VARIANT_PRESETS = {
    "default": DEFAULT_VARIANTS,
    "main3": MAIN3_VARIANTS,
}


class Variant(NamedTuple):
    name: str
    mode: str
    scale: float
    rot_enabled: bool
    trans_enabled: bool
    force_mode: str
    rot_prior_std: float | None = None
    rot_prior_std_when_translation: float | None = None
    imu_factor_mode: str = "legacy_pose_prior"
    gravity_pose_source: str | None = None
    velocity_feedback_enable: bool | None = None
    vio_cov_scale: float | None = None
    gravity_rp_correction_gain: float | None = None
    gravity_rp_acc_tol: float | None = None
    gravity_rp_window: int | None = None
    vio_alpha_p: float | None = None
    vio_alpha_v: float | None = None
    vio_alpha_R: float | None = None
    local_ba_window_size: int | None = None
    local_ba_writeback: str | None = None
    local_ba_fix_first_frame: bool | None = None
    gravity_handling: str | None = None


VARIANTS = {
    "pure_macvo": Variant("pure_macvo", "off", 1.0, False, False, "pure_macvo"),
    "rotation_only": Variant("rotation_only", "off", 1.0, True, False, "rotation_only"),
    "rotation_only_rotstd02": Variant("rotation_only_rotstd02", "off", 1.0, True, False, "rotation_only", 0.02),
    "rotation_only_rotstd03": Variant("rotation_only_rotstd03", "off", 1.0, True, False, "rotation_only", 0.03),
    "aimvo_damping_s005": Variant("aimvo_damping_s005", "damping_delta_p", 0.05, True, True, ""),
    "aimvo_damping_s005_rotstd02": Variant("aimvo_damping_s005_rotstd02", "damping_delta_p", 0.05, True, True, "", 0.02),
    "aimvo_damping_s005_rotstd03": Variant("aimvo_damping_s005_rotstd03", "damping_delta_p", 0.05, True, True, "", 0.03),
    "aimvo_damping_s005_fullrotstd02": Variant("aimvo_damping_s005_fullrotstd02", "damping_delta_p", 0.05, True, True, "", None, 0.02),
    "aimvo_damping_s005_fullrotstd03": Variant("aimvo_damping_s005_fullrotstd03", "damping_delta_p", 0.05, True, True, "", None, 0.03),
    "aimvo_damping_s1": Variant("aimvo_damping_s1", "damping_delta_p", 1.0, True, True, ""),
    "aimvo_visualvel_s1": Variant("aimvo_visualvel_s1", "visual_velocity_composed", 1.0, True, True, ""),
    "aimvo_imuvel_s1": Variant("aimvo_imuvel_s1", "imu_velocity_composed", 1.0, True, True, ""),
    "vio_preintegrated": Variant(
        "vio_preintegrated",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "",
        None,
        None,
        "preintegrated_vio",
    ),
    "vio_preintegrated_full": Variant(
        "vio_preintegrated_full",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
    ),
    "vio_preintegrated_full_gtgravity": Variant(
        "vio_preintegrated_full_gtgravity",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "gt_ref",
    ),
    "vio_preintegrated_full_no_velfb": Variant(
        "vio_preintegrated_full_no_velfb",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        None,
        False,
    ),
    "vio_preintegrated_full_cov1000": Variant(
        "vio_preintegrated_full_cov1000",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        None,
        None,
        1000.0,
    ),
    "vio_preintegrated_full_gravityrp_weak": Variant(
        "vio_preintegrated_full_gravityrp_weak",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "imu_gravity_rp",
        True,
        25.0,
        1.0,
        0.15,
        24,
    ),
    "vio_preintegrated_full_imuatt_estinit": Variant(
        "vio_preintegrated_full_imuatt_estinit",
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
    "vio_preintegrated_full_residual_gravity": Variant(
        name="vio_preintegrated_full_residual_gravity",
        mode="imu_velocity_composed",
        scale=1.0,
        rot_enabled=True,
        trans_enabled=True,
        force_mode="full_imu",
        imu_factor_mode="preintegrated_vio",
        gravity_pose_source="estimated",
        gravity_handling="residual",
    ),
    "vio_preintegrated_full_imuatt_Ronly": Variant(
        "vio_preintegrated_full_imuatt_Ronly",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "imu_integrated_estinit",
        None,
        None,
        None,
        None,
        None,
        0.0,
        0.0,
        1.0,
    ),
    "vio_preintegrated_full_imuatt_pvonly": Variant(
        "vio_preintegrated_full_imuatt_pvonly",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "imu_integrated_estinit",
        None,
        None,
        None,
        None,
        None,
        1.0,
        1.0,
        0.0,
    ),
    "vio_preintegrated_full_imuatt_pv01": Variant(
        "vio_preintegrated_full_imuatt_pv01",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "imu_integrated_estinit",
        None,
        None,
        None,
        None,
        None,
        0.1,
        0.1,
        1.0,
    ),
    "vio_preintegrated_full_imuatt_pv001": Variant(
        "vio_preintegrated_full_imuatt_pv001",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "imu_integrated_estinit",
        None,
        None,
        None,
        None,
        None,
        0.01,
        0.01,
        1.0,
    ),
    "vio_preintegrated_full_imuatt_R01": Variant(
        "vio_preintegrated_full_imuatt_R01",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "imu_integrated_estinit",
        None,
        None,
        None,
        None,
        None,
        1.0,
        1.0,
        0.1,
    ),
    "vio_preintegrated_full_imuatt_all01": Variant(
        "vio_preintegrated_full_imuatt_all01",
        "imu_velocity_composed",
        1.0,
        True,
        True,
        "full_imu",
        None,
        None,
        "preintegrated_vio",
        "imu_integrated_estinit",
        None,
        None,
        None,
        None,
        None,
        0.1,
        0.1,
        0.1,
    ),
    "vio_local_ba_w2_imuatt": Variant(
        name="vio_local_ba_w2_imuatt",
        mode="imu_velocity_composed",
        scale=1.0,
        rot_enabled=True,
        trans_enabled=True,
        force_mode="full_imu",
        imu_factor_mode="local_inertial_ba",
        gravity_pose_source="imu_integrated_estinit",
        vio_alpha_p=1.0,
        vio_alpha_v=1.0,
        vio_alpha_R=1.0,
        local_ba_window_size=2,
        local_ba_writeback="current",
        local_ba_fix_first_frame=True,
    ),
    "vio_local_ba_w2_imuatt_all": Variant(
        name="vio_local_ba_w2_imuatt_all",
        mode="imu_velocity_composed",
        scale=1.0,
        rot_enabled=True,
        trans_enabled=True,
        force_mode="full_imu",
        imu_factor_mode="local_inertial_ba",
        gravity_pose_source="imu_integrated_estinit",
        vio_alpha_p=1.0,
        vio_alpha_v=1.0,
        vio_alpha_R=1.0,
        local_ba_window_size=2,
        local_ba_writeback="all_optimized",
        local_ba_fix_first_frame=True,
    ),
    "vio_local_ba_w3_imuatt": Variant(
        name="vio_local_ba_w3_imuatt",
        mode="imu_velocity_composed",
        scale=1.0,
        rot_enabled=True,
        trans_enabled=True,
        force_mode="full_imu",
        imu_factor_mode="local_inertial_ba",
        gravity_pose_source="imu_integrated_estinit",
        vio_alpha_p=1.0,
        vio_alpha_v=1.0,
        vio_alpha_R=1.0,
        local_ba_window_size=3,
        local_ba_writeback="current",
        local_ba_fix_first_frame=True,
    ),
    "vio_local_ba_w3_imuatt_all": Variant(
        name="vio_local_ba_w3_imuatt_all",
        mode="imu_velocity_composed",
        scale=1.0,
        rot_enabled=True,
        trans_enabled=True,
        force_mode="full_imu",
        imu_factor_mode="local_inertial_ba",
        gravity_pose_source="imu_integrated_estinit",
        vio_alpha_p=1.0,
        vio_alpha_v=1.0,
        vio_alpha_R=1.0,
        local_ba_window_size=3,
        local_ba_writeback="all_optimized",
        local_ba_fix_first_frame=True,
    ),
    "vio_local_ba_w5_imuatt": Variant(
        name="vio_local_ba_w5_imuatt",
        mode="imu_velocity_composed",
        scale=1.0,
        rot_enabled=True,
        trans_enabled=True,
        force_mode="full_imu",
        imu_factor_mode="local_inertial_ba",
        gravity_pose_source="imu_integrated_estinit",
        vio_alpha_p=1.0,
        vio_alpha_v=1.0,
        vio_alpha_R=1.0,
        local_ba_window_size=5,
        local_ba_writeback="current",
        local_ba_fix_first_frame=True,
    ),
}

ROTFLOOR_VARIANTS = [
    "pure_macvo",
    "rotation_only",
    "rotation_only_rotstd02",
    "rotation_only_rotstd03",
    "aimvo_damping_s005",
    "aimvo_damping_s005_rotstd02",
    "aimvo_damping_s005_rotstd03",
    "aimvo_damping_s005_fullrotstd02",
    "aimvo_damping_s005_fullrotstd03",
]
VARIANT_PRESETS["rotfloor"] = ROTFLOOR_VARIANTS


class RunSpec(NamedTuple):
    trial: int
    scene: str
    scene_root: Path
    variant: Variant
    result_dir: Path


def cpb_fd_only_args(force_mode: str = "") -> list[str]:
    args = [
        "--adaptive-v3b",
        "--v3b-vc-mode",
        "two_level",
        "--v3b-vc-severe-thr",
        "30",
        "--v3b-vc-severe-sustain",
        "1",
        "--v3b-vc-mild-thr",
        "50",
        "--v3b-vc-mild-sustain",
        "5",
        "--v3b-fd-cooldown",
        "30",
    ]
    if force_mode:
        args += ["--v3b-force", force_mode]
    return args


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def variant_uses_autodiff(variant: Variant, autodiff: bool) -> bool:
    return bool(autodiff or variant.imu_factor_mode in {"preintegrated_vio", "local_inertial_ba"})


def variant_effective_imu_factor_mode(variant: Variant) -> str:
    return variant.imu_factor_mode


def variant_effective_local_ba_window_size(variant: Variant) -> str:
    return "" if variant.local_ba_window_size is None else str(variant.local_ba_window_size)


def variant_effective_local_ba_writeback(variant: Variant) -> str:
    return "" if variant.local_ba_writeback is None else str(variant.local_ba_writeback)


def variant_effective_local_ba_fix_first_frame(variant: Variant) -> str:
    return "" if variant.local_ba_fix_first_frame is None else str(int(variant.local_ba_fix_first_frame))


def variant_effective_gravity_pose_source(variant: Variant) -> str:
    return "estimated" if variant.gravity_pose_source is None else str(variant.gravity_pose_source)


def variant_effective_gravity_handling(variant: Variant) -> str:
    return "preintegration" if variant.gravity_handling is None else str(variant.gravity_handling)


def make_odom_cfg(variant: Variant, tmpdir: Path, *, autodiff: bool = False) -> Path:
    cfg = load_yaml(BASE_ODOM_CFG)
    odom_args = cfg["Odometry"]["args"]
    optimizer_args = cfg["Odometry"]["optimizer"]["args"]
    effective_imu_factor_mode = variant_effective_imu_factor_mode(variant)

    odom_args["mapping"] = False
    odom_args["imu_rot_prior_enable"] = variant.rot_enabled
    odom_args["imu_trans_prior_enable"] = variant.trans_enabled
    odom_args["imu_trans_prior_mode"] = variant.mode
    odom_args["imu_pose_fusion_enable"] = False

    optimizer_args["post_imu_fusion_enable"] = False
    optimizer_args["post_imu_fusion_mode"] = "none"
    optimizer_args["autodiff"] = variant_uses_autodiff(variant, autodiff)
    optimizer_args["imu_factor_mode"] = effective_imu_factor_mode
    optimizer_args["imu_rot_prior"] = variant.rot_enabled
    optimizer_args["imu_trans_prior_scale"] = float(variant.scale)
    if variant.rot_prior_std is not None:
        odom_args["imu_rot_prior_std"] = float(variant.rot_prior_std)
    if variant.rot_prior_std_when_translation is not None:
        odom_args["imu_rot_prior_std_when_translation"] = float(variant.rot_prior_std_when_translation)
    if variant.gravity_pose_source is not None:
        odom_args["imu_vio_gravity_pose_source"] = str(variant.gravity_pose_source)
    if variant.gravity_handling is not None:
        odom_args["imu_vio_gravity_handling"] = str(variant.gravity_handling)
    if variant.velocity_feedback_enable is not None:
        odom_args["imu_vio_velocity_feedback_enable"] = bool(variant.velocity_feedback_enable)
    if variant.vio_cov_scale is not None:
        optimizer_args["imu_vio_cov_scale"] = float(variant.vio_cov_scale)
    if variant.gravity_rp_correction_gain is not None:
        odom_args["imu_gravity_rp_correction_gain"] = float(variant.gravity_rp_correction_gain)
    if variant.gravity_rp_acc_tol is not None:
        odom_args["imu_gravity_rp_acc_tol"] = float(variant.gravity_rp_acc_tol)
    if variant.gravity_rp_window is not None:
        odom_args["imu_gravity_rp_window"] = int(variant.gravity_rp_window)
    if variant.vio_alpha_p is not None:
        optimizer_args["imu_vio_alpha_p"] = float(variant.vio_alpha_p)
    if variant.vio_alpha_v is not None:
        optimizer_args["imu_vio_alpha_v"] = float(variant.vio_alpha_v)
    if variant.vio_alpha_R is not None:
        optimizer_args["imu_vio_alpha_R"] = float(variant.vio_alpha_R)
    if variant.local_ba_window_size is not None:
        optimizer_args["local_ba_window_size"] = int(variant.local_ba_window_size)
    if variant.local_ba_writeback is not None:
        optimizer_args["local_ba_writeback"] = str(variant.local_ba_writeback)
    if variant.local_ba_fix_first_frame is not None:
        optimizer_args["local_ba_fix_first_frame"] = bool(variant.local_ba_fix_first_frame)

    out = tmpdir / f"odom_{variant.name}.yaml"
    write_yaml(out, cfg)
    return out


def make_seq_cfg(spec: RunSpec, tmpdir: Path) -> Path:
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(spec.scene_root)
    cfg["args"]["batch_root"] = str(spec.scene_root.parent)
    cfg["args"]["scene"] = spec.scene
    out = tmpdir / f"seq_trial{spec.trial}_{spec.variant.name}_{spec.scene}.yaml"
    write_yaml(out, cfg)
    return out


def build_specs(
    *,
    scenes: list[str],
    variants: list[str],
    trials: int,
    result_root: Path,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for scene in scenes:
        if scene not in SCENE_ROOTS:
            raise KeyError(f"Unknown scene {scene!r}. Known scenes: {', '.join(sorted(SCENE_ROOTS))}")
        for trial in range(1, trials + 1):
            for variant_name in variants:
                if variant_name not in VARIANTS:
                    raise KeyError(f"Unknown variant {variant_name!r}. Known variants: {', '.join(sorted(VARIANTS))}")
                variant = VARIANTS[variant_name]
                specs.append(
                    RunSpec(
                        trial=trial,
                        scene=scene,
                        scene_root=SCENE_ROOTS[scene],
                        variant=variant,
                        result_dir=result_root / f"trial_{trial}" / variant.name / scene,
                    )
                )
    return specs


def sanity_check(specs: list[RunSpec]) -> bool:
    ok = True
    checked: set[Path] = set()
    for spec in specs:
        if spec.scene_root in checked:
            continue
        checked.add(spec.scene_root)
        if not spec.scene_root.exists():
            print(f"ERROR: missing scene directory: {spec.scene_root}")
            ok = False
            continue
        for subdir in ("left", "right"):
            if not (spec.scene_root / subdir).is_dir():
                print(f"ERROR: missing {subdir}/ for {spec.scene}: {spec.scene_root / subdir}")
                ok = False
        for filename in ("imu_data.csv", "ref_pose.csv", "metadata.json"):
            if not (spec.scene_root / filename).exists():
                print(f"ERROR: missing {filename} for {spec.scene}: {spec.scene_root / filename}")
                ok = False
        metadata_path = spec.scene_root / "metadata.json"
        if metadata_path.exists():
            for error in validate_scene_metadata_conventions(metadata_path):
                print(f"ERROR: invalid metadata for {spec.scene}: {error}")
                ok = False
    return ok


def _metadata_get(meta: dict, dotted_key: str) -> object:
    cur: object = meta
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_scene_metadata_conventions(metadata_path: Path) -> list[str]:
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        return [f"{metadata_path}: cannot read metadata.json ({exc})"]

    errors: list[str] = []
    for key, expected in SCENE_METADATA_EXPECTED_EXACT.items():
        actual = _metadata_get(meta, key)
        if str(actual).strip() != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    for key, expected_substring in SCENE_METADATA_EXPECTED_CONTAINS.items():
        actual = str(_metadata_get(meta, key) or "")
        if expected_substring.lower() not in actual.lower():
            errors.append(f"{key}={actual!r}, expected to contain {expected_substring!r}")
    if _metadata_get(meta, "imu.acc_includes_gravity") is not True:
        errors.append("imu.acc_includes_gravity must be true")
    try:
        float(str(_metadata_get(meta, "imu.gravity_m_s2")).strip())
    except ValueError:
        errors.append("imu.gravity_m_s2 must be numeric")
    return errors


def scene_metadata_time_sync_fields(scene_root: Path) -> tuple[str, str]:
    metadata_path = scene_root / "metadata.json"
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return "", ""
    if _metadata_get(meta, "time_synchronization.camera_imu_time_offset_ns") is not None:
        return (
            str(_metadata_get(meta, "time_synchronization.camera_imu_time_offset_ns")),
            "metadata.time_synchronization.camera_imu_time_offset_ns",
        )
    if _metadata_get(meta, "time_synchronization.imu_time_offset_ns") is not None:
        return (
            str(_metadata_get(meta, "time_synchronization.imu_time_offset_ns")),
            "metadata.time_synchronization.imu_time_offset_ns",
        )
    return "", ""


def has_completed_pose(result_dir: Path) -> bool:
    if (result_dir / "poses.csv").exists() and (result_dir / "pose_coordinate_frame.txt").exists():
        return True
    return any(
        poses.parent.joinpath("pose_coordinate_frame.txt").exists()
        for poses in result_dir.rglob("poses.csv")
    )


def has_active_vio_diagnostics(result_dir: Path) -> bool:
    paths = [result_dir / "frame_pair_diagnostics.csv"]
    paths.extend(
        p for p in sorted(result_dir.rglob("frame_pair_diagnostics.csv"))
        if p != paths[0]
    )
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        for row in rows:
            if is_active_full_vio_row(row):
                return True
    return False


def _diagnostics_file_has_active_full_vio(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    return any(is_active_full_vio_row(row) for row in rows)


def has_complete_full_vio_bundle(result_dir: Path) -> bool:
    bundle = find_output_bundle(
        result_dir,
        require_same_dir_diagnostics=True,
        diagnostics_validator=_diagnostics_file_has_active_full_vio,
    )
    if not bundle.poses_path.exists():
        return False
    if not bundle.pose_frame_path.exists():
        return False
    if _diagnostics_file_has_active_full_vio(bundle.diagnostics_path):
        return True
    return False


def has_completed_run(spec: RunSpec) -> bool:
    if not has_completed_pose(spec.result_dir):
        return False
    if spec.variant.imu_factor_mode in {"preintegrated_vio", "local_inertial_ba"} and spec.variant.force_mode == "full_imu":
        return has_complete_full_vio_bundle(spec.result_dir)
    return True


def flatten_nested(result_dir: Path) -> None:
    direct_pose = result_dir / "poses.csv"
    direct_pose_frame = result_dir / "pose_coordinate_frame.txt"
    if direct_pose.exists() and direct_pose_frame.exists():
        return
    nested_poses = [
        p for p in sorted(result_dir.rglob("poses.csv"))
        if p.parent != result_dir and p.parent.joinpath("pose_coordinate_frame.txt").exists()
    ]
    if not nested_poses:
        return
    nested_dir = nested_poses[0].parent
    for src in nested_dir.iterdir():
        if not src.is_file():
            continue
        dest = result_dir / src.name
        # Replace incomplete direct outputs as one bundle; otherwise stale poses
        # can be paired with a fresh coordinate-frame sidecar after resume.
        os.replace(str(src), str(dest))


def write_manifest(result_root: Path, specs: list[RunSpec], seq_to: int | None, *, autodiff: bool = False) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    with (result_root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial",
                "scene",
                "variant",
                "imu_trans_prior_mode",
                "imu_trans_prior_scale",
                "imu_rot_prior_std",
                "imu_rot_prior_std_when_translation",
                "imu_factor_mode",
                "effective_imu_factor_mode",
                "imu_vio_gravity_pose_source",
                "imu_vio_gravity_handling",
                "force_mode",
                "imu_vio_alpha_p",
                "imu_vio_alpha_v",
                "imu_vio_alpha_R",
                "local_ba_window_size",
                "local_ba_writeback",
                "local_ba_fix_first_frame",
                "effective_local_ba_window_size",
                "effective_local_ba_writeback",
                "effective_local_ba_fix_first_frame",
                "rot_enabled",
                "trans_enabled",
                "scene_root",
                "result_dir",
                "seq_to",
                "autodiff",
                "metadata_camera_imu_time_offset_ns",
                "metadata_time_offset_source",
                "args",
                "created_at",
            ]
        )
        for spec in specs:
            metadata_time_offset_ns, metadata_time_offset_source = scene_metadata_time_sync_fields(spec.scene_root)
            writer.writerow(
                [
                    spec.trial,
                    spec.scene,
                    spec.variant.name,
                    spec.variant.mode,
                    spec.variant.scale,
                    "" if spec.variant.rot_prior_std is None else spec.variant.rot_prior_std,
                    "" if spec.variant.rot_prior_std_when_translation is None else spec.variant.rot_prior_std_when_translation,
                    spec.variant.imu_factor_mode,
                    variant_effective_imu_factor_mode(spec.variant),
                    variant_effective_gravity_pose_source(spec.variant),
                    variant_effective_gravity_handling(spec.variant),
                    spec.variant.force_mode,
                    "" if spec.variant.vio_alpha_p is None else spec.variant.vio_alpha_p,
                    "" if spec.variant.vio_alpha_v is None else spec.variant.vio_alpha_v,
                    "" if spec.variant.vio_alpha_R is None else spec.variant.vio_alpha_R,
                    "" if spec.variant.local_ba_window_size is None else spec.variant.local_ba_window_size,
                    "" if spec.variant.local_ba_writeback is None else spec.variant.local_ba_writeback,
                    "" if spec.variant.local_ba_fix_first_frame is None else int(spec.variant.local_ba_fix_first_frame),
                    variant_effective_local_ba_window_size(spec.variant),
                    variant_effective_local_ba_writeback(spec.variant),
                    variant_effective_local_ba_fix_first_frame(spec.variant),
                    int(spec.variant.rot_enabled),
                    int(spec.variant.trans_enabled),
                    str(spec.scene_root),
                    str(spec.result_dir),
                    "" if seq_to is None else seq_to,
                    int(variant_uses_autodiff(spec.variant, autodiff)),
                    metadata_time_offset_ns,
                    metadata_time_offset_source,
                    " ".join(cpb_fd_only_args(spec.variant.force_mode)),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )


def _manifest_row_for_spec(spec: RunSpec, seq_to: int | None, *, autodiff: bool = False) -> dict[str, str]:
    metadata_time_offset_ns, metadata_time_offset_source = scene_metadata_time_sync_fields(spec.scene_root)
    return {
        "trial": str(spec.trial),
        "scene": spec.scene,
        "variant": spec.variant.name,
        "imu_trans_prior_mode": spec.variant.mode,
        "imu_trans_prior_scale": str(spec.variant.scale),
        "imu_rot_prior_std": "" if spec.variant.rot_prior_std is None else str(spec.variant.rot_prior_std),
        "imu_rot_prior_std_when_translation": (
            "" if spec.variant.rot_prior_std_when_translation is None else str(spec.variant.rot_prior_std_when_translation)
        ),
        "imu_factor_mode": spec.variant.imu_factor_mode,
        "effective_imu_factor_mode": variant_effective_imu_factor_mode(spec.variant),
        "imu_vio_gravity_pose_source": variant_effective_gravity_pose_source(spec.variant),
        "imu_vio_gravity_handling": variant_effective_gravity_handling(spec.variant),
        "force_mode": spec.variant.force_mode,
        "imu_vio_alpha_p": "" if spec.variant.vio_alpha_p is None else str(spec.variant.vio_alpha_p),
        "imu_vio_alpha_v": "" if spec.variant.vio_alpha_v is None else str(spec.variant.vio_alpha_v),
        "imu_vio_alpha_R": "" if spec.variant.vio_alpha_R is None else str(spec.variant.vio_alpha_R),
        "local_ba_window_size": "" if spec.variant.local_ba_window_size is None else str(spec.variant.local_ba_window_size),
        "local_ba_writeback": "" if spec.variant.local_ba_writeback is None else str(spec.variant.local_ba_writeback),
        "local_ba_fix_first_frame": (
            "" if spec.variant.local_ba_fix_first_frame is None else str(int(spec.variant.local_ba_fix_first_frame))
        ),
        "effective_local_ba_window_size": variant_effective_local_ba_window_size(spec.variant),
        "effective_local_ba_writeback": variant_effective_local_ba_writeback(spec.variant),
        "effective_local_ba_fix_first_frame": variant_effective_local_ba_fix_first_frame(spec.variant),
        "rot_enabled": str(int(spec.variant.rot_enabled)),
        "trans_enabled": str(int(spec.variant.trans_enabled)),
        "scene_root": str(spec.scene_root),
        "result_dir": str(spec.result_dir),
        "seq_to": "" if seq_to is None else str(seq_to),
        "autodiff": str(int(variant_uses_autodiff(spec.variant, autodiff))),
        "metadata_camera_imu_time_offset_ns": metadata_time_offset_ns,
        "metadata_time_offset_source": metadata_time_offset_source,
        "args": " ".join(cpb_fd_only_args(spec.variant.force_mode)),
    }


def manifest_matches_specs(result_root: Path, specs: list[RunSpec], seq_to: int | None, *, autodiff: bool = False) -> bool:
    manifest = result_root / "run_manifest.csv"
    if not manifest.exists():
        return False
    try:
        with manifest.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    if len(rows) != len(specs):
        return False

    expected_rows = [_manifest_row_for_spec(spec, seq_to, autodiff=autodiff) for spec in specs]
    for actual, expected in zip(rows, expected_rows):
        for key, expected_value in expected.items():
            if str(actual.get(key, "")) != expected_value:
                return False
    return True


def write_manifest_guarded(
    result_root: Path,
    specs: list[RunSpec],
    seq_to: int | None,
    *,
    autodiff: bool = False,
    overwrite: bool = False,
) -> bool:
    manifest = result_root / "run_manifest.csv"
    if manifest.exists() and not overwrite:
        return False
    write_manifest(result_root, specs, seq_to, autodiff=autodiff)
    return True


def append_progress(result_root: Path, row: dict) -> None:
    progress = result_root / "progress.csv"
    with PROGRESS_LOCK:
        exists = progress.exists()
        with progress.open("a", newline="", encoding="utf-8") as f:
            fieldnames = ["trial", "scene", "variant", "status", "return_code", "runtime_s", "result_dir"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_one(spec: RunSpec, odom_cfg: Path, seq_cfg: Path, result_root: Path, timeout_s: int, seq_to: int | None) -> int:
    spec.result_dir.mkdir(parents=True, exist_ok=True)
    if has_completed_run(spec):
        print(f"  -> SKIP existing complete output: {spec.result_dir}")
        return 0

    cmd = [
        sys.executable,
        str(WORKDIR / "MACVO.py"),
        "--odom",
        str(odom_cfg),
        "--data",
        str(seq_cfg),
        "--resultRoot",
        str(spec.result_dir),
    ] + cpb_fd_only_args(spec.variant.force_mode)
    if seq_to is not None:
        cmd += ["--seq_to", str(seq_to)]

    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(WORKDIR), stdout=None, stderr=None, timeout=timeout_s)
        flatten_nested(spec.result_dir)
        elapsed = time.time() - started
        if proc.returncode == 0 and not has_completed_run(spec):
            status = "incomplete_output"
            return_code = INCOMPLETE_OUTPUT_RC
            print(
                f"  -> INCOMPLETE_OUTPUT rc={return_code} "
                f"(process rc=0, missing required output diagnostics; {elapsed:.1f}s)"
            )
        else:
            status = "ok" if proc.returncode == 0 else "failed"
            return_code = proc.returncode
            print(f"  -> {status.upper()} rc={return_code} ({elapsed:.1f}s)")
        append_progress(
            result_root,
            {
                "trial": spec.trial,
                "scene": spec.scene,
                "variant": spec.variant.name,
                "status": status,
                "return_code": return_code,
                "runtime_s": f"{elapsed:.1f}",
                "result_dir": str(spec.result_dir),
            },
        )
        return return_code
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(f"  -> TIMEOUT ({timeout_s}s)")
        append_progress(
            result_root,
            {
                "trial": spec.trial,
                "scene": spec.scene,
                "variant": spec.variant.name,
                "status": "timeout",
                "return_code": "",
                "runtime_s": f"{elapsed:.1f}",
                "result_dir": str(spec.result_dir),
            },
        )
        return 124


RunCallable = Callable[[RunSpec, Path, Path, Path, int, int | None], int]


def execute_run_schedule(
    specs: list[RunSpec],
    odom_cfgs: dict[str, Path],
    tmpdir: Path,
    result_root: Path,
    *,
    timeout_s: int,
    seq_to: int | None,
    jobs: int = 1,
    runner: RunCallable = run_one,
) -> list[tuple[RunSpec, int]]:
    failures: list[tuple[RunSpec, int]] = []
    total = len(specs)
    max_workers = max(1, int(jobs))

    def execute_one(idx: int, spec: RunSpec) -> tuple[int, RunSpec, int]:
        seq_cfg = make_seq_cfg(spec, tmpdir)
        print(f"\n-- Run {idx}/{total}: trial={spec.trial} scene={spec.scene} variant={spec.variant.name} --")
        rc = runner(spec, odom_cfgs[spec.variant.name], seq_cfg, result_root, timeout_s, seq_to)
        return idx, spec, rc

    if max_workers == 1:
        for idx, spec in enumerate(specs, start=1):
            _, finished_spec, rc = execute_one(idx, spec)
            if rc != 0:
                failures.append((finished_spec, rc))
        return failures

    print(f"\nParallel execution: {max_workers} jobs")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(execute_one, idx, spec) for idx, spec in enumerate(specs, start=1)]
        for future in as_completed(futures):
            _, finished_spec, rc = future.result()
            if rc != 0:
                failures.append((finished_spec, rc))
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small AIM-VO IMU prior mode diagnostic grid.")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--preset",
        choices=sorted(SCENE_PRESETS),
        default="smoke3",
        help="Scene preset. Use kept12 for the retained paper-scene diagnostic set.",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="Override preset scenes explicitly.")
    parser.add_argument(
        "--variant-preset",
        choices=sorted(VARIANT_PRESETS),
        default="default",
        help="Variant preset. Use main3 for pure/rotation/damping repeated paper-facing diagnostics.",
    )
    parser.add_argument("--variants", nargs="*", default=None, help="Override variant preset explicitly.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seq-to", type=int, default=None, help="Optional frame crop for quick diagnosis, e.g. 600.")
    parser.add_argument("--autodiff", action="store_true", help="Use autograd Jacobians instead of analytic Jacobians.")
    parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT_S)
    parser.add_argument("--jobs", type=int, default=1, help="Number of MACVO subprocesses to run concurrently.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-only", action="store_true", help="Write the run manifest and exit without running MACVO.")
    parser.add_argument("--overwrite-manifest", action="store_true", help="Rewrite run_manifest.csv during a normal run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = list(args.scenes) if args.scenes is not None else list(SCENE_PRESETS[args.preset])
    variants = list(args.variants) if args.variants is not None else list(VARIANT_PRESETS[args.variant_preset])
    specs = build_specs(
        scenes=scenes,
        variants=variants,
        trials=int(args.trials),
        result_root=args.result_root,
    )

    print("=" * 78)
    print("  AIM-VO IMU translation-prior diagnostic grid")
    print(f"  Results:  {args.result_root}")
    print(f"  Preset:   {args.preset if args.scenes is None else 'custom'}")
    print(f"  Scenes:   {', '.join(scenes)}")
    print(f"  Variants: {', '.join(variants)}")
    print(f"  VarPreset:{args.variant_preset if args.variants is None else 'custom'}")
    print(f"  Trials:   {args.trials}")
    print(f"  Seq-to:   {args.seq_to if args.seq_to is not None else 'full sequence'}")
    forced_autodiff = any(variant_uses_autodiff(VARIANTS[v], bool(args.autodiff)) for v in variants)
    autodiff_note = " (enabled by selected variants)" if forced_autodiff and not bool(args.autodiff) else ""
    print(f"  Autodiff: {bool(args.autodiff)}{autodiff_note}")
    print(f"  Jobs:     {max(1, int(args.jobs))}")
    print(f"  Runs:     {len(specs)}")
    print("=" * 78)

    if not sanity_check(specs):
        return 1

    existing_manifest = args.result_root / "run_manifest.csv"
    if existing_manifest.exists() and not bool(args.manifest_only or args.overwrite_manifest):
        if not manifest_matches_specs(args.result_root, specs, args.seq_to, autodiff=bool(args.autodiff)):
            print(f"\nERROR: existing manifest does not match requested schedule: {existing_manifest}")
            print("       Use --overwrite-manifest to replace it, or choose a fresh --result-root.")
            return 1

    if args.dry_run:
        print("\nDRY RUN - no MACVO process will be started.")
        for idx, spec in enumerate(specs, start=1):
            marker = "SKIP" if has_completed_run(spec) else "RUN"
            print(
                f"  [{idx:02d}/{len(specs):02d}] {marker} "
                f"trial={spec.trial} scene={spec.scene} variant={spec.variant.name} -> {spec.result_dir}"
            )
        return 0

    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest_written = write_manifest_guarded(
        args.result_root,
        specs,
        args.seq_to,
        autodiff=bool(args.autodiff),
        overwrite=bool(args.manifest_only or args.overwrite_manifest),
    )
    if args.manifest_only:
        print(f"\nMANIFEST ONLY - wrote {len(specs)} planned runs to {args.result_root / 'run_manifest.csv'}")
        return 0
    if manifest_written:
        print(f"\nWrote manifest: {args.result_root / 'run_manifest.csv'}")
    else:
        print(f"\nKeeping existing manifest: {args.result_root / 'run_manifest.csv'}")

    tmpdir = Path(tempfile.mkdtemp(prefix="vio_imu_prior_grid_"))
    print(f"Temp config dir: {tmpdir}")
    odom_cfgs = {
        variant: make_odom_cfg(VARIANTS[variant], tmpdir, autodiff=bool(args.autodiff))
        for variant in variants
    }

    started_all = time.time()
    failures = execute_run_schedule(
        specs,
        odom_cfgs,
        tmpdir,
        args.result_root,
        timeout_s=args.timeout,
        seq_to=args.seq_to,
        jobs=args.jobs,
    )

    elapsed_all = time.time() - started_all
    print("\n" + "=" * 78)
    print(f"  Attempted schedule in {elapsed_all / 60:.1f} min")
    print(f"  Results:  {args.result_root}")
    print(f"  Manifest: {args.result_root / 'run_manifest.csv'}")
    print(f"  Progress: {args.result_root / 'progress.csv'}")
    if failures:
        print(f"  Failures: {len(failures)}")
        for spec, rc in failures:
            print(f"    - trial={spec.trial} scene={spec.scene} variant={spec.variant.name} rc={rc}")
    else:
        print("  No failed return codes in this run.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

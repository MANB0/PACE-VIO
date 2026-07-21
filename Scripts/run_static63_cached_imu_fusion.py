#!/usr/bin/env python3
"""Replay three unique visual caches against all four IMU data variants."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Scripts.run_vio_imu_prior_mode_grid import (  # noqa: E402
    RunSpec,
    flatten_nested,
    has_completed_run,
    make_odom_cfg,
    make_seq_cfg,
)
from Scripts.run_visual_factor_cache_batch import (  # noqa: E402
    LATEST_IMUATT_METHOD,
    RETAINED_VARIANTS,
    switch_dashboard,
)


WORKDIR = Path("/home/admin1/macvo-dev")
PYTHON = Path("/home/admin1/miniconda3/envs/macvo/bin/python")
BATCH_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
CACHE_ROOT = WORKDIR / "VisualCache" / "static63_unique_visual_20260713"
DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "static63_cached_imu_fusion_four_configs_20260713"
DEFAULT_LOG = WORKDIR / "logs" / "static63_cached_imu_fusion_four_configs_20260713.log"
CALIBRATED_STATICINIT_METHOD = "vio_preintegrated_full_imuatt_staticinit_calibrated"
CALIBRATED_RESULT_ROOT = (
    WORKDIR / "Results" / "static63_cached_imu_fusion_staticinit_calibrated_20260713"
)
CALIBRATED_LOG = (
    WORKDIR / "logs" / "static63_cached_imu_fusion_staticinit_calibrated_20260713.log"
)
TWO_STATE_FIXED_LAG_METHOD = "vio_two_state_fixed_lag_staticinit_calibrated"
TWO_STATE_FIXED_LAG_RESULT_ROOT = (
    WORKDIR / "Results" / "static63_two_state_fixed_lag_20260714"
)
TWO_STATE_FIXED_LAG_LOG = (
    WORKDIR / "logs" / "static63_two_state_fixed_lag_20260714.log"
)
IMU_VIO_GRAVITY_HANDLING_CHOICES = (
    "legacy_external_attitude_gravity_compensation",
    "standard_local_frame_preintegration",
)
TWO_STATE_COVARIANCE_MODE_CHOICES = (
    "current_independent_step",
    "sampling_aware",
    "sampling_aware_cross_edge",
)
METHOD_CHOICES = {
    "legacy": LATEST_IMUATT_METHOD,
    "calibrated-staticinit": CALIBRATED_STATICINIT_METHOD,
    "two-state-fixed-lag": TWO_STATE_FIXED_LAG_METHOD,
}


@dataclass(frozen=True)
class ReplayTask:
    trajectory: str
    imu_config: str
    dataset_scene: str
    cache_scene: str
    cache_dir_override: Path | None = None

    @property
    def scene_root(self) -> Path:
        return BATCH_ROOT / self.dataset_scene

    @property
    def cache_dir(self) -> Path:
        return (
            self.cache_dir_override
            if self.cache_dir_override is not None
            else CACHE_ROOT / self.cache_scene
        )


TASKS = tuple(
    ReplayTask(trajectory, imu_config, f"{prefix}_{suffix}", cache_scene)
    for trajectory, prefix, cache_scene in (
        ("circle", "clear_circle_truth", "clear_circle_truth_normal_noise"),
        (
            "stop_turn_rectangle",
            "clear_stop_turn_rectangle_truth",
            "clear_stop_turn_rectangle_truth_normal_noise",
        ),
        ("straight", "clear_straight_truth", "clear_straight_truth_normal_noise"),
    )
    for imu_config, suffix in (
        ("normal_noise", "normal_noise"),
        ("bias_no_noise", "bias_no_noise"),
        ("noise_no_bias", "noise_no_bias"),
        ("no_noise_no_bias", "no_noise_no_bias"),
    )
)
PROGRESS_LOCK = threading.Lock()


def result_dir(result_root: Path, task: ReplayTask, method_name: str) -> Path:
    return result_root / "trial_1" / method_name / task.dataset_scene


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def expected_frames(task: ReplayTask, seq_to: int | None) -> int:
    total = count_rows(task.scene_root / "ref_pose.csv")
    return total if seq_to is None else min(total, int(seq_to))


def has_complete_task_output(spec: RunSpec, expected: int) -> bool:
    if not spec.result_dir.exists():
        return False
    flatten_nested(spec.result_dir)
    poses = spec.result_dir / "poses.csv"
    return (
        poses.exists()
        and count_rows(poses) == int(expected)
        and has_completed_run(spec)
    )


def reset_task_output(result_root: Path, output: Path) -> None:
    root = result_root.resolve()
    target = output.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to clear task output outside result root: {target}")
    if target.exists():
        shutil.rmtree(target)


def select_tasks(scene_names: list[str] | None) -> tuple[ReplayTask, ...]:
    if scene_names is None:
        return TASKS
    requested = set(scene_names)
    selected = tuple(task for task in TASKS if task.dataset_scene in requested)
    unknown = sorted(requested - {task.dataset_scene for task in selected})
    if unknown:
        raise ValueError("--scenes contains unknown Static63 datasets: " + ", ".join(unknown))
    if not selected:
        raise ValueError("--scenes did not match any known Static63 dataset scene")
    return selected


def validate_inputs(
    tasks: tuple[ReplayTask, ...] = TASKS,
    *,
    require_relative_pose_factors: bool = False,
    require_compressed_uvd_factors: bool = False,
) -> None:
    errors: list[str] = []
    for task in tasks:
        for path in (
            task.scene_root / "imu_data.csv",
            task.scene_root / "ref_pose.csv",
            task.scene_root / "metadata.json",
            task.cache_dir / "manifest.json",
        ):
            if not path.exists():
                errors.append(f"missing input: {path}")
        if require_relative_pose_factors:
            sidecar = task.cache_dir / "relative_pose_factors.npz"
            if not sidecar.exists():
                errors.append(
                    f"missing relative-pose sidecar: {sidecar}; run "
                    "Scripts/build_relative_pose_factor_cache.py first"
                )
        if require_compressed_uvd_factors:
            sidecar = task.cache_dir / "compressed_uvd_pose_factors.npz"
            if not sidecar.exists():
                errors.append(
                    f"missing compressed-UVD sidecar: {sidecar}; run "
                    "Scripts/build_compressed_uvd_pose_factor_cache.py first"
                )
    if errors:
        raise FileNotFoundError("\n".join(errors))


def write_manifest(
    result_root: Path,
    seq_to: int | None,
    *,
    tasks: tuple[ReplayTask, ...] = TASKS,
    method_name: str,
    static_initialization: bool,
    imu_vio_cov_diagonal_floor: float | None = None,
    two_state_fixed_lag: bool = False,
    imu_vio_gravity_handling: str | None = None,
    two_state_visual_factor_mode: str = "relative_pose",
    two_state_warm_start: str = "macvo_pose",
    two_state_covariance_mode: str = "current_independent_step",
    sa_v2_rank_aware_imu_whitening: bool = False,
) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / "run_manifest.csv"
    fields = [
        "trial",
        "scene",
        "variant",
        "trajectory",
        "imu_config",
        "scene_root",
        "cache_scene",
        "cache_dir",
        "result_dir",
        "seq_to",
        "args",
        "created_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "trial": 1,
                    "scene": task.dataset_scene,
                    "variant": method_name,
                    "trajectory": task.trajectory,
                    "imu_config": task.imu_config,
                    "scene_root": task.scene_root,
                    "cache_scene": task.cache_scene,
                    "cache_dir": task.cache_dir,
                    "result_dir": result_dir(result_root, task, method_name),
                    "seq_to": "" if seq_to is None else int(seq_to),
                    "args": (
                        "visual-cache replay; cache scene name retained for strict validation; "
                        f"static_initialization={static_initialization}; "
                        "corrected_bias_persistence=true; "
                        f"imu_factor_mode={'two_state_fixed_lag' if two_state_fixed_lag else 'preintegrated_vio'}; "
                        f"imu_vio_cov_diagonal_floor={imu_vio_cov_diagonal_floor}; "
                        f"imu_vio_gravity_handling={imu_vio_gravity_handling}; "
                        f"two_state_visual_factor_mode={two_state_visual_factor_mode}; "
                        f"two_state_warm_start={two_state_warm_start}; "
                        f"two_state_covariance_mode={two_state_covariance_mode}; "
                        "sa_v2_rank_aware_imu_whitening="
                        f"{sa_v2_rank_aware_imu_whitening}"
                    ),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )


def configure_odometry(
    odom_cfg: Path,
    *,
    static_initialization: bool,
    imu_vio_cov_diagonal_floor: float | None,
    two_state_fixed_lag: bool = False,
    imu_vio_gravity_handling: str | None = None,
    two_state_visual_factor_mode: str = "relative_pose",
    two_state_warm_start: str = "macvo_pose",
    two_state_uvd_huber_delta: float = 0.1,
    two_state_covariance_mode: str = "current_independent_step",
    sa_v2_rank_aware_imu_whitening: bool = False,
) -> None:
    with odom_cfg.open("r", encoding="utf-8") as stream:
        odom_config = yaml.safe_load(stream)
    if static_initialization:
        odom_args = odom_config["Odometry"]["args"]
        odom_args["imu_static_initialization_enable"] = True
        odom_args["imu_static_initialization_duration_s"] = 3.0
        odom_args["imu_static_sigma_multiplier"] = 5.0
        odom_args["imu_static_gyro_mean_norm_max"] = 0.03
        odom_args["imu_static_acc_norm_error_max"] = 0.6
    if imu_vio_cov_diagonal_floor is not None:
        optimizer_args = odom_config["Odometry"]["optimizer"]["args"]
        optimizer_args["imu_vio_cov_diagonal_floor"] = float(imu_vio_cov_diagonal_floor)
    if imu_vio_gravity_handling is not None:
        odom_config["Odometry"]["args"]["imu_vio_gravity_handling"] = (
            imu_vio_gravity_handling
        )
    if two_state_fixed_lag:
        optimizer_args = odom_config["Odometry"]["optimizer"]["args"]
        optimizer_args["imu_factor_mode"] = "two_state_fixed_lag"
        optimizer_args["graph_type"] = "disp"
        optimizer_args["autodiff"] = True
        optimizer_args["parallel"] = False
        optimizer_args["post_imu_fusion_enable"] = False
        optimizer_args["post_imu_fusion_mode"] = "none"
        optimizer_args["two_state_max_iterations"] = 20
        optimizer_args["two_state_visual_huber_delta"] = 3.0
        optimizer_args["two_state_visual_factor_mode"] = str(two_state_visual_factor_mode)
        optimizer_args["two_state_warm_start"] = str(two_state_warm_start)
        optimizer_args["two_state_uvd_huber_delta"] = float(two_state_uvd_huber_delta)
        optimizer_args["two_state_initial_pose_translation_std"] = 1.0e-5
        optimizer_args["two_state_initial_pose_rotation_std"] = 1.0e-5
        optimizer_args["two_state_initial_velocity_std"] = 5.0e-2
        optimizer_args["two_state_initial_acc_bias_std"] = 2.0e-1
        optimizer_args["two_state_initial_gyro_bias_std"] = 2.0e-2
        optimizer_args["two_state_cross_edge_rank_aware_imu_whitening"] = bool(
            sa_v2_rank_aware_imu_whitening
        )
        odom_config["Odometry"]["args"]["two_state_covariance_mode"] = str(
            two_state_covariance_mode
        )
    with odom_cfg.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(odom_config, stream, sort_keys=False)


def append_progress(
    result_root: Path,
    task: ReplayTask,
    *,
    method_name: str,
    status: str,
    return_code: int | str = "",
    runtime_s: float | str = "",
) -> None:
    path = result_root / "progress.csv"
    fields = ["trial", "scene", "variant", "status", "return_code", "runtime_s", "result_dir"]
    with PROGRESS_LOCK:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "trial": 1,
                    "scene": task.dataset_scene,
                    "variant": method_name,
                    "status": status,
                    "return_code": return_code,
                    "runtime_s": runtime_s,
                    "result_dir": result_dir(result_root, task, method_name),
                }
            )


def run_task(
    task: ReplayTask,
    *,
    result_root: Path,
    odom_cfg: Path,
    config_root: Path,
    timeout_s: int,
    seq_to: int | None,
    force: bool,
    method_name: str,
    two_state_fixed_lag: bool,
    sa_v2_reset_prior_at_frame: int | None = None,
    sa_v2_checkpoint_frames: tuple[int, ...] = (),
    sa_v2_rank_aware_imu_whitening: bool = False,
) -> int:
    output = result_dir(result_root, task, method_name)
    variant = RETAINED_VARIANTS[LATEST_IMUATT_METHOD]._replace(
        name=method_name,
        imu_factor_mode=(
            "two_state_fixed_lag"
            if two_state_fixed_lag
            else RETAINED_VARIANTS[LATEST_IMUATT_METHOD].imu_factor_mode
        ),
    )
    cache_named_spec = RunSpec(
        trial=1,
        scene=task.cache_scene,
        scene_root=task.scene_root,
        variant=variant,
        result_dir=output,
    )
    expected = expected_frames(task, seq_to)
    completed = has_complete_task_output(cache_named_spec, expected)
    if not force and completed:
        print(f"SKIP complete: {task.dataset_scene}")
        append_progress(
            result_root,
            task,
            method_name=method_name,
            status="ok",
            return_code=0,
            runtime_s="0.0",
        )
        return 0

    if output.exists():
        print(f"CLEAR incomplete/stale output: {task.dataset_scene}")
        reset_task_output(result_root, output)
    output.mkdir(parents=True, exist_ok=True)
    task_config_root = config_root / task.dataset_scene
    task_config_root.mkdir(parents=True, exist_ok=True)
    seq_cfg = make_seq_cfg(cache_named_spec, task_config_root)
    command = [
        str(PYTHON),
        str(WORKDIR / "MACVO.py"),
        "--odom",
        str(odom_cfg),
        "--data",
        str(seq_cfg),
        "--resultRoot",
        str(output),
        "--visual-cache-mode",
        "replay",
        "--visual-cache-path",
        str(task.cache_dir),
    ]
    if seq_to is not None:
        command.extend(["--seq_to", str(int(seq_to))])

    task_log = result_root / "logs" / f"{method_name}__{task.dataset_scene}.log"
    task_log.parent.mkdir(parents=True, exist_ok=True)
    append_progress(result_root, task, method_name=method_name, status="running")
    started = time.monotonic()
    print(f"RUN {task.dataset_scene} using cache {task.cache_scene}")
    try:
        environment = os.environ.copy()
        if sa_v2_reset_prior_at_frame is not None:
            environment["MACVO_SA_V2_RESET_PRIOR_AT_FRAME"] = str(
                int(sa_v2_reset_prior_at_frame)
            )
        if sa_v2_checkpoint_frames:
            environment["MACVO_SA_V2_CHECKPOINT_FRAMES"] = ",".join(
                str(int(frame)) for frame in sa_v2_checkpoint_frames
            )
            environment["MACVO_SA_V2_CHECKPOINT_DIR"] = str(
                output / "sa_v2_checkpoints"
            )
        environment["MACVO_SA_V2_RANK_AWARE_IMU_WHITENING"] = (
            "true" if sa_v2_rank_aware_imu_whitening else "false"
        )
        with task_log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                cwd=str(WORKDIR),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=int(timeout_s),
                check=False,
            )
        return_code = int(process.returncode)
    except subprocess.TimeoutExpired:
        return_code = 124
    elapsed = time.monotonic() - started
    flatten_nested(output)

    poses = output / "poses.csv"
    if return_code == 0 and (not poses.exists() or count_rows(poses) != expected):
        return_code = 2
    if return_code == 0 and not has_completed_run(cache_named_spec):
        return_code = 2
    status = "ok" if return_code == 0 else ("timeout" if return_code == 124 else "failed")
    append_progress(
        result_root,
        task,
        method_name=method_name,
        status=status,
        return_code=return_code,
        runtime_s=f"{elapsed:.1f}",
    )
    print(f"{status.upper()} {task.dataset_scene}: rc={return_code}, {elapsed:.1f}s")
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_CHOICES),
        default="calibrated-staticinit",
        help="Run the corrected static-initialized method by default; legacy remains reproducible.",
    )
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--imu-vio-cov-diagonal-floor",
        type=float,
        default=None,
        help="Optional additive diagonal regularization for the full 9x9 IMU covariance.",
    )
    parser.add_argument(
        "--variant-name",
        default=None,
        help="Optional explicit result/method label for a controlled experiment.",
    )
    parser.add_argument(
        "--imu-vio-gravity-handling",
        choices=IMU_VIO_GRAVITY_HANDLING_CHOICES,
        default=None,
        help=(
            "Optional exact preintegration architecture. This controlled runner rejects "
            "ambiguous aliases such as 'preintegration' and 'residual'."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional dataset-scene subset; defaults to all twelve replay tasks.",
    )
    parser.add_argument(
        "--visual-cache-path-override",
        type=Path,
        default=None,
        help="Audit-only cache override; requires exactly one selected scene.",
    )
    parser.add_argument(
        "--two-state-visual-factor-mode",
        choices=("relative_pose", "direct_uvd", "compressed_uvd"),
        default="relative_pose",
    )
    parser.add_argument(
        "--two-state-warm-start",
        choices=("macvo_pose", "imu_propagation"),
        default="macvo_pose",
    )
    parser.add_argument("--two-state-uvd-huber-delta", type=float, default=0.1)
    parser.add_argument(
        "--two-state-covariance-mode",
        choices=TWO_STATE_COVARIANCE_MODE_CHOICES,
        default="current_independent_step",
        help=(
            "IMU preintegration covariance contract used by the two-state backend. "
            "The default preserves the historical Current U1 behavior."
        ),
    )
    parser.add_argument(
        "--sa-v2-reset-prior-at-frame",
        type=int,
        default=None,
        help="Audit-only: replace the carried SA-v2 prior by the initial diagonal prior at this source frame.",
    )
    parser.add_argument(
        "--sa-v2-checkpoint-frames",
        type=int,
        nargs="*",
        default=(),
        help="Audit-only: save the SA-v2 state/prior after these frame indices.",
    )
    parser.add_argument(
        "--sa-v2-rank-aware-imu-whitening",
        action="store_true",
        help=(
            "Whiten the singular SA-v2 unique IMU covariance only in its "
            "positive-eigenvalue subspace; legacy hard-floor whitening remains "
            "the default for controlled A/B replay."
        ),
    )
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method_name = str(args.variant_name or METHOD_CHOICES[args.method])
    two_state_fixed_lag = args.method == "two-state-fixed-lag"
    static_initialization = args.method in {"calibrated-staticinit", "two-state-fixed-lag"}
    if args.imu_vio_cov_diagonal_floor is not None and args.imu_vio_cov_diagonal_floor < 0.0:
        raise ValueError("--imu-vio-cov-diagonal-floor must be non-negative")
    if args.two_state_uvd_huber_delta <= 0.0:
        raise ValueError("--two-state-uvd-huber-delta must be positive")
    if two_state_fixed_lag:
        default_result_root = TWO_STATE_FIXED_LAG_RESULT_ROOT
        default_log = TWO_STATE_FIXED_LAG_LOG
    elif static_initialization:
        default_result_root = CALIBRATED_RESULT_ROOT
        default_log = CALIBRATED_LOG
    else:
        default_result_root = DEFAULT_RESULT_ROOT
        default_log = DEFAULT_LOG
    result_root = (args.result_root or default_result_root).expanduser().resolve()
    selected_tasks = select_tasks(args.scenes)
    if args.visual_cache_path_override is not None:
        if len(selected_tasks) != 1:
            raise ValueError("--visual-cache-path-override requires exactly one scene")
        override = args.visual_cache_path_override.expanduser().resolve()
        selected_tasks = (
            replace(selected_tasks[0], cache_dir_override=override),
        )
    validate_inputs(
        selected_tasks,
        require_relative_pose_factors=(
            two_state_fixed_lag
            and (
                args.two_state_visual_factor_mode == "relative_pose"
                or (
                    args.two_state_warm_start == "macvo_pose"
                    and args.two_state_visual_factor_mode != "compressed_uvd"
                )
            )
        ),
        require_compressed_uvd_factors=(
            two_state_fixed_lag
            and args.two_state_visual_factor_mode == "compressed_uvd"
        ),
    )
    write_manifest(
        result_root,
        args.seq_to,
        tasks=selected_tasks,
        method_name=method_name,
        static_initialization=static_initialization,
        imu_vio_cov_diagonal_floor=args.imu_vio_cov_diagonal_floor,
        two_state_fixed_lag=two_state_fixed_lag,
        imu_vio_gravity_handling=args.imu_vio_gravity_handling,
        two_state_visual_factor_mode=args.two_state_visual_factor_mode,
        two_state_warm_start=args.two_state_warm_start,
        two_state_covariance_mode=args.two_state_covariance_mode,
        sa_v2_rank_aware_imu_whitening=bool(
            args.sa_v2_rank_aware_imu_whitening
        ),
    )
    print(f"Static63 cached IMU fusion: {len(selected_tasks)} selected of {len(TASKS)} runs")
    print(f"Method: {method_name}")
    print(f"Result root: {result_root}")
    for task in selected_tasks:
        print(f"  {task.dataset_scene} <- {task.cache_scene}")
    failures = 0
    with tempfile.TemporaryDirectory(prefix="static63_cached_imu_fusion_") as temporary:
        config_root = Path(temporary)
        variant = RETAINED_VARIANTS[LATEST_IMUATT_METHOD]._replace(name=method_name)
        odom_cfg = make_odom_cfg(variant, config_root)
        configure_odometry(
            odom_cfg,
            static_initialization=static_initialization,
            imu_vio_cov_diagonal_floor=args.imu_vio_cov_diagonal_floor,
            two_state_fixed_lag=two_state_fixed_lag,
            imu_vio_gravity_handling=args.imu_vio_gravity_handling,
            two_state_visual_factor_mode=args.two_state_visual_factor_mode,
            two_state_warm_start=args.two_state_warm_start,
            two_state_uvd_huber_delta=args.two_state_uvd_huber_delta,
            two_state_covariance_mode=args.two_state_covariance_mode,
            sa_v2_rank_aware_imu_whitening=bool(
                args.sa_v2_rank_aware_imu_whitening
            ),
        )
        config_snapshot = result_root / "configs" / "odometry.yaml"
        config_snapshot.parent.mkdir(parents=True, exist_ok=True)
        config_snapshot.write_text(odom_cfg.read_text(encoding="utf-8"), encoding="utf-8")
        if args.dry_run:
            print(f"Dry run only; effective odometry config: {config_snapshot}")
            return 0
        if not args.no_dashboard:
            dashboard_log = default_log
            if len(selected_tasks) == 1:
                dashboard_log = (
                    result_root
                    / "logs"
                    / f"{method_name}__{selected_tasks[0].dataset_scene}.log"
                )
            switch_dashboard(result_root, dashboard_log, port=int(args.dashboard_port))
        with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
            futures = {
                executor.submit(
                    run_task,
                    task,
                    result_root=result_root,
                    odom_cfg=odom_cfg,
                    config_root=config_root,
                    timeout_s=int(args.timeout),
                    seq_to=args.seq_to,
                    force=bool(args.force),
                    method_name=method_name,
                    two_state_fixed_lag=two_state_fixed_lag,
                    sa_v2_reset_prior_at_frame=args.sa_v2_reset_prior_at_frame,
                    sa_v2_checkpoint_frames=tuple(args.sa_v2_checkpoint_frames),
                    sa_v2_rank_aware_imu_whitening=bool(
                        args.sa_v2_rank_aware_imu_whitening
                    ),
                ): task
                for task in selected_tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    failures += future.result() != 0
                except Exception as error:
                    failures += 1
                    append_progress(
                        result_root,
                        task,
                        method_name=method_name,
                        status="failed",
                        return_code=2,
                    )
                    print(f"FAILED {task.dataset_scene}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

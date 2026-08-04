#!/usr/bin/env python3
"""Run the complete PACE-VIO paper experiment matrix.

Only dataset paths and initialization policies live in the manifest.  Method
definitions are fixed here so every scene uses the same full-sequence protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class MethodSpec:
    key: str
    paper_name: str
    backend: str
    visual_factor: str
    detector: str = "off"


METHODS = (
    MethodSpec("pose_isam2", "Pose", "isam2", "pose"),
    MethodSpec("uvd_isam2", "UVD", "isam2", "uvd"),
    MethodSpec("pace_two_state", "PACE-Two", "two_state", "pace"),
    MethodSpec("pace_isam2", "PACE-iSAM2", "isam2", "pace"),
    MethodSpec("pace_vio", "PACE-VIO", "isam2", "pace", "v2"),
)

VISUAL_CACHE_STAGE_KEY = "visual_cache"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("manifest.datasets must be a non-empty list")
    labels: set[str] = set()
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError("each dataset entry must be an object")
        label = str(item.get("scenario", "")).strip()
        dataset = Path(str(item.get("path", ""))).expanduser()
        if not label or label in labels:
            raise ValueError(f"scenario labels must be non-empty and unique: {label!r}")
        if not dataset.is_dir():
            raise FileNotFoundError(f"dataset does not exist: {dataset}")
        labels.add(label)
        initialization = item.get("static_initialization", {})
        mode = str(initialization.get("mode", "adaptive"))
        if mode not in {"adaptive", "fixed", "off"}:
            raise ValueError(f"unsupported static initialization mode: {mode}")
        duration = initialization.get("duration_s")
        if mode == "fixed" and (duration is None or float(duration) <= 0.0):
            raise ValueError(f"{label}: fixed initialization requires duration_s > 0")
        if mode != "fixed" and duration is not None:
            raise ValueError(f"{label}: duration_s is valid only for fixed initialization")
        paper_evaluation = item.get("paper_evaluation", True)
        if not isinstance(paper_evaluation, bool):
            raise ValueError(f"{label}: paper_evaluation must be a boolean")
    return manifest


def _paper_evaluation_enabled(dataset: dict[str, Any]) -> bool:
    return bool(dataset.get("paper_evaluation", True))


def build_command(
    *,
    dataset: dict[str, Any],
    method: MethodSpec,
    output_root: Path,
    model: Path,
    runtime: dict[str, Any],
    dry_run: bool,
    visual_cache_mode: str = "live",
    visual_cache_path: Path | None = None,
) -> list[str]:
    dataset_path = Path(str(dataset["path"])).expanduser().resolve()
    initialization = dataset.get("static_initialization", {})
    mode = str(initialization.get("mode", "adaptive"))
    command = [
        sys.executable,
        str(ROOT / "Scripts/run_pace_vio.py"),
        "--dataset", str(dataset_path),
        "--output", str((output_root / method.key).resolve()),
        "--model", str(model.resolve()),
        "--mode", str(runtime.get("mode", "pipeline")),
        "--cpu-threads", str(int(runtime.get("cpu_threads", 4))),
        "--timeout", str(int(runtime.get("timeout_s", 21600))),
        "--vio-backend", method.backend,
        "--visual-factor", method.visual_factor,
        "--near-zero-velocity-detector", method.detector,
        "--static-init-mode", mode,
        "--no-live-display",
    ]
    command.append(
        "--paper-evaluation"
        if _paper_evaluation_enabled(dataset)
        else "--no-paper-evaluation"
    )
    if visual_cache_mode != "live":
        if visual_cache_path is None:
            raise ValueError(f"{visual_cache_mode} requires a visual cache path")
        command += [
            "--visual-cache-mode", visual_cache_mode,
            "--visual-cache-path", str(visual_cache_path.resolve()),
        ]
    if mode == "fixed":
        command += ["--static-init-duration-s", str(float(initialization["duration_s"]))]
    alignment = dataset.get("evaluation_alignment_json")
    if alignment:
        command += ["--evaluation-alignment-json", str(Path(str(alignment)).expanduser().resolve())]
    motion_reference = dataset.get("motion_reference_csv")
    if not motion_reference:
        default_motion_reference = dataset_path / "motion_reference.csv"
        if default_motion_reference.is_file():
            motion_reference = default_motion_reference
    if motion_reference:
        command += ["--motion-reference-csv", str(Path(str(motion_reference)).expanduser().resolve())]
    if dry_run:
        command.append("--dry-run")
    # Deliberately do not pass --seq-to: every paper experiment uses all frames.
    return command


def build_cache_record_command(
    *,
    dataset: dict[str, Any],
    output_root: Path,
    cache_path: Path,
    model: Path,
    runtime: dict[str, Any],
    dry_run: bool,
) -> list[str]:
    command = build_command(
        dataset=dataset,
        # Backend/factor values are syntactically required by the shared CLI,
        # but record mode replaces them with the pure visual recorder config.
        method=MethodSpec("cache_record", "Visual cache", "two_state", "pace"),
        output_root=output_root / "_cache_record",
        model=model,
        runtime=runtime,
        dry_run=dry_run,
        visual_cache_mode="record",
        visual_cache_path=cache_path,
    )
    command = [
        value
        for value in command
        if value not in {"--paper-evaluation", "--no-paper-evaluation"}
    ]
    command.append("--no-paper-evaluation")
    return command


def _shared_cache_path(cache_root: Path, dataset: dict[str, Any]) -> Path:
    dataset_name = Path(str(dataset["path"])).expanduser().resolve().name
    return cache_root / dataset_name


def _result_dir(output_root: Path, dataset: dict[str, Any], method: MethodSpec) -> Path:
    return output_root / method.key / Path(str(dataset["path"])).expanduser().resolve().name


def _append_status(
    path: Path,
    *,
    scenario: str,
    method: str,
    status: str,
    started: str,
    finished: str,
    log: Path,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(
            (scenario, method, status, started, finished, log)
        )


def _completed_result(result_dir: Path, *, paper_evaluation: bool) -> bool:
    if paper_evaluation:
        return (result_dir / "paper_evaluation/metrics_summary.json").is_file()
    execution = result_dir / "run_execution.json"
    if not execution.is_file():
        return False
    try:
        payload = _read_json(execution)
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("return_code", -1)) == 0 and payload.get("status") == "ok"


def _publish_pure_macvo(
    cache_path: Path,
    output_root: Path,
    dataset: dict[str, Any],
) -> Path:
    dataset_name = Path(str(dataset["path"])).expanduser().resolve().name
    destination = output_root / "pure_macvo" / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "pure_macvo_poses_camera.csv": "poses_camera.csv",
        "pure_macvo_poses_imu.csv": "poses_imu.csv",
    }
    for source_name, destination_name in files.items():
        source = cache_path / source_name
        if not source.is_file():
            raise RuntimeError(f"visual cache is missing Pure MACVO output: {source}")
        shutil.copy2(source, destination / destination_name)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenario": str(dataset["scenario"]),
                "dataset": str(Path(str(dataset["path"])).expanduser().resolve()),
                "visual_cache": str(cache_path.resolve()),
                "camera_trajectory": "poses_camera.csv",
                "imu_center_trajectory": "poses_imu.csv",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _metric(summary: dict[str, Any], trajectory: str, *keys: str) -> Any:
    value: Any = summary["trajectories"][trajectory]
    for key in keys:
        value = value[key]
    return value


def _timing(summary: dict[str, Any], field: str, statistic: str) -> Any:
    return summary.get("timing", {}).get(field, {}).get(statistic)


def summarize(output_root: Path, manifest: dict[str, Any]) -> Path:
    rows: list[dict[str, Any]] = []
    bundles: dict[str, Any] = {}
    for dataset in manifest["datasets"]:
        scenario = str(dataset["scenario"])
        bundles[scenario] = {}
        pure_macvo = (
            output_root
            / "pure_macvo"
            / Path(str(dataset["path"])).expanduser().resolve().name
        )
        if (pure_macvo / "poses_imu.csv").is_file():
            bundles[scenario]["macvo"] = str(pure_macvo)
        for method in METHODS:
            result_dir = _result_dir(output_root, dataset, method)
            bundle = result_dir / "paper_evaluation"
            summary_path = bundle / "metrics_summary.json"
            if not summary_path.is_file():
                continue
            summary = _read_json(summary_path)
            bundles[scenario][method.key] = str(bundle)
            rows.append({
                "scenario": scenario,
                "method_key": method.key,
                "method": method.paper_name,
                "backend": method.backend,
                "visual_factor": method.visual_factor,
                "condition_velocity_factor": int(method.detector != "off"),
                "trajectory": "final",
                "state_count": _metric(summary, "final", "state_count"),
                "edge_count": _metric(summary, "final", "edge_count"),
                "xy_ape_rmse_m": _metric(summary, "final", "ape", "xy_m", "rmse"),
                "xy_ape_p95_m": _metric(summary, "final", "ape", "xy_m", "p95"),
                "translation_ape_rmse_m": _metric(summary, "final", "ape", "translation_m", "rmse"),
                "translation_ape_p95_m": _metric(summary, "final", "ape", "translation_m", "p95"),
                "rotation_ape_rmse_deg": _metric(summary, "final", "ape", "rotation_deg", "rmse"),
                "rotation_ape_p95_deg": _metric(summary, "final", "ape", "rotation_deg", "p95"),
                "translation_rpe_rmse_m": _metric(summary, "final", "rpe", "translation_m", "rmse"),
                "translation_rpe_p95_m": _metric(summary, "final", "rpe", "translation_m", "p95"),
                "rotation_rpe_rmse_deg": _metric(summary, "final", "rpe", "rotation_deg", "rmse"),
                "rotation_rpe_p95_deg": _metric(summary, "final", "rpe", "rotation_deg", "p95"),
                "frontend_median_ms": _timing(summary, "frontend_ms", "median"),
                "frontend_p95_ms": _timing(summary, "frontend_ms", "p95"),
                "factor_build_median_ms": _timing(summary, "factor_build_ms", "median"),
                "factor_build_p95_ms": _timing(summary, "factor_build_ms", "p95"),
                "backend_update_median_ms": _timing(summary, "backend_update_ms", "median"),
                "backend_update_p95_ms": _timing(summary, "backend_update_ms", "p95"),
                "total_compute_median_ms": _timing(summary, "total_compute_ms", "median"),
                "total_compute_p95_ms": _timing(summary, "total_compute_ms", "p95"),
                "end_to_end_frame_median_ms": _timing(summary, "end_to_end_frame_ms", "median"),
                "end_to_end_frame_p95_ms": _timing(summary, "end_to_end_frame_ms", "p95"),
                "convergence_rate": summary.get("timing", {}).get("convergence_rate"),
                "bundle": str(bundle),
            })
            if method.key == "pace_vio" and "macvo_raw" in summary.get("trajectories", {}):
                rows.append({
                    "scenario": scenario,
                    "method_key": "macvo",
                    "method": "MACVO",
                    "backend": "none",
                    "visual_factor": "raw",
                    "condition_velocity_factor": 0,
                    "trajectory": "macvo_raw",
                    "state_count": _metric(summary, "macvo_raw", "state_count"),
                    "edge_count": _metric(summary, "macvo_raw", "edge_count"),
                    "xy_ape_rmse_m": _metric(summary, "macvo_raw", "ape", "xy_m", "rmse"),
                    "xy_ape_p95_m": _metric(summary, "macvo_raw", "ape", "xy_m", "p95"),
                    "translation_ape_rmse_m": _metric(summary, "macvo_raw", "ape", "translation_m", "rmse"),
                    "translation_ape_p95_m": _metric(summary, "macvo_raw", "ape", "translation_m", "p95"),
                    "rotation_ape_rmse_deg": _metric(summary, "macvo_raw", "ape", "rotation_deg", "rmse"),
                    "rotation_ape_p95_deg": _metric(summary, "macvo_raw", "ape", "rotation_deg", "p95"),
                    "translation_rpe_rmse_m": _metric(summary, "macvo_raw", "rpe", "translation_m", "rmse"),
                    "translation_rpe_p95_m": _metric(summary, "macvo_raw", "rpe", "translation_m", "p95"),
                    "rotation_rpe_rmse_deg": _metric(summary, "macvo_raw", "rpe", "rotation_deg", "rmse"),
                    "rotation_rpe_p95_deg": _metric(summary, "macvo_raw", "rpe", "rotation_deg", "p95"),
                    "frontend_median_ms": None,
                    "frontend_p95_ms": None,
                    "factor_build_median_ms": None,
                    "factor_build_p95_ms": None,
                    "backend_update_median_ms": None,
                    "backend_update_p95_ms": None,
                    "total_compute_median_ms": None,
                    "total_compute_p95_ms": None,
                    "end_to_end_frame_median_ms": None,
                    "end_to_end_frame_p95_ms": None,
                    "convergence_rate": None,
                    "bundle": str(bundle),
                })
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "paper_run_summary.csv"
    fields = tuple(rows[0]) if rows else (
        "scenario", "method_key", "method", "backend", "visual_factor",
        "condition_velocity_factor", "trajectory", "state_count", "edge_count",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    source_map = {
        "schema_version": 1,
        "sequence_scope": "complete active sequence",
        "runs": bundles,
        "paper_comparisons": {
            "overall": ["macvo", "pace_two_state", "pace_isam2", "pace_vio"],
            "visual_factor": ["pose_isam2", "uvd_isam2", "pace_isam2"],
            "backend": ["pace_two_state", "pace_isam2"],
            "condition_factor": ["pace_isam2", "pace_vio"],
        },
    }
    (output_root / "paper_comparison_sources.json").write_text(
        json.dumps(source_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--visual-cache-policy",
        choices=("shared", "live"),
        default="shared",
        help=(
            "shared records one pure-MACVO cache per scene and replays it for every "
            "method; live independently runs the neural frontend for every method"
        ),
    )
    parser.add_argument(
        "--visual-cache-root",
        type=Path,
        help="Shared cache root; defaults to OUTPUT/visual_cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    output_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path(str(manifest.get("output", ROOT / "Results/paper_experiments"))).expanduser().resolve()
    )
    model = (
        args.model.expanduser().resolve()
        if args.model is not None
        else Path(str(manifest.get("model", ROOT / "Model/MACVO_FrontendCov.pth"))).expanduser().resolve()
    )
    runtime = manifest.get("runtime", {})
    cache_root = (
        args.visual_cache_root.expanduser().resolve()
        if args.visual_cache_root is not None
        else output_root / "visual_cache"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "experiment_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "method", "status", "started_utc", "finished_utc", "log"))

    failed = False
    for dataset in manifest["datasets"]:
        scenario = str(dataset["scenario"])
        shared_cache: Path | None = None
        cache_failed = False
        if args.visual_cache_policy == "shared":
            shared_cache = _shared_cache_path(cache_root, dataset)
            cache_log = output_root / "logs" / scenario / f"{VISUAL_CACHE_STAGE_KEY}.log"
            cache_log.parent.mkdir(parents=True, exist_ok=True)
            cache_started = datetime.now(timezone.utc).isoformat()
            cache_valid = False
            cache_status = "pending"
            if shared_cache.exists():
                try:
                    from Scripts.record_visual_factor_cache import validate_visual_cache_bundle

                    validate_visual_cache_bundle(shared_cache)
                    cache_valid = True
                    cache_status = "skipped_complete"
                    print(
                        f"[{scenario}] Visual cache: reuse {shared_cache}",
                        flush=True,
                    )
                except Exception as error:
                    print(
                        f"[{scenario}] Existing visual cache is invalid: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    cache_failed = True
                    failed = True
                    cache_status = "failed:invalid_cache"
            if not cache_valid and not cache_failed:
                record_command = build_cache_record_command(
                    dataset=dataset,
                    output_root=output_root,
                    cache_path=shared_cache,
                    model=model,
                    runtime=runtime,
                    dry_run=args.dry_run,
                )
                print(
                    f"[{scenario}] Visual cache: {' '.join(record_command)}",
                    flush=True,
                )
                _append_status(
                    status_path,
                    scenario=scenario,
                    method=VISUAL_CACHE_STAGE_KEY,
                    status="running",
                    started=cache_started,
                    finished="",
                    log=cache_log,
                )
                if not args.dry_run:
                    with cache_log.open("w", encoding="utf-8") as log:
                        process = subprocess.run(
                            record_command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                    if process.returncode != 0:
                        cache_failed = True
                        failed = True
                        cache_status = f"failed:{process.returncode}"
                        print(f"Visual cache recording failed; inspect {cache_log}", file=sys.stderr)
                    else:
                        from Scripts.record_visual_factor_cache import validate_visual_cache_bundle

                        validate_visual_cache_bundle(shared_cache)
                        cache_valid = True
                        cache_status = "ok"
                else:
                    cache_status = "ok"
            if cache_valid and not args.dry_run:
                try:
                    pure_result = _publish_pure_macvo(shared_cache, output_root, dataset)
                    print(f"[{scenario}] Pure MACVO: {pure_result}", flush=True)
                except Exception as error:
                    print(
                        f"[{scenario}] Pure MACVO publication failed: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    cache_failed = True
                    failed = True
                    cache_status = "failed:pure_macvo"
            cache_finished = datetime.now(timezone.utc).isoformat()
            _append_status(
                status_path,
                scenario=scenario,
                method=VISUAL_CACHE_STAGE_KEY,
                status=cache_status,
                started=cache_started,
                finished=cache_finished,
                log=cache_log,
            )
            if cache_failed:
                if not args.continue_on_error:
                    return 1
                continue
        for method in METHODS:
            result_dir = _result_dir(output_root, dataset, method)
            paper_evaluation = _paper_evaluation_enabled(dataset)
            completed = _completed_result(
                result_dir,
                paper_evaluation=paper_evaluation,
            )
            log_path = output_root / "logs" / scenario / f"{method.key}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            method_failed = False
            if completed and not args.no_resume and not args.dry_run:
                status = "skipped_complete"
                started = finished = datetime.now(timezone.utc).isoformat()
            else:
                command = build_command(
                    dataset=dataset,
                    method=method,
                    output_root=output_root,
                    model=model,
                    runtime=runtime,
                    dry_run=args.dry_run,
                    visual_cache_mode=(
                        "replay" if shared_cache is not None else "live"
                    ),
                    visual_cache_path=shared_cache,
                )
                started = datetime.now(timezone.utc).isoformat()
                print(f"[{scenario}] {method.paper_name}: {' '.join(command)}", flush=True)
                _append_status(
                    status_path,
                    scenario=scenario,
                    method=method.key,
                    status="running",
                    started=started,
                    finished="",
                    log=log_path,
                )
                if args.dry_run:
                    process = subprocess.CompletedProcess(command, 0)
                else:
                    with log_path.open("w", encoding="utf-8") as log:
                        process = subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                finished = datetime.now(timezone.utc).isoformat()
                status = "ok" if process.returncode == 0 else f"failed:{process.returncode}"
                if process.returncode != 0:
                    method_failed = True
                    failed = True
            _append_status(
                status_path,
                scenario=scenario,
                method=method.key,
                status=status,
                started=started,
                finished=finished,
                log=log_path,
            )
            if method_failed and not args.continue_on_error:
                print(f"Experiment failed; inspect {log_path}", file=sys.stderr)
                return 1
    if not args.dry_run:
        summary_path = summarize(output_root, manifest)
        print(f"Paper summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

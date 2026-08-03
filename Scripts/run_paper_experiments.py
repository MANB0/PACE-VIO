#!/usr/bin/env python3
"""Run the complete PACE-VIO paper experiment matrix.

Only dataset paths and initialization policies live in the manifest.  Method
definitions are fixed here so every scene uses the same full-sequence protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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
    return manifest


def build_command(
    *,
    dataset: dict[str, Any],
    method: MethodSpec,
    output_root: Path,
    model: Path,
    runtime: dict[str, Any],
    dry_run: bool,
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
        "--paper-evaluation",
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


def _result_dir(output_root: Path, dataset: dict[str, Any], method: MethodSpec) -> Path:
    return output_root / method.key / Path(str(dataset["path"])).expanduser().resolve().name


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
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "experiment_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "method", "status", "started_utc", "finished_utc", "log"))

    failed = False
    for dataset in manifest["datasets"]:
        for method in METHODS:
            result_dir = _result_dir(output_root, dataset, method)
            completed = result_dir / "paper_evaluation/metrics_summary.json"
            log_path = output_root / "logs" / str(dataset["scenario"]) / f"{method.key}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if completed.is_file() and not args.no_resume and not args.dry_run:
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
                )
                started = datetime.now(timezone.utc).isoformat()
                print(f"[{dataset['scenario']}] {method.paper_name}: {' '.join(command)}", flush=True)
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
                    failed = True
            with status_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    (dataset["scenario"], method.key, status, started, finished, log_path)
                )
            if failed and not args.continue_on_error:
                print(f"Experiment failed; inspect {log_path}", file=sys.stderr)
                return 1
    if not args.dry_run:
        summary_path = summarize(output_root, manifest)
        print(f"Paper summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

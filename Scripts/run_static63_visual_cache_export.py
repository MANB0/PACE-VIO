#!/usr/bin/env python3
"""Run pure-MACVO source passes and export visual-factor caches for static63 scenes.

This batch is intentionally narrow: it only produces the frontend/optimizer
visual inputs used by cache replay.  It does not run IMU replay, VIO variants,
or method comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Scripts.export_visual_factor_cache import (  # noqa: E402
    ExportVisualFactorCacheError,
    export_result_to_visual_cache,
)
from Scripts.run_vio_imu_prior_mode_grid import (  # noqa: E402
    RunSpec,
    VARIANTS,
    flatten_nested,
    has_completed_run,
    make_odom_cfg,
    make_seq_cfg,
    run_one,
    sanity_check,
)
from Scripts.run_visual_factor_cache_batch import switch_dashboard  # noqa: E402


WORKDIR = Path("/home/admin1/macvo-dev")
BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_clear_truth_paths_20260713_static63_variants")
DEFAULT_SOURCE_ROOT = WORKDIR / "Results" / "visual_factor_cache_static63_unique_source_20260713"
DEFAULT_CACHE_ROOT = WORKDIR / "VisualCache" / "static63_unique_visual_20260713"
DEFAULT_CONTROL_ROOT = WORKDIR / "Results" / "visual_factor_cache_static63_control_20260713"
DEFAULT_LOG = WORKDIR / "logs" / "visual_factor_cache_static63_three_geometry_20260713.log"
UNIQUE_VISUAL_SCENES = (
    "clear_circle_truth_normal_noise",
    "clear_stop_turn_rectangle_truth_normal_noise",
    "clear_straight_truth_normal_noise",
)
ALL_VARIANT_SCENES = (
    "clear_circle_truth_normal_noise",
    "clear_circle_truth_bias_no_noise",
    "clear_circle_truth_noise_no_bias",
    "clear_circle_truth_no_noise_no_bias",
    "clear_stop_turn_rectangle_truth_normal_noise",
    "clear_stop_turn_rectangle_truth_bias_no_noise",
    "clear_stop_turn_rectangle_truth_noise_no_bias",
    "clear_stop_turn_rectangle_truth_no_noise_no_bias",
    "clear_straight_truth_normal_noise",
    "clear_straight_truth_bias_no_noise",
    "clear_straight_truth_noise_no_bias",
    "clear_straight_truth_no_noise_no_bias",
)


def _scene_roots(scenes: list[str]) -> dict[str, Path]:
    return {scene: BATCH_ROOT / scene for scene in scenes}


def _build_source_specs(scenes: list[str], source_root: Path) -> list[RunSpec]:
    pure = VARIANTS["pure_macvo"]
    roots = _scene_roots(scenes)
    return [
        RunSpec(
            trial=1,
            scene=scene,
            scene_root=roots[scene],
            variant=pure,
            result_dir=source_root / "trial_1" / "pure_macvo" / scene,
        )
        for scene in scenes
    ]


def _append_stage(control_root: Path, row: dict[str, object]) -> None:
    control_root.mkdir(parents=True, exist_ok=True)
    path = control_root / "cache_progress.csv"
    exists = path.exists()
    fieldnames = [
        "stage",
        "scene",
        "status",
        "runtime_s",
        "source_dir",
        "cache_dir",
        "message",
    ]
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_dashboard_manifest(control_root: Path, specs: list[RunSpec], seq_to: int | None) -> None:
    control_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial",
        "scene",
        "variant",
        "scene_root",
        "result_dir",
        "seq_to",
        "created_at",
    ]
    with (control_root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "trial": spec.trial,
                    "scene": spec.scene,
                    "variant": "pure_macvo_cache_export",
                    "scene_root": spec.scene_root,
                    "result_dir": spec.result_dir,
                    "seq_to": "" if seq_to is None else int(seq_to),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
    (control_root / "progress.csv").write_text(
        "trial,scene,variant,status,return_code,runtime_s,result_dir\n",
        encoding="utf-8",
    )


def _append_dashboard_progress(
    control_root: Path,
    spec: RunSpec,
    *,
    status: str,
    return_code: int | str = "",
    runtime_s: float | str = "",
) -> None:
    with (control_root / "progress.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["trial", "scene", "variant", "status", "return_code", "runtime_s", "result_dir"],
        )
        writer.writerow(
            {
                "trial": spec.trial,
                "scene": spec.scene,
                "variant": "pure_macvo_cache_export",
                "status": status,
                "return_code": return_code,
                "runtime_s": runtime_s,
                "result_dir": spec.result_dir,
            }
        )


def _cache_complete(cache_dir: Path, scene: str, scene_root: Path) -> bool:
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("scene") != scene:
        return False
    source = data.get("source")
    if not isinstance(source, dict):
        return False
    if str(source.get("dataset", "")) != str(scene_root.resolve()):
        return False
    frame_count = data.get("frame_count")
    pairs = data.get("pairs")
    if not isinstance(frame_count, int) or not isinstance(pairs, list):
        return False
    return frame_count > 1 and len(pairs) == frame_count - 1


def _export_cache(
    *,
    spec: RunSpec,
    cache_root: Path,
    control_root: Path,
    force_export: bool,
    validation_atol: float,
) -> int:
    cache_dir = cache_root / spec.scene
    if not force_export and _cache_complete(cache_dir, spec.scene, spec.scene_root):
        print(f"  -> SKIP existing cache: {cache_dir}")
        _append_stage(
            control_root,
            {
                "stage": "export",
                "scene": spec.scene,
                "status": "skipped",
                "source_dir": spec.result_dir,
                "cache_dir": cache_dir,
                "message": "existing complete cache",
            },
        )
        return 0

    started = time.time()
    try:
        export_result_to_visual_cache(
            spec.result_dir,
            cache_dir,
            spec.scene,
            spec.scene_root,
            validation_atol=validation_atol,
        )
    except ExportVisualFactorCacheError as exc:
        elapsed = time.time() - started
        print(f"  -> EXPORT FAILED ({elapsed:.1f}s): {exc}")
        _append_stage(
            control_root,
            {
                "stage": "export",
                "scene": spec.scene,
                "status": "failed",
                "runtime_s": f"{elapsed:.1f}",
                "source_dir": spec.result_dir,
                "cache_dir": cache_dir,
                "message": str(exc),
            },
        )
        return 1

    elapsed = time.time() - started
    print(f"  -> EXPORT OK ({elapsed:.1f}s): {cache_dir}")
    _append_stage(
        control_root,
        {
            "stage": "export",
            "scene": spec.scene,
            "status": "ok",
            "runtime_s": f"{elapsed:.1f}",
            "source_dir": spec.result_dir,
            "cache_dir": cache_dir,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--all-variant-scenes",
        action="store_true",
        help="Run all 8 IMU-variant scene directories instead of the two unique visual trajectories.",
    )
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--force-source", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--validation-atol", type=float, default=1e-3)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scenes is not None:
        scenes = list(args.scenes)
    elif args.all_variant_scenes:
        scenes = list(ALL_VARIANT_SCENES)
    else:
        scenes = list(UNIQUE_VISUAL_SCENES)
    specs = _build_source_specs(scenes, args.source_root)

    print("=" * 78)
    print("  Static63 visual-factor cache export")
    print(f"  Batch root: {BATCH_ROOT}")
    print(f"  Source:     {args.source_root}")
    print(f"  Cache:      {args.cache_root}")
    print(f"  Control:    {args.control_root}")
    print(f"  Scenes:     {len(specs)}")
    print(f"  Seq-to:     {args.seq_to if args.seq_to is not None else 'full sequence'}")
    print("=" * 78)

    if not sanity_check(specs):
        return 2

    for spec in specs:
        print(f"  - {spec.scene}: {spec.scene_root}")

    if args.dry_run:
        return 0

    _write_dashboard_manifest(args.control_root, specs, args.seq_to)
    if not args.no_dashboard:
        switch_dashboard(args.control_root, DEFAULT_LOG, port=int(args.dashboard_port))

    failures = 0
    with tempfile.TemporaryDirectory(prefix="static63_cache_cfg_") as tmp:
        tmpdir = Path(tmp)
        odom_cfg = make_odom_cfg(VARIANTS["pure_macvo"], tmpdir, autodiff=False)

        for idx, spec in enumerate(specs, start=1):
            print(f"\n[{idx}/{len(specs)}] source: {spec.scene}")
            started = time.monotonic()
            _append_dashboard_progress(args.control_root, spec, status="running")
            if args.force_source and spec.result_dir.exists():
                print(f"  -> force-source requested, existing output will be reused only after MACVO overwrites files: {spec.result_dir}")
            seq_cfg = make_seq_cfg(spec, tmpdir)
            if not args.force_source and has_completed_run(spec):
                flatten_nested(spec.result_dir)
                print(f"  -> SKIP existing source: {spec.result_dir}")
                _append_stage(
                    args.control_root,
                    {
                        "stage": "source",
                        "scene": spec.scene,
                        "status": "skipped",
                        "source_dir": spec.result_dir,
                        "message": "existing complete source",
                    },
                )
                source_rc = 0
            else:
                source_rc = run_one(
                    spec,
                    odom_cfg,
                    seq_cfg,
                    args.source_root,
                    int(args.timeout),
                    args.seq_to,
                )
                _append_stage(
                    args.control_root,
                    {
                        "stage": "source",
                        "scene": spec.scene,
                        "status": "ok" if source_rc == 0 else "failed",
                        "source_dir": spec.result_dir,
                        "message": f"return_code={source_rc}",
                    },
                )
            if source_rc != 0:
                failures += 1
                _append_dashboard_progress(
                    args.control_root,
                    spec,
                    status="failed",
                    return_code=source_rc,
                    runtime_s=f"{time.monotonic() - started:.1f}",
                )
                continue

            print(f"[{idx}/{len(specs)}] export: {spec.scene}")
            export_rc = _export_cache(
                spec=spec,
                cache_root=args.cache_root,
                control_root=args.control_root,
                force_export=bool(args.force_export),
                validation_atol=float(args.validation_atol),
            )
            failures += export_rc
            _append_dashboard_progress(
                args.control_root,
                spec,
                status="ok" if export_rc == 0 else "failed",
                return_code=export_rc,
                runtime_s=f"{time.monotonic() - started:.1f}",
            )

    print("\nDone.")
    print(f"Source root: {args.source_root}")
    print(f"Cache root:  {args.cache_root}")
    print(f"Progress:    {args.control_root / 'cache_progress.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

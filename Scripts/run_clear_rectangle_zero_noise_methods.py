#!/usr/bin/env python3
"""Run pure MACVO and full preintegrated VIO on the zero-noise rectangle scene."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import run_vio_imu_prior_mode_grid as grid


SCENE_ROOTS = {
    "clear_rectangle_zero_noise": Path(
        "/mnt/e/文档/holoocean/code/recordings/"
        "batch_clear_rectangle_zero_noise_20260704/clear_rectangle_path"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=WORKDIR / "Results" / "clear_rectangle_zero_noise_methods_20260704",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["pure_macvo", "vio_preintegrated_full"],
        help="Variants from Scripts/run_vio_imu_prior_mode_grid.py.",
    )
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=grid.RUN_TIMEOUT_S)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid.SCENE_ROOTS.update(SCENE_ROOTS)
    scenes = list(SCENE_ROOTS)
    specs = grid.build_specs(
        scenes=scenes,
        variants=list(args.variants),
        trials=1,
        result_root=args.result_root,
    )

    print("=" * 78)
    print("  Clear-rectangle zero-noise methods run")
    print(f"  Results:  {args.result_root}")
    print(f"  Scenes:   {', '.join(scenes)}")
    print(f"  Variants: {', '.join(args.variants)}")
    print(f"  Seq-to:   {args.seq_to if args.seq_to is not None else 'full sequence'}")
    print(f"  Jobs:     {max(1, int(args.jobs))}")
    print(f"  Runs:     {len(specs)}")
    print("=" * 78)

    if not grid.sanity_check(specs):
        return 1

    if args.dry_run:
        print("\nDRY RUN - no MACVO process will be started.")
        for idx, spec in enumerate(specs, start=1):
            marker = "SKIP" if grid.has_completed_run(spec) else "RUN"
            print(
                f"  [{idx:02d}/{len(specs):02d}] {marker} "
                f"scene={spec.scene} variant={spec.variant.name} -> {spec.result_dir}"
            )
        return 0

    args.result_root.mkdir(parents=True, exist_ok=True)
    grid.write_manifest_guarded(
        args.result_root,
        specs,
        args.seq_to,
        autodiff=False,
        overwrite=bool(args.overwrite_manifest),
    )
    print(f"\nManifest: {args.result_root / 'run_manifest.csv'}")

    tmpdir = Path(tempfile.mkdtemp(prefix="clear_rectangle_zero_noise_"))
    print(f"Temp config dir: {tmpdir}")
    odom_cfgs = {
        variant: grid.make_odom_cfg(grid.VARIANTS[variant], tmpdir, autodiff=False)
        for variant in args.variants
    }

    started = time.time()
    failures = grid.execute_run_schedule(
        specs,
        odom_cfgs,
        tmpdir,
        args.result_root,
        timeout_s=int(args.timeout),
        seq_to=args.seq_to,
        jobs=int(args.jobs),
    )

    elapsed = time.time() - started
    print("\n" + "=" * 78)
    print(f"  Attempted schedule in {elapsed / 60:.1f} min")
    print(f"  Results:  {args.result_root}")
    print(f"  Progress: {args.result_root / 'progress.csv'}")
    if failures:
        print(f"  Failures: {len(failures)}")
        for spec, rc in failures:
            print(f"    - scene={spec.scene} variant={spec.variant.name} rc={rc}")
    else:
        print("  No failed return codes in this run.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

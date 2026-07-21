#!/usr/bin/env python3
"""Run the /home/admin1/macvo MACVO.py baseline on HoloOcean scenes.

This script intentionally executes the original workspace entrypoint:
    /home/admin1/macvo/MACVO.py

Only temporary sequence YAMLs are generated; tracked configs are not mutated.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
import statistics
from datetime import datetime
from pathlib import Path

import yaml

DEV_ROOT = Path("/home/admin1/macvo-dev")
MACVO_ROOT = Path("/home/admin1/macvo")
BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260508_134929")
ODOM_CFG = MACVO_ROOT / "Config/Experiment/MACVO/MACVO_Fast.yaml"

sys.path.insert(0, str(DEV_ROOT))
from Scripts.eval_qa_vif import evaluate_trajectory, evaluate_trajectory_direct  # noqa: E402


def write_sequence_config(scene_root: Path, tmpdir: Path) -> Path:
    cfg = {
        "type": "GeneralStereo",
        "name": "holoocean_macvo_original",
        "args": {
            "root": str(scene_root),
            "camera": {
                "fx": 320.0,
                "fy": 320.0,
                "cx": 320.0,
                "cy": 240.0,
            },
            "bl": 0.225,
            "format": "png",
            "pose_output_frame": "NWU",
        },
    }
    out = tmpdir / f"seq_{scene_root.name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def latest_result(result_root: Path, scene: str) -> Path:
    project_root = result_root / "MACVO-Fast@holoocean_macvo_original"
    candidates = sorted(
        project_root.glob(f"*batch_20260508_134929_{scene}/poses.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            project_root.glob(f"*batch_20260508_134929_{scene}/poses.npy"),
            key=lambda p: p.stat().st_mtime,
        )
    if not candidates:
        raise FileNotFoundError(f"No poses.csv or poses.npy found for {scene} under {project_root}")
    return candidates[-1]


def run_one(scene_root: Path, seq_cfg: Path, result_root: Path, seq_to: int | None, timeout_s: int) -> dict:
    cmd = [
        sys.executable,
        "MACVO.py",
        "--odom",
        str(ODOM_CFG),
        "--data",
        str(seq_cfg),
        "--resultRoot",
        str(result_root),
        "--noRR",
    ]
    if seq_to is not None:
        cmd.extend(["--seq_to", str(seq_to)])

    started = time.time()
    proc = subprocess.run(cmd, cwd=MACVO_ROOT, text=True, capture_output=True, timeout=timeout_s)
    elapsed = time.time() - started
    result = {
        "scene": scene_root.name,
        "ok": proc.returncode == 0,
        "elapsed_s": round(elapsed, 2),
        "cmd": cmd,
    }
    if proc.returncode != 0:
        result["stdout_tail"] = proc.stdout[-4000:]
        result["stderr_tail"] = proc.stderr[-4000:]
        return result

    poses = latest_result(result_root, scene_root.name)
    metrics = evaluate_trajectory_direct(
        poses,
        scene_root / "ref_pose.csv",
        f"macvo_original_direct_{scene_root.name}",
        vo_frame="auto",
        eval_frame="NWU",
    )
    result["poses"] = str(poses)
    result["metrics"] = metrics
    result["ate_rmse"] = metrics["ate"]["ate_rmse"]
    result["rpe_trans_rmse"] = metrics["rpe"]["rpe_trans_rmse"]
    result["scale"] = metrics["ate"].get("scale")
    result["alignment"] = metrics.get("alignment", metrics["ate"].get("alignment"))
    result["matching"] = metrics.get("matching")
    return result


def summarize(results: dict[str, dict]) -> dict:
    ok_items = [r for r in results.values() if r.get("ok")]
    ates = [float(r["ate_rmse"]) for r in ok_items if math.isfinite(float(r["ate_rmse"]))]
    rpes = [float(r["rpe_trans_rmse"]) for r in ok_items if math.isfinite(float(r["rpe_trans_rmse"]))]
    if not ates:
        return {"num_ok": 0}
    return {
        "num_ok": len(ates),
        "mean_ate": statistics.mean(ates),
        "median_ate": statistics.median(ates),
        "mean_rpe_trans": statistics.mean(rpes) if rpes else None,
        "median_rpe_trans": statistics.median(rpes) if rpes else None,
        "max_ate": max(ates),
        "min_ate": min(ates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run /home/admin1/macvo full HoloOcean baseline")
    parser.add_argument("--batch", type=Path, default=BATCH_ROOT)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--seq_to", type=int, default=None)
    parser.add_argument("--timeout_s", type=int, default=12000)
    parser.add_argument(
        "--result_root",
        type=Path,
        default=MACVO_ROOT / "Results/original_full_baseline_20260512",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEV_ROOT / "Results/macvo_original_full_baseline_20260512.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = sorted([p for p in args.batch.iterdir() if p.is_dir()])
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [p for p in scenes if p.name in wanted]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "macvo_root": str(MACVO_ROOT),
        "odom_cfg": str(ODOM_CFG),
        "batch": str(args.batch),
        "result_root": str(args.result_root),
        "evaluation": {
            "alignment": "none",
            "metric": "direct_position_rmse",
            "eval_frame": "NWU",
            "vo_frame": "auto",
            "matching": "exact timestamp when available, otherwise index truncation",
        },
        "results": {},
        "summary": {},
    }

    with tempfile.TemporaryDirectory(prefix="macvo_original_baseline_") as td:
        tmpdir = Path(td)
        for scene_root in scenes:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] scene={scene_root.name}")
            seq_cfg = write_sequence_config(scene_root, tmpdir)
            result = run_one(scene_root, seq_cfg, args.result_root, args.seq_to, args.timeout_s)
            payload["results"][scene_root.name] = result
            payload["summary"] = summarize(payload["results"])
            args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            if result.get("ok"):
                print(
                    f"  ATE={result['ate_rmse']:.4f} "
                    f"RPE_t={result['rpe_trans_rmse']:.5f} "
                    f"alignment={result['alignment']} "
                    f"match={result.get('matching', {}).get('mode')} "
                    f"elapsed={result['elapsed_s']:.1f}s"
                )
            else:
                print(f"  FAILED elapsed={result['elapsed_s']:.1f}s")
                print(result.get("stderr_tail", "")[-1000:])

    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved original MACVO baseline to {args.output}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

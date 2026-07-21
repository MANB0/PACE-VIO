#!/usr/bin/env python3
"""Batch benchmark MACVO variants on the seven HoloOcean scenes.

The script writes temporary sequence/odometry configs and never mutates the
tracked configs under Config/.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

WORKDIR = Path("/home/admin1/macvo-dev")
BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260508_134929")
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"
BASELINE_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_Fast.yaml"
IMU_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_Fast_IMU.yaml"

sys.path.insert(0, str(WORKDIR))
from Scripts.eval_qa_vif import evaluate_trajectory, evaluate_trajectory_direct  # noqa: E402
from Utility.Config import IncludeLoader  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def latest_result(project: str, scene: str) -> Path:
    result_root = WORKDIR / "Results" / project
    candidates = sorted(result_root.glob(f"*batch_20260508_134929_{scene}/poses.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(result_root.glob(f"*batch_20260508_134929_{scene}/poses.npy"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(result_root.glob("*/poses.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(result_root.glob("*/poses.npy"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No poses.csv or poses.npy found under {result_root}")
    return candidates[-1]


def make_sequence_config(scene_root: Path, tmpdir: Path) -> Path:
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(scene_root)
    out = tmpdir / f"seq_{scene_root.name}.yaml"
    write_yaml(out, cfg)
    return out


def make_imu_config(candidate: str, tmpdir: Path) -> Path:
    cfg = load_yaml(IMU_CFG)
    odom = cfg["Odometry"]
    opt = odom["optimizer"]["args"]

    odom.setdefault("motion", {})
    if odom["motion"].get("args") is None:
        odom["motion"]["args"] = {}
    if odom["motion"].get("type") != "StaticMotionModel":
        odom["motion"]["args"]["device"] = cfg.get("Common", {}).get("device", "cuda")
    opt["imu_rot_prior"] = False

    if candidate == "no_post":
        opt["post_imu_fusion_enable"] = False
        opt["post_imu_fusion_mode"] = "none"
    elif candidate == "uraf":
        # Reproduce the legacy full-sequence "URAF(lambda=0.5)" table.
        #
        # The saved strong runs were labelled URAF, but their config.yaml files
        # had post_imu_fusion_enable=false.  Keep this candidate bound to that
        # historical behavior so benchmark results stay comparable.
        odom["args"]["imu_pose_fusion_enable"] = False
        odom["args"]["imu_pose_fusion_alpha"] = 0.5
        odom["args"]["imu_legacy_gyro_prior_enable"] = False
        opt["autodiff"] = True
        opt["imu_rot_prior"] = False
        opt["imu_trans_prior_scale"] = 0.05
        opt["post_imu_fusion_enable"] = False
        opt["post_imu_fusion_prepose_enable"] = False
        opt["post_imu_fusion_mode"] = "uraf"
        opt["uraf_lambda"] = 0.5
        opt["post_imu_fusion_imu_trans_enable"] = False
        opt["post_imu_fusion_imu_trans_z_only"] = False
    elif candidate == "post_uraf":
        odom["args"]["imu_pose_fusion_enable"] = False
        odom["args"]["imu_pose_fusion_alpha"] = 0.5
        opt["post_imu_fusion_enable"] = True
        opt["post_imu_fusion_prepose_enable"] = False
        opt["post_imu_fusion_mode"] = "uraf"
        opt["uraf_lambda"] = 0.5
        opt["post_imu_fusion_imu_trans_enable"] = False
        opt["post_imu_fusion_imu_trans_z_only"] = False
    elif candidate == "csga":
        # Coordinate-Safe Gyro Assist: after the HoloOcean FLU→NED fix, use only
        # IMU rotation unless a separate experiment proves acceleration-based
        # translation helps. This prevents coordinate/initial-velocity mistakes
        # from creating misleadingly good scores.
        odom["args"]["imu_pose_fusion_enable"] = False
        odom["args"]["imu_trans_prior_enable"] = False
        odom["args"]["imu_legacy_gyro_prior_enable"] = False
        opt["imu_rot_prior"] = False
        opt["post_imu_fusion_enable"] = True
        opt["post_imu_fusion_prepose_enable"] = False
        opt["post_imu_fusion_mode"] = "dua"
        opt["post_imu_fusion_imu_trans_enable"] = False
        opt["post_imu_fusion_imu_rot_enable"] = True
        opt["post_imu_fusion_yaw_residual_enable"] = True
        opt["post_imu_fusion_yaw_residual_gain"] = 0.25
        opt["post_imu_fusion_yaw_residual_max_corr"] = 0.04
        opt["post_imu_fusion_yaw_trans_couple_enable"] = False
        opt["post_imu_fusion_xy_turn_residual_enable"] = False
    else:
        opt["post_imu_fusion_enable"] = True
        opt["post_imu_fusion_mode"] = candidate

    out = tmpdir / f"odom_{candidate}.yaml"
    write_yaml(out, cfg)
    return out


def run_macvo(odom_cfg: Path, seq_cfg: Path, seq_to: int | None, timeout_s: int) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "MACVO.py",
        "--odom",
        str(odom_cfg),
        "--data",
        str(seq_cfg),
    ]
    if seq_to is not None:
        cmd.extend(["--seq_to", str(seq_to)])
    return subprocess.run(cmd, cwd=WORKDIR, text=True, capture_output=True, timeout=timeout_s)


def run_one(candidate: str, scene_root: Path, seq_cfg: Path, tmpdir: Path, seq_to: int | None, timeout_s: int, alignment: str = "umeyama") -> dict:
    if candidate == "baseline":
        odom_cfg = BASELINE_CFG
        project = "MACVO-Fast@holoocean_imu"
    else:
        odom_cfg = make_imu_config(candidate, tmpdir)
        project = "MACVO-Fast-IMU@holoocean_imu"

    started = time.time()
    proc = run_macvo(odom_cfg, seq_cfg, seq_to, timeout_s)
    elapsed = time.time() - started
    result = {
        "candidate": candidate,
        "scene": scene_root.name,
        "ok": proc.returncode == 0,
        "elapsed_s": round(elapsed, 2),
    }
    if proc.returncode != 0:
        result["stderr_tail"] = proc.stderr[-3000:]
        result["stdout_tail"] = proc.stdout[-3000:]
        return result

    poses = latest_result(project, scene_root.name)
    if alignment == "direct":
        metrics = evaluate_trajectory_direct(poses, scene_root / "ref_pose.csv", f"{candidate}_{scene_root.name}")
    else:
        metrics = evaluate_trajectory(poses, scene_root / "ref_pose.csv", f"{candidate}_{scene_root.name}")
    result["poses"] = str(poses)
    result["metrics"] = metrics
    result["ate_rmse"] = metrics["ate"]["ate_rmse"]
    result["rpe_trans_rmse"] = metrics["rpe"]["rpe_trans_rmse"]
    result["scale"] = metrics["ate"].get("scale", None)
    result["alignment"] = alignment
    return result


def summarize(results: dict[str, dict[str, dict]]) -> dict:
    summary: dict[str, dict] = {}
    baselines = {
        scene: scene_results["baseline"]["ate_rmse"]
        for scene, scene_results in results.items()
        if "baseline" in scene_results and scene_results["baseline"].get("ok")
    }
    candidates = sorted({cand for scene_results in results.values() for cand in scene_results})
    for cand in candidates:
        ratios = []
        ates = []
        max_regression = 0.0
        for scene, scene_results in results.items():
            item = scene_results.get(cand)
            if not item or not item.get("ok"):
                continue
            ate = float(item["ate_rmse"])
            if math.isnan(ate) or math.isinf(ate):
                continue
            ates.append(ate)
            if scene in baselines:
                ratio = ate / max(float(baselines[scene]), 1e-9)
                ratios.append(ratio)
                max_regression = max(max_regression, ratio - 1.0)
        if ates:
            summary[cand] = {
                "mean_ate": sum(ates) / len(ates),
                "mean_ratio_to_baseline": sum(ratios) / len(ratios) if ratios else None,
                "max_regression_pct": max_regression * 100.0,
                "num_ok": len(ates),
            }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MACVO HoloOcean variants")
    parser.add_argument("--batch", type=Path, default=BATCH_ROOT)
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names; default all scenes")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["baseline", "uraf", "dua"],
        choices=["baseline", "no_post", "uraf", "post_uraf", "qavif", "padf", "bagf", "dua", "csga"],
    )
    parser.add_argument("--seq_to", type=int, default=None, help="Optional frame crop")
    parser.add_argument("--timeout_s", type=int, default=5400)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--alignment",
        type=str,
        default="umeyama",
        choices=["umeyama", "direct"],
        help="ATE alignment mode: umeyama=Sim(3) aligned, direct=no alignment (default: umeyama)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = sorted([p for p in args.batch.iterdir() if p.is_dir()])
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [p for p in scenes if p.name in wanted]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or WORKDIR / "Results" / f"holoocean_bench_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict]] = {}
    with tempfile.TemporaryDirectory(prefix="macvo_bench_") as td:
        tmpdir = Path(td)
        for scene_root in scenes:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] scene={scene_root.name}")
            seq_cfg = make_sequence_config(scene_root, tmpdir)
            all_results[scene_root.name] = {}
            for candidate in args.candidates:
                print(f"  running {candidate}...")
                result = run_one(candidate, scene_root, seq_cfg, tmpdir, args.seq_to, args.timeout_s, args.alignment)
                all_results[scene_root.name][candidate] = result
                if result.get("ok"):
                    scale_str = f" scale={result['scale']:.4f}" if result.get("scale") is not None else ""
                    print(f"    ATE={result['ate_rmse']:.4f}{scale_str} elapsed={result['elapsed_s']:.1f}s")
                else:
                    print(f"    FAILED elapsed={result['elapsed_s']:.1f}s")
                payload = {"results": all_results, "summary": summarize(all_results)}
                output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {"results": all_results, "summary": summarize(all_results)}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved benchmark results to {output}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

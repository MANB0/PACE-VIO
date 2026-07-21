#!/usr/bin/env python3
"""
Comprehensive Ablation Study for MACVO Visual-Inertial Fusion.

Evaluates multiple configurations against reference trajectory:
  1. Baseline (no IMU, StaticMotionModel, no prior, no post-fusion)
  2. QA-VIF (post-fusion only, rotation-only, conservative floors)
  3. PADF (post-fusion only, per-axis decoupled)
  4. In-Graph Prior (scale-corrected, no post-fusion)
  5. In-Graph + PADF (combined)
  6. In-Graph + PADF + IMU Translation

Usage:
    python eval_ablation.py --data_root <path> --ref_pose <path> [--datasets D1 D2 D3]
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np


# ── Dataset definitions ──────────────────────────────────────────────────────
DATASETS = {
    "D1": "/mnt/e/文档/holoocean/code/recordings/20260507_162841",
    "D2": "/mnt/e/文档/holoocean/code/recordings/20260507_183026",
    "D3": "/mnt/e/文档/holoocean/code/recordings/20260507_192710",
    "D4": "/mnt/e/文档/holoocean/code/recordings/20260507_212316",
}

# ── Configuration templates ──────────────────────────────────────────────────
BASE_CONFIG = "Config/Experiment/MACVO/MACVO_Fast.yaml"
IMU_CONFIG = "Config/Experiment/MACVO/MACVO_Fast_IMU.yaml"
SEQ_CONFIG = "Config/Sequence/holoocean_imu.yaml"


def update_seq_root(root: str) -> None:
    """Update the sequence config root path."""
    import yaml
    with open(SEQ_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    config["args"]["root"] = root
    with open(SEQ_CONFIG, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def update_imu_config(overrides: dict) -> None:
    """Update IMU config YAML with specific overrides."""
    import yaml
    with open(IMU_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    opt_args = config["Odometry"]["optimizer"]["args"]
    for key, value in overrides.items():
        keys = key.split(".")
        target = opt_args
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value

    with open(IMU_CONFIG, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def run_macvo(config_path: str) -> str:
    """Run MACVO and return the result directory path."""
    cmd = [
        "python", "MACVO.py",
        "--odom", config_path,
        "--data", SEQ_CONFIG,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/admin1/macvo-dev")
    # Find latest result directory
    results_dir = Path("/home/admin1/macvo-dev/Results")
    pattern = config_path.split("/")[-1].replace(".yaml", "")
    matching = sorted(results_dir.glob(f"*{pattern}*/*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matching:
        return str(matching[0])
    raise RuntimeError(f"No result directory found for {config_path}")


def evaluate(poses_npy: str, ref_csv: str, name: str) -> dict:
    """Run evaluation and return metrics dict."""
    cmd = [
        "python", "Scripts/eval_qa_vif.py",
        "--vo_result", poses_npy,
        "--ref_pose", ref_csv,
        "--name", name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/admin1/macvo-dev")

    # Parse output
    metrics = {"name": name}
    for line in result.stdout.split("\n"):
        if "RMSE:" in line:
            try:
                metrics["ate_rmse"] = float(line.split("RMSE:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Trans RMSE:" in line:
            try:
                metrics["rpe_trans_rmse"] = float(line.split("Trans RMSE:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Rot RMSE:" in line:
            try:
                metrics["rpe_rot_rmse_deg"] = float(line.split("Rot RMSE:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    # Also load from saved JSON if available
    json_path = Path(f"metrics_{name}.json")
    if json_path.exists():
        with open(json_path) as f:
            saved = json.load(f)
            metrics.update(saved)

    return metrics


def result_poses_path(res_dir: str) -> str:
    result_dir = Path(res_dir)
    for filename in ("poses.csv", "poses.npy"):
        candidate = result_dir / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"No poses.csv or poses.npy found under {result_dir}")


def run_ablation(dataset_name: str, data_root: str, ref_csv: str) -> list[dict]:
    """Run full ablation on one dataset."""
    print(f"\n{'='*70}")
    print(f"  Ablation Study: {dataset_name} ({data_root})")
    print(f"{'='*70}")

    update_seq_root(data_root)
    results = []

    # ── 1. Baseline: MACVO_Fast (no IMU) ─────────────────────────────────
    print("\n[1/5] Running Baseline (no IMU)...")
    try:
        res_dir = run_macvo(BASE_CONFIG)
        poses = result_poses_path(res_dir)
        metrics = evaluate(poses, ref_csv, f"BL_{dataset_name}")
        metrics["config"] = "Baseline (no IMU)"
        results.append(metrics)
        print(f"  ATE RMSE: {metrics.get('ate_rmse', 'N/A')}")
    except Exception as e:
        print(f"  Baseline failed: {e}")

    # ── 2. QA-VIF (post-fusion, rotation-only) ──────────────────────────
    print("\n[2/5] Running QA-VIF (post-fusion, rotation-only)...")
    try:
        update_imu_config({
            "imu_rot_prior": False,
            "post_imu_fusion_enable": True,
            "qa_vif_enable": True,
            "padf_enable": False,
        })
        res_dir = run_macvo(IMU_CONFIG)
        poses = result_poses_path(res_dir)
        metrics = evaluate(poses, ref_csv, f"QAVIF_{dataset_name}")
        metrics["config"] = "QA-VIF (post-fusion)"
        results.append(metrics)
        print(f"  ATE RMSE: {metrics.get('ate_rmse', 'N/A')}")
    except Exception as e:
        print(f"  QA-VIF failed: {e}")

    # ── 3. PADF (post-fusion, per-axis) ─────────────────────────────────
    print("\n[3/5] Running PADF (post-fusion, per-axis)...")
    try:
        update_imu_config({
            "imu_rot_prior": False,
            "post_imu_fusion_enable": True,
            "qa_vif_enable": False,
            "padf_enable": True,
        })
        res_dir = run_macvo(IMU_CONFIG)
        poses = result_poses_path(res_dir)
        metrics = evaluate(poses, ref_csv, f"PADF_{dataset_name}")
        metrics["config"] = "PADF (post-fusion)"
        results.append(metrics)
        print(f"  ATE RMSE: {metrics.get('ate_rmse', 'N/A')}")
    except Exception as e:
        print(f"  PADF failed: {e}")

    # ── 4. In-Graph Prior (scale-corrected, optimal) ────────────────────
    print("\n[4/5] Running In-Graph IMU Prior (scale=10)...")
    try:
        update_imu_config({
            "imu_rot_prior": True,
            "imu_rot_prior_scale": 10.0,
            "post_imu_fusion_enable": False,
            "qa_vif_enable": False,
            "padf_enable": False,
        })
        res_dir = run_macvo(IMU_CONFIG)
        poses = result_poses_path(res_dir)
        metrics = evaluate(poses, ref_csv, f"InGraph_{dataset_name}")
        metrics["config"] = "In-Graph Prior (scale=10)"
        results.append(metrics)
        print(f"  ATE RMSE: {metrics.get('ate_rmse', 'N/A')}")
    except Exception as e:
        print(f"  In-Graph failed: {e}")

    # ── 5. In-Graph + PADF (combined) ───────────────────────────────────
    print("\n[5/5] Running In-Graph + PADF (combined)...")
    try:
        update_imu_config({
            "imu_rot_prior": True,
            "imu_rot_prior_scale": 20.0,
            "post_imu_fusion_enable": True,
            "qa_vif_enable": False,
            "padf_enable": True,
        })
        res_dir = run_macvo(IMU_CONFIG)
        poses = result_poses_path(res_dir)
        metrics = evaluate(poses, ref_csv, f"Combo_{dataset_name}")
        metrics["config"] = "In-Graph + PADF"
        results.append(metrics)
        print(f"  ATE RMSE: {metrics.get('ate_rmse', 'N/A')}")
    except Exception as e:
        print(f"  Combined failed: {e}")

    return results


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """Print formatted summary table."""
    print(f"\n{'='*80}")
    print("  ABLATION STUDY SUMMARY")
    print(f"{'='*80}")

    for dataset, results in all_results.items():
        print(f"\n── {dataset} ──")
        print(f"  {'Config':<30} {'ATE RMSE':>10} {'RPE Trans':>10} {'RPE Rot':>10}")
        print(f"  {'-'*60}")
        baseline_ate = None
        for r in results:
            ate = r.get("ate_rmse", float("nan"))
            rpe_t = r.get("rpe_trans_rmse", float("nan"))
            rpe_r = r.get("rpe_rot_rmse_deg", float("nan"))
            if r["config"] == "Baseline (no IMU)":
                baseline_ate = ate
                print(f"  {r['config']:<30} {ate:>10.4f} {rpe_t:>10.4f} {rpe_r:>10.4f}")
            else:
                impr = ""
                if baseline_ate and not math.isnan(ate) and not math.isnan(baseline_ate):
                    impr = f" ({-(1 - ate / baseline_ate) * 100:+.1f}%)"
                print(f"  {r['config']:<30} {ate:>10.4f} {rpe_t:>10.4f} {rpe_r:>10.4f}{impr}")


def main():
    parser = argparse.ArgumentParser(description="MACVO Visual-Inertial Ablation Study")
    parser.add_argument("--datasets", nargs="+", default=["D4"],
                        help="Dataset keys to evaluate (D1, D2, D3, D4)")
    parser.add_argument("--output", type=str, default="ablation_results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"ablation_{timestamp}.json")

    for ds_key in args.datasets:
        if ds_key not in DATASETS:
            print(f"Unknown dataset: {ds_key}. Available: {list(DATASETS.keys())}")
            continue
        data_root = DATASETS[ds_key]
        ref_csv = str(Path(data_root) / "ref_pose.csv")
        if not Path(ref_csv).exists():
            print(f"Reference pose not found: {ref_csv}, skipping {ds_key}")
            continue

        results = run_ablation(ds_key, data_root, ref_csv)
        all_results[ds_key] = results

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Holdout Validation Experiments
================================
3 new scenes × 4 fixed modes × 3 trials + 3 scenes × Rule B × 3 trials = 45 runs total.

Matches the config approach from:
  - Scripts/run_7x4_experiments.py (fixed baseline, MACVO_HoloOcean_IMU.yaml base)
  - Scripts/run_ruleB_sanity_sequential.py (Rule B adaptive)

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/run_holdout_experiments.py
"""

from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Utility.Config import IncludeLoader

WORKDIR = Path("/home/admin1/macvo-dev")
BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401")
RESULT = WORKDIR / "Results" / "holdout_validation"

BASE_ODOM_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"

SCENES = ["moderate_turbidity", "open_water_overcast", "twilight_coast"]
TRIALS = 3

# (name, imu_rot_prior_enable, imu_trans_prior_enable, imu_rot_prior)
FIXED_MODES = [
    ("pure_macvo",       False, False, False),
    ("rotation_only",    True,  False, True),
    ("translation_only", False, True,  True),  # imu_rot_prior=True needed for translation prior
    ("full_imu",         True,  True,  True),
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def make_sequence_config(scene: str, tmpdir: Path) -> Path:
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(BATCH / scene)
    out = tmpdir / f"seq_{scene}.yaml"
    write_yaml(out, cfg)
    return out


def make_method_config(method: str, base: dict) -> dict:
    """Mirrors make_method_config() in Scripts/run_7x4_experiments.py."""
    cfg = copy.deepcopy(base)
    odom = cfg["Odometry"]
    opt = odom["optimizer"]["args"]

    opt["post_imu_fusion_enable"] = False
    opt["post_imu_fusion_mode"] = "none"
    opt["autodiff"] = False

    if method == "pure_macvo":
        odom["args"]["imu_rot_prior_enable"] = False
        odom["args"]["imu_trans_prior_enable"] = False
        opt["imu_rot_prior"] = False
    elif method == "rotation_only":
        odom["args"]["imu_rot_prior_enable"] = True
        odom["args"]["imu_trans_prior_enable"] = False
        opt["imu_rot_prior"] = True
    elif method == "translation_only":
        odom["args"]["imu_rot_prior_enable"] = False
        odom["args"]["imu_trans_prior_enable"] = True
        opt["imu_rot_prior"] = True  # needed for get_graph_data to read translation prior
    elif method == "full_imu":
        odom["args"]["imu_rot_prior_enable"] = True
        odom["args"]["imu_trans_prior_enable"] = True
        opt["imu_rot_prior"] = True
    else:
        raise ValueError(f"Unknown method: {method}")
    return cfg


def make_ruleB_config(base: dict) -> dict:
    """Mirrors build_config() in Scripts/run_ruleB_sanity_sequential.py."""
    cfg = copy.deepcopy(base)
    odom = cfg["Odometry"]
    opt = odom["optimizer"]["args"]

    opt["post_imu_fusion_enable"] = False
    opt["post_imu_fusion_mode"] = "none"
    opt["autodiff"] = False
    odom["args"]["imu_rot_prior_enable"] = True
    odom["args"]["imu_trans_prior_enable"] = True
    opt["imu_rot_prior"] = True
    odom["args"]["mapping"] = False
    return cfg


def run_macvo(odom_cfg: Path, seq_cfg: Path, result_root: Path, extra_args=None) -> int:
    """Run MACVO.py; stdout/stderr inherited so tqdm is visible."""
    cmd = [
        sys.executable, str(WORKDIR / "MACVO.py"),
        "--odom", str(odom_cfg),
        "--data", str(seq_cfg),
        "--resultRoot", str(result_root),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=str(WORKDIR), stdout=None, stderr=None).returncode


def main():
    total_fixed = len(SCENES) * len(FIXED_MODES) * TRIALS
    total_ruleB = len(SCENES) * TRIALS
    total = total_fixed + total_ruleB

    print("=" * 60)
    print("  Holdout Validation Experiments")
    print(f"  Scenes: {SCENES}")
    print(f"  Fixed modes: {[m[0] for m in FIXED_MODES]}")
    print(f"  Trials per config: {TRIALS}")
    print(f"  Fixed baseline: {total_fixed} runs")
    print(f"  Rule B adaptive: {total_ruleB} runs")
    print(f"  TOTAL: {total} runs")
    print("=" * 60)

    base_odom = load_yaml(BASE_ODOM_CFG)

    tmpdir = Path(tempfile.mkdtemp(prefix="holdout_"))
    print(f"\nTemp config dir: {tmpdir}")

    seq_cfgs = {s: make_sequence_config(s, tmpdir) for s in SCENES}

    odom_cfgs = {}
    for mode_name, rot, trans, rot_prior in FIXED_MODES:
        cfg = make_method_config(mode_name, base_odom)
        p = tmpdir / f"odom_{mode_name}.yaml"
        write_yaml(p, cfg)
        odom_cfgs[mode_name] = p

    ruleB_cfg = make_ruleB_config(base_odom)
    ruleB_odom = tmpdir / "odom_ruleB.yaml"
    write_yaml(ruleB_odom, ruleB_cfg)

    start_time = time.time()
    run_id = 0
    failures = []

    # ===== PHASE 1: Fixed Baseline (36 runs) =====
    print("\n" + "=" * 60)
    print("  PHASE 1: Fixed Baseline")
    print("=" * 60)

    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            for mode_name, _, _, _ in FIXED_MODES:
                run_id += 1
                print(f"\n-- Run {run_id}/{total}: trial={trial} scene={scene} mode={mode_name} --")
                t0 = time.time()
                rc = run_macvo(odom_cfgs[mode_name], seq_cfgs[scene], RESULT / "fixed_baseline")
                dt = time.time() - t0
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  -> {status}  ({dt:.1f}s)")
                if rc != 0:
                    failures.append(f"[fixed] t={trial} {scene} {mode_name}  rc={rc}")

    # ===== PHASE 2: Rule B Adaptive (9 runs) =====
    print("\n" + "=" * 60)
    print("  PHASE 2: Rule B Adaptive")
    print("=" * 60)

    ruleB_args = [
        "--adaptive-v3b",
        "--v3b-vc-mode", "two_level",
        "--v3b-vc-severe-thr", "30", "--v3b-vc-severe-sustain", "1",
        "--v3b-vc-mild-thr", "50", "--v3b-vc-mild-sustain", "5",
    ]

    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            run_id += 1
            print(f"\n-- Run {run_id}/{total}: trial={trial} scene={scene} mode=ruleB --")
            t0 = time.time()
            rc = run_macvo(ruleB_odom, seq_cfgs[scene], RESULT / "ruleB", extra_args=ruleB_args)
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  -> {status}  ({dt:.1f}s)")
            if rc != 0:
                failures.append(f"[ruleB] t={trial} {scene}  rc={rc}")

    # ── Summary ──
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ALL {run_id} RUNS COMPLETE  ({total_time/60:.1f} min total)")
    if failures:
        print(f"  FAILURES: {len(failures)}/{run_id}")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  All runs succeeded!")
    print(f"  Results: {RESULT}")
    print(f"  Temp configs: {tmpdir}")
    print("=" * 60)
    return int(len(failures) > 0)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
V3b++ Validation: 3 scenes × (4 fixed modes + Rule B + CP-B-FD-only) × 3 trials = 54 runs.

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/run_v3bpp_validation_3scene_54runs.py
"""

from __future__ import annotations

import copy, subprocess, sys, tempfile, time
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Utility.Config import IncludeLoader

WORKDIR = Path("/home/admin1/macvo-dev")
BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707")

SCENES = ["validation_moderate_harbor", "validation_transient_dropout", "validation_twilight_structure"]

TRIALS = 3
RESULT_ROOT = WORKDIR / "Results" / "v3bpp_validation_3scene_54runs"
BASE_ODOM = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"

FIXED_MODES = [
    ("pure_macvo",       False, False, False),
    ("rotation_only",    True,  False, True),
    ("translation_only", False, True,  True),
    ("full_imu",         True,  True,  True),
]


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f: return yaml.load(f, IncludeLoader)

def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def make_seq_cfg(scene, tmpdir):
    cfg = load_yaml(SEQ_TEMPLATE); cfg["args"]["root"] = str(BATCH / scene)
    out = tmpdir / f"seq_{scene}.yaml"; write_yaml(out, cfg); return out

def make_fixed_odom(mode_name, imu_rot, imu_trans, imu_rot_prior, tmpdir):
    cfg = load_yaml(BASE_ODOM)
    od = cfg["Odometry"]; opt = od["optimizer"]["args"]
    opt["post_imu_fusion_enable"] = False; opt["post_imu_fusion_mode"] = "none"; opt["autodiff"] = False
    od["args"]["imu_rot_prior_enable"] = imu_rot
    od["args"]["imu_trans_prior_enable"] = imu_trans
    opt["imu_rot_prior"] = imu_rot_prior
    out = tmpdir / f"odom_{mode_name}.yaml"; write_yaml(out, cfg); return out

def make_adaptive_odom(tmpdir):
    cfg = load_yaml(BASE_ODOM)
    od = cfg["Odometry"]; opt = od["optimizer"]["args"]
    opt["post_imu_fusion_enable"] = False; opt["post_imu_fusion_mode"] = "none"; opt["autodiff"] = False
    od["args"]["imu_rot_prior_enable"] = True; od["args"]["imu_trans_prior_enable"] = True
    opt["imu_rot_prior"] = True; od["args"]["mapping"] = False
    out = tmpdir / "odom_adaptive.yaml"; write_yaml(out, cfg); return out

def run_macvo(odom, seq, result_root, extra_args=None):
    cmd = [sys.executable, str(WORKDIR / "MACVO.py"),
           "--odom", str(odom), "--data", str(seq), "--resultRoot", str(result_root)]
    if extra_args: cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=str(WORKDIR), stdout=None, stderr=None).returncode


def main():
    total_fixed = len(SCENES) * len(FIXED_MODES) * TRIALS
    total_ruleB = len(SCENES) * TRIALS
    total_cpb = len(SCENES) * TRIALS
    total = total_fixed + total_ruleB + total_cpb

    print("=" * 70)
    print("  V3b++ Validation: 3 scenes × 54 runs")
    print(f"  Fixed baseline: {total_fixed}  |  Rule B: {total_ruleB}  |  CP-B: {total_cpb}")
    print(f"  Total: {total}")
    print("=" * 70)

    tmpdir = Path(tempfile.mkdtemp(prefix="v3bpp_val_"))
    print(f"Temp config dir: {tmpdir}")

    seq_cfgs = {s: make_seq_cfg(s, tmpdir) for s in SCENES}
    fixed_odoms = {m[0]: make_fixed_odom(*m, tmpdir) for m in FIXED_MODES}
    adaptive_odom = make_adaptive_odom(tmpdir)

    ruleB_args = [
        "--adaptive-v3b", "--v3b-vc-mode", "two_level",
        "--v3b-vc-severe-thr", "30", "--v3b-vc-severe-sustain", "1",
        "--v3b-vc-mild-thr", "50", "--v3b-vc-mild-sustain", "5",
    ]
    cpb_args = ruleB_args + ["--v3b-fd-cooldown", "30"]

    start = time.time(); run_id = 0; failures = []

    # ── PHASE A: Fixed Baseline (36 runs) ──────────────────────
    print("\n" + "=" * 70)
    print("  PHASE A: Fixed Baseline")
    print("=" * 70)
    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            for mode_name, _, _, _ in FIXED_MODES:
                run_id += 1
                print(f"\n-- Run {run_id}/{total}: [FIXED] trial={trial} scene={scene} mode={mode_name} --")
                t0 = time.time()
                rc = run_macvo(fixed_odoms[mode_name], seq_cfgs[scene],
                              RESULT_ROOT / "fixed_baseline")
                dt = time.time() - t0
                st = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  -> {st}  ({dt:.1f}s)")
                if rc != 0: failures.append(f"[FIXED] t={trial} {scene} {mode_name} rc={rc}")

    # ── PHASE B: Rule B Baseline (9 runs) ──────────────────────
    print("\n" + "=" * 70)
    print("  PHASE B: Rule B Baseline")
    print("=" * 70)
    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            run_id += 1
            print(f"\n-- Run {run_id}/{total}: [RuleB] trial={trial} scene={scene} --")
            t0 = time.time()
            rc = run_macvo(adaptive_odom, seq_cfgs[scene],
                          RESULT_ROOT / "ruleB_baseline", ruleB_args)
            dt = time.time() - t0
            st = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  -> {st}  ({dt:.1f}s)")
            if rc != 0: failures.append(f"[RuleB] t={trial} {scene} rc={rc}")

    # ── PHASE C: CP-B-FD-only (9 runs) ─────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE C: CP-B-FD-only")
    print("=" * 70)
    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            run_id += 1
            print(f"\n-- Run {run_id}/{total}: [CPB] trial={trial} scene={scene} --")
            t0 = time.time()
            rc = run_macvo(adaptive_odom, seq_cfgs[scene],
                          RESULT_ROOT / "cpb_fd_only", cpb_args)
            dt = time.time() - t0
            st = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  -> {st}  ({dt:.1f}s)")
            if rc != 0: failures.append(f"[CPB] t={trial} {scene} rc={rc}")

    tt = time.time() - start
    print(f"\n{'='*70}\n  ALL {run_id} RUNS ({tt/60:.1f} min)")
    print(f"  Results: {RESULT_ROOT}\n  Temp: {tmpdir}")
    if failures:
        print(f"  FAILURES: {len(failures)}/{run_id}")
        for f in failures: print(f"    - {f}")
    else:
        print("  All runs succeeded!")
    print("=" * 70)
    return int(len(failures) > 0)


if __name__ == "__main__":
    sys.exit(main())

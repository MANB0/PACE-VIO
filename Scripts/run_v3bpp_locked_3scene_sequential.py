#!/usr/bin/env python3
"""
V3b++ Locked Held-Out: 3 scenes x (4 fixed modes + Rule B + CP-B-FD-only) x 3 trials = 54 runs.

This script intentionally follows Scripts/run_v3bpp_validation_3scene_54runs.py:
  - strict sequential for-loops
  - no outer tqdm/rich/progress
  - no PTY/tee wrapper
  - MACVO.py stdout/stderr are left attached to the terminal, so its original tqdm is preserved

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/run_v3bpp_locked_3scene_sequential.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Utility.Config import IncludeLoader

WORKDIR = Path("/home/admin1/macvo-dev")
BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853")

SCENES = [
    "locked_murky_entry_help",
    "locked_clear_imu_harm",
    "locked_quality_degrade_no_dropout",
]

TRIALS = 3
RESULT_ROOT = WORKDIR / "Results" / "v3bpp_locked_3scene_54runs"
BASE_ODOM = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"

FIXED_MODES = [
    ("pure_macvo", False, False, False),
    ("rotation_only", True, False, True),
    ("translation_only", False, True, True),
    ("full_imu", True, True, True),
]

FIXED_MODE_NAMES = [mode[0] for mode in FIXED_MODES]


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def sanity_check_batch():
    ok = True
    for scene in SCENES:
        root = BATCH / scene
        if not root.exists():
            print(f"ERROR: missing scene directory: {root}")
            ok = False
            continue
        for subdir in ("left", "right"):
            if not (root / subdir).is_dir():
                print(f"ERROR: missing {subdir}/ for {scene}: {root / subdir}")
                ok = False
        for filename in ("imu_data.csv", "ref_pose.csv", "metadata.json"):
            if not (root / filename).exists():
                print(f"ERROR: missing {filename} for {scene}: {root / filename}")
                ok = False
    return ok


def scene_from_config(cfg):
    try:
        return cfg["Data"]["args"]["args"]["scene"]
    except Exception:
        return None


def fixed_method_from_config(cfg):
    args = cfg["Odometry"]["args"]
    imu_rot = args.get("imu_rot_prior_enable", False)
    imu_trans = args.get("imu_trans_prior_enable", False)
    if not imu_rot and not imu_trans:
        return "pure_macvo"
    if imu_rot and not imu_trans:
        return "rotation_only"
    if not imu_rot and imu_trans:
        return "translation_only"
    return "full_imu"


def count_completed_runs():
    """Count completed poses.csv runs by phase, scene, and method for resume."""
    counts = defaultdict(int)
    roots = [
        ("fixed_baseline", None),
        ("ruleB_baseline", "ruleB"),
        ("cpb_fd_only", "cpb_fd_only"),
    ]
    for phase, forced_method in roots:
        root = RESULT_ROOT / phase
        if not root.exists():
            continue
        for run_dir in root.glob("*/*"):
            if not (run_dir / "poses.csv").exists():
                continue
            cfg_path = run_dir / "config.yaml"
            if not cfg_path.exists():
                continue
            try:
                cfg = load_yaml(cfg_path)
            except Exception:
                continue
            scene = scene_from_config(cfg)
            if scene not in SCENES:
                continue
            method = forced_method if forced_method else fixed_method_from_config(cfg)
            if method not in FIXED_MODE_NAMES + ["ruleB", "cpb_fd_only"]:
                continue
            counts[(phase, scene, method)] += 1
    return counts


def list_incomplete_runs():
    """Return run directories with config but without poses.csv."""
    incomplete = []
    for phase in ("fixed_baseline", "ruleB_baseline", "cpb_fd_only"):
        root = RESULT_ROOT / phase
        if not root.exists():
            continue
        for run_dir in root.glob("*/*"):
            if (run_dir / "config.yaml").exists() and not (run_dir / "poses.csv").exists():
                incomplete.append(run_dir)
    return sorted(incomplete)


def make_seq_cfg(scene, tmpdir):
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(BATCH / scene)
    cfg["args"]["batch_root"] = str(BATCH)
    cfg["args"]["scene"] = scene
    out = tmpdir / f"seq_{scene}.yaml"
    write_yaml(out, cfg)
    return out


def make_fixed_odom(mode_name, imu_rot, imu_trans, imu_rot_prior, tmpdir):
    cfg = load_yaml(BASE_ODOM)
    od = cfg["Odometry"]
    opt = od["optimizer"]["args"]
    opt["post_imu_fusion_enable"] = False
    opt["post_imu_fusion_mode"] = "none"
    opt["autodiff"] = False
    od["args"]["imu_rot_prior_enable"] = imu_rot
    od["args"]["imu_trans_prior_enable"] = imu_trans
    opt["imu_rot_prior"] = imu_rot_prior
    out = tmpdir / f"odom_{mode_name}.yaml"
    write_yaml(out, cfg)
    return out


def make_adaptive_odom(tmpdir):
    cfg = load_yaml(BASE_ODOM)
    od = cfg["Odometry"]
    opt = od["optimizer"]["args"]
    opt["post_imu_fusion_enable"] = False
    opt["post_imu_fusion_mode"] = "none"
    opt["autodiff"] = False
    od["args"]["imu_rot_prior_enable"] = True
    od["args"]["imu_trans_prior_enable"] = True
    opt["imu_rot_prior"] = True
    od["args"]["mapping"] = False
    out = tmpdir / "odom_adaptive.yaml"
    write_yaml(out, cfg)
    return out


def run_macvo(odom, seq, result_root, extra_args=None):
    cmd = [
        sys.executable,
        str(WORKDIR / "MACVO.py"),
        "--odom",
        str(odom),
        "--data",
        str(seq),
        "--resultRoot",
        str(result_root),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=str(WORKDIR), stdout=None, stderr=None).returncode


def main():
    if not sanity_check_batch():
        return 1

    total_fixed = len(SCENES) * len(FIXED_MODES) * TRIALS
    total_rule_b = len(SCENES) * TRIALS
    total_cpb = len(SCENES) * TRIALS
    total = total_fixed + total_rule_b + total_cpb

    print("=" * 70)
    print("  V3b++ Locked Held-Out: 3 scenes x 54 runs")
    print(f"  Batch: {BATCH}")
    print(f"  Scenes: {SCENES}")
    print(f"  Fixed baseline: {total_fixed}  |  Rule B: {total_rule_b}  |  CP-B: {total_cpb}")
    print(f"  Total: {total}")
    print(f"  Results: {RESULT_ROOT}")
    print("=" * 70)

    tmpdir = Path(tempfile.mkdtemp(prefix="v3bpp_locked_"))
    print(f"Temp config dir: {tmpdir}")

    completed_counts = count_completed_runs()
    required_counts = defaultdict(int)
    if completed_counts:
        print("\nResume scan: completed runs found")
        for (phase, scene, method), count in sorted(completed_counts.items()):
            print(f"  [DONE] {phase} {scene} {method}: {count}/{TRIALS}")
    incomplete = list_incomplete_runs()
    if incomplete:
        print("\nResume scan: incomplete run dirs found (ignored; no poses.csv)")
        for run_dir in incomplete:
            print(f"  [INCOMPLETE] {run_dir}")

    seq_cfgs = {scene: make_seq_cfg(scene, tmpdir) for scene in SCENES}
    fixed_odoms = {mode[0]: make_fixed_odom(*mode, tmpdir) for mode in FIXED_MODES}
    adaptive_odom = make_adaptive_odom(tmpdir)

    rule_b_args = [
        "--adaptive-v3b",
        "--v3b-vc-mode",
        "two_level",
        "--v3b-vc-severe-thr",
        "30",
        "--v3b-vc-severe-sustain",
        "1",
        "--v3b-vc-mild-thr",
        "50",
        "--v3b-vc-mild-sustain",
        "5",
    ]
    cpb_args = rule_b_args + ["--v3b-fd-cooldown", "30"]

    start = time.time()
    run_id = 0
    failures = []

    def should_skip_completed(phase, scene, method):
        key = (phase, scene, method)
        required_counts[key] += 1
        if completed_counts.get(key, 0) >= required_counts[key]:
            print(f"  -> SKIP existing completed run ({completed_counts[key]}/{TRIALS})")
            return True
        return False

    print("\n" + "=" * 70)
    print("  PHASE A: Fixed Baseline")
    print("=" * 70)
    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            for mode_name, _, _, _ in FIXED_MODES:
                run_id += 1
                print(f"\n-- Run {run_id}/{total}: [FIXED] trial={trial} scene={scene} mode={mode_name} --")
                print(f"   resultRoot={RESULT_ROOT / 'fixed_baseline'}")
                if should_skip_completed("fixed_baseline", scene, mode_name):
                    continue
                t0 = time.time()
                rc = run_macvo(
                    fixed_odoms[mode_name],
                    seq_cfgs[scene],
                    RESULT_ROOT / "fixed_baseline",
                )
                dt = time.time() - t0
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  -> {status}  ({dt:.1f}s)")
                if rc != 0:
                    failures.append(f"[FIXED] t={trial} {scene} {mode_name} rc={rc}")

    print("\n" + "=" * 70)
    print("  PHASE B: Rule B Baseline")
    print("=" * 70)
    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            run_id += 1
            print(f"\n-- Run {run_id}/{total}: [RuleB] trial={trial} scene={scene} --")
            print(f"   resultRoot={RESULT_ROOT / 'ruleB_baseline'}")
            if should_skip_completed("ruleB_baseline", scene, "ruleB"):
                continue
            t0 = time.time()
            rc = run_macvo(
                adaptive_odom,
                seq_cfgs[scene],
                RESULT_ROOT / "ruleB_baseline",
                rule_b_args,
            )
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  -> {status}  ({dt:.1f}s)")
            if rc != 0:
                failures.append(f"[RuleB] t={trial} {scene} rc={rc}")

    print("\n" + "=" * 70)
    print("  PHASE C: CP-B-FD-only")
    print("=" * 70)
    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            run_id += 1
            print(f"\n-- Run {run_id}/{total}: [CPB] trial={trial} scene={scene} --")
            print(f"   resultRoot={RESULT_ROOT / 'cpb_fd_only'}")
            if should_skip_completed("cpb_fd_only", scene, "cpb_fd_only"):
                continue
            t0 = time.time()
            rc = run_macvo(
                adaptive_odom,
                seq_cfgs[scene],
                RESULT_ROOT / "cpb_fd_only",
                cpb_args,
            )
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  -> {status}  ({dt:.1f}s)")
            if rc != 0:
                failures.append(f"[CPB] t={trial} {scene} rc={rc}")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  ALL {run_id} RUNS ({elapsed / 60:.1f} min)")
    print(f"  Results: {RESULT_ROOT}")
    print(f"  Temp: {tmpdir}")
    if failures:
        print(f"  FAILURES: {len(failures)}/{run_id}")
        for failure in failures:
            print(f"    - {failure}")
    else:
        print("  All runs succeeded!")
    print("=" * 70)
    return int(len(failures) > 0)


if __name__ == "__main__":
    sys.exit(main())

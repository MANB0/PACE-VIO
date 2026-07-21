#!/usr/bin/env python3
"""
V3b++ Phase 1a: FD-E grace=30 — 5-scene × 3-trial sequential run script.

Scenes:
  1. moderate_turbidity  ×3  (primary: verify FD no longer kills full_imu)
  2. murky_coast         ×3  (regression guard: must keep full_imu from pair 1)
  3. open_water          ×3  (safety guard: no full_imu false positive)
  4. open_water_overcast ×3  (safety guard: gate stays in rotation_only)
  5. twilight_coast      ×3  (observation only; not a pass/fail criterion)

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/run_v3bpp_phase1a_fd_e_grace30.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Utility.Config import IncludeLoader

WORKDIR = Path("/home/admin1/macvo-dev")

# ── Data paths ───────────────────────────────────────────────────
OLD7_BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
NEW3_BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401")

SCENES_CONFIGS = [
    # (scene_name, batch_root)
    ("moderate_turbidity",  NEW3_BATCH),
    ("murky_coast",         OLD7_BATCH),
    ("open_water",          OLD7_BATCH),
    ("open_water_overcast", NEW3_BATCH),
    ("twilight_coast",      NEW3_BATCH),
]

TRIALS = 3
RESULT_ROOT = WORKDIR / "Results" / "v3bpp_phase1a_fd_e_grace30_5scene_3x"

# ── Base configs ─────────────────────────────────────────────────
BASE_ODOM_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def make_ruleB_odom_config(tmpdir: Path) -> Path:
    """Create Rule B odom config: full_imu base with mapping=False."""
    cfg = load_yaml(BASE_ODOM_CFG)
    od = cfg["Odometry"]
    opt = od["optimizer"]["args"]

    opt["post_imu_fusion_enable"] = False
    opt["post_imu_fusion_mode"] = "none"
    opt["autodiff"] = False
    od["args"]["imu_rot_prior_enable"] = True
    od["args"]["imu_trans_prior_enable"] = True
    opt["imu_rot_prior"] = True
    od["args"]["mapping"] = False

    out = tmpdir / "odom_ruleB.yaml"
    write_yaml(out, cfg)
    return out


def make_sequence_config(scene: str, batch: Path, tmpdir: Path) -> Path:
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(batch / scene)
    out = tmpdir / f"seq_{scene}.yaml"
    write_yaml(out, cfg)
    return out


def run_macvo(odom_cfg: Path, seq_cfg: Path, result_root: Path, extra_args: list | None = None) -> int:
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
    total = len(SCENES_CONFIGS) * TRIALS
    print("=" * 70)
    print("  V3b++ Phase 1a: FD-E grace=30 — 5-scene × 3-trial")
    print(f"  Total runs: {total}")
    print("=" * 70)

    # ── Generate temp configs ────────────────────────────────────
    tmpdir = Path(tempfile.mkdtemp(prefix="v3bpp_phase1a_"))
    print(f"Temp config dir: {tmpdir}")

    odom_cfg = make_ruleB_odom_config(tmpdir)
    seq_cfgs = {}
    for scene, batch in SCENES_CONFIGS:
        seq_cfgs[scene] = make_sequence_config(scene, batch, tmpdir)

    # ── FD-E grace args ──────────────────────────────────────────
    fd_grace_args = [
        "--adaptive-v3b",
        "--v3b-vc-mode", "two_level",
        "--v3b-vc-severe-thr", "30",
        "--v3b-vc-severe-sustain", "1",
        "--v3b-vc-mild-thr", "50",
        "--v3b-vc-mild-sustain", "5",
        "--v3b-fd-grace-enabled",
        "--v3b-fd-grace-period", "30",
    ]

    start_time = time.time()
    run_id = 0
    failures = []

    for trial in range(1, TRIALS + 1):
        for scene, batch in SCENES_CONFIGS:
            run_id += 1
            label = "OBSERVE" if scene == "twilight_coast" else "EVAL"
            print(f"\n── Run {run_id}/{total}: trial={trial} scene={scene} [{label}] ──")
            t0 = time.time()

            rc = run_macvo(
                odom_cfg,
                seq_cfgs[scene],
                RESULT_ROOT,
                extra_args=fd_grace_args,
            )

            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  -> {status}  ({dt:.1f}s)")
            if rc != 0:
                failures.append(f"[{label}] trial={trial} scene={scene}  rc={rc}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  ALL {run_id} RUNS COMPLETE  ({total_time/60:.1f} min)")
    if failures:
        print(f"  FAILURES: {len(failures)}/{run_id}")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  All runs succeeded!")
    print(f"  Results: {RESULT_ROOT}")
    print(f"  Temp configs: {tmpdir}")
    print("=" * 70)

    return int(len(failures) > 0)


if __name__ == "__main__":
    sys.exit(main())

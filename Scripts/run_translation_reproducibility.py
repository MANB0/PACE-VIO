#!/usr/bin/env python3
"""Run repeated translation_only experiments for reproducibility check."""
import subprocess, sys, time, os, json
os.chdir("/home/admin1/macvo-dev"); sys.path.insert(0, ".")
from pathlib import Path

RESULTS_ROOT = Path("Results") / "holoocean_translation_reproducibility"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "new_fixed_trans": "/tmp/odom_fixed_translation_only.yaml",
    "force_trans": "/tmp/odom_force_translation_only.yaml",
}
SEQ = "/tmp/seq_open_water.yaml"

for method, cfg_path in CONFIGS.items():
    for run in range(3):
        result_dir = RESULTS_ROOT / f"{method}_run{run}"
        result_dir.mkdir(exist_ok=True)
        print(f"[{method} run{run}] Starting...", flush=True)
        ts = time.time()

        proc = subprocess.run([
            sys.executable, "MACVO.py",
            "--odom", cfg_path, "--data", SEQ,
            "--resultRoot", str(result_dir),
            "--noeval"
        ], cwd=str(os.getcwd()), text=True, capture_output=True, timeout=1800)

        elapsed = time.time() - ts
        import shutil
        for poses_path in sorted(result_dir.rglob("poses.csv")):
            nested = poses_path.parent
            if nested != result_dir:
                for f in nested.iterdir():
                    if f.is_file() and not (result_dir / f.name).exists():
                        shutil.move(str(f), str(result_dir / f.name))

        status = "ok" if proc.returncode == 0 else f"rc={proc.returncode}"
        poses_csv = result_dir / "poses.csv"
        if poses_csv.exists():
            n = len(open(poses_csv).readlines()) - 1
        else:
            n = 0
        print(f"  {status} in {elapsed:.0f}s, {n} poses", flush=True)

print(f"\nAll done. Results in {RESULTS_ROOT}")

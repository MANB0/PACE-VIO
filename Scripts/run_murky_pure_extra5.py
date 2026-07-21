#!/usr/bin/env python3
"""Run murky_coast pure_macvo 5 extra trials for determinism verification."""
import sys, subprocess, time, os, yaml, copy
from pathlib import Path
from datetime import datetime

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader

RESULT = WORKDIR / "Results" / "murky_pure_extra5"
RESULT.mkdir(parents=True, exist_ok=True)
N = 5

# Build config once
with open(WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml") as f:
    cfg = yaml.load(f, IncludeLoader)
with open(WORKDIR / "Config/Sequence/holoocean_imu.yaml") as f:
    seq = yaml.load(f, IncludeLoader)

cfg = copy.deepcopy(cfg)
od = cfg["Odometry"]; op = od["optimizer"]["args"]
op["post_imu_fusion_enable"] = False; op["post_imu_fusion_mode"] = "none"
op["autodiff"] = False
od["args"]["imu_rot_prior_enable"] = False
od["args"]["imu_trans_prior_enable"] = False
op["imu_rot_prior"] = False
seq["args"]["root"] = "/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/murky_coast"

print(f"Running murky_coast pure_macvo × {N} extra trials...")
print(f"Output: {RESULT}\n")

for trial in range(1, N+1):
    d = RESULT / f"trial_{trial}"
    d.mkdir(parents=True, exist_ok=True)

    if (d / "poses.csv").exists():
        print(f"  [{trial}/{N}] SKIP (already exists)")
        continue

    tmp = d / ".tmp"; tmp.mkdir(exist_ok=True)
    (tmp / "odom.yaml").write_text(yaml.safe_dump(cfg))
    (tmp / "seq.yaml").write_text(yaml.safe_dump(seq))

    print(f"  [{trial}/{N}] Running... ", end="", flush=True)
    start = time.time()

    proc = subprocess.run(
        [sys.executable, str(WORKDIR / "MACVO.py"),
         "--odom", str(tmp / "odom.yaml"),
         "--data", str(tmp / "seq.yaml"),
         "--resultRoot", str(d)],
        cwd=str(WORKDIR),
        capture_output=True,
        timeout=3600
    )
    elapsed = time.time() - start

    # Flatten
    for nested in sorted(d.rglob("poses.csv")):
        nd = nested.parent
        if nd == d: continue
        for f in nd.iterdir():
            if f.is_file():
                dest = d / f.name
                if not dest.exists():
                    os.rename(str(f), str(dest))
        try: nd.rmdir()
        except: pass

    if (d / "poses.csv").exists():
        print(f"DONE ({elapsed:.0f}s)")
    else:
        print(f"FAILED (exit={proc.returncode})")

print(f"\nDone. Results in: {RESULT}")
print(f"Check: md5sum {RESULT}/trial_*/poses.csv | sort | uniq -c -w32")

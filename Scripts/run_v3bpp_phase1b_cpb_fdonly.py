#!/usr/bin/env python3
"""
V3b++ Phase 1b Experiment A: CP-B-FD-only (shortened FD cooldown, no FD-E grace).

Config: Rule B VC + FD cooldown=30 + FD-E disabled.

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/run_v3bpp_phase1b_cpb_fdonly.py
"""

from __future__ import annotations

import subprocess, sys, tempfile, time
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Utility.Config import IncludeLoader

WORKDIR = Path("/home/admin1/macvo-dev")
OLD7_BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
NEW3_BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401")

SCENES_CONFIGS = [
    ("moderate_turbidity",  NEW3_BATCH),
    ("murky_coast",         OLD7_BATCH),
    ("open_water",          OLD7_BATCH),
    ("open_water_overcast", NEW3_BATCH),
    ("twilight_coast",      NEW3_BATCH),  # observe only
]

TRIALS = 3
RESULT_ROOT = WORKDIR / "Results" / "v3bpp_phase1b_cpb_fdonly_5scene_3x"
BASE_ODOM_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"


def load_yaml(path):
    with path.open("r",encoding="utf-8") as f: return yaml.load(f,IncludeLoader)
def write_yaml(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f: yaml.safe_dump(data,f,sort_keys=False,allow_unicode=True)

def make_odom_cfg(tmpdir):
    cfg=load_yaml(BASE_ODOM_CFG)
    od=cfg["Odometry"];opt=od["optimizer"]["args"]
    opt["post_imu_fusion_enable"]=False;opt["post_imu_fusion_mode"]="none";opt["autodiff"]=False
    od["args"]["imu_rot_prior_enable"]=True;od["args"]["imu_trans_prior_enable"]=True
    opt["imu_rot_prior"]=True;od["args"]["mapping"]=False
    out=tmpdir/"odom.yaml";write_yaml(out,cfg);return out

def make_seq_cfg(scene,batch,tmpdir):
    cfg=load_yaml(SEQ_TEMPLATE);cfg["args"]["root"]=str(batch/scene)
    out=tmpdir/f"seq_{scene}.yaml";write_yaml(out,cfg);return out

def main():
    total=len(SCENES_CONFIGS)*TRIALS
    print("="*70)
    print("  V3b++ Phase 1b Experiment A: CP-B-FD-only (FD cooldown=30)")
    print(f"  Total: {total} runs")
    print("="*70)

    tmpdir=Path(tempfile.mkdtemp(prefix="v3bpp_ph1b_A_"))
    odom_cfg=make_odom_cfg(tmpdir)
    seq_cfgs={s:make_seq_cfg(s,b,tmpdir) for s,b in SCENES_CONFIGS}

    args_base=[
        "--adaptive-v3b","--v3b-vc-mode","two_level",
        "--v3b-vc-severe-thr","30","--v3b-vc-severe-sustain","1",
        "--v3b-vc-mild-thr","50","--v3b-vc-mild-sustain","5",
        "--v3b-fd-cooldown","30",
    ]

    start=time.time();run_id=0;failures=[]
    for trial in range(1,TRIALS+1):
        for scene,batch in SCENES_CONFIGS:
            run_id+=1
            lbl="OBSERVE" if scene=="twilight_coast" else "EVAL"
            print(f"\n-- Run {run_id}/{total}: trial={trial} scene={scene} [{lbl}] --")
            t0=time.time()
            rc=subprocess.run(
                [sys.executable,str(WORKDIR/"MACVO.py"),"--odom",str(odom_cfg),
                 "--data",str(seq_cfgs[scene]),"--resultRoot",str(RESULT_ROOT)]+args_base,
                cwd=str(WORKDIR),stdout=None,stderr=None).returncode
            dt=time.time()-t0
            st="OK" if rc==0 else f"FAIL(rc={rc})"
            print(f"  -> {st}  ({dt:.1f}s)")
            if rc!=0: failures.append(f"[{lbl}] t={trial} {scene} rc={rc}")

    tt=time.time()-start
    print(f"\n{'='*70}\n  ALL {run_id} RUNS ({tt/60:.1f} min)\n  Results: {RESULT_ROOT}\n  Temp: {tmpdir}")
    if failures:
        print(f"  FAILURES: {len(failures)}")
        for f in failures: print(f"    - {f}")
    print("="*70)
    return int(len(failures)>0)

if __name__=="__main__":sys.exit(main())

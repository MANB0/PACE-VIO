#!/usr/bin/env python3
"""Controlled switching experiments for open_water only."""
import subprocess, sys, os, csv, math, yaml
import numpy as np
from pathlib import Path
from datetime import datetime

WORKDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader

BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
SCENE = "open_water"
BASE_ODOM = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TMPL = WORKDIR / "Config/Sequence/holoocean_imu.yaml"
RESULT_BASE = WORKDIR / "Results" / f"open_water_controlled_switching_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULT_BASE.mkdir(parents=True, exist_ok=True)

def load_yaml(p):
    with open(p) as f: return yaml.load(f, IncludeLoader)
def write_yaml(p, d):
    p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding='utf-8')

def make_exp_config(method):
    """Build odometry config. For fixed methods, use same approach as run_7x4.
    For v3a forced modes, use the adaptive v3a path with force flags.
    """
    import copy
    cfg = copy.deepcopy(load_yaml(BASE_ODOM))
    odom = cfg["Odometry"]
    opt = odom["optimizer"]["args"]
    opt["post_imu_fusion_enable"] = False
    opt["post_imu_fusion_mode"] = "none"
    opt["autodiff"] = False
    opt["imu_rot_prior"] = True

    if method in ("current_fixed_pure",):
        odom["args"]["imu_rot_prior_enable"] = False
        odom["args"]["imu_trans_prior_enable"] = False
        opt["imu_rot_prior"] = False
    elif method in ("current_fixed_full",):
        odom["args"]["imu_rot_prior_enable"] = True
        odom["args"]["imu_trans_prior_enable"] = True
    elif method.startswith("v3a_"):
        odom["args"]["imu_rot_prior_enable"] = True
        odom["args"]["imu_trans_prior_enable"] = True
    else:
        raise ValueError(f"Unknown method: {method}")
    return cfg

def make_seq_config():
    cfg = load_yaml(SEQ_TMPL)
    cfg["args"]["root"] = str(BATCH / SCENE)
    return cfg

def run_experiment(method, v3a_force=""):
    """Run one open_water experiment. Returns result_dir."""
    result_dir = RESULT_BASE / method
    result_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    poses_exist = any(result_dir.rglob("poses.csv"))
    if poses_exist:
        print(f"  [{method}] Already has poses, skipping")
        return result_dir

    odom_cfg = make_exp_config(method)
    seq_cfg = make_seq_config()

    tmpdir = result_dir / ".tmp"
    tmpdir.mkdir(exist_ok=True)
    write_yaml(tmpdir / "odom.yaml", odom_cfg)
    write_yaml(tmpdir / "seq.yaml", seq_cfg)

    cmd = [sys.executable, str(WORKDIR / "MACVO.py"),
           "--odom", str(tmpdir / "odom.yaml"),
           "--data", str(tmpdir / "seq.yaml"),
           "--resultRoot", str(result_dir)]

    if method.startswith("v3a_"):
        cmd.append("--adaptive-v3a")
        if v3a_force:
            cmd.extend(["--v3a-force", v3a_force])
    # No --adaptive flag for fixed methods

    run_log = result_dir / "run.log"
    stderr_log = result_dir / "stderr.log"

    print(f"  [{method}] Running... (force={v3a_force or 'none'})")
    with open(run_log, "w") as out_f, open(stderr_log, "w") as err_f:
        proc = subprocess.run(cmd, cwd=str(WORKDIR), text=True,
                              stdout=out_f, stderr=err_f, timeout=7200)

    if proc.returncode != 0:
        print(f"  [{method}] FAILED: return code {proc.returncode}")
        return result_dir

    # Flatten nested results
    for nested in sorted(result_dir.rglob("poses.csv")):
        nd = nested.parent
        if nd == result_dir: continue
        for f in nd.iterdir():
            if f.is_file():
                dest = result_dir / f.name
                if not dest.exists():
                    os.rename(str(f), str(dest))
        try: nd.rmdir()
        except: pass

    print(f"  [{method}] DONE")
    return result_dir

def evaluate(result_dir, method_name):
    """Compute clean evaluation for a result directory."""
    poses_path = None
    for p in result_dir.rglob("poses.csv"): poses_path = p; break
    if not poses_path:
        return {"method": method_name, "status": "no_poses"}

    ref_path = BATCH / SCENE / "ref_pose.csv"
    est = np.genfromtxt(poses_path, delimiter=',', dtype=float, skip_header=1)
    if est.ndim == 1: est = est.reshape(1,-1)
    gt = np.genfromtxt(ref_path, delimiter=',', dtype=float, skip_header=1)
    if gt.ndim == 1: gt = gt.reshape(1,-1)

    n = min(len(est), len(gt))
    e = est[:n, 1:4]; g = gt[:n, 1:4]

    ate_d = float(np.sqrt(np.mean(np.sum((e-g)**2, axis=1))))
    ate_s = float(np.sqrt(np.mean(np.sum(((e-e[0])-(g-g[0]))**2, axis=1))))

    try:
        ec = e - e.mean(axis=0); gc = g - g.mean(axis=0)
        C = ec.T @ gc; U, _, Vt = np.linalg.svd(C)
        R = U @ Vt
        if np.linalg.det(R) < 0: R[:,-1] *= -1
        t = g.mean(axis=0) - R @ e.mean(axis=0)
        ea = (R @ e.T).T + t
        ate_se3 = float(np.sqrt(np.mean(np.sum((ea-g)**2, axis=1))))
        s = np.trace(R.T @ C) / max(np.trace(ec.T @ ec), 1e-12)
        eas = s * (R @ e.T).T + t
        ate_s3 = float(np.sqrt(np.mean(np.sum((eas-g)**2, axis=1))))
    except:
        ate_se3 = np.nan; ate_s3 = np.nan

    rpe_ts = [float(np.linalg.norm((e[i+1]-e[i])-(g[i+1]-g[i]))) for i in range(n-1)]
    rpe_rs = []
    for i in range(n-1):
        er = e[i+1]-e[i]; gr = g[i+1]-g[i]
        if np.linalg.norm(er)>1e-6 and np.linalg.norm(gr)>1e-6:
            ca = max(-1., min(1., float(np.dot(er,gr)/(np.linalg.norm(er)*np.linalg.norm(gr)))))
            rpe_rs.append(math.acos(ca))

    est_len = float(np.sum(np.sqrt(np.sum(np.diff(e,axis=0)**2, axis=1))))
    gt_len = float(np.sum(np.sqrt(np.sum(np.diff(g,axis=0)**2, axis=1))))
    pos_errs = np.sqrt(np.sum((e-g)**2, axis=1))

    dec_path = result_dir / "adaptive_decisions.csv"
    num_pairs = 0
    if dec_path.exists():
        with open(dec_path) as f: num_pairs = sum(1 for _ in f) - 1

    return {
        "method": method_name, "result_dir": str(result_dir), "num_poses": n, "num_pairs": num_pairs,
        "direct_ATE": round(ate_d,4), "start_aligned_ATE": round(ate_s,4),
        "RPE_translation_mean": round(float(np.mean(rpe_ts)),6) if rpe_ts else 0,
        "RPE_translation_median": round(float(np.median(rpe_ts)),6) if rpe_ts else 0,
        "RPE_rotation_mean_deg": round(float(np.mean(rpe_rs)*180/math.pi),4) if rpe_rs else 0,
        "RPE_rotation_median_deg": round(float(np.median(rpe_rs)*180/math.pi),4) if rpe_rs else 0,
        "trajectory_length": round(est_len,2), "gt_trajectory_length": round(gt_len,2),
        "trajectory_length_ratio": round(est_len/max(gt_len,1e-12),4),
        "final_position_error": round(float(pos_errs[-1]),4), "max_position_error": round(float(np.max(pos_errs)),4),
        "p95_position_error": round(float(np.percentile(pos_errs,95)),4),
        "SE3_aligned_ATE": round(ate_se3,4) if not np.isnan(ate_se3) else np.nan,
        "Sim3_aligned_ATE": round(ate_s3,4) if not np.isnan(ate_s3) else np.nan,
    }

# ================================================================
# Main
# ================================================================
experiments = [
    ("current_fixed_pure", ""),
    ("current_fixed_full", ""),
    ("v3a_force_full", "full"),
    ("v3a_pure2_latch_full", "pure2latch"),
    ("v3a_latched_first_trigger", "latch"),
]

print(f"Result base: {RESULT_BASE}")
print(f"Running {len(experiments)} experiments on {SCENE}...\n")

for method, v3a_force in experiments:
    run_experiment(method, v3a_force)

# Evaluate all
print("\n=== Evaluation ===")
all_evals = []
for method, _ in experiments:
    ev = evaluate(RESULT_BASE / method, method)
    all_evals.append(ev)
    print(f"  {method:35s}: ATE={ev.get('direct_ATE','N/A')}")

# Also read v3a_normal from full experiment
v3a_normal_dir = WORKDIR / "Results/holoocean_adaptive_v3a_20260517_235544/open_water"
if v3a_normal_dir.exists():
    ev_normal = evaluate(v3a_normal_dir, "adaptive_v3a_normal")
    all_evals.append(ev_normal)
    print(f"  {'adaptive_v3a_normal':35s}: ATE={ev_normal.get('direct_ATE','N/A')}")

# Read existing baselines
for old_method, old_path in [("old_fixed_pure", "Results/holoocean_7x4_20260515_211733/open_water/pure_macvo"),
                               ("old_fixed_full", "Results/holoocean_7x4_20260515_211733/open_water/full_imu")]:
    old_dir = WORKDIR / old_path
    if old_dir.exists():
        ev_old = evaluate(old_dir, old_method)
        all_evals.append(ev_old)
        print(f"  {old_method:35s}: ATE={ev_old.get('direct_ATE','N/A')}")

# Write evaluation CSV
if all_evals:
    import csv
    with open(RESULT_BASE/"open_water_controlled_switching_clean_evaluation.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_evals[0].keys()))
        w.writeheader(); w.writerows(all_evals)
    print(f"\n✅ Clean evaluation → {RESULT_BASE}/open_water_controlled_switching_clean_evaluation.csv")

print("\nDone. Result dir:", RESULT_BASE)

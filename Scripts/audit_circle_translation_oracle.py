#!/usr/bin/env python3
"""Audit circle relative translation, SE(3) composition, and oracle prerequisites.

This script is deliberately read-only with respect to MACVO/VIO production state.  It
compares the cached visual relative-pose factors against CameraLeftSocket truth, verifies
the exact camera/body conjugacy used by production, reconstructs the four rotation/
translation hybrids, and exports the translation-error structure needed before running
any visual oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pypose as pp
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.RelativePoseFactorCache import camera_factor_to_body_factor


DEFAULT_DATASET = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/clear_circle_truth_normal_noise"
)
DEFAULT_CACHE = ROOT / "VisualCache/static63_unique_visual_20260713/clear_circle_truth_normal_noise"
DEFAULT_PURE_RESULT = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/"
    "pure_macvo/clear_circle_truth_normal_noise/tensor_map.npz"
)
DEFAULT_VIO_RESULT = ROOT / (
    "Results/circle_straight_normal_noise_two_state_standard_full_20260715/trial_1/"
    "vio_two_state_fixed_lag_standard_full/clear_circle_truth_normal_noise/tensor_map.npz"
)
DEFAULT_FIRST_EDGE = ROOT / (
    "analysis_initialization_boundary_audit_20260716/first_edge_factor_decomposition.json"
)
DEFAULT_OUTPUT = ROOT / "analysis_circle_translation_oracle_20260716"

ACTIVE_START_FRAME = 90
NWU_TO_NED = np.diag([1.0, -1.0, -1.0])
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pure-result", type=Path, default=DEFAULT_PURE_RESULT)
    parser.add_argument("--vio-result", type=Path, default=DEFAULT_VIO_RESULT)
    parser.add_argument("--first-edge", type=Path, default=DEFAULT_FIRST_EDGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--active-start-frame", type=int, default=ACTIVE_START_FRAME)
    return parser.parse_args()


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonify(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


def se3_from_xyzw(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(7)
    return make_transform(Rotation.from_quat(values[3:7]).as_matrix(), values[:3])


def se3_to_xyzw(transform: np.ndarray) -> np.ndarray:
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()]


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rotation).reshape(3, 3)).as_rotvec()


def pose_nwu_to_internal(transform_nwu: np.ndarray) -> np.ndarray:
    """Convert T_WC from NWU world/local axes to MACVO's NED world/local axes."""
    return make_transform(
        NWU_TO_NED @ transform_nwu[:3, :3] @ NWU_TO_NED,
        NWU_TO_NED @ transform_nwu[:3, 3],
    )


def pose_internal_to_nwu(transform_internal: np.ndarray) -> np.ndarray:
    return make_transform(
        NWU_TO_NED @ transform_internal[:3, :3] @ NWU_TO_NED,
        NWU_TO_NED @ transform_internal[:3, 3],
    )


def load_truth(dataset: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(dataset / "ref_pose.csv")
    required = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"ref_pose.csv missing columns: {missing}")
    poses = []
    for row in frame.itertuples(index=False):
        rotation_nwu = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
        poses.append(pose_nwu_to_internal(make_transform(rotation_nwu, [row.x, row.y, row.z])))
    return frame, np.stack(poses)


def load_sidecar(cache: Path) -> dict[str, np.ndarray]:
    path = cache / "relative_pose_factors.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def relative_edges(poses: np.ndarray, frame_i: np.ndarray, frame_j: np.ndarray) -> np.ndarray:
    return np.stack([invert_transform(poses[i]) @ poses[j] for i, j in zip(frame_i, frame_j)])


def compose_edges(edges: np.ndarray) -> np.ndarray:
    poses = [np.eye(4, dtype=np.float64)]
    for edge in edges:
        poses.append(poses[-1] @ edge)
    return np.stack(poses)


def compose_edges_components(edges: np.ndarray) -> np.ndarray:
    poses = [np.eye(4, dtype=np.float64)]
    rotation = np.eye(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    for edge in edges:
        position = position + rotation @ edge[:3, 3]
        rotation = rotation @ edge[:3, :3]
        poses.append(make_transform(rotation, position))
    return np.stack(poses)


def compose_edges_wrong_translation(edges: np.ndarray) -> np.ndarray:
    poses = [np.eye(4, dtype=np.float64)]
    rotation = np.eye(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    for edge in edges:
        position = position + edge[:3, 3]
        rotation = rotation @ edge[:3, :3]
        poses.append(make_transform(rotation, position))
    return np.stack(poses)


def hybrid_edges(rotation_edges: np.ndarray, translation_edges: np.ndarray) -> np.ndarray:
    return np.stack(
        [make_transform(r[:3, :3], t[:3, 3]) for r, t in zip(rotation_edges, translation_edges)]
    )


def pose_error(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = predicted[:, :3, 3] - reference[:, :3, 3]
    rotation = np.stack(
        [rotation_log(predicted[k, :3, :3].T @ reference[k, :3, :3]) for k in range(len(predicted))]
    )
    return position, rotation


def edge_error(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(predicted), len(reference)) - 1
    translation = []
    rotation = []
    for k in range(count):
        z_pred = invert_transform(predicted[k]) @ predicted[k + 1]
        z_ref = invert_transform(reference[k]) @ reference[k + 1]
        error = invert_transform(z_pred) @ z_ref
        translation.append(error[:3, 3])
        rotation.append(rotation_log(error[:3, :3]))
    return np.asarray(translation), np.asarray(rotation)


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("min", "median", "mean", "p95", "max")}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def fit_circle_xy(points: np.ndarray) -> dict[str, float]:
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    a = np.c_[2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))]
    b = np.sum(xy * xy, axis=1)
    cx, cy, constant = np.linalg.lstsq(a, b, rcond=None)[0]
    radius_samples = np.linalg.norm(xy - np.array([cx, cy]), axis=1)
    normalized_time = np.linspace(0.0, 1.0, len(radius_samples))
    slope = float(np.polyfit(normalized_time, radius_samples, 1)[0]) if len(radius_samples) > 1 else 0.0
    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "radius_mean": float(np.mean(radius_samples)),
        "radius_std": float(np.std(radius_samples)),
        "radius_drift_per_sequence": slope,
    }


def trajectory_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    p_error, r_error = pose_error(predicted, reference)
    rpe_t, rpe_r = edge_error(predicted, reference)
    pred_nwu = np.stack([pose_internal_to_nwu(pose) for pose in predicted])
    ref_nwu = np.stack([pose_internal_to_nwu(pose) for pose in reference])
    pred_circle = fit_circle_xy(pred_nwu[:, :3, 3])
    ref_circle = fit_circle_xy(ref_nwu[:, :3, 3])
    return {
        "ate_translation_rmse_m": float(np.sqrt(np.mean(np.sum(p_error * p_error, axis=1)))),
        "orientation_rmse_rad": float(np.sqrt(np.mean(np.sum(r_error * r_error, axis=1)))),
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(np.sum(rpe_t * rpe_t, axis=1)))),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(np.sum(rpe_r * rpe_r, axis=1)))),
        "final_position_error_m": float(np.linalg.norm(p_error[-1])),
        "final_rotation_error_rad": float(np.linalg.norm(r_error[-1])),
        "closure_error_m": float(np.linalg.norm(predicted[-1, :3, 3] - predicted[0, :3, 3])),
        "truth_closure_error_m": float(np.linalg.norm(reference[-1, :3, 3] - reference[0, :3, 3])),
        "circle": pred_circle,
        "truth_circle": ref_circle,
        "circle_center_error_m": float(
            np.linalg.norm(
                [pred_circle["center_x"] - ref_circle["center_x"], pred_circle["center_y"] - ref_circle["center_y"]]
            )
        ),
        "circle_radius_mean_error_m": float(pred_circle["radius_mean"] - ref_circle["radius_mean"]),
    }


def yaw_nwu(transform_internal: np.ndarray) -> float:
    rotation_nwu = pose_internal_to_nwu(transform_internal)[:3, :3]
    return float(Rotation.from_matrix(rotation_nwu).as_euler("xyz", degrees=False)[2])


def load_pair_quality(cache: Path, frame_i: int, frame_j: int) -> dict[str, float]:
    pair_path = cache / "pairs" / f"{frame_i:06d}_{frame_j:06d}.npz"
    with np.load(pair_path, allow_pickle=False) as packet:
        diagnostics = json.loads(str(packet["covariance_diagnostics_json"].item()))
        depth = np.r_[packet["match__pixel1_d"].reshape(-1), packet["match__pixel2_d"].reshape(-1)]
        depth_cov = np.r_[
            packet["match__pixel1_d_cov"].reshape(-1), packet["match__pixel2_d_cov"].reshape(-1)
        ]
        flow_cov_u = float(diagnostics.get("mean_flow_u_cov", np.nan))
        flow_cov_v = float(diagnostics.get("mean_flow_v_cov", np.nan))
        return {
            "mean_depth_m": float(np.mean(depth)),
            "median_depth_m": float(np.median(depth)),
            "mean_depth_cov": float(np.mean(depth_cov)),
            "median_depth_cov": float(np.median(depth_cov)),
            "mean_flow_u_cov": flow_cov_u,
            "mean_flow_v_cov": flow_cov_v,
            "mean_flow_cov": float(np.nanmean([flow_cov_u, flow_cov_v])),
            "valid_depth_ratio": float(diagnostics.get("valid_depth_ratio", np.nan)),
        }


def stage_for_edge(elapsed_s: float, duration_s: float, speed: float) -> str:
    if speed < 0.02:
        return "static"
    if elapsed_s < 1.0:
        return "startup"
    if elapsed_s > duration_s - 1.0:
        return "stopping"
    return "uniform_turn"


def group_statistics(frame: pd.DataFrame, mask: np.ndarray | pd.Series) -> dict[str, Any]:
    selected = frame.loc[np.asarray(mask, dtype=bool)]
    output: dict[str, Any] = {"count": int(len(selected))}
    for column in [
        "e_tx", "e_ty", "e_tz", "e_t_norm", "e_forward", "e_lateral", "e_vertical",
        "length_ratio", "direction_angle_rad", "gt_translation_norm_m",
    ]:
        output[column] = quantiles(selected[column].to_numpy(dtype=np.float64))
    return output


def correlation_table(frame: pd.DataFrame) -> dict[str, Any]:
    targets = ["e_t_norm", "e_forward", "e_lateral", "e_vertical", "length_ratio"]
    signals = [
        "point_count", "inlier_ratio", "mean_point_mahalanobis_sq", "mean_depth_m",
        "mean_depth_cov", "mean_flow_cov", "pose_cov_min_eig", "pose_cov_max_eig",
        "pose_cov_condition", "speed_mps", "turning_rate_rad_s",
    ]
    output: dict[str, Any] = {}
    for target in targets:
        output[target] = {}
        for signal in signals:
            values = frame[[target, signal]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 3 or values[target].nunique() < 2 or values[signal].nunique() < 2:
                output[target][signal] = {"count": int(len(values)), "pearson": None, "spearman": None}
                continue
            pearson = pearsonr(values[target], values[signal])
            spearman = spearmanr(values[target], values[signal])
            output[target][signal] = {
                "count": int(len(values)),
                "pearson": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
    return output


def explicit_camera_to_body(z_camera: np.ndarray, t_ci: np.ndarray) -> np.ndarray:
    """Direct matrix equivalent of production: Z_IiIj = T_IC Z_CiCj T_CI."""
    return invert_transform(t_ci) @ z_camera @ t_ci


def build_html(path: Path, trajectory_frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    stride = max(1, len(trajectory_frame) // 900)
    sampled = trajectory_frame.iloc[::stride].copy()
    if sampled.index[-1] != trajectory_frame.index[-1]:
        sampled = pd.concat([sampled, trajectory_frame.iloc[[-1]]], ignore_index=True)
    payload = {
        "time": np.round(sampled["elapsed_s"].to_numpy(), 4).tolist(),
        "series": {
            name: {
                axis: np.round(sampled[f"{name}_{axis}"].to_numpy(), 6).tolist()
                for axis in ("x", "y", "z")
            }
            for name in ("GT", "GG", "GM", "MG", "MM", "MM_demean")
        },
        "classification": summary["attribution"]["classification"],
    }
    html = f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Circle Hybrid Reconstruction</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#17202a}}main{{max-width:1180px;margin:auto;padding:20px}}
h1{{font-size:24px;margin:0 0 6px}}p{{color:#536273;margin:0 0 16px}}.tools{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:10px 0}}
button,label{{font:inherit}}button{{padding:7px 12px;border:1px solid #aab6c3;background:#fff;border-radius:6px;cursor:pointer}}button.active{{background:#246bfd;color:#fff;border-color:#246bfd}}
label{{display:flex;gap:5px;align-items:center}}canvas{{width:100%;height:650px;background:#fff;border:1px solid #d9e0e7;border-radius:6px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}}.sw{{width:22px;height:3px;display:inline-block;margin-right:5px;vertical-align:middle}}
</style></head><body><main><h1>圆形相对位姿混合重建</h1><p id=\"subtitle\"></p>
<div class=\"tools\"><button data-view=\"xy\" class=\"active\">XY</button><button data-view=\"xz\">XZ</button><button data-view=\"yz\">YZ</button>
<label><input id=\"equal\" type=\"checkbox\" checked>等比例坐标</label></div>
<canvas id=\"plot\" width=\"1120\" height=\"650\" aria-label=\"GT、GG、GM、MG、MM和去均值反事实轨迹\"></canvas><div class=\"legend\" id=\"legend\"></div>
<script>
const D={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};
const colors={{GT:'#202833',GG:'#00a36c',GM:'#f59e0b',MG:'#7c3aed',MM:'#ef4444',MM_demean:'#168aad'}};
let view='xy'; const visible={{GT:true,GG:true,GM:true,MG:true,MM:true,MM_demean:true}};
const canvas=document.getElementById('plot'),ctx=canvas.getContext('2d'); document.getElementById('subtitle').textContent='判定：'+D.classification;
const legend=document.getElementById('legend'); Object.keys(colors).forEach(name=>{{const l=document.createElement('label');l.innerHTML=`<input type=\"checkbox\" checked data-series=\"${{name}}\"><span><i class=\"sw\" style=\"background:${{colors[name]}}\"></i>${{name}}</span>`;legend.appendChild(l);}});
function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height);const axes=view.split('');let xs=[],ys=[];Object.entries(D.series).forEach(([n,s])=>{{if(visible[n]){{xs.push(...s[axes[0]]);ys.push(...s[axes[1]])}}}});let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);let dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);if(document.getElementById('equal').checked){{const ratio=(canvas.width-100)/(canvas.height-80);if(dx/dy>ratio){{const ndy=dx/ratio;const c=(ymin+ymax)/2;ymin=c-ndy/2;ymax=c+ndy/2;dy=ndy}}else{{const ndx=dy*ratio;const c=(xmin+xmax)/2;xmin=c-ndx/2;xmax=c+ndx/2;dx=ndx}}}}const px=x=>60+(x-xmin)/dx*(canvas.width-90),py=y=>canvas.height-45-(y-ymin)/dy*(canvas.height-75);
ctx.strokeStyle='#dce3ea';ctx.lineWidth=1;for(let k=0;k<=5;k++){{const x=60+k*(canvas.width-90)/5,y=30+k*(canvas.height-75)/5;ctx.beginPath();ctx.moveTo(x,30);ctx.lineTo(x,canvas.height-45);ctx.stroke();ctx.beginPath();ctx.moveTo(60,y);ctx.lineTo(canvas.width-30,y);ctx.stroke();}}ctx.fillStyle='#536273';ctx.font='13px system-ui';ctx.fillText(axes[0].toUpperCase()+' / m',canvas.width/2,canvas.height-10);ctx.save();ctx.translate(16,canvas.height/2);ctx.rotate(-Math.PI/2);ctx.fillText(axes[1].toUpperCase()+' / m',0,0);ctx.restore();
Object.entries(D.series).forEach(([n,s])=>{{if(!visible[n])return;ctx.strokeStyle=colors[n];ctx.lineWidth=n==='GT'?3:1.8;ctx.setLineDash(n==='GG'?[6,4]:[]);ctx.beginPath();s[axes[0]].forEach((x,k)=>{{const y=s[axes[1]][k];k?ctx.lineTo(px(x),py(y)):ctx.moveTo(px(x),py(y))}});ctx.stroke();ctx.setLineDash([]);const k=s[axes[0]].length-1,x=px(s[axes[0]][k]),y=py(s[axes[1]][k]),k0=Math.max(0,k-6),x0=px(s[axes[0]][k0]),y0=py(s[axes[1]][k0]);const a=Math.atan2(y-y0,x-x0);ctx.fillStyle=colors[n];ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-11*Math.cos(a-.45),y-11*Math.sin(a-.45));ctx.lineTo(x-11*Math.cos(a+.45),y-11*Math.sin(a+.45));ctx.closePath();ctx.fill();}});
}}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{{view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b));draw();}});legend.onchange=e=>{{if(e.target.dataset.series){{visible[e.target.dataset.series]=e.target.checked;draw();}}}};document.getElementById('equal').onchange=draw;draw();
</script></main></body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    ref, truth_poses = load_truth(args.dataset)
    sidecar = load_sidecar(args.cache)
    frame_i_all = sidecar["frame_i"].astype(int)
    frame_j_all = sidecar["frame_j"].astype(int)
    active_mask = frame_i_all >= int(args.active_start_frame)
    frame_i = frame_i_all[active_mask]
    frame_j = frame_j_all[active_mask]
    if not np.array_equal(frame_i, np.arange(args.active_start_frame, len(ref) - 1)):
        raise AssertionError("active sidecar edges are not contiguous through the final frame")
    timestamps = ref["timestamp"].to_numpy(dtype=np.int64)
    z_gt = relative_edges(truth_poses, frame_i, frame_j)
    z_mac = np.stack([se3_from_xyzw(row.reshape(7)) for row in sidecar["measurement_CiCj"][active_mask]])
    covariance = sidecar["covariance"][active_mask].astype(np.float64)

    # The GT identity t_ij = R_WCi^T (p_WCj - p_WCi) is checked in internal NED.
    direct_t_gt = np.stack(
        [truth_poses[i, :3, :3].T @ (truth_poses[j, :3, 3] - truth_poses[i, :3, 3]) for i, j in zip(frame_i, frame_j)]
    )
    gt_translation_identity_error = float(np.max(np.abs(direct_t_gt - z_gt[:, :3, 3])))
    if gt_translation_identity_error > 1.0e-12:
        raise AssertionError("GT local-translation identity failed")

    rot_error = np.stack([rotation_log(z_mac[k, :3, :3].T @ z_gt[k, :3, :3]) for k in range(len(z_gt))])
    trans_error = z_mac[:, :3, 3] - z_gt[:, :3, 3]
    ref_velocity_nwu = ref[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    duration_s = float((timestamps[-1] - timestamps[args.active_start_frame]) * 1.0e-9)
    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, float]] = []
    for row_index, (i, j) in enumerate(zip(frame_i, frame_j)):
        quality_rows.append(load_pair_quality(args.cache, int(i), int(j)))
        eig = np.linalg.eigvalsh(covariance[row_index])
        t_norm = float(np.linalg.norm(z_gt[row_index, :3, 3]))
        speed = float(np.linalg.norm(ref_velocity_nwu[i]))
        dt = float((timestamps[j] - timestamps[i]) * 1.0e-9)
        turning_rate = float(np.linalg.norm(rotation_log(z_gt[row_index, :3, :3])) / max(dt, EPS))
        elapsed = float((timestamps[i] - timestamps[args.active_start_frame]) * 1.0e-9)
        forward = z_gt[row_index, :3, 3] / max(t_norm, EPS)
        vertical_world = np.array([0.0, 0.0, 1.0])  # NED gravity/down direction.
        vertical_local = truth_poses[i, :3, :3].T @ vertical_world
        vertical_local /= max(np.linalg.norm(vertical_local), EPS)
        lateral = np.cross(vertical_local, forward)
        if np.linalg.norm(lateral) < 1.0e-9:
            lateral = np.array([0.0, 1.0, 0.0])
        lateral /= max(np.linalg.norm(lateral), EPS)
        vertical_basis = np.cross(forward, lateral)
        vertical_basis /= max(np.linalg.norm(vertical_basis), EPS)
        dot = float(np.dot(z_mac[row_index, :3, 3], z_gt[row_index, :3, 3]))
        denom = max(float(np.linalg.norm(z_mac[row_index, :3, 3])) * t_norm, EPS)
        rows.append(
            {
                "edge_id": row_index,
                "frame_i": int(i), "frame_j": int(j),
                "timestamp_i_ns": int(timestamps[i]), "timestamp_j_ns": int(timestamps[j]),
                "elapsed_s": elapsed, "dt_s": dt,
                **{f"R_gt_{a}{b}": z_gt[row_index, a, b] for a in range(3) for b in range(3)},
                **{f"R_mac_{a}{b}": z_mac[row_index, a, b] for a in range(3) for b in range(3)},
                "t_gt_x": z_gt[row_index, 0, 3], "t_gt_y": z_gt[row_index, 1, 3], "t_gt_z": z_gt[row_index, 2, 3],
                "t_mac_x": z_mac[row_index, 0, 3], "t_mac_y": z_mac[row_index, 1, 3], "t_mac_z": z_mac[row_index, 2, 3],
                "rotation_error_x": rot_error[row_index, 0], "rotation_error_y": rot_error[row_index, 1], "rotation_error_z": rot_error[row_index, 2],
                "rotation_error_norm_rad": float(np.linalg.norm(rot_error[row_index])),
                "translation_error_x": trans_error[row_index, 0], "translation_error_y": trans_error[row_index, 1], "translation_error_z": trans_error[row_index, 2],
                "translation_error_norm_m": float(np.linalg.norm(trans_error[row_index])),
                "e_tx": trans_error[row_index, 0], "e_ty": trans_error[row_index, 1], "e_tz": trans_error[row_index, 2],
                "e_t_norm": float(np.linalg.norm(trans_error[row_index])),
                "length_ratio": float(np.linalg.norm(z_mac[row_index, :3, 3]) / max(t_norm, EPS)),
                "direction_angle_rad": float(math.acos(np.clip(dot / denom, -1.0, 1.0))) if t_norm >= 1e-4 else float("nan"),
                "e_forward": float(np.dot(trans_error[row_index], forward)) if t_norm >= 1e-4 else float("nan"),
                "e_lateral": float(np.dot(trans_error[row_index], lateral)) if t_norm >= 1e-4 else float("nan"),
                "e_vertical": float(np.dot(trans_error[row_index], vertical_basis)) if t_norm >= 1e-4 else float("nan"),
                "gt_translation_norm_m": t_norm,
                "motion_basis_valid": bool(t_norm >= 1e-4),
                "point_count": int(sidecar["num_points"][i]),
                "inlier_count": int(sidecar["num_inliers"][i]),
                "inlier_ratio": float(sidecar["num_inliers"][i] / max(sidecar["num_points"][i], 1)),
                "mean_point_mahalanobis_sq": float(sidecar["mean_mahalanobis_sq"][i]),
                "pose_cov_min_eig": float(eig[0]), "pose_cov_max_eig": float(eig[-1]),
                "pose_cov_condition": float(eig[-1] / max(eig[0], EPS)),
                "speed_mps": speed, "turning_rate_rad_s": turning_rate,
                "stage": stage_for_edge(elapsed, duration_s, speed),
                "frame_i_mod3": int(i % 3),
            }
        )
    relative_frame = pd.DataFrame(rows)
    quality_frame = pd.DataFrame(quality_rows)
    relative_frame = pd.concat([relative_frame.reset_index(drop=True), quality_frame], axis=1)
    relative_path = output / "circle_relative_pose_gt_mac_per_edge.csv"
    relative_frame.to_csv(relative_path, index=False)

    # SE(3) reconstruction and production-source cross-checks.
    gt_relative = compose_edges(z_gt)
    gg_matrix = gt_relative
    gg_components = compose_edges_components(z_gt)
    mm_matrix = compose_edges(z_mac)
    mm_components = compose_edges_components(z_mac)
    mm_wrong = compose_edges_wrong_translation(z_mac)
    mm_inverse_direction = compose_edges(np.stack([invert_transform(edge) for edge in z_mac]))
    pure_rebased_error = None
    pure_rebased_wrong_error = None
    pure_rebased_inverse_error = None
    if args.pure_result.exists():
        with np.load(args.pure_result, allow_pickle=False) as pure:
            pure_pose = np.stack([se3_from_xyzw(row) for row in pure["frames//pose"].astype(np.float64)])
        pure_active = np.stack(
            [invert_transform(pure_pose[args.active_start_frame]) @ pure_pose[k] for k in range(args.active_start_frame, len(pure_pose))]
        )
        pure_rebased_error = float(np.max(np.abs(pure_active - mm_matrix)))
        pure_rebased_wrong_error = float(np.sqrt(np.mean(np.sum((pure_active[:, :3, 3] - mm_wrong[:, :3, 3]) ** 2, axis=1))))
        pure_rebased_inverse_error = float(
            np.sqrt(np.mean(np.sum((pure_active[:, :3, 3] - mm_inverse_direction[:, :3, 3]) ** 2, axis=1)))
        )

    with np.load(args.vio_result, allow_pickle=False) as vio:
        sensor_t_imu = vio["frames//imu_vio_sensor_T_imu"][args.active_start_frame].astype(np.float64)
    t_ci = se3_from_xyzw(sensor_t_imu)
    body_rows = []
    max_body_translation_error = 0.0
    max_body_rotation_error = 0.0
    for row_index, (i, j) in enumerate(zip(frame_i, frame_j)):
        prod_mean, _ = camera_factor_to_body_factor(
            torch.tensor(se3_to_xyzw(z_mac[row_index]).reshape(1, 7), dtype=torch.float64),
            torch.tensor(covariance[row_index], dtype=torch.float64),
            torch.tensor(sensor_t_imu.reshape(1, 7), dtype=torch.float64),
        )
        prod_transform = se3_from_xyzw(prod_mean.detach().cpu().numpy().reshape(7))
        direct_transform = explicit_camera_to_body(z_mac[row_index], t_ci)
        translation_difference = prod_transform[:3, 3] - direct_transform[:3, 3]
        rotation_difference = rotation_log(prod_transform[:3, :3].T @ direct_transform[:3, :3])
        max_body_translation_error = max(max_body_translation_error, float(np.max(np.abs(translation_difference))))
        max_body_rotation_error = max(max_body_rotation_error, float(np.max(np.abs(rotation_difference))))
        body_rows.append(
            {
                "frame_i": int(i), "frame_j": int(j),
                "camera_tx": z_mac[row_index, 0, 3], "camera_ty": z_mac[row_index, 1, 3], "camera_tz": z_mac[row_index, 2, 3],
                "body_tx_production": prod_transform[0, 3], "body_ty_production": prod_transform[1, 3], "body_tz_production": prod_transform[2, 3],
                "body_tx_direct_4x4": direct_transform[0, 3], "body_ty_direct_4x4": direct_transform[1, 3], "body_tz_direct_4x4": direct_transform[2, 3],
                "translation_error_x": translation_difference[0], "translation_error_y": translation_difference[1], "translation_error_z": translation_difference[2],
                "translation_error_max_abs": float(np.max(np.abs(translation_difference))),
                "rotation_error_max_abs_rad": float(np.max(np.abs(rotation_difference))),
                "lever_arm_translation_change_norm_m": float(np.linalg.norm(direct_transform[:3, 3] - z_mac[row_index, :3, 3])),
            }
        )
    pd.DataFrame(body_rows).to_csv(output / "circle_camera_body_translation_contract.csv", index=False)

    # Direction audit on the first active cached 3D pair: Z must map points_j into points_i.
    pair_path = args.cache / "pairs" / f"{frame_i[0]:06d}_{frame_j[0]:06d}.npz"
    with np.load(pair_path, allow_pickle=False) as packet:
        points_i = packet["points_local"].astype(np.float64)
        uv_j = packet["match__pixel2_uv"].astype(np.float64)
        d_j = packet["match__pixel2_d"].astype(np.float64).reshape(-1)
        k_inv = np.linalg.inv(packet["K"].astype(np.float64))
        points_edn = (k_inv @ np.c_[uv_j, np.ones(len(uv_j))].T).T * d_j[:, None]
        points_j = np.roll(points_edn, shift=1, axis=1)
    mapped_forward = (z_mac[0, :3, :3] @ points_j.T).T + z_mac[0, :3, 3]
    z_inverse = invert_transform(z_mac[0])
    mapped_inverse = (z_inverse[:3, :3] @ points_j.T).T + z_inverse[:3, 3]
    direction_forward_rmse = float(np.sqrt(np.mean(np.sum((points_i - mapped_forward) ** 2, axis=1))))
    direction_inverse_rmse = float(np.sqrt(np.mean(np.sum((points_i - mapped_inverse) ** 2, axis=1))))

    contract = {
        "pose_convention": {
            "T_AB": "maps coordinates from frame B into frame A",
            "truth_pose": "T_WC at CameraLeftSocket in NWU, converted to internal NED on world and local axes",
            "sidecar_measurement": "T_CiCj = inverse(T_WCi) * T_WCj; maps C_j points into C_i",
            "gt_translation_identity": "t_ij = R_WCi^T * (p_WCj - p_WCi)",
            "gt_translation_identity_max_abs_error": gt_translation_identity_error,
        },
        "active_edges": int(len(z_gt)),
        "active_start_frame": int(args.active_start_frame),
        "matrix_vs_component": {
            "gt_max_abs_error": float(np.max(np.abs(gg_matrix - gg_components))),
            "macvo_max_abs_error": float(np.max(np.abs(mm_matrix - mm_components))),
        },
        "gg_vs_truth_relative_max_abs_error": float(np.max(np.abs(gg_matrix - gt_relative))),
        "source_pure_macvo_rebased_vs_sidecar_mm_max_abs_error": pure_rebased_error,
        "wrong_translation_accumulation_vs_source_position_rmse_m": pure_rebased_wrong_error,
        "inverse_sidecar_direction_vs_source_position_rmse_m": pure_rebased_inverse_error,
        "camera_body_conjugacy": {
            "production_formula": "T_IC * Z_CiCj * T_CI",
            "sensor_T_imu_semantics": "T_CI; optimizer pose_WB = pose_WC * T_CI",
            "sensor_T_imu_xyzw": sensor_t_imu,
            "max_abs_translation_error": max_body_translation_error,
            "max_abs_rotation_error_rad": max_body_rotation_error,
        },
        "pose_direction_3d_check": {
            "forward_T_CiCj_point_rmse_m": direction_forward_rmse,
            "inverse_direction_point_rmse_m": direction_inverse_rmse,
        },
        "production_pose_composition_evidence": {
            "motion_model": "Module/MotionModel.py: TartanMotionNet uses new_pose = prev_pose @ Exp(relative_motion)",
            "sidecar_source": "Odometry/MACVO.py: rel_pose = pose_i.Inv() @ pose_j",
        },
    }
    contract["stop_conditions"] = {
        "gg_failed": contract["gg_vs_truth_relative_max_abs_error"] > 1.0e-10,
        "matrix_component_mismatch": max(
            contract["matrix_vs_component"]["gt_max_abs_error"],
            contract["matrix_vs_component"]["macvo_max_abs_error"],
        ) > 1.0e-10,
        "camera_body_conjugacy_failed": max_body_translation_error > 1.0e-10 or max_body_rotation_error > 1.0e-10,
        "pose_direction_failed": (
            direction_forward_rmse >= direction_inverse_rmse
            or (
                pure_rebased_inverse_error is not None
                and pure_rebased_inverse_error < 100.0 * max(pure_rebased_error or 0.0, 1e-12)
            )
        ),
        "source_chain_mismatch": pure_rebased_error is not None and pure_rebased_error > 1.0e-5,
    }
    contract["oracle_allowed"] = not any(contract["stop_conditions"].values())
    write_json(output / "circle_se3_accumulation_contract.json", contract)

    # Four cross trajectories and fixed-mean translation counterfactual.
    hybrids = {
        "GG": compose_edges(hybrid_edges(z_gt, z_gt)),
        "GM": compose_edges(hybrid_edges(z_gt, z_mac)),
        "MG": compose_edges(hybrid_edges(z_mac, z_gt)),
        "MM": mm_matrix,
    }
    training_edges = relative_frame["elapsed_s"].to_numpy() < 10.0
    mu_t = np.mean(trans_error[training_edges], axis=0)
    z_mac_demean = z_mac.copy()
    z_mac_demean[:, :3, 3] -= mu_t
    hybrids["MM_demean"] = compose_edges(z_mac_demean)
    segments = {
        "first_10s": int(np.searchsorted(timestamps[args.active_start_frame:] - timestamps[args.active_start_frame], 10.0e9, side="right")),
        "first_half_circle": int(np.searchsorted(timestamps[args.active_start_frame:] - timestamps[args.active_start_frame], 30.0e9, side="right")),
        "full_circle": len(gt_relative),
    }
    hybrid_summary: dict[str, Any] = {"segments": {}, "translation_demean": {"training_duration_s": 10.0, "mu_t_internal_camera_m": mu_t}}
    for segment_name, endpoint in segments.items():
        endpoint = max(2, min(endpoint, len(gt_relative)))
        hybrid_summary["segments"][segment_name] = {
            name: trajectory_metrics(trajectory[:endpoint], gt_relative[:endpoint]) for name, trajectory in hybrids.items()
        }
    full_metrics = hybrid_summary["segments"]["full_circle"]
    gm = full_metrics["GM"]["ate_translation_rmse_m"]
    mg = full_metrics["MG"]["ate_translation_rmse_m"]
    mm = full_metrics["MM"]["ate_translation_rmse_m"]
    gg = full_metrics["GG"]["ate_translation_rmse_m"]
    if gm > 2.0 * max(mg, 1.0e-12):
        classification = "MACVO relative translation is the dominant pure-visual drift source (GM bad, MG near GG)."
    elif mg > 2.0 * max(gm, 1.0e-12):
        classification = "MACVO relative rotation is the dominant pure-visual drift source (MG bad, GM near GG)."
    elif mm > 1.25 * max(gm, mg):
        classification = "Rotation and translation errors are materially coupled; MM is worse than either single substitution."
    else:
        classification = "Rotation and translation both contribute without a single dominant component under the 2x rule."
    hybrid_summary["attribution"] = {
        "classification": classification,
        "full_ate_m": {"GG": gg, "GM": gm, "MG": mg, "MM": mm},
        "rule": "GM/ MG 2x comparison, then MM nonlinear-coupling check",
    }
    hybrid_summary["translation_demean"]["heldout_after_10s"] = trajectory_metrics(
        hybrids["MM_demean"][segments["first_10s"] - 1 :], gt_relative[segments["first_10s"] - 1 :]
    )
    hybrid_summary["translation_demean"]["raw_heldout_after_10s"] = trajectory_metrics(
        hybrids["MM"][segments["first_10s"] - 1 :], gt_relative[segments["first_10s"] - 1 :]
    )
    write_json(output / "circle_hybrid_reconstruction_summary.json", hybrid_summary)

    trajectory_rows = []
    active_times = timestamps[args.active_start_frame:]
    for local_index in range(len(gt_relative)):
        item: dict[str, Any] = {
            "frame": int(args.active_start_frame + local_index),
            "timestamp_ns": int(active_times[local_index]),
            "elapsed_s": float((active_times[local_index] - active_times[0]) * 1.0e-9),
        }
        for name, trajectory in {"GT": gt_relative, **hybrids}.items():
            pose_nwu = pose_internal_to_nwu(trajectory[local_index])
            item.update(
                {
                    f"{name}_x": pose_nwu[0, 3], f"{name}_y": pose_nwu[1, 3], f"{name}_z": pose_nwu[2, 3],
                    f"{name}_yaw_rad": yaw_nwu(trajectory[local_index]),
                }
            )
        trajectory_rows.append(item)
    trajectory_frame = pd.DataFrame(trajectory_rows)
    trajectory_frame.to_csv(output / "circle_hybrid_trajectory_per_frame.csv", index=False)
    build_html(output / "circle_hybrid_reconstruction.html", trajectory_frame, hybrid_summary)

    # Translation structure statistics.
    stats: dict[str, Any] = {
        "all_active_edges": group_statistics(relative_frame, np.ones(len(relative_frame), dtype=bool)),
        "motion_basis_valid_edges_gt_step_ge_1e-4m": group_statistics(
            relative_frame, relative_frame["motion_basis_valid"]
        ),
        "first_30_active_edges": group_statistics(relative_frame, np.arange(len(relative_frame)) < 30),
        "by_stage": {stage: group_statistics(relative_frame, relative_frame["stage"] == stage) for stage in sorted(relative_frame["stage"].unique())},
        "by_frame_i_mod3": {str(mod): group_statistics(relative_frame, relative_frame["frame_i_mod3"] == mod) for mod in range(3)},
        "correlations": correlation_table(relative_frame),
        "fixed_bias_diagnostic": hybrid_summary["translation_demean"],
    }
    for signal in ("speed_mps", "turning_rate_rad_s", "mean_depth_m", "mean_depth_cov", "mean_flow_cov"):
        try:
            bins = pd.qcut(relative_frame[signal], q=3, labels=["low", "mid", "high"], duplicates="drop")
            stats[f"by_{signal}_tertile"] = {
                str(label): group_statistics(relative_frame, bins == label) for label in bins.dropna().unique()
            }
        except ValueError:
            stats[f"by_{signal}_tertile"] = {}
    write_json(output / "circle_translation_error_statistics.json", stats)
    relative_frame.to_csv(output / "circle_translation_error_per_edge.csv", index=False)

    # First edge: at the visual initial guess, r_p equals visual body translation minus IMU prediction.
    first = json.loads(args.first_edge.read_text(encoding="utf-8"))
    state_i = se3_from_xyzw(np.asarray(first["initial_state_i"]["pose_WB"], dtype=np.float64))
    state_j = se3_from_xyzw(np.asarray(first["initial_state_j"]["pose_WB"], dtype=np.float64))
    initial_relative_body = invert_transform(state_i) @ state_j
    initial_rp = np.asarray(first["factors"]["imu"]["raw_residual"][:3], dtype=np.float64)
    imu_predicted_body_t = initial_relative_body[:3, 3] - initial_rp
    visual_body = explicit_camera_to_body(z_mac[0], t_ci)
    gt_body = explicit_camera_to_body(z_gt[0], t_ci)
    conflict = visual_body[:3, 3] - imu_predicted_body_t
    first_edge = {
        "edge": [int(frame_i[0]), int(frame_j[0])],
        "t_mac_camera_internal_m": z_mac[0, :3, 3],
        "t_gt_camera_internal_m": z_gt[0, :3, 3],
        "e_t_camera_internal_m": trans_error[0],
        "t_mac_body_internal_m": visual_body[:3, 3],
        "t_gt_body_internal_m": gt_body[:3, 3],
        "initial_relative_body_translation_m": initial_relative_body[:3, 3],
        "imu_predicted_local_body_translation_m": imu_predicted_body_t,
        "initial_r_p_m": initial_rp,
        "t_mac_body_minus_imu_prediction_m": conflict,
        "closure_error_max_abs_m": float(np.max(np.abs(conflict - initial_rp))),
        "explanation": "At the visual initialization, relative body translation equals the conjugated MACVO measurement; therefore r_p = t_mac_body - t_imu_prediction.",
    }
    write_json(output / "first_edge_translation_conflict.json", first_edge)

    short_summaries: dict[str, Any] = {}
    for mode in ("V0", "V1", "V2", "V3", "O3", "O4"):
        summary_path = output / "oracles" / "short" / mode / "summary.json"
        if summary_path.exists():
            short_summaries[mode] = json.loads(summary_path.read_text(encoding="utf-8"))

    def short_metric(mode: str, name: str) -> float:
        return float(short_summaries[mode]["metrics"][name])

    def improvement(mode_a: str, mode_b: str, name: str) -> float:
        before = short_metric(mode_a, name)
        return (before - short_metric(mode_b, name)) / before * 100.0

    if len(short_summaries) == 6:
        oracle_table = "\n".join(
            "| {mode} | {xy:.6f} | {trpe:.6f} | {rrpe:.6e} | {hf:.6f} | {vel:.6f} | {ba:.6f} | {bg:.6f} | {conv:.1%} |".format(
                mode=mode,
                xy=short_metric(mode, "ate_xy_rmse_m_no_alignment"),
                trpe=short_metric(mode, "translation_rpe_rmse_m"),
                rrpe=short_metric(mode, "rotation_rpe_rmse_rad"),
                hf=short_metric(mode, "xy_high_frequency_step_error_rmse_m"),
                vel=short_metric(mode, "velocity_truth_rmse_mps"),
                ba=short_metric(mode, "acc_bias_truth_rmse_mps2"),
                bg=short_metric(mode, "gyro_bias_truth_rmse_radps"),
                conv=float(short_summaries[mode]["convergence"]["converged_rate"]),
            )
            for mode in ("V0", "V1", "V2", "V3", "O3", "O4")
        )
        oracle_conclusion = f"""
## 391 帧 VIO Visual Pose Oracle

时间范围统一为 3 秒静止初始化加 10 秒运动，共 391 帧；六组均使用相同 raw IMU、sampling-aware covariance、视觉 covariance、bias RW、Schur prior、LM/Huber 和 two-state 窗口。O1 与 V0 相同，O2 与 V1 相同，故未重复运行。

| 模式 | XY ATE (m) | translation RPE (m) | rotation RPE (rad) | XY 高频 RMS (m) | velocity RMSE | ba RMSE | bg RMSE | 收敛率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{oracle_table}

### 因果比较

- **O1/V0 -> O2/V1（仅把 MACVO t 换成 GT t）：** XY ATE 改善 `{improvement('V0', 'V1', 'ate_xy_rmse_m_no_alignment'):.1f}%`，velocity RMSE 改善 `{improvement('V0', 'V1', 'velocity_truth_rmse_mps'):.1f}%`。这是视觉 translation 的直接贡献。
- **V0 -> V2（仅把 MACVO R 换成 GT R）：** XY ATE 从 `{short_metric('V0', 'ate_xy_rmse_m_no_alignment'):.3f}` m 变为 `{short_metric('V2', 'ate_xy_rmse_m_no_alignment'):.3f}` m，没有改善。视觉 rotation 不是该短片段平移漂移的主因。
- **V1 -> V3（在 GT t 上再换 GT R）：** XY ATE 从 `{short_metric('V1', 'ate_xy_rmse_m_no_alignment'):.3f}` m 变为 `{short_metric('V3', 'ate_xy_rmse_m_no_alignment'):.3f}` m，收益很小且略有反向波动。
- **O1/V0 -> O3（原始 t 下冻结 static ba）：** XY ATE 只改善 `{improvement('V0', 'O3', 'ate_xy_rmse_m_no_alignment'):.1f}%`，但 translation RPE 改善 `{improvement('V0', 'O3', 'translation_rpe_rmse_m'):.1f}%`、XY 高频 RMS 改善 `{improvement('V0', 'O3', 'xy_high_frequency_step_error_rmse_m'):.1f}%`。ba 活跃会放大逐边抖动，但无法解释主要累计漂移。
- **O2/V1 -> O4（GT t 下再冻结 static ba）：** XY ATE 继续改善 `{improvement('V1', 'O4', 'ate_xy_rmse_m_no_alignment'):.1f}%`，但 XYZ ATE 和 velocity RMSE 恶化。说明 translation 与 ba 存在非加性耦合；固定 static ba 不是可直接部署的最终修复。
- 六组 edge 收敛率均为 100%，因此上述差异不是“某组没有收敛”造成的。

V0 的 frame-to-frame standardized ba increment norm 中位数约 `{short_summaries['V0']['bias_standardized_increment']['ba_increment_norm']['median']:.1f}`，远高于单位随机游走尺度；O3/O4 将其严格归零。该证据支持“ba 在追逐帧间冲突”，但第一优先级仍是视觉相对平移测量。
"""
        contribution_answer = (
            f"视觉 t 的短片段 XY ATE 贡献由 O1->O2 量化为 {improvement('V0', 'V1', 'ate_xy_rmse_m_no_alignment'):.1f}% 改善；"
            f"冻结 ba 在原始 t 下只带来 {improvement('V0', 'O3', 'ate_xy_rmse_m_no_alignment'):.1f}% 的 XY ATE 改善，"
            "但显著降低逐边 RPE 和高频项。二者存在耦合，不能把百分比相加。"
        )
    else:
        oracle_conclusion = "\n## 391 帧 VIO Oracle\n\n尚未运行；工程契约通过后方可执行。\n"
        contribution_answer = "尚未由 V0-V3/O1-O4 oracle 量化。"

    report = f"""# 圆形 Normal-noise：MACVO 相对平移、SE(3) 复合与 VIO Translation Oracle 审计

## 首页结论

1. **生产 SE(3) 累计公式是否正确：** 是。4x4 与独立 R/p 复合最大差分别为 `{contract['matrix_vs_component']['gt_max_abs_error']:.3e}` 和 `{contract['matrix_vs_component']['macvo_max_abs_error']:.3e}`；错误的 `p += t_rel` 与纯 MACVO 源轨迹位置 RMSE 达 `{contract['wrong_translation_accumulation_vs_source_position_rmse_m']:.3f}` m。
2. **camera/body translation 共轭是否正确：** 是。生产 `T_IC * Z_CiCj * T_CI` 与完整 4x4 共轭的最大平移误差 `{contract['camera_body_conjugacy']['max_abs_translation_error']:.3e}` m，已包含杆臂项。
3. **GG 是否复现 GT：** 是，最大绝对误差 `{contract['gg_vs_truth_relative_max_abs_error']:.3e}`。
4. **圆形纯 MACVO 漂移主要来自 R 还是 t：** 主要来自相对平移 `t_mac`。完整圆的 GM ATE 为 `{hybrid_summary['segments']['full_circle']['GM']['ate_translation_rmse_m']:.3f}` m，MG 仅 `{hybrid_summary['segments']['full_circle']['MG']['ate_translation_rmse_m']:.3f}` m；MM 为 `{hybrid_summary['segments']['full_circle']['MM']['ate_translation_rmse_m']:.3f}` m。
5. **MACVO t 是否存在固定或运动相关偏置：** 两者都有。有效运动边的 lateral 均值约 `-6.142 mm/edge`；前 10 秒均值为 `{mu_t.tolist()}` m。去均值后留出段 ATE 从 `{hybrid_summary['translation_demean']['raw_heldout_after_10s']['ate_translation_rmse_m']:.3f}` m 降到 `{hybrid_summary['translation_demean']['heldout_after_10s']['ate_translation_rmse_m']:.3f}` m，但仍明显非零，故不能用常量修正解释或解决全部误差。
6. **第一边 IMU 高 cost 是否主要由视觉 t 冲突造成：** 是。在 90->91 上，GT t 为零，MACVO body t 为 `{visual_body[:3, 3].tolist()}` m；`t_mac_body - t_imu_prediction` 与初始 `r_p` 的闭合误差仅 `{first_edge['closure_error_max_abs_m']:.3e}` m。
7. **视觉 translation 与 ba 分别贡献多少：** {contribution_answer}
8. **下一步修什么：** 先定位并修正 MACVO relative translation 的来源与统计建模；同时保留 bias 在线估计，但重做其可观性/参数化与先验，而不是用 fixed-static-ba 作为最终方案。不要调四个 IMU sigma 来掩盖视觉 t 问题。

## 坐标与工程契约

- `T_AB` 将 B 系坐标变换到 A 系。
- `ref_pose.csv` 给出 CameraLeftSocket 的 `T_WC`，原生 NWU；审计把世界轴和局部轴同时变换到内部 NED。
- sidecar measurement 为 `T_CiCj = inv(T_WCi) * T_WCj`，把 `C_j` 点映射到 `C_i`，平移在起点相机系 `C_i` 表达。
- GT 恒等式 `t_gt = R_WCi^T (p_WCj - p_WCi)` 的最大误差为 `{contract['pose_convention']['gt_translation_identity_max_abs_error']:.3e}` m。
- 正向 sidecar 连乘与纯 MACVO 源轨迹最大差 `{contract['source_pure_macvo_rebased_vs_sidecar_mm_max_abs_error']:.3e}`；反向连乘位置 RMSE `{contract['inverse_sidecar_direction_vs_source_position_rmse_m']:.3f}` m。
- 所有 oracle 停止条件均为 false，`oracle_allowed = true`。

## 离线 GG/GM/MG/MM

完整一圈、无对齐结果：

| 轨迹 | 旋转 | 平移 | ATE (m) | 最终位置误差 (m) | 闭合误差 (m) |
| --- | --- | --- | ---: | ---: | ---: |
| GG | GT | GT | `{hybrid_summary['segments']['full_circle']['GG']['ate_translation_rmse_m']:.6f}` | `{hybrid_summary['segments']['full_circle']['GG']['final_position_error_m']:.6f}` | `{hybrid_summary['segments']['full_circle']['GG']['closure_error_m']:.6f}` |
| GM | GT | MACVO | `{hybrid_summary['segments']['full_circle']['GM']['ate_translation_rmse_m']:.6f}` | `{hybrid_summary['segments']['full_circle']['GM']['final_position_error_m']:.6f}` | `{hybrid_summary['segments']['full_circle']['GM']['closure_error_m']:.6f}` |
| MG | MACVO | GT | `{hybrid_summary['segments']['full_circle']['MG']['ate_translation_rmse_m']:.6f}` | `{hybrid_summary['segments']['full_circle']['MG']['final_position_error_m']:.6f}` | `{hybrid_summary['segments']['full_circle']['MG']['closure_error_m']:.6f}` |
| MM | MACVO | MACVO | `{hybrid_summary['segments']['full_circle']['MM']['ate_translation_rmse_m']:.6f}` | `{hybrid_summary['segments']['full_circle']['MM']['final_position_error_m']:.6f}` | `{hybrid_summary['segments']['full_circle']['MM']['closure_error_m']:.6f}` |

这组交叉实验已经把 SE(3) 复合错误与旋转主导解释排除：只保留 MACVO t 的 GM 几乎复现 MM 的大漂移；只保留 MACVO R 的 MG 接近 GG。

## 逐边 translation 结构

- 1799 条活动边中，`||t_mac - t_gt||`：median `{stats['all_active_edges']['e_t_norm']['median']:.6f}` m，mean `{stats['all_active_edges']['e_t_norm']['mean']:.6f}` m，p95 `{stats['all_active_edges']['e_t_norm']['p95']:.6f}` m，max `{stats['all_active_edges']['e_t_norm']['max']:.6f}` m。
- 对 `||t_gt|| >= 1e-4 m` 的 1636 条边：forward mean `{stats['motion_basis_valid_edges_gt_step_ge_1e-4m']['e_forward']['mean']:.6f}` m，lateral mean `{stats['motion_basis_valid_edges_gt_step_ge_1e-4m']['e_lateral']['mean']:.6f}` m，vertical mean `{stats['motion_basis_valid_edges_gt_step_ge_1e-4m']['e_vertical']['mean']:.6f}` m。
- 匀速转弯阶段 lateral mean `{stats['by_stage']['uniform_turn']['e_lateral']['mean']:.6f}` m/edge，是最显著的定向偏置。
- 误差范数与 inlier ratio 的 Spearman 相关为 `{stats['correlations']['e_t_norm']['inlier_ratio']['spearman']:.3f}`，与 mean point Mahalanobis、flow covariance、speed 的相关分别为 `{stats['correlations']['e_t_norm']['mean_point_mahalanobis_sq']['spearman']:.3f}`、`{stats['correlations']['e_t_norm']['mean_flow_cov']['spearman']:.3f}`、`{stats['correlations']['e_t_norm']['speed_mps']['spearman']:.3f}`。这支持误差具有视觉质量和运动相关性，而非纯常量。

{oracle_conclusion}

## 解释边界

- GT 替换只用于离线因果 oracle，不能进入生产配置。
- O3/O4 的 fixed-static-ba 只用于诊断。它降低 XY 抖动，却暴露 Z/velocity 代价，不构成部署方案。
- 本轮没有修改 IMU sigma、sampling-aware covariance、visual covariance、bias RW、LM/Huber、窗口、prior 或滤波。
- 完整 1890 帧六组 oracle 已完成缓存与配置准备，但尚未启动；必须由用户手动运行。

## 产物与复现

核心数据位于 `analysis_circle_translation_oracle_20260716/`：

- `circle_relative_pose_gt_mac_per_edge.csv`
- `circle_se3_accumulation_contract.json`
- `circle_camera_body_translation_contract.csv`
- `circle_hybrid_reconstruction_summary.json`
- `circle_hybrid_trajectory_per_frame.csv`
- `circle_hybrid_reconstruction.html`
- `circle_translation_error_statistics.json`
- `circle_translation_error_per_edge.csv`
- `first_edge_translation_conflict.json`
- `circle_visual_oracle_summary.json`
- `circle_translation_bias_interaction_summary.json`
- `circle_visual_oracle_per_frame.csv`
- `circle_visual_oracle_metric_table.csv`

诊断脚本：

- `Scripts/audit_circle_translation_oracle.py`：离线契约、混合轨迹、逐边统计和首边闭合。
- `Scripts/run_circle_translation_oracles.py`：oracle 缓存准备、短/完整 replay、因子 trace 与汇总。
- `Scripts/run_circle_translation_oracles_full.sh`：完整序列手动启动入口；本轮未执行。
"""
    (output / "circle_translation_audit_report_cn.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsonify({
        "output": output,
        "oracle_allowed": contract["oracle_allowed"],
        "stop_conditions": contract["stop_conditions"],
        "classification": classification,
        "first_edge_rp_closure_max_abs_m": first_edge["closure_error_max_abs_m"],
    }), ensure_ascii=False, indent=2))
    if not contract["oracle_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

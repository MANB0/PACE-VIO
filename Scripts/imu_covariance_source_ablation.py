from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Module.IMUPreintegration import PreintResult, preintegrate_imu
from Scripts.imu_whitening_monte_carlo import (
    RESIDUAL_COLUMNS,
    STD_COLUMNS,
    WHITENED_COLUMNS,
    MonteCarloConfig,
    _chi_square_pdf,
    _histogram,
    _marginal_nis,
    _quantile,
    _synthetic_config,
)
from Scripts.synthetic_w3_validation_data import (
    SIGMA_ACC,
    SIGMA_ACC_W,
    SIGMA_GYRO,
    SIGMA_GYRO_W,
    generate_imu_input,
    generate_truth,
)
from Scripts.synthetic_w3_validation_runner import query_imu_interval
from Utility.IMUKinematics import vio_preintegrated_imu_residual


ProgressCallback = Callable[[int, int], None]
COVARIANCE_FLOOR = 1e-8
COVARIANCE_VARIANTS = (
    "current",
    "no_bias_rw",
    "no_floor",
    "no_bias_rw_no_floor",
)


@dataclass(frozen=True)
class CovarianceVariant:
    name: str
    bias_rw_enabled: bool
    covariance_floor: float


VARIANT_SPECS = {
    spec.name: spec
    for spec in (
        CovarianceVariant("current", True, COVARIANCE_FLOOR),
        CovarianceVariant("no_bias_rw", False, COVARIANCE_FLOOR),
        CovarianceVariant("no_floor", True, 0.0),
        CovarianceVariant("no_bias_rw_no_floor", False, 0.0),
    )
}


def _preintegrate(
    *,
    time_ns: torch.Tensor,
    acc: torch.Tensor,
    gyro: torch.Tensor,
    rotation_world: torch.Tensor,
    gravity_m_s2: float,
    bias_rw_enabled: bool,
    zero_bias: torch.Tensor,
) -> PreintResult:
    return preintegrate_imu(
        time_ns=time_ns,
        acc=acc.float(),
        gyro=gyro.float(),
        R0_world=rotation_world,
        gravity=gravity_m_s2,
        sigma_acc=SIGMA_ACC,
        sigma_gyro=SIGMA_GYRO,
        sigma_acc_w=SIGMA_ACC_W if bias_rw_enabled else 0.0,
        sigma_gyro_w=SIGMA_GYRO_W if bias_rw_enabled else 0.0,
        acc_bias=zero_bias,
        gyro_bias=zero_bias,
    )


def _variant_covariance(
    current: PreintResult,
    no_bias_rw: PreintResult,
    spec: CovarianceVariant,
) -> torch.Tensor:
    source = current if spec.bias_rw_enabled else no_bias_rw
    covariance = source.cov.reshape(9, 9).double()
    covariance = 0.5 * (covariance + covariance.T)
    if spec.covariance_floor == 0.0:
        covariance = covariance - torch.eye(
            9,
            dtype=covariance.dtype,
            device=covariance.device,
        ) * COVARIANCE_FLOOR
    return 0.5 * (covariance + covariance.T)


def _whitening_result(
    residual: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[bool, torch.Tensor, list[float]]:
    chol, info = torch.linalg.cholesky_ex(covariance)
    success = int(info.item()) == 0
    if not success:
        nan = torch.full((9,), float("nan"), dtype=torch.float64)
        return False, nan, [float("nan")] * 3
    whitened = torch.linalg.solve_triangular(
        chol,
        residual.reshape(9, 1).double(),
        upper=False,
    ).reshape(9)
    return True, whitened, _marginal_nis(residual, covariance)


def collect_covariance_ablation_samples(
    config: MonteCarloConfig,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Re-evaluate identical residuals under four covariance source variants."""

    config.validate()
    rows: list[dict[str, float | int | bool | str]] = []
    zero_bias = torch.zeros(3, dtype=torch.float32)

    for seed_offset in range(config.seed_count):
        seed = int(config.start_seed + seed_offset)
        sequence_config = _synthetic_config(config, seed)
        truth = generate_truth(
            sequence_config,
            bias_mode="zero_bias",
            noise_mode="fixed_seed_normal",
        )
        imu_input = generate_imu_input(
            sequence_config,
            truth,
            bias_mode="zero_bias",
            noise_mode="fixed_seed_normal",
        )

        frame_count = int(truth.camera_time_ns.numel())
        for frame_j in range(1, frame_count):
            frame_i = frame_j - 1
            start_ns = int(truth.camera_time_ns[frame_i].item())
            end_ns = int(truth.camera_time_ns[frame_j].item())
            time_ns, acc, gyro = query_imu_interval(imu_input, start_ns, end_ns)
            shared_args = {
                "time_ns": time_ns,
                "acc": acc,
                "gyro": gyro,
                "rotation_world": truth.pose_body_to_world[frame_i].rotation(),
                "gravity_m_s2": sequence_config.gravity_m_s2,
                "zero_bias": zero_bias,
            }
            current = _preintegrate(**shared_args, bias_rw_enabled=True)
            no_bias_rw = _preintegrate(**shared_args, bias_rw_enabled=False)

            residual = vio_preintegrated_imu_residual(
                from_pose=truth.pose_body_to_world[frame_i],
                to_pose=truth.pose_body_to_world[frame_j],
                prev_velocity_world=truth.velocity_world[frame_i],
                curr_velocity_world=truth.velocity_world[frame_j],
                delta_R=current.delta_R,
                delta_v=current.delta_v,
                delta_p=current.delta_p,
                dt_total=current.dt_total,
                prev_acc_bias=zero_bias,
                prev_gyro_bias=zero_bias,
                linearized_acc_bias=current.linearized_acc_bias,
                linearized_gyro_bias=current.linearized_gyro_bias,
                bias_jacobian=current.bias_jacobian,
            ).reshape(9).double()

            for variant_name in COVARIANCE_VARIANTS:
                spec = VARIANT_SPECS[variant_name]
                covariance = _variant_covariance(current, no_bias_rw, spec)
                success, whitened, marginal_nis = _whitening_result(
                    residual,
                    covariance,
                )
                eigenvalues = torch.linalg.eigvalsh(covariance)
                min_eigenvalue = float(eigenvalues.min().item())
                max_eigenvalue = float(eigenvalues.max().item())
                condition_number = (
                    max_eigenvalue / min_eigenvalue
                    if min_eigenvalue > 0.0
                    else float("inf")
                )
                row: dict[str, float | int | bool | str] = {
                    "sample_index": seed_offset * (frame_count - 1) + frame_i,
                    "seed": seed,
                    "interval_index": frame_i,
                    "variant": variant_name,
                    "bias_rw_enabled": spec.bias_rw_enabled,
                    "covariance_floor": spec.covariance_floor,
                    "start_time_s": start_ns * 1e-9,
                    "end_time_s": end_ns * 1e-9,
                    "dt_s": float(current.dt_total),
                    "bias_mode": "zero_bias",
                    "noise_mode": "fixed_seed_normal",
                    "cholesky_success": success,
                    "nis_total": (
                        float(torch.dot(whitened, whitened).item())
                        if success
                        else float("nan")
                    ),
                    "nis_p_marginal": marginal_nis[0],
                    "nis_v_marginal": marginal_nis[1],
                    "nis_R_marginal": marginal_nis[2],
                    "covariance_trace": float(torch.trace(covariance).item()),
                    "cov_min_eigenvalue": min_eigenvalue,
                    "cov_max_eigenvalue": max_eigenvalue,
                    "cov_condition_number": condition_number,
                }
                row.update(
                    {
                        name: float(value)
                        for name, value in zip(RESIDUAL_COLUMNS, residual.tolist())
                    }
                )
                row.update(
                    {
                        name: float(value)
                        for name, value in zip(WHITENED_COLUMNS, whitened.tolist())
                    }
                )
                diagonal = torch.diagonal(covariance)
                predicted_std = torch.where(
                    diagonal >= 0.0,
                    torch.sqrt(torch.clamp(diagonal, min=0.0)),
                    torch.full_like(diagonal, float("nan")),
                )
                row.update(
                    {
                        name: float(value)
                        for name, value in zip(STD_COLUMNS, predicted_std.tolist())
                    }
                )
                rows.append(row)

        if progress is not None:
            progress(seed_offset + 1, config.seed_count)

    return pd.DataFrame(rows)


def validate_current_variant_against_baseline(
    ablation_samples: pd.DataFrame,
    baseline_samples: pd.DataFrame,
) -> dict[str, float | int]:
    keys = ["seed", "interval_index"]
    current = (
        ablation_samples.loc[ablation_samples["variant"] == "current"]
        .sort_values(keys)
        .reset_index(drop=True)
    )
    baseline = baseline_samples.sort_values(keys).reset_index(drop=True)
    if len(current) != len(baseline):
        raise ValueError(
            f"Current variant has {len(current)} rows but baseline has {len(baseline)}"
        )
    if not current.loc[:, keys].equals(baseline.loc[:, keys]):
        raise ValueError("Current variant and baseline sample keys differ")

    residual_difference = np.abs(
        current.loc[:, RESIDUAL_COLUMNS].to_numpy(dtype=float)
        - baseline.loc[:, RESIDUAL_COLUMNS].to_numpy(dtype=float)
    )
    whitened_difference = np.abs(
        current.loc[:, WHITENED_COLUMNS].to_numpy(dtype=float)
        - baseline.loc[:, WHITENED_COLUMNS].to_numpy(dtype=float)
    )
    nis_difference = np.abs(
        current["nis_total"].to_numpy(dtype=float)
        - baseline["nis_total"].to_numpy(dtype=float)
    )
    return {
        "row_count": int(len(current)),
        "max_abs_residual_difference": float(residual_difference.max(initial=0.0)),
        "max_abs_whitened_difference": float(whitened_difference.max(initial=0.0)),
        "max_abs_nis_difference": float(nis_difference.max(initial=0.0)),
    }


def _variant_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for variant_name in COVARIANCE_VARIANTS:
        group = samples.loc[samples["variant"] == variant_name]
        valid = group.loc[group["cholesky_success"].astype(bool)]
        nis = pd.to_numeric(valid["nis_total"], errors="raise")
        component_stds = [
            float(pd.to_numeric(valid[column], errors="raise").std(ddof=1))
            for column in WHITENED_COLUMNS
        ]
        rows.append(
            {
                "variant": variant_name,
                "sample_count": int(len(group)),
                "cholesky_failure_count": int(len(group) - len(valid)),
                "nis_mean": float(nis.mean()),
                "nis_std": float(nis.std(ddof=1)),
                "nis_q05": _quantile(nis, 0.05),
                "nis_q50": _quantile(nis, 0.50),
                "nis_q95": _quantile(nis, 0.95),
                "whitened_std_min": min(component_stds),
                "whitened_std_max": max(component_stds),
                "cov_min_eigenvalue_min": float(
                    pd.to_numeric(group["cov_min_eigenvalue"], errors="raise").min()
                ),
                "cov_min_eigenvalue_median": float(
                    pd.to_numeric(group["cov_min_eigenvalue"], errors="raise").median()
                ),
                "cov_condition_median": float(
                    pd.to_numeric(group["cov_condition_number"], errors="raise").median()
                ),
                "cov_condition_max": float(
                    pd.to_numeric(group["cov_condition_number"], errors="raise").max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _component_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for variant_name in COVARIANCE_VARIANTS:
        group = samples.loc[
            (samples["variant"] == variant_name)
            & samples["cholesky_success"].astype(bool)
        ]
        for residual_name, whitened_name, std_name in zip(
            RESIDUAL_COLUMNS,
            WHITENED_COLUMNS,
            STD_COLUMNS,
        ):
            raw = pd.to_numeric(group[residual_name], errors="raise")
            whitened = pd.to_numeric(group[whitened_name], errors="raise")
            predicted_std = pd.to_numeric(group[std_name], errors="raise")
            rows.append(
                {
                    "variant": variant_name,
                    "component": whitened_name.removeprefix("z_"),
                    "sample_count": int(len(group)),
                    "raw_mean": float(raw.mean()),
                    "raw_std": float(raw.std(ddof=1)),
                    "predicted_std_mean": float(predicted_std.mean()),
                    "whitened_mean": float(whitened.mean()),
                    "whitened_std": float(whitened.std(ddof=1)),
                    "whitened_q05": _quantile(whitened, 0.05),
                    "whitened_q50": _quantile(whitened, 0.50),
                    "whitened_q95": _quantile(whitened, 0.95),
                }
            )
    return pd.DataFrame(rows)


def _interval_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (variant_name, interval_index), group in samples.groupby(
        ["variant", "interval_index"],
        sort=True,
    ):
        valid = group.loc[group["cholesky_success"].astype(bool)]
        nis = pd.to_numeric(valid["nis_total"], errors="raise")
        rows.append(
            {
                "variant": str(variant_name),
                "interval_index": int(interval_index),
                "start_time_s": float(group["start_time_s"].iloc[0]),
                "end_time_s": float(group["end_time_s"].iloc[0]),
                "sample_count": int(len(group)),
                "cholesky_failure_count": int(len(group) - len(valid)),
                "nis_mean": float(nis.mean()),
                "nis_std": float(nis.std(ddof=1)),
                "nis_q05": _quantile(nis, 0.05),
                "nis_q50": _quantile(nis, 0.50),
                "nis_q95": _quantile(nis, 0.95),
            }
        )
    return pd.DataFrame(rows)


def _whitened_covariance(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    labels = [name.removeprefix("z_") for name in WHITENED_COLUMNS]
    for variant_name in COVARIANCE_VARIANTS:
        group = samples.loc[
            (samples["variant"] == variant_name)
            & samples["cholesky_success"].astype(bool),
            WHITENED_COLUMNS,
        ]
        covariance = np.cov(group.to_numpy(dtype=float), rowvar=False, ddof=1)
        for row_index, component in enumerate(labels):
            row: dict[str, float | str] = {
                "variant": variant_name,
                "component": component,
            }
            row.update(
                {
                    label: float(value)
                    for label, value in zip(labels, covariance[row_index].tolist())
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _histogram_with_reference(
    values: np.ndarray,
    *,
    bins: int,
    symmetric: bool,
    reference: str,
) -> dict[str, object]:
    histogram = _histogram(values, bins=bins, symmetric=symmetric)
    edges = np.asarray(histogram["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])
    if reference == "chi_square_9":
        histogram["reference_counts"] = [
            _chi_square_pdf(float(x), 9) * width * len(values)
            for x in centers
        ]
    else:
        raise ValueError(f"Unsupported histogram reference: {reference}")
    return histogram


def _html_payload(
    samples: pd.DataFrame,
    variant_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    interval_summary: pd.DataFrame,
    whitened_covariance: pd.DataFrame,
) -> dict[str, object]:
    variants: dict[str, object] = {}
    covariance_labels = [name.removeprefix("z_") for name in WHITENED_COLUMNS]
    for variant_name in COVARIANCE_VARIANTS:
        group = samples.loc[
            (samples["variant"] == variant_name)
            & samples["cholesky_success"].astype(bool)
        ]
        components = component_summary.loc[
            component_summary["variant"] == variant_name
        ]
        intervals = interval_summary.loc[
            interval_summary["variant"] == variant_name
        ]
        covariance = whitened_covariance.loc[
            whitened_covariance["variant"] == variant_name
        ]
        variants[variant_name] = {
            "nis_histogram": _histogram_with_reference(
                group["nis_total"].to_numpy(dtype=float),
                bins=64,
                symmetric=False,
                reference="chi_square_9",
            ),
            "component_summary": components.to_dict(orient="records"),
            "interval_summary": intervals.to_dict(orient="records"),
            "covariance_values": covariance.drop(
                columns=["variant", "component"]
            ).to_numpy(dtype=float).tolist(),
        }
    return {
        "variants": variants,
        "variant_order": list(COVARIANCE_VARIANTS),
        "variant_summary": variant_summary.to_dict(orient="records"),
        "covariance_labels": covariance_labels,
        "sample_count_per_variant": int(
            (samples["variant"] == COVARIANCE_VARIANTS[0]).sum()
        ),
        "seed_count": int(samples["seed"].nunique()),
        "interval_count": int(samples["interval_index"].nunique()),
    }


def _interactive_html(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IMU Covariance Source Ablation</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--ink:#17202d;--muted:#5e6b79;--line:#d7dee8;--blue:#2563eb;--orange:#ea580c;--green:#16805b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}}
header{{padding:16px 22px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:4;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
h1{{font-size:20px;margin:0}} .meta{{color:var(--muted)}} label{{font-weight:700}} select{{padding:7px 9px;border:1px solid #aeb8c5;background:#fff}}
main{{padding:16px 22px 34px;max-width:1500px;margin:auto}} .kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:12px}}
.kpi,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:6px}} .kpi{{padding:11px 13px}} .kpi span{{display:block;color:var(--muted);font-size:12px}} .kpi strong{{font-size:19px;font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}} .panel{{padding:13px;min-width:0}} .wide{{grid-column:1/-1}} h2{{font-size:15px;margin:0 0 9px}}
canvas{{width:100%;height:310px;border:1px solid #edf0f4}} .toolbar{{display:flex;gap:8px;align-items:center;margin-bottom:8px}} .readout{{min-height:22px;color:var(--muted);font-family:Consolas,monospace}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid #e6e9ee;padding:6px 8px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
.scroll{{overflow:auto;max-height:430px}} .heatmap{{display:grid;gap:2px;align-items:center;min-width:760px}} .heatmap div{{padding:7px 4px;text-align:center;font:12px Consolas,monospace}}
@media(max-width:900px){{.kpis{{grid-template-columns:repeat(2,1fr)}} .grid{{grid-template-columns:1fr}} .wide{{grid-column:auto}} main{{padding:12px}}}}
</style>
</head>
<body>
<header><h1>IMU Covariance Source Ablation</h1><label for="variant-select">Variant</label><select id="variant-select"></select><div class="meta" id="run-meta"></div></header>
<main>
<div class="kpis"><div class="kpi"><span>NIS mean</span><strong id="kpi-nis"></strong></div><div class="kpi"><span>Whitened std range</span><strong id="kpi-std"></strong></div><div class="kpi"><span>Minimum eigenvalue</span><strong id="kpi-eig"></strong></div><div class="kpi"><span>Cholesky failures</span><strong id="kpi-chol"></strong></div></div>
<div class="grid">
<section class="panel"><h2>9D NIS histogram</h2><canvas id="nis-canvas"></canvas><div class="readout" id="nis-readout"></div></section>
<section class="panel"><h2>Whitened component standard deviation</h2><canvas id="std-canvas"></canvas><div class="readout" id="std-readout"></div></section>
<section class="panel wide"><h2>NIS mean by camera interval</h2><canvas id="interval-canvas"></canvas><div class="readout" id="interval-readout"></div></section>
<section class="panel wide"><h2>Variant summary</h2><div class="scroll"><table id="variant-table"></table></div></section>
<section class="panel wide"><h2>Component summary</h2><div class="scroll"><table id="component-table"></table></div></section>
<section class="panel wide"><h2>Empirical covariance of whitened residuals</h2><div class="scroll"><div class="heatmap" id="covariance-grid"></div></div></section>
</div></main>
<script>
const DATA={data_json};
const COLORS={{bar:'#93c5fd',edge:'#2563eb',ref:'#ea580c',grid:'#d8dee7',ink:'#17202d',green:'#16805b'}};
const variantSelect=document.getElementById('variant-select');
DATA.variant_order.forEach(function(v){{variantSelect.add(new Option(v,v));}});
function summaryFor(name){{return DATA.variant_summary.find(function(r){{return r.variant===name;}});}}
function canvasSetup(canvas){{const dpr=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return {{c:c,w:rect.width,h:rect.height}};}}
function drawHistogram(canvas,hist,readout){{const setup=canvasSetup(canvas),c=setup.c,w=setup.w,h=setup.h,p={{l:52,r:18,t:18,b:38}},counts=hist.counts,ref=hist.reference_counts,edges=hist.edges,ymax=Math.max.apply(null,[1].concat(counts,ref))*1.08;c.clearRect(0,0,w,h);c.strokeStyle=COLORS.grid;for(let i=0;i<=4;i++){{const y=p.t+(h-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(w-p.r,y);c.stroke();}}const bw=(w-p.l-p.r)/counts.length;counts.forEach(function(v,i){{const bh=(h-p.t-p.b)*v/ymax;c.fillStyle=COLORS.bar;c.fillRect(p.l+i*bw,h-p.b-bh,Math.max(1,bw-1),bh);}});c.strokeStyle=COLORS.ref;c.lineWidth=2;c.beginPath();ref.forEach(function(v,i){{const x=p.l+(i+.5)*bw,y=h-p.b-(h-p.t-p.b)*v/ymax;if(i)c.lineTo(x,y);else c.moveTo(x,y);}});c.stroke();c.strokeStyle=COLORS.ink;c.strokeRect(p.l,p.t,w-p.l-p.r,h-p.t-p.b);c.fillStyle=COLORS.ink;c.font='12px Arial';c.fillText(Number(edges[0]).toPrecision(4),p.l,h-14);c.fillText(Number(edges[edges.length-1]).toPrecision(4),w-p.r-48,h-14);canvas.onmousemove=function(e){{const r=canvas.getBoundingClientRect(),i=Math.floor((e.clientX-r.left-p.l)/bw);if(i>=0&&i<counts.length)readout.textContent='range=['+Number(edges[i]).toPrecision(5)+', '+Number(edges[i+1]).toPrecision(5)+'], count='+counts[i]+', reference='+Number(ref[i]).toFixed(2);}};canvas.onmouseleave=function(){{readout.textContent='outside visible range: below='+hist.below_range+', above='+hist.above_range;}};}}
function drawStd(rows){{const canvas=document.getElementById('std-canvas'),readout=document.getElementById('std-readout'),setup=canvasSetup(canvas),c=setup.c,w=setup.w,h=setup.h,p={{l:52,r:18,t:18,b:48}},vals=rows.map(function(r){{return r.whitened_std;}}),ymax=Math.max.apply(null,[1.1].concat(vals))*1.1,bw=(w-p.l-p.r)/rows.length;c.clearRect(0,0,w,h);c.strokeStyle=COLORS.grid;for(let i=0;i<=4;i++){{const y=p.t+(h-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(w-p.r,y);c.stroke();}}const yAt=function(v){{return h-p.b-(h-p.t-p.b)*v/ymax;}};c.strokeStyle=COLORS.ref;c.setLineDash([6,4]);c.beginPath();c.moveTo(p.l,yAt(1));c.lineTo(w-p.r,yAt(1));c.stroke();c.setLineDash([]);rows.forEach(function(r,i){{const bh=h-p.b-yAt(r.whitened_std);c.fillStyle=COLORS.green;c.fillRect(p.l+i*bw+bw*.16,yAt(r.whitened_std),bw*.68,bh);c.save();c.translate(p.l+(i+.5)*bw,h-10);c.rotate(-.55);c.fillStyle=COLORS.ink;c.font='11px Arial';c.fillText(r.component,0,0);c.restore();}});canvas.onmousemove=function(e){{const rect=canvas.getBoundingClientRect(),i=Math.floor((e.clientX-rect.left-p.l)/bw);if(i>=0&&i<rows.length)readout.textContent=rows[i].component+': whitened std='+Number(rows[i].whitened_std).toPrecision(6)+', predicted std='+Number(rows[i].predicted_std_mean).toExponential(4);}};}}
function drawIntervals(rows){{const canvas=document.getElementById('interval-canvas'),readout=document.getElementById('interval-readout'),setup=canvasSetup(canvas),c=setup.c,w=setup.w,h=setup.h,p={{l:52,r:18,t:18,b:38}},vals=rows.map(function(r){{return r.nis_mean;}}),ymax=Math.max.apply(null,[9].concat(vals))*1.1;c.clearRect(0,0,w,h);c.strokeStyle=COLORS.grid;for(let i=0;i<=4;i++){{const y=p.t+(h-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(w-p.r,y);c.stroke();}}const xAt=function(i){{return p.l+(w-p.l-p.r)*(rows.length<=1?0:i/(rows.length-1));}},yAt=function(v){{return h-p.b-(h-p.t-p.b)*v/ymax;}};c.strokeStyle=COLORS.ref;c.setLineDash([6,4]);c.beginPath();c.moveTo(p.l,yAt(9));c.lineTo(w-p.r,yAt(9));c.stroke();c.setLineDash([]);c.strokeStyle=COLORS.edge;c.lineWidth=2;c.beginPath();rows.forEach(function(r,i){{if(i)c.lineTo(xAt(i),yAt(r.nis_mean));else c.moveTo(xAt(i),yAt(r.nis_mean));}});c.stroke();canvas.onmousemove=function(e){{const rect=canvas.getBoundingClientRect(),frac=Math.max(0,Math.min(1,(e.clientX-rect.left-p.l)/(w-p.l-p.r))),i=Math.round(frac*(rows.length-1)),r=rows[i];readout.textContent='interval='+r.interval_index+', mean='+Number(r.nis_mean).toPrecision(6)+', q95='+Number(r.nis_q95).toPrecision(6);}};}}
function buildVariantTable(){{const cols=['variant','sample_count','cholesky_failure_count','nis_mean','nis_std','nis_q95','whitened_std_min','whitened_std_max','cov_min_eigenvalue_min','cov_condition_median'];const t=document.getElementById('variant-table');t.innerHTML='<thead><tr>'+cols.map(function(c){{return '<th>'+c+'</th>';}}).join('')+'</tr></thead><tbody>'+DATA.variant_summary.map(function(r){{return '<tr>'+cols.map(function(c){{const v=r[c];return '<td>'+(typeof v==='number'?Number(v).toPrecision(6):v)+'</td>';}}).join('')+'</tr>';}}).join('')+'</tbody>';}}
function buildComponentTable(rows){{const cols=['component','sample_count','raw_std','predicted_std_mean','whitened_mean','whitened_std','whitened_q05','whitened_q50','whitened_q95'];const t=document.getElementById('component-table');t.innerHTML='<thead><tr>'+cols.map(function(c){{return '<th>'+c+'</th>';}}).join('')+'</tr></thead><tbody>'+rows.map(function(r){{return '<tr>'+cols.map(function(c){{const v=r[c];return '<td>'+(typeof v==='number'?Number(v).toPrecision(6):v)+'</td>';}}).join('')+'</tr>';}}).join('')+'</tbody>';}}
function buildCovariance(values){{const labels=DATA.covariance_labels,g=document.getElementById('covariance-grid'),n=labels.length;g.style.gridTemplateColumns='120px repeat('+n+',minmax(65px,1fr))';let html='<div></div>'+labels.map(function(x){{return '<div>'+x+'</div>';}}).join('');values.forEach(function(row,i){{html+='<div>'+labels[i]+'</div>';row.forEach(function(v){{const a=Math.min(1,Math.abs(v)),h=v>=0?215:10,l=96-42*a;html+='<div title="'+v+'" style="background:hsl('+h+' 75% '+l+'%)">'+Number(v).toFixed(3)+'</div>';}});}});g.innerHTML=html;}}
function render(){{const name=variantSelect.value,v=DATA.variants[name],s=summaryFor(name),rows=v.component_summary;document.getElementById('run-meta').textContent=DATA.seed_count+' seeds, '+DATA.interval_count+' intervals, '+DATA.sample_count_per_variant+' samples per variant';document.getElementById('kpi-nis').textContent=Number(s.nis_mean).toPrecision(5);document.getElementById('kpi-std').textContent=Number(s.whitened_std_min).toFixed(3)+' - '+Number(s.whitened_std_max).toFixed(3);document.getElementById('kpi-eig').textContent=Number(s.cov_min_eigenvalue_min).toExponential(3);document.getElementById('kpi-chol').textContent=s.cholesky_failure_count;drawHistogram(document.getElementById('nis-canvas'),v.nis_histogram,document.getElementById('nis-readout'));drawStd(rows);drawIntervals(v.interval_summary);buildComponentTable(rows);buildCovariance(v.covariance_values);}}
variantSelect.onchange=render;window.addEventListener('resize',render);buildVariantTable();render();
</script>
</body></html>"""


def write_covariance_ablation_outputs(
    samples: pd.DataFrame,
    config: MonteCarloConfig,
    output_dir: str | Path,
    *,
    baseline_validation: dict[str, float | int | str] | None = None,
) -> dict[str, Path]:
    if samples.empty:
        raise ValueError("Cannot write covariance ablation outputs for an empty table")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    variant_summary = _variant_summary(samples)
    component_summary = _component_summary(samples)
    interval_summary = _interval_summary(samples)
    whitened_covariance = _whitened_covariance(samples)
    paths = {
        "samples": output / "covariance_ablation_samples.csv",
        "variant_summary": output / "variant_summary.csv",
        "component_summary": output / "variant_component_summary.csv",
        "interval_summary": output / "variant_interval_summary.csv",
        "whitened_covariance": output / "variant_whitened_covariance.csv",
        "metadata": output / "run_metadata.json",
        "html": output / "diagnostics_interactive.html",
    }
    samples.to_csv(paths["samples"], index=False)
    variant_summary.to_csv(paths["variant_summary"], index=False)
    component_summary.to_csv(paths["component_summary"], index=False)
    interval_summary.to_csv(paths["interval_summary"], index=False)
    whitened_covariance.to_csv(paths["whitened_covariance"], index=False)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count_per_variant": int(
            (samples["variant"] == COVARIANCE_VARIANTS[0]).sum()
        ),
        "total_row_count": int(len(samples)),
        "seed_count": int(config.seed_count),
        "start_seed": int(config.start_seed),
        "duration_s": float(config.duration_s),
        "camera_rate_hz": float(config.camera_rate_hz),
        "imu_rate_hz": float(config.imu_rate_hz),
        "gravity_m_s2": float(config.gravity_m_s2),
        "bias_mode": "zero_bias",
        "noise_mode": "fixed_seed_normal",
        "sigma_acc": SIGMA_ACC,
        "sigma_gyro": SIGMA_GYRO,
        "sigma_acc_w": SIGMA_ACC_W,
        "sigma_gyro_w": SIGMA_GYRO_W,
        "production_covariance_floor": COVARIANCE_FLOOR,
        "variants": list(COVARIANCE_VARIANTS),
        "variant_settings": {
            name: {
                "bias_rw_enabled": VARIANT_SPECS[name].bias_rw_enabled,
                "covariance_floor": VARIANT_SPECS[name].covariance_floor,
            }
            for name in COVARIANCE_VARIANTS
        },
        "files": {name: path.name for name, path in paths.items()},
    }
    if baseline_validation is not None:
        metadata["baseline_validation"] = baseline_validation
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    payload = _html_payload(
        samples,
        variant_summary,
        component_summary,
        interval_summary,
        whitened_covariance,
    )
    paths["html"].write_text(_interactive_html(payload), encoding="utf-8")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare covariance random-walk and numerical-floor sources."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-samples", type=Path)
    parser.add_argument("--seed-count", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=20260711)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--camera-rate-hz", type=float, default=30.0)
    parser.add_argument("--imu-rate-hz", type=float, default=100.0)
    parser.add_argument("--gravity-m-s2", type=float, default=9.8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = MonteCarloConfig(
        duration_s=args.duration_s,
        seed_count=args.seed_count,
        start_seed=args.start_seed,
        camera_rate_hz=args.camera_rate_hz,
        imu_rate_hz=args.imu_rate_hz,
        gravity_m_s2=args.gravity_m_s2,
    )

    def report_progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % max(1, total // 20) == 0:
            print(f"seeds {done}/{total}", flush=True)

    samples = collect_covariance_ablation_samples(config, progress=report_progress)
    baseline_validation = None
    if args.baseline_samples is not None:
        baseline = pd.read_csv(args.baseline_samples)
        baseline_validation = validate_current_variant_against_baseline(
            samples,
            baseline,
        )
        baseline_validation["baseline_samples"] = str(
            args.baseline_samples.resolve()
        )
    paths = write_covariance_ablation_outputs(
        samples,
        config,
        args.output_dir,
        baseline_validation=baseline_validation,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

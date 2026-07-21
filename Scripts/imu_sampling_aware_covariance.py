from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Module.IMUPreintegration import preintegrate_imu
from Scripts.imu_whitening_monte_carlo import (
    RESIDUAL_COLUMNS,
    STD_COLUMNS,
    WHITENED_COLUMNS,
    MonteCarloConfig,
    _chi_square_pdf,
    _component_summary,
    _histogram,
    _interval_summary,
    _marginal_nis,
    _normal_pdf,
    _quantile,
    _synthetic_config,
    _whitened_covariance,
)
from Scripts.synthetic_w3_validation_data import (
    SIGMA_ACC,
    SIGMA_GYRO,
    EstimatorIMUInput,
    generate_imu_input,
    generate_truth,
)
from Scripts.synthetic_w3_validation_runner import query_imu_interval
from Utility.IMUKinematics import vio_preintegrated_imu_residual


ProgressCallback = Callable[[int, int], None]
SOURCE_RESIDUAL_COLUMNS = tuple(f"source_{name}" for name in RESIDUAL_COLUMNS)
NOMINAL_RESIDUAL_COLUMNS = tuple(f"nominal_{name}" for name in RESIDUAL_COLUMNS)
COMPONENT_LABELS = tuple(name.removeprefix("z_") for name in WHITENED_COLUMNS)


@dataclass(frozen=True)
class SamplingAwareIntervalReference:
    interval_index: int
    start_time_s: float
    end_time_s: float
    dt_s: float
    raw_sample_count: int
    nominal_residual: torch.Tensor
    residual_jacobian: torch.Tensor
    covariance: torch.Tensor


def _residual_from_raw_noise(
    raw_noise: torch.Tensor,
    *,
    nominal_imu: EstimatorIMUInput,
    truth,
    frame_i: int,
    gravity_m_s2: float,
) -> torch.Tensor:
    frame_j = frame_i + 1
    raw_noise = raw_noise.reshape(nominal_imu.time_ns.numel(), 6)
    perturbed = EstimatorIMUInput(
        time_ns=nominal_imu.time_ns,
        measured_acc_body=nominal_imu.measured_acc_body + raw_noise[:, 0:3],
        measured_gyro_body=nominal_imu.measured_gyro_body + raw_noise[:, 3:6],
    )
    start_ns = int(truth.camera_time_ns[frame_i].item())
    end_ns = int(truth.camera_time_ns[frame_j].item())
    time_ns, acc, gyro = query_imu_interval(perturbed, start_ns, end_ns)
    zero_bias = torch.zeros(3, dtype=torch.float32)
    preintegrated = preintegrate_imu(
        time_ns=time_ns,
        acc=acc.float(),
        gyro=gyro.float(),
        R0_world=truth.pose_body_to_world[frame_i].rotation(),
        gravity=gravity_m_s2,
        sigma_acc=SIGMA_ACC,
        sigma_gyro=SIGMA_GYRO,
        sigma_acc_w=0.0,
        sigma_gyro_w=0.0,
        acc_bias=zero_bias,
        gyro_bias=zero_bias,
    )
    return vio_preintegrated_imu_residual(
        from_pose=truth.pose_body_to_world[frame_i],
        to_pose=truth.pose_body_to_world[frame_j],
        prev_velocity_world=truth.velocity_world[frame_i],
        curr_velocity_world=truth.velocity_world[frame_j],
        delta_R=preintegrated.delta_R,
        delta_v=preintegrated.delta_v,
        delta_p=preintegrated.delta_p,
        dt_total=preintegrated.dt_total,
    ).reshape(9).double()


def compute_sampling_aware_references(
    config: MonteCarloConfig,
    progress: ProgressCallback | None = None,
) -> list[SamplingAwareIntervalReference]:
    """Linearize the complete sampled-IMU-to-residual chain for every interval."""

    config.validate()
    sequence_config = _synthetic_config(config, config.start_seed)
    truth = generate_truth(
        sequence_config,
        bias_mode="zero_bias",
        noise_mode="mean_measurement",
    )
    nominal_imu = generate_imu_input(
        sequence_config,
        truth,
        bias_mode="zero_bias",
        noise_mode="mean_measurement",
    )
    raw_sample_count = int(nominal_imu.time_ns.numel())
    raw_variance = torch.tensor(
        [SIGMA_ACC**2 * config.imu_rate_hz] * 3
        + [SIGMA_GYRO**2 * config.imu_rate_hz] * 3,
        dtype=torch.float64,
    ).repeat(raw_sample_count)
    zero_noise = torch.zeros(
        raw_sample_count * 6,
        dtype=torch.float64,
        requires_grad=True,
    )

    interval_count = int(truth.camera_time_ns.numel()) - 1
    references: list[SamplingAwareIntervalReference] = []
    for frame_i in range(interval_count):
        def residual_function(noise: torch.Tensor) -> torch.Tensor:
            return _residual_from_raw_noise(
                noise,
                nominal_imu=nominal_imu,
                truth=truth,
                frame_i=frame_i,
                gravity_m_s2=config.gravity_m_s2,
            )

        nominal_residual = residual_function(zero_noise).detach()
        jacobian = torch.autograd.functional.jacobian(
            residual_function,
            zero_noise,
            create_graph=False,
            strict=False,
            vectorize=False,
        ).reshape(9, raw_sample_count * 6).detach().double()
        covariance = (jacobian * raw_variance.unsqueeze(0)) @ jacobian.T
        covariance = 0.5 * (covariance + covariance.T)

        start_ns = int(truth.camera_time_ns[frame_i].item())
        end_ns = int(truth.camera_time_ns[frame_i + 1].item())
        references.append(
            SamplingAwareIntervalReference(
                interval_index=frame_i,
                start_time_s=start_ns * 1e-9,
                end_time_s=end_ns * 1e-9,
                dt_s=(end_ns - start_ns) * 1e-9,
                raw_sample_count=raw_sample_count,
                nominal_residual=nominal_residual,
                residual_jacobian=jacobian,
                covariance=covariance,
            )
        )
        if progress is not None:
            progress(frame_i + 1, interval_count)
    return references


def _reference_map(
    references: Sequence[SamplingAwareIntervalReference],
) -> dict[int, SamplingAwareIntervalReference]:
    by_interval = {reference.interval_index: reference for reference in references}
    if len(by_interval) != len(references):
        raise ValueError("Sampling-aware references contain duplicate interval indices")
    return by_interval


def apply_sampling_aware_covariance(
    baseline_samples: pd.DataFrame,
    references: Sequence[SamplingAwareIntervalReference],
) -> pd.DataFrame:
    """Re-whiten existing residual samples using interval-specific J Q J^T."""

    if baseline_samples.empty:
        raise ValueError("Cannot whiten an empty baseline sample table")
    required = {"seed", "interval_index", *RESIDUAL_COLUMNS}
    missing = required - set(baseline_samples.columns)
    if missing:
        raise ValueError(f"Baseline sample table is missing columns: {sorted(missing)}")

    by_interval = _reference_map(references)
    sample_intervals = set(
        pd.to_numeric(baseline_samples["interval_index"], errors="raise").astype(int)
    )
    if sample_intervals != set(by_interval):
        raise ValueError(
            "Baseline intervals and sampling-aware references differ: "
            f"samples={sorted(sample_intervals)} references={sorted(by_interval)}"
        )

    samples = baseline_samples.copy().reset_index(drop=True)
    source_values = samples.loc[:, RESIDUAL_COLUMNS].to_numpy(dtype=float).copy()
    for source_name, residual_index in zip(SOURCE_RESIDUAL_COLUMNS, range(9)):
        samples[source_name] = source_values[:, residual_index]

    samples["variant"] = "sampling_aware"
    samples["cholesky_success"] = False
    samples["covariance_floor"] = 0.0
    samples["bias_rw_enabled"] = False
    samples["covariance_trace"] = np.nan

    for interval_index, reference in by_interval.items():
        row_indices = samples.index[samples["interval_index"] == interval_index].to_numpy()
        covariance = reference.covariance.detach().cpu().numpy().astype(float)
        covariance = 0.5 * (covariance + covariance.T)
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            for column in (*RESIDUAL_COLUMNS, *WHITENED_COLUMNS, *STD_COLUMNS):
                samples.loc[row_indices, column] = np.nan
            for name, value in zip(
                NOMINAL_RESIDUAL_COLUMNS,
                reference.nominal_residual.tolist(),
            ):
                samples.loc[row_indices, name] = float(value)
            continue

        nominal = reference.nominal_residual.detach().cpu().numpy().astype(float)
        centered = source_values[row_indices] - nominal.reshape(1, 9)
        whitened = np.linalg.solve(chol, centered.T).T
        eigenvalues = np.linalg.eigvalsh(covariance)
        predicted_std = np.sqrt(np.diag(covariance))

        for column_index, column in enumerate(RESIDUAL_COLUMNS):
            samples.loc[row_indices, column] = centered[:, column_index]
        for column_index, column in enumerate(WHITENED_COLUMNS):
            samples.loc[row_indices, column] = whitened[:, column_index]
        for column_index, column in enumerate(STD_COLUMNS):
            samples.loc[row_indices, column] = predicted_std[column_index]
        for name, value in zip(NOMINAL_RESIDUAL_COLUMNS, nominal.tolist()):
            samples.loc[row_indices, name] = float(value)

        samples.loc[row_indices, "cholesky_success"] = True
        samples.loc[row_indices, "nis_total"] = np.square(whitened).sum(axis=1)
        for block_index, column in enumerate(
            ("nis_p_marginal", "nis_v_marginal", "nis_R_marginal")
        ):
            block = slice(3 * block_index, 3 * (block_index + 1))
            block_chol = np.linalg.cholesky(covariance[block, block])
            block_z = np.linalg.solve(block_chol, centered[:, block].T).T
            samples.loc[row_indices, column] = np.square(block_z).sum(axis=1)
        samples.loc[row_indices, "covariance_trace"] = float(np.trace(covariance))
        samples.loc[row_indices, "cov_min_eigenvalue"] = float(eigenvalues.min())
        samples.loc[row_indices, "cov_max_eigenvalue"] = float(eigenvalues.max())
        samples.loc[row_indices, "cov_condition_number"] = float(
            eigenvalues.max() / eigenvalues.min()
        )

    return samples


def _sampling_variant_summary(samples: pd.DataFrame) -> pd.DataFrame:
    valid = samples.loc[samples["cholesky_success"].astype(bool)]
    nis = pd.to_numeric(valid["nis_total"], errors="raise")
    component_stds = [
        float(pd.to_numeric(valid[column], errors="raise").std(ddof=1))
        for column in WHITENED_COLUMNS
    ]
    return pd.DataFrame(
        [
            {
                "variant": "sampling_aware",
                "sample_count": int(len(samples)),
                "cholesky_failure_count": int(len(samples) - len(valid)),
                "nis_mean": float(nis.mean()),
                "nis_std": float(nis.std(ddof=1)),
                "nis_q05": _quantile(nis, 0.05),
                "nis_q50": _quantile(nis, 0.50),
                "nis_q95": _quantile(nis, 0.95),
                "whitened_std_min": min(component_stds),
                "whitened_std_max": max(component_stds),
                "cov_min_eigenvalue_min": float(
                    pd.to_numeric(samples["cov_min_eigenvalue"], errors="raise").min()
                ),
                "cov_min_eigenvalue_median": float(
                    pd.to_numeric(samples["cov_min_eigenvalue"], errors="raise").median()
                ),
                "cov_condition_median": float(
                    pd.to_numeric(samples["cov_condition_number"], errors="raise").median()
                ),
                "cov_condition_max": float(
                    pd.to_numeric(samples["cov_condition_number"], errors="raise").max()
                ),
            }
        ]
    )


def _comparison_summary(
    samples: pd.DataFrame,
    ablation_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    sampling = _sampling_variant_summary(samples)
    if ablation_summary is None:
        return sampling
    required = {
        "variant",
        "sample_count",
        "cholesky_failure_count",
        "nis_mean",
        "nis_std",
        "nis_q05",
        "nis_q50",
        "nis_q95",
    }
    missing = required - set(ablation_summary.columns)
    if missing:
        raise ValueError(f"Ablation summary is missing columns: {sorted(missing)}")
    selected = (
        ablation_summary.set_index("variant")
        .loc[["current", "no_floor"]]
        .reset_index()
    )
    columns = list(dict.fromkeys([*selected.columns, *sampling.columns]))
    return pd.concat(
        [selected.reindex(columns=columns), sampling.reindex(columns=columns)],
        ignore_index=True,
    )


def _interval_reference_frame(
    references: Sequence[SamplingAwareIntervalReference],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    for reference in references:
        covariance = reference.covariance.detach().cpu().numpy().astype(float)
        eigenvalues = np.linalg.eigvalsh(covariance)
        _, info = torch.linalg.cholesky_ex(reference.covariance)
        row: dict[str, float | int | bool] = {
            "interval_index": int(reference.interval_index),
            "start_time_s": float(reference.start_time_s),
            "end_time_s": float(reference.end_time_s),
            "dt_s": float(reference.dt_s),
            "raw_sample_count": int(reference.raw_sample_count),
            "jacobian_frobenius_norm": float(
                torch.linalg.vector_norm(reference.residual_jacobian).item()
            ),
            "covariance_trace": float(np.trace(covariance)),
            "cov_min_eigenvalue": float(eigenvalues.min()),
            "cov_max_eigenvalue": float(eigenvalues.max()),
            "cov_condition_number": float(eigenvalues.max() / eigenvalues.min()),
            "cholesky_success": int(info.item()) == 0,
        }
        row.update(
            {
                name: float(value)
                for name, value in zip(
                    NOMINAL_RESIDUAL_COLUMNS,
                    reference.nominal_residual.tolist(),
                )
            }
        )
        row.update(
            {
                name: float(value)
                for name, value in zip(STD_COLUMNS, np.sqrt(np.diag(covariance)).tolist())
            }
        )
        for row_index, row_label in enumerate(COMPONENT_LABELS):
            for column_index, column_label in enumerate(COMPONENT_LABELS):
                row[f"cov_{row_label}__{column_label}"] = float(
                    covariance[row_index, column_index]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _html_payload(
    samples: pd.DataFrame,
    component_summary: pd.DataFrame,
    interval_summary: pd.DataFrame,
    whitened_covariance: pd.DataFrame,
    comparison_summary: pd.DataFrame,
) -> dict[str, object]:
    nis_histogram = _histogram(
        samples["nis_total"].to_numpy(dtype=float),
        bins=64,
        symmetric=False,
    )
    nis_edges = np.asarray(nis_histogram["edges"], dtype=float)
    nis_centers = 0.5 * (nis_edges[:-1] + nis_edges[1:])
    nis_width = float(nis_edges[1] - nis_edges[0])
    nis_histogram["reference_counts"] = [
        _chi_square_pdf(float(x), 9) * nis_width * len(samples)
        for x in nis_centers
    ]

    component_histograms: dict[str, dict[str, object]] = {}
    for column in WHITENED_COLUMNS:
        histogram = _histogram(
            samples[column].to_numpy(dtype=float),
            bins=64,
            symmetric=True,
        )
        edges = np.asarray(histogram["edges"], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(edges[1] - edges[0])
        histogram["reference_counts"] = [
            _normal_pdf(float(x)) * width * len(samples) for x in centers
        ]
        component_histograms[column.removeprefix("z_")] = histogram

    covariance_labels = whitened_covariance["component"].astype(str).tolist()
    covariance_values = whitened_covariance.drop(columns=["component"]).to_numpy(
        dtype=float
    )
    return {
        "sample_count": int(len(samples)),
        "seed_count": int(samples["seed"].nunique()),
        "interval_count": int(samples["interval_index"].nunique()),
        "comparison_summary": comparison_summary.to_dict(orient="records"),
        "nis_histogram": nis_histogram,
        "component_histograms": component_histograms,
        "component_summary": component_summary.to_dict(orient="records"),
        "interval_summary": interval_summary.to_dict(orient="records"),
        "covariance_labels": covariance_labels,
        "covariance_values": covariance_values.tolist(),
    }


def _interactive_html(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sampling-aware IMU Covariance</title>
<style>
:root{{--bg:#f4f6f8;--surface:#fff;--ink:#18212f;--muted:#5d6b7b;--line:#d8dee7;--blue:#2563eb;--orange:#ea580c;--green:#15803d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}}
header{{padding:18px 24px;border-bottom:1px solid var(--line);background:var(--surface)}}
h1{{font-size:21px;margin:0 0 6px;letter-spacing:0}} .meta{{color:var(--muted)}}
main{{padding:18px 24px 36px;max-width:1500px;margin:auto}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px;min-width:0}} .wide{{grid-column:1/-1}}
h2{{font-size:15px;margin:0 0 10px;letter-spacing:0}} canvas{{width:100%;height:310px;border:1px solid #edf0f4}}
select{{padding:7px 9px;border:1px solid #aeb8c5;background:#fff;margin-bottom:10px}} .readout{{min-height:22px;color:var(--muted);font-family:Consolas,monospace}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid #e6e9ee;padding:6px 8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} .scroll{{overflow:auto;max-height:430px}} .heatmap{{display:grid;gap:2px;align-items:center;min-width:760px}}
.heatmap div{{padding:7px 4px;text-align:center;font:12px Consolas,monospace}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}} .wide{{grid-column:auto}} main{{padding:12px}}}}
</style>
</head>
<body>
<header><h1>Sampling-aware IMU Covariance</h1><div class="meta" id="run-meta"></div></header>
<main><div class="grid">
<section class="panel wide"><h2>Covariance model comparison: mean 9D NIS</h2><canvas id="comparison-canvas" aria-label="Mean NIS comparison"></canvas><div class="readout" id="comparison-readout"></div></section>
<section class="panel"><h2>Sampling-aware 9D NIS distribution</h2><canvas id="nis-canvas" aria-label="NIS histogram"></canvas><div class="readout" id="nis-readout"></div></section>
<section class="panel"><h2>Whitened component distribution</h2><select id="component-select" aria-label="Whitened component"></select><canvas id="component-canvas" aria-label="Whitened component histogram"></canvas><div class="readout" id="component-readout"></div></section>
<section class="panel wide"><h2>Mean NIS by camera interval</h2><canvas id="interval-canvas" aria-label="Interval NIS"></canvas><div class="readout" id="interval-readout"></div></section>
<section class="panel wide"><h2>Whitened component statistics</h2><div class="scroll"><table id="summary-table"></table></div></section>
<section class="panel wide"><h2>Empirical covariance of whitened residuals</h2><div class="scroll"><div class="heatmap" id="covariance-grid"></div></div></section>
</div></main>
<script>
const DATA={data_json};
const COLORS={{bar:'#93c5fd',edge:'#2563eb',ref:'#ea580c',green:'#15803d',grid:'#d8dee7',ink:'#18212f'}};
function setup(canvas){{const dpr=window.devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(r.width*dpr));canvas.height=Math.max(1,Math.round(r.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return {{c,w:r.width,h:r.height}}}}
function histogram(canvas,hist,readout,label){{const {{c,w,h}}=setup(canvas),p={{l:52,r:18,t:18,b:38}},counts=hist.counts,ref=hist.reference_counts,edges=hist.edges,ymax=Math.max(1,...counts,...ref)*1.08;c.clearRect(0,0,w,h);c.strokeStyle=COLORS.grid;for(let i=0;i<=4;i++){{const y=p.t+(h-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(w-p.r,y);c.stroke()}}const bw=(w-p.l-p.r)/counts.length;counts.forEach((v,i)=>{{const bh=(h-p.t-p.b)*v/ymax;c.fillStyle=COLORS.bar;c.fillRect(p.l+i*bw,h-p.b-bh,Math.max(1,bw-1),bh)}});c.strokeStyle=COLORS.ref;c.lineWidth=2;c.beginPath();ref.forEach((v,i)=>{{const x=p.l+(i+.5)*bw,y=h-p.b-(h-p.t-p.b)*v/ymax;i?c.lineTo(x,y):c.moveTo(x,y)}});c.stroke();c.strokeStyle=COLORS.ink;c.strokeRect(p.l,p.t,w-p.l-p.r,h-p.t-p.b);c.fillStyle=COLORS.ink;c.font='12px Arial';c.fillText(edges[0].toPrecision(4),p.l,h-14);c.fillText(edges.at(-1).toPrecision(4),w-p.r-48,h-14);canvas.onmousemove=e=>{{const r=canvas.getBoundingClientRect(),i=Math.floor((e.clientX-r.left-p.l)/bw);if(i>=0&&i<counts.length)readout.textContent=`${{label}} [${{edges[i].toPrecision(5)}}, ${{edges[i+1].toPrecision(5)}}): count=${{counts[i]}}, reference=${{ref[i].toFixed(2)}}`;}};canvas.onmouseleave=()=>readout.textContent=`visible range excludes below=${{hist.below_range}}, above=${{hist.above_range}}`;}}
function comparison(){{const rows=DATA.comparison_summary,canvas=document.getElementById('comparison-canvas'),readout=document.getElementById('comparison-readout'),{{c,w,h}}=setup(canvas),p={{l:58,r:20,t:20,b:60}},ymax=Math.max(9,...rows.map(r=>r.nis_mean))*1.18,bw=(w-p.l-p.r)/Math.max(1,rows.length);c.clearRect(0,0,w,h);c.strokeStyle=COLORS.grid;for(let i=0;i<=4;i++){{const y=p.t+(h-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(w-p.r,y);c.stroke()}}const yAt=v=>h-p.b-(h-p.t-p.b)*v/ymax;c.strokeStyle=COLORS.ref;c.setLineDash([6,4]);c.beginPath();c.moveTo(p.l,yAt(9));c.lineTo(w-p.r,yAt(9));c.stroke();c.setLineDash([]);rows.forEach((r,i)=>{{const x=p.l+i*bw+bw*.2,barW=bw*.6,y=yAt(r.nis_mean);c.fillStyle=r.variant==='sampling_aware'?COLORS.green:COLORS.edge;c.fillRect(x,y,barW,h-p.b-y);c.fillStyle=COLORS.ink;c.textAlign='center';c.font='12px Arial';c.fillText(r.nis_mean.toFixed(3),x+barW/2,y-7);c.fillText(r.variant,x+barW/2,h-p.b+22)}});c.textAlign='left';c.fillText('NIS',12,p.t+8);canvas.onmousemove=e=>{{const r=canvas.getBoundingClientRect(),i=Math.floor((e.clientX-r.left-p.l)/bw);if(i>=0&&i<rows.length){{const v=rows[i];readout.textContent=`${{v.variant}}: mean=${{v.nis_mean.toFixed(6)}}, q05=${{v.nis_q05.toFixed(6)}}, median=${{v.nis_q50.toFixed(6)}}, q95=${{v.nis_q95.toFixed(6)}}`;}}}}}}
function intervals(){{const rows=DATA.interval_summary,canvas=document.getElementById('interval-canvas'),readout=document.getElementById('interval-readout'),{{c,w,h}}=setup(canvas),p={{l:52,r:18,t:18,b:38}},vals=rows.map(r=>r.nis_mean),ymax=Math.max(9,...vals)*1.1;c.clearRect(0,0,w,h);c.strokeStyle=COLORS.grid;for(let i=0;i<=4;i++){{const y=p.t+(h-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(w-p.r,y);c.stroke()}}const xAt=i=>p.l+(w-p.l-p.r)*(rows.length<=1?0:i/(rows.length-1)),yAt=v=>h-p.b-(h-p.t-p.b)*v/ymax;c.strokeStyle=COLORS.ref;c.setLineDash([6,4]);c.beginPath();c.moveTo(p.l,yAt(9));c.lineTo(w-p.r,yAt(9));c.stroke();c.setLineDash([]);c.strokeStyle=COLORS.edge;c.lineWidth=2;c.beginPath();rows.forEach((r,i)=>i?c.lineTo(xAt(i),yAt(r.nis_mean)):c.moveTo(xAt(i),yAt(r.nis_mean)));c.stroke();canvas.onmousemove=e=>{{const r=canvas.getBoundingClientRect(),f=Math.max(0,Math.min(1,(e.clientX-r.left-p.l)/(w-p.l-p.r))),i=Math.round(f*(rows.length-1)),v=rows[i];readout.textContent=`interval=${{v.interval_index}}, time=${{v.start_time_s.toFixed(4)}}-${{v.end_time_s.toFixed(4)}} s, mean=${{v.nis_mean.toFixed(6)}}`;}}}}
function summary(){{const cols=['component','sample_count','raw_mean','raw_std','predicted_std_mean','whitened_mean','whitened_std','whitened_q05','whitened_q50','whitened_q95'],t=document.getElementById('summary-table');t.innerHTML='<thead><tr>'+cols.map(c=>`<th>${{c}}</th>`).join('')+'</tr></thead><tbody>'+DATA.component_summary.map(r=>'<tr>'+cols.map(c=>`<td>${{typeof r[c]==='number'?r[c].toPrecision(6):r[c]}}</td>`).join('')+'</tr>').join('')+'</tbody>'}}
function covariance(){{const labels=DATA.covariance_labels,values=DATA.covariance_values,g=document.getElementById('covariance-grid'),n=labels.length;g.style.gridTemplateColumns=`120px repeat(${{n}},minmax(65px,1fr))`;let html='<div></div>'+labels.map(x=>`<div>${{x}}</div>`).join('');values.forEach((row,i)=>{{html+=`<div>${{labels[i]}}</div>`;row.forEach(v=>{{const a=Math.min(1,Math.abs(v)),h=v>=0?215:10,l=96-42*a;html+=`<div aria-label="${{labels[i]}} ${{v}}" style="background:hsl(${{h}} 75% ${{l}}%)">${{v.toFixed(3)}}</div>`}})}});g.innerHTML=html}}
function render(){{document.getElementById('run-meta').textContent=`${{DATA.seed_count}} seeds, ${{DATA.interval_count}} intervals, ${{DATA.sample_count}} reused residual samples`;comparison();histogram(document.getElementById('nis-canvas'),DATA.nis_histogram,document.getElementById('nis-readout'),'NIS');const select=document.getElementById('component-select');if(!select.options.length)Object.keys(DATA.component_histograms).forEach(k=>select.add(new Option(k,k)));const redraw=()=>histogram(document.getElementById('component-canvas'),DATA.component_histograms[select.value],document.getElementById('component-readout'),select.value);select.onchange=redraw;redraw();intervals();summary();covariance()}}
window.addEventListener('resize',render);render();
</script>
</body>
</html>"""


def write_sampling_aware_outputs(
    samples: pd.DataFrame,
    references: Sequence[SamplingAwareIntervalReference],
    config: MonteCarloConfig,
    output_dir: str | Path,
    *,
    ablation_summary: pd.DataFrame | None = None,
) -> dict[str, Path]:
    if samples.empty:
        raise ValueError("Cannot write sampling-aware outputs for an empty table")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    component_summary = _component_summary(samples)
    interval_summary = _interval_summary(samples)
    whitened_covariance = _whitened_covariance(samples)
    comparison_summary = _comparison_summary(samples, ablation_summary)
    interval_references = _interval_reference_frame(references)
    paths = {
        "samples": output / "sampling_aware_samples.csv",
        "interval_references": output / "interval_covariance_references.csv",
        "component_summary": output / "sampling_aware_component_summary.csv",
        "interval_summary": output / "sampling_aware_interval_summary.csv",
        "whitened_covariance": output / "sampling_aware_whitened_covariance.csv",
        "comparison_summary": output / "comparison_variant_summary.csv",
        "metadata": output / "run_metadata.json",
        "html": output / "diagnostics_interactive.html",
    }
    samples.to_csv(paths["samples"], index=False)
    interval_references.to_csv(paths["interval_references"], index=False)
    component_summary.to_csv(paths["component_summary"], index=False)
    interval_summary.to_csv(paths["interval_summary"], index=False)
    whitened_covariance.to_csv(paths["whitened_covariance"], index=False)
    comparison_summary.to_csv(paths["comparison_summary"], index=False)

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": int(len(samples)),
        "seed_count": int(samples["seed"].nunique()),
        "interval_count": int(len(references)),
        "duration_s": float(config.duration_s),
        "camera_rate_hz": float(config.camera_rate_hz),
        "imu_rate_hz": float(config.imu_rate_hz),
        "gravity_m_s2": float(config.gravity_m_s2),
        "bias_mode": "zero_bias",
        "noise_mode": "fixed_seed_normal",
        "covariance_model": "raw_sample_jacobian_jqjt",
        "raw_sample_noise_model": "independent_density_discretized_at_imu_rate",
        "residual_centering": "subtract_zero_noise_nominal_residual_per_interval",
        "source_samples_reused": True,
        "sigma_acc": SIGMA_ACC,
        "sigma_gyro": SIGMA_GYRO,
        "files": {name: path.name for name, path in paths.items()},
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    payload = _html_payload(
        samples,
        component_summary,
        interval_summary,
        whitened_covariance,
        comparison_summary,
    )
    paths["html"].write_text(_interactive_html(payload), encoding="utf-8")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-whiten existing zero-Bias Monte Carlo residuals with an exact "
            "sampling/interpolation-aware J Q J^T covariance reference."
        )
    )
    parser.add_argument("--baseline-samples", type=Path, required=True)
    parser.add_argument("--ablation-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        print(f"interval references {done}/{total}", flush=True)

    baseline_samples = pd.read_csv(args.baseline_samples)
    ablation_summary = (
        pd.read_csv(args.ablation_summary) if args.ablation_summary is not None else None
    )
    references = compute_sampling_aware_references(config, progress=report_progress)
    samples = apply_sampling_aware_covariance(baseline_samples, references)
    paths = write_sampling_aware_outputs(
        samples,
        references,
        config,
        args.output_dir,
        ablation_summary=ablation_summary,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

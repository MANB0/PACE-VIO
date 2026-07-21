from __future__ import annotations

import argparse
import json
import math
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

from Module.IMUPreintegration import preintegrate_imu
from Scripts.synthetic_w3_validation_data import (
    SIGMA_ACC,
    SIGMA_ACC_W,
    SIGMA_GYRO,
    SIGMA_GYRO_W,
    SyntheticSequenceConfig,
    generate_imu_input,
    generate_truth,
)
from Scripts.synthetic_w3_validation_runner import query_imu_interval
from Utility.IMUKinematics import vio_preintegrated_imu_residual


ProgressCallback = Callable[[int, int], None]

BLOCKS = ("p", "v", "R")
AXES = ("x", "y", "z")
RESIDUAL_COLUMNS = tuple(
    f"r_{block}_{axis}" for block in BLOCKS for axis in AXES
)
WHITENED_COLUMNS = tuple(
    f"z_{block}_{axis}" for block in BLOCKS for axis in AXES
)
STD_COLUMNS = tuple(
    f"predicted_std_{block}_{axis}" for block in BLOCKS for axis in AXES
)


@dataclass(frozen=True)
class MonteCarloConfig:
    duration_s: float = 1.0
    seed_count: int = 1000
    start_seed: int = 20260711
    camera_rate_hz: float = 30.0
    imu_rate_hz: float = 100.0
    gravity_m_s2: float = 9.8

    def validate(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.seed_count <= 0:
            raise ValueError("seed_count must be positive")
        if self.camera_rate_hz <= 0.0:
            raise ValueError("camera_rate_hz must be positive")
        if self.imu_rate_hz <= 0.0:
            raise ValueError("imu_rate_hz must be positive")


def _synthetic_config(config: MonteCarloConfig, seed: int) -> SyntheticSequenceConfig:
    return SyntheticSequenceConfig(
        duration_s=float(config.duration_s),
        camera_rate_hz=float(config.camera_rate_hz),
        imu_rate_hz=float(config.imu_rate_hz),
        gravity_m_s2=float(config.gravity_m_s2),
        seed=int(seed),
    )


def _whiten(residual: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    cov = covariance.reshape(9, 9).double()
    cov = 0.5 * (cov + cov.T)
    chol = torch.linalg.cholesky(cov)
    return torch.linalg.solve_triangular(
        chol,
        residual.reshape(9, 1).double(),
        upper=False,
    ).reshape(9)


def _marginal_nis(residual: torch.Tensor, covariance: torch.Tensor) -> list[float]:
    values: list[float] = []
    for block_index in range(3):
        sl = slice(3 * block_index, 3 * (block_index + 1))
        block_z = _whiten_block(residual[sl], covariance[sl, sl])
        values.append(float(torch.dot(block_z, block_z).item()))
    return values


def _whiten_block(residual: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    cov = covariance.reshape(3, 3).double()
    cov = 0.5 * (cov + cov.T)
    chol = torch.linalg.cholesky(cov)
    return torch.linalg.solve_triangular(
        chol,
        residual.reshape(3, 1).double(),
        upper=False,
    ).reshape(3)


def collect_whitening_samples(
    config: MonteCarloConfig,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Collect GT-evaluated production IMU residuals for independent noise seeds."""

    config.validate()
    rows: list[dict[str, float | int | str]] = []
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
            preintegrated = preintegrate_imu(
                time_ns=time_ns,
                acc=acc.float(),
                gyro=gyro.float(),
                R0_world=truth.pose_body_to_world[frame_i].rotation(),
                gravity=sequence_config.gravity_m_s2,
                sigma_acc=SIGMA_ACC,
                sigma_gyro=SIGMA_GYRO,
                sigma_acc_w=SIGMA_ACC_W,
                sigma_gyro_w=SIGMA_GYRO_W,
                acc_bias=zero_bias,
                gyro_bias=zero_bias,
            )
            residual = vio_preintegrated_imu_residual(
                from_pose=truth.pose_body_to_world[frame_i],
                to_pose=truth.pose_body_to_world[frame_j],
                prev_velocity_world=truth.velocity_world[frame_i],
                curr_velocity_world=truth.velocity_world[frame_j],
                delta_R=preintegrated.delta_R,
                delta_v=preintegrated.delta_v,
                delta_p=preintegrated.delta_p,
                dt_total=preintegrated.dt_total,
                prev_acc_bias=zero_bias,
                prev_gyro_bias=zero_bias,
                linearized_acc_bias=preintegrated.linearized_acc_bias,
                linearized_gyro_bias=preintegrated.linearized_gyro_bias,
                bias_jacobian=preintegrated.bias_jacobian,
            ).reshape(9).double()
            covariance = preintegrated.cov.reshape(9, 9).double()
            whitened = _whiten(residual, covariance)
            marginal_nis = _marginal_nis(residual, covariance)
            eigenvalues = torch.linalg.eigvalsh(0.5 * (covariance + covariance.T))

            row: dict[str, float | int | str] = {
                "seed": seed,
                "interval_index": frame_i,
                "start_time_s": start_ns * 1e-9,
                "end_time_s": end_ns * 1e-9,
                "dt_s": float(preintegrated.dt_total),
                "bias_mode": "zero_bias",
                "noise_mode": "fixed_seed_normal",
                "nis_total": float(torch.dot(whitened, whitened).item()),
                "nis_p_marginal": marginal_nis[0],
                "nis_v_marginal": marginal_nis[1],
                "nis_R_marginal": marginal_nis[2],
                "cov_min_eigenvalue": float(eigenvalues.min().item()),
                "cov_max_eigenvalue": float(eigenvalues.max().item()),
                "cov_condition_number": float(
                    (eigenvalues.max() / eigenvalues.min()).item()
                ),
            }
            row.update(
                {name: float(value) for name, value in zip(RESIDUAL_COLUMNS, residual.tolist())}
            )
            row.update(
                {name: float(value) for name, value in zip(WHITENED_COLUMNS, whitened.tolist())}
            )
            row.update(
                {
                    name: float(value)
                    for name, value in zip(
                        STD_COLUMNS,
                        torch.sqrt(torch.diagonal(covariance)).tolist(),
                    )
                }
            )
            rows.append(row)

        if progress is not None:
            progress(seed_offset + 1, config.seed_count)

    return pd.DataFrame(rows)


def _quantile(values: pd.Series, probability: float) -> float:
    return float(pd.to_numeric(values, errors="raise").quantile(probability))


def _component_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for residual_name, whitened_name, std_name in zip(
        RESIDUAL_COLUMNS,
        WHITENED_COLUMNS,
        STD_COLUMNS,
    ):
        raw = pd.to_numeric(samples[residual_name], errors="raise")
        whitened = pd.to_numeric(samples[whitened_name], errors="raise")
        predicted_std = pd.to_numeric(samples[std_name], errors="raise")
        rows.append(
            {
                "component": whitened_name.removeprefix("z_"),
                "sample_count": int(len(samples)),
                "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std(ddof=1)),
                "predicted_std_mean": float(predicted_std.mean()),
                "whitened_mean": float(whitened.mean()),
                "whitened_std": float(whitened.std(ddof=1)),
                "whitened_q01": _quantile(whitened, 0.01),
                "whitened_q05": _quantile(whitened, 0.05),
                "whitened_q50": _quantile(whitened, 0.50),
                "whitened_q95": _quantile(whitened, 0.95),
                "whitened_q99": _quantile(whitened, 0.99),
            }
        )
    return pd.DataFrame(rows)


def _interval_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for interval_index, group in samples.groupby("interval_index", sort=True):
        nis = pd.to_numeric(group["nis_total"], errors="raise")
        rows.append(
            {
                "interval_index": int(interval_index),
                "start_time_s": float(group["start_time_s"].iloc[0]),
                "end_time_s": float(group["end_time_s"].iloc[0]),
                "sample_count": int(len(group)),
                "nis_mean": float(nis.mean()),
                "nis_std": float(nis.std(ddof=1)),
                "nis_q05": _quantile(nis, 0.05),
                "nis_q50": _quantile(nis, 0.50),
                "nis_q95": _quantile(nis, 0.95),
            }
        )
    return pd.DataFrame(rows)


def _whitened_covariance(samples: pd.DataFrame) -> pd.DataFrame:
    values = samples.loc[:, WHITENED_COLUMNS].to_numpy(dtype=float)
    covariance = np.cov(values, rowvar=False, ddof=1)
    labels = [name.removeprefix("z_") for name in WHITENED_COLUMNS]
    frame = pd.DataFrame(covariance, columns=labels)
    frame.insert(0, "component", labels)
    return frame


def _histogram(values: np.ndarray, *, bins: int, symmetric: bool) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Histogram input contains no finite samples")

    if symmetric:
        visible = max(float(np.quantile(np.abs(finite), 0.995)), 1e-12)
        lower, upper = -visible, visible
    else:
        lower = min(0.0, float(finite.min()))
        upper = max(float(np.quantile(finite, 0.995)), lower + 1e-12)
    counts, edges = np.histogram(finite, bins=bins, range=(lower, upper))
    return {
        "counts": counts.astype(int).tolist(),
        "edges": edges.astype(float).tolist(),
        "sample_count": int(finite.size),
        "below_range": int(np.count_nonzero(finite < lower)),
        "above_range": int(np.count_nonzero(finite > upper)),
    }


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _chi_square_pdf(x: float, degrees_of_freedom: int) -> float:
    if x <= 0.0:
        return 0.0
    half_k = 0.5 * degrees_of_freedom
    return (
        x ** (half_k - 1.0)
        * math.exp(-0.5 * x)
        / (2.0**half_k * math.gamma(half_k))
    )


def _html_payload(
    samples: pd.DataFrame,
    component_summary: pd.DataFrame,
    interval_summary: pd.DataFrame,
    whitened_covariance: pd.DataFrame,
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
            _normal_pdf(float(x)) * width * len(samples)
            for x in centers
        ]
        component_histograms[column.removeprefix("z_")] = histogram

    cov_labels = whitened_covariance["component"].astype(str).tolist()
    cov_values = whitened_covariance.drop(columns=["component"]).to_numpy(dtype=float)
    return {
        "sample_count": int(len(samples)),
        "seed_count": int(samples["seed"].nunique()),
        "interval_count": int(samples["interval_index"].nunique()),
        "nis_histogram": nis_histogram,
        "component_histograms": component_histograms,
        "component_summary": component_summary.to_dict(orient="records"),
        "interval_summary": interval_summary.to_dict(orient="records"),
        "covariance_labels": cov_labels,
        "covariance_values": cov_values.tolist(),
    }


def _interactive_html(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IMU Whitening Monte Carlo</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--ink:#18212f;--muted:#5d6b7b;--line:#d8dee7;--blue:#2563eb;--orange:#ea580c}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}}
header{{padding:18px 24px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:3}}
h1{{font-size:21px;margin:0 0 6px}} .meta{{color:var(--muted)}} main{{padding:18px 24px 36px;max-width:1500px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} .panel{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px;min-width:0}}
.wide{{grid-column:1/-1}} h2{{font-size:15px;margin:0 0 10px}} canvas{{width:100%;height:330px;border:1px solid #edf0f4}}
select{{padding:7px 9px;border:1px solid #aeb8c5;background:#fff;margin-bottom:10px}} .readout{{min-height:22px;color:var(--muted);font-family:Consolas,monospace}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid #e6e9ee;padding:6px 8px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
.scroll{{overflow:auto;max-height:430px}} .heatmap{{display:grid;gap:2px;align-items:center;min-width:760px}} .heatmap div{{padding:7px 4px;text-align:center;font:12px Consolas,monospace}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}} .wide{{grid-column:auto}} main{{padding:12px}}}}
</style>
</head>
<body>
<header><h1>IMU Whitening Monte Carlo</h1><div class="meta" id="run-meta"></div></header>
<main><div class="grid">
<section class="panel"><h2>9D NIS histogram</h2><canvas id="nis-canvas"></canvas><div class="readout" id="nis-readout"></div></section>
<section class="panel"><h2>Whitened component histogram</h2><select id="component-select"></select><canvas id="component-canvas"></canvas><div class="readout" id="component-readout"></div></section>
<section class="panel wide"><h2>NIS mean by camera interval</h2><canvas id="interval-canvas"></canvas><div class="readout" id="interval-readout"></div></section>
<section class="panel wide"><h2>Whitened component summary</h2><div class="scroll"><table id="summary-table"></table></div></section>
<section class="panel wide"><h2>Empirical covariance of whitened residuals</h2><div class="scroll"><div class="heatmap" id="covariance-grid"></div></div></section>
</div></main>
<script>
const DATA={data_json};
const COLORS={{bar:'#93c5fd',edge:'#2563eb',ref:'#ea580c',grid:'#d8dee7',ink:'#18212f'}};
function canvasSetup(canvas){{const dpr=window.devicePixelRatio||1;const rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return {{c,w:rect.width,h:rect.height}}}}
function drawHistogram(canvas,hist,readout,label){{const {{c,w,h}}=canvasSetup(canvas),pad={{l:52,r:18,t:18,b:38}};const counts=hist.counts,ref=hist.reference_counts,edges=hist.edges;const ymax=Math.max(1,...counts,...ref)*1.08;c.clearRect(0,0,w,h);c.strokeStyle=COLORS.grid;c.lineWidth=1;for(let i=0;i<=4;i++){{const y=pad.t+(h-pad.t-pad.b)*i/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke()}}const bw=(w-pad.l-pad.r)/counts.length;counts.forEach((v,i)=>{{const bh=(h-pad.t-pad.b)*v/ymax;c.fillStyle=COLORS.bar;c.fillRect(pad.l+i*bw,h-pad.b-bh,Math.max(1,bw-1),bh)}});c.strokeStyle=COLORS.ref;c.lineWidth=2;c.beginPath();ref.forEach((v,i)=>{{const x=pad.l+(i+.5)*bw,y=h-pad.b-(h-pad.t-pad.b)*v/ymax;i?c.lineTo(x,y):c.moveTo(x,y)}});c.stroke();c.strokeStyle=COLORS.ink;c.strokeRect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);c.fillStyle=COLORS.ink;c.font='12px Arial';c.fillText(edges[0].toPrecision(4),pad.l,h-14);c.fillText(edges[edges.length-1].toPrecision(4),w-pad.r-48,h-14);c.fillText('count',8,pad.t+8);canvas.onmousemove=e=>{{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left-pad.l,i=Math.floor(x/bw);if(i>=0&&i<counts.length)readout.textContent=`${{label}} [${{edges[i].toPrecision(5)}}, ${{edges[i+1].toPrecision(5)}}): count=${{counts[i]}}, reference=${{ref[i].toFixed(2)}}`;}};canvas.onmouseleave=()=>readout.textContent=`visible range excludes below=${{hist.below_range}}, above=${{hist.above_range}}`;}}
function drawIntervals(){{const canvas=document.getElementById('interval-canvas'),readout=document.getElementById('interval-readout'),rows=DATA.interval_summary;const {{c,w,h}}=canvasSetup(canvas),pad={{l:52,r:18,t:18,b:38}};const vals=rows.map(r=>r.nis_mean),ymax=Math.max(9,...vals)*1.1;c.clearRect(0,0,w,h);c.strokeStyle=COLORS.grid;for(let i=0;i<=4;i++){{const y=pad.t+(h-pad.t-pad.b)*i/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke()}}const xAt=i=>pad.l+(w-pad.l-pad.r)*(rows.length<=1?0:i/(rows.length-1));const yAt=v=>h-pad.b-(h-pad.t-pad.b)*v/ymax;c.strokeStyle=COLORS.orange;c.setLineDash([6,4]);c.beginPath();c.moveTo(pad.l,yAt(9));c.lineTo(w-pad.r,yAt(9));c.stroke();c.setLineDash([]);c.strokeStyle=COLORS.edge;c.lineWidth=2;c.beginPath();rows.forEach((r,i)=>i?c.lineTo(xAt(i),yAt(r.nis_mean)):c.moveTo(xAt(i),yAt(r.nis_mean)));c.stroke();c.fillStyle=COLORS.ink;c.font='12px Arial';c.fillText('interval',w/2,h-10);c.fillText('NIS',10,pad.t+8);canvas.onmousemove=e=>{{const rect=canvas.getBoundingClientRect(),frac=Math.max(0,Math.min(1,(e.clientX-rect.left-pad.l)/(w-pad.l-pad.r))),i=Math.round(frac*(rows.length-1)),r=rows[i];readout.textContent=`interval=${{r.interval_index}}, time=${{r.start_time_s.toFixed(4)}}-${{r.end_time_s.toFixed(4)}} s, mean=${{r.nis_mean.toFixed(5)}}, q05=${{r.nis_q05.toFixed(5)}}, q95=${{r.nis_q95.toFixed(5)}}`;}}}}
function buildSummary(){{const cols=['component','sample_count','raw_mean','raw_std','predicted_std_mean','whitened_mean','whitened_std','whitened_q05','whitened_q50','whitened_q95'];const t=document.getElementById('summary-table');t.innerHTML='<thead><tr>'+cols.map(c=>`<th>${{c}}</th>`).join('')+'</tr></thead><tbody>'+DATA.component_summary.map(r=>'<tr>'+cols.map(c=>`<td>${{typeof r[c]==='number'?r[c].toPrecision(6):r[c]}}</td>`).join('')+'</tr>').join('')+'</tbody>'}}
function buildCovariance(){{const labels=DATA.covariance_labels,values=DATA.covariance_values,g=document.getElementById('covariance-grid'),n=labels.length;g.style.gridTemplateColumns=`120px repeat(${{n}},minmax(65px,1fr))`;let html='<div></div>'+labels.map(x=>`<div>${{x}}</div>`).join('');values.forEach((row,i)=>{{html+=`<div>${{labels[i]}}</div>`;row.forEach(v=>{{const a=Math.min(1,Math.abs(v)),h=v>=0?215:10,l=96-42*a;html+=`<div title="${{v}}" style="background:hsl(${{h}} 75% ${{l}}%)">${{v.toFixed(3)}}</div>`}})}});g.innerHTML=html}}
function render(){{document.getElementById('run-meta').textContent=`${{DATA.seed_count}} seeds, ${{DATA.interval_count}} intervals, ${{DATA.sample_count}} residual samples`;drawHistogram(document.getElementById('nis-canvas'),DATA.nis_histogram,document.getElementById('nis-readout'),'NIS');const select=document.getElementById('component-select');Object.keys(DATA.component_histograms).forEach(k=>select.add(new Option(k,k)));const redraw=()=>drawHistogram(document.getElementById('component-canvas'),DATA.component_histograms[select.value],document.getElementById('component-readout'),select.value);select.onchange=redraw;redraw();drawIntervals();buildSummary();buildCovariance()}}
window.addEventListener('resize',()=>render());render();
</script>
</body></html>"""


def write_analysis_outputs(
    samples: pd.DataFrame,
    config: MonteCarloConfig,
    output_dir: str | Path,
) -> dict[str, Path]:
    if samples.empty:
        raise ValueError("Cannot write analysis outputs for an empty sample table")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    component_summary = _component_summary(samples)
    interval_summary = _interval_summary(samples)
    whitened_covariance = _whitened_covariance(samples)
    paths = {
        "samples": output / "whitening_samples.csv",
        "component_summary": output / "whitened_component_summary.csv",
        "interval_summary": output / "interval_summary.csv",
        "whitened_covariance": output / "whitened_covariance.csv",
        "metadata": output / "run_metadata.json",
        "html": output / "diagnostics_interactive.html",
    }
    samples.to_csv(paths["samples"], index=False)
    component_summary.to_csv(paths["component_summary"], index=False)
    interval_summary.to_csv(paths["interval_summary"], index=False)
    whitened_covariance.to_csv(paths["whitened_covariance"], index=False)

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": int(len(samples)),
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
    )
    paths["html"].write_text(_interactive_html(payload), encoding="utf-8")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate zero-Bias normal-noise IMU whitening Monte Carlo evidence."
    )
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
        if done == 1 or done == total or done % max(1, total // 20) == 0:
            print(f"seeds {done}/{total}", flush=True)

    samples = collect_whitening_samples(config, progress=report_progress)
    paths = write_analysis_outputs(samples, config, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

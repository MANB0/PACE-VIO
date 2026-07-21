#!/usr/bin/env python3
"""Analyze clear-circle paired VIO runs against HoloOcean ref_pose ground truth."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Utility.RunOutputBundle import find_output_bundle


DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "clear_circle_pair_vio_20260704"
DEFAULT_OUTDIR = WORKDIR / "analysis_clear_circle_pair_vio_20260704"


TRACE_COLORS = [
    "#d95f02",
    "#1b9e77",
    "#1f78b4",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
]


def read_manifest(result_root: Path) -> list[dict[str, str]]:
    path = result_root / "run_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_manifests(result_roots: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result_root in result_roots:
        rows.extend(read_manifest(result_root))
    return rows


def make_trace_label(row: dict[str, object]) -> str:
    return f"{row['scene']} / {row['variant']}"


def color_for_label(label: str, index: int) -> str:
    preferred = {
        "clear_circle_normal_noise / vio_preintegrated_full": "#d95f02",
        "clear_circle_zero_noise / vio_preintegrated_full": "#1b9e77",
        "clear_circle_normal_noise / pure_macvo": "#9467bd",
        "clear_circle_zero_noise / pure_macvo": "#1f78b4",
    }
    return preferred.get(label, TRACE_COLORS[index % len(TRACE_COLORS)])


def load_pose_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename: dict[str, str] = {}
    if "timestamp" in df.columns:
        rename["timestamp"] = "timestamp_ns"
    if "x" in df.columns:
        rename["x"] = "tx"
    if "y" in df.columns:
        rename["y"] = "ty"
    if "z" in df.columns:
        rename["z"] = "tz"
    df = df.rename(columns=rename)
    required = {"timestamp_ns", "tx", "ty", "tz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    return df


def read_pose_frame(bundle_dir: Path) -> str:
    path = bundle_dir / "pose_coordinate_frame.txt"
    if not path.exists():
        return "NWU"
    return path.read_text(encoding="utf-8").strip().upper() or "NWU"


def xyz_to_nwu(xyz: np.ndarray, frame: str) -> np.ndarray:
    frame = frame.strip().upper()
    out = np.asarray(xyz, dtype=float).copy()
    if frame == "NWU":
        return out
    if frame == "NED":
        out[:, 1] *= -1.0
        out[:, 2] *= -1.0
        return out
    raise ValueError(f"Unsupported pose frame {frame!r}")


def align_est_gt(est: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    joined = est.merge(gt, on="timestamp_ns", suffixes=("_est", "_gt"))
    if not joined.empty:
        return joined
    n = min(len(est), len(gt))
    if n <= 0:
        raise ValueError("Cannot align empty estimated/GT trajectory")
    data = {
        "timestamp_ns": gt["timestamp_ns"].iloc[:n].to_numpy(),
        "tx_est": est["tx"].iloc[:n].to_numpy(),
        "ty_est": est["ty"].iloc[:n].to_numpy(),
        "tz_est": est["tz"].iloc[:n].to_numpy(),
        "tx_gt": gt["tx"].iloc[:n].to_numpy(),
        "ty_gt": gt["ty"].iloc[:n].to_numpy(),
        "tz_gt": gt["tz"].iloc[:n].to_numpy(),
    }
    for col in ("qx", "qy", "qz", "qw"):
        if col in est.columns and col in gt.columns:
            data[f"{col}_est"] = est[col].iloc[:n].to_numpy()
            data[f"{col}_gt"] = gt[col].iloc[:n].to_numpy()
    return pd.DataFrame(data)


def evaluate_run(row: dict[str, str]) -> tuple[dict[str, object], pd.DataFrame]:
    result_dir = Path(row["result_dir"])
    scene_root = Path(row["scene_root"])
    bundle = find_output_bundle(
        result_dir,
        require_same_dir_diagnostics=row.get("imu_factor_mode") == "preintegrated_vio",
    )
    est = load_pose_table(bundle.poses_path)
    gt = load_pose_table(scene_root / "ref_pose.csv")

    pose_frame = read_pose_frame(bundle.bundle_dir)
    est_xyz = xyz_to_nwu(est[["tx", "ty", "tz"]].to_numpy(float), pose_frame)
    est = est.copy()
    est[["tx", "ty", "tz"]] = est_xyz

    joined = align_est_gt(est, gt)
    est_pos = joined[["tx_est", "ty_est", "tz_est"]].to_numpy(float)
    gt_pos = joined[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(float)
    err = est_pos - gt_pos
    err_norm = np.linalg.norm(err, axis=1)

    span = gt_pos.max(axis=0) - gt_pos.min(axis=0)
    path_len = float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum()) if len(gt_pos) > 1 else 0.0
    summary = {
        "scene": row["scene"],
        "variant": row["variant"],
        "label": make_trace_label(row),
        "matched_frames": int(len(joined)),
        "pose_frame": pose_frame,
        "ate_rmse_m": float(np.sqrt(np.mean(err_norm**2))),
        "ate_median_m": float(np.median(err_norm)),
        "ate_final_m": float(err_norm[-1]),
        "ate_max_m": float(np.max(err_norm)),
        "gt_path_length_m": path_len,
        "gt_span_x_m": float(span[0]),
        "gt_span_y_m": float(span[1]),
        "gt_span_z_m": float(span[2]),
        "result_dir": str(result_dir),
        "bundle_dir": str(bundle.bundle_dir),
        "poses_path": str(bundle.poses_path),
        "gt_path": str(scene_root / "ref_pose.csv"),
    }
    joined = joined.copy()
    joined["err_m"] = err_norm
    return summary, joined


def plot_xy(trajs: dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=180)
    first = next(iter(trajs.values()))
    ax.plot(first["tx_gt"], first["ty_gt"], color="black", lw=2.4, label="GT")
    for idx, (name, df) in enumerate(trajs.items()):
        ax.plot(df["tx_est"], df["ty_est"], lw=1.6, color=color_for_label(name, idx), label=name)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / m (NWU)")
    ax.set_ylabel("y / m (NWU)")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "trajectory_xy_gt_vs_est.png")
    plt.close(fig)


def plot_xy_gt_region(trajs: dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=180)
    first = next(iter(trajs.values()))
    gt_x = first["tx_gt"].to_numpy(float)
    gt_y = first["ty_gt"].to_numpy(float)
    ax.plot(gt_x, gt_y, color="black", lw=2.4, label="GT")
    for idx, (name, df) in enumerate(trajs.items()):
        ax.plot(df["tx_est"], df["ty_est"], lw=1.6, color=color_for_label(name, idx), label=name)

    margin = 1.5
    ax.set_xlim(float(gt_x.min()) - margin, float(gt_x.max()) + margin)
    ax.set_ylim(float(gt_y.min()) - margin, float(gt_y.max()) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m (NWU)")
    ax.set_ylabel("y / m (NWU)")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "trajectory_xy_gt_region.png")
    plt.close(fig)


def plot_xz(trajs: dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    first = next(iter(trajs.values()))
    ax.plot(first["tx_gt"], first["tz_gt"], color="black", lw=2.4, label="GT")
    for idx, (name, df) in enumerate(trajs.items()):
        ax.plot(df["tx_est"], df["tz_est"], lw=1.6, color=color_for_label(name, idx), label=name)
    ax.set_xlabel("x / m (NWU)")
    ax.set_ylabel("z / m (NWU)")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "trajectory_xz_gt_vs_est.png")
    plt.close(fig)


def plot_error(trajs: dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180)
    for name, df in trajs.items():
        t = (df["timestamp_ns"].to_numpy(float) - float(df["timestamp_ns"].iloc[0])) * 1e-9
        ax.plot(t, df["err_m"], lw=1.4, label=name)
    ax.set_xlabel("time / s")
    ax.set_ylabel("position error / m")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "position_error_over_time.png")
    plt.close(fig)


def _round_float_list(values: np.ndarray) -> list[float]:
    return [round(float(v), 6) for v in values]


def write_interactive_html(trajs: dict[str, pd.DataFrame], outdir: Path) -> None:
    first = next(iter(trajs.values()))
    gt = {
        "name": "GT",
        "color": "#111111",
        "t": _round_float_list((first["timestamp_ns"].to_numpy(float) - float(first["timestamp_ns"].iloc[0])) * 1e-9),
        "x": _round_float_list(first["tx_gt"].to_numpy(float)),
        "y": _round_float_list(first["ty_gt"].to_numpy(float)),
        "z": _round_float_list(first["tz_gt"].to_numpy(float)),
    }
    traces = []
    for idx, (name, df) in enumerate(trajs.items()):
        traces.append(
            {
                "name": name,
                "color": color_for_label(name, idx),
                "t": _round_float_list((df["timestamp_ns"].to_numpy(float) - float(df["timestamp_ns"].iloc[0])) * 1e-9),
                "x": _round_float_list(df["tx_est"].to_numpy(float)),
                "y": _round_float_list(df["ty_est"].to_numpy(float)),
                "z": _round_float_list(df["tz_est"].to_numpy(float)),
                "err": _round_float_list(df["err_m"].to_numpy(float)),
            }
        )

    payload = json.dumps({"gt": gt, "traces": traces}, ensure_ascii=False)
    page_title = "Clear-circle VIO interactive trajectory"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(page_title)}</title>
<style>
  :root {{
    font-family: Arial, Helvetica, sans-serif;
    color: #1f2933;
    background: #f5f7fa;
  }}
  body {{
    margin: 0;
    padding: 20px;
  }}
  .panel {{
    max-width: 1180px;
    margin: 0 auto;
    background: #fff;
    border: 1px solid #d8dee6;
    box-shadow: 0 8px 24px rgba(31, 41, 51, 0.08);
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    align-items: center;
    padding: 12px 14px;
    border-bottom: 1px solid #d8dee6;
  }}
  h1 {{
    font-size: 16px;
    line-height: 1.25;
    margin: 0 18px 0 0;
    font-weight: 700;
  }}
  button {{
    border: 1px solid #aab4c0;
    background: #f8fafc;
    color: #1f2933;
    border-radius: 4px;
    padding: 6px 10px;
    cursor: pointer;
    font-size: 13px;
  }}
  button.active {{
    background: #1f6feb;
    border-color: #1f6feb;
    color: white;
  }}
  label {{
    display: inline-flex;
    gap: 5px;
    align-items: center;
    font-size: 13px;
    white-space: nowrap;
  }}
  #plot-wrap {{
    position: relative;
    height: 690px;
  }}
  svg {{
    display: block;
    width: 100%;
    height: 100%;
    background: white;
    touch-action: none;
    cursor: crosshair;
  }}
  .grid {{
    stroke: #d1d5db;
    stroke-width: 1;
  }}
  .axis {{
    stroke: #111827;
    stroke-width: 1.3;
  }}
  .tick-label, .axis-label {{
    fill: #111827;
    font-size: 12px;
  }}
  .axis-label {{
    font-size: 14px;
    font-weight: 600;
  }}
  .trace {{
    fill: none;
    stroke-width: 2.4;
    stroke-linejoin: round;
    stroke-linecap: round;
  }}
  .gt {{
    stroke-width: 3.8;
  }}
  #tooltip {{
    position: absolute;
    display: none;
    pointer-events: none;
    padding: 7px 9px;
    border: 1px solid #9aa6b2;
    background: rgba(255, 255, 255, 0.96);
    font-size: 12px;
    line-height: 1.35;
    box-shadow: 0 4px 16px rgba(31, 41, 51, 0.16);
    max-width: 260px;
  }}
  .hint {{
    padding: 10px 14px;
    border-top: 1px solid #d8dee6;
    font-size: 12px;
    color: #52606d;
  }}
</style>
</head>
<body>
<div class="panel">
  <div class="toolbar">
    <h1>{html.escape(page_title)}</h1>
    <button id="view-xy" class="active">XY</button>
    <button id="view-xz">XZ</button>
    <button id="view-error">Error-Time</button>
    <button id="range-full" class="active">Full range</button>
    <button id="range-gt">GT region</button>
    <button id="reset">Reset view</button>
    <span id="legend"></span>
  </div>
  <div id="plot-wrap">
    <svg id="plot" role="img" aria-label="Interactive trajectory plot"></svg>
    <div id="tooltip"></div>
  </div>
  <div class="hint">Mouse wheel: zoom. Drag: pan. Double click or Reset view: reset. Toggle traces in the legend.</div>
</div>
<script>
const DATA = {payload};
const svg = document.getElementById("plot");
const wrap = document.getElementById("plot-wrap");
const tooltip = document.getElementById("tooltip");
const NS = "http://www.w3.org/2000/svg";
const state = {{
  view: "xy",
  rangeMode: "full",
  domain: null,
  visible: Object.fromEntries(DATA.traces.map(t => [t.name, true])),
  dragging: false,
  dragStart: null,
}};
const margin = {{left: 72, right: 26, top: 24, bottom: 62}};

function el(tag, attrs = {{}}, parent = svg) {{
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent.appendChild(node);
  return node;
}}

function extent(arrays) {{
  let lo = Infinity, hi = -Infinity;
  for (const arr of arrays) {{
    for (const v of arr) {{
      if (Number.isFinite(v)) {{
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }}
    }}
  }}
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (Math.abs(hi - lo) < 1e-9) return [lo - 1, hi + 1];
  return [lo, hi];
}}

function padDomain([lo, hi], ratio = 0.08, minPad = 0.2) {{
  const span = hi - lo;
  const pad = Math.max(Math.abs(span) * ratio, minPad);
  return [lo - pad, hi + pad];
}}

function naturalDomain() {{
  if (state.view === "error") {{
    const visible = DATA.traces.filter(t => state.visible[t.name]);
    return {{
      x: padDomain(extent(visible.map(t => t.t)), 0.02, 0.1),
      y: padDomain(extent(visible.map(t => t.err)), 0.08, 0.5),
    }};
  }}
  const xKey = "x";
  const yKey = state.view === "xy" ? "y" : "z";
  if (state.rangeMode === "gt") {{
    return {{
      x: padDomain(extent([DATA.gt[xKey]]), 0.16, 1.5),
      y: padDomain(extent([DATA.gt[yKey]]), 0.16, 1.5),
    }};
  }}
  const visible = DATA.traces.filter(t => state.visible[t.name]);
  return {{
    x: padDomain(extent([DATA.gt[xKey], ...visible.map(t => t[xKey])]), 0.06, 0.5),
    y: padDomain(extent([DATA.gt[yKey], ...visible.map(t => t[yKey])]), 0.06, 0.5),
  }};
}}

function setButtons() {{
  for (const id of ["view-xy", "view-xz", "view-error", "range-full", "range-gt"]) {{
    document.getElementById(id).classList.remove("active");
  }}
  document.getElementById("view-" + (state.view === "error" ? "error" : state.view)).classList.add("active");
  document.getElementById("range-" + (state.rangeMode === "gt" ? "gt" : "full")).classList.add("active");
  document.getElementById("range-gt").disabled = state.view === "error";
}}

function setView(view) {{
  state.view = view;
  if (view === "error") state.rangeMode = "full";
  state.domain = null;
  draw();
}}

function setRange(mode) {{
  if (state.view === "error" && mode === "gt") return;
  state.rangeMode = mode;
  state.domain = null;
  draw();
}}

function xMap(x, domain, w) {{
  return margin.left + (x - domain.x[0]) / (domain.x[1] - domain.x[0]) * (w - margin.left - margin.right);
}}

function yMap(y, domain, h) {{
  return margin.top + (domain.y[1] - y) / (domain.y[1] - domain.y[0]) * (h - margin.top - margin.bottom);
}}

function niceTicks([lo, hi], count = 8) {{
  const span = hi - lo;
  const raw = span / Math.max(count, 1);
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-12))));
  const candidates = [1, 2, 5, 10].map(v => v * pow);
  const step = candidates.reduce((best, v) => Math.abs(v - raw) < Math.abs(best - raw) ? v : best, candidates[0]);
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.5; v += step) ticks.push(Math.abs(v) < 1e-10 ? 0 : v);
  return ticks;
}}

function formatTick(v) {{
  const av = Math.abs(v);
  if (av >= 100) return v.toFixed(0);
  if (av >= 10) return v.toFixed(1);
  return v.toFixed(2).replace(/0+$/, "").replace(/\\.$/, "");
}}

function makePath(xs, ys, domain, w, h) {{
  let d = "";
  for (let i = 0; i < xs.length; i++) {{
    const x = xMap(xs[i], domain, w);
    const y = yMap(ys[i], domain, h);
    d += (i === 0 ? "M" : "L") + x.toFixed(2) + "," + y.toFixed(2);
  }}
  return d;
}}

function drawAxes(domain, w, h) {{
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  for (const tx of niceTicks(domain.x)) {{
    const x = xMap(tx, domain, w);
    el("line", {{x1: x, y1: margin.top, x2: x, y2: h - margin.bottom, class: "grid"}});
    el("text", {{x: x, y: h - margin.bottom + 20, "text-anchor": "middle", class: "tick-label"}}).textContent = formatTick(tx);
  }}
  for (const ty of niceTicks(domain.y)) {{
    const y = yMap(ty, domain, h);
    el("line", {{x1: margin.left, y1: y, x2: w - margin.right, y2: y, class: "grid"}});
    el("text", {{x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "tick-label"}}).textContent = formatTick(ty);
  }}
  el("line", {{x1: margin.left, y1: h - margin.bottom, x2: w - margin.right, y2: h - margin.bottom, class: "axis"}});
  el("line", {{x1: margin.left, y1: margin.top, x2: margin.left, y2: h - margin.bottom, class: "axis"}});
  const xLabel = state.view === "error" ? "time / s" : "x / m (NWU)";
  const yLabel = state.view === "xy" ? "y / m (NWU)" : state.view === "xz" ? "z / m (NWU)" : "position error / m";
  el("text", {{x: margin.left + plotW / 2, y: h - 18, "text-anchor": "middle", class: "axis-label"}}).textContent = xLabel;
  const yText = el("text", {{x: 20, y: margin.top + plotH / 2, "text-anchor": "middle", class: "axis-label", transform: `rotate(-90 20 ${{margin.top + plotH / 2}})`}});
  yText.textContent = yLabel;
}}

function draw() {{
  setButtons();
  svg.replaceChildren();
  const w = wrap.clientWidth || 1000;
  const h = wrap.clientHeight || 690;
  svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const domain = state.domain || naturalDomain();
  state.domain = domain;
  drawAxes(domain, w, h);

  if (state.view !== "error") {{
    const yKey = state.view === "xy" ? "y" : "z";
    el("path", {{d: makePath(DATA.gt.x, DATA.gt[yKey], domain, w, h), stroke: DATA.gt.color, class: "trace gt"}});
    for (const t of DATA.traces) {{
      if (!state.visible[t.name]) continue;
      el("path", {{d: makePath(t.x, t[yKey], domain, w, h), stroke: t.color, class: "trace"}});
    }}
  }} else {{
    for (const t of DATA.traces) {{
      if (!state.visible[t.name]) continue;
      el("path", {{d: makePath(t.t, t.err, domain, w, h), stroke: t.color, class: "trace"}});
    }}
  }}
}}

function buildLegend() {{
  const legend = document.getElementById("legend");
  legend.replaceChildren();
  for (const t of DATA.traces) {{
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.visible[t.name];
    input.addEventListener("change", () => {{
      state.visible[t.name] = input.checked;
      state.domain = null;
      draw();
    }});
    const swatch = document.createElement("span");
    swatch.style.display = "inline-block";
    swatch.style.width = "18px";
    swatch.style.height = "3px";
    swatch.style.background = t.color;
    const text = document.createElement("span");
    text.textContent = t.name;
    label.append(input, swatch, text);
    legend.appendChild(label);
  }}
}}

function invertX(px, domain, w) {{
  return domain.x[0] + (px - margin.left) / (w - margin.left - margin.right) * (domain.x[1] - domain.x[0]);
}}

function invertY(py, domain, h) {{
  return domain.y[1] - (py - margin.top) / (h - margin.top - margin.bottom) * (domain.y[1] - domain.y[0]);
}}

function zoomAt(factor, px, py) {{
  const w = wrap.clientWidth || 1000;
  const h = wrap.clientHeight || 690;
  const d = state.domain;
  const cx = invertX(px, d, w);
  const cy = invertY(py, d, h);
  state.domain = {{
    x: [cx + (d.x[0] - cx) * factor, cx + (d.x[1] - cx) * factor],
    y: [cy + (d.y[0] - cy) * factor, cy + (d.y[1] - cy) * factor],
  }};
  draw();
}}

function findNearest(px, py) {{
  const w = wrap.clientWidth || 1000;
  const h = wrap.clientHeight || 690;
  const d = state.domain;
  const candidates = [];
  if (state.view !== "error") {{
    const yKey = state.view === "xy" ? "y" : "z";
    candidates.push({{name: DATA.gt.name, color: DATA.gt.color, t: DATA.gt.t, x: DATA.gt.x, y: DATA.gt[yKey], z: DATA.gt.z, rawX: DATA.gt.x, rawY: DATA.gt.y}});
    for (const t of DATA.traces) {{
      if (state.visible[t.name]) candidates.push({{name: t.name, color: t.color, t: t.t, x: t.x, y: t[yKey], z: t.z, rawX: t.x, rawY: t.y, err: t.err}});
    }}
  }} else {{
    for (const t of DATA.traces) {{
      if (state.visible[t.name]) candidates.push({{name: t.name, color: t.color, t: t.t, x: t.t, y: t.err, err: t.err}});
    }}
  }}
  let best = null;
  for (const c of candidates) {{
    for (let i = 0; i < c.x.length; i++) {{
      const sx = xMap(c.x[i], d, w);
      const sy = yMap(c.y[i], d, h);
      const dist = Math.hypot(sx - px, sy - py);
      if (!best || dist < best.dist) best = {{...c, i, sx, sy, dist}};
    }}
  }}
  return best && best.dist < 18 ? best : null;
}}

svg.addEventListener("wheel", ev => {{
  ev.preventDefault();
  const rect = svg.getBoundingClientRect();
  zoomAt(ev.deltaY < 0 ? 0.82 : 1.22, ev.clientX - rect.left, ev.clientY - rect.top);
}}, {{passive: false}});

svg.addEventListener("pointerdown", ev => {{
  const rect = svg.getBoundingClientRect();
  state.dragging = true;
  state.dragStart = {{x: ev.clientX - rect.left, y: ev.clientY - rect.top, domain: JSON.parse(JSON.stringify(state.domain))}};
  svg.setPointerCapture(ev.pointerId);
}});

svg.addEventListener("pointermove", ev => {{
  const rect = svg.getBoundingClientRect();
  const px = ev.clientX - rect.left;
  const py = ev.clientY - rect.top;
  if (state.dragging && state.dragStart) {{
    const w = wrap.clientWidth || 1000;
    const h = wrap.clientHeight || 690;
    const dx = (px - state.dragStart.x) / (w - margin.left - margin.right) * (state.dragStart.domain.x[1] - state.dragStart.domain.x[0]);
    const dy = (py - state.dragStart.y) / (h - margin.top - margin.bottom) * (state.dragStart.domain.y[1] - state.dragStart.domain.y[0]);
    state.domain = {{
      x: [state.dragStart.domain.x[0] - dx, state.dragStart.domain.x[1] - dx],
      y: [state.dragStart.domain.y[0] + dy, state.dragStart.domain.y[1] + dy],
    }};
    draw();
    return;
  }}
  const near = findNearest(px, py);
  if (!near) {{
    tooltip.style.display = "none";
    return;
  }}
  const i = near.i;
  const parts = [
    `<b style="color:${{near.color}}">${{near.name}}</b>`,
    `frame: ${{i}}`,
    `t: ${{near.t[i].toFixed(3)}} s`,
  ];
  if (state.view === "error") {{
    parts.push(`error: ${{near.err[i].toFixed(3)}} m`);
  }} else {{
    parts.push(`x: ${{near.rawX[i].toFixed(3)}} m`);
    parts.push(`y: ${{near.rawY[i].toFixed(3)}} m`);
    parts.push(`z: ${{near.z[i].toFixed(3)}} m`);
    if (near.err) parts.push(`error: ${{near.err[i].toFixed(3)}} m`);
  }}
  tooltip.innerHTML = parts.join("<br>");
  tooltip.style.left = Math.min(px + 14, wrap.clientWidth - 250) + "px";
  tooltip.style.top = Math.max(py + 14, 8) + "px";
  tooltip.style.display = "block";
}});

svg.addEventListener("pointerup", ev => {{
  state.dragging = false;
  state.dragStart = null;
  try {{ svg.releasePointerCapture(ev.pointerId); }} catch (_) {{}}
}});
svg.addEventListener("pointerleave", () => {{
  state.dragging = false;
  state.dragStart = null;
  tooltip.style.display = "none";
}});
svg.addEventListener("dblclick", () => {{
  state.domain = null;
  draw();
}});

document.getElementById("view-xy").addEventListener("click", () => setView("xy"));
document.getElementById("view-xz").addEventListener("click", () => setView("xz"));
document.getElementById("view-error").addEventListener("click", () => setView("error"));
document.getElementById("range-full").addEventListener("click", () => setRange("full"));
document.getElementById("range-gt").addEventListener("click", () => setRange("gt"));
document.getElementById("reset").addEventListener("click", () => {{
  state.domain = null;
  draw();
}});
window.addEventListener("resize", () => {{
  state.domain = null;
  draw();
}});

buildLegend();
draw();
</script>
</body>
</html>
"""
    (outdir / "interactive_trajectory_gt_vs_est.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, nargs="+", default=[DEFAULT_RESULT_ROOT])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = read_manifests(list(args.result_root))
    summaries: list[dict[str, object]] = []
    trajs: dict[str, pd.DataFrame] = {}
    for row in rows:
        summary, joined = evaluate_run(row)
        summaries.append(summary)
        trajs[str(summary["label"])] = joined

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.outdir / "summary.csv", index=False)
    plot_xy(trajs, args.outdir)
    plot_xy_gt_region(trajs, args.outdir)
    plot_xz(trajs, args.outdir)
    plot_error(trajs, args.outdir)
    write_interactive_html(trajs, args.outdir)

    cols = [
        "scene",
        "variant",
        "label",
        "matched_frames",
        "ate_rmse_m",
        "ate_median_m",
        "ate_final_m",
        "ate_max_m",
        "gt_path_length_m",
        "result_dir",
    ]
    md = [
        "# Clear-circle paired VIO analysis",
        "",
        summary_df[cols].to_markdown(index=False, floatfmt=".6g"),
        "",
        f"- Summary CSV: `{args.outdir / 'summary.csv'}`",
        f"- XY plot: `{args.outdir / 'trajectory_xy_gt_vs_est.png'}`",
        f"- XY GT-region plot: `{args.outdir / 'trajectory_xy_gt_region.png'}`",
        f"- XZ plot: `{args.outdir / 'trajectory_xz_gt_vs_est.png'}`",
        f"- Error plot: `{args.outdir / 'position_error_over_time.png'}`",
        f"- Interactive plot: `{args.outdir / 'interactive_trajectory_gt_vs_est.html'}`",
    ]
    (args.outdir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(summary_df[cols].to_string(index=False))
    print(f"Wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

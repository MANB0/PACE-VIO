#!/usr/bin/env python3
"""Create an interactive three-geometry trajectory comparison viewer."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


WORKDIR = Path("/home/admin1/macvo-dev")
BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_clear_truth_paths_20260713_static63_variants")
SOURCE_ROOT = WORKDIR / "Results" / "visual_factor_cache_static63_unique_source_20260713" / "trial_1" / "pure_macvo"
FUSION_ROOT = (
    WORKDIR
    / "Results"
    / "static63_cached_imu_fusion_four_configs_20260713"
    / "trial_1"
    / "vio_preintegrated_full_imuatt_estinit"
)
IMU_ONLY_ROOT = (
    WORKDIR
    / "Results"
    / "static63_calibrated_imu_only_four_configs_20260713"
    / "trajectories"
)
OUTDIR = WORKDIR / "analysis_static63_three_geometry_20260713"
SCENES = (
    "clear_circle_truth_normal_noise",
    "clear_stop_turn_rectangle_truth_normal_noise",
    "clear_straight_truth_normal_noise",
)
IMU_CONFIGS = (
    ("normal_noise", "Normal noise", "#2563eb"),
    ("bias_no_noise", "Bias only", "#16a34a"),
    ("noise_no_bias", "White noise only", "#9333ea"),
    ("no_noise_no_bias", "No bias / no noise", "#dc2626"),
)


def read_xyz(path: Path) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            timestamp = int(float(row.get("timestamp_ns") or row.get("timestamp") or 0))
            rows.append((timestamp, float(row["x"] if "x" in row else row["tx"]),
                         float(row["y"] if "y" in row else row["ty"]),
                         float(row["z"] if "z" in row else row["tz"])))
    return rows


def read_forward_axes(path: Path) -> list[list[float]]:
    """Read each pose's local +X (vehicle forward) axis in world coordinates."""
    axes: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        quaternion_fields = ("qx", "qy", "qz", "qw")
        if reader.fieldnames is None or not all(
            field in reader.fieldnames for field in quaternion_fields
        ):
            raise ValueError(f"pose CSV does not contain qx/qy/qz/qw: {path}")
        for row in reader:
            x, y, z, w = (float(row[field]) for field in quaternion_fields)
            norm = math.sqrt(x * x + y * y + z * z + w * w)
            if not math.isfinite(norm) or norm < 1e-12:
                axes.append([float("nan"), float("nan"), float("nan")])
                continue
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
            axes.append(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y + w * z),
                    2.0 * (x * z - w * y),
                ]
            )
    return axes


def metrics(gt: list[tuple[int, float, float, float]], est: list[tuple[int, float, float, float]]) -> dict[str, float | int]:
    n = min(len(gt), len(est))
    errors = []
    for i in range(n):
        dx = est[i][1] - gt[i][1]
        dy = est[i][2] - gt[i][2]
        dz = est[i][3] - gt[i][3]
        errors.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    rmse = math.sqrt(sum(e * e for e in errors) / n) if n else float("nan")
    return {
        "frames": n,
        "rmse_m": rmse,
        "mean_m": sum(errors) / n if n else float("nan"),
        "max_m": max(errors) if errors else float("nan"),
        "final_m": errors[-1] if errors else float("nan"),
    }


def payload_for_scene(
    scene: str,
    *,
    fusion_sources: list[tuple[str, Path, str]],
    include_imu_only: bool = False,
) -> dict:
    gt_path = BATCH_ROOT / scene / "ref_pose.csv"
    est_path = SOURCE_ROOT / scene / "poses.csv"
    gt_rows = read_xyz(gt_path)
    est_rows = read_xyz(est_path)
    gt_forward = read_forward_axes(gt_path)
    est_forward = read_forward_axes(est_path)
    n = min(len(gt_rows), len(est_rows))
    gt = [[x, y, z] for _, x, y, z in gt_rows[:n]]
    est = [[x, y, z] for _, x, y, z in est_rows[:n]]
    t0 = gt_rows[0][0] if gt_rows else 0
    time_s = [(gt_rows[i][0] - t0) / 1e9 for i in range(n)]
    err = [
        math.sqrt((est[i][0] - gt[i][0]) ** 2 + (est[i][1] - gt[i][1]) ** 2 + (est[i][2] - gt[i][2]) ** 2)
        for i in range(n)
    ]
    if scene.startswith("clear_circle"):
        prefix = "clear_circle_truth"
    elif scene.startswith("clear_stop_turn_rectangle"):
        prefix = "clear_stop_turn_rectangle_truth"
    elif scene.startswith("clear_straight"):
        prefix = "clear_straight_truth"
    else:
        raise ValueError(f"unsupported scene: {scene}")
    fusion = []
    imu_only = []
    for source_index, (source_label, fusion_root, dasharray) in enumerate(fusion_sources):
        for suffix, label, color in IMU_CONFIGS:
            fusion_scene = f"{prefix}_{suffix}"
            fusion_path = fusion_root / fusion_scene / "poses.csv"
            if not fusion_path.exists():
                raise FileNotFoundError(f"missing cached IMU fusion result: {fusion_path}")
            fusion_rows = read_xyz(fusion_path)
            fusion_forward = read_forward_axes(fusion_path)
            if len(fusion_rows) < n:
                raise ValueError(
                    f"fusion result is shorter than GT/MACVO for {fusion_scene}: "
                    f"{len(fusion_rows)} < {n}"
                )
            fusion_xyz = [[x, y, z] for _, x, y, z in fusion_rows[:n]]
            fusion_error = [
                math.sqrt(
                    (fusion_xyz[i][0] - gt[i][0]) ** 2
                    + (fusion_xyz[i][1] - gt[i][1]) ** 2
                    + (fusion_xyz[i][2] - gt[i][2]) ** 2
                )
                for i in range(n)
            ]
            fusion.append(
                {
                    "key": f"{source_index}_{suffix}",
                    "source": source_label,
                    "config": suffix,
                    "label": f"{source_label} · {label}",
                    "color": color,
                    "dasharray": dasharray,
                    "scene": fusion_scene,
                    "xyz": fusion_xyz,
                    "forward": fusion_forward[:n],
                    "error_m": fusion_error,
                    "metrics": metrics(gt_rows, fusion_rows),
                    "path": str(fusion_path),
                }
            )
    for suffix, label, color in IMU_CONFIGS:
        fusion_scene = f"{prefix}_{suffix}"
        if include_imu_only:
            imu_only_path = (
                IMU_ONLY_ROOT
                / f"{fusion_scene}_imu_only_staticinit_calibrated_poses.csv"
            )
            if not imu_only_path.exists():
                raise FileNotFoundError(f"missing IMU-only result: {imu_only_path}")
            imu_only_rows = read_xyz(imu_only_path)
            imu_only_forward = read_forward_axes(imu_only_path)
            if len(imu_only_rows) < n:
                raise ValueError(
                    f"IMU-only result is shorter than GT/MACVO for {fusion_scene}: "
                    f"{len(imu_only_rows)} < {n}"
                )
            imu_only_xyz = [[x, y, z] for _, x, y, z in imu_only_rows[:n]]
            imu_only_error = [
                math.sqrt(
                    (imu_only_xyz[i][0] - gt[i][0]) ** 2
                    + (imu_only_xyz[i][1] - gt[i][1]) ** 2
                    + (imu_only_xyz[i][2] - gt[i][2]) ** 2
                )
                for i in range(n)
            ]
            imu_only.append(
                {
                    "key": suffix,
                    "config": suffix,
                    "label": f"Staticinit calibrated / {label}",
                    "color": color,
                    "scene": fusion_scene,
                    "xyz": imu_only_xyz,
                    "forward": imu_only_forward[:n],
                    "error_m": imu_only_error,
                    "metrics": metrics(gt_rows, imu_only_rows),
                    "path": str(imu_only_path),
                }
            )
    return {
        "scene": scene,
        "gt": gt,
        "gt_forward": gt_forward[:n],
        "macvo": est,
        "macvo_forward": est_forward[:n],
        "time_s": time_s,
        "error_m": err,
        "metrics": metrics(gt_rows, est_rows),
        "fusion": fusion,
        "imu_only": imu_only,
        "gt_path": str(gt_path),
        "macvo_path": str(est_path),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Circle, stop-turn rectangle, and straight trajectory comparison</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #1f2933; background: #f4f6f8; }}
    header {{ padding: 18px 24px 10px; background: #fff; border-bottom: 1px solid #d8dee6; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ color: #52606d; font-size: 13px; }}
    main {{ padding: 18px 24px 28px; display: grid; gap: 18px; }}
    .card {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; }}
    .bar {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 12px 14px; border-bottom: 1px solid #d8dee6; }}
    .title {{ font-weight: 700; margin-right: 12px; }}
    button, select {{ border: 1px solid #bcccdc; background: #f8fafc; border-radius: 6px; padding: 7px 12px; font-size: 14px; cursor: pointer; }}
    button.active {{ background: #2f80ed; color: white; border-color: #2f80ed; }}
    label {{ display: inline-flex; align-items: center; gap: 5px; font-size: 14px; }}
    input[type=checkbox] {{ width: 16px; height: 16px; }}
    svg {{ width: 100%; aspect-ratio: 1000 / 620; display: block; background: #f9fbfc; touch-action: none; }}
    .axis-label {{ font-size: 13px; fill: #334e68; font-weight: 600; }}
    .tick {{ font-size: 11px; fill: #52606d; }}
    .grid {{ stroke: #d9e2ec; stroke-width: 1; }}
    .gt {{ fill: none; stroke: #111827; stroke-width: 3.0; vector-effect: non-scaling-stroke; }}
    .macvo {{ fill: none; stroke: #e8590c; stroke-width: 2.2; vector-effect: non-scaling-stroke; }}
    .fusion {{ fill: none; stroke-width: 2.0; vector-effect: non-scaling-stroke; }}
    .imu-only {{ fill: none; stroke-width: 2.0; stroke-dasharray: 8 5; vector-effect: non-scaling-stroke; }}
    .gt-head {{ fill: #111827; stroke: #fff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }}
    .macvo-head {{ fill: #e8590c; stroke: #fff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }}
    .fusion-head {{ stroke: #fff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }}
    .imu-only-head {{ stroke: #fff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }}
    .heading-arrow {{ pointer-events: none; }}
    .playback {{ display: flex; align-items: center; gap: 9px; flex: 1 1 420px; min-width: 260px; }}
    .playback input[type=range] {{ flex: 1; min-width: 140px; }}
    .time-readout {{ min-width: 118px; color: #52606d; font-variant-numeric: tabular-nums; font-size: 13px; }}
    .legend {{ display: flex; gap: 18px; align-items: center; padding: 10px 14px; border-top: 1px solid #d8dee6; font-size: 14px; flex-wrap: wrap; }}
    .swatch {{ display: inline-block; width: 26px; height: 0; border-top: 4px solid; margin-right: 6px; vertical-align: middle; }}
    .stats {{ margin-left: auto; color: #52606d; }}
    .hint {{ color: #627d98; font-size: 13px; padding: 0 14px 12px; }}
  </style>
</head>
<body>
<header>
  <h1>Circle, stop-turn rectangle, and straight trajectory comparison</h1>
  <div class="meta">__METHOD_SCOPE__ in NWU. No alignment, no SE(3) fitting, no scale correction.</div>
</header>
<main id="app"></main>
<script>
const DATA = __DATA__;

function extent(values) {{
  let min = Infinity, max = -Infinity;
  for (const v of values) {{ if (v < min) min = v; if (v > max) max = v; }}
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (Math.abs(max - min) < 1e-9) {{ min -= 0.5; max += 0.5; }}
  return [min, max];
}}

function linePath(points, map) {{
  return points.map((p, i) => `${{i ? "L" : "M"}}${{map.x(p[0]).toFixed(2)}} ${{map.y(p[1]).toFixed(2)}}`).join(" ");
}}

function headingMarker(point, forward, map, color, cssClass, fallbackRadius) {{
  const cx = map.x(point[0]), cy = map.y(point[1]);
  if (!forward || !Number.isFinite(forward[0]) || !Number.isFinite(forward[1])) {{
    return `<circle class="${{cssClass}}" fill="${{color}}" cx="${{cx}}" cy="${{cy}}" r="${{fallbackRadius}}"/>`;
  }}
  const projectedX = map.x(point[0] + forward[0]) - cx;
  const projectedY = map.y(point[1] + forward[1]) - cy;
  const norm = Math.hypot(projectedX, projectedY);
  if (norm < 1e-7) {{
    return `<circle class="${{cssClass}}" fill="${{color}}" cx="${{cx}}" cy="${{cy}}" r="${{fallbackRadius}}"/>`;
  }}
  const dx = projectedX / norm, dy = projectedY / norm;
  const px = -dy, py = dx;
  const tailX = cx - 8 * dx, tailY = cy - 8 * dy;
  const tipX = cx + 15 * dx, tipY = cy + 15 * dy;
  const baseX = cx + 5 * dx, baseY = cy + 5 * dy;
  const wing = 6;
  const arrowPoints = [
    `${{tipX.toFixed(2)}},${{tipY.toFixed(2)}}`,
    `${{(baseX + wing * px).toFixed(2)}},${{(baseY + wing * py).toFixed(2)}}`,
    `${{(baseX - wing * px).toFixed(2)}},${{(baseY - wing * py).toFixed(2)}}`,
  ].join(" ");
  return `<g class="heading-arrow ${{cssClass}}">
    <line x1="${{tailX}}" y1="${{tailY}}" x2="${{tipX}}" y2="${{tipY}}" stroke="#fff" stroke-width="7" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
    <line x1="${{tailX}}" y1="${{tailY}}" x2="${{tipX}}" y2="${{tipY}}" stroke="${{color}}" stroke-width="3.5" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
    <polygon points="${{arrowPoints}}" fill="${{color}}" stroke="#fff" stroke-width="1.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
  </g>`;
}}

function makeChart(sceneData) {{
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML = `
    <div class="bar">
      <span class="title">${{sceneData.scene}}</span>
      <button data-view="xy" class="active">XY</button>
      <button data-view="xz">XZ</button>
      <button data-view="yz">YZ</button>
      <button data-view="err">Error-Time</button>
      <button data-range="full" class="active">Full range</button>
      <button data-range="gt">GT region</button>
      <button data-reset>Reset</button>
      <label><input type="checkbox" data-show-gt checked> GT</label>
      <label><input type="checkbox" data-show-macvo checked> MACVO</label>
      <label>IMU data
        <select data-config-filter aria-label="Filter trajectories by IMU data configuration">
          <option value="all">All configurations</option>
          <option value="normal_noise">Normal noise + bias</option>
          <option value="bias_no_noise">Bias only</option>
          <option value="noise_no_bias">White noise only</option>
          <option value="no_noise_no_bias">No bias / no noise</option>
        </select>
      </label>
      ${{sceneData.fusion.map(item => `<label data-trace-config="${{item.config}}"><input type="checkbox" data-show-fusion="${{item.key}}" checked> Fusion · ${{item.label}}</label>`).join("")}}
      ${{sceneData.imu_only.map(item => `<label data-trace-config="${{item.config}}"><input type="checkbox" data-show-imu-only="${{item.key}}" checked> IMU-only · ${{item.label}}</label>`).join("")}}
      <div class="playback">
        <button data-play aria-label="Play synchronized trajectories">Play</button>
        <input data-frame type="range" min="0" max="${{sceneData.metrics.frames - 1}}" value="${{sceneData.metrics.frames - 1}}" step="1" aria-label="Trajectory frame">
        <span class="time-readout" data-time></span>
        <select data-speed aria-label="Playback speed">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
    </div>
    <svg viewBox="0 0 1000 620" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Synchronized GT and MACVO trajectory playback with equal metric axis scale"></svg>
    <div class="legend">
      <span><span class="swatch" style="border-color:#111827"></span>GT</span>
      <span><span class="swatch" style="border-color:#e8590c"></span>Pure MACVO</span>
      ${{sceneData.fusion.map(item => `<span data-legend-config="${{item.config}}"><span class="swatch" style="border-color:${{item.color}};border-top-style:${{item.dasharray ? "dashed" : "solid"}}"></span>Fusion · ${{item.label}}</span>`).join("")}}
      ${{sceneData.imu_only.map(item => `<span data-legend-config="${{item.config}}"><span class="swatch" style="border-color:${{item.color}};border-top-style:dashed"></span>IMU-only · ${{item.label}}</span>`).join("")}}
      <span class="stats">MACVO RMSE ${{sceneData.metrics.rmse_m.toFixed(3)}} m · frames ${{sceneData.metrics.frames}}</span>
    </div>
    <div class="hint">__LINE_NOTE__ Play or drag the timeline to compare the same instant. XY, XZ and YZ use an equal metric scale.</div>
  `;
  const svg = card.querySelector("svg");
  const frameSlider = card.querySelector("[data-frame]");
  const playButton = card.querySelector("[data-play]");
  const speedSelect = card.querySelector("[data-speed]");
  const configFilter = card.querySelector("[data-config-filter]");
  const timeReadout = card.querySelector("[data-time]");
  let state = {{
    view: "xy", range: "full", xlim: null, ylim: null, dragging: false, last: null,
    frame: sceneData.metrics.frames - 1, playing: false, animationId: null, lastAnimationTime: null, playTime: 0,
    configFilter: "all",
  }};

  function configVisible(item) {{
    return state.configFilter === "all" || item.config === state.configFilter;
  }}

  function updateFilterVisibility() {{
    card.querySelectorAll("[data-trace-config]").forEach(node => {{
      node.hidden = state.configFilter !== "all" && node.dataset.traceConfig !== state.configFilter;
    }});
    card.querySelectorAll("[data-legend-config]").forEach(node => {{
      node.hidden = state.configFilter !== "all" && node.dataset.legendConfig !== state.configFilter;
    }});
  }}

  function getSeries() {{
    if (state.view === "err") {{
      return {{
        gt: null,
        gtForward: null,
        macvo: sceneData.time_s.map((t, i) => [t, sceneData.error_m[i]]),
        macvoForward: null,
        fusion: sceneData.fusion.map(item => ({{
          ...item,
          points: sceneData.time_s.map((t, i) => [t, item.error_m[i]]),
        }})),
        imuOnly: sceneData.imu_only.map(item => ({{
          ...item,
          points: sceneData.time_s.map((t, i) => [t, item.error_m[i]]),
        }})),
        xlabel: "time / s",
        ylabel: "3D error / m",
      }};
    }}
    const idx = state.view === "xy" ? [0, 1] : state.view === "xz" ? [0, 2] : [1, 2];
    const labels = ["x / m (NWU)", "y / m (NWU)", "z / m (NWU)"];
    return {{
      gt: sceneData.gt.map(p => [p[idx[0]], p[idx[1]]]),
      gtForward: (sceneData.gt_forward || []).map(p => [p[idx[0]], p[idx[1]]]),
      macvo: sceneData.macvo.map(p => [p[idx[0]], p[idx[1]]]),
      macvoForward: (sceneData.macvo_forward || []).map(p => [p[idx[0]], p[idx[1]]]),
      fusion: sceneData.fusion.map(item => ({{
        ...item,
        points: item.xyz.map(p => [p[idx[0]], p[idx[1]]]),
        forward: (item.forward || []).map(p => [p[idx[0]], p[idx[1]]]),
      }})),
      imuOnly: sceneData.imu_only.map(item => ({{
        ...item,
        points: item.xyz.map(p => [p[idx[0]], p[idx[1]]]),
        forward: (item.forward || []).map(p => [p[idx[0]], p[idx[1]]]),
      }})),
      xlabel: labels[idx[0]],
      ylabel: labels[idx[1]],
    }};
  }}

  function computeLimits(series) {{
    const curves = [];
    if (state.range === "gt" && series.gt) curves.push(series.gt);
    else {{
      if (series.gt) curves.push(series.gt);
      curves.push(series.macvo);
      curves.push(...series.fusion.filter(configVisible).map(item => item.points));
      curves.push(...series.imuOnly.filter(configVisible).map(item => item.points));
    }}
    const xs = curves.flatMap(c => c.map(p => p[0]));
    const ys = curves.flatMap(c => c.map(p => p[1]));
    let [xmin, xmax] = extent(xs), [ymin, ymax] = extent(ys);
    const padX = (xmax - xmin) * 0.08;
    const padY = (ymax - ymin) * 0.08;
    xmin -= padX; xmax += padX; ymin -= padY; ymax += padY;
    if (state.view !== "err") {{
      const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
      const plotAspect = 904 / 536;
      let spanX = xmax - xmin;
      let spanY = ymax - ymin;
      if (spanX / spanY < plotAspect) spanX = spanY * plotAspect;
      else spanY = spanX / plotAspect;
      xmin = cx - spanX / 2; xmax = cx + spanX / 2;
      ymin = cy - spanY / 2; ymax = cy + spanY / 2;
    }}
    return {{ xlim: [xmin, xmax], ylim: [ymin, ymax] }};
  }}

  function resetLimits() {{
    const limits = computeLimits(getSeries());
    state.xlim = limits.xlim;
    state.ylim = limits.ylim;
  }}

  function render() {{
    const series = getSeries();
    if (!state.xlim || !state.ylim) resetLimits();
    const W = 1000, H = 620, L = 72, R = 24, T = 26, B = 58;
    const pw = W - L - R, ph = H - T - B;
    const [xmin, xmax] = state.xlim, [ymin, ymax] = state.ylim;
    const map = {{
      x: v => L + (v - xmin) / (xmax - xmin) * pw,
      y: v => T + (1 - (v - ymin) / (ymax - ymin)) * ph,
    }};
    let html = "";
    for (let i = 0; i <= 8; i++) {{
      const x = L + i * pw / 8, y = T + i * ph / 8;
      const xv = xmin + i * (xmax - xmin) / 8;
      const yv = ymax - i * (ymax - ymin) / 8;
      html += `<line class="grid" x1="${{x}}" y1="${{T}}" x2="${{x}}" y2="${{T+ph}}"/>`;
      html += `<line class="grid" x1="${{L}}" y1="${{y}}" x2="${{L+pw}}" y2="${{y}}"/>`;
      html += `<text class="tick" x="${{x}}" y="${{T+ph+22}}" text-anchor="middle">${{xv.toFixed(1)}}</text>`;
      html += `<text class="tick" x="${{L-10}}" y="${{y+4}}" text-anchor="end">${{yv.toFixed(1)}}</text>`;
    }}
    const lastFrame = Math.max(0, Math.min(state.frame, sceneData.metrics.frames - 1));
    const visibleGt = series.gt ? series.gt.slice(0, lastFrame + 1) : null;
    const visibleMacvo = series.macvo.slice(0, lastFrame + 1);
    if (visibleGt && card.querySelector("[data-show-gt]").checked) {{
      html += `<path class="gt" d="${{linePath(visibleGt, map)}}"/>`;
      const head = visibleGt[visibleGt.length - 1];
      html += headingMarker(head, series.gtForward?.[lastFrame], map, "#111827", "gt-head", 5);
    }}
    if (card.querySelector("[data-show-macvo]").checked) {{
      html += `<path class="macvo" d="${{linePath(visibleMacvo, map)}}"/>`;
      const head = visibleMacvo[visibleMacvo.length - 1];
      html += headingMarker(head, series.macvoForward?.[lastFrame], map, "#e8590c", "macvo-head", 5);
    }}
    for (const item of series.fusion) {{
      if (!configVisible(item)) continue;
      const checkbox = card.querySelector(`[data-show-fusion="${{item.key}}"]`);
      if (!checkbox || !checkbox.checked) continue;
      const visible = item.points.slice(0, lastFrame + 1);
      html += `<path class="fusion" stroke="${{item.color}}" stroke-dasharray="${{item.dasharray}}" d="${{linePath(visible, map)}}"/>`;
      const head = visible[visible.length - 1];
      html += headingMarker(head, item.forward?.[lastFrame], map, item.color, "fusion-head", 4.5);
    }}
    for (const item of series.imuOnly) {{
      if (!configVisible(item)) continue;
      const checkbox = card.querySelector(`[data-show-imu-only="${{item.key}}"]`);
      if (!checkbox || !checkbox.checked) continue;
      const visible = item.points.slice(0, lastFrame + 1);
      html += `<path class="imu-only" stroke="${{item.color}}" d="${{linePath(visible, map)}}"/>`;
      const head = visible[visible.length - 1];
      html += headingMarker(head, item.forward?.[lastFrame], map, item.color, "imu-only-head", 4);
    }}
    if (state.view === "err") {{
      const cursorX = map.x(sceneData.time_s[lastFrame]);
      html += `<line x1="${{cursorX}}" y1="${{T}}" x2="${{cursorX}}" y2="${{T+ph}}" stroke="#52606d" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
    }}
    html += `<text class="axis-label" x="${{L+pw/2}}" y="${{H-18}}" text-anchor="middle">${{series.xlabel}}</text>`;
    html += `<text class="axis-label" transform="translate(20 ${{T+ph/2}}) rotate(-90)" text-anchor="middle">${{series.ylabel}}</text>`;
    svg.innerHTML = html;
    frameSlider.value = String(lastFrame);
    const currentTime = sceneData.time_s[lastFrame] || 0;
    const totalTime = sceneData.time_s[sceneData.time_s.length - 1] || 0;
    timeReadout.textContent = `${{currentTime.toFixed(2)}} / ${{totalTime.toFixed(2)}} s`;
  }}

  function stopPlayback() {{
    state.playing = false;
    state.lastAnimationTime = null;
    playButton.textContent = "Play";
    playButton.classList.remove("active");
    if (state.animationId !== null) cancelAnimationFrame(state.animationId);
    state.animationId = null;
  }}

  function animationStep(now) {{
    if (!state.playing) return;
    if (state.lastAnimationTime === null) state.lastAnimationTime = now;
    const elapsed = (now - state.lastAnimationTime) / 1000 * Number(speedSelect.value);
    state.lastAnimationTime = now;
    state.playTime += elapsed;
    let nextFrame = state.frame;
    while (nextFrame + 1 < sceneData.time_s.length && sceneData.time_s[nextFrame + 1] <= state.playTime) nextFrame++;
    state.frame = nextFrame;
    render();
    if (state.frame >= sceneData.metrics.frames - 1) stopPlayback();
    else state.animationId = requestAnimationFrame(animationStep);
  }}

  playButton.onclick = () => {{
    if (state.playing) {{ stopPlayback(); return; }}
    if (state.frame >= sceneData.metrics.frames - 1) state.frame = 0;
    state.playTime = sceneData.time_s[state.frame] || 0;
    state.playing = true;
    state.lastAnimationTime = null;
    playButton.textContent = "Pause";
    playButton.classList.add("active");
    render();
    state.animationId = requestAnimationFrame(animationStep);
  }};
  frameSlider.oninput = () => {{ stopPlayback(); state.frame = Number(frameSlider.value); render(); }};

  card.querySelectorAll("button[data-view]").forEach(btn => btn.onclick = () => {{
    card.querySelectorAll("button[data-view]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.view = btn.dataset.view;
    state.xlim = state.ylim = null;
    render();
  }});
  card.querySelectorAll("button[data-range]").forEach(btn => btn.onclick = () => {{
    card.querySelectorAll("button[data-range]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.range = btn.dataset.range;
    state.xlim = state.ylim = null;
    render();
  }});
  card.querySelector("[data-reset]").onclick = () => {{ state.xlim = state.ylim = null; render(); }};
  card.querySelector("[data-show-gt]").onchange = render;
  card.querySelector("[data-show-macvo]").onchange = render;
  configFilter.onchange = () => {{
    state.configFilter = configFilter.value;
    state.xlim = state.ylim = null;
    updateFilterVisibility();
    render();
  }};
  card.querySelectorAll("[data-show-fusion]").forEach(input => input.onchange = render);
  card.querySelectorAll("[data-show-imu-only]").forEach(input => input.onchange = render);
  svg.addEventListener("wheel", e => {{
    e.preventDefault();
    const factor = e.deltaY < 0 ? 0.85 : 1.18;
    const rect = svg.getBoundingClientRect();
    const rx = (e.clientX - rect.left) / rect.width;
    const ry = (e.clientY - rect.top) / rect.height;
    const cx = state.xlim[0] + rx * (state.xlim[1] - state.xlim[0]);
    const cy = state.ylim[1] - ry * (state.ylim[1] - state.ylim[0]);
    state.xlim = [cx + (state.xlim[0] - cx) * factor, cx + (state.xlim[1] - cx) * factor];
    state.ylim = [cy + (state.ylim[0] - cy) * factor, cy + (state.ylim[1] - cy) * factor];
    render();
  }});
  svg.addEventListener("pointerdown", e => {{ state.dragging = true; state.last = [e.clientX, e.clientY]; svg.setPointerCapture(e.pointerId); }});
  svg.addEventListener("pointermove", e => {{
    if (!state.dragging) return;
    const dx = e.clientX - state.last[0], dy = e.clientY - state.last[1];
    state.last = [e.clientX, e.clientY];
    const rect = svg.getBoundingClientRect();
    const sx = (state.xlim[1] - state.xlim[0]) / rect.width;
    const sy = (state.ylim[1] - state.ylim[0]) / rect.height;
    state.xlim = [state.xlim[0] - dx * sx, state.xlim[1] - dx * sx];
    state.ylim = [state.ylim[0] + dy * sy, state.ylim[1] + dy * sy];
    render();
  }});
  svg.addEventListener("pointerup", () => state.dragging = false);
  svg.addEventListener("pointercancel", () => state.dragging = false);
  resetLimits();
  updateFilterVisibility();
  render();
  return card;
}}

const app = document.getElementById("app");
for (const scene of DATA.scenes) app.appendChild(makeChart(scene));
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion-root",
        type=Path,
        default=FUSION_ROOT,
        help="Directory containing one fusion result subdirectory per IMU configuration.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUTDIR,
        help="Output directory for the interactive HTML and summary CSV.",
    )
    parser.add_argument(
        "--fusion-label",
        default="cached IMU fusion",
        help="Short label used to identify the selected fusion method in the page.",
    )
    parser.add_argument(
        "--comparison-fusion",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional historical fusion root to draw with a dashed line. Repeatable.",
    )
    parser.add_argument(
        "--include-imu-only",
        action="store_true",
        help="Include completed IMU-only trajectories in addition to cached imuatt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fusion_sources = [(args.fusion_label, args.fusion_root, "")]
    for index, item in enumerate(args.comparison_fusion):
        if "=" not in item:
            raise ValueError(f"comparison fusion must be LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        dasharray = "8 5" if index == 0 else "3 4"
        fusion_sources.append((label.strip(), Path(path).expanduser(), dasharray))
    args.outdir.mkdir(parents=True, exist_ok=True)
    scenes = [
        payload_for_scene(
            scene,
            fusion_sources=fusion_sources,
            include_imu_only=args.include_imu_only,
        )
        for scene in SCENES
    ]

    summary_path = args.outdir / "gt_vs_macvo_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scene",
                "method",
                "frames",
                "rmse_m",
                "mean_m",
                "max_m",
                "final_m",
                "gt_path",
                "estimate_path",
            ],
        )
        writer.writeheader()
        for item in scenes:
            writer.writerow(
                {
                    "scene": item["scene"],
                    "method": "pure_macvo",
                    **item["metrics"],
                    "gt_path": item["gt_path"],
                    "estimate_path": item["macvo_path"],
                }
            )
            for fusion in item["fusion"]:
                writer.writerow(
                    {
                        "scene": item["scene"],
                        "method": f"{fusion['source']}_{fusion['config']}",
                        **fusion["metrics"],
                        "gt_path": item["gt_path"],
                        "estimate_path": fusion["path"],
                    }
                )
            for imu_only in item["imu_only"]:
                writer.writerow(
                    {
                        "scene": item["scene"],
                        "method": f"imu_only_staticinit_calibrated_{imu_only['key']}",
                        **imu_only["metrics"],
                        "gt_path": item["gt_path"],
                        "estimate_path": imu_only["path"],
                    }
                )

    html_template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    if args.include_imu_only:
        source_names = ", ".join(label for label, _, _ in fusion_sources)
        method_scope = f"GT, Pure MACVO, {source_names} and calibrated IMU-only"
        line_note = "Fusion line style distinguishes the selected version; calibrated IMU-only uses dashed lines."
    else:
        source_names = ", ".join(label for label, _, _ in fusion_sources)
        method_scope = f"GT, Pure MACVO and {source_names}"
        line_note = "Current fusion uses solid lines; historical fusion uses dashed lines."
    html_template = html_template.replace("__METHOD_SCOPE__", method_scope)
    html_template = html_template.replace("__LINE_NOTE__", line_note)
    html = html_template.replace("__DATA__", json.dumps({"scenes": scenes}, ensure_ascii=False))
    html_path = args.outdir / "interactive_gt_vs_macvo.html"
    html_path.write_text(html, encoding="utf-8")
    print(html_path)
    print(summary_path)


if __name__ == "__main__":
    main()

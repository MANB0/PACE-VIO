#!/usr/bin/env python3
"""Summarize per-frame VIO residual diagnostics for closed-path scenes."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import pandas as pd

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Utility.RunOutputBundle import find_output_bundle


SCENES = [
    "clear_circle_normal_noise",
    "clear_rectangle_normal_noise",
    "clear_circle_zero_noise",
    "clear_rectangle_zero_noise",
]

DIAGNOSTIC_METRICS = [
    "energy_visual_change",
    "energy_imu_change",
    "energy_p_change",
    "energy_v_change",
    "energy_R_change",
    "initial_energy_visual_weighted",
    "initial_energy_imu_weighted",
    "energy_imu_to_visual_ratio",
    "energy_pv_to_visual_ratio",
    "energy_R_to_visual_ratio",
    "energy_visual_weighted",
    "energy_imu_weighted",
    "energy_pv_weighted",
    "energy_R_weighted",
    "r_p_whitened_norm",
    "r_v_whitened_norm",
    "r_R_whitened_norm",
    "imu_vio_whitened_norm",
    "visual_loss_per_residual",
    "imu_vio_weight_diag_max",
    "imu_vio_acc_bias_norm",
    "imu_vio_gyro_bias_norm",
    "influence_imu_to_visual_grad_ratio",
    "influence_imu_to_visual_hessian_ratio",
    "influence_grad_cosine",
    "counterfactual_visual_to_imu_cosine",
    "actual_to_visual_step_cosine",
    "actual_to_imu_step_cosine",
    "actual_to_full_step_cosine",
    "predicted_visual_change_on_actual_step",
    "predicted_imu_change_on_actual_step",
    "init_over_gt_translation_ratio",
    "cos_init_gt_translation",
    "init_rotation_error_angle",
    "imu_rotation_error_angle",
    "rotation_error_angle",
    "init_velocity_error_norm",
    "est_velocity_error_norm",
    "update_pose_translation_norm",
    "update_pose_rotation_norm",
    "update_velocity_norm",
    "update_acc_bias_norm",
    "update_gyro_bias_norm",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    return values[values.notna() & values.map(math.isfinite)]


def classify_factor_victory(rows: pd.DataFrame) -> pd.Series:
    """Classify which factor benefited from the actual nonlinear update."""
    visual = _numeric(rows.get("energy_visual_change", pd.Series(float("nan"), index=rows.index)))
    imu = _numeric(rows.get("energy_imu_change", pd.Series(float("nan"), index=rows.index)))
    outcome = pd.Series("undetermined", index=rows.index, dtype="object")
    finite = visual.map(math.isfinite) & imu.map(math.isfinite)
    eps = 1e-9
    outcome.loc[finite & (imu < -eps) & (visual > eps)] = "imu_wins"
    outcome.loc[finite & (visual < -eps) & (imu > eps)] = "visual_wins"
    outcome.loc[finite & (visual < -eps) & (imu < -eps)] = "compatible"
    outcome.loc[finite & (visual > eps) & (imu > eps)] = "both_worse"
    return outcome


def summarize_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per scene/method diagnostic group."""
    if rows.empty:
        return pd.DataFrame()

    summaries: list[dict[str, object]] = []
    for (scene, method), group in rows.groupby(["scene", "method"], dropna=False):
        factor_outcome = classify_factor_victory(group)
        outcome_counts = factor_outcome.value_counts()
        imu_wins = int(outcome_counts.get("imu_wins", 0))
        visual_wins = int(outcome_counts.get("visual_wins", 0))
        if imu_wins > visual_wins:
            majority = "imu_wins"
        elif visual_wins > imu_wins:
            majority = "visual_wins"
        else:
            majority = "tie"
        row: dict[str, object] = {
            "scene": scene,
            "method": method,
            "pairs": int(len(group)),
            "vio_active_pairs": int(_numeric(group.get("vio_factor_active", pd.Series(dtype=float))).fillna(0).sum()),
            "influence_sampled_pairs": int(
                _numeric(group.get("influence_sampled", pd.Series(dtype=float))).fillna(0).sum()
            ),
            "imu_wins_pairs": imu_wins,
            "visual_wins_pairs": visual_wins,
            "compatible_pairs": int(outcome_counts.get("compatible", 0)),
            "both_worse_pairs": int(outcome_counts.get("both_worse", 0)),
            "undetermined_pairs": int(outcome_counts.get("undetermined", 0)),
            "factor_victory_majority": majority,
        }
        for metric in DIAGNOSTIC_METRICS:
            if metric not in group:
                row[f"{metric}_median"] = float("nan")
                row[f"{metric}_p95"] = float("nan")
                row[f"{metric}_max"] = float("nan")
                row[f"{metric}_final"] = float("nan")
                continue
            values = _finite(group[metric])
            row[f"{metric}_median"] = float(values.median()) if len(values) else float("nan")
            row[f"{metric}_p95"] = float(values.quantile(0.95)) if len(values) else float("nan")
            row[f"{metric}_max"] = float(values.max()) if len(values) else float("nan")
            row[f"{metric}_final"] = float(values.iloc[-1]) if len(values) else float("nan")
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values(["scene", "method"]).reset_index(drop=True)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def augment_energy_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Add cost-scale audit columns, estimating them for legacy diagnostics."""
    df = rows.copy()

    def ensure_energy(norm_col: str, energy_col: str) -> None:
        if energy_col in df:
            df[energy_col] = _numeric(df[energy_col])
        else:
            df[energy_col] = pd.Series(float("nan"), index=df.index)
        if norm_col in df:
            missing = df[energy_col].isna()
            norm = _numeric(df[norm_col])
            df.loc[missing, energy_col] = norm[missing] ** 2

    ensure_energy("r_p_whitened_norm", "energy_p_weighted")
    ensure_energy("r_v_whitened_norm", "energy_v_weighted")
    ensure_energy("r_R_whitened_norm", "energy_R_weighted")
    ensure_energy("imu_vio_whitened_norm", "energy_imu_weighted")

    if "energy_pv_weighted" in df:
        df["energy_pv_weighted"] = _numeric(df["energy_pv_weighted"])
    else:
        df["energy_pv_weighted"] = pd.Series(float("nan"), index=df.index)
    pv_missing = df["energy_pv_weighted"].isna()
    df.loc[pv_missing, "energy_pv_weighted"] = (
        _numeric(df["energy_p_weighted"])[pv_missing].fillna(0.0)
        + _numeric(df["energy_v_weighted"])[pv_missing].fillna(0.0)
    )

    if "energy_imu_diag_weighted" in df:
        df["energy_imu_diag_weighted"] = _numeric(df["energy_imu_diag_weighted"])
    else:
        df["energy_imu_diag_weighted"] = pd.Series(float("nan"), index=df.index)
    diag_missing = df["energy_imu_diag_weighted"].isna()
    df.loc[diag_missing, "energy_imu_diag_weighted"] = (
        _numeric(df["energy_pv_weighted"])[diag_missing].fillna(0.0)
        + _numeric(df["energy_R_weighted"])[diag_missing].fillna(0.0)
    )

    if "energy_visual_weighted" in df:
        df["energy_visual_weighted"] = _numeric(df["energy_visual_weighted"])
    else:
        df["energy_visual_weighted"] = pd.Series(float("nan"), index=df.index)

    for ratio_col in [
        "energy_imu_to_visual_ratio",
        "energy_pv_to_visual_ratio",
        "energy_R_to_visual_ratio",
    ]:
        if ratio_col in df:
            df[ratio_col] = _numeric(df[ratio_col])
        else:
            df[ratio_col] = pd.Series(float("nan"), index=df.index)

    visual = _numeric(df["energy_visual_weighted"])
    valid_visual = visual.notna() & (visual.abs() > 1e-12)
    for ratio_col, numerator_col in [
        ("energy_imu_to_visual_ratio", "energy_imu_weighted"),
        ("energy_pv_to_visual_ratio", "energy_pv_weighted"),
        ("energy_R_to_visual_ratio", "energy_R_weighted"),
    ]:
        missing = df[ratio_col].isna() & valid_visual
        df.loc[missing, ratio_col] = _numeric(df[numerator_col])[missing] / visual[missing]

    return df


def load_diagnostics(result_root: Path) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    rows: list[pd.DataFrame] = []
    skipped: list[tuple[str, str, str]] = []
    manifest_path = result_root / "run_manifest.csv"
    for row in read_manifest(manifest_path):
        scene = row.get("scene", "")
        method = row.get("variant", "")
        result_dir = Path(row.get("result_dir", ""))
        if not result_dir.is_absolute():
            result_dir = WORKDIR / result_dir
        bundle = find_output_bundle(result_dir, require_same_dir_diagnostics=True)
        diag_path = bundle.diagnostics_path
        if not diag_path.exists():
            skipped.append((scene, method, f"missing diagnostics: {diag_path}"))
            continue
        try:
            df = pd.read_csv(diag_path)
        except Exception as exc:
            skipped.append((scene, method, f"cannot read diagnostics: {exc}"))
            continue
        if df.empty:
            skipped.append((scene, method, "empty diagnostics"))
            continue
        df["scene"] = scene or df.get("scene", "")
        df["method"] = method or df.get("method", "")
        df["diagnostics_path"] = str(diag_path)
        rows.append(df)
    if not rows:
        return pd.DataFrame(), skipped
    combined = augment_energy_columns(pd.concat(rows, ignore_index=True, sort=False))
    combined["factor_victory"] = classify_factor_victory(combined)
    return combined, skipped


def top_residual_frames(rows: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if rows.empty or "imu_vio_whitened_norm" not in rows:
        return pd.DataFrame()
    df = rows.copy()
    df["_rank_metric"] = _numeric(df["imu_vio_whitened_norm"])
    keep = [
        "scene",
        "method",
        "pair_id",
        "frame_i",
        "frame_j",
        "timestamp_i",
        "timestamp_j",
        "r_p_whitened_norm",
        "r_v_whitened_norm",
        "r_R_whitened_norm",
        "imu_vio_whitened_norm",
        "energy_imu_to_visual_ratio",
        "energy_pv_to_visual_ratio",
        "energy_R_to_visual_ratio",
        "energy_visual_weighted",
        "energy_imu_weighted",
        "energy_pv_weighted",
        "energy_R_weighted",
        "visual_loss_per_residual",
        "imu_vio_weight_diag_max",
        "imu_vio_acc_bias_norm",
        "imu_vio_gyro_bias_norm",
        "energy_visual_change",
        "energy_imu_change",
        "influence_imu_to_visual_grad_ratio",
        "influence_imu_to_visual_hessian_ratio",
        "actual_to_visual_step_cosine",
        "actual_to_imu_step_cosine",
        "init_rotation_error_angle",
        "imu_rotation_error_angle",
        "rotation_error_angle",
        "init_velocity_error_norm",
        "est_velocity_error_norm",
        "factor_victory",
        "diagnostics_path",
    ]
    return (
        df.sort_values("_rank_metric", ascending=False)
        .head(top_n)
        .loc[:, [c for c in keep if c in df.columns]]
        .reset_index(drop=True)
    )


def _json_safe_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def write_scene_html(scene: str, rows: pd.DataFrame, output_root: Path) -> Path:
    scene_dir = output_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in DIAGNOSTIC_METRICS if m in rows.columns]
    data: dict[str, dict[str, list[float | None]]] = {}
    for method, group in rows.groupby("method", dropna=False):
        ordered = group.sort_values("pair_id" if "pair_id" in group.columns else "frame_j")
        data[str(method)] = {
            "x": [_json_safe_number(v) for v in ordered.get("frame_j", pd.Series(range(len(ordered))))],
        }
        for metric in metrics:
            data[str(method)][metric] = [_json_safe_number(v) for v in ordered[metric]]

    summary = summarize_diagnostics(rows)
    top = top_residual_frames(rows, top_n=20)
    top.to_csv(scene_dir / "top_residual_frames.csv", index=False)

    metric_checks = "\n".join(
        f'<label><input type="checkbox" class="metric" value="{html.escape(metric)}" '
        f'{"checked" if metric in {"energy_visual_change", "energy_imu_change", "actual_to_visual_step_cosine", "actual_to_imu_step_cosine"} else ""}> '
        f'{html.escape(metric)}</label>'
        for metric in metrics
    )
    summary_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(col, '')))}</td>"
            for col in [
                "method",
                "pairs",
                "vio_active_pairs",
                "influence_sampled_pairs",
                "imu_wins_pairs",
                "visual_wins_pairs",
                "compatible_pairs",
                "both_worse_pairs",
                "undetermined_pairs",
                "factor_victory_majority",
                "imu_vio_whitened_norm_median",
                "imu_vio_whitened_norm_p95",
                "imu_vio_whitened_norm_max",
                "energy_imu_to_visual_ratio_median",
                "energy_pv_to_visual_ratio_median",
                "energy_R_to_visual_ratio_median",
                "r_p_whitened_norm_max",
                "r_v_whitened_norm_max",
                "r_R_whitened_norm_max",
            ]
        )
        + "</tr>"
        for _, row in summary.iterrows()
    )
    top_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(col, '')))}</td>"
            for col in [
                "method",
                "pair_id",
                "frame_j",
                "imu_vio_whitened_norm",
                "energy_imu_to_visual_ratio",
                "energy_pv_to_visual_ratio",
                "energy_R_to_visual_ratio",
                "r_p_whitened_norm",
                "r_v_whitened_norm",
                "r_R_whitened_norm",
                "visual_loss_per_residual",
            ]
        )
        + "</tr>"
        for _, row in top.iterrows()
    )

    page = scene_dir / "diagnostics_interactive.html"
    page.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(scene)} VIO diagnostics</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#f5f7fa;color:#1f2933}}
main{{max-width:1200px;margin:auto;background:white;border:1px solid #d8dee6;padding:16px}}
canvas{{width:100%;height:520px;border:1px solid #d8dee6;background:#fff}}
.controls{{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0 14px}}
label{{white-space:nowrap}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:12px}}
th,td{{border:1px solid #d8dee6;padding:5px 7px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#eef2f7}}
</style>
</head>
<body><main>
<h1>{html.escape(scene)} VIO residual diagnostics</h1>
<h2>Factor-victory audit</h2>
<p>The primary decision uses the signs of the actual nonlinear visual and IMU energy changes. The chart uses signed log10(1 + abs(value)) so opposing update directions remain visible.</p>
<div class="controls">{metric_checks}</div>
<canvas id="chart" width="1180" height="520"></canvas>
<h2>Summary</h2>
<table><thead><tr>
<th>method</th><th>pairs</th><th>active</th><th>sampled</th><th>IMU wins</th><th>visual wins</th><th>compatible</th><th>both worse</th><th>unknown</th><th>majority</th><th>vio median</th><th>vio p95</th><th>vio max</th><th>IMU/vis med</th><th>pv/vis med</th><th>R/vis med</th><th>p max</th><th>v max</th><th>R max</th>
</tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Top residual frames</h2>
<table><thead><tr>
<th>method</th><th>pair</th><th>frame_j</th><th>vio</th><th>IMU/vis</th><th>pv/vis</th><th>R/vis</th><th>p</th><th>v</th><th>R</th><th>visual</th>
</tr></thead><tbody>{top_rows}</tbody></table>
<script>
const DATA = {json.dumps(data, ensure_ascii=False)};
const METRICS = {json.dumps(metrics)};
const COLORS = ["#2563eb","#dc2626","#16a34a","#9333ea","#ea580c","#0891b2","#4f46e5","#be123c"];
function checkedMetrics() {{
  return Array.from(document.querySelectorAll(".metric")).filter(x => x.checked).map(x => x.value);
}}
function logv(v) {{
  if (v === null || !Number.isFinite(v)) return null;
  return Math.sign(v) * Math.log10(1 + Math.abs(v));
}}
function draw() {{
  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const pad = {{l:58,r:18,t:20,b:42}};
  const metrics = checkedMetrics();
  let pts = [];
  for (const [method, series] of Object.entries(DATA)) {{
    for (const metric of metrics) {{
      const xs = series.x || [];
      const ys = series[metric] || [];
      for (let i = 0; i < Math.min(xs.length, ys.length); i++) {{
        const y = logv(ys[i]);
        if (xs[i] !== null && y !== null) pts.push([xs[i], y]);
      }}
    }}
  }}
  if (!pts.length) return;
  const minX = Math.min(...pts.map(p => p[0]));
  const maxX = Math.max(...pts.map(p => p[0]));
  const minY = Math.min(...pts.map(p => p[1]));
  const maxY = Math.max(...pts.map(p => p[1]));
  const sx = x => pad.l + (x - minX) / Math.max(maxX - minX, 1e-9) * (canvas.width - pad.l - pad.r);
  const sy = y => canvas.height - pad.b - (y - minY) / Math.max(maxY - minY, 1e-9) * (canvas.height - pad.t - pad.b);
  ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, canvas.height-pad.b); ctx.lineTo(canvas.width-pad.r, canvas.height-pad.b); ctx.stroke();
  ctx.fillStyle = "#334155"; ctx.font = "12px Arial";
  ctx.fillText("frame", canvas.width/2, canvas.height-10);
  ctx.save(); ctx.translate(14, canvas.height/2); ctx.rotate(-Math.PI/2); ctx.fillText("signed log10(1 + abs(value))", 0, 0); ctx.restore();
  let colorIdx = 0;
  const legend = [];
  for (const [method, series] of Object.entries(DATA)) {{
    for (const metric of metrics) {{
      const color = COLORS[colorIdx++ % COLORS.length];
      legend.push([color, method + " / " + metric]);
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
      let open = false;
      const xs = series.x || [], ys = series[metric] || [];
      for (let i = 0; i < Math.min(xs.length, ys.length); i++) {{
        const y = logv(ys[i]);
        if (xs[i] === null || y === null) {{ open = false; continue; }}
        const px = sx(xs[i]), py = sy(y);
        if (!open) {{ ctx.moveTo(px, py); open = true; }} else {{ ctx.lineTo(px, py); }}
      }}
      ctx.stroke();
    }}
  }}
  let ly = 20;
  for (const [color, label] of legend.slice(0, 18)) {{
    ctx.fillStyle = color; ctx.fillRect(canvas.width - 360, ly - 9, 18, 3);
    ctx.fillStyle = "#334155"; ctx.fillText(label, canvas.width - 336, ly - 5);
    ly += 16;
  }}
}}
document.querySelectorAll(".metric").forEach(x => x.addEventListener("change", draw));
draw();
</script>
</main></body></html>
""",
        encoding="utf-8",
    )
    return page


def write_index(output_root: Path, summary: pd.DataFrame, scene_pages: dict[str, Path], skipped: list[tuple[str, str, str]]) -> Path:
    rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(col, '')))}</td>"
            for col in [
                "scene",
                "method",
                "pairs",
                "vio_active_pairs",
                "imu_vio_whitened_norm_median",
                "imu_vio_whitened_norm_p95",
                "imu_vio_whitened_norm_max",
                "energy_imu_to_visual_ratio_median",
                "energy_pv_to_visual_ratio_median",
                "energy_R_to_visual_ratio_median",
                "r_p_whitened_norm_max",
                "r_v_whitened_norm_max",
                "r_R_whitened_norm_max",
            ]
        )
        + "</tr>"
        for _, row in summary.iterrows()
    )
    links = "\n".join(
        f'<li><a href="{html.escape(str(page.relative_to(output_root)))}">{html.escape(scene)}</a></li>'
        for scene, page in sorted(scene_pages.items())
    )
    skipped_html = ""
    if skipped:
        skipped_html = "<h2>Skipped</h2><ul>" + "".join(
            f"<li>{html.escape(scene)} / {html.escape(method)}: {html.escape(reason)}</li>"
            for scene, method, reason in skipped
        ) + "</ul>"
    index = output_root / "index.html"
    index.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Closed-path VIO residual diagnostics</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#f5f7fa;color:#1f2933}}
main{{max-width:1200px;margin:auto;background:white;border:1px solid #d8dee6;padding:16px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:12px}}
th,td{{border:1px solid #d8dee6;padding:5px 7px;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{background:#eef2f7}}
</style></head><body><main>
<h1>Closed-path VIO residual diagnostics</h1>
<h2>Scene pages</h2><ul>{links}</ul>
<h2>Summary</h2>
<table><thead><tr>
<th>scene</th><th>method</th><th>pairs</th><th>active</th><th>vio median</th><th>vio p95</th><th>vio max</th><th>IMU/vis med</th><th>pv/vis med</th><th>R/vis med</th><th>p max</th><th>v max</th><th>R max</th>
</tr></thead><tbody>{rows}</tbody></table>
{skipped_html}
</main></body></html>
""",
        encoding="utf-8",
    )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, skipped = load_diagnostics(args.result_root)
    if rows.empty:
        print("No diagnostics loaded.")
        for scene, method, reason in skipped:
            print(f"  - {scene} / {method}: {reason}")
        return 1

    rows = rows[rows["scene"].isin(args.scenes)].copy()
    rows.to_csv(args.output_root / "diagnostics_all_pairs.csv", index=False)
    summary = summarize_diagnostics(rows)
    summary.to_csv(args.output_root / "diagnostics_summary.csv", index=False)
    top = top_residual_frames(rows, top_n=100)
    top.to_csv(args.output_root / "top_residual_frames.csv", index=False)

    scene_pages: dict[str, Path] = {}
    for scene in args.scenes:
        scene_rows = rows[rows["scene"] == scene].copy()
        if scene_rows.empty:
            continue
        scene_pages[scene] = write_scene_html(scene, scene_rows, args.output_root)
    index = write_index(args.output_root, summary, scene_pages, skipped)

    print(f"Wrote all pairs: {args.output_root / 'diagnostics_all_pairs.csv'}")
    print(f"Wrote summary:   {args.output_root / 'diagnostics_summary.csv'}")
    print(f"Wrote top frames:{args.output_root / 'top_residual_frames.csv'}")
    print(f"Wrote index:     {index}")
    for scene, page in sorted(scene_pages.items()):
        print(f"  {scene}: {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

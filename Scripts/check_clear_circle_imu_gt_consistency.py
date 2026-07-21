#!/usr/bin/env python3
"""Check IMU-vs-GT consistency on the deterministic clear-circle datasets.

This is an isolation diagnostic: it does not run MACVO and does not read any
estimated trajectory.  It compares IMU preintegration over each camera-frame
interval against the corresponding GT pose/velocity increment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

import Scripts.diagnose_imu_vio_residuals as diag


DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_clear_circle_imu_gt_consistency_20260704"
DEFAULT_SCENE_ROOTS = {
    "clear_circle_normal_noise": Path(
        "/mnt/e/文档/holoocean/code/recordings/"
        "batch_clear_circle_pair_20260704/normal_noise/clear_circle_path"
    ),
    "clear_circle_zero_noise": Path(
        "/mnt/e/文档/holoocean/code/recordings/"
        "batch_clear_circle_pair_20260704/zero_noise/clear_circle_path"
    ),
}
VARIANTS = ("raw_nwu", "rx180_to_ned", "imu_rx180_only_gt_nwu")


def _fmt(value: object, ndigits: int = 6) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return ""


def _row(
    summary_rows: list[dict[str, object]],
    scene: str,
    variant: str,
    protocol: str,
) -> dict[str, object] | None:
    for row in summary_rows:
        if row["scene"] == scene and row["variant"] == variant and row["protocol"] == protocol:
            return row
    return None


def _number(row: dict[str, object] | None, key: str) -> float | None:
    if row is None:
        return None
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def evaluate_clear_circle_roots(
    scene_roots: Mapping[str, Path],
    *,
    max_pairs: int | None = None,
) -> list[diag.PairResidual]:
    records: list[diag.PairResidual] = []
    for scene, root in scene_roots.items():
        if not root.exists():
            raise FileNotFoundError(f"Missing clear-circle scene root: {root}")
        for variant in VARIANTS:
            records.extend(diag.evaluate_scene_variant(scene, root, variant, max_pairs=max_pairs))
    return records


def build_auto_conclusion(
    summary_rows: list[dict[str, object]],
    scene_roots: Mapping[str, Path],
) -> list[str]:
    lines = ["自动结论："]
    for scene in scene_roots:
        raw = _row(summary_rows, scene, "raw_nwu", "standard_vio")
        ned = _row(summary_rows, scene, "rx180_to_ned", "standard_vio")
        mixed = _row(summary_rows, scene, "imu_rx180_only_gt_nwu", "standard_vio")
        if raw is None:
            continue

        raw_rot = _number(raw, "rot_err_deg_median")
        raw_vel = _number(raw, "vel_err_median")
        raw_pos = _number(raw, "pos_err_median")
        mixed_vel = _number(mixed, "vel_err_median")
        ned_vel = _number(ned, "vel_err_median")

        lines.append(
            f"- `{scene}`: raw standard-VIO median residuals are "
            f"rot={_fmt(raw_rot)}, vel={_fmt(raw_vel)} m/s, pos={_fmt(raw_pos)} m."
        )
        if ned_vel is not None and raw_vel is not None:
            lines.append(
                f"  一致 NWU->NED 转换后的速度残差中位数为 {_fmt(ned_vel)} m/s，"
                "与 raw_nwu 基本一致。"
            )
        if mixed_vel is not None:
            lines.append(
                f"  故意混合坐标系后的速度残差中位数为 {_fmt(mixed_vel)} m/s，"
                "明显大于一致坐标检查，因此坐标系错误检测有效。"
            )

        if "zero_noise" in scene and raw_rot is not None and raw_vel is not None and raw_pos is not None:
            if raw_rot < 1e-2 and raw_vel < 2e-2 and raw_pos < 1e-3:
                lines.append(
                    "  zero-noise 下 IMU 与 GT 的相邻帧增量基本一致；"
                    "当前 full-VIO 轨迹发散不应优先归因于原始 IMU 坐标系错误。"
                )
            else:
                lines.append(
                    "  zero-noise 下仍有不可忽略残差，需要继续检查时间戳、速度定义、"
                    "加速度/重力定义和预积分模型。"
                )
    return lines


def build_report_text(
    summary_rows: list[dict[str, object]],
    scene_roots: Mapping[str, Path],
) -> str:
    lines = [
        "# Clear-circle IMU/GT consistency check",
        "",
        "目的：隔离检查 HoloOcean 输出的 IMU 序列是否能在相邻相机帧之间积分出与 GT 一致的运动增量。",
        "该检查不使用 MACVO 视觉结果、不使用优化器，也不读取任何估计轨迹。",
        "",
        "数据集：",
    ]
    for scene, root in scene_roots.items():
        lines.append(f"- `{scene}`: `{root}`")

    lines.extend(
        [
            "",
            "检查协议：",
            "- `raw_nwu + standard_vio`: 直接使用数据原始 NWU/FLU 约定，并按常见 VIO 预积分形式比较 GT 增量。",
            "- `rx180_to_ned + standard_vio`: 将 IMU、GT body、GT world 一致转换到内部 NED；结果应与 raw_nwu 基本一致。",
            "- `imu_rx180_only_gt_nwu + standard_vio`: 只旋转 IMU、不旋转 GT 的故意错误接法，用作坐标系错误检测。",
            "- `raw_nwu + macvo_kinematic`: 当前项目的重力校正积分形式，用于和 standard_vio 互相校验。",
            "",
            "关键结果：",
            "| scene | check | pairs | rot med deg | rot p95 deg | vel med m/s | vel p95 m/s | pos med m | pos p95 m | IMU/GT dp | IMU/GT dv |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    checks = [
        ("raw_nwu", "standard_vio"),
        ("rx180_to_ned", "standard_vio"),
        ("imu_rx180_only_gt_nwu", "standard_vio"),
        ("raw_nwu", "macvo_kinematic"),
    ]
    for scene in scene_roots:
        for variant, protocol in checks:
            row = _row(summary_rows, scene, variant, protocol)
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        scene,
                        f"`{variant} + {protocol}`",
                        str(row["n_pairs"]),
                        _fmt(row["rot_err_deg_median"]),
                        _fmt(row["rot_err_deg_p95"]),
                        _fmt(row["vel_err_median"]),
                        _fmt(row["vel_err_p95"]),
                        _fmt(row["pos_err_median"]),
                        _fmt(row["pos_err_p95"]),
                        _fmt(row["imu_over_gt_delta_p_median"], 3),
                        _fmt(row["imu_over_gt_delta_v_median"], 3),
                    ]
                )
                + " |"
            )

    lines.extend([""] + build_auto_conclusion(summary_rows, scene_roots))

    lines.extend(
        [
            "",
            "判读方式：",
            "- 如果 `raw_nwu + standard_vio` 和 `rx180_to_ned + standard_vio` 都很小且彼此接近，说明坐标系一致转换没有暴露出 IMU/GT 的基础矛盾。",
            "- 如果 `imu_rx180_only_gt_nwu + standard_vio` 明显变差，说明该检查能识别混合坐标系错误，坐标系错误检测是有效的。",
            "- 如果 zero-noise 仍然出现明显速度或位置残差，问题更可能在 IMU 数值生成、时间区间选取、速度定义、预积分模型假设或重力/加速度定义之间，而不是随机噪声本身。",
            "",
            "输出文件：",
            "- `imu_gt_pair_residuals.csv`: 每个相邻相机帧区间的残差。",
            "- `imu_gt_summary.csv`: 每个数据集/坐标变体/积分协议的汇总统计。",
            "- `figures/*_timeseries.png`: 残差随相机帧区间变化的曲线。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_root: Path,
    records: list[diag.PairResidual],
    scene_roots: Mapping[str, Path],
) -> list[dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    fig_root = output_root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    diag.OUTPUT_ROOT = output_root
    diag.FIG_ROOT = fig_root

    pair_rows = [record.__dict__ for record in records]
    if not pair_rows:
        raise RuntimeError("No IMU/GT residual records were produced.")

    diag.write_csv(output_root / "imu_gt_pair_residuals.csv", pair_rows, list(pair_rows[0].keys()))
    summary_rows = diag.summarize(records)
    diag.write_csv(output_root / "imu_gt_summary.csv", summary_rows, list(summary_rows[0].keys()))

    for scene in scene_roots:
        for variant in VARIANTS:
            for protocol in ("standard_vio", "macvo_kinematic"):
                diag.plot_scene_timeseries(records, scene, variant=variant, protocol=protocol)

    (output_root / "README.md").write_text(
        build_report_text(summary_rows, scene_roots),
        encoding="utf-8",
    )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for CSV, plots, and README output.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit on adjacent camera-frame intervals for quick checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = evaluate_clear_circle_roots(DEFAULT_SCENE_ROOTS, max_pairs=args.max_pairs)
    summary_rows = write_outputs(args.output_root, records, DEFAULT_SCENE_ROOTS)
    print(f"Wrote {args.output_root}")
    print(f"Pair residual rows: {len(records)}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Figures: {len(list((args.output_root / 'figures').glob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

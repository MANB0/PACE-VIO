#!/usr/bin/env python3
"""Plot the completed full-circle translation-oracle VIO runs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.plot_static63_gt_macvo import (
    HTML_TEMPLATE,
    metrics,
    read_forward_axes,
    read_xyz,
)


SCENE = "clear_circle_truth_normal_noise"
DATASET = (
    Path("/mnt/e")
    / "\u6587\u6863"
    / "holoocean/code/recordings/batch_clear_truth_paths_20260713_static63_variants"
    / SCENE
)
MACVO = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713"
    / "trial_1/pure_macvo"
    / SCENE
    / "poses.csv"
)
OUT = WORKDIR / "analysis_circle_translation_oracle_20260716"
FULL = OUT / "oracles/full"
MODES = {
    "V0": ("MAC R + MAC t / optimize ba,bg", "optimize_ba_bg", "#2563eb", ""),
    "V1": ("MAC R + GT t / optimize ba,bg", "optimize_ba_bg", "#0891b2", ""),
    "V2": ("GT R + MAC t / optimize ba,bg", "optimize_ba_bg", "#7c3aed", ""),
    "V3": ("GT R + GT t / optimize ba,bg", "optimize_ba_bg", "#059669", ""),
    "O3": ("MAC R + MAC t / fixed ba", "fixed_ba", "#db2777", "8 5"),
    "O4": ("MAC R + GT t / fixed ba", "fixed_ba", "#ca8a04", "8 5"),
}


def xyz(rows: list[tuple[int, float, float, float]]) -> list[list[float]]:
    return [[x, y, z] for _, x, y, z in rows]


def errors(
    gt: list[tuple[int, float, float, float]],
    estimate: list[tuple[int, float, float, float]],
) -> list[float]:
    return [
        math.sqrt(
            (est[1] - ref[1]) ** 2
            + (est[2] - ref[2]) ** 2
            + (est[3] - ref[3]) ** 2
        )
        for ref, est in zip(gt, estimate)
    ]


def assert_aligned(
    name: str,
    reference: list[tuple[int, float, float, float]],
    estimate: list[tuple[int, float, float, float]],
) -> None:
    if len(reference) != len(estimate):
        raise ValueError(f"{name}: row count mismatch {len(estimate)} != {len(reference)}")
    ref_time = [row[0] for row in reference]
    est_time = [row[0] for row in estimate]
    if ref_time != est_time:
        raise ValueError(f"{name}: timestamps do not match GT")


def main() -> None:
    gt_path = DATASET / "ref_pose.csv"
    gt = read_xyz(gt_path)
    pure_macvo = read_xyz(MACVO)
    assert_aligned("Pure MACVO", gt, pure_macvo)
    gt_forward = read_forward_axes(gt_path)
    macvo_forward = read_forward_axes(MACVO)

    fusion = []
    for mode, (label, config, color, dasharray) in MODES.items():
        summary = json.loads((FULL / mode / "summary.json").read_text(encoding="utf-8"))
        pose_path = Path(summary["artifacts"]["poses"])
        rows = read_xyz(pose_path)
        assert_aligned(mode, gt, rows)
        forward = read_forward_axes(pose_path)
        if len(forward) != len(gt):
            raise ValueError(f"{mode}: forward-axis count mismatch")
        fusion.append(
            {
                "key": mode,
                "source": "full_circle_translation_oracle",
                "config": config,
                "label": f"{mode} / {label}",
                "color": color,
                "dasharray": dasharray,
                "scene": SCENE,
                "xyz": xyz(rows),
                "forward": forward,
                "error_m": errors(gt, rows),
                "metrics": metrics(gt, rows),
                "path": str(pose_path),
            }
        )

    payload = {
        "scene": "Circle / Normal noise / Full translation-oracle VIO",
        "gt": xyz(gt),
        "gt_forward": gt_forward,
        "macvo": xyz(pure_macvo),
        "macvo_forward": macvo_forward,
        "time_s": [(row[0] - gt[0][0]) / 1e9 for row in gt],
        "error_m": errors(gt, pure_macvo),
        "metrics": metrics(gt, pure_macvo),
        "fusion": fusion,
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(MACVO),
    }

    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Full-circle translation-oracle VIO comparison",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO and six completed full-circle VIO oracle runs",
    )
    template = template.replace(
        "__LINE_NOTE__",
        "Timestamp-matched NWU trajectories with no alignment, fitting or scale correction. GT substitutions are diagnostic only.",
    )
    html = template.replace(
        "__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False)
    )
    output = OUT / "interactive_full_circle_translation_oracles.html"
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

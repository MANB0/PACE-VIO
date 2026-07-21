#!/usr/bin/env python3
"""Gate 4: run the frozen N=2 slice with only P_imu changed."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts import freeze_normal_noise_sampling_baseline as baseline  # noqa: E402


OUT = baseline.DEFAULT_OUTPUT
RUN_OUT = OUT / "sampling_aware_n2"


def nested_ratio(current: dict, sampling: dict) -> dict:
    result = {}
    for name, current_value in current.items():
        sampling_value = sampling.get(name)
        if isinstance(current_value, (int, float)) and isinstance(sampling_value, (int, float)):
            result[name] = {
                "current": current_value,
                "sampling_aware": sampling_value,
                "relative_change_percent": (
                    (sampling_value / current_value - 1.0) * 100.0
                    if current_value != 0.0
                    else None
                ),
            }
    return result


def main() -> int:
    baseline_summary_path = OUT / "baseline_normal_noise_summary.json"
    if not baseline_summary_path.exists():
        raise FileNotFoundError("Gate 0 baseline must be frozen before Gate 4")
    if RUN_OUT.exists():
        resolved = RUN_OUT.resolve()
        if OUT.resolve() not in resolved.parents:
            raise RuntimeError("refusing to clear sampling run outside the audit directory")
        shutil.rmtree(RUN_OUT)
    RUN_OUT.mkdir(parents=True)

    config = yaml.safe_load(baseline.ODOM_SOURCE.read_text(encoding="utf-8"))
    config["Odometry"]["args"]["two_state_covariance_mode"] = "sampling_aware"
    sampling_source = RUN_OUT / "sampling_aware_odometry_source.yaml"
    sampling_source.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    replay = baseline.replay
    replay.OUT = RUN_OUT
    replay.FRAME_LIMIT = baseline.FRAME_LIMIT
    replay.ITERATION_LIMIT = baseline.ITERATION_LIMIT
    replay.SOURCE_CACHE = baseline.SOURCE_CACHE
    replay.PREFIX_CACHE = RUN_OUT / "sampling_cache_rectangle_300"
    replay.SOURCE_RESULT = baseline.SOURCE_RESULT
    replay.ODOM_SOURCE = sampling_source
    replay.GT_PATH = baseline.DATASET_DIR / "ref_pose.csv"
    replay.RUN_RESULT_ROOT = RUN_OUT / "sampling_run_result"
    replay.TRACE_PATH = RUN_OUT / "baseline_factor_per_edge.csv"
    replay.ODOM_CONFIG = RUN_OUT / "sampling_odometry.yaml"
    replay.DATA_CONFIG = RUN_OUT / "sampling_data.yaml"
    replay.TRACE_FIELDS = baseline.TRACE_FIELDS
    replay.audited_solve = baseline.audited_solve
    replay.summarize_run = lambda: baseline.summarize_baseline(RUN_OUT)
    replay.GATE_BY_EDGE.clear()
    status = replay.main()

    sampling_summary_path = RUN_OUT / "baseline_normal_noise_summary.json"
    sampling_summary = json.loads(sampling_summary_path.read_text(encoding="utf-8"))
    sampling_summary["gate"] = 4
    sampling_summary["covariance_mode"] = "sampling_aware"
    sampling_summary["production_code_modified"] = True
    (RUN_OUT / "sampling_n2_summary.json").write_text(
        json.dumps(sampling_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for source_name, target_name in (
        ("baseline_state_per_frame.csv", "sampling_n2_state_per_frame.csv"),
        ("baseline_factor_per_edge.csv", "sampling_n2_factor_per_edge.csv"),
        ("baseline_trajectory.csv", "sampling_n2_trajectory.csv"),
    ):
        shutil.copy2(RUN_OUT / source_name, RUN_OUT / target_name)

    current_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    comparison = {
        "gate": 4,
        "scene": baseline.SCENE,
        "frame_range": [0, baseline.FRAME_LIMIT - 1],
        "edge_count": baseline.EXPECTED_EDGES,
        "S0": "current_independent_step",
        "S1": "sampling_aware",
        "controlled_variables": {
            "delta_R_delta_v_delta_p": "unchanged",
            "continuous_sigmas": current_summary["continuous_sigmas"],
            "visual_measurement_and_covariance": "unchanged",
            "bias_random_walk": "unchanged",
            "window_states": 2,
            "LM_Huber_prior_initialization": "unchanged",
        },
        "truth_metrics": nested_ratio(
            current_summary["truth_metrics_valid_frames"],
            sampling_summary["truth_metrics_valid_frames"],
        ),
        "high_frequency_metrics": nested_ratio(
            current_summary["high_frequency_metrics_valid_frames"],
            sampling_summary["high_frequency_metrics_valid_frames"],
        ),
        "solver": {
            "current": current_summary["solver"],
            "sampling_aware": sampling_summary["solver"],
        },
        "factor_cost_statistics": {
            "current": current_summary["factor_cost_statistics"],
            "sampling_aware": sampling_summary["factor_cost_statistics"],
        },
        "input_hashes": {
            "current": current_summary["input_hashes"],
            "sampling_aware": sampling_summary["input_hashes"],
        },
    }
    (OUT / "sampling_covariance_n2_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

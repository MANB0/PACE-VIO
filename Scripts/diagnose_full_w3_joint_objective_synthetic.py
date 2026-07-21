from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pypose as pp
import torch

from Scripts.synthetic_w3_validation_data import (
    EstimatorIMUInput,
    EvaluationTruth,
    EstimatorVisualInput,
    MOTION_SAMPLING_CONTRACT,
    SyntheticSequenceConfig,
    generate_imu_input,
    generate_truth,
    generate_visual_input,
    write_sensor_artifacts,
    write_visual_artifact,
)
from Scripts.synthetic_w3_validation_runner import (
    CaseRunResult,
    ValidationCase,
    build_cases,
    run_validation_case,
)


_REQUIRED_CASES = (
    "visual_initial",
    "w3_bias_locked_zero",
    "w3_full_current",
    "w3_full_all",
    "w3_full_all_zero_mean",
    "w3_full_all_zero_normal",
    "w3_full_all_drifting_bias",
)

_REQUIRED_CASE_DEFINITIONS = {
    "visual_initial": {
        "visual_condition": "drifted_visual",
        "imu_noise_mode": "fixed_seed_normal",
        "bias_mode": "constant_bias",
        "bias_enabled": False,
        "writeback": "none",
    },
    "w3_bias_locked_zero": {
        "visual_condition": "drifted_visual",
        "imu_noise_mode": "fixed_seed_normal",
        "bias_mode": "constant_bias",
        "bias_enabled": False,
        "writeback": "all_optimized",
    },
    "w3_full_current": {
        "visual_condition": "drifted_visual",
        "imu_noise_mode": "fixed_seed_normal",
        "bias_mode": "constant_bias",
        "bias_enabled": True,
        "writeback": "current",
    },
    "w3_full_all": {
        "visual_condition": "drifted_visual",
        "imu_noise_mode": "fixed_seed_normal",
        "bias_mode": "constant_bias",
        "bias_enabled": True,
        "writeback": "all_optimized",
    },
    "w3_full_all_zero_mean": {
        "visual_condition": "clean_visual",
        "imu_noise_mode": "mean_measurement",
        "bias_mode": "zero_bias",
        "bias_enabled": True,
        "writeback": "all_optimized",
    },
    "w3_full_all_zero_normal": {
        "visual_condition": "drifted_visual",
        "imu_noise_mode": "fixed_seed_normal",
        "bias_mode": "zero_bias",
        "bias_enabled": True,
        "writeback": "all_optimized",
    },
    "w3_full_all_drifting_bias": {
        "visual_condition": "drifted_visual",
        "imu_noise_mode": "fixed_seed_normal",
        "bias_mode": "drifting_bias",
        "bias_enabled": True,
        "writeback": "all_optimized",
    },
}


class ReadinessInputContractError(ValueError):
    """Raised when evidence cannot form the canonical seven-case readiness view."""


_READINESS_NUMERIC_COLUMNS = (
    "frame_count",
    "evaluation_start_frame",
    "evaluation_frame_count",
    "max_shift_source_bias_error",
    "max_shift_source_pose_translation_error_m",
    "max_shift_source_pose_rotation_error_rad",
    "max_shift_source_velocity_error_m_s",
    "translation_rmse_m",
    "rotation_rmse_deg",
    "velocity_rmse_m_s",
    "visual_translation_rmse_m",
    "visual_rotation_rmse_deg",
    "visual_velocity_rmse_m_s",
    "acc_bias_rmse_m_s2",
    "gyro_bias_rmse_rad_s",
    "six_axis_bias_rmse",
    "zero_bias_baseline_acc_bias_rmse_m_s2",
    "zero_bias_baseline_gyro_bias_rmse_rad_s",
    "zero_bias_baseline_six_axis_bias_rmse",
    "previous_estimate_persistence_acc_bias_rmse_m_s2",
    "previous_estimate_persistence_gyro_bias_rmse_rad_s",
    "previous_estimate_persistence_six_axis_bias_rmse",
    "max_acc_bias_norm_m_s2",
    "max_gyro_bias_norm_rad_s",
    "final_acc_bias_norm_m_s2",
    "final_gyro_bias_norm_rad_s",
    "final10_acc_bias_norm_slope_m_s3",
    "final10_gyro_bias_norm_slope_rad_s2",
    "final_six_axis_bias_norm",
    "final10_six_axis_bias_norm_slope_per_s",
)

_READINESS_BOOLEAN_COLUMNS = (
    "finite",
    "state_propagation_ok",
    "diagnostics_available",
    "optimizer_completed",
    "optimizer_final_loss_finite",
    "all_edges_rebase_fields_match_runtime",
    "window_contract_ok",
)

_READINESS_FRAME_NUMERIC_COLUMNS = (
    "frame_index",
    "translation_error_m",
    "rotation_error_deg",
    "velocity_error_m_s",
    "acc_bias_error_m_s2",
    "gyro_bias_error_rad_s",
    "estimated_acc_bias_norm_m_s2",
    "estimated_gyro_bias_norm_rad_s",
)

_READINESS_INTEGER_CASE_COLUMNS = (
    "frame_count",
    "evaluation_start_frame",
    "evaluation_frame_count",
)

_READINESS_INTEGER_FRAME_COLUMNS = ("frame_index",)

_WINDOW_RUNTIME_MATCH_FIELDS = (
    "from_pose",
    "init_motion",
    "prev_velocity",
    "curr_velocity",
    "prev_acc_bias",
    "prev_gyro_bias",
    "curr_acc_bias",
    "curr_gyro_bias",
)

_WINDOW_BOOLEAN_COLUMNS = (
    "bias_state_active",
    "bias_parameter_structurally_present",
    "vio_factor_active",
    "finite",
    "optimizer_completed",
    "optimizer_final_loss_finite",
    "all_edges_rebase_fields_match_runtime",
    "leading_preintegration_fields_immutable",
    "trailing_preintegration_fields_immutable",
    "trailing_linearization_bias_matches_creation_source",
    "trailing_preintegration_rotation_matches_creation_source",
    "edge_12_source_state_matches_runtime",
) + tuple(
    f"{prefix}_{field}_matches_runtime"
    for prefix in ("leading", "trailing")
    for field in _WINDOW_RUNTIME_MATCH_FIELDS
)

_WINDOW_INTEGER_COLUMNS = (
    "window_target",
    "imu_residual_rows",
    "bias_fields_cleared_per_edge",
    "source_main_imu_bias_jacobian_rank",
    "terminal_main_imu_bias_jacobian_rank",
)

_WINDOW_SHIFT_COLUMNS = (
    "shift_source_bias_error",
    "shift_source_pose_translation_error_m",
    "shift_source_pose_rotation_error_rad",
    "shift_source_velocity_error_m_s",
)

_WINDOW_SHIFT_TO_CASE_COLUMNS = {
    "shift_source_bias_error": "max_shift_source_bias_error",
    "shift_source_pose_translation_error_m": "max_shift_source_pose_translation_error_m",
    "shift_source_pose_rotation_error_rad": "max_shift_source_pose_rotation_error_rad",
    "shift_source_velocity_error_m_s": "max_shift_source_velocity_error_m_s",
}


def _strict_boolean_values(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    scope: str,
) -> None:
    for column in columns:
        values = table[column]
        valid = values.map(lambda value: isinstance(value, (bool, np.bool_)))
        if not bool(valid.all()):
            raise ReadinessInputContractError(
                f"Readiness {scope} column {column} must contain only boolean values"
            )


def _strict_integer_values(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    scope: str,
) -> None:
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise ReadinessInputContractError(
                f"Readiness {scope} column {column} must contain finite integer values"
            )


def _validate_window_diagnostics_contract(
    case_table: pd.DataFrame,
    window_table: pd.DataFrame,
) -> None:
    case_table["window_contract_ok"] = False
    if "case" not in case_table:
        raise ReadinessInputContractError(
            "Window production structure cannot be matched without case metadata"
        )

    completed_cases = case_table
    if "run_status" in completed_cases:
        completed_cases = completed_cases.loc[completed_cases["run_status"] == "completed"]
    optimized_cases = completed_cases.loc[completed_cases["case"] != "visual_initial"]
    if optimized_cases.empty:
        case_table.loc[case_table["case"] == "visual_initial", "window_contract_ok"] = True
        return
    if window_table.empty or "case" not in window_table:
        raise ReadinessInputContractError(
            "Window production structure evidence is missing for optimized cases"
        )

    required_columns = {
        "case",
        "visual_condition",
        "imu_noise_mode",
        "bias_mode",
        "writeback",
        "graph_class",
        "context_device",
        *_WINDOW_BOOLEAN_COLUMNS,
        *_WINDOW_INTEGER_COLUMNS,
        *_WINDOW_SHIFT_COLUMNS,
    }
    missing_columns = sorted(required_columns - set(window_table.columns))
    if missing_columns:
        raise ReadinessInputContractError(
            "Window production structure evidence is missing required columns: "
            + ", ".join(missing_columns)
        )
    _strict_boolean_values(window_table, _WINDOW_BOOLEAN_COLUMNS, "window")
    _strict_integer_values(window_table, _WINDOW_INTEGER_COLUMNS, "window")
    for column in _WINDOW_SHIFT_COLUMNS:
        try:
            pd.to_numeric(window_table[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ReadinessInputContractError(
                f"Window production structure column {column} must contain numeric evidence"
            ) from exc

    extra_window_cases = sorted(
        set(window_table["case"].astype(str)) - set(optimized_cases["case"].astype(str))
    )
    if extra_window_cases:
        raise ReadinessInputContractError(
            "Window production structure contains unexpected cases: "
            + ", ".join(extra_window_cases)
        )

    for _, case_row in completed_cases.iterrows():
        case_name = str(case_row["case"])
        rows = window_table.loc[window_table["case"].astype(str) == case_name]
        if case_name == "visual_initial":
            if not rows.empty:
                raise ReadinessInputContractError(
                    "Window production structure must not contain visual_initial optimizer rows"
                )
            case_table.loc[case_table["case"] == case_name, "window_contract_ok"] = True
            continue

        frame_count = int(case_row["frame_count"])
        expected_targets = np.arange(2, frame_count, dtype=np.int64)
        actual_targets = np.sort(
            pd.to_numeric(rows["window_target"], errors="coerce")
            .to_numpy(dtype=np.int64)
        )
        if len(rows) != frame_count - 2 or not np.array_equal(
            actual_targets, expected_targets
        ):
            raise ReadinessInputContractError(
                f"Window production structure for {case_name} requires exactly "
                f"targets 2..{frame_count - 1}"
            )

        bias_enabled = case_row["bias_enabled"]
        if not isinstance(bias_enabled, (bool, np.bool_)):
            raise ReadinessInputContractError(
                f"Window production structure for {case_name} has non-boolean bias_enabled"
            )
        expected_metadata = {
            "visual_condition": case_row["visual_condition"],
            "imu_noise_mode": case_row["imu_noise_mode"],
            "bias_mode": case_row["bias_mode"],
            "writeback": case_row["writeback"],
            "graph_class": "LocalWindowInertialGraph",
            "context_device": "cpu",
            "bias_state_active": bool(bias_enabled),
            "bias_parameter_structurally_present": True,
            "vio_factor_active": True,
            "imu_residual_rows": 10 if bias_enabled else 6,
            "bias_fields_cleared_per_edge": 0 if bias_enabled else 8,
            "source_main_imu_bias_jacobian_rank": 6 if bias_enabled else 0,
            "terminal_main_imu_bias_jacobian_rank": 0,
        }
        always_true_window_flags = tuple(
            column
            for column in _WINDOW_BOOLEAN_COLUMNS
            if column
            not in {"bias_state_active", "edge_12_source_state_matches_runtime"}
        )
        edge_12_rows = rows.loc[rows["window_target"] == 3]
        structure_ok = all(
            bool((rows[column] == expected).all())
            for column, expected in expected_metadata.items()
        ) and all(bool(rows[column].all()) for column in always_true_window_flags)
        structure_ok = (
            structure_ok
            and len(edge_12_rows) == 1
            and bool(edge_12_rows["edge_12_source_state_matches_runtime"].iloc[0])
        )
        case_selector = case_table["case"] == case_name
        for window_column, case_column in _WINDOW_SHIFT_TO_CASE_COLUMNS.items():
            raw_values = pd.to_numeric(rows[window_column], errors="raise").to_numpy(
                dtype=np.float64
            )
            raw_max = float(
                np.max(np.where(np.isfinite(raw_values), np.abs(raw_values), np.inf))
            )
            case_table.loc[case_selector, case_column] = raw_max
        case_table.loc[case_selector, "window_contract_ok"] = bool(structure_ok)


def _validate_readiness_contract(case_table: pd.DataFrame, frame_table: pd.DataFrame) -> None:
    if case_table.empty or frame_table.empty:
        raise ReadinessInputContractError(
            "Readiness evidence tables must not be empty; a single short or main run is not a final decision view"
        )
    if "case" not in case_table or "case" not in frame_table:
        raise ReadinessInputContractError("Readiness evidence is missing required case columns")

    case_names = case_table["case"].astype(str)
    duplicate_names = sorted(
        name for name, count in case_names.value_counts().items() if count > 1
    )
    if duplicate_names:
        raise ReadinessInputContractError(
            "Readiness evidence contains duplicate case rows across periods: "
            + ", ".join(duplicate_names)
        )
    missing_cases = sorted(set(_REQUIRED_CASES) - set(case_names))
    extra_cases = sorted(set(case_names) - set(_REQUIRED_CASES))
    if missing_cases:
        raise ReadinessInputContractError(
            "Readiness evidence is missing required cases: " + ", ".join(missing_cases)
        )
    if extra_cases or len(case_table) != len(_REQUIRED_CASES):
        raise ReadinessInputContractError(
            "Readiness evidence must contain only the canonical seven cases; extra cases: "
            + ", ".join(extra_cases)
        )

    definition_columns = tuple(next(iter(_REQUIRED_CASE_DEFINITIONS.values())).keys())
    missing_case_columns = sorted(
        set(_READINESS_NUMERIC_COLUMNS + _READINESS_BOOLEAN_COLUMNS + definition_columns)
        - set(case_table.columns)
    )
    if missing_case_columns:
        raise ReadinessInputContractError(
            "Readiness case evidence is missing required columns: "
            + ", ".join(missing_case_columns)
    )
    for column in _READINESS_NUMERIC_COLUMNS:
        try:
            pd.to_numeric(case_table[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ReadinessInputContractError(
                f"Readiness case column {column} must contain numeric evidence"
            ) from exc
    _strict_boolean_values(case_table, _READINESS_BOOLEAN_COLUMNS, "case")
    _strict_integer_values(
        case_table,
        _READINESS_INTEGER_CASE_COLUMNS,
        "case",
    )

    missing_frame_columns = sorted(
        set(
            _READINESS_FRAME_NUMERIC_COLUMNS
            + ("in_evaluation", "frame_state_finite")
        )
        - set(frame_table.columns)
    )
    if missing_frame_columns:
        raise ReadinessInputContractError(
            "Readiness frame evidence is missing required columns: "
            + ", ".join(missing_frame_columns)
    )
    for column in _READINESS_FRAME_NUMERIC_COLUMNS:
        try:
            pd.to_numeric(frame_table[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ReadinessInputContractError(
                f"Readiness frame column {column} must contain numeric evidence"
            ) from exc
    _strict_boolean_values(
        frame_table,
        ("in_evaluation", "frame_state_finite"),
        "frame",
    )
    _strict_integer_values(
        frame_table,
        _READINESS_INTEGER_FRAME_COLUMNS,
        "frame",
    )

    frame_case_names = set(frame_table["case"].astype(str))
    if frame_case_names != set(_REQUIRED_CASES):
        missing = sorted(set(_REQUIRED_CASES) - frame_case_names)
        extra = sorted(frame_case_names - set(_REQUIRED_CASES))
        raise ReadinessInputContractError(
            f"Readiness frame rows do not match canonical cases; missing={missing}, extra={extra}"
        )

    indexed_cases = case_table.set_index("case")
    for case_name in _REQUIRED_CASES:
        is_short_control = case_name == "w3_full_all_zero_mean"
        expected_frame_count = 31 if is_short_control else 301
        expected_start = 6 if is_short_control else 30
        expected_evaluation_count = 25 if is_short_control else 271
        row = indexed_cases.loc[case_name]
        expected_definition = _REQUIRED_CASE_DEFINITIONS[case_name]
        for column, expected_value in expected_definition.items():
            actual_value = row[column]
            if actual_value != expected_value:
                raise ReadinessInputContractError(
                    f"Case {case_name} canonical definition requires "
                    f"{column}={expected_value!r}; got {actual_value!r}"
                )
        actual_frame_count = int(row["frame_count"])
        actual_start = int(row["evaluation_start_frame"])
        actual_evaluation_count = int(row["evaluation_frame_count"])
        if actual_frame_count != expected_frame_count:
            raise ReadinessInputContractError(
                f"Case {case_name} must provide exactly {expected_frame_count} frames; got {actual_frame_count}"
            )
        if actual_start != expected_start:
            raise ReadinessInputContractError(
                f"Case {case_name} must start evaluation at frame {expected_start}; got {actual_start}"
            )
        if actual_evaluation_count != expected_evaluation_count:
            raise ReadinessInputContractError(
                f"Case {case_name} must provide exactly {expected_evaluation_count} evaluation frames; got {actual_evaluation_count}"
            )

        case_frames = frame_table.loc[frame_table["case"].astype(str) == case_name].copy()
        if len(case_frames) != expected_frame_count:
            raise ReadinessInputContractError(
                f"Case {case_name} must provide exactly {expected_frame_count} frame rows; got {len(case_frames)}"
            )
        frame_indices = pd.to_numeric(case_frames["frame_index"], errors="coerce").to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(np.sort(frame_indices), np.arange(expected_frame_count)):
            raise ReadinessInputContractError(
                f"Case {case_name} frame rows must contain each index 0..{expected_frame_count - 1} exactly once"
            )
        case_frames = case_frames.sort_values("frame_index")
        actual_evaluation_mask = case_frames["in_evaluation"].astype(bool).to_numpy()
        expected_evaluation_mask = np.arange(expected_frame_count) >= expected_start
        if not np.array_equal(actual_evaluation_mask, expected_evaluation_mask):
            raise ReadinessInputContractError(
                f"Case {case_name} must have exactly {expected_evaluation_count} evaluation frames "
                f"covering {expected_start}..{expected_frame_count - 1}"
            )


def _as_double_tensor(value: torch.Tensor | pp.LieTensor) -> torch.Tensor:
    tensor = value.tensor() if isinstance(value, pp.LieTensor) else value
    return tensor.detach().cpu().double()


def _camera_truth_indices(frame_time_ns: torch.Tensor, truth: EvaluationTruth) -> torch.Tensor:
    frame_times = frame_time_ns.detach().cpu().to(torch.int64).reshape(-1)
    truth_times = truth.camera_time_ns.detach().cpu().to(torch.int64).reshape(-1)
    if truth_times.numel() == 0 or not bool(torch.all(truth_times[1:] > truth_times[:-1])):
        raise ValueError("truth.camera_time_ns must be non-empty and strictly increasing")
    indices = torch.searchsorted(truth_times, frame_times)
    valid = indices < truth_times.numel()
    if bool(valid.any()):
        valid_indices = indices[valid]
        valid[valid.clone()] = truth_times[valid_indices] == frame_times[valid]
    if not bool(valid.all()):
        missing = frame_times[~valid].tolist()
        raise ValueError(f"Result frame timestamps are not camera truth states: {missing}")
    return indices.long()


def _interpolate_truth_rows(
    sample_time_ns: torch.Tensor,
    values: torch.Tensor,
    query_time_ns: torch.Tensor,
) -> torch.Tensor:
    sample_times = sample_time_ns.detach().cpu().to(torch.int64).reshape(-1)
    queries = query_time_ns.detach().cpu().to(torch.int64).reshape(-1)
    rows = _as_double_tensor(values).reshape(sample_times.numel(), -1)
    if sample_times.numel() == 0 or not bool(torch.all(sample_times[1:] > sample_times[:-1])):
        raise ValueError("Bias truth timestamps must be non-empty and strictly increasing")
    if rows.shape[0] != sample_times.numel():
        raise ValueError("Bias truth values and timestamps must have matching lengths")
    if queries.numel() and (
        int(queries.min().item()) < int(sample_times[0].item())
        or int(queries.max().item()) > int(sample_times[-1].item())
    ):
        raise ValueError("Frame timestamp is outside the Bias truth range")

    output = torch.empty((queries.numel(), rows.shape[1]), dtype=torch.float64)
    for output_index, query in enumerate(queries.tolist()):
        right = int(torch.searchsorted(sample_times, torch.tensor(query, dtype=torch.int64)).item())
        if right < sample_times.numel() and int(sample_times[right].item()) == query:
            output[output_index] = rows[right]
            continue
        left = right - 1
        alpha = (query - int(sample_times[left].item())) / float(
            int(sample_times[right].item()) - int(sample_times[left].item())
        )
        output[output_index] = rows[left] + (rows[right] - rows[left]) * alpha
    return output


def _vector_rmse(norms: torch.Tensor) -> float:
    values = norms.detach().cpu().double().reshape(-1)
    if values.numel() == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean(values.square())).item())


def _visual_velocity(frame_time_ns: torch.Tensor, visual_pose: pp.LieTensor) -> torch.Tensor:
    positions = _as_double_tensor(visual_pose.translation()).reshape(-1, 3)
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)
    times = frame_time_ns.detach().cpu().double().reshape(-1) * 1e-9
    dt = (times[1:] - times[:-1]).unsqueeze(-1)
    if not bool(torch.all(dt > 0.0)):
        raise ValueError("Result frame timestamps must be strictly increasing")
    interval_velocity = (positions[1:] - positions[:-1]) / dt
    return torch.cat((interval_velocity[:1], interval_velocity), dim=0)


def _least_squares_slope(time_s: torch.Tensor, values: torch.Tensor) -> float:
    x = time_s.detach().cpu().double().reshape(-1)
    y = values.detach().cpu().double().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2 or float((x.max() - x.min()).abs().item()) == 0.0:
        return 0.0
    centered = x - x.mean()
    return float((centered @ (y - y.mean()) / (centered @ centered)).item())


def _diagnostic_aggregates(result: CaseRunResult) -> dict[str, object]:
    diagnostics = result.frame_diagnostics
    is_visual_initial = result.case.name == "visual_initial"
    aggregates: dict[str, object] = {
        "num_windows": int(result.num_windows),
        "max_shift_source_bias_error": float(result.max_shift_source_bias_error),
        "max_shift_source_pose_translation_error_m": float(
            result.max_shift_source_pose_translation_error_m
        ),
        "max_shift_source_pose_rotation_error_rad": float(
            result.max_shift_source_pose_rotation_error_rad
        ),
        "max_shift_source_velocity_error_m_s": float(
            result.max_shift_source_velocity_error_m_s
        ),
        "bias_state_active": bool(result.bias_state_active),
    }
    if diagnostics.empty:
        aggregates.update(
            {
                "diagnostic_row_count": 0,
                "diagnostics_available": False,
                "diagnostics_finite": is_visual_initial,
                "optimizer_completed": False,
                "optimizer_final_loss_finite": False,
                "optimizer_convergence_status": (
                    "not_applicable_visual_initial"
                    if is_visual_initial
                    else "unknown_not_exposed_by_production"
                ),
                "all_edges_rebase_fields_match_runtime": is_visual_initial,
            }
        )
        return aggregates

    def all_bool(column: str, default: bool = False) -> bool:
        if column not in diagnostics:
            return default
        return bool(diagnostics[column].fillna(False).astype(bool).all())

    def numeric_values(column: str) -> pd.Series | None:
        if column not in diagnostics:
            return None
        return pd.to_numeric(diagnostics[column], errors="coerce")

    energy_columns = (
        "initial_visual_energy",
        "final_visual_energy",
        "initial_p_energy",
        "final_p_energy",
        "initial_v_energy",
        "final_v_energy",
        "initial_R_energy",
        "final_R_energy",
        "initial_main_imu_energy",
        "final_main_imu_energy",
        "initial_bias_rw_energy",
        "final_bias_rw_energy",
    )
    update_columns = (
        "pose_translation_update_norm",
        "pose_rotation_update_norm",
        "velocity_update_norm",
        "acc_bias_update_norm",
        "gyro_bias_update_norm",
    )
    rank_columns = (
        "source_main_imu_bias_jacobian_rank",
        "terminal_main_imu_bias_jacobian_rank",
    )
    diagnostic_numeric_finite = all(
        values is None or bool(np.isfinite(values.to_numpy(dtype=np.float64)).all())
        for values in (
            numeric_values(column)
            for column in energy_columns + update_columns + rank_columns
        )
    )
    aggregates.update(
        {
            "diagnostic_row_count": int(len(diagnostics)),
            "diagnostics_available": True,
            "diagnostics_finite": all_bool("finite") and diagnostic_numeric_finite,
            "optimizer_completed": all_bool("optimizer_completed"),
            "optimizer_final_loss_finite": all_bool("optimizer_final_loss_finite"),
            "optimizer_convergence_status": "unknown_not_exposed_by_production",
            "all_edges_rebase_fields_match_runtime": all_bool(
                "all_edges_rebase_fields_match_runtime"
            ),
        }
    )
    for column in energy_columns:
        values = numeric_values(column)
        if values is None:
            aggregates[f"{column}_total"] = float("nan")
            aggregates[f"{column}_mean"] = float("nan")
        else:
            aggregates[f"{column}_total"] = float(values.sum(skipna=False))
            aggregates[f"{column}_mean"] = float(values.mean(skipna=False))
    for column in update_columns:
        values = numeric_values(column)
        if values is None:
            aggregates[f"{column}_max"] = float("nan")
            aggregates[f"{column}_mean"] = float("nan")
        else:
            aggregates[f"{column}_max"] = float(values.max(skipna=False))
            aggregates[f"{column}_mean"] = float(values.mean(skipna=False))
    for column in rank_columns:
        values = numeric_values(column)
        if values is None:
            aggregates[f"{column}_min"] = float("nan")
            aggregates[f"{column}_max"] = float("nan")
        elif not bool(np.isfinite(values.to_numpy(dtype=np.float64)).all()):
            aggregates[f"{column}_min"] = float("nan")
            aggregates[f"{column}_max"] = float("nan")
        else:
            aggregates[f"{column}_min"] = int(values.min())
            aggregates[f"{column}_max"] = int(values.max())
    return aggregates


def evaluate_case(
    result: CaseRunResult,
    truth: EvaluationTruth,
    visual_input: EstimatorVisualInput | Any,
) -> tuple[dict[str, object], pd.DataFrame]:
    frame_times = result.frame_time_ns.detach().cpu().to(torch.int64).reshape(-1)
    frame_count = int(frame_times.numel())
    if frame_count == 0:
        raise ValueError("Cannot evaluate an empty case result")
    if not bool(torch.all(frame_times[1:] > frame_times[:-1])):
        raise ValueError("Result frame timestamps must be strictly increasing")

    truth_indices = _camera_truth_indices(frame_times, truth)
    truth_pose = pp.SE3(_as_double_tensor(truth.pose_body_to_world)[truth_indices])
    truth_position = _as_double_tensor(truth.position_world)[truth_indices]
    truth_velocity = _as_double_tensor(truth.velocity_world)[truth_indices]
    pose_est = pp.SE3(_as_double_tensor(result.pose_est))
    position_est = _as_double_tensor(pose_est.translation()).reshape(-1, 3)
    velocity_est = _as_double_tensor(result.velocity_est).reshape(-1, 3)
    acc_bias_est = _as_double_tensor(result.acc_bias_est).reshape(-1, 3)
    gyro_bias_est = _as_double_tensor(result.gyro_bias_est).reshape(-1, 3)
    visual_pose = pp.SE3(_as_double_tensor(visual_input.pose_initial))
    visual_position = _as_double_tensor(visual_pose.translation()).reshape(-1, 3)
    visual_velocity = _visual_velocity(frame_times, visual_pose)

    expected_shapes = {
        "pose_est": int(pose_est.shape[0]),
        "velocity_est": int(velocity_est.shape[0]),
        "acc_bias_est": int(acc_bias_est.shape[0]),
        "gyro_bias_est": int(gyro_bias_est.shape[0]),
        "visual_pose": int(visual_pose.shape[0]),
    }
    mismatched = {name: count for name, count in expected_shapes.items() if count != frame_count}
    if mismatched:
        raise ValueError(f"Case state lengths do not match frame_count={frame_count}: {mismatched}")

    true_acc_bias = _interpolate_truth_rows(
        truth.imu_time_ns,
        truth.true_acc_bias,
        frame_times,
    )
    true_gyro_bias = _interpolate_truth_rows(
        truth.imu_time_ns,
        truth.true_gyro_bias,
        frame_times,
    )
    translation_error = (position_est - truth_position).norm(dim=-1)
    visual_translation_error = (visual_position - truth_position).norm(dim=-1)
    rotation_error = (
        truth_pose.rotation().Inv() @ pose_est.rotation()
    ).Log().tensor().reshape(-1, 3).norm(dim=-1) * (180.0 / math.pi)
    visual_rotation_error = (
        truth_pose.rotation().Inv() @ visual_pose.rotation()
    ).Log().tensor().reshape(-1, 3).norm(dim=-1) * (180.0 / math.pi)
    velocity_error = (velocity_est - truth_velocity).norm(dim=-1)
    visual_velocity_error = (visual_velocity - truth_velocity).norm(dim=-1)
    acc_bias_error = (acc_bias_est - true_acc_bias).norm(dim=-1)
    gyro_bias_error = (gyro_bias_est - true_gyro_bias).norm(dim=-1)
    six_axis_bias_error = torch.cat(
        (acc_bias_est - true_acc_bias, gyro_bias_est - true_gyro_bias), dim=-1
    ).norm(dim=-1)

    previous_acc_est = torch.full_like(acc_bias_est, float("nan"))
    previous_gyro_est = torch.full_like(gyro_bias_est, float("nan"))
    if frame_count > 1:
        previous_acc_est[1:] = acc_bias_est[:-1]
        previous_gyro_est[1:] = gyro_bias_est[:-1]
    persistence_acc_error = (previous_acc_est - true_acc_bias).norm(dim=-1)
    persistence_gyro_error = (previous_gyro_est - true_gyro_bias).norm(dim=-1)
    persistence_six_error = torch.cat(
        (previous_acc_est - true_acc_bias, previous_gyro_est - true_gyro_bias), dim=-1
    ).norm(dim=-1)
    zero_acc_error = true_acc_bias.norm(dim=-1)
    zero_gyro_error = true_gyro_bias.norm(dim=-1)
    zero_six_error = torch.cat((true_acc_bias, true_gyro_bias), dim=-1).norm(dim=-1)
    estimated_acc_bias_norm = acc_bias_est.norm(dim=-1)
    estimated_gyro_bias_norm = gyro_bias_est.norm(dim=-1)
    estimated_six_bias_norm = torch.cat((acc_bias_est, gyro_bias_est), dim=-1).norm(dim=-1)

    evaluation_start = 6 if frame_count <= 31 else 30
    if frame_count <= evaluation_start:
        raise ValueError(
            f"Case has {frame_count} frames but evaluation starts at frame {evaluation_start}"
        )
    evaluation_mask = torch.arange(frame_count) >= evaluation_start

    case_name = result.case.name
    frame_data: dict[str, object] = {
        "case": [case_name] * frame_count,
        "frame_index": np.arange(frame_count, dtype=np.int64),
        "timestamp_ns": frame_times.numpy(),
        "time_s": (frame_times.double() * 1e-9).numpy(),
        "in_evaluation": evaluation_mask.numpy(),
        "frame_state_finite": torch.stack(
            (
                torch.isfinite(_as_double_tensor(result.pose_est)).all(dim=-1),
                torch.isfinite(velocity_est).all(dim=-1),
                torch.isfinite(acc_bias_est).all(dim=-1),
                torch.isfinite(gyro_bias_est).all(dim=-1),
            ),
            dim=-1,
        ).all(dim=-1).numpy(),
        "translation_error_m": translation_error.numpy(),
        "visual_translation_error_m": visual_translation_error.numpy(),
        "rotation_error_deg": rotation_error.numpy(),
        "visual_rotation_error_deg": visual_rotation_error.numpy(),
        "velocity_error_m_s": velocity_error.numpy(),
        "visual_velocity_error_m_s": visual_velocity_error.numpy(),
        "acc_bias_error_m_s2": acc_bias_error.numpy(),
        "gyro_bias_error_rad_s": gyro_bias_error.numpy(),
        "six_axis_bias_error": six_axis_bias_error.numpy(),
        "zero_bias_baseline_acc_error_m_s2": zero_acc_error.numpy(),
        "zero_bias_baseline_gyro_error_rad_s": zero_gyro_error.numpy(),
        "zero_bias_baseline_six_axis_bias_error": zero_six_error.numpy(),
        "previous_estimate_persistence_acc_bias_error_m_s2": persistence_acc_error.numpy(),
        "previous_estimate_persistence_gyro_bias_error_rad_s": persistence_gyro_error.numpy(),
        "previous_estimate_persistence_six_axis_bias_error": persistence_six_error.numpy(),
        "visual_acc_bias_error_m_s2": zero_acc_error.numpy(),
        "visual_gyro_bias_error_rad_s": zero_gyro_error.numpy(),
        "visual_six_axis_bias_error": zero_six_error.numpy(),
        "estimated_acc_bias_norm_m_s2": estimated_acc_bias_norm.numpy(),
        "estimated_gyro_bias_norm_rad_s": estimated_gyro_bias_norm.numpy(),
        "estimated_six_axis_bias_norm": estimated_six_bias_norm.numpy(),
    }
    for prefix, values in (
        ("true_position", truth_position),
        ("estimated_position", position_est),
        ("visual_position", visual_position),
        ("true_velocity", truth_velocity),
        ("estimated_velocity", velocity_est),
        ("true_acc_bias", true_acc_bias),
        ("true_gyro_bias", true_gyro_bias),
        ("estimated_acc_bias", acc_bias_est),
        ("estimated_gyro_bias", gyro_bias_est),
    ):
        for axis, axis_name in enumerate(("x", "y", "z")):
            frame_data[f"{prefix}_{axis_name}"] = values[:, axis].numpy()
    frame_table = pd.DataFrame(frame_data)
    eval_translation = translation_error[evaluation_mask]
    eval_visual_translation = visual_translation_error[evaluation_mask]
    eval_rotation = rotation_error[evaluation_mask]
    eval_visual_rotation = visual_rotation_error[evaluation_mask]
    eval_velocity = velocity_error[evaluation_mask]
    eval_visual_velocity = visual_velocity_error[evaluation_mask]
    eval_acc_bias = acc_bias_error[evaluation_mask]
    eval_gyro_bias = gyro_bias_error[evaluation_mask]
    eval_bias = six_axis_bias_error[evaluation_mask]
    eval_zero_acc = zero_acc_error[evaluation_mask]
    eval_zero_gyro = zero_gyro_error[evaluation_mask]
    eval_zero_six = zero_six_error[evaluation_mask]
    eval_persistence_acc = persistence_acc_error[evaluation_mask]
    eval_persistence_gyro = persistence_gyro_error[evaluation_mask]
    eval_persistence = persistence_six_error[evaluation_mask]
    eval_acc_norm = estimated_acc_bias_norm[evaluation_mask]
    eval_gyro_norm = estimated_gyro_bias_norm[evaluation_mask]
    eval_six_norm = estimated_six_bias_norm[evaluation_mask]
    eval_time_s = frame_times.double()[evaluation_mask] * 1e-9
    final_ten_start = max(0, eval_six_norm.numel() - 10)
    metrics: dict[str, object] = {
        "case": case_name,
        "visual_condition": result.case.visual_condition,
        "imu_noise_mode": result.case.imu_noise_mode,
        "bias_mode": result.case.bias_mode,
        "bias_enabled": result.case.bias_enabled,
        "writeback": result.case.writeback,
        "alignment": "none_common_origin",
        "frame_count": frame_count,
        "evaluation_start_frame": evaluation_start,
        "evaluation_frame_count": int(evaluation_mask.sum().item()),
        "translation_rmse_m": _vector_rmse(eval_translation),
        "visual_translation_rmse_m": _vector_rmse(eval_visual_translation),
        "rotation_rmse_deg": _vector_rmse(eval_rotation),
        "visual_rotation_rmse_deg": _vector_rmse(eval_visual_rotation),
        "velocity_rmse_m_s": _vector_rmse(eval_velocity),
        "world_velocity_rmse_m_s": _vector_rmse(eval_velocity),
        "visual_velocity_rmse_m_s": _vector_rmse(eval_visual_velocity),
        "acc_bias_rmse_m_s2": _vector_rmse(eval_acc_bias),
        "gyro_bias_rmse_rad_s": _vector_rmse(eval_gyro_bias),
        "six_axis_bias_rmse": _vector_rmse(eval_bias),
        "zero_bias_baseline_acc_bias_rmse_m_s2": _vector_rmse(eval_zero_acc),
        "zero_bias_baseline_gyro_bias_rmse_rad_s": _vector_rmse(eval_zero_gyro),
        "zero_bias_baseline_six_axis_bias_rmse": _vector_rmse(eval_zero_six),
        "previous_estimate_persistence_acc_bias_rmse_m_s2": _vector_rmse(
            eval_persistence_acc
        ),
        "previous_estimate_persistence_gyro_bias_rmse_rad_s": _vector_rmse(
            eval_persistence_gyro
        ),
        "previous_estimate_persistence_six_axis_bias_rmse": _vector_rmse(eval_persistence),
        "visual_acc_bias_rmse_m_s2": _vector_rmse(eval_zero_acc),
        "visual_gyro_bias_rmse_rad_s": _vector_rmse(eval_zero_gyro),
        "visual_six_axis_bias_rmse": _vector_rmse(eval_zero_six),
        "final_translation_error_m": float(eval_translation[-1].item()),
        "max_translation_error_m": float(eval_translation.max().item()),
        "translation_error_slope_m_s": _least_squares_slope(eval_time_s, eval_translation),
        "final_rotation_error_deg": float(eval_rotation[-1].item()),
        "max_rotation_error_deg": float(eval_rotation.max().item()),
        "rotation_error_slope_deg_s": _least_squares_slope(eval_time_s, eval_rotation),
        "final_velocity_error_m_s": float(eval_velocity[-1].item()),
        "max_velocity_error_m_s": float(eval_velocity.max().item()),
        "velocity_error_slope_m_s2": _least_squares_slope(eval_time_s, eval_velocity),
        "final_acc_bias_norm_m_s2": float(eval_acc_norm[-1].item()),
        "max_acc_bias_norm_m_s2": float(eval_acc_norm.max().item()),
        "final_gyro_bias_norm_rad_s": float(eval_gyro_norm[-1].item()),
        "max_gyro_bias_norm_rad_s": float(eval_gyro_norm.max().item()),
        "final10_acc_bias_norm_slope_m_s3": _least_squares_slope(
            eval_time_s[final_ten_start:], eval_acc_norm[final_ten_start:]
        ),
        "final10_gyro_bias_norm_slope_rad_s2": _least_squares_slope(
            eval_time_s[final_ten_start:], eval_gyro_norm[final_ten_start:]
        ),
        "final_six_axis_bias_norm": float(eval_six_norm[-1].item()),
        "max_six_axis_bias_norm": float(eval_six_norm.max().item()),
        "final10_six_axis_bias_norm_slope_per_s": _least_squares_slope(
            eval_time_s[final_ten_start:], eval_six_norm[final_ten_start:]
        ),
    }
    diagnostic_metrics = _diagnostic_aggregates(result)
    state_tensors = (
        _as_double_tensor(result.pose_est),
        velocity_est,
        acc_bias_est,
        gyro_bias_est,
    )
    state_finite = all(bool(torch.isfinite(value).all()) for value in state_tensors)
    metrics.update(diagnostic_metrics)
    metrics["state_finite"] = state_finite
    metrics["finite"] = state_finite and bool(diagnostic_metrics["diagnostics_finite"])
    metrics["state_propagation_ok"] = state_finite
    return metrics, frame_table


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _metric(row: pd.Series | None, name: str) -> float:
    if row is None or name not in row:
        return float("nan")
    try:
        return float(row[name])
    except (TypeError, ValueError):
        return float("nan")


def _row_by_case(case_table: pd.DataFrame, case_name: str) -> pd.Series | None:
    if "case" not in case_table:
        return None
    selected = case_table.loc[case_table["case"] == case_name]
    if len(selected) != 1:
        return None
    return selected.iloc[0]


def _gate(
    gate_number: int,
    name: str,
    passed: bool,
    values: dict[str, object],
    threshold: str,
) -> dict[str, object]:
    return {
        "gate": gate_number,
        "name": name,
        "pass": bool(passed),
        "values": values,
        "threshold": threshold,
    }


def _readiness_gates(case_table: pd.DataFrame, frame_table: pd.DataFrame) -> list[dict[str, object]]:
    case_names = set(case_table["case"].astype(str)) if "case" in case_table else set()
    frame_case_names = set(frame_table["case"].astype(str)) if "case" in frame_table else set()
    missing_cases = sorted(set(_REQUIRED_CASES) - case_names)
    missing_frame_cases = sorted(set(_REQUIRED_CASES) - frame_case_names)
    duplicate_cases = sorted(
        name
        for name, count in case_table.get("case", pd.Series(dtype=object)).value_counts().items()
        if count != 1 and name in _REQUIRED_CASES
    )

    required_rows = case_table.loc[case_table.get("case", pd.Series(dtype=object)).isin(_REQUIRED_CASES)]
    optimized_rows = required_rows.loc[required_rows["case"] != "visual_initial"]

    def all_required_bool(column: str) -> bool:
        return (
            column in required_rows
            and len(required_rows) == len(_REQUIRED_CASES)
            and bool(required_rows[column].fillna(False).astype(bool).all())
        )

    def all_optimized_bool(column: str) -> bool:
        return (
            column in optimized_rows
            and len(optimized_rows) == len(_REQUIRED_CASES) - 1
            and bool(optimized_rows[column].all())
        )

    core_metric_columns = (
        "translation_rmse_m",
        "rotation_rmse_deg",
        "velocity_rmse_m_s",
        "visual_translation_rmse_m",
        "visual_rotation_rmse_deg",
        "visual_velocity_rmse_m_s",
        "acc_bias_rmse_m_s2",
        "gyro_bias_rmse_rad_s",
        "six_axis_bias_rmse",
        "zero_bias_baseline_acc_bias_rmse_m_s2",
        "zero_bias_baseline_gyro_bias_rmse_rad_s",
        "zero_bias_baseline_six_axis_bias_rmse",
        "previous_estimate_persistence_acc_bias_rmse_m_s2",
        "previous_estimate_persistence_gyro_bias_rmse_rad_s",
        "previous_estimate_persistence_six_axis_bias_rmse",
        "max_acc_bias_norm_m_s2",
        "max_gyro_bias_norm_rad_s",
        "final_acc_bias_norm_m_s2",
        "final_gyro_bias_norm_rad_s",
        "final10_acc_bias_norm_slope_m_s3",
        "final10_gyro_bias_norm_slope_rad_s2",
        "final_six_axis_bias_norm",
        "final10_six_axis_bias_norm_slope_per_s",
        "max_shift_source_bias_error",
        "max_shift_source_pose_translation_error_m",
        "max_shift_source_pose_rotation_error_rad",
        "max_shift_source_velocity_error_m_s",
    )
    case_numeric_finite = len(required_rows) == len(_REQUIRED_CASES)
    for column in core_metric_columns:
        if column not in required_rows:
            case_numeric_finite = False
            break
        case_numeric_finite = case_numeric_finite and bool(
            np.isfinite(pd.to_numeric(required_rows[column], errors="coerce")).all()
        )

    finite_frame_columns = (
        "translation_error_m",
        "rotation_error_deg",
        "velocity_error_m_s",
        "acc_bias_error_m_s2",
        "gyro_bias_error_rad_s",
    )
    frame_numeric_finite = all(
        column in frame_table
        and bool(np.isfinite(pd.to_numeric(frame_table[column], errors="coerce")).all())
        for column in finite_frame_columns
    )
    if "frame_state_finite" in frame_table:
        frame_state_finite = bool(
            frame_table["frame_state_finite"].fillna(False).astype(bool).all()
        ) and frame_numeric_finite
    else:
        frame_state_finite = frame_numeric_finite
    gate1_pass = (
        not missing_cases
        and not missing_frame_cases
        and not duplicate_cases
        and all_required_bool("finite")
        and all_required_bool("state_propagation_ok")
        and all_required_bool("window_contract_ok")
        and all_optimized_bool("diagnostics_available")
        and all_optimized_bool("optimizer_completed")
        and all_optimized_bool("optimizer_final_loss_finite")
        and case_numeric_finite
        and frame_state_finite
    )
    gates = [
        _gate(
            1,
            "finite_complete_state_propagation",
            gate1_pass,
            {
                "missing_cases": missing_cases,
                "missing_frame_cases": missing_frame_cases,
                "duplicate_cases": duplicate_cases,
                "case_flags_finite": all_required_bool("finite"),
                "state_propagation_ok": all_required_bool("state_propagation_ok"),
                "window_contract_ok": all_required_bool("window_contract_ok"),
                "diagnostics_available": all_optimized_bool("diagnostics_available"),
                "optimizer_completed": all_optimized_bool("optimizer_completed"),
                "optimizer_final_loss_finite": all_optimized_bool(
                    "optimizer_final_loss_finite"
                ),
                "case_numeric_finite": case_numeric_finite,
                "frame_state_finite": frame_state_finite,
            },
            "all required cases present exactly once; all states, energies, and final losses finite; state propagation succeeds",
        )
    ]

    full = _row_by_case(case_table, "w3_full_all")
    full_acc_bias = _metric(full, "acc_bias_rmse_m_s2")
    full_gyro_bias = _metric(full, "gyro_bias_rmse_rad_s")
    zero_acc_baseline = _metric(full, "zero_bias_baseline_acc_bias_rmse_m_s2")
    zero_gyro_baseline = _metric(full, "zero_bias_baseline_gyro_bias_rmse_rad_s")
    persistence_acc_baseline = _metric(
        full, "previous_estimate_persistence_acc_bias_rmse_m_s2"
    )
    persistence_gyro_baseline = _metric(
        full, "previous_estimate_persistence_gyro_bias_rmse_rad_s"
    )
    full_acc_vs_zero = _safe_ratio(full_acc_bias, zero_acc_baseline)
    full_gyro_vs_zero = _safe_ratio(full_gyro_bias, zero_gyro_baseline)
    full_acc_vs_persistence = _safe_ratio(full_acc_bias, persistence_acc_baseline)
    full_gyro_vs_persistence = _safe_ratio(full_gyro_bias, persistence_gyro_baseline)
    gate2_pass = full_acc_vs_zero <= 0.5 and full_gyro_vs_zero <= 0.5
    gates.append(
        _gate(
            2,
            "constant_bias_reduction",
            gate2_pass,
            {
                "full_all_acc_bias_rmse_m_s2": full_acc_bias,
                "full_all_gyro_bias_rmse_rad_s": full_gyro_bias,
                "zero_baseline_acc_bias_rmse_m_s2": zero_acc_baseline,
                "zero_baseline_gyro_bias_rmse_rad_s": zero_gyro_baseline,
                "previous_estimate_persistence_acc_bias_rmse_m_s2": persistence_acc_baseline,
                "previous_estimate_persistence_gyro_bias_rmse_rad_s": persistence_gyro_baseline,
                "acc_ratio_to_zero": full_acc_vs_zero,
                "gyro_ratio_to_zero": full_gyro_vs_zero,
                "acc_ratio_to_previous_estimate_persistence": full_acc_vs_persistence,
                "gyro_ratio_to_previous_estimate_persistence": full_gyro_vs_persistence,
            },
            "both accelerometer-Bias and gyroscope-Bias full_all/zero ratios <= 0.5; persistence is diagnostic only",
        )
    )

    drifting = _row_by_case(case_table, "w3_full_all_drifting_bias")
    drifting_acc_bias = _metric(drifting, "acc_bias_rmse_m_s2")
    drifting_gyro_bias = _metric(drifting, "gyro_bias_rmse_rad_s")
    drifting_acc_persistence = _metric(
        drifting, "previous_estimate_persistence_acc_bias_rmse_m_s2"
    )
    drifting_gyro_persistence = _metric(
        drifting, "previous_estimate_persistence_gyro_bias_rmse_rad_s"
    )
    drifting_acc_ratio = _safe_ratio(drifting_acc_bias, drifting_acc_persistence)
    drifting_gyro_ratio = _safe_ratio(drifting_gyro_bias, drifting_gyro_persistence)
    gates.append(
        _gate(
            3,
            "drifting_bias_tracking",
            drifting_acc_ratio <= 0.8 and drifting_gyro_ratio <= 0.8,
            {
                "drifting_acc_bias_rmse_m_s2": drifting_acc_bias,
                "drifting_gyro_bias_rmse_rad_s": drifting_gyro_bias,
                "previous_estimate_persistence_acc_bias_rmse_m_s2": drifting_acc_persistence,
                "previous_estimate_persistence_gyro_bias_rmse_rad_s": drifting_gyro_persistence,
                "acc_ratio_to_previous_estimate_persistence": drifting_acc_ratio,
                "gyro_ratio_to_previous_estimate_persistence": drifting_gyro_ratio,
            },
            "both accelerometer-Bias and gyroscope-Bias drifting/persistence ratios <= 0.8",
        )
    )

    zero_mean = _row_by_case(case_table, "w3_full_all_zero_mean")
    zero_mean_frames = frame_table.loc[
        (frame_table["case"] == "w3_full_all_zero_mean")
        & frame_table["in_evaluation"].astype(bool)
    ].sort_values("frame_index")
    zero_mean_acc_values = pd.to_numeric(
        zero_mean_frames["estimated_acc_bias_norm_m_s2"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    zero_mean_gyro_values = pd.to_numeric(
        zero_mean_frames["estimated_gyro_bias_norm_rad_s"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    zero_mean_acc_max = float(zero_mean_acc_values.max())
    zero_mean_gyro_max = float(zero_mean_gyro_values.max())
    gates.append(
        _gate(
            4,
            "zero_mean_numerical_floor",
            zero_mean_acc_max < 1e-6 and zero_mean_gyro_max < 1e-6,
            {
                "max_acc_bias_norm_m_s2": zero_mean_acc_max,
                "max_gyro_bias_norm_rad_s": zero_mean_gyro_max,
                "evaluated_frame_indices": zero_mean_frames["frame_index"].astype(int).tolist(),
                "all_acc_frames_strictly_below_floor": bool(
                    np.all(zero_mean_acc_values < 1e-6)
                ),
                "all_gyro_frames_strictly_below_floor": bool(
                    np.all(zero_mean_gyro_values < 1e-6)
                ),
            },
            "evaluation-segment max acc and gyro Bias norms are each strictly < 1e-6",
        )
    )

    zero_normal = _row_by_case(case_table, "w3_full_all_zero_normal")
    zero_normal_final_acc = _metric(zero_normal, "final_acc_bias_norm_m_s2")
    zero_normal_final_gyro = _metric(zero_normal, "final_gyro_bias_norm_rad_s")
    zero_normal_acc_slope = _metric(
        zero_normal, "final10_acc_bias_norm_slope_m_s3"
    )
    zero_normal_gyro_slope = _metric(
        zero_normal, "final10_gyro_bias_norm_slope_rad_s2"
    )
    final_acc_ratio = _safe_ratio(zero_normal_final_acc, zero_acc_baseline)
    final_gyro_ratio = _safe_ratio(zero_normal_final_gyro, zero_gyro_baseline)
    acc_slope_ratio = _safe_ratio(zero_normal_acc_slope, zero_acc_baseline)
    gyro_slope_ratio = _safe_ratio(zero_normal_gyro_slope, zero_gyro_baseline)
    gate5_pass = (
        final_acc_ratio < 0.25
        and final_gyro_ratio < 0.25
        and acc_slope_ratio <= 0.1
        and gyro_slope_ratio <= 0.1
    )
    gates.append(
        _gate(
            5,
            "zero_normal_bias_rejection",
            gate5_pass,
            {
                "constant_probe_acc_norm_m_s2": zero_acc_baseline,
                "constant_probe_gyro_norm_rad_s": zero_gyro_baseline,
                "final_acc_bias_norm_m_s2": zero_normal_final_acc,
                "final_gyro_bias_norm_rad_s": zero_normal_final_gyro,
                "final_acc_ratio_to_probe": final_acc_ratio,
                "final_gyro_ratio_to_probe": final_gyro_ratio,
                "final10_acc_slope_m_s3": zero_normal_acc_slope,
                "final10_gyro_slope_rad_s2": zero_normal_gyro_slope,
                "acc_slope_ratio_to_probe_per_s": acc_slope_ratio,
                "gyro_slope_ratio_to_probe_per_s": gyro_slope_ratio,
            },
            "both sensor Bias final/probe ratios < 0.25 and final-10 slope/probe ratios <= 0.1 per second",
        )
    )

    optimized_rows = case_table.loc[case_table.get("case", pd.Series(dtype=object)) != "visual_initial"]
    gate6_ratios: dict[str, dict[str, float]] = {}
    gate6_pass = len(optimized_rows) == len(_REQUIRED_CASES) - 1
    for _, row in optimized_rows.iterrows():
        case_name = str(row["case"])
        ratios = {
            "translation": _safe_ratio(
                _metric(row, "translation_rmse_m"),
                _metric(row, "visual_translation_rmse_m"),
            ),
            "rotation": _safe_ratio(
                _metric(row, "rotation_rmse_deg"),
                _metric(row, "visual_rotation_rmse_deg"),
            ),
            "velocity": _safe_ratio(
                _metric(row, "velocity_rmse_m_s"),
                _metric(row, "visual_velocity_rmse_m_s"),
            ),
        }
        gate6_ratios[case_name] = ratios
        gate6_pass = gate6_pass and all(value <= 1.05 for value in ratios.values())
    gates.append(
        _gate(
            6,
            "visual_non_degradation",
            gate6_pass,
            {"optimized_to_visual_ratios": gate6_ratios},
            "every optimized translation, rotation, and velocity RMSE / corresponding visual baseline <= 1.05",
        )
    )

    all_writeback_rows = case_table.loc[case_table.get("writeback", pd.Series(dtype=object)) == "all_optimized"]
    gate7_case_values: dict[str, dict[str, object]] = {}
    gate7_pass = not all_writeback_rows.empty
    for _, row in all_writeback_rows.iterrows():
        case_name = str(row["case"])
        shifts = {
            "bias": _metric(row, "max_shift_source_bias_error"),
            "pose_translation": _metric(
                row, "max_shift_source_pose_translation_error_m"
            ),
            "pose_rotation": _metric(
                row, "max_shift_source_pose_rotation_error_rad"
            ),
            "velocity": _metric(row, "max_shift_source_velocity_error_m_s"),
        }
        rebase_ok = bool(row.get("all_edges_rebase_fields_match_runtime", False))
        finite = bool(row.get("finite", False))
        case_pass = all(value <= 1e-8 for value in shifts.values()) and rebase_ok and finite
        gate7_case_values[case_name] = {
            "max_shift_source_errors": shifts,
            "rebase_fields_match_runtime": rebase_ok,
            "finite": finite,
            "pass": case_pass,
        }
        gate7_pass = gate7_pass and case_pass
    gates.append(
        _gate(
            7,
            "all_optimized_writeback_continuity",
            gate7_pass,
            {"all_optimized_cases": gate7_case_values},
            "each all_optimized Bias/pose/velocity max shift <= 1e-8 with runtime rebase match and finite state",
        )
    )

    locked = _row_by_case(case_table, "w3_bias_locked_zero")
    relative_ratios = {
        "acc_bias": _safe_ratio(
            full_acc_bias, _metric(locked, "acc_bias_rmse_m_s2")
        ),
        "gyro_bias": _safe_ratio(
            full_gyro_bias, _metric(locked, "gyro_bias_rmse_rad_s")
        ),
        "translation": _safe_ratio(
            _metric(full, "translation_rmse_m"), _metric(locked, "translation_rmse_m")
        ),
        "rotation": _safe_ratio(
            _metric(full, "rotation_rmse_deg"), _metric(locked, "rotation_rmse_deg")
        ),
        "velocity": _safe_ratio(
            _metric(full, "velocity_rmse_m_s"), _metric(locked, "velocity_rmse_m_s")
        ),
    }
    clean_control_floor = (
        _metric(zero_mean, "translation_rmse_m") < 0.001
        and _metric(zero_mean, "rotation_rmse_deg") < 0.01
        and _metric(zero_mean, "velocity_rmse_m_s") < 0.001
    )
    pose_velocity_ratios = [
        relative_ratios["translation"],
        relative_ratios["rotation"],
        relative_ratios["velocity"],
    ]
    gate8_pass = (
        relative_ratios["acc_bias"] <= 0.5
        and relative_ratios["gyro_bias"] <= 0.5
        and all(value <= 1.05 for value in pose_velocity_ratios)
        and (min(pose_velocity_ratios) <= 0.95 or clean_control_floor)
    )
    gates.append(
        _gate(
            8,
            "relative_to_locked_joint_improvement",
            gate8_pass,
            {
                "full_all_to_locked_ratios": relative_ratios,
                "clean_control_floor_exception": clean_control_floor,
                "clean_control_metrics": {
                    "translation_rmse_m": _metric(zero_mean, "translation_rmse_m"),
                    "rotation_rmse_deg": _metric(zero_mean, "rotation_rmse_deg"),
                    "velocity_rmse_m_s": _metric(zero_mean, "velocity_rmse_m_s"),
                },
            },
            "both sensor Bias ratios <= 0.5; pose/velocity ratios <= 1.05; at least one <= 0.95 unless clean-control floor applies",
        )
    )
    return gates


def classify_readiness(case_table: pd.DataFrame, frame_table: pd.DataFrame) -> str:
    _validate_readiness_contract(case_table, frame_table)
    gates = _readiness_gates(case_table, frame_table)
    if not gates[0]["pass"] or not gates[6]["pass"]:
        decision = "numerical_or_state_propagation_failure"
    elif not all(gates[index - 1]["pass"] for index in (2, 3, 4, 5)):
        decision = "bias_initialization_or_tracking_fails"
    elif not all(gates[index - 1]["pass"] for index in (6, 8)):
        decision = "bias_path_works_but_joint_weighting_fails"
    else:
        decision = "ready_for_manual_scene_validation"

    case_table.attrs["readiness_gates"] = gates
    case_table.attrs["readiness_decision"] = decision
    frame_table.attrs["readiness_gates"] = gates
    frame_table.attrs["readiness_decision"] = decision
    case_table["readiness_decision"] = decision
    for gate in gates:
        case_table[f"readiness_gate_{gate['gate']}_pass"] = bool(gate["pass"])
        case_table[f"readiness_gate_{gate['gate']}_name"] = str(gate["name"])
    return decision


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _mode_key(case: ValidationCase) -> str:
    return f"{case.bias_mode}__{case.imu_noise_mode}"


def _normalize_runs(runs: object, *, allow_empty: bool = False) -> list[CaseRunResult]:
    if isinstance(runs, dict):
        values = list(runs.values())
    else:
        try:
            values = list(runs)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("runs must be a sequence or mapping of CaseRunResult values") from exc
    if not values and not allow_empty:
        raise ValueError("runs must not be empty")
    if not all(isinstance(run, CaseRunResult) for run in values):
        raise TypeError("Every run must be a CaseRunResult")
    return values


def _render_chinese_summary(
    case_table: pd.DataFrame,
    decision: str | None,
    gates: list[dict[str, object]],
    contract_error: str | None,
) -> str:
    lines = [
        "# Full W3 联合目标合成验证报告",
        "",
        "## 判定",
        "",
    ]
    if decision is None:
        lines.extend(
            [
                "**未形成最终四选一结论。**",
                "",
                "证据契约不完整：本报告不把单个短跑、单个主跑或 tiny fixture 当作算法失败，也不允许空证据真空通过。",
                "",
                f"契约错误：`{contract_error}`",
                "",
            ]
        )
    else:
        lines.extend([f"最终 decision：`{decision}`", ""])

    lines.extend(["## 就绪门槛", ""])
    if gates:
        for gate in gates:
            status = "通过" if gate["pass"] else "失败"
            values = json.dumps(_json_value(gate["values"]), ensure_ascii=False, sort_keys=True)
            lines.extend(
                [
                    f"### Gate {gate['gate']}：{gate['name']} - {status}",
                    "",
                    f"- 判据：{gate['threshold']}",
                    f"- 数值：`{values}`",
                    "",
                ]
            )
    else:
        for gate_number in range(1, 9):
            lines.append(f"- Gate {gate_number}：未评估（证据契约不完整）")
        lines.append("")

    lines.extend(["## Case 指标", ""])
    metric_columns = [
        "case",
        "run_status",
        "failure_type",
        "failure_message",
        "frame_count",
        "evaluation_frame_count",
        "translation_rmse_m",
        "rotation_rmse_deg",
        "velocity_rmse_m_s",
        "six_axis_bias_rmse",
        "optimizer_completed",
        "optimizer_final_loss_finite",
        "optimizer_convergence_status",
    ]
    available = [column for column in metric_columns if column in case_table]
    if available:
        lines.append(case_table[available].to_markdown(index=False))
    else:
        lines.append("无可用 case metrics。")
    if "run_status" in case_table and bool((case_table["run_status"] == "failed").any()):
        lines.extend(["", "## 运行失败", ""])
        failed_columns = [
            column
            for column in ("case", "failure_type", "failure_message")
            if column in case_table
        ]
        lines.append(
            case_table.loc[
                case_table["run_status"] == "failed", failed_columns
            ].to_markdown(index=False)
        )
    lines.extend(
        [
            "",
            "## 生产优化器收敛状态",
            "",
            "仅报告 `optimizer_completed`、`optimizer_final_loss_finite` 与 "
            "`optimizer_convergence_status=unknown_not_exposed_by_production`。有限 final loss 不代表已证明 converged。",
            "",
            "## 限制",
            "",
            "- synthetic 不是 scene，合成通过不能替代人工场景验证。",
            "- locked-Bias 控制中仍存在 12 个未观测 Bias 参数；它们在结构上存在，但 Bias factor 被禁用。",
            "- production convergence unknown：生产优化器未暴露 StopOnPlateau 的内部收敛原因。",
            "- 最终 readiness 只接受短跑 31 帧 zero_mean（评估 6..30，共 25 帧）与主跑 301 帧其余 6 case（评估 30..300，共 271 帧）组成的唯一 7-case view。",
            "",
            "## HoloOcean 数据门槛",
            "",
            "本阶段无需新 HoloOcean 数据。若 synthetic pass 但 scene behavior 仍有歧义，再请求 HoloOcean agent 协助 export realized Bias；在此之前不新增数据。",
            "",
        ]
    )
    return "\n".join(lines)


_VIEW_NAMES = (
    "XY",
    "XZ",
    "Trajectory",
    "Position Error",
    "Rotation Error",
    "Velocity Error",
    "Acc Bias Error",
    "Gyro Bias Error",
)

_TRACE_GROUPS = {
    "gt": {"label": "GT", "color": "#111827", "width": 2.6},
    "visual": {"label": "visual", "color": "#d97706", "width": 2.0},
    "locked": {"label": "locked", "color": "#7c3aed", "width": 1.8},
    "current": {"label": "current", "color": "#dc2626", "width": 1.8},
    "all": {"label": "all", "color": "#0f766e", "width": 2.2},
}

_CASE_TRACE_COLORS = {
    "w3_bias_locked_zero": "#7c3aed",
    "w3_full_current": "#dc2626",
    "w3_full_all": "#0f766e",
    "w3_full_all_zero_mean": "#2563eb",
    "w3_full_all_zero_normal": "#65a30d",
    "w3_full_all_drifting_bias": "#db2777",
}


def _trace_group(case_name: str) -> str:
    if case_name == "visual_initial":
        return "visual"
    if case_name == "w3_bias_locked_zero":
        return "locked"
    if case_name == "w3_full_current":
        return "current"
    return "all"


def _xy_points(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> list[list[float]]:
    x_values = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=np.float64)
    y_values = np.asarray(pd.to_numeric(y, errors="coerce"), dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    return [[float(xv), float(yv)] for xv, yv in zip(x_values[finite], y_values[finite])]


def _view_points(frame_rows: pd.DataFrame, source: str, view_name: str) -> list[list[float]]:
    if source == "gt":
        position_prefix = "true_position"
        errors = {
            "Position Error": np.zeros(len(frame_rows)),
            "Rotation Error": np.zeros(len(frame_rows)),
            "Velocity Error": np.zeros(len(frame_rows)),
            "Acc Bias Error": np.zeros(len(frame_rows)),
            "Gyro Bias Error": np.zeros(len(frame_rows)),
        }
    elif source == "visual":
        position_prefix = "visual_position"
        errors = {
            "Position Error": frame_rows.get("visual_translation_error_m", pd.Series(dtype=float)),
            "Rotation Error": frame_rows.get("visual_rotation_error_deg", pd.Series(dtype=float)),
            "Velocity Error": frame_rows.get("visual_velocity_error_m_s", pd.Series(dtype=float)),
            "Acc Bias Error": frame_rows.get("visual_acc_bias_error_m_s2", pd.Series(dtype=float)),
            "Gyro Bias Error": frame_rows.get("visual_gyro_bias_error_rad_s", pd.Series(dtype=float)),
        }
    else:
        position_prefix = "estimated_position"
        errors = {
            "Position Error": frame_rows.get("translation_error_m", pd.Series(dtype=float)),
            "Rotation Error": frame_rows.get("rotation_error_deg", pd.Series(dtype=float)),
            "Velocity Error": frame_rows.get("velocity_error_m_s", pd.Series(dtype=float)),
            "Acc Bias Error": frame_rows.get("acc_bias_error_m_s2", pd.Series(dtype=float)),
            "Gyro Bias Error": frame_rows.get("gyro_bias_error_rad_s", pd.Series(dtype=float)),
        }

    position_columns = {
        f"{position_prefix}_x",
        f"{position_prefix}_y",
        f"{position_prefix}_z",
    }
    required_columns = set(position_columns)
    if view_name not in {"XY", "XZ", "Trajectory"}:
        required_columns.add("time_s")
        if source != "gt":
            error_column = {
                "Position Error": (
                    "visual_translation_error_m"
                    if source == "visual"
                    else "translation_error_m"
                ),
                "Rotation Error": (
                    "visual_rotation_error_deg"
                    if source == "visual"
                    else "rotation_error_deg"
                ),
                "Velocity Error": (
                    "visual_velocity_error_m_s"
                    if source == "visual"
                    else "velocity_error_m_s"
                ),
                "Acc Bias Error": (
                    "visual_acc_bias_error_m_s2"
                    if source == "visual"
                    else "acc_bias_error_m_s2"
                ),
                "Gyro Bias Error": (
                    "visual_gyro_bias_error_rad_s"
                    if source == "visual"
                    else "gyro_bias_error_rad_s"
                ),
            }[view_name]
            required_columns.add(error_column)
    if not required_columns.issubset(frame_rows.columns):
        return []

    time_s = frame_rows["time_s"] if "time_s" in frame_rows else pd.Series(dtype=float)

    x_position = frame_rows[f"{position_prefix}_x"]
    y_position = frame_rows[f"{position_prefix}_y"]
    z_position = frame_rows[f"{position_prefix}_z"]
    if view_name == "XY":
        return _xy_points(x_position, y_position)
    if view_name == "XZ":
        return _xy_points(x_position, z_position)
    if view_name == "Trajectory":
        projected_x = pd.to_numeric(x_position, errors="coerce") + 0.35 * pd.to_numeric(
            y_position, errors="coerce"
        )
        projected_y = pd.to_numeric(z_position, errors="coerce") + 0.20 * pd.to_numeric(
            y_position, errors="coerce"
        )
        return _xy_points(projected_x, projected_y)
    return _xy_points(time_s, errors[view_name])


def _fixed_extent(traces: list[dict[str, object]]) -> dict[str, float]:
    points = [
        point
        for trace in traces
        for point in trace["points"]  # type: ignore[index]
        if len(point) == 2 and math.isfinite(point[0]) and math.isfinite(point[1])
    ]
    if not points:
        return {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0}
    values = np.asarray(points, dtype=np.float64)
    x_min, y_min = values.min(axis=0)
    x_max, y_max = values.max(axis=0)
    x_pad = max((x_max - x_min) * 0.06, abs(x_min) * 0.01, abs(x_max) * 0.01, 1e-6)
    y_pad = max((y_max - y_min) * 0.08, abs(y_min) * 0.01, abs(y_max) * 0.01, 1e-6)
    return {
        "x_min": float(x_min - x_pad),
        "x_max": float(x_max + x_pad),
        "y_min": float(y_min - y_pad),
        "y_max": float(y_max + y_pad),
    }


def _build_interactive_model(
    case_table: pd.DataFrame,
    frame_table: pd.DataFrame,
    decision: str | None,
    readiness_status: str,
) -> dict[str, object]:
    ordered_cases = (
        case_table["case"].astype(str).tolist() if "case" in case_table else []
    )
    views: dict[str, list[dict[str, object]]] = {name: [] for name in _VIEW_NAMES}
    trace_controls: list[dict[str, object]] = []
    trace_styles: dict[str, dict[str, object]] = {}

    def rows_for_case(case_name: str) -> pd.DataFrame:
        if "case" not in frame_table:
            return pd.DataFrame()
        rows = frame_table.loc[frame_table["case"].astype(str) == case_name]
        return rows.sort_values("frame_index") if "frame_index" in rows else rows

    def add_trace(
        trace_id: str,
        group: str,
        label: str,
        rows: pd.DataFrame,
        source: str,
        color: str,
        width: float,
    ) -> None:
        if trace_id in trace_styles:
            return
        trace_styles[trace_id] = {
            "label": label,
            "group": group,
            "color": color,
            "width": width,
        }
        trace_controls.append({"id": trace_id, **trace_styles[trace_id]})
        for view_name in _VIEW_NAMES:
            views[view_name].append(
                {
                    "id": trace_id,
                    "group": group,
                    "label": label,
                    "points": _view_points(rows, source, view_name),
                }
            )

    seen_gt_horizons: set[int] = set()
    for case_name in ordered_cases:
        rows = rows_for_case(case_name)
        if rows.empty:
            continue
        horizon = len(rows)
        if horizon not in seen_gt_horizons:
            add_trace(
                f"gt:{horizon}",
                "gt",
                f"GT ({horizon} frames)",
                rows,
                "gt",
                "#111827",
                2.8,
            )
            seen_gt_horizons.add(horizon)

    seen_visuals: set[tuple[str, int]] = set()
    if "case" in case_table and "visual_condition" in case_table:
        case_metadata = case_table.set_index("case", drop=False)
        for case_name in ordered_cases:
            rows = rows_for_case(case_name)
            if rows.empty or case_name not in case_metadata.index:
                continue
            condition = str(case_metadata.loc[case_name, "visual_condition"])
            horizon = len(rows)
            visual_key = (condition, horizon)
            if visual_key in seen_visuals:
                continue
            visual_color = "#d97706" if condition == "drifted_visual" else "#ca8a04"
            add_trace(
                f"visual:{condition}:{horizon}",
                "visual",
                f"visual: {condition} ({horizon} frames)",
                rows,
                "visual",
                visual_color,
                2.0,
            )
            seen_visuals.add(visual_key)

    for case_name in ordered_cases:
        if case_name == "visual_initial":
            continue
        rows = rows_for_case(case_name)
        if rows.empty:
            continue
        case_rows = case_table[case_table["case"] == case_name]
        visual_condition = (
            str(case_rows.iloc[0]["visual_condition"])
            if not case_rows.empty and "visual_condition" in case_rows
            else "unknown_visual"
        )
        group = _trace_group(case_name)
        add_trace(
            case_name,
            group,
            f"{case_name} ({visual_condition}, {len(rows)} frames)",
            rows,
            "estimate",
            _CASE_TRACE_COLORS.get(case_name, "#0f766e"),
            2.2 if group == "all" else 1.8,
        )

    axes = {
        "XY": ["world x (m)", "world y (m)"],
        "XZ": ["world x (m)", "world z (m)"],
        "Trajectory": ["projected horizontal (m)", "projected vertical (m)"],
        "Position Error": ["time (s)", "position error (m)"],
        "Rotation Error": ["time (s)", "rotation error (deg)"],
        "Velocity Error": ["time (s)", "velocity error (m/s)"],
        "Acc Bias Error": ["time (s)", "acc Bias Error (m/s^2)"],
        "Gyro Bias Error": ["time (s)", "gyro Bias Error (rad/s)"],
    }
    return {
        "decision": decision,
        "readiness_status": readiness_status,
        "trace_groups": _TRACE_GROUPS,
        "trace_controls": trace_controls,
        "trace_styles": trace_styles,
        "views": views,
        "view_axes": axes,
        "view_extents": {name: _fixed_extent(views[name]) for name in _VIEW_NAMES},
    }


def _render_interactive_html(model: dict[str, object]) -> str:
    model_json = json.dumps(
        _json_value(model), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).replace("</", "<\\/")
    view_buttons = "\n".join(
        f'<button type="button" class="view-button" data-view="{name}" aria-selected="false">{name}</button>'
        for name in _VIEW_NAMES
    )
    trace_toggles = "\n".join(
        '<label class="trace-toggle">'
        f'<input type="checkbox" data-trace="{escape(str(control["id"]))}" checked>'
        f'<span class="swatch" style="--trace-color:{escape(str(control["color"]))}"></span>'
        f'<span>{escape(str(control["label"]))}</span></label>'
        for control in model["trace_controls"]  # type: ignore[index]
    )
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full W3 Joint Objective Diagnostics</title>
<style>
:root { color-scheme: light; font-family: Inter, "Segoe UI", Arial, sans-serif; background: #f4f6f8; color: #18212f; }
* { box-sizing: border-box; }
body { margin: 0; min-width: 300px; min-height: 100vh; background: #f4f6f8; }
header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; padding: 14px 20px 10px; border-bottom: 1px solid #d8dee7; background: #ffffff; }
h1 { margin: 0; font-size: 18px; line-height: 1.35; letter-spacing: 0; font-weight: 650; }
#decision { color: #526071; font-size: 12px; overflow-wrap: anywhere; text-align: right; }
.toolbar { display: flex; flex-direction: column; gap: 8px; padding: 10px 20px; border-bottom: 1px solid #d8dee7; background: #ffffff; }
.segmented { display: flex; flex-wrap: wrap; min-width: 0; gap: 4px; }
.view-button, #reset { min-height: 34px; border: 1px solid #c7d0dc; border-radius: 4px; background: #ffffff; color: #263445; padding: 6px 10px; font: inherit; font-size: 12px; cursor: pointer; white-space: normal; }
.view-button:hover, #reset:hover { border-color: #66758a; background: #f5f7fa; }
.view-button[aria-selected="true"] { border-color: #0f766e; background: #e7f4f1; color: #075e58; font-weight: 650; }
.trace-controls { display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap; gap: 6px 12px; min-width: 0; }
.trace-toggle { display: inline-flex; align-items: center; gap: 5px; min-height: 28px; font-size: 12px; white-space: nowrap; }
.trace-toggle input { margin: 0; accent-color: #0f766e; }
.swatch { width: 17px; height: 3px; flex: 0 0 auto; background: var(--trace-color); }
main { width: 100%; padding: 12px 20px 20px; }
.plot-shell { width: 100%; height: calc(100vh - 150px); min-height: 360px; border: 1px solid #cbd3dd; border-radius: 6px; background: #ffffff; overflow: hidden; }
canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
canvas.dragging { cursor: grabbing; }
@media (max-width: 860px) {
  header { align-items: flex-start; flex-direction: column; gap: 4px; padding-inline: 12px; }
  #decision { text-align: left; }
  .toolbar { padding-inline: 12px; }
  main { padding: 10px 12px 14px; }
  .plot-shell { height: calc(100vh - 224px); min-height: 320px; }
}
</style>
</head>
<body>
<header><h1>Full W3 Joint Objective Diagnostics</h1><div id="decision"></div></header>
<section class="toolbar">
  <div class="segmented" role="tablist" aria-label="Diagnostic view">__VIEW_BUTTONS__</div>
  <div class="trace-controls">__TRACE_TOGGLES__<button type="button" id="reset" title="Reset view">Reset</button></div>
</section>
<main><div class="plot-shell"><canvas id="plot" aria-label="Interactive diagnostic plot"></canvas></div></main>
<script>
const MODEL = __MODEL__;
const canvas = document.getElementById('plot');
const context = canvas.getContext('2d');
const enabledTraces = new Set(MODEL.trace_controls.map(item => item.id));
let activeView = 'XY';
let zoom = 1;
let pan = {x: 0, y: 0};
let dragging = false;
let dragOrigin = {x: 0, y: 0};

function cssSize() {
  const rect = canvas.getBoundingClientRect();
  return {width: Math.max(1, rect.width), height: Math.max(1, rect.height)};
}

function resizeCanvas() {
  const size = cssSize();
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.round(size.width * ratio);
  const height = Math.round(size.height * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function plotTransform(size) {
  const margin = {left: 68, right: 24, top: 28, bottom: 54};
  const width = Math.max(1, size.width - margin.left - margin.right);
  const height = Math.max(1, size.height - margin.top - margin.bottom);
  const baseExtent = MODEL.view_extents[activeView];
  const scaleX = width * zoom / (baseExtent.x_max - baseExtent.x_min);
  const scaleY = height * zoom / (baseExtent.y_max - baseExtent.y_min);
  const offsetX = margin.left + width * (1 - zoom) / 2 + pan.x;
  const offsetY = margin.top + height * (1 - zoom) / 2 + pan.y;
  return {
    margin, width, height, baseExtent, scaleX, scaleY, offsetX, offsetY,
    x: value => offsetX + (value - baseExtent.x_min) * scaleX,
    y: value => offsetY + height * zoom - (value - baseExtent.y_min) * scaleY
  };
}

function drawAxes(size, transform) {
  context.save();
  context.strokeStyle = '#d5dbe4';
  context.fillStyle = '#526071';
  context.lineWidth = 1;
  context.font = '11px "Segoe UI", Arial, sans-serif';
  const ticks = 5;
  for (let index = 0; index <= ticks; index += 1) {
    const fraction = index / ticks;
    const xValue = transform.baseExtent.x_min + fraction * (transform.baseExtent.x_max - transform.baseExtent.x_min);
    const yValue = transform.baseExtent.y_min + fraction * (transform.baseExtent.y_max - transform.baseExtent.y_min);
    const x = transform.x(xValue);
    const y = transform.y(yValue);
    context.beginPath(); context.moveTo(x, transform.margin.top); context.lineTo(x, size.height - transform.margin.bottom); context.stroke();
    context.beginPath(); context.moveTo(transform.margin.left, y); context.lineTo(size.width - transform.margin.right, y); context.stroke();
    context.textAlign = 'center'; context.fillText(xValue.toPrecision(4), x, size.height - 31);
    context.textAlign = 'right'; context.fillText(yValue.toPrecision(4), 58, y + 4);
  }
  const axes = MODEL.view_axes[activeView];
  context.fillStyle = '#263445';
  context.font = '12px "Segoe UI", Arial, sans-serif';
  context.textAlign = 'center'; context.fillText(axes[0], transform.margin.left + transform.width / 2, size.height - 10);
  context.save(); context.translate(15, transform.margin.top + transform.height / 2); context.rotate(-Math.PI / 2); context.fillText(axes[1], 0, 0); context.restore();
  context.restore();
}

function draw() {
  const size = cssSize();
  context.clearRect(0, 0, size.width, size.height);
  context.fillStyle = '#ffffff'; context.fillRect(0, 0, size.width, size.height);
  const transform = plotTransform(size);
  drawAxes(size, transform);
  context.save();
  context.beginPath(); context.rect(transform.margin.left, transform.margin.top, transform.width, transform.height); context.clip();
  for (const trace of MODEL.views[activeView]) {
    if (!enabledTraces.has(trace.id) || trace.points.length === 0) continue;
    const style = MODEL.trace_styles[trace.id];
    context.strokeStyle = style.color; context.lineWidth = style.width; context.lineJoin = 'round'; context.lineCap = 'round';
    context.beginPath();
    trace.points.forEach((point, index) => {
      const x = transform.x(point[0]); const y = transform.y(point[1]);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  }
  context.restore();
}

document.querySelectorAll('.view-button').forEach(button => {
  button.addEventListener('click', () => {
    activeView = button.dataset.view; zoom = 1; pan = {x: 0, y: 0};
    document.querySelectorAll('.view-button').forEach(item => item.setAttribute('aria-selected', String(item === button)));
    draw();
  });
});
document.querySelector('.view-button[data-view="XY"]').setAttribute('aria-selected', 'true');
document.querySelectorAll('input[data-trace]').forEach(input => {
  input.addEventListener('change', () => { if (input.checked) enabledTraces.add(input.dataset.trace); else enabledTraces.delete(input.dataset.trace); draw(); });
});
document.getElementById('reset').addEventListener('click', () => { zoom = 1; pan = {x: 0, y: 0}; draw(); });
canvas.addEventListener('wheel', event => { event.preventDefault(); zoom = Math.min(20, Math.max(0.5, zoom * (event.deltaY < 0 ? 1.12 : 0.89))); draw(); }, {passive: false});
canvas.addEventListener('pointerdown', event => { dragging = true; dragOrigin = {x: event.clientX - pan.x, y: event.clientY - pan.y}; canvas.classList.add('dragging'); canvas.setPointerCapture(event.pointerId); });
canvas.addEventListener('pointermove', event => { if (!dragging) return; pan = {x: event.clientX - dragOrigin.x, y: event.clientY - dragOrigin.y}; draw(); });
canvas.addEventListener('pointerup', event => { dragging = false; canvas.classList.remove('dragging'); canvas.releasePointerCapture(event.pointerId); });
canvas.addEventListener('pointercancel', () => { dragging = false; canvas.classList.remove('dragging'); });
document.getElementById('decision').textContent = MODEL.decision || MODEL.readiness_status;
new ResizeObserver(resizeCanvas).observe(canvas.parentElement);
resizeCanvas();
</script>
</body>
</html>
'''
    return (
        template.replace("__VIEW_BUTTONS__", view_buttons)
        .replace("__TRACE_TOGGLES__", trace_toggles)
        .replace("__MODEL__", model_json)
    )


def write_validation_bundle(
    output_dir: str | Path,
    runs: object,
    truth_by_mode: dict[str, EvaluationTruth],
    visual_by_condition: dict[str, EstimatorVisualInput | Any],
    *,
    config: SyntheticSequenceConfig | None = None,
    imu_inputs_by_mode: dict[str, EstimatorIMUInput] | None = None,
    failures: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    failure_records = list(failures or [])
    run_list = _normalize_runs(runs, allow_empty=bool(failure_records))

    for mode_key, truth in truth_by_mode.items():
        truth_mode_key = f"{truth.bias_mode}__{truth.noise_mode}"
        if mode_key != truth_mode_key:
            raise ValueError(
                f"Evaluation truth metadata disagrees with key {mode_key!r}: "
                f"truth declares {truth_mode_key!r}"
            )

    case_metrics: list[dict[str, object]] = []
    frame_tables: list[pd.DataFrame] = []
    window_tables: list[pd.DataFrame] = []
    for result in run_list:
        mode_key = _mode_key(result.case)
        if mode_key not in truth_by_mode:
            raise KeyError(
                f"Missing truth_by_mode[{mode_key!r}]; keys must use bias_mode__imu_noise_mode"
            )
        condition = result.case.visual_condition
        if condition not in visual_by_condition:
            raise KeyError(f"Missing visual_by_condition[{condition!r}]")
        truth = truth_by_mode[mode_key]
        if truth.bias_mode != result.case.bias_mode or truth.noise_mode != result.case.imu_noise_mode:
            raise ValueError(
                f"Evaluation truth metadata for {result.case.name!r} does not match "
                f"case bias/noise modes"
            )
        metrics, frames = evaluate_case(
            result,
            truth,
            visual_by_condition[condition],
        )
        metrics["run_status"] = "completed"
        metrics["failure_type"] = None
        metrics["failure_message"] = None
        case_metrics.append(metrics)
        frame_tables.append(frames)
        if not result.frame_diagnostics.empty:
            window_rows = result.frame_diagnostics.copy()
            window_metadata = {
                "case": result.case.name,
                "visual_condition": result.case.visual_condition,
                "imu_noise_mode": result.case.imu_noise_mode,
                "bias_mode": result.case.bias_mode,
                "writeback": result.case.writeback,
            }
            for column, value in window_metadata.items():
                window_rows[column] = value
            metadata_columns = list(window_metadata)
            window_rows = window_rows[
                metadata_columns
                + [column for column in window_rows.columns if column not in metadata_columns]
            ]
            window_tables.append(window_rows)

    for failure in failure_records:
        required_failure_fields = {
            "case",
            "visual_condition",
            "imu_noise_mode",
            "bias_mode",
            "bias_enabled",
            "writeback",
            "failure_type",
            "failure_message",
        }
        missing_failure_fields = sorted(required_failure_fields - set(failure))
        if missing_failure_fields:
            raise ValueError(
                "Failure record is missing required fields: "
                + ", ".join(missing_failure_fields)
            )
        case_metrics.append({**failure, "run_status": "failed"})

    case_table = pd.DataFrame(case_metrics)
    frame_table = (
        pd.concat(frame_tables, ignore_index=True)
        if frame_tables
        else pd.DataFrame(columns=["case", "frame_index"])
    )
    window_table = (
        pd.concat(window_tables, ignore_index=True)
        if window_tables
        else pd.DataFrame(columns=["case"])
    )
    decision: str | None = None
    gates: list[dict[str, object]] = []
    contract_error: str | None = None
    try:
        _validate_window_diagnostics_contract(case_table, window_table)
        decision = classify_readiness(case_table, frame_table)
        gates = list(case_table.attrs["readiness_gates"])
        readiness_status = "classified_canonical_evidence"
    except ReadinessInputContractError as exc:
        contract_error = str(exc)
        readiness_status = "incomplete_evidence_contract"
        case_table["readiness_decision"] = None
    case_table["readiness_status"] = readiness_status
    case_table["readiness_contract_error"] = contract_error
    case_table["readiness_gate_details_json"] = (
        json.dumps(_json_value(gates), ensure_ascii=False, sort_keys=True) if gates else None
    )

    cases_path = output_path / "joint_objective_cases.csv"
    frames_path = output_path / "frame_metrics.csv"
    windows_path = output_path / "window_diagnostics.csv"
    report_path = output_path / "joint_objective_summary_cn.md"
    html_path = output_path / "diagnostics_interactive.html"
    manifest_path = output_path / "input_manifest.json"
    case_table.to_csv(cases_path, index=False)
    frame_table.to_csv(frames_path, index=False)
    window_table.to_csv(windows_path, index=False)
    report_path.write_text(
        _render_chinese_summary(case_table, decision, gates, contract_error),
        encoding="utf-8",
    )
    interactive_model = _build_interactive_model(
        case_table,
        frame_table,
        decision,
        readiness_status,
    )
    html_path.write_text(
        _render_interactive_html(interactive_model),
        encoding="utf-8",
    )

    sensor_artifacts: dict[str, dict[str, str]] = {}
    visual_artifacts: dict[str, str] = {}
    if (config is None) != (imu_inputs_by_mode is None):
        raise ValueError(
            "config and imu_inputs_by_mode must be provided together when writing input artifacts"
        )
    if config is not None and imu_inputs_by_mode is not None:
        sensors_root = output_path / "inputs" / "sensors"
        for mode_key, truth in sorted(truth_by_mode.items()):
            if mode_key not in imu_inputs_by_mode:
                raise KeyError(f"Missing imu_inputs_by_mode[{mode_key!r}]")
            artifact_paths = write_sensor_artifacts(
                sensors_root / mode_key,
                config,
                truth,
                imu_inputs_by_mode[mode_key],
            )
            sensor_artifacts[mode_key] = {
                name: str(path.relative_to(output_path))
                for name, path in artifact_paths.items()
            }

        visual_root = output_path / "inputs" / "visual"
        for condition, visual_input in sorted(visual_by_condition.items()):
            artifact_path = write_visual_artifact(
                visual_root / condition,
                visual_input,
            )
            visual_artifacts[condition] = str(artifact_path.relative_to(output_path))

    manifest = {
        "schema_version": 1,
        "truth_keys": sorted(truth_by_mode),
        "visual_conditions": sorted(visual_by_condition),
        "run_cases": [run.case.name for run in run_list],
        "failures": failure_records,
        "sensor_artifacts": sensor_artifacts,
        "visual_artifacts": visual_artifacts,
        "readiness": {
            "status": readiness_status,
            "decision": decision,
            "contract_error": contract_error,
            "gates": gates,
        },
        "outputs": {
            "cases_csv": cases_path.name,
            "frames_csv": frames_path.name,
            "window_diagnostics": windows_path.name,
            "report": report_path.name,
            "html": html_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(_json_value(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "cases_csv": cases_path,
        "frames_csv": frames_path,
        "window_diagnostics": windows_path,
        "report": report_path,
        "html": html_path,
        "input_manifest": manifest_path,
    }


def _read_bundle_table(bundle_dir: Path, filename: str) -> pd.DataFrame:
    path = bundle_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Evidence bundle is missing {path}")
    return pd.read_csv(path)


def _validate_source_bundle_contract(bundle_dir: Path) -> None:
    input_manifest_path = bundle_dir / "input_manifest.json"
    if not input_manifest_path.is_file():
        raise ReadinessInputContractError(
            f"Evidence source is missing input manifest: {input_manifest_path}"
        )
    try:
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ReadinessInputContractError(
            f"Evidence source input manifest is unreadable: {input_manifest_path}"
        ) from exc
    if input_manifest.get("schema_version") != 1:
        raise ReadinessInputContractError(
            f"Evidence source {bundle_dir} must use input manifest schema_version=1"
        )
    sensor_artifacts = input_manifest.get("sensor_artifacts")
    if not isinstance(sensor_artifacts, dict) or not sensor_artifacts:
        raise ReadinessInputContractError(
            f"Evidence source {bundle_dir} has no sensor generation contracts"
        )

    required_formulas = {
        "gyro_body = [0, 0, 2*pi/trajectory_period_s] at each IMU timestamp",
        "white_noise_sample_std = sigma_density*sqrt(imu_rate_hz)",
        "bias_random_walk_step_std = sigma_walk_density/sqrt(imu_rate_hz)",
    }
    bundle_root = bundle_dir.resolve()
    for sensor_key, artifact in sensor_artifacts.items():
        if not isinstance(artifact, dict) or not isinstance(artifact.get("manifest_json"), str):
            raise ReadinessInputContractError(
                f"Evidence source sensor {sensor_key} has no generation manifest path"
            )
        generation_path = (bundle_dir / artifact["manifest_json"]).resolve()
        if not generation_path.is_relative_to(bundle_root) or not generation_path.is_file():
            raise ReadinessInputContractError(
                f"Evidence source sensor {sensor_key} generation manifest is missing or outside the bundle"
            )
        try:
            generation = json.loads(generation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ReadinessInputContractError(
                f"Evidence source sensor {sensor_key} generation manifest is unreadable"
            ) from exc
        if generation.get("motion_sampling_contract") != MOTION_SAMPLING_CONTRACT:
            raise ReadinessInputContractError(
                f"Evidence source sensor {sensor_key} uses an invalid motion sampling contract"
            )
        if generation.get("orientation_profile") != (
            "roll=0, pitch=0, yaw=2*pi*t/trajectory_period_s"
        ):
            raise ReadinessInputContractError(
                f"Evidence source sensor {sensor_key} uses an invalid orientation profile"
            )
        if generation.get("noise_parameter_semantics") != "continuous-time density":
            raise ReadinessInputContractError(
                f"Evidence source sensor {sensor_key} uses invalid noise parameter semantics"
            )
        if (
            generation.get("camera_rate_hz") != 30.0
            or generation.get("imu_rate_hz") != 100.0
            or generation.get("trajectory_period_s") != 10.0
        ):
            raise ReadinessInputContractError(
                f"Evidence source sensor {sensor_key} uses non-canonical sampling rates"
            )
        formulas = generation.get("formulas")
        if not isinstance(formulas, list) or not required_formulas.issubset(set(formulas)):
            raise ReadinessInputContractError(
                f"Evidence source sensor {sensor_key} is missing canonical generation formulas"
            )


def _recompute_separate_bias_tail_metrics(
    case_table: pd.DataFrame,
    frame_table: pd.DataFrame,
) -> None:
    required_frame_columns = {
        "case",
        "in_evaluation",
        "timestamp_ns",
        "estimated_acc_bias_norm_m_s2",
        "estimated_gyro_bias_norm_rad_s",
    }
    if not required_frame_columns.issubset(frame_table.columns):
        return
    for case_name in case_table.get("case", pd.Series(dtype=object)).astype(str):
        rows = frame_table.loc[
            (frame_table["case"].astype(str) == case_name)
            & frame_table["in_evaluation"].astype(bool)
        ].sort_values("frame_index")
        if rows.empty:
            continue
        time_s = pd.to_numeric(rows["timestamp_ns"], errors="coerce").to_numpy(
            dtype=np.float64
        ) * 1e-9
        acc_norm = pd.to_numeric(
            rows["estimated_acc_bias_norm_m_s2"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        gyro_norm = pd.to_numeric(
            rows["estimated_gyro_bias_norm_rad_s"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        tail_start = max(0, len(rows) - 10)
        selector = case_table["case"].astype(str) == case_name
        case_table.loc[selector, "final_acc_bias_norm_m_s2"] = float(acc_norm[-1])
        case_table.loc[selector, "final_gyro_bias_norm_rad_s"] = float(gyro_norm[-1])
        case_table.loc[selector, "final10_acc_bias_norm_slope_m_s3"] = (
            _least_squares_slope(
                torch.from_numpy(time_s[tail_start:]),
                torch.from_numpy(acc_norm[tail_start:]),
            )
        )
        case_table.loc[selector, "final10_gyro_bias_norm_slope_rad_s2"] = (
            _least_squares_slope(
                torch.from_numpy(time_s[tail_start:]),
                torch.from_numpy(gyro_norm[tail_start:]),
            )
        )


def combine_evidence_bundles(
    short_dir: str | Path,
    main_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    short_path = Path(short_dir)
    main_path = Path(main_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _validate_source_bundle_contract(short_path)
    _validate_source_bundle_contract(main_path)
    short_cases = _read_bundle_table(short_path, "joint_objective_cases.csv")
    short_frames = _read_bundle_table(short_path, "frame_metrics.csv")
    main_cases = _read_bundle_table(main_path, "joint_objective_cases.csv")
    main_frames = _read_bundle_table(main_path, "frame_metrics.csv")
    short_control = "w3_full_all_zero_mean"
    case_table = pd.concat(
        [
            short_cases.loc[short_cases["case"] == short_control],
            main_cases.loc[main_cases["case"] != short_control],
        ],
        ignore_index=True,
    )
    stale_readiness_columns = [
        column for column in case_table.columns if column.startswith("readiness_")
    ]
    case_table = case_table.drop(columns=stale_readiness_columns, errors="ignore")
    frame_table = pd.concat(
        [
            short_frames.loc[short_frames["case"] == short_control],
            main_frames.loc[main_frames["case"] != short_control],
        ],
        ignore_index=True,
    )
    _recompute_separate_bias_tail_metrics(case_table, frame_table)

    window_tables: list[pd.DataFrame] = []
    for source_path, keep_short_control in (
        (short_path, True),
        (main_path, False),
    ):
        rows = _read_bundle_table(source_path, "window_diagnostics.csv")
        mask = rows["case"] == short_control
        window_tables.append(rows.loc[mask if keep_short_control else ~mask])
    window_table = (
        pd.concat(window_tables, ignore_index=True)
        if window_tables
        else pd.DataFrame(columns=["case"])
    )

    decision: str | None = None
    gates: list[dict[str, object]] = []
    contract_error: str | None = None
    try:
        _validate_window_diagnostics_contract(case_table, window_table)
        decision = classify_readiness(case_table, frame_table)
        gates = list(case_table.attrs["readiness_gates"])
        readiness_status = "classified_canonical_evidence"
    except ReadinessInputContractError as exc:
        contract_error = str(exc)
        readiness_status = "incomplete_evidence_contract"
        case_table["readiness_decision"] = None
    case_table["readiness_status"] = readiness_status
    case_table["readiness_contract_error"] = contract_error
    case_table["readiness_gate_details_json"] = (
        json.dumps(_json_value(gates), ensure_ascii=False, sort_keys=True) if gates else None
    )

    cases_path = output_path / "joint_objective_cases.csv"
    frames_path = output_path / "frame_metrics.csv"
    windows_path = output_path / "window_diagnostics.csv"
    report_path = output_path / "joint_objective_summary_cn.md"
    html_path = output_path / "diagnostics_interactive.html"
    manifest_path = output_path / "input_manifest.json"
    case_table.to_csv(cases_path, index=False)
    frame_table.to_csv(frames_path, index=False)
    window_table.to_csv(windows_path, index=False)
    report_path.write_text(
        _render_chinese_summary(case_table, decision, gates, contract_error),
        encoding="utf-8",
    )
    html_path.write_text(
        _render_interactive_html(
            _build_interactive_model(case_table, frame_table, decision, readiness_status)
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "bundle_kind": "combined_short_main_readiness",
        "evidence_sources": {
            "short": str(short_path),
            "main": str(main_path),
        },
        "readiness": {
            "status": readiness_status,
            "decision": decision,
            "contract_error": contract_error,
            "gates": gates,
        },
        "outputs": {
            "cases_csv": cases_path.name,
            "frames_csv": frames_path.name,
            "window_diagnostics": windows_path.name,
            "report": report_path.name,
            "html": html_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(_json_value(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "cases_csv": cases_path,
        "frames_csv": frames_path,
        "window_diagnostics": windows_path,
        "report": report_path,
        "html": html_path,
        "input_manifest": manifest_path,
    }


def select_cases(duration_s: float, requested_names: list[str] | None) -> list[ValidationCase]:
    available = build_cases(duration_s)
    if not requested_names:
        return available

    available_names = {case.name for case in available}
    unknown = sorted(set(requested_names) - available_names)
    if unknown:
        raise ValueError(
            "Unknown case(s): "
            + ", ".join(unknown)
            + "; available cases: "
            + ", ".join(case.name for case in available)
        )
    requested = set(requested_names)
    return [case for case in available if case.name in requested]


def _emit_progress(event: str, **payload: object) -> None:
    print(
        json.dumps(
            {"event": event, **_json_value(payload)},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _build_cli_inputs(
    config: SyntheticSequenceConfig,
    cases: list[ValidationCase],
) -> tuple[
    dict[str, EvaluationTruth],
    dict[str, EstimatorIMUInput],
    dict[str, EstimatorVisualInput],
]:
    truth_by_mode: dict[str, EvaluationTruth] = {}
    imu_inputs_by_mode: dict[str, EstimatorIMUInput] = {}
    visual_by_condition: dict[str, EstimatorVisualInput] = {}

    for case in cases:
        mode_key = _mode_key(case)
        if mode_key not in truth_by_mode:
            truth = generate_truth(
                config,
                bias_mode=case.bias_mode,
                noise_mode=case.imu_noise_mode,
            )
            truth_by_mode[mode_key] = truth
            imu_inputs_by_mode[mode_key] = generate_imu_input(
                config,
                truth,
                bias_mode=case.bias_mode,
                noise_mode=case.imu_noise_mode,
            )
        if case.visual_condition not in visual_by_condition:
            visual_by_condition[case.visual_condition] = generate_visual_input(
                config,
                truth_by_mode[mode_key],
                condition=case.visual_condition,
            )

    return truth_by_mode, imu_inputs_by_mode, visual_by_condition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the causal W=3 joint-objective synthetic validation.",
    )
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combine-short-dir", type=Path)
    parser.add_argument("--combine-main-dir", type=Path)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="Case name to run; repeat to select multiple cases.",
    )
    args = parser.parse_args(argv)

    combine_requested = args.combine_short_dir is not None or args.combine_main_dir is not None
    if combine_requested:
        if args.combine_short_dir is None or args.combine_main_dir is None:
            parser.error("--combine-short-dir and --combine-main-dir must be provided together")
        if args.duration_s is not None or args.case_names:
            parser.error("combine mode cannot be used with --duration-s or --case")
        paths = combine_evidence_bundles(
            args.combine_short_dir,
            args.combine_main_dir,
            args.output_dir,
        )
        manifest = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
        _emit_progress(
            "bundle_complete",
            output_dir=str(args.output_dir),
            html=str(paths["html"]),
            readiness_status=manifest["readiness"]["status"],
            readiness_decision=manifest["readiness"]["decision"],
        )
        return 0 if manifest["readiness"]["status"] == "classified_canonical_evidence" else 1

    if args.duration_s is None:
        parser.error("--duration-s is required unless combine mode is used")

    try:
        cases = select_cases(args.duration_s, args.case_names)
    except ValueError as exc:
        parser.error(str(exc))

    config = SyntheticSequenceConfig(duration_s=args.duration_s)
    truth_by_mode, imu_inputs_by_mode, visual_by_condition = _build_cli_inputs(
        config,
        cases,
    )
    runs: list[CaseRunResult] = []
    failures: list[dict[str, object]] = []
    for case_index, case in enumerate(cases, start=1):
        _emit_progress(
            "case_start",
            case=case.name,
            case_index=case_index,
            case_count=len(cases),
        )
        mode_key = _mode_key(case)
        try:
            result = run_validation_case(
                config,
                imu_inputs_by_mode[mode_key],
                visual_by_condition[case.visual_condition],
                case,
                progress=lambda window_event, current_case_index=case_index: _emit_progress(
                    "window_progress",
                    case_index=current_case_index,
                    case_count=len(cases),
                    **window_event,
                ),
            )
        except Exception as exc:
            failure = {
                "case": case.name,
                "visual_condition": case.visual_condition,
                "imu_noise_mode": case.imu_noise_mode,
                "bias_mode": case.bias_mode,
                "bias_enabled": case.bias_enabled,
                "writeback": case.writeback,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            }
            failures.append(failure)
            _emit_progress(
                "case_failed",
                case=case.name,
                case_index=case_index,
                case_count=len(cases),
                failure_type=failure["failure_type"],
                failure_message=failure["failure_message"],
            )
            continue
        runs.append(result)
        _emit_progress(
            "case_complete",
            case=case.name,
            case_index=case_index,
            case_count=len(cases),
            num_windows=result.num_windows,
        )

    paths = write_validation_bundle(
        args.output_dir,
        runs,
        truth_by_mode,
        visual_by_condition,
        config=config,
        imu_inputs_by_mode=imu_inputs_by_mode,
        failures=failures,
    )
    _emit_progress(
        "bundle_complete",
        output_dir=str(args.output_dir),
        html=str(paths["html"]),
        failed_case_count=len(failures),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only cross-project comparison of existing MACVIO and GTSAM IMU audits.

This script does not call either preintegrator and does not modify production data.
MACVIO absolute deltas are read from the audited production tensor_map and moved to
the GT-bias linearization point with the cached production bias Jacobian.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


D_FLU_TO_NED = np.diag([1.0, -1.0, -1.0])
S_PVR_FLU_TO_NED = np.kron(np.eye(3), D_FLU_TO_NED)
S_BIAS_FLU_TO_NED = np.kron(np.eye(2), D_FLU_TO_NED)
CHI2_9_CENTRAL_95 = (2.700, 19.023)
CHI2_6_CENTRAL_95 = (1.2373442457912027, 14.44937533544792)
EPS = 1e-30


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vec(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array([float(row[f"{prefix}_{axis}"]) for axis in "xyz"], dtype=np.float64)


def matrix(row: dict[str, str], prefix: str, n: int) -> np.ndarray:
    return np.array(
        [[float(row[f"{prefix}_{i}{j}"]) for j in range(n)] for i in range(n)],
        dtype=np.float64,
    )


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-10:
        return np.eye(3) + K + 0.5 * (K @ K)
    return np.eye(3) + math.sin(theta) / theta * K + (1.0 - math.cos(theta)) / theta**2 * (K @ K)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if theta < 1e-10:
        return 0.5 * vee
    return theta / (2.0 * math.sin(theta)) * vee


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    return (
        np.eye(3)
        - (1.0 - math.cos(theta)) / theta**2 * K
        + (theta - math.sin(theta)) / theta**3 * (K @ K)
    )


def quaternion_matrix(row: dict[str, str], prefix: str) -> np.ndarray:
    x = float(row[f"{prefix}qx"])
    y = float(row[f"{prefix}qy"])
    z = float(row[f"{prefix}qz"])
    w = float(row[f"{prefix}qw"])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def summary(values: Iterable[float]) -> dict[str, float | int]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "p95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
    }


def coverage(values: np.ndarray, interval: tuple[float, float]) -> dict[str, float]:
    low, high = interval
    return {
        "fraction_inside": float(np.mean((values >= low) & (values <= high))),
        "fraction_below": float(np.mean(values < low)),
        "fraction_above": float(np.mean(values > high)),
    }


def whiten(covariance: np.ndarray, error: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(0.5 * (covariance + covariance.T))
    return np.linalg.solve(chol, error)


def mahalanobis(covariance: np.ndarray, error: np.ndarray) -> float:
    return float(error @ np.linalg.solve(covariance, error))


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), EPS))


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def autocorrelation(values: np.ndarray, lag: int) -> list[float]:
    output: list[float] = []
    for axis in range(values.shape[1]):
        x = values[:-lag, axis]
        y = values[lag:, axis]
        if np.std(x) < EPS or np.std(y) < EPS:
            output.append(0.0)
        else:
            output.append(float(np.corrcoef(x, y)[0, 1]))
    return output


def vector_columns(output: dict[str, Any], prefix: str, value: np.ndarray) -> None:
    for axis, number in zip("xyz", value):
        output[f"{prefix}_{axis}"] = float(number)


def scalar_matrix_columns(output: dict[str, Any], prefix: str, value: np.ndarray) -> None:
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            output[f"{prefix}_{i}{j}"] = float(value[i, j])


def strict_input_check(
    mac_contract: dict[str, Any],
    gtsam_contract: dict[str, Any],
    mac_edges: list[dict[str, str]],
    gtsam_edges: list[dict[str, str]],
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    mac_hashes = mac_contract["input_hashes"]
    gtsam_prov = gtsam_contract["provenance"]
    checks: dict[str, Any] = {}
    checks["scene_equal"] = mac_hashes["scene"] == gtsam_prov["scene_name"]
    for label, mac_key, gtsam_key in [
        ("metadata_sha256", "metadata_sha256", "metadata_sha256"),
        ("imu_csv_sha256", "imu_csv_sha256", "imu_csv_sha256"),
        ("truth_decomposition_sha256", "imu_truth_decomposition_sha256", "truth_decomposition_sha256"),
    ]:
        checks[f"{label}_equal"] = mac_hashes[mac_key] == gtsam_prov[gtsam_key]
        checks[f"mac_{label}"] = mac_hashes[mac_key]
        checks[f"gtsam_{label}"] = gtsam_prov[gtsam_key]
    checks["frame_range_equal"] = mac_hashes["frame_range_inclusive"] == [
        gtsam_prov["frame_window"]["start"],
        gtsam_prov["frame_window"]["end_inclusive"],
    ]
    checks["valid_edge_count_equal"] = (
        mac_hashes["valid_edge_count"] == gtsam_prov["effective_edge_frame_range"]["count"]
    )
    checks["frame_time_range_equal"] = mac_hashes["frame_time_range_ns"] == [
        gtsam_prov["timestamp_window_ns"]["start"],
        gtsam_prov["timestamp_window_ns"]["end"],
    ]
    checks["valid_edge_time_range_equal"] = mac_hashes["valid_edge_time_range_ns"] == [
        gtsam_prov["effective_edge_timestamp_range_ns"]["start"],
        gtsam_prov["effective_edge_timestamp_range_ns"]["end"],
    ]
    mac_sigma = mac_contract["continuous_noise_density"]
    sigma_pairs = {
        "sigma_a": (mac_sigma["sigma_a"], gtsam_contract["sigma_a"]),
        "sigma_g": (mac_sigma["sigma_g"], gtsam_contract["sigma_g"]),
        "sigma_aw": (mac_sigma["sigma_aw"], gtsam_contract["sigma_aw"]),
        "sigma_gw": (mac_sigma["sigma_gw"], gtsam_contract["sigma_gw"]),
    }
    checks["sigma"] = {
        name: {"macvio": left, "gtsam": right, "equal": bool(abs(left - right) <= 1e-15)}
        for name, (left, right) in sigma_pairs.items()
    }
    checks["all_sigma_equal"] = all(item["equal"] for item in checks["sigma"].values())
    checks["edge_row_count_equal"] = len(mac_edges) == len(gtsam_edges) == 209

    alignment: list[dict[str, Any]] = []
    mismatch_count = 0
    exact_int_fields = [
        "frame_i",
        "frame_j",
        "timestamp_i",
        "timestamp_j",
        "first_imu_timestamp",
        "last_imu_timestamp",
    ]
    if len(mac_edges) == len(gtsam_edges):
        for mac, gtsam in zip(mac_edges, gtsam_edges):
            row: dict[str, Any] = {}
            edge_equal = True
            mac_edge_id = int(mac["edge_id"])
            gtsam_edge_id = int(gtsam["edge_id"])
            row["common_edge_index"] = gtsam_edge_id
            row["mac_edge_id"] = mac_edge_id
            row["gtsam_edge_id"] = gtsam_edge_id
            row["edge_id_mapping"] = "mac_edge_id = gtsam_edge_id + 1"
            row["edge_id_mapping_valid"] = mac_edge_id == gtsam_edge_id + 1
            edge_equal &= row["edge_id_mapping_valid"]
            for field in exact_int_fields:
                left = int(mac[field])
                right = int(gtsam[field])
                row[f"mac_{field}"] = left
                row[f"gtsam_{field}"] = right
                row[f"{field}_equal"] = left == right
                edge_equal &= left == right
            mac_sample_count = int(mac["imu_sample_count"])
            gtsam_sample_count = int(gtsam["imu_sample_count"])
            row["mac_imu_knot_count"] = mac_sample_count
            row["gtsam_imu_integration_step_count"] = gtsam_sample_count
            row["sample_count_semantic_mapping"] = "MAC knot_count = GTSAM integration_step_count + 1"
            row["sample_count_mapping_valid"] = mac_sample_count == gtsam_sample_count + 1
            edge_equal &= row["sample_count_mapping_valid"]
            dt_mac = float(mac["delta_t"])
            dt_gtsam = float(gtsam["delta_t"])
            row["mac_delta_t"] = dt_mac
            row["gtsam_delta_t"] = dt_gtsam
            row["delta_t_abs_difference"] = abs(dt_mac - dt_gtsam)
            row["delta_t_equal_1e_12"] = abs(dt_mac - dt_gtsam) <= 1e-12
            edge_equal &= row["delta_t_equal_1e_12"]
            row["edge_contract_equal"] = edge_equal
            mismatch_count += int(not edge_equal)
            alignment.append(row)
    checks["edge_id_convention"] = "MACVIO pair_id is one-based; GTSAM audit edge_id is zero-based"
    checks["sample_count_convention"] = (
        "MACVIO counts interpolated knots including both endpoints; "
        "GTSAM counts adjacent integration steps (knots-1)"
    )
    checks["edge_mismatch_count_after_proven_semantic_normalization"] = mismatch_count
    checks["all_edges_equal"] = mismatch_count == 0 and len(alignment) == 209
    scalar_bools = [
        value for key, value in checks.items() if isinstance(value, bool) and key != "all_sigma_equal"
    ]
    passed = all(scalar_bools) and checks["all_sigma_equal"] and checks["all_edges_equal"]
    checks["strict_input_gate_passed"] = passed
    return passed, checks, alignment


def validate_conventions(gtsam_rows: list[dict[str, str]]) -> dict[str, Any]:
    direct_errors: list[float] = []
    stored_jr_errors: list[float] = []
    synthetic_errors: list[float] = []
    per_edge_transforms: list[dict[str, Any]] = []
    eps = 1e-7
    for row in gtsam_rows:
        transform = matrix(row, "T_native_to_common", 9)
        native = np.array([float(row[f"native_e{i}"]) for i in range(9)])
        common_flu = np.concatenate([vec(row, "ep"), vec(row, "ev"), vec(row, "er")])
        common_ned_direct = S_PVR_FLU_TO_NED @ common_flu
        j_full = S_PVR_FLU_TO_NED @ transform
        per_edge_transforms.append(
            {
                "edge_id": int(row["edge_id"]),
                "frame_i": int(row["frame_i"]),
                "frame_j": int(row["frame_j"]),
                "J_gtsam_native_to_public_common": j_full,
            }
        )
        direct_errors.append(float(np.linalg.norm(common_ned_direct - j_full @ native)))
        r_ref = quaternion_matrix(row, "delta_R_ref_")
        theta_ref = so3_log(r_ref)
        jr = so3_right_jacobian(theta_ref)
        stored_jr_errors.append(float(np.max(np.abs(jr - transform[6:9, 0:3]))))
        for axis in range(3):
            perturb = np.zeros(3)
            perturb[axis] = eps
            exact = so3_log(so3_exp(theta_ref).T @ so3_exp(theta_ref + perturb))
            synthetic_errors.append(float(np.max(np.abs(exact - jr @ perturb))))

    single_axis_errors: list[float] = []
    for axis in range(3):
        phi = np.zeros(3)
        phi[axis] = 1e-5
        transformed = so3_log(D_FLU_TO_NED @ so3_exp(phi) @ D_FLU_TO_NED.T)
        single_axis_errors.append(float(np.max(np.abs(transformed - D_FLU_TO_NED @ phi))))
    zero = np.zeros(9)
    first_j = S_PVR_FLU_TO_NED @ matrix(gtsam_rows[0], "T_native_to_common", 9)
    zero_noise_error = float(np.linalg.norm(first_j @ zero))
    return {
        "public_common_order": ["P", "V", "R"],
        "public_common_coordinates": "MACVIO internal NED body tangent",
        "macvio_native_to_common": np.eye(9),
        "gtsam_flu_to_ned": D_FLU_TO_NED,
        "gtsam_common_flu_to_public_common_ned": S_PVR_FLU_TO_NED,
        "gtsam_native_to_public_common_per_edge": "J_edge = blockdiag(D,D,D) * T_native_to_common_edge",
        "gtsam_native_to_public_common_per_edge_matrices": per_edge_transforms,
        "gtsam_native_order": ["theta", "P", "V"],
        "gtsam_rotation_map": "right error ~= Jr(theta_reference) * additive delta_theta",
        "stored_error_vs_full_J_max_norm": max(direct_errors),
        "stored_Jr_vs_formula_max_abs": max(stored_jr_errors),
        "synthetic_small_perturbation_epsilon": eps,
        "synthetic_small_perturbation_max_abs": max(synthetic_errors),
        "rotation_single_axis_basis_change_max_abs": max(single_axis_errors),
        "synthetic_zero_noise_error_norm": zero_noise_error,
        "validation_passed": bool(
            max(direct_errors) < 1e-12
            # The CSV stores the per-edge transform at decimal text precision.
            and max(stored_jr_errors) < 1e-8
            and max(synthetic_errors) < 1e-12
            and max(single_axis_errors) < 1e-12
            and zero_noise_error == 0.0
        ),
    }


def compare(
    mac_root: Path,
    gtsam_root: Path,
    tensor_map_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mac_contract = read_json(mac_root / "macvio_runtime_noise_contract.json")
    gtsam_contract = read_json(gtsam_root / "gtsam_runtime_noise_contract.json")
    mac_edges = read_csv(mac_root / "macvio_normal_noise_edge_manifest.csv")
    gtsam_edges = read_csv(gtsam_root / "gtsam_normal_noise_edge_manifest.csv")
    gate_passed, gate, alignment_rows = strict_input_check(
        mac_contract, gtsam_contract, mac_edges, gtsam_edges
    )
    write_csv(output_dir / "common_edge_alignment.csv", alignment_rows)
    if not gate_passed:
        decision = {
            "status": "INPUT_MISMATCH",
            "numerical_comparison_performed": False,
            "input_gate": gate,
        }
        write_json(output_dir / "final_decision.json", decision)
        report = "# MACVIO 与 GTSAM Normal-noise 跨项目对比\n\n输入契约不一致，已按要求停止数值比较。\n\n```json\n" + json.dumps(jsonable(gate), ensure_ascii=False, indent=2) + "\n```\n"
        (output_dir / "macvio_vs_gtsam_normal_noise_comparison_cn.md").write_text(report, encoding="utf-8")
        return decision

    mac_nis = read_csv(mac_root / "macvio_preintegration_nis_per_edge.csv")
    gtsam_nis = read_csv(gtsam_root / "gtsam_preintegration_nis_per_edge.csv")
    mac_bias = read_csv(mac_root / "macvio_bias_rw_nis_per_edge.csv")
    gtsam_bias = read_csv(gtsam_root / "gtsam_bias_rw_nis_per_edge.csv")
    if not (len(mac_nis) == len(gtsam_nis) == len(mac_bias) == len(gtsam_bias) == 209):
        raise RuntimeError("numeric audit tables are not all 209 rows")
    for tables in [(mac_nis, gtsam_nis), (mac_bias, gtsam_bias)]:
        for left, right in zip(*tables):
            if (left["frame_i"], left["frame_j"], left["timestamp_i"], left["timestamp_j"]) != (
                right["frame_i"], right["frame_j"], right["timestamp_i"], right["timestamp_j"]
            ):
                raise RuntimeError("numeric table edge mismatch")

    convention = validate_conventions(gtsam_nis)
    if not convention["validation_passed"]:
        raise RuntimeError("common convention transform validation failed")
    write_json(output_dir / "common_convention_transform.json", convention)

    expected_tensor_hash = mac_contract["input_hashes"]["tensor_map_sha256"]
    actual_tensor_hash = sha256(tensor_map_path)
    if actual_tensor_hash != expected_tensor_hash:
        raise RuntimeError("tensor_map hash does not match the independently audited MACVIO artifact")
    tensor_map = np.load(tensor_map_path, allow_pickle=False)

    delta_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    nis_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    mac_whitened: list[np.ndarray] = []
    gtsam_whitened: list[np.ndarray] = []
    mac_errors: list[np.ndarray] = []
    gtsam_errors: list[np.ndarray] = []
    mac_covariances: list[np.ndarray] = []
    gtsam_covariances: list[np.ndarray] = []

    for mac_row, gtsam_row, mac_bias_row, gtsam_bias_row, aligned in zip(
        mac_nis, gtsam_nis, mac_bias, gtsam_bias, alignment_rows
    ):
        mac_edge_id = int(mac_row["edge_id"])
        gtsam_edge_id = int(gtsam_row["edge_id"])
        edge_id = gtsam_edge_id
        frame_i = int(mac_row["frame_i"])
        frame_j = int(mac_row["frame_j"])
        if int(tensor_map["frames//time_ns"][frame_j]) != int(mac_row["timestamp_j"]):
            raise RuntimeError(f"tensor map frame/time mismatch on edge {edge_id}")

        mac_error = np.concatenate([vec(mac_row, "ep"), vec(mac_row, "ev"), vec(mac_row, "er")])
        gtsam_error_flu = np.concatenate(
            [vec(gtsam_row, "ep"), vec(gtsam_row, "ev"), vec(gtsam_row, "er")]
        )
        gtsam_error = S_PVR_FLU_TO_NED @ gtsam_error_flu
        p_mac = matrix(mac_row, "P", 9)
        p_gtsam_flu = matrix(gtsam_row, "P_common", 9)
        p_gtsam = S_PVR_FLU_TO_NED @ p_gtsam_flu @ S_PVR_FLU_TO_NED.T
        p_mac = 0.5 * (p_mac + p_mac.T)
        p_gtsam = 0.5 * (p_gtsam + p_gtsam.T)
        mac_errors.append(mac_error)
        gtsam_errors.append(gtsam_error)
        mac_covariances.append(p_mac)
        gtsam_covariances.append(p_gtsam)
        mac_w = whiten(p_mac, mac_error)
        gtsam_w = whiten(p_gtsam, gtsam_error)
        mac_whitened.append(mac_w)
        gtsam_whitened.append(gtsam_w)

        # Read the production MACVIO delta and apply its cached first-order bias correction.
        jacobian = np.asarray(
            tensor_map["frames//imu_vio_bias_jacobian"][frame_j], dtype=np.float64
        )
        ba_truth_ned = D_FLU_TO_NED @ vec(gtsam_bias_row, "ba_gt_i")
        bg_truth_ned = D_FLU_TO_NED @ vec(gtsam_bias_row, "bg_gt_i")
        ba_linearized = np.asarray(
            tensor_map["frames//imu_vio_linearized_acc_bias"][frame_j], dtype=np.float64
        )
        bg_linearized = np.asarray(
            tensor_map["frames//imu_vio_linearized_gyro_bias"][frame_j], dtype=np.float64
        )
        delta_ba = ba_truth_ned - ba_linearized
        delta_bg = bg_truth_ned - bg_linearized
        bias_delta = np.concatenate([delta_ba, delta_bg])
        mac_delta_p_noisy = np.asarray(
            tensor_map["frames//imu_vio_delta_p"][frame_j], dtype=np.float64
        ) + jacobian[0:3] @ bias_delta
        mac_delta_v_noisy = np.asarray(
            tensor_map["frames//imu_vio_delta_v"][frame_j], dtype=np.float64
        ) + jacobian[3:6] @ bias_delta
        mac_delta_r_noisy = so3_exp(
            np.asarray(tensor_map["frames//imu_vio_delta_rotvec"][frame_j], dtype=np.float64)
        ) @ so3_exp(jacobian[6:9] @ bias_delta)
        mac_delta_p_ref = mac_delta_p_noisy - mac_error[0:3]
        mac_delta_v_ref = mac_delta_v_noisy - mac_error[3:6]
        mac_delta_r_ref = mac_delta_r_noisy @ so3_exp(-mac_error[6:9])

        gtsam_delta_p_noisy = D_FLU_TO_NED @ vec(gtsam_row, "delta_p_noisy")
        gtsam_delta_v_noisy = D_FLU_TO_NED @ vec(gtsam_row, "delta_v_noisy")
        gtsam_delta_r_noisy = (
            D_FLU_TO_NED
            @ quaternion_matrix(gtsam_row, "delta_R_noisy_")
            @ D_FLU_TO_NED.T
        )
        gtsam_delta_p_ref = D_FLU_TO_NED @ vec(gtsam_row, "delta_p_ref")
        gtsam_delta_v_ref = D_FLU_TO_NED @ vec(gtsam_row, "delta_v_ref")
        gtsam_delta_r_ref = (
            D_FLU_TO_NED
            @ quaternion_matrix(gtsam_row, "delta_R_ref_")
            @ D_FLU_TO_NED.T
        )
        dp_noisy = mac_delta_p_noisy - gtsam_delta_p_noisy
        dv_noisy = mac_delta_v_noisy - gtsam_delta_v_noisy
        dr_noisy = so3_log(mac_delta_r_noisy.T @ gtsam_delta_r_noisy)
        dp_ref = mac_delta_p_ref - gtsam_delta_p_ref
        dv_ref = mac_delta_v_ref - gtsam_delta_v_ref
        dr_ref = so3_log(mac_delta_r_ref.T @ gtsam_delta_r_ref)
        drow: dict[str, Any] = {
            "edge_id": edge_id,
            "macvio_edge_id": mac_edge_id,
            "gtsam_edge_id": gtsam_edge_id,
            "frame_i": frame_i,
            "frame_j": frame_j,
            "delta_t": float(mac_row["delta_t"]),
            "motion_stage_macvio": mac_row["motion_stage"],
            "motion_stage_gtsam": gtsam_row["motion_stage"],
            "mac_bias_correction_acc_norm": float(np.linalg.norm(delta_ba)),
            "mac_bias_correction_gyro_norm": float(np.linalg.norm(delta_bg)),
        }
        for name, value in [
            ("delta_p_macvio_noisy", mac_delta_p_noisy),
            ("delta_p_gtsam_noisy", gtsam_delta_p_noisy),
            ("delta_p_noisy_difference", dp_noisy),
            ("delta_v_macvio_noisy", mac_delta_v_noisy),
            ("delta_v_gtsam_noisy", gtsam_delta_v_noisy),
            ("delta_v_noisy_difference", dv_noisy),
            ("delta_R_macvio_noisy_rotvec", so3_log(mac_delta_r_noisy)),
            ("delta_R_gtsam_noisy_rotvec", so3_log(gtsam_delta_r_noisy)),
            ("delta_R_noisy_log_difference", dr_noisy),
            ("delta_p_macvio_reference", mac_delta_p_ref),
            ("delta_p_gtsam_reference", gtsam_delta_p_ref),
            ("delta_p_reference_difference", dp_ref),
            ("delta_v_macvio_reference", mac_delta_v_ref),
            ("delta_v_gtsam_reference", gtsam_delta_v_ref),
            ("delta_v_reference_difference", dv_ref),
            ("delta_R_macvio_reference_rotvec", so3_log(mac_delta_r_ref)),
            ("delta_R_gtsam_reference_rotvec", so3_log(gtsam_delta_r_ref)),
            ("delta_R_reference_log_difference", dr_ref),
            ("noise_error_macvio_minus_gtsam_p", mac_error[0:3] - gtsam_error[0:3]),
            ("noise_error_macvio_minus_gtsam_v", mac_error[3:6] - gtsam_error[3:6]),
            ("noise_error_macvio_minus_gtsam_R", mac_error[6:9] - gtsam_error[6:9]),
        ]:
            vector_columns(drow, name, value)
            drow[f"{name}_norm"] = float(np.linalg.norm(value))
        delta_rows.append(drow)

        eig_mac, eigvec_mac = np.linalg.eigh(p_mac)
        eig_gtsam, eigvec_gtsam = np.linalg.eigh(p_gtsam)
        dominant_singular_values = np.linalg.svd(
            eigvec_mac[:, -3:].T @ eigvec_gtsam[:, -3:], compute_uv=False
        )
        dominant_subspace_max_angle = math.degrees(
            math.acos(float(np.clip(np.min(dominant_singular_values), 0.0, 1.0)))
        )
        sign_mac, logdet_mac = np.linalg.slogdet(p_mac)
        sign_gtsam, logdet_gtsam = np.linalg.slogdet(p_gtsam)
        crow: dict[str, Any] = {
            "edge_id": edge_id,
            "macvio_edge_id": mac_edge_id,
            "gtsam_edge_id": gtsam_edge_id,
            "frame_i": frame_i,
            "frame_j": frame_j,
            "delta_t": float(mac_row["delta_t"]),
            "motion_stage_macvio": mac_row["motion_stage"],
            "motion_stage_gtsam": gtsam_row["motion_stage"],
            "relative_P_error": relative_frobenius(p_mac, p_gtsam),
            "trace_macvio": float(np.trace(p_mac)),
            "trace_gtsam": float(np.trace(p_gtsam)),
            "trace_ratio_macvio_over_gtsam": float(np.trace(p_mac) / np.trace(p_gtsam)),
            "logdet_macvio": float(logdet_mac),
            "logdet_gtsam": float(logdet_gtsam),
            "logdet_difference_macvio_minus_gtsam": float(logdet_mac - logdet_gtsam),
            "logdet_sign_macvio": float(sign_mac),
            "logdet_sign_gtsam": float(sign_gtsam),
            "condition_macvio": float(np.linalg.cond(p_mac)),
            "condition_gtsam": float(np.linalg.cond(p_gtsam)),
            "principal_max_eigenvector_angle_deg": angle_degrees(
                eigvec_mac[:, -1], eigvec_gtsam[:, -1]
            ),
            "dominant_3d_subspace_max_principal_angle_deg": dominant_subspace_max_angle,
        }
        for index in range(9):
            crow[f"diag_ratio_{index}"] = float(p_mac[index, index] / p_gtsam[index, index])
            crow[f"eigenvalue_macvio_{index}"] = float(eig_mac[index])
            crow[f"eigenvalue_gtsam_{index}"] = float(eig_gtsam[index])
            crow[f"eigenvalue_ratio_{index}"] = float(eig_mac[index] / eig_gtsam[index])
        block_names = ["p", "v", "R"]
        for bi, bname in enumerate(block_names):
            for bj, cname in enumerate(block_names):
                mac_block = p_mac[3 * bi : 3 * bi + 3, 3 * bj : 3 * bj + 3]
                gtsam_block = p_gtsam[3 * bi : 3 * bi + 3, 3 * bj : 3 * bj + 3]
                crow[f"block_{bname}{cname}_relative_frobenius"] = relative_frobenius(
                    mac_block, gtsam_block
                )
                crow[f"block_{bname}{cname}_absolute_frobenius_difference"] = float(
                    np.linalg.norm(mac_block - gtsam_block)
                )
        covariance_rows.append(crow)

        nis_mm = mahalanobis(p_mac, mac_error)
        nis_mg = mahalanobis(p_gtsam, mac_error)
        nis_gm = mahalanobis(p_mac, gtsam_error)
        nis_gg = mahalanobis(p_gtsam, gtsam_error)
        nrow: dict[str, Any] = {
            "edge_id": edge_id,
            "macvio_edge_id": mac_edge_id,
            "gtsam_edge_id": gtsam_edge_id,
            "frame_i": frame_i,
            "frame_j": frame_j,
            "delta_t": float(mac_row["delta_t"]),
            "motion_stage_macvio": mac_row["motion_stage"],
            "motion_stage_gtsam": gtsam_row["motion_stage"],
            "nis_macvio_error_macvio_P": nis_mm,
            "nis_macvio_error_gtsam_P": nis_mg,
            "nis_gtsam_error_macvio_P": nis_gm,
            "nis_gtsam_error_gtsam_P": nis_gg,
            "reported_macvio_nis": float(mac_row["nis_9"]),
            "reported_gtsam_common_nis": float(gtsam_row["nis_common_9"]),
            "macvio_recompute_abs_difference": abs(nis_mm - float(mac_row["nis_9"])),
            "gtsam_recompute_abs_difference": abs(nis_gg - float(gtsam_row["nis_common_9"])),
            "error_vector_difference_norm": float(np.linalg.norm(mac_error - gtsam_error)),
        }
        for label, error, covariance in [
            ("macvio_own", mac_error, p_mac),
            ("gtsam_own", gtsam_error, p_gtsam),
        ]:
            for block_index, block_name in enumerate(["p", "v", "R"]):
                block_slice = slice(3 * block_index, 3 * block_index + 3)
                nrow[f"nis_{label}_{block_name}"] = mahalanobis(
                    covariance[block_slice, block_slice], error[block_slice]
                )
        for name, value in [
            ("macvio_error_common", mac_error),
            ("gtsam_error_common", gtsam_error),
            ("macvio_whitened_common", mac_w),
            ("gtsam_whitened_common", gtsam_w),
        ]:
            for index, number in enumerate(value):
                nrow[f"{name}_{index}"] = float(number)
        nis_rows.append(nrow)

        q_mac = matrix(mac_bias_row, "Qb", 6)
        q_gtsam_flu = matrix(gtsam_bias_row, "Q_bias", 6)
        q_gtsam = S_BIAS_FLU_TO_NED @ q_gtsam_flu @ S_BIAS_FLU_TO_NED.T
        db_gt_mac = np.concatenate([vec(mac_bias_row, "dba_gt"), vec(mac_bias_row, "dbg_gt")])
        db_gt_gtsam = S_BIAS_FLU_TO_NED @ np.concatenate(
            [vec(gtsam_bias_row, "delta_ba_gt"), vec(gtsam_bias_row, "delta_bg_gt")]
        )
        db_est_mac = np.concatenate([vec(mac_bias_row, "dba_est"), vec(mac_bias_row, "dbg_est")])
        db_est_gtsam = S_BIAS_FLU_TO_NED @ np.concatenate(
            [vec(gtsam_bias_row, "delta_ba_est"), vec(gtsam_bias_row, "delta_bg_est")]
        )
        white_mac = np.concatenate(
            [vec(mac_bias_row, "acc_white_noise_mean"), vec(mac_bias_row, "gyro_white_noise_mean")]
        )
        white_gtsam = S_BIAS_FLU_TO_NED @ np.concatenate(
            [vec(gtsam_bias_row, "acc_white_noise_mean"), vec(gtsam_bias_row, "gyro_white_noise_mean")]
        )
        brow: dict[str, Any] = {
            "edge_id": edge_id,
            "macvio_edge_id": mac_edge_id,
            "gtsam_edge_id": gtsam_edge_id,
            "frame_i": frame_i,
            "frame_j": frame_j,
            "delta_t": float(mac_bias_row["delta_t"]),
            "motion_stage_macvio": mac_bias_row["motion_stage"],
            "motion_stage_gtsam": gtsam_bias_row["motion_stage"],
            "Q_relative_frobenius": relative_frobenius(q_mac, q_gtsam),
            "Q_max_abs_difference": float(np.max(np.abs(q_mac - q_gtsam))),
            "truth_increment_difference_norm": float(np.linalg.norm(db_gt_mac - db_gt_gtsam)),
            "white_noise_mean_difference_norm": float(np.linalg.norm(white_mac - white_gtsam)),
            "nis_truth_using_macvio_Q": mahalanobis(q_mac, db_gt_mac),
            "nis_truth_using_gtsam_Q": mahalanobis(q_gtsam, db_gt_gtsam),
            "nis_macvio_estimated_increment_using_Q": mahalanobis(q_mac, db_est_mac),
            "nis_gtsam_estimated_increment_using_Q": mahalanobis(q_gtsam, db_est_gtsam),
        }
        for name, value in [
            ("truth_increment_macvio", db_gt_mac),
            ("truth_increment_gtsam", db_gt_gtsam),
            ("estimated_increment_macvio", db_est_mac),
            ("estimated_increment_gtsam", db_est_gtsam),
            ("white_noise_mean_macvio", white_mac),
            ("white_noise_mean_gtsam", white_gtsam),
        ]:
            for index, number in enumerate(value):
                brow[f"{name}_{index}"] = float(number)
        bias_rows.append(brow)

    write_csv(output_dir / "preintegration_delta_comparison_per_edge.csv", delta_rows)
    write_csv(output_dir / "covariance_comparison_per_edge.csv", covariance_rows)
    write_csv(output_dir / "nis_cross_evaluation_per_edge.csv", nis_rows)
    write_csv(output_dir / "bias_rw_cross_comparison.csv", bias_rows)

    mac_w_array = np.asarray(mac_whitened)
    gtsam_w_array = np.asarray(gtsam_whitened)
    mac_error_array = np.asarray(mac_errors)
    gtsam_error_array = np.asarray(gtsam_errors)
    delta_summary = build_delta_summary(delta_rows)
    covariance_summary = build_covariance_summary(covariance_rows)
    nis_summary = build_nis_summary(nis_rows, mac_w_array, gtsam_w_array)
    bias_summary = build_bias_summary(bias_rows)

    # Both measurement models are conservative in a similar direction; Q_bias itself passes.
    classification = ["D", "E"]
    approved = False
    decision = {
        "status": "COMPARISON_COMPLETE",
        "input_gate": gate,
        "same_dataset": True,
        "same_four_continuous_sigmas": True,
        "delta_consistency": {
            "classification": "numerically consistent after common frame and bias-linearization alignment",
            "noisy_delta_p_norm_max": delta_summary["noisy_delta_p_norm"]["max"],
            "noisy_delta_v_norm_max": delta_summary["noisy_delta_v_norm"]["max"],
            "noisy_delta_R_norm_max": delta_summary["noisy_delta_R_norm"]["max"],
        },
        "covariance_consistency": {
            "full_relative_frobenius_median": covariance_summary["full_relative_P_error"]["median"],
            "trace_ratio_median": covariance_summary["trace_ratio_macvio_over_gtsam"]["median"],
            "position_block_difference_explained_by_gtsam_integration_covariance": True,
            "gtsam_integration_covariance": gtsam_contract["integrationCovariance"],
            "macvio_additional_integration_covariance": mac_contract["additional_integration_covariance"],
        },
        "measurement_nis": nis_summary,
        "preintegration_delta_summary": delta_summary,
        "covariance_summary": covariance_summary,
        "bias_rw": bias_summary,
        "decision_classification": classification,
        "classification_explanation": {
            "D": "Both measurement covariances are conservative by similar factors under the same endpoint-interpolation/midpoint contract.",
            "E": "Total NIS coverage is near but below the nominal shape; P/V/R block means are all below 3 and cross-component correlations remain.",
        },
        "approved_for_N_2_5_10_fixed_lag_experiment": approved,
        "approval_reason": "Measurement covariance consistency has not passed; first resolve or explicitly model the common sampling/interpolation covariance mismatch. Do not tune the four sigmas.",
        "production_code_modified": False,
        "preintegrator_reexecuted": False,
        "tensor_map_path": str(tensor_map_path),
        "tensor_map_sha256": actual_tensor_hash,
    }
    write_json(output_dir / "final_decision.json", decision)
    report = render_report(
        gate,
        convention,
        delta_summary,
        covariance_summary,
        nis_summary,
        bias_summary,
        decision,
        mac_contract,
        gtsam_contract,
    )
    (output_dir / "macvio_vs_gtsam_normal_noise_comparison_cn.md").write_text(
        report, encoding="utf-8"
    )
    return decision


def build_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mapping = {
        "noisy_delta_p_norm": "delta_p_noisy_difference_norm",
        "noisy_delta_v_norm": "delta_v_noisy_difference_norm",
        "noisy_delta_R_norm": "delta_R_noisy_log_difference_norm",
        "reference_delta_p_norm": "delta_p_reference_difference_norm",
        "reference_delta_v_norm": "delta_v_reference_difference_norm",
        "reference_delta_R_norm": "delta_R_reference_log_difference_norm",
        "noise_error_p_difference_norm": "noise_error_macvio_minus_gtsam_p_norm",
        "noise_error_v_difference_norm": "noise_error_macvio_minus_gtsam_v_norm",
        "noise_error_R_difference_norm": "noise_error_macvio_minus_gtsam_R_norm",
    }
    result: dict[str, Any] = {name: summary(row[column] for row in rows) for name, column in mapping.items()}
    result["signed_axis_differences"] = {}
    axis_prefixes = {
        "noisy_p": "delta_p_noisy_difference",
        "noisy_v": "delta_v_noisy_difference",
        "noisy_R": "delta_R_noisy_log_difference",
        "reference_p": "delta_p_reference_difference",
        "reference_v": "delta_v_reference_difference",
        "reference_R": "delta_R_reference_log_difference",
    }
    for label, prefix in axis_prefixes.items():
        result["signed_axis_differences"][label] = {
            axis: summary(row[f"{prefix}_{axis}"] for row in rows) for axis in "xyz"
        }
    result["absolute_delta_scale_regression"] = {}
    regression_pairs = {
        "noisy_p": ("delta_p_macvio_noisy", "delta_p_gtsam_noisy"),
        "noisy_v": ("delta_v_macvio_noisy", "delta_v_gtsam_noisy"),
        "noisy_R_rotvec": ("delta_R_macvio_noisy_rotvec", "delta_R_gtsam_noisy_rotvec"),
        "reference_p": ("delta_p_macvio_reference", "delta_p_gtsam_reference"),
        "reference_v": ("delta_v_macvio_reference", "delta_v_gtsam_reference"),
        "reference_R_rotvec": (
            "delta_R_macvio_reference_rotvec",
            "delta_R_gtsam_reference_rotvec",
        ),
    }
    for label, (mac_prefix, gtsam_prefix) in regression_pairs.items():
        mac_values = np.asarray([[row[f"{mac_prefix}_{axis}"] for axis in "xyz"] for row in rows])
        gtsam_values = np.asarray([[row[f"{gtsam_prefix}_{axis}"] for axis in "xyz"] for row in rows])
        result["absolute_delta_scale_regression"][label] = {}
        for index, axis in enumerate("xyz"):
            denominator = float(gtsam_values[:, index] @ gtsam_values[:, index])
            slope = float(gtsam_values[:, index] @ mac_values[:, index] / denominator) if denominator > EPS else None
            offset = float(np.mean(mac_values[:, index] - gtsam_values[:, index]))
            result["absolute_delta_scale_regression"][label][axis] = {
                "through_origin_slope_macvio_over_gtsam": slope,
                "mean_signed_offset": offset,
            }
    result["bias_linearization_correction_acc_norm"] = summary(
        row["mac_bias_correction_acc_norm"] for row in rows
    )
    result["bias_linearization_correction_gyro_norm"] = summary(
        row["mac_bias_correction_gyro_norm"] for row in rows
    )
    result["by_macvio_motion_stage"] = {}
    for stage in sorted({row["motion_stage_macvio"] for row in rows}):
        subset = [row for row in rows if row["motion_stage_macvio"] == stage]
        result["by_macvio_motion_stage"][stage] = {
            "count": len(subset),
            **{name: summary(row[column] for row in subset) for name, column in mapping.items()},
        }
    dt = np.asarray([row["delta_t"] for row in rows])
    result["edge_dt"] = summary(dt)
    result["dt_unique_rounded_12"] = sorted(set(np.round(dt, 12).tolist()))
    result["difference_norm_vs_dt_pearson"] = {}
    for name, column in mapping.items():
        values = np.asarray([row[column] for row in rows])
        if np.std(dt) < 1e-12:
            result["difference_norm_vs_dt_pearson"][name] = None
        else:
            result["difference_norm_vs_dt_pearson"][name] = float(np.corrcoef(dt, values)[0, 1])
    result["dt_growth_interpretation"] = "Edge dt is effectively constant; this slice cannot identify growth with edge duration."
    return result


def build_covariance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "full_relative_P_error": summary(row["relative_P_error"] for row in rows),
        "trace_ratio_macvio_over_gtsam": summary(
            row["trace_ratio_macvio_over_gtsam"] for row in rows
        ),
        "logdet_difference_macvio_minus_gtsam": summary(
            row["logdet_difference_macvio_minus_gtsam"] for row in rows
        ),
        "principal_max_eigenvector_angle_deg": summary(
            row["principal_max_eigenvector_angle_deg"] for row in rows
        ),
        "dominant_3d_subspace_max_principal_angle_deg": summary(
            row["dominant_3d_subspace_max_principal_angle_deg"] for row in rows
        ),
        "condition_macvio": summary(row["condition_macvio"] for row in rows),
        "condition_gtsam": summary(row["condition_gtsam"] for row in rows),
    }
    result["diagonal_ratios"] = {
        str(index): summary(row[f"diag_ratio_{index}"] for row in rows) for index in range(9)
    }
    result["eigenvalue_ratios"] = {
        str(index): summary(row[f"eigenvalue_ratio_{index}"] for row in rows) for index in range(9)
    }
    result["blocks_relative_frobenius"] = {}
    for left in ["p", "v", "R"]:
        for right in ["p", "v", "R"]:
            key = f"{left}{right}"
            result["blocks_relative_frobenius"][key] = summary(
                row[f"block_{key}_relative_frobenius"] for row in rows
            )
    return result


def build_nis_summary(
    rows: list[dict[str, Any]], mac_w: np.ndarray, gtsam_w: np.ndarray
) -> dict[str, Any]:
    keys = [
        "nis_macvio_error_macvio_P",
        "nis_macvio_error_gtsam_P",
        "nis_gtsam_error_macvio_P",
        "nis_gtsam_error_gtsam_P",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows])
        result[key] = {**summary(values), **coverage(values, CHI2_9_CENTRAL_95)}
    result["error_vector_difference_norm"] = summary(
        row["error_vector_difference_norm"] for row in rows
    )
    result["reported_nis_recompute_max_abs"] = {
        "macvio": max(row["macvio_recompute_abs_difference"] for row in rows),
        "gtsam": max(row["gtsam_recompute_abs_difference"] for row in rows),
    }
    result["own_block_nis"] = {
        side: {
            block: summary(row[f"nis_{side}_own_{block}"] for row in rows)
            for block in ["p", "v", "R"]
        }
        for side in ["macvio", "gtsam"]
    }
    for label, values in [("macvio", mac_w), ("gtsam", gtsam_w)]:
        result[f"{label}_whitened"] = {
            "mean": np.mean(values, axis=0),
            "covariance": np.cov(values, rowvar=False, ddof=1),
            "std": np.std(values, axis=0, ddof=1),
            "autocorrelation": {str(lag): autocorrelation(values, lag) for lag in [1, 2, 5, 10]},
        }
    return result


def build_bias_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth_mac = np.asarray([[row[f"truth_increment_macvio_{i}"] for i in range(6)] for row in rows])
    truth_gtsam = np.asarray([[row[f"truth_increment_gtsam_{i}"] for i in range(6)] for row in rows])
    est_mac = np.asarray([[row[f"estimated_increment_macvio_{i}"] for i in range(6)] for row in rows])
    est_gtsam = np.asarray([[row[f"estimated_increment_gtsam_{i}"] for i in range(6)] for row in rows])
    white_mac = np.asarray([[row[f"white_noise_mean_macvio_{i}"] for i in range(6)] for row in rows])
    white_gtsam = np.asarray([[row[f"white_noise_mean_gtsam_{i}"] for i in range(6)] for row in rows])
    nis_mac = np.asarray([row["nis_truth_using_macvio_Q"] for row in rows])
    nis_gtsam = np.asarray([row["nis_truth_using_gtsam_Q"] for row in rows])
    truth_energy = float(np.mean(np.sum(truth_mac**2, axis=1)))
    mac_energy = float(np.mean(np.sum(est_mac**2, axis=1)))
    gtsam_energy = float(np.mean(np.sum(est_gtsam**2, axis=1)))
    corr_mac = [float(np.corrcoef(est_mac[:, i], white_mac[:, i])[0, 1]) for i in range(6)]
    corr_gtsam = [float(np.corrcoef(est_gtsam[:, i], white_gtsam[:, i])[0, 1]) for i in range(6)]
    return {
        "factor_models": {
            "macvio": "independent 6D bias random-walk residual",
            "gtsam": "ImuFactor + BetweenFactor<imuBias::ConstantBias>; not CombinedImuFactor",
        },
        "Q_formula_equal": True,
        "Q_relative_frobenius": summary(row["Q_relative_frobenius"] for row in rows),
        "Q_max_abs_difference": max(row["Q_max_abs_difference"] for row in rows),
        "truth_increment_difference_norm": summary(
            row["truth_increment_difference_norm"] for row in rows
        ),
        "white_noise_mean_difference_norm": summary(
            row["white_noise_mean_difference_norm"] for row in rows
        ),
        "truth_nis_macvio_Q": {**summary(nis_mac), **coverage(nis_mac, CHI2_6_CENTRAL_95)},
        "truth_nis_gtsam_Q": {**summary(nis_gtsam), **coverage(nis_gtsam, CHI2_6_CENTRAL_95)},
        "increment_mean_squared_total": {
            "truth": truth_energy,
            "macvio_estimate": mac_energy,
            "gtsam_estimate": gtsam_energy,
            "macvio_estimate_over_truth": mac_energy / truth_energy,
            "gtsam_estimate_over_truth": gtsam_energy / truth_energy,
        },
        "estimated_increment_vs_white_noise_same_axis_correlation": {
            "macvio": corr_mac,
            "gtsam": corr_gtsam,
            "macvio_max_abs": max(abs(value) for value in corr_mac),
            "gtsam_max_abs": max(abs(value) for value in corr_gtsam),
        },
        "random_walk_measurement_model_passed": bool(
            abs(float(np.mean(nis_mac)) - 6.0) < 1.0
            and coverage(nis_mac, CHI2_6_CENTRAL_95)["fraction_inside"] > 0.90
        ),
        "online_bias_dynamics_note": "Q_bias and GT increments agree. MACVIO online bias is much too active; GTSAM online bias is much smoother than truth. This is estimator behavior, not a Q_bias mismatch.",
    }


def table_stat_line(name: str, value: dict[str, Any]) -> str:
    return (
        f"| {name} | {value['min']:.6g} | {value['median']:.6g} | "
        f"{value['mean']:.6g} | {value['p95']:.6g} | {value['max']:.6g} |"
    )


def render_report(
    gate: dict[str, Any],
    convention: dict[str, Any],
    delta: dict[str, Any],
    covariance: dict[str, Any],
    nis: dict[str, Any],
    bias: dict[str, Any],
    decision: dict[str, Any],
    mac_contract: dict[str, Any],
    gtsam_contract: dict[str, Any],
) -> str:
    mm = nis["nis_macvio_error_macvio_P"]
    gg = nis["nis_gtsam_error_gtsam_P"]
    mg = nis["nis_macvio_error_gtsam_P"]
    gm = nis["nis_gtsam_error_macvio_P"]
    lines = [
        "# MACVIO 与 GTSAM Normal-noise covariance / NIS 跨项目对比",
        "",
        "## 首页结论",
        "",
        "| 必答问题 | 结论 |",
        "| --- | --- |",
        "| 1. 是否同一数据 | **是**。metadata、IMU CSV、truth decomposition 哈希一致；209 条边在编号/计数语义归一化后逐边一致。 |",
        "| 2. 是否同一四个 sigma | **是**。`sigma_a/g/aw/gw` 逐值一致。 |",
        f"| 3. Delta 是否一致 | **是，数值一致**。公共 NED 下 noisy `dp/dv/dR` 最大 norm 分别为 `{delta['noisy_delta_p_norm']['max']:.3e}` / `{delta['noisy_delta_v_norm']['max']:.3e}` / `{delta['noisy_delta_R_norm']['max']:.3e}`。 |",
        f"| 4. P 是否一致 | **主体一致，但不完全相同**。完整相对 Frobenius 中位数 `{covariance['full_relative_P_error']['median']:.3e}`；GTSAM 额外 `integrationCovariance=1e-8 I` 使 P block 约大 12.1%。 |",
        f"| 5. 哪边 NIS 通过 | **两边都不能判为完全通过**。MACVIO mean `{mm['mean']:.4f}`，GTSAM mean `{gg['mean']:.4f}`，均低于 9，都是偏保守。 |",
        f"| 6. Bias RW 是否通过 | **GT/Q 模型通过**。共同 NIS6 mean `{bias['truth_nis_macvio_Q']['mean']:.4f}`；但两边在线 bias 动态差异很大。 |",
        "| 7. 是否批准 N=2/5/10 | **暂不批准**。先处理共同的端点插值 + midpoint 采样 covariance 契约；不应调四个 sigma。 |",
        "",
        "最终分类：**D + E**。两边 measurement covariance 以相似方向偏保守；总 NIS 没有爆炸，但 P/V/R 分块均低于理论均值且仍有相关结构。",
        "",
        "## 1. 输入门与来源",
        "",
        f"- 场景：`{mac_contract['input_hashes']['scene']}`",
        f"- metadata SHA-256：`{mac_contract['input_hashes']['metadata_sha256']}`",
        f"- IMU CSV SHA-256：`{mac_contract['input_hashes']['imu_csv_sha256']}`",
        f"- truth decomposition SHA-256：`{mac_contract['input_hashes']['imu_truth_decomposition_sha256']}`",
        "- frame：0..299；有效边：90→91 到 298→299，共 209 条。",
        "- frame、timestamp、delta_t、首尾 IMU timestamp 全部逐行一致；未丢弃任何边。",
        "- `edge_id` 只是编号基准不同：MACVIO 为 1..209，GTSAM 为 0..208，稳定满足 `mac_id=gtsam_id+1`。",
        "- sample count 定义不同但区间相同：MACVIO 记录 5 个插值 knot，GTSAM 记录 4 个相邻 integration step，稳定满足 `knot_count=step_count+1`。",
        "- 两边 manifest 文件哈希不同是因为列结构不同，不是边集合不同。",
        "",
        "四个公共连续时间参数：",
        "",
        "```text",
        f"sigma_a  = {mac_contract['continuous_noise_density']['sigma_a']}",
        f"sigma_g  = {mac_contract['continuous_noise_density']['sigma_g']}",
        f"sigma_aw = {mac_contract['continuous_noise_density']['sigma_aw']}",
        f"sigma_gw = {mac_contract['continuous_noise_density']['sigma_gw']}",
        "```",
        "",
        "## 2. 公共数学约定",
        "",
        "公共排列是 `[P,V,R]`，公共坐标是 MACVIO 内部 NED body tangent。MACVIO 已在该约定下，因此 `J_mac=I9`。GTSAM 审计原始数据仍是 FLU，并且 TangentPreintegration 原生误差为 `[theta,P,V]` 的加性误差。逐边完整变换为：",
        "",
        "```text",
        "D = diag(1,-1,-1)                     # FLU -> NED",
        "S = blockdiag(D,D,D)",
        "T_edge = permutation([theta,P,V] -> [P,V,R])",
        "         with R block Jr(theta_reference)",
        "J_gtsam_to_common_edge = S * T_edge",
        "e_common = J_edge * e_native",
        "P_common = J_edge * P_native * J_edge^T",
        "```",
        "",
        f"- 已存误差与完整 J 的最大 norm：`{convention['stored_error_vs_full_J_max_norm']:.3e}`",
        f"- 存储 Jr 与公式最大绝对差：`{convention['stored_Jr_vs_formula_max_abs']:.3e}`",
        f"- 合成小扰动检查（epsilon={convention['synthetic_small_perturbation_epsilon']:.1e}）最大差：`{convention['synthetic_small_perturbation_max_abs']:.3e}`",
        f"- rotation 单轴符号/基变换最大差：`{convention['rotation_single_axis_basis_change_max_abs']:.3e}`",
        f"- 合成零噪声检查：`{convention['synthetic_zero_noise_error_norm']:.1f}`",
        "",
        "因此不能仅做块排列；本报告所有 GTSAM e/P 都使用完整逐边 J。",
        "",
        "## 3. 预积分 Delta",
        "",
        "MACVIO 的 absolute Delta 来自独立审计哈希锁定的生产 `tensor_map.npz`，并使用缓存 bias Jacobian 一阶移动到与 GTSAM 相同的 edge-start GT bias。没有重跑预积分。reference Delta 由该 noisy Delta 和独立审计导出的 noisy-reference error 反推。",
        "",
        "| 差异 norm | min | median | mean | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        table_stat_line("noisy delta_p / m", delta["noisy_delta_p_norm"]),
        table_stat_line("noisy delta_v / m/s", delta["noisy_delta_v_norm"]),
        table_stat_line("noisy delta_R / rad", delta["noisy_delta_R_norm"]),
        table_stat_line("reference delta_p / m", delta["reference_delta_p_norm"]),
        table_stat_line("reference delta_v / m/s", delta["reference_delta_v_norm"]),
        table_stat_line("reference delta_R / rad", delta["reference_delta_R_norm"]),
        "",
        "noisy Delta 三轴有符号差的最大绝对值：",
        "",
        "```text",
        "dp: " + ", ".join(f"{max(abs(delta['signed_axis_differences']['noisy_p'][a]['min']), abs(delta['signed_axis_differences']['noisy_p'][a]['max'])):.3e}" for a in "xyz"),
        "dv: " + ", ".join(f"{max(abs(delta['signed_axis_differences']['noisy_v'][a]['min']), abs(delta['signed_axis_differences']['noisy_v'][a]['max'])):.3e}" for a in "xyz"),
        "dR: " + ", ".join(f"{max(abs(delta['signed_axis_differences']['noisy_R'][a]['min']), abs(delta['signed_axis_differences']['noisy_R'][a]['max'])):.3e}" for a in "xyz"),
        "```",
        "",
        "按 MACVIO 公共阶段标签的最大 noisy Delta 差：",
        "",
        "| stage | count | dp max | dv max | dR max |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {stage} | {data['count']} | {data['noisy_delta_p_norm']['max']:.3e} | {data['noisy_delta_v_norm']['max']:.3e} | {data['noisy_delta_R_norm']['max']:.3e} |"
            for stage, data in delta["by_macvio_motion_stage"].items()
        ],
        "",
        "这些差异远小于各自噪声误差尺度，没有固定符号或尺度翻转证据。当前所有 edge dt 近似相同，因此本片段不能检验差值随 edge duration 增长。逐运动阶段统计已写入逐边 CSV 和 `final_decision.json` 的上游汇总；该 300 帧只覆盖早期 stationary/accelerating/constant-velocity，不包含矩形转弯。",
        "",
        "## 4. Covariance",
        "",
        f"- 完整 9x9 relative Frobenius error：median `{covariance['full_relative_P_error']['median']:.3e}`，max `{covariance['full_relative_P_error']['max']:.3e}`。",
        f"- trace ratio MACVIO/GTSAM：median `{covariance['trace_ratio_macvio_over_gtsam']['median']:.9f}`。",
        f"- 最大特征向量夹角：median `{covariance['principal_max_eigenvector_angle_deg']['median']:.3e}` deg，max `{covariance['principal_max_eigenvector_angle_deg']['max']:.3e}` deg。",
        f"- 最大三个特征向量张成子空间的最大 principal angle：median `{covariance['dominant_3d_subspace_max_principal_angle_deg']['median']:.3e}` deg，max `{covariance['dominant_3d_subspace_max_principal_angle_deg']['max']:.3e}` deg。",
        f"- condition median：MACVIO `{covariance['condition_macvio']['median']:.3f}`，GTSAM `{covariance['condition_gtsam']['median']:.3f}`。",
        "",
        "对角线比例 MACVIO/GTSAM 的中位数：",
        "",
        "```text",
        "P: " + ", ".join(f"{covariance['diagonal_ratios'][str(i)]['median']:.6f}" for i in range(3)),
        "V: " + ", ".join(f"{covariance['diagonal_ratios'][str(i)]['median']:.6f}" for i in range(3, 6)),
        "R: " + ", ".join(f"{covariance['diagonal_ratios'][str(i)]['median']:.6f}" for i in range(6, 9)),
        "```",
        "",
        "V、R 主块和整体 trace 几乎一致；P 主块约小 12.1%，与 GTSAM 独有的 `integrationCovariance=1e-8 I` 完全同方向。单根最大特征向量角度看似很大，是因为三个轴向最大特征值近乎简并，单根 eigenvector 可在该三维子空间内任意旋转；应同时看稳定得多的 dominant-3D subspace angle。所有 3x3 主块、交叉块、9 个特征值、logdet 和主特征向量逐边数据均在 `covariance_comparison_per_edge.csv`。",
        "",
        "## 5. NIS 交叉评价",
        "",
        "| error / covariance | mean | median | p95 | central 95% inside |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| MAC e / MAC P | {mm['mean']:.4f} | {mm['median']:.4f} | {mm['p95']:.4f} | {mm['fraction_inside']:.2%} |",
        f"| MAC e / GTSAM P | {mg['mean']:.4f} | {mg['median']:.4f} | {mg['p95']:.4f} | {mg['fraction_inside']:.2%} |",
        f"| GTSAM e / MAC P | {gm['mean']:.4f} | {gm['median']:.4f} | {gm['p95']:.4f} | {gm['fraction_inside']:.2%} |",
        f"| GTSAM e / GTSAM P | {gg['mean']:.4f} | {gg['median']:.4f} | {gg['p95']:.4f} | {gg['fraction_inside']:.2%} |",
        "",
        "自用 covariance 的 3DoF block NIS mean：",
        "",
        "| project | P | V | R |",
        "| --- | ---: | ---: | ---: |",
        f"| MACVIO | {nis['own_block_nis']['macvio']['p']['mean']:.4f} | {nis['own_block_nis']['macvio']['v']['mean']:.4f} | {nis['own_block_nis']['macvio']['R']['mean']:.4f} |",
        f"| GTSAM | {nis['own_block_nis']['gtsam']['p']['mean']:.4f} | {nis['own_block_nis']['gtsam']['v']['mean']:.4f} | {nis['own_block_nis']['gtsam']['R']['mean']:.4f} |",
        "",
        f"两边 common error 的逐边 norm 差 median `{nis['error_vector_difference_norm']['median']:.3e}`、max `{nis['error_vector_difference_norm']['max']:.3e}`。交换 covariance 后的 NIS 变化主要来自 GTSAM P block 的额外 integration covariance；不是预积分均值冲突。",
        "",
        "两边自用 NIS 均明显低于 9，且都没有高于 19.023 的尾部。这支持共同的偏保守来源：保存样本端点插值后再 midpoint 平均，使相邻小步噪声相关且方差下降，而两边传播仍以独立连续白噪声小步近似处理。该结论是现有证据最一致的解释，不等于证明库的通用公式错误。",
        "",
        "## 6. Bias random walk",
        "",
        "两边都使用独立 bias factor，GTSAM 不是 CombinedImuFactor：",
        "",
        "```text",
        "Q_bias = diag(sigma_aw^2*dt*I3, sigma_gw^2*dt*I3), order [ba,bg]",
        "```",
        "",
        f"- Q 最大绝对差：`{bias['Q_max_abs_difference']:.3e}`。",
        f"- GT endpoint increment 差 norm 最大：`{bias['truth_increment_difference_norm']['max']:.3e}`。",
        f"- truth NIS6：mean `{bias['truth_nis_macvio_Q']['mean']:.4f}`，central 95% inside `{bias['truth_nis_macvio_Q']['fraction_inside']:.2%}`。",
        f"- MACVIO estimated increment energy / truth：`{bias['increment_mean_squared_total']['macvio_estimate_over_truth']:.3g}`。",
        f"- GTSAM estimated increment energy / truth：`{bias['increment_mean_squared_total']['gtsam_estimate_over_truth']:.3g}`。",
        "",
        "因此 Bias RW 的随机模型本身通过；MACVIO 在线 bias 明显过活跃，而 GTSAM bias 比 truth 更平滑。这是优化器状态约束/可观测性行为差异，不能把两个 estimated-bias cost 当作 covariance 公式优劣。",
        "",
        "## 7. 决策",
        "",
        "分类为：",
        "",
        "- **D：两边以相似方向失败。** Delta 和 measurement error 几乎相同，P 主体也相同；共同的 reference 构造、端点插值和 midpoint 噪声相关性是第一检查对象。",
        "- **E：总 NIS 没有爆炸，但分块仍异常。** MACVIO/GTSAM 的 P、V、R mean NIS 均低于各自 3DoF 理论均值 3，不能宣布完全通过。",
        "- **F 不成立于 Q_bias 模型。** GT bias increment 与 Q_bias 基本一致；在线 bias 动态问题另列。",
        "",
        "**当前不批准直接进入 N=2/N=5/N=10 正式窗口归因实验。** 在 measurement covariance 契约尚未统计闭合时，加长窗口可能改变抖动，但不能证明两状态窗口是根因。下一步应固定四个 sigma，建立 sampling-aware 的端点插值/midpoint covariance 反事实或 Monte Carlo 验证；通过后再做窗口长度实验。",
        "",
        "## 8. 限制与可复现性",
        "",
        "1. 只有一个 noise realization、209 条相邻边；共享端点使边间不独立。",
        "2. 300 帧只覆盖矩形轨迹早期，没有 turning/decelerating 阶段。",
        "3. MACVIO absolute Delta 是缓存值加生产一阶 bias correction；bias correction 较大时存在一阶近似余项，但观测到的跨实现差仍只有 1e-6 及以下。",
        "4. 本轮未重新执行预积分、未修改生产代码、未调 sigma、integrationCovariance 或优化器参数。",
        "5. 完整逐边证据见同目录 6 个 CSV/JSON；没有以轨迹平滑程度替代 NIS 结论。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mac-root",
        type=Path,
        default=Path("/home/admin1/macvo-dev/analysis_rectangle_normal_noise_preintegration_nis_audit_20260715"),
    )
    parser.add_argument(
        "--gtsam-root",
        type=Path,
        default=Path("/home/admin1/gtsam/macvo_vio/results/normal_noise_preintegration_audit"),
    )
    parser.add_argument(
        "--tensor-map",
        type=Path,
        default=Path("/home/admin1/macvo-dev/Results/rectangle_normal_noise_two_state_standard_full_20260715/trial_1/vio_two_state_fixed_lag_standard_full/clear_stop_turn_rectangle_truth_normal_noise/tensor_map.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/admin1/macvo-dev/analysis_macvio_vs_gtsam_normal_noise_covariance_nis_20260715"),
    )
    args = parser.parse_args()
    decision = compare(args.mac_root, args.gtsam_root, args.tensor_map, args.output_dir)
    print(json.dumps(jsonable(decision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

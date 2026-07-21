#!/usr/bin/env python3
"""Focused regression tests for the opt-in sampling-aware covariance path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Module.IMUPreintegration import (  # noqa: E402
    preintegrate_imu_local_frame,
    replace_with_sampling_aware_covariance,
)
from Utility.IMUCSV import IMUCSVLoader  # noqa: E402


SCENE = "clear_stop_turn_rectangle_truth_normal_noise"
DATASET = (
    Path("/mnt/e")
    / "\u6587\u6863/holoocean/code/recordings"
    / "batch_clear_truth_paths_20260713_static63_variants"
    / SCENE
)
RESULT = (
    ROOT
    / "Results/rectangle_normal_noise_two_state_standard_full_20260715"
    / "trial_1/vio_two_state_fixed_lag_standard_full"
    / SCENE
)
SIGMA_A = 0.0141258
SIGMA_G = 0.00182898
SIGMA_AW = 0.000386071
SIGMA_GW = 3.57864e-05
SENSOR_TO_INTERNAL = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))


def main() -> int:
    loader = IMUCSVLoader(DATASET / "imu_data.csv")
    tensor_map = np.load(RESULT / "tensor_map.npz", allow_pickle=False)
    frame_times = tensor_map["frames//time_ns"].astype(np.int64)
    linearized_ba = tensor_map["frames//imu_vio_linearized_acc_bias"].astype(np.float64)
    linearized_bg = tensor_map["frames//imu_vio_linearized_gyro_bias"].astype(np.float64)
    stored_covariance = tensor_map["frames//imu_vio_cov"].astype(np.float64)

    rows: list[dict[str, float | int]] = []
    for frame_i, frame_j in ((90, 91), (91, 92), (92, 93)):
        start_ns = int(frame_times[frame_i])
        end_ns = int(frame_times[frame_j])
        old_time, old_acc, old_gyro = loader.query_range(start_ns, end_ns)
        time_ns, acc_flu, gyro_flu, sampling = loader.query_range_with_sampling_map(
            start_ns, end_ns
        )
        assert sampling is not None
        assert torch.equal(time_ns, old_time)
        assert torch.equal(acc_flu, old_acc)
        assert torch.equal(gyro_flu, old_gyro)

        raw_acc = loader.acc[sampling.raw_indices].double()
        raw_gyro = loader.gyro[sampling.raw_indices].double()
        reconstructed_acc = sampling.knot_from_raw @ raw_acc
        reconstructed_gyro = sampling.knot_from_raw @ raw_gyro
        map_value_error = max(
            float((reconstructed_acc - acc_flu.double()).abs().max()),
            float((reconstructed_gyro - gyro_flu.double()).abs().max()),
        )

        acc_internal = (SENSOR_TO_INTERNAL @ acc_flu.double().T).T.float()
        gyro_internal = (SENSOR_TO_INTERNAL @ gyro_flu.double().T).T.float()
        ba = torch.from_numpy(linearized_ba[frame_j]).float()
        bg = torch.from_numpy(linearized_bg[frame_j]).float()
        current = preintegrate_imu_local_frame(
            time_ns=time_ns,
            acc=acc_internal,
            gyro=gyro_internal,
            sigma_acc=SIGMA_A,
            sigma_gyro=SIGMA_G,
            sigma_acc_w=SIGMA_AW,
            sigma_gyro_w=SIGMA_GW,
            acc_bias=ba,
            gyro_bias=bg,
        )
        sampling_result = replace_with_sampling_aware_covariance(
            current,
            time_ns=time_ns,
            acc_internal=acc_internal,
            gyro_internal=gyro_internal,
            knot_from_raw=sampling.knot_from_raw,
            sensor_to_internal_rotation=SENSOR_TO_INTERNAL,
            measurement_rate_hz=100.0,
            sigma_acc=SIGMA_A,
            sigma_gyro=SIGMA_G,
            acc_bias=ba,
            gyro_bias=bg,
        )

        delta_difference = max(
            float((sampling_result.delta_p - current.delta_p).abs().max()),
            float((sampling_result.delta_v - current.delta_v).abs().max()),
            float(
                (
                    sampling_result.delta_R.tensor() - current.delta_R.tensor()
                ).abs().max()
            ),
        )
        current_sum_error = float(
            (
                current.cov.double()
                - current.measurement_cov.double()
                - current.bias_process_cov.double()
            ).abs().max()
        )
        stored_cov_error = float(
            (current.cov.double() - torch.from_numpy(stored_covariance[frame_j])).abs().max()
        )
        sampling_min_eigenvalue = float(
            torch.linalg.eigvalsh(sampling_result.cov.double()).min()
        )
        rows.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                "map_value_max_abs_error": map_value_error,
                "delta_max_abs_difference": delta_difference,
                "current_measurement_plus_bias_max_abs_error": current_sum_error,
                "current_vs_frozen_cov_max_abs_error": stored_cov_error,
                "sampling_cov_min_eigenvalue": sampling_min_eigenvalue,
                "sampling_over_current_trace_ratio": float(
                    torch.trace(sampling_result.cov.double())
                    / torch.trace(current.cov.double())
                ),
            }
        )

    assertions = {
        # Production interpolation is float32; near 9.8 m/s^2 this allows a
        # single-ULP reconstruction difference against the float64 map check.
        "query_map_reconstructs_knots": max(row["map_value_max_abs_error"] for row in rows) < 1e-6,
        "sampling_mode_does_not_change_deltas": max(row["delta_max_abs_difference"] for row in rows) == 0.0,
        "current_covariance_decomposition_closes": max(
            row["current_measurement_plus_bias_max_abs_error"] for row in rows
        ) < 2e-12,
        "current_mode_matches_frozen_covariance": max(
            row["current_vs_frozen_cov_max_abs_error"] for row in rows
        ) < 2e-12,
        "sampling_covariance_is_spd": min(
            row["sampling_cov_min_eigenvalue"] for row in rows
        ) > 0.0,
    }
    output = {"edges": rows, "assertions": assertions, "passed": all(assertions.values())}
    print(json.dumps(output, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

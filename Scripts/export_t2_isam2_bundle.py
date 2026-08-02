#!/usr/bin/env python3
"""Export archived PACE factors for the standalone C++ iSAM2 backend.

The bundle preserves the production optimizer's internal NED/IMU-centre
contract.  It contains no ground truth and does not modify the source tensor
map.  Reference residuals and Jacobians are included only for cross-language
regression tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pypose as pp
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.IMUKinematics import vio_preintegrated_imu_residual
from Utility.T2HistorySmoother import (
    compressed_factor_equivalence,
    load_t2_history_archive,
)
from Utility.TwoStateVIO import (
    NavigationState,
    PAIR_DOF,
    STATE_DOF,
    _linearized_visual_residual_and_analytic_jacobian,
    retract_state,
)


POSE_PERMUTATION_TR_TO_RT = np.block(
    [
        [np.zeros((3, 3)), np.eye(3)],
        [np.eye(3), np.zeros((3, 3))],
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=90)
    parser.add_argument("--end-frame", type=int, default=299)
    return parser.parse_args()


def _flatten(prefix: str, value: np.ndarray) -> tuple[list[str], list[float]]:
    array = np.asarray(value)
    names: list[str] = []
    if array.ndim == 1:
        names = [f"{prefix}_{index}" for index in range(array.shape[0])]
    elif array.ndim == 2:
        names = [
            f"{prefix}_{row}_{col}"
            for row in range(array.shape[0])
            for col in range(array.shape[1])
        ]
    else:
        raise ValueError(f"cannot flatten {prefix} with shape {array.shape}")
    return names, array.reshape(-1).astype(np.float64).tolist()


def _rotation_vector(delta_rotation: torch.Tensor) -> np.ndarray:
    value = delta_rotation.detach().reshape(-1)
    if value.numel() == 3:
        return value.cpu().numpy().astype(np.float64, copy=False)
    if value.numel() == 4:
        return pp.SO3(value.reshape(1, 4)).Log().tensor().reshape(3).cpu().numpy()
    raise ValueError(f"unsupported delta rotation shape {tuple(delta_rotation.shape)}")


def _normalized_pose(pose: torch.Tensor) -> torch.Tensor:
    result = pose.detach().reshape(7).to(dtype=torch.float64).clone()
    result[3:7] = result[3:7] / result[3:7].norm().clamp_min(1.0e-15)
    return result.reshape(1, 7)


def _normalized_state(state: NavigationState) -> NavigationState:
    return NavigationState(
        pose_WB=_normalized_pose(state.pose_WB),
        velocity_W=state.velocity_W.detach().to(dtype=torch.float64).clone(),
        acc_bias=state.acc_bias.detach().to(dtype=torch.float64).clone(),
        gyro_bias=state.gyro_bias.detach().to(dtype=torch.float64).clone(),
    )


def _raw_imu_residual_and_jacobian(state_i, state_j, imu) -> tuple[np.ndarray, np.ndarray]:
    zero = torch.zeros(PAIR_DOF, dtype=torch.float64, requires_grad=True)

    def residual(increment: torch.Tensor) -> torch.Tensor:
        candidate_i = retract_state(state_i, increment[:STATE_DOF])
        candidate_j = retract_state(state_j, increment[STATE_DOF:])
        return vio_preintegrated_imu_residual(
            from_pose=pp.SE3(candidate_i.pose_WB),
            to_pose=pp.SE3(candidate_j.pose_WB),
            prev_velocity_world=candidate_i.velocity_W,
            curr_velocity_world=candidate_j.velocity_W,
            delta_R=imu.delta_rotation,
            delta_v=imu.delta_velocity,
            delta_p=imu.delta_position,
            dt_total=imu.dt,
            prev_acc_bias=candidate_i.acc_bias,
            prev_gyro_bias=candidate_i.gyro_bias,
            curr_acc_bias=candidate_j.acc_bias,
            curr_gyro_bias=candidate_j.gyro_bias,
            linearized_acc_bias=imu.linearized_acc_bias,
            linearized_gyro_bias=imu.linearized_gyro_bias,
            bias_jacobian=imu.bias_jacobian,
            gravity_world=imu.gravity_world,
            gravity_handling=imu.gravity_handling,
        ).reshape(9)

    value = residual(zero)
    jacobian = torch.autograd.functional.jacobian(
        residual, zero, create_graph=False, vectorize=True
    )
    return value.detach().cpu().numpy(), jacobian.detach().cpu().numpy()


def _pose_jacobian_to_gtsam(jacobian_tr: np.ndarray) -> np.ndarray:
    return np.asarray(jacobian_tr, dtype=np.float64) @ POSE_PERMUTATION_TR_TO_RT


def write_states(path: Path, archive) -> None:
    header = [
        "local_index", "frame", "timestamp_ns",
        "tx", "ty", "tz", "qx", "qy", "qz", "qw",
        "vx", "vy", "vz",
        "ba_x", "ba_y", "ba_z", "bg_x", "bg_y", "bg_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for local_index, (frame, timestamp, state) in enumerate(
            zip(archive.frame_indices, archive.timestamps_ns, archive.online_states)
        ):
            writer.writerow(
                [int(local_index), int(frame), int(timestamp)]
                + _normalized_pose(state.pose_WB).reshape(7).cpu().numpy().tolist()
                + state.velocity_W.detach().cpu().numpy().tolist()
                + state.acc_bias.detach().cpu().numpy().tolist()
                + state.gyro_bias.detach().cpu().numpy().tolist()
            )


def write_edges(path: Path, archive) -> None:
    rows: list[list[float | int]] = []
    header: list[str] | None = None
    for local_edge, edge in enumerate(archive.edges):
        state_i = _normalized_state(archive.online_states[local_edge])
        state_j = _normalized_state(archive.online_states[local_edge + 1])
        imu = edge.imu
        normalized_visual = replace(
            edge.visual,
            reference_relative_CjCi=_normalized_pose(
                edge.visual.reference_relative_CjCi
            ),
            extrinsic_CI=_normalized_pose(edge.visual.extrinsic_CI),
        )
        visual_residual, visual_jacobian = _linearized_visual_residual_and_analytic_jacobian(
            state_i, state_j, normalized_visual
        )
        raw_imu_residual, raw_imu_jacobian = _raw_imu_residual_and_jacobian(
            state_i, state_j, imu
        )

        rank = int(edge.visual.sqrt_information.shape[0])
        visual_a = np.zeros((6, 6), dtype=np.float64)
        visual_c = np.zeros(6, dtype=np.float64)
        expected_visual_residual = np.zeros(6, dtype=np.float64)
        expected_visual_j_i = np.zeros((6, 6), dtype=np.float64)
        expected_visual_j_j = np.zeros((6, 6), dtype=np.float64)
        visual_a[:rank] = edge.visual.sqrt_information.detach().cpu().numpy()
        visual_c[:rank] = edge.visual.residual_offset.detach().cpu().numpy()
        expected_visual_residual[:rank] = visual_residual.detach().cpu().numpy()
        expected_visual_j_i[:rank] = _pose_jacobian_to_gtsam(
            visual_jacobian[:, :6].detach().cpu().numpy()
        )
        expected_visual_j_j[:rank] = _pose_jacobian_to_gtsam(
            visual_jacobian[:, STATE_DOF : STATE_DOF + 6].detach().cpu().numpy()
        )

        gravity = (
            np.zeros(3, dtype=np.float64)
            if imu.gravity_world is None
            else imu.gravity_world.detach().cpu().numpy()
        )
        values: list[float | int] = [
            local_edge,
            local_edge + 1,
            int(edge.frame_i),
            int(edge.frame_j),
            float(imu.dt),
            int(imu.gravity_world is not None),
            rank,
        ]
        names = [
            "local_i", "local_j", "frame_i", "frame_j", "dt",
            "gravity_in_residual", "visual_rank",
        ]

        fields = [
            ("delta_rotation_vector", _rotation_vector(imu.delta_rotation)),
            ("delta_velocity", imu.delta_velocity.detach().cpu().numpy()),
            ("delta_position", imu.delta_position.detach().cpu().numpy()),
            ("imu_covariance", imu.covariance.detach().cpu().numpy()),
            ("bias_jacobian", imu.bias_jacobian.detach().cpu().numpy()),
            ("linearized_acc_bias", imu.linearized_acc_bias.detach().cpu().numpy()),
            ("linearized_gyro_bias", imu.linearized_gyro_bias.detach().cpu().numpy()),
            ("bias_rw_covariance", imu.bias_rw_covariance.detach().cpu().numpy()),
            ("gravity_world", gravity),
            ("visual_reference_CjCi", normalized_visual.reference_relative_CjCi.detach().reshape(7).cpu().numpy()),
            ("visual_A", visual_a),
            ("visual_c", visual_c),
            ("visual_H", edge.cached_hessian.detach().cpu().numpy()),
            ("visual_g", edge.cached_gradient.detach().cpu().numpy()),
            ("expected_visual_residual", expected_visual_residual),
            ("expected_visual_J_i", expected_visual_j_i),
            ("expected_visual_J_j", expected_visual_j_j),
            ("expected_imu_residual", raw_imu_residual),
            ("expected_imu_J_Xi", _pose_jacobian_to_gtsam(raw_imu_jacobian[:, :6])),
            ("expected_imu_J_Vi", raw_imu_jacobian[:, 6:9]),
            ("expected_imu_J_Bi", raw_imu_jacobian[:, 9:15]),
            ("expected_imu_J_Xj", _pose_jacobian_to_gtsam(raw_imu_jacobian[:, STATE_DOF : STATE_DOF + 6])),
            ("expected_imu_J_Vj", raw_imu_jacobian[:, STATE_DOF + 6 : STATE_DOF + 9]),
        ]
        for prefix, value in fields:
            field_names, field_values = _flatten(prefix, np.asarray(value))
            names.extend(field_names)
            values.extend(field_values)
        if header is None:
            header = names
        elif names != header:
            raise RuntimeError("edge bundle schema changed between rows")
        rows.append(values)

    if header is None:
        raise RuntimeError("cannot export an empty PACE-VIO edge range")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = load_t2_history_archive(
        args.tensor_map,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    write_states(output / "states.csv", archive)
    write_edges(output / "edges.csv", archive)

    prior_sigma = torch.diag(archive.initial_prior.sqrt_information).reciprocal()
    with (output / "contract.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["extrinsic_tx", "extrinsic_ty", "extrinsic_tz",
             "extrinsic_qx", "extrinsic_qy", "extrinsic_qz", "extrinsic_qw"]
            + [f"prior_sigma_{index}" for index in range(STATE_DOF)]
        )
        writer.writerow(
            _normalized_pose(archive.extrinsic_CI).reshape(7).cpu().numpy().tolist()
            + prior_sigma.detach().cpu().numpy().tolist()
        )
    metadata = {
        "schema_version": 1,
        "source_tensor_map": str(archive.source_path),
        "source_tensor_map_sha256": archive.source_sha256,
        "frame_start": int(archive.frame_indices[0]),
        "frame_end": int(archive.frame_indices[-1]),
        "state_count": len(archive.online_states),
        "edge_count": len(archive.edges),
        "optimizer_world": "MACVO internal NED",
        "reference_point": "IMU origin",
        "t2_pose_tangent": "right [translation, rotation]",
        "gtsam_pose_tangent": "right [rotation, translation]",
        "imu_residual_order": "[position, velocity, rotation]",
        "bias_order": "[accelerometer, gyroscope]",
        "extrinsic_CI_xyzw": _normalized_pose(archive.extrinsic_CI).reshape(7).cpu().numpy().tolist(),
        "initial_prior_sigma_t2_order": prior_sigma.detach().cpu().numpy().tolist(),
        "compressed_factor_equivalence": compressed_factor_equivalence(archive),
        "contains_ground_truth": False,
        "source_archive_modified": False,
        "rotation_import_policy": "unit-normalize quaternion before constructing GTSAM Rot3",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

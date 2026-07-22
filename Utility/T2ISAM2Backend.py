from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

import torch

from Utility.T2FactorPacket import T2FactorPacket
from Utility.TwoStateVIO import NavigationState


def _load_extension():
    try:
        return importlib.import_module("t2_isam2_backend")
    except ModuleNotFoundError as original_error:
        project_root = Path(__file__).resolve().parents[1]
        module_dir = project_root / "build" / "t2_isam2" / "python"
        if module_dir.is_dir():
            sys.path.insert(0, str(module_dir))
            try:
                return importlib.import_module("t2_isam2_backend")
            except ModuleNotFoundError:
                pass
        raise RuntimeError(
            "T2 iSAM2 extension is not built. Run "
            "'cmake -S cpp/t2_isam2 -B build/t2_isam2 "
            "-DPython_EXECUTABLE=$CONDA_PREFIX/bin/python && "
            "cmake --build build/t2_isam2 -j'."
        ) from original_error


@dataclass(frozen=True)
class T2ISAM2Update:
    previous_state: NavigationState
    state: NavigationState
    frame_idx: int
    local_index: int
    update_ms: float
    imu_cost: float
    bias_cost: float
    visual_cost: float
    total_edge_cost: float
    initial_pose_mismatch_norm: float
    initial_velocity_mismatch_norm: float
    initial_bias_mismatch_norm: float


def _navigation_state(payload: dict[str, Any]) -> NavigationState:
    return NavigationState(
        pose_WB=torch.as_tensor(payload["pose_WB"], dtype=torch.float64).reshape(1, 7),
        velocity_W=torch.as_tensor(payload["velocity_W"], dtype=torch.float64).reshape(3),
        acc_bias=torch.as_tensor(payload["acc_bias"], dtype=torch.float64).reshape(3),
        gyro_bias=torch.as_tensor(payload["gyro_bias"], dtype=torch.float64).reshape(3),
    )


class IncrementalT2ISAM2Backend:
    """In-process iSAM2 consumer for the exact T2FactorPacket contract."""

    def __init__(
        self,
        *,
        initial_prior_std: dict[str, float],
        relinearize_threshold: float = 0.01,
        relinearize_skip: int = 1,
        covariance_floor: float = 1.0e-12,
    ) -> None:
        self._extension = _load_extension()
        self._configuration = {
            "relinearize_threshold": float(relinearize_threshold),
            "relinearize_skip": int(relinearize_skip),
            "covariance_floor": float(covariance_floor),
        }
        self._initial_prior_sigma = torch.tensor(
            [float(initial_prior_std["pose_translation_std"])] * 3
            + [float(initial_prior_std["pose_rotation_std"])] * 3
            + [float(initial_prior_std["velocity_std"])] * 3
            + [float(initial_prior_std["acc_bias_std"])] * 3
            + [float(initial_prior_std["gyro_bias_std"])] * 3,
            dtype=torch.float64,
        ).numpy()
        if not bool(torch.as_tensor(self._initial_prior_sigma > 0.0).all()):
            raise ValueError("all T2 iSAM2 initial prior standard deviations must be positive")
        self._backend = self._make_backend()

    def _make_backend(self):
        return self._extension.T2ISAM2Backend(
            self._configuration["relinearize_threshold"],
            self._configuration["relinearize_skip"],
            self._configuration["covariance_floor"],
        )

    @property
    def initialized(self) -> bool:
        return bool(self._backend.initialized)

    @property
    def state_count(self) -> int:
        return int(self._backend.state_count)

    @property
    def latest_frame(self) -> int | None:
        return int(self._backend.latest_frame) if self.initialized else None

    def reset(self) -> None:
        self._backend = self._make_backend()

    def consume(self, packet: T2FactorPacket) -> T2ISAM2Update:
        packet.validate()
        payload = packet.incremental_payload()
        if not self.initialized:
            self._backend.reset(payload, self._initial_prior_sigma)
        elif self.latest_frame != packet.frame_i:
            raise ValueError(
                "T2 iSAM2 packet discontinuity: "
                f"latest={self.latest_frame}, incoming={packet.frame_i}->{packet.frame_j}"
            )
        raw = self._backend.add_edge(payload)
        return T2ISAM2Update(
            previous_state=_navigation_state(raw["previous_state"]),
            state=_navigation_state(raw),
            frame_idx=int(raw["frame_idx"]),
            local_index=int(raw["local_index"]),
            update_ms=float(raw["update_ms"]),
            imu_cost=float(raw["imu_cost"]),
            bias_cost=float(raw["bias_cost"]),
            visual_cost=float(raw["visual_cost"]),
            total_edge_cost=float(raw["total_edge_cost"]),
            initial_pose_mismatch_norm=float(raw["initial_pose_mismatch_norm"]),
            initial_velocity_mismatch_norm=float(
                raw["initial_velocity_mismatch_norm"]
            ),
            initial_bias_mismatch_norm=float(raw["initial_bias_mismatch_norm"]),
        )

    def history(self) -> list[tuple[int, NavigationState]]:
        return [
            (int(raw["frame_idx"]), _navigation_state(raw))
            for raw in self._backend.history()
        ]

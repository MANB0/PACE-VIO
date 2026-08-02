from __future__ import annotations

import ctypes
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import torch

from Utility.PACEFactorPacket import PACEFactorPacket
from Utility.TwoStateVIO import NavigationState


def _preload_project_local_gtsam(project_root: Path) -> None:
    """Make a relocatable project-local GTSAM build visible to the extension."""
    library_dir = project_root / ".deps" / "gtsam-install" / "lib"
    if not library_dir.is_dir():
        return

    load_mode = getattr(ctypes, "RTLD_GLOBAL", getattr(os, "RTLD_GLOBAL", 0))
    for library_name in (
        "libmetis-gtsam.so",
        "libcephes-gtsam.so.1",
        "libgtsam.so.4",
    ):
        library_path = library_dir / library_name
        if not library_path.exists():
            continue
        try:
            ctypes.CDLL(str(library_path), mode=load_mode)
        except OSError:
            # The extension import below reports the complete missing dependency.
            continue


def _load_extension():
    project_root = Path(__file__).resolve().parents[1]
    _preload_project_local_gtsam(project_root)
    try:
        return importlib.import_module("pace_vio_isam2_backend")
    except ModuleNotFoundError as original_error:
        module_dirs = (
            project_root / "build" / "pace_vio_isam2" / "python",
            project_root / "build" / "t2_isam2" / "python",
        )
        for module_dir in module_dirs:
            if module_dir.is_dir() and str(module_dir) not in sys.path:
                sys.path.insert(0, str(module_dir))
        try:
            return importlib.import_module("pace_vio_isam2_backend")
        except ModuleNotFoundError:
            try:
                return importlib.import_module("t2_isam2_backend")
            except ModuleNotFoundError:
                pass
        raise RuntimeError(
            "PACE-VIO iSAM2 extension is not built. Run "
            "'bash Scripts/build_pace_vio_isam2.sh'."
        ) from original_error


@dataclass(frozen=True)
class PACEISAM2Update:
    previous_state: NavigationState
    state: NavigationState
    frame_idx: int
    local_index: int
    update_ms: float
    imu_cost: float
    bias_cost: float
    visual_cost: float
    velocity_prior_cost: float
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


class IncrementalPACEISAM2Backend:
    """In-process iSAM2 consumer for native Pose, UVD and PACE packets."""

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
            raise ValueError(
                "all PACE-VIO iSAM2 initial prior standard deviations must be positive"
            )
        self._backend = self._make_backend()

    def _make_backend(self):
        backend_type = getattr(
            self._extension,
            "PACEISAM2Backend",
            getattr(self._extension, "T2ISAM2Backend", None),
        )
        if backend_type is None:
            raise RuntimeError("PACE-VIO iSAM2 extension exports no supported backend")
        return backend_type(
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

    def consume(
        self,
        packet: PACEFactorPacket,
        *,
        velocity_prior_mean_W: torch.Tensor | None = None,
        velocity_prior_covariance_W: torch.Tensor | None = None,
    ) -> PACEISAM2Update:
        packet.validate()
        payload = packet.incremental_payload()
        if (velocity_prior_mean_W is None) != (
            velocity_prior_covariance_W is None
        ):
            raise ValueError(
                "velocity prior requires both mean and covariance"
            )
        if velocity_prior_mean_W is not None:
            mean = torch.as_tensor(
                velocity_prior_mean_W, dtype=torch.float64
            ).reshape(3)
            covariance = torch.as_tensor(
                velocity_prior_covariance_W, dtype=torch.float64
            ).reshape(3, 3)
            if not bool(torch.isfinite(mean).all()) or not bool(
                torch.isfinite(covariance).all()
            ):
                raise ValueError("velocity prior contains NaN/Inf")
            payload["velocity_prior_mean_W"] = mean.numpy()
            payload["velocity_prior_covariance_W"] = covariance.numpy()
        if not self.initialized:
            self._backend.reset(payload, self._initial_prior_sigma)
        elif self.latest_frame != packet.frame_i:
            raise ValueError(
                "PACE-VIO iSAM2 packet discontinuity: "
                f"latest={self.latest_frame}, incoming={packet.frame_i}->{packet.frame_j}"
            )
        raw = self._backend.add_edge(payload)
        return PACEISAM2Update(
            previous_state=_navigation_state(raw["previous_state"]),
            state=_navigation_state(raw),
            frame_idx=int(raw["frame_idx"]),
            local_index=int(raw["local_index"]),
            update_ms=float(raw["update_ms"]),
            imu_cost=float(raw["imu_cost"]),
            bias_cost=float(raw["bias_cost"]),
            visual_cost=float(raw["visual_cost"]),
            velocity_prior_cost=float(raw["velocity_prior_cost"]),
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


# Backward-compatible API for archived callers.
T2ISAM2Update = PACEISAM2Update
IncrementalT2ISAM2Backend = IncrementalPACEISAM2Backend

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class IMUSamplingMap:
    """Linear map from independent CSV samples to the returned interval knots."""

    raw_indices: torch.Tensor
    raw_time_ns: torch.Tensor
    knot_from_raw: torch.Tensor


class IMUCSVLoader:
    """Load IMU CSV rows and provide exact interval endpoint interpolation."""

    def __init__(self, csv_path: Path) -> None:
        assert csv_path.exists(), f"IMU csv file does not exist: {csv_path}"
        raw = np.genfromtxt(str(csv_path), delimiter=",", names=True)
        assert raw.size > 0, f"No IMU rows in {csv_path}"

        self.time_ns = torch.from_numpy(
            raw[self._pick_field(raw.dtype.names, ["timestamp", "time_ns", "time"])]
        ).long()
        gx = raw[self._pick_field(raw.dtype.names, ["ang_vel_x", "gyro_x", "wx"])]
        gy = raw[self._pick_field(raw.dtype.names, ["ang_vel_y", "gyro_y", "wy"])]
        gz = raw[self._pick_field(raw.dtype.names, ["ang_vel_z", "gyro_z", "wz"])]
        ax = raw[self._pick_field(raw.dtype.names, ["lin_acc_x", "acc_x", "ax"])]
        ay = raw[self._pick_field(raw.dtype.names, ["lin_acc_y", "acc_y", "ay"])]
        az = raw[self._pick_field(raw.dtype.names, ["lin_acc_z", "acc_z", "az"])]

        self.gyro = torch.from_numpy(np.stack([gx, gy, gz], axis=-1)).float()
        self.acc = torch.from_numpy(np.stack([ax, ay, az], axis=-1)).float()

    @staticmethod
    def _pick_field(fields: tuple[str, ...], candidates: list[str]) -> str:
        field_set = {field.lower(): field for field in fields}
        for candidate in candidates:
            if candidate.lower() in field_set:
                return field_set[candidate.lower()]
        raise KeyError(f"Cannot find any of {candidates} in csv fields {fields}")

    def query_range(
        self,
        start_ns: int,
        end_ns: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if end_ns < start_ns:
            start_ns, end_ns = end_ns, start_ns

        if self.time_ns.numel() == 0:
            return self.time_ns[:0], self.acc[:0], self.gyro[:0]

        first_ns = int(self.time_ns[0].item())
        last_ns = int(self.time_ns[-1].item())
        if end_ns < first_ns or start_ns > last_ns:
            return self.time_ns[:0], self.acc[:0], self.gyro[:0]

        start_ns = max(int(start_ns), first_ns)
        end_ns = min(int(end_ns), last_ns)

        start_time, start_acc, start_gyro = self._interpolate_sample(start_ns)
        if start_ns == end_ns:
            return start_time, start_acc, start_gyro

        i0 = int(
            torch.searchsorted(self.time_ns, torch.tensor(start_ns), right=True).item()
        )
        i1 = int(
            torch.searchsorted(self.time_ns, torch.tensor(end_ns), right=False).item()
        )
        i0 = max(0, min(i0, self.time_ns.numel()))
        i1 = max(0, min(i1, self.time_ns.numel()))

        end_time, end_acc, end_gyro = self._interpolate_sample(end_ns)
        return (
            torch.cat([start_time, self.time_ns[i0:i1], end_time], dim=0),
            torch.cat([start_acc, self.acc[i0:i1], end_acc], dim=0),
            torch.cat([start_gyro, self.gyro[i0:i1], end_gyro], dim=0),
        )

    def query_range_with_sampling_map(
        self,
        start_ns: int,
        end_ns: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, IMUSamplingMap | None]:
        """Return the production interval and its exact endpoint interpolation map.

        ``knot_from_raw[k, m]`` is the linear weight of raw CSV sample ``m``
        in returned knot ``k``. Interior knots are one-hot rows; interpolated
        camera-time endpoints have the two usual linear interpolation weights.
        """
        knot_time_ns, knot_acc, knot_gyro = self.query_range(start_ns, end_ns)
        if knot_time_ns.numel() == 0:
            return knot_time_ns, knot_acc, knot_gyro, None

        supports = [self._interpolation_support(int(value.item())) for value in knot_time_ns]
        raw_indices = torch.tensor(
            sorted({index for support in supports for index, _ in support}),
            dtype=torch.long,
        )
        local_index = {int(index): column for column, index in enumerate(raw_indices.tolist())}
        knot_from_raw = torch.zeros(
            (len(supports), raw_indices.numel()), dtype=torch.float64
        )
        for row, support in enumerate(supports):
            for index, weight in support:
                knot_from_raw[row, local_index[index]] += float(weight)
        return knot_time_ns, knot_acc, knot_gyro, IMUSamplingMap(
            raw_indices=raw_indices,
            raw_time_ns=self.time_ns[raw_indices].clone(),
            knot_from_raw=knot_from_raw,
        )

    def _interpolation_support(self, target_ns: int) -> list[tuple[int, float]]:
        index = int(
            torch.searchsorted(
                self.time_ns,
                torch.tensor(int(target_ns)),
                right=False,
            ).item()
        )
        if index < self.time_ns.numel() and int(self.time_ns[index].item()) == int(target_ns):
            return [(index, 1.0)]
        if index <= 0:
            return [(0, 1.0)]
        if index >= self.time_ns.numel():
            return [(self.time_ns.numel() - 1, 1.0)]
        left = index - 1
        right = index
        left_ns = int(self.time_ns[left].item())
        right_ns = int(self.time_ns[right].item())
        alpha = (int(target_ns) - left_ns) / float(right_ns - left_ns)
        return [(left, 1.0 - alpha), (right, alpha)]

    def _interpolate_sample(
        self,
        target_ns: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(
            torch.searchsorted(
                self.time_ns,
                torch.tensor(int(target_ns)),
                right=False,
            ).item()
        )
        if idx < self.time_ns.numel() and int(self.time_ns[idx].item()) == int(target_ns):
            return (
                self.time_ns[idx : idx + 1],
                self.acc[idx : idx + 1],
                self.gyro[idx : idx + 1],
            )
        if idx <= 0:
            return self.time_ns[:1], self.acc[:1], self.gyro[:1]
        if idx >= self.time_ns.numel():
            return self.time_ns[-1:], self.acc[-1:], self.gyro[-1:]

        left = idx - 1
        right = idx
        left_ns = int(self.time_ns[left].item())
        right_ns = int(self.time_ns[right].item())
        alpha = (int(target_ns) - left_ns) / float(right_ns - left_ns)
        acc = self.acc[left : left + 1] + (
            self.acc[right : right + 1] - self.acc[left : left + 1]
        ) * alpha
        gyro = self.gyro[left : left + 1] + (
            self.gyro[right : right + 1] - self.gyro[left : left + 1]
        ) * alpha
        return self.time_ns.new_tensor([int(target_ns)]), acc, gyro

    def query_nearest(
        self,
        target_ns: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(
            torch.searchsorted(self.time_ns, torch.tensor(target_ns), right=False).item()
        )
        if idx <= 0:
            nearest = 0
        elif idx >= self.time_ns.numel():
            nearest = self.time_ns.numel() - 1
        else:
            left = idx - 1
            right = idx
            nearest = (
                left
                if abs(int(self.time_ns[left].item()) - target_ns)
                <= abs(int(self.time_ns[right].item()) - target_ns)
                else right
            )

        return (
            nearest,
            self.time_ns[nearest : nearest + 1],
            self.acc[nearest : nearest + 1],
            self.gyro[nearest : nearest + 1],
        )

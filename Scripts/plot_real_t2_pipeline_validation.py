"""Plot the real-data T2 serial/pipeline audit in the IMU-center frame."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in (
            "serial_x", "serial_y", "serial_z",
            "pipeline_x", "pipeline_y", "pipeline_z",
            "gt_imu_x", "gt_imu_y", "gt_imu_z",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, x_name, y_name, title, xlabel, ylabel in (
        (axes[0], "*_x", "*_y", "XY", "x / m (NWU)", "y / m (NWU)"),
        (axes[1], "*_x", "*_z", "XZ", "x / m (NWU)", "z / m (NWU)"),
        (axes[2], "*_y", "*_z", "YZ", "y / m (NWU)", "z / m (NWU)"),
    ):
        suffix_x = x_name[1:]
        suffix_y = y_name[1:]
        ax.plot(data["gt_imu" + suffix_x], data["gt_imu" + suffix_y], color="#1f2937", linewidth=2.3, label="GT (IMU center)")
        ax.plot(data["serial" + suffix_x], data["serial" + suffix_y], color="#2563eb", linewidth=1.4, label="T2 serial (IMU center)")
        ax.plot(data["pipeline" + suffix_x], data["pipeline" + suffix_y], color="#059669", linewidth=1.4, label="T2 pipeline (IMU center)")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.axis("equal")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Real images + raw IMU: T2 serial vs pipeline, all trajectories at IMU center")
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()

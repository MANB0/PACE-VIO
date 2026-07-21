#!/usr/bin/env python3
"""Export DUA/BAGF fusion diagnostics from a MACVO result folder.

The optimizer stores compact per-frame diagnostics in tensor_map.npz so paper
figures can show when the method enabled IMU constraints and why.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = [
    "time_ns",
    "fusion_visual_quality",
    "fusion_degrade_score",
    "fusion_trans_switch",
    "fusion_rot_switch",
    "fusion_xy_weight",
    "fusion_z_weight",
    "fusion_rot_weight",
]


def load_array(data: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    npz_key = f"frames//{key}"
    if npz_key in data:
        arr = data[npz_key]
    else:
        arr = np.full((n,), np.nan, dtype=np.float64)
    if arr.ndim > 1 and arr.shape[-1] == 1:
        arr = arr.reshape(-1)
    return arr


def export_csv(result_dir: Path, csv_out: Path) -> dict:
    tensor_map = result_dir / "tensor_map.npz"
    if not tensor_map.exists():
        raise FileNotFoundError(f"Missing {tensor_map}")

    with np.load(tensor_map) as data:
        n = int(data["frames//pose"].shape[0])
        arrays = {field: load_array(data, field, n) for field in FIELDS}
        flags = data["frames//fusion_gate_flags"] if "frames//fusion_gate_flags" in data else np.zeros((n, 4))

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS + ["gate_active", "gate_xy", "gate_z", "gate_rot"])
        for i in range(n):
            writer.writerow(
                [arrays[field][i].item() if hasattr(arrays[field][i], "item") else arrays[field][i] for field in FIELDS]
                + [float(flags[i, j]) for j in range(4)]
            )

    active = flags[:, 0] > 0.5 if flags.size else np.zeros((n,), dtype=bool)
    summary = {
        "result_dir": str(result_dir),
        "num_frames": n,
        "num_active_frames": int(active.sum()),
        "active_ratio": float(active.mean()) if n > 0 else 0.0,
        "csv": str(csv_out),
    }

    log_path = result_dir / "fusion_log.json"
    if log_path.exists():
        try:
            logs = json.loads(log_path.read_text(encoding="utf-8"))
            summary["num_fusion_log_entries"] = len(logs)
            summary["num_skipped_entries"] = sum(1 for item in logs if item.get("skipped"))
        except json.JSONDecodeError:
            summary["fusion_log_parse_error"] = str(log_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MACVO fusion diagnostics to CSV")
    parser.add_argument("result_dir", type=Path, help="Result folder containing tensor_map.npz")
    parser.add_argument("--csv_out", type=Path, default=None)
    parser.add_argument("--summary_out", type=Path, default=None)
    args = parser.parse_args()

    csv_out = args.csv_out or (args.result_dir / "fusion_diagnostics.csv")
    summary = export_csv(args.result_dir, csv_out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

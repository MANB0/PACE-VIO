#!/usr/bin/env python3
"""Migrate legacy HoloOcean socket translations to one full 4x4 T_CI."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def migrate(path: Path, *, backup: bool = True) -> np.ndarray:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    extrinsics = metadata.get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise ValueError(f"{path}: missing extrinsics object")

    existing = extrinsics.get("T_CI")
    if existing is not None:
        matrix = np.asarray(existing, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"{path}: existing T_CI is not 4x4")
    else:
        try:
            t_BI = np.asarray(
                extrinsics["T_body_imu"]["translation_body_nwu_m"],
                dtype=np.float64,
            )
            t_BC = np.asarray(
                extrinsics["T_body_camera"]["translation_body_nwu_m"],
                dtype=np.float64,
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"{path}: expected legacy T_body_imu and T_body_camera translations"
            ) from error
        if t_BI.shape != (3,) or t_BC.shape != (3,):
            raise ValueError(f"{path}: legacy translations must each contain three values")

        # Raw IMU I is FLU; MACVO camera C is FRD/NED. The lever arm from camera
        # origin to IMU origin is first computed in body FLU and then expressed in C.
        rotation_CI = np.diag([1.0, -1.0, -1.0])
        translation_CI = rotation_CI @ (t_BI - t_BC)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation_CI
        matrix[:3, 3] = translation_CI

    if backup:
        backup_path = path.with_name("metadata.pre_t_ci_4x4.json")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    metadata["extrinsics"] = {"T_CI": matrix.tolist()}
    metadata.pop("coordinate_convention", None)
    metadata.pop("time_synchronization", None)
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="dataset root or a directory containing datasets")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    paths = [root / "metadata.json"] if (root / "metadata.json").exists() else sorted(root.rglob("metadata.json"))
    if not paths:
        raise FileNotFoundError(f"no metadata.json found under {root}")
    for path in paths:
        matrix = migrate(path, backup=not args.no_backup)
        print(f"{path}: T_CI={matrix.tolist()}")


if __name__ == "__main__":
    main()

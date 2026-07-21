#!/usr/bin/env python3
"""Check integrity of new HoloOcean dataset (batch_20260515_155653 format).

Usage:
    python Scripts/check_new_holoocean_dataset.py /path/to/batch_root
    python Scripts/check_new_holoocean_dataset.py /path/to/batch_root/scene_name
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


REQUIRED_ENTRIES = ["left", "right", "ref_pose.csv", "imu_data.csv", "metadata.json"]


def check_quat_normalized(qx, qy, qz, qw) -> float:
    return (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5


def check_scene(scene_path: Path) -> dict:
    name = scene_path.name
    result = {"name": name, "status": "UNKNOWN", "checks": []}

    # ── Required entries ──────────────────────────────────────────────
    for entry in REQUIRED_ENTRIES:
        p = scene_path / entry
        ok = p.exists() and (not entry.endswith("/") or p.is_dir())
        result["checks"].append(f"{'✅' if ok else '❌'} {entry}")
        if not ok:
            result["status"] = "INCOMPLETE"

    if result["status"] == "INCOMPLETE":
        return result

    left_dir = scene_path / "left"
    right_dir = scene_path / "right"
    ref_csv = scene_path / "ref_pose.csv"
    imu_csv = scene_path / "imu_data.csv"
    meta_json = scene_path / "metadata.json"

    # ── Image counts ──────────────────────────────────────────────────
    n_left = len(list(left_dir.glob("*.png")))
    n_right = len(list(right_dir.glob("*.png")))
    match = n_left == n_right
    result["checks"].append(f"{'✅' if match else '❌'} images: left={n_left} right={n_right}")
    if not match:
        result["status"] = "FAIL"

    # ── ref_pose.csv ──────────────────────────────────────────────────
    try:
        ref_data = np.genfromtxt(str(ref_csv), delimiter=",", names=True, dtype=None, encoding="utf-8")
        n_ref = len(ref_data)
        ref_close = abs(n_ref - n_left) <= 1
        result["checks"].append(f"{'✅' if ref_close else '⚠️'} ref_pose rows={n_ref} vs images={n_left}")
        ts_ref = ref_data["timestamp"].astype(np.int64)
        ts_mono = np.all(np.diff(ts_ref) > 0)
        result["checks"].append(f"{'✅' if ts_mono else '❌'} ref_pose timestamps monotonic")
        # Check quaternion if available
        has_quat = all(c in ref_data.dtype.names for c in ["qx", "qy", "qz", "qw"])
        if has_quat:
            q_norms = check_quat_normalized(
                ref_data["qx"].astype(float)[:100],
                ref_data["qy"].astype(float)[:100],
                ref_data["qz"].astype(float)[:100],
                ref_data["qw"].astype(float)[:100],
            )
            q_ok = np.all(np.abs(q_norms - 1.0) < 0.01)
            result["checks"].append(f"{'✅' if q_ok else '⚠️'} quaternion norms ~1 (first 100)")
        else:
            result["checks"].append("ℹ️  ref_pose: positions only (no quaternion)")
    except Exception as e:
        result["checks"].append(f"❌ ref_pose.csv read error: {e}")
        result["status"] = "FAIL"

    # ── imu_data.csv ─────────────────────────────────────────────────
    try:
        imu_data = np.genfromtxt(str(imu_csv), delimiter=",", names=True, dtype=None, encoding="utf-8")
        n_imu = len(imu_data)
        ts_imu = imu_data["timestamp"].astype(np.int64)
        ts_imu_mono = np.all(np.diff(ts_imu) > 0)
        result["checks"].append(f"{'✅' if ts_imu_mono else '❌'} imu timestamps monotonic ({n_imu} samples)")
        # WARNING if IMU quaternion fields exist (should be ignored for preintegration)
        imu_has_quat = all(c in imu_data.dtype.names for c in ["qx", "qy", "qz", "qw"])
        if imu_has_quat:
            result["checks"].append("⚠️  IMU CSV contains qx,qy,qz,qw — must be IGNORED for preintegration (only ang_vel/lin_acc used)")
        # Check gyro/acc fields exist
        has_gyro = any(c in imu_data.dtype.names for c in ["ang_vel_x", "gyro_x", "wx"])
        has_acc  = any(c in imu_data.dtype.names for c in ["lin_acc_x", "acc_x", "ax"])
        result["checks"].append(f"{'✅' if has_gyro else '❌'} IMU gyro field found")
        result["checks"].append(f"{'✅' if has_acc else '❌'} IMU acc field found")
    except Exception as e:
        result["checks"].append(f"❌ imu_data.csv read error: {e}")
        result["status"] = "FAIL"

    # ── metadata.json ─────────────────────────────────────────────────
    try:
        with open(meta_json) as f:
            meta = json.load(f)
        required_sections = ["dataset", "camera", "imu", "extrinsics", "ground_truth",
                             "coordinate_convention", "time_synchronization"]
        for sec in required_sections:
            ok = sec in meta
            result["checks"].append(f"{'✅' if ok else '❌'} metadata.{sec}")
            if not ok:
                result["status"] = "WARNING"

        cam = meta.get("camera", {})
        cam_keys = ["fx", "fy", "cx", "cy", "baseline_m", "image_width", "image_height"]
        cam_ok = all(k in cam for k in cam_keys)
        result["checks"].append(f"{'✅' if cam_ok else '❌'} metadata.camera has all intrinsics keys")

        imu_meta = meta.get("imu", {})
        # Coordinate convention checks
        cc = meta.get("coordinate_convention", {})
        cc_keys = ["export_world_frame", "body_frame", "camera_frame", "imu_frame",
                    "imu_measurement_frame", "ref_pose_position_frame",
                    "R_holocean_to_export", "R_sensor_to_flu"]
        for k in cc_keys:
            v = cc.get(k, None)
            if v is None:
                result["checks"].append(f"⚠️  coordinate_convention.{k} MISSING")
            elif "identity" in str(v).lower():
                result["checks"].append(f"✅ coordinate_convention.{k} = identity")
            else:
                result["checks"].append(f"ℹ️  coordinate_convention.{k} = {str(v)[:100]}")
        # Check if FLU→NED conversion needed
        imu_frame = cc.get("imu_frame", "")
        if "FLU" in str(imu_frame):
            result["checks"].append("⚠️  IMU frame=FLU, MACVO expects NED → FLU→NED R_x(180°) required in loader")

        # Unit checks
        ts_meta = meta.get("time_synchronization", {})
        ts_unit = ts_meta.get("timestamp_unit", "?")
        result["checks"].append(f"{'✅' if ts_unit=='ns' else '⚠️'} timestamp_unit={ts_unit}")

        imu_m = meta.get("imu", {})
        acc_unit = imu_m.get("acc_unit", "?")
        gyro_unit = imu_m.get("gyro_unit", "?")
        result["checks"].append(f"{'✅' if acc_unit=='m/s^2' else '⚠️'} acc_unit={acc_unit}")
        result["checks"].append(f"{'✅' if gyro_unit=='rad/s' else '⚠️'} gyro_unit={gyro_unit}")
        # IMU keys completeness
        imu_keys = ["rate_hz", "frame", "acc_unit", "gyro_unit", "acc_includes_gravity"]
        imu_ok = all(k in imu_m for k in imu_keys)
        result["checks"].append(f"{'✅' if imu_ok else '❌'} metadata.imu has all keys")

        # Rate checks
        cam_rate = cam.get("camera_rate_hz", 30)
        imu_rate = imu_m.get("rate_hz", 300)
        if n_left > 0 and n_ref > 0:
            cam_dt_ns = float(np.diff(ts_ref).mean()) if len(ts_ref) > 1 else 0
            cam_rate_obs = 1e9 / cam_dt_ns if cam_dt_ns > 0 else 0
            cam_rate_ok = abs(cam_rate_obs - cam_rate) < 5
            result["checks"].append(f"{'✅' if cam_rate_ok else '⚠️'} camera rate obs={cam_rate_obs:.1f}Hz vs metadata={cam_rate}Hz")
        if n_imu > 0:
            imu_dt_ns = float(np.diff(ts_imu).mean()) if len(ts_imu) > 1 else 0
            imu_rate_obs = 1e9 / imu_dt_ns if imu_dt_ns > 0 else 0
            imu_rate_ok = abs(imu_rate_obs - imu_rate) < imu_rate * 0.15
            result["checks"].append(f"{'✅' if imu_rate_ok else '⚠️'} IMU rate obs={imu_rate_obs:.1f}Hz vs metadata={imu_rate}Hz")
    except Exception as e:
        result["checks"].append(f"❌ metadata.json read error: {e}")
        result["status"] = "FAIL"

    if result["status"] == "UNKNOWN":
        result["status"] = "COMPLETE"

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_new_holoocean_dataset.py <batch_root_or_scene_path>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.exists():
        print(f"❌ Path does not exist: {root}")
        sys.exit(1)

    # Determine if root is a batch root (contains scene folders) or a single scene
    if (root / "left").exists() and (root / "metadata.json").exists():
        scenes = [root]
    else:
        scenes = sorted([d for d in root.iterdir() if d.is_dir() and (d / "metadata.json").exists()])
        if not scenes:
            print(f"❌ No scene folders found under {root}")
            sys.exit(1)

    print(f"Checking {len(scenes)} scene(s) under {root}")
    print("=" * 60)

    summary = []
    for scene_path in scenes:
        result = check_scene(scene_path)
        summary.append(result)
        print(f"\n📁 {result['name']}  [{result['status']}]")
        for check in result["checks"]:
            print(f"  {check}")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    statuses = {}
    for r in summary:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    for s, c in statuses.items():
        print(f"  {s}: {c} scene(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()

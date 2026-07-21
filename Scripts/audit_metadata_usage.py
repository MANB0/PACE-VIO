#!/usr/bin/env python3
"""Audit metadata usage across macvo-dev.

Usage:
    python Scripts/audit_metadata_usage.py /path/to/scene_folder
    python Scripts/audit_metadata_usage.py /path/to/batch_root
    python Scripts/audit_metadata_usage.py --from-trace /path/to/metadata_usage_trace.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Utility.MetadataUsageTrace import generate_metadata_usage_trace


HIGH_RISK_FIELDS = [
    ("camera", "baseline_m", "Disparity depth/scale computation may not use baseline"),
    ("imu", "AngVelSigma", "IMU covariance & rot floor may miss gyro noise"),
    ("imu", "AccelSigma", "IMU covariance may miss accel noise"),
    ("imu", "acc_unit", "Accel unit not checked; if not m/s², preintegration is wrong"),
    ("imu", "gyro_unit", "Gyro unit not checked; if not rad/s, preintegration is wrong"),
    ("imu", "acc_includes_gravity", "Gravity correction assumption not verified"),
    ("imu", "gravity_m_s2", "Gravity value may conflict with config"),
    ("extrinsics", "T_imu_camera.translation", "IMU extrinsic translation may not affect residuals"),
    ("extrinsics", "T_imu_camera.rotation", "IMU extrinsic rotation (FLU→NED) missing"),
    ("coordinate_convention", "R_sensor_to_flu", "FLU→NED conversion may be missing"),
    ("time_synchronization", "camera_imu_time_offset_ns", "Time offset not applied"),
    ("time_synchronization", "timestamp_unit", "Timestamp unit mismatch risk"),
]


def audit_trace(trace: dict) -> dict:
    """Check a metadata usage trace and flag suspicious entries."""
    issues = []

    for section, field, desc in HIGH_RISK_FIELDS:
        sec = trace.get(section, {})
        entry = sec.get(field, {})
        if not entry:
            issues.append({"severity": "FAIL", "field": f"{section}.{field}", "reason": f"MISSING in trace"})
            continue
        status = entry.get("status", "?")
        used_in = entry.get("used_in", [])
        value = entry.get("value", None)

        if section == "extrinsics" and field == "T_imu_camera.translation":
            body_imu = trace.get("extrinsics", {}).get("T_body_imu", {})
            if (
                body_imu.get("status") == "USED_IN_COMPUTATION"
                and "imu_T_BS_translation" in body_imu.get("used_in", [])
            ):
                continue

        if status == "READ_ONLY" and not used_in:
            issues.append({"severity": "WARNING", "field": f"{section}.{field}",
                           "reason": f"{desc} — status=READ_ONLY, used_in=[]",
                           "value": value})
        elif status == "USED_IN_COMPUTATION":
            # OK
            pass
        else:
            issues.append({"severity": "WARNING", "field": f"{section}.{field}",
                           "reason": f"Unexpected status={status}", "value": value})

    return {"issues": issues, "warnings_from_trace": trace.get("warnings", [])}


def main():
    args = sys.argv[1:]
    if "--from-trace" in args:
        idx = args.index("--from-trace")
        trace_path = Path(args[idx + 1])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    elif len(args) >= 1:
        root = Path(args[0])
        if not root.exists():
            print(f"FAIL: path does not exist: {root}")
            sys.exit(1)

        # Determine scenes
        if (root / "metadata.json").exists():
            scenes = [root]
        else:
            scenes = sorted([d for d in root.iterdir() if d.is_dir() and (d / "metadata.json").exists()])
        if not scenes:
            print(f"FAIL: no scene folders with metadata.json found under {root}")
            sys.exit(1)

        for scene_path in scenes:
            print(f"\n{'='*60}")
            print(f"Auditing: {scene_path.name}")
            print(f"{'='*60}")

            meta = json.load(open(scene_path / "metadata.json"))
            cam = meta.get("camera")
            imu_calib = meta.get("imu")
            imu_extrinsic = meta.get("extrinsics")

            trace = generate_metadata_usage_trace(scene_path, meta, cam, imu_calib, imu_extrinsic)
            result = audit_trace(trace)

            # Print summary by section
            STATUS_ORDER = {"USED_IN_COMPUTATION": 0, "READ_ONLY": 1}
            for section in ["camera", "imu", "extrinsics", "coordinate_convention", "time_synchronization"]:
                sec_data = trace.get(section, {})
                used = [(k, v) for k, v in sec_data.items() if v.get("status") == "USED_IN_COMPUTATION"]
                read = [(k, v) for k, v in sec_data.items() if v.get("status") == "READ_ONLY"]
                print(f"\n[{section}]")
                if used:
                    print(f"  USED_IN_COMPUTATION ({len(used)}): {', '.join(k for k,_ in used)}")
                if read:
                    print(f"  READ_ONLY ({len(read)}): {', '.join(k for k,_ in read)}")

            # Issues
            if result["issues"]:
                print(f"\n⚠️  ISSUES ({len(result['issues'])}):")
                for iss in result["issues"]:
                    print(f"  [{iss['severity']}] {iss['field']}: {iss['reason']}  (value={iss.get('value')})")
            else:
                print(f"\n✅ No high-risk issues found")

            if trace.get("warnings"):
                print(f"\n📋 Warnings from trace:")
                for w in trace["warnings"]:
                    print(f"  - {w}")

    print(f"\n{'='*60}")
    print("Audit complete.")


if __name__ == "__main__":
    main()

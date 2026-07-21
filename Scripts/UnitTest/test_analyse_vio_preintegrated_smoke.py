import csv
import subprocess
import sys
from pathlib import Path

import pytest

from Scripts.analyse_vio_preintegrated_smoke import (
    evaluate_result_root,
    summarize_vio_diagnostics,
    write_outputs,
)
from Utility.FramePairDiagnostics import CSV_HEADER


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_smoke_rejects_empty_manifest(tmp_path: Path):
    result_root = tmp_path / "results"
    _write_csv(
        result_root / "run_manifest.csv",
        ["trial", "scene", "variant", "scene_root", "result_dir"],
        [],
    )

    with pytest.raises(ValueError, match="contains no runs"):
        evaluate_result_root(result_root)


def test_evaluate_smoke_manifest_reports_ate_and_missing_runs(tmp_path: Path):
    scene_root = tmp_path / "scene"
    ref_rows = [
        {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1, "vx": 0, "vy": 0, "vz": 0, "wx": 0, "wy": 0, "wz": 0},
        {"timestamp": 1, "x": 1.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1, "vx": 0, "vy": 0, "vz": 0, "wx": 0, "wy": 0, "wz": 0},
        {"timestamp": 2, "x": 2.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1, "vx": 0, "vy": 0, "vz": 0, "wx": 0, "wy": 0, "wz": 0},
    ]
    _write_csv(scene_root / "ref_pose.csv", list(ref_rows[0]), ref_rows)

    result_root = tmp_path / "results"
    complete_dir = result_root / "trial_1" / "vio_preintegrated" / "clear_shallow"
    missing_dir = result_root / "trial_1" / "pure_macvo" / "clear_shallow"
    pose_rows = [
        {"timestamp_ns": 0, "tx": 10.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 1, "tx": 11.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 2, "tx": 12.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(complete_dir / "poses.csv", list(pose_rows[0]), pose_rows)
    (complete_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    manifest_rows = [
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "vio_preintegrated",
            "scene_root": str(scene_root),
            "result_dir": str(complete_dir),
            "seq_to": 3,
            "autodiff": 1,
            "imu_factor_mode": "preintegrated_vio",
        },
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "pure_macvo",
            "scene_root": str(scene_root),
            "result_dir": str(missing_dir),
            "seq_to": 3,
            "autodiff": 0,
            "imu_factor_mode": "legacy_pose_prior",
        },
    ]
    _write_csv(result_root / "run_manifest.csv", list(manifest_rows[0]), manifest_rows)

    records = evaluate_result_root(result_root)
    by_variant = {record.variant: record for record in records}

    assert by_variant["vio_preintegrated"].status == "complete"
    assert by_variant["vio_preintegrated"].frames == 3
    assert by_variant["vio_preintegrated"].direct_raw_ate == pytest.approx(10.0)
    assert by_variant["vio_preintegrated"].direct_origin_ate == pytest.approx(0.0)
    assert by_variant["pure_macvo"].status == "missing_poses"

    outdir = tmp_path / "analysis"
    csv_path, md_path = write_outputs(records, outdir)
    assert csv_path.exists()
    assert md_path.exists()
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    by_variant_out = {row["variant"]: row for row in rows}
    assert by_variant_out["vio_preintegrated"]["imu_factor_mode"] == "preintegrated_vio"
    assert by_variant_out["vio_preintegrated"]["autodiff"] == "1"
    assert by_variant_out["vio_preintegrated"]["scene_root"] == str(scene_root)
    assert by_variant_out["vio_preintegrated"]["result_dir"] == str(complete_dir)
    md_text = md_path.read_text(encoding="utf-8")
    assert "vio_preintegrated" in md_text
    assert "preintegrated_vio" in md_text
    assert "message" in md_text
    assert "missing" in md_text


def test_evaluate_smoke_converts_ned_pose_csv_to_holoocean_nwu_gt(tmp_path: Path):
    scene_root = tmp_path / "scene"
    ref_rows = [
        {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1, "vx": 0, "vy": 0, "vz": 0, "wx": 0, "wy": 0, "wz": 0},
        {"timestamp": 1, "x": 1.0, "y": 2.0, "z": 3.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1, "vx": 0, "vy": 0, "vz": 0, "wx": 0, "wy": 0, "wz": 0},
        {"timestamp": 2, "x": 2.0, "y": 4.0, "z": 6.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1, "vx": 0, "vy": 0, "vz": 0, "wx": 0, "wy": 0, "wz": 0},
    ]
    _write_csv(scene_root / "ref_pose.csv", list(ref_rows[0]), ref_rows)

    result_root = tmp_path / "results"
    result_dir = result_root / "trial_1" / "vio_preintegrated" / "clear_shallow"
    pose_rows_ned = [
        {"timestamp_ns": 0, "tx": 0.0, "ty": -0.0, "tz": -0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 1, "tx": 1.0, "ty": -2.0, "tz": -3.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 2, "tx": 2.0, "ty": -4.0, "tz": -6.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(result_dir / "poses.csv", list(pose_rows_ned[0]), pose_rows_ned)
    (result_dir / "pose_coordinate_frame.txt").write_text("NED\n", encoding="utf-8")

    manifest_rows = [
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "vio_preintegrated",
            "scene_root": str(scene_root),
            "result_dir": str(result_dir),
            "seq_to": 3,
            "autodiff": 1,
        }
    ]
    _write_csv(result_root / "run_manifest.csv", list(manifest_rows[0]), manifest_rows)

    [record] = evaluate_result_root(result_root)

    assert record.status == "complete"
    assert record.direct_raw_ate == pytest.approx(0.0)
    assert record.direct_origin_ate == pytest.approx(0.0)


def test_evaluate_smoke_requires_pose_coordinate_frame_sidecar(tmp_path: Path):
    scene_root = tmp_path / "scene"
    ref_rows = [
        {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp": 1, "x": 1.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(scene_root / "ref_pose.csv", list(ref_rows[0]), ref_rows)

    result_root = tmp_path / "results"
    result_dir = result_root / "trial_1" / "vio_preintegrated" / "clear_shallow"
    pose_rows = [
        {"timestamp_ns": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 1, "tx": 1.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(result_dir / "poses.csv", list(pose_rows[0]), pose_rows)

    manifest_rows = [
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "vio_preintegrated",
            "scene_root": str(scene_root),
            "result_dir": str(result_dir),
            "seq_to": 2,
            "autodiff": 1,
            "imu_factor_mode": "preintegrated_vio",
        }
    ]
    _write_csv(result_root / "run_manifest.csv", list(manifest_rows[0]), manifest_rows)

    [record] = evaluate_result_root(result_root)

    assert record.status == "missing_pose_frame"
    assert "pose_coordinate_frame.txt" in record.message


def test_full_vio_smoke_requires_active_vio_factor_diagnostics(tmp_path: Path):
    scene_root = tmp_path / "scene"
    ref_rows = [
        {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp": 1, "x": 1.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(scene_root / "ref_pose.csv", list(ref_rows[0]), ref_rows)

    result_root = tmp_path / "results"
    result_dir = result_root / "trial_1" / "vio_preintegrated_full" / "clear_shallow"
    pose_rows = [
        {"timestamp_ns": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 1, "tx": 1.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(result_dir / "poses.csv", list(pose_rows[0]), pose_rows)
    (result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    manifest_rows = [
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "vio_preintegrated_full",
            "scene_root": str(scene_root),
            "result_dir": str(result_dir),
            "seq_to": 2,
            "autodiff": 1,
            "imu_factor_mode": "preintegrated_vio",
            "force_mode": "full_imu",
        }
    ]
    _write_csv(result_root / "run_manifest.csv", list(manifest_rows[0]), manifest_rows)

    [record] = evaluate_result_root(result_root)
    assert record.status == "missing_vio_diagnostics"
    assert record.diagnostics_status == "missing"

    diag_rows = [
        {"pair_id": 1, "vio_factor_active": 1, "imu_factor_mode": "preintegrated_vio", "imu_residual_rows": 3},
    ]
    _write_csv(result_dir / "frame_pair_diagnostics.csv", list(diag_rows[0]), diag_rows)

    [record] = evaluate_result_root(result_root)
    assert record.status == "missing_vio_convention_diagnostics"
    assert record.diagnostics_status == "missing_convention_columns"

    diag_rows = [
        {
            "pair_id": 1,
            "vio_factor_active": 1,
            "imu_factor_mode": "preintegrated_vio",
            "imu_residual_rows": 3,
            "imu_source_world_frame": "NWU",
            "imu_source_measurement_frame": "FLU",
            "imu_internal_world_frame": "NED",
            "imu_internal_measurement_frame": "NED",
            "imu_acc_unit": "m/s^2",
            "imu_gyro_unit": "rad/s",
            "imu_timestamp_unit": "ns",
            "imu_time_offset_ns": "0",
            "imu_time_offset_source": "metadata.time_synchronization.camera_imu_time_offset_ns",
            "imu_gravity_source": "metadata.json",
            "imu_metadata_gravity_m_s2": "9.8",
            "imu_preintegration_gravity_z": "9.8",
        },
    ]
    _write_csv(result_dir / "frame_pair_diagnostics.csv", list(diag_rows[0]), diag_rows)

    [record] = evaluate_result_root(result_root)
    assert record.status == "complete"
    assert record.force_mode == "full_imu"
    assert record.diagnostics_status == "ok"
    assert record.diagnostics_rows == 1
    assert record.vio_factor_active_rows == 1

    csv_path, md_path = write_outputs([record], tmp_path / "analysis")
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert rows[0]["force_mode"] == "full_imu"
    assert rows[0]["diagnostics_status"] == "ok"
    assert rows[0]["vio_factor_active_rows"] == "1"
    assert rows[0]["imu_time_offset_ns"] == "0"
    assert rows[0]["imu_time_offset_source"] == "metadata.time_synchronization.camera_imu_time_offset_ns"
    md_text = md_path.read_text(encoding="utf-8")
    assert "vio-active" in md_text
    assert "metadata.time_synchronization.camera_imu_time_offset_ns" in md_text


def test_full_vio_variant_requires_diagnostics_even_without_force_mode_column(tmp_path: Path):
    scene_root = tmp_path / "scene"
    ref_rows = [
        {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp": 1, "x": 1.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(scene_root / "ref_pose.csv", list(ref_rows[0]), ref_rows)

    result_root = tmp_path / "results"
    result_dir = result_root / "trial_1" / "vio_preintegrated_full" / "clear_shallow"
    pose_rows = [
        {"timestamp_ns": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 1, "tx": 1.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(result_dir / "poses.csv", list(pose_rows[0]), pose_rows)
    (result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    manifest_rows = [
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "vio_preintegrated_full",
            "scene_root": str(scene_root),
            "result_dir": str(result_dir),
            "seq_to": 2,
            "autodiff": 1,
            "imu_factor_mode": "preintegrated_vio",
        }
    ]
    _write_csv(result_root / "run_manifest.csv", list(manifest_rows[0]), manifest_rows)

    [record] = evaluate_result_root(result_root)

    assert record.status == "missing_vio_diagnostics"
    assert record.diagnostics_status == "missing"


def test_full_vio_smoke_finds_nested_macvo_output_bundle(tmp_path: Path):
    scene_root = tmp_path / "scene"
    ref_rows = [
        {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp": 1, "x": 1.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(scene_root / "ref_pose.csv", list(ref_rows[0]), ref_rows)

    result_root = tmp_path / "results"
    result_dir = result_root / "trial_1" / "vio_preintegrated_full" / "clear_shallow"
    nested_dir = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    pose_rows = [
        {"timestamp_ns": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 1, "tx": 1.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(nested_dir / "poses.csv", list(pose_rows[0]), pose_rows)
    (nested_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    diag_rows = [
        {
            "pair_id": 1,
            "vio_factor_active": 1,
            "imu_factor_mode": "preintegrated_vio",
            "imu_residual_rows": 3,
            "imu_source_world_frame": "NWU",
            "imu_source_measurement_frame": "FLU",
            "imu_internal_world_frame": "NED",
            "imu_internal_measurement_frame": "NED",
            "imu_acc_unit": "m/s^2",
            "imu_gyro_unit": "rad/s",
            "imu_timestamp_unit": "ns",
            "imu_time_offset_ns": "0",
            "imu_time_offset_source": "metadata.time_synchronization.camera_imu_time_offset_ns",
            "imu_gravity_source": "metadata.json",
            "imu_metadata_gravity_m_s2": "9.8",
            "imu_preintegration_gravity_z": "9.8",
        },
    ]
    _write_csv(nested_dir / "frame_pair_diagnostics.csv", list(diag_rows[0]), diag_rows)

    manifest_rows = [
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "vio_preintegrated_full",
            "scene_root": str(scene_root),
            "result_dir": str(result_dir),
            "seq_to": 2,
            "autodiff": 1,
            "imu_factor_mode": "preintegrated_vio",
            "force_mode": "full_imu",
        }
    ]
    _write_csv(result_root / "run_manifest.csv", list(manifest_rows[0]), manifest_rows)

    [record] = evaluate_result_root(result_root)

    assert record.status == "complete"
    assert record.poses_path == nested_dir / "poses.csv"
    assert record.diagnostics_path == nested_dir / "frame_pair_diagnostics.csv"


def test_full_vio_smoke_rejects_split_pose_and_diagnostics_bundles(tmp_path: Path):
    scene_root = tmp_path / "scene"
    ref_rows = [
        {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp": 1, "x": 1.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(scene_root / "ref_pose.csv", list(ref_rows[0]), ref_rows)

    result_root = tmp_path / "results"
    result_dir = result_root / "trial_1" / "vio_preintegrated_full" / "clear_shallow"
    pose_rows = [
        {"timestamp_ns": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 1, "tx": 1.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(result_dir / "poses.csv", list(pose_rows[0]), pose_rows)
    (result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    nested_dir = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    diag_rows = [
        {
            "pair_id": 1,
            "vio_factor_active": 1,
            "imu_factor_mode": "preintegrated_vio",
            "imu_residual_rows": 3,
            "imu_source_world_frame": "NWU",
            "imu_source_measurement_frame": "FLU",
            "imu_internal_world_frame": "NED",
            "imu_internal_measurement_frame": "NED",
            "imu_acc_unit": "m/s^2",
            "imu_gyro_unit": "rad/s",
            "imu_timestamp_unit": "ns",
            "imu_time_offset_ns": "0",
            "imu_time_offset_source": "metadata.time_synchronization.camera_imu_time_offset_ns",
            "imu_gravity_source": "metadata.json",
            "imu_metadata_gravity_m_s2": "9.8",
            "imu_preintegration_gravity_z": "9.8",
        },
    ]
    _write_csv(nested_dir / "frame_pair_diagnostics.csv", list(diag_rows[0]), diag_rows)

    manifest_rows = [
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "vio_preintegrated_full",
            "scene_root": str(scene_root),
            "result_dir": str(result_dir),
            "seq_to": 2,
            "autodiff": 1,
            "imu_factor_mode": "preintegrated_vio",
            "force_mode": "full_imu",
        }
    ]
    _write_csv(result_root / "run_manifest.csv", list(manifest_rows[0]), manifest_rows)

    [record] = evaluate_result_root(result_root)

    assert record.status == "missing_vio_diagnostics"
    assert record.diagnostics_path == result_dir / "frame_pair_diagnostics.csv"


def test_full_vio_smoke_prefers_nested_valid_bundle_over_stale_direct_invalid(tmp_path: Path):
    scene_root = tmp_path / "scene"
    ref_rows = [
        {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp": 1, "x": 1.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(scene_root / "ref_pose.csv", list(ref_rows[0]), ref_rows)

    result_root = tmp_path / "results"
    result_dir = result_root / "trial_1" / "vio_preintegrated_full" / "clear_shallow"
    pose_rows = [
        {"timestamp_ns": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        {"timestamp_ns": 1, "tx": 1.0, "ty": 0.0, "tz": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
    ]
    _write_csv(result_dir / "poses.csv", list(pose_rows[0]), pose_rows)
    (result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    _write_csv(
        result_dir / "frame_pair_diagnostics.csv",
        ["pair_id", "imu_factor_mode", "vio_factor_active", "imu_residual_rows"],
        [{"pair_id": 1, "imu_factor_mode": "preintegrated_vio", "vio_factor_active": 1, "imu_residual_rows": 3}],
    )

    nested_dir = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    _write_csv(nested_dir / "poses.csv", list(pose_rows[0]), pose_rows)
    (nested_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    diag_rows = [
        {
            "pair_id": 1,
            "vio_factor_active": 1,
            "imu_factor_mode": "preintegrated_vio",
            "imu_residual_rows": 3,
            "imu_source_world_frame": "NWU",
            "imu_source_measurement_frame": "FLU",
            "imu_internal_world_frame": "NED",
            "imu_internal_measurement_frame": "NED",
            "imu_acc_unit": "m/s^2",
            "imu_gyro_unit": "rad/s",
            "imu_timestamp_unit": "ns",
            "imu_time_offset_ns": "0",
            "imu_time_offset_source": "metadata.time_synchronization.camera_imu_time_offset_ns",
            "imu_gravity_source": "metadata.json",
            "imu_metadata_gravity_m_s2": "9.8",
            "imu_preintegration_gravity_z": "9.8",
        },
    ]
    _write_csv(nested_dir / "frame_pair_diagnostics.csv", list(diag_rows[0]), diag_rows)

    manifest_rows = [
        {
            "trial": 1,
            "scene": "clear_shallow",
            "variant": "vio_preintegrated_full",
            "scene_root": str(scene_root),
            "result_dir": str(result_dir),
            "seq_to": 2,
            "autodiff": 1,
            "imu_factor_mode": "preintegrated_vio",
            "force_mode": "full_imu",
        }
    ]
    _write_csv(result_root / "run_manifest.csv", list(manifest_rows[0]), manifest_rows)

    [record] = evaluate_result_root(result_root)

    assert record.status == "complete"
    assert record.poses_path == nested_dir / "poses.csv"
    assert record.diagnostics_path == nested_dir / "frame_pair_diagnostics.csv"


def test_vio_diagnostics_accepts_new_rows_appended_after_legacy_header(tmp_path: Path):
    legacy_header = [field for field in CSV_HEADER if field not in {"imu_factor_mode", "vio_factor_active", "imu_residual_rows"}]
    path = tmp_path / "frame_pair_diagnostics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(legacy_header)
        writer.writerow(["" for _ in legacy_header])

        row = {field: "" for field in CSV_HEADER}
        row["pair_id"] = "1"
        row["imu_factor_mode"] = "preintegrated_vio"
        row["vio_factor_active"] = "1"
        row["imu_residual_rows"] = "3"
        writer.writerow([row[field] for field in CSV_HEADER])

    status, rows, active_rows, message = summarize_vio_diagnostics(path)

    assert status == "ok"
    assert rows == 2
    assert active_rows == 1
    assert message == ""


def test_analyse_script_can_run_as_direct_entrypoint():
    result = subprocess.run(
        [sys.executable, "Scripts/analyse_vio_preintegrated_smoke.py", "--help"],
        cwd="/home/admin1/macvo-dev",
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--result-root" in result.stdout
    assert "Results/vio_preintegrated_full_smoke_30f" in result.stdout
    assert "analysis_vio_preintegrated_full_smoke_30f" in result.stdout

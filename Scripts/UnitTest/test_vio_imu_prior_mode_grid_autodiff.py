import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from Scripts import run_vio_imu_prior_mode_grid as grid


def _minimal_scene_metadata(**overrides):
    metadata = {
        "dataset": {
            "timestamp_unit": "ns",
            "simulator": "HoloOcean",
        },
        "imu": {
            "frame": "FLU",
            "acc_unit": "m/s^2",
            "gyro_unit": "rad/s",
            "acc_includes_gravity": True,
            "gravity_m_s2": 9.8,
        },
        "ground_truth": {
            "velocity_frame": "world NWU",
            "angular_velocity_frame": "body NWU",
        },
        "coordinate_convention": {
            "holocean_world_frame": "NWU",
            "export_world_frame": "NWU",
            "ref_pose_position_frame": "world NWU",
            "ref_pose_velocity_frame": "world NWU",
            "ref_pose_angular_velocity_frame": "body NWU",
            "imu_measurement_frame": "FLU",
        },
        "time_synchronization": {
            "timestamp_unit": "ns",
            "camera_imu_time_offset_ns": 0,
        },
    }
    for dotted_key, value in overrides.items():
        target = metadata
        parts = dotted_key.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return metadata


def _write_scene_files(scene_root: Path, metadata: dict) -> None:
    (scene_root / "left").mkdir(parents=True)
    (scene_root / "right").mkdir(parents=True)
    (scene_root / "imu_data.csv").write_text("timestamp,qx,qy,qz,qw,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n", encoding="utf-8")
    (scene_root / "ref_pose.csv").write_text("timestamp,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz\n", encoding="utf-8")
    (scene_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _full_vio_diagnostics_row(**overrides):
    row = {
        "pair_id": "1",
        "imu_factor_mode": "preintegrated_vio",
        "vio_factor_active": "1",
        "imu_residual_rows": "3",
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
    }
    row.update(overrides)
    return row


def _write_diagnostics_csv(path: Path, row: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_autodiff_flag_is_written_to_odom_config_and_manifest(tmp_path: Path):
    odom_cfg = grid.make_odom_cfg(grid.VARIANTS["aimvo_visualvel_s1"], tmp_path, autodiff=True)
    cfg = grid.load_yaml(odom_cfg)

    assert cfg["Odometry"]["optimizer"]["args"]["autodiff"] is True

    specs = grid.build_specs(
        scenes=["murky_coast"],
        variants=["aimvo_visualvel_s1"],
        trials=1,
        result_root=tmp_path / "results",
    )
    grid.write_manifest(tmp_path, specs, seq_to=120, autodiff=True)

    rows = list(csv.DictReader((tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["autodiff"] == "1"


def test_preintegrated_vio_variant_forces_autodiff_and_vio_factor_mode(tmp_path: Path):
    odom_cfg = grid.make_odom_cfg(grid.VARIANTS["vio_preintegrated"], tmp_path, autodiff=False)
    cfg = grid.load_yaml(odom_cfg)
    optimizer_args = cfg["Odometry"]["optimizer"]["args"]

    assert optimizer_args["autodiff"] is True
    assert optimizer_args["imu_factor_mode"] == "preintegrated_vio"
    assert optimizer_args["imu_rot_prior"] is True

    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated"],
        trials=1,
        result_root=tmp_path / "results",
    )
    grid.write_manifest(tmp_path, specs, seq_to=30, autodiff=False)
    rows = list(csv.DictReader((tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["autodiff"] == "1"
    assert rows[0]["imu_factor_mode"] == "preintegrated_vio"


def test_preintegrated_vio_full_variant_forces_full_imu_and_autodiff(tmp_path: Path):
    odom_cfg = grid.make_odom_cfg(grid.VARIANTS["vio_preintegrated_full"], tmp_path, autodiff=False)
    cfg = grid.load_yaml(odom_cfg)
    optimizer_args = cfg["Odometry"]["optimizer"]["args"]

    assert optimizer_args["autodiff"] is True
    assert optimizer_args["imu_factor_mode"] == "preintegrated_vio"
    assert grid.VARIANTS["vio_preintegrated_full"].force_mode == "full_imu"

    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated_full"],
        trials=1,
        result_root=tmp_path / "results",
    )
    grid.write_manifest(tmp_path, specs, seq_to=30, autodiff=False)
    rows = list(csv.DictReader((tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["autodiff"] == "1"
    assert rows[0]["imu_factor_mode"] == "preintegrated_vio"
    assert rows[0]["force_mode"] == "full_imu"


def test_local_inertial_ba_variant_writes_window_controls(tmp_path: Path):
    odom_cfg_w2 = grid.make_odom_cfg(grid.VARIANTS["vio_local_ba_w2_imuatt"], tmp_path, autodiff=False)
    cfg_w2 = grid.load_yaml(odom_cfg_w2)
    optimizer_args_w2 = cfg_w2["Odometry"]["optimizer"]["args"]
    assert optimizer_args_w2["imu_factor_mode"] == "local_inertial_ba"
    assert optimizer_args_w2["local_ba_window_size"] == 2

    specs_w2 = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_local_ba_w2_imuatt"],
        trials=1,
        result_root=tmp_path / "results_w2",
    )
    grid.write_manifest(tmp_path, specs_w2, seq_to=30, autodiff=False)
    rows_w2 = list(csv.DictReader((tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8")))
    assert rows_w2[0]["imu_factor_mode"] == "local_inertial_ba"
    assert rows_w2[0]["effective_imu_factor_mode"] == "local_inertial_ba"
    assert rows_w2[0]["local_ba_window_size"] == "2"
    assert rows_w2[0]["local_ba_writeback"] == "current"
    assert rows_w2[0]["local_ba_fix_first_frame"] == "1"
    assert rows_w2[0]["effective_local_ba_window_size"] == "2"
    assert rows_w2[0]["effective_local_ba_writeback"] == "current"
    assert rows_w2[0]["effective_local_ba_fix_first_frame"] == "1"

    odom_cfg = grid.make_odom_cfg(grid.VARIANTS["vio_local_ba_w5_imuatt"], tmp_path, autodiff=False)
    cfg = grid.load_yaml(odom_cfg)
    odom_args = cfg["Odometry"]["args"]
    optimizer_args = cfg["Odometry"]["optimizer"]["args"]

    assert optimizer_args["autodiff"] is True
    assert optimizer_args["imu_factor_mode"] == "local_inertial_ba"
    assert optimizer_args["local_ba_window_size"] == 5
    assert optimizer_args["local_ba_writeback"] == "current"
    assert optimizer_args["local_ba_fix_first_frame"] is True
    assert odom_args["imu_vio_gravity_pose_source"] == "imu_integrated_estinit"

    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_local_ba_w5_imuatt"],
        trials=1,
        result_root=tmp_path / "results",
    )
    grid.write_manifest(tmp_path, specs, seq_to=30, autodiff=False)
    rows = list(csv.DictReader((tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8")))

    assert rows[0]["autodiff"] == "1"
    assert rows[0]["imu_factor_mode"] == "local_inertial_ba"
    assert rows[0]["effective_imu_factor_mode"] == "local_inertial_ba"
    assert rows[0]["local_ba_window_size"] == "5"
    assert rows[0]["local_ba_writeback"] == "current"
    assert rows[0]["local_ba_fix_first_frame"] == "1"
    assert rows[0]["effective_local_ba_window_size"] == "5"
    assert rows[0]["effective_local_ba_writeback"] == "current"
    assert rows[0]["effective_local_ba_fix_first_frame"] == "1"


def test_vio_isolation_variants_write_diagnostic_controls(tmp_path: Path):
    expected = {
        "vio_preintegrated_full_gtgravity": {
            "odom": {"imu_vio_gravity_pose_source": "gt_ref"},
            "optimizer": {},
        },
        "vio_preintegrated_full_no_velfb": {
            "odom": {"imu_vio_velocity_feedback_enable": False},
            "optimizer": {},
        },
        "vio_preintegrated_full_cov1000": {
            "odom": {},
            "optimizer": {"imu_vio_cov_scale": 1000.0},
        },
        "vio_preintegrated_full_gravityrp_weak": {
            "odom": {
                "imu_vio_gravity_pose_source": "imu_gravity_rp",
                "imu_gravity_rp_correction_gain": 1.0,
                "imu_gravity_rp_acc_tol": 0.15,
                "imu_vio_velocity_feedback_enable": True,
            },
            "optimizer": {"imu_vio_cov_scale": 25.0},
        },
        "vio_preintegrated_full_imuatt_estinit": {
            "odom": {"imu_vio_gravity_pose_source": "imu_integrated_estinit"},
            "optimizer": {},
        },
        "vio_preintegrated_full_imuatt_Ronly": {
            "odom": {"imu_vio_gravity_pose_source": "imu_integrated_estinit"},
            "optimizer": {"imu_vio_alpha_p": 0.0, "imu_vio_alpha_v": 0.0, "imu_vio_alpha_R": 1.0},
        },
        "vio_preintegrated_full_imuatt_pvonly": {
            "odom": {"imu_vio_gravity_pose_source": "imu_integrated_estinit"},
            "optimizer": {"imu_vio_alpha_p": 1.0, "imu_vio_alpha_v": 1.0, "imu_vio_alpha_R": 0.0},
        },
        "vio_preintegrated_full_imuatt_pv01": {
            "odom": {"imu_vio_gravity_pose_source": "imu_integrated_estinit"},
            "optimizer": {"imu_vio_alpha_p": 0.1, "imu_vio_alpha_v": 0.1, "imu_vio_alpha_R": 1.0},
        },
        "vio_preintegrated_full_imuatt_pv001": {
            "odom": {"imu_vio_gravity_pose_source": "imu_integrated_estinit"},
            "optimizer": {"imu_vio_alpha_p": 0.01, "imu_vio_alpha_v": 0.01, "imu_vio_alpha_R": 1.0},
        },
        "vio_preintegrated_full_imuatt_R01": {
            "odom": {"imu_vio_gravity_pose_source": "imu_integrated_estinit"},
            "optimizer": {"imu_vio_alpha_p": 1.0, "imu_vio_alpha_v": 1.0, "imu_vio_alpha_R": 0.1},
        },
        "vio_preintegrated_full_imuatt_all01": {
            "odom": {"imu_vio_gravity_pose_source": "imu_integrated_estinit"},
            "optimizer": {"imu_vio_alpha_p": 0.1, "imu_vio_alpha_v": 0.1, "imu_vio_alpha_R": 0.1},
        },
    }

    for variant_name, fields in expected.items():
        odom_cfg = grid.make_odom_cfg(grid.VARIANTS[variant_name], tmp_path, autodiff=False)
        cfg = grid.load_yaml(odom_cfg)
        odom_args = cfg["Odometry"]["args"]
        optimizer_args = cfg["Odometry"]["optimizer"]["args"]

        assert optimizer_args["autodiff"] is True
        assert optimizer_args["imu_factor_mode"] == "preintegrated_vio"
        assert grid.VARIANTS[variant_name].force_mode == "full_imu"
        for key, value in fields["odom"].items():
            assert odom_args[key] == value
        for key, value in fields["optimizer"].items():
            assert optimizer_args[key] == value

    imuatt = grid.VARIANTS["vio_preintegrated_full_imuatt_estinit"]
    assert imuatt.velocity_feedback_enable is None
    assert imuatt.vio_cov_scale is None
    assert imuatt.force_mode == "full_imu"
    assert imuatt.imu_factor_mode == "preintegrated_vio"


def test_vio_alpha_values_are_recorded_in_manifest(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated_full_imuatt_Ronly"],
        trials=1,
        result_root=tmp_path / "results",
    )
    grid.write_manifest(tmp_path, specs, seq_to=30, autodiff=False)

    rows = list(csv.DictReader((tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8")))

    assert rows[0]["imu_vio_alpha_p"] == "0.0"
    assert rows[0]["imu_vio_alpha_v"] == "0.0"
    assert rows[0]["imu_vio_alpha_R"] == "1.0"


def test_parallel_run_schedule_executes_all_specs_and_reports_failures(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["turbid_harbor"],
        variants=["pure_macvo", "rotation_only", "aimvo_damping_s005"],
        trials=1,
        result_root=tmp_path / "results",
    )
    odom_cfgs = {spec.variant.name: tmp_path / f"{spec.variant.name}.yaml" for spec in specs}
    calls: list[str] = []

    def fake_runner(spec, odom_cfg, seq_cfg, result_root, timeout_s, seq_to):
        calls.append(spec.variant.name)
        assert odom_cfg == odom_cfgs[spec.variant.name]
        assert seq_cfg.exists()
        assert result_root == tmp_path / "results"
        assert timeout_s == 123
        assert seq_to == 300
        return 7 if spec.variant.name == "rotation_only" else 0

    failures = grid.execute_run_schedule(
        specs,
        odom_cfgs,
        tmp_path,
        tmp_path / "results",
        timeout_s=123,
        seq_to=300,
        jobs=2,
        runner=fake_runner,
    )

    assert sorted(calls) == ["aimvo_damping_s005", "pure_macvo", "rotation_only"]
    assert [(spec.variant.name, rc) for spec, rc in failures] == [("rotation_only", 7)]


def test_manifest_write_is_guarded_against_subset_clobber(tmp_path: Path):
    result_root = tmp_path / "results"
    full_specs = grid.build_specs(
        scenes=["turbid_harbor"],
        variants=["pure_macvo", "rotation_only", "aimvo_damping_s005"],
        trials=1,
        result_root=result_root,
    )
    subset_specs = grid.build_specs(
        scenes=["turbid_harbor"],
        variants=["pure_macvo"],
        trials=1,
        result_root=result_root,
    )

    assert grid.write_manifest_guarded(result_root, full_specs, seq_to=300, autodiff=False) is True
    assert grid.write_manifest_guarded(result_root, subset_specs, seq_to=300, autodiff=False) is False

    rows = list(csv.DictReader((result_root / "run_manifest.csv").open(newline="", encoding="utf-8")))
    assert [row["variant"] for row in rows] == ["pure_macvo", "rotation_only", "aimvo_damping_s005"]

    assert grid.write_manifest_guarded(
        result_root,
        subset_specs,
        seq_to=300,
        autodiff=False,
        overwrite=True,
    ) is True
    rows = list(csv.DictReader((result_root / "run_manifest.csv").open(newline="", encoding="utf-8")))
    assert [row["variant"] for row in rows] == ["pure_macvo"]


def test_manifest_matches_specs_accepts_same_schedule_and_rejects_different_seq_to(tmp_path: Path):
    result_root = tmp_path / "results"
    specs = grid.build_specs(
        scenes=["turbid_harbor"],
        variants=["pure_macvo", "rotation_only"],
        trials=1,
        result_root=result_root,
    )
    grid.write_manifest(result_root, specs, seq_to=300, autodiff=False)

    assert grid.manifest_matches_specs(result_root, specs, seq_to=300, autodiff=False) is True
    assert grid.manifest_matches_specs(result_root, specs, seq_to=30, autodiff=False) is False


def test_main_refuses_to_run_when_existing_manifest_mismatches_current_specs(tmp_path: Path, monkeypatch):
    result_root = tmp_path / "results"
    stale_specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["pure_macvo"],
        trials=1,
        result_root=result_root,
    )
    grid.write_manifest(result_root, stale_specs, seq_to=30, autodiff=False)

    monkeypatch.setattr(grid, "sanity_check", lambda specs: True)

    def fail_execute(*args, **kwargs):
        raise AssertionError("execute_run_schedule should not be called with a stale manifest")

    monkeypatch.setattr(grid, "execute_run_schedule", fail_execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vio_imu_prior_mode_grid.py",
            "--result-root",
            str(result_root),
            "--preset",
            "smoke3",
            "--variants",
            "vio_preintegrated_full",
            "--trials",
            "1",
            "--seq-to",
            "30",
        ],
    )

    assert grid.main() == 1


def test_dry_run_refuses_mismatching_existing_manifest(tmp_path: Path, monkeypatch):
    result_root = tmp_path / "results"
    stale_specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["pure_macvo"],
        trials=1,
        result_root=result_root,
    )
    grid.write_manifest(result_root, stale_specs, seq_to=30, autodiff=False)

    monkeypatch.setattr(grid, "sanity_check", lambda specs: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vio_imu_prior_mode_grid.py",
            "--result-root",
            str(result_root),
            "--preset",
            "smoke3",
            "--variants",
            "vio_preintegrated_full",
            "--trials",
            "1",
            "--seq-to",
            "30",
            "--dry-run",
        ],
    )

    assert grid.main() == 1


def test_completed_pose_requires_coordinate_sidecar(tmp_path: Path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")

    assert grid.has_completed_pose(result_dir) is False

    (result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    assert grid.has_completed_pose(result_dir) is True


def test_sanity_check_rejects_scene_metadata_with_wrong_imu_frame(tmp_path: Path):
    scene_root = tmp_path / "bad_scene"
    _write_scene_files(scene_root, _minimal_scene_metadata(imu__frame="NED"))
    spec = grid.RunSpec(
        trial=1,
        scene="bad_scene",
        scene_root=scene_root,
        variant=grid.VARIANTS["vio_preintegrated_full"],
        result_dir=tmp_path / "results" / "bad_scene",
    )

    assert grid.sanity_check([spec]) is False


def test_sanity_check_rejects_endpoint_estimated_camera_imu_offset_metadata(tmp_path: Path):
    scene_root = tmp_path / "bad_scene"
    _write_scene_files(
        scene_root,
        _minimal_scene_metadata(time_synchronization__camera_imu_time_offset_ns=-15_000_000),
    )
    spec = grid.RunSpec(
        trial=1,
        scene="bad_scene",
        scene_root=scene_root,
        variant=grid.VARIANTS["vio_preintegrated_full"],
        result_dir=tmp_path / "results" / "bad_scene",
    )

    assert grid.sanity_check([spec]) is False


def test_manifest_records_scene_metadata_time_sync_evidence(tmp_path: Path, monkeypatch):
    scene_root = tmp_path / "scene"
    _write_scene_files(scene_root, _minimal_scene_metadata())
    monkeypatch.setitem(grid.SCENE_ROOTS, "tmp_scene", scene_root)
    specs = grid.build_specs(
        scenes=["tmp_scene"],
        variants=["vio_preintegrated_full"],
        trials=1,
        result_root=tmp_path / "results",
    )

    grid.write_manifest(tmp_path, specs, seq_to=30, autodiff=False)

    rows = list(csv.DictReader((tmp_path / "run_manifest.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["metadata_camera_imu_time_offset_ns"] == "0"
    assert rows[0]["metadata_time_offset_source"] == "metadata.time_synchronization.camera_imu_time_offset_ns"


def test_full_vio_completed_run_requires_active_diagnostics(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated_full"],
        trials=1,
        result_root=tmp_path / "results",
    )
    spec = specs[0]
    spec.result_dir.mkdir(parents=True)
    (spec.result_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (spec.result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    assert grid.has_completed_pose(spec.result_dir) is True
    assert grid.has_completed_run(spec) is False


def test_full_vio_completed_run_accepts_active_diagnostics(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated_full"],
        trials=1,
        result_root=tmp_path / "results",
    )
    spec = specs[0]
    spec.result_dir.mkdir(parents=True)
    (spec.result_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (spec.result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    _write_diagnostics_csv(spec.result_dir / "frame_pair_diagnostics.csv", _full_vio_diagnostics_row())

    assert grid.has_completed_run(spec) is True


def test_local_inertial_ba_completed_run_requires_active_diagnostics(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_local_ba_w3_imuatt"],
        trials=1,
        result_root=tmp_path / "results",
    )
    spec = specs[0]
    spec.result_dir.mkdir(parents=True)
    (spec.result_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (spec.result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    assert grid.has_completed_run(spec) is False

    _write_diagnostics_csv(
        spec.result_dir / "frame_pair_diagnostics.csv",
        _full_vio_diagnostics_row(imu_factor_mode="local_inertial_ba", imu_residual_rows="10"),
    )

    assert grid.has_completed_run(spec) is True


def test_full_vio_completed_run_rejects_active_diagnostics_without_conventions(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated_full"],
        trials=1,
        result_root=tmp_path / "results",
    )
    spec = specs[0]
    spec.result_dir.mkdir(parents=True)
    (spec.result_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (spec.result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    (spec.result_dir / "frame_pair_diagnostics.csv").write_text(
        "pair_id,imu_factor_mode,vio_factor_active,imu_residual_rows\n"
        "1,preintegrated_vio,1,3\n",
        encoding="utf-8",
    )

    assert grid.has_completed_run(spec) is False


def test_full_vio_completed_run_accepts_nested_active_diagnostics(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated_full"],
        trials=1,
        result_root=tmp_path / "results",
    )
    spec = specs[0]
    nested_dir = spec.result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    nested_dir.mkdir(parents=True)
    (nested_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (nested_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    _write_diagnostics_csv(nested_dir / "frame_pair_diagnostics.csv", _full_vio_diagnostics_row())

    assert grid.has_completed_pose(spec.result_dir) is True
    assert grid.has_completed_run(spec) is True


def test_full_vio_completed_run_rejects_split_pose_and_diagnostics_bundles(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated_full"],
        trials=1,
        result_root=tmp_path / "results",
    )
    spec = specs[0]
    spec.result_dir.mkdir(parents=True)
    (spec.result_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (spec.result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    nested_dir = spec.result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    nested_dir.mkdir(parents=True)
    _write_diagnostics_csv(nested_dir / "frame_pair_diagnostics.csv", _full_vio_diagnostics_row())

    assert grid.has_completed_run(spec) is False


def test_full_vio_completed_run_prefers_nested_valid_bundle_over_stale_direct_invalid(tmp_path: Path):
    specs = grid.build_specs(
        scenes=["clear_shallow"],
        variants=["vio_preintegrated_full"],
        trials=1,
        result_root=tmp_path / "results",
    )
    spec = specs[0]
    spec.result_dir.mkdir(parents=True)
    (spec.result_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (spec.result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    (spec.result_dir / "frame_pair_diagnostics.csv").write_text(
        "pair_id,imu_factor_mode,vio_factor_active,imu_residual_rows\n"
        "1,preintegrated_vio,1,3\n",
        encoding="utf-8",
    )

    nested_dir = spec.result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    nested_dir.mkdir(parents=True)
    (nested_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (nested_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    _write_diagnostics_csv(nested_dir / "frame_pair_diagnostics.csv", _full_vio_diagnostics_row())

    assert grid.has_completed_run(spec) is True


def test_run_one_reports_incomplete_full_vio_output_after_zero_return(tmp_path: Path, monkeypatch):
    result_root = tmp_path / "results"
    spec = grid.RunSpec(
        trial=1,
        scene="clear_shallow",
        scene_root=tmp_path / "scene",
        variant=grid.VARIANTS["vio_preintegrated_full"],
        result_dir=result_root / "trial_1" / "vio_preintegrated_full" / "clear_shallow",
    )

    def fake_subprocess_run(*args, **kwargs):
        spec.result_dir.mkdir(parents=True, exist_ok=True)
        (spec.result_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
        (spec.result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(grid.subprocess, "run", fake_subprocess_run)

    rc = grid.run_one(
        spec,
        odom_cfg=tmp_path / "odom.yaml",
        seq_cfg=tmp_path / "seq.yaml",
        result_root=result_root,
        timeout_s=30,
        seq_to=30,
    )
    rows = list(csv.DictReader((result_root / "progress.csv").open(newline="", encoding="utf-8")))

    assert rc != 0
    assert rows[-1]["status"] == "incomplete_output"


def test_flatten_nested_moves_complete_pose_bundle(tmp_path: Path):
    result_dir = tmp_path / "run"
    nested_dir = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    nested_dir.mkdir(parents=True)
    (nested_dir / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (nested_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    (nested_dir / "frame_pair_diagnostics.csv").write_text("a,b\n", encoding="utf-8")

    grid.flatten_nested(result_dir)

    assert (result_dir / "poses.csv").exists()
    assert (result_dir / "pose_coordinate_frame.txt").read_text(encoding="utf-8").strip() == "NWU"
    assert (result_dir / "frame_pair_diagnostics.csv").exists()
    assert grid.has_completed_pose(result_dir) is True


def test_flatten_nested_replaces_stale_direct_pose_without_sidecar(tmp_path: Path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "poses.csv").write_text("stale\n", encoding="utf-8")

    nested_dir = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    nested_dir.mkdir(parents=True)
    (nested_dir / "poses.csv").write_text("fresh\n", encoding="utf-8")
    (nested_dir / "pose_coordinate_frame.txt").write_text("NED\n", encoding="utf-8")

    grid.flatten_nested(result_dir)

    assert (result_dir / "poses.csv").read_text(encoding="utf-8") == "fresh\n"
    assert (result_dir / "pose_coordinate_frame.txt").read_text(encoding="utf-8").strip() == "NED"
    assert grid.has_completed_pose(result_dir) is True

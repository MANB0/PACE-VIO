import subprocess
import sys
import csv
from pathlib import Path

import yaml

from Scripts import run_vio_imu_prior_mode_grid as grid
from Scripts import run_gravity_handling_ablation as ablation


WORKDIR = Path("/home/admin1/macvo-dev")
RUNNER = WORKDIR / "Scripts/run_gravity_handling_ablation.py"


def test_residual_gravity_variant_changes_only_gravity_contract(tmp_path):
    legacy = grid.VARIANTS["vio_preintegrated_full_imuatt_estinit"]
    residual = grid.VARIANTS["vio_preintegrated_full_residual_gravity"]

    assert residual.mode == legacy.mode
    assert residual.scale == legacy.scale
    assert residual.rot_enabled == legacy.rot_enabled
    assert residual.trans_enabled == legacy.trans_enabled
    assert residual.force_mode == legacy.force_mode
    assert residual.imu_factor_mode == legacy.imu_factor_mode
    assert residual.gravity_handling == "residual"
    assert residual.gravity_pose_source == "estimated"

    cfg_path = grid.make_odom_cfg(residual, tmp_path, autodiff=False)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    odom_args = cfg["Odometry"]["args"]
    optimizer_args = cfg["Odometry"]["optimizer"]["args"]

    assert odom_args["imu_vio_gravity_handling"] == "residual"
    assert odom_args["imu_vio_gravity_pose_source"] == "estimated"
    assert odom_args["mapping"] is False
    assert optimizer_args["imu_factor_mode"] == "preintegrated_vio"
    assert optimizer_args["autodiff"] is True


def test_manual_launcher_dry_run_schedules_two_modes_over_two_scenes(tmp_path):
    assert RUNNER.exists(), f"missing launcher: {RUNNER}"
    result_root = tmp_path / "results"
    output_root = tmp_path / "analysis"
    log_path = tmp_path / "run.log"

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dry-run",
            "--no-dashboard",
            "--result-root",
            str(result_root),
            "--output-root",
            str(output_root),
            "--log-path",
            str(log_path),
        ],
        cwd=str(WORKDIR),
        text=True,
        capture_output=True,
        check=False,
    )

    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "clear_rectangle_zero_noise" in output
    assert "clear_rectangle_normal_noise" in output
    assert "vio_preintegrated_full_imuatt_estinit" in output
    assert "vio_preintegrated_full_residual_gravity" in output
    assert "Runs:     4" in output
    assert "http://127.0.0.1:8765/" in output
    assert not (result_root / "progress.csv").exists()


def test_run_manifest_records_gravity_contract(tmp_path):
    scene_root = tmp_path / "scene"
    scene_root.mkdir()
    result_root = tmp_path / "results"
    variant = grid.VARIANTS["vio_preintegrated_full_residual_gravity"]
    spec = grid.RunSpec(
        trial=1,
        scene="synthetic_scene",
        scene_root=scene_root,
        variant=variant,
        result_dir=result_root / "trial_1" / variant.name / "synthetic_scene",
    )

    grid.write_manifest(result_root, [spec], seq_to=None, autodiff=False)
    with (result_root / "run_manifest.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["imu_vio_gravity_handling"] == "residual"
    assert row["imu_vio_gravity_pose_source"] == "estimated"


def _write_visual_fingerprint_run(result_dir: Path, rows: list[tuple[int, int, str]]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "frame_pair_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_i", "frame_j", "visual_input_sha256"])
        writer.writeheader()
        for frame_i, frame_j, fingerprint in rows:
            writer.writerow(
                {
                    "frame_i": frame_i,
                    "frame_j": frame_j,
                    "visual_input_sha256": fingerprint,
                }
            )


def _write_ablation_manifest(result_root: Path, legacy_dir: Path, residual_dir: Path) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    with (result_root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "variant", "result_dir"])
        writer.writeheader()
        writer.writerow(
            {
                "scene": "clear_rectangle_normal_noise",
                "variant": ablation.VARIANTS[0],
                "result_dir": legacy_dir,
            }
        )
        writer.writerow(
            {
                "scene": "clear_rectangle_normal_noise",
                "variant": ablation.VARIANTS[1],
                "result_dir": residual_dir,
            }
        )


def test_visual_fingerprint_validation_accepts_exactly_matching_frontend_inputs(tmp_path):
    result_root = tmp_path / "results"
    legacy_dir = result_root / "legacy"
    residual_dir = result_root / "residual"
    _write_ablation_manifest(result_root, legacy_dir, residual_dir)
    rows = [(0, 1, "a" * 64), (1, 2, "b" * 64)]
    _write_visual_fingerprint_run(legacy_dir, rows)
    _write_visual_fingerprint_run(residual_dir, rows)

    report_path, exact_match = ablation.validate_visual_input_identity(result_root)

    assert exact_match is True
    with report_path.open(newline="", encoding="utf-8") as f:
        report = next(csv.DictReader(f))
    assert report["compared_pairs"] == "2"
    assert report["mismatched_pairs"] == "0"
    assert report["exact_match"] == "1"


def test_visual_fingerprint_validation_rejects_frontend_mismatch(tmp_path):
    result_root = tmp_path / "results"
    legacy_dir = result_root / "legacy"
    residual_dir = result_root / "residual"
    _write_ablation_manifest(result_root, legacy_dir, residual_dir)
    _write_visual_fingerprint_run(legacy_dir, [(0, 1, "a" * 64)])
    _write_visual_fingerprint_run(residual_dir, [(0, 1, "c" * 64)])

    _, exact_match = ablation.validate_visual_input_identity(result_root)

    assert exact_match is False

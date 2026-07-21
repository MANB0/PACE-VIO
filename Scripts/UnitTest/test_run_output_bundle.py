from pathlib import Path

from Utility.RunOutputBundle import find_output_bundle


def _write_pose_bundle(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "poses.csv").write_text("timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n", encoding="utf-8")
    (path / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")


def test_find_output_bundle_prefers_direct_pose_bundle():
    result_dir = Path("/tmp/result")

    bundle = find_output_bundle(result_dir)

    assert bundle.poses_path == result_dir / "poses.csv"
    assert bundle.pose_frame_path == result_dir / "pose_coordinate_frame.txt"
    assert bundle.diagnostics_path == result_dir / "frame_pair_diagnostics.csv"


def test_find_output_bundle_uses_nested_complete_pose_bundle(tmp_path: Path):
    result_dir = tmp_path / "result"
    nested = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    _write_pose_bundle(nested)

    bundle = find_output_bundle(result_dir)

    assert bundle.bundle_dir == nested
    assert bundle.poses_path == nested / "poses.csv"
    assert bundle.pose_frame_path == nested / "pose_coordinate_frame.txt"


def test_find_output_bundle_requires_diagnostics_in_same_directory(tmp_path: Path):
    result_dir = tmp_path / "result"
    _write_pose_bundle(result_dir)
    nested = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    nested.mkdir(parents=True)
    (nested / "frame_pair_diagnostics.csv").write_text("pair_id\n", encoding="utf-8")

    bundle = find_output_bundle(result_dir, require_same_dir_diagnostics=True)

    assert bundle.bundle_dir == result_dir
    assert bundle.diagnostics_path == result_dir / "frame_pair_diagnostics.csv"


def test_find_output_bundle_selects_nested_complete_diagnostics_bundle(tmp_path: Path):
    result_dir = tmp_path / "result"
    nested = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    _write_pose_bundle(nested)
    (nested / "frame_pair_diagnostics.csv").write_text("pair_id\n", encoding="utf-8")

    bundle = find_output_bundle(result_dir, require_same_dir_diagnostics=True)

    assert bundle.bundle_dir == nested
    assert bundle.diagnostics_path == nested / "frame_pair_diagnostics.csv"


def test_find_output_bundle_can_skip_invalid_direct_diagnostics_bundle(tmp_path: Path):
    result_dir = tmp_path / "result"
    _write_pose_bundle(result_dir)
    (result_dir / "frame_pair_diagnostics.csv").write_text("invalid\n", encoding="utf-8")

    nested = result_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "trial"
    _write_pose_bundle(nested)
    (nested / "frame_pair_diagnostics.csv").write_text("valid\n", encoding="utf-8")

    bundle = find_output_bundle(
        result_dir,
        require_same_dir_diagnostics=True,
        diagnostics_validator=lambda path: path.read_text(encoding="utf-8") == "valid\n",
    )

    assert bundle.bundle_dir == nested
    assert bundle.diagnostics_path == nested / "frame_pair_diagnostics.csv"

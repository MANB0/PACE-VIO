from __future__ import annotations

import csv
from pathlib import Path

from Scripts.run_progress_dashboard import collect_dashboard_state, read_tail


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_ref_pose(scene_root: Path, frames: int) -> None:
    rows = [{"timestamp": idx, "x": idx, "y": 0, "z": 0} for idx in range(frames)]
    _write_csv(scene_root / "ref_pose.csv", ["timestamp", "x", "y", "z"], rows)


def test_collect_dashboard_state_merges_manifest_progress_and_frame_progress(tmp_path: Path) -> None:
    result_root = tmp_path / "Results" / "batch"
    scene_root = tmp_path / "recordings" / "scene_a"
    run_dir = result_root / "trial_1" / "variant_a" / "scene_a"
    bundle_dir = run_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "latest"
    _write_ref_pose(scene_root, frames=10)
    _write_csv(
        result_root / "run_manifest.csv",
        ["trial", "scene", "variant", "scene_root", "result_dir"],
        [
            {
                "trial": 1,
                "scene": "scene_a",
                "variant": "variant_a",
                "scene_root": scene_root,
                "result_dir": run_dir,
            },
            {
                "trial": 1,
                "scene": "scene_a",
                "variant": "variant_b",
                "scene_root": scene_root,
                "result_dir": result_root / "trial_1" / "variant_b" / "scene_a",
            },
        ],
    )
    _write_csv(
        result_root / "progress.csv",
        ["trial", "scene", "variant", "status", "return_code", "runtime_s", "result_dir"],
        [
            {
                "trial": 1,
                "scene": "scene_a",
                "variant": "variant_a",
                "status": "ok",
                "return_code": 0,
                "runtime_s": 12.3,
                "result_dir": run_dir,
            }
        ],
    )
    _write_csv(
        bundle_dir / "frame_pair_diagnostics.csv",
        ["frame_j", "imu_factor_mode"],
        [{"frame_j": idx, "imu_factor_mode": "local_inertial_ba"} for idx in range(5)],
    )
    (bundle_dir / "poses.csv").write_text("frame,x,y,z\n0,0,0,0\n", encoding="utf-8")
    (bundle_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    state = collect_dashboard_state(result_root, include_system=False, now=100.0)

    assert state["counters"]["total"] == 2
    assert state["counters"]["ok"] == 1
    assert state["counters"]["pending"] == 1
    run_a = next(run for run in state["runs"] if run["variant"] == "variant_a")
    assert run_a["status"] == "ok"
    assert run_a["current_frame"] == 4
    assert run_a["total_frames"] == 10
    assert run_a["percent"] == 50.0
    assert run_a["has_poses"] is True
    assert run_a["has_pose_frame"] is True


def test_collect_dashboard_state_marks_complete_unlogged_when_pose_bundle_exists(tmp_path: Path) -> None:
    result_root = tmp_path / "Results" / "batch"
    scene_root = tmp_path / "recordings" / "scene_b"
    run_dir = result_root / "trial_1" / "variant_a" / "scene_b"
    _write_ref_pose(scene_root, frames=3)
    _write_csv(
        result_root / "run_manifest.csv",
        ["trial", "scene", "variant", "scene_root", "result_dir"],
        [
            {
                "trial": 1,
                "scene": "scene_b",
                "variant": "variant_a",
                "scene_root": scene_root,
                "result_dir": run_dir,
            }
        ],
    )
    (run_dir / "poses.csv").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "poses.csv").write_text("frame,x,y,z\n0,0,0,0\n", encoding="utf-8")
    (run_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")

    state = collect_dashboard_state(result_root, include_system=False)

    assert state["counters"]["complete_unlogged"] == 1
    assert state["runs"][0]["status"] == "complete_unlogged"


def test_collect_dashboard_state_marks_partial_when_diagnostics_exist_without_pose(tmp_path: Path) -> None:
    result_root = tmp_path / "Results" / "batch"
    scene_root = tmp_path / "recordings" / "scene_c"
    run_dir = result_root / "trial_1" / "variant_a" / "scene_c"
    bundle_dir = run_dir / "MACVO-HoloOcean-IMU@holoocean_imu" / "latest"
    _write_ref_pose(scene_root, frames=4)
    _write_csv(
        result_root / "run_manifest.csv",
        ["trial", "scene", "variant", "scene_root", "result_dir"],
        [
            {
                "trial": 1,
                "scene": "scene_c",
                "variant": "variant_a",
                "scene_root": scene_root,
                "result_dir": run_dir,
            }
        ],
    )
    _write_csv(
        bundle_dir / "frame_pair_diagnostics.csv",
        ["frame_idx", "imu_factor_mode"],
        [{"frame_idx": 0, "imu_factor_mode": "local_inertial_ba"}],
    )

    state = collect_dashboard_state(result_root, include_system=False)

    assert state["counters"]["partial"] == 1
    assert state["runs"][0]["status"] == "partial"


def test_collect_dashboard_state_tracks_running_staging_artifact(tmp_path: Path) -> None:
    result_root = tmp_path / "Results" / "batch"
    scene_root = tmp_path / "recordings" / "scene_d"
    final_dir = tmp_path / "Results" / "final" / "scene_d"
    active_dir = result_root / "staging" / "source" / "scene_d" / "attempt" / "artifact"
    _write_ref_pose(scene_root, frames=10)
    _write_csv(
        result_root / "run_manifest.csv",
        ["trial", "scene", "variant", "scene_root", "result_dir"],
        [
            {
                "trial": 1,
                "scene": "scene_d",
                "variant": "source_pure_macvo",
                "scene_root": scene_root,
                "result_dir": final_dir,
            }
        ],
    )
    _write_csv(
        result_root / "progress.csv",
        [
            "trial",
            "scene",
            "variant",
            "status",
            "result_dir",
            "active_result_dir",
        ],
        [
            {
                "trial": 1,
                "scene": "scene_d",
                "variant": "source_pure_macvo",
                "status": "running",
                "result_dir": final_dir,
                "active_result_dir": active_dir,
            }
        ],
    )
    _write_csv(
        active_dir / "frame_pair_diagnostics.csv",
        ["frame_j", "imu_factor_mode"],
        [{"frame_j": idx, "imu_factor_mode": "legacy_pose_prior"} for idx in range(4)],
    )

    state = collect_dashboard_state(result_root, include_system=False)

    run = state["runs"][0]
    assert run["status"] == "running"
    assert run["result_dir"] == str(final_dir)
    assert run["active_result_dir"] == str(active_dir)
    assert run["diagnostics_rows"] == 4
    assert run["current_frame"] == 3
    assert run["percent"] == 40.0


def test_collect_dashboard_state_uses_manifest_seq_to_as_progress_total(tmp_path: Path) -> None:
    result_root = tmp_path / "Results" / "batch"
    scene_root = tmp_path / "recordings" / "scene_e"
    run_dir = result_root / "trial_1" / "variant_a" / "scene_e"
    _write_ref_pose(scene_root, frames=10)
    _write_csv(
        result_root / "run_manifest.csv",
        ["trial", "scene", "variant", "scene_root", "result_dir", "seq_to"],
        [
            {
                "trial": 1,
                "scene": "scene_e",
                "variant": "variant_a",
                "scene_root": scene_root,
                "result_dir": run_dir,
                "seq_to": 4,
            }
        ],
    )
    _write_csv(
        run_dir / "frame_pair_diagnostics.csv",
        ["frame_j"],
        [{"frame_j": index} for index in range(1, 4)],
    )

    state = collect_dashboard_state(result_root, include_system=False)

    run = state["runs"][0]
    assert run["total_frames"] == 4
    assert run["current_frame"] == 3
    assert run["percent"] == 100.0


def test_read_tail_returns_only_requested_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text("\n".join(f"line {idx}" for idx in range(10)) + "\n", encoding="utf-8")

    assert read_tail(log_path, max_lines=3) == ["line 7", "line 8", "line 9"]

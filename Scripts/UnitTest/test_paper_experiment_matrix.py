import json
from pathlib import Path

from Scripts.run_paper_experiments import METHODS, build_command, load_manifest, summarize


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    dataset = tmp_path / "circle_data"
    dataset.mkdir()
    manifest = {
        "datasets": [{
            "scenario": "Circle",
            "path": str(dataset),
            "static_initialization": {"mode": "fixed", "duration_s": 3.0},
        }],
        "runtime": {"mode": "pipeline", "cpu_threads": 2, "timeout_s": 60},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_matrix_uses_all_required_method_combinations_and_full_sequences(tmp_path: Path):
    path, manifest = _manifest(tmp_path)
    loaded = load_manifest(path)
    commands = [
        build_command(
            dataset=loaded["datasets"][0],
            method=method,
            output_root=tmp_path / "results",
            model=tmp_path / "model.pth",
            runtime=loaded["runtime"],
            dry_run=True,
        )
        for method in METHODS
    ]

    assert [method.key for method in METHODS] == [
        "pose_isam2", "uvd_isam2", "pace_two_state", "pace_isam2", "pace_vio"
    ]
    assert all("--seq-to" not in command for command in commands)
    assert all("--paper-evaluation" in command for command in commands)
    assert commands[-1][commands[-1].index("--near-zero-velocity-detector") + 1] == "v2"


def test_summary_maps_every_paper_comparison_to_traceable_bundles(tmp_path: Path):
    _, manifest = _manifest(tmp_path)
    output = tmp_path / "results"
    for method in METHODS:
        bundle = output / method.key / "circle_data" / "paper_evaluation"
        bundle.mkdir(parents=True)
        trajectory = {
            "state_count": 4,
            "edge_count": 3,
            "ape": {
                "xy_m": {"rmse": 1.0, "p95": 2.0},
                "translation_m": {"rmse": 1.1, "p95": 2.1},
                "rotation_deg": {"rmse": 0.1, "p95": 0.2},
            },
            "rpe": {
                "translation_m": {"rmse": 0.01, "p95": 0.02},
                "rotation_deg": {"rmse": 0.001, "p95": 0.002},
            },
        }
        trajectories = {"final": trajectory}
        if method.key == "pace_vio":
            trajectories["macvo_raw"] = trajectory
        payload = {
            "trajectories": trajectories,
            "timing": {
                "factor_build_ms": {"median": 1.0, "p95": 2.0},
                "backend_update_ms": {"median": 3.0, "p95": 4.0},
                "convergence_rate": 1.0,
            },
        }
        (bundle / "metrics_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize(output, manifest)
    rows = summary.read_text(encoding="utf-8").splitlines()
    sources = json.loads((output / "paper_comparison_sources.json").read_text(encoding="utf-8"))

    assert len(rows) == 7  # header + five experiment rows + one MACVO row
    assert sources["sequence_scope"] == "complete active sequence"
    assert sources["paper_comparisons"]["visual_factor"] == [
        "pose_isam2", "uvd_isam2", "pace_isam2"
    ]
    assert sources["paper_comparisons"]["condition_factor"] == [
        "pace_isam2", "pace_vio"
    ]

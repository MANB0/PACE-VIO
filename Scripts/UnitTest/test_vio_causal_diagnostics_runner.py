from pathlib import Path
from types import SimpleNamespace

import yaml

import Scripts.run_vio_causal_diagnostics_rectangle_normal as runner


def _args(tmp_path: Path, *, dry_run: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        result_root=tmp_path / "Results" / "causal",
        seq_to=None,
        timeout=123,
        influence_interval=5,
        dashboard_port=8765,
        no_dashboard=False,
        overwrite_manifest=True,
        dry_run=dry_run,
    )


def test_schedule_contains_one_full_sequence_method(tmp_path: Path):
    specs = runner.build_specs(tmp_path / "Results" / "causal")

    assert len(specs) == 1
    spec = specs[0]
    assert spec.scene == "clear_rectangle_normal_noise"
    assert spec.variant.name == "vio_preintegrated_full_imuatt_estinit"
    assert spec.trial == 1


def test_causal_config_enables_read_only_audit_with_requested_interval(tmp_path: Path):
    path = runner.make_causal_odom_cfg(tmp_path, influence_interval=7)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    optimizer = cfg["Odometry"]["optimizer"]["args"]

    assert optimizer["imu_factor_mode"] == "preintegrated_vio"
    assert optimizer["vio_causal_diagnostics_enable"] is True
    assert optimizer["vio_causal_diagnostics_interval"] == 7
    assert optimizer["imu_vio_alpha_p"] == 1.0
    assert optimizer["imu_vio_alpha_v"] == 1.0
    assert optimizer["imu_vio_alpha_R"] == 1.0


def test_dashboard_command_points_to_same_result_root(tmp_path: Path):
    result_root = tmp_path / "Results" / "causal"
    cmd = runner.build_dashboard_command(result_root, port=8765)

    assert "Scripts/run_progress_dashboard.py" in cmd
    assert cmd[cmd.index("--result-root") + 1] == str(result_root)
    assert cmd[cmd.index("--port") + 1] == "8765"


def test_dry_run_never_starts_dashboard_or_macvo(tmp_path: Path, monkeypatch):
    args = _args(tmp_path, dry_run=True)
    monkeypatch.setattr(runner.grid, "sanity_check", lambda specs: True)
    monkeypatch.setattr(
        runner,
        "switch_dashboard",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("dashboard started during dry-run")),
    )
    monkeypatch.setattr(
        runner.grid,
        "execute_run_schedule",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("MACVO schedule executed during dry-run")),
    )

    assert runner.run_batch(args) == 0
    assert not args.result_root.exists()

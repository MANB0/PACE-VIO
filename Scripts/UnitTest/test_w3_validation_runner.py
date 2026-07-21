from pathlib import Path
from types import SimpleNamespace

import Scripts.run_w3_after_fix_validation as runner


def _args(tmp_path: Path):
    return SimpleNamespace(
        result_root=tmp_path / "Results" / "w3",
        output_root=tmp_path / "analysis_w3",
        log_path=tmp_path / "logs" / "w3.log",
        jobs=1,
        timeout=123,
        seq_to=None,
        overwrite_manifest=True,
    )


def test_build_run_command_runs_only_w3_variants_for_rectangle_normal_noise(tmp_path: Path):
    args = _args(tmp_path)

    cmd = runner.build_run_command(args)

    assert "Scripts/run_latest_closed_paths_methods.py" in cmd
    assert cmd[cmd.index("--scenes") + 1] == "clear_rectangle_normal_noise"
    variants = cmd[cmd.index("--variants") + 1 : cmd.index("--jobs")]
    assert variants == ["vio_local_ba_w3_imuatt", "vio_local_ba_w3_imuatt_all"]
    assert str(args.result_root) in cmd
    assert str(args.output_root) in cmd


def test_dashboard_command_uses_same_result_root_and_log_path(tmp_path: Path):
    args = _args(tmp_path)

    cmd = runner.build_dashboard_command(args.result_root, args.log_path, port=8765)

    assert "Scripts/run_progress_dashboard.py" in cmd
    assert cmd[cmd.index("--result-root") + 1] == str(args.result_root)
    assert cmd[cmd.index("--log") + 1] == str(args.log_path)
    assert cmd[cmd.index("--port") + 1] == "8765"


def test_combined_comparison_sources_include_reused_and_new_methods(tmp_path: Path):
    args = _args(tmp_path)

    specs = runner.comparison_specs(args.output_root)

    assert [spec.method for spec in specs] == [
        "pure_macvo_reused",
        "two_frame_imuatt",
        "w2_current",
        "w3_current",
        "w3_all_optimized",
    ]
    assert specs[-2].root == args.output_root
    assert specs[-1].root == args.output_root

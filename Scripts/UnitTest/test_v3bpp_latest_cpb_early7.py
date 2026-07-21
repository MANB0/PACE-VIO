from __future__ import annotations

import importlib.util
import math
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[2]


def _load_script(relative_path: str):
    path = WORKDIR / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_defines_latest_cpb_early7_protocol():
    runner = _load_script("Scripts/run_v3bpp_latest_cpb_fdonly_early7_3x.py")

    assert runner.SOURCE_NAME == "latest_cpb_early7"
    assert runner.TRIALS == 3
    assert runner.RESULT_ROOT.name == "v3bpp_latest_cpb_fdonly_early7_3x"
    assert [spec.scene for spec in runner.build_run_specs()] == [
        "turbid_harbor",
        "clear_shallow",
        "deep_dark",
        "caustic_shallow",
        "dam_inspection",
        "murky_coast",
        "open_water",
    ] * 3

    args = runner.cpb_fd_only_args()
    assert args == [
        "--adaptive-v3b",
        "--v3b-vc-mode",
        "two_level",
        "--v3b-vc-severe-thr",
        "30",
        "--v3b-vc-severe-sustain",
        "1",
        "--v3b-vc-mild-thr",
        "50",
        "--v3b-vc-mild-sustain",
        "5",
        "--v3b-fd-cooldown",
        "30",
    ]
    assert "--v3b-fd-grace" not in args


def test_analyzer_computes_latest_cpb_summary_against_pure_and_fixed_best():
    analyzer = _load_script("Scripts/analyse_v3bpp_latest_cpb_fdonly_early7.py")

    summary = analyzer.summarize_scene(
        scene="murky_coast",
        cpb_ates=[20.0, 18.0, 22.0],
        pure_median=55.0,
        fixed_best_method="full_imu",
        fixed_best_median=9.0,
    )

    assert summary["source"] == "latest_cpb_early7"
    assert summary["method"] == "CP-B-FD-only"
    assert summary["trial_1"] == 20.0
    assert summary["trial_2"] == 18.0
    assert summary["trial_3"] == 22.0
    assert summary["median_ATE"] == 20.0
    assert summary["min_ATE"] == 18.0
    assert summary["max_ATE"] == 22.0
    assert math.isclose(summary["std_ATE"], 1.632993161855452)
    assert math.isclose(summary["cv"], 0.0816496580927726)
    assert summary["median_range"] == "20.00 [18.00-22.00]"
    assert summary["delta_vs_pure"] == -35.0
    assert math.isclose(summary["relative_gain_vs_pure_pct"], 63.63636363636363)
    assert summary["fixed_best_method"] == "full_imu"
    assert math.isclose(summary["gap_to_fixed_best_x"], 20.0 / 9.0)

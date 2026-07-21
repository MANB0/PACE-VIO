from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "build_v3bpp_final_paper_results_v1.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_v3bpp_final_paper_results_v1", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_cpb_source_mapping_covers_all_16_scenes():
    builder = load_builder_module()

    expected_scenes = {
        "turbid_harbor",
        "clear_shallow",
        "deep_dark",
        "caustic_shallow",
        "dam_inspection",
        "murky_coast",
        "open_water",
        "moderate_turbidity",
        "open_water_overcast",
        "twilight_coast",
        "validation_moderate_harbor",
        "validation_transient_dropout",
        "validation_twilight_structure",
        "locked_murky_entry_help",
        "locked_clear_imu_harm",
        "locked_quality_degrade_no_dropout",
    }

    assert set(builder.FINAL_CPB_SOURCE_BY_SCENE) == expected_scenes
    assert builder.FINAL_CPB_SOURCE_BY_SCENE["turbid_harbor"] == "latest_cpb_early7"
    assert builder.FINAL_CPB_SOURCE_BY_SCENE["moderate_turbidity"] == "phase1b_cpb"
    assert builder.FINAL_CPB_SOURCE_BY_SCENE["validation_moderate_harbor"] == "validation_3scene"
    assert builder.FINAL_CPB_SOURCE_BY_SCENE["locked_murky_entry_help"] == "locked_3scene"


def test_interpretation_labels_preserve_conservative_paper_reading():
    builder = load_builder_module()

    assert builder.classify_scene("murky_coast", 65.25) == "clear_improvement"
    assert builder.classify_scene("deep_dark", -1.31) == "near_tie_slight_worse"
    assert builder.classify_scene("dam_inspection", -7.36) == "degradation_or_failure"
    assert builder.classify_scene("validation_moderate_harbor", -21.5) == "degradation_or_failure"


def test_median_range_format_is_paper_ready():
    builder = load_builder_module()

    assert builder.format_median_range(0.8772, 0.8670, 0.9109) == "0.88 [0.87-0.91]"
    assert builder.format_median_range(226.7247, 210.0484, 228.1909) == "226.72 [210.05-228.19]"


def test_fd_e_variant_is_not_counted_as_final_cpb_fd_only():
    builder = load_builder_module()

    assert builder.is_cpb_fd_only_method("CP-B-FD-only")
    assert builder.is_cpb_fd_only_method("cpb_fd_only")
    assert not builder.is_cpb_fd_only_method("FD-E+CP-B")


def test_user_selected_four_scenes_are_excluded_from_main_table():
    builder = load_builder_module()

    assert builder.EXCLUDED_FROM_MAIN_SCENES == {
        "open_water",
        "moderate_turbidity",
        "twilight_coast",
        "validation_moderate_harbor",
    }

    rows = builder.build_final_rows()
    main_rows = [row for row in rows if row["paper_role"] == "main"]
    excluded_rows = [row for row in rows if row["paper_role"] == "excluded_from_main"]

    assert len(main_rows) == 12
    assert {row["scene"] for row in excluded_rows} == builder.EXCLUDED_FROM_MAIN_SCENES
    assert all(row["exclusion_reason"] for row in excluded_rows)


def test_large_ate_exclusions_are_not_mislabeled_as_degraded():
    builder = load_builder_module()

    assert builder.exclusion_reason_for_scene("moderate_turbidity") == "large_absolute_ate_or_unstable"
    assert builder.exclusion_reason_for_scene("twilight_coast") == "large_absolute_ate"

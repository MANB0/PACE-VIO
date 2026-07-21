import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Scripts.imu_covariance_source_ablation import (
    COVARIANCE_VARIANTS,
    collect_covariance_ablation_samples,
    validate_current_variant_against_baseline,
    write_covariance_ablation_outputs,
)
from Scripts.imu_whitening_monte_carlo import (
    MonteCarloConfig,
    RESIDUAL_COLUMNS,
    collect_whitening_samples,
)


def test_covariance_ablation_reuses_residuals_and_toggles_declared_sources():
    config = MonteCarloConfig(duration_s=0.1, seed_count=2, start_seed=41)

    samples = collect_covariance_ablation_samples(config)

    assert len(samples) == 2 * 3 * len(COVARIANCE_VARIANTS)
    assert set(samples["variant"]) == set(COVARIANCE_VARIANTS)
    assert samples["cholesky_success"].all()

    for _, group in samples.groupby(["seed", "interval_index"], sort=False):
        assert len(group) == len(COVARIANCE_VARIANTS)
        np.testing.assert_allclose(
            group[list(RESIDUAL_COLUMNS)].to_numpy(dtype=float),
            np.repeat(
                group.iloc[[0]][list(RESIDUAL_COLUMNS)].to_numpy(dtype=float),
                len(group),
                axis=0,
            ),
            rtol=0.0,
            atol=0.0,
        )

        by_variant = group.set_index("variant")
        assert bool(by_variant.loc["current", "bias_rw_enabled"])
        assert not bool(by_variant.loc["no_bias_rw", "bias_rw_enabled"])
        assert float(by_variant.loc["current", "covariance_floor"]) == 1e-8
        assert float(by_variant.loc["no_floor", "covariance_floor"]) == 0.0
        assert np.isclose(
            float(by_variant.loc["current", "covariance_trace"])
            - float(by_variant.loc["no_floor", "covariance_trace"]),
            9e-8,
            rtol=1e-5,
            atol=1e-13,
        )
        assert np.isclose(
            float(by_variant.loc["no_bias_rw", "covariance_trace"])
            - float(by_variant.loc["no_bias_rw_no_floor", "covariance_trace"]),
            9e-8,
            rtol=1e-5,
            atol=1e-13,
        )


def test_current_variant_matches_original_whitening_samples():
    config = MonteCarloConfig(duration_s=0.1, seed_count=1, start_seed=73)
    baseline = collect_whitening_samples(config)
    ablation = collect_covariance_ablation_samples(config)

    validation = validate_current_variant_against_baseline(ablation, baseline)

    assert validation["row_count"] == len(baseline)
    assert validation["max_abs_residual_difference"] <= 1e-12
    assert validation["max_abs_whitened_difference"] <= 1e-12
    assert validation["max_abs_nis_difference"] <= 1e-12


def test_write_covariance_ablation_outputs_creates_comparison_without_decision(tmp_path):
    config = MonteCarloConfig(duration_s=0.1, seed_count=2, start_seed=89)
    samples = collect_covariance_ablation_samples(config)

    paths = write_covariance_ablation_outputs(samples, config, tmp_path)

    assert set(paths) == {
        "samples",
        "variant_summary",
        "component_summary",
        "interval_summary",
        "whitened_covariance",
        "metadata",
        "html",
    }
    assert all(path.is_file() for path in paths.values())

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["sample_count_per_variant"] == 2 * 3
    assert metadata["total_row_count"] == 2 * 3 * len(COVARIANCE_VARIANTS)
    assert metadata["variants"] == list(COVARIANCE_VARIANTS)
    assert not ({"pass", "fail", "decision", "classification"} & set(metadata))

    summary = pd.read_csv(paths["variant_summary"])
    assert set(summary["variant"]) == set(COVARIANCE_VARIANTS)
    assert summary["sample_count"].eq(2 * 3).all()

    html = paths["html"].read_text(encoding="utf-8")
    assert "IMU Covariance Source Ablation" in html
    assert 'id="variant-select"' in html
    assert 'id="nis-canvas"' in html
    assert all(variant in html for variant in COVARIANCE_VARIANTS)
    assert "https://" not in html


def test_covariance_ablation_script_entrypoint_can_load_repository_modules():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "Scripts" / "imu_covariance_source_ablation.py"),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--baseline-samples" in completed.stdout

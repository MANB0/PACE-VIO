import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from Scripts.imu_sampling_aware_covariance import (
    NOMINAL_RESIDUAL_COLUMNS,
    SOURCE_RESIDUAL_COLUMNS,
    apply_sampling_aware_covariance,
    compute_sampling_aware_references,
    write_sampling_aware_outputs,
)
from Scripts.imu_whitening_monte_carlo import (
    MonteCarloConfig,
    RESIDUAL_COLUMNS,
    WHITENED_COLUMNS,
    collect_whitening_samples,
)


@pytest.fixture(scope="module")
def small_config() -> MonteCarloConfig:
    return MonteCarloConfig(duration_s=0.1, seed_count=2, start_seed=131)


@pytest.fixture(scope="module")
def sampling_references(small_config):
    return compute_sampling_aware_references(small_config)


def test_sampling_aware_reference_uses_full_residual_jacobian_and_is_spd(
    small_config,
    sampling_references,
):
    assert len(sampling_references) == 3
    expected_raw_sample_count = int(round(small_config.duration_s * small_config.imu_rate_hz)) + 1

    for interval_index, reference in enumerate(sampling_references):
        assert reference.interval_index == interval_index
        assert reference.nominal_residual.shape == (9,)
        assert reference.residual_jacobian.shape == (9, expected_raw_sample_count * 6)
        assert reference.covariance.shape == (9, 9)
        assert torch.isfinite(reference.nominal_residual).all()
        assert torch.isfinite(reference.residual_jacobian).all()
        assert torch.isfinite(reference.covariance).all()
        torch.testing.assert_close(reference.covariance, reference.covariance.T)
        assert torch.linalg.eigvalsh(reference.covariance).min().item() > 0.0
        _, info = torch.linalg.cholesky_ex(reference.covariance)
        assert int(info.item()) == 0


def test_sampling_aware_whitening_reuses_source_samples_and_centers_nominal_residual(
    small_config,
    sampling_references,
):
    baseline = collect_whitening_samples(small_config)

    samples = apply_sampling_aware_covariance(baseline, sampling_references)

    assert len(samples) == len(baseline)
    assert samples[["seed", "interval_index"]].equals(
        baseline[["seed", "interval_index"]]
    )
    assert samples["variant"].eq("sampling_aware").all()
    np.testing.assert_allclose(
        samples[list(SOURCE_RESIDUAL_COLUMNS)].to_numpy(dtype=float),
        baseline[list(RESIDUAL_COLUMNS)].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    )

    source = samples[list(SOURCE_RESIDUAL_COLUMNS)].to_numpy(dtype=float)
    nominal = samples[list(NOMINAL_RESIDUAL_COLUMNS)].to_numpy(dtype=float)
    centered = samples[list(RESIDUAL_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(centered, source - nominal, rtol=0.0, atol=1e-15)

    whitened = samples[list(WHITENED_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(
        samples["nis_total"].to_numpy(dtype=float),
        np.square(whitened).sum(axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert samples["cholesky_success"].all()


def test_write_sampling_aware_outputs_includes_comparison_without_decision(
    tmp_path,
    small_config,
    sampling_references,
):
    baseline = collect_whitening_samples(small_config)
    samples = apply_sampling_aware_covariance(baseline, sampling_references)
    ablation_summary = pd.DataFrame(
        [
            {
                "variant": "current",
                "sample_count": len(samples),
                "cholesky_failure_count": 0,
                "nis_mean": 5.0,
                "nis_std": 2.0,
                "nis_q05": 1.0,
                "nis_q50": 4.5,
                "nis_q95": 9.0,
            },
            {
                "variant": "no_floor",
                "sample_count": len(samples),
                "cholesky_failure_count": 0,
                "nis_mean": 6.8,
                "nis_std": 3.0,
                "nis_q05": 1.2,
                "nis_q50": 6.1,
                "nis_q95": 12.0,
            },
        ]
    )

    paths = write_sampling_aware_outputs(
        samples,
        sampling_references,
        small_config,
        tmp_path,
        ablation_summary=ablation_summary,
    )

    assert set(paths) == {
        "samples",
        "interval_references",
        "component_summary",
        "interval_summary",
        "whitened_covariance",
        "comparison_summary",
        "metadata",
        "html",
    }
    assert all(path.is_file() for path in paths.values())

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["sample_count"] == len(samples)
    assert metadata["interval_count"] == 3
    assert metadata["covariance_model"] == "raw_sample_jacobian_jqjt"
    assert metadata["source_samples_reused"] is True
    assert not ({"pass", "fail", "decision", "classification"} & set(metadata))

    comparison = pd.read_csv(paths["comparison_summary"])
    assert comparison["variant"].tolist() == [
        "current",
        "no_floor",
        "sampling_aware",
    ]
    assert comparison.loc[
        comparison["variant"] == "sampling_aware", "sample_count"
    ].item() == len(samples)

    html = paths["html"].read_text(encoding="utf-8")
    assert "Sampling-aware IMU Covariance" in html
    assert 'id="comparison-canvas"' in html
    assert 'id="component-select"' in html
    assert all(name in html for name in ("current", "no_floor", "sampling_aware"))
    assert "https://" not in html


def test_sampling_aware_script_entrypoint_can_load_repository_modules():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "Scripts" / "imu_sampling_aware_covariance.py"),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--baseline-samples" in completed.stdout
    assert "--ablation-summary" in completed.stdout

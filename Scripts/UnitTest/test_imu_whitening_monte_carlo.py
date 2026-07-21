import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from Scripts.imu_whitening_monte_carlo import (
    MonteCarloConfig,
    collect_whitening_samples,
    write_analysis_outputs,
)


def test_collect_whitening_samples_uses_every_seed_and_camera_interval():
    config = MonteCarloConfig(
        duration_s=0.1,
        seed_count=3,
        start_seed=17,
    )

    samples = collect_whitening_samples(config)

    assert len(samples) == 3 * 3
    assert samples["seed"].nunique() == 3
    assert samples.groupby("seed")["interval_index"].nunique().eq(3).all()

    whitened_columns = [
        f"z_{block}_{axis}"
        for block in ("p", "v", "R")
        for axis in ("x", "y", "z")
    ]
    expected_nis = np.square(samples[whitened_columns].to_numpy(dtype=float)).sum(axis=1)
    np.testing.assert_allclose(samples["nis_total"], expected_nis, rtol=1e-10, atol=1e-10)

    assert np.isfinite(samples[whitened_columns + ["nis_total"]].to_numpy()).all()
    assert math.isclose(float(samples["dt_s"].min()), 1.0 / 30.0, abs_tol=1e-8)
    assert math.isclose(float(samples["dt_s"].max()), 1.0 / 30.0, abs_tol=1e-8)


def test_write_analysis_outputs_creates_evidence_without_classification(tmp_path):
    config = MonteCarloConfig(duration_s=0.1, seed_count=2, start_seed=29)
    samples = collect_whitening_samples(config)

    paths = write_analysis_outputs(samples, config, tmp_path)

    assert set(paths) == {
        "samples",
        "component_summary",
        "interval_summary",
        "whitened_covariance",
        "metadata",
        "html",
    }
    assert all(path.is_file() for path in paths.values())

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["sample_count"] == len(samples)
    assert metadata["seed_count"] == 2
    assert metadata["bias_mode"] == "zero_bias"
    assert metadata["noise_mode"] == "fixed_seed_normal"
    assert not ({"pass", "fail", "decision", "classification"} & set(metadata))

    html = paths["html"].read_text(encoding="utf-8")
    assert "IMU Whitening Monte Carlo" in html
    assert 'id="nis-canvas"' in html
    assert 'id="component-select"' in html
    assert "https://" not in html


def test_direct_script_entrypoint_can_load_repository_modules():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "Scripts" / "imu_whitening_monte_carlo.py"),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--seed-count" in completed.stdout

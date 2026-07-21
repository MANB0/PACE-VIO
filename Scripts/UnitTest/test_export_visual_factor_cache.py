from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pypose as pp
import pytest
import torch
import yaml

from Scripts import export_visual_factor_cache as exporter
from Scripts.export_visual_factor_cache import ExportVisualFactorCacheError, export_result_to_visual_cache, main
from Utility.Point import pixel2point_NED
from Utility.VisualFactorCache import MATCH_FIELDS, VisualFactorCacheReader
from Utility.VisualInputFingerprint import visual_input_sha256


DIAGNOSTIC_FIELDS = (
    "median_kp0_depth_cov",
    "p90_kp0_depth_cov",
    "mean_kp0_depth_cov",
    "median_kp1_depth_cov",
    "p90_kp1_depth_cov",
    "mean_kp1_depth_cov",
    "median_flow_u_cov",
    "p90_flow_u_cov",
    "mean_flow_u_cov",
    "median_flow_v_cov",
    "p90_flow_v_cov",
    "mean_flow_v_cov",
    "valid_depth_ratio",
    "num_selected_keypoints",
)


def test_git_revision_marks_dirty_worktree_with_reproducible_digest(monkeypatch):
    outputs = {
        ("git", "rev-parse", "HEAD"): "abc123\n",
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"): " M MACVO.py\n",
        ("git", "diff", "--binary", "HEAD"): "diff --git a/MACVO.py b/MACVO.py\n",
    }

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=outputs[tuple(command)], stderr="")

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)

    first = exporter._git_revision()
    second = exporter._git_revision()

    assert first == second
    assert first.startswith("abc123+dirty:")
    assert len(first.rsplit(":", 1)[1]) == 64


def _match_arrays(frame_count: int) -> dict[str, np.ndarray]:
    rows = 2 * (frame_count - 1)
    dtype = np.float64
    arrays = {
        "pixel1_uv": np.array(
            [[60.0 + index, 45.0 + index] for index in range(rows)], dtype=np.float32
        ),
        "pixel2_uv": np.array(
            [[61.0 + index, 46.0 + index] for index in range(rows)], dtype=np.float32
        ),
        "pixel1_d": np.array([[2.0 + 0.1 * index] for index in range(rows)], dtype=np.float32),
        "pixel2_d": np.array([[2.1 + 0.1 * index] for index in range(rows)], dtype=np.float32),
        "pixel1_disp": np.full((rows, 1), 5.0, dtype=np.float32),
        "pixel2_disp": np.full((rows, 1), 6.0, dtype=np.float32),
        "pixel1_disp_cov": np.full((rows, 1), 0.1, dtype=np.float32),
        "pixel2_disp_cov": np.full((rows, 1), 0.2, dtype=np.float32),
        "pixel1_d_cov": np.arange(1, 2 * rows + 1, 2, dtype=np.float32).reshape(rows, 1) / 10.0,
        "pixel2_d_cov": np.arange(2, 2 * rows + 1, 2, dtype=np.float32).reshape(rows, 1) / 10.0,
        "pixel1_uv_cov": np.tile(np.array([[0.1, 0.2, 0.0]], dtype=np.float32), (rows, 1)),
        "pixel2_uv_cov": np.array(
            [[1.0 + 2 * index, 2.0 + 2 * index, 0.0] for index in range(rows)], dtype=np.float32
        ),
        "obs1_covTc": np.tile(np.diag([0.01, 0.02, 0.03]).astype(dtype), (rows, 1, 1)),
        "obs2_covTc": np.tile(np.diag([0.04, 0.05, 0.06]).astype(dtype), (rows, 1, 1)),
    }
    assert set(arrays) == set(MATCH_FIELDS)
    return arrays


def _write_synthetic_result(tmp_path: Path, *, frame_count: int = 3) -> tuple[Path, Path, dict[str, np.ndarray]]:
    result_dir = tmp_path / "result"
    dataset_root = tmp_path / "dataset"
    result_dir.mkdir()
    dataset_root.mkdir()
    config = {
        "Odometry": {
            "args": {"mapping": False},
            "motion": {"type": "StaticMotionModel"},
            "keyframe": {"type": "AllKeyframe"},
        }
    }
    (result_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    K = np.array([[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            [2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )[:frame_count]
    fields = _match_arrays(frame_count)
    rows = len(fields["pixel1_uv"])
    frame1 = np.repeat(np.arange(frame_count - 1, dtype=np.int64), 2)
    frame2 = frame1 + 1
    local_points = pixel2point_NED(
        torch.from_numpy(fields["pixel1_uv"]).double(),
        torch.from_numpy(fields["pixel1_d"]).squeeze(-1).double(),
        torch.from_numpy(K).double(),
    )
    point_positions = torch.empty((rows, 3), dtype=torch.float64)
    point_covariances = torch.empty((rows, 3, 3), dtype=torch.float64)
    for index, source_frame in enumerate(frame1):
        pose = pp.SE3(torch.from_numpy(poses[source_frame]).double())
        rotation = pose.rotation().matrix()
        point_positions[index] = pose.Act(local_points[index])
        covariance = torch.from_numpy(fields["obs1_covTc"][index])
        point_covariances[index] = rotation @ covariance @ rotation.transpose(-1, -2)

    tensor_map = {
        "frames//K": np.repeat(K[None], frame_count, axis=0),
        "frames//baseline": np.full(frame_count, 0.12, dtype=np.float32),
        "frames//pose": poses,
        "frames//time_ns": np.arange(frame_count, dtype=np.int64) * 10 + 10,
        "edge/match2frame1/mapping": frame1,
        "edge/match2frame2/mapping": frame2,
        "edge/match2point/mapping": np.arange(rows, dtype=np.int64),
        "points//pos_Tc": local_points.numpy().astype(np.float32),
        "points//pos_Tw": point_positions.numpy().astype(np.float32),
        "points//cov_Tw": point_covariances.numpy(),
        "points//color": np.array(
            [[10 + 3 * index, 20 + 3 * index, 30 + 3 * index] for index in range(rows)], dtype=np.uint8
        ),
    }
    tensor_map.update({f"match//{name}": value for name, value in fields.items()})
    np.savez_compressed(result_dir / "tensor_map.npz", **tensor_map)
    return result_dir, dataset_root, tensor_map


def _load_tensor_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def _rewrite_tensor_map(result_dir: Path, mutate) -> None:
    tensor_map = _load_tensor_map(result_dir / "tensor_map.npz")
    mutate(tensor_map)
    np.savez_compressed(result_dir / "tensor_map.npz", **tensor_map)


def _drop_final_pair(tensor_map: dict[str, np.ndarray]) -> None:
    for name in MATCH_FIELDS:
        tensor_map[f"match//{name}"] = tensor_map[f"match//{name}"][:2]
    for name in (
        "edge/match2frame1/mapping",
        "edge/match2frame2/mapping",
        "edge/match2point/mapping",
        "points//pos_Tc",
        "points//pos_Tw",
        "points//cov_Tw",
        "points//color",
    ):
        tensor_map[name] = tensor_map[name][:2]


def _diagnostics_row(frame_i: int, frame_j: int, base: int, *, nan_field: str | None = None) -> dict:
    row = {
        name: float(base + index)
        for index, name in enumerate(DIAGNOSTIC_FIELDS[:-2])
    }
    row.update(
        {
            "frame_i": frame_i,
            "frame_j": frame_j,
            "valid_depth_ratio": float(base) / 100.0,
            "num_selected_keypoints": base + 100,
        }
    )
    if nan_field is not None:
        row[nan_field] = float("nan")
    return row


def _write_diagnostics_csv(result_dir: Path, rows: list[dict]) -> None:
    with (result_dir / "frame_pair_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_i", "frame_j", *DIAGNOSTIC_FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def _visual_diagnostics_row(
    frame_i: int,
    frame_j: int,
    base: int,
    timestamp_i: int,
    timestamp_j: int,
    *,
    nan_field: str | None = None,
) -> dict:
    return {
        **_diagnostics_row(frame_i, frame_j, base, nan_field=nan_field),
        "timestamp_i": timestamp_i,
        "timestamp_j": timestamp_j,
    }


def _write_visual_diagnostics_csv(
    result_dir: Path,
    rows: list[dict],
    *,
    include_timestamps: bool = True,
) -> None:
    with np.load(result_dir / "tensor_map.npz", allow_pickle=False) as data:
        tensor_map = {name: np.asarray(data[name]) for name in data.files}
    frame1 = tensor_map["edge/match2frame1/mapping"]
    frame2 = tensor_map["edge/match2frame2/mapping"]
    normalized_rows = []
    for source_row in rows:
        row = dict(source_row)
        pair = (int(row["frame_i"]), int(row["frame_j"]))
        selected = np.flatnonzero((frame1 == pair[0]) & (frame2 == pair[1]))
        if "visual_input_sha256" not in row:
            if len(selected):
                fields = {
                    name: torch.from_numpy(tensor_map[f"match//{name}"][selected])
                    for name in MATCH_FIELDS
                }
                row["visual_input_sha256"] = visual_input_sha256(fields)
            else:
                row["visual_input_sha256"] = "unavailable"
        normalized_rows.append(row)
    identity_fields = ["frame_i", "frame_j"]
    if include_timestamps:
        identity_fields.extend(["timestamp_i", "timestamp_j"])
    identity_fields.append("visual_input_sha256")
    with (result_dir / "visual_factor_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*identity_fields, *DIAGNOSTIC_FIELDS])
        writer.writeheader()
        writer.writerows(normalized_rows)


def _assert_diagnostics_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for name, expected_value in expected.items():
        if isinstance(expected_value, float) and math.isnan(expected_value):
            assert math.isnan(actual[name])
        else:
            assert actual[name] == expected_value


def _derived_diagnostics(tensor_map: dict[str, np.ndarray], frame_i: int) -> dict[str, float | int]:
    rows = slice(frame_i * 2, frame_i * 2 + 2)

    def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
        filtered = values.reshape(-1)
        filtered = filtered[~np.isnan(filtered)]
        if len(filtered) == 0:
            return {
                f"median_{prefix}": float("nan"),
                f"p90_{prefix}": float("nan"),
                f"mean_{prefix}": float("nan"),
            }
        return {
            f"median_{prefix}": float(np.median(filtered)),
            f"p90_{prefix}": float(np.percentile(filtered, 90)),
            f"mean_{prefix}": float(np.mean(filtered)),
        }

    pixel1_depth = tensor_map["match//pixel1_d"][rows].reshape(-1)
    valid_depth = np.isfinite(pixel1_depth) & (pixel1_depth > 0)
    diagnostics = {
        **stats(tensor_map["match//pixel1_d_cov"][rows], "kp0_depth_cov"),
        **stats(tensor_map["match//pixel2_d_cov"][rows], "kp1_depth_cov"),
        **stats(tensor_map["match//pixel2_uv_cov"][rows, 0], "flow_u_cov"),
        **stats(tensor_map["match//pixel2_uv_cov"][rows, 1], "flow_v_cov"),
        "valid_depth_ratio": float(np.mean(valid_depth)),
        "num_selected_keypoints": 2,
    }
    assert set(diagnostics) == set(DIAGNOSTIC_FIELDS)
    return diagnostics


def test_export_imports_exact_csv_diagnostics_by_pair_without_changing_visual_inputs(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    baseline_cache = export_result_to_visual_cache(
        result_dir, tmp_path / "baseline-cache", "synthetic-scene", dataset_root
    )
    expected_first = _diagnostics_row(0, 1, 10, nan_field="p90_flow_v_cov")
    expected_final = _diagnostics_row(1, 2, 40)
    _write_diagnostics_csv(result_dir, [expected_final, expected_first])

    csv_cache = export_result_to_visual_cache(result_dir, tmp_path / "csv-cache", "synthetic-scene", dataset_root)

    baseline_reader = VisualFactorCacheReader(baseline_cache)
    csv_reader = VisualFactorCacheReader(csv_cache)
    for frame_i, expected in ((0, expected_first), (1, expected_final)):
        timestamp_i = 10 + frame_i * 10
        baseline_packet = baseline_reader.load_pair(frame_i, frame_i + 1, timestamp_i, timestamp_i + 10)
        csv_packet = csv_reader.load_pair(frame_i, frame_i + 1, timestamp_i, timestamp_i + 10)
        expected = {name: expected[name] for name in DIAGNOSTIC_FIELDS}

        _assert_diagnostics_equal(csv_packet.covariance_diagnostics, expected)
        assert csv_packet.visual_sha256 == baseline_packet.visual_sha256
        assert csv_packet.visual_sha256 == visual_input_sha256(csv_packet.match_fields)
        for name in MATCH_FIELDS:
            assert torch.equal(csv_packet.match_fields[name], baseline_packet.match_fields[name])
        for name in ("K", "relative_pose_init", "points_local", "points_cov_local", "point_colors"):
            assert torch.equal(getattr(csv_packet, name), getattr(baseline_packet, name))


def test_export_preserves_exact_runtime_local_points_instead_of_reprojecting(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)

    def perturb_exact_runtime_point(tensor_map: dict[str, np.ndarray]) -> None:
        delta = np.float32(3.814697265625e-06)
        tensor_map["points//pos_Tc"][0, 0] += delta
        tensor_map["points//pos_Tw"][0, 0] += delta

    _rewrite_tensor_map(result_dir, perturb_exact_runtime_point)
    cache_dir = export_result_to_visual_cache(
        result_dir,
        tmp_path / "cache",
        "synthetic-scene",
        dataset_root,
    )

    with np.load(result_dir / "tensor_map.npz", allow_pickle=False) as tensor_map:
        expected = torch.from_numpy(np.array(tensor_map["points//pos_Tc"][:2], copy=True))
    packet = VisualFactorCacheReader(cache_dir).load_pair(0, 1, 10, 20)

    assert torch.equal(packet.points_local, expected)


def test_export_rejects_source_without_exact_runtime_local_points(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)

    def remove_exact_runtime_points(tensor_map: dict[str, np.ndarray]) -> None:
        tensor_map.pop("points//pos_Tc")

    _rewrite_tensor_map(result_dir, remove_exact_runtime_points)

    with pytest.raises(ExportVisualFactorCacheError, match=r"points//pos_Tc"):
        export_result_to_visual_cache(
            result_dir,
            tmp_path / "cache",
            "synthetic-scene",
            dataset_root,
        )


def test_authoritative_visual_diagnostics_requires_complete_pair_coverage(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    _write_diagnostics_csv(
        result_dir,
        [_diagnostics_row(0, 1, 70), _diagnostics_row(1, 2, 90)],
    )
    _write_visual_diagnostics_csv(
        result_dir,
        [_visual_diagnostics_row(0, 1, 10, 10, 20)],
    )

    with pytest.raises(ExportVisualFactorCacheError, match=r"visual_factor_diagnostics.*missing.*\(1, 2\)"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


@pytest.mark.parametrize("extra_pair", [(-1, 0), (0, 2), (2, 3)])
def test_authoritative_visual_diagnostics_rejects_unexpected_pairs(tmp_path: Path, extra_pair: tuple[int, int]):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    rows = [
        _visual_diagnostics_row(0, 1, 10, 10, 20),
        _visual_diagnostics_row(1, 2, 40, 20, 30),
        _visual_diagnostics_row(*extra_pair, 70, 30, 40),
    ]
    _write_visual_diagnostics_csv(result_dir, rows)

    with pytest.raises(ExportVisualFactorCacheError, match=r"visual_factor_diagnostics.*line 4.*pair"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


def test_authoritative_visual_diagnostics_requires_timestamp_columns(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    _write_visual_diagnostics_csv(
        result_dir,
        [_diagnostics_row(0, 1, 10), _diagnostics_row(1, 2, 40)],
        include_timestamps=False,
    )

    with pytest.raises(ExportVisualFactorCacheError, match=r"visual_factor_diagnostics.*timestamp_i.*timestamp_j"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [("timestamp_i", 11), ("timestamp_j", 21)],
)
def test_authoritative_visual_diagnostics_requires_exact_timestamps(
    tmp_path: Path,
    field: str,
    wrong_value: int,
):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    first = _visual_diagnostics_row(0, 1, 10, 10, 20)
    first[field] = wrong_value
    _write_visual_diagnostics_csv(
        result_dir,
        [first, _visual_diagnostics_row(1, 2, 40, 20, 30)],
    )

    with pytest.raises(ExportVisualFactorCacheError, match=rf"{field}.*visual_factor_diagnostics.*line 2"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


def test_authoritative_visual_diagnostics_rejects_hash_that_differs_from_tensor_map(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    first = _visual_diagnostics_row(0, 1, 10, 10, 20)
    first["visual_input_sha256"] = "0" * 64
    _write_visual_diagnostics_csv(
        result_dir,
        [first, _visual_diagnostics_row(1, 2, 40, 20, 30)],
    )

    with pytest.raises(ExportVisualFactorCacheError, match=r"visual.*hash.*line 2"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


@pytest.mark.parametrize(
    ("field", "missing_value"),
    [
        ("mean_flow_v_cov", float("nan")),
        ("mean_flow_v_cov", ""),
        ("num_selected_keypoints", float("nan")),
        ("num_selected_keypoints", ""),
    ],
)
def test_authoritative_visual_diagnostics_preserves_nan_and_missing_semantics(
    tmp_path: Path,
    field: str,
    missing_value: float | str,
):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    expected_first = _visual_diagnostics_row(0, 1, 10, 10, 20)
    expected_first[field] = missing_value
    expected_final = _visual_diagnostics_row(1, 2, 40, 20, 30)
    _write_visual_diagnostics_csv(result_dir, [expected_first, expected_final])

    cache_dir = export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)
    packet = VisualFactorCacheReader(cache_dir).load_pair(0, 1, 10, 20)

    assert math.isnan(packet.covariance_diagnostics[field])


@pytest.mark.parametrize("unexpected_pair", [(-1, 0), (0, 2), (2, 3)])
def test_legacy_diagnostics_rejects_rows_outside_expected_pair_set(
    tmp_path: Path,
    unexpected_pair: tuple[int, int],
):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    _write_diagnostics_csv(
        result_dir,
        [_diagnostics_row(0, 1, 10), _diagnostics_row(*unexpected_pair, 70)],
    )

    with pytest.raises(ExportVisualFactorCacheError, match=r"frame_pair_diagnostics.*line 3.*pair"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


@pytest.mark.parametrize("authoritative", [False, True])
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
@pytest.mark.parametrize("field", DIAGNOSTIC_FIELDS)
def test_diagnostics_rejects_infinite_float_with_line_specific_error(
    tmp_path: Path,
    authoritative: bool,
    value: float,
    field: str,
):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    first = _diagnostics_row(0, 1, 10)
    first[field] = value
    if authoritative:
        _write_visual_diagnostics_csv(
            result_dir,
            [
                {**first, "timestamp_i": 10, "timestamp_j": 20},
                _visual_diagnostics_row(1, 2, 40, 20, 30),
            ],
        )
        source_name = "visual_factor_diagnostics"
    else:
        _write_diagnostics_csv(result_dir, [first])
        source_name = "frame_pair_diagnostics"

    with pytest.raises(
        ExportVisualFactorCacheError,
        match=rf"{field}.*{source_name}.*line 2",
    ):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


def test_export_derives_missing_final_pair_diagnostics(tmp_path: Path):
    result_dir, dataset_root, tensor_map = _write_synthetic_result(tmp_path)
    _write_diagnostics_csv(result_dir, [_diagnostics_row(0, 1, 10)])

    cache_dir = export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)
    final = VisualFactorCacheReader(cache_dir).load_pair(1, 2, 20, 30)

    _assert_diagnostics_equal(final.covariance_diagnostics, _derived_diagnostics(tensor_map, 1))


def test_export_derives_every_pair_diagnostics_when_csv_is_absent(tmp_path: Path):
    result_dir, dataset_root, tensor_map = _write_synthetic_result(tmp_path)

    cache_dir = export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)

    reader = VisualFactorCacheReader(cache_dir)
    first = reader.load_pair(0, 1, 10, 20)
    final = reader.load_pair(1, 2, 20, 30)
    expected_points_local = pixel2point_NED(
        torch.from_numpy(tensor_map["match//pixel1_uv"][:2]),
        torch.from_numpy(tensor_map["match//pixel1_d"][:2]).squeeze(-1),
        torch.from_numpy(tensor_map["frames//K"][0]),
    )

    assert [pair["path"] for pair in reader.manifest.pairs] == [
        "pairs/000000_000001.npz",
        "pairs/000001_000002.npz",
    ]
    assert first.points_local.dtype == torch.float32
    assert torch.equal(first.points_local, expected_points_local)
    assert torch.equal(first.points_cov_local, first.match_fields["obs1_covTc"])
    assert first.point_colors.tolist() == [[10, 20, 30], [13, 23, 33]]
    assert first.visual_sha256 == visual_input_sha256(first.match_fields)
    assert final.frame_i == 1 and final.frame_j == 2
    _assert_diagnostics_equal(first.covariance_diagnostics, _derived_diagnostics(tensor_map, 0))
    _assert_diagnostics_equal(final.covariance_diagnostics, _derived_diagnostics(tensor_map, 1))
    assert all(value is None or isinstance(value, (str, int, float, bool)) for value in reader.manifest.source.values())


def test_export_static_motion_initialization_is_identity_despite_finalized_source_poses(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)

    cache_dir = export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)

    reader = VisualFactorCacheReader(cache_dir)
    expected = torch.eye(4, dtype=torch.float64)
    assert torch.equal(reader.load_pair(0, 1, 10, 20).relative_pose_init, expected)
    assert torch.equal(reader.load_pair(1, 2, 20, 30).relative_pose_init, expected)


@pytest.mark.parametrize("authoritative", [False, True])
def test_export_rejects_duplicate_diagnostics_pair_rows(tmp_path: Path, authoritative: bool):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    if authoritative:
        _write_visual_diagnostics_csv(
            result_dir,
            [
                _visual_diagnostics_row(0, 1, 10, 10, 20),
                _visual_diagnostics_row(0, 1, 20, 10, 20),
                _visual_diagnostics_row(1, 2, 40, 20, 30),
            ],
        )
    else:
        _write_diagnostics_csv(
            result_dir,
            [_diagnostics_row(0, 1, 10), _diagnostics_row(0, 1, 20)],
        )

    with pytest.raises(ExportVisualFactorCacheError, match=r"duplicate.*\(0, 1\)"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


def test_cli_exports_the_same_cache(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    cache_dir = tmp_path / "cache"

    assert main([str(result_dir), str(cache_dir), "synthetic-scene", str(dataset_root)]) == 0
    assert VisualFactorCacheReader(cache_dir).load_pair(1, 2, 20, 30).frame_j == 2


def test_cli_script_is_directly_invokable(tmp_path: Path):
    script_path = Path(__file__).parents[1] / "export_visual_factor_cache.py"

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_final_pair, "missing matches"),
        (
            lambda data: data["edge/match2frame2/mapping"].__setitem__(1, 2),
            "consecutive",
        ),
        (
            lambda data: data.__setitem__("match//pixel2_d", data["match//pixel2_d"][:-1]),
            "row count",
        ),
        (
            lambda data: data["points//pos_Tw"].__setitem__((0, 0), data["points//pos_Tw"][0, 0] + 0.0011),
            "world point",
        ),
        (
            lambda data: data["points//cov_Tw"].__setitem__((0, 0, 0), data["points//cov_Tw"][0, 0, 0] + 0.0011),
            "world covariance",
        ),
    ],
)
def test_export_rejects_invalid_tensor_map(tmp_path: Path, mutate, message: str):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    _rewrite_tensor_map(result_dir, mutate)

    with pytest.raises(ExportVisualFactorCacheError, match=message):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


@pytest.mark.parametrize(
    ("config_path", "replacement", "message"),
    [
        ("Odometry.motion.type", "DynamicMotionModel", "StaticMotionModel"),
        ("Odometry.keyframe.type", "SomeKeyframeSelector", "AllKeyframe"),
    ],
)
def test_export_requires_compatible_source_config(tmp_path: Path, config_path: str, replacement: str, message: str):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))
    parent, leaf = config_path.rsplit(".", 1)
    target = config
    for key in parent.split("."):
        target = target[key]
    target[leaf] = replacement
    (result_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ExportVisualFactorCacheError, match=message):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


def test_export_requires_explicit_false_mapping_config(tmp_path: Path):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))
    del config["Odometry"]["args"]["mapping"]
    (result_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ExportVisualFactorCacheError, match="args.mapping must be exactly False"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)


@pytest.mark.parametrize("mapping", [True, "false", 0, 1, None, [], {}])
def test_export_rejects_nonfalse_or_nonboolean_mapping_config(tmp_path: Path, mapping):
    result_dir, dataset_root, _ = _write_synthetic_result(tmp_path)
    config = yaml.safe_load((result_dir / "config.yaml").read_text(encoding="utf-8"))
    config["Odometry"]["args"]["mapping"] = mapping
    (result_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ExportVisualFactorCacheError, match="args.mapping must be exactly False"):
        export_result_to_visual_cache(result_dir, tmp_path / "cache", "synthetic-scene", dataset_root)

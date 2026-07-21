import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from Utility.VisualFactorCache import (
    MATCH_FIELDS,
    VisualFactorCacheError,
    VisualFactorCacheReader,
    VisualFactorPacket,
    _cpu_float64_calibration,
    packet_sha256,
    write_visual_factor_cache,
)


def make_packet(frame_i: int = 0, frame_j: int = 1) -> VisualFactorPacket:
    rows = 2
    fields = {
        "pixel1_uv": torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        "pixel2_uv": torch.tensor([[11.0, 21.0], [31.0, 41.0]]),
        "pixel1_d": torch.full((rows, 1), 2.0),
        "pixel2_d": torch.full((rows, 1), 3.0),
        "pixel1_disp": torch.full((rows, 1), 5.0),
        "pixel2_disp": torch.full((rows, 1), 6.0),
        "pixel1_disp_cov": torch.full((rows, 1), 0.1),
        "pixel2_disp_cov": torch.full((rows, 1), 0.2),
        "pixel1_d_cov": torch.full((rows, 1), 0.3),
        "pixel2_d_cov": torch.full((rows, 1), 0.4),
        "pixel1_uv_cov": torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]]),
        "pixel2_uv_cov": torch.tensor([[0.2, 0.3, 0.0], [0.4, 0.5, 0.0]]),
        "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(rows, 1, 1),
        "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(rows, 1, 1) * 2.0,
    }
    return VisualFactorPacket(
        frame_i=frame_i,
        frame_j=frame_j,
        timestamp_i_ns=10 + frame_i * 10,
        timestamp_j_ns=10 + frame_j * 10,
        K=torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
        baseline_m=0.12,
        relative_pose_init=torch.eye(4, dtype=torch.float64),
        points_local=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        points_cov_local=torch.eye(3, dtype=torch.float64).repeat(rows, 1, 1),
        point_colors=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8),
        match_fields=fields,
        covariance_diagnostics={"mean_trace": 1.5, "rows": rows},
        visual_sha256="visual-input-sha",
    )


def write_one_packet_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    write_visual_factor_cache(cache_dir, "scene", [make_packet()], source={"frame_count": 2})
    return cache_dir


def replace_covariance_diagnostics_json(cache_dir: Path, payload: str) -> None:
    pair_path = cache_dir / "pairs" / "000000_000001.npz"
    with np.load(pair_path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    arrays["covariance_diagnostics_json"] = np.asarray(payload)
    np.savez_compressed(pair_path, **arrays)

    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pairs"][0]["sha256"] = packet_sha256(pair_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_visual_factor_packet_round_trip_preserves_every_tensor(tmp_path: Path):
    packet = make_packet()
    cache_dir = write_one_packet_cache(tmp_path)

    loaded = VisualFactorCacheReader(cache_dir).load_pair(0, 1, 10, 20)

    assert loaded.visual_sha256 == packet.visual_sha256
    assert loaded.covariance_diagnostics == packet.covariance_diagnostics
    for name in MATCH_FIELDS:
        assert torch.equal(loaded.match_fields[name], packet.match_fields[name])
    for name in ("K", "relative_pose_init", "points_local", "points_cov_local", "point_colors"):
        assert torch.equal(getattr(loaded, name), getattr(packet, name))


def test_schema_v1_writer_rejects_nonidentity_relative_pose_initialization(tmp_path: Path):
    relative_pose = torch.eye(4, dtype=torch.float64)
    relative_pose[0, 3] = 0.25
    packet = replace(make_packet(), relative_pose_init=relative_pose)

    with pytest.raises(VisualFactorCacheError, match="relative_pose_init.*identity"):
        write_visual_factor_cache(tmp_path / "cache", "scene", [packet], source={"frame_count": 2})


def test_schema_v1_reader_rejects_nonidentity_relative_pose_initialization(tmp_path: Path):
    cache_dir = write_one_packet_cache(tmp_path)
    pair_path = cache_dir / "pairs" / "000000_000001.npz"
    with np.load(pair_path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    arrays["relative_pose_init"][0, 3] = 0.25
    np.savez_compressed(pair_path, **arrays)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pairs"][0]["sha256"] = packet_sha256(pair_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VisualFactorCacheError, match="relative_pose_init.*identity"):
        VisualFactorCacheReader(cache_dir).load_pair(0, 1, 10, 20)


def test_covariance_diagnostics_nan_round_trip_preserves_missing_semantics(tmp_path: Path):
    packet = replace(
        make_packet(),
        covariance_diagnostics={"mean_trace": float("nan"), "rows": 2},
    )
    cache_dir = tmp_path / "cache"

    write_visual_factor_cache(cache_dir, "scene", [packet], source={"frame_count": 2})
    loaded = VisualFactorCacheReader(cache_dir).load_pair(0, 1, 10, 20)

    assert math.isnan(loaded.covariance_diagnostics["mean_trace"])
    assert loaded.covariance_diagnostics["rows"] == 2


def test_covariance_diagnostics_nan_serializes_as_json_null(tmp_path: Path):
    packet = replace(make_packet(), covariance_diagnostics={"missing": float("nan")})
    cache_dir = tmp_path / "cache"

    write_visual_factor_cache(cache_dir, "scene", [packet], source={"frame_count": 2})

    with np.load(cache_dir / "pairs" / "000000_000001.npz", allow_pickle=False) as data:
        payload = str(data["covariance_diagnostics_json"].item())
    assert json.loads(payload) == {"missing": None}
    assert "NaN" not in payload


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_reader_rejects_nonstandard_covariance_json_constants(tmp_path: Path, constant: str):
    cache_dir = write_one_packet_cache(tmp_path)
    replace_covariance_diagnostics_json(cache_dir, f'{{"invalid": {constant}}}')

    with pytest.raises(VisualFactorCacheError, match="non-standard JSON constant"):
        VisualFactorCacheReader(cache_dir).load_pair(0, 1, 10, 20)


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), True, "missing", [], {}, torch.tensor(1.0)],
)
def test_writer_rejects_invalid_covariance_diagnostic_values(tmp_path: Path, value):
    packet = replace(make_packet(), covariance_diagnostics={"invalid": value})

    with pytest.raises(VisualFactorCacheError, match="covariance diagnostics"):
        write_visual_factor_cache(tmp_path / "cache", "scene", [packet], source={"frame_count": 2})


def test_reader_rejects_packet_checksum_mismatch(tmp_path: Path):
    cache_dir = write_one_packet_cache(tmp_path)
    pair_path = cache_dir / "pairs" / "000000_000001.npz"
    pair_path.write_bytes(pair_path.read_bytes() + b"tampered")

    with pytest.raises(VisualFactorCacheError, match="checksum"):
        VisualFactorCacheReader(cache_dir).load_pair(0, 1, 10, 20)


def test_packet_sha256_hashes_exact_pair_bytes(tmp_path: Path):
    cache_dir = write_one_packet_cache(tmp_path)
    pair_path = cache_dir / "pairs" / "000000_000001.npz"

    assert packet_sha256(pair_path) == hashlib.sha256(pair_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda packet: packet.match_fields.pop("pixel1_uv"), "match fields"),
        (lambda packet: packet.match_fields.__setitem__("unexpected", torch.ones((2, 1))), "match fields"),
        (lambda packet: packet.match_fields.__setitem__("pixel2_uv", torch.ones((2, 3))), "shape"),
        (lambda packet: packet.match_fields.__setitem__("pixel2_d", torch.ones((3, 1))), "row count"),
        (lambda packet: packet.match_fields.__setitem__("pixel2_d", torch.tensor([[float("nan")], [1.0]])), "finite"),
        (
            lambda packet: packet.match_fields.__setitem__(
                "obs2_covTc", torch.tensor([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
            ),
            "symmetric",
        ),
    ],
)
def test_writer_rejects_invalid_packet_content(tmp_path: Path, mutate, message: str):
    packet = make_packet()
    mutate(packet)

    with pytest.raises(VisualFactorCacheError, match=message):
        write_visual_factor_cache(tmp_path / "cache", "scene", [packet], source={"frame_count": 2})


def test_writer_rejects_noncontiguous_pairs(tmp_path: Path):
    with pytest.raises(VisualFactorCacheError, match="contiguous"):
        write_visual_factor_cache(
            tmp_path / "cache",
            "scene",
            [make_packet(0, 1), make_packet(2, 3)],
            source={"frame_count": 4},
        )


def test_writer_serializes_only_allowlisted_scalar_source_provenance(tmp_path: Path):
    source = {
        "frame_count": 2,
        "dataset": "synthetic-clear",
        "result": 17,
        "config": 0.5,
        "git": True,
        "exporter": None,
        "motion": "circle",
        "keyframes": 1,
        "timestamps": 20,
        "checksums": "images-sha",
    }

    write_visual_factor_cache(tmp_path / "cache", "scene", [make_packet()], source=source)

    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == source


@pytest.mark.parametrize(
    ("source", "description"),
    [
        ({"frame_count": 2, "dataset": {"id": torch.eye(4)}}, "hidden 4x4 pose"),
        ({"frame_count": 2, "checksums": torch.tensor([1.0, 2.0, 3.0])}, "numeric vector"),
        ({"frame_count": 2, "config": {"id": "config-1"}}, "dict"),
        ({"frame_count": 2, "keyframes": [0, 1]}, "list"),
        ({"frame_count": 2, "motion": ("circle",)}, "tuple"),
        ({"frame_count": 2, "result": np.array([1, 2])}, "array"),
    ],
)
def test_writer_rejects_non_scalar_source_provenance(tmp_path: Path, source: dict, description: str):
    with pytest.raises(VisualFactorCacheError, match="scalar provenance"):
        write_visual_factor_cache(tmp_path / "cache", "scene", [make_packet()], source=source)


def test_writer_rejects_unknown_source_metadata_key(tmp_path: Path):
    with pytest.raises(VisualFactorCacheError, match="source"):
        write_visual_factor_cache(
            tmp_path / "cache",
            "scene",
            [make_packet()],
            source={"frame_count": 2, "unrelated": "metadata"},
        )


def test_reader_validate_run_rejects_metadata_mismatches(tmp_path: Path):
    cache_dir = write_one_packet_cache(tmp_path)
    reader = VisualFactorCacheReader(cache_dir)
    packet = make_packet()

    reader.validate_run(
        scene="scene",
        frame_count=2,
        timestamps_ns=[10, 20],
        K=packet.K,
        baseline_m=packet.baseline_m,
    )
    with pytest.raises(VisualFactorCacheError, match="scene"):
        reader.validate_run(scene="other", frame_count=2, timestamps_ns=[10, 20], K=packet.K, baseline_m=packet.baseline_m)
    with pytest.raises(VisualFactorCacheError, match="frame_count"):
        reader.validate_run(scene="scene", frame_count=3, timestamps_ns=[10, 20], K=packet.K, baseline_m=packet.baseline_m)
    with pytest.raises(VisualFactorCacheError, match="timestamps"):
        reader.validate_run(scene="scene", frame_count=2, timestamps_ns=[10, 21], K=packet.K, baseline_m=packet.baseline_m)
    with pytest.raises(VisualFactorCacheError, match="K"):
        reader.validate_run(scene="scene", frame_count=2, timestamps_ns=[10, 20], K=packet.K * 2, baseline_m=packet.baseline_m)
    with pytest.raises(VisualFactorCacheError, match="baseline"):
        reader.validate_run(scene="scene", frame_count=2, timestamps_ns=[10, 20], K=packet.K, baseline_m=0.2)


def test_reader_validate_run_rejects_incompatible_calibration_shape(tmp_path: Path):
    reader = VisualFactorCacheReader(write_one_packet_cache(tmp_path))

    with pytest.raises(VisualFactorCacheError, match="K"):
        reader.validate_run(
            scene="scene",
            frame_count=2,
            timestamps_ns=[10, 20],
            K=torch.eye(4),
            baseline_m=0.12,
        )


def test_cpu_calibration_helper_normalizes_dtype_and_device():
    normalized = _cpu_float64_calibration(torch.eye(3, dtype=torch.float32))

    assert normalized.device.type == "cpu"
    assert normalized.dtype == torch.float64
    assert torch.equal(normalized, torch.eye(3, dtype=torch.float64))


def test_reader_validate_run_wraps_malformed_calibration_as_cache_error(tmp_path: Path):
    reader = VisualFactorCacheReader(write_one_packet_cache(tmp_path))

    with pytest.raises(VisualFactorCacheError, match="K"):
        reader.validate_run(
            scene="scene",
            frame_count=2,
            timestamps_ns=[10, 20],
            K=object(),
            baseline_m=0.12,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_reader_validate_run_accepts_cuda_resident_calibration(tmp_path: Path):
    reader = VisualFactorCacheReader(write_one_packet_cache(tmp_path))

    reader.validate_run(
        scene="scene",
        frame_count=2,
        timestamps_ns=[10, 20],
        K=make_packet().K.to(device="cuda"),
        baseline_m=0.12,
    )


def test_reader_rejects_manifest_with_invalid_checksum_metadata(tmp_path: Path):
    cache_dir = write_one_packet_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pairs"][0]["sha256"] = "not-a-sha256"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VisualFactorCacheError, match="checksum"):
        VisualFactorCacheReader(cache_dir)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.__setitem__("backend_pose", torch.eye(4).tolist()),
        lambda manifest: manifest.__setitem__("world_points", [[1.0, 2.0, 3.0]]),
        lambda manifest: manifest.__setitem__("GraphInput", "backend-state"),
        lambda manifest: manifest.pop("scene"),
    ],
)
def test_reader_rejects_nonexact_manifest_top_level_keys(tmp_path: Path, mutate):
    cache_dir = write_one_packet_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VisualFactorCacheError, match="manifest top-level keys"):
        VisualFactorCacheReader(cache_dir)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pair: pair.__setitem__("backend_pose", torch.eye(4).tolist()),
        lambda pair: pair.__setitem__("world_points", [[1.0, 2.0, 3.0]]),
        lambda pair: pair.__setitem__("GraphInput", "backend-state"),
        lambda pair: pair.pop("path"),
    ],
)
def test_reader_rejects_nonexact_manifest_pair_keys(tmp_path: Path, mutate):
    cache_dir = write_one_packet_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest["pairs"][0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VisualFactorCacheError, match="manifest pair keys"):
        VisualFactorCacheReader(cache_dir)


def test_writer_rejects_fractional_packet_timestamp(tmp_path: Path):
    packet = replace(make_packet(), timestamp_i_ns=10.5)

    with pytest.raises(VisualFactorCacheError, match="timestamp_i_ns.*integer"):
        write_visual_factor_cache(tmp_path / "cache", "scene", [packet], source={"frame_count": 2})


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda manifest: manifest["timestamps_ns"].__setitem__(0, 10.5), "timestamps_ns"),
        (lambda manifest: manifest.__setitem__("schema_version", 1.0), "schema_version"),
        (lambda manifest: manifest.__setitem__("frame_count", 2.0), "frame_count"),
        (lambda manifest: manifest["pairs"][0].__setitem__("frame_i", 0.0), "frame_i"),
    ],
)
def test_reader_rejects_fractional_manifest_integer_fields(tmp_path: Path, mutate, field: str):
    cache_dir = write_one_packet_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VisualFactorCacheError, match=field):
        VisualFactorCacheReader(cache_dir)


def test_writer_accepts_numpy_integer_packet_and_source_fields(tmp_path: Path):
    packet = replace(
        make_packet(),
        frame_i=np.int64(0),
        frame_j=np.int64(1),
        timestamp_i_ns=np.int64(10),
        timestamp_j_ns=np.int64(20),
    )
    cache_dir = tmp_path / "cache"

    write_visual_factor_cache(cache_dir, "scene", [packet], source={"frame_count": np.int64(2)})

    loaded = VisualFactorCacheReader(cache_dir).load_pair(0, 1, 10, 20)
    assert loaded.frame_i == 0
    assert loaded.timestamp_j_ns == 20

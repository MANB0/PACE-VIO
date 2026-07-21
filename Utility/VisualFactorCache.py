"""Portable cache format for replaying visual factors without the frontend."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 1
MATCH_FIELDS = (
    "pixel1_uv", "pixel2_uv", "pixel1_d", "pixel2_d",
    "pixel1_disp", "pixel2_disp",
    "pixel1_disp_cov", "pixel2_disp_cov",
    "pixel1_d_cov", "pixel2_d_cov",
    "pixel1_uv_cov", "pixel2_uv_cov",
    "obs1_covTc", "obs2_covTc",
)
FRONTEND_COVARIANCE_DIAGNOSTIC_FIELDS = (
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

_MATCH_SHAPES = {
    "pixel1_uv": (2,),
    "pixel2_uv": (2,),
    "pixel1_d": (1,),
    "pixel2_d": (1,),
    "pixel1_disp": (1,),
    "pixel2_disp": (1,),
    "pixel1_disp_cov": (1,),
    "pixel2_disp_cov": (1,),
    "pixel1_d_cov": (1,),
    "pixel2_d_cov": (1,),
    "pixel1_uv_cov": (3,),
    "pixel2_uv_cov": (3,),
    "obs1_covTc": (3, 3),
    "obs2_covTc": (3, 3),
}
_TENSOR_NAMES = ("K", "relative_pose_init", "points_local", "points_cov_local", "point_colors")
_SOURCE_PROVENANCE_FIELDS = frozenset({
    "dataset", "result", "config", "git", "exporter", "motion", "keyframes", "timestamps", "checksums",
})
_MANIFEST_FIELDS = frozenset({
    "schema_version", "scene", "source", "frame_count", "timestamps_ns", "K", "baseline_m", "pairs",
})
_MANIFEST_PAIR_FIELDS = frozenset({"frame_i", "frame_j", "path", "sha256"})


class VisualFactorCacheError(RuntimeError):
    """Raised when a visual factor cache is malformed or incompatible."""


@dataclass(frozen=True)
class VisualFactorPacket:
    frame_i: int
    frame_j: int
    timestamp_i_ns: int
    timestamp_j_ns: int
    K: torch.Tensor
    baseline_m: float
    relative_pose_init: torch.Tensor
    points_local: torch.Tensor
    points_cov_local: torch.Tensor
    point_colors: torch.Tensor
    match_fields: dict[str, torch.Tensor]
    covariance_diagnostics: dict[str, float | int]
    visual_sha256: str


@dataclass(frozen=True)
class SourceMetadata:
    """Flat scalar run provenance that is safe to store in a cache manifest."""

    frame_count: int
    provenance: dict[str, str | int | float | bool | None]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "SourceMetadata":
        if not isinstance(source, Mapping):
            raise VisualFactorCacheError("source metadata must be a mapping")
        source_keys = set(source)
        allowed_keys = _SOURCE_PROVENANCE_FIELDS | {"frame_count"}
        if not source_keys <= allowed_keys:
            raise VisualFactorCacheError("source metadata contains an unallowlisted key")
        try:
            frame_count = source["frame_count"]
        except KeyError as error:
            raise VisualFactorCacheError("source metadata requires frame_count") from error
        frame_count = _require_integer(frame_count, "source frame_count")
        if frame_count < 1:
            raise VisualFactorCacheError("source frame_count must be a positive integer")
        provenance = {
            key: _validate_provenance_scalar(value)
            for key, value in source.items()
            if key != "frame_count"
        }
        return cls(frame_count=frame_count, provenance=provenance)

    def to_dict(self) -> dict[str, Any]:
        return {"frame_count": self.frame_count, **self.provenance}


@dataclass(frozen=True)
class VisualFactorCacheManifest:
    schema_version: int
    scene: str
    source: dict[str, Any]
    frame_count: int
    timestamps_ns: tuple[int, ...]
    K: tuple[tuple[float, ...], ...]
    baseline_m: float
    pairs: tuple[dict[str, Any], ...]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "VisualFactorCacheManifest":
        if not isinstance(data, Mapping) or set(data) != _MANIFEST_FIELDS:
            raise VisualFactorCacheError("manifest top-level keys do not match cache schema")
        try:
            raw_pairs = data["pairs"]
            if not isinstance(raw_pairs, list):
                raise TypeError("manifest pairs must be a list")
            if any(not isinstance(pair, Mapping) or set(pair) != _MANIFEST_PAIR_FIELDS for pair in raw_pairs):
                raise VisualFactorCacheError("manifest pair keys do not match cache schema")
            source = SourceMetadata.from_mapping(data["source"])
            pairs = tuple(_normalize_manifest_pair(pair) for pair in raw_pairs)
            manifest = cls(
                schema_version=_require_integer(data["schema_version"], "schema_version"),
                scene=str(data["scene"]),
                source=source.to_dict(),
                frame_count=_require_integer(data["frame_count"], "frame_count"),
                timestamps_ns=tuple(_require_integer(value, "timestamps_ns") for value in data["timestamps_ns"]),
                K=tuple(tuple(float(value) for value in row) for row in data["K"]),
                baseline_m=float(data["baseline_m"]),
                pairs=pairs,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VisualFactorCacheError("invalid manifest") from error
        _validate_manifest(manifest)
        return manifest

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene": self.scene,
            "source": self.source,
            "frame_count": self.frame_count,
            "timestamps_ns": list(self.timestamps_ns),
            "K": [list(row) for row in self.K],
            "baseline_m": self.baseline_m,
            "pairs": list(self.pairs),
        }


def packet_sha256(path_or_bytes: str | Path | bytes | bytearray) -> str:
    """Return the SHA-256 digest of exact serialized NPZ bytes."""
    if isinstance(path_or_bytes, (str, Path)):
        payload = Path(path_or_bytes).read_bytes()
    else:
        payload = bytes(path_or_bytes)
    return hashlib.sha256(payload).hexdigest()


def write_visual_factor_cache(
    cache_dir: str | Path,
    scene: str,
    packets: Sequence[VisualFactorPacket],
    *,
    source: Mapping[str, Any],
) -> None:
    """Atomically write a complete visual-factor cache and its manifest."""
    cache_path = Path(cache_dir)
    ordered_packets = list(packets)
    source_metadata = SourceMetadata.from_mapping(source)
    _validate_packet_sequence(ordered_packets, source_metadata.frame_count)

    cache_path.mkdir(parents=True, exist_ok=True)
    pairs_path = cache_path / "pairs"
    pairs_path.mkdir(exist_ok=True)
    manifest_pairs: list[dict[str, Any]] = []
    for packet in ordered_packets:
        frame_i = _require_integer(packet.frame_i, "packet frame_i")
        frame_j = _require_integer(packet.frame_j, "packet frame_j")
        pair_name = f"{frame_i:06d}_{frame_j:06d}.npz"
        pair_path = pairs_path / pair_name
        _atomic_save_npz(pair_path, _packet_arrays(packet))
        manifest_pairs.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                "path": f"pairs/{pair_name}",
                "sha256": packet_sha256(pair_path),
            }
        )

    first = ordered_packets[0]
    manifest = VisualFactorCacheManifest(
        schema_version=SCHEMA_VERSION,
        scene=str(scene),
        source=source_metadata.to_dict(),
        frame_count=source_metadata.frame_count,
        timestamps_ns=tuple(
            [_require_integer(first.timestamp_i_ns, "packet timestamp_i_ns")]
            + [_require_integer(packet.timestamp_j_ns, "packet timestamp_j_ns") for packet in ordered_packets]
        ),
        K=tuple(tuple(float(value) for value in row) for row in first.K.detach().cpu().tolist()),
        baseline_m=float(first.baseline_m),
        pairs=tuple(manifest_pairs),
    )
    _atomic_write_json(cache_path / "manifest.json", manifest.to_json())


class VisualFactorCacheReader:
    def __init__(self, cache_dir: str | Path):
        """Load and validate manifest.json from cache_dir."""
        self.cache_dir = Path(cache_dir)
        try:
            data = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VisualFactorCacheError("unable to read manifest") from error
        self.manifest = VisualFactorCacheManifest.from_json(data)

    def validate_run(
        self,
        *,
        scene: str,
        frame_count: int,
        timestamps_ns: list[int],
        K: torch.Tensor,
        baseline_m: float,
    ) -> None:
        try:
            run_scene = str(scene)
            run_frame_count = _require_integer(frame_count, "frame_count")
            run_timestamps = tuple(_require_integer(value, "timestamps_ns") for value in timestamps_ns)
            run_baseline_m = float(baseline_m)
        except Exception as error:
            raise VisualFactorCacheError("invalid run metadata") from error
        run_K = _cpu_float64_calibration(K)
        try:
            manifest_K = _manifest_K(self.manifest)
            _validate_tensor("K", run_K, (3, 3))
            matching_K = torch.allclose(run_K, manifest_K, rtol=0.0, atol=0.0)
        except VisualFactorCacheError:
            raise
        except Exception as error:
            raise VisualFactorCacheError("invalid K calibration") from error
        if run_scene != self.manifest.scene:
            raise VisualFactorCacheError("scene differs from cache manifest")
        if run_frame_count != self.manifest.frame_count:
            raise VisualFactorCacheError("frame_count differs from cache manifest")
        if run_timestamps != self.manifest.timestamps_ns:
            raise VisualFactorCacheError("timestamps differ from cache manifest")
        if not matching_K:
            raise VisualFactorCacheError("K differs from cache manifest")
        if run_baseline_m != self.manifest.baseline_m:
            raise VisualFactorCacheError("baseline differs from cache manifest")

    def load_pair(
        self,
        frame_i: int,
        frame_j: int,
        timestamp_i_ns: int,
        timestamp_j_ns: int,
    ) -> VisualFactorPacket:
        requested_frame_i = _require_integer(frame_i, "frame_i")
        requested_frame_j = _require_integer(frame_j, "frame_j")
        requested_timestamp_i_ns = _require_integer(timestamp_i_ns, "timestamp_i_ns")
        requested_timestamp_j_ns = _require_integer(timestamp_j_ns, "timestamp_j_ns")
        pair = next(
            (
                pair for pair in self.manifest.pairs
                if pair["frame_i"] == requested_frame_i and pair["frame_j"] == requested_frame_j
            ),
            None,
        )
        if pair is None:
            raise VisualFactorCacheError("requested pair is not present in cache")
        if (
            self.manifest.timestamps_ns[requested_frame_i] != requested_timestamp_i_ns
            or self.manifest.timestamps_ns[requested_frame_j] != requested_timestamp_j_ns
        ):
            raise VisualFactorCacheError("pair timestamps differ from cache manifest")
        pair_path = self.cache_dir / pair["path"]
        try:
            actual_sha256 = packet_sha256(pair_path)
        except OSError as error:
            raise VisualFactorCacheError("unable to read packet checksum") from error
        if actual_sha256 != pair["sha256"]:
            raise VisualFactorCacheError("packet checksum mismatch")
        try:
            with np.load(pair_path, allow_pickle=False) as data:
                packet = _packet_from_arrays(data)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise VisualFactorCacheError("invalid packet payload") from error
        _validate_packet(packet)
        if (packet.frame_i, packet.frame_j) != (requested_frame_i, requested_frame_j):
            raise VisualFactorCacheError("packet pair indices differ from manifest")
        if (packet.timestamp_i_ns, packet.timestamp_j_ns) != (requested_timestamp_i_ns, requested_timestamp_j_ns):
            raise VisualFactorCacheError("packet timestamps differ from manifest")
        if not torch.equal(packet.K.to(torch.float64), _manifest_K(self.manifest)):
            raise VisualFactorCacheError("packet K differs from manifest")
        if float(packet.baseline_m) != self.manifest.baseline_m:
            raise VisualFactorCacheError("packet baseline differs from manifest")
        return packet


def _validate_packet_sequence(packets: Sequence[VisualFactorPacket], frame_count: int) -> None:
    if not packets:
        raise VisualFactorCacheError("cache requires at least one packet")
    if frame_count != len(packets) + 1:
        raise VisualFactorCacheError("source frame_count is inconsistent with contiguous pairs")
    first = packets[0]
    for expected_i, packet in enumerate(packets):
        _validate_packet(packet)
        if (packet.frame_i, packet.frame_j) != (expected_i, expected_i + 1):
            raise VisualFactorCacheError("packet pair indices must be contiguous")
        if expected_i and packet.timestamp_i_ns != packets[expected_i - 1].timestamp_j_ns:
            raise VisualFactorCacheError("packet timestamps must be contiguous")
        if not torch.equal(packet.K.to(torch.float64), first.K.to(torch.float64)):
            raise VisualFactorCacheError("packet K calibration differs")
        if float(packet.baseline_m) != float(first.baseline_m):
            raise VisualFactorCacheError("packet baseline calibration differs")


def _validate_packet(packet: VisualFactorPacket) -> None:
    frame_i = _require_integer(packet.frame_i, "packet frame_i")
    frame_j = _require_integer(packet.frame_j, "packet frame_j")
    timestamp_i_ns = _require_integer(packet.timestamp_i_ns, "packet timestamp_i_ns")
    timestamp_j_ns = _require_integer(packet.timestamp_j_ns, "packet timestamp_j_ns")
    if frame_i < 0 or frame_j != frame_i + 1:
        raise VisualFactorCacheError("packet pair indices must be consecutive")
    if timestamp_j_ns <= timestamp_i_ns:
        raise VisualFactorCacheError("packet timestamps must increase")
    _validate_tensor("K", packet.K, (3, 3))
    if not np.isfinite(packet.baseline_m) or packet.baseline_m <= 0.0:
        raise VisualFactorCacheError("baseline must be positive and finite")
    _validate_tensor("relative_pose_init", packet.relative_pose_init, (4, 4))
    relative_pose_init = torch.as_tensor(packet.relative_pose_init).detach().to(device="cpu", dtype=torch.float64)
    if not torch.equal(relative_pose_init, torch.eye(4, dtype=torch.float64)):
        raise VisualFactorCacheError("schema v1 relative_pose_init must be identity")
    _validate_tensor("points_local", packet.points_local, (None, 3))
    rows = int(packet.points_local.shape[0])
    _validate_tensor("points_cov_local", packet.points_cov_local, (rows, 3, 3))
    _validate_symmetric("points_cov_local", packet.points_cov_local)
    _validate_tensor("point_colors", packet.point_colors, (rows, 3))
    if set(packet.match_fields) != set(MATCH_FIELDS):
        raise VisualFactorCacheError("packet match fields do not match cache schema")
    for name in MATCH_FIELDS:
        _validate_tensor(name, packet.match_fields[name], (rows,) + _MATCH_SHAPES[name])
    _validate_symmetric("obs1_covTc", packet.match_fields["obs1_covTc"])
    _validate_symmetric("obs2_covTc", packet.match_fields["obs2_covTc"])
    if not isinstance(packet.visual_sha256, str) or not packet.visual_sha256:
        raise VisualFactorCacheError("visual_sha256 must be a non-empty string")
    if not isinstance(packet.covariance_diagnostics, dict):
        raise VisualFactorCacheError("covariance diagnostics must be a dictionary")
    for value in packet.covariance_diagnostics.values():
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise VisualFactorCacheError("covariance diagnostics must be int or float scalars")
        if np.isinf(value):
            raise VisualFactorCacheError("covariance diagnostics must not contain infinity")


def _validate_tensor(name: str, value: torch.Tensor, shape: tuple[int | None, ...]) -> None:
    tensor = torch.as_tensor(value)
    if tensor.ndim != len(shape):
        raise VisualFactorCacheError(f"{name} has invalid shape")
    if shape[0] is not None and tensor.shape[0] != shape[0]:
        raise VisualFactorCacheError(f"{name} has invalid row count")
    if any(expected is not None and tensor.shape[index] != expected for index, expected in enumerate(shape)):
        raise VisualFactorCacheError(f"{name} has invalid shape")
    if not torch.isfinite(tensor).all():
        raise VisualFactorCacheError(f"{name} must contain finite values")


def _validate_symmetric(name: str, value: torch.Tensor) -> None:
    tensor = torch.as_tensor(value)
    if not torch.allclose(tensor, tensor.transpose(-1, -2), rtol=0.0, atol=1e-8):
        raise VisualFactorCacheError(f"{name} covariance must be symmetric")


def _packet_arrays(packet: VisualFactorPacket) -> dict[str, np.ndarray]:
    arrays = {
        "frame_i": np.asarray(_require_integer(packet.frame_i, "packet frame_i"), dtype=np.int64),
        "frame_j": np.asarray(_require_integer(packet.frame_j, "packet frame_j"), dtype=np.int64),
        "timestamp_i_ns": np.asarray(_require_integer(packet.timestamp_i_ns, "packet timestamp_i_ns"), dtype=np.int64),
        "timestamp_j_ns": np.asarray(_require_integer(packet.timestamp_j_ns, "packet timestamp_j_ns"), dtype=np.int64),
        "baseline_m": np.asarray(packet.baseline_m, dtype=np.float64),
        "visual_sha256": np.asarray(packet.visual_sha256),
        "covariance_diagnostics_json": np.asarray(_serialize_covariance_diagnostics(packet.covariance_diagnostics)),
    }
    arrays.update({name: _to_numpy(getattr(packet, name)) for name in _TENSOR_NAMES})
    arrays.update({f"match__{name}": _to_numpy(packet.match_fields[name]) for name in MATCH_FIELDS})
    return arrays


def _packet_from_arrays(data: Mapping[str, np.ndarray]) -> VisualFactorPacket:
    required = {
        "frame_i", "frame_j", "timestamp_i_ns", "timestamp_j_ns", "baseline_m", "visual_sha256", "covariance_diagnostics_json",
        *_TENSOR_NAMES,
        *(f"match__{name}" for name in MATCH_FIELDS),
    }
    if set(data.keys()) != required:
        raise VisualFactorCacheError("packet field names do not match cache schema")
    return VisualFactorPacket(
        frame_i=_require_integer(data["frame_i"].item(), "packet frame_i"),
        frame_j=_require_integer(data["frame_j"].item(), "packet frame_j"),
        timestamp_i_ns=_require_integer(data["timestamp_i_ns"].item(), "packet timestamp_i_ns"),
        timestamp_j_ns=_require_integer(data["timestamp_j_ns"].item(), "packet timestamp_j_ns"),
        K=_to_tensor(data["K"]),
        baseline_m=float(data["baseline_m"].item()),
        relative_pose_init=_to_tensor(data["relative_pose_init"]),
        points_local=_to_tensor(data["points_local"]),
        points_cov_local=_to_tensor(data["points_cov_local"]),
        point_colors=_to_tensor(data["point_colors"]),
        match_fields={name: _to_tensor(data[f"match__{name}"]) for name in MATCH_FIELDS},
        covariance_diagnostics=_deserialize_covariance_diagnostics(
            str(data["covariance_diagnostics_json"].item())
        ),
        visual_sha256=str(data["visual_sha256"].item()),
    )


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return torch.as_tensor(value).detach().cpu().contiguous().numpy()


def _serialize_covariance_diagnostics(values: Mapping[str, float | int]) -> str:
    json_values = {
        name: None if isinstance(value, float) and np.isnan(value) else value
        for name, value in values.items()
    }
    return json.dumps(json_values, sort_keys=True, allow_nan=False)


def _deserialize_covariance_diagnostics(payload: str) -> dict[str, float | int]:
    values = json.loads(payload, parse_constant=_reject_nonstandard_json_constant)
    if not isinstance(values, dict):
        raise VisualFactorCacheError("covariance diagnostics payload must be a dictionary")
    return {
        name: float("nan") if value is None else value
        for name, value in values.items()
    }


def _reject_nonstandard_json_constant(value: str) -> None:
    raise VisualFactorCacheError(f"non-standard JSON constant is not allowed: {value}")


def _to_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(value, copy=True))


def _cpu_float64_calibration(value: Any) -> torch.Tensor:
    try:
        return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    except Exception as error:
        raise VisualFactorCacheError("invalid K calibration") from error


def _manifest_K(manifest: VisualFactorCacheManifest) -> torch.Tensor:
    return torch.tensor(manifest.K, dtype=torch.float64)


def _validate_manifest(manifest: VisualFactorCacheManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise VisualFactorCacheError("unsupported cache schema version")
    if not manifest.scene:
        raise VisualFactorCacheError("manifest scene is required")
    if SourceMetadata.from_mapping(manifest.source).frame_count != manifest.frame_count:
        raise VisualFactorCacheError("manifest source frame_count is inconsistent")
    if manifest.frame_count != len(manifest.timestamps_ns) or manifest.frame_count != len(manifest.pairs) + 1:
        raise VisualFactorCacheError("manifest frame_count is inconsistent")
    if any(later <= earlier for earlier, later in zip(manifest.timestamps_ns, manifest.timestamps_ns[1:])):
        raise VisualFactorCacheError("manifest timestamps must increase")
    _validate_tensor("manifest K", _manifest_K(manifest), (3, 3))
    if not np.isfinite(manifest.baseline_m) or manifest.baseline_m <= 0.0:
        raise VisualFactorCacheError("manifest baseline must be positive and finite")
    for expected_i, pair in enumerate(manifest.pairs):
        if set(pair) != _MANIFEST_PAIR_FIELDS:
            raise VisualFactorCacheError("manifest pair keys do not match cache schema")
        try:
            frame_i = _require_integer(pair["frame_i"], "pair frame_i")
            frame_j = _require_integer(pair["frame_j"], "pair frame_j")
            path = str(pair["path"])
            checksum = str(pair["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise VisualFactorCacheError("invalid manifest pair") from error
        if (frame_i, frame_j) != (expected_i, expected_i + 1):
            raise VisualFactorCacheError("manifest pair indices must be contiguous")
        if path != f"pairs/{frame_i:06d}_{frame_j:06d}.npz":
            raise VisualFactorCacheError("manifest pair path is invalid")
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise VisualFactorCacheError("manifest packet checksum is invalid")


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            np.savez_compressed(temp_file, **arrays)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise


def _validate_provenance_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise VisualFactorCacheError("source metadata must contain finite values")
        return value
    raise VisualFactorCacheError("source metadata only permits scalar provenance values")


def _require_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise VisualFactorCacheError(f"{field} must be an integer")
    return int(value)


def _normalize_manifest_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **pair,
        "frame_i": _require_integer(pair["frame_i"], "pair frame_i"),
        "frame_j": _require_integer(pair["frame_j"], "pair frame_j"),
    }


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            json.dump(data, temp_file, sort_keys=True, separators=(",", ":"))
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

"""Export a pure-MACVO tensor map into the portable visual-factor cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pypose as pp
import torch
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Utility.VisualFactorCache import (
    FRONTEND_COVARIANCE_DIAGNOSTIC_FIELDS,
    MATCH_FIELDS,
    VisualFactorPacket,
    write_visual_factor_cache,
)
from Utility.VisualInputFingerprint import visual_input_sha256


DEFAULT_VALIDATION_ATOL = 1e-3


class ExportVisualFactorCacheError(RuntimeError):
    """Raised when a result cannot be replayed as a visual-factor cache."""


def export_result_to_visual_cache(
    result_dir: str | Path,
    cache_dir: str | Path,
    scene: str,
    dataset_root: str | Path,
    *,
    validation_atol: float = DEFAULT_VALIDATION_ATOL,
) -> Path:
    """Export every contiguous pure-MACVO pair from ``result_dir`` to ``cache_dir``.

    The exported visual tensors are derived exclusively from ``tensor_map.npz``.
    Exact visual-factor diagnostics are authoritative when present. Legacy
    frame-pair diagnostics may be supplemented from filtered cached matches.
    """
    if not np.isfinite(validation_atol) or validation_atol < 0.0:
        raise ExportVisualFactorCacheError("validation_atol must be finite and non-negative")

    result_path = Path(result_dir)
    cache_path = Path(cache_dir)
    config_path = result_path / "config.yaml"
    tensor_map_path = result_path / "tensor_map.npz"
    config = _read_config(config_path)
    motion_type, keyframe_type = _require_replay_compatible_config(config)
    tensor_map = _read_tensor_map(tensor_map_path)
    timestamps = _tensor_map_array(tensor_map, "frames//time_ns", (None,))
    visual_diagnostics_path = result_path / "visual_factor_diagnostics.csv"
    authoritative_diagnostics = visual_diagnostics_path.exists()
    diagnostics_by_pair = _read_covariance_diagnostics(
        visual_diagnostics_path
        if authoritative_diagnostics
        else result_path / "frame_pair_diagnostics.csv",
        timestamps,
        authoritative=authoritative_diagnostics,
    )
    packets, frame_count, timestamps = _build_packets(
        tensor_map,
        diagnostics_by_pair,
        validation_atol,
        require_complete_diagnostics=authoritative_diagnostics,
    )

    write_visual_factor_cache(
        cache_path,
        str(scene),
        packets,
        source={
            "frame_count": frame_count,
            "dataset": str(Path(dataset_root).resolve()),
            "result": str(result_path.resolve()),
            "config": _sha256_file(config_path),
            "git": _source_git_revision(result_path),
            "exporter": "Scripts.export_visual_factor_cache",
            "motion": motion_type,
            "keyframes": keyframe_type,
            "timestamps": _sha256_array(timestamps),
            "checksums": _sha256_file(tensor_map_path),
        },
    )
    return cache_path


def _read_config(path: Path) -> Mapping[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ExportVisualFactorCacheError(f"unable to read config: {path}") from error
    if not isinstance(data, Mapping):
        raise ExportVisualFactorCacheError("config must be a mapping")
    return data


def _require_replay_compatible_config(config: Mapping[str, Any]) -> tuple[str, str]:
    odometry = config.get("Odometry", config)
    if not isinstance(odometry, Mapping):
        raise ExportVisualFactorCacheError("config Odometry section must be a mapping")
    args = odometry.get("args")
    motion = odometry.get("motion")
    keyframe = odometry.get("keyframe")
    mapping = args.get("mapping") if isinstance(args, Mapping) else None
    motion_type = motion.get("type") if isinstance(motion, Mapping) else None
    keyframe_type = keyframe.get("type") if isinstance(keyframe, Mapping) else None
    if mapping is not False:
        raise ExportVisualFactorCacheError("source config Odometry.args.mapping must be exactly False")
    if motion_type != "StaticMotionModel":
        raise ExportVisualFactorCacheError("source config motion.type must be StaticMotionModel")
    if keyframe_type != "AllKeyframe":
        raise ExportVisualFactorCacheError("source config keyframe.type must be AllKeyframe")
    return str(motion_type), str(keyframe_type)


def _read_tensor_map(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: np.array(data[name], copy=True) for name in data.files}
    except (OSError, ValueError) as error:
        raise ExportVisualFactorCacheError(f"unable to read tensor map: {path}") from error


def _read_covariance_diagnostics(
    path: Path,
    timestamps: np.ndarray,
    *,
    authoritative: bool,
) -> dict[tuple[int, int], dict[str, float | int]]:
    if not path.exists():
        return {}
    source_name = path.name
    expected_pairs = tuple((frame_i, frame_i + 1) for frame_i in range(max(len(timestamps) - 1, 0)))
    expected_pair_set = set(expected_pairs)
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required_fields = {"frame_i", "frame_j", *FRONTEND_COVARIANCE_DIAGNOSTIC_FIELDS}
            if authoritative:
                required_fields.update(("timestamp_i", "timestamp_j", "visual_input_sha256"))
            available_fields = set(reader.fieldnames or ())
            missing_fields = sorted(required_fields - available_fields)
            if missing_fields:
                raise ExportVisualFactorCacheError(
                    f"{source_name} is missing required columns: {', '.join(missing_fields)}"
                )
            diagnostics: dict[tuple[int, int], dict[str, float | int]] = {}
            for line_number, row in enumerate(reader, start=2):
                pair = (
                    _parse_csv_integer(row.get("frame_i"), "frame_i", line_number, source_name),
                    _parse_csv_integer(row.get("frame_j"), "frame_j", line_number, source_name),
                )
                if pair not in expected_pair_set:
                    raise ExportVisualFactorCacheError(
                        f"{source_name} line {line_number} has unexpected pair {pair}"
                    )
                if pair in diagnostics:
                    raise ExportVisualFactorCacheError(
                        f"{source_name} line {line_number} has duplicate diagnostics pair {pair}"
                    )
                if authoritative:
                    for field, frame_index in (("timestamp_i", pair[0]), ("timestamp_j", pair[1])):
                        timestamp = _parse_csv_integer(row.get(field), field, line_number, source_name)
                        if timestamp != timestamps[frame_index]:
                            raise ExportVisualFactorCacheError(
                                f"{field} does not match tensor map in {source_name} line {line_number}"
                            )
                diagnostics[pair] = {
                    name: _parse_diagnostic_scalar(row.get(name), name, line_number, source_name)
                    for name in FRONTEND_COVARIANCE_DIAGNOSTIC_FIELDS
                }
                if authoritative:
                    visual_hash = str(row.get("visual_input_sha256", "")).strip().lower()
                    try:
                        valid_visual_hash = len(visual_hash) == 64 and int(visual_hash, 16) >= 0
                    except ValueError:
                        valid_visual_hash = False
                    if not valid_visual_hash:
                        raise ExportVisualFactorCacheError(
                            f"invalid visual hash in {source_name} line {line_number}"
                        )
                    diagnostics[pair]["__visual_input_sha256"] = visual_hash
                    diagnostics[pair]["__source_line"] = line_number
            if authoritative:
                missing_pairs = [pair for pair in expected_pairs if pair not in diagnostics]
                if missing_pairs:
                    raise ExportVisualFactorCacheError(
                        f"{source_name} is missing required pair rows: {missing_pairs}"
                    )
            return diagnostics
    except ExportVisualFactorCacheError:
        raise
    except (OSError, csv.Error) as error:
        raise ExportVisualFactorCacheError(f"unable to read diagnostics: {path}") from error


def _parse_csv_integer(value: str | None, field: str, line_number: int, source_name: str) -> int:
    try:
        text = value.strip() if value is not None else ""
        if not text:
            raise ValueError
        return int(text)
    except ValueError as error:
        raise ExportVisualFactorCacheError(
            f"invalid {field} in {source_name} line {line_number}"
        ) from error


def _parse_diagnostic_scalar(
    value: str | None,
    field: str,
    line_number: int,
    source_name: str,
) -> float | int:
    text = value.strip() if value is not None else ""
    if not text:
        return float("nan")
    try:
        numeric = float(text)
    except ValueError as error:
        raise ExportVisualFactorCacheError(
            f"invalid {field} in {source_name} line {line_number}"
        ) from error
    if np.isinf(numeric):
        raise ExportVisualFactorCacheError(
            f"invalid {field} in {source_name} line {line_number}: infinity is not allowed"
        )
    if field == "num_selected_keypoints" and not np.isnan(numeric):
        if not numeric.is_integer():
            raise ExportVisualFactorCacheError(
                f"invalid {field} in {source_name} line {line_number}"
            )
        return int(numeric)
    return numeric


def _build_packets(
    tensor_map: Mapping[str, np.ndarray],
    diagnostics_by_pair: Mapping[tuple[int, int], dict[str, float | int]],
    validation_atol: float,
    *,
    require_complete_diagnostics: bool = False,
) -> tuple[list[VisualFactorPacket], int, np.ndarray]:
    frame_K = _tensor_map_array(tensor_map, "frames//K", (None, 3, 3))
    frame_baseline = _tensor_map_array(tensor_map, "frames//baseline", (None,))
    frame_pose = _tensor_map_array(tensor_map, "frames//pose", (None, 7))
    timestamps = _tensor_map_array(tensor_map, "frames//time_ns", (None,))
    frame_count = len(timestamps)
    if frame_count < 2:
        raise ExportVisualFactorCacheError("tensor map requires at least two frames")
    if any(len(values) != frame_count for values in (frame_K, frame_baseline, frame_pose)):
        raise ExportVisualFactorCacheError("frame tensor row counts must agree")
    if not np.issubdtype(timestamps.dtype, np.integer) or np.any(np.diff(timestamps) <= 0):
        raise ExportVisualFactorCacheError("frame timestamps must be strictly increasing integers")
    if not np.isfinite(frame_K).all() or not np.isfinite(frame_baseline).all() or not np.isfinite(frame_pose).all():
        raise ExportVisualFactorCacheError("frame tensors must be finite")
    if not np.array_equal(frame_K, np.repeat(frame_K[:1], frame_count, axis=0)):
        raise ExportVisualFactorCacheError("all frame K calibrations must match")
    if not np.array_equal(frame_baseline, np.repeat(frame_baseline[:1], frame_count, axis=0)):
        raise ExportVisualFactorCacheError("all frame baselines must match")

    match_fields = {
        name: _tensor_map_array(tensor_map, f"match//{name}", None)
        for name in MATCH_FIELDS
    }
    match_rows = len(match_fields["pixel1_uv"])
    if match_rows == 0:
        raise ExportVisualFactorCacheError("tensor map must contain matches")
    if any(len(values) != match_rows for values in match_fields.values()):
        raise ExportVisualFactorCacheError("match field row count differs from pixel1_uv")
    frame1_mapping = _integer_mapping(tensor_map, "edge/match2frame1/mapping", match_rows)
    frame2_mapping = _integer_mapping(tensor_map, "edge/match2frame2/mapping", match_rows)
    point_mapping = _integer_mapping(tensor_map, "edge/match2point/mapping", match_rows)
    point_local_positions = _tensor_map_array(tensor_map, "points//pos_Tc", (None, 3))
    point_positions = _tensor_map_array(tensor_map, "points//pos_Tw", (None, 3))
    point_covariances = _tensor_map_array(tensor_map, "points//cov_Tw", (None, 3, 3))
    point_colors = _tensor_map_array(tensor_map, "points//color", (None, 3))
    point_count = len(point_positions)
    if any(len(values) != point_count for values in (point_local_positions, point_covariances, point_colors)):
        raise ExportVisualFactorCacheError("point tensor row counts must agree")
    if np.any(frame1_mapping < 0) or np.any(frame2_mapping >= frame_count):
        raise ExportVisualFactorCacheError("match frame mapping is out of bounds")
    if np.any(frame2_mapping != frame1_mapping + 1):
        raise ExportVisualFactorCacheError("match frame mappings must be consecutive pairs")
    if np.any(point_mapping < 0) or np.any(point_mapping >= point_count):
        raise ExportVisualFactorCacheError("match point mapping is out of bounds")

    packets: list[VisualFactorPacket] = []
    for frame_i in range(frame_count - 1):
        rows = np.flatnonzero((frame1_mapping == frame_i) & (frame2_mapping == frame_i + 1))
        if len(rows) == 0:
            raise ExportVisualFactorCacheError(f"pair ({frame_i}, {frame_i + 1}) has missing matches")
        packet_fields = {name: _to_tensor(values[rows]) for name, values in match_fields.items()}
        K = _to_tensor(frame_K[frame_i]).to(dtype=torch.float64)
        points_local = _to_tensor(point_local_positions[point_mapping[rows]])
        points_cov_local = packet_fields["obs1_covTc"]
        pose_i = pp.SE3(_to_tensor(frame_pose[frame_i]).to(dtype=torch.float64))
        _validate_source_geometry(
            points_local,
            points_cov_local,
            pose_i,
            _to_tensor(point_positions[point_mapping[rows]]).to(dtype=torch.float64),
            _to_tensor(point_covariances[point_mapping[rows]]).to(dtype=torch.float64),
            validation_atol,
            frame_i,
        )
        pair = (frame_i, frame_i + 1)
        covariance_diagnostics = diagnostics_by_pair.get(pair)
        if covariance_diagnostics is None:
            if require_complete_diagnostics:
                raise ExportVisualFactorCacheError(
                    f"authoritative visual diagnostics are missing pair {pair}"
                )
            covariance_diagnostics = _derive_covariance_diagnostics(packet_fields)
        covariance_diagnostics = dict(covariance_diagnostics)
        expected_visual_hash = covariance_diagnostics.pop("__visual_input_sha256", None)
        source_line = covariance_diagnostics.pop("__source_line", None)
        packet_visual_hash = visual_input_sha256(packet_fields)
        if expected_visual_hash is not None and expected_visual_hash != packet_visual_hash:
            raise ExportVisualFactorCacheError(
                "visual hash differs from tensor map in visual_factor_diagnostics.csv "
                f"line {source_line} for pair {pair}"
            )
        packets.append(
            VisualFactorPacket(
                frame_i=frame_i,
                frame_j=frame_i + 1,
                timestamp_i_ns=int(timestamps[frame_i]),
                timestamp_j_ns=int(timestamps[frame_i + 1]),
                K=K,
                baseline_m=float(frame_baseline[frame_i]),
                relative_pose_init=torch.eye(4, dtype=torch.float64),
                points_local=points_local,
                points_cov_local=points_cov_local,
                point_colors=_to_tensor(point_colors[point_mapping[rows]]),
                match_fields=packet_fields,
                covariance_diagnostics=covariance_diagnostics,
                visual_sha256=packet_visual_hash,
            )
        )
    return packets, frame_count, timestamps


def _derive_covariance_diagnostics(match_fields: Mapping[str, torch.Tensor]) -> dict[str, float | int]:
    def stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
        array = values.detach().cpu().float().numpy().ravel()
        filtered = array[~np.isnan(array)]
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

    flow_covariance = match_fields["pixel2_uv_cov"]
    cached_depths = match_fields["pixel1_d"].detach().cpu().float().numpy().ravel()
    valid_depths = np.isfinite(cached_depths) & (cached_depths > 0)
    valid_depth_ratio = float(np.mean(valid_depths)) if len(valid_depths) else float("nan")
    diagnostics: dict[str, float | int] = {
        **stats(match_fields["pixel1_d_cov"], "kp0_depth_cov"),
        **stats(match_fields["pixel2_d_cov"], "kp1_depth_cov"),
        **stats(flow_covariance[:, 0], "flow_u_cov"),
        **stats(flow_covariance[:, 1], "flow_v_cov"),
        "valid_depth_ratio": valid_depth_ratio,
        "num_selected_keypoints": int(match_fields["pixel1_uv"].shape[0]),
    }
    return diagnostics


def _tensor_map_array(
    tensor_map: Mapping[str, np.ndarray], name: str, shape: tuple[int | None, ...] | None
) -> np.ndarray:
    try:
        value = np.asarray(tensor_map[name])
    except KeyError as error:
        raise ExportVisualFactorCacheError(f"tensor map is missing {name}") from error
    if shape is not None and (value.ndim != len(shape) or any(
        expected is not None and value.shape[index] != expected
        for index, expected in enumerate(shape)
    )):
        raise ExportVisualFactorCacheError(f"{name} has invalid shape")
    if not np.isfinite(value).all():
        raise ExportVisualFactorCacheError(f"{name} must contain finite values")
    return value


def _integer_mapping(tensor_map: Mapping[str, np.ndarray], name: str, rows: int) -> np.ndarray:
    value = _tensor_map_array(tensor_map, name, (None,))
    if len(value) != rows:
        raise ExportVisualFactorCacheError(f"{name} row count differs from match fields")
    if not np.issubdtype(value.dtype, np.integer):
        raise ExportVisualFactorCacheError(f"{name} must contain integers")
    return value.astype(np.int64, copy=False)


def _validate_source_geometry(
    points_local: torch.Tensor,
    covariance_local: torch.Tensor,
    source_pose: pp.LieTensor,
    source_world_points: torch.Tensor,
    source_world_covariances: torch.Tensor,
    atol: float,
    frame_i: int,
) -> None:
    point_pose = pp.SE3(source_pose.tensor().to(dtype=points_local.dtype))
    expected_world_points = point_pose.Act(points_local)
    rotation = source_pose.rotation().matrix().to(dtype=covariance_local.dtype)
    expected_world_covariances = rotation @ covariance_local @ rotation.transpose(-1, -2)
    if not torch.allclose(
        expected_world_points,
        source_world_points.to(dtype=expected_world_points.dtype),
        rtol=0.0,
        atol=atol,
    ):
        raise ExportVisualFactorCacheError(f"pair ({frame_i}, {frame_i + 1}) world point validation failed")
    if not torch.allclose(
        expected_world_covariances,
        source_world_covariances.to(dtype=expected_world_covariances.dtype),
        rtol=0.0,
        atol=atol,
    ):
        raise ExportVisualFactorCacheError(f"pair ({frame_i}, {frame_i + 1}) world covariance validation failed")


def _to_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(value, copy=True))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _source_git_revision(result_path: Path) -> str:
    metadata_path = result_path / "metadata.yaml"
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return _git_revision()
    if not isinstance(metadata, Mapping):
        return _git_revision()
    revision = str(metadata.get("git_version", "")).strip()
    return revision or _git_revision()


def _git_revision() -> str:
    repository = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if not status.strip():
            return head
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty_digest = hashlib.sha256(
            (status + "\0" + diff).encode("utf-8", errors="surrogateescape")
        ).hexdigest()
        return f"{head}+dirty:{dirty_digest}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("scene")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--validation-atol", type=float, default=DEFAULT_VALIDATION_ATOL)
    args = parser.parse_args(argv)
    try:
        cache_path = export_result_to_visual_cache(
            args.result_dir,
            args.cache_dir,
            args.scene,
            args.dataset_root,
            validation_atol=args.validation_atol,
        )
    except ExportVisualFactorCacheError as error:
        parser.error(str(error))
    print(cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

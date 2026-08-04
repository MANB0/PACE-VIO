#!/usr/bin/env python3
"""Record a complete pure-MACVO visual cache from a finished live run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.build_compressed_uvd_pose_factor_cache import (
    build_cache as build_compressed_cache,
)
from Scripts.build_relative_pose_factor_cache import build_cache as build_relative_cache
from Scripts.export_visual_factor_cache import export_result_to_visual_cache
from Utility.CompressedUVDFactorCache import (
    COMPRESSED_UVD_FACTOR_FILENAME,
    CompressedUVDFactorCacheReader,
)
from Utility.RelativePoseFactorCache import (
    RELATIVE_POSE_FACTOR_FILENAME,
    RelativePoseFactorCacheReader,
)
from Utility.VisualFactorCache import MATCH_FIELDS, VisualFactorCacheReader


CACHE_READY_FILENAME = "cache_ready.json"
PURE_MACVO_TRAJECTORIES = {
    "macvo_raw_poses_camera.csv": "pure_macvo_poses_camera.csv",
    "macvo_raw_poses_imu.csv": "pure_macvo_poses_imu.csv",
}
FRAME_FIELDS = ("frames//K", "frames//baseline", "frames//time_ns")
EDGE_FIELDS = (
    "edge/match2frame1/mapping",
    "edge/match2frame2/mapping",
    "edge/match2point/mapping",
)
POINT_FIELDS = ("points//pos_Tc", "points//color")


def _resolve_result_dir(path: Path) -> Path:
    if (path / "tensor_map.npz").is_file():
        return path
    candidates = sorted(path.rglob("tensor_map.npz")) if path.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one completed MACVO result below {path}, found {len(candidates)}"
        )
    return candidates[0].parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_archive(result: Path) -> dict[str, np.ndarray]:
    tensor_map = result / "tensor_map.npz"
    try:
        with np.load(tensor_map, allow_pickle=False) as stream:
            return {name: np.array(stream[name], copy=True) for name in stream.files}
    except (OSError, ValueError) as error:
        raise RuntimeError(f"unable to read tensor map: {tensor_map}") from error


def _raw_camera_poses(result: Path, timestamps: np.ndarray) -> np.ndarray:
    pose_path = result / "macvo_raw_poses_camera.csv"
    try:
        table = pd.read_csv(pose_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"unable to read pure-MACVO pose track: {pose_path}") from error
    required = ("timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw")
    if any(name not in table.columns for name in required):
        raise RuntimeError(f"pure-MACVO pose table is malformed: {pose_path}")
    raw_time = table["timestamp_ns"].to_numpy(np.int64)
    if not np.array_equal(raw_time, timestamps):
        raise RuntimeError("pure-MACVO timestamps differ from tensor_map timestamps")
    poses = table[list(required[1:])].to_numpy(np.float64)
    norms = np.linalg.norm(poses[:, 3:7], axis=1)
    if np.any(~np.isfinite(poses)) or np.any(norms < 1.0e-12):
        raise RuntimeError("pure-MACVO pose table contains invalid poses")
    poses[:, 3:7] /= norms[:, None]
    return poses


def _rebuild_world_geometry(
    archive: dict[str, np.ndarray], poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    point_local = np.asarray(archive["points//pos_Tc"], dtype=np.float64)
    point_count = point_local.shape[0]
    point_mapping = np.asarray(archive["edge/match2point/mapping"], dtype=np.int64)
    frame_mapping = np.asarray(archive["edge/match2frame1/mapping"], dtype=np.int64)
    match_covariance = np.asarray(archive["match//obs1_covTc"], dtype=np.float64)
    if point_mapping.shape != frame_mapping.shape or len(point_mapping) != len(match_covariance):
        raise RuntimeError("match-to-point mappings are inconsistent")

    point_frame = np.full(point_count, -1, dtype=np.int64)
    point_covariance_local = np.zeros((point_count, 3, 3), dtype=np.float64)
    for match_index, (point_index, frame_index) in enumerate(zip(point_mapping, frame_mapping)):
        previous = point_frame[point_index]
        if previous >= 0 and previous != frame_index:
            raise RuntimeError(f"point {point_index} is attached to multiple source frames")
        point_frame[point_index] = frame_index
        point_covariance_local[point_index] = match_covariance[match_index]
    if np.any(point_frame < 0):
        raise RuntimeError("tensor map contains points without a source-frame match")

    rotations = Rotation.from_quat(poses[:, 3:7]).as_matrix()
    point_rotation = rotations[point_frame]
    point_world = (
        np.einsum("nij,nj->ni", point_rotation, point_local)
        + poses[point_frame, :3]
    )
    covariance_world = np.einsum(
        "nij,njk,nlk->nil",
        point_rotation,
        point_covariance_local,
        point_rotation,
    )
    return point_world, covariance_world


def _write_pure_macvo_source(
    result: Path,
    destination: Path,
    *,
    scene: str,
    dataset: Path,
) -> None:
    archive = _load_archive(result)
    timestamps = np.asarray(archive["frames//time_ns"], dtype=np.int64)
    poses = _raw_camera_poses(result, timestamps)
    point_world, covariance_world = _rebuild_world_geometry(archive, poses)

    destination.mkdir(parents=True, exist_ok=False)
    exported: dict[str, np.ndarray] = {
        name: archive[name] for name in FRAME_FIELDS + EDGE_FIELDS + POINT_FIELDS
    }
    exported["frames//pose"] = poses
    exported["points//pos_Tw"] = point_world
    exported["points//cov_Tw"] = covariance_world
    for name in MATCH_FIELDS:
        exported[f"match//{name}"] = archive[f"match//{name}"]
    np.savez_compressed(destination / "tensor_map.npz", **exported)

    for name in ("config.yaml", "visual_factor_diagnostics.csv"):
        source = result / name
        if not source.is_file():
            raise RuntimeError(f"cache recording requires {source}")
        shutil.copy2(source, destination / name)
    metadata = result / "metadata.yaml"
    if metadata.is_file():
        shutil.copy2(metadata, destination / metadata.name)
    for source_name in PURE_MACVO_TRAJECTORIES:
        source = result / source_name
        if not source.is_file():
            raise RuntimeError(f"cache recording requires Pure MACVO trajectory {source}")
        shutil.copy2(source, destination / source_name)
    _atomic_write_json(
        destination / "provenance.json",
        {
            "schema_version": 1,
            "scene": scene,
            "dataset": str(dataset.resolve()),
            "source_live_result": str(result.resolve()),
            "pose_source": "macvo_raw_poses_camera.csv",
            "reference_point": "left camera center",
            "local_observations_modified": False,
        },
    )


def validate_visual_cache_bundle(cache_dir: str | Path, visual_factor: str | None = None) -> Path:
    cache = Path(cache_dir).expanduser().resolve()
    reader = VisualFactorCacheReader(cache)
    factor = None if visual_factor is None else str(visual_factor).strip().lower()
    if factor not in {None, "pose", "uvd", "pace"}:
        raise ValueError(f"unsupported visual factor: {visual_factor!r}")
    if factor in {None, "pose"}:
        RelativePoseFactorCacheReader(cache)
    if factor in {None, "pace"}:
        CompressedUVDFactorCacheReader(cache)
    for filename in PURE_MACVO_TRAJECTORIES.values():
        trajectory = cache / filename
        if not trajectory.is_file():
            raise RuntimeError(f"visual cache is missing Pure MACVO trajectory: {trajectory}")

    ready_path = cache / CACHE_READY_FILENAME
    if ready_path.is_file():
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"unable to read cache readiness marker: {ready_path}") from error
        if int(ready.get("schema_version", -1)) != 1:
            raise RuntimeError("unsupported cache readiness schema")
        if str(ready.get("scene", "")) != reader.manifest.scene:
            raise RuntimeError("cache readiness scene differs from manifest")
        for name, expected in ready.get("sha256", {}).items():
            path = cache / str(name)
            if not path.is_file() or _sha256(path) != str(expected):
                raise RuntimeError(f"cache readiness checksum differs: {name}")
    return cache


def record_visual_factor_cache(
    result_dir: str | Path,
    cache_dir: str | Path,
    scene: str,
    dataset_root: str | Path,
) -> Path:
    result_root = Path(result_dir).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    if cache.exists():
        raise FileExistsError(
            f"visual cache target already exists: {cache}; choose a new path"
        )
    if not result_root.is_dir() or not dataset.is_dir():
        raise FileNotFoundError("result and dataset directories must exist")
    result = _resolve_result_dir(result_root)

    try:
        cache.mkdir(parents=True, exist_ok=False)
        source = cache / "source"
        _write_pure_macvo_source(
            result,
            source,
            scene=str(scene),
            dataset=dataset,
        )
        export_result_to_visual_cache(source, cache, str(scene), dataset)
        for source_name, destination_name in PURE_MACVO_TRAJECTORIES.items():
            shutil.copy2(source / source_name, cache / destination_name)
        relative = build_relative_cache(cache, force=True, huber_delta=3.0)
        compressed = build_compressed_cache(cache, force=True, huber_delta=0.1)
        reader = VisualFactorCacheReader(cache)
        files = [
            cache / "manifest.json",
            relative,
            compressed,
            *(cache / name for name in PURE_MACVO_TRAJECTORIES.values()),
        ]
        _atomic_write_json(
            cache / CACHE_READY_FILENAME,
            {
                "schema_version": 1,
                "scene": reader.manifest.scene,
                "dataset": str(dataset),
                "frame_count": reader.manifest.frame_count,
                "pair_count": len(reader.manifest.pairs),
                "source_reference": "pure MACVO left-camera center",
                "pure_macvo_trajectories": {
                    "camera_center": PURE_MACVO_TRAJECTORIES["macvo_raw_poses_camera.csv"],
                    "imu_center": PURE_MACVO_TRAJECTORIES["macvo_raw_poses_imu.csv"],
                },
                "sha256": {path.name: _sha256(path) for path in files},
            },
        )
        validate_visual_cache_bundle(cache)
    except Exception:
        (cache / CACHE_READY_FILENAME).unlink(missing_ok=True)
        raise
    return cache


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("scene")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args(argv)
    cache = record_visual_factor_cache(
        args.result_dir,
        args.cache_dir,
        args.scene,
        args.dataset_root,
    )
    print(f"Visual cache ready: {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

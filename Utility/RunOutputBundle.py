"""Resolve MACVO run output files as a single directory bundle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOutputBundle:
    bundle_dir: Path
    poses_path: Path
    pose_frame_path: Path
    diagnostics_path: Path


def _pose_bundle_dirs(result_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    if result_dir.joinpath("poses.csv").exists() and result_dir.joinpath("pose_coordinate_frame.txt").exists():
        dirs.append(result_dir)
    dirs.extend(
        p.parent for p in sorted(result_dir.rglob("poses.csv"))
        if p.parent != result_dir and p.parent.joinpath("pose_coordinate_frame.txt").exists()
    )

    seen: set[Path] = set()
    unique_dirs: list[Path] = []
    for directory in dirs:
        if directory in seen:
            continue
        seen.add(directory)
        unique_dirs.append(directory)
    return unique_dirs


def _bundle_for_dir(directory: Path) -> RunOutputBundle:
    return RunOutputBundle(
        bundle_dir=directory,
        poses_path=directory / "poses.csv",
        pose_frame_path=directory / "pose_coordinate_frame.txt",
        diagnostics_path=directory / "frame_pair_diagnostics.csv",
    )


def find_output_bundle(
    result_dir: Path,
    *,
    require_same_dir_diagnostics: bool = False,
    diagnostics_validator: Callable[[Path], bool] | None = None,
) -> RunOutputBundle:
    pose_dirs = _pose_bundle_dirs(result_dir)
    if require_same_dir_diagnostics:
        for directory in pose_dirs:
            diagnostics_path = directory.joinpath("frame_pair_diagnostics.csv")
            if diagnostics_path.exists() and (diagnostics_validator is None or diagnostics_validator(diagnostics_path)):
                return _bundle_for_dir(directory)
    if pose_dirs:
        return _bundle_for_dir(pose_dirs[0])
    return _bundle_for_dir(result_dir)

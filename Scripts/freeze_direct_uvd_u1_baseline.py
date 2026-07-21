#!/usr/bin/env python3
"""Freeze the current Direct-UVD U1 production baseline before factor research."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Baselines/direct_uvd_u1_standard_20260719"
CONFIG_SOURCE = (
    ROOT
    / "analysis_u1_counterfactual_branches_20260719"
    / "effective_u1_odometry.yaml"
)
EVIDENCE_ROOT = ROOT / "analysis_u1_counterfactual_branches_20260719"

SOURCE_FILES = (
    "Utility/TwoStateVIO.py",
    "Utility/IMUKinematics.py",
    "Module/IMUPreintegration.py",
    "Module/Optimization/TwoFramePGO/Graphs.py",
    "Module/Optimization/TwoFramePGO/Optimizer.py",
    "Scripts/run_direct_uvd_short_experiments.py",
    "Scripts/run_u1_counterfactual_branches.py",
    "Scripts/UnitTest/test_two_state_uvd_factor.py",
)

EVIDENCE_FILES = (
    "capture_manifest.json",
    "truth_contract.json",
    "captured_u1_problems.pt",
    "immediate_counterfactual_per_edge.csv",
    "lookahead_counterfactual_per_seed.csv",
    "u1_counterfactual_summary.json",
    "u1_counterfactual_report_cn.md",
    "adaptive_mode_decision_analysis.json",
    "u1_adaptive_mode_decision_cn.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def copy_relative(source: Path, destination_root: Path, relative: Path) -> Path:
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"baseline already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copied: list[Path] = []
    copied.append(
        copy_relative(CONFIG_SOURCE, output, Path("configs/odometry.yaml"))
    )
    for relative_text in SOURCE_FILES:
        relative = Path(relative_text)
        copied.append(
            copy_relative(ROOT / relative, output / "source_snapshot", relative)
        )
    for name in EVIDENCE_FILES:
        copied.append(
            copy_relative(EVIDENCE_ROOT / name, output, Path("evidence") / name)
        )

    replay_command = (
        "cd /home/admin1/macvo-dev\n"
        "/home/admin1/miniconda3/envs/macvo/bin/python "
        "Scripts/run_u1_counterfactual_branches.py --skip-capture "
        "--max-edges 300 --lookahead 5 --lookahead-stride 10 "
        "--top-seeds-per-mode 5 --workers 5 --immediate-batch-edges 30\n"
    )
    command_path = output / "commands/replay_300_edges.sh"
    command_path.parent.mkdir(parents=True)
    command_path.write_text(replay_command, encoding="utf-8")
    copied.append(command_path)

    manifest = {
        "schema_version": 1,
        "name": "Direct UVD U1 Standard production baseline",
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(git_value("status", "--porcelain")),
        "visual_factor": "direct_uvd_full",
        "warm_start": "macvo_pose",
        "imu_preintegration": "standard_local_frame_preintegration",
        "sampling_covariance": "current_independent_step",
        "active_start_frame": 90,
        "captured_edges": 300,
        "full_replay_max_state_boxminus_norm": 5.0887187096214415e-20,
        "estimate_reference_point": "IMU center",
        "files": {},
    }
    for path in copied:
        relative = path.relative_to(output).as_posix()
        manifest["files"][relative] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest_path = output / "baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checksum_lines = [
        f"{details['sha256']}  {relative}"
        for relative, details in sorted(manifest["files"].items())
    ]
    checksum_lines.append(f"{sha256(manifest_path)}  baseline_manifest.json")
    (output / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    readme = """# Direct UVD U1 Standard baseline

This immutable snapshot freezes the production baseline before UVD Schur-marginal factor research.

The active production contract is:

- full point-level Direct UVD visual factor;
- MACVO-pose warm start;
- standard local-frame IMU preintegration;
- current-independent-step IMU covariance;
- two-state fixed-lag solver with the incoming 15D Schur prior;
- three-second static IMU initialization;
- pose evaluation at the IMU center.

The `source_snapshot` directory is for audit and recovery only. Development must continue in the normal source tree. The heuristic `rotation_only` and `translation_only` branches present in the snapshot are diagnostic modes; `full` is the frozen production default.

The evidence packet contains 300 exact incoming production problems. Replaying the FULL branch differs from the captured production output by at most `5.09e-20` in 15D state boxminus norm.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

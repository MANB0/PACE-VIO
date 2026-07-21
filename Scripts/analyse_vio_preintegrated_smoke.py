#!/usr/bin/env python3
"""Validate short VIO preintegrated smoke-run outputs.

This script is intended for the manual smoke runs launched from
``Scripts/run_vio_imu_prior_mode_grid.py``. It does not start MACVO; it only
checks already-produced ``poses.csv`` files against each scene's ``ref_pose``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORKDIR = Path(__file__).resolve().parents[1]
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

DEFAULT_RESULT_ROOT = Path("Results/vio_preintegrated_full_smoke_30f")
DEFAULT_OUTDIR = Path("analysis_vio_preintegrated_full_smoke_30f")

from Utility.FramePairDiagnostics import CSV_HEADER
from Utility.RunOutputBundle import find_output_bundle
from Utility.VIOConventionDiagnostics import (
    FULL_VIO_CONVENTION_FIELDS as VIO_CONVENTION_FIELDS,
    is_active_full_vio_row,
    validate_full_vio_convention,
)


@dataclass
class SmokeRecord:
    trial: str
    scene: str
    variant: str
    imu_factor_mode: str
    autodiff: str
    seq_to: str
    force_mode: str
    diagnostics_status: str
    diagnostics_rows: int
    vio_factor_active_rows: int
    imu_time_offset_ns: str
    imu_time_offset_source: str
    status: str
    frames: int
    direct_origin_ate: float
    direct_raw_ate: float
    end_error: float
    pose_frame: str
    scene_root: Path
    result_dir: Path
    poses_path: Path
    gt_path: Path
    diagnostics_path: Path
    message: str = ""


def read_manifest(result_root: Path) -> list[dict[str, str]]:
    manifest = result_root / "run_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest contains no runs: {manifest}")
    return rows


def _resolve_path(raw: str | None, *, result_root: Path) -> Path:
    if not raw:
        return Path("")
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [
        (WORKDIR / path),
        (result_root / path),
        path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return WORKDIR / path


def load_xyz(path: Path, *, kind: str) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return np.empty((0, 3), dtype=float)

    names = set(rows[0].keys())
    if {"tx", "ty", "tz"}.issubset(names):
        cols = ("tx", "ty", "tz")
    elif {"x", "y", "z"}.issubset(names):
        cols = ("x", "y", "z")
    else:
        raise ValueError(f"{kind} file lacks xyz/tx ty tz columns: {path}")

    xyz = np.array([[float(row[c]) for c in cols] for row in rows], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid xyz array from {path}: shape={xyz.shape}")
    return xyz


def read_pose_frame(frame_path: Path) -> str:
    if not frame_path.exists():
        return "NWU"
    frame = frame_path.read_text(encoding="utf-8").strip().upper()
    return frame or "NWU"


def convert_xyz_to_nwu(xyz: np.ndarray, source_frame: str) -> np.ndarray:
    frame = source_frame.strip().upper()
    converted = np.asarray(xyz, dtype=float).copy()
    if frame == "NWU":
        return converted
    if frame == "NED":
        converted[:, 1] *= -1.0
        converted[:, 2] *= -1.0
        return converted
    raise ValueError(f"Unsupported pose coordinate frame={source_frame!r}; expected NWU or NED")


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _requires_active_vio(row: dict[str, str]) -> bool:
    if str(row.get("imu_factor_mode", "")).strip().lower() != "preintegrated_vio":
        return False
    force_mode = str(row.get("force_mode", "")).strip().lower()
    variant = str(row.get("variant", "")).strip().lower()
    return force_mode == "full_imu" or "vio_preintegrated_full" in variant


def _read_diagnostic_rows(path: Path) -> tuple[list[dict[str, str]], int, set[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.reader(f))
    if not raw_rows:
        return [], 0, set()

    header = raw_rows[0]
    data_rows = raw_rows[1:]
    fieldnames = set(header)
    required = {"vio_factor_active", "imu_factor_mode", "imu_residual_rows"}
    if required.issubset(fieldnames):
        return [dict(zip(header, row)) for row in data_rows], len(data_rows), fieldnames

    # Compatibility for result directories created before the VIO diagnostic
    # columns were introduced. A new run may append wider rows under the old
    # header; those rows can still be decoded by the current canonical header.
    canonical_rows = [
        dict(zip(CSV_HEADER, row))
        for row in data_rows
        if len(row) == len(CSV_HEADER)
    ]
    if canonical_rows:
        return canonical_rows, len(data_rows), set(CSV_HEADER)

    return [dict(zip(header, row)) for row in data_rows], len(data_rows), fieldnames


def summarize_vio_diagnostics(path: Path, *, require_convention: bool = False) -> tuple[str, int, int, str]:
    if not path.exists():
        return "missing", 0, 0, f"missing {path}"
    rows, total_data_rows, fieldnames = _read_diagnostic_rows(path)
    if not rows:
        return "empty", 0, 0, f"empty {path}"

    required = {"vio_factor_active", "imu_factor_mode", "imu_residual_rows"}
    missing = sorted(required - fieldnames)
    if missing:
        return "missing_columns", len(rows), 0, f"missing diagnostic columns: {', '.join(missing)}"
    if require_convention:
        missing_convention = sorted(VIO_CONVENTION_FIELDS - fieldnames)
        if missing_convention:
            return (
                "missing_convention_columns",
                len(rows),
                0,
                f"missing convention diagnostic columns: {', '.join(missing_convention)}",
            )

    active_rows = 0
    convention_errors: list[str] = []
    for row in rows:
        try:
            residual_rows = int(float(row.get("imu_residual_rows", "0") or "0"))
        except ValueError:
            residual_rows = 0
        mode = str(row.get("imu_factor_mode", "")).strip().lower()
        if _truthy(row.get("vio_factor_active")) and mode == "preintegrated_vio" and residual_rows >= 3:
            if require_convention:
                valid_convention, convention_message = validate_full_vio_convention(row)
                if not valid_convention:
                    convention_errors.append(convention_message)
                    continue
            active_rows += 1
    if require_convention and convention_errors and active_rows <= 0:
        return "invalid_convention", total_data_rows, 0, "; ".join(convention_errors[:3])
    return "ok", total_data_rows, active_rows, ""


def _diagnostics_file_has_active_full_vio(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    return any(is_active_full_vio_row(row) for row in rows)


def _active_full_vio_time_sync(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return "", ""
    for row in rows:
        if is_active_full_vio_row(row):
            return str(row.get("imu_time_offset_ns", "")), str(row.get("imu_time_offset_source", ""))
    return "", ""


def rmse(errors: np.ndarray) -> float:
    if errors.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def evaluate_run(row: dict[str, str], result_root: Path) -> SmokeRecord:
    scene_root = _resolve_path(row.get("scene_root"), result_root=result_root)
    result_dir = _resolve_path(row.get("result_dir"), result_root=result_root)
    require_active_vio = _requires_active_vio(row)
    output_bundle = find_output_bundle(
        result_dir,
        require_same_dir_diagnostics=require_active_vio,
        diagnostics_validator=_diagnostics_file_has_active_full_vio if require_active_vio else None,
    )
    poses_path = output_bundle.poses_path
    pose_frame_path = output_bundle.pose_frame_path
    diagnostics_path = output_bundle.diagnostics_path
    gt_path = scene_root / "ref_pose.csv"

    base = dict(
        trial=row.get("trial", ""),
        scene=row.get("scene", ""),
        variant=row.get("variant", ""),
        imu_factor_mode=row.get("imu_factor_mode", ""),
        autodiff=row.get("autodiff", ""),
        seq_to=row.get("seq_to", ""),
        force_mode=row.get("force_mode", ""),
        diagnostics_status="not_evaluated",
        diagnostics_rows=0,
        vio_factor_active_rows=0,
        imu_time_offset_ns="",
        imu_time_offset_source="",
        frames=0,
        direct_origin_ate=math.nan,
        direct_raw_ate=math.nan,
        end_error=math.nan,
        pose_frame="",
        scene_root=scene_root,
        result_dir=result_dir,
        poses_path=poses_path,
        gt_path=gt_path,
        diagnostics_path=diagnostics_path,
    )

    if not gt_path.exists():
        return SmokeRecord(status="missing_gt", message=f"missing {gt_path}", **base)
    if not poses_path.exists():
        return SmokeRecord(status="missing_poses", message=f"missing {poses_path}", **base)
    if not pose_frame_path.exists():
        return SmokeRecord(status="missing_pose_frame", message=f"missing {pose_frame_path}", **base)

    try:
        est = load_xyz(poses_path, kind="poses")
        gt = load_xyz(gt_path, kind="gt")
        pose_frame = read_pose_frame(pose_frame_path)
        est = convert_xyz_to_nwu(est, pose_frame)
    except Exception as exc:
        return SmokeRecord(status="read_error", message=str(exc), **base)

    n = min(len(est), len(gt))
    if n <= 0:
        return SmokeRecord(status="empty", message="no overlapping trajectory rows", **base)

    est = est[:n]
    gt = gt[:n]
    direct_raw = rmse(est - gt)
    direct_origin = rmse((est - est[0]) - (gt - gt[0]))
    end_error = float(np.linalg.norm((est[-1] - est[0]) - (gt[-1] - gt[0])))
    diagnostics_status, diagnostics_rows, vio_factor_active_rows, diagnostics_message = summarize_vio_diagnostics(
        diagnostics_path,
        require_convention=require_active_vio,
    )
    imu_time_offset_ns, imu_time_offset_source = (
        _active_full_vio_time_sync(diagnostics_path) if require_active_vio and diagnostics_status == "ok" else ("", "")
    )
    if require_active_vio and diagnostics_status != "ok":
        status = (
            "missing_vio_convention_diagnostics"
            if diagnostics_status in {"missing_convention_columns", "invalid_convention"}
            else "missing_vio_diagnostics"
        )
        return SmokeRecord(
            status=status,
            message=diagnostics_message,
            frames=n,
            direct_origin_ate=direct_origin,
            direct_raw_ate=direct_raw,
            end_error=end_error,
            pose_frame=pose_frame,
            diagnostics_status=diagnostics_status,
            diagnostics_rows=diagnostics_rows,
            vio_factor_active_rows=vio_factor_active_rows,
            imu_time_offset_ns="",
            imu_time_offset_source="",
            **{k: v for k, v in base.items() if k not in {
                "status", "message", "frames", "direct_origin_ate", "direct_raw_ate",
                "end_error", "pose_frame", "diagnostics_status", "diagnostics_rows",
                "vio_factor_active_rows", "imu_time_offset_ns", "imu_time_offset_source",
            }},
        )
    if require_active_vio and vio_factor_active_rows <= 0:
        return SmokeRecord(
            status="vio_factor_inactive",
            message="no active preintegrated VIO factor rows in frame_pair_diagnostics.csv",
            frames=n,
            direct_origin_ate=direct_origin,
            direct_raw_ate=direct_raw,
            end_error=end_error,
            pose_frame=pose_frame,
            diagnostics_status=diagnostics_status,
            diagnostics_rows=diagnostics_rows,
            vio_factor_active_rows=vio_factor_active_rows,
            imu_time_offset_ns="",
            imu_time_offset_source="",
            **{k: v for k, v in base.items() if k not in {
                "status", "message", "frames", "direct_origin_ate", "direct_raw_ate",
                "end_error", "pose_frame", "diagnostics_status", "diagnostics_rows",
                "vio_factor_active_rows", "imu_time_offset_ns", "imu_time_offset_source",
            }},
        )

    if not require_active_vio and diagnostics_status == "missing":
        diagnostics_status = "not_required"
        diagnostics_message = ""

    return SmokeRecord(
        status="complete",
        message=diagnostics_message,
        frames=n,
        direct_origin_ate=direct_origin,
        direct_raw_ate=direct_raw,
        end_error=end_error,
        pose_frame=pose_frame,
        scene_root=scene_root,
        result_dir=result_dir,
        trial=row.get("trial", ""),
        scene=row.get("scene", ""),
        variant=row.get("variant", ""),
        imu_factor_mode=row.get("imu_factor_mode", ""),
        autodiff=row.get("autodiff", ""),
        seq_to=row.get("seq_to", ""),
        force_mode=row.get("force_mode", ""),
        diagnostics_status=diagnostics_status,
        diagnostics_rows=diagnostics_rows,
        vio_factor_active_rows=vio_factor_active_rows,
        imu_time_offset_ns=imu_time_offset_ns,
        imu_time_offset_source=imu_time_offset_source,
        poses_path=poses_path,
        gt_path=gt_path,
        diagnostics_path=diagnostics_path,
    )


def evaluate_result_root(result_root: Path) -> list[SmokeRecord]:
    result_root = result_root.resolve()
    return [evaluate_run(row, result_root) for row in read_manifest(result_root)]


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def write_outputs(records: list[SmokeRecord], outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "smoke_summary.csv"
    md_path = outdir / "smoke_summary.md"

    fieldnames = [
        "trial",
        "scene",
        "variant",
        "imu_factor_mode",
        "autodiff",
        "seq_to",
        "force_mode",
        "diagnostics_status",
        "diagnostics_rows",
        "vio_factor_active_rows",
        "imu_time_offset_ns",
        "imu_time_offset_source",
        "status",
        "frames",
        "direct_origin_ate",
        "direct_raw_ate",
        "end_error",
        "pose_frame",
        "scene_root",
        "result_dir",
        "poses_path",
        "gt_path",
        "diagnostics_path",
        "message",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "trial": record.trial,
                    "scene": record.scene,
                    "variant": record.variant,
                    "imu_factor_mode": record.imu_factor_mode,
                    "autodiff": record.autodiff,
                    "seq_to": record.seq_to,
                    "force_mode": record.force_mode,
                    "diagnostics_status": record.diagnostics_status,
                    "diagnostics_rows": record.diagnostics_rows,
                    "vio_factor_active_rows": record.vio_factor_active_rows,
                    "imu_time_offset_ns": record.imu_time_offset_ns,
                    "imu_time_offset_source": record.imu_time_offset_source,
                    "status": record.status,
                    "frames": record.frames,
                    "direct_origin_ate": _fmt(record.direct_origin_ate),
                    "direct_raw_ate": _fmt(record.direct_raw_ate),
                    "end_error": _fmt(record.end_error),
                    "pose_frame": record.pose_frame,
                    "scene_root": str(record.scene_root),
                    "result_dir": str(record.result_dir),
                    "poses_path": str(record.poses_path),
                    "gt_path": str(record.gt_path),
                    "diagnostics_path": str(record.diagnostics_path),
                    "message": record.message,
                }
            )

    complete = sum(1 for record in records if record.status == "complete")
    lines = [
        "# VIO Preintegrated Smoke Summary",
        "",
        f"- Runs in manifest: {len(records)}",
        f"- Complete runs: {complete}",
        "",
        "| trial | scene | variant | imu factor | force | autodiff | seq-to | diag | vio-active | imu offset ns | imu offset source | status | frames | pose frame | direct-origin ATE | raw ATE | end error | message |",
        "|---:|---|---|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                _md_cell(value) for value in [
                    record.trial,
                    record.scene,
                    record.variant,
                    record.imu_factor_mode,
                    record.force_mode,
                    record.autodiff,
                    record.seq_to,
                    record.diagnostics_status,
                    str(record.vio_factor_active_rows),
                    record.imu_time_offset_ns,
                    record.imu_time_offset_source,
                    record.status,
                    str(record.frames),
                    record.pose_frame,
                    _fmt(record.direct_origin_ate),
                    _fmt(record.direct_raw_ate),
                    _fmt(record.end_error),
                    record.message,
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT, help="Smoke result root to analyse.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory for summary CSV and Markdown outputs.")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    args = parser.parse_args()

    records = evaluate_result_root(args.result_root)
    csv_path, md_path = write_outputs(records, args.outdir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    incomplete = [record for record in records if record.status != "complete"]
    if args.fail_on_incomplete and incomplete:
        print(f"Incomplete runs: {len(incomplete)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export full-sequence paper metrics from a completed PACE-VIO run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.PaperEvaluation import export_paper_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alignment-json", type=Path)
    parser.add_argument("--motion-reference-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = export_paper_evaluation(
        project_root=ROOT,
        dataset_root=args.dataset,
        result_root=args.result_root,
        evaluation_dir=args.output,
        alignment_path=args.alignment_json,
        motion_reference_path=args.motion_reference_csv,
    )
    summary = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    print(f"Paper evaluation: {output}")
    print(json.dumps(summary["trajectories"].get("final", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

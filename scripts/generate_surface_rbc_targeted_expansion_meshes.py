#!/usr/bin/env python
"""Generate watertight STL meshes for the surface RBC targeted top-up plan.

This wrapper keeps the older RBC pilot mesh generator unchanged while allowing
24-row calibration batches to produce per-sample mesh metrics. Calibration
failures must be classified by the top-up orchestrator, not hidden by a legacy
30-sample pilot threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_true_3d_rbc_pilot_watertight_meshes as mesh_stage  # noqa: E402


DEFAULT_PLAN = ROOT / "results/metrics/surface_rbc_targeted_expansion_plan.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_rbc_targeted_expansion_mesh_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/surface_rbc_targeted_expansion_mesh_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate surface RBC targeted expansion STL meshes.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--stage-label", default="surface RBC targeted expansion top-up")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--min-success", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    mesh_stage.MIN_SUCCESS = max(1, min(args.min_success, args.max_samples))
    mesh_stage.FULL_SUCCESS = args.max_samples
    return mesh_stage.run(args)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

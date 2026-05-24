#!/usr/bin/env python
"""Generate watertight STL meshes for the 20.74 v2_120 top-up plan."""

from __future__ import annotations

import argparse
from pathlib import Path

import generate_true_3d_rbc_pilot_watertight_meshes as batch


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PLAN = ROOT / "results/metrics/true_3d_rbc_dataset_120_topup_plan.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_120_topup_mesh_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_rbc_dataset_120_topup_mesh_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 20.74 v2_120 top-up watertight meshes.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--stage-label", default="20.74 true 3D RBC v2_120 top-up")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    batch.DEFAULT_PLAN = DEFAULT_PLAN
    batch.DEFAULT_SUMMARY = DEFAULT_SUMMARY
    batch.DEFAULT_METRICS = DEFAULT_METRICS
    batch.MIN_SUCCESS = 64
    batch.FULL_SUCCESS = 72
    return batch.run(args)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

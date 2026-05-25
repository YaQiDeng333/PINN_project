"""Generate watertight meshes for the 20.76 v3_240 top-up plan."""

from __future__ import annotations

import argparse
from pathlib import Path

import generate_true_3d_rbc_pilot_watertight_meshes as batch_mesh


DEFAULT_PLAN = Path("results/metrics/true_3d_rbc_dataset_240_topup_plan.csv")
DEFAULT_SUMMARY = Path("results/summaries/true_3d_rbc_dataset_240_topup_mesh_summary.txt")
DEFAULT_METRICS = Path("results/metrics/true_3d_rbc_dataset_240_topup_mesh_metrics.csv")
DEFAULT_STL_DIR = Path("data/comsol_mfl/generated/temp_true_3d_rbc_dataset_240_topup_meshes")
MIN_PASS = 128


def run(args: argparse.Namespace) -> None:
    ns = argparse.Namespace(
        plan_csv=args.plan_csv,
        summary=args.summary,
        metrics=args.metrics_csv,
        stage_label="20.76 true 3D RBC v3_240 top-up",
        max_samples=160,
        overwrite=args.overwrite,
    )
    batch_mesh.run(ns)

    rows = batch_mesh.read_csv(args.metrics_csv)
    pass_count = sum(str(row.get("mesh_validation_pass", "")).lower() == "true" for row in rows)
    if pass_count < MIN_PASS:
        raise SystemExit(f"top-up mesh pass count {pass_count} below required {MIN_PASS}")
    print(f"top-up mesh pass count={pass_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--stl-dir", type=Path, default=DEFAULT_STL_DIR)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

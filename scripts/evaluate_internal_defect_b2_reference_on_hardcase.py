#!/usr/bin/env python
"""Replay the fixed 21.9 B2 artifact on the 22.3 v3_hardcase dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from internal_defect_hardcase_utils import (
    B2_MANIFEST,
    DATASET_ID,
    METRIC_FIELDS,
    PREDICTION_FIELDS,
    TAIL_FIELDS,
    load_old_b2_on_dataset,
    metric_rows_for_model,
    prediction_rows,
    prepare_dataset,
)
from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY = ROOT / "results/summaries/internal_defect_hardcase_b2_reference_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_hardcase_b2_reference_metrics.csv"
TAIL = ROOT / "results/metrics/internal_defect_hardcase_b2_reference_tail_metrics.csv"
PREDICTIONS = ROOT / "results/metrics/internal_defect_hardcase_b2_reference_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate old B2 artifact on v3_hardcase.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--artifact-manifest", type=Path, default=B2_MANIFEST)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail", type=Path, default=TAIL)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared = prepare_dataset(args.dataset_id)
    dataset = prepared["dataset"]
    y_std = prepared["y_std"].reshape(-1)
    pred, shape_pred, manifest = load_old_b2_on_dataset(prepared, args.artifact_manifest)
    metric_rows, tail_rows = metric_rows_for_model("old_B2_v2_artifact", False, manifest.get("seed", "2026"), dataset, prepared["splits"], pred, shape_pred, y_std)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.tail, tail_rows, TAIL_FIELDS)
    write_csv(args.predictions, prediction_rows("old_B2_v2_artifact", manifest.get("seed", "2026"), dataset, pred, shape_pred, y_std), PREDICTION_FIELDS)

    test_all = next(row for row in metric_rows if row["split"] == "test" and row["subset"] == "all")
    test_tail = next(row for row in tail_rows if row["split"] == "test" and row["subset"] == "all")
    test_source = next(row for row in metric_rows if row["split"] == "test" and row["subset"] == "source_v2")
    test_topup = next(row for row in metric_rows if row["split"] == "test" and row["subset"] == "hardcase_topup")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "22.3 old B2 reference replay on hard-case augmented dataset",
                f"dataset_id: {args.dataset_id}",
                f"artifact_manifest: {args.artifact_manifest}",
                "model: 21.9 B2_feature_fusion_burial_head checkpoint, no retraining",
                "input_policy: delta_b/BxByBz plus delta_b-derived features only",
                f"test_total_normalized_mae: {float(test_all['total_normalized_mae']):.6f}",
                f"test_LWD_mae_mm: {float(test_all['L_mae_mm']):.3f} / {float(test_all['W_mae_mm']):.3f} / {float(test_all['D_mae_mm']):.3f}",
                f"test_burial_depth_mae_mm: {float(test_all['burial_depth_mae_mm']):.3f}",
                f"test_center_xyz_component_mae_mm: {float(test_all['center_xyz_component_mae_mm']):.3f}",
                f"test_shape_accuracy_f1: {float(test_all['shape_accuracy']):.6f} / {float(test_all['shape_macro_f1']):.6f}",
                f"test_catastrophic_failure_count_rate: {test_tail['catastrophic_failure_count']} / {float(test_tail['catastrophic_failure_rate']):.6f}",
                f"test_geometry_branch_failure_count_rate: {test_tail['geometry_branch_failure_count']} / {float(test_tail['geometry_branch_failure_rate']):.6f}",
                f"test_center_p95_max_mm: {float(test_tail['center_xyz_error_p95_mm']):.3f} / {float(test_tail['center_xyz_error_max_mm']):.3f}",
                f"test_burial_p95_max_mm: {float(test_tail['burial_depth_error_p95_mm']):.3f} / {float(test_tail['burial_depth_error_max_mm']):.3f}",
                f"test_source_v2_total_mae: {float(test_source['total_normalized_mae']):.6f}",
                f"test_hardcase_topup_total_mae: {float(test_topup['total_normalized_mae']):.6f}",
                "checkpoint_written: false",
                "npz_written: false",
                "current_baseline_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

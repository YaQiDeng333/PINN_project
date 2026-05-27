#!/usr/bin/env python
"""Evaluate the fixed 20.85 baseline artifact on the 20.91b liftoff pack."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

import audit_true_3d_rbc_observation_perturbation_robustness as obs
import load_true_3d_rbc_liftoff_aug_dataset as liftoff
import load_true_3d_rbc_pilot_dataset as pilot
from run_true_3d_rbc_formal_benchmark_20_77_candidate import add_profile_error_rows


ROOT = liftoff.ROOT
ARTIFACT_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_baseline_evaluation_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_baseline_metrics.csv"
BY_LIFTOFF = ROOT / "results/metrics/true_3d_rbc_liftoff_baseline_by_liftoff.csv"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None, ""}]
    return float(np.mean(values)) if values else math.nan


def aggregate(rows: list[dict[str, Any]], split: str, subset_name: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
    if not subset:
        return {"candidate": "C0_reference_20_85_baseline", "split": split, "liftoff_subset": subset_name, "sample_count": 0}
    return {
        "candidate": "C0_reference_20_85_baseline",
        "split": split,
        "liftoff_subset": subset_name,
        "sample_count": len(subset),
        "normalized_param_mae": mean(subset, "normalized_param_mae_mean"),
        "dimension_mae_norm": mean(subset, "dimension_param_mae_norm"),
        "curvature_mae_norm": mean(subset, "curvature_param_mae_norm"),
        "L_mae_mm": mean(subset, "L_mae_mm"),
        "W_mae_mm": mean(subset, "W_mae_mm"),
        "D_mae_mm": mean(subset, "D_mae_mm"),
        "wLD_abs_error": mean(subset, "wLD_abs_error"),
        "wWD_abs_error": mean(subset, "wWD_abs_error"),
        "wLW_abs_error": mean(subset, "wLW_abs_error"),
        "wMAE_auxiliary": mean(subset, "curvature_mae_mean"),
        "projected_mask_iou": mean(subset, "projected_mask_iou"),
        "projected_mask_dice": mean(subset, "projected_mask_dice"),
        "profile_depth_rmse_m": mean(subset, "profile_depth_rmse_m"),
        "er_like_profile_error": mean(subset, "er_like_profile_error"),
        "max_depth_error_m": mean(subset, "max_depth_error_m"),
        "volume_proxy_rel_error": mean(subset, "volume_proxy_rel_error"),
    }


def run(dataset_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = liftoff.load_liftoff_dataset(dataset_id)
    artifact, checkpoint, model = obs.load_artifact(ARTIFACT_MANIFEST)
    ctx = {
        "x_mean": checkpoint["normalization"]["x_mean"],
        "x_std": checkpoint["normalization"]["x_std"],
        "y_mean": checkpoint["normalization"]["y_mean"],
        "y_std": checkpoint["normalization"]["y_std"],
    }
    pred_raw = obs.predict(model, dataset.x_channels, ctx)
    row_metrics = pilot.evaluate_param_predictions(dataset, pred_raw, stats={"y_mean": ctx["y_mean"], "y_std": ctx["y_std"]})
    profile_rows = add_profile_error_rows(dataset, pred_raw, row_metrics)
    for idx, row in enumerate(profile_rows):
        row["candidate"] = "C0_reference_20_85_baseline"
        row["sensor_z_m"] = float(dataset.sensor_z_m[idx])
        row["base_sample_id"] = str(dataset.base_sample_ids[idx])
        row["variant_name"] = str(dataset.variant_name[idx])
        row["selected_seed"] = True
        row["seed"] = int(artifact.get("seed", 42))

    metric_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in profile_rows if row["split"] == split]
        metric_rows.append(aggregate(profile_rows, split, "all_liftoff", split_rows))
        metric_rows.append(aggregate(profile_rows, split, "nominal_0p008", [row for row in split_rows if round(float(row["sensor_z_m"]), 3) == 0.008]))
        metric_rows.append(aggregate(profile_rows, split, "non_nominal", [row for row in split_rows if round(float(row["sensor_z_m"]), 3) != 0.008]))

    by_liftoff: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in profile_rows if row["split"] == split]
        for z in sorted({round(float(row["sensor_z_m"]), 3) for row in split_rows}):
            by_liftoff.append(aggregate(profile_rows, split, f"sensor_z_{z:.3f}", [row for row in split_rows if round(float(row["sensor_z_m"]), 3) == z]))

    write_csv(METRICS, metric_rows)
    write_csv(BY_LIFTOFF, by_liftoff)
    test_all = next(row for row in metric_rows if row["split"] == "test" and row["liftoff_subset"] == "all_liftoff")
    test_nom = next(row for row in metric_rows if row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008")
    test_non = next(row for row in metric_rows if row["split"] == "test" and row["liftoff_subset"] == "non_nominal")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.92 fixed 20.85 baseline evaluation on liftoff augmentation pack",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"baseline_artifact_manifest: {ARTIFACT_MANIFEST}",
                "model: 20.77/20.85 small Conv1D + MLP six-parameter head, seed=42",
                "training_run: false",
                "COMSOL_run: false",
                "CURRENT_BASELINE_update: false",
                "",
                f"test_all_profile_depth_rmse_m: {float(test_all['profile_depth_rmse_m']):.9f}",
                f"test_nominal_0p008_profile_depth_rmse_m: {float(test_nom['profile_depth_rmse_m']):.9f}",
                f"test_non_nominal_profile_depth_rmse_m: {float(test_non['profile_depth_rmse_m']):.9f}",
                f"test_all_projected_mask_dice: {float(test_all['projected_mask_dice']):.6f}",
                f"test_non_nominal_projected_mask_dice: {float(test_non['projected_mask_dice']):.6f}",
                f"test_non_nominal_LWD_MAE_mm: {float(test_non['L_mae_mm']):.3f} / {float(test_non['W_mae_mm']):.3f} / {float(test_non['D_mae_mm']):.3f}",
                f"test_non_nominal_wMAE_auxiliary: {float(test_non['wMAE_auxiliary']):.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metric_rows, by_liftoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=liftoff.DATASET_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.dataset_id)
    print(f"wrote {SUMMARY}")
    print(f"wrote {METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

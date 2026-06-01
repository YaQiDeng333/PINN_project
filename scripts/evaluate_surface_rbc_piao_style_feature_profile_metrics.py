#!/usr/bin/env python
"""Evaluate profile and projected-mask metrics for the 24.1 NLS-lite baseline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    V3_240_DATASET_ID,
    aggregate_prediction_rows,
    check_no_overwrite,
    clip_params_to_train_bounds,
    depth_grid_from_params,
    evaluate_param_predictions,
    write_csv,
)
from train_surface_rbc_piao_style_feature_baseline import run_model_selection


SUMMARY = ROOT / "results/summaries/surface_rbc_piao_style_feature_profile_eval_summary.txt"
PROFILE_METRICS = ROOT / "results/metrics/surface_rbc_piao_style_feature_profile_metrics.csv"
VS_REFERENCE = ROOT / "results/metrics/surface_rbc_piao_style_feature_vs_reference.csv"
REFERENCE_MATRIX = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_comparison_matrix.csv"

PROFILE_FIELDS = [
    "row_type",
    "model",
    "family",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "sample_id",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "size_bin",
    "normalized_param_mae_mean",
    "dimension_param_mae_norm",
    "curvature_param_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "curvature_mae_mean",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "max_depth_error_m",
    "projected_mask_iou",
    "projected_mask_dice",
    "projected_mask_area_error",
    "projected_mask_center_error_px",
    "volume_proxy_rel_error",
    "clip_applied",
    "notes",
]

REFERENCE_FIELDS = [
    "metric",
    "reference_label",
    "current_label",
    "reference_value",
    "current_value",
    "delta",
    "improved",
    "notes",
]

REFERENCE_LABELS = {
    "20.85_formal_rerun_20.77_protocol": "20.85_formal_rerun_20.77_protocol",
    "20.77_original_candidate": "20.77_original_candidate",
    "20.81_feature_fusion": "20.81_feature_fusion_visual_comparator",
}

METRIC_MAP = [
    ("total_normalized_mae", "test_total_mae", "normalized_param_mae_mean_mean", "lower"),
    ("L_mae_mm", "test_L_mae_mm", "L_mae_mm_mean", "lower"),
    ("W_mae_mm", "test_W_mae_mm", "W_mae_mm_mean", "lower"),
    ("D_mae_mm", "test_D_mae_mm", "D_mae_mm_mean", "lower"),
    ("wMAE_auxiliary", "test_wMAE_auxiliary", "curvature_mae_mean_mean", "lower"),
    ("wLD_abs_error", "test_wLD_abs_error", "wLD_abs_error_mean", "lower"),
    ("wWD_abs_error", "test_wWD_abs_error", "wWD_abs_error_mean", "lower"),
    ("wLW_abs_error", "test_wLW_abs_error", "wLW_abs_error_mean", "lower"),
    ("profile_depth_rmse_m", "test_profile_depth_rmse_m", "profile_depth_rmse_m_mean", "lower"),
    ("er_like_profile_error", "test_er_like_profile_error", "er_like_profile_error_mean", "lower"),
    ("projected_mask_iou", "test_projected_mask_iou", "projected_mask_iou_mean", "higher"),
    ("projected_mask_dice", "test_projected_mask_dice", "projected_mask_dice_mean", "higher"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [f(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def add_profile_error_rows(dataset: Any, pred_params_raw: np.ndarray, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pred_params, _ = clip_params_to_train_bounds(pred_params_raw, dataset)
    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        pred_depth = depth_grid_from_params(pred_params[idx])
        true_depth = dataset.profile_depth_grid_m[idx]
        denom = float(np.sum(true_depth**2))
        er_like = 0.0 if denom <= 1.0e-20 else float(np.sqrt(np.sum((pred_depth - true_depth) ** 2) / denom))
        updated = dict(row)
        updated["er_like_profile_error"] = er_like
        updated["max_depth_error_m"] = abs(float(pred_params[idx, 2] - dataset.rbc_params[idx, 2]))
        enriched.append(updated)
    return enriched


def split_aggregate(rows: list[dict[str, Any]], split_name: str, model_name: str) -> dict[str, Any]:
    agg = aggregate_prediction_rows(rows, model_name, split_name)
    subset = [row for row in rows if row["split"] == split_name]
    agg["er_like_profile_error_mean"] = mean(subset, "er_like_profile_error")
    agg["max_depth_error_m_mean"] = mean(subset, "max_depth_error_m")
    return agg


def profile_metric_rows(model: str, family: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {field: "" for field in PROFILE_FIELDS}
        item.update(row)
        item["row_type"] = "per_sample"
        item["model"] = model
        item["family"] = family
        item["group_field"] = ""
        item["group_value"] = ""
        item["notes"] = "labels used only for diagnostic evaluation"
        out.append(item)

    metric_keys = [
        "normalized_param_mae_mean",
        "dimension_param_mae_norm",
        "curvature_param_mae_norm",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "wLD_abs_error",
        "wWD_abs_error",
        "wLW_abs_error",
        "curvature_mae_mean",
        "profile_depth_rmse_m",
        "er_like_profile_error",
        "max_depth_error_m",
        "projected_mask_iou",
        "projected_mask_dice",
        "projected_mask_area_error",
        "projected_mask_center_error_px",
        "volume_proxy_rel_error",
        "clip_applied",
    ]
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        for group_field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
            for group_value in sorted({str(row[group_field]) for row in split_rows}):
                subset = [row for row in split_rows if str(row[group_field]) == group_value]
                if not subset:
                    continue
                item = {field: "" for field in PROFILE_FIELDS}
                item.update(
                    {
                        "row_type": "group_summary",
                        "model": model,
                        "family": family,
                        "split": split_name,
                        "group_field": group_field,
                        "group_value": group_value,
                        "sample_id": "",
                        "sample_count": len(subset),
                        "notes": "group metric means; labels used only for diagnostic evaluation",
                    }
                )
                for key in metric_keys:
                    item[key] = mean(subset, key)
                out.append(item)
    return out


def load_reference_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    return {row["candidate_id"]: row for row in rows if row.get("candidate_id") in REFERENCE_LABELS}


def reference_comparison_rows(current_label: str, current_test: dict[str, Any], references: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate_id, reference_label in REFERENCE_LABELS.items():
        ref = references.get(candidate_id, {})
        for metric, ref_key, current_key, direction in METRIC_MAP:
            ref_value = f(ref.get(ref_key))
            current_value = f(current_test.get(current_key))
            if math.isfinite(ref_value) and math.isfinite(current_value):
                delta = current_value - ref_value
                improved = delta < 0.0 if direction == "lower" else delta > 0.0
            else:
                delta = math.nan
                improved = ""
            out.append(
                {
                    "metric": metric,
                    "reference_label": reference_label,
                    "current_label": current_label,
                    "reference_value": ref_value,
                    "current_value": current_value,
                    "delta": delta,
                    "improved": improved,
                    "notes": "test split final only; lower is better except IoU/Dice" if direction == "lower" else "test split final only; higher is better",
                }
            )
    return out


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.profile_metrics, args.vs_reference], args.overwrite)
    result = run_model_selection(args.dataset_id)
    selected = result.selected
    base_rows = evaluate_param_predictions(result.dataset, selected.pred_raw, result.stats)
    rows = add_profile_error_rows(result.dataset, selected.pred_raw, base_rows)
    profile_rows = profile_metric_rows(selected.model, selected.family, rows)
    test_agg = split_aggregate(rows, "test", selected.model)
    val_agg = split_aggregate(rows, "val", selected.model)
    train_agg = split_aggregate(rows, "train", selected.model)
    references = load_reference_rows(REFERENCE_MATRIX)
    comparison_rows = reference_comparison_rows(selected.model, test_agg, references)

    write_csv(args.profile_metrics, profile_rows, PROFILE_FIELDS)
    write_csv(args.vs_reference, comparison_rows, REFERENCE_FIELDS)

    test_group_rows = [row for row in profile_rows if row["row_type"] == "group_summary" and row["split"] == "test"]
    worst_profile = max(test_group_rows, key=lambda row: f(row.get("profile_depth_rmse_m")), default={})
    best_dice = max(test_group_rows, key=lambda row: f(row.get("projected_mask_dice")), default={})
    improved_counts: dict[str, int] = {}
    for reference_label in REFERENCE_LABELS.values():
        improved_counts[reference_label] = sum(
            1 for row in comparison_rows if row["reference_label"] == reference_label and str(row["improved"]).lower() == "true"
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface_rbc_piao_style_feature_profile_eval_summary",
                "stage: 24.1",
                "",
                f"dataset_id: {result.dataset.dataset_id}",
                f"selected_model: {selected.model}",
                f"selected_family: {selected.family}",
                "model_input: nlslite_* only",
                "labels_used_for_training_targets_and_diagnostic_metrics_only: true",
                "COMSOL_run: false",
                "data_or_NPZ_modified: false",
                "CURRENT_BASELINE_update: false",
                f"train_total_normalized_mae: {train_agg['normalized_param_mae_mean_mean']:.6f}",
                f"val_total_normalized_mae: {val_agg['normalized_param_mae_mean_mean']:.6f}",
                f"test_total_normalized_mae: {test_agg['normalized_param_mae_mean_mean']:.6f}",
                f"test_LWD_mae_mm: L={test_agg['L_mae_mm_mean']:.6f}, W={test_agg['W_mae_mm_mean']:.6f}, D={test_agg['D_mae_mm_mean']:.6f}",
                f"test_wMAE_auxiliary: {test_agg['curvature_mae_mean_mean']:.6f}",
                f"test_wLD_wWD_wLW: {test_agg['wLD_abs_error_mean']:.6f}, {test_agg['wWD_abs_error_mean']:.6f}, {test_agg['wLW_abs_error_mean']:.6f}",
                f"test_profile_depth_rmse_m: {test_agg['profile_depth_rmse_m_mean']:.9f}",
                f"test_er_like_profile_error: {test_agg['er_like_profile_error_mean']:.6f}",
                f"test_projected_mask_iou_dice: {test_agg['projected_mask_iou_mean']:.6f}, {test_agg['projected_mask_dice_mean']:.6f}",
                f"test_volume_proxy_rel_error: {test_agg['volume_proxy_rel_error_mean']:.6f}",
                f"worst_test_group_by_profile_rmse: {worst_profile.get('group_field', '')}={worst_profile.get('group_value', '')}, rmse={f(worst_profile.get('profile_depth_rmse_m')):.9f}",
                f"best_test_group_by_dice: {best_dice.get('group_field', '')}={best_dice.get('group_value', '')}, dice={f(best_dice.get('projected_mask_dice')):.6f}",
        f"metrics_improved_vs_20_85_count: {improved_counts['20.85_formal_rerun_20.77_protocol']}",
        f"metrics_improved_vs_20_77_count: {improved_counts['20.77_original_candidate']}",
        f"metrics_improved_vs_20_81_count: {improved_counts['20.81_feature_fusion_visual_comparator']}",
                "docs_sync_completed: EXPERIMENT_LOG.md, NEXT_STEP.md, PINN优化路线.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--profile-metrics", type=Path, default=PROFILE_METRICS)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

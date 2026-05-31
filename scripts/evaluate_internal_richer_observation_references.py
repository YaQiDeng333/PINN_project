#!/usr/bin/env python
"""23.0/23.1 richer-observation reference evaluation。

该脚本先补齐 23.0 richer-observation evaluation gate 的 route decision，
再为 23.1 training gate 输出 reference metrics。它只读取 22.9 diagnostic
pack，不运行 COMSOL，不写 data/NPZ。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_richer_observation_dataset import (
    DATASET_ID,
    OBSERVATION_CONFIGS,
    ROOT,
    build_inputs,
    denormalize_y,
    load_dataset,
    metric_row,
    normalize_y,
    selection_score,
    shape_metrics,
    split_indices,
    standardize_matrix,
    target_scaler,
    tail_metrics,
    write_csv,
)


SUMMARY = ROOT / "results/summaries/internal_richer_observation_reference_summary.txt"
METRICS = ROOT / "results/metrics/internal_richer_observation_reference_metrics.csv"
EVAL_SUMMARY = ROOT / "results/summaries/internal_richer_observation_evaluation_summary.txt"
EVAL_ROUTE = ROOT / "results/summaries/internal_richer_observation_evaluation_route_decision_summary.txt"
EVAL_MATRIX = ROOT / "results/metrics/internal_richer_observation_evaluation_decision_matrix.csv"

METRIC_FIELDS = [
    "model",
    "observation_config",
    "split",
    "sample_count",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_component_mae_mm",
    "center_xyz_euclidean_mean_mm",
    "shape_accuracy",
    "shape_macro_f1",
    "center_xyz_error_mean_mm",
    "center_xyz_error_median_mm",
    "center_xyz_error_p90_mm",
    "center_xyz_error_p95_mm",
    "center_xyz_error_max_mm",
    "burial_depth_error_mean_mm",
    "burial_depth_error_median_mm",
    "burial_depth_error_p90_mm",
    "burial_depth_error_p95_mm",
    "burial_depth_error_max_mm",
    "catastrophic_failure_count",
    "catastrophic_failure_rate",
    "geometry_branch_failure_count",
    "geometry_branch_failure_rate",
    "shape_misclassified_count",
    "full_shift_failure_count",
]
DECISION_FIELDS = [
    "observation_config",
    "selected_for_23_1_training",
    "variants",
    "val_selection_score",
    "test_total_normalized_mae",
    "test_burial_depth_mae_mm",
    "test_center_xyz_component_mae_mm",
    "test_center_p95_mm",
    "test_center_max_mm",
    "test_burial_p95_mm",
    "test_burial_max_mm",
    "test_catastrophic_failure_count",
    "test_geometry_branch_failure_count",
    "test_shape_macro_f1",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate richer-observation reference probes and write 23.0 route decision.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--evaluation-summary", type=Path, default=EVAL_SUMMARY)
    parser.add_argument("--route-summary", type=Path, default=EVAL_ROUTE)
    parser.add_argument("--decision-matrix", type=Path, default=EVAL_MATRIX)
    return parser.parse_args()


def ridge_fit_predict(x: np.ndarray, y_norm: np.ndarray, train_idx: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    xt = x_aug[train_idx]
    yt = y_norm[train_idx]
    ident = np.eye(xt.shape[1], dtype=np.float32)
    ident[-1, -1] = 0.0
    coef = np.linalg.solve(xt.T @ xt + alpha * ident, xt.T @ yt)
    return (x_aug @ coef).astype(np.float32)


def centroid_predict(x: np.ndarray, shape: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    centroids = []
    global_mean = x[train_idx].mean(axis=0)
    for label in range(3):
        idx = train_idx[shape[train_idx] == label]
        centroids.append(x[idx].mean(axis=0) if idx.size else global_mean)
    c = np.stack(centroids)
    dist = ((x[:, None, :] - c[None, :, :]) ** 2).mean(axis=2)
    return np.argmin(dist, axis=1).astype(np.int64)


def evaluate_config(dataset: Any, config: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw, features, variants = build_inputs(dataset, config)
    splits = split_indices(dataset.split)
    x_std, _, _ = standardize_matrix(features, splits["train"])
    y_mean, y_std = target_scaler(dataset.y, splits["train"])
    y_norm = normalize_y(dataset.y, y_mean, y_std)
    pred_norm = ridge_fit_predict(x_std, y_norm, splits["train"], alpha=1.0)
    pred = denormalize_y(pred_norm, y_mean, y_std)
    shape_pred = centroid_predict(x_std, dataset.shape_label, splits["train"])
    rows = [metric_row("ridge_feature_probe", config, split, idx, dataset.y, pred, dataset.shape_label, shape_pred, y_std.reshape(-1)) for split, idx in splits.items()]
    val_row = next(row for row in rows if row["split"] == "val")
    test_row = next(row for row in rows if row["split"] == "test")
    decision = {
        "observation_config": config,
        "variants": ";".join(variants),
        "val_selection_score": selection_score(val_row),
        "test_total_normalized_mae": test_row["total_normalized_mae"],
        "test_burial_depth_mae_mm": test_row["burial_depth_mae_mm"],
        "test_center_xyz_component_mae_mm": test_row["center_xyz_component_mae_mm"],
        "test_center_p95_mm": test_row["center_xyz_error_p95_mm"],
        "test_center_max_mm": test_row["center_xyz_error_max_mm"],
        "test_burial_p95_mm": test_row["burial_depth_error_p95_mm"],
        "test_burial_max_mm": test_row["burial_depth_error_max_mm"],
        "test_catastrophic_failure_count": test_row["catastrophic_failure_count"],
        "test_geometry_branch_failure_count": test_row["geometry_branch_failure_count"],
        "test_shape_macro_f1": test_row["shape_macro_f1"],
    }
    return rows, decision


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id)
    metric_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for config in OBSERVATION_CONFIGS:
        rows, decision = evaluate_config(dataset, config)
        metric_rows.extend(rows)
        decision_rows.append(decision)
    config_rank = {config: i for i, config in enumerate(OBSERVATION_CONFIGS)}
    decision_rows.sort(key=lambda row: (float(row["val_selection_score"]), config_rank[row["observation_config"]]))
    selected = decision_rows[0]["observation_config"]
    for row in decision_rows:
        row["selected_for_23_1_training"] = row["observation_config"] == selected
        row["decision"] = "selected" if row["selected_for_23_1_training"] else "not_selected"
        if row["observation_config"] == selected:
            row["notes"] = "23.0 evaluation gate recommends this config for 23.1 training; selection uses validation score, test is reported final only."
        else:
            row["notes"] = "reference diagnostic only"
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.decision_matrix, decision_rows, DECISION_FIELDS)
    selected_row = next(row for row in decision_rows if row["selected_for_23_1_training"])
    text = "\n".join(
        [
            "# 23.0 internal richer-observation evaluation summary",
            "",
            f"- dataset_id: {args.dataset_id}",
            f"- base_count: {len(dataset.base_ids)}",
            "- method: train-only standardized delta_b-derived feature probe; validation score selects the observation config; test metrics are final report only.",
            f"- recommended_23_1_observation_config: {selected}",
            f"- selected_variants: {selected_row['variants']}",
            f"- selected_val_score: {float(selected_row['val_selection_score']):.6f}",
            f"- selected_test_total_normalized_mae: {float(selected_row['test_total_normalized_mae']):.6f}",
            f"- selected_test_center_p95/max_mm: {float(selected_row['test_center_p95_mm']):.3f} / {float(selected_row['test_center_max_mm']):.3f}",
            f"- selected_test_burial_p95/max_mm: {float(selected_row['test_burial_p95_mm']):.3f} / {float(selected_row['test_burial_max_mm']):.3f}",
            f"- selected_test_catastrophic/geometry_count: {selected_row['test_catastrophic_failure_count']} / {selected_row['test_geometry_branch_failure_count']}",
            "- conclusion: 23.1 may proceed only with the selected config above; R3/R4 remain out of scope.",
        ]
    ) + "\n"
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(text, encoding="utf-8")
    args.evaluation_summary.write_text(text, encoding="utf-8")
    route_text = "\n".join(
        [
            "# 23.0 internal richer-observation evaluation route decision",
            "",
            "route_decision: proceed_to_23_1_training_gate",
            f"selected_observation_config: {selected}",
            f"selected_variants: {selected_row['variants']}",
            "allowed_training_configs: R1_5line_z0p008, R1_9line_z0p008, R2_5line_multi_liftoff, R1_plus_R2_combined",
            "r3_r4_allowed: false",
            "selection_rule: validation score from richer-observation feature probe; test metrics are not used for selection.",
            "baseline_update: false",
            "",
            "理由：23.0 已经明确选择一个 R1/R2/R1+R2 范围内的配置，满足 23.1 的训练前置门槛。",
        ]
    ) + "\n"
    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    args.route_summary.write_text(route_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

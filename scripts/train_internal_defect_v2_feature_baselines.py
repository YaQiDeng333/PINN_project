#!/usr/bin/env python
"""训练 21.4 internal defect v2_240 的 delta_b-derived feature baselines。

特征只从 Bx/By/Bz delta_b 计算；labels 和 metadata 只用于 supervision、
validation selection、test-final metrics 和分组诊断。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import (
    ROOT,
    classification_metrics,
    load_dataset,
    regression_metrics,
    split_indices,
    train_target_scaler,
    write_csv,
)
from train_internal_defect_feature_baselines import (
    GROUP_FIELDS,
    METRIC_FIELDS,
    build_models,
    evaluate_split,
    extract_features,
    group_rows,
    model_predictions,
    selection_score,
    standardize_features,
)


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SUMMARY = ROOT / "results/summaries/internal_defect_v2_feature_baseline_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_v2_feature_baseline_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_v2_feature_baseline_group_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 internal defect v2_240 feature baselines。")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    x_feat, feature_names = extract_features(dataset.delta_b)
    x_feat, _, _ = standardize_features(x_feat, splits["train"])
    y = dataset.y_regression
    y_mean, y_std = train_target_scaler(y, splits["train"])
    y_norm = (y - y_mean) / y_std
    shape = dataset.shape_label
    metric_rows: list[dict[str, Any]] = []
    all_preds: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    best_name = ""
    best_score = float("inf")

    models = build_models()
    for name, reg_model, cls_model in models:
        reg_model.fit(x_feat[splits["train"]], y_norm[splits["train"]])
        cls_model.fit(x_feat[splits["train"]], shape[splits["train"]])
        pred_norm = model_predictions(reg_model, x_feat)
        pred = pred_norm * y_std + y_mean
        shape_pred = np.asarray(cls_model.predict(x_feat), dtype=np.int64)
        val_reg = regression_metrics(y[splits["val"]], pred[splits["val"]], y_std.reshape(-1))
        val_cls = classification_metrics(shape[splits["val"]], shape_pred[splits["val"]])
        score = selection_score(val_reg["total_normalized_mae"], val_cls["shape_accuracy"])
        all_preds[name] = (pred, shape_pred, score)
        if score < best_score:
            best_name = name
            best_score = score
        for split_name, idx in splits.items():
            metric_rows.append(evaluate_split(name, "", split_name, idx, y, pred, shape, shape_pred, y_std.reshape(-1), score))

    selected_pred, selected_shape_pred, _selected_score = all_preds[best_name]
    for row in metric_rows:
        if row["model"] == best_name:
            row["selected_model"] = best_name
    group_values = {
        "shape_type": dataset.shape_type,
        "burial_depth_level": dataset.burial_depth_level,
        "size_level": dataset.size_level,
        "aspect_bin": dataset.aspect_bin,
    }
    group_summary_rows: list[dict[str, Any]] = []
    for split_name, idx in splits.items():
        group_summary_rows.extend(
            group_rows(best_name, best_name, split_name, idx, y, selected_pred, shape, selected_shape_pred, y_std.reshape(-1), group_values)
        )

    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_summary_rows, GROUP_FIELDS)
    test_reg = regression_metrics(y[splits["test"]], selected_pred[splits["test"]], y_std.reshape(-1))
    test_cls = classification_metrics(shape[splits["test"]], selected_shape_pred[splits["test"]])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.4 internal defect v2_240 feature baseline 摘要",
                f"dataset_id: {args.dataset_id}",
                "feature_source: 仅来自 delta_b/BxByBz",
                f"feature_count: {len(feature_names)}",
                f"candidate_models: {', '.join(name for name, _, _ in models)}",
                f"selected_model: {best_name}",
                f"validation_selection_score: {best_score:.6f}",
                f"test_total_normalized_mae: {test_reg['total_normalized_mae']:.6f}",
                f"test_LWD_mae_mm: {test_reg['L_mae_mm']:.3f} / {test_reg['W_mae_mm']:.3f} / {test_reg['D_mae_mm']:.3f}",
                f"test_burial_depth_mae_mm: {test_reg['burial_depth_mae_mm']:.3f}",
                f"test_center_xyz_mae_mm: {test_reg['center_xyz_mae_mm']:.3f}",
                f"test_shape_accuracy: {test_cls['shape_accuracy']:.6f}",
                f"test_shape_macro_f1: {test_cls['shape_macro_f1']:.6f}",
                "label_leakage: false; split/sample_id/shape/depth/size/aspect metadata 未作为特征输入。",
                "selection_protocol: validation-only model selection; test final only.",
                "current_baseline_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

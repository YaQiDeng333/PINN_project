#!/usr/bin/env python
"""23.3 lightweight diagnostic probes for single-vs-dual scan directions.

这些 probe 只用于诊断，不是正式模型候选：不保存 checkpoint，不更新 baseline。
输入只来自 delta_b 派生特征；shape/burial/size/aspect/sample_id 不作为输入。
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_internal_multi_scan_direction_pairs import (
    CONFIGS,
    PARAM_NAMES,
    ROOT,
    build_feature_matrix,
    load_dataset,
    metric_row,
    selection_score,
    split_indices,
    tail_metrics,
    target_scaler,
    train_standardize,
    write_csv,
)


SUMMARY = ROOT / "results/summaries/internal_multi_scan_direction_diagnostic_probe_summary.txt"
METRICS = ROOT / "results/metrics/internal_multi_scan_direction_diagnostic_probe_metrics.csv"
TAIL_METRICS = ROOT / "results/metrics/internal_multi_scan_direction_diagnostic_probe_tail_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_multi_scan_direction_diagnostic_probe_group_summary.csv"

MODEL_NAMES = ["mean_baseline", "ridge_logreg", "svr_rbf_C10", "random_forest_small"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight 23.3 diagnostic probes.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail-metrics", type=Path, default=TAIL_METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    return parser.parse_args()


def mode_label(values: np.ndarray) -> int:
    counter = Counter([int(v) for v in values.tolist()])
    return int(counter.most_common(1)[0][0])


def encode_labels(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes = sorted({str(v) for v in values.tolist()})
    mapping = {name: i for i, name in enumerate(classes)}
    return np.asarray([mapping[str(v)] for v in values.tolist()], dtype=np.int64), classes


def _fit_predict_model(model_name: str, x: np.ndarray, y: np.ndarray, shape: np.ndarray, aspect: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = split_indices(split)
    train_idx = splits["train"]
    x_std, _, _ = train_standardize(x, train_idx)
    y_mean, y_std = target_scaler(y, train_idx)
    y_norm = ((y - y_mean) / y_std).astype(np.float32)
    aspect_label, _aspect_classes = encode_labels(aspect)

    if model_name == "mean_baseline":
        pred_norm = np.zeros_like(y_norm)
        shape_pred = np.full(shape.shape, mode_label(shape[train_idx]), dtype=np.int64)
        aspect_pred = np.full(aspect_label.shape, mode_label(aspect_label[train_idx]), dtype=np.int64)
        return (pred_norm * y_std + y_mean).astype(np.float32), shape_pred, aspect_pred

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVC, SVR

    if model_name == "ridge_logreg":
        reg = Ridge(alpha=1.0)
        cls = LogisticRegression(max_iter=1000, class_weight="balanced")
        asp = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_name == "svr_rbf_C10":
        reg = MultiOutputRegressor(SVR(C=10.0, kernel="rbf", gamma="scale"))
        cls = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced")
        asp = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced")
    elif model_name == "random_forest_small":
        reg = RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=2, random_state=42)
        cls = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=2, random_state=42, class_weight="balanced")
        asp = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=2, random_state=42, class_weight="balanced")
    else:
        raise KeyError(model_name)

    reg.fit(x_std[train_idx], y_norm[train_idx])
    cls.fit(x_std[train_idx], shape[train_idx])
    asp.fit(x_std[train_idx], aspect_label[train_idx])
    pred_norm = np.asarray(reg.predict(x_std), dtype=np.float32)
    shape_pred = np.asarray(cls.predict(x_std), dtype=np.int64)
    aspect_pred = np.asarray(asp.predict(x_std), dtype=np.int64)
    return (pred_norm * y_std + y_mean).astype(np.float32), shape_pred, aspect_pred


def extended_metric_row(model: str, config: str, split_name: str, idx: np.ndarray, dataset: Any, y_pred: np.ndarray, shape_pred: np.ndarray, aspect_pred: np.ndarray, y_std: np.ndarray) -> dict[str, Any]:
    row = metric_row(model, config, split_name, idx, dataset.y, y_pred, dataset.shape_label, shape_pred, y_std.reshape(-1))
    shape_true = dataset.shape_label[idx]
    shape_p = shape_pred[idx]
    cuboid, ellipsoid = 0, 1
    ce_mask = np.isin(shape_true, [cuboid, ellipsoid])
    ce_conf = ce_mask & np.isin(shape_p, [cuboid, ellipsoid]) & (shape_true != shape_p)
    aspect_label, _ = encode_labels(dataset.aspect_bin)
    aspect_true = aspect_label[idx]
    aspect_p = aspect_pred[idx]
    elongated_names = sorted({str(v) for v in dataset.aspect_bin.tolist()})
    elongated_ids = [i for i, name in enumerate(elongated_names) if name in {"elongated_x", "elongated_y"}]
    elong_mask = np.isin(aspect_true, elongated_ids)
    elong_conf = elong_mask & np.isin(aspect_p, elongated_ids) & (aspect_true != aspect_p)
    row["cuboid_ellipsoid_confusion_count"] = int(ce_conf.sum())
    row["cuboid_ellipsoid_confusion_rate"] = float(ce_conf.sum() / max(1, ce_mask.sum()))
    row["aspect_accuracy"] = float(np.mean(aspect_true == aspect_p)) if aspect_true.size else 0.0
    row["elongated_aspect_confusion_count"] = int(elong_conf.sum())
    row["elongated_aspect_confusion_rate"] = float(elong_conf.sum() / max(1, elong_mask.sum()))
    row["diagnostic_probe_only"] = True
    row["input_boundary"] = "delta_b_derived_features_only"
    return row


def group_rows(model: str, config: str, dataset: Any, y_pred: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray, split_name: str = "test") -> list[dict[str, Any]]:
    idx_split = split_indices(dataset.split)[split_name]
    rows: list[dict[str, Any]] = []
    for field, values in [
        ("shape_type", dataset.shape_type),
        ("burial_depth_level", dataset.burial_depth_level),
        ("size_level", dataset.size_level),
        ("aspect_bin", dataset.aspect_bin),
    ]:
        for value in sorted(set(values[idx_split].tolist())):
            idx = idx_split[values[idx_split] == value]
            if idx.size == 0:
                continue
            row = metric_row(model, config, split_name, idx, dataset.y, y_pred, dataset.shape_label, shape_pred, y_std.reshape(-1))
            rows.append(
                {
                    "model": model,
                    "observation_config": config,
                    "split": split_name,
                    "group_field": field,
                    "group_value": str(value),
                    "sample_count": int(idx.size),
                    "total_normalized_mae": row["total_normalized_mae"],
                    "burial_depth_mae_mm": row["burial_depth_mae_mm"],
                    "center_xyz_component_mae_mm": row["center_xyz_component_mae_mm"],
                    "shape_macro_f1": row["shape_macro_f1"],
                    "catastrophic_failure_count": row["catastrophic_failure_count"],
                    "geometry_branch_failure_count": row["geometry_branch_failure_count"],
                }
            )
    return rows


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    splits = split_indices(dataset.split)
    metric_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    group_summary: list[dict[str, Any]] = []
    selected_by_config: dict[str, dict[str, Any]] = {}
    y_mean, y_std = target_scaler(dataset.y, splits["train"])

    for config in CONFIGS:
        features, _names = build_feature_matrix(dataset, config)
        candidate_rows: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]] = []
        for model_name in MODEL_NAMES:
            y_pred, shape_pred, aspect_pred = _fit_predict_model(model_name, features, dataset.y, dataset.shape_label, dataset.aspect_bin, dataset.split)
            rows = [
                extended_metric_row(model_name, config, split_name, idx, dataset, y_pred, shape_pred, aspect_pred, y_std)
                for split_name, idx in splits.items()
            ]
            val_row = next(row for row in rows if row["split"] == "val")
            candidate_rows.append((selection_score(val_row), model_name, y_pred, shape_pred, aspect_pred, rows))
            for row in rows:
                row["selected_model"] = False
                if row["split"] != "test":
                    metric_rows.append(row)
        candidate_rows.sort(key=lambda item: (item[0], MODEL_NAMES.index(item[1])))
        _score, selected_model, y_pred, shape_pred, aspect_pred, rows = candidate_rows[0]
        selected_by_config[config] = {"model": selected_model, "score": _score}
        for row in rows:
            row["selected_model"] = True
            metric_rows.append(row)
            if row["split"] == "test":
                tail = tail_metrics(dataset.y[splits["test"]], y_pred[splits["test"]], dataset.shape_label[splits["test"]], shape_pred[splits["test"]])
                tail_rows.append({"model": selected_model, "observation_config": config, "split": "test", **tail})
        group_summary.extend(group_rows(selected_model, config, dataset, y_pred, shape_pred, y_std, "test"))

    metric_fields = [
        "model",
        "observation_config",
        "selected_model",
        "split",
        "sample_count",
        "selection_score",
        "total_normalized_mae",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "burial_depth_mae_mm",
        "center_xyz_component_mae_mm",
        "center_xyz_euclidean_mean_mm",
        "shape_accuracy",
        "shape_macro_f1",
        "center_xyz_error_p90_mm",
        "center_xyz_error_p95_mm",
        "center_xyz_error_max_mm",
        "burial_depth_error_p90_mm",
        "burial_depth_error_p95_mm",
        "burial_depth_error_max_mm",
        "catastrophic_failure_count",
        "catastrophic_failure_rate",
        "geometry_branch_failure_count",
        "geometry_branch_failure_rate",
        "shape_misclassified_count",
        "full_shift_failure_count",
        "cuboid_ellipsoid_confusion_count",
        "cuboid_ellipsoid_confusion_rate",
        "aspect_accuracy",
        "elongated_aspect_confusion_count",
        "elongated_aspect_confusion_rate",
        "diagnostic_probe_only",
        "input_boundary",
    ]
    tail_fields = [
        "model",
        "observation_config",
        "split",
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
    write_csv(args.metrics, metric_rows, metric_fields)
    write_csv(args.tail_metrics, tail_rows, tail_fields)
    write_csv(
        args.group_summary,
        group_summary,
        [
            "model",
            "observation_config",
            "split",
            "group_field",
            "group_value",
            "sample_count",
            "total_normalized_mae",
            "burial_depth_mae_mm",
            "center_xyz_component_mae_mm",
            "shape_macro_f1",
            "catastrophic_failure_count",
            "geometry_branch_failure_count",
        ],
    )
    selected_test = [row for row in metric_rows if row["selected_model"] and row["split"] == "test"]
    selected_test.sort(key=lambda row: (selection_score(row), str(row["observation_config"])))
    best = selected_test[0]
    lines = [
        "23.3 internal multi-scan-direction lightweight diagnostic probe summary",
        "",
        "scope: diagnostic probe only; no formal model candidate, no checkpoint, no COMSOL, no data/NPZ mutation.",
        f"base split: {dict(Counter(dataset.split.tolist()))}",
        f"best_validation_selected_test_config_by_score: {best['observation_config']}",
        f"best_selected_model: {best['model']}",
        f"test_total_normalized_mae: {float(best['total_normalized_mae']):.6f}",
        f"test_shape_accuracy/F1: {float(best['shape_accuracy']):.6f} / {float(best['shape_macro_f1']):.6f}",
        f"test_center_p95/max_mm: {float(best['center_xyz_error_p95_mm']):.3f} / {float(best['center_xyz_error_max_mm']):.3f}",
        f"test_burial_p95/max_mm: {float(best['burial_depth_error_p95_mm']):.3f} / {float(best['burial_depth_error_max_mm']):.3f}",
        f"test_catastrophic/geometry_count: {best['catastrophic_failure_count']} / {best['geometry_branch_failure_count']}",
        "",
        "结论：probe 仅用于判断 dual-direction 是否值得进入 23.4；不得把 probe 写成正式模型候选。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

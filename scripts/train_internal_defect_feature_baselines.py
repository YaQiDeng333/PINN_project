#!/usr/bin/env python
"""Train delta_b-derived feature baselines for the 21.2 internal defect gate.

Features are computed only from Bx/By/Bz delta_b. Labels and metadata are used
only for supervision, validation selection, and metrics.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import (
    CENTER_NAMES,
    DATASET_ID,
    PARAM_NAMES,
    ROOT,
    SHAPE_CLASSES,
    classification_metrics,
    load_dataset,
    regression_metrics,
    split_indices,
    train_target_scaler,
    write_csv,
)


SUMMARY = ROOT / "results/summaries/internal_defect_feature_baseline_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_feature_baseline_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_feature_baseline_group_summary.csv"

METRIC_FIELDS = [
    "model",
    "selected_model",
    "split",
    "sample_count",
    "selection_score",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
]
GROUP_FIELDS = [
    "model",
    "selected_model",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_mae_mm",
    "shape_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train internal defect feature baselines.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def half_width(signal: np.ndarray) -> float:
    values = np.abs(signal)
    peak = float(values.max())
    if peak <= 0:
        return 0.0
    return float((values >= peak * 0.5).sum()) / float(values.size)


def center_of_energy(signal: np.ndarray) -> float:
    values = np.abs(signal)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    idx = np.arange(values.size, dtype=np.float64)
    return float((idx * values).sum() / total / max(values.size - 1, 1))


def extract_features(delta_b: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    names: list[str] = []
    base_names: list[str] = []
    for axis in range(3):
        for line in range(3):
            prefix = f"a{axis}_l{line}"
            base_names.extend(
                [
                    f"{prefix}_max",
                    f"{prefix}_min",
                    f"{prefix}_ptp",
                    f"{prefix}_abs_peak",
                    f"{prefix}_mean",
                    f"{prefix}_std",
                    f"{prefix}_energy",
                    f"{prefix}_grad_energy",
                    f"{prefix}_half_width",
                    f"{prefix}_center_energy",
                    f"{prefix}_arg_abs_peak",
                ]
            )
    base_names.extend([f"axis_{axis}_energy" for axis in range(3)])
    base_names.extend(["energy_bx_by_ratio", "energy_bx_bz_ratio", "energy_by_bz_ratio"])
    base_names.extend([f"line_{line}_energy" for line in range(3)])
    base_names.extend(["line_0_1_energy_ratio", "line_2_1_energy_ratio"])
    names = base_names
    eps = 1e-12
    for sample in delta_b:
        feats: list[float] = []
        for axis in range(3):
            for line in range(3):
                s = np.asarray(sample[axis, line], dtype=np.float64)
                g = np.diff(s)
                feats.extend(
                    [
                        float(s.max()),
                        float(s.min()),
                        float(np.ptp(s)),
                        float(np.max(np.abs(s))),
                        float(s.mean()),
                        float(s.std()),
                        float(np.sqrt(np.mean(s * s))),
                        float(np.sqrt(np.mean(g * g))) if g.size else 0.0,
                        half_width(s),
                        center_of_energy(s),
                        float(np.argmax(np.abs(s))) / max(len(s) - 1, 1),
                    ]
                )
        axis_energy = [float(np.sqrt(np.mean(sample[axis] ** 2))) for axis in range(3)]
        feats.extend(axis_energy)
        feats.extend([axis_energy[0] / (axis_energy[1] + eps), axis_energy[0] / (axis_energy[2] + eps), axis_energy[1] / (axis_energy[2] + eps)])
        line_energy = [float(np.sqrt(np.mean(sample[:, line, :] ** 2))) for line in range(3)]
        feats.extend(line_energy)
        feats.extend([line_energy[0] / (line_energy[1] + eps), line_energy[2] / (line_energy[1] + eps)])
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32), names


class MeanRegressor:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "MeanRegressor":
        self.mean_ = y.mean(axis=0, keepdims=True)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.mean_, x.shape[0], axis=0)


class MajorityClassifier:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "MajorityClassifier":
        counts = Counter(y.tolist())
        self.cls_ = int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.cls_, dtype=np.int64)


def standardize_features(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_models() -> list[tuple[str, Any, Any]]:
    models: list[tuple[str, Any, Any]] = [("mean_baseline", MeanRegressor(), MajorityClassifier())]
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.linear_model import Ridge, RidgeClassifier
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVC, SVR

        models.extend(
            [
                ("ridge_alpha1", Ridge(alpha=1.0), RidgeClassifier(alpha=1.0)),
                ("kernel_ridge_rbf", KernelRidge(alpha=0.05, kernel="rbf", gamma=0.2), SVC(C=3.0, gamma="scale", kernel="rbf")),
                ("svr_rbf_C10", MultiOutputRegressor(SVR(C=10.0, epsilon=0.03, gamma="scale")), SVC(C=10.0, gamma="scale", kernel="rbf")),
                (
                    "random_forest_cheap",
                    RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42, min_samples_leaf=2),
                    RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42, min_samples_leaf=2),
                ),
            ]
        )
    except Exception:
        pass
    return models


def model_predictions(model: Any, x: np.ndarray) -> np.ndarray:
    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float32)


def selection_score(total_norm_mae: float, shape_acc: float) -> float:
    return float(total_norm_mae + 0.35 * (1.0 - shape_acc))


def evaluate_split(model_name: str, selected_model: str, split_name: str, idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, shape_true: np.ndarray, shape_pred: np.ndarray, y_scale: np.ndarray, score: float) -> dict[str, Any]:
    reg = regression_metrics(y_true[idx], y_pred[idx], y_scale)
    cls = classification_metrics(shape_true[idx], shape_pred[idx])
    return {
        "model": model_name,
        "selected_model": selected_model,
        "split": split_name,
        "sample_count": int(idx.size),
        "selection_score": score if split_name == "val" else "",
        **reg,
        **cls,
    }


def group_rows(model_name: str, selected_model: str, split_name: str, idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, shape_true: np.ndarray, shape_pred: np.ndarray, y_scale: np.ndarray, group_values: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, values in group_values.items():
        for value in sorted(set(values[idx].tolist())):
            sub = idx[values[idx] == value]
            if sub.size == 0:
                continue
            reg = regression_metrics(y_true[sub], y_pred[sub], y_scale)
            cls = classification_metrics(shape_true[sub], shape_pred[sub])
            rows.append(
                {
                    "model": model_name,
                    "selected_model": selected_model,
                    "split": split_name,
                    "group_field": field,
                    "group_value": value,
                    "sample_count": int(sub.size),
                    **reg,
                    "shape_accuracy": cls["shape_accuracy"],
                }
            )
    return rows


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

    for name, reg_model, cls_model in build_models():
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

    selected_pred, selected_shape_pred, selected_score = all_preds[best_name]
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
        group_summary_rows.extend(group_rows(best_name, best_name, split_name, idx, y, selected_pred, shape, selected_shape_pred, y_std.reshape(-1), group_values))

    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_summary_rows, GROUP_FIELDS)
    test_reg = regression_metrics(y[splits["test"]], selected_pred[splits["test"]], y_std.reshape(-1))
    test_cls = classification_metrics(shape[splits["test"]], selected_shape_pred[splits["test"]])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.2 内部/埋藏缺陷 feature baseline 摘要",
                "",
                f"dataset_id: {args.dataset_id}",
                "feature_source: 仅来自 delta_b/BxByBz",
                f"feature_count: {len(feature_names)}",
                f"candidate_models: {', '.join(name for name, _, _ in build_models())}",
                f"selected_model: {best_name}",
                f"validation_selection_score: {best_score:.6f}",
                f"test_total_normalized_mae: {test_reg['total_normalized_mae']:.6f}",
                f"test_LWD_mae_mm: {test_reg['L_mae_mm']:.3f} / {test_reg['W_mae_mm']:.3f} / {test_reg['D_mae_mm']:.3f}",
                f"test_burial_depth_mae_mm: {test_reg['burial_depth_mae_mm']:.3f}",
                f"test_center_xyz_mae_mm: {test_reg['center_xyz_mae_mm']:.3f}",
                f"test_shape_accuracy: {test_cls['shape_accuracy']:.6f}",
                f"test_shape_macro_f1: {test_cls['shape_macro_f1']:.6f}",
                "label_leakage: false；split/sample_id/shape/depth metadata 未作为特征输入。",
                "test_final_only: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

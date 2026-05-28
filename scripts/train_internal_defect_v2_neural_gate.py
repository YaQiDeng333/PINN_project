#!/usr/bin/env python
"""训练 21.4 internal defect v2_240 neural gate。

模型输入只允许 delta_b/BxByBz，形状为 (N,9,201)。labels 和 metadata
只用于 supervision、validation-only selection 和 test-final metrics。
本脚本不保存 checkpoint，不更新 CURRENT_BASELINE.md。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import (
    ROOT,
    classification_metrics,
    denormalize_y,
    load_dataset,
    normalize_x,
    normalize_y,
    regression_metrics,
    split_indices,
    train_normalization,
    train_target_scaler,
    write_csv,
)
from train_internal_defect_neural_gate import (
    COMPARE_FIELDS,
    EPOCH_FIELDS,
    GROUP_FIELDS,
    METRIC_FIELDS,
    SEED_FIELDS,
    eval_split,
    group_rows,
    predict,
    train_one_seed,
)


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SUMMARY = ROOT / "results/summaries/internal_defect_v2_neural_training_gate_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_v2_neural_seed_summary.csv"
METRICS = ROOT / "results/metrics/internal_defect_v2_neural_metrics.csv"
EPOCH_LOG = ROOT / "results/metrics/internal_defect_v2_neural_epoch_log.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_v2_neural_group_summary.csv"
VS_FEATURE = ROOT / "results/metrics/internal_defect_v2_vs_feature_baseline.csv"
FEATURE_METRICS = ROOT / "results/metrics/internal_defect_v2_feature_baseline_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 internal defect v2_240 neural gate。")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--epoch-log", type=Path, default=EPOCH_LOG)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--vs-feature", type=Path, default=VS_FEATURE)
    parser.add_argument("--feature-metrics", type=Path, default=FEATURE_METRICS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_feature_comparison(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "test" and (row.get("model") == "mean_baseline" or row.get("selected_model")):
                rows.append(
                    {
                        "model": row.get("model", ""),
                        "source": "feature_baseline",
                        "selected": bool(row.get("selected_model")),
                        "split": "test",
                        "sample_count": row.get("sample_count", ""),
                        "total_normalized_mae": row.get("total_normalized_mae", ""),
                        "L_mae_mm": row.get("L_mae_mm", ""),
                        "W_mae_mm": row.get("W_mae_mm", ""),
                        "D_mae_mm": row.get("D_mae_mm", ""),
                        "burial_depth_mae_mm": row.get("burial_depth_mae_mm", ""),
                        "center_xyz_mae_mm": row.get("center_xyz_mae_mm", ""),
                        "shape_accuracy": row.get("shape_accuracy", ""),
                        "shape_macro_f1": row.get("shape_macro_f1", ""),
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    x_mean, x_std = train_normalization(dataset.x_channels, splits["train"])
    x = normalize_x(dataset.x_channels, x_mean, x_std)
    y = dataset.y_regression
    y_mean, y_std = train_target_scaler(y, splits["train"])
    y_norm = normalize_y(y, y_mean, y_std)
    shape = dataset.shape_label
    seed_results = [train_one_seed(seed, args, x, y_norm, shape, splits) for seed in [42, 123, 2026]]

    candidate = "internal_v2_conv1d_multitask"
    selected = min(seed_results, key=lambda item: item["best_score"])
    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    selected_test: dict[str, Any] | None = None
    selected_train: dict[str, Any] | None = None
    selected_val: dict[str, Any] | None = None
    group_values = {
        "shape_type": dataset.shape_type,
        "burial_depth_level": dataset.burial_depth_level,
        "size_level": dataset.size_level,
        "aspect_bin": dataset.aspect_bin,
    }

    for result in seed_results:
        seed = int(result["seed"])
        is_selected = seed == int(selected["seed"])
        pred_norm, shape_pred = predict(result["model"], x)
        pred = denormalize_y(pred_norm, y_mean, y_std)
        split_eval: dict[str, dict[str, Any]] = {}
        for split_name, idx in splits.items():
            score = result["best_score"] if split_name == "val" else ""
            row = eval_split(candidate, is_selected, seed, split_name, idx, y, pred, shape, shape_pred, y_std.reshape(-1), score)
            metric_rows.append(row)
            split_eval[split_name] = row
            if is_selected and split_name == "train":
                selected_train = row
            if is_selected and split_name == "val":
                selected_val = row
            if is_selected and split_name == "test":
                selected_test = row
            if is_selected:
                group_summary_rows.extend(group_rows(candidate, is_selected, seed, split_name, idx, y, pred, shape, shape_pred, y_std.reshape(-1), group_values))
        seed_rows.append(
            {
                "candidate": candidate,
                "selected_seed": is_selected,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_val_selection_score": result["best_score"],
                "train_total_normalized_mae": split_eval["train"]["total_normalized_mae"],
                "val_total_normalized_mae": split_eval["val"]["total_normalized_mae"],
                "test_total_normalized_mae": split_eval["test"]["total_normalized_mae"],
                "train_shape_accuracy": split_eval["train"]["shape_accuracy"],
                "val_shape_accuracy": split_eval["val"]["shape_accuracy"],
                "test_shape_accuracy": split_eval["test"]["shape_accuracy"],
                "test_L_mae_mm": split_eval["test"]["L_mae_mm"],
                "test_W_mae_mm": split_eval["test"]["W_mae_mm"],
                "test_D_mae_mm": split_eval["test"]["D_mae_mm"],
                "test_burial_depth_mae_mm": split_eval["test"]["burial_depth_mae_mm"],
                "test_center_xyz_mae_mm": split_eval["test"]["center_xyz_mae_mm"],
            }
        )
        epoch_rows.extend(result["logs"])

    if selected_test is None or selected_train is None or selected_val is None:
        raise RuntimeError("selected train/val/test metrics were not created")
    compare_rows = read_feature_comparison(args.feature_metrics)
    compare_rows.append(
        {
            "model": candidate,
            "source": "neural_gate",
            "selected": True,
            "split": "test",
            "sample_count": selected_test["sample_count"],
            "total_normalized_mae": selected_test["total_normalized_mae"],
            "L_mae_mm": selected_test["L_mae_mm"],
            "W_mae_mm": selected_test["W_mae_mm"],
            "D_mae_mm": selected_test["D_mae_mm"],
            "burial_depth_mae_mm": selected_test["burial_depth_mae_mm"],
            "center_xyz_mae_mm": selected_test["center_xyz_mae_mm"],
            "shape_accuracy": selected_test["shape_accuracy"],
            "shape_macro_f1": selected_test["shape_macro_f1"],
        }
    )

    write_csv(args.seed_summary, seed_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
    write_csv(args.group_summary, group_summary_rows, GROUP_FIELDS)
    write_csv(args.vs_feature, compare_rows, COMPARE_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.4 internal defect v2_240 neural training gate 摘要",
                f"dataset_id: {args.dataset_id}",
                "model_input: only delta_b/BxByBz, shape=(N,9,201)",
                "supervision_labels: L/W/D, burial_depth, center_xyz, shape_type; labels only used for loss/metrics.",
                "metadata_leakage: false; shape_type/burial_depth_bin/size_bin/aspect_bin/split/sample_id 未作为模型输入。",
                "selection_protocol: validation-only epoch selection and validation-only seed selection; test final only.",
                "seeds: 42, 123, 2026",
                f"selected_seed: {selected['seed']}",
                f"selected_best_epoch: {selected['best_epoch']}",
                f"selected_val_score: {selected['best_score']:.6f}",
                f"train_total_normalized_mae: {float(selected_train['total_normalized_mae']):.6f}",
                f"val_total_normalized_mae: {float(selected_val['total_normalized_mae']):.6f}",
                f"test_total_normalized_mae: {float(selected_test['total_normalized_mae']):.6f}",
                f"test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.3f} / {float(selected_test['W_mae_mm']):.3f} / {float(selected_test['D_mae_mm']):.3f}",
                f"test_burial_depth_mae_mm: {float(selected_test['burial_depth_mae_mm']):.3f}",
                f"test_center_xyz_mae_mm: {float(selected_test['center_xyz_mae_mm']):.3f}",
                f"test_shape_accuracy: {float(selected_test['shape_accuracy']):.6f}",
                f"test_shape_macro_f1: {float(selected_test['shape_macro_f1']):.6f}",
                "checkpoint_saved: false",
                "current_baseline_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""21.7 fixed B2 internal defect benchmark rerun.

固定 21.6 的 B2_feature_fusion_burial_head 结构和训练协议：
delta_b/BxByBz + train-normalized delta_b-derived features，train-only
normalization，validation-only epoch/seed selection，test final only。
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
    normalize_x,
    normalize_y,
    regression_metrics,
    split_indices,
    train_normalization,
    train_target_scaler,
    write_csv,
)
from train_internal_defect_feature_baselines import extract_features, standardize_features
from train_internal_defect_burial_depth_candidates import (
    GROUP_FIELDS,
    METRIC_FIELDS as BASE_METRIC_FIELDS,
    candidate_selection_score,
    group_rows,
    load_reference_metrics,
    metric_row,
    safe_float,
    train_one_candidate,
)
from train_internal_defect_burial_depth_refined_model import SEED_FIELDS, VS_FIELDS, build_vs_rows


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
CANDIDATE = "B2_feature_fusion_burial_head"
SUMMARY = ROOT / "results/summaries/internal_defect_benchmark_rerun_b2_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_seed_summary.csv"
METRICS = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_group_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_vs_reference.csv"
METRIC_FIELDS = ["candidate", "selected", *BASE_METRIC_FIELDS[1:]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed B2 internal defect benchmark rerun.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    return parser.parse_args()


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
    feature_raw, _feature_names = extract_features(dataset.delta_b)
    features, _, _ = standardize_features(feature_raw, splits["train"])
    refs = load_reference_metrics()

    seed_results = [
        train_one_candidate(CANDIDATE, seed, args.epochs, args.batch_size, x, y_norm, y, y_mean, y_std, shape, splits, features)
        for seed in [42, 123, 2026]
    ]
    selected = min(seed_results, key=lambda result: (result["best_score"], result["best_val"].get("burial_depth_mae_mm", 999.0)))

    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []
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
        split_metrics: dict[str, dict[str, Any]] = {}
        for split_name, idx in splits.items():
            reg = regression_metrics(y[idx], result["pred"][idx], y_std.reshape(-1))
            cls = classification_metrics(shape[idx], result["shape_pred"][idx])
            score = candidate_selection_score(reg, cls) if split_name == "val" else ""
            row = metric_row(
                CANDIDATE,
                is_selected,
                True,
                seed,
                split_name,
                idx,
                y,
                result["pred"],
                shape,
                result["shape_pred"],
                y_std.reshape(-1),
                score,
                "fixed B2 formal rerun; validation-only selection; test final only",
            )
            metric_rows.append(row)
            row["selected"] = row.get("selected_candidate")
            split_metrics[split_name] = row
            if is_selected and split_name == "train":
                selected_train = row
            if is_selected and split_name == "val":
                selected_val = row
            if is_selected and split_name == "test":
                selected_test = row
            if is_selected:
                group_summary_rows.extend(
                    group_rows(CANDIDATE, True, seed, split_name, idx, y, result["pred"], shape, result["shape_pred"], y_std.reshape(-1), group_values)
                )
        seed_rows.append(
            {
                "candidate": CANDIDATE,
                "selected_seed": is_selected,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_val_selection_score": result["best_score"],
                "train_total_normalized_mae": split_metrics["train"]["total_normalized_mae"],
                "val_total_normalized_mae": split_metrics["val"]["total_normalized_mae"],
                "test_total_normalized_mae": split_metrics["test"]["total_normalized_mae"],
                "train_burial_depth_mae_mm": split_metrics["train"]["burial_depth_mae_mm"],
                "val_burial_depth_mae_mm": split_metrics["val"]["burial_depth_mae_mm"],
                "test_burial_depth_mae_mm": split_metrics["test"]["burial_depth_mae_mm"],
                "test_L_mae_mm": split_metrics["test"]["L_mae_mm"],
                "test_W_mae_mm": split_metrics["test"]["W_mae_mm"],
                "test_D_mae_mm": split_metrics["test"]["D_mae_mm"],
                "test_center_xyz_mae_mm": split_metrics["test"]["center_xyz_mae_mm"],
                "test_shape_accuracy": split_metrics["test"]["shape_accuracy"],
                "test_shape_macro_f1": split_metrics["test"]["shape_macro_f1"],
            }
        )

    if selected_test is None or selected_train is None or selected_val is None:
        raise RuntimeError("selected train/val/test metrics missing")
    vs_rows = build_vs_rows(CANDIDATE, selected_test, refs)
    for row in vs_rows:
        if row.get("model") == CANDIDATE:
            row["source"] = "21.7_B2_formal_rerun"

    write_csv(args.seed_summary, seed_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_summary_rows, GROUP_FIELDS)
    write_csv(args.vs_reference, vs_rows, VS_FIELDS)

    b0_test = refs["B0_reference_neural"]["test"]
    feature_test = refs["feature_baseline_svr_rbf_C10"]["test"]
    all_seed_burials = [safe_float(row["test_burial_depth_mae_mm"]) for row in seed_rows]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.7 internal defect B2 formal benchmark rerun",
                f"dataset_id: {args.dataset_id}",
                "candidate: B2_feature_fusion_burial_head",
                "input_policy: delta_b/BxByBz plus train-normalized delta_b-derived features only.",
                "forbidden_inputs: true shape_type, burial_bin, size_bin, aspect_bin, split, sample_id.",
                "selection_protocol: validation-only epoch and seed selection; test final only.",
                "seeds: 42, 123, 2026",
                f"selected_seed: {selected['seed']}",
                f"selected_best_epoch: {selected['best_epoch']}",
                f"train_total_normalized_mae: {safe_float(selected_train.get('total_normalized_mae')):.6f}",
                f"val_total_normalized_mae: {safe_float(selected_val.get('total_normalized_mae')):.6f}",
                f"test_total_normalized_mae: {safe_float(selected_test.get('total_normalized_mae')):.6f}",
                f"test_LWD_mae_mm: {safe_float(selected_test.get('L_mae_mm')):.3f} / {safe_float(selected_test.get('W_mae_mm')):.3f} / {safe_float(selected_test.get('D_mae_mm')):.3f}",
                f"test_burial_depth_mae_mm: {safe_float(selected_test.get('burial_depth_mae_mm')):.3f}",
                f"test_center_xyz_mae_mm: {safe_float(selected_test.get('center_xyz_mae_mm')):.3f}",
                f"test_shape_accuracy: {safe_float(selected_test.get('shape_accuracy')):.6f}",
                f"test_shape_macro_f1: {safe_float(selected_test.get('shape_macro_f1')):.6f}",
                f"all_seed_test_burial_depth_mae_mm: {', '.join(f'{value:.3f}' for value in all_seed_burials)}",
                f"burial_vs_21_4_neural_delta_mm: {safe_float(selected_test.get('burial_depth_mae_mm')) - safe_float(b0_test.get('burial_depth_mae_mm')):.3f}",
                f"burial_vs_feature_baseline_delta_mm: {safe_float(selected_test.get('burial_depth_mae_mm')) - safe_float(feature_test.get('burial_depth_mae_mm')):.3f}",
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

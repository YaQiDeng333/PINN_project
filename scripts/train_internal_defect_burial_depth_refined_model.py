#!/usr/bin/env python
"""21.6 selected burial-depth candidate multi-seed run.

仅在 candidate screen 产生 validation-selected 有效候选时执行。训练仍只使用
delta_b/BxByBz 和候选明确允许的 delta_b-derived features；test 只做最终评估。
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
from train_internal_defect_feature_baselines import extract_features, standardize_features
from train_internal_defect_burial_depth_candidates import (
    GROUP_FIELDS,
    METRIC_FIELDS,
    candidate_selection_score,
    group_rows,
    load_reference_metrics,
    metric_row,
    safe_float,
    train_one_candidate,
)


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SCREEN_METRICS = ROOT / "results/metrics/internal_defect_burial_depth_candidate_screen_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_burial_depth_refined_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_burial_depth_refined_seed_summary.csv"
METRICS = ROOT / "results/metrics/internal_defect_burial_depth_refined_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_burial_depth_refined_group_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/internal_defect_burial_depth_refined_vs_reference.csv"


SEED_FIELDS = [
    "candidate",
    "selected_seed",
    "seed",
    "best_epoch",
    "best_val_selection_score",
    "train_total_normalized_mae",
    "val_total_normalized_mae",
    "test_total_normalized_mae",
    "train_burial_depth_mae_mm",
    "val_burial_depth_mae_mm",
    "test_burial_depth_mae_mm",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_center_xyz_mae_mm",
    "test_shape_accuracy",
    "test_shape_macro_f1",
]

VS_FIELDS = [
    "model",
    "source",
    "selected",
    "split",
    "sample_count",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
    "burial_delta_vs_21_4_neural_mm",
    "burial_delta_vs_feature_baseline_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selected 21.6 burial-depth refined model.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--screen-metrics", type=Path, default=SCREEN_METRICS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def selected_candidate(path: Path) -> str:
    rows = read_csv(path)
    selected = sorted({row.get("candidate", "") for row in rows if row.get("selected_candidate") == "True" and row.get("candidate") != "B0_reference_neural"})
    return selected[0] if selected else ""


def write_skip_outputs(args: argparse.Namespace, reason: str) -> None:
    write_csv(args.seed_summary, [], SEED_FIELDS)
    write_csv(args.metrics, [], METRIC_FIELDS)
    write_csv(args.group_summary, [], GROUP_FIELDS)
    write_csv(args.vs_reference, [], VS_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.6 internal defect burial-depth refined multi-seed",
                "status: skipped",
                f"reason: {reason}",
                "current_baseline_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_vs_rows(candidate: str, selected_test: dict[str, Any], refs: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    b0 = refs["B0_reference_neural"]["test"]
    feature = refs["feature_baseline_svr_rbf_C10"]["test"]
    rows: list[dict[str, Any]] = []
    for model, source, ref, selected in [
        ("B0_reference_neural", "21.4_neural_reference", False, b0),
        ("svr_rbf_C10", "21.4_feature_baseline", False, feature),
        (candidate, "21.6_refined_model", True, selected_test),
    ]:
        row = {
            "model": model,
            "source": source,
            "selected": ref,
            "split": "test",
            "sample_count": int(safe_float(selected.get("sample_count", 40))),
            "total_normalized_mae": selected.get("total_normalized_mae", ""),
            "L_mae_mm": selected.get("L_mae_mm", ""),
            "W_mae_mm": selected.get("W_mae_mm", ""),
            "D_mae_mm": selected.get("D_mae_mm", ""),
            "burial_depth_mae_mm": selected.get("burial_depth_mae_mm", ""),
            "center_xyz_mae_mm": selected.get("center_xyz_mae_mm", ""),
            "shape_accuracy": selected.get("shape_accuracy", ""),
            "shape_macro_f1": selected.get("shape_macro_f1", ""),
            "burial_delta_vs_21_4_neural_mm": safe_float(selected.get("burial_depth_mae_mm")) - safe_float(b0.get("burial_depth_mae_mm")),
            "burial_delta_vs_feature_baseline_mm": safe_float(selected.get("burial_depth_mae_mm")) - safe_float(feature.get("burial_depth_mae_mm")),
        }
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    candidate = selected_candidate(args.screen_metrics)
    if not candidate:
        write_skip_outputs(args, "candidate screen did not produce a validation-eligible candidate")
        return 0

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
        train_one_candidate(candidate, seed, args.epochs, args.batch_size, x, y_norm, y, y_mean, y_std, shape, splits, features)
        for seed in [42, 123, 2026]
    ]
    selected = min(seed_results, key=lambda result: (result["best_score"], result["best_val"].get("burial_depth_mae_mm", 999.0)))
    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []
    selected_test: dict[str, Any] | None = None
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
                candidate,
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
                "multi-seed validation-only selection; test final only",
            )
            metric_rows.append(row)
            split_metrics[split_name] = row
            if is_selected and split_name == "test":
                selected_test = row
            if is_selected:
                group_summary_rows.extend(
                    group_rows(candidate, True, seed, split_name, idx, y, result["pred"], shape, result["shape_pred"], y_std.reshape(-1), group_values)
                )
        seed_rows.append(
            {
                "candidate": candidate,
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

    if selected_test is None:
        raise RuntimeError("selected test metrics missing")
    vs_rows = build_vs_rows(candidate, selected_test, refs)
    write_csv(args.seed_summary, seed_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_summary_rows, GROUP_FIELDS)
    write_csv(args.vs_reference, vs_rows, VS_FIELDS)
    b0_test = refs["B0_reference_neural"]["test"]
    feature_test = refs["feature_baseline_svr_rbf_C10"]["test"]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.6 internal defect burial-depth refined multi-seed",
                f"dataset_id: {args.dataset_id}",
                f"selected_candidate_from_screen: {candidate}",
                "seeds: 42, 123, 2026",
                f"selected_seed: {selected['seed']}",
                f"selected_best_epoch: {selected['best_epoch']}",
                "selection_protocol: validation-only seed/model selection; test final only.",
                f"test_total_normalized_mae: {safe_float(selected_test.get('total_normalized_mae')):.6f}",
                f"test_LWD_mae_mm: {safe_float(selected_test.get('L_mae_mm')):.3f} / {safe_float(selected_test.get('W_mae_mm')):.3f} / {safe_float(selected_test.get('D_mae_mm')):.3f}",
                f"test_burial_depth_mae_mm: {safe_float(selected_test.get('burial_depth_mae_mm')):.3f}",
                f"test_center_xyz_mae_mm: {safe_float(selected_test.get('center_xyz_mae_mm')):.3f}",
                f"test_shape_accuracy: {safe_float(selected_test.get('shape_accuracy')):.6f}",
                f"test_shape_macro_f1: {safe_float(selected_test.get('shape_macro_f1')):.6f}",
                f"burial_delta_vs_21_4_neural_mm: {safe_float(selected_test.get('burial_depth_mae_mm')) - safe_float(b0_test.get('burial_depth_mae_mm')):.3f}",
                f"burial_delta_vs_feature_baseline_mm: {safe_float(selected_test.get('burial_depth_mae_mm')) - safe_float(feature_test.get('burial_depth_mae_mm')):.3f}",
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

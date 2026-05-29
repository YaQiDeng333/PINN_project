#!/usr/bin/env python
"""22.1 selected shape-conditioned internal defect model multi-seed run."""

from __future__ import annotations

import argparse
import csv
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
from train_internal_defect_shape_conditioned_candidates import (
    B2_REPLAY,
    METRIC_FIELDS,
    TAIL_FIELDS,
    load_b2_predictions,
    metric_row,
    per_sample_errors,
    safe_float,
    selection_score,
    tail_metrics,
    train_candidate,
)


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SCREEN_METRICS = ROOT / "results/metrics/internal_defect_shape_conditioned_candidate_screen_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_shape_conditioned_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_shape_conditioned_seed_summary.csv"
METRICS = ROOT / "results/metrics/internal_defect_shape_conditioned_metrics.csv"
TAIL_METRICS = ROOT / "results/metrics/internal_defect_shape_conditioned_tail_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_shape_conditioned_group_summary.csv"
VS_B2 = ROOT / "results/metrics/internal_defect_shape_conditioned_vs_b2.csv"


SEED_FIELDS = [
    "candidate",
    "selected_seed",
    "seed",
    "best_epoch",
    "best_val_selection_score",
    "train_total_normalized_mae",
    "val_total_normalized_mae",
    "test_total_normalized_mae",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_burial_depth_mae_mm",
    "test_center_xyz_component_mae_mm",
    "test_center_xyz_error_p95_mm",
    "test_center_xyz_error_max_mm",
    "test_burial_depth_error_p95_mm",
    "test_burial_depth_error_max_mm",
    "test_catastrophic_failure_count",
    "test_geometry_branch_failure_count",
    "test_shape_accuracy",
    "test_shape_macro_f1",
]

GROUP_FIELDS = [
    "candidate",
    "selected_seed",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_component_mae_mm",
    "center_xyz_error_p95_mm",
    "center_xyz_error_max_mm",
    "catastrophic_failure_count",
    "geometry_branch_failure_count",
    "shape_accuracy",
    "shape_macro_f1",
]

VS_FIELDS = [
    "metric",
    "b2_reference",
    "selected_candidate",
    "delta_selected_minus_b2",
    "improvement_direction",
    "passes_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selected 22.1 shape-conditioned model.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--screen-metrics", type=Path, default=SCREEN_METRICS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail-metrics", type=Path, default=TAIL_METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--vs-b2", type=Path, default=VS_B2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def selected_candidate(path: Path) -> str:
    selected = sorted({row["candidate"] for row in read_csv(path) if row.get("selected_candidate") == "True" and row.get("candidate_role") == "official_candidate"})
    return selected[0] if selected else ""


def write_skip(args: argparse.Namespace, reason: str) -> None:
    write_csv(args.seed_summary, [], SEED_FIELDS)
    write_csv(args.metrics, [], METRIC_FIELDS)
    write_csv(args.tail_metrics, [], TAIL_FIELDS)
    write_csv(args.group_summary, [], GROUP_FIELDS)
    write_csv(args.vs_b2, [], VS_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(["22.1 selected shape-conditioned internal model multi-seed", "status: skipped", f"reason: {reason}", "current_baseline_update: false"]) + "\n",
        encoding="utf-8",
    )


def group_rows(
    candidate: str,
    selected_seed: int,
    split: str,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    y_std: np.ndarray,
    group_field: str,
    values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(set(values[idx].tolist())):
        sub = idx[values[idx] == value]
        reg = regression_metrics(y_true[sub], y_pred[sub], y_std)
        cls = classification_metrics(shape_true[sub], shape_pred[sub])
        errors = per_sample_errors(y_true[sub], y_pred[sub], shape_true[sub], shape_pred[sub], y_std)
        rows.append(
            {
                "candidate": candidate,
                "selected_seed": selected_seed,
                "split": split,
                "group_field": group_field,
                "group_value": value,
                "sample_count": int(sub.size),
                "total_normalized_mae": reg["total_normalized_mae"],
                "L_mae_mm": reg["L_mae_mm"],
                "W_mae_mm": reg["W_mae_mm"],
                "D_mae_mm": reg["D_mae_mm"],
                "burial_depth_mae_mm": reg["burial_depth_mae_mm"],
                "center_xyz_component_mae_mm": reg["center_xyz_mae_mm"],
                "center_xyz_error_p95_mm": float(np.percentile(errors["center"], 95)),
                "center_xyz_error_max_mm": float(np.max(errors["center"])),
                "catastrophic_failure_count": int(np.sum(errors["catastrophic"])),
                "geometry_branch_failure_count": int(np.sum(errors["geometry_branch"])),
                "shape_accuracy": cls["shape_accuracy"],
                "shape_macro_f1": cls["shape_macro_f1"],
            }
        )
    return rows


def b2_test_rows(dataset: Any, splits: dict[str, np.ndarray], y_std: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    b2_pred, b2_shape_pred = load_b2_predictions(B2_REPLAY, dataset)
    idx = splits["test"]
    metric = metric_row("T0_B2_reference", False, False, 2026, "test", idx, dataset.y_regression, b2_pred, dataset.shape_label, b2_shape_pred, y_std, "", "", "B2 reference")
    tail = tail_metrics("T0_B2_reference", False, False, 2026, "test", idx, dataset.y_regression, b2_pred, dataset.shape_label, b2_shape_pred, y_std)
    return metric, tail


def build_vs_rows(b2_metric: dict[str, Any], b2_tail: dict[str, Any], selected_metric: dict[str, Any], selected_tail: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("total_normalized_mae", b2_metric["total_normalized_mae"], selected_metric["total_normalized_mae"], "lower", safe_float(selected_metric["total_normalized_mae"]) <= safe_float(b2_metric["total_normalized_mae"]) * 1.10),
        ("L_mae_mm", b2_metric["L_mae_mm"], selected_metric["L_mae_mm"], "lower", safe_float(selected_metric["L_mae_mm"]) <= safe_float(b2_metric["L_mae_mm"]) * 1.10),
        ("W_mae_mm", b2_metric["W_mae_mm"], selected_metric["W_mae_mm"], "lower", safe_float(selected_metric["W_mae_mm"]) <= safe_float(b2_metric["W_mae_mm"]) * 1.10),
        ("D_mae_mm", b2_metric["D_mae_mm"], selected_metric["D_mae_mm"], "lower", safe_float(selected_metric["D_mae_mm"]) <= safe_float(b2_metric["D_mae_mm"]) * 1.10),
        ("burial_depth_mae_mm", b2_metric["burial_depth_mae_mm"], selected_metric["burial_depth_mae_mm"], "lower", safe_float(selected_metric["burial_depth_mae_mm"]) <= safe_float(b2_metric["burial_depth_mae_mm"]) * 1.10),
        ("center_xyz_error_p95_mm", b2_tail["center_xyz_error_p95_mm"], selected_tail["center_xyz_error_p95_mm"], "lower", safe_float(selected_tail["center_xyz_error_p95_mm"]) < safe_float(b2_tail["center_xyz_error_p95_mm"])),
        ("center_xyz_error_max_mm", b2_tail["center_xyz_error_max_mm"], selected_tail["center_xyz_error_max_mm"], "lower", safe_float(selected_tail["center_xyz_error_max_mm"]) < safe_float(b2_tail["center_xyz_error_max_mm"])),
        ("burial_depth_error_p95_mm", b2_tail["burial_depth_error_p95_mm"], selected_tail["burial_depth_error_p95_mm"], "lower", safe_float(selected_tail["burial_depth_error_p95_mm"]) < safe_float(b2_tail["burial_depth_error_p95_mm"])),
        ("catastrophic_failure_count", b2_tail["catastrophic_failure_count"], selected_tail["catastrophic_failure_count"], "lower", safe_float(selected_tail["catastrophic_failure_count"]) < safe_float(b2_tail["catastrophic_failure_count"])),
        ("geometry_branch_failure_count", b2_tail["geometry_branch_failure_count"], selected_tail["geometry_branch_failure_count"], "lower", safe_float(selected_tail["geometry_branch_failure_count"]) < safe_float(b2_tail["geometry_branch_failure_count"])),
        ("shape_accuracy", b2_metric["shape_accuracy"], selected_metric["shape_accuracy"], "higher", safe_float(selected_metric["shape_accuracy"]) >= safe_float(b2_metric["shape_accuracy"]) - 0.05),
    ]
    return [
        {
            "metric": name,
            "b2_reference": b2,
            "selected_candidate": val,
            "delta_selected_minus_b2": safe_float(val) - safe_float(b2),
            "improvement_direction": direction,
            "passes_gate": passed,
        }
        for name, b2, val, direction, passed in specs
    ]


def main() -> int:
    args = parse_args()
    candidate = selected_candidate(args.screen_metrics)
    if not candidate:
        write_skip(args, "candidate screen did not produce a validation-eligible official candidate")
        return 0
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    x_mean, x_std = train_normalization(dataset.x_channels, splits["train"])
    x = normalize_x(dataset.x_channels, x_mean, x_std)
    y = dataset.y_regression
    y_mean, y_std = train_target_scaler(y, splits["train"])
    y_norm = normalize_y(y, y_mean, y_std)
    features_raw, _ = extract_features(dataset.delta_b)
    features, _, _ = standardize_features(features_raw, splits["train"])
    results = [train_candidate(candidate, seed, args.epochs, args.batch_size, x, features, y_norm, y, y_mean, y_std, dataset.shape_label, splits) for seed in [42, 123, 2026]]
    selected = min(results, key=lambda row: row["best_score"])
    metric_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    selected_metric: dict[str, Any] | None = None
    selected_tail: dict[str, Any] | None = None
    group_summary: list[dict[str, Any]] = []
    group_values = {
        "shape_type": dataset.shape_type,
        "burial_depth_level": dataset.burial_depth_level,
        "size_level": dataset.size_level,
        "aspect_bin": dataset.aspect_bin,
    }
    for result in results:
        seed = int(result["seed"])
        is_selected = seed == int(selected["seed"])
        split_metric: dict[str, dict[str, Any]] = {}
        split_tail: dict[str, dict[str, Any]] = {}
        for split, idx in splits.items():
            metric = metric_row(candidate, is_selected, True, seed, split, idx, y, result["pred"], dataset.shape_label, result["shape_pred"], y_std.reshape(-1), result["best_score"] if split == "val" else "", result["best_epoch"], "multi-seed validation-only selection; test final only")
            tail = tail_metrics(candidate, is_selected, True, seed, split, idx, y, result["pred"], dataset.shape_label, result["shape_pred"], y_std.reshape(-1))
            metric_rows.append(metric)
            tail_rows.append(tail)
            split_metric[split] = metric
            split_tail[split] = tail
            if is_selected and split == "test":
                selected_metric = metric
                selected_tail = tail
            if is_selected:
                for field, values in group_values.items():
                    group_summary.extend(group_rows(candidate, seed, split, idx, y, result["pred"], dataset.shape_label, result["shape_pred"], y_std.reshape(-1), field, values))
        seed_rows.append(
            {
                "candidate": candidate,
                "selected_seed": is_selected,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_val_selection_score": result["best_score"],
                "train_total_normalized_mae": split_metric["train"]["total_normalized_mae"],
                "val_total_normalized_mae": split_metric["val"]["total_normalized_mae"],
                "test_total_normalized_mae": split_metric["test"]["total_normalized_mae"],
                "test_L_mae_mm": split_metric["test"]["L_mae_mm"],
                "test_W_mae_mm": split_metric["test"]["W_mae_mm"],
                "test_D_mae_mm": split_metric["test"]["D_mae_mm"],
                "test_burial_depth_mae_mm": split_metric["test"]["burial_depth_mae_mm"],
                "test_center_xyz_component_mae_mm": split_metric["test"]["center_xyz_component_mae_mm"],
                "test_center_xyz_error_p95_mm": split_tail["test"]["center_xyz_error_p95_mm"],
                "test_center_xyz_error_max_mm": split_tail["test"]["center_xyz_error_max_mm"],
                "test_burial_depth_error_p95_mm": split_tail["test"]["burial_depth_error_p95_mm"],
                "test_burial_depth_error_max_mm": split_tail["test"]["burial_depth_error_max_mm"],
                "test_catastrophic_failure_count": split_tail["test"]["catastrophic_failure_count"],
                "test_geometry_branch_failure_count": split_tail["test"]["geometry_branch_failure_count"],
                "test_shape_accuracy": split_metric["test"]["shape_accuracy"],
                "test_shape_macro_f1": split_metric["test"]["shape_macro_f1"],
            }
        )
    if selected_metric is None or selected_tail is None:
        raise RuntimeError("selected test metrics missing")
    b2_metric, b2_tail = b2_test_rows(dataset, splits, y_std.reshape(-1))
    vs_rows = build_vs_rows(b2_metric, b2_tail, selected_metric, selected_tail)
    write_csv(args.seed_summary, seed_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.tail_metrics, tail_rows, TAIL_FIELDS)
    write_csv(args.group_summary, group_summary, GROUP_FIELDS)
    write_csv(args.vs_b2, vs_rows, VS_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "22.1 selected shape-conditioned internal model multi-seed",
                f"dataset_id: {args.dataset_id}",
                f"selected_candidate: {candidate}",
                "seeds: 42, 123, 2026",
                f"selected_seed: {selected['seed']}",
                f"selected_best_epoch: {selected['best_epoch']}",
                "selection_protocol: validation-only seed/model selection; test final only.",
                f"test_total_normalized_mae: {safe_float(selected_metric['total_normalized_mae']):.6f}",
                f"test_LWD_mae_mm: {safe_float(selected_metric['L_mae_mm']):.3f} / {safe_float(selected_metric['W_mae_mm']):.3f} / {safe_float(selected_metric['D_mae_mm']):.3f}",
                f"test_burial_depth_mae_mm: {safe_float(selected_metric['burial_depth_mae_mm']):.3f}",
                f"test_center_xyz_component_mae_mm: {safe_float(selected_metric['center_xyz_component_mae_mm']):.3f}",
                f"test_shape_accuracy_f1: {safe_float(selected_metric['shape_accuracy']):.6f} / {safe_float(selected_metric['shape_macro_f1']):.6f}",
                f"test_center_p95_max_mm: {safe_float(selected_tail['center_xyz_error_p95_mm']):.3f} / {safe_float(selected_tail['center_xyz_error_max_mm']):.3f}",
                f"test_burial_p95_max_mm: {safe_float(selected_tail['burial_depth_error_p95_mm']):.3f} / {safe_float(selected_tail['burial_depth_error_max_mm']):.3f}",
                f"test_catastrophic_failure_count: {selected_tail['catastrophic_failure_count']}",
                f"test_geometry_branch_failure_count: {selected_tail['geometry_branch_failure_count']}",
                f"vs_B2_catastrophic_delta: {safe_float(selected_tail['catastrophic_failure_count']) - safe_float(b2_tail['catastrophic_failure_count']):.0f}",
                f"vs_B2_geometry_branch_delta: {safe_float(selected_tail['geometry_branch_failure_count']) - safe_float(b2_tail['geometry_branch_failure_count']):.0f}",
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

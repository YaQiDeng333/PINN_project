#!/usr/bin/env python
"""22.5 selected freeze-shape tail-regression multi-seed training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from internal_defect_hardcase_utils import METRIC_FIELDS, PREDICTION_FIELDS, TAIL_FIELDS, metric_row, prediction_rows, safe_float, tail_row
from load_internal_defect_pilot_dataset import ROOT, classification_metrics, regression_metrics, write_csv
from train_internal_defect_freeze_shape_tail_candidates import (
    DATASET_ID,
    FORMAL_CANDIDATES,
    load_frozen_b2_context,
    selection_score,
    train_tail_candidate,
)


SCREEN_METRICS = ROOT / "results/metrics/internal_defect_freeze_shape_candidate_screen_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_freeze_shape_tail_training_summary.txt"
SEEDS = ROOT / "results/metrics/internal_defect_freeze_shape_tail_seed_summary.csv"
METRICS = ROOT / "results/metrics/internal_defect_freeze_shape_tail_metrics.csv"
TAIL = ROOT / "results/metrics/internal_defect_freeze_shape_tail_tail_metrics.csv"
GROUP = ROOT / "results/metrics/internal_defect_freeze_shape_tail_group_summary.csv"
VS_REF = ROOT / "results/metrics/internal_defect_freeze_shape_tail_vs_reference.csv"
SELECTED_PRED = ROOT / "results/metrics/internal_defect_freeze_shape_tail_selected_predictions.csv"
REFERENCE_METRICS = ROOT / "results/metrics/internal_defect_freeze_shape_reference_metrics.csv"
REFERENCE_TAIL = ROOT / "results/metrics/internal_defect_freeze_shape_reference_tail_metrics.csv"

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
    "test_shape_accuracy",
    "test_shape_macro_f1",
    "test_center_p90_mm",
    "test_center_p95_mm",
    "test_center_max_mm",
    "test_burial_p90_mm",
    "test_burial_p95_mm",
    "test_burial_max_mm",
    "test_catastrophic_failure_count",
    "test_catastrophic_failure_rate",
    "test_geometry_branch_failure_count",
    "test_geometry_branch_failure_rate",
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
    "shape_accuracy",
    "shape_macro_f1",
    "center_p95_mm",
    "center_max_mm",
    "burial_p95_mm",
    "burial_max_mm",
    "catastrophic_failure_count",
    "geometry_branch_failure_count",
]

VS_FIELDS = ["metric", "B2_reference", "H2_reference", "freeze_shape_candidate", "delta_vs_B2", "delta_vs_H2", "passes_gate", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selected 22.5 freeze-shape model.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--screen-metrics", type=Path, default=SCREEN_METRICS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEEDS)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail", type=Path, default=TAIL)
    parser.add_argument("--group-summary", type=Path, default=GROUP)
    parser.add_argument("--vs-reference", type=Path, default=VS_REF)
    parser.add_argument("--selected-predictions", type=Path, default=SELECTED_PRED)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def selected_candidate(path: Path) -> str:
    for row in read_csv(path):
        if row.get("selected_model") == "True" and row.get("candidate_role") == "official_candidate" and row.get("model") in FORMAL_CANDIDATES:
            return row["model"]
    return ""


def group_rows(candidate: str, selected_seed: int, ctx: Any, pred: np.ndarray, shape_pred: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # These metadata fields are only for reporting; they are never model inputs.
    groups = {
        "shape_type": ctx.dataset.shape_type,
        "burial_depth_level": ctx.dataset.burial_depth_level,
        "size_level": ctx.dataset.size_level,
        "aspect_bin": ctx.dataset.aspect_bin,
        "row_origin": ctx.dataset.row_origin,
    }
    y_std = ctx.y_std.reshape(-1)
    for split, split_idx in ctx.splits.items():
        for field, values in groups.items():
            for value in sorted(set(values[split_idx].tolist())):
                idx = split_idx[values[split_idx] == value]
                reg = regression_metrics(ctx.y[idx], pred[idx], y_std)
                cls = classification_metrics(ctx.shape[idx], shape_pred[idx])
                tail = tail_row(candidate, True, selected_seed, split, "all", idx, ctx.y, pred, ctx.shape, shape_pred, y_std)
                rows.append(
                    {
                        "candidate": candidate,
                        "selected_seed": selected_seed,
                        "split": split,
                        "group_field": field,
                        "group_value": value,
                        "sample_count": int(idx.size),
                        "total_normalized_mae": reg["total_normalized_mae"],
                        "L_mae_mm": reg["L_mae_mm"],
                        "W_mae_mm": reg["W_mae_mm"],
                        "D_mae_mm": reg["D_mae_mm"],
                        "burial_depth_mae_mm": reg["burial_depth_mae_mm"],
                        "center_xyz_component_mae_mm": reg["center_xyz_mae_mm"],
                        "shape_accuracy": cls["shape_accuracy"],
                        "shape_macro_f1": cls["shape_macro_f1"],
                        "center_p95_mm": tail["center_xyz_error_p95_mm"],
                        "center_max_mm": tail["center_xyz_error_max_mm"],
                        "burial_p95_mm": tail["burial_depth_error_p95_mm"],
                        "burial_max_mm": tail["burial_depth_error_max_mm"],
                        "catastrophic_failure_count": tail["catastrophic_failure_count"],
                        "geometry_branch_failure_count": tail["geometry_branch_failure_count"],
                    }
                )
    return rows


def reference_lookup(path: Path, model_name: str) -> dict[str, dict[str, str]]:
    rows = [row for row in read_csv(path) if row.get("model") == model_name and row.get("split") == "test" and row.get("subset") == "all"]
    return rows[0] if rows else {}


def build_vs(candidate_metric: dict[str, Any], candidate_tail: dict[str, Any]) -> list[dict[str, Any]]:
    b2_m = reference_lookup(REFERENCE_METRICS, "B2_feature_fusion_burial_head_reference")
    h2_m = reference_lookup(REFERENCE_METRICS, "H2_B2_hardcase_tail_weighted_reference")
    b2_t = reference_lookup(REFERENCE_TAIL, "B2_feature_fusion_burial_head_reference")
    h2_t = reference_lookup(REFERENCE_TAIL, "H2_B2_hardcase_tail_weighted_reference")
    specs = [
        ("total_normalized_mae", b2_m.get("total_normalized_mae"), h2_m.get("total_normalized_mae"), candidate_metric["total_normalized_mae"], "lower", "mean total guard"),
        ("L_mae_mm", b2_m.get("L_mae_mm"), h2_m.get("L_mae_mm"), candidate_metric["L_mae_mm"], "lower", "L guard"),
        ("W_mae_mm", b2_m.get("W_mae_mm"), h2_m.get("W_mae_mm"), candidate_metric["W_mae_mm"], "lower", "W guard"),
        ("D_mae_mm", b2_m.get("D_mae_mm"), h2_m.get("D_mae_mm"), candidate_metric["D_mae_mm"], "lower", "D guard"),
        ("burial_depth_mae_mm", b2_m.get("burial_depth_mae_mm"), h2_m.get("burial_depth_mae_mm"), candidate_metric["burial_depth_mae_mm"], "lower", "burial mean guard"),
        ("center_xyz_component_mae_mm", b2_m.get("center_xyz_component_mae_mm"), h2_m.get("center_xyz_component_mae_mm"), candidate_metric["center_xyz_component_mae_mm"], "lower", "center mean guard"),
        ("shape_macro_f1", b2_m.get("shape_macro_f1"), h2_m.get("shape_macro_f1"), candidate_metric["shape_macro_f1"], "higher", "shape branch must be preserved"),
        ("center_p95_mm", b2_t.get("center_xyz_error_p95_mm"), h2_t.get("center_xyz_error_p95_mm"), candidate_tail["center_xyz_error_p95_mm"], "lower", "center p95 should improve"),
        ("center_max_mm", b2_t.get("center_xyz_error_max_mm"), h2_t.get("center_xyz_error_max_mm"), candidate_tail["center_xyz_error_max_mm"], "lower", "center max should improve"),
        ("burial_p95_mm", b2_t.get("burial_depth_error_p95_mm"), h2_t.get("burial_depth_error_p95_mm"), candidate_tail["burial_depth_error_p95_mm"], "lower", "burial p95 should not regress"),
        ("burial_max_mm", b2_t.get("burial_depth_error_max_mm"), h2_t.get("burial_depth_error_max_mm"), candidate_tail["burial_depth_error_max_mm"], "lower", "burial max should not regress"),
        ("catastrophic_failure_count", b2_t.get("catastrophic_failure_count"), h2_t.get("catastrophic_failure_count"), candidate_tail["catastrophic_failure_count"], "lower", "catastrophic below both references"),
        ("geometry_branch_failure_count", b2_t.get("geometry_branch_failure_count"), h2_t.get("geometry_branch_failure_count"), candidate_tail["geometry_branch_failure_count"], "lower", "geometry branch below both references"),
    ]
    rows: list[dict[str, Any]] = []
    for metric, b2, h2, val, direction, notes in specs:
        b2f = safe_float(b2)
        h2f = safe_float(h2)
        vf = safe_float(val)
        if direction == "higher":
            passed = vf >= b2f - 0.03 and vf >= h2f
        elif metric == "catastrophic_failure_count":
            passed = vf < b2f and vf < h2f and vf <= 3
        elif metric == "geometry_branch_failure_count":
            passed = vf < b2f and vf < h2f
        else:
            passed = vf <= b2f and vf <= h2f
        rows.append(
            {
                "metric": metric,
                "B2_reference": b2,
                "H2_reference": h2,
                "freeze_shape_candidate": val,
                "delta_vs_B2": vf - b2f,
                "delta_vs_H2": vf - h2f,
                "passes_gate": passed,
                "notes": notes,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    candidate = selected_candidate(args.screen_metrics)
    if not candidate:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text("22.5 freeze-shape tail model skipped\nreason: candidate screen produced no formal selected candidate\ncurrent_baseline_update: false\n", encoding="utf-8")
        write_csv(args.seed_summary, [], SEED_FIELDS)
        write_csv(args.metrics, [], METRIC_FIELDS)
        write_csv(args.tail, [], TAIL_FIELDS)
        write_csv(args.group_summary, [], GROUP_FIELDS)
        write_csv(args.vs_reference, [], VS_FIELDS)
        return 0
    ctx = load_frozen_b2_context(args.dataset_id)
    results = [train_tail_candidate(candidate, seed, args.epochs, args.batch_size, ctx) for seed in [42, 123, 2026]]
    selected = min(
        results,
        key=lambda r: selection_score(
            metric_row(candidate, False, r["seed"], "val", "all", ctx.splits["val"], ctx.y, r["pred"], ctx.shape, r["shape_pred"], ctx.y_std.reshape(-1)),
            tail_row(candidate, False, r["seed"], "val", "all", ctx.splits["val"], ctx.y, r["pred"], ctx.shape, r["shape_pred"], ctx.y_std.reshape(-1)),
        ),
    )
    metric_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    selected_metric: dict[str, Any] | None = None
    selected_tail: dict[str, Any] | None = None
    for result in results:
        is_selected = int(result["seed"]) == int(selected["seed"])
        split_metrics: dict[str, dict[str, Any]] = {}
        split_tails: dict[str, dict[str, Any]] = {}
        for split, idx in ctx.splits.items():
            metric = metric_row(candidate, is_selected, result["seed"], split, "all", idx, ctx.y, result["pred"], ctx.shape, result["shape_pred"], ctx.y_std.reshape(-1), result["best_score"] if split == "val" else "", result["best_epoch"])
            tail = tail_row(candidate, is_selected, result["seed"], split, "all", idx, ctx.y, result["pred"], ctx.shape, result["shape_pred"], ctx.y_std.reshape(-1))
            metric_rows.append(metric)
            tail_rows.append(tail)
            split_metrics[split] = metric
            split_tails[split] = tail
            if is_selected and split == "test":
                selected_metric = metric
                selected_tail = tail
        seed_rows.append(
            {
                "candidate": candidate,
                "selected_seed": is_selected,
                "seed": result["seed"],
                "best_epoch": result["best_epoch"],
                "best_val_selection_score": result["best_score"],
                "train_total_normalized_mae": split_metrics["train"]["total_normalized_mae"],
                "val_total_normalized_mae": split_metrics["val"]["total_normalized_mae"],
                "test_total_normalized_mae": split_metrics["test"]["total_normalized_mae"],
                "test_L_mae_mm": split_metrics["test"]["L_mae_mm"],
                "test_W_mae_mm": split_metrics["test"]["W_mae_mm"],
                "test_D_mae_mm": split_metrics["test"]["D_mae_mm"],
                "test_burial_depth_mae_mm": split_metrics["test"]["burial_depth_mae_mm"],
                "test_center_xyz_component_mae_mm": split_metrics["test"]["center_xyz_component_mae_mm"],
                "test_shape_accuracy": split_metrics["test"]["shape_accuracy"],
                "test_shape_macro_f1": split_metrics["test"]["shape_macro_f1"],
                "test_center_p90_mm": split_tails["test"]["center_xyz_error_p90_mm"],
                "test_center_p95_mm": split_tails["test"]["center_xyz_error_p95_mm"],
                "test_center_max_mm": split_tails["test"]["center_xyz_error_max_mm"],
                "test_burial_p90_mm": split_tails["test"]["burial_depth_error_p90_mm"],
                "test_burial_p95_mm": split_tails["test"]["burial_depth_error_p95_mm"],
                "test_burial_max_mm": split_tails["test"]["burial_depth_error_max_mm"],
                "test_catastrophic_failure_count": split_tails["test"]["catastrophic_failure_count"],
                "test_catastrophic_failure_rate": split_tails["test"]["catastrophic_failure_rate"],
                "test_geometry_branch_failure_count": split_tails["test"]["geometry_branch_failure_count"],
                "test_geometry_branch_failure_rate": split_tails["test"]["geometry_branch_failure_rate"],
            }
        )
    if selected_metric is None or selected_tail is None:
        raise RuntimeError("selected test metrics missing")
    group = group_rows(candidate, int(selected["seed"]), ctx, selected["pred"], selected["shape_pred"])
    write_csv(args.seed_summary, seed_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.tail, tail_rows, TAIL_FIELDS)
    write_csv(args.group_summary, group, GROUP_FIELDS)
    write_csv(args.vs_reference, build_vs(selected_metric, selected_tail), VS_FIELDS)
    write_csv(args.selected_predictions, prediction_rows(candidate, selected["seed"], ctx.dataset, selected["pred"], selected["shape_pred"], ctx.y_std.reshape(-1)), PREDICTION_FIELDS)
    lines = [
        "22.5 freeze-shape tail-regression multi-seed training summary",
        f"dataset_id: {args.dataset_id}",
        f"selected_candidate: {candidate}",
        f"selected_seed: {selected['seed']}",
        f"selected_best_epoch: {selected['best_epoch']}",
        "freeze_policy: B2 encoder/trunk/shape outputs are frozen; only center/burial tail residual heads are trained.",
        "input_policy: delta_b/BxByBz-derived frozen B2 latent/logits/predictions and delta_b-derived feature latent only; no true shape/split/sample_id metadata as formal input.",
        "selection_protocol: validation-only seed selection; test final only.",
        f"test_total_normalized_mae: {safe_float(selected_metric['total_normalized_mae']):.6f}",
        f"test_LWD_mae_mm: {safe_float(selected_metric['L_mae_mm']):.3f} / {safe_float(selected_metric['W_mae_mm']):.3f} / {safe_float(selected_metric['D_mae_mm']):.3f}",
        f"test_burial_depth_mae_mm: {safe_float(selected_metric['burial_depth_mae_mm']):.3f}",
        f"test_center_xyz_component_mae_mm: {safe_float(selected_metric['center_xyz_component_mae_mm']):.3f}",
        f"test_shape_accuracy_f1: {safe_float(selected_metric['shape_accuracy']):.6f} / {safe_float(selected_metric['shape_macro_f1']):.6f}",
        f"test_center_p90_p95_max_mm: {safe_float(selected_tail['center_xyz_error_p90_mm']):.3f} / {safe_float(selected_tail['center_xyz_error_p95_mm']):.3f} / {safe_float(selected_tail['center_xyz_error_max_mm']):.3f}",
        f"test_burial_p90_p95_max_mm: {safe_float(selected_tail['burial_depth_error_p90_mm']):.3f} / {safe_float(selected_tail['burial_depth_error_p95_mm']):.3f} / {safe_float(selected_tail['burial_depth_error_max_mm']):.3f}",
        f"test_catastrophic_failure_count_rate: {selected_tail['catastrophic_failure_count']} / {safe_float(selected_tail['catastrophic_failure_rate']):.6f}",
        f"test_geometry_branch_failure_count_rate: {selected_tail['geometry_branch_failure_count']} / {safe_float(selected_tail['geometry_branch_failure_rate']):.6f}",
        "checkpoint_saved: false",
        "npz_written: false",
        "current_baseline_update: false",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

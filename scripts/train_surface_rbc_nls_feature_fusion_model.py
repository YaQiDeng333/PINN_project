#!/usr/bin/env python
"""Multi-seed run for the validation-selected surface RBC NLS-lite fusion model."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_surface_rbc_nls_feature_fusion_dataset import build_inputs
from load_true_3d_rbc_pilot_dataset import ROOT, V3_240_DATASET_ID, check_no_overwrite, write_csv
from train_surface_rbc_nls_feature_fusion_candidates import (
    REF_20_77,
    REF_20_81,
    REF_20_85,
    REF_24_1,
    CandidateConfig,
    aggregate_eval_rows,
    aggregate_selection_score,
    candidate_configs,
    evaluate_subset,
    predict_raw_for_indices,
    row_from_aggregate,
    train_candidate,
)


SCREEN_SUMMARY = ROOT / "results/summaries/surface_rbc_nls_feature_fusion_candidate_screen_summary.txt"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_feature_fusion_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_seed_summary.csv"
METRICS = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_group_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_vs_reference.csv"

SEED_FIELDS = [
    "candidate",
    "seed",
    "selected_seed",
    "feature_count",
    "best_epoch",
    "best_val_epoch_score",
    "validation_selection_score",
    "min_train_normalized_param_mae",
    "train_total_mae",
    "val_total_mae",
    "test_total_mae",
    "train_wMAE",
    "val_wMAE",
    "test_wMAE",
    "train_profile_depth_rmse_m",
    "val_profile_depth_rmse_m",
    "test_profile_depth_rmse_m",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_wLD_abs_error",
    "test_wWD_abs_error",
    "test_wLW_abs_error",
    "test_er_like_profile_error",
    "test_projected_mask_iou",
    "test_projected_mask_dice",
    "test_final_only",
]

METRIC_FIELDS = [
    "candidate",
    "seed",
    "selected_seed",
    "split",
    "sample_count",
    "test_final_only",
    "normalized_param_mae",
    "dimension_mae_norm",
    "curvature_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "wMAE",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_iou",
    "projected_mask_dice",
    "notes",
]

GROUP_FIELDS = [
    "candidate",
    "seed",
    "selected_seed",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "normalized_param_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "wMAE",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_iou",
    "projected_mask_dice",
]

VS_FIELDS = ["metric", "reference_label", "current_label", "reference_value", "current_value", "delta", "improved", "direction", "notes"]


def selected_candidate_from_summary(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    selected = re.search(r"^selected_candidate:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not selected:
        raise RuntimeError(f"selected_candidate not found in {path}")
    eligible = re.search(r"^eligible_for_multiseed:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if eligible and eligible.group(1).strip().lower() != "true":
        raise RuntimeError(f"Stage C gate is false; not running Stage D: eligible_for_multiseed={eligible.group(1).strip()}")
    return selected.group(1).strip()


def config_by_name(name: str) -> CandidateConfig:
    for config in candidate_configs():
        if config.name == name:
            return config
    raise RuntimeError(f"unknown candidate selected by Stage C: {name}")


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_row_from_screen_row(row: dict[str, Any], selected_seed: bool) -> dict[str, Any]:
    return {
        "candidate": row["candidate"],
        "seed": row["seed"],
        "selected_seed": selected_seed,
        "split": row["split"],
        "sample_count": row["sample_count"],
        "test_final_only": row["test_final_only"],
        "normalized_param_mae": row["normalized_param_mae"],
        "dimension_mae_norm": row["dimension_mae_norm"],
        "curvature_mae_norm": row["curvature_mae_norm"],
        "L_mae_mm": row["L_mae_mm"],
        "W_mae_mm": row["W_mae_mm"],
        "D_mae_mm": row["D_mae_mm"],
        "wLD_abs_error": row["wLD_abs_error"],
        "wWD_abs_error": row["wWD_abs_error"],
        "wLW_abs_error": row["wLW_abs_error"],
        "wMAE": row["wMAE"],
        "profile_depth_rmse_m": row["profile_depth_rmse_m"],
        "er_like_profile_error": row["er_like_profile_error"],
        "projected_mask_iou": row["projected_mask_iou"],
        "projected_mask_dice": row["projected_mask_dice"],
        "notes": row["notes"],
    }


def finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [f(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def group_rows(candidate: str, seed: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        for field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
            for value in sorted({str(row[field]) for row in split_rows}):
                subset = [row for row in split_rows if str(row[field]) == value]
                if not subset:
                    continue
                out.append(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "selected_seed": True,
                        "split": split,
                        "group_field": field,
                        "group_value": value,
                        "sample_count": len(subset),
                        "normalized_param_mae": finite_mean(subset, "normalized_param_mae_mean"),
                        "L_mae_mm": finite_mean(subset, "L_mae_mm"),
                        "W_mae_mm": finite_mean(subset, "W_mae_mm"),
                        "D_mae_mm": finite_mean(subset, "D_mae_mm"),
                        "wLD_abs_error": finite_mean(subset, "wLD_abs_error"),
                        "wWD_abs_error": finite_mean(subset, "wWD_abs_error"),
                        "wLW_abs_error": finite_mean(subset, "wLW_abs_error"),
                        "wMAE": finite_mean(subset, "curvature_mae_mean"),
                        "profile_depth_rmse_m": finite_mean(subset, "profile_depth_rmse_m"),
                        "er_like_profile_error": finite_mean(subset, "er_like_profile_error"),
                        "projected_mask_iou": finite_mean(subset, "projected_mask_iou"),
                        "projected_mask_dice": finite_mean(subset, "projected_mask_dice"),
                    }
                )
    return out


def comparison_rows(selected_test: dict[str, Any]) -> list[dict[str, Any]]:
    references = [REF_20_85, REF_20_77, REF_24_1, REF_20_81]
    metric_map = [
        ("total_normalized_mae", "normalized_param_mae", "total", "lower"),
        ("L_mae_mm", "L_mae_mm", "L_mm", "lower"),
        ("W_mae_mm", "W_mae_mm", "W_mm", "lower"),
        ("D_mae_mm", "D_mae_mm", "D_mm", "lower"),
        ("wMAE", "wMAE", "wMAE", "lower"),
        ("wLD_abs_error", "wLD_abs_error", "wLD", "lower"),
        ("wWD_abs_error", "wWD_abs_error", "wWD", "lower"),
        ("wLW_abs_error", "wLW_abs_error", "wLW", "lower"),
        ("profile_depth_rmse_m", "profile_depth_rmse_m", "profile_rmse", "lower"),
        ("er_like_profile_error", "er_like_profile_error", "er_like", "lower"),
        ("projected_mask_iou", "projected_mask_iou", "iou", "higher"),
        ("projected_mask_dice", "projected_mask_dice", "dice", "higher"),
    ]
    out: list[dict[str, Any]] = []
    for metric, current_key, ref_key, direction in metric_map:
        current = f(selected_test.get(current_key))
        for ref in references:
            reference = f(ref.get(ref_key))
            if not (math.isfinite(reference) and math.isfinite(current)):
                improved: bool | str = ""
                delta: float | str = ""
            else:
                delta = current - reference
                improved = delta < 0.0 if direction == "lower" else delta > 0.0
            out.append(
                {
                    "metric": metric,
                    "reference_label": ref["label"],
                    "current_label": f"{selected_test['candidate']}::seed{selected_test['seed']}",
                    "reference_value": reference,
                    "current_value": current,
                    "delta": delta,
                    "improved": improved,
                    "direction": direction,
                    "notes": "validation-only seed selection; test final only for selected seed",
                }
            )
    return out


def surface_candidate_gate(test_row: dict[str, Any]) -> dict[str, bool]:
    return {
        "profile_not_obviously_worse_than_20_85": f(test_row["profile_depth_rmse_m"]) <= REF_20_85["profile_rmse"] + 0.00008,
        "LWD_not_obviously_degraded": (
            f(test_row["L_mae_mm"]) <= REF_20_85["L_mm"] + 0.45
            and f(test_row["W_mae_mm"]) <= REF_20_85["W_mm"] + 0.45
            and f(test_row["D_mae_mm"]) <= REF_20_85["D_mm"] + 0.25
        ),
        "w_or_mask_improved": f(test_row["wMAE"]) < min(REF_20_85["wMAE"], REF_24_1["wMAE"]) or f(test_row["projected_mask_dice"]) >= REF_24_1["dice"],
        "dice_not_obviously_worse_than_20_85": f(test_row["projected_mask_dice"]) >= REF_20_85["dice"] - 0.02,
    }


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.seed_summary, args.metrics, args.group_summary, args.vs_reference], args.overwrite)
    selected_name = selected_candidate_from_summary(args.screen_summary)
    config = config_by_name(selected_name)
    inputs = build_inputs(args.dataset_id)
    seed_outputs: list[dict[str, Any]] = []

    for seed in args.seeds:
        out = train_candidate(config, seed, inputs, args.epochs, args.batch_size, args.lr, args.weight_decay)
        train_val_idx = np.concatenate([inputs.splits["train"], inputs.splits["val"]])
        train_val_pred = predict_raw_for_indices(out["model"], inputs, train_val_idx)
        train_val_rows = evaluate_subset(inputs.dataset, train_val_pred, inputs.stats, train_val_idx)
        train_agg = aggregate_eval_rows(train_val_rows, config.name, "train")
        val_agg = aggregate_eval_rows(train_val_rows, config.name, "val")
        train_row = row_from_aggregate(config.name, seed, config.role, len(inputs.feature_names), "train", train_agg, "", out["best_epoch"], False, True, False, "multi-seed train screen")
        val_row = row_from_aggregate(config.name, seed, config.role, len(inputs.feature_names), "val", val_agg, "", out["best_epoch"], False, True, False, "multi-seed validation screen")
        val_row["selection_score"] = aggregate_selection_score(val_row)
        seed_outputs.append({"seed": seed, "out": out, "train_row": train_row, "val_row": val_row})

    selected = min(seed_outputs, key=lambda item: f(item["val_row"]["selection_score"]))
    selected_seed = int(selected["seed"])
    all_idx = np.arange(len(inputs.dataset.sample_ids))
    selected_pred = predict_raw_for_indices(selected["out"]["model"], inputs, all_idx)
    selected_rows = evaluate_subset(inputs.dataset, selected_pred, inputs.stats, all_idx)
    metric_rows: list[dict[str, Any]] = []
    selected_split_rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        agg = aggregate_eval_rows(selected_rows, config.name, split)
        row = row_from_aggregate(
            config.name,
            selected_seed,
            config.role,
            len(inputs.feature_names),
            split,
            agg,
            selected["val_row"]["selection_score"] if split == "val" else "",
            selected["out"]["best_epoch"],
            True,
            True,
            split == "test",
            "selected seed; test final only" if split == "test" else "selected seed",
        )
        selected_split_rows[split] = row
        metric_rows.append(metric_row_from_screen_row(row, True))

    selected_test = selected_split_rows["test"]
    seed_summary_rows: list[dict[str, Any]] = []
    for item in seed_outputs:
        seed = int(item["seed"])
        is_selected = seed == selected_seed
        train_row = selected_split_rows["train"] if is_selected else item["train_row"]
        val_row = selected_split_rows["val"] if is_selected else item["val_row"]
        test_row = selected_test if is_selected else {}
        seed_summary_rows.append(
            {
                "candidate": config.name,
                "seed": seed,
                "selected_seed": is_selected,
                "feature_count": len(inputs.feature_names),
                "best_epoch": item["out"]["best_epoch"],
                "best_val_epoch_score": item["out"]["best_val_epoch_score"],
                "validation_selection_score": val_row["selection_score"],
                "min_train_normalized_param_mae": item["out"]["min_train_normalized_param_mae"],
                "train_total_mae": train_row["normalized_param_mae"],
                "val_total_mae": val_row["normalized_param_mae"],
                "test_total_mae": test_row.get("normalized_param_mae", ""),
                "train_wMAE": train_row["wMAE"],
                "val_wMAE": val_row["wMAE"],
                "test_wMAE": test_row.get("wMAE", ""),
                "train_profile_depth_rmse_m": train_row["profile_depth_rmse_m"],
                "val_profile_depth_rmse_m": val_row["profile_depth_rmse_m"],
                "test_profile_depth_rmse_m": test_row.get("profile_depth_rmse_m", ""),
                "test_L_mae_mm": test_row.get("L_mae_mm", ""),
                "test_W_mae_mm": test_row.get("W_mae_mm", ""),
                "test_D_mae_mm": test_row.get("D_mae_mm", ""),
                "test_wLD_abs_error": test_row.get("wLD_abs_error", ""),
                "test_wWD_abs_error": test_row.get("wWD_abs_error", ""),
                "test_wLW_abs_error": test_row.get("wLW_abs_error", ""),
                "test_er_like_profile_error": test_row.get("er_like_profile_error", ""),
                "test_projected_mask_iou": test_row.get("projected_mask_iou", ""),
                "test_projected_mask_dice": test_row.get("projected_mask_dice", ""),
                "test_final_only": is_selected,
            }
        )

    gate = surface_candidate_gate(selected_test)
    surface_candidate = all(gate.values())
    write_csv(args.seed_summary, seed_summary_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_rows(config.name, selected_seed, selected_rows), GROUP_FIELDS)
    write_csv(args.vs_reference, comparison_rows(selected_test), VS_FIELDS)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface_rbc_nls_feature_fusion_training_summary",
                "stage: 24.2 Stage D",
                "",
                f"dataset_id: {inputs.dataset.dataset_id}",
                f"selected_candidate: {config.name}",
                f"selected_candidate_role: {config.role}",
                f"feature_count: {len(inputs.feature_names)}",
                f"seeds: {list(args.seeds)}",
                f"epochs: {args.epochs}",
                f"batch_size: {args.batch_size}",
                f"selected_seed: {selected_seed}",
                "model_selection: validation-only seed selection; test final only for selected seed.",
                "model_inputs: delta_b/BxByBz + train-scaled nlslite_* features",
                "COMSOL_run: false",
                "data_or_NPZ_modified: false",
                "CURRENT_BASELINE_update: false",
                "",
                f"selected_test_total_mae: {f(selected_test['normalized_param_mae']):.6f}",
                f"selected_test_LWD_mae_mm: {f(selected_test['L_mae_mm']):.6f}/{f(selected_test['W_mae_mm']):.6f}/{f(selected_test['D_mae_mm']):.6f}",
                f"selected_test_wMAE: {f(selected_test['wMAE']):.6f}",
                f"selected_test_wLD_wWD_wLW: {f(selected_test['wLD_abs_error']):.6f}/{f(selected_test['wWD_abs_error']):.6f}/{f(selected_test['wLW_abs_error']):.6f}",
                f"selected_test_profile_depth_rmse_m: {f(selected_test['profile_depth_rmse_m']):.9f}",
                f"selected_test_er_like_profile_error: {f(selected_test['er_like_profile_error']):.6f}",
                f"selected_test_projected_mask_iou_dice: {f(selected_test['projected_mask_iou']):.6f}/{f(selected_test['projected_mask_dice']):.6f}",
                f"delta_vs_20_85_total: {f(selected_test['normalized_param_mae']) - REF_20_85['total']:.6f}",
                f"delta_vs_20_85_wMAE: {f(selected_test['wMAE']) - REF_20_85['wMAE']:.6f}",
                f"delta_vs_20_85_profile_rmse_m: {f(selected_test['profile_depth_rmse_m']) - REF_20_85['profile_rmse']:.9f}",
                f"delta_vs_20_85_dice: {f(selected_test['projected_mask_dice']) - REF_20_85['dice']:.6f}",
                f"delta_vs_24_1_total: {f(selected_test['normalized_param_mae']) - REF_24_1['total']:.6f}",
                f"delta_vs_24_1_wMAE: {f(selected_test['wMAE']) - REF_24_1['wMAE']:.6f}",
                f"delta_vs_24_1_profile_rmse_m: {f(selected_test['profile_depth_rmse_m']) - REF_24_1['profile_rmse']:.9f}",
                f"delta_vs_24_1_dice: {f(selected_test['projected_mask_dice']) - REF_24_1['dice']:.6f}",
                f"surface_candidate_gate: {gate}",
                f"surface_feature_fusion_candidate: {surface_candidate}",
                "boundary: diagnostic candidate only; CURRENT_BASELINE remains unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--screen-summary", type=Path, default=SCREEN_SUMMARY)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

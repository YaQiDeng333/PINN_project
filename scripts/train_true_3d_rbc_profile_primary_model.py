#!/usr/bin/env python
"""Multi-seed run for the selected Stage 20.83 profile-primary candidate."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    V3_240_DATASET_ID,
    PARAM_NAMES,
    ROOT,
    check_no_overwrite,
    denormalize_y,
    load_dataset,
    normalize_x,
    normalize_y,
    split_indices,
    train_normalization,
    write_csv,
)
from train_true_3d_rbc_profile_primary_candidates import (
    REF_FEATURE,
    REF_FUSION,
    REF_NEURAL,
    GROUP_FIELDS,
    PROFILE_FIELDS,
    CandidateConfig,
    aggregate_rows,
    candidate_configs,
    candidate_selection_score,
    evaluate_predictions,
    group_rows,
    reference_metrics,
    selected_reference_profile,
    train_candidate,
)


SCREEN_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_candidate_screen_summary.txt"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_seed_summary.csv"
METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_metrics.csv"
EPOCH_LOG = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_epoch_log.csv"
GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_group_summary.csv"
PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_profile_metrics.csv"
VS_REFERENCE = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_vs_reference.csv"

SEED_FIELDS = [
    "variant",
    "seed",
    "selected_seed",
    "profile_weight",
    "dimension_weight",
    "curvature_aux_weight",
    "soft_mask_weight",
    "best_epoch",
    "best_val_score",
    "candidate_val_selection_score",
    "min_train_normalized_param_mae",
    "train_normalized_param_mae",
    "val_normalized_param_mae",
    "test_normalized_param_mae",
    "train_dimension_mae_norm",
    "val_dimension_mae_norm",
    "test_dimension_mae_norm",
    "train_curvature_mae_norm",
    "val_curvature_mae_norm",
    "test_curvature_mae_norm",
    "train_profile_depth_rmse_m",
    "val_profile_depth_rmse_m",
    "test_profile_depth_rmse_m",
    "train_er_like_profile_error",
    "val_er_like_profile_error",
    "test_er_like_profile_error",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_wLD_abs_error",
    "test_wWD_abs_error",
    "test_wLW_abs_error",
    "test_projected_mask_iou",
    "test_projected_mask_dice",
    "test_max_depth_error_m",
    "test_volume_proxy_rel_error",
    "test_final_only",
]
METRIC_FIELDS = [
    "variant",
    "seed",
    "selected_seed",
    "split",
    "param",
    "sample_count",
    "normalized_mae",
    "physical_mae",
    "physical_mae_mm",
    "relative_mae",
]
EPOCH_FIELDS = [
    "variant",
    "seed",
    "epoch",
    "train_loss",
    "train_profile_depth_rmse_m",
    "val_profile_depth_rmse_m",
    "train_er_like_profile_error",
    "val_er_like_profile_error",
    "train_dimension_mae_norm",
    "val_dimension_mae_norm",
    "train_curvature_mae_norm",
    "val_curvature_mae_norm",
    "train_projected_mask_dice",
    "val_projected_mask_dice",
    "val_selection_score",
]
VS_FIELDS = ["metric", "reference_label", "current_label", "reference_value", "current_value", "delta", "improved", "notes"]


def selected_candidate_from_summary(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^selected_candidate:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"selected_candidate not found in {path}")
    eligible = re.search(r"^eligible_for_multiseed:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if eligible and eligible.group(1).strip().lower() != "true":
        raise RuntimeError(f"candidate screen did not pass Stage C gate: eligible_for_multiseed={eligible.group(1).strip()}")
    return match.group(1).strip()


def screen_gate_failed(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    eligible = re.search(r"^eligible_for_multiseed:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return bool(eligible and eligible.group(1).strip().lower() != "true")


def config_by_name(name: str) -> CandidateConfig:
    for config in candidate_configs():
        if config.name == name:
            return config
    raise RuntimeError(f"unknown profile-primary candidate: {name}")


def compute_param_rows(variant: str, seed: int, selected: bool, y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_norm: np.ndarray, y_pred_norm: np.ndarray, splits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, idx in splits.items():
        true_raw = y_true_raw[idx]
        pred_raw = y_pred_raw[idx]
        true_norm = y_true_norm[idx]
        pred_norm = y_pred_norm[idx]
        for pidx, param_name in enumerate(PARAM_NAMES):
            abs_raw = np.abs(pred_raw[:, pidx] - true_raw[:, pidx])
            abs_norm = np.abs(pred_norm[:, pidx] - true_norm[:, pidx])
            denom = np.maximum(np.abs(true_raw[:, pidx]), 1.0e-12)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "selected_seed": selected,
                    "split": split_name,
                    "param": param_name,
                    "sample_count": len(idx),
                    "normalized_mae": float(abs_norm.mean()),
                    "physical_mae": float(abs_raw.mean()),
                    "physical_mae_mm": float(abs_raw.mean() * 1000.0) if pidx < 3 else "",
                    "relative_mae": float(np.mean(abs_raw / denom)) if pidx < 3 else "",
                }
            )
        abs_norm_all = np.abs(pred_norm - true_norm)
        abs_raw_all = np.abs(pred_raw - true_raw)
        rows.extend(
            [
                {"variant": variant, "seed": seed, "selected_seed": selected, "split": split_name, "param": "ALL", "sample_count": len(idx), "normalized_mae": float(abs_norm_all.mean()), "physical_mae": float(abs_raw_all.mean()), "physical_mae_mm": "", "relative_mae": ""},
                {"variant": variant, "seed": seed, "selected_seed": selected, "split": split_name, "param": "DIMENSION_MEAN", "sample_count": len(idx), "normalized_mae": float(abs_norm_all[:, :3].mean()), "physical_mae": float(abs_raw_all[:, :3].mean()), "physical_mae_mm": float(abs_raw_all[:, :3].mean() * 1000.0), "relative_mae": ""},
                {"variant": variant, "seed": seed, "selected_seed": selected, "split": split_name, "param": "CURVATURE_MEAN", "sample_count": len(idx), "normalized_mae": float(abs_norm_all[:, 3:].mean()), "physical_mae": float(abs_raw_all[:, 3:].mean()), "physical_mae_mm": "", "relative_mae": ""},
            ]
        )
    return rows


def comparison_rows(selected_test: dict[str, Any]) -> list[dict[str, Any]]:
    refs = [REF_NEURAL, REF_FUSION, REF_FEATURE]
    mapping = [
        ("total_normalized_mae", "normalized_param_mae", "total", True),
        ("L_mae_mm", "L_mae_mm", "L_mm", True),
        ("W_mae_mm", "W_mae_mm", "W_mm", True),
        ("D_mae_mm", "D_mae_mm", "D_mm", True),
        ("curvature_aux_mae", "curvature_mae", "curvature", True),
        ("wLD_abs_error_aux", "wLD_abs_error", "wLD", True),
        ("wWD_abs_error_aux", "wWD_abs_error", "wWD", True),
        ("wLW_abs_error_aux", "wLW_abs_error", "wLW", True),
        ("projected_mask_iou", "projected_mask_iou", "iou", False),
        ("projected_mask_dice", "projected_mask_dice", "dice", False),
        ("profile_depth_rmse_m", "profile_depth_rmse_m", "profile_rmse", True),
    ]
    out: list[dict[str, Any]] = []
    for metric, current_key, ref_key, lower_better in mapping:
        current = float(selected_test[current_key])
        for ref in refs:
            reference = float(ref[ref_key])
            delta = current - reference
            improved = delta < 0.0 if lower_better else delta > 0.0
            out.append(
                {
                    "metric": metric,
                    "reference_label": ref["label"],
                    "current_label": f"{selected_test['variant']}::seed{selected_test['seed']}",
                    "reference_value": reference,
                    "current_value": current,
                    "delta": delta,
                    "improved": improved,
                    "notes": "selected by validation profile-primary score; test final only",
                }
            )
    return out


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.seed_summary, args.metrics, args.epoch_log, args.group_summary, args.profile_metrics, args.vs_reference], args.overwrite)
    if args.variant and screen_gate_failed(args.screen_summary) and not args.force_after_failed_gate:
        raise RuntimeError("refusing to bypass failed Stage C gate with --variant; pass --force-after-failed-gate only for an explicitly documented diagnostic rerun")
    selected_name = args.variant or selected_candidate_from_summary(args.screen_summary)
    config = config_by_name(selected_name)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    stats = train_normalization(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    depth_scale = float(np.max(dataset.rbc_params[splits["train"], 2]) - np.min(dataset.rbc_params[splits["train"], 2]))
    if depth_scale <= 1.0e-12:
        depth_scale = float(np.mean(dataset.rbc_params[splits["train"], 2]))
    ref_val = reference_metrics(selected_reference_profile(), "val", depth_scale)
    seed_outputs: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        out = train_candidate(config, seed, x_norm, y_norm, dataset, stats, splits, args.epochs, args.batch_size, args.lr, args.weight_decay, depth_scale)
        pred_raw = denormalize_y(out["pred_norm"], stats)
        rows = evaluate_predictions(dataset, pred_raw, stats)
        train_agg = aggregate_rows(rows, config.name, seed, config, False, False, "train", "", depth_scale)
        val_agg = aggregate_rows(rows, config.name, seed, config, False, False, "val", "", depth_scale)
        val_score = candidate_selection_score(val_agg, ref_val)
        out["pred_raw"] = pred_raw
        out["rows"] = rows
        out["train_agg"] = train_agg
        out["val_agg"] = val_agg
        out["candidate_val_selection_score"] = val_score
        seed_outputs.append(out)
        epoch_rows.extend(out["epoch_rows"])
    selected = min(seed_outputs, key=lambda item: float(item["candidate_val_selection_score"]))
    selected_seed = int(selected["seed"])
    selected_rows = selected["rows"]
    for row in selected_rows:
        row["variant"] = config.name
        row["seed"] = selected_seed
        row["selected_by_validation"] = True
    selected_train = aggregate_rows(selected_rows, config.name, selected_seed, config, True, True, "train", "", depth_scale)
    selected_val = aggregate_rows(selected_rows, config.name, selected_seed, config, True, True, "val", selected["candidate_val_selection_score"], depth_scale)
    selected_test = aggregate_rows(selected_rows, config.name, selected_seed, config, True, True, "test", "", depth_scale, "selected seed test final only", True)
    for item in seed_outputs:
        seed = int(item["seed"])
        is_selected = seed == selected_seed
        train_agg = selected_train if is_selected else item["train_agg"]
        val_agg = selected_val if is_selected else item["val_agg"]
        test_agg = selected_test if is_selected else {}
        seed_rows.append(
            {
                "variant": config.name,
                "seed": seed,
                "selected_seed": is_selected,
                "profile_weight": config.profile_weight,
                "dimension_weight": config.dimension_weight,
                "curvature_aux_weight": config.curvature_aux_weight,
                "soft_mask_weight": config.soft_mask_weight,
                "best_epoch": item["best_epoch"],
                "best_val_score": item["best_val_score"],
                "candidate_val_selection_score": item["candidate_val_selection_score"],
                "min_train_normalized_param_mae": item["min_train_normalized_param_mae"],
                "train_normalized_param_mae": train_agg["normalized_param_mae"],
                "val_normalized_param_mae": val_agg["normalized_param_mae"],
                "test_normalized_param_mae": test_agg.get("normalized_param_mae", ""),
                "train_dimension_mae_norm": train_agg["dimension_mae_norm"],
                "val_dimension_mae_norm": val_agg["dimension_mae_norm"],
                "test_dimension_mae_norm": test_agg.get("dimension_mae_norm", ""),
                "train_curvature_mae_norm": train_agg["curvature_mae_norm"],
                "val_curvature_mae_norm": val_agg["curvature_mae_norm"],
                "test_curvature_mae_norm": test_agg.get("curvature_mae_norm", ""),
                "train_profile_depth_rmse_m": train_agg["profile_depth_rmse_m"],
                "val_profile_depth_rmse_m": val_agg["profile_depth_rmse_m"],
                "test_profile_depth_rmse_m": test_agg.get("profile_depth_rmse_m", ""),
                "train_er_like_profile_error": train_agg["er_like_profile_error"],
                "val_er_like_profile_error": val_agg["er_like_profile_error"],
                "test_er_like_profile_error": test_agg.get("er_like_profile_error", ""),
                "test_L_mae_mm": test_agg.get("L_mae_mm", ""),
                "test_W_mae_mm": test_agg.get("W_mae_mm", ""),
                "test_D_mae_mm": test_agg.get("D_mae_mm", ""),
                "test_wLD_abs_error": test_agg.get("wLD_abs_error", ""),
                "test_wWD_abs_error": test_agg.get("wWD_abs_error", ""),
                "test_wLW_abs_error": test_agg.get("wLW_abs_error", ""),
                "test_projected_mask_iou": test_agg.get("projected_mask_iou", ""),
                "test_projected_mask_dice": test_agg.get("projected_mask_dice", ""),
                "test_max_depth_error_m": test_agg.get("max_depth_error_m", ""),
                "test_volume_proxy_rel_error": test_agg.get("volume_proxy_rel_error", ""),
                "test_final_only": is_selected,
            }
        )
    selected_pred_norm = (selected["pred_raw"] - stats["y_mean"]) / stats["y_std"]
    metric_rows = compute_param_rows(config.name, selected_seed, True, dataset.rbc_params, selected["pred_raw"], y_norm, selected_pred_norm, splits)
    group_out = group_rows(selected_rows, config.name, selected_seed, True)
    write_csv(args.seed_summary, seed_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
    write_csv(args.group_summary, group_out, GROUP_FIELDS)
    write_csv(args.profile_metrics, selected_rows, PROFILE_FIELDS)
    write_csv(args.vs_reference, comparison_rows(selected_test), VS_FIELDS)
    profile_improved_5pct = float(selected_test["profile_depth_rmse_m"]) <= REF_NEURAL["profile_rmse"] * 0.95
    dice_stable = float(selected_test["projected_mask_dice"]) >= REF_NEURAL["dice"] - 0.01
    dimension_stable = float(selected_test["dimension_mae_norm"]) <= 0.488785 + 0.05
    promising = profile_improved_5pct and dice_stable and dimension_stable
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 profile-primary training summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"selected_candidate: {config.name}",
                f"loss_weights: profile={config.profile_weight}, dimension={config.dimension_weight}, curvature_aux={config.curvature_aux_weight}, soft_mask={config.soft_mask_weight}",
                f"seeds: {list(args.seeds)}",
                f"selected_seed: {selected_seed}",
                "model_selection: validation-only profile-primary score; test final only for selected seed.",
                f"selected_test_total_mae: {float(selected_test['normalized_param_mae']):.6f}",
                f"selected_test_dimension_mae_norm: {float(selected_test['dimension_mae_norm']):.6f}",
                f"selected_test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.6f}/{float(selected_test['W_mae_mm']):.6f}/{float(selected_test['D_mae_mm']):.6f}",
                f"selected_test_wLD_wWD_wLW_aux: {float(selected_test['wLD_abs_error']):.6f}/{float(selected_test['wWD_abs_error']):.6f}/{float(selected_test['wLW_abs_error']):.6f}",
                f"selected_test_curvature_aux_mae: {float(selected_test['curvature_mae']):.6f}",
                f"selected_test_profile_depth_rmse_m: {float(selected_test['profile_depth_rmse_m']):.9f}",
                f"selected_test_er_like_profile_error: {float(selected_test['er_like_profile_error']):.6f}",
                f"selected_test_max_depth_error_m: {float(selected_test['max_depth_error_m']):.9f}",
                f"selected_test_volume_proxy_rel_error: {float(selected_test['volume_proxy_rel_error']):.6f}",
                f"selected_test_projected_mask_iou_dice: {float(selected_test['projected_mask_iou']):.6f}/{float(selected_test['projected_mask_dice']):.6f}",
                f"vs_20_77_profile_rmse_delta: {float(selected_test['profile_depth_rmse_m']) - REF_NEURAL['profile_rmse']:.9f}",
                f"vs_20_81_profile_rmse_delta: {float(selected_test['profile_depth_rmse_m']) - REF_FUSION['profile_rmse']:.9f}",
                f"profile_improved_5pct_vs_20_77: {profile_improved_5pct}",
                f"dice_stable_vs_20_77: {dice_stable}",
                f"dimension_stable_guard: {dimension_stable}",
                f"promising_profile_primary_gate: {promising}",
                "wMAE_status: auxiliary diagnostic only, not the primary pass/fail criterion.",
                "boundary: no checkpoint is written; no COMSOL, no new data, no NPZ modification, no baseline update.",
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
    parser.add_argument("--variant")
    parser.add_argument("--force-after-failed-gate", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--epoch-log", type=Path, default=EPOCH_LOG)
    parser.add_argument("--group-summary", type=Path, default=GROUP)
    parser.add_argument("--profile-metrics", type=Path, default=PROFILE)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

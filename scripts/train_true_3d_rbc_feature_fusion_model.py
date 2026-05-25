#!/usr/bin/env python
"""Multi-seed run for the validation-selected Stage 20.81 fusion candidate."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    V3_240_DATASET_ID,
    ROOT,
    aggregate_prediction_rows,
    check_no_overwrite,
    denormalize_y,
    evaluate_param_predictions,
    load_dataset,
    normalize_x,
    normalize_y,
    split_indices,
    train_normalization,
    write_csv,
)
from train_true_3d_rbc_feature_fusion_candidates import (
    FEATURES,
    REF_FEATURE,
    REF_NEURAL,
    REF_REFINED,
    CandidateConfig,
    aggregate_rows,
    candidate_configs,
    candidate_selection_score,
    evaluate_subset,
    feature_indices,
    feature_matrix,
    group_rows,
    read_features,
    train_candidate,
)


SCREEN_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_candidate_screen_summary.txt"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_seed_summary.csv"
METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_metrics.csv"
EPOCH_LOG = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_epoch_log.csv"
GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_group_summary.csv"
PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_profile_metrics.csv"
VS_REFERENCE = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_vs_reference.csv"

SEED_FIELDS = [
    "variant",
    "seed",
    "selected_seed",
    "feature_set",
    "feature_count",
    "curvature_weight",
    "best_epoch",
    "best_val_epoch_score",
    "candidate_val_selection_score",
    "min_train_normalized_param_mae",
    "train_normalized_param_mae",
    "val_normalized_param_mae",
    "test_normalized_param_mae",
    "train_dimension_mae_norm",
    "val_dimension_mae_norm",
    "test_dimension_mae_norm",
    "train_curvature_mae",
    "val_curvature_mae",
    "test_curvature_mae",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_wLD_abs_error",
    "test_wWD_abs_error",
    "test_wLW_abs_error",
    "test_projected_mask_iou",
    "test_projected_mask_dice",
    "test_profile_depth_rmse_m",
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
    "train_total_mae",
    "val_total_mae",
    "train_dimension_mae_norm",
    "val_dimension_mae_norm",
    "train_curvature_mae_norm",
    "val_curvature_mae_norm",
    "val_epoch_selection_score",
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


def config_by_name(name: str) -> CandidateConfig:
    for config in candidate_configs():
        if config.name == name:
            return config
    raise RuntimeError(f"unknown fusion candidate: {name}")


def compute_param_rows(variant: str, seed: int, selected: bool, y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_norm: np.ndarray, y_pred_norm: np.ndarray, splits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    param_names = ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW"]
    rows: list[dict[str, Any]] = []
    for split_name, idx in splits.items():
        true_raw = y_true_raw[idx]
        pred_raw = y_pred_raw[idx]
        true_norm = y_true_norm[idx]
        pred_norm = y_pred_norm[idx]
        for pidx, param_name in enumerate(param_names):
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
                {
                    "variant": variant,
                    "seed": seed,
                    "selected_seed": selected,
                    "split": split_name,
                    "param": "ALL",
                    "sample_count": len(idx),
                    "normalized_mae": float(abs_norm_all.mean()),
                    "physical_mae": float(abs_raw_all.mean()),
                    "physical_mae_mm": "",
                    "relative_mae": "",
                },
                {
                    "variant": variant,
                    "seed": seed,
                    "selected_seed": selected,
                    "split": split_name,
                    "param": "DIMENSION_MEAN",
                    "sample_count": len(idx),
                    "normalized_mae": float(abs_norm_all[:, :3].mean()),
                    "physical_mae": float(abs_raw_all[:, :3].mean()),
                    "physical_mae_mm": float(abs_raw_all[:, :3].mean() * 1000.0),
                    "relative_mae": "",
                },
                {
                    "variant": variant,
                    "seed": seed,
                    "selected_seed": selected,
                    "split": split_name,
                    "param": "CURVATURE_MEAN",
                    "sample_count": len(idx),
                    "normalized_mae": float(abs_norm_all[:, 3:].mean()),
                    "physical_mae": float(abs_raw_all[:, 3:].mean()),
                    "physical_mae_mm": "",
                    "relative_mae": "",
                },
            ]
        )
    return rows


def comparison_rows(selected_test: dict[str, Any]) -> list[dict[str, Any]]:
    refs = [REF_NEURAL, REF_FEATURE, REF_REFINED]
    mapping = [
        ("total_normalized_mae", "normalized_param_mae", "total", True),
        ("L_mae_mm", "L_mae_mm", "L_mm", True),
        ("W_mae_mm", "W_mae_mm", "W_mm", True),
        ("D_mae_mm", "D_mae_mm", "D_mm", True),
        ("curvature_mae", "curvature_mae", "curvature", True),
        ("wLD_abs_error", "wLD_abs_error", "wLD", True),
        ("wWD_abs_error", "wWD_abs_error", "wWD", True),
        ("wLW_abs_error", "wLW_abs_error", "wLW", True),
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
                    "notes": "selected by validation; test final only",
                }
            )
    return out


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.seed_summary, args.metrics, args.epoch_log, args.group_summary, args.profile_metrics, args.vs_reference], args.overwrite)
    selected_name = args.variant or selected_candidate_from_summary(args.screen_summary)
    config = config_by_name(selected_name)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    stats = train_normalization(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    feature_names, raw_features, sample_ids, feature_split = read_features(args.features)
    if sample_ids != [str(x) for x in dataset.sample_ids]:
        raise RuntimeError("feature sample_id mismatch")
    if feature_split != [str(x) for x in dataset.split]:
        raise RuntimeError("feature split mismatch")
    idx = feature_indices(feature_names, config.prefixes)
    features_norm, _transform = feature_matrix(raw_features, splits["train"], idx)

    seed_outputs: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        out = train_candidate(config, seed, x_norm, y_norm, features_norm, splits, args.epochs, args.batch_size, args.lr, args.weight_decay)
        pred_raw = denormalize_y(out["pred_norm"], stats)
        train_val_rows = evaluate_subset(dataset, pred_raw, stats, np.concatenate([splits["train"], splits["val"]]))
        train_agg = aggregate_rows(train_val_rows, config.name, seed, config, False, False, "train", "", f"feature_count={len(idx)}")
        val_agg = aggregate_rows(train_val_rows, config.name, seed, config, False, False, "val", "", f"feature_count={len(idx)}")
        val_score = candidate_selection_score(val_agg)
        out["pred_raw"] = pred_raw
        out["train_rows"] = train_val_rows
        out["train_agg"] = train_agg
        out["val_agg"] = val_agg
        out["candidate_val_selection_score"] = val_score
        seed_outputs.append(out)
        epoch_rows.extend(out["epoch_rows"])
    selected = min(seed_outputs, key=lambda item: float(item["candidate_val_selection_score"]))
    selected_seed = int(selected["seed"])
    selected_rows = evaluate_param_predictions(dataset, selected["pred_raw"], stats)
    selected_test = aggregate_rows(selected_rows, config.name, selected_seed, config, True, True, "test", "", f"feature_count={len(idx)}; selected seed test final only", True)
    selected_train = aggregate_rows(selected_rows, config.name, selected_seed, config, True, True, "train", "", f"feature_count={len(idx)}")
    selected_val = aggregate_rows(selected_rows, config.name, selected_seed, config, True, True, "val", selected["candidate_val_selection_score"], f"feature_count={len(idx)}")

    for item in seed_outputs:
        seed = int(item["seed"])
        is_selected = seed == selected_seed
        train_agg = selected_train if is_selected else item["train_agg"]
        val_agg = selected_val if is_selected else item["val_agg"]
        if is_selected:
            test_agg = selected_test
        else:
            test_agg = {key: "" for key in selected_test}
        seed_summary_rows.append(
            {
                "variant": config.name,
                "seed": seed,
                "selected_seed": is_selected,
                "feature_set": config.feature_set,
                "feature_count": len(idx),
                "curvature_weight": config.curvature_weight,
                "best_epoch": item["best_epoch"],
                "best_val_epoch_score": item["best_val_epoch_score"],
                "candidate_val_selection_score": item["candidate_val_selection_score"],
                "min_train_normalized_param_mae": item["min_train_normalized_param_mae"],
                "train_normalized_param_mae": train_agg["normalized_param_mae"],
                "val_normalized_param_mae": val_agg["normalized_param_mae"],
                "test_normalized_param_mae": test_agg.get("normalized_param_mae", ""),
                "train_dimension_mae_norm": train_agg["dimension_mae_norm"],
                "val_dimension_mae_norm": val_agg["dimension_mae_norm"],
                "test_dimension_mae_norm": test_agg.get("dimension_mae_norm", ""),
                "train_curvature_mae": train_agg["curvature_mae"],
                "val_curvature_mae": val_agg["curvature_mae"],
                "test_curvature_mae": test_agg.get("curvature_mae", ""),
                "test_L_mae_mm": test_agg.get("L_mae_mm", ""),
                "test_W_mae_mm": test_agg.get("W_mae_mm", ""),
                "test_D_mae_mm": test_agg.get("D_mae_mm", ""),
                "test_wLD_abs_error": test_agg.get("wLD_abs_error", ""),
                "test_wWD_abs_error": test_agg.get("wWD_abs_error", ""),
                "test_wLW_abs_error": test_agg.get("wLW_abs_error", ""),
                "test_projected_mask_iou": test_agg.get("projected_mask_iou", ""),
                "test_projected_mask_dice": test_agg.get("projected_mask_dice", ""),
                "test_profile_depth_rmse_m": test_agg.get("profile_depth_rmse_m", ""),
                "test_final_only": is_selected,
            }
        )

    metric_rows = compute_param_rows(config.name, selected_seed, True, dataset.rbc_params, selected["pred_raw"], y_norm, (selected["pred_raw"] - stats["y_mean"]) / stats["y_std"], splits)
    for row in selected_rows:
        row["variant"] = config.name
        row["seed"] = selected_seed
        row["feature_set"] = config.feature_set
        row["selected_by_validation"] = True
    profile_rows = selected_rows
    group_out = group_rows(selected_rows, config.name, selected_seed, config.feature_set, True)

    write_csv(args.seed_summary, seed_summary_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
    write_csv(args.group_summary, group_out, [
        "variant",
        "seed",
        "feature_set",
        "selected_by_validation",
        "split",
        "group_field",
        "group_value",
        "sample_count",
        "normalized_param_mae",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "curvature_mae",
        "wLD_abs_error",
        "wWD_abs_error",
        "wLW_abs_error",
        "projected_mask_iou",
        "projected_mask_dice",
        "profile_depth_rmse_m",
    ])
    write_csv(args.profile_metrics, profile_rows, [
        "variant",
        "seed",
        "feature_set",
        "selected_by_validation",
        "sample_id",
        "split",
        "curvature_template",
        "depth_bin",
        "aspect_bin",
        "size_bin",
        "normalized_param_mae_mean",
        "dimension_param_mae_norm",
        "curvature_param_mae_norm",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "wLD_abs_error",
        "wWD_abs_error",
        "wLW_abs_error",
        "curvature_mae_mean",
        "projected_mask_iou",
        "projected_mask_dice",
        "profile_depth_rmse_m",
    ])
    write_csv(args.vs_reference, comparison_rows(selected_test), VS_FIELDS)

    promising = (
        float(selected_test["curvature_mae"]) <= REF_NEURAL["curvature"] - 0.01
        and float(selected_test["normalized_param_mae"]) <= REF_NEURAL["total"] + 0.02
        and float(selected_test["projected_mask_dice"]) >= REF_NEURAL["dice"] - 0.01
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 feature-fusion training summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"selected_candidate: {config.name}",
                f"selected_feature_set: {config.feature_set}",
                f"selected_feature_count: {len(idx)}",
                f"curvature_weight: {config.curvature_weight}",
                f"seeds: {list(args.seeds)}",
                f"selected_seed: {selected_seed}",
                "model_selection: validation-only seed selection; test final only for selected seed.",
                f"selected_test_total_mae: {float(selected_test['normalized_param_mae']):.6f}",
                f"selected_test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.6f}/{float(selected_test['W_mae_mm']):.6f}/{float(selected_test['D_mae_mm']):.6f}",
                f"selected_test_curvature_mae: {float(selected_test['curvature_mae']):.6f}",
                f"selected_test_wLD_wWD_wLW: {float(selected_test['wLD_abs_error']):.6f}/{float(selected_test['wWD_abs_error']):.6f}/{float(selected_test['wLW_abs_error']):.6f}",
                f"selected_test_projected_mask_iou_dice: {float(selected_test['projected_mask_iou']):.6f}/{float(selected_test['projected_mask_dice']):.6f}",
                f"selected_test_profile_depth_rmse_m: {float(selected_test['profile_depth_rmse_m']):.9f}",
                f"vs_20_77_total_delta: {float(selected_test['normalized_param_mae']) - REF_NEURAL['total']:.6f}",
                f"vs_20_77_curvature_delta: {float(selected_test['curvature_mae']) - REF_NEURAL['curvature']:.6f}",
                f"vs_20_77_dice_delta: {float(selected_test['projected_mask_dice']) - REF_NEURAL['dice']:.6f}",
                f"promising_feature_fusion: {promising}",
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
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--screen-summary", type=Path, default=SCREEN_SUMMARY)
    parser.add_argument("--variant")
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

"""Multi-seed run for the selected v3_240 curvature-aware variant."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    PARAM_NAMES,
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
from train_true_3d_rbc_feature_baselines import extract_signal_features, fit_feature_scaler, transform_features
from train_true_3d_rbc_curvature_aware_candidates import (
    GROUP_FIELDS,
    METRIC_FIELDS as CANDIDATE_METRIC_FIELDS,
    PROFILE_FIELDS,
    VariantConfig,
    aggregate_variant_rows,
    group_variant_rows,
    profile_selection_score,
    train_variant,
)


DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
CANDIDATE_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_candidate_screen_metrics.csv"

SUMMARY_OUT = ROOT / "results/summaries/true_3d_rbc_v3_240_curvature_refined_training_summary.txt"
SEED_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_seed_summary.csv"
METRICS_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_metrics.csv"
EPOCH_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_epoch_log.csv"
GROUP_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_group_summary.csv"
PROFILE_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_profile_metrics.csv"
COMPARE_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_vs_reference.csv"

SEED_FIELDS = [
    "variant",
    "seed",
    "selected_seed",
    "best_epoch",
    "best_val_epoch_score",
    "val_selection_score",
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
]

PARAM_FIELDS = [
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

COMPARE_FIELDS = ["metric", "reference_value", "refined_value", "delta", "improved", "notes"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def selected_variant_from_screen() -> str:
    rows = read_csv(CANDIDATE_METRICS)
    selected = [r for r in rows if r.get("split") == "val" and r.get("selected_by_validation") == "True"]
    if not selected:
        raise RuntimeError("selected candidate missing from screen metrics")
    return selected[0]["variant"]


def config_for_variant(name: str) -> VariantConfig:
    if name == "C1_split_heads":
        return VariantConfig(name, curvature_weight=1.0)
    if name == "C2_split_heads_curv_weight_1p5":
        return VariantConfig(name, curvature_weight=1.5)
    if name == "C3_stronger_encoder_feature_fusion":
        return VariantConfig(name, stronger_encoder=True, use_features=True, curvature_weight=1.0)
    raise RuntimeError(f"unsupported selected variant for multi-seed run: {name}")


def param_metric_rows(variant: str, seed: int, selected: bool, y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_norm: np.ndarray, y_pred_norm: np.ndarray, splits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, idx in splits.items():
        abs_norm = np.abs(y_true_norm[idx] - y_pred_norm[idx])
        abs_raw = np.abs(y_true_raw[idx] - y_pred_raw[idx])
        for pidx, name in enumerate(PARAM_NAMES):
            denom = np.maximum(np.abs(y_true_raw[idx, pidx]), 1.0e-12)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "selected_seed": selected,
                    "split": split,
                    "param": name,
                    "sample_count": len(idx),
                    "normalized_mae": float(abs_norm[:, pidx].mean()),
                    "physical_mae": float(abs_raw[:, pidx].mean()),
                    "physical_mae_mm": float(abs_raw[:, pidx].mean() * 1000.0) if pidx < 3 else "",
                    "relative_mae": float((abs_raw[:, pidx] / denom).mean()) if pidx < 3 else "",
                }
            )
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "selected_seed": selected,
                "split": split,
                "param": "ALL",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm.mean()),
                "physical_mae": float(abs_raw.mean()),
            }
        )
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "selected_seed": selected,
                "split": split,
                "param": "DIMENSION_MEAN",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm[:, :3].mean()),
                "physical_mae": float(abs_raw[:, :3].mean()),
                "physical_mae_mm": float(abs_raw[:, :3].mean() * 1000.0),
            }
        )
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "selected_seed": selected,
                "split": split,
                "param": "CURVATURE_MEAN",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm[:, 3:].mean()),
                "physical_mae": float(abs_raw[:, 3:].mean()),
            }
        )
    return rows


def reference_aggregate_values() -> dict[str, float]:
    rows = read_csv(ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_candidate_screen_metrics.csv")
    ref_test = [r for r in rows if r["variant"] == "C0_reference_20_77" and r["split"] == "test"][0]
    return {
        "normalized_param_mae": float(ref_test["normalized_param_mae"]),
        "dimension_mae_norm": float(ref_test["dimension_mae_norm"]),
        "curvature_mae": float(ref_test["curvature_mae"]),
        "L_mae_mm": float(ref_test["L_mae_mm"]),
        "W_mae_mm": float(ref_test["W_mae_mm"]),
        "D_mae_mm": float(ref_test["D_mae_mm"]),
        "wLD_abs_error": float(ref_test["wLD_abs_error"]),
        "wWD_abs_error": float(ref_test["wWD_abs_error"]),
        "wLW_abs_error": float(ref_test["wLW_abs_error"]),
        "projected_mask_iou": float(ref_test["projected_mask_iou"]),
        "projected_mask_dice": float(ref_test["projected_mask_dice"]),
        "profile_depth_rmse_m": float(ref_test["profile_depth_rmse_m"]),
    }


def comparison_rows(ref: dict[str, float], test: dict[str, Any]) -> list[dict[str, Any]]:
    lower = {
        "normalized_param_mae",
        "dimension_mae_norm",
        "curvature_mae",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "wLD_abs_error",
        "wWD_abs_error",
        "wLW_abs_error",
        "profile_depth_rmse_m",
    }
    metrics = [
        "normalized_param_mae",
        "dimension_mae_norm",
        "curvature_mae",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "wLD_abs_error",
        "wWD_abs_error",
        "wLW_abs_error",
        "projected_mask_iou",
        "projected_mask_dice",
        "profile_depth_rmse_m",
    ]
    test_keys = {
        "normalized_param_mae": "test_normalized_param_mae",
        "dimension_mae_norm": "test_dimension_mae_norm",
        "curvature_mae": "test_curvature_mae",
        "L_mae_mm": "test_L_mae_mm",
        "W_mae_mm": "test_W_mae_mm",
        "D_mae_mm": "test_D_mae_mm",
        "wLD_abs_error": "test_wLD_abs_error",
        "wWD_abs_error": "test_wWD_abs_error",
        "wLW_abs_error": "test_wLW_abs_error",
        "projected_mask_iou": "test_projected_mask_iou",
        "projected_mask_dice": "test_projected_mask_dice",
        "profile_depth_rmse_m": "test_profile_depth_rmse_m",
    }
    out = []
    for key in metrics:
        refined = float(test[test_keys[key]])
        delta = refined - ref[key]
        improved = delta < 0 if key in lower else delta > 0
        out.append({"metric": key, "reference_value": ref[key], "refined_value": refined, "delta": delta, "improved": improved, "notes": "test split selected seed"})
    return out


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> int:
    outputs = [args.summary, args.seed_summary, args.metrics, args.epoch_log, args.group_summary, args.profile_metrics, args.comparison]
    check_no_overwrite(outputs, args.overwrite)
    selected_variant = selected_variant_from_screen()
    config = config_for_variant(selected_variant)
    dataset = load_dataset(args.dataset_id)
    stats = train_normalization(dataset)
    splits = split_indices(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    raw_features, _ = extract_signal_features(dataset.x_channels)
    scaler = fit_feature_scaler(raw_features, splits["train"])
    feature_norm = transform_features(raw_features, scaler)

    trained = [
        train_variant(config, seed, x_norm, y_norm, splits, feature_norm, args.epochs, args.batch_size, args.lr, args.weight_decay)
        for seed in args.seeds
    ]
    ref_for_selection_rows = read_csv(ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_candidate_screen_metrics.csv")
    ref_val = [r for r in ref_for_selection_rows if r["variant"] == "C0_reference_20_77" and r["split"] == "val"][0]
    ref_val_dice = float(ref_val["projected_mask_dice"])

    profile_by_seed: dict[int, list[dict[str, Any]]] = {}
    seed_scores: dict[int, float] = {}
    param_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for item in trained:
        seed = int(item["seed"])
        pred_raw = denormalize_y(item["pred_norm"], stats)
        rows = evaluate_param_predictions(dataset, pred_raw, stats)
        for row in rows:
            row["variant"] = selected_variant
            row["seed"] = seed
            row["selected_seed"] = False
        profile_by_seed[seed] = rows
        val_agg = aggregate_prediction_rows(rows, selected_variant, "val")
        seed_scores[seed] = profile_selection_score(val_agg, ref_val_dice)
        y_true = dataset.rbc_params
        y_true_norm = y_norm
        param_rows.extend(param_metric_rows(selected_variant, seed, False, y_true, pred_raw, y_true_norm, item["pred_norm"], splits))
        epoch_rows.extend(item["epoch_rows"])

    selected_seed = min(seed_scores, key=seed_scores.get)
    all_profile: list[dict[str, Any]] = []
    all_group: list[dict[str, Any]] = []
    for seed, rows in profile_by_seed.items():
        selected = seed == selected_seed
        for row in rows:
            row["selected_seed"] = selected
            all_profile.append(row)
        all_group.extend(group_variant_rows(selected_variant, seed, selected, rows))
        train = aggregate_prediction_rows(rows, selected_variant, "train")
        val = aggregate_prediction_rows(rows, selected_variant, "val")
        test = aggregate_prediction_rows(rows, selected_variant, "test")
        seed_rows.append(
            {
                "variant": selected_variant,
                "seed": seed,
                "selected_seed": selected,
                "best_epoch": [t["best_epoch"] for t in trained if int(t["seed"]) == seed][0],
                "best_val_epoch_score": [t["best_val_epoch_score"] for t in trained if int(t["seed"]) == seed][0],
                "val_selection_score": seed_scores[seed],
                "min_train_normalized_param_mae": [t["min_train_normalized_param_mae"] for t in trained if int(t["seed"]) == seed][0],
                "train_normalized_param_mae": train["normalized_param_mae_mean_mean"],
                "val_normalized_param_mae": val["normalized_param_mae_mean_mean"],
                "test_normalized_param_mae": test["normalized_param_mae_mean_mean"],
                "train_dimension_mae_norm": train["dimension_param_mae_norm_mean"],
                "val_dimension_mae_norm": val["dimension_param_mae_norm_mean"],
                "test_dimension_mae_norm": test["dimension_param_mae_norm_mean"],
                "train_curvature_mae": train["curvature_mae_mean_mean"],
                "val_curvature_mae": val["curvature_mae_mean_mean"],
                "test_curvature_mae": test["curvature_mae_mean_mean"],
                "test_L_mae_mm": test["L_mae_mm_mean"],
                "test_W_mae_mm": test["W_mae_mm_mean"],
                "test_D_mae_mm": test["D_mae_mm_mean"],
                "test_wLD_abs_error": test["wLD_abs_error_mean"],
                "test_wWD_abs_error": test["wWD_abs_error_mean"],
                "test_wLW_abs_error": test["wLW_abs_error_mean"],
                "test_projected_mask_iou": test["projected_mask_iou_mean"],
                "test_projected_mask_dice": test["projected_mask_dice_mean"],
                "test_profile_depth_rmse_m": test["profile_depth_rmse_m_mean"],
            }
        )
    for row in param_rows:
        if int(row["seed"]) == selected_seed:
            row["selected_seed"] = True
    for row in epoch_rows:
        row["selected_seed"] = int(row["seed"]) == selected_seed

    selected_test = [r for r in seed_rows if r["selected_seed"]][0]
    ref = reference_aggregate_values()
    comp = comparison_rows(ref, selected_test)

    write_rows(args.seed_summary, seed_rows, SEED_FIELDS)
    write_rows(args.metrics, param_rows, PARAM_FIELDS)
    write_rows(args.epoch_log, epoch_rows, ["variant", "seed", "selected_seed", "epoch", "train_loss", "train_total_mae", "val_total_mae", "train_dimension_mae", "val_dimension_mae", "train_curvature_mae", "val_curvature_mae", "val_selection_score"])
    write_rows(args.group_summary, all_group, GROUP_FIELDS)
    write_rows(args.profile_metrics, all_profile, PROFILE_FIELDS)
    write_rows(args.comparison, comp, COMPARE_FIELDS)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 curvature refined training summary",
                "",
                f"dataset_id: {args.dataset_id}",
                f"selected_variant_from_screen: {selected_variant}",
                f"seeds: {list(args.seeds)}",
                f"selected_seed_by_validation: {selected_seed}",
                f"selected_val_selection_score: {seed_scores[selected_seed]:.6f}",
                f"test_normalized_mae: {float(selected_test['test_normalized_param_mae']):.6f}",
                f"test_LWD_mae_mm: {float(selected_test['test_L_mae_mm']):.3f}/{float(selected_test['test_W_mae_mm']):.3f}/{float(selected_test['test_D_mae_mm']):.3f}",
                f"test_curvature_mae: {float(selected_test['test_curvature_mae']):.6f}",
                f"test_wLD_wWD_wLW: {float(selected_test['test_wLD_abs_error']):.6f}/{float(selected_test['test_wWD_abs_error']):.6f}/{float(selected_test['test_wLW_abs_error']):.6f}",
                f"test_mask_iou_dice: {float(selected_test['test_projected_mask_iou']):.6f}/{float(selected_test['test_projected_mask_dice']):.6f}",
                f"test_profile_depth_rmse_m: {float(selected_test['test_profile_depth_rmse_m']):.9f}",
                f"curvature_delta_vs_20_77: {float(selected_test['test_curvature_mae']) - ref['curvature_mae']:.6f}",
                f"dimension_delta_vs_20_77: {float(selected_test['test_dimension_mae_norm']) - ref['dimension_mae_norm']:.6f}",
                f"dice_delta_vs_20_77: {float(selected_test['test_projected_mask_dice']) - ref['projected_mask_dice']:.6f}",
                "selection_boundary: validation-only seed selection; test final only.",
                "data_boundary: no COMSOL, no data generation, no NPZ modification, no checkpoint written.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summary", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--seed-summary", type=Path, default=SEED_OUT)
    parser.add_argument("--metrics", type=Path, default=METRICS_OUT)
    parser.add_argument("--epoch-log", type=Path, default=EPOCH_OUT)
    parser.add_argument("--group-summary", type=Path, default=GROUP_OUT)
    parser.add_argument("--profile-metrics", type=Path, default=PROFILE_OUT)
    parser.add_argument("--comparison", type=Path, default=COMPARE_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))

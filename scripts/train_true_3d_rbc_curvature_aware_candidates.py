"""Seed-42 curvature-aware candidate screen for v3_240.

The screen trains only bounded model variants on the explicit v3_240 dataset.
Candidate selection uses validation metrics only; test metrics are reported but
never used to select the variant.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
REFERENCE_SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_seed_summary.csv"
REFERENCE_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_profile_metrics.csv"

SUMMARY_OUT = ROOT / "results/summaries/true_3d_rbc_v3_240_curvature_candidate_screen_summary.txt"
METRICS_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_candidate_screen_metrics.csv"
GROUP_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_candidate_group_summary.csv"
PROFILE_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_candidate_profile_metrics.csv"

METRIC_FIELDS = [
    "variant",
    "seed",
    "selected_by_validation",
    "split",
    "sample_count",
    "selection_score",
    "normalized_param_mae",
    "dimension_mae_norm",
    "curvature_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "curvature_mae",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
    "curvature_delta_vs_reference",
    "dimension_delta_vs_reference",
    "dice_delta_vs_reference",
    "notes",
]

GROUP_FIELDS = [
    "variant",
    "seed",
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
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
]

PROFILE_FIELDS = [
    "variant",
    "seed",
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
]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    stronger_encoder: bool = False
    use_features: bool = False
    curvature_weight: float = 1.0


class SplitHeadRegressor(nn.Module):
    def __init__(self, stronger_encoder: bool = False, feature_dim: int = 0) -> None:
        super().__init__()
        if stronger_encoder:
            channels = [9, 48, 64, 96]
            latent_channels = 96
        else:
            channels = [9, 32, 48, 64]
            latent_channels = 64
        self.encoder = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(channels[1], channels[2], kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[2], channels[3], kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        latent_dim = latent_channels * 8 + feature_dim
        hidden = 128 if stronger_encoder else 96
        self.shared = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU())
        self.dim_head = nn.Sequential(nn.Linear(hidden, 48), nn.GELU(), nn.Linear(48, 3))
        self.curv_head = nn.Sequential(nn.Linear(hidden, 48), nn.GELU(), nn.Linear(48, 3))

    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        latent = torch.flatten(self.encoder(x), 1)
        if features is not None:
            latent = torch.cat([latent, features], dim=1)
        shared = self.shared(latent)
        return torch.cat([self.dim_head(shared), self.curv_head(shared)], dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def predict(model: nn.Module, x: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32)
        ft = torch.as_tensor(features, dtype=torch.float32) if features is not None else None
        return model(xt, ft).cpu().numpy().astype(np.float32)


def components(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.abs(y_true - y_pred)
    return {
        "total": float(err.mean()),
        "dimension": float(err[:, :3].mean()),
        "curvature": float(err[:, 3:].mean()),
    }


def epoch_selection(comp: dict[str, float]) -> float:
    return comp["total"] + 0.50 * comp["curvature"] + 0.20 * comp["dimension"]


def profile_selection_score(val_agg: dict[str, Any], ref_val_dice: float) -> float:
    dice_drop = max(0.0, ref_val_dice - float(val_agg["projected_mask_dice_mean"]))
    return (
        float(val_agg["normalized_param_mae_mean_mean"])
        + 0.50 * float(val_agg["curvature_mae_mean_mean"])
        + 0.20 * float(val_agg["dimension_param_mae_norm_mean"])
        + 0.10 * dice_drop
    )


def loss_fn(pred: torch.Tensor, target: torch.Tensor, curvature_weight: float) -> torch.Tensor:
    dim = torch.nn.functional.smooth_l1_loss(pred[:, :3], target[:, :3])
    curv = torch.nn.functional.smooth_l1_loss(pred[:, 3:], target[:, 3:])
    return dim + float(curvature_weight) * curv


def train_variant(
    config: VariantConfig,
    seed: int,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    splits: dict[str, np.ndarray],
    feature_norm: np.ndarray | None,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    set_seed(seed)
    feature_dim = 0 if not config.use_features or feature_norm is None else feature_norm.shape[1]
    model = SplitHeadRegressor(config.stronger_encoder, feature_dim=feature_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = splits["train"]
    if config.use_features and feature_norm is not None:
        ds = TensorDataset(
            torch.as_tensor(x_norm[train_idx], dtype=torch.float32),
            torch.as_tensor(feature_norm[train_idx], dtype=torch.float32),
            torch.as_tensor(y_norm[train_idx], dtype=torch.float32),
        )
    else:
        ds = TensorDataset(
            torch.as_tensor(x_norm[train_idx], dtype=torch.float32),
            torch.as_tensor(y_norm[train_idx], dtype=torch.float32),
        )
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = math.inf
    min_train = math.inf
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            if config.use_features and feature_norm is not None:
                xb, fb, yb = batch
                pred = model(xb, fb)
            else:
                xb, yb = batch
                pred = model(xb)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(pred, yb, config.curvature_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        pred_train = predict(model, x_norm[train_idx], feature_norm[train_idx] if config.use_features and feature_norm is not None else None)
        pred_val = predict(model, x_norm[splits["val"]], feature_norm[splits["val"]] if config.use_features and feature_norm is not None else None)
        train_comp = components(y_norm[train_idx], pred_train)
        val_comp = components(y_norm[splits["val"]], pred_val)
        score = epoch_selection(val_comp)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        min_train = min(min_train, train_comp["total"])
        epoch_rows.append(
            {
                "variant": config.name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_total_mae": train_comp["total"],
                "val_total_mae": val_comp["total"],
                "train_dimension_mae": train_comp["dimension"],
                "val_dimension_mae": val_comp["dimension"],
                "train_curvature_mae": train_comp["curvature"],
                "val_curvature_mae": val_comp["curvature"],
                "val_selection_score": score,
            }
        )
    if best_state is None:
        raise RuntimeError(f"no best state for {config.name}")
    model.load_state_dict(best_state)
    pred_all = predict(model, x_norm, feature_norm if config.use_features and feature_norm is not None else None)
    return {
        "variant": config.name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_epoch_score": best_score,
        "min_train_normalized_param_mae": min_train,
        "pred_norm": pred_all,
        "epoch_rows": epoch_rows,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reference_profile_rows() -> list[dict[str, Any]]:
    rows = [
        dict(r)
        for r in read_csv(REFERENCE_PROFILE)
        if r.get("selected_seed") == "True"
    ]
    for row in rows:
        row["variant"] = "C0_reference_20_77"
        row["seed"] = 42
        row["selected_by_validation"] = False
        row["L_mae_m"] = float(row["L_mae_mm"]) / 1000.0
        row["W_mae_m"] = float(row["W_mae_mm"]) / 1000.0
        row["D_mae_m"] = float(row["D_mae_mm"]) / 1000.0
        row["clip_applied"] = 1.0 if str(row.get("clip_applied", "")).lower() == "true" else 0.0
    return rows


def reference_val_dice() -> float:
    val = [r for r in reference_profile_rows() if r.get("split") == "val"]
    return float(np.mean([float(r["projected_mask_dice"]) for r in val]))


def aggregate_variant_rows(variant: str, seed: int, selected: bool, rows: list[dict[str, Any]], ref: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for split in ["train", "val", "test"]:
        agg = aggregate_prediction_rows(rows, variant, split)
        if not agg.get("sample_count"):
            continue
        selection = profile_selection_score(agg, ref["val_dice"]) if split == "val" else ""
        out.append(
            {
                "variant": variant,
                "seed": seed,
                "selected_by_validation": selected,
                "split": split,
                "sample_count": agg["sample_count"],
                "selection_score": selection,
                "normalized_param_mae": agg["normalized_param_mae_mean_mean"],
                "dimension_mae_norm": agg["dimension_param_mae_norm_mean"],
                "curvature_mae_norm": agg["curvature_param_mae_norm_mean"],
                "L_mae_mm": agg["L_mae_mm_mean"],
                "W_mae_mm": agg["W_mae_mm_mean"],
                "D_mae_mm": agg["D_mae_mm_mean"],
                "wLD_abs_error": agg["wLD_abs_error_mean"],
                "wWD_abs_error": agg["wWD_abs_error_mean"],
                "wLW_abs_error": agg["wLW_abs_error_mean"],
                "curvature_mae": agg["curvature_mae_mean_mean"],
                "projected_mask_iou": agg["projected_mask_iou_mean"],
                "projected_mask_dice": agg["projected_mask_dice_mean"],
                "profile_depth_rmse_m": agg["profile_depth_rmse_m_mean"],
                "curvature_delta_vs_reference": agg["curvature_mae_mean_mean"] - ref[f"{split}_curvature"],
                "dimension_delta_vs_reference": agg["dimension_param_mae_norm_mean"] - ref[f"{split}_dimension"],
                "dice_delta_vs_reference": agg["projected_mask_dice_mean"] - ref[f"{split}_dice"],
                "notes": "",
            }
        )
    return out


def group_variant_rows(variant: str, seed: int, selected: bool, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for split in ["train", "val", "test"]:
        subset = [r for r in rows if r["split"] == split]
        for field in ["curvature_template", "depth_bin", "aspect_bin"]:
            values = sorted({r[field] for r in subset})
            for value in values:
                group = [r for r in subset if r[field] == value]
                if not group:
                    continue
                def avg(key: str) -> float:
                    return float(np.mean([float(r[key]) for r in group]))
                out.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "selected_by_validation": selected,
                        "split": split,
                        "group_field": field,
                        "group_value": value,
                        "sample_count": len(group),
                        "normalized_param_mae": avg("normalized_param_mae_mean"),
                        "L_mae_mm": avg("L_mae_mm"),
                        "W_mae_mm": avg("W_mae_mm"),
                        "D_mae_mm": avg("D_mae_mm"),
                        "curvature_mae": avg("curvature_mae_mean"),
                        "projected_mask_iou": avg("projected_mask_iou"),
                        "projected_mask_dice": avg("projected_mask_dice"),
                        "profile_depth_rmse_m": avg("profile_depth_rmse_m"),
                    }
                )
    return out


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_reference_values() -> dict[str, float]:
    rows = reference_profile_rows()
    ref: dict[str, float] = {}
    for split in ["train", "val", "test"]:
        agg = aggregate_prediction_rows(rows, "C0_reference_20_77", split)
        ref[f"{split}_curvature"] = float(agg["curvature_mae_mean_mean"])
        ref[f"{split}_dimension"] = float(agg["dimension_param_mae_norm_mean"])
        ref[f"{split}_dice"] = float(agg["projected_mask_dice_mean"])
    ref["val_dice"] = ref["val_dice"]
    return ref


def run_screen(args: argparse.Namespace) -> str:
    outputs = [args.summary, args.metrics, args.group_summary, args.profile_metrics]
    check_no_overwrite(outputs, args.overwrite)
    dataset = load_dataset(args.dataset_id)
    stats = train_normalization(dataset)
    splits = split_indices(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    raw_features, _ = extract_signal_features(dataset.x_channels)
    feature_scaler = fit_feature_scaler(raw_features, splits["train"])
    feature_norm = transform_features(raw_features, feature_scaler)
    ref = build_reference_values()

    variants = [
        VariantConfig("C1_split_heads", curvature_weight=1.0),
        VariantConfig("C2_split_heads_curv_weight_1p5", curvature_weight=1.5),
        VariantConfig("C3_stronger_encoder_feature_fusion", stronger_encoder=True, use_features=True, curvature_weight=1.0),
    ]
    trained = [train_variant(v, args.seed, x_norm, y_norm, splits, feature_norm, args.epochs, args.batch_size, args.lr, args.weight_decay) for v in variants]
    profile_by_variant: dict[str, list[dict[str, Any]]] = {"C0_reference_20_77": reference_profile_rows()}
    for item in trained:
        pred_raw = denormalize_y(item["pred_norm"], stats)
        rows = evaluate_param_predictions(dataset, pred_raw, stats)
        for row in rows:
            row["variant"] = item["variant"]
            row["seed"] = args.seed
            row["selected_by_validation"] = False
        profile_by_variant[item["variant"]] = rows

    metric_rows: list[dict[str, Any]] = []
    for variant, rows in profile_by_variant.items():
        metric_rows.extend(aggregate_variant_rows(variant, args.seed, False, rows, ref))
    candidate_val = [r for r in metric_rows if r["split"] == "val" and r["variant"] != "C0_reference_20_77"]
    selected_variant = min(candidate_val, key=lambda r: float(r["selection_score"]))["variant"]
    for row in metric_rows:
        row["selected_by_validation"] = row["variant"] == selected_variant

    profile_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for variant, rows in profile_by_variant.items():
        selected = variant == selected_variant
        for row in rows:
            row["selected_by_validation"] = selected
            profile_rows.append(row)
        group_rows.extend(group_variant_rows(variant, args.seed, selected, rows))

    write_rows(args.metrics, metric_rows, METRIC_FIELDS)
    write_rows(args.group_summary, group_rows, GROUP_FIELDS)
    write_rows(args.profile_metrics, profile_rows, PROFILE_FIELDS)

    selected_test = [r for r in metric_rows if r["variant"] == selected_variant and r["split"] == "test"][0]
    selected_val = [r for r in metric_rows if r["variant"] == selected_variant and r["split"] == "val"][0]
    ref_test = [r for r in metric_rows if r["variant"] == "C0_reference_20_77" and r["split"] == "test"][0]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 curvature candidate screen summary",
                "",
                f"dataset_id: {args.dataset_id}",
                f"seed: {args.seed}",
                "input: delta_b only; optional C3 features are derived only from delta_b and train-scaled.",
                "candidate_selection: validation-only; test metrics are report-only.",
                f"selected_candidate: {selected_variant}",
                f"selected_val_selection_score: {float(selected_val['selection_score']):.6f}",
                f"selected_test_normalized_mae: {float(selected_test['normalized_param_mae']):.6f}",
                f"selected_test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.3f}/{float(selected_test['W_mae_mm']):.3f}/{float(selected_test['D_mae_mm']):.3f}",
                f"selected_test_curvature_mae: {float(selected_test['curvature_mae']):.6f}",
                f"selected_test_wLD_wWD_wLW: {float(selected_test['wLD_abs_error']):.6f}/{float(selected_test['wWD_abs_error']):.6f}/{float(selected_test['wLW_abs_error']):.6f}",
                f"selected_test_mask_iou_dice: {float(selected_test['projected_mask_iou']):.6f}/{float(selected_test['projected_mask_dice']):.6f}",
                f"selected_test_profile_depth_rmse_m: {float(selected_test['profile_depth_rmse_m']):.9f}",
                "",
                f"reference_test_normalized_mae: {float(ref_test['normalized_param_mae']):.6f}",
                f"reference_test_curvature_mae: {float(ref_test['curvature_mae']):.6f}",
                f"curvature_delta_vs_reference: {float(selected_test['curvature_delta_vs_reference']):.6f}",
                f"dice_delta_vs_reference: {float(selected_test['dice_delta_vs_reference']):.6f}",
                "boundary: no COMSOL, no data generation, no NPZ modification, no checkpoint written.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return selected_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summary", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--metrics", type=Path, default=METRICS_OUT)
    parser.add_argument("--group-summary", type=Path, default=GROUP_OUT)
    parser.add_argument("--profile-metrics", type=Path, default=PROFILE_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_screen(parse_args())

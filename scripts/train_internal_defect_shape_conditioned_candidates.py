#!/usr/bin/env python
"""22.1 shape-conditioned / two-stage internal defect candidate screen.

正式候选只使用 delta_b/BxByBz 与 delta_b-derived features。true shape_type
只用于 supervision/metrics；T4 oracle 单独标记为诊断，不允许被选为正式候选。
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

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


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
B2_REPLAY = ROOT / "results/metrics/internal_defect_b2_inference_replay_metrics.csv"
B2_REFERENCE = ROOT / "results/metrics/internal_defect_shape_conditioned_reference_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_shape_conditioned_candidate_screen_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_shape_conditioned_candidate_screen_metrics.csv"
TAIL_METRICS = ROOT / "results/metrics/internal_defect_shape_conditioned_candidate_tail_metrics.csv"


METRIC_FIELDS = [
    "candidate",
    "candidate_role",
    "selected_candidate",
    "valid_for_multiseed",
    "seed",
    "split",
    "sample_count",
    "selection_score",
    "best_epoch",
    "total_normalized_mae",
    "dimension_mae_mm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_component_mae_mm",
    "center_x_mae_mm",
    "center_y_mae_mm",
    "center_z_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
    "selection_notes",
]

TAIL_FIELDS = [
    "candidate",
    "candidate_role",
    "selected_candidate",
    "valid_for_multiseed",
    "seed",
    "split",
    "sample_count",
    "total_error_mean",
    "total_error_median",
    "total_error_p90",
    "total_error_p95",
    "total_error_max",
    "burial_depth_error_mean_mm",
    "burial_depth_error_median_mm",
    "burial_depth_error_p90_mm",
    "burial_depth_error_p95_mm",
    "burial_depth_error_max_mm",
    "center_xyz_error_mean_mm",
    "center_xyz_error_median_mm",
    "center_xyz_error_p90_mm",
    "center_xyz_error_p95_mm",
    "center_xyz_error_max_mm",
    "dimension_outlier_count",
    "catastrophic_failure_count",
    "catastrophic_failure_rate",
    "geometry_branch_failure_count",
    "geometry_branch_failure_rate",
    "shape_error_rate",
]


CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {
    "T1_shape_latent_conditioned": {"mode": "latent", "oracle": False, "pretrain_shape_epochs": 0, "burial_weight": 3.0, "center_weight": 1.8, "shape_weight": 0.45},
    "T2_two_stage_soft_shape": {"mode": "two_stage", "oracle": False, "pretrain_shape_epochs": 70, "burial_weight": 3.2, "center_weight": 2.0, "shape_weight": 0.45},
    "T3_shape_specific_heads": {"mode": "shape_heads", "oracle": False, "pretrain_shape_epochs": 0, "burial_weight": 3.2, "center_weight": 2.0, "shape_weight": 0.45},
    "T4_oracle_true_shape_diagnostic": {"mode": "shape_heads", "oracle": True, "pretrain_shape_epochs": 0, "burial_weight": 3.2, "center_weight": 2.0, "shape_weight": 0.45},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 22.1 shape-conditioned internal defect candidates.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail-metrics", type=Path, default=TAIL_METRICS)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = indices.copy()
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, order.size, batch_size)]


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def shape_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return classification_metrics(y_true, y_pred)["shape_macro_f1"]


class ShapeConditionedNet(nn.Module):
    def __init__(self, feature_dim: int, mode: str) -> None:
        super().__init__()
        if mode not in {"latent", "two_stage", "shape_heads"}:
            raise ValueError(mode)
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.conv_mlp = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8, 96), nn.GELU(), nn.Dropout(0.05))
        self.feature_mlp = nn.Sequential(nn.Linear(feature_dim, 64), nn.GELU(), nn.Dropout(0.05))
        self.shared = nn.Sequential(nn.Linear(160, 128), nn.GELU(), nn.Dropout(0.05))
        self.shape_head = nn.Linear(128, 3)
        if mode == "latent":
            self.reg_head = nn.Sequential(nn.Linear(128 + 3, 96), nn.GELU(), nn.Linear(96, 7))
        else:
            self.reg_heads = nn.ModuleList([nn.Sequential(nn.Linear(128, 80), nn.GELU(), nn.Linear(80, 7)) for _ in range(3)])

    def latent(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        conv = self.conv_mlp(self.encoder(x))
        feat = self.feature_mlp(features)
        return self.shared(torch.cat([conv, feat], dim=1))

    def forward(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        true_shape: torch.Tensor | None = None,
        oracle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.latent(x, features)
        logits = self.shape_head(hidden)
        probs = torch.softmax(logits, dim=1)
        if self.mode == "latent":
            return self.reg_head(torch.cat([hidden, probs], dim=1)), logits
        head_outputs = torch.stack([head(hidden) for head in self.reg_heads], dim=1)
        if oracle:
            if true_shape is None:
                raise RuntimeError("oracle mode requires true_shape")
            weights = torch.nn.functional.one_hot(true_shape, num_classes=3).float()
        else:
            weights = probs
        return torch.sum(head_outputs * weights.unsqueeze(-1), dim=1), logits


def candidate_role(candidate: str) -> str:
    if candidate == "T0_B2_reference":
        return "reference"
    if candidate.startswith("T4_"):
        return "oracle_diagnostic"
    return "official_candidate"


def per_sample_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, np.ndarray]:
    err = np.abs(y_true - y_pred)
    total = np.mean(err / y_std.reshape(1, -1), axis=1)
    center = np.linalg.norm(err[:, 4:7], axis=1) * 1000.0
    burial = err[:, 3] * 1000.0
    dim_err = err[:, :3] * 1000.0
    dim_rel = dim_err / np.maximum(np.abs(y_true[:, :3]) * 1000.0, 1e-6)
    shape_mis = shape_true != shape_pred
    center_out = center > 3.0
    burial_out = burial > 1.0
    dim_out = (np.max(dim_err, axis=1) > 2.0) | (np.max(dim_rel, axis=1) > 0.30)
    return {
        "total": total,
        "center": center,
        "burial": burial,
        "dimension_outlier": dim_out,
        "catastrophic": center_out & burial_out,
        "geometry_branch": shape_mis & center_out & burial_out,
        "shape_error": shape_mis,
    }


def tail_metrics(
    candidate: str,
    selected: bool,
    valid: bool,
    seed: int,
    split: str,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, Any]:
    errors = per_sample_errors(y_true[idx], y_pred[idx], shape_true[idx], shape_pred[idx], y_std)
    total = quantiles(errors["total"])
    burial = quantiles(errors["burial"])
    center = quantiles(errors["center"])
    n = int(idx.size)
    catastrophic = int(np.sum(errors["catastrophic"]))
    branch = int(np.sum(errors["geometry_branch"]))
    shape_err = int(np.sum(errors["shape_error"]))
    return {
        "candidate": candidate,
        "candidate_role": candidate_role(candidate),
        "selected_candidate": selected,
        "valid_for_multiseed": valid,
        "seed": seed,
        "split": split,
        "sample_count": n,
        "total_error_mean": total["mean"],
        "total_error_median": total["median"],
        "total_error_p90": total["p90"],
        "total_error_p95": total["p95"],
        "total_error_max": total["max"],
        "burial_depth_error_mean_mm": burial["mean"],
        "burial_depth_error_median_mm": burial["median"],
        "burial_depth_error_p90_mm": burial["p90"],
        "burial_depth_error_p95_mm": burial["p95"],
        "burial_depth_error_max_mm": burial["max"],
        "center_xyz_error_mean_mm": center["mean"],
        "center_xyz_error_median_mm": center["median"],
        "center_xyz_error_p90_mm": center["p90"],
        "center_xyz_error_p95_mm": center["p95"],
        "center_xyz_error_max_mm": center["max"],
        "dimension_outlier_count": int(np.sum(errors["dimension_outlier"])),
        "catastrophic_failure_count": catastrophic,
        "catastrophic_failure_rate": catastrophic / n,
        "geometry_branch_failure_count": branch,
        "geometry_branch_failure_rate": branch / n,
        "shape_error_rate": shape_err / n,
    }


def selection_score(metric: dict[str, Any], tail: dict[str, Any]) -> float:
    return float(
        safe_float(metric["total_normalized_mae"])
        + 0.5 * safe_float(metric["burial_depth_mae_mm"])
        + 0.5 * safe_float(tail["center_xyz_error_mean_mm"])
        + 0.3 * safe_float(tail["catastrophic_failure_rate"])
        + 0.2 * safe_float(tail["geometry_branch_failure_rate"])
        + 0.1 * (1.0 - safe_float(metric["shape_accuracy"]))
    )


def metric_row(
    candidate: str,
    selected: bool,
    valid: bool,
    seed: int,
    split: str,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    y_std: np.ndarray,
    score: float | str,
    best_epoch: int | str,
    notes: str,
) -> dict[str, Any]:
    reg = regression_metrics(y_true[idx], y_pred[idx], y_std)
    cls = classification_metrics(shape_true[idx], shape_pred[idx])
    return {
        "candidate": candidate,
        "candidate_role": candidate_role(candidate),
        "selected_candidate": selected,
        "valid_for_multiseed": valid,
        "seed": seed,
        "split": split,
        "sample_count": int(idx.size),
        "selection_score": score,
        "best_epoch": best_epoch,
        "total_normalized_mae": reg["total_normalized_mae"],
        "dimension_mae_mm": reg["dimension_mae_mm"],
        "L_mae_mm": reg["L_mae_mm"],
        "W_mae_mm": reg["W_mae_mm"],
        "D_mae_mm": reg["D_mae_mm"],
        "burial_depth_mae_mm": reg["burial_depth_mae_mm"],
        "center_xyz_component_mae_mm": reg["center_xyz_mae_mm"],
        "center_x_mae_mm": reg["center_x_mae_mm"],
        "center_y_mae_mm": reg["center_y_mae_mm"],
        "center_z_mae_mm": reg["center_z_mae_mm"],
        "shape_accuracy": cls["shape_accuracy"],
        "shape_macro_f1": cls["shape_macro_f1"],
        "selection_notes": notes,
    }


def load_b2_predictions(path: Path, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    by_id = {row["sample_id"]: row for row in rows}
    y_pred: list[list[float]] = []
    shape_pred: list[int] = []
    mapping = {"internal_cuboid": 0, "internal_ellipsoid": 1, "internal_sphere": 2}
    for sample_id in dataset.sample_ids:
        row = by_id[str(sample_id)]
        y_pred.append(
            [
                safe_float(row["pred_L_mm"]) / 1000.0,
                safe_float(row["pred_W_mm"]) / 1000.0,
                safe_float(row["pred_D_mm"]) / 1000.0,
                safe_float(row["pred_burial_depth_mm"]) / 1000.0,
                safe_float(row["pred_center_x_mm"]) / 1000.0,
                safe_float(row["pred_center_y_mm"]) / 1000.0,
                safe_float(row["pred_center_z_mm"]) / 1000.0,
            ]
        )
        shape_pred.append(mapping[row["pred_shape_type"]])
    return np.asarray(y_pred, dtype=np.float32), np.asarray(shape_pred, dtype=np.int64)


def loss_weights(y_norm: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    train_burial = np.abs(y_norm[train_idx, 3])
    train_center = np.linalg.norm(y_norm[train_idx, 4:7], axis=1)
    raw = 1.0 + 0.25 * train_burial + 0.15 * train_center
    out = np.ones(y_norm.shape[0], dtype=np.float32)
    out[train_idx] = np.clip(raw, 1.0, 1.8).astype(np.float32)
    return out


def predict(model: ShapeConditionedNet, x: np.ndarray, features: np.ndarray, shape: np.ndarray | None, oracle: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 64):
            xb = torch.from_numpy(x[start : start + 64])
            fb = torch.from_numpy(features[start : start + 64])
            sb = torch.from_numpy(shape[start : start + 64]) if shape is not None else None
            reg, logits = model(xb, fb, sb, oracle=oracle)
            preds.append(reg.cpu().numpy())
            shapes.append(torch.argmax(logits, dim=1).cpu().numpy())
    shape_pred = np.concatenate(shapes, axis=0).astype(np.int64)
    if oracle and shape is not None:
        shape_pred = shape.copy()
    return np.concatenate(preds, axis=0).astype(np.float32), shape_pred


def train_candidate(
    candidate: str,
    seed: int,
    epochs: int,
    batch_size: int,
    x: np.ndarray,
    features: np.ndarray,
    y_norm: np.ndarray,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    shape: np.ndarray,
    splits: dict[str, np.ndarray],
) -> dict[str, Any]:
    cfg = CANDIDATE_CONFIGS[candidate]
    set_seed(seed)
    model = ShapeConditionedNet(features.shape[1], str(cfg["mode"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    x_t = torch.from_numpy(x)
    f_t = torch.from_numpy(features.astype(np.float32))
    y_t = torch.from_numpy(y_norm.astype(np.float32))
    shape_t = torch.from_numpy(shape.astype(np.int64))
    sample_w = torch.from_numpy(loss_weights(y_norm, splits["train"]))
    param_w = torch.tensor([1.0, 1.0, 1.0, float(cfg["burial_weight"]), float(cfg["center_weight"]), float(cfg["center_weight"]), float(cfg["center_weight"])], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metric: dict[str, Any] = {}
    best_val_tail: dict[str, Any] = {}
    oracle = bool(cfg["oracle"])
    pretrain_shape_epochs = int(cfg["pretrain_shape_epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx in batches(splits["train"], batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            reg, logits = model(x_t[batch_idx], f_t[batch_idx], shape_t[batch_idx] if oracle else None, oracle=oracle)
            if epoch <= pretrain_shape_epochs:
                loss = ce(logits, shape_t[batch_idx])
            else:
                raw_reg = nn.functional.smooth_l1_loss(reg, y_t[batch_idx], reduction="none")
                weighted_reg = raw_reg * param_w.reshape(1, -1) * sample_w[batch_idx].reshape(-1, 1)
                loss = weighted_reg.mean() + float(cfg["shape_weight"]) * ce(logits, shape_t[batch_idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        pred_norm, shape_pred = predict(model, x, features, shape if oracle else None, oracle)
        pred = denormalize_y(pred_norm, y_mean, y_std)
        val_metric = metric_row(candidate, False, False, seed, "val", splits["val"], y_true, pred, shape, shape_pred, y_std.reshape(-1), "", epoch, "")
        val_tail = tail_metrics(candidate, False, False, seed, "val", splits["val"], y_true, pred, shape, shape_pred, y_std.reshape(-1))
        score = selection_score(val_metric, val_tail)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metric = val_metric
            best_val_tail = val_tail
    if best_state is None:
        raise RuntimeError(f"no best state for {candidate}")
    model.load_state_dict(best_state)
    pred_norm, shape_pred = predict(model, x, features, shape if oracle else None, oracle)
    pred = denormalize_y(pred_norm, y_mean, y_std)
    return {
        "candidate": candidate,
        "seed": seed,
        "model": model,
        "pred": pred,
        "shape_pred": shape_pred,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_metric": best_val_metric,
        "best_val_tail": best_val_tail,
        "oracle": oracle,
    }


def reference_rows(dataset: Any, splits: dict[str, np.ndarray], y_std: np.ndarray, b2_pred: np.ndarray, b2_shape_pred: np.ndarray, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    for split, idx in splits.items():
        metrics.append(metric_row("T0_B2_reference", False, False, seed, split, idx, dataset.y_regression, b2_pred, dataset.shape_label, b2_shape_pred, y_std, "", "", "22.0 fixed B2 reference; not retrained"))
        tails.append(tail_metrics("T0_B2_reference", False, False, seed, split, idx, dataset.y_regression, b2_pred, dataset.shape_label, b2_shape_pred, y_std))
    return metrics, tails


def validity(val_metric: dict[str, Any], val_tail: dict[str, Any], b2_val_metric: dict[str, Any], b2_val_tail: dict[str, Any], oracle: bool) -> tuple[bool, str]:
    if oracle:
        return False, "oracle diagnostic 使用 true shape 条件，不能作为正式候选。"
    checks = [
        safe_float(val_tail["catastrophic_failure_count"]) <= safe_float(b2_val_tail["catastrophic_failure_count"]),
        safe_float(val_tail["geometry_branch_failure_count"]) <= safe_float(b2_val_tail["geometry_branch_failure_count"]),
        safe_float(val_tail["center_xyz_error_p95_mm"]) <= safe_float(b2_val_tail["center_xyz_error_p95_mm"]) * 1.25,
        safe_float(val_tail["burial_depth_error_p95_mm"]) <= safe_float(b2_val_tail["burial_depth_error_p95_mm"]) * 1.15,
        safe_float(val_metric["total_normalized_mae"]) <= safe_float(b2_val_metric["total_normalized_mae"]) * 1.20,
        safe_float(val_metric["shape_macro_f1"]) >= max(0.90, safe_float(b2_val_metric["shape_macro_f1"]) - 0.08),
    ]
    notes = (
        f"val catastrophic {val_tail['catastrophic_failure_count']} vs B2 {b2_val_tail['catastrophic_failure_count']}; "
        f"geometry {val_tail['geometry_branch_failure_count']} vs B2 {b2_val_tail['geometry_branch_failure_count']}; "
        f"center_p95 {safe_float(val_tail['center_xyz_error_p95_mm']):.3f}mm vs B2 {safe_float(b2_val_tail['center_xyz_error_p95_mm']):.3f}mm; "
        f"burial_p95 {safe_float(val_tail['burial_depth_error_p95_mm']):.3f}mm vs B2 {safe_float(b2_val_tail['burial_depth_error_p95_mm']):.3f}mm."
    )
    return all(checks), notes


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    x_mean, x_std = train_normalization(dataset.x_channels, splits["train"])
    x = normalize_x(dataset.x_channels, x_mean, x_std)
    y = dataset.y_regression
    y_mean, y_std = train_target_scaler(y, splits["train"])
    y_norm = normalize_y(y, y_mean, y_std)
    features_raw, _feature_names = extract_features(dataset.delta_b)
    features, _, _ = standardize_features(features_raw, splits["train"])
    b2_pred, b2_shape_pred = load_b2_predictions(B2_REPLAY, dataset)
    metric_rows, tail_rows = reference_rows(dataset, splits, y_std.reshape(-1), b2_pred, b2_shape_pred, args.seed)
    b2_val_metric = next(row for row in metric_rows if row["candidate"] == "T0_B2_reference" and row["split"] == "val")
    b2_val_tail = next(row for row in tail_rows if row["candidate"] == "T0_B2_reference" and row["split"] == "val")
    results: list[dict[str, Any]] = []

    for candidate in CANDIDATE_CONFIGS:
        result = train_candidate(candidate, args.seed, args.epochs, args.batch_size, x, features, y_norm, y, y_mean, y_std, dataset.shape_label, splits)
        val_metric = metric_row(candidate, False, False, args.seed, "val", splits["val"], y, result["pred"], dataset.shape_label, result["shape_pred"], y_std.reshape(-1), "", result["best_epoch"], "")
        val_tail = tail_metrics(candidate, False, False, args.seed, "val", splits["val"], y, result["pred"], dataset.shape_label, result["shape_pred"], y_std.reshape(-1))
        valid, notes = validity(val_metric, val_tail, b2_val_metric, b2_val_tail, bool(result["oracle"]))
        result["valid_for_multiseed"] = valid
        result["selection_notes"] = notes
        results.append(result)
        score = selection_score(val_metric, val_tail)
        for split, idx in splits.items():
            metric_rows.append(metric_row(candidate, False, valid, args.seed, split, idx, y, result["pred"], dataset.shape_label, result["shape_pred"], y_std.reshape(-1), score if split == "val" else "", result["best_epoch"], notes))
            tail_rows.append(tail_metrics(candidate, False, valid, args.seed, split, idx, y, result["pred"], dataset.shape_label, result["shape_pred"], y_std.reshape(-1)))

    official = [r for r in results if r["valid_for_multiseed"] and not r["oracle"]]
    selected = min(official, key=lambda r: r["best_score"]) if official else None
    if selected:
        for row in metric_rows:
            if row["candidate"] == selected["candidate"]:
                row["selected_candidate"] = True
        for row in tail_rows:
            if row["candidate"] == selected["candidate"]:
                row["selected_candidate"] = True
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.tail_metrics, tail_rows, TAIL_FIELDS)

    lines = [
        "22.1 shape-conditioned / two-stage internal defect candidate screen",
        f"dataset_id: {args.dataset_id}",
        "seed: 42",
        "输入规则: 正式候选只使用 delta_b/BxByBz 和 train-normalized delta_b-derived features。",
        "禁止输入: true shape_type、burial_bin、size/aspect、split、sample_id。",
        "selection_protocol: validation-only candidate selection；screen 中的 test 指标只作诊断记录，不参与选择。",
        f"B2_val_catastrophic_failure_count: {b2_val_tail['catastrophic_failure_count']}",
        f"B2_val_center_p95_mm: {safe_float(b2_val_tail['center_xyz_error_p95_mm']):.3f}",
    ]
    for result in results:
        test_tail = next(row for row in tail_rows if row["candidate"] == result["candidate"] and row["split"] == "test")
        test_metric = next(row for row in metric_rows if row["candidate"] == result["candidate"] and row["split"] == "test")
        lines.append(
            f"{result['candidate']}: role={candidate_role(result['candidate'])}; valid_for_multiseed={result['valid_for_multiseed']}; "
            f"best_epoch={result['best_epoch']}; test_total={safe_float(test_metric['total_normalized_mae']):.6f}; "
            f"test_catastrophic={test_tail['catastrophic_failure_count']}; test_geometry_branch={test_tail['geometry_branch_failure_count']}; "
            f"test_center_p95={safe_float(test_tail['center_xyz_error_p95_mm']):.3f}mm; test_burial_p95={safe_float(test_tail['burial_depth_error_p95_mm']):.3f}mm."
        )
    if selected:
        lines.append(f"selected_candidate: {selected['candidate']}")
        lines.append("stage_d_allowed: true")
    else:
        lines.append("selected_candidate: none")
        lines.append("stage_d_allowed: false")
    lines.append("current_baseline_update: false")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"selected_candidate": selected["candidate"] if selected else "", "selected_valid": bool(selected)}


def main() -> int:
    args = parse_args()
    run_screen(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Train the 21.2 internal defect neural gate.

Only Bx/By/Bz delta_b is used as model input. Labels and metadata are used for
supervision, validation selection, and metrics.
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
    DATASET_ID,
    PARAM_NAMES,
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


SUMMARY = ROOT / "results/summaries/internal_defect_neural_training_gate_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_neural_seed_summary.csv"
METRICS = ROOT / "results/metrics/internal_defect_neural_metrics.csv"
EPOCH_LOG = ROOT / "results/metrics/internal_defect_neural_epoch_log.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_neural_group_summary.csv"
VS_FEATURE = ROOT / "results/metrics/internal_defect_vs_feature_baseline.csv"
FEATURE_METRICS = ROOT / "results/metrics/internal_defect_feature_baseline_metrics.csv"

METRIC_FIELDS = [
    "candidate",
    "selected_seed",
    "seed",
    "split",
    "sample_count",
    "selection_score",
    "total_normalized_mae",
    "dimension_mae_mm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_mae_mm",
    "center_x_mae_mm",
    "center_y_mae_mm",
    "center_z_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
]
SEED_FIELDS = [
    "candidate",
    "selected_seed",
    "seed",
    "best_epoch",
    "best_val_selection_score",
    "train_total_normalized_mae",
    "val_total_normalized_mae",
    "test_total_normalized_mae",
    "train_shape_accuracy",
    "val_shape_accuracy",
    "test_shape_accuracy",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_burial_depth_mae_mm",
    "test_center_xyz_mae_mm",
]
EPOCH_FIELDS = ["candidate", "seed", "epoch", "train_loss", "val_selection_score", "val_total_normalized_mae", "val_shape_accuracy"]
GROUP_FIELDS = [
    "candidate",
    "selected_seed",
    "seed",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_mae_mm",
    "shape_accuracy",
]
COMPARE_FIELDS = [
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
]


class InternalDefectNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8, 96), nn.GELU(), nn.Dropout(0.05))
        self.reg_head = nn.Linear(96, 7)
        self.shape_head = nn.Linear(96, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.trunk(self.encoder(x))
        return self.reg_head(latent), self.shape_head(latent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train internal defect neural gate.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--epoch-log", type=Path, default=EPOCH_LOG)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--vs-feature", type=Path, default=VS_FEATURE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def selection_score(total_norm_mae: float, shape_acc: float) -> float:
    return float(total_norm_mae + 0.35 * (1.0 - shape_acc))


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = indices.copy()
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, order.size, batch_size)]


def predict(model: nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 64):
            xb = torch.from_numpy(x[start : start + 64])
            reg, logits = model(xb)
            preds.append(reg.cpu().numpy())
            shapes.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32), np.concatenate(shapes, axis=0).astype(np.int64)


def eval_split(
    candidate: str,
    selected_seed: bool,
    seed: int,
    split_name: str,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    y_std: np.ndarray,
    score: float | str = "",
) -> dict[str, Any]:
    reg = regression_metrics(y_true[idx], y_pred[idx], y_std)
    cls = classification_metrics(shape_true[idx], shape_pred[idx])
    return {
        "candidate": candidate,
        "selected_seed": selected_seed,
        "seed": seed,
        "split": split_name,
        "sample_count": int(idx.size),
        "selection_score": score,
        **reg,
        **cls,
    }


def group_rows(
    candidate: str,
    selected_seed: bool,
    seed: int,
    split_name: str,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    y_std: np.ndarray,
    group_values: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, values in group_values.items():
        for value in sorted(set(values[idx].tolist())):
            sub = idx[values[idx] == value]
            reg = regression_metrics(y_true[sub], y_pred[sub], y_std)
            cls = classification_metrics(shape_true[sub], shape_pred[sub])
            rows.append(
                {
                    "candidate": candidate,
                    "selected_seed": selected_seed,
                    "seed": seed,
                    "split": split_name,
                    "group_field": field,
                    "group_value": value,
                    "sample_count": int(sub.size),
                    **reg,
                    "shape_accuracy": cls["shape_accuracy"],
                }
            )
    return rows


def train_one_seed(seed: int, args: argparse.Namespace, x: np.ndarray, y_norm: np.ndarray, shape: np.ndarray, splits: dict[str, np.ndarray]) -> dict[str, Any]:
    set_seed(seed)
    torch.set_num_threads(1)
    model = InternalDefectNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    reg_loss = nn.SmoothL1Loss()
    cls_loss = nn.CrossEntropyLoss()
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y_norm.astype(np.float32))
    shape_t = torch.from_numpy(shape.astype(np.int64))
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = float("inf")
    logs: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_idx in batches(splits["train"], args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            reg, logits = model(x_t[batch_idx])
            loss = reg_loss(reg, y_t[batch_idx]) + 0.35 * cls_loss(logits, shape_t[batch_idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        pred_norm, shape_pred = predict(model, x)
        val_reg = regression_metrics(y_norm[splits["val"]], pred_norm[splits["val"]])
        val_cls = classification_metrics(shape[splits["val"]], shape_pred[splits["val"]])
        score = selection_score(val_reg["total_normalized_mae"], val_cls["shape_accuracy"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        logs.append(
            {
                "candidate": "internal_conv1d_multitask",
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_selection_score": score,
                "val_total_normalized_mae": val_reg["total_normalized_mae"],
                "val_shape_accuracy": val_cls["shape_accuracy"],
            }
        )
    if best_state is None:
        raise RuntimeError("no best neural state was selected")
    model.load_state_dict(best_state)
    return {"seed": seed, "model": model, "best_epoch": best_epoch, "best_score": best_score, "logs": logs}


def read_feature_comparison() -> list[dict[str, Any]]:
    if not FEATURE_METRICS.exists():
        return []
    rows: list[dict[str, Any]] = []
    with FEATURE_METRICS.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "test" and (row.get("model") == "mean_baseline" or row.get("selected_model")):
                rows.append(
                    {
                        "model": row.get("model", ""),
                        "source": "feature_baseline",
                        "selected": bool(row.get("selected_model")),
                        "split": "test",
                        "sample_count": row.get("sample_count", ""),
                        "total_normalized_mae": row.get("total_normalized_mae", ""),
                        "L_mae_mm": row.get("L_mae_mm", ""),
                        "W_mae_mm": row.get("W_mae_mm", ""),
                        "D_mae_mm": row.get("D_mae_mm", ""),
                        "burial_depth_mae_mm": row.get("burial_depth_mae_mm", ""),
                        "center_xyz_mae_mm": row.get("center_xyz_mae_mm", ""),
                        "shape_accuracy": row.get("shape_accuracy", ""),
                        "shape_macro_f1": row.get("shape_macro_f1", ""),
                    }
                )
    return rows


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
    seed_results = [train_one_seed(seed, args, x, y_norm, shape, splits) for seed in [42, 123, 2026]]

    candidate = "internal_conv1d_multitask"
    selected = min(seed_results, key=lambda item: item["best_score"])
    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
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
        pred_norm, shape_pred = predict(result["model"], x)
        pred = denormalize_y(pred_norm, y_mean, y_std)
        split_eval: dict[str, dict[str, Any]] = {}
        for split_name, idx in splits.items():
            score = result["best_score"] if split_name == "val" else ""
            row = eval_split(candidate, is_selected, seed, split_name, idx, y, pred, shape, shape_pred, y_std.reshape(-1), score)
            metric_rows.append(row)
            split_eval[split_name] = row
            if is_selected and split_name == "test":
                selected_test = row
            if is_selected:
                group_summary_rows.extend(group_rows(candidate, is_selected, seed, split_name, idx, y, pred, shape, shape_pred, y_std.reshape(-1), group_values))
        seed_rows.append(
            {
                "candidate": candidate,
                "selected_seed": is_selected,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_val_selection_score": result["best_score"],
                "train_total_normalized_mae": split_eval["train"]["total_normalized_mae"],
                "val_total_normalized_mae": split_eval["val"]["total_normalized_mae"],
                "test_total_normalized_mae": split_eval["test"]["total_normalized_mae"],
                "train_shape_accuracy": split_eval["train"]["shape_accuracy"],
                "val_shape_accuracy": split_eval["val"]["shape_accuracy"],
                "test_shape_accuracy": split_eval["test"]["shape_accuracy"],
                "test_L_mae_mm": split_eval["test"]["L_mae_mm"],
                "test_W_mae_mm": split_eval["test"]["W_mae_mm"],
                "test_D_mae_mm": split_eval["test"]["D_mae_mm"],
                "test_burial_depth_mae_mm": split_eval["test"]["burial_depth_mae_mm"],
                "test_center_xyz_mae_mm": split_eval["test"]["center_xyz_mae_mm"],
            }
        )
        epoch_rows.extend(result["logs"])

    if selected_test is None:
        raise RuntimeError("selected test metrics were not created")
    compare_rows = read_feature_comparison()
    compare_rows.append(
        {
            "model": candidate,
            "source": "neural_gate",
            "selected": True,
            "split": "test",
            "sample_count": selected_test["sample_count"],
            "total_normalized_mae": selected_test["total_normalized_mae"],
            "L_mae_mm": selected_test["L_mae_mm"],
            "W_mae_mm": selected_test["W_mae_mm"],
            "D_mae_mm": selected_test["D_mae_mm"],
            "burial_depth_mae_mm": selected_test["burial_depth_mae_mm"],
            "center_xyz_mae_mm": selected_test["center_xyz_mae_mm"],
            "shape_accuracy": selected_test["shape_accuracy"],
            "shape_macro_f1": selected_test["shape_macro_f1"],
        }
    )

    write_csv(args.seed_summary, seed_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
    write_csv(args.group_summary, group_summary_rows, GROUP_FIELDS)
    write_csv(args.vs_feature, compare_rows, COMPARE_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.2 内部/埋藏缺陷 neural training gate 摘要",
                "",
                f"dataset_id: {args.dataset_id}",
                "模型输入: 仅 delta_b/BxByBz，形状为 (N,9,201)。",
                "监督标签: L/W/D、burial_depth、center_xyz、shape_type，仅用于 loss 和 metrics。",
                "metadata_leakage: false；shape_type、burial_depth_bin、split、sample_id 未作为模型输入。",
                "selection_protocol: 每个 seed 用 validation 选 epoch，再用 validation score 选 seed；test final only。",
                f"seeds: 42, 123, 2026",
                f"selected_seed: {selected['seed']}",
                f"selected_best_epoch: {selected['best_epoch']}",
                f"selected_val_score: {selected['best_score']:.6f}",
                f"test_total_normalized_mae: {float(selected_test['total_normalized_mae']):.6f}",
                f"test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.3f} / {float(selected_test['W_mae_mm']):.3f} / {float(selected_test['D_mae_mm']):.3f}",
                f"test_burial_depth_mae_mm: {float(selected_test['burial_depth_mae_mm']):.3f}",
                f"test_center_xyz_mae_mm: {float(selected_test['center_xyz_mae_mm']):.3f}",
                f"test_shape_accuracy: {float(selected_test['shape_accuracy']):.6f}",
                f"test_shape_macro_f1: {float(selected_test['shape_macro_f1']):.6f}",
                "current_baseline_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

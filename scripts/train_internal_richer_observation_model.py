#!/usr/bin/env python
"""23.1 richer-observation internal defect training gate。"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from load_internal_richer_observation_dataset import (
    DATASET_ID,
    OBSERVATION_CONFIGS,
    ROOT,
    build_inputs,
    denormalize_y,
    load_dataset,
    metric_row,
    normalize_y,
    read_csv,
    selection_score,
    split_indices,
    standardize_matrix,
    target_scaler,
    train_scaler,
    write_csv,
)


SUMMARY = ROOT / "results/summaries/internal_richer_observation_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/internal_richer_observation_seed_summary.csv"
METRICS = ROOT / "results/metrics/internal_richer_observation_metrics.csv"
TAIL_METRICS = ROOT / "results/metrics/internal_richer_observation_tail_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_richer_observation_group_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/internal_richer_observation_vs_reference.csv"
ROUTE_MATRIX = ROOT / "results/metrics/internal_richer_observation_evaluation_decision_matrix.csv"
REFERENCE_METRICS = ROOT / "results/metrics/internal_richer_observation_reference_metrics.csv"

METRIC_FIELDS = [
    "model",
    "observation_config",
    "seed",
    "best_epoch",
    "selected_model",
    "split",
    "sample_count",
    "selection_score",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_component_mae_mm",
    "center_xyz_euclidean_mean_mm",
    "shape_accuracy",
    "shape_macro_f1",
    "center_xyz_error_p95_mm",
    "center_xyz_error_max_mm",
    "burial_depth_error_p95_mm",
    "burial_depth_error_max_mm",
    "catastrophic_failure_count",
    "catastrophic_failure_rate",
    "geometry_branch_failure_count",
    "geometry_branch_failure_rate",
]
TAIL_FIELDS = [
    "model",
    "observation_config",
    "seed",
    "selected_model",
    "split",
    "sample_count",
    "center_xyz_error_mean_mm",
    "center_xyz_error_median_mm",
    "center_xyz_error_p90_mm",
    "center_xyz_error_p95_mm",
    "center_xyz_error_max_mm",
    "burial_depth_error_mean_mm",
    "burial_depth_error_median_mm",
    "burial_depth_error_p90_mm",
    "burial_depth_error_p95_mm",
    "burial_depth_error_max_mm",
    "catastrophic_failure_count",
    "catastrophic_failure_rate",
    "geometry_branch_failure_count",
    "geometry_branch_failure_rate",
    "shape_misclassified_count",
    "full_shift_failure_count",
]
GROUP_FIELDS = [
    "model",
    "observation_config",
    "seed",
    "selected_model",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "total_normalized_mae",
    "burial_depth_mae_mm",
    "center_xyz_component_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
    "catastrophic_failure_count",
    "geometry_branch_failure_count",
]
VS_FIELDS = ["metric", "reference", "selected_model", "reference_value", "selected_value", "delta_selected_minus_reference", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 23.1 richer-observation internal model.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--observation-config", default="")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail-metrics", type=Path, default=TAIL_METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    return parser.parse_args()


class RicherObservationNet(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int, use_features: bool) -> None:
        super().__init__()
        self.use_features = use_features
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(96 * 8, 128),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.feature_mlp = nn.Sequential(nn.Linear(feature_dim, 64), nn.GELU(), nn.Dropout(0.05)) if use_features else None
        head_in = 128 + (64 if use_features else 0)
        self.reg_head = nn.Sequential(nn.Linear(head_in, 96), nn.GELU(), nn.Linear(96, 7))
        self.shape_head = nn.Linear(head_in, 3)

    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        pieces = [latent]
        if self.use_features:
            if features is None or self.feature_mlp is None:
                raise RuntimeError("feature-fusion model requires feature tensor")
            pieces.append(self.feature_mlp(features))
        fused = torch.cat(pieces, dim=1)
        return self.reg_head(fused), self.shape_head(fused)


def recommended_config() -> str:
    rows = read_csv(ROUTE_MATRIX)
    selected = [row for row in rows if row.get("selected_for_23_1_training", "").lower() == "true"]
    return selected[0].get("observation_config", "") if selected else ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = indices.copy()
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, order.size, batch_size)]


def predict(model: nn.Module, x: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 64):
            xb = torch.from_numpy(x[start : start + 64])
            fb = torch.from_numpy(features[start : start + 64])
            reg, logits = model(xb, fb)
            preds.append(reg.detach().cpu().numpy())
            shapes.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32), np.concatenate(shapes).astype(np.int64)


def train_one(model_name: str, seed: int, epochs: int, batch_size: int, x: np.ndarray, features: np.ndarray, y: np.ndarray, shape: np.ndarray, split: np.ndarray, tail_weighted: bool) -> dict[str, Any]:
    splits = split_indices(split)
    x_mean, x_std = train_scaler(x, splits["train"], axes=(0, 2))
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    f_norm, _, _ = standardize_matrix(features, splits["train"])
    y_mean, y_std = target_scaler(y, splits["train"])
    y_norm = normalize_y(y, y_mean, y_std)
    set_seed(seed)
    model = RicherObservationNet(in_channels=x.shape[1], feature_dim=f_norm.shape[1], use_features=model_name != "O1_richer_observation_conv1d")
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    x_t = torch.from_numpy(x_norm)
    f_t = torch.from_numpy(f_norm)
    y_t = torch.from_numpy(y_norm)
    shape_t = torch.from_numpy(shape.astype(np.int64))
    param_w = torch.tensor([1.0, 1.0, 1.0, 2.5 if tail_weighted else 1.8, 1.4 if tail_weighted else 1.0, 1.4 if tail_weighted else 1.0, 1.4 if tail_weighted else 1.0], dtype=torch.float32)
    sample_w = torch.ones(y.shape[0], dtype=torch.float32)
    if tail_weighted:
        train_idx = splits["train"]
        center_z = torch.from_numpy(np.abs((y[:, 6] - y[train_idx, 6].mean()) / (y[train_idx, 6].std() + 1e-8)).astype(np.float32))
        burial_z = torch.from_numpy(np.abs((y[:, 3] - y[train_idx, 3].mean()) / (y[train_idx, 3].std() + 1e-8)).astype(np.float32))
        sample_w[train_idx] = 1.0 + 0.20 * torch.clamp(center_z[train_idx] + burial_z[train_idx], 0.0, 3.0)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx in batches(splits["train"], batch_size, rng):
            opt.zero_grad(set_to_none=True)
            reg, logits = model(x_t[batch_idx], f_t[batch_idx])
            raw = nn.functional.smooth_l1_loss(reg, y_t[batch_idx], reduction="none")
            reg_loss = (raw * param_w.reshape(1, -1) * sample_w[batch_idx].reshape(-1, 1)).mean()
            loss = reg_loss + 0.35 * ce(logits, shape_t[batch_idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pred_norm, shape_pred = predict(model, x_norm, f_norm)
        pred = denormalize_y(pred_norm, y_mean, y_std)
        val = metric_row(model_name, "", "val", splits["val"], y, pred, shape, shape_pred, y_std.reshape(-1))
        score = selection_score(val)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no model state selected")
    model.load_state_dict(best_state)
    pred_norm, shape_pred = predict(model, x_norm, f_norm)
    pred = denormalize_y(pred_norm, y_mean, y_std)
    return {"model": model_name, "seed": seed, "best_epoch": best_epoch, "best_score": best_score, "pred": pred, "shape_pred": shape_pred, "y_std": y_std.reshape(-1)}


def metric_rows_for(result: dict[str, Any], config: str, selected: bool, y: np.ndarray, shape: np.ndarray, split: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for split_name, idx in split_indices(split).items():
        row = metric_row(result["model"], config, split_name, idx, y, result["pred"], shape, result["shape_pred"], result["y_std"])
        row.update({"seed": result["seed"], "best_epoch": result["best_epoch"], "selected_model": selected, "selection_score": selection_score(row)})
        rows.append(row)
        tail_rows.append({key: row.get(key, "") for key in TAIL_FIELDS} | {"model": result["model"], "observation_config": config, "seed": result["seed"], "selected_model": selected, "split": split_name, "sample_count": int(idx.size)})
    return rows, tail_rows


def group_rows(result: dict[str, Any], config: str, selected: bool, dataset: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = {
        "shape_type": dataset.shape_type,
        "burial_depth_level": dataset.burial_depth_level,
        "size_level": dataset.size_level,
        "aspect_bin": dataset.aspect_bin,
    }
    for split_name, split_idx in split_indices(dataset.split).items():
        for field, values in fields.items():
            for value in sorted(set(values[split_idx].tolist())):
                idx = split_idx[values[split_idx] == value]
                row = metric_row(result["model"], config, split_name, idx, dataset.y, result["pred"], dataset.shape_label, result["shape_pred"], result["y_std"])
                rows.append(
                    {
                        "model": result["model"],
                        "observation_config": config,
                        "seed": result["seed"],
                        "selected_model": selected,
                        "split": split_name,
                        "group_field": field,
                        "group_value": value,
                        "sample_count": int(idx.size),
                        "total_normalized_mae": row["total_normalized_mae"],
                        "burial_depth_mae_mm": row["burial_depth_mae_mm"],
                        "center_xyz_component_mae_mm": row["center_xyz_component_mae_mm"],
                        "shape_accuracy": row["shape_accuracy"],
                        "shape_macro_f1": row["shape_macro_f1"],
                        "catastrophic_failure_count": row["catastrophic_failure_count"],
                        "geometry_branch_failure_count": row["geometry_branch_failure_count"],
                    }
                )
    return rows


def reference_compare_rows(selected_metric: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ref_rows = read_csv(REFERENCE_METRICS)
    same_ref = [row for row in ref_rows if row.get("observation_config") == selected_metric["observation_config"] and row.get("split") == "test"]
    refs = [("23.0_ridge_feature_probe_same_config", same_ref[0])] if same_ref else []
    hardcase_refs = read_csv(ROOT / "results/metrics/internal_defect_hardcase_b2_reference_metrics.csv")
    b2 = [row for row in hardcase_refs if row.get("split") == "test" and row.get("subset", "all") == "all"]
    if b2:
        refs.append(("old_B2_v3_hardcase_scope_mismatch", b2[0]))
    for ref_name, ref in refs:
        for metric in ["total_normalized_mae", "burial_depth_mae_mm", "center_xyz_component_mae_mm", "shape_macro_f1"]:
            if metric not in ref or metric not in selected_metric:
                continue
            ref_val = float(ref[metric])
            sel_val = float(selected_metric[metric])
            rows.append(
                {
                    "metric": metric,
                    "reference": ref_name,
                    "selected_model": selected_metric["model"],
                    "reference_value": ref_val,
                    "selected_value": sel_val,
                    "delta_selected_minus_reference": sel_val - ref_val,
                    "notes": "old B2 reference uses v3_hardcase 60-row scope" if "scope_mismatch" in ref_name else "same richer-observation split/config reference",
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    config = args.observation_config or recommended_config()
    if config not in OBSERVATION_CONFIGS:
        raise RuntimeError(f"23.0 route decision did not select an allowed config: {config}")
    dataset = load_dataset(args.dataset_id)
    x, features, variants = build_inputs(dataset, config)
    candidates = [
        ("O1_richer_observation_conv1d", False),
        ("O2_richer_observation_feature_fusion", False),
        ("O3_richer_observation_tail_aware", True),
    ]
    results: list[dict[str, Any]] = []
    for model_name, tail_weighted in candidates:
        for seed in [42, 123, 2026]:
            results.append(train_one(model_name, seed, args.epochs, args.batch_size, x, features, dataset.y, dataset.shape_label, dataset.split, tail_weighted))
    all_metrics: list[dict[str, Any]] = []
    all_tail: list[dict[str, Any]] = []
    selection_rows: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for result in results:
        metrics, tail = metric_rows_for(result, config, False, dataset.y, dataset.shape_label, dataset.split)
        all_metrics.extend(metrics)
        all_tail.extend(tail)
        val = next(row for row in metrics if row["split"] == "val")
        selection_rows.append((selection_score(val), result, val))
    selection_rows.sort(key=lambda item: item[0])
    selected = selection_rows[0][1]
    selected_metrics, selected_tail = metric_rows_for(selected, config, True, dataset.y, dataset.shape_label, dataset.split)
    all_metrics.extend(selected_metrics)
    all_tail.extend(selected_tail)
    group = group_rows(selected, config, True, dataset)
    test_selected = next(row for row in selected_metrics if row["split"] == "test")
    vs_rows = reference_compare_rows(test_selected)
    seed_rows = []
    for score, result, val in selection_rows:
        test = [row for row in all_metrics if row["model"] == result["model"] and row["seed"] == result["seed"] and row["split"] == "test" and not row["selected_model"]][0]
        seed_rows.append(
            {
                "model": result["model"],
                "observation_config": config,
                "seed": result["seed"],
                "best_epoch": result["best_epoch"],
                "selected_model": result is selected,
                "val_selection_score": score,
                "test_total_normalized_mae": test["total_normalized_mae"],
                "test_burial_depth_mae_mm": test["burial_depth_mae_mm"],
                "test_center_xyz_component_mae_mm": test["center_xyz_component_mae_mm"],
                "test_shape_macro_f1": test["shape_macro_f1"],
                "test_catastrophic_failure_count": test["catastrophic_failure_count"],
                "test_geometry_branch_failure_count": test["geometry_branch_failure_count"],
            }
        )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.metrics, all_metrics, METRIC_FIELDS)
    write_csv(args.tail_metrics, all_tail, TAIL_FIELDS)
    write_csv(args.group_summary, group, GROUP_FIELDS)
    write_csv(args.vs_reference, vs_rows, VS_FIELDS)
    write_csv(args.seed_summary, seed_rows, list(seed_rows[0].keys()))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "# 23.1 internal richer-observation training summary",
                "",
                f"- dataset_id: {args.dataset_id}",
                f"- selected_observation_config_from_23_0: {config}",
                f"- selected_variants: {variants}",
                f"- selected_model: {selected['model']}",
                f"- selected_seed: {selected['seed']}",
                f"- best_epoch: {selected['best_epoch']}",
                f"- train/val/test split: {dict((name, int(idx.size)) for name, idx in split_indices(dataset.split).items())}",
                f"- test_total_normalized_mae: {float(test_selected['total_normalized_mae']):.6f}",
                f"- test_L/W/D_mae_mm: {float(test_selected['L_mae_mm']):.3f} / {float(test_selected['W_mae_mm']):.3f} / {float(test_selected['D_mae_mm']):.3f}",
                f"- test_burial_depth_mae_mm: {float(test_selected['burial_depth_mae_mm']):.3f}",
                f"- test_center_xyz_component_mae_mm: {float(test_selected['center_xyz_component_mae_mm']):.3f}",
                f"- test_shape_accuracy/F1: {float(test_selected['shape_accuracy']):.6f} / {float(test_selected['shape_macro_f1']):.6f}",
                f"- test_center_p95/max_mm: {float(test_selected['center_xyz_error_p95_mm']):.3f} / {float(test_selected['center_xyz_error_max_mm']):.3f}",
                f"- test_burial_p95/max_mm: {float(test_selected['burial_depth_error_p95_mm']):.3f} / {float(test_selected['burial_depth_error_max_mm']):.3f}",
                f"- test_catastrophic/geometry_failure_count: {test_selected['catastrophic_failure_count']} / {test_selected['geometry_branch_failure_count']}",
                "- selection: validation-only; test final only.",
                "- input boundary: delta_b/BxByBz, scan_line_mask/sensor_z_m, and delta_b-derived features only; no true shape/burial/size/aspect/split/sample_id model input.",
                "- baseline_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

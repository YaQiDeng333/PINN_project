#!/usr/bin/env python
"""21.6 burial-depth focused controlled candidate screen.

模型输入仅使用 delta_b/BxByBz，以及 B2 中明确允许的 delta_b-derived
features。shape_type、burial bin、size/aspect、split、sample_id 等 metadata
不进入模型输入；labels 只用于 loss 和 metrics。
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
SUMMARY = ROOT / "results/summaries/internal_defect_burial_depth_candidate_screen_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_burial_depth_candidate_screen_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_burial_depth_candidate_group_summary.csv"
REF_NEURAL = ROOT / "results/metrics/internal_defect_v2_neural_metrics.csv"
REF_GROUP = ROOT / "results/metrics/internal_defect_v2_neural_group_summary.csv"
REF_FEATURE = ROOT / "results/metrics/internal_defect_v2_feature_baseline_metrics.csv"


METRIC_FIELDS = [
    "candidate",
    "selected_candidate",
    "valid_for_multiseed",
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
    "selection_notes",
]

GROUP_FIELDS = [
    "candidate",
    "selected_candidate",
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


CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {
    "B1_burial_weighted_loss": {
        "feature_fusion": False,
        "shape_conditioned": False,
        "burial_weight": 4.0,
        "description": "same Conv1D encoder; burial_depth loss weight increased",
    },
    "B2_feature_fusion_burial_head": {
        "feature_fusion": True,
        "shape_conditioned": False,
        "burial_weight": 3.5,
        "description": "Conv1D latent plus delta_b-derived feature MLP feeding burial head",
    },
    "B3_shape_conditioned_burial_head": {
        "feature_fusion": False,
        "shape_conditioned": True,
        "burial_weight": 3.0,
        "description": "burial head conditioned on predicted shape logits, not true shape labels",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 21.6 burial-depth candidate screen.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = indices.copy()
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, order.size, batch_size)]


class BurialDepthNet(nn.Module):
    def __init__(self, feature_dim: int = 0, feature_fusion: bool = False, shape_conditioned: bool = False) -> None:
        super().__init__()
        self.feature_fusion = feature_fusion
        self.shape_conditioned = shape_conditioned
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
        self.feature_mlp = nn.Sequential(nn.Linear(feature_dim, 64), nn.GELU(), nn.Dropout(0.05)) if feature_fusion else None
        self.shape_head = nn.Linear(96, 3)
        self.reg_base = nn.Linear(96, 7)
        burial_in = 96
        if feature_fusion:
            burial_in += 64
        if shape_conditioned:
            burial_in += 3
        self.burial_head = nn.Sequential(nn.Linear(burial_in, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.trunk(self.encoder(x))
        logits = self.shape_head(latent)
        reg = self.reg_base(latent)
        pieces = [latent]
        if self.feature_fusion:
            if features is None or self.feature_mlp is None:
                raise RuntimeError("feature_fusion candidate requires feature tensor")
            pieces.append(self.feature_mlp(features))
        if self.shape_conditioned:
            pieces.append(logits)
        burial = self.burial_head(torch.cat(pieces, dim=1))
        reg = reg.clone()
        reg[:, 3:4] = burial
        return reg, logits


def candidate_selection_score(reg: dict[str, float], cls: dict[str, float]) -> float:
    return float(
        reg["burial_depth_mae_mm"]
        + 0.15 * reg["total_normalized_mae"]
        + 0.05 * reg["center_xyz_mae_mm"]
        + 1.0 * (1.0 - cls["shape_macro_f1"])
    )


def predict(model: nn.Module, x: np.ndarray, features: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 64):
            xb = torch.from_numpy(x[start : start + 64])
            fb = torch.from_numpy(features[start : start + 64]) if features is not None else None
            reg, logits = model(xb, fb)
            preds.append(reg.cpu().numpy())
            shapes.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32), np.concatenate(shapes, axis=0).astype(np.int64)


def metric_row(
    candidate: str,
    selected_candidate: bool,
    valid_for_multiseed: bool,
    seed: int,
    split_name: str,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    y_std: np.ndarray,
    score: float | str,
    notes: str,
) -> dict[str, Any]:
    reg = regression_metrics(y_true[idx], y_pred[idx], y_std)
    cls = classification_metrics(shape_true[idx], shape_pred[idx])
    return {
        "candidate": candidate,
        "selected_candidate": selected_candidate,
        "valid_for_multiseed": valid_for_multiseed,
        "seed": seed,
        "split": split_name,
        "sample_count": int(idx.size),
        "selection_score": score,
        **reg,
        **cls,
        "selection_notes": notes,
    }


def group_rows(
    candidate: str,
    selected_candidate: bool,
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
            if sub.size == 0:
                continue
            reg = regression_metrics(y_true[sub], y_pred[sub], y_std)
            cls = classification_metrics(shape_true[sub], shape_pred[sub])
            rows.append(
                {
                    "candidate": candidate,
                    "selected_candidate": selected_candidate,
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


def load_reference_metrics() -> dict[str, dict[str, dict[str, float]]]:
    refs: dict[str, dict[str, dict[str, float]]] = {"B0_reference_neural": {}, "feature_baseline_svr_rbf_C10": {}}
    for row in read_csv(REF_NEURAL):
        if row.get("selected_seed") == "True":
            refs["B0_reference_neural"][row["split"]] = {k: safe_float(v) for k, v in row.items() if k not in {"candidate", "selected_seed", "split"}}
    for row in read_csv(REF_FEATURE):
        if row.get("model") == "svr_rbf_C10":
            refs["feature_baseline_svr_rbf_C10"][row["split"]] = {k: safe_float(v) for k, v in row.items() if k not in {"model", "selected_model", "split"}}
    return refs


def load_reference_group_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(REF_GROUP):
        if row.get("selected_seed") != "True":
            continue
        rows.append(
            {
                "candidate": "B0_reference_neural",
                "selected_candidate": False,
                "seed": int(safe_float(row.get("seed"), 42)),
                "split": row.get("split", ""),
                "group_field": row.get("group_field", ""),
                "group_value": row.get("group_value", ""),
                "sample_count": row.get("sample_count", ""),
                "total_normalized_mae": row.get("total_normalized_mae", ""),
                "L_mae_mm": row.get("L_mae_mm", ""),
                "W_mae_mm": row.get("W_mae_mm", ""),
                "D_mae_mm": row.get("D_mae_mm", ""),
                "burial_depth_mae_mm": row.get("burial_depth_mae_mm", ""),
                "center_xyz_mae_mm": row.get("center_xyz_mae_mm", ""),
                "shape_accuracy": row.get("shape_accuracy", ""),
            }
        )
    return rows


def train_one_candidate(
    candidate: str,
    seed: int,
    epochs: int,
    batch_size: int,
    x: np.ndarray,
    y_norm: np.ndarray,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    shape: np.ndarray,
    splits: dict[str, np.ndarray],
    features: np.ndarray,
) -> dict[str, Any]:
    config = CANDIDATE_CONFIGS[candidate]
    set_seed(seed)
    torch.set_num_threads(1)
    feature_input = features if config["feature_fusion"] else None
    model = BurialDepthNet(
        feature_dim=features.shape[1],
        feature_fusion=bool(config["feature_fusion"]),
        shape_conditioned=bool(config["shape_conditioned"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    cls_loss = nn.CrossEntropyLoss()
    x_t = torch.from_numpy(x)
    f_t = torch.from_numpy(features.astype(np.float32)) if config["feature_fusion"] else None
    y_t = torch.from_numpy(y_norm.astype(np.float32))
    shape_t = torch.from_numpy(shape.astype(np.int64))
    weights = torch.tensor([1.0, 1.0, 1.0, float(config["burial_weight"]), 1.0, 1.0, 1.0], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = float("inf")
    best_val: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx in batches(splits["train"], batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            fb = f_t[batch_idx] if f_t is not None else None
            reg, logits = model(x_t[batch_idx], fb)
            raw_reg = nn.functional.smooth_l1_loss(reg, y_t[batch_idx], reduction="none")
            reg_loss = (raw_reg * weights.reshape(1, -1)).mean()
            loss = reg_loss + 0.35 * cls_loss(logits, shape_t[batch_idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        pred_norm, shape_pred = predict(model, x, feature_input)
        pred = denormalize_y(pred_norm, y_mean, y_std)
        val_reg = regression_metrics(y_true[splits["val"]], pred[splits["val"]], y_std.reshape(-1))
        val_cls = classification_metrics(shape[splits["val"]], shape_pred[splits["val"]])
        score = candidate_selection_score(val_reg, val_cls)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_val = {**val_reg, **val_cls}
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"no best state selected for {candidate}")
    model.load_state_dict(best_state)
    pred_norm, shape_pred = predict(model, x, feature_input)
    pred = denormalize_y(pred_norm, y_mean, y_std)
    return {
        "candidate": candidate,
        "seed": seed,
        "model": model,
        "pred": pred,
        "shape_pred": shape_pred,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val": best_val,
        "feature_input": feature_input,
    }


def b0_rows(refs: dict[str, dict[str, dict[str, float]]], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        ref = refs["B0_reference_neural"].get(split_name, {})
        rows.append(
            {
                "candidate": "B0_reference_neural",
                "selected_candidate": False,
                "valid_for_multiseed": False,
                "seed": seed,
                "split": split_name,
                "sample_count": int(ref.get("sample_count", 0)),
                "selection_score": "",
                "total_normalized_mae": ref.get("total_normalized_mae", ""),
                "dimension_mae_mm": ref.get("dimension_mae_mm", ""),
                "L_mae_mm": ref.get("L_mae_mm", ""),
                "W_mae_mm": ref.get("W_mae_mm", ""),
                "D_mae_mm": ref.get("D_mae_mm", ""),
                "burial_depth_mae_mm": ref.get("burial_depth_mae_mm", ""),
                "center_xyz_mae_mm": ref.get("center_xyz_mae_mm", ""),
                "center_x_mae_mm": ref.get("center_x_mae_mm", ""),
                "center_y_mae_mm": ref.get("center_y_mae_mm", ""),
                "center_z_mae_mm": ref.get("center_z_mae_mm", ""),
                "shape_accuracy": ref.get("shape_accuracy", ""),
                "shape_macro_f1": ref.get("shape_macro_f1", ""),
                "selection_notes": "21.4 selected neural reference; not retrained in this screen",
            }
        )
    return rows


def candidate_validity(candidate_val: dict[str, Any], b0_val: dict[str, float]) -> tuple[bool, str]:
    val_burial = safe_float(candidate_val.get("burial_depth_mae_mm"))
    val_total = safe_float(candidate_val.get("total_normalized_mae"))
    val_center = safe_float(candidate_val.get("center_xyz_mae_mm"))
    val_f1 = safe_float(candidate_val.get("shape_macro_f1"))
    b0_burial = safe_float(b0_val.get("burial_depth_mae_mm"))
    b0_total = safe_float(b0_val.get("total_normalized_mae"))
    b0_center = safe_float(b0_val.get("center_xyz_mae_mm"))
    checks = [
        val_burial <= b0_burial * 1.05,
        val_total <= b0_total * 1.20,
        val_center <= b0_center * 1.25,
        val_f1 >= 0.95,
    ]
    notes = (
        f"val_burial={val_burial:.3f}mm vs B0={b0_burial:.3f}mm; "
        f"val_total={val_total:.6f}; val_center={val_center:.3f}mm; val_shape_f1={val_f1:.3f}"
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
    shape = dataset.shape_label
    feature_raw, _feature_names = extract_features(dataset.delta_b)
    features, _, _ = standardize_features(feature_raw, splits["train"])
    refs = load_reference_metrics()
    b0_val = refs["B0_reference_neural"]["val"]
    metric_rows = b0_rows(refs, args.seed)
    group_summary_rows = load_reference_group_rows()
    candidate_results: list[dict[str, Any]] = []

    group_values = {
        "shape_type": dataset.shape_type,
        "burial_depth_level": dataset.burial_depth_level,
        "size_level": dataset.size_level,
        "aspect_bin": dataset.aspect_bin,
    }

    for candidate in CANDIDATE_CONFIGS:
        result = train_one_candidate(candidate, args.seed, args.epochs, args.batch_size, x, y_norm, y, y_mean, y_std, shape, splits, features)
        val_row = metric_row(
            candidate,
            False,
            False,
            args.seed,
            "val",
            splits["val"],
            y,
            result["pred"],
            shape,
            result["shape_pred"],
            y_std.reshape(-1),
            result["best_score"],
            "",
        )
        valid, notes = candidate_validity(val_row, b0_val)
        result["valid_for_multiseed"] = valid
        result["selection_notes"] = notes
        candidate_results.append(result)
        for split_name, idx in splits.items():
            row = metric_row(
                candidate,
                False,
                valid,
                args.seed,
                split_name,
                idx,
                y,
                result["pred"],
                shape,
                result["shape_pred"],
                y_std.reshape(-1),
                result["best_score"] if split_name == "val" else "",
                notes,
            )
            metric_rows.append(row)
        for split_name, idx in splits.items():
            group_summary_rows.extend(
                group_rows(candidate, False, args.seed, split_name, idx, y, result["pred"], shape, result["shape_pred"], y_std.reshape(-1), group_values)
            )

    valid_results = [r for r in candidate_results if r["valid_for_multiseed"]]
    selected: dict[str, Any] | None = None
    if valid_results:
        selected = min(
            valid_results,
            key=lambda r: (
                safe_float(r["best_val"].get("burial_depth_mae_mm")),
                safe_float(r["best_val"].get("total_normalized_mae")),
            ),
        )
        for row in metric_rows:
            if row["candidate"] == selected["candidate"]:
                row["selected_candidate"] = True
        for row in group_summary_rows:
            if row["candidate"] == selected["candidate"]:
                row["selected_candidate"] = True

    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_summary_rows, GROUP_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "21.6 internal defect burial-depth candidate screen",
        f"dataset_id: {args.dataset_id}",
        "input_policy: B0/B1/B3 use only delta_b/BxByBz; B2 additionally uses train-normalized delta_b-derived features.",
        "forbidden_inputs: labels, true shape_type, burial_depth_bin, size_bin, aspect_bin, split, sample_id.",
        "selection_protocol: seed=42 screen; validation-only candidate selection; test final only after validation selection.",
        f"reference_B0_val_burial_depth_mae_mm: {safe_float(b0_val.get('burial_depth_mae_mm')):.3f}",
    ]
    for result in candidate_results:
        val = result["best_val"]
        test_row = next(row for row in metric_rows if row["candidate"] == result["candidate"] and row["split"] == "test")
        lines.append(
            f"{result['candidate']}: valid_for_multiseed={result['valid_for_multiseed']}; "
            f"best_epoch={result['best_epoch']}; val_burial={safe_float(val.get('burial_depth_mae_mm')):.3f}mm; "
            f"test_burial={safe_float(test_row.get('burial_depth_mae_mm')):.3f}mm; "
            f"test_total={safe_float(test_row.get('total_normalized_mae')):.6f}; "
            f"test_shape_f1={safe_float(test_row.get('shape_macro_f1')):.6f}."
        )
    if selected:
        lines.append(f"selected_candidate: {selected['candidate']}")
        lines.append("stage_d_allowed: true")
    else:
        lines.append("selected_candidate: none")
        lines.append("stage_d_allowed: false")
    lines.append("current_baseline_update: false")
    lines.append("note: 非 selected candidate 的 test 指标只作为诊断记录；candidate 选择只使用 validation burial_depth MAE 及 total/center/shape guard，没有用 test 反选。")
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "selected_candidate": selected["candidate"] if selected else "",
        "selected_valid": bool(selected),
        "candidate_results": candidate_results,
    }


def main() -> int:
    args = parse_args()
    run_screen(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

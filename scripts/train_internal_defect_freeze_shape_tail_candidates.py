#!/usr/bin/env python
"""22.5 seed=42 freeze-shape tail-regression candidate screen.

Formal candidates use frozen B2 encoder/shape outputs plus delta_b-derived
features. True shape labels are used only for supervision/metrics; F4 is an
oracle diagnostic and is never a selectable model.
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from internal_defect_hardcase_utils import (
    B2_MANIFEST,
    DATASET_ID,
    METRIC_FIELDS,
    TAIL_FIELDS,
    metric_row,
    prediction_rows,
    prepare_dataset,
    safe_float,
    tail_row,
)
from load_internal_defect_pilot_dataset import ROOT, classification_metrics, denormalize_y, regression_metrics, write_csv
from train_internal_defect_burial_depth_candidates import BurialDepthNet, predict as predict_b2_net
from train_internal_defect_feature_baselines import extract_features, standardize_features


SUMMARY = ROOT / "results/summaries/internal_defect_freeze_shape_candidate_screen_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_freeze_shape_candidate_screen_metrics.csv"
TAIL = ROOT / "results/metrics/internal_defect_freeze_shape_candidate_tail_metrics.csv"

TAIL_IDXS = np.asarray([3, 4, 5, 6], dtype=np.int64)
LWD_IDXS = np.asarray([0, 1, 2], dtype=np.int64)

SCREEN_FIELDS = METRIC_FIELDS + ["candidate_role", "valid_for_multiseed", "selection_notes"]
SCREEN_TAIL_FIELDS = TAIL_FIELDS + ["candidate_role", "best_epoch", "valid_for_multiseed", "selection_notes"]

FORMAL_CANDIDATES = {
    "F1_freeze_encoder_shape_train_tail_heads",
    "F2_freeze_shape_train_center_burial_with_residual",
    "F3_soft_shape_conditioned_tail_heads",
}


@dataclass
class FrozenB2Context:
    model: BurialDepthNet
    x: np.ndarray
    features: np.ndarray
    y: np.ndarray
    y_norm: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    shape: np.ndarray
    base_norm: np.ndarray
    base_pred: np.ndarray
    base_shape: np.ndarray
    base_logits: np.ndarray
    latent: np.ndarray
    feature_latent: np.ndarray
    splits: dict[str, np.ndarray]
    dataset: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 22.5 freeze-shape candidate screen.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail", type=Path, default=TAIL)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = indices.copy()
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, order.size, batch_size)]


def load_frozen_b2_context(dataset_id: str = DATASET_ID) -> FrozenB2Context:
    prepared = prepare_dataset(dataset_id)
    dataset = prepared["dataset"]
    features_raw, _ = extract_features(dataset.delta_b)
    features, _, _ = standardize_features(features_raw, prepared["splits"]["train"])
    checkpoint = torch.load(Path(__import__("json").loads(B2_MANIFEST.read_text(encoding="utf-8"))["checkpoint_path"]), map_location="cpu", weights_only=False)
    model = BurialDepthNet(feature_dim=int(features.shape[1]), feature_fusion=True, shape_conditioned=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    x_t = torch.from_numpy(prepared["x"])
    f_t = torch.from_numpy(features.astype(np.float32))
    with torch.no_grad():
        latent = model.trunk(model.encoder(x_t))
        feature_latent = model.feature_mlp(f_t)
        base_norm_t, logits_t = model(x_t, f_t)
    base_norm = base_norm_t.cpu().numpy().astype(np.float32)
    return FrozenB2Context(
        model=model,
        x=prepared["x"],
        features=features.astype(np.float32),
        y=prepared["y"],
        y_norm=prepared["y_norm"],
        y_mean=prepared["y_mean"],
        y_std=prepared["y_std"],
        shape=dataset.shape_label,
        base_norm=base_norm,
        base_pred=denormalize_y(base_norm, prepared["y_mean"], prepared["y_std"]),
        base_shape=torch.argmax(logits_t, dim=1).cpu().numpy().astype(np.int64),
        base_logits=logits_t.cpu().numpy().astype(np.float32),
        latent=latent.cpu().numpy().astype(np.float32),
        feature_latent=feature_latent.cpu().numpy().astype(np.float32),
        splits=prepared["splits"],
        dataset=dataset,
    )


class TailResidualHead(nn.Module):
    def __init__(self, input_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.shared = nn.Sequential(nn.Linear(input_dim, 96), nn.GELU(), nn.Dropout(0.05), nn.Linear(96, 64), nn.GELU())
        if mode in {"F3_soft_shape_conditioned_tail_heads", "F4_true_shape_oracle_tail_diagnostic"}:
            self.heads = nn.ModuleList([nn.Linear(64, 4) for _ in range(3)])
        else:
            self.head = nn.Linear(64, 4)

    def forward(self, z: torch.Tensor, shape_probs: torch.Tensor | None = None) -> torch.Tensor:
        h = self.shared(z)
        if self.mode in {"F3_soft_shape_conditioned_tail_heads", "F4_true_shape_oracle_tail_diagnostic"}:
            if shape_probs is None:
                raise RuntimeError("shape probabilities required")
            stacked = torch.stack([head(h) for head in self.heads], dim=1)
            return torch.sum(stacked * shape_probs.unsqueeze(-1), dim=1)
        return self.head(h)


def feature_matrix(ctx: FrozenB2Context, candidate: str) -> tuple[np.ndarray, np.ndarray | None]:
    probs = torch.softmax(torch.from_numpy(ctx.base_logits), dim=1).numpy().astype(np.float32)
    true_onehot = np.eye(3, dtype=np.float32)[ctx.shape]
    if candidate == "F1_freeze_encoder_shape_train_tail_heads":
        z = np.concatenate([ctx.latent, ctx.base_logits, ctx.feature_latent], axis=1)
        return z.astype(np.float32), None
    if candidate == "F2_freeze_shape_train_center_burial_with_residual":
        z = np.concatenate([ctx.base_norm, ctx.base_logits, ctx.feature_latent], axis=1)
        return z.astype(np.float32), None
    if candidate == "F3_soft_shape_conditioned_tail_heads":
        z = np.concatenate([ctx.latent, ctx.base_norm, ctx.feature_latent], axis=1)
        return z.astype(np.float32), probs
    if candidate == "F4_true_shape_oracle_tail_diagnostic":
        z = np.concatenate([ctx.latent, ctx.base_norm, ctx.feature_latent], axis=1)
        return z.astype(np.float32), true_onehot
    raise ValueError(candidate)


def sample_weights(ctx: FrozenB2Context) -> np.ndarray:
    err = np.abs(ctx.y - ctx.base_pred)
    center = np.linalg.norm(err[:, 4:7], axis=1) * 1000.0
    burial = err[:, 3] * 1000.0
    weights = np.ones(ctx.y.shape[0], dtype=np.float32)
    train = ctx.splits["train"]
    hard = train[(center[train] > 3.0) | (burial[train] > 1.0)]
    weights[hard] = 1.75
    return weights


def apply_residual(ctx: FrozenB2Context, residual: np.ndarray) -> np.ndarray:
    out = ctx.base_norm.copy()
    out[:, TAIL_IDXS] = out[:, TAIL_IDXS] + residual
    return denormalize_y(out, ctx.y_mean, ctx.y_std)


def train_tail_candidate(candidate: str, seed: int, epochs: int, batch_size: int, ctx: FrozenB2Context) -> dict[str, Any]:
    set_seed(seed)
    z, shape_probs = feature_matrix(ctx, candidate)
    head = TailResidualHead(z.shape[1], candidate)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2.0e-3, weight_decay=5e-4)
    z_t = torch.from_numpy(z)
    target_residual = torch.from_numpy((ctx.y_norm[:, TAIL_IDXS] - ctx.base_norm[:, TAIL_IDXS]).astype(np.float32))
    w_t = torch.from_numpy(sample_weights(ctx))
    probs_t = torch.from_numpy(shape_probs) if shape_probs is not None else None
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        head.train()
        for batch in batches(ctx.splits["train"], batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            pb = probs_t[batch] if probs_t is not None else None
            residual = head(z_t[batch], pb)
            raw = nn.functional.smooth_l1_loss(residual, target_residual[batch], reduction="none")
            loss = (raw * torch.tensor([2.0, 1.0, 1.0, 1.0]).reshape(1, -1) * w_t[batch].reshape(-1, 1)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
        pred, shape_pred = predict_tail(head, z, probs_t, ctx)
        val_metric = metric_row(candidate, False, seed, "val", "all", ctx.splits["val"], ctx.y, pred, ctx.shape, shape_pred, ctx.y_std.reshape(-1))
        val_tail = tail_row(candidate, False, seed, "val", "all", ctx.splits["val"], ctx.y, pred, ctx.shape, shape_pred, ctx.y_std.reshape(-1))
        score = selection_score(val_metric, val_tail)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
    if best_state is None:
        raise RuntimeError(f"no best state for {candidate}")
    head.load_state_dict(best_state)
    pred, shape_pred = predict_tail(head, z, probs_t, ctx)
    return {"candidate": candidate, "seed": seed, "head": head, "pred": pred, "shape_pred": shape_pred, "best_epoch": best_epoch, "best_score": best_score}


def predict_tail(head: TailResidualHead, z: np.ndarray, probs_t: torch.Tensor | None, ctx: FrozenB2Context) -> tuple[np.ndarray, np.ndarray]:
    head.eval()
    residuals: list[np.ndarray] = []
    with torch.no_grad():
        z_t = torch.from_numpy(z)
        for start in range(0, z.shape[0], 128):
            pb = probs_t[start : start + 128] if probs_t is not None else None
            residuals.append(head(z_t[start : start + 128], pb).cpu().numpy())
    residual = np.concatenate(residuals, axis=0).astype(np.float32)
    return apply_residual(ctx, residual), ctx.base_shape.copy()


def selection_score(metric: dict[str, Any], tail: dict[str, Any]) -> float:
    return float(
        safe_float(tail["center_xyz_error_p95_mm"])
        + 0.25 * safe_float(tail["center_xyz_error_max_mm"])
        + safe_float(tail["burial_depth_error_p95_mm"])
        + 0.25 * safe_float(tail["burial_depth_error_max_mm"])
        + 6.0 * safe_float(tail["catastrophic_failure_rate"])
        + 3.0 * safe_float(tail["geometry_branch_failure_rate"])
        + 2.0 * max(0.0, 0.82 - safe_float(metric["shape_macro_f1"]))
    )


def official_guard(candidate: str, val_metric: dict[str, Any], val_tail: dict[str, Any], b2_metric: dict[str, Any], b2_tail: dict[str, Any]) -> tuple[bool, str]:
    if candidate == "F4_true_shape_oracle_tail_diagnostic":
        return False, "oracle diagnostic only; true shape labels are not allowed as formal model input"
    checks = [
        safe_float(val_metric["shape_macro_f1"]) >= safe_float(b2_metric["shape_macro_f1"]) - 0.05,
        safe_float(val_tail["center_xyz_error_p95_mm"]) < safe_float(b2_tail["center_xyz_error_p95_mm"]),
        safe_float(val_tail["center_xyz_error_max_mm"]) < safe_float(b2_tail["center_xyz_error_max_mm"]),
        safe_float(val_tail["burial_depth_error_p95_mm"]) <= safe_float(b2_tail["burial_depth_error_p95_mm"]) * 1.05,
        safe_float(val_tail["burial_depth_error_max_mm"]) <= safe_float(b2_tail["burial_depth_error_max_mm"]) * 1.10,
        safe_float(val_metric["total_normalized_mae"]) <= safe_float(b2_metric["total_normalized_mae"]) * 1.15,
    ]
    note = (
        f"shape_f1={safe_float(val_metric['shape_macro_f1']):.3f}; "
        f"center_p95/max={safe_float(val_tail['center_xyz_error_p95_mm']):.3f}/{safe_float(val_tail['center_xyz_error_max_mm']):.3f}; "
        f"burial_p95/max={safe_float(val_tail['burial_depth_error_p95_mm']):.3f}/{safe_float(val_tail['burial_depth_error_max_mm']):.3f}; "
        f"total={safe_float(val_metric['total_normalized_mae']):.6f}"
    )
    return all(checks), note


def rows_for_candidate(result: dict[str, Any], selected: bool, valid: bool, note: str, ctx: FrozenB2Context) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    role = "oracle_diagnostic" if result["candidate"] == "F4_true_shape_oracle_tail_diagnostic" else "official_candidate"
    for split, idx in ctx.splits.items():
        metric = metric_row(result["candidate"], selected, result["seed"], split, "all", idx, ctx.y, result["pred"], ctx.shape, result["shape_pred"], ctx.y_std.reshape(-1), result["best_score"] if split == "val" else "", result["best_epoch"])
        tail = tail_row(result["candidate"], selected, result["seed"], split, "all", idx, ctx.y, result["pred"], ctx.shape, result["shape_pred"], ctx.y_std.reshape(-1))
        metric.update({"candidate_role": role, "best_epoch": result["best_epoch"], "valid_for_multiseed": valid, "selection_notes": note})
        tail.update({"candidate_role": role, "best_epoch": result["best_epoch"], "valid_for_multiseed": valid, "selection_notes": note})
        metrics.append(metric)
        tails.append(tail)
    return metrics, tails


def selected_summary(metrics: list[dict[str, Any]], tails: list[dict[str, Any]]) -> tuple[str, bool]:
    candidates = []
    for metric in metrics:
        if metric["split"] == "val" and metric["candidate_role"] == "official_candidate" and metric["valid_for_multiseed"]:
            tail = next(row for row in tails if row["model"] == metric["model"] and row["split"] == "val")
            candidates.append((selection_score(metric, tail), metric["model"]))
    if not candidates:
        return "", False
    return sorted(candidates)[0][1], True


def main() -> int:
    args = parse_args()
    ctx = load_frozen_b2_context(args.dataset_id)
    b2_val_metric = metric_row("F0_B2_reference", False, 2026, "val", "all", ctx.splits["val"], ctx.y, ctx.base_pred, ctx.shape, ctx.base_shape, ctx.y_std.reshape(-1))
    b2_val_tail = tail_row("F0_B2_reference", False, 2026, "val", "all", ctx.splits["val"], ctx.y, ctx.base_pred, ctx.shape, ctx.base_shape, ctx.y_std.reshape(-1))
    all_metrics: list[dict[str, Any]] = []
    all_tails: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for candidate in [
        "F1_freeze_encoder_shape_train_tail_heads",
        "F2_freeze_shape_train_center_burial_with_residual",
        "F3_soft_shape_conditioned_tail_heads",
        "F4_true_shape_oracle_tail_diagnostic",
    ]:
        result = train_tail_candidate(candidate, args.seed, args.epochs, args.batch_size, ctx)
        val_metric = metric_row(candidate, False, args.seed, "val", "all", ctx.splits["val"], ctx.y, result["pred"], ctx.shape, result["shape_pred"], ctx.y_std.reshape(-1))
        val_tail = tail_row(candidate, False, args.seed, "val", "all", ctx.splits["val"], ctx.y, result["pred"], ctx.shape, result["shape_pred"], ctx.y_std.reshape(-1))
        valid, note = official_guard(candidate, val_metric, val_tail, b2_val_metric, b2_val_tail)
        result["valid_for_multiseed"] = valid
        result["selection_notes"] = note
        all_results.append(result)
    valid_candidates = [r for r in all_results if r["valid_for_multiseed"] and r["candidate"] in FORMAL_CANDIDATES]
    selected_name = ""
    if valid_candidates:
        selected_name = min(
            valid_candidates,
            key=lambda r: selection_score(
                metric_row(r["candidate"], False, args.seed, "val", "all", ctx.splits["val"], ctx.y, r["pred"], ctx.shape, r["shape_pred"], ctx.y_std.reshape(-1)),
                tail_row(r["candidate"], False, args.seed, "val", "all", ctx.splits["val"], ctx.y, r["pred"], ctx.shape, r["shape_pred"], ctx.y_std.reshape(-1)),
            ),
        )["candidate"]
    for result in all_results:
        metrics, tails = rows_for_candidate(result, result["candidate"] == selected_name, bool(result["valid_for_multiseed"]), result["selection_notes"], ctx)
        all_metrics.extend(metrics)
        all_tails.extend(tails)
    write_csv(args.metrics, all_metrics, SCREEN_FIELDS)
    write_csv(args.tail, all_tails, SCREEN_TAIL_FIELDS)
    lines = [
        "22.5 freeze-shape tail-regression candidate screen",
        f"dataset_id: {args.dataset_id}",
        f"seed: {args.seed}",
        "frozen_base: 21.9 B2 encoder/trunk/shape classifier/regression output; no B2 parameter updates.",
        "formal_input_policy: frozen B2 latent/logits/predictions plus delta_b-derived feature latent only; no true shape/split/sample_id metadata as formal input.",
        "oracle_policy: F4 uses true shape only as diagnostic and cannot be selected.",
        f"selected_candidate: {selected_name if selected_name else 'none'}",
        f"stage_d_allowed: {bool(selected_name)}",
    ]
    for result in all_results:
        test_metric = next(row for row in all_metrics if row["model"] == result["candidate"] and row["split"] == "test")
        test_tail = next(row for row in all_tails if row["model"] == result["candidate"] and row["split"] == "test")
        lines.append(
            f"{result['candidate']}: valid={result['valid_for_multiseed']}; best_epoch={result['best_epoch']}; "
            f"test_total={safe_float(test_metric['total_normalized_mae']):.6f}; "
            f"shape_f1={safe_float(test_metric['shape_macro_f1']):.6f}; "
            f"center_p95/max={safe_float(test_tail['center_xyz_error_p95_mm']):.3f}/{safe_float(test_tail['center_xyz_error_max_mm']):.3f}; "
            f"burial_p95/max={safe_float(test_tail['burial_depth_error_p95_mm']):.3f}/{safe_float(test_tail['burial_depth_error_max_mm']):.3f}; "
            f"cat={test_tail['catastrophic_failure_count']}; geometry={test_tail['geometry_branch_failure_count']}; note={result['selection_notes']}"
        )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

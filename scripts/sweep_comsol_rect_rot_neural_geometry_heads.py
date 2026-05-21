from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = base.DEFAULT_NPZ
DEFAULT_LABELS = base.DEFAULT_LABELS
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_geometry_head_sweep_summary.txt"
DEFAULT_AUDIT = PROJECT_ROOT / "results/summaries/comsol_rect_rot_geometry_head_sweep_failure_audit_summary.txt"
DEFAULT_CANDIDATES = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_head_sweep_candidates.csv"
DEFAULT_SELECTED_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_head_sweep_selected_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_head_sweep_epoch_log.csv"
DEFAULT_GROUP_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_head_sweep_group_summary.csv"
DEFAULT_GEOMETRY_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_head_sweep_geometry_summary.csv"
DEFAULT_FAILURE_CASES = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_head_sweep_failure_cases.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_rect_rot_geometry_head_sweep_best"

SEED = 42
EPOCHS = 150
BATCH_SIZE = 8
MAX_ANGLE_RAD = math.radians(35.0)
RESIDUAL_SCALE = 0.25
RESIDUAL_L2_WEIGHT = 1e-4
THRESHOLDS = base.THRESHOLDS

REF_2048_IOU = 0.5908
REF_2048_DICE = 0.7385
REF_2048_TYPE_ACC = 0.6364
REF_2048_ANGLE_MAE = 20.14
REF_2049_IOU = 0.5702
REF_2049_DICE = 0.7207
REF_2049_TYPE_ACC = 0.6061
REF_2049_ANGLE_MAE = 19.12
PIAO_WEAK_IOU = base.PIAO_WEAK_IOU
PIAO_WEAK_DICE = base.PIAO_WEAK_DICE
PIAO_WEAK_TYPE_ACC = base.PIAO_WEAK_TYPE_ACC
PIAO_WEAK_ANGLE_MAE = base.PIAO_WEAK_ANGLE_MAE
DENSE_SINGLE_BASELINE_IOU = base.DENSE_SINGLE_BASELINE_IOU
DENSE_SINGLE_BASELINE_DICE = base.DENSE_SINGLE_BASELINE_DICE


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    angle_mode: str
    routing: str
    residual: bool
    type_weight: float
    angle_weight: float


CANDIDATES = [
    CandidateConfig("shared_sincos_softmix", "sincos", "softmix", False, 0.3, 0.1),
    CandidateConfig("shared_boundedangle_softmix", "bounded", "softmix", False, 0.3, 0.15),
    CandidateConfig("shared_boundedangle_strongtype", "bounded", "softmix", False, 0.6, 0.15),
    CandidateConfig("shared_trunk_type_residual", "bounded", "softmix", True, 0.5, 0.2),
    CandidateConfig("unified_rotated_box_no_typemix", "bounded", "unified", False, 0.3, 0.15),
]

METRIC_FIELDS = ["candidate"] + base.METRIC_FIELDS
EPOCH_FIELDS = [
    "candidate",
    "epoch",
    "train_loss",
    "train_bce_loss",
    "train_dice_loss",
    "train_type_loss",
    "train_center_loss",
    "train_size_loss",
    "train_depth_loss",
    "train_angle_loss",
    "train_residual_l2_loss",
    "best_val_threshold",
    "val_iou",
    "val_dice",
    "val_area_error",
    "val_type_accuracy",
    "val_angle_mae_deg",
    "val_score",
]

CANDIDATE_FIELDS = [
    "candidate",
    "best_epoch",
    "selected_threshold",
    "val_score",
    "val_iou",
    "val_dice",
    "val_area_error",
    "val_type_accuracy",
    "val_rect_precision",
    "val_rect_recall",
    "val_rotated_precision",
    "val_rotated_recall",
    "val_center_mae_m",
    "val_width_mae_m",
    "val_length_mae_m",
    "val_depth_mae_m",
    "val_angle_mae_deg",
    "train_iou",
    "train_dice",
    "train_area_error",
    "train_type_accuracy",
    "score_formula",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CandidateModel(nn.Module):
    def __init__(self, config: CandidateConfig, latent_dim: int = 128):
        super().__init__()
        self.config = config
        self.encoder = base.BzEncoder(latent_dim=latent_dim)
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.type_head = nn.Linear(64, 2)
        out_dim = 7 if config.angle_mode == "sincos" else 6
        self.geom_head = nn.Linear(64, out_dim)
        if config.residual:
            self.rect_residual = nn.Linear(64, out_dim)
            self.rot_residual = nn.Linear(64, out_dim)
        else:
            self.rect_residual = None
            self.rot_residual = None

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(self.encoder(signal))
        type_logits = self.type_head(hidden)
        type_prob = torch.softmax(type_logits, dim=1)
        raw = self.geom_head(hidden)
        residual_l2 = raw.new_tensor(0.0)
        if self.config.residual:
            rect_res = self.rect_residual(hidden)
            rot_res = self.rot_residual(hidden)
            weighted = type_prob[:, 0:1] * rect_res + type_prob[:, 1:2] * rot_res
            raw = raw + RESIDUAL_SCALE * weighted
            residual_l2 = weighted.square().mean()

        geom_norm = raw[:, :5]
        if self.config.angle_mode == "sincos":
            angle_raw = raw[:, 5:7]
            angle_norm = torch.sqrt(angle_raw.square().sum(dim=1, keepdim=True) + 1e-8)
            angle_sincos = angle_raw / angle_norm
            angle_rad = torch.atan2(angle_sincos[:, 0], angle_sincos[:, 1])
            angle_repr = angle_sincos
        else:
            bounded = torch.tanh(raw[:, 5]) * MAX_ANGLE_RAD
            angle_rad = bounded
            angle_repr = (bounded / MAX_ANGLE_RAD).unsqueeze(1)
        return {
            "type_logits": type_logits,
            "type_prob": type_prob,
            "geom_norm": geom_norm,
            "angle_rad": angle_rad,
            "angle_repr": angle_repr,
            "residual_l2": residual_l2,
        }


def decode_outputs(
    outputs: dict[str, torch.Tensor],
    arrays: dict[str, Any],
    config: CandidateConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    geom_mean = torch.tensor(arrays["geom_mean"], device=device).view(1, -1)
    geom_std = torch.tensor(arrays["geom_std"], device=device).view(1, -1)
    raw_geom = outputs["geom_norm"] * geom_std + geom_mean
    mask_x = torch.tensor(arrays["mask_x"], device=device)
    mask_y = torch.tensor(arrays["mask_y"], device=device)
    cx = raw_geom[:, 0].clamp(float(mask_x.min()), float(mask_x.max()))
    cy = raw_geom[:, 1].clamp(float(mask_y.min()), float(mask_y.max()))
    width = raw_geom[:, 2].clamp(0.001, 0.025)
    length = raw_geom[:, 3].clamp(0.001, 0.020)
    depth = raw_geom[:, 4].clamp(0.0001, 0.004)
    type_prob = outputs["type_prob"]
    rect_prob = base.soft_rect_mask(mask_x, mask_y, cx, cy, width, length, torch.zeros_like(outputs["angle_rad"]))
    rot_prob = base.soft_rect_mask(mask_x, mask_y, cx, cy, width, length, outputs["angle_rad"])
    if config.routing == "unified":
        mask_prob = rot_prob
    else:
        mask_prob = type_prob[:, 0].view(-1, 1, 1) * rect_prob + type_prob[:, 1].view(-1, 1, 1) * rot_prob
    return {
        "raw_geom": torch.stack([cx, cy, width, length, depth], dim=1),
        "angle_rad": outputs["angle_rad"],
        "type_prob": type_prob,
        "mask_prob": mask_prob,
        "mask_logits": torch.logit(mask_prob.clamp(1e-4, 1.0 - 1e-4)),
    }


def batch_loss(
    model_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    arrays: dict[str, Any],
    config: CandidateConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    decoded = decode_outputs(model_outputs, arrays, config, device)
    target_mask = batch["mask"].to(device)
    bce = F.binary_cross_entropy_with_logits(decoded["mask_logits"], target_mask)
    dice = base.soft_dice_loss(decoded["mask_prob"], target_mask)
    type_loss = F.cross_entropy(model_outputs["type_logits"], batch["type_target"].to(device))
    geom_target = batch["geom_target"].to(device)
    center_loss = F.smooth_l1_loss(model_outputs["geom_norm"][:, :2], geom_target[:, :2])
    size_loss = F.smooth_l1_loss(model_outputs["geom_norm"][:, 2:4], geom_target[:, 2:4])
    depth_loss = F.smooth_l1_loss(model_outputs["geom_norm"][:, 4:5], geom_target[:, 4:5])
    if config.angle_mode == "sincos":
        angle_target = batch["angle_target"].to(device)
        angle_loss = F.smooth_l1_loss(model_outputs["angle_repr"], angle_target)
    else:
        true_angle = torch.atan2(batch["angle_target"].to(device)[:, 0], batch["angle_target"].to(device)[:, 1])
        angle_loss = F.smooth_l1_loss(model_outputs["angle_repr"].squeeze(1), true_angle / MAX_ANGLE_RAD)
    residual_l2 = model_outputs["residual_l2"]
    loss = (
        bce
        + dice
        + config.type_weight * type_loss
        + 0.15 * center_loss
        + 0.15 * size_loss
        + 0.05 * depth_loss
        + config.angle_weight * angle_loss
        + (RESIDUAL_L2_WEIGHT * residual_l2 if config.residual else 0.0)
    )
    return loss, {
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "type": float(type_loss.detach().cpu()),
        "center": float(center_loss.detach().cpu()),
        "size": float(size_loss.detach().cpu()),
        "depth": float(depth_loss.detach().cpu()),
        "angle": float(angle_loss.detach().cpu()),
        "residual_l2": float(residual_l2.detach().cpu()),
    }


def predict(
    model: nn.Module,
    dataset: base.RectRotDataset,
    arrays: dict[str, Any],
    config: CandidateConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["signal"].to(device))
            decoded = decode_outputs(outputs, arrays, config, device)
            chunks["indices"].append(batch["source_index"].cpu().numpy())
            chunks["mask_prob"].append(decoded["mask_prob"].cpu().numpy())
            chunks["type_prob"].append(decoded["type_prob"].cpu().numpy())
            chunks["raw_geom"].append(decoded["raw_geom"].cpu().numpy())
            chunks["angle_rad"].append(decoded["angle_rad"].cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def metric_rows(
    candidate: str,
    pred: dict[str, np.ndarray],
    arrays: dict[str, Any],
    split: str,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, local_idx_raw in enumerate(pred["indices"]):
        local_idx = int(local_idx_raw)
        true_geom = arrays["raw_geom"][local_idx]
        pred_geom = pred["raw_geom"][order]
        true_angle = math.atan2(float(arrays["angle_targets"][local_idx, 0]), float(arrays["angle_targets"][local_idx, 1]))
        pred_angle = float(pred["angle_rad"][order])
        true_angle_deg = math.degrees(true_angle)
        pred_angle_deg = math.degrees(pred_angle)
        type_prob = pred["type_prob"][order]
        pred_type = base.TYPE_NAMES[int(np.argmax(type_prob))]
        true_type = str(arrays["defect_types"][local_idx])
        mask_metric = base.mask_metric(pred["mask_prob"][order], arrays["masks"][local_idx], threshold)
        angle_error = (
            base.circular_angle_error_deg(pred_angle_deg, true_angle_deg)
            if true_type == "rotated_rect"
            else math.nan
        )
        sin_cos_error = float(
            np.linalg.norm(np.array([math.sin(pred_angle), math.cos(pred_angle)]) - arrays["angle_targets"][local_idx])
        )
        rows.append(
            {
                "candidate": candidate,
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": split,
                "defect_type": true_type,
                "source_pack": str(arrays["source_packs"][local_idx]),
                "threshold": threshold,
                **mask_metric,
                "type_prob_rectangular_notch": float(type_prob[0]),
                "type_prob_rotated_rect": float(type_prob[1]),
                "pred_defect_type": pred_type,
                "type_correct": float(pred_type == true_type),
                "true_center_x": float(true_geom[0]),
                "true_center_y": float(true_geom[1]),
                "pred_center_x": float(pred_geom[0]),
                "pred_center_y": float(pred_geom[1]),
                "center_mae_m": float(math.hypot(pred_geom[0] - true_geom[0], pred_geom[1] - true_geom[1])),
                "true_width": float(true_geom[2]),
                "pred_width": float(pred_geom[2]),
                "width_abs_error_m": float(abs(pred_geom[2] - true_geom[2])),
                "true_length": float(true_geom[3]),
                "pred_length": float(pred_geom[3]),
                "length_abs_error_m": float(abs(pred_geom[3] - true_geom[3])),
                "true_depth": float(true_geom[4]),
                "pred_depth": float(pred_geom[4]),
                "depth_abs_error_m": float(abs(pred_geom[4] - true_geom[4])),
                "true_angle_deg": true_angle_deg,
                "pred_angle_deg": pred_angle_deg,
                "angle_abs_error_deg": angle_error,
                "sin_cos_error": sin_cos_error,
                "notes": "",
            }
        )
    return rows


def score_pred(pred: dict[str, np.ndarray], arrays: dict[str, Any], threshold: float) -> dict[str, float]:
    rows = metric_rows("score", pred, arrays, "val", threshold)
    stats = base.split_summary(rows, "val")
    score = stats["iou"] + stats["dice"] - stats["area_error"] + 0.10 * stats["type_accuracy"] - 0.003 * stats["angle_mae_deg"]
    return {
        "threshold": threshold,
        "iou": stats["iou"],
        "dice": stats["dice"],
        "area_error": stats["area_error"],
        "type_accuracy": stats["type_accuracy"],
        "angle_mae_deg": stats["angle_mae_deg"],
        "score": score,
    }


def best_threshold(pred: dict[str, np.ndarray], arrays: dict[str, Any]) -> dict[str, float]:
    best = {"score": -math.inf}
    for threshold in THRESHOLDS:
        current = score_pred(pred, arrays, threshold)
        if current["score"] > best["score"]:
            best = current
    return best


def type_report(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    return base.type_report(rows, split)


def candidate_summary_row(
    candidate: str,
    best_epoch: int,
    best_val: dict[str, float],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    val_stats = base.split_summary(val_rows, "val")
    train_stats = base.split_summary(train_rows, "train")
    val_type = type_report(val_rows, "val")
    return {
        "candidate": candidate,
        "best_epoch": best_epoch,
        "selected_threshold": best_val["threshold"],
        "val_score": best_val["score"],
        "val_iou": val_stats["iou"],
        "val_dice": val_stats["dice"],
        "val_area_error": val_stats["area_error"],
        "val_type_accuracy": val_stats["type_accuracy"],
        "val_rect_precision": val_type["rect_precision"],
        "val_rect_recall": val_type["rect_recall"],
        "val_rotated_precision": val_type["rotated_precision"],
        "val_rotated_recall": val_type["rotated_recall"],
        "val_center_mae_m": val_stats["center_mae_m"],
        "val_width_mae_m": val_stats["width_mae_m"],
        "val_length_mae_m": val_stats["length_mae_m"],
        "val_depth_mae_m": val_stats["depth_mae_m"],
        "val_angle_mae_deg": val_stats["angle_mae_deg"],
        "train_iou": train_stats["iou"],
        "train_dice": train_stats["dice"],
        "train_area_error": train_stats["area_error"],
        "train_type_accuracy": train_stats["type_accuracy"],
        "score_formula": "IoU + Dice - area_error + 0.10*type_accuracy - 0.003*rotated_angle_mae_deg",
    }


def train_candidate(
    config: CandidateConfig,
    arrays: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor], CandidateModel]:
    set_seed(args.seed)
    train_ds = base.RectRotDataset(arrays["split_indices"]["train"], arrays)
    val_ds = base.RectRotDataset(arrays["split_indices"]["val"], arrays)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = CandidateModel(config, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val = {"score": -math.inf, "threshold": 0.5}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        train_loss = 0.0
        n_batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["signal"].to(device))
            loss, loss_parts = batch_loss(outputs, batch, arrays, config, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            for key, value in loss_parts.items():
                totals[key] += value
            n_batches += 1
        val_pred = predict(model, val_ds, arrays, config, device, args.batch_size)
        val_best = best_threshold(val_pred, arrays)
        if val_best["score"] > best_val["score"]:
            best_val = val_best
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "candidate": config.name,
                "epoch": epoch,
                "train_loss": train_loss / max(n_batches, 1),
                "train_bce_loss": totals["bce"] / max(n_batches, 1),
                "train_dice_loss": totals["dice"] / max(n_batches, 1),
                "train_type_loss": totals["type"] / max(n_batches, 1),
                "train_center_loss": totals["center"] / max(n_batches, 1),
                "train_size_loss": totals["size"] / max(n_batches, 1),
                "train_depth_loss": totals["depth"] / max(n_batches, 1),
                "train_angle_loss": totals["angle"] / max(n_batches, 1),
                "train_residual_l2_loss": totals["residual_l2"] / max(n_batches, 1),
                "best_val_threshold": val_best["threshold"],
                "val_iou": val_best["iou"],
                "val_dice": val_best["dice"],
                "val_area_error": val_best["area_error"],
                "val_type_accuracy": val_best["type_accuracy"],
                "val_angle_mae_deg": val_best["angle_mae_deg"],
                "val_score": val_best["score"],
            }
        )
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"{config.name} epoch={epoch:03d} loss={epoch_rows[-1]['train_loss']:.4f} "
                f"val_score={val_best['score']:.4f} thr={val_best['threshold']:.2f}"
            )

    if best_state is None:
        raise RuntimeError(f"No validation checkpoint selected for {config.name}")
    model.load_state_dict(best_state)
    train_pred = predict(model, train_ds, arrays, config, device, args.batch_size)
    val_pred = predict(model, val_ds, arrays, config, device, args.batch_size)
    train_rows = metric_rows(config.name, train_pred, arrays, "train", float(best_val["threshold"]))
    val_rows = metric_rows(config.name, val_pred, arrays, "val", float(best_val["threshold"]))
    summary = candidate_summary_row(config.name, best_epoch, best_val, train_rows, val_rows)
    return summary, epoch_rows, best_state, model


def select_best(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidate_rows,
        key=lambda row: (
            base.to_float(row["val_score"]),
            base.to_float(row["val_type_accuracy"]),
            -base.to_float(row["val_angle_mae_deg"]),
            base.to_float(row["val_dice"]),
        ),
        reverse=True,
    )[0]


def build_group_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows, geometry_rows = base.build_group_summaries(rows)
    return group_rows, geometry_rows


def failure_category(row: dict[str, Any]) -> str:
    return base.failure_category(row)


def build_failure_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in sorted([r for r in rows if r["split"] == "test"], key=lambda item: base.to_float(item["iou"]))[:40]:
        out = dict(row)
        out["failure_category"] = failure_category(row)
        cases.append(out)
    return cases


def preview(rows: list[dict[str, Any]], arrays: dict[str, Any], preview_dir: Path, max_count: int = 24) -> int:
    return base.preview(rows, arrays, preview_dir, max_count=max_count)


def write_summary(
    args: argparse.Namespace,
    diagnostics: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    selected_name: str,
    selected_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    preview_count: int,
    device: torch.device,
) -> dict[str, Any]:
    train_stats = base.split_summary(selected_rows, "train")
    val_stats = base.split_summary(selected_rows, "val")
    test_stats = base.split_summary(selected_rows, "test")
    test_type = type_report(selected_rows, "test")
    selected = next(row for row in candidate_rows if row["candidate"] == selected_name)
    promising = (
        test_stats["type_accuracy"] >= 0.72
        and test_stats["angle_mae_deg"] <= 18.0
        and (test_stats["iou"] >= REF_2048_IOU + 0.015 or test_stats["dice"] >= REF_2048_DICE + 0.012)
        and test_stats["area_error"] <= 0.28
    )
    recommendation = "A. Add lightweight forward consistency." if promising else "B. Feature-assisted geometry head."

    lines = [
        "COMSOL rect/rot neural geometry head controlled architecture sweep summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Input labels: {args.labels}",
        f"Device: {device}",
        "Scope: rectangular_notch + rotated_rect only; polygon parsed/reported but excluded.",
        "Input policy: delta_bz only. No defect_type / geometry_params / source_pack / angle / vertex_count as model input.",
        "Supervision: type labels, geometry labels, and true masks are labels only.",
        "Rasterizer: fixed PyTorch soft rotated-rectangle SDF, temperature=0.0005.",
        "This is a controlled architecture sweep, not a baseline.",
        "",
        "Stage A gate:",
        f"- rect+rotated N = {diagnostics['n_rect_rot']}",
        f"- split counts = {diagnostics['split_counts']}",
        "- rasterizer validation passed separately with true-geometry IoU=1.0000.",
        "",
        "Candidates:",
    ]
    for row in candidate_rows:
        lines.append(
            f"- {row['candidate']}: val_score={base.to_float(row['val_score']):.4f}, "
            f"val IoU/Dice={base.to_float(row['val_iou']):.4f}/{base.to_float(row['val_dice']):.4f}, "
            f"type_acc={base.to_float(row['val_type_accuracy']):.4f}, "
            f"angle_mae={base.to_float(row['val_angle_mae_deg']):.2f}, "
            f"threshold={row['selected_threshold']}, epoch={row['best_epoch']}"
        )
    lines.extend(
        [
            "",
            f"Selected best candidate: {selected_name}",
            "Selection source: validation only.",
            "Selection objective: highest val_score; tie-breakers are val type accuracy, lower angle MAE, higher Dice.",
            f"Selected validation row: {selected}",
            "",
            "Selected candidate metrics:",
            f"- train IoU/Dice/area_error = {train_stats['iou']:.4f} / {train_stats['dice']:.4f} / {train_stats['area_error']:.4f}",
            f"- val IoU/Dice/area_error = {val_stats['iou']:.4f} / {val_stats['dice']:.4f} / {val_stats['area_error']:.4f}",
            f"- test IoU/Dice/area_error = {test_stats['iou']:.4f} / {test_stats['dice']:.4f} / {test_stats['area_error']:.4f}",
            f"- test type accuracy = {test_stats['type_accuracy']:.4f}",
            f"- test rect precision / recall = {test_type['rect_precision']:.4f} / {test_type['rect_recall']:.4f}",
            f"- test rotated precision / recall = {test_type['rotated_precision']:.4f} / {test_type['rotated_recall']:.4f}",
            f"- test center MAE m = {test_stats['center_mae_m']:.6f}",
            f"- test width MAE m = {test_stats['width_mae_m']:.6f}",
            f"- test length MAE m = {test_stats['length_mae_m']:.6f}",
            f"- test depth MAE m = {test_stats['depth_mae_m']:.6f}",
            f"- test rotated angle MAE deg = {test_stats['angle_mae_deg']:.4f}",
            "",
            "Comparisons:",
            f"- 20.48 test IoU/Dice/type/angle = {REF_2048_IOU:.4f} / {REF_2048_DICE:.4f} / {REF_2048_TYPE_ACC:.4f} / {REF_2048_ANGLE_MAE:.2f}",
            f"- 20.49 test IoU/Dice/type/angle = {REF_2049_IOU:.4f} / {REF_2049_DICE:.4f} / {REF_2049_TYPE_ACC:.4f} / {REF_2049_ANGLE_MAE:.2f}",
            f"- selected gain vs 20.48 IoU/Dice/type/angle-improvement = {test_stats['iou'] - REF_2048_IOU:.4f} / {test_stats['dice'] - REF_2048_DICE:.4f} / {test_stats['type_accuracy'] - REF_2048_TYPE_ACC:.4f} / {REF_2048_ANGLE_MAE - test_stats['angle_mae_deg']:.2f}",
            f"- selected gain vs 20.49 IoU/Dice/type/angle-improvement = {test_stats['iou'] - REF_2049_IOU:.4f} / {test_stats['dice'] - REF_2049_DICE:.4f} / {test_stats['type_accuracy'] - REF_2049_TYPE_ACC:.4f} / {REF_2049_ANGLE_MAE - test_stats['angle_mae_deg']:.2f}",
            f"- Piao weak adaptation IoU/Dice/type/angle = {PIAO_WEAK_IOU:.4f} / {PIAO_WEAK_DICE:.4f} / {PIAO_WEAK_TYPE_ACC:.4f} / {PIAO_WEAK_ANGLE_MAE:.2f}",
            f"- dense COMSOL single-defect baseline IoU/Dice = {DENSE_SINGLE_BASELINE_IOU:.4f} / {DENSE_SINGLE_BASELINE_DICE:.4f}",
            "",
            f"Preview PNG generated for selected best: {preview_count} (not for submission)",
            f"Acceptance passed: {promising}",
            f"Next recommendation: {recommendation}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    categories = defaultdict(int)
    for row in selected_rows:
        if row["split"] == "test":
            categories[failure_category(row)] += 1
    audit_lines = [
        "COMSOL rect/rot geometry head sweep failure audit summary",
        "",
        f"Selected candidate: {selected_name}",
        f"Failure categories on selected test split: {dict(sorted(categories.items()))}",
        f"Type report on selected test split: {test_type}",
        "",
        "Conclusion:",
        (
            "- Best candidate passed acceptance and is eligible for forward consistency."
            if promising
            else "- No candidate passed acceptance; do not proceed to forward consistency."
        ),
        "",
        "Worst selected-candidate test cases:",
    ]
    for row in failure_rows[:20]:
        audit_lines.append(
            f"- {row['sample_id']}: {row.get('failure_category', failure_category(row))}, "
            f"IoU={base.to_float(row['iou']):.3f}, Dice={base.to_float(row['dice']):.3f}, "
            f"type={row['defect_type']}->{row['pred_defect_type']}, "
            f"angle_error={base.to_float(row['angle_abs_error_deg']):.2f}"
        )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return {"promising": promising, "recommendation": recommendation}


def run(args: argparse.Namespace) -> dict[str, Any]:
    arrays, diagnostics = base.load_arrays(args.npz, args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    candidate_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, torch.Tensor]] = {}

    for config in CANDIDATES:
        summary, epochs, state, _ = train_candidate(config, arrays, device, args)
        candidate_rows.append(summary)
        epoch_rows.extend(epochs)
        states[config.name] = state

    selected = select_best(candidate_rows)
    selected_name = str(selected["candidate"])
    selected_config = next(config for config in CANDIDATES if config.name == selected_name)
    selected_model = CandidateModel(selected_config, latent_dim=args.latent_dim).to(device)
    selected_model.load_state_dict(states[selected_name])
    selected_threshold = float(selected["selected_threshold"])

    selected_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        ds = base.RectRotDataset(arrays["split_indices"][split], arrays)
        pred = predict(selected_model, ds, arrays, selected_config, device, args.batch_size)
        selected_rows.extend(metric_rows(selected_name, pred, arrays, split, selected_threshold))

    group_rows, geometry_rows = build_group_summaries(selected_rows)
    failure_rows = build_failure_cases(selected_rows)
    preview_count = preview(selected_rows, arrays, args.preview_dir, max_count=24)

    base.write_csv(args.candidates, candidate_rows, CANDIDATE_FIELDS)
    base.write_csv(args.selected_metrics, selected_rows, METRIC_FIELDS)
    base.write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
    base.write_csv(args.group_summary, group_rows, base.GROUP_FIELDS)
    base.write_csv(args.geometry_summary, geometry_rows, base.GROUP_FIELDS)
    base.write_csv(args.failure_cases, failure_rows, list(failure_rows[0].keys()) if failure_rows else METRIC_FIELDS)
    return write_summary(args, diagnostics, candidate_rows, selected_name, selected_rows, failure_rows, preview_count, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--selected-metrics", type=Path, default=DEFAULT_SELECTED_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY_SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE_CASES)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

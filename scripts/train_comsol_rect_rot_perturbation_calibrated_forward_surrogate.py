from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_rect_rot_local_perturbation_forward_pack_v1.npz"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_perturbation_forward_surrogate_summary.txt"
DEFAULT_RESIDUAL_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_perturbation_residual_objective_audit_summary.txt"
)
DEFAULT_CANDIDATES = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_forward_surrogate_candidates.csv"
)
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_forward_surrogate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_forward_surrogate_epoch_log.csv"
DEFAULT_ORDERING = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_forward_surrogate_ordering_audit.csv"
)
DEFAULT_RESIDUAL_AUDIT = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_residual_objective_audit.csv"
)

SEED = 42
OLD_S2_VAL_MISMATCH_RATE = 0.3030
OLD_S2_TEST_MISMATCH_RATE = 0.3939


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train perturbation-calibrated COMSOL rect/rot forward surrogates and audit residual ordering."
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--residual-summary", type=Path, default=DEFAULT_RESIDUAL_SUMMARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--ordering-audit", type=Path, default=DEFAULT_ORDERING)
    parser.add_argument("--residual-audit", type=Path, default=DEFAULT_RESIDUAL_AUDIT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_corr(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or y.size < 2:
        return math.nan
    if float(x.std()) <= 1.0e-12 or float(y.std()) <= 1.0e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_corr(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3:
        return math.nan
    return safe_corr(rankdata(x), rankdata(y))


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def gradient_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
    target_grad = target[:, :, 1:] - target[:, :, :-1]
    return F.mse_loss(pred_grad, target_grad)


def peak_region_weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    abs_target = torch.abs(target)
    threshold = 0.5 * abs_target.amax(dim=2, keepdim=True).clamp_min(1.0e-8)
    weights = 1.0 + (abs_target >= threshold).float()
    return ((pred - target).square() * weights).mean()


def waveform_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred - target
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = math.sqrt(mse)
    denom = float(np.std(target))
    nrmse = rmse / denom if denom > 1.0e-12 else math.nan
    corr = safe_corr(pred.reshape(-1), target.reshape(-1))
    grad_diff = (pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])
    gradient = float(np.mean(grad_diff**2))
    amp_pred = np.max(np.abs(pred), axis=2)
    amp_target = np.max(np.abs(target), axis=2)
    amplitude_error = float(np.mean(np.abs(amp_pred - amp_target)))
    peak_pred = np.argmax(np.abs(pred), axis=2)
    peak_target = np.argmax(np.abs(target), axis=2)
    peak_index_error = float(np.mean(np.abs(peak_pred - peak_target)))
    per_line_mse = np.mean(diff**2, axis=(0, 2))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "correlation": corr,
        "gradient_mse": gradient,
        "amplitude_error": amplitude_error,
        "peak_index_error": peak_index_error,
        "line0_mse": float(per_line_mse[0]),
        "line1_mse": float(per_line_mse[1]),
        "line2_mse": float(per_line_mse[2]),
    }


def residual_nrmse(signal: np.ndarray, observed: np.ndarray) -> float:
    diff = signal - observed
    rmse = math.sqrt(float(np.mean(diff**2)))
    denom = float(np.std(observed))
    return rmse / denom if denom > 1.0e-12 else math.nan


def mask_feature_rows(masks: np.ndarray, mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    features: list[list[float]] = []
    x_grid, y_grid = np.meshgrid(mask_x, mask_y)
    for mask in masks:
        weights = mask.astype(np.float64)
        total = float(weights.sum())
        if total <= 0:
            features.append([0.0] * 9)
            continue
        x_mean = float((weights * x_grid).sum() / total)
        y_mean = float((weights * y_grid).sum() / total)
        dx = x_grid - x_mean
        dy = y_grid - y_mean
        xx = float((weights * dx * dx).sum() / total)
        yy = float((weights * dy * dy).sum() / total)
        xy = float((weights * dx * dy).sum() / total)
        angle = 0.5 * math.atan2(2.0 * xy, xx - yy + 1.0e-18)
        ys, xs = np.where(mask > 0)
        bbox_w = float(mask_x[xs.max()] - mask_x[xs.min()]) if xs.size else 0.0
        bbox_h = float(mask_y[ys.max()] - mask_y[ys.min()]) if ys.size else 0.0
        area_frac = total / float(mask.size)
        features.append([area_frac, x_mean, y_mean, bbox_w, bbox_h, xx, yy, math.sin(angle), math.cos(angle)])
    return np.asarray(features, dtype=np.float32)


def load_arrays(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    required = [
        "delta_bz",
        "masks",
        "reference_observed_delta_bz",
        "split",
        "defect_types",
        "variant_types",
        "base_sample_ids",
        "sample_ids",
        "geometry_params",
        "geometry_quality_to_true",
        "expected_quality_rank",
        "generated_real_forward",
        "mask_x",
        "mask_y",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing perturbation pack keys: {missing}")
    split = data["split"].astype(str)
    train_idx = np.where(split == "train")[0]
    if train_idx.size == 0:
        raise RuntimeError("No train rows in perturbation pack")

    geom_rows = [parse_json(raw) for raw in data["geometry_params"]]
    qualities = [parse_json(raw) for raw in data["geometry_quality_to_true"]]
    defect_types = data["defect_types"].astype(str)
    type_onehot = np.zeros((len(defect_types), 2), dtype=np.float32)
    type_onehot[:, 0] = defect_types == "rectangular_notch"
    type_onehot[:, 1] = defect_types == "rotated_rect"
    geom = np.array(
        [
            [
                float(g["center_x_m"]),
                float(g["center_y_m"]),
                float(g["width_m"]),
                float(g["length_m"]),
                float(g["depth_m"]),
                float(g["angle_rad"]),
                math.sin(float(g["angle_rad"])),
                math.cos(float(g["angle_rad"])),
            ]
            for g in geom_rows
        ],
        dtype=np.float32,
    )
    mask_features = mask_feature_rows(data["masks"], data["mask_x"].astype(np.float64), data["mask_y"].astype(np.float64))
    target = data["delta_bz"].astype(np.float32)
    observed = data["reference_observed_delta_bz"].astype(np.float32)

    geom_mean = geom[train_idx].mean(axis=0)
    geom_std = geom[train_idx].std(axis=0)
    geom_std = np.where(geom_std <= 1.0e-12, 1.0, geom_std).astype(np.float32)
    mask_mean = mask_features[train_idx].mean(axis=0)
    mask_std = mask_features[train_idx].std(axis=0)
    mask_std = np.where(mask_std <= 1.0e-12, 1.0, mask_std).astype(np.float32)
    target_mean = float(target[train_idx].mean())
    target_std = float(target[train_idx].std())
    if target_std <= 1.0e-12:
        target_std = 1.0
    arrays = {
        "target_norm": ((target - target_mean) / target_std).astype(np.float32),
        "observed_norm": ((observed - target_mean) / target_std).astype(np.float32),
        "target_raw": target,
        "observed_raw": observed,
        "base_input": np.concatenate([type_onehot, (geom - geom_mean) / geom_std], axis=1).astype(np.float32),
        "mask_input": ((mask_features - mask_mean) / mask_std).astype(np.float32),
        "split": split,
        "sample_ids": data["sample_ids"].astype(str),
        "base_sample_ids": data["base_sample_ids"].astype(str),
        "defect_types": defect_types,
        "variant_types": data["variant_types"].astype(str),
        "expected_quality_rank": data["expected_quality_rank"].astype(np.int64),
        "generated_real_forward": data["generated_real_forward"].astype(bool),
        "mask_iou": np.array([float(q.get("mask_iou", q.get("geometry_mask_iou_vs_true", math.nan))) for q in qualities]),
        "mask_dice": np.array([float(q.get("mask_dice", q.get("geometry_mask_dice_vs_true", math.nan))) for q in qualities]),
        "area_error": np.array([float(q.get("area_error", math.nan)) for q in qualities]),
        "target_mean": target_mean,
        "target_std": target_std,
        "geom_input_dim": int(type_onehot.shape[1] + geom.shape[1]),
        "mask_input_dim": int(mask_features.shape[1]),
    }
    arrays["split_indices"] = {name: np.where(split == name)[0].astype(np.int64) for name in ["train", "val", "test"]}
    return arrays


class ForwardDataset(Dataset):
    def __init__(self, indices: np.ndarray, arrays: dict[str, Any], candidate: str):
        self.indices = indices.astype(np.int64)
        self.arrays = arrays
        self.candidate = candidate

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        if self.candidate == "S1_perturb_geom_mlp":
            x = self.arrays["base_input"][i]
        else:
            x = np.concatenate([self.arrays["base_input"][i], self.arrays["mask_input"][i]], axis=0)
        return {
            "x": torch.from_numpy(x).float(),
            "target": torch.from_numpy(self.arrays["target_norm"][i]).float(),
            "source_index": torch.tensor(i, dtype=torch.long),
        }


class ForwardMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Linear(192, 603),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 3, 201)


@dataclass
class Bundle:
    name: str
    model: nn.Module
    device: torch.device
    best_epoch: int
    best_val_nrmse: float


def loss_for_candidate(name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(pred, target) + 0.2 * F.l1_loss(pred, target) + 0.1 * gradient_mse(pred, target)
    if name == "S2_perturb_geom_mask_mlp":
        loss = loss + 0.1 * peak_region_weighted_mse(pred, target)
    return loss


def predict_norm(bundle: Bundle, arrays: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    bundle.model.eval()
    ds = ForwardDataset(indices, arrays, bundle.name)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = bundle.model(batch["x"].to(bundle.device)).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def train_candidate(name: str, arrays: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[Bundle, list[dict[str, Any]]]:
    input_dim = arrays["geom_input_dim"] if name == "S1_perturb_geom_mlp" else arrays["geom_input_dim"] + arrays["mask_input_dim"]
    model = ForwardMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    train_ds = ForwardDataset(arrays["split_indices"]["train"], arrays, name)
    val_idx = arrays["split_indices"]["val"]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_nrmse = math.inf
    best_epoch = -1
    epoch_rows: list[dict[str, Any]] = []
    patience = 70
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(device)
            pred = model(batch["x"].to(device))
            loss = loss_for_candidate(name, pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        pred_val = predict_norm(Bundle(name, model, device, epoch, best_val_nrmse), arrays, val_idx)
        val_stats = waveform_stats(pred_val, arrays["target_norm"][val_idx])
        improved = val_stats["nrmse"] < best_val_nrmse
        if improved:
            best_val_nrmse = val_stats["nrmse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        epoch_rows.append(
            {
                "candidate": name,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_mse": val_stats["mse"],
                "val_mae": val_stats["mae"],
                "val_nrmse": val_stats["nrmse"],
                "val_correlation": val_stats["correlation"],
                "best_epoch_so_far": best_epoch,
            }
        )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"No best state found for {name}")
    model.load_state_dict(best_state)
    return Bundle(name, model, device, best_epoch, best_val_nrmse), epoch_rows


def evaluate_waveform(bundle: Bundle, arrays: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    pred_by_split: dict[str, np.ndarray] = {}
    for split, idx in arrays["split_indices"].items():
        pred = predict_norm(bundle, arrays, idx)
        pred_by_split[split] = pred
        stats = waveform_stats(pred, arrays["target_norm"][idx])
        rows.append({"candidate": bundle.name, "split": split, "n": int(idx.size), **stats})
    return rows, pred_by_split


def ordering_for_split(
    arrays: dict[str, Any],
    split: str,
    surrogate_pred_norm: np.ndarray | None,
    candidate: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    indices = arrays["split_indices"][split]
    by_base: dict[str, list[int]] = {}
    for i in indices:
        by_base.setdefault(str(arrays["base_sample_ids"][i]), []).append(int(i))

    residual_rows: list[dict[str, Any]] = []
    for i in indices:
        local_pos = int(np.where(indices == i)[0][0])
        oracle_res = residual_nrmse(arrays["target_norm"][i], arrays["observed_norm"][i])
        if surrogate_pred_norm is None:
            surrogate_res = math.nan
        else:
            surrogate_res = residual_nrmse(surrogate_pred_norm[local_pos], arrays["observed_norm"][i])
        residual_rows.append(
            {
                "candidate": candidate,
                "split": split,
                "sample_id": arrays["sample_ids"][i],
                "base_sample_id": arrays["base_sample_ids"][i],
                "defect_type": arrays["defect_types"][i],
                "variant_type": arrays["variant_types"][i],
                "expected_quality_rank": int(arrays["expected_quality_rank"][i]),
                "generated_real_forward": bool(arrays["generated_real_forward"][i]),
                "mask_iou": float(arrays["mask_iou"][i]),
                "mask_dice": float(arrays["mask_dice"][i]),
                "area_error": float(arrays["area_error"][i]),
                "oracle_residual_nrmse": oracle_res,
                "surrogate_residual_nrmse": surrogate_res,
            }
        )

    row_by_idx = {int(i): row for i, row in zip(indices, residual_rows)}
    oracle_hits: list[float] = []
    surrogate_hits: list[float] = []
    oracle_mismatches: list[float] = []
    surrogate_mismatches: list[float] = []
    for base_indices in by_base.values():
        for a_pos in range(len(base_indices)):
            for b_pos in range(a_pos + 1, len(base_indices)):
                ia = base_indices[a_pos]
                ib = base_indices[b_pos]
                qa = float(arrays["mask_iou"][ia])
                qb = float(arrays["mask_iou"][ib])
                if abs(qa - qb) < 1.0e-9:
                    continue
                better, worse = (ia, ib) if qa > qb else (ib, ia)
                better_row = row_by_idx[better]
                worse_row = row_by_idx[worse]
                oracle_hit = float(better_row["oracle_residual_nrmse"] < worse_row["oracle_residual_nrmse"])
                oracle_hits.append(oracle_hit)
                oracle_mismatches.append(1.0 - oracle_hit)
                if surrogate_pred_norm is not None:
                    s_hit = float(better_row["surrogate_residual_nrmse"] < worse_row["surrogate_residual_nrmse"])
                    surrogate_hits.append(s_hit)
                    surrogate_mismatches.append(1.0 - s_hit)

    oracle_residuals = [row["oracle_residual_nrmse"] for row in residual_rows]
    surrogate_residuals = [row["surrogate_residual_nrmse"] for row in residual_rows]
    mask_errors = [1.0 - row["mask_iou"] for row in residual_rows]
    area_errors = [row["area_error"] for row in residual_rows]
    stats = {
        "oracle_residual_ordering_accuracy": float(np.mean(oracle_hits)) if oracle_hits else math.nan,
        "oracle_mismatch_rate": float(np.mean(oracle_mismatches)) if oracle_mismatches else math.nan,
        "oracle_residual_error_correlation": safe_corr(oracle_residuals, mask_errors),
        "oracle_residual_error_spearman": spearman_corr(oracle_residuals, mask_errors),
        "oracle_residual_area_correlation": safe_corr(oracle_residuals, area_errors),
        "surrogate_residual_ordering_accuracy": float(np.mean(surrogate_hits)) if surrogate_hits else math.nan,
        "surrogate_mismatch_rate": float(np.mean(surrogate_mismatches)) if surrogate_mismatches else math.nan,
        "surrogate_residual_error_correlation": safe_corr(surrogate_residuals, mask_errors)
        if surrogate_pred_norm is not None
        else math.nan,
        "surrogate_residual_error_spearman": spearman_corr(surrogate_residuals, mask_errors)
        if surrogate_pred_norm is not None
        else math.nan,
        "surrogate_vs_oracle_residual_correlation": safe_corr(surrogate_residuals, oracle_residuals)
        if surrogate_pred_norm is not None
        else math.nan,
        "pair_count": len(oracle_hits),
    }
    return stats, residual_rows


def summarize_candidate(
    candidate: str,
    waveform_rows: list[dict[str, Any]],
    ordering_stats_by_split: dict[str, dict[str, float]],
    best_epoch: int,
) -> dict[str, Any]:
    val_w = next(row for row in waveform_rows if row["candidate"] == candidate and row["split"] == "val")
    test_w = next(row for row in waveform_rows if row["candidate"] == candidate and row["split"] == "test")
    val_o = ordering_stats_by_split["val"]
    test_o = ordering_stats_by_split["test"]
    val_corr = val_o["surrogate_residual_error_correlation"]
    val_corr = 0.0 if not math.isfinite(val_corr) else val_corr
    val_nrmse_norm = val_w["nrmse"]
    score = (
        0.40 * val_o["surrogate_residual_ordering_accuracy"]
        + 0.30 * val_corr
        - 0.20 * val_o["surrogate_mismatch_rate"]
        - 0.10 * val_nrmse_norm
    )
    usable = (
        math.isfinite(test_w["nrmse"])
        and val_o["oracle_residual_ordering_accuracy"] >= 0.60
        and val_o["surrogate_residual_ordering_accuracy"] > 0.50
        and val_o["surrogate_mismatch_rate"] < OLD_S2_VAL_MISMATCH_RATE
        and test_o["surrogate_residual_ordering_accuracy"] > 0.50
        and val_corr > 0.05
    )
    return {
        "candidate": candidate,
        "best_epoch": best_epoch,
        "val_nrmse": val_w["nrmse"],
        "test_nrmse": test_w["nrmse"],
        "val_correlation": val_w["correlation"],
        "test_correlation": test_w["correlation"],
        "val_oracle_ordering_accuracy": val_o["oracle_residual_ordering_accuracy"],
        "test_oracle_ordering_accuracy": test_o["oracle_residual_ordering_accuracy"],
        "val_surrogate_ordering_accuracy": val_o["surrogate_residual_ordering_accuracy"],
        "test_surrogate_ordering_accuracy": test_o["surrogate_residual_ordering_accuracy"],
        "val_surrogate_mismatch_rate": val_o["surrogate_mismatch_rate"],
        "test_surrogate_mismatch_rate": test_o["surrogate_mismatch_rate"],
        "val_surrogate_residual_error_correlation": val_o["surrogate_residual_error_correlation"],
        "test_surrogate_residual_error_correlation": test_o["surrogate_residual_error_correlation"],
        "val_surrogate_vs_oracle_residual_correlation": val_o["surrogate_vs_oracle_residual_correlation"],
        "test_surrogate_vs_oracle_residual_correlation": test_o["surrogate_vs_oracle_residual_correlation"],
        "selection_score": score,
        "usable": usable,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(SEED)
    arrays = load_arrays(args.npz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_names = ["S1_perturb_geom_mlp", "S2_perturb_geom_mask_mlp"]
    all_metrics: list[dict[str, Any]] = []
    all_epoch_rows: list[dict[str, Any]] = []
    all_ordering_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    residual_summary_rows: list[dict[str, Any]] = []
    bundles: dict[str, Bundle] = {}

    # Oracle-only audit is independent of candidate training.
    oracle_stats_by_split: dict[str, dict[str, float]] = {}
    for split in ["train", "val", "test"]:
        stats, rows = ordering_for_split(arrays, split, None, "COMSOL_oracle")
        oracle_stats_by_split[split] = stats
        all_ordering_rows.extend(rows)
        residual_summary_rows.append({"candidate": "COMSOL_oracle", "split": split, **stats})

    for name in candidate_names:
        bundle, epoch_rows = train_candidate(name, arrays, args, device)
        bundles[name] = bundle
        all_epoch_rows.extend(epoch_rows)
        waveform_rows, pred_by_split = evaluate_waveform(bundle, arrays)
        all_metrics.extend(waveform_rows)
        ordering_stats_by_split: dict[str, dict[str, float]] = {}
        for split in ["train", "val", "test"]:
            stats, rows = ordering_for_split(arrays, split, pred_by_split[split], name)
            ordering_stats_by_split[split] = stats
            all_ordering_rows.extend(rows)
            residual_summary_rows.append({"candidate": name, "split": split, **stats})
        candidate_rows.append(summarize_candidate(name, all_metrics, ordering_stats_by_split, bundle.best_epoch))

    selected = max(candidate_rows, key=lambda row: row["selection_score"])
    write_csv(args.candidates, candidate_rows)
    write_csv(args.metrics, all_metrics)
    write_csv(args.epoch_log, all_epoch_rows)
    write_csv(args.ordering_audit, all_ordering_rows)
    write_csv(args.residual_audit, residual_summary_rows)

    summary_lines = [
        "COMSOL rect/rot perturbation-calibrated forward surrogate summary",
        "",
        f"Input perturbation NPZ: {args.npz}",
        "Scope: rectangular_notch + rotated_rect local perturbation pack only; polygon excluded.",
        "This stage trains forward surrogates only. It does not train inverse geometry heads and does not run refinement.",
        "Surrogate inputs are geometry / type and optional rasterized-geometry-derived mask features; observed delta_bz is never an input.",
        "",
        f"Rows by split: { {k: int(v.size) for k, v in arrays['split_indices'].items()} }",
        f"Rows by defect type: {dict(Counter(str(x) for x in arrays['defect_types']))}",
        f"Rows by variant: {dict(Counter(str(x) for x in arrays['variant_types']))}",
        f"Generated real COMSOL forward rows: {int(arrays['generated_real_forward'].sum())}",
        f"Reused true reference rows: {int((~arrays['generated_real_forward']).sum())}",
        "",
        "Oracle residual ordering:",
    ]
    for split in ["train", "val", "test"]:
        stats = oracle_stats_by_split[split]
        summary_lines.extend(
            [
                f"- {split}: ordering={stats['oracle_residual_ordering_accuracy']:.6f}, "
                f"mismatch={stats['oracle_mismatch_rate']:.6f}, "
                f"residual-error corr={stats['oracle_residual_error_correlation']:.6f}, "
                f"spearman={stats['oracle_residual_error_spearman']:.6f}",
            ]
        )
    summary_lines.extend(["", "Candidate results:"])
    for row in candidate_rows:
        summary_lines.append(
            f"- {row['candidate']}: val/test NRMSE={row['val_nrmse']:.6f}/{row['test_nrmse']:.6f}, "
            f"val/test ordering={row['val_surrogate_ordering_accuracy']:.6f}/{row['test_surrogate_ordering_accuracy']:.6f}, "
            f"val/test mismatch={row['val_surrogate_mismatch_rate']:.6f}/{row['test_surrogate_mismatch_rate']:.6f}, "
            f"val corr={row['val_surrogate_residual_error_correlation']:.6f}, "
            f"score={row['selection_score']:.6f}, usable={row['usable']}"
        )
    summary_lines.extend(
        [
            "",
            f"Selected perturbation-calibrated surrogate: {selected['candidate']}",
            f"Selected usable for future refinement: {selected['usable']}",
            "",
            "Acceptance interpretation:",
            f"- oracle residual ordering not poor on val: {selected['val_oracle_ordering_accuracy'] >= 0.60}",
            f"- surrogate val ordering above random: {selected['val_surrogate_ordering_accuracy'] > 0.50}",
            f"- surrogate val mismatch lower than 20.55 S2 ({OLD_S2_VAL_MISMATCH_RATE}): {selected['val_surrogate_mismatch_rate'] < OLD_S2_VAL_MISMATCH_RATE}",
            f"- test ordering does not collapse: {selected['test_surrogate_ordering_accuracy'] > 0.50}",
            f"- residual-error correlation positive on val: {selected['val_surrogate_residual_error_correlation'] > 0.05}",
            f"- residual-error correlation positive on test: {selected['test_surrogate_residual_error_correlation'] > 0.05}",
            "- Caveat: this POC primarily validates pairwise local ordering; residual-error correlation can still be split-sensitive on the small partial pack.",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    residual_lines = [
        "COMSOL rect/rot perturbation residual objective audit summary",
        "",
        "Question 1: Does real COMSOL oracle residual rank geometry quality?",
        f"- val/test oracle ordering accuracy = {oracle_stats_by_split['val']['oracle_residual_ordering_accuracy']:.6f} / {oracle_stats_by_split['test']['oracle_residual_ordering_accuracy']:.6f}",
        f"- val/test oracle residual-error correlation = {oracle_stats_by_split['val']['oracle_residual_error_correlation']:.6f} / {oracle_stats_by_split['test']['oracle_residual_error_correlation']:.6f}",
        "",
        "Question 2: Does surrogate residual approximate oracle ordering?",
        f"- selected surrogate = {selected['candidate']}",
        f"- val/test surrogate ordering accuracy = {selected['val_surrogate_ordering_accuracy']:.6f} / {selected['test_surrogate_ordering_accuracy']:.6f}",
        f"- val/test surrogate-vs-oracle residual correlation = {selected['val_surrogate_vs_oracle_residual_correlation']:.6f} / {selected['test_surrogate_vs_oracle_residual_correlation']:.6f}",
        "",
        "Question 3: Is prior surrogate mismatch caused by lack of perturbation data?",
        f"- selected val mismatch rate = {selected['val_surrogate_mismatch_rate']:.6f}; 20.55 S2 val mismatch = {OLD_S2_VAL_MISMATCH_RATE:.6f}",
        f"- selected test mismatch rate = {selected['test_surrogate_mismatch_rate']:.6f}; 20.55 S2 test mismatch = {OLD_S2_TEST_MISMATCH_RATE:.6f}",
        "",
        "Question 4: Is local perturbation forward data enough to make surrogate useful?",
        f"- usable gate passed = {selected['usable']}",
        f"- caveat: selected test residual-error correlation = {selected['test_surrogate_residual_error_correlation']:.6f}; treat next refinement as controlled retry, not baseline evidence.",
        "",
        "Question 5: Recommended next step:",
    ]
    if selected["usable"]:
        recommendation = "A. Use the perturbation-calibrated surrogate in a controlled Priewald refinement retry."
    elif oracle_stats_by_split["val"]["oracle_residual_ordering_accuracy"] < 0.60:
        recommendation = "D. Richer observations / non-identifiability audit before further surrogate tuning."
    elif selected["val_surrogate_ordering_accuracy"] <= 0.50:
        recommendation = "B. Expand perturbation data or improve surrogate architecture; oracle is useful but surrogate ordering is weak."
    else:
        recommendation = "B. Expand perturbation data; current partial pack helps but does not fully clear surrogate mismatch."
    residual_lines.append(f"- {recommendation}")
    args.residual_summary.parent.mkdir(parents=True, exist_ok=True)
    args.residual_summary.write_text("\n".join(residual_lines) + "\n", encoding="utf-8")

    return {
        "selected": selected,
        "candidate_rows": candidate_rows,
        "oracle_stats": oracle_stats_by_split,
        "recommendation": recommendation,
    }


def main() -> None:
    result = run(parse_args())
    selected = result["selected"]
    print(f"Selected surrogate: {selected['candidate']}")
    print(
        "val/test ordering: "
        f"{selected['val_surrogate_ordering_accuracy']:.4f} / {selected['test_surrogate_ordering_accuracy']:.4f}"
    )
    print(f"usable: {selected['usable']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_rect_rot_profile_perturbation_forward_pack_v1.npz"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_profile_perturbation_forward_surrogate_summary.txt"
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_profile_perturbation_residual_objective_audit_summary.txt"
DEFAULT_CANDIDATES = PROJECT_ROOT / "results/metrics/comsol_profile_perturbation_forward_surrogate_candidates.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_profile_perturbation_forward_surrogate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_profile_perturbation_forward_surrogate_epoch_log.csv"
DEFAULT_ORDERING = PROJECT_ROOT / "results/metrics/comsol_profile_perturbation_forward_surrogate_ordering_audit.csv"
DEFAULT_AUDIT = PROJECT_ROOT / "results/metrics/comsol_profile_perturbation_residual_objective_audit.csv"

SEED = 42
K_STATIONS = 8
OLD_2059_VAL_MISMATCH = 0.3393
CANDIDATES = ("PPF1_profile_station_mlp", "PPF2_profile_raster_sequence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train profile-perturbation-calibrated forward surrogates and audit residual ordering."
    )
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--ordering-audit", type=Path, default=DEFAULT_ORDERING)
    parser.add_argument("--residual-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_corr(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or y.size < 2:
        return math.nan
    if float(x.std()) <= 1.0e-12 or float(y.std()) <= 1.0e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def safe_spearman(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        return math.nan
    return safe_corr(rank_values(x), rank_values(y))


def gradient_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def peak_region_weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    abs_target = torch.abs(target)
    threshold = 0.5 * abs_target.amax(dim=2, keepdim=True).clamp_min(1.0e-8)
    weights = 1.0 + (abs_target >= threshold).float()
    return ((pred - target).square() * weights).mean()


def waveform_loss(pred: torch.Tensor, target: torch.Tensor, peak_weighted: bool) -> torch.Tensor:
    loss = F.mse_loss(pred, target) + 0.2 * F.l1_loss(pred, target) + 0.1 * gradient_mse(pred, target)
    if peak_weighted:
        loss = loss + 0.1 * peak_region_weighted_mse(pred, target)
    return loss


def waveform_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred - target
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = math.sqrt(mse)
    denom = float(np.std(target))
    nrmse = rmse / denom if denom > 1.0e-12 else math.nan
    grad_diff = (pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])
    amp_pred = np.max(np.abs(pred), axis=2)
    amp_target = np.max(np.abs(target), axis=2)
    peak_pred = np.argmax(np.abs(pred), axis=2)
    peak_target = np.argmax(np.abs(target), axis=2)
    per_line_mse = np.mean(diff**2, axis=(0, 2))
    per_line_corr = [
        safe_corr(pred[:, line, :].reshape(-1), target[:, line, :].reshape(-1)) for line in range(pred.shape[1])
    ]
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "correlation": safe_corr(pred.reshape(-1), target.reshape(-1)),
        "gradient_mse": float(np.mean(grad_diff**2)),
        "amplitude_error": float(np.mean(np.abs(amp_pred - amp_target))),
        "peak_index_error": float(np.mean(np.abs(peak_pred - peak_target))),
        "peak_amplitude_error": float(np.mean(np.abs(amp_pred - amp_target))),
        "line0_mse": float(per_line_mse[0]),
        "line1_mse": float(per_line_mse[1]),
        "line2_mse": float(per_line_mse[2]),
        "line0_corr": per_line_corr[0],
        "line1_corr": per_line_corr[1],
        "line2_corr": per_line_corr[2],
    }


def residual_nrmse(signal: np.ndarray, observed: np.ndarray) -> float:
    diff = signal - observed
    rmse = math.sqrt(float(np.mean(diff**2)))
    denom = float(np.std(observed))
    return rmse / denom if denom > 1.0e-12 else math.nan


def json_load(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def profile_vector(profile: dict[str, Any], quality: dict[str, Any]) -> np.ndarray:
    half = np.asarray(profile["half_widths"], dtype=np.float64)
    offset = np.asarray(profile["center_offsets"], dtype=np.float64)
    occup = np.asarray(profile.get("occupancy", [1.0] * K_STATIONS), dtype=np.float64)
    u = np.asarray(profile["u_stations"], dtype=np.float64)
    length = float(profile["length"])
    depth = float(profile.get("depth_proxy", 0.0))
    area_proxy = float(profile.get("area_proxy", float(np.trapezoid(2.0 * half, u))))
    roughness = float(profile.get("roughness", float(np.mean(np.abs(np.diff(half, n=2)))) if half.size > 2 else 0.0))
    angle_rad = float(profile["angle_rad"])
    scalars = [
        float(profile["center_x"]),
        float(profile["center_y"]),
        math.sin(angle_rad),
        math.cos(angle_rad),
        angle_rad,
        length,
        depth,
        area_proxy,
        roughness,
        float(half.mean()),
        float(half.std()),
        float(half.min()),
        float(half.max()),
        float(np.abs(offset).mean()),
        float(np.abs(offset).max()),
        float(occup.mean()),
        float(quality.get("area_error", math.nan)),
    ]
    # quality area_error is included only as a geometry-derived diagnostic feature? No: avoid quality labels as input.
    scalars = scalars[:-1]
    return np.concatenate([np.asarray(scalars), u, half, offset, occup]).astype(np.float32)


def downsample_mask(mask: np.ndarray) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    return mask_f.reshape(16, 4, 32, 4).mean(axis=(1, 3)).astype(np.float32)


@dataclass
class PackData:
    sample_ids: np.ndarray
    base_ids: np.ndarray
    split: np.ndarray
    source_type: np.ndarray
    variant: np.ndarray
    reused: np.ndarray
    real_forward: np.ndarray
    vector: np.ndarray
    raster: np.ndarray
    target: np.ndarray
    reference: np.ndarray
    quality_score: np.ndarray
    quality_error: np.ndarray
    quality_iou: np.ndarray
    quality_dice: np.ndarray
    quality_area_error: np.ndarray
    target_norm: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    vector_norm: np.ndarray
    vector_mean: np.ndarray
    vector_std: np.ndarray


def load_pack(path: Path) -> PackData:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    split = z["split"].astype(str)
    train_mask = split == "train"
    vectors: list[np.ndarray] = []
    rasters: list[np.ndarray] = []
    quality_score: list[float] = []
    quality_error: list[float] = []
    quality_iou: list[float] = []
    quality_dice: list[float] = []
    quality_area_error: list[float] = []
    for i in range(len(split)):
        profile = json_load(z["profile_params_json"][i])
        quality = json_load(z["quality_to_true"][i])
        vectors.append(profile_vector(profile, quality))
        rasters.append(downsample_mask(z["masks"][i]))
        iou = float(quality.get("iou", math.nan))
        dice = float(quality.get("dice", math.nan))
        area = float(quality.get("area_error", math.nan))
        quality_iou.append(iou)
        quality_dice.append(dice)
        quality_area_error.append(area)
        quality_score.append(iou + dice - area)
        quality_error.append((1.0 - iou) + (1.0 - dice) + area)
    vector = np.stack(vectors).astype(np.float32)
    raster = np.stack(rasters).astype(np.float32)
    target = z["delta_bz"].astype(np.float32)
    reference = z["reference_observed_delta_bz"].astype(np.float32)
    target_mean = target[train_mask].mean(axis=0, keepdims=True)
    target_std = target[train_mask].std(axis=0, keepdims=True)
    target_std = np.where(target_std < 1.0e-8, 1.0, target_std)
    target_norm = ((target - target_mean) / target_std).astype(np.float32)
    vector_mean = vector[train_mask].mean(axis=0, keepdims=True)
    vector_std = vector[train_mask].std(axis=0, keepdims=True)
    vector_std = np.where(vector_std < 1.0e-8, 1.0, vector_std)
    vector_norm = ((vector - vector_mean) / vector_std).astype(np.float32)
    if not np.isfinite(vector_norm).all() or not np.isfinite(target_norm).all():
        raise ValueError("Non-finite normalized features or targets")
    return PackData(
        sample_ids=z["sample_ids"].astype(str),
        base_ids=z["base_sample_ids"].astype(str),
        split=split,
        source_type=z["source_defect_types"].astype(str),
        variant=z["variant_types"].astype(str),
        reused=z["reused_original"].astype(bool),
        real_forward=z["generated_real_forward"].astype(bool),
        vector=vector,
        raster=raster,
        target=target,
        reference=reference,
        quality_score=np.asarray(quality_score, dtype=np.float64),
        quality_error=np.asarray(quality_error, dtype=np.float64),
        quality_iou=np.asarray(quality_iou, dtype=np.float64),
        quality_dice=np.asarray(quality_dice, dtype=np.float64),
        quality_area_error=np.asarray(quality_area_error, dtype=np.float64),
        target_norm=target_norm,
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        vector_norm=vector_norm,
        vector_mean=vector_mean.astype(np.float32),
        vector_std=vector_std.astype(np.float32),
    )


class ProfileDataset(Dataset):
    def __init__(self, data: PackData, indices: np.ndarray) -> None:
        self.vector = torch.from_numpy(data.vector_norm[indices]).float()
        self.raster = torch.from_numpy(data.raster[indices, None, :, :]).float()
        self.target = torch.from_numpy(data.target_norm[indices]).float()

    def __len__(self) -> int:
        return int(self.vector.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.vector[idx], self.raster[idx], self.target[idx]


class PPF1ProfileStationMLP(nn.Module):
    def __init__(self, vector_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vector_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 201),
        )

    def forward(self, vector: torch.Tensor, raster: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(vector).view(-1, 3, 201)


class PPF2ProfileRasterSequence(nn.Module):
    def __init__(self, vector_dim: int) -> None:
        super().__init__()
        self.raster_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 8, 80),
            nn.ReLU(),
        )
        self.vector_encoder = nn.Sequential(nn.Linear(vector_dim, 96), nn.ReLU(), nn.Linear(96, 80), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 201),
        )

    def forward(self, vector: torch.Tensor, raster: torch.Tensor) -> torch.Tensor:
        latent = torch.cat([self.vector_encoder(vector), self.raster_encoder(raster)], dim=1)
        return self.head(latent).view(-1, 3, 201)


def make_model(name: str, vector_dim: int) -> nn.Module:
    if name == "PPF1_profile_station_mlp":
        return PPF1ProfileStationMLP(vector_dim)
    if name == "PPF2_profile_raster_sequence":
        return PPF2ProfileRasterSequence(vector_dim)
    raise ValueError(name)


def denormalize(pred_norm: np.ndarray, data: PackData) -> np.ndarray:
    return pred_norm * data.target_std + data.target_mean


def predict(model: nn.Module, dataset: ProfileDataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for vector, raster, _target in loader:
            vector = vector.to(device)
            raster = raster.to(device)
            outputs.append(model(vector, raster).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_ordering(
    data: PackData,
    indices: np.ndarray,
    surrogate_signal: np.ndarray | None,
    candidate: str,
    split_name: str,
) -> dict[str, Any]:
    oracle_residual = np.asarray([residual_nrmse(data.target[i], data.reference[i]) for i in indices], dtype=np.float64)
    if surrogate_signal is None:
        surrogate_residual = np.full_like(oracle_residual, math.nan)
    else:
        surrogate_residual = np.asarray(
            [residual_nrmse(surrogate_signal[j], data.reference[i]) for j, i in enumerate(indices)], dtype=np.float64
        )
    q_score = data.quality_score[indices]
    q_error = data.quality_error[indices]
    base_ids = data.base_ids[indices]

    oracle_total = surrogate_total = 0
    oracle_ok = surrogate_ok = 0
    for base_id in sorted(set(base_ids.tolist())):
        local = np.where(base_ids == base_id)[0]
        for a_pos in range(len(local)):
            for b_pos in range(a_pos + 1, len(local)):
                a = local[a_pos]
                b = local[b_pos]
                if abs(q_score[a] - q_score[b]) <= 1.0e-12:
                    continue
                a_better = q_score[a] > q_score[b]
                oracle_prefers_a = oracle_residual[a] < oracle_residual[b]
                oracle_ok += int(oracle_prefers_a == a_better)
                oracle_total += 1
                if np.isfinite(surrogate_residual[a]) and np.isfinite(surrogate_residual[b]):
                    surrogate_prefers_a = surrogate_residual[a] < surrogate_residual[b]
                    surrogate_ok += int(surrogate_prefers_a == a_better)
                    surrogate_total += 1

    oracle_acc = oracle_ok / oracle_total if oracle_total else math.nan
    surrogate_acc = surrogate_ok / surrogate_total if surrogate_total else math.nan
    mismatch = 1.0 - surrogate_acc if np.isfinite(surrogate_acc) else math.nan
    by_variant: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for idx in indices:
        by_variant[data.variant[idx]] = by_variant.get(data.variant[idx], 0) + 1
        by_type[data.source_type[idx]] = by_type.get(data.source_type[idx], 0) + 1
    return {
        "candidate": candidate,
        "split": split_name,
        "rows": int(len(indices)),
        "base_samples": int(len(set(base_ids.tolist()))),
        "oracle_pair_count": int(oracle_total),
        "surrogate_pair_count": int(surrogate_total),
        "oracle_ordering_accuracy": oracle_acc,
        "surrogate_ordering_accuracy": surrogate_acc,
        "mismatch_rate": mismatch,
        "oracle_residual_error_correlation": safe_corr(oracle_residual, q_error),
        "surrogate_residual_error_correlation": safe_corr(surrogate_residual, q_error),
        "oracle_residual_error_spearman": safe_spearman(oracle_residual, q_error),
        "surrogate_residual_error_spearman": safe_spearman(surrogate_residual, q_error),
        "surrogate_vs_oracle_residual_correlation": safe_corr(surrogate_residual, oracle_residual),
        "mean_oracle_residual": float(np.nanmean(oracle_residual)),
        "mean_surrogate_residual": (
            float(np.nanmean(surrogate_residual)) if np.isfinite(surrogate_residual).any() else math.nan
        ),
        "mean_quality_iou": float(np.nanmean(data.quality_iou[indices])),
        "mean_quality_dice": float(np.nanmean(data.quality_dice[indices])),
        "mean_quality_area_error": float(np.nanmean(data.quality_area_error[indices])),
        "rows_by_variant": json.dumps(by_variant, sort_keys=True),
        "rows_by_source_defect_type": json.dumps(by_type, sort_keys=True),
    }


def train_candidate(
    candidate: str,
    data: PackData,
    split_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    set_seed(args.seed)
    model = make_model(candidate, data.vector_norm.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_ds = ProfileDataset(data, split_indices["train"])
    val_ds = ProfileDataset(data, split_indices["val"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    peak_weighted = candidate == "PPF2_profile_raster_sequence"
    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    best_epoch = -1
    no_improve = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for vector, raster, target in train_loader:
            vector = vector.to(device)
            raster = raster.to(device)
            target = target.to(device)
            opt.zero_grad(set_to_none=True)
            loss = waveform_loss(model(vector, raster), target, peak_weighted)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for vector, raster, target in val_loader:
                vector = vector.to(device)
                raster = raster.to(device)
                target = target.to(device)
                val_losses.append(float(waveform_loss(model(vector, raster), target, peak_weighted).cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        logs.append({"candidate": candidate, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss + 1.0e-8 < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch >= 80 and no_improve >= 60:
            break
    if best_state is None:
        raise RuntimeError(f"No best state for {candidate}")
    model.load_state_dict(best_state)
    logs.append({"candidate": candidate, "epoch": "best", "train_loss": math.nan, "val_loss": best_val, "best_epoch": best_epoch})
    return model, logs


def summarize_pack(data: PackData) -> dict[str, Any]:
    rows_by_split = {name: int((data.split == name).sum()) for name in ("train", "val", "test")}
    rows_by_type: dict[str, int] = {}
    rows_by_variant: dict[str, int] = {}
    for value in data.source_type:
        key = str(value)
        rows_by_type[key] = rows_by_type.get(key, 0) + 1
    for value in data.variant:
        key = str(value)
        rows_by_variant[key] = rows_by_variant.get(key, 0) + 1
    return {
        "total_rows": int(len(data.split)),
        "reused_original_rows": int(data.reused.sum()),
        "real_comsol_forward_rows": int(data.real_forward.sum()),
        "represented_base_samples": int(len(set(data.base_ids.tolist()))),
        "rows_by_split": rows_by_split,
        "rows_by_source_defect_type": rows_by_type,
        "rows_by_variant": rows_by_variant,
    }


def write_summaries(
    args: argparse.Namespace,
    data: PackData,
    pack_summary: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    ordering_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    gate_pass: bool,
) -> None:
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    sel_name = selected["candidate"]
    selected_order = {
        row["split"]: row for row in ordering_rows if row["candidate"] == sel_name and row["split"] in {"train", "val", "test"}
    }
    selected_metrics = {
        row["split"]: row for row in candidate_rows if row["candidate"] == sel_name and row["split"] in {"train", "val", "test"}
    }
    with args.summary.open("w", encoding="utf-8") as f:
        f.write("COMSOL profile perturbation forward surrogate calibration summary\n\n")
        f.write("No profile refinement, no inverse model training, no checkpoint writing, and no baseline update.\n")
        f.write("Inputs are profile-native parameters and optional profile raster features; observed base delta_bz is not a model input.\n\n")
        for key, value in pack_summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(f"selected_candidate: {sel_name}\n")
        f.write(f"selected_by_validation_score: {selected['selection_score']:.6f}\n")
        f.write(f"stage_c_gate_passed: {gate_pass}\n")
        f.write("\nWaveform metrics for selected candidate:\n")
        for split_name in ("train", "val", "test"):
            row = selected_metrics[split_name]
            f.write(
                f"- {split_name}: NRMSE={row['nrmse']:.6f}, corr={row['correlation']:.6f}, "
                f"MAE={row['mae']:.6e}, peak_index_error={row['peak_index_error']:.3f}\n"
            )
        f.write("\nOrdering metrics for selected candidate:\n")
        for split_name in ("train", "val", "test"):
            row = selected_order[split_name]
            f.write(
                f"- {split_name}: oracle_ordering={row['oracle_ordering_accuracy']:.6f}, "
                f"surrogate_ordering={row['surrogate_ordering_accuracy']:.6f}, "
                f"mismatch_rate={row['mismatch_rate']:.6f}, "
                f"residual_error_corr={row['surrogate_residual_error_correlation']:.6f}\n"
            )
        f.write("\nGate criteria:\n")
        f.write("- oracle ordering should not be poor on validation.\n")
        f.write("- validation surrogate ordering should be meaningfully above random.\n")
        f.write(f"- validation mismatch_rate should be materially below 20.59 value {OLD_2059_VAL_MISMATCH:.4f}.\n")
        f.write("- validation residual-error correlation should be non-negative and meaningful.\n")
        f.write("- test ordering audit should not collapse.\n")
        f.write("\nConclusion:\n")
        if gate_pass:
            f.write("Profile perturbation data produced a usable profile-compatible surrogate for a future refinement retry.\n")
        else:
            f.write("The selected surrogate did not pass all usability gates; do not run profile refinement in this stage.\n")

    audit_by_split = {row["split"]: row for row in ordering_rows if row["candidate"] == sel_name and row["split"] in {"train", "val", "test"}}
    with args.audit_summary.open("w", encoding="utf-8") as f:
        f.write("COMSOL profile perturbation residual objective audit summary\n\n")
        f.write("Question 1: Does real COMSOL oracle residual rank profile quality?\n")
        f.write(
            f"- Validation oracle ordering accuracy = {audit_by_split['val']['oracle_ordering_accuracy']:.6f}; "
            f"test = {audit_by_split['test']['oracle_ordering_accuracy']:.6f}.\n"
        )
        f.write("Question 2: Does surrogate residual approximate oracle ordering?\n")
        f.write(
            f"- Selected {sel_name} validation surrogate ordering = {audit_by_split['val']['surrogate_ordering_accuracy']:.6f}; "
            f"test = {audit_by_split['test']['surrogate_ordering_accuracy']:.6f}.\n"
        )
        f.write("Question 3: Was the 20.59 failure caused by lack of profile-native perturbation data?\n")
        if gate_pass:
            f.write("- Evidence is positive: validation mismatch dropped below the 20.59 reference and residual correlation is usable.\n")
        else:
            f.write("- Evidence is incomplete: the pack is profile-native, but one or more validation gates remain weak.\n")
        f.write("Question 4: Is profile perturbation forward data enough to make surrogate useful?\n")
        f.write("- Yes for waveform fit if NRMSE/correlation are stable; residual ordering usefulness follows the gate result above.\n")
        f.write("Question 5: Next step?\n")
        if gate_pass:
            f.write("- Use the calibrated profile surrogate in a controlled profile-forward refinement retry, with validation-only selection.\n")
        else:
            f.write("- Expand profile perturbation data or improve the profile surrogate; do not run another refinement small tweak yet.\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data = load_pack(args.pack)
    split_indices = {name: np.where(data.split == name)[0] for name in ("train", "val", "test")}
    pack_summary = summarize_pack(data)
    metrics_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    candidate_summary: list[dict[str, Any]] = []

    oracle_rows = [
        evaluate_ordering(data, split_indices[split_name], None, "COMSOL_oracle", split_name)
        for split_name in ("train", "val", "test")
    ]
    ordering_rows.extend(oracle_rows)

    predictions_by_candidate: dict[str, dict[str, np.ndarray]] = {}
    for candidate in CANDIDATES:
        model, logs = train_candidate(candidate, data, split_indices, args, device)
        epoch_rows.extend(logs)
        predictions_by_candidate[candidate] = {}
        for split_name, indices in split_indices.items():
            ds = ProfileDataset(data, indices)
            pred_norm = predict(model, ds, device)
            pred = denormalize(pred_norm, data)
            predictions_by_candidate[candidate][split_name] = pred
            stats = waveform_stats(pred, data.target[indices])
            metrics_rows.append({"candidate": candidate, "split": split_name, "rows": int(len(indices)), **stats})
            ordering_rows.append(evaluate_ordering(data, indices, pred, candidate, split_name))

    val_ordering = {row["candidate"]: row for row in ordering_rows if row["split"] == "val"}
    val_metrics = {row["candidate"]: row for row in metrics_rows if row["split"] == "val"}
    selected_name = ""
    best_score = -math.inf
    for candidate in CANDIDATES:
        order = val_ordering[candidate]
        metric = val_metrics[candidate]
        corr = order["surrogate_residual_error_correlation"]
        corr_term = corr if np.isfinite(corr) else -1.0
        score = (
            0.40 * order["surrogate_ordering_accuracy"]
            + 0.30 * corr_term
            - 0.20 * order["mismatch_rate"]
            - 0.10 * metric["nrmse"]
        )
        gate = (
            order["oracle_ordering_accuracy"] >= 0.55
            and order["surrogate_ordering_accuracy"] >= 0.60
            and order["mismatch_rate"] < OLD_2059_VAL_MISMATCH
            and corr_term >= 0.10
            and val_metrics[candidate]["nrmse"] < 0.75
        )
        test_order = next(row for row in ordering_rows if row["candidate"] == candidate and row["split"] == "test")
        gate = bool(gate and test_order["surrogate_ordering_accuracy"] >= 0.55)
        candidate_summary.append(
            {
                "candidate": candidate,
                "selection_score": score,
                "selected": False,
                "gate_pass": gate,
                "val_nrmse": metric["nrmse"],
                "val_correlation": metric["correlation"],
                "val_ordering_accuracy": order["surrogate_ordering_accuracy"],
                "val_mismatch_rate": order["mismatch_rate"],
                "val_residual_error_correlation": order["surrogate_residual_error_correlation"],
                "val_oracle_ordering_accuracy": order["oracle_ordering_accuracy"],
                "test_ordering_accuracy": test_order["surrogate_ordering_accuracy"],
                "test_mismatch_rate": test_order["mismatch_rate"],
                "test_residual_error_correlation": test_order["surrogate_residual_error_correlation"],
                "notes": "validation-only selection; no refinement run",
            }
        )
        if score > best_score:
            best_score = score
            selected_name = candidate

    selected_row = next(row for row in candidate_summary if row["candidate"] == selected_name)
    selected_row["selected"] = True
    gate_pass = bool(selected_row["gate_pass"])

    residual_rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        selected_order = next(row for row in ordering_rows if row["candidate"] == selected_name and row["split"] == split_name)
        oracle_order = next(row for row in ordering_rows if row["candidate"] == "COMSOL_oracle" and row["split"] == split_name)
        residual_rows.append(
            {
                "selected_candidate": selected_name,
                "split": split_name,
                "oracle_ordering_accuracy": selected_order["oracle_ordering_accuracy"],
                "surrogate_ordering_accuracy": selected_order["surrogate_ordering_accuracy"],
                "mismatch_rate": selected_order["mismatch_rate"],
                "oracle_residual_error_correlation": selected_order["oracle_residual_error_correlation"],
                "surrogate_residual_error_correlation": selected_order["surrogate_residual_error_correlation"],
                "oracle_pair_count": selected_order["oracle_pair_count"],
                "surrogate_pair_count": selected_order["surrogate_pair_count"],
                "oracle_only_check_ordering_accuracy": oracle_order["oracle_ordering_accuracy"],
                "rows": selected_order["rows"],
                "base_samples": selected_order["base_samples"],
                "rows_by_variant": selected_order["rows_by_variant"],
                "rows_by_source_defect_type": selected_order["rows_by_source_defect_type"],
                "stage_c_gate_passed": gate_pass,
            }
        )

    write_csv(args.candidates, candidate_summary)
    write_csv(args.metrics, metrics_rows)
    write_csv(args.epoch_log, epoch_rows)
    write_csv(args.ordering_audit, ordering_rows)
    write_csv(args.residual_audit, residual_rows)
    write_summaries(args, data, pack_summary, metrics_rows, ordering_rows, selected_row, gate_pass)
    print(f"selected_candidate={selected_name}")
    print(f"gate_pass={gate_pass}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()

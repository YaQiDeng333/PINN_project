#!/usr/bin/env python
"""20.94 candidate screen for nominal-preserving liftoff adapters."""

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

import audit_true_3d_rbc_observation_perturbation_robustness as obs
import load_true_3d_rbc_liftoff_aug_dataset as liftoff
import train_true_3d_rbc_liftoff_aware_models as liftoff_train
import train_true_3d_rbc_neural_parameter_gate as gate
from replay_true_3d_rbc_baseline_on_liftoff_pack import ARTIFACT_MANIFEST, run as replay_baseline
from true_3d_rbc_profile_generator import clip_params_torch, depth_grid_torch, soft_bound_penalty


ROOT = liftoff.ROOT
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_adapter_candidate_screen_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_candidate_screen_metrics.csv"
BY_LIFTOFF = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_candidate_by_liftoff.csv"
SELECTED_JSON = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_candidate_screen_selected.csv"


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    kind: str
    description: str
    profile_weight: float = 1.0
    dimension_weight: float = 0.35
    curvature_aux_weight: float = 0.05
    nominal_zero_weight: float = 0.75
    paired_consistency_weight: float = 0.10
    bound_weight: float = 0.02


class ResidualAdapter(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )

    def forward(self, features: torch.Tensor, baseline_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.net(features)
        return baseline_norm + residual, residual


class SensorZConditionedRegressor(nn.Module):
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
        self.z_head = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 8), nn.GELU())
        self.head = nn.Sequential(nn.Linear(64 * 8 + 8, 96), nn.GELU(), nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 6))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x).flatten(1)
        return self.head(torch.cat([enc, self.z_head(z)], dim=1))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def candidate_configs(include_full_model: bool = True) -> list[CandidateConfig]:
    items = [
        CandidateConfig("A1_output_residual_adapter", "output_residual", "baseline six params + sensor_z embedding -> residual"),
        CandidateConfig("A2_latent_residual_adapter", "latent_residual", "frozen baseline latent + baseline six params + sensor_z embedding -> residual"),
    ]
    if include_full_model:
        items.append(CandidateConfig("A3_revised_sensor_z_conditioned_full_model", "full_sensor_z", "delta_b + sensor_z full model with nominal-preserving validation"))
    return items


def load_baseline_model() -> tuple[dict[str, Any], dict[str, Any], gate.RBCConvRegressor]:
    artifact, checkpoint, model = obs.load_artifact(ARTIFACT_MANIFEST)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return artifact, checkpoint, model


def baseline_context(checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "x_mean": np.asarray(checkpoint["normalization"]["x_mean"], dtype=np.float32),
        "x_std": np.asarray(checkpoint["normalization"]["x_std"], dtype=np.float32),
        "y_mean": np.asarray(checkpoint["normalization"]["y_mean"], dtype=np.float32),
        "y_std": np.asarray(checkpoint["normalization"]["y_std"], dtype=np.float32),
    }


def baseline_arrays(dataset: Any, stats: dict[str, np.ndarray], checkpoint: dict[str, Any], model: gate.RBCConvRegressor) -> dict[str, np.ndarray]:
    ctx = baseline_context(checkpoint)
    pred_raw = obs.predict(model, dataset.x_channels, ctx)
    pred_norm = ((pred_raw - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
    x_for_baseline = ((dataset.x_channels - ctx["x_mean"]) / ctx["x_std"]).astype(np.float32)
    with torch.no_grad():
        latent = model.encoder(torch.as_tensor(x_for_baseline, dtype=torch.float32)).flatten(1).cpu().numpy().astype(np.float32)
    return {"pred_raw": pred_raw.astype(np.float32), "pred_norm": pred_norm, "latent": latent}


def base_group_ids(dataset: Any) -> np.ndarray:
    mapping = {base_id: idx for idx, base_id in enumerate(sorted(set(dataset.base_sample_ids.astype(str))))}
    return np.asarray([mapping[str(base)] for base in dataset.base_sample_ids], dtype=np.int64)


def depth_scale(dataset: Any, splits: dict[str, np.ndarray]) -> float:
    train_d = dataset.rbc_params[splits["train"], 2]
    scale = float(np.max(train_d) - np.min(train_d))
    return scale if scale > 1.0e-12 else float(np.mean(train_d))


def param_bounds(dataset: Any, splits: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    train = dataset.rbc_params[splits["train"]]
    return torch.as_tensor(train.min(axis=0), dtype=torch.float32), torch.as_tensor(train.max(axis=0), dtype=torch.float32)


def denormalize_torch(pred_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return pred_norm * y_std.to(pred_norm.device) + y_mean.to(pred_norm.device)


def paired_consistency_loss(pred_norm: torch.Tensor, pred_grid: torch.Tensor, base_gid: torch.Tensor, scale: float) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for gid in torch.unique(base_gid):
        mask = base_gid == gid
        if int(mask.sum().item()) < 2:
            continue
        group_params = pred_norm[mask]
        group_grid = pred_grid[mask] / max(scale, 1.0e-12)
        losses.append(torch.mean((group_params - group_params.mean(dim=0, keepdim=True)) ** 2))
        losses.append(torch.mean((group_grid - group_grid.mean(dim=0, keepdim=True)) ** 2))
    if not losses:
        return torch.tensor(0.0, dtype=pred_norm.dtype, device=pred_norm.device)
    return torch.stack(losses).mean()


def adapter_loss(
    pred_norm: torch.Tensor,
    residual_norm: torch.Tensor,
    target_norm: torch.Tensor,
    baseline_norm: torch.Tensor,
    true_grid: torch.Tensor,
    sensor_z: torch.Tensor,
    base_gid: torch.Tensor,
    config: CandidateConfig,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    pred_raw = denormalize_torch(pred_norm, y_mean, y_std)
    pred_clipped = clip_params_torch(pred_raw, lower, upper)
    pred_grid = depth_grid_torch(pred_clipped)
    profile = torch.nn.functional.smooth_l1_loss(pred_grid / scale, true_grid / scale)
    dim = torch.nn.functional.smooth_l1_loss(pred_norm[:, :3], target_norm[:, :3])
    curv = torch.nn.functional.smooth_l1_loss(pred_norm[:, 3:], target_norm[:, 3:])
    nominal = torch.abs(sensor_z.reshape(-1) - 0.008) < 5.0e-4
    if torch.any(nominal):
        nominal_zero = torch.nn.functional.smooth_l1_loss(residual_norm[nominal], torch.zeros_like(residual_norm[nominal]))
        nominal_anchor = torch.nn.functional.smooth_l1_loss(pred_norm[nominal], baseline_norm[nominal])
        nominal_loss = 0.5 * nominal_zero + 0.5 * nominal_anchor
    else:
        nominal_loss = torch.tensor(0.0, dtype=pred_norm.dtype, device=pred_norm.device)
    paired = paired_consistency_loss(pred_norm, pred_grid, base_gid, scale)
    bound = soft_bound_penalty(pred_raw, lower, upper)
    return (
        config.profile_weight * profile
        + config.dimension_weight * dim
        + config.curvature_aux_weight * curv
        + config.nominal_zero_weight * nominal_loss
        + config.paired_consistency_weight * paired
        + config.bound_weight * bound
    )


def model_for(config: CandidateConfig, input_dim: int | None = None) -> nn.Module:
    if config.kind in {"output_residual", "latent_residual"}:
        if input_dim is None:
            raise ValueError("input_dim required for residual adapter")
        return ResidualAdapter(input_dim)
    if config.kind == "full_sensor_z":
        return SensorZConditionedRegressor()
    raise ValueError(config.kind)


def features_for(config: CandidateConfig, arrays: dict[str, np.ndarray], z_norm: np.ndarray) -> np.ndarray | None:
    if config.kind == "output_residual":
        return np.concatenate([arrays["pred_norm"], z_norm], axis=1).astype(np.float32)
    if config.kind == "latent_residual":
        return np.concatenate([arrays["latent"], arrays["pred_norm"], z_norm], axis=1).astype(np.float32)
    return None


def predict_candidate(model: nn.Module, config: CandidateConfig, x_norm: np.ndarray, z_norm: np.ndarray, features: np.ndarray | None, baseline_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        if config.kind in {"output_residual", "latent_residual"}:
            pred, residual = model(torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(baseline_norm, dtype=torch.float32))
        else:
            pred = model(torch.as_tensor(x_norm, dtype=torch.float32), torch.as_tensor(z_norm, dtype=torch.float32))
            residual = pred - torch.as_tensor(baseline_norm, dtype=torch.float32)
    return pred.cpu().numpy().astype(np.float32), residual.cpu().numpy().astype(np.float32)


def fast_metrics_for_selection(
    pred_norm: np.ndarray,
    dataset: Any,
    stats: dict[str, np.ndarray],
    indices: np.ndarray,
    scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, float]:
    y_norm = liftoff.normalize_y(dataset, stats)
    with torch.no_grad():
        pred = torch.as_tensor(pred_norm[indices], dtype=torch.float32)
        pred_raw = clip_params_torch(
            denormalize_torch(pred, torch.as_tensor(stats["y_mean"], dtype=torch.float32), torch.as_tensor(stats["y_std"], dtype=torch.float32)),
            lower,
            upper,
        )
        pred_grid = depth_grid_torch(pred_raw)
        true_grid = torch.as_tensor(dataset.profile_depth_grid_m[indices], dtype=torch.float32)
        diff = pred_grid - true_grid
        rmse = torch.sqrt(torch.mean(diff * diff, dim=(1, 2))).cpu().numpy()
        er = torch.sqrt(torch.sum(diff * diff, dim=(1, 2)) / torch.clamp(torch.sum(true_grid * true_grid, dim=(1, 2)), min=1.0e-20)).cpu().numpy()
    abs_norm = np.abs(pred_norm[indices] - y_norm[indices])
    return {
        "profile_depth_rmse_m": float(np.mean(rmse)),
        "profile_depth_rmse_norm": float(np.mean(rmse)) / max(scale, 1.0e-12),
        "er_like_profile_error": float(np.mean(er)),
        "dimension_mae_norm": float(abs_norm[:, :3].mean()),
        "curvature_mae_norm": float(abs_norm[:, 3:].mean()),
    }


def nominal_preserving_selection_score(
    val_nom: dict[str, float],
    val_non: dict[str, float],
    c0_nom: dict[str, float],
    c0_non: dict[str, float],
) -> float:
    nominal_limit = 1.10 * float(c0_nom["profile_depth_rmse_m"])
    nominal_penalty = max(0.0, float(val_nom["profile_depth_rmse_m"]) - nominal_limit) / max(nominal_limit, 1.0e-12)
    non_penalty = max(0.0, float(val_non["profile_depth_rmse_m"]) - float(c0_non["profile_depth_rmse_m"])) / max(float(c0_non["profile_depth_rmse_m"]), 1.0e-12)
    return (
        val_non["profile_depth_rmse_norm"]
        + 0.75 * val_nom["profile_depth_rmse_norm"]
        + 1.50 * nominal_penalty
        + 0.50 * non_penalty
        + 0.25 * val_non["er_like_profile_error"]
        + 0.20 * (val_nom["dimension_mae_norm"] + val_non["dimension_mae_norm"])
        + 0.05 * val_non["curvature_mae_norm"]
    )


def train_one(
    config: CandidateConfig,
    seed: int,
    dataset: Any,
    stats: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    baseline_metrics: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_seed(seed)
    splits = liftoff.split_indices(dataset)
    x_norm = liftoff.normalize_x(dataset, stats)
    z_norm = liftoff.normalize_sensor_z(dataset, stats)
    y_norm = liftoff.normalize_y(dataset, stats)
    group_ids = base_group_ids(dataset)
    lower, upper = param_bounds(dataset, splits)
    scale = depth_scale(dataset, splits)
    features = features_for(config, baseline, z_norm)
    input_dim = None if features is None else int(features.shape[1])
    model = model_for(config, input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = splits["train"]
    train_order = np.asarray(sorted(train_idx.tolist(), key=lambda idx: (str(dataset.base_sample_ids[idx]), float(dataset.sensor_z_m[idx]))), dtype=np.int64)
    if features is None:
        train_ds = TensorDataset(
            torch.as_tensor(x_norm[train_order], dtype=torch.float32),
            torch.as_tensor(z_norm[train_order], dtype=torch.float32),
            torch.as_tensor(y_norm[train_order], dtype=torch.float32),
            torch.as_tensor(baseline["pred_norm"][train_order], dtype=torch.float32),
            torch.as_tensor(dataset.profile_depth_grid_m[train_order], dtype=torch.float32),
            torch.as_tensor(dataset.sensor_z_m[train_order], dtype=torch.float32),
            torch.as_tensor(group_ids[train_order], dtype=torch.long),
        )
    else:
        train_ds = TensorDataset(
            torch.as_tensor(features[train_order], dtype=torch.float32),
            torch.as_tensor(y_norm[train_order], dtype=torch.float32),
            torch.as_tensor(baseline["pred_norm"][train_order], dtype=torch.float32),
            torch.as_tensor(dataset.profile_depth_grid_m[train_order], dtype=torch.float32),
            torch.as_tensor(dataset.sensor_z_m[train_order], dtype=torch.float32),
            torch.as_tensor(group_ids[train_order], dtype=torch.long),
        )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    y_mean = torch.as_tensor(stats["y_mean"], dtype=torch.float32)
    y_std = torch.as_tensor(stats["y_std"], dtype=torch.float32)
    c0_val_nom = next(row for row in baseline_metrics if row["split"] == "val" and row["liftoff_subset"] == "nominal_0p008")
    c0_val_non = next(row for row in baseline_metrics if row["split"] == "val" and row["liftoff_subset"] == "non_nominal")
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            if features is None:
                xb, zb, yb, bb, gb, sz, bg = batch
                pred = model(xb, zb)
                residual = pred - bb
            else:
                fb, yb, bb, gb, sz, bg = batch
                pred, residual = model(fb, bb)
            loss = adapter_loss(pred, residual, yb, bb, gb, sz, bg, config, y_mean, y_std, lower, upper, scale)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        pred_norm, _resid = predict_candidate(model, config, x_norm, z_norm, features, baseline["pred_norm"])
        val_idx = splits["val"]
        val_nom_idx = val_idx[np.isclose(dataset.sensor_z_m[val_idx], 0.008, atol=5.0e-4)]
        val_non_idx = val_idx[~np.isclose(dataset.sensor_z_m[val_idx], 0.008, atol=5.0e-4)]
        val_nom = fast_metrics_for_selection(pred_norm, dataset, stats, val_nom_idx, scale, lower, upper)
        val_non = fast_metrics_for_selection(pred_norm, dataset, stats, val_non_idx, scale, lower, upper)
        score = nominal_preserving_selection_score(val_nom, val_non, c0_val_nom, c0_val_non)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "candidate": config.name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_nominal_profile_depth_rmse_m": val_nom["profile_depth_rmse_m"],
                "val_non_nominal_profile_depth_rmse_m": val_non["profile_depth_rmse_m"],
                "val_non_nominal_er_like_profile_error": val_non["er_like_profile_error"],
                "val_nominal_dimension_mae_norm": val_nom["dimension_mae_norm"],
                "val_non_nominal_dimension_mae_norm": val_non["dimension_mae_norm"],
                "val_selection_score": score,
            }
        )
    if best_state is None:
        raise RuntimeError(f"no validation checkpoint for {config.name} seed={seed}")
    model.load_state_dict(best_state)
    pred_norm, residual_norm = predict_candidate(model, config, x_norm, z_norm, features, baseline["pred_norm"])
    pred_raw = liftoff.denormalize_y(pred_norm, stats)
    return {
        "candidate": config.name,
        "seed": seed,
        "description": config.description,
        "model": model,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "pred_norm": pred_norm,
        "pred_raw": pred_raw,
        "residual_norm": residual_norm,
        "epoch_rows": epoch_rows,
    }


def evaluate_result(dataset: Any, stats: dict[str, np.ndarray], result: dict[str, Any], selected: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, by_liftoff = liftoff_train.evaluate_run(dataset, stats, result, selected)
    for row in metrics + by_liftoff:
        row["selection_score"] = result["best_val_score"] if row["split"] == "val" else ""
        row["validation_only_selection"] = True
        row["test_final_only"] = row["split"] == "test"
    return metrics, by_liftoff


def baseline_as_rows(rows: list[dict[str, Any]], selected: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cur = dict(row)
        cur["candidate"] = "A0_baseline_replay"
        cur["seed"] = 42
        cur["selected_seed"] = selected
        cur["best_epoch"] = ""
        cur["best_val_selection_metric"] = ""
        cur["selection_score"] = ""
        cur["validation_only_selection"] = True
        cur["test_final_only"] = cur.get("split") == "test"
        out.append(cur)
    return out


def select_candidate(metrics: list[dict[str, Any]]) -> tuple[str, int, bool]:
    val_rows = [row for row in metrics if row["split"] == "val" and row["liftoff_subset"] == "non_nominal" and str(row["candidate"]).startswith("A")]
    val_rows = [row for row in val_rows if row["candidate"] != "A0_baseline_replay"]
    if not val_rows:
        raise RuntimeError("no adapter validation rows")
    selected = min(val_rows, key=lambda row: float(row["best_val_selection_metric"]))
    candidate = str(selected["candidate"])
    seed = int(selected["seed"])
    c0_nom = next(row for row in metrics if row["candidate"] == "A0_baseline_replay" and row["split"] == "val" and row["liftoff_subset"] == "nominal_0p008")
    c0_non = next(row for row in metrics if row["candidate"] == "A0_baseline_replay" and row["split"] == "val" and row["liftoff_subset"] == "non_nominal")
    sel_nom = next(row for row in metrics if row["candidate"] == candidate and int(row["seed"]) == seed and row["split"] == "val" and row["liftoff_subset"] == "nominal_0p008")
    sel_non = next(row for row in metrics if row["candidate"] == candidate and int(row["seed"]) == seed and row["split"] == "val" and row["liftoff_subset"] == "non_nominal")
    nominal_ok = float(sel_nom["profile_depth_rmse_m"]) <= 1.10 * float(c0_nom["profile_depth_rmse_m"])
    non_improve = float(sel_non["profile_depth_rmse_m"]) <= 0.80 * float(c0_non["profile_depth_rmse_m"])
    return candidate, seed, bool(nominal_ok and non_improve)


def seed_summary(metrics: list[dict[str, Any]], selected_candidate: str, selected_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted({str(row["candidate"]) for row in metrics}):
        for seed in sorted({str(row["seed"]) for row in metrics if row["candidate"] == candidate}):
            subset = [row for row in metrics if row["candidate"] == candidate and str(row["seed"]) == seed]
            test_nom = next((row for row in subset if row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008"), None)
            test_non = next((row for row in subset if row["split"] == "test" and row["liftoff_subset"] == "non_nominal"), None)
            val_non = next((row for row in subset if row["split"] == "val" and row["liftoff_subset"] == "non_nominal"), None)
            if not test_nom or not test_non:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "selected_robustness_candidate": candidate == selected_candidate and str(seed) == str(selected_seed),
                    "best_epoch": test_non.get("best_epoch", ""),
                    "best_val_selection_metric": test_non.get("best_val_selection_metric", ""),
                    "val_non_nominal_profile_depth_rmse_m": val_non.get("profile_depth_rmse_m", "") if val_non else "",
                    "test_nominal_profile_depth_rmse_m": test_nom["profile_depth_rmse_m"],
                    "test_non_nominal_profile_depth_rmse_m": test_non["profile_depth_rmse_m"],
                    "test_non_nominal_projected_mask_dice": test_non["projected_mask_dice"],
                    "test_non_nominal_L_mae_mm": test_non["L_mae_mm"],
                    "test_non_nominal_W_mae_mm": test_non["W_mae_mm"],
                    "test_non_nominal_D_mae_mm": test_non["D_mae_mm"],
                    "test_non_nominal_wMAE_auxiliary": test_non["wMAE_auxiliary"],
                }
            )
    return rows


def vs_baseline_rows(metrics: list[dict[str, Any]], selected_candidate: str, selected_seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for subset in ("all_liftoff", "nominal_0p008", "non_nominal"):
        ref = next(row for row in metrics if row["candidate"] == "A0_baseline_replay" and row["split"] == "test" and row["liftoff_subset"] == subset)
        cur = next(row for row in metrics if row["candidate"] == selected_candidate and str(row["seed"]) == str(selected_seed) and row["split"] == "test" and row["liftoff_subset"] == subset)
        for metric in ["profile_depth_rmse_m", "er_like_profile_error", "normalized_param_mae", "L_mae_mm", "W_mae_mm", "D_mae_mm", "wMAE_auxiliary", "projected_mask_iou", "projected_mask_dice"]:
            ref_value = float(ref[metric])
            cur_value = float(cur[metric])
            lower_better = metric not in {"projected_mask_iou", "projected_mask_dice"}
            out.append(
                {
                    "liftoff_subset": subset,
                    "metric": metric,
                    "reference_candidate": "A0_baseline_replay",
                    "current_candidate": selected_candidate,
                    "current_seed": selected_seed,
                    "reference_value": ref_value,
                    "current_value": cur_value,
                    "delta": cur_value - ref_value,
                    "relative_change_pct": 0.0 if abs(ref_value) < 1.0e-20 else 100.0 * (cur_value - ref_value) / ref_value,
                    "improved": cur_value < ref_value if lower_better else cur_value > ref_value,
                }
            )
    return out


def run_training(
    output_summary: Path,
    output_metrics: Path,
    output_by_liftoff: Path,
    seed_output: Path | None,
    vs_output: Path | None,
    selected_output: Path | None,
    selected_only: str | None,
    seeds: list[int],
    args: argparse.Namespace,
) -> tuple[str, int, bool, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = liftoff.load_liftoff_dataset(args.dataset_id)
    if liftoff.base_split_leakage(dataset):
        raise RuntimeError(f"base split leakage detected: {liftoff.base_split_leakage(dataset)}")
    if not liftoff.paired_liftoff_complete(dataset):
        raise RuntimeError("paired liftoff completeness failed")
    stats = liftoff.train_normalization(dataset)
    baseline_rows, baseline_by = replay_baseline(args.dataset_id)
    _artifact, checkpoint, baseline_model = load_baseline_model()
    baseline = baseline_arrays(dataset, stats, checkpoint, baseline_model)
    metric_rows: list[dict[str, Any]] = baseline_as_rows(baseline_rows)
    by_rows: list[dict[str, Any]] = baseline_as_rows(baseline_by)
    configs = candidate_configs(include_full_model=args.include_full_model)
    if selected_only:
        configs = [cfg for cfg in configs if cfg.name == selected_only]
        if not configs:
            raise RuntimeError(f"selected candidate config not found: {selected_only}")
    trained: list[dict[str, Any]] = []
    for config in configs:
        for seed in seeds:
            result = train_one(config, seed, dataset, stats, baseline, baseline_rows, args)
            trained.append(result)
            cur_metrics, cur_by = evaluate_result(dataset, stats, result, selected=False)
            metric_rows.extend(cur_metrics)
            by_rows.extend(cur_by)
    selected_candidate, selected_seed, eligible = select_candidate(metric_rows)
    for row in metric_rows + by_rows:
        row["selected_seed"] = row["candidate"] == selected_candidate and str(row["seed"]) == str(selected_seed)
        row["eligible_for_multiseed_or_candidate"] = eligible
    summary_lines = summary_lines_for(metric_rows, selected_candidate, selected_seed, eligible, args, selected_only is not None)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_csv(output_metrics, metric_rows)
    write_csv(output_by_liftoff, by_rows)
    seed_rows = seed_summary(metric_rows, selected_candidate, selected_seed)
    vs_rows = vs_baseline_rows(metric_rows, selected_candidate, selected_seed)
    if seed_output:
        write_csv(seed_output, seed_rows)
    if vs_output:
        write_csv(vs_output, vs_rows)
    if selected_output:
        write_csv(selected_output, [{"selected_candidate": selected_candidate, "selected_seed": selected_seed, "eligible": eligible}])
    return selected_candidate, selected_seed, eligible, metric_rows, by_rows, vs_rows


def summary_lines_for(metrics: list[dict[str, Any]], selected_candidate: str, selected_seed: int, eligible: bool, args: argparse.Namespace, multiseed: bool) -> list[str]:
    sel = [row for row in metrics if row["candidate"] == selected_candidate and str(row["seed"]) == str(selected_seed) and row["split"] == "test"]
    c0 = [row for row in metrics if row["candidate"] == "A0_baseline_replay" and row["split"] == "test"]
    nom = next(row for row in sel if row["liftoff_subset"] == "nominal_0p008")
    non = next(row for row in sel if row["liftoff_subset"] == "non_nominal")
    c0_nom = next(row for row in c0 if row["liftoff_subset"] == "nominal_0p008")
    c0_non = next(row for row in c0 if row["liftoff_subset"] == "non_nominal")
    non_change = 100.0 * (float(non["profile_depth_rmse_m"]) - float(c0_non["profile_depth_rmse_m"])) / float(c0_non["profile_depth_rmse_m"])
    nom_change = 100.0 * (float(nom["profile_depth_rmse_m"]) - float(c0_nom["profile_depth_rmse_m"])) / float(c0_nom["profile_depth_rmse_m"])
    title = "20.94 liftoff adapter selected candidate multi-seed training" if multiseed else "20.94 liftoff adapter candidate screen"
    return [
        title,
        "",
        f"dataset_id: {args.dataset_id}",
        "COMSOL_run: false",
        "data_or_npz_write: false",
        "CURRENT_BASELINE_update: false",
        "checkpoint_saved_or_committed: false",
        "selection: validation-only nominal-preserving score; test final only",
        "base_sample_id_usage: split grouping and paired-consistency regularization only; not a model input feature",
        f"seeds: {','.join(str(seed) for seed in args.seeds)}",
        f"selected_candidate: {selected_candidate}",
        f"selected_seed: {selected_seed}",
        f"eligible: {eligible}",
        f"test_nominal_profile_depth_rmse_m: {float(nom['profile_depth_rmse_m']):.9f}",
        f"test_nominal_profile_rmse_change_vs_C0_pct: {nom_change:.3f}",
        f"test_non_nominal_profile_depth_rmse_m: {float(non['profile_depth_rmse_m']):.9f}",
        f"test_non_nominal_profile_rmse_change_vs_C0_pct: {non_change:.3f}",
        f"test_non_nominal_projected_mask_iou_dice: {float(non['projected_mask_iou']):.6f}/{float(non['projected_mask_dice']):.6f}",
        f"test_non_nominal_LWD_MAE_mm: {float(non['L_mae_mm']):.3f}/{float(non['W_mae_mm']):.3f}/{float(non['D_mae_mm']):.3f}",
        f"test_non_nominal_wMAE_auxiliary: {float(non['wMAE_auxiliary']):.6f}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=liftoff.DATASET_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--include-full-model", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.seeds = [args.seed] if args.seeds is None else args.seeds
    selected_candidate, selected_seed, eligible, _metrics, _by, _vs = run_training(
        SUMMARY,
        METRICS,
        BY_LIFTOFF,
        None,
        None,
        SELECTED_JSON,
        None,
        args.seeds,
        args,
    )
    print(f"selected={selected_candidate} seed={selected_seed} eligible={eligible}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

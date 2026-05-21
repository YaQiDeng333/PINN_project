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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_rect_rot_geometry_forward_surrogate as old_forward  # noqa: E402
import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = base.DEFAULT_NPZ
DEFAULT_LABELS = base.DEFAULT_LABELS
DEFAULT_PROPOSAL = PROJECT_ROOT / "results/metrics/comsol_rect_rot_proposal_extraction_selected_geometry.csv"
DEFAULT_2054_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_improved_dense_priewald_refinement_metrics.csv"
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_forward_surrogate_mismatch_audit_summary.txt"
DEFAULT_AUDIT = PROJECT_ROOT / "results/metrics/comsol_rect_rot_forward_surrogate_mismatch_audit.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_calibrated_forward_surrogate_summary.txt"
DEFAULT_CANDIDATES = PROJECT_ROOT / "results/metrics/comsol_rect_rot_calibrated_forward_surrogate_candidates.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_calibrated_forward_surrogate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_calibrated_forward_surrogate_epoch_log.csv"
DEFAULT_CALIBRATION = PROJECT_ROOT / "results/metrics/comsol_rect_rot_calibrated_forward_surrogate_calibration.csv"

TARGET_SHAPE = (3, 201)
MASK_FEATURE_SHAPE = (8, 16)
MAX_ANGLE_RAD = math.radians(35.0)

CANDIDATE_NAMES = ["S1_geom_mlp_waveform", "S2_geom_mask_mlp_waveform", "S3_geom_mlp_peakaware"]


@dataclass
class CandidateBundle:
    name: str
    model: nn.Module
    arrays: dict[str, Any]
    diagnostics: dict[str, Any]
    best_epoch: int
    best_val_loss: float
    device: torch.device


class ForwardDataset(Dataset):
    def __init__(self, indices: np.ndarray, arrays: dict[str, Any], candidate: str):
        self.indices = indices.astype(np.int64)
        self.inputs = build_candidate_input_numpy(arrays, self.indices, candidate)
        self.targets = arrays["signals_norm"][self.indices].astype(np.float32).reshape(len(self.indices), -1)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "source_index": torch.tensor(int(self.indices[idx]), dtype=torch.long),
            "input": torch.from_numpy(self.inputs[idx]).float(),
            "target": torch.from_numpy(self.targets[idx]).float(),
        }


class CalibratedForwardSurrogate(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 603):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [as_float(row, key) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray([v for v in xs], dtype=np.float64)
    y = np.asarray([v for v in ys], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def load_arrays(npz: Path, labels: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays, diagnostics = base.load_arrays(npz, labels)
    arrays = dict(arrays)
    arrays["true_angle"] = np.arctan2(arrays["angle_targets"][:, 0], arrays["angle_targets"][:, 1]).astype(np.float32)
    diagnostics = dict(diagnostics)
    diagnostics["target_shape"] = TARGET_SHAPE
    return arrays, diagnostics


def geom_input_np(arrays: dict[str, Any], indices: np.ndarray, geom: np.ndarray, angle: np.ndarray, type_prob: np.ndarray) -> np.ndarray:
    geom_norm = (geom - arrays["geom_mean"].reshape(1, -1)) / np.maximum(arrays["geom_std"].reshape(1, -1), 1e-8)
    return np.concatenate([type_prob, geom_norm, np.sin(angle)[:, None], np.cos(angle)[:, None]], axis=1).astype(np.float32)


def geom_input_torch(
    arrays: dict[str, Any],
    geom: torch.Tensor,
    angle: torch.Tensor,
    type_prob: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    geom_mean = torch.tensor(arrays["geom_mean"], dtype=torch.float32, device=device).view(1, -1)
    geom_std = torch.tensor(arrays["geom_std"], dtype=torch.float32, device=device).view(1, -1)
    geom_norm = (geom - geom_mean) / geom_std.clamp_min(1e-8)
    return torch.cat([type_prob, geom_norm, torch.sin(angle).unsqueeze(1), torch.cos(angle).unsqueeze(1)], dim=1)


def mask_features_torch(
    arrays: dict[str, Any],
    geom: torch.Tensor,
    angle: torch.Tensor,
    type_prob: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    mask_x = torch.tensor(arrays["mask_x"], dtype=torch.float32, device=device)
    mask_y = torch.tensor(arrays["mask_y"], dtype=torch.float32, device=device)
    rect = base.soft_rect_mask(mask_x, mask_y, geom[:, 0], geom[:, 1], geom[:, 2], geom[:, 3], torch.zeros_like(angle))
    rot = base.soft_rect_mask(mask_x, mask_y, geom[:, 0], geom[:, 1], geom[:, 2], geom[:, 3], angle)
    mask = type_prob[:, 0].view(-1, 1, 1) * rect + type_prob[:, 1].view(-1, 1, 1) * rot
    pooled = F.interpolate(mask.unsqueeze(1), size=MASK_FEATURE_SHAPE, mode="area")
    return pooled.flatten(1)


def build_input_from_params_torch(
    arrays: dict[str, Any],
    geom: torch.Tensor,
    angle: torch.Tensor,
    type_prob: torch.Tensor,
    candidate: str,
    device: torch.device,
) -> torch.Tensor:
    base_input = geom_input_torch(arrays, geom, angle, type_prob, device)
    if candidate == "S2_geom_mask_mlp_waveform":
        return torch.cat([base_input, mask_features_torch(arrays, geom, angle, type_prob, device)], dim=1)
    return base_input


def build_candidate_input_numpy(arrays: dict[str, Any], indices: np.ndarray, candidate: str) -> np.ndarray:
    geom = arrays["raw_geom"][indices].astype(np.float32)
    angle = arrays["true_angle"][indices].astype(np.float32)
    type_prob = np.eye(2, dtype=np.float32)[arrays["type_targets"][indices]]
    base_input = geom_input_np(arrays, indices, geom, angle, type_prob)
    if candidate != "S2_geom_mask_mlp_waveform":
        return base_input
    device = torch.device("cpu")
    with torch.no_grad():
        mask_feat = mask_features_torch(
            arrays,
            torch.from_numpy(geom).float(),
            torch.from_numpy(angle).float(),
            torch.from_numpy(type_prob).float(),
            device,
        ).numpy()
    return np.concatenate([base_input, mask_feat], axis=1).astype(np.float32)


def forward_loss(candidate: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_sig = pred.view(-1, *TARGET_SHAPE)
    target_sig = target.view(-1, *TARGET_SHAPE)
    mse = F.mse_loss(pred_sig, target_sig)
    mae = F.l1_loss(pred_sig, target_sig)
    grad_loss = F.mse_loss(pred_sig[:, :, 1:] - pred_sig[:, :, :-1], target_sig[:, :, 1:] - target_sig[:, :, :-1])
    if candidate in {"S1_geom_mlp_waveform", "S2_geom_mask_mlp_waveform"}:
        return mse + 0.2 * mae + 0.1 * grad_loss
    pred_amp = pred_sig.amax(dim=2) - pred_sig.amin(dim=2)
    true_amp = target_sig.amax(dim=2) - target_sig.amin(dim=2)
    amp_loss = F.l1_loss(pred_amp, true_amp)
    abs_target = target_sig.abs()
    weights = 1.0 + 3.0 * abs_target / abs_target.amax(dim=2, keepdim=True).clamp_min(1e-6)
    peak_loss = torch.mean(weights * (pred_sig - target_sig) ** 2)
    return mse + 0.2 * amp_loss + 0.1 * grad_loss + 0.1 * peak_loss


def waveform_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    base_metrics = old_forward.signal_metrics(pred, target)
    grad_mse = float(np.mean(((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])) ** 2))
    pred_amp = pred.max(axis=1) - pred.min(axis=1)
    true_amp = target.max(axis=1) - target.min(axis=1)
    peak_amp_err = []
    line_corr = []
    for line in range(TARGET_SHAPE[0]):
        p = pred[line]
        t = target[line]
        peak_amp_err.append(abs(float(p[int(np.argmax(np.abs(t)))]) - float(t[int(np.argmax(np.abs(t)))])))
        line_corr.append(corr(list(p), list(t)))
    return {
        **base_metrics,
        "gradient_mse": grad_mse,
        "peak_amplitude_abs_error": float(np.mean(peak_amp_err)),
        "line0_correlation": line_corr[0],
        "line1_correlation": line_corr[1],
        "line2_correlation": line_corr[2],
    }


def predict(model: nn.Module, ds: ForwardDataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["input"].to(device))
            chunks["indices"].append(batch["source_index"].cpu().numpy())
            chunks["pred"].append(pred.cpu().numpy())
            chunks["target"].append(batch["target"].cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def metric_rows(candidate: str, pred: dict[str, np.ndarray], arrays: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, local_idx_raw in enumerate(pred["indices"]):
        local_idx = int(local_idx_raw)
        stats = waveform_metrics(pred["pred"][order].reshape(TARGET_SHAPE), pred["target"][order].reshape(TARGET_SHAPE))
        rows.append(
            {
                "candidate": candidate,
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": split,
                "defect_type": str(arrays["defect_types"][local_idx]),
                "source_pack": str(arrays["source_packs"][local_idx]),
                **stats,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], split: str, candidate: str | None = None) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split and (candidate is None or row["candidate"] == candidate)]
    keys = [
        "mse",
        "mae",
        "rmse",
        "nrmse",
        "correlation",
        "amplitude_abs_error",
        "abs_peak_index_error_mean",
        "gradient_mse",
        "peak_amplitude_abs_error",
        "line0_correlation",
        "line1_correlation",
        "line2_correlation",
    ]
    return {key: safe_mean(subset, key) for key in keys} | {"sample_count": float(len(subset))}


def train_candidate(candidate: str, args: argparse.Namespace, arrays: dict[str, Any], diagnostics: dict[str, Any]) -> CandidateBundle:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = ForwardDataset(arrays["split_indices"]["train"], arrays, candidate)
    val_ds = ForwardDataset(arrays["split_indices"]["val"], arrays, candidate)
    input_dim = int(train_ds.inputs.shape[1])
    model = CalibratedForwardSurrogate(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["input"].to(device))
            target = batch["target"].to(device)
            loss = forward_loss(candidate, pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                losses.append(float(forward_loss(candidate, model(batch["input"].to(device)), batch["target"].to(device)).cpu()))
        val_loss = float(np.mean(losses))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"{candidate} epoch={epoch:03d} val_loss={val_loss:.5f}")
    if best_state is None:
        raise RuntimeError(f"No validation checkpoint selected for {candidate}")
    model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return CandidateBundle(candidate, model, arrays, diagnostics, best_epoch, best_val_loss, device)


def evaluate_bundle(bundle: CandidateBundle, batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        ds = ForwardDataset(bundle.arrays["split_indices"][split], bundle.arrays, bundle.name)
        rows.extend(metric_rows(bundle.name, predict(bundle.model, ds, bundle.device, batch_size), bundle.arrays, split))
    return rows


def proposal_rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in read_csv(path)}


def refinement_rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in read_csv(path)}


def state_arrays(
    state: str,
    local_idx: int,
    arrays: dict[str, Any],
    proposal_by_id: dict[str, dict[str, str]],
    refine_by_id: dict[str, dict[str, str]],
) -> tuple[np.ndarray, float, np.ndarray, float, float, float]:
    sample_id = str(arrays["sample_ids"][local_idx])
    true_geom = arrays["raw_geom"][local_idx].astype(np.float32)
    true_angle = float(arrays["true_angle"][local_idx])
    true_type = np.eye(2, dtype=np.float32)[arrays["type_targets"][local_idx]]
    if state == "true":
        return true_geom, true_angle, true_type, 1.0, 1.0, 0.0
    if state == "proposal":
        row = proposal_by_id[sample_id]
        geom = np.array(
            [
                as_float(row, "pred_center_x"),
                as_float(row, "pred_center_y"),
                as_float(row, "pred_width"),
                as_float(row, "pred_length"),
                as_float(row, "pred_depth"),
            ],
            dtype=np.float32,
        )
        type_prob = np.array(
            [as_float(row, "type_prob_rectangular_notch", 0.5), as_float(row, "type_prob_rotated_rect", 0.5)],
            dtype=np.float32,
        )
        type_prob = type_prob / max(float(type_prob.sum()), 1e-8)
        return geom, math.radians(as_float(row, "pred_angle_deg", 0.0)), type_prob, as_float(row, "geometry_iou"), as_float(row, "geometry_dice"), as_float(row, "geometry_area_error")
    if state == "refined_20_54":
        row = refine_by_id[sample_id]
        geom = np.array(
            [
                as_float(row, "refined_center_x"),
                as_float(row, "refined_center_y"),
                as_float(row, "refined_width"),
                as_float(row, "refined_length"),
                as_float(row, "refined_depth"),
            ],
            dtype=np.float32,
        )
        type_prob = np.array(
            [as_float(row, "type_prob_rectangular_notch", 0.5), as_float(row, "type_prob_rotated_rect", 0.5)],
            dtype=np.float32,
        )
        type_prob = type_prob / max(float(type_prob.sum()), 1e-8)
        return geom, math.radians(as_float(row, "refined_angle_deg", 0.0)), type_prob, as_float(row, "refined_iou"), as_float(row, "refined_dice"), as_float(row, "refined_area_error")
    if state == "jittered_bad":
        geom = true_geom.copy()
        geom[0] = np.clip(geom[0] + 0.0012, arrays["mask_x"].min(), arrays["mask_x"].max())
        geom[1] = np.clip(geom[1] - 0.0007, arrays["mask_y"].min(), arrays["mask_y"].max())
        geom[2] = np.clip(geom[2] * 1.20, 0.001, 0.025)
        geom[3] = np.clip(geom[3] * 0.85, 0.001, 0.020)
        geom[4] = np.clip(geom[4] * 0.80, 0.0001, 0.004)
        angle = float(np.clip(true_angle + math.radians(12.0), -MAX_ANGLE_RAD, MAX_ANGLE_RAD))
        iou, dice, area = geometry_mask_metric(arrays, local_idx, geom, angle, true_type)
        return geom, angle, true_type, iou, dice, area
    raise ValueError(state)


def geometry_mask_metric(arrays: dict[str, Any], local_idx: int, geom: np.ndarray, angle: float, type_prob: np.ndarray) -> tuple[float, float, float]:
    device = torch.device("cpu")
    with torch.no_grad():
        mask = mask_from_params_torch(
            arrays,
            torch.from_numpy(geom[None, :]).float(),
            torch.tensor([angle], dtype=torch.float32),
            torch.from_numpy(type_prob[None, :]).float(),
            device,
        )[0].numpy()
    metric = base.mask_metric(mask, arrays["masks"][local_idx], 0.5)
    return metric["iou"], metric["dice"], metric["area_error"]


def mask_from_params_torch(
    arrays: dict[str, Any],
    geom: torch.Tensor,
    angle: torch.Tensor,
    type_prob: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    mask_x = torch.tensor(arrays["mask_x"], dtype=torch.float32, device=device)
    mask_y = torch.tensor(arrays["mask_y"], dtype=torch.float32, device=device)
    rect = base.soft_rect_mask(mask_x, mask_y, geom[:, 0], geom[:, 1], geom[:, 2], geom[:, 3], torch.zeros_like(angle))
    rot = base.soft_rect_mask(mask_x, mask_y, geom[:, 0], geom[:, 1], geom[:, 2], geom[:, 3], angle)
    return type_prob[:, 0].view(-1, 1, 1) * rect + type_prob[:, 1].view(-1, 1, 1) * rot


def residual_for_state(bundle: CandidateBundle, local_idx: int, geom: np.ndarray, angle: float, type_prob: np.ndarray) -> dict[str, float]:
    device = bundle.device
    with torch.no_grad():
        inp = build_input_from_params_torch(
            bundle.arrays,
            torch.from_numpy(geom[None, :]).float().to(device),
            torch.tensor([angle], dtype=torch.float32, device=device),
            torch.from_numpy(type_prob[None, :]).float().to(device),
            bundle.name,
            device,
        )
        pred = bundle.model(inp).view(1, *TARGET_SHAPE)[0].cpu().numpy()
    obs = bundle.arrays["signals_norm"][local_idx]
    return waveform_metrics(pred, obs)


def calibration_rows(bundle: CandidateBundle, proposal_path: Path, refinement_path: Path) -> list[dict[str, Any]]:
    proposal_by_id = proposal_rows_by_id(proposal_path)
    refine_by_id = refinement_rows_by_id(refinement_path)
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        for local_idx in bundle.arrays["split_indices"][split]:
            local_idx = int(local_idx)
            for state in ["true", "proposal", "refined_20_54", "jittered_bad"]:
                geom, angle, type_prob, iou, dice, area = state_arrays(state, local_idx, bundle.arrays, proposal_by_id, refine_by_id)
                residual = residual_for_state(bundle, local_idx, geom, angle, type_prob)
                rows.append(
                    {
                        "candidate": bundle.name,
                        "sample_id": str(bundle.arrays["sample_ids"][local_idx]),
                        "source_index": int(bundle.arrays["source_indices"][local_idx]),
                        "split": split,
                        "defect_type": str(bundle.arrays["defect_types"][local_idx]),
                        "state": state,
                        "mask_iou": iou,
                        "mask_dice": dice,
                        "area_error": area,
                        "residual_mse": residual["mse"],
                        "residual_nrmse": residual["nrmse"],
                        "residual_correlation": residual["correlation"],
                        "peak_index_error": residual["abs_peak_index_error_mean"],
                        "amplitude_error": residual["amplitude_abs_error"],
                    }
                )
    return rows


def calibration_summary(cal_rows: list[dict[str, Any]], refinement_path: Path, candidate: str, split: str) -> dict[str, float]:
    subset = [row for row in cal_rows if row["candidate"] == candidate and row["split"] == split]
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in subset:
        by_sample[row["sample_id"]][row["state"]] = row
    order_hits = []
    mismatch = []
    residual_down = []
    delta_iou = []
    delta_dice = []
    residual_delta = []
    drift = []
    ref_by_id = refinement_rows_by_id(refinement_path)
    for sample_id, states in by_sample.items():
        if "true" in states:
            true_res = as_float(states["true"], "residual_nrmse")
            for other in ["proposal", "refined_20_54", "jittered_bad"]:
                if other in states:
                    order_hits.append(float(true_res < as_float(states[other], "residual_nrmse")))
        if "proposal" in states and "refined_20_54" in states:
            proposal_res = as_float(states["proposal"], "residual_nrmse")
            refined_res = as_float(states["refined_20_54"], "residual_nrmse")
            row = ref_by_id[sample_id]
            d_iou = as_float(row, "delta_iou")
            d_dice = as_float(row, "delta_dice")
            down = refined_res < proposal_res
            residual_down.append(float(down))
            mismatch.append(float(down and (d_iou < 0 or d_dice < 0)))
            delta_iou.append(d_iou)
            delta_dice.append(d_dice)
            residual_delta.append(proposal_res - refined_res)
            drift.append(as_float(row, "parameter_drift_norm"))
    residuals = [as_float(row, "residual_nrmse") for row in subset]
    mask_errors = [1.0 - as_float(row, "mask_iou") for row in subset]
    area_errors = [as_float(row, "area_error") for row in subset]
    return {
        "residual_ordering_accuracy": float(np.mean(order_hits)) if order_hits else math.nan,
        "mismatch_rate": float(np.mean(mismatch)) if mismatch else math.nan,
        "residual_down_rate": float(np.mean(residual_down)) if residual_down else math.nan,
        "residual_error_correlation": corr(residuals, mask_errors),
        "residual_area_correlation": corr(residuals, area_errors),
        "residual_reduction_iou_delta_correlation": corr(residual_delta, delta_iou),
        "residual_reduction_dice_delta_correlation": corr(residual_delta, delta_dice),
        "residual_reduction_drift_correlation": corr(residual_delta, drift),
    }


def write_mismatch_audit(args: argparse.Namespace) -> None:
    rows = read_csv(args.refinement_metrics)
    out_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        for defect in ["overall", "rectangular_notch", "rotated_rect"]:
            subset = [r for r in rows if r["split"] == split and (defect == "overall" or r["defect_type"] == defect)]
            if not subset:
                continue
            fwd_change = [as_float(r, "forward_nrmse_reduction") for r in subset]
            iou_delta = [as_float(r, "delta_iou") for r in subset]
            dice_delta = [as_float(r, "delta_dice") for r in subset]
            area_delta = [as_float(r, "delta_area_error") for r in subset]
            angle_delta = [as_float(r, "angle_error_delta") for r in subset if r["defect_type"] == "rotated_rect"]
            drift = [as_float(r, "parameter_drift_norm") for r in subset]
            patterns = {
                "residual_down_mask_up": 0,
                "residual_down_mask_down": 0,
                "residual_up_mask_up": 0,
                "residual_up_mask_down": 0,
                "high_forward_reduction_high_geometry_drift": 0,
                "high_forward_reduction_angle_worse": 0,
            }
            for r in subset:
                down = as_float(r, "forward_nrmse_reduction") > 0
                mask_up = as_float(r, "delta_iou") > 0 or as_float(r, "delta_dice") > 0
                if down and mask_up:
                    patterns["residual_down_mask_up"] += 1
                elif down and not mask_up:
                    patterns["residual_down_mask_down"] += 1
                elif (not down) and mask_up:
                    patterns["residual_up_mask_up"] += 1
                else:
                    patterns["residual_up_mask_down"] += 1
                if down and as_float(r, "parameter_drift_norm") > 0.75:
                    patterns["high_forward_reduction_high_geometry_drift"] += 1
                if down and r["defect_type"] == "rotated_rect" and as_float(r, "angle_error_delta") > 0:
                    patterns["high_forward_reduction_angle_worse"] += 1
            out_rows.append(
                {
                    "split": split,
                    "defect_type": defect,
                    "sample_count": len(subset),
                    "initial_forward_nrmse_mean": safe_mean(subset, "initial_forward_nrmse"),
                    "refined_forward_nrmse_mean": safe_mean(subset, "refined_forward_nrmse"),
                    "forward_nrmse_reduction_mean": float(np.nanmean(fwd_change)),
                    "delta_iou_mean": float(np.nanmean(iou_delta)),
                    "delta_dice_mean": float(np.nanmean(dice_delta)),
                    "delta_area_error_mean": float(np.nanmean(area_delta)),
                    "angle_error_delta_mean": float(np.nanmean(angle_delta)) if angle_delta else math.nan,
                    "parameter_drift_norm_mean": float(np.nanmean(drift)),
                    "forward_nrmse_vs_1_minus_initial_iou_corr": corr([as_float(r, "initial_forward_nrmse") for r in subset], [1.0 - as_float(r, "initial_iou") for r in subset]),
                    "forward_nrmse_vs_1_minus_refined_iou_corr": corr([as_float(r, "refined_forward_nrmse") for r in subset], [1.0 - as_float(r, "refined_iou") for r in subset]),
                    "forward_reduction_vs_iou_delta_corr": corr(fwd_change, iou_delta),
                    "forward_reduction_vs_dice_delta_corr": corr(fwd_change, dice_delta),
                    "forward_reduction_vs_area_delta_corr": corr(fwd_change, area_delta),
                    "forward_reduction_vs_drift_corr": corr(fwd_change, drift),
                    **patterns,
                }
            )
    write_csv(args.audit, out_rows)
    test_overall = next(row for row in out_rows if row["split"] == "test" and row["defect_type"] == "overall")
    lines = [
        "COMSOL rect/rot forward surrogate mismatch audit summary",
        "",
        f"Input 20.54 metrics: {args.refinement_metrics}",
        "Audit uses 20.54 pre-refine and post-refine rows only; no training or COMSOL run.",
        "",
        "Test overall:",
        f"- forward NRMSE reduction mean = {test_overall['forward_nrmse_reduction_mean']:.4f}",
        f"- delta IoU/Dice = {test_overall['delta_iou_mean']:.4f} / {test_overall['delta_dice_mean']:.4f}",
        f"- delta area_error = {test_overall['delta_area_error_mean']:.4f}",
        f"- forward reduction vs IoU delta corr = {test_overall['forward_reduction_vs_iou_delta_corr']:.4f}",
        f"- forward reduction vs Dice delta corr = {test_overall['forward_reduction_vs_dice_delta_corr']:.4f}",
        f"- residual_down_mask_down count = {test_overall['residual_down_mask_down']}",
        f"- residual_down_mask_up count = {test_overall['residual_down_mask_up']}",
        "",
        "Interpretation:",
        "20.54 residual decreased on most samples, but the mean IoU/Dice delta was negative. This confirms a residual-objective mismatch risk rather than an initializer/proposal blocker.",
    ]
    args.audit_summary.parent.mkdir(parents=True, exist_ok=True)
    args.audit_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace, write_outputs: bool = True) -> dict[str, Any]:
    write_mismatch_audit(args)
    arrays, diagnostics = load_arrays(args.npz, args.labels)
    all_metric_rows: list[dict[str, Any]] = []
    all_epoch_rows: list[dict[str, Any]] = []
    all_cal_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    bundles: dict[str, CandidateBundle] = {}
    for name in CANDIDATE_NAMES:
        bundle = train_candidate(name, args, arrays, diagnostics)
        bundles[name] = bundle
        metric = evaluate_bundle(bundle, args.batch_size)
        cal_rows = calibration_rows(bundle, args.proposal_geometry, args.refinement_metrics)
        all_metric_rows.extend(metric)
        all_cal_rows.extend(cal_rows)
        val_wave = summarize(metric, "val", name)
        test_wave = summarize(metric, "test", name)
        val_cal = calibration_summary(cal_rows, args.refinement_metrics, name, "val")
        test_cal = calibration_summary(cal_rows, args.refinement_metrics, name, "test")
        peak_norm = val_wave["abs_peak_index_error_mean"] / TARGET_SHAPE[1]
        corr_value = val_cal["residual_error_correlation"]
        corr_value = 0.0 if not math.isfinite(corr_value) else corr_value
        score = (
            -val_wave["nrmse"]
            + 0.30 * val_cal["residual_ordering_accuracy"]
            + 0.20 * corr_value
            - 0.20 * val_cal["mismatch_rate"]
            - 0.05 * peak_norm
        )
        usable = (
            val_wave["nrmse"] <= 0.463 + 0.05
            and val_cal["residual_ordering_accuracy"] > 0.50
            and val_cal["mismatch_rate"] < 0.60
            and corr_value > 0.05
        )
        candidate_rows.append(
            {
                "candidate": name,
                "best_epoch": bundle.best_epoch,
                "best_val_loss": bundle.best_val_loss,
                "val_nrmse": val_wave["nrmse"],
                "test_nrmse": test_wave["nrmse"],
                "val_correlation": val_wave["correlation"],
                "test_correlation": test_wave["correlation"],
                "val_peak_index_error": val_wave["abs_peak_index_error_mean"],
                "test_peak_index_error": test_wave["abs_peak_index_error_mean"],
                "val_gradient_mse": val_wave["gradient_mse"],
                "test_gradient_mse": test_wave["gradient_mse"],
                "val_residual_ordering_accuracy": val_cal["residual_ordering_accuracy"],
                "test_residual_ordering_accuracy": test_cal["residual_ordering_accuracy"],
                "val_residual_error_correlation": val_cal["residual_error_correlation"],
                "test_residual_error_correlation": test_cal["residual_error_correlation"],
                "val_mismatch_rate": val_cal["mismatch_rate"],
                "test_mismatch_rate": test_cal["mismatch_rate"],
                "val_residual_reduction_iou_delta_correlation": val_cal["residual_reduction_iou_delta_correlation"],
                "test_residual_reduction_iou_delta_correlation": test_cal["residual_reduction_iou_delta_correlation"],
                "surrogate_selection_score": score,
                "usable": usable,
            }
        )
        for row in metric:
            all_epoch_rows.append(
                {
                    "candidate": name,
                    "event": "best_checkpoint",
                    "epoch": bundle.best_epoch,
                    "best_val_loss": bundle.best_val_loss,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                }
            )
    selected = sorted(candidate_rows, key=lambda row: (row["usable"], row["surrogate_selection_score"]), reverse=True)[0]
    if write_outputs:
        write_csv(args.candidates, candidate_rows)
        write_csv(args.metrics, all_metric_rows)
        write_csv(args.epoch_log, all_epoch_rows)
        write_csv(args.calibration, all_cal_rows)
        write_summary(args, candidate_rows, selected)
    return {"selected_candidate": selected["candidate"], "candidate_rows": candidate_rows, "bundles": bundles}


def write_summary(args: argparse.Namespace, candidate_rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    lines = [
        "COMSOL rect/rot calibrated forward surrogate summary",
        "",
        f"Input NPZ: {args.npz}",
        f"20.54 proposal geometry: {args.proposal_geometry}",
        f"20.54 refinement metrics: {args.refinement_metrics}",
        "No COMSOL run; no new data; no checkpoint written.",
        "Surrogate inputs are geometry / rasterized-geometry-derived values only, not observed delta_bz.",
        "",
        "Candidates:",
    ]
    for row in candidate_rows:
        lines.append(
            f"- {row['candidate']}: val/test NRMSE = {row['val_nrmse']:.4f} / {row['test_nrmse']:.4f}, "
            f"val ordering={row['val_residual_ordering_accuracy']:.4f}, "
            f"val mismatch={row['val_mismatch_rate']:.4f}, "
            f"val residual-error corr={row['val_residual_error_correlation']:.4f}, "
            f"score={row['surrogate_selection_score']:.4f}, usable={row['usable']}"
        )
    lines.extend(
        [
            "",
            f"Selected surrogate: {selected['candidate']}",
            f"Selection score: {selected['surrogate_selection_score']:.6f}",
            f"Selected val/test NRMSE: {selected['val_nrmse']:.6f} / {selected['test_nrmse']:.6f}",
            f"Selected val/test residual ordering accuracy: {selected['val_residual_ordering_accuracy']:.6f} / {selected['test_residual_ordering_accuracy']:.6f}",
            f"Selected val/test mismatch rate: {selected['val_mismatch_rate']:.6f} / {selected['test_mismatch_rate']:.6f}",
            "",
            "Stage B acceptance:",
            f"- validation NRMSE comparable with old reference around 0.463: {selected['val_nrmse'] <= 0.513}",
            f"- ordering accuracy above random on val: {selected['val_residual_ordering_accuracy'] > 0.50}",
            f"- mismatch rate lower than conservative 0.60 gate: {selected['val_mismatch_rate'] < 0.60}",
            f"- residual-error correlation nontrivial positive: {selected['val_residual_error_correlation'] > 0.05}",
            f"- usable for Stage C: {selected['usable']}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--proposal-geometry", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--refinement-metrics", type=Path, default=DEFAULT_2054_METRICS)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args(), write_outputs=True)


if __name__ == "__main__":
    main()

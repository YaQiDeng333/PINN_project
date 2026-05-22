from __future__ import annotations

import argparse
import csv
import math
import random
import sys
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

import build_comsol_rect_rot_profile_forward_dataset as dataset_builder  # noqa: E402
import extract_comsol_rect_rot_profile_basis_from_dense as profile_extract  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_NPZ = dataset_builder.DEFAULT_ORIGINAL_NPZ
DEFAULT_PERTURB_NPZ = dataset_builder.DEFAULT_PERTURB_NPZ
DEFAULT_ORIGINAL_CSV = dataset_builder.DEFAULT_ORIGINAL_CSV
DEFAULT_PERTURB_CSV = dataset_builder.DEFAULT_PERTURB_CSV
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_forward_surrogate_summary.txt"
DEFAULT_CANDIDATES = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_forward_surrogate_candidates.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_forward_surrogate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_forward_surrogate_epoch_log.csv"
DEFAULT_ORDERING = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_forward_surrogate_ordering_audit.csv"

SEED = 42
K_STATIONS = 8
CANDIDATES = [
    "PFS1_profile_mlp_waveform",
    "PFS2_profile_raster_encoder",
    "PFS3_profile_station_sequence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train profile-compatible COMSOL rect/rot forward surrogates.")
    parser.add_argument("--original-npz", type=Path, default=DEFAULT_ORIGINAL_NPZ)
    parser.add_argument("--perturb-npz", type=Path, default=DEFAULT_PERTURB_NPZ)
    parser.add_argument("--original-csv", type=Path, default=DEFAULT_ORIGINAL_CSV)
    parser.add_argument("--perturb-csv", type=Path, default=DEFAULT_PERTURB_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--ordering-audit", type=Path, default=DEFAULT_ORDERING)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def gradient_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def peak_region_weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    abs_target = torch.abs(target)
    threshold = 0.5 * abs_target.amax(dim=2, keepdim=True).clamp_min(1.0e-8)
    weights = 1.0 + (abs_target >= threshold).float()
    return ((pred - target).square() * weights).mean()


def waveform_loss(pred: torch.Tensor, target: torch.Tensor, peak: bool = False) -> torch.Tensor:
    loss = F.mse_loss(pred, target) + 0.2 * F.l1_loss(pred, target) + 0.1 * gradient_mse(pred, target)
    if peak:
        loss = loss + 0.1 * peak_region_weighted_mse(pred, target)
    return loss


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
    per_line_corr = [
        safe_corr(pred[:, line, :].reshape(-1), target[:, line, :].reshape(-1)) for line in range(pred.shape[1])
    ]
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "correlation": corr,
        "gradient_mse": gradient,
        "amplitude_error": amplitude_error,
        "peak_index_error": peak_index_error,
        "peak_amplitude_error": amplitude_error,
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


def profile_feature_fields() -> list[str]:
    return dataset_builder.profile_feature_fields()


def row_profile_arrays(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray([to_float(row[f"u_station_{i}"]) for i in range(K_STATIONS)], dtype=np.float64)
    half = np.asarray([to_float(row[f"half_width_{i}"]) for i in range(K_STATIONS)], dtype=np.float64)
    off = np.asarray([to_float(row[f"center_offset_{i}"]) for i in range(K_STATIONS)], dtype=np.float64)
    return u, half, off


def raster_feature(row: dict[str, str], mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    u, half, off = row_profile_arrays(row)
    prob = profile_extract.rasterize_profile_np(
        mask_x,
        mask_y,
        to_float(row["center_x"]),
        to_float(row["center_y"]),
        to_float(row["angle_rad"]),
        u,
        half,
        off,
    )
    # 64x128 -> 16x32 block average.
    return prob.reshape(16, 4, 32, 4).mean(axis=(1, 3)).astype(np.float32)


def station_feature(row: dict[str, str]) -> np.ndarray:
    stations = []
    for i in range(K_STATIONS):
        stations.append(
            [
                to_float(row[f"u_station_{i}"]),
                to_float(row[f"half_width_{i}"]),
                to_float(row[f"center_offset_{i}"]),
                to_float(row[f"occupancy_{i}"]),
            ]
        )
    return np.asarray(stations, dtype=np.float32).T


@dataclass
class ProfileArrays:
    rows: list[dict[str, str]]
    scalar: np.ndarray
    raster: np.ndarray
    station: np.ndarray
    global_scalar: np.ndarray
    target_norm: np.ndarray
    target_raw: np.ndarray
    observed_norm: np.ndarray
    observed_raw: np.ndarray
    split: np.ndarray
    dataset: np.ndarray
    sample_ids: np.ndarray
    base_sample_ids: np.ndarray
    defect_types: np.ndarray
    variant_types: np.ndarray
    mask_iou: np.ndarray
    mask_dice: np.ndarray
    area_error: np.ndarray
    target_mean: float
    target_std: float
    scalar_mean: np.ndarray
    scalar_std: np.ndarray
    station_mean: np.ndarray
    station_std: np.ndarray
    global_mean: np.ndarray
    global_std: np.ndarray
    split_indices: dict[str, np.ndarray]


def load_arrays(args: argparse.Namespace) -> ProfileArrays:
    original_rows = read_csv(args.original_csv)
    perturb_rows = read_csv(args.perturb_csv) if args.perturb_csv.exists() else []
    rows = original_rows + perturb_rows
    if not rows:
        raise RuntimeError("No profile-forward rows found")
    original_npz = np.load(args.original_npz, allow_pickle=True)
    perturb_npz = np.load(args.perturb_npz, allow_pickle=True) if perturb_rows else None
    mask_x = original_npz["mask_x"].astype(np.float64)
    mask_y = original_npz["mask_y"].astype(np.float64)
    fields = profile_feature_fields()
    scalar = np.asarray([[to_float(row[field]) for field in fields] for row in rows], dtype=np.float32)
    raster = np.stack([raster_feature(row, mask_x, mask_y) for row in rows]).astype(np.float32)
    station = np.stack([station_feature(row) for row in rows]).astype(np.float32)
    global_fields = ["center_x", "center_y", "angle_sin", "angle_cos", "length", "depth_proxy", "profile_area_proxy"]
    global_scalar = np.asarray([[to_float(row[field]) for field in global_fields] for row in rows], dtype=np.float32)
    target_raw = []
    observed_raw = []
    for row in rows:
        idx = int(to_float(row["target_index"]))
        if row["dataset"] == "original":
            delta = original_npz["delta_bz"][idx].astype(np.float32)
            obs = delta
        elif row["dataset"] == "perturb":
            if perturb_npz is None:
                raise RuntimeError("Perturb rows present but perturb NPZ unavailable")
            delta = perturb_npz["delta_bz"][idx].astype(np.float32)
            obs = perturb_npz["reference_observed_delta_bz"][idx].astype(np.float32)
        else:
            raise ValueError(f"Unknown dataset: {row['dataset']}")
        target_raw.append(delta)
        observed_raw.append(obs)
    target_raw_arr = np.stack(target_raw).astype(np.float32)
    observed_raw_arr = np.stack(observed_raw).astype(np.float32)
    split = np.asarray([row["split"] for row in rows])
    train_idx = np.where(split == "train")[0]
    if train_idx.size == 0:
        raise RuntimeError("No train rows")
    target_mean = float(target_raw_arr[train_idx].mean())
    target_std = float(target_raw_arr[train_idx].std())
    if target_std <= 1.0e-12:
        target_std = 1.0
    target_norm = ((target_raw_arr - target_mean) / target_std).astype(np.float32)
    observed_norm = ((observed_raw_arr - target_mean) / target_std).astype(np.float32)
    scalar_mean = scalar[train_idx].mean(axis=0)
    scalar_std = scalar[train_idx].std(axis=0)
    scalar_std = np.where(scalar_std <= 1.0e-12, 1.0, scalar_std).astype(np.float32)
    station_mean = station[train_idx].mean(axis=(0, 2), keepdims=True)
    station_std = station[train_idx].std(axis=(0, 2), keepdims=True)
    station_std = np.where(station_std <= 1.0e-12, 1.0, station_std).astype(np.float32)
    global_mean = global_scalar[train_idx].mean(axis=0)
    global_std = global_scalar[train_idx].std(axis=0)
    global_std = np.where(global_std <= 1.0e-12, 1.0, global_std).astype(np.float32)
    scalar = ((scalar - scalar_mean) / scalar_std).astype(np.float32)
    station = ((station - station_mean) / station_std).astype(np.float32)
    global_scalar = ((global_scalar - global_mean) / global_std).astype(np.float32)
    return ProfileArrays(
        rows=rows,
        scalar=scalar,
        raster=raster[:, None, :, :],
        station=station,
        global_scalar=global_scalar,
        target_norm=target_norm,
        target_raw=target_raw_arr,
        observed_norm=observed_norm,
        observed_raw=observed_raw_arr,
        split=split,
        dataset=np.asarray([row["dataset"] for row in rows]),
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        base_sample_ids=np.asarray([row["base_sample_id"] for row in rows]),
        defect_types=np.asarray([row["defect_type"] for row in rows]),
        variant_types=np.asarray([row["variant_type"] for row in rows]),
        mask_iou=np.asarray([to_float(row["true_mask_iou"]) for row in rows], dtype=np.float32),
        mask_dice=np.asarray([to_float(row["true_mask_dice"]) for row in rows], dtype=np.float32),
        area_error=np.asarray([to_float(row["true_area_error"]) for row in rows], dtype=np.float32),
        target_mean=target_mean,
        target_std=target_std,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
        station_mean=station_mean,
        station_std=station_std,
        global_mean=global_mean,
        global_std=global_std,
        split_indices={name: np.where(split == name)[0].astype(np.int64) for name in ["train", "val", "test"]},
    )


class ForwardDataset(Dataset):
    def __init__(self, indices: np.ndarray, arrays: ProfileArrays, candidate: str):
        self.indices = indices.astype(np.int64)
        self.arrays = arrays
        self.candidate = candidate

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        item = {
            "target": torch.from_numpy(self.arrays.target_norm[i]).float(),
            "scalar": torch.from_numpy(self.arrays.scalar[i]).float(),
            "raster": torch.from_numpy(self.arrays.raster[i]).float(),
            "station": torch.from_numpy(self.arrays.station[i]).float(),
            "global_scalar": torch.from_numpy(self.arrays.global_scalar[i]).float(),
        }
        return item


class ProfileMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, 256),
            nn.GELU(),
            nn.Linear(256, 603),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["scalar"]).view(-1, 3, 201)


class ProfileRasterEncoder(nn.Module):
    def __init__(self, scalar_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
        )
        self.scalar = nn.Sequential(nn.Linear(scalar_dim, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(24 * 4 * 8 + 64, 256), nn.GELU(), nn.Linear(256, 603))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([self.conv(batch["raster"]), self.scalar(batch["scalar"])], dim=1)
        return self.head(x).view(-1, 3, 201)


class ProfileStationSequence(nn.Module):
    def __init__(self, global_dim: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 48, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.global_net = nn.Sequential(nn.Linear(global_dim, 48), nn.GELU())
        self.head = nn.Sequential(nn.Linear(96, 192), nn.GELU(), nn.Linear(192, 603))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([self.seq(batch["station"]), self.global_net(batch["global_scalar"])], dim=1)
        return self.head(x).view(-1, 3, 201)


def make_model(candidate: str, arrays: ProfileArrays) -> nn.Module:
    if candidate == "PFS1_profile_mlp_waveform":
        return ProfileMLP(arrays.scalar.shape[1])
    if candidate == "PFS2_profile_raster_encoder":
        return ProfileRasterEncoder(arrays.scalar.shape[1])
    if candidate == "PFS3_profile_station_sequence":
        return ProfileStationSequence(arrays.global_scalar.shape[1])
    raise ValueError(candidate)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def predict(model: nn.Module, ds: Dataset, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(move_batch(batch, device)).cpu().numpy()
            outs.append(pred)
    return np.concatenate(outs, axis=0)


def split_waveform_metrics(
    model: nn.Module,
    arrays: ProfileArrays,
    candidate: str,
    split: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    idx = arrays.split_indices[split]
    ds = ForwardDataset(idx, arrays, candidate)
    pred_norm = predict(model, ds, device, batch_size)
    pred_raw = pred_norm * arrays.target_std + arrays.target_mean
    return waveform_stats(pred_raw, arrays.target_raw[idx])


def ordering_stats(
    model: nn.Module,
    arrays: ProfileArrays,
    candidate: str,
    split: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    split_idx = arrays.split_indices[split]
    idx = split_idx[arrays.dataset[split_idx] == "perturb"]
    if idx.size == 0:
        return {
            "ordering_accuracy": math.nan,
            "mismatch_rate": math.nan,
            "residual_error_correlation": math.nan,
            "surrogate_vs_oracle_residual_correlation": math.nan,
            "oracle_ordering_accuracy": math.nan,
            "oracle_residual_error_correlation": math.nan,
            "n_pairs": 0,
        }
    ds = ForwardDataset(idx, arrays, candidate)
    pred_norm = predict(model, ds, device, batch_size)
    pred_raw = pred_norm * arrays.target_std + arrays.target_mean
    target_raw = arrays.target_raw[idx]
    observed = arrays.observed_raw[idx]
    surrogate_residual = np.asarray([residual_nrmse(pred_raw[i], observed[i]) for i in range(idx.size)], dtype=np.float64)
    oracle_residual = np.asarray([residual_nrmse(target_raw[i], observed[i]) for i in range(idx.size)], dtype=np.float64)
    quality = arrays.mask_iou[idx].astype(np.float64)
    base_ids = arrays.base_sample_ids[idx]
    correct = 0
    oracle_correct = 0
    total = 0
    for base in sorted(set(base_ids)):
        local = np.where(base_ids == base)[0]
        for a_pos in range(local.size):
            for b_pos in range(a_pos + 1, local.size):
                a = local[a_pos]
                b = local[b_pos]
                if abs(quality[a] - quality[b]) <= 1.0e-12:
                    continue
                better, worse = (a, b) if quality[a] > quality[b] else (b, a)
                correct += int(surrogate_residual[better] < surrogate_residual[worse])
                oracle_correct += int(oracle_residual[better] < oracle_residual[worse])
                total += 1
    ordering = correct / total if total else math.nan
    oracle_ordering = oracle_correct / total if total else math.nan
    error = 1.0 - quality
    corr = safe_corr(surrogate_residual, error)
    oracle_corr = safe_corr(oracle_residual, error)
    vs_oracle = safe_corr(surrogate_residual, oracle_residual)
    return {
        "ordering_accuracy": ordering,
        "mismatch_rate": 1.0 - ordering if math.isfinite(ordering) else math.nan,
        "residual_error_correlation": corr,
        "surrogate_vs_oracle_residual_correlation": vs_oracle,
        "oracle_ordering_accuracy": oracle_ordering,
        "oracle_residual_error_correlation": oracle_corr,
        "n_pairs": total,
    }


def train_candidate(
    candidate: str,
    arrays: ProfileArrays,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    model = make_model(candidate, arrays).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    train_ds = ForwardDataset(arrays.split_indices["train"], arrays, candidate)
    val_ds = ForwardDataset(arrays.split_indices["val"], arrays, candidate)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = waveform_loss(pred, batch["target"], peak=(candidate == "PFS2_profile_raster_encoder"))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict(model, val_ds, device, args.batch_size)
        val_stats = waveform_stats(val_pred * arrays.target_std + arrays.target_mean, arrays.target_raw[arrays.split_indices["val"]])
        val_loss = val_stats["nrmse"]
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            ord_val = ordering_stats(model, arrays, candidate, "val", device, args.batch_size)
            epoch_rows.append(
                {
                    "candidate": candidate,
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_nrmse": val_stats["nrmse"],
                    "val_correlation": val_stats["correlation"],
                    "val_ordering_accuracy": ord_val["ordering_accuracy"],
                    "val_mismatch_rate": ord_val["mismatch_rate"],
                }
            )
    if best_state is None:
        raise RuntimeError(f"No best state for {candidate}")
    model.load_state_dict(best_state)
    return model, epoch_rows, {"best_val_nrmse": best_val}


def selection_score(wave: dict[str, float], order: dict[str, float]) -> float:
    nrmse = wave["nrmse"]
    waveform_quality = max(-1.0, min(1.0, 1.0 - nrmse))
    ordering = 0.0 if not math.isfinite(order["ordering_accuracy"]) else order["ordering_accuracy"]
    corr = 0.0 if not math.isfinite(order["residual_error_correlation"]) else order["residual_error_correlation"]
    mismatch = 1.0 if not math.isfinite(order["mismatch_rate"]) else order["mismatch_rate"]
    peak_norm = min(1.0, wave["peak_index_error"] / 40.0)
    return 0.30 * waveform_quality + 0.35 * ordering + 0.20 * corr - 0.25 * mismatch - 0.10 * peak_norm


def gate_pass(wave: dict[str, float], order: dict[str, float]) -> bool:
    return (
        math.isfinite(wave["nrmse"])
        and wave["nrmse"] <= 0.75
        and order["ordering_accuracy"] >= 0.70
        and order["mismatch_rate"] <= 0.30
        and order["residual_error_correlation"] >= 0.05
        and order["surrogate_vs_oracle_residual_correlation"] >= 0.20
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    arrays = load_arrays(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    candidate_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    trained: dict[str, nn.Module] = {}
    for candidate in CANDIDATES:
        model, epochs, _info = train_candidate(candidate, arrays, args, device)
        trained[candidate] = model
        epoch_rows.extend(epochs)
        train_wave = split_waveform_metrics(model, arrays, candidate, "train", device, args.batch_size)
        val_wave = split_waveform_metrics(model, arrays, candidate, "val", device, args.batch_size)
        val_order = ordering_stats(model, arrays, candidate, "val", device, args.batch_size)
        score = selection_score(val_wave, val_order)
        passed = gate_pass(val_wave, val_order)
        candidate_rows.append(
            {
                "candidate": candidate,
                "selected": False,
                "gate_pass": passed,
                "val_score": score,
                "val_nrmse": val_wave["nrmse"],
                "val_correlation": val_wave["correlation"],
                "val_peak_index_error": val_wave["peak_index_error"],
                "val_ordering_accuracy": val_order["ordering_accuracy"],
                "val_mismatch_rate": val_order["mismatch_rate"],
                "val_residual_error_correlation": val_order["residual_error_correlation"],
                "val_surrogate_vs_oracle_residual_correlation": val_order["surrogate_vs_oracle_residual_correlation"],
                "notes": "profile-compatible direct input; validation-only candidate selection",
            }
        )
        for split, wave in [("train", train_wave), ("val", val_wave)]:
            metrics_rows.append({"candidate": candidate, "split": split, "selected": False, **wave})
        ordering_rows.append({"candidate": candidate, "split": "val", "selected": False, **val_order})
    selected_row = max(candidate_rows, key=lambda row: float(row["val_score"]))
    selected = str(selected_row["candidate"])
    for row in candidate_rows:
        row["selected"] = row["candidate"] == selected
    selected_model = trained[selected]
    for split in ["test"]:
        wave = split_waveform_metrics(selected_model, arrays, selected, split, device, args.batch_size)
        metrics_rows.append({"candidate": selected, "split": split, "selected": True, **wave})
        ordering_rows.append({"candidate": selected, "split": split, "selected": True, **ordering_stats(selected_model, arrays, selected, split, device, args.batch_size)})
    for row in metrics_rows:
        if row["candidate"] == selected:
            row["selected"] = True
    for row in ordering_rows:
        if row["candidate"] == selected:
            row["selected"] = True
    write_csv(args.candidates, candidate_rows)
    write_csv(args.metrics, metrics_rows)
    write_csv(args.epoch_log, epoch_rows)
    write_csv(args.ordering_audit, ordering_rows)
    selected_val = next(row for row in candidate_rows if row["candidate"] == selected)
    selected_test_order = next(row for row in ordering_rows if row["candidate"] == selected and row["split"] == "test")
    selected_test_wave = next(row for row in metrics_rows if row["candidate"] == selected and row["split"] == "test")
    usable = bool(selected_val["gate_pass"]) and selected_test_order["ordering_accuracy"] >= 0.65 and selected_test_order["mismatch_rate"] <= 0.35
    lines = [
        "COMSOL rect/rot profile-compatible forward surrogate summary",
        "",
        "No COMSOL run, no inverse geometry head, and no checkpoint written.",
        "Surrogate inputs are profile/basis parameters or rasterized-profile-derived features, not a compressed single rotated rectangle.",
        "defect_type / variant_type are used only for metrics, not as model inputs.",
        "",
        f"Rows by split: train={arrays.split_indices['train'].size}, val={arrays.split_indices['val'].size}, test={arrays.split_indices['test'].size}",
        f"Selected candidate by validation score: {selected}",
        f"Selected val NRMSE/corr/score: {selected_val['val_nrmse']:.6f} / {selected_val['val_correlation']:.6f} / {selected_val['val_score']:.6f}",
        f"Selected val ordering/mismatch/residual_corr/oracle_corr: {selected_val['val_ordering_accuracy']:.6f} / {selected_val['val_mismatch_rate']:.6f} / {selected_val['val_residual_error_correlation']:.6f} / {selected_val['val_surrogate_vs_oracle_residual_correlation']:.6f}",
        f"Selected test NRMSE/corr: {selected_test_wave['nrmse']:.6f} / {selected_test_wave['correlation']:.6f}",
        f"Selected test ordering/mismatch/residual_corr/oracle_corr: {selected_test_order['ordering_accuracy']:.6f} / {selected_test_order['mismatch_rate']:.6f} / {selected_test_order['residual_error_correlation']:.6f} / {selected_test_order['surrogate_vs_oracle_residual_correlation']:.6f}",
        f"Stage B gate passed: {usable}",
        "",
        "Gate rule: refinement retry is allowed only if Stage B gate passes. If false, Stage C must be skipped.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "selected": selected,
        "usable": usable,
        "candidate_rows": candidate_rows,
        "metrics_rows": metrics_rows,
        "ordering_rows": ordering_rows,
    }


def main() -> None:
    result = run(parse_args())
    print(f"Selected profile-compatible surrogate: {result['selected']} usable={result['usable']}")


if __name__ == "__main__":
    main()

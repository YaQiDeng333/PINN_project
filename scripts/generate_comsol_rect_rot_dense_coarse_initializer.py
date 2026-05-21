from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from collections import defaultdict, deque
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

import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = base.DEFAULT_NPZ
DEFAULT_LABELS = base.DEFAULT_LABELS
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_dense_coarse_initializer_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_coarse_initializer_metrics.csv"
DEFAULT_GEOMETRY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_coarse_initializer_geometry.csv"

SEED = 42
EPOCHS = 200
BATCH_SIZE = 8
THRESHOLDS = base.THRESHOLDS
MAX_ANGLE_DEG = 35.0

METRIC_FIELDS = [
    "sample_id",
    "source_index",
    "split",
    "defect_type",
    "source_pack",
    "threshold",
    "dense_threshold",
    "geometry_threshold",
    "dense_iou",
    "dense_dice",
    "dense_area_error",
    "dense_center_error_px",
    "dense_pred_area",
    "true_area",
    "geometry_iou",
    "geometry_dice",
    "geometry_area_error",
    "geometry_center_error_px",
    "geometry_pred_area",
    "empty_prediction",
    "fallback_used",
    "component_area_px",
    "notes",
]

GEOMETRY_FIELDS = [
    "sample_id",
    "source_index",
    "split",
    "defect_type",
    "source_pack",
    "threshold",
    "dense_threshold",
    "geometry_threshold",
    "pred_center_x",
    "pred_center_y",
    "pred_width",
    "pred_length",
    "pred_depth",
    "pred_angle_deg",
    "pred_angle_rad",
    "type_prob_rectangular_notch",
    "type_prob_rotated_rect",
    "pred_defect_type",
    "dense_iou",
    "dense_dice",
    "dense_area_error",
    "geometry_iou",
    "geometry_dice",
    "geometry_area_error",
    "center_abs_error_m",
    "width_abs_error_m",
    "length_abs_error_m",
    "depth_abs_error_m",
    "angle_abs_error_deg",
    "empty_prediction",
    "fallback_used",
    "component_area_px",
    "depth_source",
    "type_probability_source",
    "notes",
]

EPOCH_FIELDS = [
    "epoch",
    "train_loss",
    "train_bce_loss",
    "train_dice_loss",
    "best_val_threshold",
    "val_iou",
    "val_dice",
    "val_area_error",
    "val_score",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    return base.safe_mean(rows, key)


class DenseMaskDataset(Dataset):
    def __init__(self, indices: np.ndarray, arrays: dict[str, Any]):
        self.indices = indices.astype(np.int64)
        self.signals = arrays["signals_norm"][self.indices]
        self.masks = arrays["masks"][self.indices]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        local_idx = int(self.indices[idx])
        return {
            "source_index": torch.tensor(local_idx, dtype=torch.long),
            "signal": torch.from_numpy(self.signals[idx]).float(),
            "mask": torch.from_numpy(self.masks[idx]).float(),
        }


class DenseMaskInitializer(nn.Module):
    def __init__(self, latent_dim: int = 192):
        super().__init__()
        self.encoder = base.BzEncoder(latent_dim=latent_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128 * 8 * 16),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(signal)
        grid = self.fc(latent).view(signal.shape[0], 128, 8, 16)
        return self.decoder(grid).squeeze(1)


def batch_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = base.soft_dice_loss(prob, target)
    return bce + dice, {"bce": float(bce.detach().cpu()), "dice": float(dice.detach().cpu())}


def predict(model: nn.Module, dataset: DenseMaskDataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["signal"].to(device))
            chunks["indices"].append(batch["source_index"].cpu().numpy())
            chunks["prob"].append(torch.sigmoid(logits).cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def rows_for_threshold(pred: dict[str, np.ndarray], arrays: dict[str, Any], split: str, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for order, local_idx_raw in enumerate(pred["indices"]):
        local_idx = int(local_idx_raw)
        metric = base.mask_metric(pred["prob"][order], arrays["masks"][local_idx], threshold)
        rows.append(
            {
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": split,
                "defect_type": str(arrays["defect_types"][local_idx]),
                "source_pack": str(arrays["source_packs"][local_idx]),
                "threshold": threshold,
                **metric,
            }
        )
    return rows


def val_score(pred: dict[str, np.ndarray], arrays: dict[str, Any]) -> dict[str, float]:
    best = {"score": -math.inf, "threshold": math.nan}
    for threshold in THRESHOLDS:
        rows = rows_for_threshold(pred, arrays, "val", threshold)
        iou = safe_mean(rows, "iou")
        dice = safe_mean(rows, "dice")
        area_error = safe_mean(rows, "area_error")
        score = iou + dice - area_error
        if score > best["score"]:
            best = {
                "score": score,
                "threshold": threshold,
                "iou": iou,
                "dice": dice,
                "area_error": area_error,
            }
    return best


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask, True
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) > len(best):
                best = component
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out, False


def top_region(prob: np.ndarray) -> np.ndarray:
    flat = prob.reshape(-1)
    count = max(16, int(round(flat.size * 0.04)))
    cutoff = np.partition(flat, -count)[-count]
    return prob >= cutoff


def normalize_angle_deg(angle_deg: float) -> float:
    angle = (angle_deg + 90.0) % 180.0 - 90.0
    if angle > 45.0:
        angle -= 90.0
    if angle < -45.0:
        angle += 90.0
    return float(np.clip(angle, -MAX_ANGLE_DEG, MAX_ANGLE_DEG))


def extract_geometry_from_prob(
    prob: np.ndarray,
    threshold: float,
    arrays: dict[str, Any],
    train_median_depth: float,
) -> dict[str, Any]:
    raw_mask = prob >= threshold
    empty = not bool(raw_mask.any())
    fallback = False
    if empty:
        raw_mask = top_region(prob)
        fallback = True
    component, component_fallback = largest_component(raw_mask)
    fallback = fallback or component_fallback
    if not component.any():
        raw_mask = top_region(prob)
        component, _ = largest_component(raw_mask)
        fallback = True
    ys, xs = np.argwhere(component).T
    mask_x = arrays["mask_x"]
    mask_y = arrays["mask_y"]
    x_coords = mask_x[xs]
    y_coords = mask_y[ys]
    coords = np.stack([x_coords, y_coords], axis=1)
    center = coords.mean(axis=0)
    if coords.shape[0] >= 3:
        centered = coords - center
        cov = centered.T @ centered / max(coords.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]
        angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
        angle = normalize_angle_deg(angle)
        theta = math.radians(angle)
        axis_major = np.array([math.cos(theta), math.sin(theta)])
        axis_minor = np.array([-math.sin(theta), math.cos(theta)])
        proj_major = centered @ axis_major
        proj_minor = centered @ axis_minor
        width = float(max(proj_major.max() - proj_major.min(), 1e-4))
        length = float(max(proj_minor.max() - proj_minor.min(), 1e-4))
    else:
        width = float(max(mask_x[min(xs.max() + 1, mask_x.size - 1)] - mask_x[max(xs.min() - 1, 0)], 1e-4))
        length = float(max(mask_y[min(ys.max() + 1, mask_y.size - 1)] - mask_y[max(ys.min() - 1, 0)], 1e-4))
        angle = 0.0
        fallback = True
    if length > width:
        width, length = length, width
        angle = normalize_angle_deg(angle + 90.0)
    width = float(np.clip(width, 0.001, 0.025))
    length = float(np.clip(length, 0.001, 0.020))
    p_rot = 1.0 / (1.0 + math.exp(-(abs(angle) - 5.0) / 3.0))
    p_rot = float(np.clip(p_rot, 0.05, 0.95))
    return {
        "center_x": float(np.clip(center[0], mask_x.min(), mask_x.max())),
        "center_y": float(np.clip(center[1], mask_y.min(), mask_y.max())),
        "width": width,
        "length": length,
        "depth": float(train_median_depth),
        "angle_deg": angle,
        "angle_rad": math.radians(angle),
        "type_prob_rectangular_notch": 1.0 - p_rot,
        "type_prob_rotated_rect": p_rot,
        "pred_defect_type": "rotated_rect" if p_rot >= 0.5 else "rectangular_notch",
        "empty_prediction": float(empty),
        "fallback_used": float(fallback),
        "component_area_px": float(component.sum()),
    }


def geometry_mask(geom: dict[str, Any], arrays: dict[str, Any]) -> np.ndarray:
    mask_x_t = torch.tensor(arrays["mask_x"], dtype=torch.float32)
    mask_y_t = torch.tensor(arrays["mask_y"], dtype=torch.float32)
    with torch.no_grad():
        rect = base.soft_rect_mask(
            mask_x_t,
            mask_y_t,
            torch.tensor([geom["center_x"]], dtype=torch.float32),
            torch.tensor([geom["center_y"]], dtype=torch.float32),
            torch.tensor([geom["width"]], dtype=torch.float32),
            torch.tensor([geom["length"]], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        )[0].numpy()
        rot = base.soft_rect_mask(
            mask_x_t,
            mask_y_t,
            torch.tensor([geom["center_x"]], dtype=torch.float32),
            torch.tensor([geom["center_y"]], dtype=torch.float32),
            torch.tensor([geom["width"]], dtype=torch.float32),
            torch.tensor([geom["length"]], dtype=torch.float32),
            torch.tensor([geom["angle_rad"]], dtype=torch.float32),
        )[0].numpy()
    return geom["type_prob_rectangular_notch"] * rect + geom["type_prob_rotated_rect"] * rot


def split_summary(rows: list[dict[str, Any]], split: str, prefix: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split]
    return {
        "iou": safe_mean(subset, f"{prefix}_iou"),
        "dice": safe_mean(subset, f"{prefix}_dice"),
        "area_error": safe_mean(subset, f"{prefix}_area_error"),
        "center_error_px": safe_mean(subset, f"{prefix}_center_error_px"),
        "pred_area": safe_mean(subset, f"{prefix}_pred_area"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    arrays, diagnostics = base.load_arrays(args.npz, args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = DenseMaskDataset(arrays["split_indices"]["train"], arrays)
    val_ds = DenseMaskDataset(arrays["split_indices"]["val"], arrays)
    test_ds = DenseMaskDataset(arrays["split_indices"]["test"], arrays)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = DenseMaskInitializer(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val = {"score": -math.inf, "threshold": 0.5}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        parts = defaultdict(float)
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["signal"].to(device))
            loss, loss_parts = batch_loss(logits, batch["mask"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            for key, value in loss_parts.items():
                parts[key] += value
            batches += 1
        val_pred = predict(model, val_ds, device, args.batch_size)
        current = val_score(val_pred, arrays)
        if current["score"] > best_val["score"]:
            best_val = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": total / max(batches, 1),
                "train_bce_loss": parts["bce"] / max(batches, 1),
                "train_dice_loss": parts["dice"] / max(batches, 1),
                "best_val_threshold": current["threshold"],
                "val_iou": current["iou"],
                "val_dice": current["dice"],
                "val_area_error": current["area_error"],
                "val_score": current["score"],
            }
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"dense init epoch={epoch:03d} loss={epoch_rows[-1]['train_loss']:.4f} "
                f"val_score={current['score']:.4f} thr={current['threshold']:.2f}"
            )

    if best_state is None:
        raise RuntimeError("No dense initializer validation checkpoint selected")
    model.load_state_dict(best_state)
    dense_threshold = float(best_val["threshold"])
    geometry_threshold = 0.50
    train_median_depth = float(np.median(arrays["raw_geom"][arrays["split_indices"]["train"], 4]))

    metric_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for split, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        pred = predict(model, dataset, device, args.batch_size)
        for order, local_idx_raw in enumerate(pred["indices"]):
            local_idx = int(local_idx_raw)
            prob = pred["prob"][order]
            dense_metric = base.mask_metric(prob, arrays["masks"][local_idx], dense_threshold)
            geom = extract_geometry_from_prob(prob, dense_threshold, arrays, train_median_depth)
            geom_prob = geometry_mask(geom, arrays)
            geom_metric = base.mask_metric(geom_prob, arrays["masks"][local_idx], geometry_threshold)
            true_geom = arrays["raw_geom"][local_idx]
            true_angle = math.atan2(float(arrays["angle_targets"][local_idx, 0]), float(arrays["angle_targets"][local_idx, 1]))
            angle_error = (
                base.circular_angle_error_deg(geom["angle_deg"], math.degrees(true_angle))
                if str(arrays["defect_types"][local_idx]) == "rotated_rect"
                else math.nan
            )
            shared = {
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": split,
                "defect_type": str(arrays["defect_types"][local_idx]),
                "source_pack": str(arrays["source_packs"][local_idx]),
                "threshold": geometry_threshold,
                "dense_threshold": dense_threshold,
                "geometry_threshold": geometry_threshold,
            }
            metric_rows.append(
                {
                    **shared,
                    "dense_iou": dense_metric["iou"],
                    "dense_dice": dense_metric["dice"],
                    "dense_area_error": dense_metric["area_error"],
                    "dense_center_error_px": dense_metric["center_error_px"],
                    "dense_pred_area": dense_metric["pred_area"],
                    "true_area": dense_metric["true_area"],
                    "geometry_iou": geom_metric["iou"],
                    "geometry_dice": geom_metric["dice"],
                    "geometry_area_error": geom_metric["area_error"],
                    "geometry_center_error_px": geom_metric["center_error_px"],
                    "geometry_pred_area": geom_metric["pred_area"],
                    "empty_prediction": geom["empty_prediction"],
                    "fallback_used": geom["fallback_used"],
                    "component_area_px": geom["component_area_px"],
                    "notes": "",
                }
            )
            geometry_rows.append(
                {
                    **shared,
                    "pred_center_x": geom["center_x"],
                    "pred_center_y": geom["center_y"],
                    "pred_width": geom["width"],
                    "pred_length": geom["length"],
                    "pred_depth": geom["depth"],
                    "pred_angle_deg": geom["angle_deg"],
                    "pred_angle_rad": geom["angle_rad"],
                    "type_prob_rectangular_notch": geom["type_prob_rectangular_notch"],
                    "type_prob_rotated_rect": geom["type_prob_rotated_rect"],
                    "pred_defect_type": geom["pred_defect_type"],
                    "dense_iou": dense_metric["iou"],
                    "dense_dice": dense_metric["dice"],
                    "dense_area_error": dense_metric["area_error"],
                    "geometry_iou": geom_metric["iou"],
                    "geometry_dice": geom_metric["dice"],
                    "geometry_area_error": geom_metric["area_error"],
                    "center_abs_error_m": float(math.hypot(geom["center_x"] - true_geom[0], geom["center_y"] - true_geom[1])),
                    "width_abs_error_m": float(abs(geom["width"] - true_geom[2])),
                    "length_abs_error_m": float(abs(geom["length"] - true_geom[3])),
                    "depth_abs_error_m": float(abs(geom["depth"] - true_geom[4])),
                    "angle_abs_error_deg": angle_error,
                    "empty_prediction": geom["empty_prediction"],
                    "fallback_used": geom["fallback_used"],
                    "component_area_px": geom["component_area_px"],
                    "depth_source": "train_median_depth",
                    "type_probability_source": "angle_magnitude_sigmoid",
                    "notes": "",
                }
            )

    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.geometry, geometry_rows, GEOMETRY_FIELDS)
    if args.epoch_log:
        write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
    summary = write_summary(args, diagnostics, metric_rows, geometry_rows, best_epoch, best_val, dense_threshold, geometry_threshold, device)
    return summary


def write_summary(
    args: argparse.Namespace,
    diagnostics: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    best_epoch: int,
    best_val: dict[str, float],
    dense_threshold: float,
    geometry_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    dense = {split: split_summary(metric_rows, split, "dense") for split in ["train", "val", "test"]}
    geom = {split: split_summary(metric_rows, split, "geometry") for split in ["train", "val", "test"]}
    test_geom_rows = [row for row in geometry_rows if row["split"] == "test"]
    test_type_accuracy = safe_mean(
        [
            {
                "type_correct": float(row["pred_defect_type"] == row["defect_type"]),
            }
            for row in test_geom_rows
        ],
        "type_correct",
    )
    test_angle = safe_mean([row for row in test_geom_rows if row["defect_type"] == "rotated_rect"], "angle_abs_error_deg")
    fallback_count = int(sum(base.to_float(row["fallback_used"], 0.0) for row in metric_rows))
    empty_count = int(sum(base.to_float(row["empty_prediction"], 0.0) for row in metric_rows))
    proposal_usable = geom["test"]["iou"] >= 0.30
    lines = [
        "COMSOL rect/rot dense/coarse mask initializer summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Geometry labels: {args.labels}",
        "Scope: rectangular_notch + rotated_rect only; polygon parsed/reported elsewhere and excluded.",
        "Dense initializer input: delta_bz only. No defect_type, geometry_params, source_pack, true mask, true angle, or true size are model inputs.",
        "True masks are used only as dense initializer supervision and final metrics.",
        "No checkpoint is written or submitted; this is only a coarse initializer, not a baseline.",
        "",
        f"Device: {device}",
        f"Seed: {args.seed}",
        f"Epochs: {args.epochs}",
        f"Best epoch: {best_epoch}",
        f"Selected dense mask threshold from validation: {dense_threshold}",
        f"Geometry raster threshold: {geometry_threshold}",
        f"Best val score IoU+Dice-area_error: {best_val['score']:.6f}",
        f"Split counts: {diagnostics['split_counts']}",
        "",
        "Dense mask initializer metrics:",
        f"- train IoU/Dice/area_error = {dense['train']['iou']:.4f} / {dense['train']['dice']:.4f} / {dense['train']['area_error']:.4f}",
        f"- val IoU/Dice/area_error = {dense['val']['iou']:.4f} / {dense['val']['dice']:.4f} / {dense['val']['area_error']:.4f}",
        f"- test IoU/Dice/area_error = {dense['test']['iou']:.4f} / {dense['test']['dice']:.4f} / {dense['test']['area_error']:.4f}",
        "",
        "Extracted geometry proposal raster metrics:",
        f"- train IoU/Dice/area_error = {geom['train']['iou']:.4f} / {geom['train']['dice']:.4f} / {geom['train']['area_error']:.4f}",
        f"- val IoU/Dice/area_error = {geom['val']['iou']:.4f} / {geom['val']['dice']:.4f} / {geom['val']['area_error']:.4f}",
        f"- test IoU/Dice/area_error = {geom['test']['iou']:.4f} / {geom['test']['dice']:.4f} / {geom['test']['area_error']:.4f}",
        f"- test extracted type accuracy = {test_type_accuracy:.4f}",
        f"- test rotated angle MAE deg = {test_angle:.4f}",
        "",
        "Geometry extraction policy:",
        "- largest connected component from dense thresholded prediction.",
        "- empty prediction fallback: top-probability region.",
        "- rotated bbox: PCA orientation and projected extents.",
        "- depth initialization: train split median depth.",
        "- type probability initialization: sigmoid of predicted angle magnitude.",
        f"- empty prediction count = {empty_count}",
        f"- fallback count = {fallback_count}",
        f"Proposal usable for refinement: {proposal_usable}",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "dense": dense,
        "geometry": geom,
        "proposal_usable": proposal_usable,
        "dense_threshold": dense_threshold,
        "geometry_threshold": geometry_threshold,
        "test_type_accuracy": test_type_accuracy,
        "test_angle_mae": test_angle,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--epoch-log", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

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
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_multiline_tiny_smoke as tiny  # noqa: E402
import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = base.DEFAULT_NPZ
DEFAULT_LABELS = base.DEFAULT_LABELS
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_strong_dense_initializer_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_strong_dense_initializer_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_strong_dense_initializer_epoch_log.csv"
DEFAULT_GROUP_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_strong_dense_initializer_group_summary.csv"

THRESHOLDS = base.THRESHOLDS

METRIC_FIELDS = [
    "sample_id",
    "source_index",
    "split",
    "defect_type",
    "source_pack",
    "threshold",
    "iou",
    "dice",
    "area_error",
    "center_error_px",
    "pred_area",
    "true_area",
    "pred_area_zero",
    "prob_min",
    "prob_max",
    "prob_mean",
]

EPOCH_FIELDS = [
    "epoch",
    "train_loss",
    "train_bce_loss",
    "train_dice_loss",
    "val_threshold",
    "val_iou",
    "val_dice",
    "val_area_error",
    "val_score",
    "best_epoch",
    "best_val_score",
]

GROUP_FIELDS = [
    "split",
    "group_name",
    "group_value",
    "sample_count",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_px_mean",
    "pred_area_mean",
    "true_area_mean",
    "pred_area_zero_sum",
]


@dataclass
class DenseInitializerBundle:
    model: nn.Module
    arrays: dict[str, Any]
    diagnostics: dict[str, Any]
    selected_threshold: float
    best_epoch: int
    best_val: dict[str, float]
    device: torch.device


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
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else math.nan


def safe_sum(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.sum(values)) if values else 0.0


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
    rows: list[dict[str, Any]] = []
    for order, local_idx_raw in enumerate(pred["indices"]):
        local_idx = int(local_idx_raw)
        prob = pred["prob"][order]
        metric = base.mask_metric(prob, arrays["masks"][local_idx], threshold)
        rows.append(
            {
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": split,
                "defect_type": str(arrays["defect_types"][local_idx]),
                "source_pack": str(arrays["source_packs"][local_idx]),
                "threshold": threshold,
                **metric,
                "prob_min": float(prob.min()),
                "prob_max": float(prob.max()),
                "prob_mean": float(prob.mean()),
            }
        )
    return rows


def threshold_score(pred: dict[str, np.ndarray], arrays: dict[str, Any], split: str) -> dict[str, float]:
    best = {"score": -math.inf, "threshold": math.nan, "iou": math.nan, "dice": math.nan, "area_error": math.nan}
    for threshold in THRESHOLDS:
        rows = rows_for_threshold(pred, arrays, split, float(threshold))
        iou = safe_mean(rows, "iou")
        dice = safe_mean(rows, "dice")
        area_error = safe_mean(rows, "area_error")
        score = iou + dice - area_error
        if score > best["score"]:
            best = {"score": score, "threshold": float(threshold), "iou": iou, "dice": dice, "area_error": area_error}
    return best


def split_stats(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split]
    return {
        "sample_count": float(len(subset)),
        "iou": safe_mean(subset, "iou"),
        "dice": safe_mean(subset, "dice"),
        "area_error": safe_mean(subset, "area_error"),
        "center_error_px": safe_mean(subset, "center_error_px"),
        "pred_area": safe_mean(subset, "pred_area"),
        "true_area": safe_mean(subset, "true_area"),
        "pred_area_zero": safe_sum(subset, "pred_area_zero"),
    }


def build_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        groups = [("overall", ["rect_rot"])]
        groups.append(("defect_type", sorted({str(row["defect_type"]) for row in split_rows})))
        groups.append(("source_pack", sorted({str(row["source_pack"]) for row in split_rows})))
        for group_name, values in groups:
            for value in values:
                subset = split_rows if group_name == "overall" else [row for row in split_rows if str(row[group_name]) == value]
                out.append(
                    {
                        "split": split,
                        "group_name": group_name,
                        "group_value": value,
                        "sample_count": len(subset),
                        "iou_mean": safe_mean(subset, "iou"),
                        "dice_mean": safe_mean(subset, "dice"),
                        "area_error_mean": safe_mean(subset, "area_error"),
                        "center_error_px_mean": safe_mean(subset, "center_error_px"),
                        "pred_area_mean": safe_mean(subset, "pred_area"),
                        "true_area_mean": safe_mean(subset, "true_area"),
                        "pred_area_zero_sum": safe_sum(subset, "pred_area_zero"),
                    }
                )
    return out


def train_dense_initializer(args: argparse.Namespace, write_outputs: bool = True) -> tuple[DenseInitializerBundle, list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(args.seed)
    arrays, diagnostics = base.load_arrays(args.npz, args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = DenseMaskDataset(arrays["split_indices"]["train"], arrays)
    val_ds = DenseMaskDataset(arrays["split_indices"]["val"], arrays)
    test_ds = DenseMaskDataset(arrays["split_indices"]["test"], arrays)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = tiny.TinyComsolMaskDecoder(arrays["signals_norm"].shape[1], arrays["masks"].shape[1], arrays["masks"].shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val = {"score": -math.inf, "threshold": 0.5, "iou": math.nan, "dice": math.nan, "area_error": math.nan}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        bce_total = 0.0
        dice_total = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["signal"].to(device))
            loss, bce, dice = tiny.loss_components(logits, batch["mask"].to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            bce_total += float(bce.detach().cpu())
            dice_total += float(dice.detach().cpu())
            batches += 1
        val_pred = predict(model, val_ds, device, args.batch_size)
        current = threshold_score(val_pred, arrays, "val")
        if current["score"] > best_val["score"]:
            best_val = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": total / max(batches, 1),
                "train_bce_loss": bce_total / max(batches, 1),
                "train_dice_loss": dice_total / max(batches, 1),
                "val_threshold": current["threshold"],
                "val_iou": current["iou"],
                "val_dice": current["dice"],
                "val_area_error": current["area_error"],
                "val_score": current["score"],
                "best_epoch": best_epoch,
                "best_val_score": best_val["score"],
            }
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"strong dense epoch={epoch:03d} loss={epoch_rows[-1]['train_loss']:.4f} "
                f"val_score={current['score']:.4f} thr={current['threshold']:.2f}"
            )

    if best_state is None:
        raise RuntimeError("No validation checkpoint selected for strong dense initializer")
    model.load_state_dict(best_state)
    selected_threshold = float(best_val["threshold"])
    metric_rows: list[dict[str, Any]] = []
    for split, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        pred = predict(model, dataset, device, args.batch_size)
        metric_rows.extend(rows_for_threshold(pred, arrays, split, selected_threshold))
    group_rows = build_group_rows(metric_rows)

    if write_outputs:
        write_csv(args.metrics, metric_rows, METRIC_FIELDS)
        write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
        write_csv(args.group_summary, group_rows, GROUP_FIELDS)
        write_summary(args, diagnostics, metric_rows, best_epoch, best_val, device)

    bundle = DenseInitializerBundle(
        model=model,
        arrays=arrays,
        diagnostics=diagnostics,
        selected_threshold=selected_threshold,
        best_epoch=best_epoch,
        best_val=best_val,
        device=device,
    )
    return bundle, metric_rows, epoch_rows


def write_summary(
    args: argparse.Namespace,
    diagnostics: dict[str, Any],
    rows: list[dict[str, Any]],
    best_epoch: int,
    best_val: dict[str, float],
    device: torch.device,
) -> None:
    stats = {split: split_stats(rows, split) for split in ["train", "val", "test"]}
    test = stats["test"]
    improves_2053 = test["iou"] > 0.5664 and test["dice"] > 0.7179
    hits_target = test["iou"] >= 0.62 or test["dice"] >= 0.76
    usable = test["iou"] >= 0.60
    lines = [
        "COMSOL rect/rot strong dense initializer summary",
        "",
        f"Input NPZ: {args.npz}",
        "Scope: rectangular_notch + rotated_rect only; polygon excluded from training/evaluation.",
        "Input policy: delta_bz only. No defect_type, geometry_params, source_pack, true angle, true size, or true mask as model input.",
        "True masks are used only as train supervision and final metrics.",
        "This initializer is only a proposal generator, not a baseline. No checkpoint is written.",
        "",
        f"Device: {device}",
        f"Seed / epochs / batch_size / lr: {args.seed} / {args.epochs} / {args.batch_size} / {args.lr}",
        f"rect+rot N and split counts: {diagnostics['n_rect_rot']} / {diagnostics['split_counts']}",
        f"Best epoch: {best_epoch}",
        f"Selected threshold from validation: {best_val['threshold']}",
        f"Best validation score IoU+Dice-area_error: {best_val['score']:.6f}",
        "",
        "Mask metrics:",
    ]
    for split in ["train", "val", "test"]:
        s = stats[split]
        lines.append(
            f"- {split} IoU/Dice/area_error/center_error_px = "
            f"{s['iou']:.4f} / {s['dice']:.4f} / {s['area_error']:.4f} / {s['center_error_px']:.4f}"
        )
    lines.extend(
        [
            "",
            "Acceptance check:",
            "- 20.53 small dense initializer reference test IoU/Dice = 0.5664 / 0.7179",
            f"- improves over 20.53: {improves_2053}",
            "- target: test IoU >= 0.62 OR Dice >= 0.76",
            f"- target hit: {hits_target}",
            f"- usable for proposal extraction gate (test IoU >= 0.60): {usable}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    train_dense_initializer(parse_args(), write_outputs=True)


if __name__ == "__main__":
    main()

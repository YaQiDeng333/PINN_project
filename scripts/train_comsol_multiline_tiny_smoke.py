from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "prepared"
    / "comsol_single_defect_multiline_forward_pack_v1_small.npz"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "summaries"
    / "comsol_multiline_tiny_training_smoke_summary.txt"
)
DEFAULT_METRICS = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_multiline_tiny_training_smoke_metrics.csv"
)
DEFAULT_EPOCH_LOG = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_multiline_tiny_training_smoke_epoch_log.csv"
)
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results" / "previews" / "comsol_multiline_tiny_smoke"

REQUIRED_FIELDS = [
    "delta_bz",
    "bz_defect",
    "bz_no_defect",
    "masks",
    "sensor_x",
    "scan_line_y",
    "mask_x",
    "mask_y",
    "defect_types",
    "sample_ids",
    "geometry_params",
    "metadata",
]

METRIC_FIELDS = [
    "sample_id",
    "split",
    "defect_type",
    "threshold",
    "iou",
    "dice",
    "area_error",
    "center_error",
    "pred_area",
    "true_area",
    "pred_area_zero",
    "bce_loss",
    "dice_loss",
    "total_loss",
    "prob_min",
    "prob_max",
    "prob_mean",
    "notes",
]

EPOCH_FIELDS = [
    "epoch",
    "train_loss",
    "train_bce",
    "train_dice_loss",
    "val_loss",
    "val_bce",
    "val_dice_loss",
    "val_iou_at_0_5",
    "val_dice_at_0_5",
    "val_area_error_at_0_5",
    "val_score_at_0_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded tiny training smoke on the COMSOL multi-line small pack."
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_json_array(array: np.ndarray) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in array.tolist():
        try:
            value = json.loads(as_text(item))
        except json.JSONDecodeError:
            value = {"raw": as_text(item)}
        parsed.append(value if isinstance(value, dict) else {"raw": value})
    return parsed


def validate_npz(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    missing = [field for field in REQUIRED_FIELDS if field not in data.files]
    if missing:
        raise RuntimeError(f"missing NPZ fields: {missing}")

    delta_bz = data["delta_bz"]
    bz_defect = data["bz_defect"]
    bz_no_defect = data["bz_no_defect"]
    masks = data["masks"]
    sensor_x = data["sensor_x"]
    scan_line_y = data["scan_line_y"]
    mask_x = data["mask_x"]
    mask_y = data["mask_y"]
    sample_ids = np.array([as_text(item) for item in data["sample_ids"].tolist()])
    suggested_split = (
        np.array([as_text(item) for item in data["suggested_split"].tolist()])
        if "suggested_split" in data.files
        else None
    )

    if delta_bz.ndim != 3:
        raise RuntimeError(f"delta_bz must be (N,n_lines,L), got {delta_bz.shape}")
    if bz_defect.shape != delta_bz.shape or bz_no_defect.shape != delta_bz.shape:
        raise RuntimeError("bz_defect / bz_no_defect shape mismatch with delta_bz")
    if masks.ndim != 3:
        raise RuntimeError(f"masks must be (N,H,W), got {masks.shape}")
    n, n_lines, signal_length = delta_bz.shape
    mask_n, mask_h, mask_w = masks.shape
    if mask_n != n:
        raise RuntimeError("masks N does not match delta_bz N")
    if sensor_x.shape != (signal_length,):
        raise RuntimeError("sensor_x length does not match signal length")
    if scan_line_y.shape != (n_lines,):
        raise RuntimeError("scan_line_y length does not match n_lines")
    if mask_x.shape != (mask_w,) or mask_y.shape != (mask_h,):
        raise RuntimeError("mask_x / mask_y shape mismatch with masks")
    if len(set(sample_ids.tolist())) != n:
        raise RuntimeError("sample_id values are not unique")

    numeric_arrays = {
        "delta_bz": delta_bz,
        "bz_defect": bz_defect,
        "bz_no_defect": bz_no_defect,
        "masks": masks,
        "sensor_x": sensor_x,
        "scan_line_y": scan_line_y,
        "mask_x": mask_x,
        "mask_y": mask_y,
    }
    finite = {name: bool(np.isfinite(value).all()) for name, value in numeric_arrays.items()}
    if not all(finite.values()):
        raise RuntimeError(f"non-finite numeric arrays: {finite}")
    delta_matches = bool(np.allclose(delta_bz, bz_defect - bz_no_defect, rtol=1e-9, atol=1e-12))
    if not delta_matches:
        raise RuntimeError("delta_bz does not match bz_defect - bz_no_defect")
    if not np.any(np.abs(delta_bz) > 0):
        raise RuntimeError("delta_bz is all zero")
    if not np.all(masks.reshape(n, -1).sum(axis=1) > 0):
        raise RuntimeError("one or more masks are empty")
    if n_lines >= 2:
        max_line_diff = max(
            float(np.max(np.abs(delta_bz[:, line_index, :] - delta_bz[:, 0, :])))
            for line_index in range(1, n_lines)
        )
    else:
        max_line_diff = 0.0
    if max_line_diff <= 1e-12:
        raise RuntimeError("scan lines are numerically identical")

    geometry_params = load_json_array(data["geometry_params"])
    geometry_mask_ious = []
    for index, geometry in enumerate(geometry_params):
        if not all(key in geometry for key in ("center_x_m", "center_y_m", "width_m", "length_m")):
            geometry_mask_ious.append(None)
            continue
        yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
        expected = (
            (np.abs(xx - float(geometry["center_x_m"])) <= float(geometry["width_m"]) / 2.0)
            & (np.abs(yy - float(geometry["center_y_m"])) <= float(geometry["length_m"]) / 2.0)
        )
        stored = masks[index].astype(bool)
        union = np.logical_or(expected, stored).sum()
        geometry_mask_ious.append(
            1.0 if union == 0 else float(np.logical_and(expected, stored).sum() / union)
        )
    if any(value is None or value < 0.999 for value in geometry_mask_ious):
        raise RuntimeError(f"geometry_params do not explain mask rasterization: {geometry_mask_ious}")

    return {
        "data": data,
        "missing": missing,
        "finite": finite,
        "delta_matches": delta_matches,
        "max_line_diff": max_line_diff,
        "geometry_mask_ious": geometry_mask_ious,
        "suggested_split": suggested_split,
    }


def split_indices(sample_count: int, suggested_split: np.ndarray | None) -> dict[str, list[int]]:
    if suggested_split is not None:
        result = {
            "train": np.where(suggested_split == "train")[0].tolist(),
            "val": np.where(suggested_split == "val")[0].tolist(),
            "test": np.where(suggested_split == "test")[0].tolist(),
        }
        if result["train"] and result["val"] and result["test"]:
            return result
    if sample_count < 8:
        raise RuntimeError("fallback smoke split expects at least 8 samples")
    return {"train": list(range(6)), "val": [6], "test": [7]}


class ComsolSmokeDataset(Dataset):
    def __init__(self, signals: np.ndarray, masks: np.ndarray, indices: list[int]):
        self.signals = torch.from_numpy(signals[indices]).float()
        self.masks = torch.from_numpy(masks[indices]).float()
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.signals[item], self.masks[item], self.indices[item]


class TinyComsolMaskDecoder(nn.Module):
    def __init__(self, n_lines: int, mask_h: int, mask_w: int):
        super().__init__()
        if (mask_h, mask_w) != (64, 128):
            raise ValueError("This tiny smoke decoder expects 64x128 masks.")
        self.encoder = nn.Sequential(
            nn.Conv1d(n_lines, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )
        self.project = nn.Sequential(
            nn.Linear(128 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * 4 * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        feature = self.project(latent).view(x.shape[0], 128, 4, 8)
        return self.decoder(feature).squeeze(1)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    intersection = (prob * target).flatten(1).sum(dim=1)
    denom = prob.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def loss_components(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    return bce + dice, bce, dice


def center_of_mass(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def sample_metrics(prob: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = prob >= threshold
    true = target >= 0.5
    pred_area = int(pred.sum())
    true_area = int(true.sum())
    intersection = int(np.logical_and(pred, true).sum())
    union = int(np.logical_or(pred, true).sum())
    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if pred_area + true_area == 0 else (2.0 * intersection) / (pred_area + true_area)
    area_error = 0.0 if true_area == 0 else abs(pred_area - true_area) / true_area
    pred_center = center_of_mass(pred)
    true_center = center_of_mass(true)
    if pred_center is None or true_center is None:
        center_error = float("nan")
    else:
        center_error = float(
            np.hypot(pred_center[0] - true_center[0], pred_center[1] - true_center[1])
        )
    return {
        "iou": float(iou),
        "dice": float(dice),
        "area_error": float(area_error),
        "center_error": center_error,
        "pred_area": pred_area,
        "true_area": true_area,
        "pred_area_zero": int(pred_area == 0),
    }


def aggregate(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        return {}
    keys = ["iou", "dice", "area_error", "center_error", "pred_area_zero", "total_loss"]
    result = {}
    for key in keys:
        values = [float(row[key]) for row in selected if str(row[key]).lower() != "nan"]
        result[f"{split}_{key}_mean"] = float(np.mean(values)) if values else float("nan")
    return result


def evaluate_model(
    model: nn.Module,
    dataset: ComsolSmokeDataset,
    device: torch.device,
    threshold: float,
    sample_ids: np.ndarray,
    defect_types: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    probs: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for signals, masks, indices in loader:
            signals = signals.to(device)
            masks = masks.to(device)
            logits = model(signals)
            total, bce, dice = loss_components(logits, masks)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
            target = masks.cpu().numpy()[0]
            index = int(indices.item())
            probs[index] = prob
            metrics = sample_metrics(prob, target, threshold)
            rows.append(
                {
                    "sample_id": as_text(sample_ids[index]),
                    "defect_type": as_text(defect_types[index]),
                    "threshold": threshold,
                    **metrics,
                    "bce_loss": float(bce.item()),
                    "dice_loss": float(dice.item()),
                    "total_loss": float(total.item()),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                    "notes": "tiny_smoke_only",
                }
            )
    return rows, probs


def evaluate_loss(model: nn.Module, dataset: ComsolSmokeDataset, device: torch.device) -> tuple[float, float, float]:
    loader = DataLoader(dataset, batch_size=max(1, len(dataset)), shuffle=False)
    model.eval()
    totals: list[float] = []
    bces: list[float] = []
    dices: list[float] = []
    with torch.no_grad():
        for signals, masks, _ in loader:
            signals = signals.to(device)
            masks = masks.to(device)
            logits = model(signals)
            total, bce, dice = loss_components(logits, masks)
            totals.append(float(total.item()))
            bces.append(float(bce.item()))
            dices.append(float(dice.item()))
    return float(np.mean(totals)), float(np.mean(bces)), float(np.mean(dices))


def select_threshold(
    model: nn.Module,
    val_dataset: ComsolSmokeDataset,
    device: torch.device,
    candidates: list[float],
) -> tuple[float, dict[float, dict[str, float]]]:
    model.eval()
    with torch.no_grad():
        signals, masks, _ = next(iter(DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)))
        prob = torch.sigmoid(model(signals.to(device))).cpu().numpy()
        target = masks.numpy()
    scores: dict[float, dict[str, float]] = {}
    for threshold in candidates:
        metrics = [sample_metrics(prob[index], target[index], threshold) for index in range(prob.shape[0])]
        mean_iou = float(np.mean([item["iou"] for item in metrics]))
        mean_dice = float(np.mean([item["dice"] for item in metrics]))
        mean_area_error = float(np.mean([item["area_error"] for item in metrics]))
        scores[threshold] = {
            "iou": mean_iou,
            "dice": mean_dice,
            "area_error": mean_area_error,
            "score": mean_iou + mean_dice - mean_area_error,
        }
    best_threshold = max(candidates, key=lambda value: scores[value]["score"])
    return best_threshold, scores


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_previews(
    preview_dir: Path,
    probs: dict[int, np.ndarray],
    masks: np.ndarray,
    signals: np.ndarray,
    sensor_x: np.ndarray,
    scan_line_y: np.ndarray,
    rows: list[dict[str, Any]],
    threshold: float,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = {row["sample_id"]: row for row in rows}
    for index, prob in probs.items():
        row = rows_by_id.get(f"sample_{index + 1:03d}")
        if row is None:
            continue
        pred = prob >= threshold
        true = masks[index] >= 0.5
        overlay = np.zeros((*true.shape, 3), dtype=np.float32)
        overlay[..., 1] = true.astype(np.float32)
        overlay[..., 0] = pred.astype(np.float32)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for line_index, y in enumerate(scan_line_y):
            axes[0, 0].plot(sensor_x, signals[index, line_index], label=f"y={y:.4g} m")
        axes[0, 0].set_title("delta_bz scan lines")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].imshow(true, cmap="gray")
        axes[0, 1].set_title("true mask")
        axes[0, 2].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[0, 2].set_title("predicted probability")
        axes[1, 0].imshow(pred, cmap="gray")
        axes[1, 0].set_title(f"pred mask @ {threshold:.2f}")
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("overlay red=pred green=true")
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"sample_id: {row['sample_id']}",
                    f"split: {row['split']}",
                    f"IoU: {float(row['iou']):.4f}",
                    f"Dice: {float(row['dice']):.4f}",
                    f"area_error: {float(row['area_error']):.4f}",
                ]
            ),
            va="top",
            fontsize=10,
        )
        for ax in axes.flat:
            if ax is not axes[0, 0] and ax is not axes[1, 2]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(preview_dir / f"{row['sample_id']}_{row['split']}.png", dpi=140)
        plt.close(fig)


def build_summary(context: dict[str, Any]) -> str:
    lines = [
        "# 第 20.9 COMSOL multiline tiny training smoke",
        "",
        "## 1. 数据读取与 schema",
        "",
        f"- small NPZ 是否可读：{context['npz_readable']}",
        f"- schema 是否完整：{context['schema_complete']}",
        f"- delta_bz 输入 shape：{context['delta_bz_shape']}",
        f"- masks shape：{context['masks_shape']}",
        f"- split 构造：{context['split_summary']}",
        f"- delta_bz 是否等于 bz_defect - bz_no_defect：{context['delta_matches']}",
        f"- 三条 scan line 是否不同：{context['scan_lines_different']}",
        f"- geometry_params 是否解释 mask：{context['geometry_mask_ious']}",
        "",
        "## 2. tiny model / train loop",
        "",
        "- 模型：轻量 Conv1d Bz encoder 处理 `(3, 201)`，latent 投影到 `4x8` feature map，再用 ConvTranspose2d 上采样到 `(64, 128)` mask logits。",
        "- loss：BCEWithLogits + soft Dice。",
        f"- epochs：{context['epochs']}",
        f"- selected threshold：{context['threshold']}",
        f"- train loop 是否跑通：{context['train_loop_ok']}",
        f"- train loss 是否下降：{context['train_loss_decreased']}，initial={context['initial_train_loss']:.6f}, final={context['final_train_loss']:.6f}",
        f"- 是否能 overfit 6 个 train samples：{context['can_overfit_train_samples']}",
        "",
        "## 3. smoke metrics",
        "",
        f"- train：{context['train_metrics']}",
        f"- val：{context['val_metrics']}",
        f"- test：{context['test_metrics']}",
        f"- 是否出现全空预测：{context['has_empty_prediction']}",
        f"- 是否出现全图预测：{context['has_full_prediction']}",
        f"- 是否出现 NaN：{context['has_nan']}",
        "",
        "## 4. preview",
        "",
        f"- preview 是否生成：{context['preview_generated']}",
        f"- preview 目录：{context['preview_dir']}",
        "",
        "## 5. 结论",
        "",
        "- 该结果只能说明 COMSOL small pack 的 schema、loader、normalization、训练/验证/测试循环和 preview 链路可以跑通。",
        "- 8 个样本不足以支持正式模型训练或性能结论，不能作为 candidate 或 baseline 依据。",
        "- 下一步应优先扩展 COMSOL 样本数，并增加 defect_type 多样性；PINN_project 侧则需要把这个 smoke loader 固化成正式 prepare/dataset loader 后再做小训练 gate。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if args.epochs < 1 or args.epochs > 100:
        raise ValueError("--epochs must be between 1 and 100 for this smoke.")

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    preview_dir = resolve(args.preview_dir)

    validation = validate_npz(npz_path)
    data = validation["data"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = np.array([as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([as_text(item) for item in data["defect_types"].tolist()])
    scan_line_y = data["scan_line_y"].astype(np.float64)
    sensor_x = data["sensor_x"].astype(np.float64)
    splits = split_indices(delta_bz.shape[0], validation["suggested_split"])

    train_mean = delta_bz[splits["train"]].mean(axis=(0, 2), keepdims=True)
    train_std = delta_bz[splits["train"]].std(axis=(0, 2), keepdims=True)
    train_std = np.maximum(train_std, 1e-8)
    normalized = (delta_bz - train_mean) / train_std

    train_dataset = ComsolSmokeDataset(normalized, masks, splits["train"])
    val_dataset = ComsolSmokeDataset(normalized, masks, splits["val"])
    test_dataset = ComsolSmokeDataset(normalized, masks, splits["test"])
    all_dataset = ComsolSmokeDataset(normalized, masks, list(range(delta_bz.shape[0])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyComsolMaskDecoder(delta_bz.shape[1], masks.shape[1], masks.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    epoch_rows: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    initial_train_loss = None
    final_train_loss = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_totals: list[float] = []
        batch_bces: list[float] = []
        batch_dices: list[float] = []
        for signals, target, _ in train_loader:
            signals = signals.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            total, bce, dice = loss_components(logits, target)
            total.backward()
            optimizer.step()
            batch_totals.append(float(total.item()))
            batch_bces.append(float(bce.item()))
            batch_dices.append(float(dice.item()))

        train_loss = float(np.mean(batch_totals))
        train_bce = float(np.mean(batch_bces))
        train_dice = float(np.mean(batch_dices))
        val_loss, val_bce, val_dice = evaluate_loss(model, val_dataset, device)
        threshold_rows, _ = evaluate_model(model, val_dataset, device, 0.5, sample_ids, defect_types)
        val_iou = float(np.mean([row["iou"] for row in threshold_rows]))
        val_dice_metric = float(np.mean([row["dice"] for row in threshold_rows]))
        val_area_error = float(np.mean([row["area_error"] for row in threshold_rows]))
        val_score = val_iou + val_dice_metric - val_area_error
        if val_score > best_score:
            best_score = val_score
            best_state = deepcopy(model.state_dict())
        if initial_train_loss is None:
            initial_train_loss = train_loss
        final_train_loss = train_loss
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_dice_loss": train_dice,
                "val_loss": val_loss,
                "val_bce": val_bce,
                "val_dice_loss": val_dice,
                "val_iou_at_0_5": val_iou,
                "val_dice_at_0_5": val_dice_metric,
                "val_area_error_at_0_5": val_area_error,
                "val_score_at_0_5": val_score,
            }
        )

    model.load_state_dict(best_state)
    candidates = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    selected_threshold, threshold_scores = select_threshold(model, val_dataset, device, candidates)

    metric_rows: list[dict[str, Any]] = []
    all_probs: dict[int, np.ndarray] = {}
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        rows, probs = evaluate_model(model, dataset, device, selected_threshold, sample_ids, defect_types)
        for row in rows:
            row["split"] = split_name
        metric_rows.extend(rows)
        all_probs.update(probs)

    all_rows, all_preview_probs = evaluate_model(model, all_dataset, device, selected_threshold, sample_ids, defect_types)
    split_by_index = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_by_index[index] = split_name
    for index, row in enumerate(all_rows):
        row["split"] = split_by_index[index]
    make_previews(
        preview_dir,
        all_preview_probs,
        masks,
        delta_bz,
        sensor_x,
        scan_line_y,
        all_rows,
        selected_threshold,
    )

    write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    write_csv(epoch_log_path, epoch_rows, EPOCH_FIELDS)

    train_metrics = aggregate(metric_rows, "train")
    val_metrics = aggregate(metric_rows, "val")
    test_metrics = aggregate(metric_rows, "test")
    train_loss_decreased = bool(final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss)
    can_overfit_train_samples = bool(
        train_loss_decreased
        and train_metrics.get("train_dice_mean", 0.0) > 0.5
        and train_metrics.get("train_iou_mean", 0.0) > 0.35
    )
    has_empty_prediction = any(int(row["pred_area_zero"]) == 1 for row in metric_rows)
    full_area = masks.shape[1] * masks.shape[2]
    has_full_prediction = any(int(row["pred_area"]) >= full_area for row in metric_rows)
    has_nan = any(not np.isfinite(float(row["total_loss"])) for row in metric_rows)

    context = {
        "npz_readable": True,
        "schema_complete": len(validation["missing"]) == 0,
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "split_summary": {name: [as_text(sample_ids[index]) for index in indices] for name, indices in splits.items()},
        "delta_matches": validation["delta_matches"],
        "scan_lines_different": validation["max_line_diff"] > 1e-12,
        "geometry_mask_ious": validation["geometry_mask_ious"],
        "epochs": args.epochs,
        "threshold": selected_threshold,
        "train_loop_ok": True,
        "train_loss_decreased": train_loss_decreased,
        "initial_train_loss": float(initial_train_loss if initial_train_loss is not None else float("nan")),
        "final_train_loss": float(final_train_loss if final_train_loss is not None else float("nan")),
        "can_overfit_train_samples": can_overfit_train_samples,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "has_empty_prediction": has_empty_prediction,
        "has_full_prediction": has_full_prediction,
        "has_nan": has_nan,
        "preview_generated": True,
        "preview_dir": str(preview_dir),
        "threshold_scores": threshold_scores,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(context), encoding="utf-8-sig")
    print(json.dumps(context, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

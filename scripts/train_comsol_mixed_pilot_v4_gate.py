from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import train_comsol_multiline_tiny_smoke as tiny


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "prepared"
    / "comsol_single_defect_multiline_forward_pack_v1_pilot_v4_mixed.npz"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_mixed_pilot_v4_training_gate_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v4_training_gate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v4_training_gate_epoch_log.csv"
DEFAULT_DEFECT_TYPE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v4_defect_type_summary.csv"
DEFAULT_ANGLE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v4_angle_summary.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_mixed_pilot_v4_gate"

THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPECTED_DEFECT_COUNTS = {"rectangular_notch": 120, "rotated_rect": 48}
EXPECTED_SPLIT_COUNTS = {"train": 112, "val": 28, "test": 28}
EXPECTED_ROTATED_ANGLES = {-30.0, -20.0, -10.0, 10.0, 20.0, 30.0}

METRIC_FIELDS = [
    "source_index",
    "sample_id",
    "split",
    "defect_type",
    "angle_deg",
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

GROUP_FIELDS = [
    "group",
    "split",
    "sample_count",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_mean",
    "pred_area_mean",
    "true_area_mean",
    "pred_area_zero_sum",
    "total_loss_mean",
]

ANGLE_FIELDS = [
    "defect_type",
    "angle_deg",
    "split",
    "sample_count",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_mean",
    "pred_area_zero_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mixed COMSOL pilot_v4 training gate.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--defect-type-summary", type=Path, default=DEFAULT_DEFECT_TYPE_SUMMARY)
    parser.add_argument("--angle-summary", type=Path, default=DEFAULT_ANGLE_SUMMARY)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_geometry(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, dict):
        return dict(value)
    parsed = json.loads(tiny.as_text(value))
    if not isinstance(parsed, dict):
        raise RuntimeError("geometry_params entry is not a JSON object")
    return parsed


def rasterize_rect_from_geometry(geometry: dict[str, Any], mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    angle_rad = float(geometry.get("angle_rad", np.deg2rad(float(geometry.get("angle", 0.0)))))
    center_x = float(geometry["center_x"])
    center_y = float(geometry["center_y"])
    width = float(geometry["width"])
    length = float(geometry["length"])
    xx, yy = np.meshgrid(mask_x, mask_y, indexing="xy")
    dx = xx - center_x
    dy = yy - center_y
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return (np.abs(local_x) <= width / 2.0) & (np.abs(local_y) <= length / 2.0)


def mask_rotation_visible(mask: np.ndarray, angle_deg: float) -> bool:
    if abs(angle_deg) < 1e-6:
        return True
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return False
    coords = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1)
    covariance = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    principal_vec = eigvecs[:, int(np.argmax(eigvals))]
    principal = np.degrees(np.arctan2(float(principal_vec[1]), float(principal_vec[0])))
    normalized = ((principal + 90.0) % 180.0) - 90.0
    distance_to_axis = min(abs(normalized), abs(abs(normalized) - 90.0))
    return distance_to_axis > 2.0


def validate_mixed_npz(npz_path: Path) -> dict[str, Any]:
    required = [
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
        "split",
    ]
    data = np.load(npz_path, allow_pickle=True)
    missing = [field for field in required if field not in data.files]
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
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    split_values = np.array([tiny.as_text(item) for item in data["split"].tolist()])
    geometry_params = [parse_geometry(item) for item in data["geometry_params"].tolist()]

    if delta_bz.shape != (168, 3, 201):
        raise RuntimeError(f"unexpected delta_bz shape: {delta_bz.shape}")
    if bz_defect.shape != delta_bz.shape or bz_no_defect.shape != delta_bz.shape:
        raise RuntimeError("bz_defect / bz_no_defect shape mismatch with delta_bz")
    if masks.shape != (168, 64, 128):
        raise RuntimeError(f"unexpected masks shape: {masks.shape}")
    if sensor_x.shape != (201,) or scan_line_y.shape != (3,):
        raise RuntimeError("sensor_x / scan_line_y shape mismatch")
    if mask_x.shape != (128,) or mask_y.shape != (64,):
        raise RuntimeError("mask_x / mask_y shape mismatch")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise RuntimeError("sample_id values are not unique")

    split_counts = {name: int(np.sum(split_values == name)) for name in ("train", "val", "test")}
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"unexpected split counts: {split_counts}")
    defect_counts = {
        name: int(np.sum(defect_types == name))
        for name in sorted(set(defect_types.tolist()))
    }
    if defect_counts != EXPECTED_DEFECT_COUNTS:
        raise RuntimeError(f"unexpected defect_type distribution: {defect_counts}")
    split_defect_counts = {
        split_name: {
            defect_name: int(np.sum((split_values == split_name) & (defect_types == defect_name)))
            for defect_name in EXPECTED_DEFECT_COUNTS
        }
        for split_name in ("train", "val", "test")
    }
    if any(any(count <= 0 for count in values.values()) for values in split_defect_counts.values()):
        raise RuntimeError(f"each split must contain both defect types: {split_defect_counts}")

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
    max_line_diff = max(
        float(np.max(np.abs(delta_bz[:, line_index, :] - delta_bz[:, 0, :])))
        for line_index in range(1, delta_bz.shape[1])
    )
    if max_line_diff <= 1e-12:
        raise RuntimeError("scan lines are numerically identical")
    if not np.all(masks.reshape(masks.shape[0], -1).sum(axis=1) > 0):
        raise RuntimeError("one or more masks are empty")
    if not np.any(np.abs(delta_bz) > 0):
        raise RuntimeError("delta_bz is all zero")
    for name, coords in (("sensor_x", sensor_x), ("scan_line_y", scan_line_y), ("mask_x", mask_x), ("mask_y", mask_y)):
        if not np.all(np.diff(coords) > 0):
            raise RuntimeError(f"{name} is not strictly increasing")

    required_geometry_keys = {
        "defect_type",
        "center_x",
        "center_y",
        "width",
        "length",
        "depth",
        "angle",
        "angle_deg",
        "angle_rad",
        "units",
        "source_sample_id",
    }
    geometry_mask_ious: list[float] = []
    rotated_visible: list[bool] = []
    angles = []
    for index, geometry in enumerate(geometry_params):
        missing_geometry = sorted(required_geometry_keys - set(geometry))
        if missing_geometry:
            raise RuntimeError(f"geometry_params missing keys for sample {index}: {missing_geometry}")
        if geometry["defect_type"] != defect_types[index]:
            raise RuntimeError(f"geometry defect_type mismatch at sample {index}")
        angle = float(geometry["angle"])
        angle_deg = float(geometry["angle_deg"])
        if abs(angle - angle_deg) > 1e-6:
            raise RuntimeError(f"angle / angle_deg mismatch at sample {index}")
        if defect_types[index] == "rectangular_notch" and abs(angle_deg) > 1e-6:
            raise RuntimeError(f"rectangular_notch angle must be zero at sample {index}")
        angles.append(angle_deg)
        expected = rasterize_rect_from_geometry(geometry, mask_x, mask_y)
        stored = masks[index].astype(bool)
        union = np.logical_or(expected, stored).sum()
        iou = 1.0 if union == 0 else float(np.logical_and(expected, stored).sum() / union)
        geometry_mask_ious.append(iou)
        if defect_types[index] == "rotated_rect":
            rotated_visible.append(mask_rotation_visible(stored, angle_deg))
    if min(geometry_mask_ious) < 0.999:
        raise RuntimeError(f"geometry_params do not explain masks: min IoU={min(geometry_mask_ious):.6f}")
    if not all(rotated_visible):
        raise RuntimeError("one or more rotated_rect masks do not visibly express angle variation")

    angles_array = np.array(angles, dtype=np.float32)
    rotated_angles = set(float(value) for value in angles_array[defect_types == "rotated_rect"].tolist())
    if rotated_angles != EXPECTED_ROTATED_ANGLES:
        raise RuntimeError(f"unexpected rotated angle values: {sorted(rotated_angles)}")

    angle_by_split = {
        split_name: sorted({float(angles_array[index]) for index in np.where(split_values == split_name)[0].tolist()})
        for split_name in ("train", "val", "test")
    }
    splits = {
        split_name: np.where(split_values == split_name)[0].tolist()
        for split_name in ("train", "val", "test")
    }
    return {
        "data": data,
        "missing": missing,
        "finite": finite,
        "delta_matches": delta_matches,
        "max_line_diff": max_line_diff,
        "geometry_mask_ious": geometry_mask_ious,
        "rotated_visible": rotated_visible,
        "angles": angles_array,
        "split_counts": split_counts,
        "defect_counts": defect_counts,
        "split_defect_counts": split_defect_counts,
        "angle_by_split": angle_by_split,
        "splits": splits,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    threshold: float,
    sample_ids: np.ndarray,
    defect_types: np.ndarray,
    angles: np.ndarray,
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
            total, bce, dice = tiny.loss_components(logits, masks)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
            target = masks.cpu().numpy()[0]
            index = int(indices.item())
            probs[index] = prob
            metrics = tiny.sample_metrics(prob, target, threshold)
            rows.append(
                {
                    "source_index": index,
                    "sample_id": tiny.as_text(sample_ids[index]),
                    "defect_type": tiny.as_text(defect_types[index]),
                    "angle_deg": float(angles[index]),
                    "threshold": threshold,
                    **metrics,
                    "bce_loss": float(bce.item()),
                    "dice_loss": float(dice.item()),
                    "total_loss": float(total.item()),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                    "notes": "mixed_pilot_v4_training_gate_only",
                }
            )
    return rows, probs


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def summarize_rows(rows: list[dict[str, Any]], group_name: str, split_name: str) -> dict[str, Any]:
    return {
        "group": group_name,
        "split": split_name,
        "sample_count": len(rows),
        "iou_mean": mean_or_nan([float(row["iou"]) for row in rows]),
        "dice_mean": mean_or_nan([float(row["dice"]) for row in rows]),
        "area_error_mean": mean_or_nan([float(row["area_error"]) for row in rows]),
        "center_error_mean": mean_or_nan([float(row["center_error"]) for row in rows if str(row["center_error"]).lower() != "nan"]),
        "pred_area_mean": mean_or_nan([float(row["pred_area"]) for row in rows]),
        "true_area_mean": mean_or_nan([float(row["true_area"]) for row in rows]),
        "pred_area_zero_sum": int(sum(int(row["pred_area_zero"]) for row in rows)),
        "total_loss_mean": mean_or_nan([float(row["total_loss"]) for row in rows]),
    }


def split_summary_rows(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: summarize_rows([row for row in metric_rows if row["split"] == split_name], "all", split_name)
        for split_name in ("train", "val", "test")
    }


def defect_type_summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        for defect_type in sorted({row["defect_type"] for row in source}):
            selected = [row for row in source if row["defect_type"] == defect_type]
            rows.append(summarize_rows(selected, defect_type, split_name))
    return rows


def angle_summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        groups = sorted({(row["defect_type"], float(row["angle_deg"])) for row in source}, key=lambda item: (item[0], item[1]))
        for defect_type, angle in groups:
            selected = [row for row in source if row["defect_type"] == defect_type and float(row["angle_deg"]) == angle]
            rows.append(
                {
                    "defect_type": defect_type,
                    "angle_deg": angle,
                    "split": split_name,
                    "sample_count": len(selected),
                    "iou_mean": mean_or_nan([float(row["iou"]) for row in selected]),
                    "dice_mean": mean_or_nan([float(row["dice"]) for row in selected]),
                    "area_error_mean": mean_or_nan([float(row["area_error"]) for row in selected]),
                    "center_error_mean": mean_or_nan([float(row["center_error"]) for row in selected if str(row["center_error"]).lower() != "nan"]),
                    "pred_area_zero_sum": int(sum(int(row["pred_area_zero"]) for row in selected)),
                }
            )
    return rows


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []

    def add_index(index: int) -> None:
        if index not in selected:
            selected.append(index)

    val_test = [row for row in metric_rows if row["split"] in {"val", "test"}]
    for defect_type in ("rectangular_notch", "rotated_rect"):
        rows = [row for row in val_test if row["defect_type"] == defect_type]
        if rows:
            add_index(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
            add_index(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))

    rotated_val_test = [row for row in val_test if row["defect_type"] == "rotated_rect"]
    for angle in sorted({float(row["angle_deg"]) for row in rotated_val_test}):
        rows = [row for row in rotated_val_test if float(row["angle_deg"]) == angle]
        if rows:
            add_index(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))

    for reverse in (True, False):
        for row in sorted(val_test, key=lambda item: float(item["dice"]), reverse=reverse)[:8]:
            add_index(int(row["source_index"]))

    train_rows = [row for row in metric_rows if row["split"] == "train"]
    for reverse in (True, False):
        for row in sorted(train_rows, key=lambda item: float(item["dice"]), reverse=reverse)[:4]:
            add_index(int(row["source_index"]))
    return selected[:20]


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
    rows_by_index = {int(row["source_index"]): row for row in rows}
    for index, prob in probs.items():
        row = rows_by_index[index]
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
                    f"type: {row['defect_type']}",
                    f"angle: {float(row['angle_deg']):.1f} deg",
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
        safe_angle = f"{float(row['angle_deg']):+04.0f}"
        fig.savefig(preview_dir / f"{row['sample_id']}_{row['split']}_{row['defect_type']}_angle_{safe_angle}.png", dpi=140)
        plt.close(fig)


def build_summary(context: dict[str, Any]) -> str:
    lines = [
        "# 第 20.17 COMSOL mixed pilot_v4 training gate",
        "",
        "## 1. NPZ / schema / mixed defect_type 检查",
        "",
        f"- mixed pilot_v4 NPZ 是否可读：{context['npz_readable']}",
        f"- schema 是否完整：{context['schema_complete']}",
        f"- split 是否为 112 / 28 / 28：{context['split_is_112_28_28']}",
        f"- defect_type 分布是否正确：{context['defect_distribution_ok']}，{context['defect_type_distribution']}",
        f"- 每个 split 是否同时包含 rectangular_notch 和 rotated_rect：{context['each_split_has_both_defect_types']}，{context['split_defect_counts']}",
        f"- angle 分布是否正确：{context['angle_distribution_ok']}，{context['angle_distribution']}",
        f"- train / val / test angle 覆盖：{context['angle_by_split']}",
        f"- mask 是否真实体现 angle variation：{context['mask_angle_variation_ok']}",
        f"- geometry_params 是否解释 mask：{context['geometry_mask_ious_summary']}",
        f"- delta_bz 输入 shape：{context['delta_bz_shape']}",
        f"- mask 输出 shape：{context['masks_shape']}",
        f"- delta_bz 是否等于 bz_defect - bz_no_defect：{context['delta_matches']}",
        f"- 三条 scan line 是否不同：{context['scan_lines_different']}",
        f"- 坐标是否单调且完整：{context['coords_valid']}",
        "",
        "## 2. Normalization / loader",
        "",
        "- normalization 方法：只使用 train split 的 delta_bz 统计量，按 n_lines 通道计算 mean/std；val/test 只复用 train mean/std。",
        f"- train mean shape：{context['train_mean_shape']}",
        f"- train std shape：{context['train_std_shape']}",
        f"- normalization 是否只使用 train split：{context['normalization_train_only']}",
        "",
        "## 3. 训练 gate",
        "",
        "- 模型：轻量 Conv1d Bz encoder 处理 `(3, 201)`，latent 投影到 `4x8` feature map，再用 ConvTranspose2d 上采样到 `(64, 128)` mask logits。",
        "- loss：BCEWithLogits + soft Dice；未使用 defect_type / angle / geometry 参数监督。",
        f"- epochs：{context['epochs']}",
        f"- batch_size：{context['batch_size']}",
        "- checkpoint selection：validation score = IoU + Dice - area_error；test 只用于最终 smoke evaluation。",
        f"- selected threshold：{context['threshold']}",
        f"- validation threshold scores：{context['threshold_scores']}",
        f"- best validation epoch：{context['best_epoch']}",
        f"- train loop 是否跑通：{context['train_loop_ok']}",
        f"- train loss 是否下降：{context['train_loss_decreased']}，initial={context['initial_train_loss']:.6f}, final={context['final_train_loss']:.6f}",
        f"- 是否能拟合 112 个 train samples：{context['can_fit_train_samples']}",
        "",
        "## 4. pilot_v4 metrics",
        "",
        f"- train：{context['train_metrics']}",
        f"- val：{context['val_metrics']}",
        f"- test：{context['test_metrics']}",
        f"- rectangular_notch / rotated_rect 分组指标文件：{context['defect_type_summary_path']}",
        f"- per-angle 指标是否有明显问题：{context['per_angle_issue']}；文件：{context['angle_summary_path']}",
        f"- mixed training 是否明显伤害某一类：{context['mixed_type_imbalance_issue']}",
        f"- 是否出现全空预测：{context['has_empty_prediction']}",
        f"- 是否出现全图预测：{context['has_full_prediction']}",
        f"- 是否出现 NaN：{context['has_nan']}",
        "",
        "## 5. Preview",
        "",
        f"- preview 是否生成：{context['preview_generated']}",
        f"- preview 目录：{context['preview_dir']}",
        f"- preview 样本：{context['preview_sample_ids']}",
        "",
        "## 6. 结论",
        "",
        "- 该结果只说明 mixed pilot_v4 数据包的读取、schema、dataset loader、train-only normalization、mixed defect_type training gate、validation threshold selection、test smoke evaluation 和 preview 链路可用。",
        "- 168-sample mixed pack 可以支持下一阶段 defect_type 多样性训练 gate，但仍不是正式泛化结论，也不更新 v3_complex CURRENT_BASELINE。",
        "- 当前限制：样本数仍是 pilot 级别；defect_type 只有 rectangular_notch + rotated_rect；未包含 polygon；未包含 multi_defect；geometry 范围仍有限。",
        f"- 下一步建议：{context['next_step_recommendation']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200 for this mixed pilot_v4 gate.")
    tiny.set_seed(args.seed)

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    defect_type_summary_path = resolve(args.defect_type_summary)
    angle_summary_path = resolve(args.angle_summary)
    preview_dir = resolve(args.preview_dir)

    validation = validate_mixed_npz(npz_path)
    data = validation["data"]
    splits = validation["splits"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    angles = validation["angles"]
    scan_line_y = data["scan_line_y"].astype(np.float64)
    sensor_x = data["sensor_x"].astype(np.float64)

    # Train-only normalization: validation/test never contribute statistics.
    train_mean = delta_bz[splits["train"]].mean(axis=(0, 2), keepdims=True)
    train_std = np.maximum(delta_bz[splits["train"]].std(axis=(0, 2), keepdims=True), 1e-8)
    normalized = (delta_bz - train_mean) / train_std

    train_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["train"])
    val_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["val"])
    test_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["test"])
    all_dataset = tiny.ComsolSmokeDataset(normalized, masks, list(range(delta_bz.shape[0])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tiny.TinyComsolMaskDecoder(delta_bz.shape[1], masks.shape[1], masks.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    epoch_rows: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    initial_train_loss: float | None = None
    final_train_loss: float | None = None

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
            total, bce, dice = tiny.loss_components(logits, target)
            total.backward()
            optimizer.step()
            batch_totals.append(float(total.item()))
            batch_bces.append(float(bce.item()))
            batch_dices.append(float(dice.item()))

        train_loss = float(np.mean(batch_totals))
        train_bce = float(np.mean(batch_bces))
        train_dice_loss = float(np.mean(batch_dices))
        val_loss, val_bce, val_dice_loss = tiny.evaluate_loss(model, val_dataset, device)
        val_rows_at_half, _ = evaluate_model(model, val_dataset, device, 0.5, sample_ids, defect_types, angles)
        val_iou = float(np.mean([row["iou"] for row in val_rows_at_half]))
        val_dice_metric = float(np.mean([row["dice"] for row in val_rows_at_half]))
        val_area_error = float(np.mean([row["area_error"] for row in val_rows_at_half]))
        val_score = val_iou + val_dice_metric - val_area_error
        if val_score > best_score:
            best_score = val_score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        if initial_train_loss is None:
            initial_train_loss = train_loss
        final_train_loss = train_loss
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_dice_loss": train_dice_loss,
                "val_loss": val_loss,
                "val_bce": val_bce,
                "val_dice_loss": val_dice_loss,
                "val_iou_at_0_5": val_iou,
                "val_dice_at_0_5": val_dice_metric,
                "val_area_error_at_0_5": val_area_error,
                "val_score_at_0_5": val_score,
            }
        )

    model.load_state_dict(best_state)
    selected_threshold, threshold_scores = tiny.select_threshold(model, val_dataset, device, THRESHOLD_CANDIDATES)

    metric_rows: list[dict[str, Any]] = []
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        rows, _ = evaluate_model(model, dataset, device, selected_threshold, sample_ids, defect_types, angles)
        for row in rows:
            row["split"] = split_name
        metric_rows.extend(rows)

    write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    write_csv(epoch_log_path, epoch_rows, tiny.EPOCH_FIELDS)
    defect_rows = defect_type_summary_rows(metric_rows)
    angle_rows = angle_summary_rows(metric_rows)
    write_csv(defect_type_summary_path, defect_rows, GROUP_FIELDS)
    write_csv(angle_summary_path, angle_rows, ANGLE_FIELDS)

    selected_preview_indices = choose_preview_indices(metric_rows)
    all_rows, all_probs = evaluate_model(model, all_dataset, device, selected_threshold, sample_ids, defect_types, angles)
    split_by_index = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_by_index[index] = split_name
    for row in all_rows:
        row["split"] = split_by_index[int(row["source_index"])]
    selected_probs = {index: all_probs[index] for index in selected_preview_indices}
    make_previews(preview_dir, selected_probs, masks, delta_bz, sensor_x, scan_line_y, all_rows, selected_threshold)

    split_metrics = split_summary_rows(metric_rows)
    train_metrics = split_metrics["train"]
    val_metrics = split_metrics["val"]
    test_metrics = split_metrics["test"]
    train_loss_decreased = bool(
        final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss
    )
    can_fit_train_samples = bool(
        train_loss_decreased
        and float(train_metrics.get("dice_mean", 0.0)) > 0.75
        and float(train_metrics.get("iou_mean", 0.0)) > 0.60
    )
    full_area = masks.shape[1] * masks.shape[2]
    all_angle_rows = [row for row in angle_rows if row["split"] == "all"]
    per_angle_issue = any(
        float(row["dice_mean"]) < 0.45 or int(row["pred_area_zero_sum"]) > 0
        for row in all_angle_rows
    )
    all_defect_rows = {row["group"]: row for row in defect_rows if row["split"] == "all"}
    dice_gap = abs(float(all_defect_rows["rectangular_notch"]["dice_mean"]) - float(all_defect_rows["rotated_rect"]["dice_mean"]))
    mixed_type_imbalance_issue = bool(dice_gap > 0.20)
    next_step = (
        "进入 mixed pilot_v4 后续训练 gate 后，可优先加入 polygon；如果要提升结论强度，再扩展 rectangular_notch + rotated_rect 样本数。"
        if not per_angle_issue and not mixed_type_imbalance_issue
        else "先检查 mixed training 中的弱类别或弱角度，再决定是扩展样本数、加入 polygon，还是调整 model capacity。"
    )

    context = {
        "npz_readable": True,
        "schema_complete": len(validation["missing"]) == 0,
        "split_is_112_28_28": validation["split_counts"] == EXPECTED_SPLIT_COUNTS,
        "defect_distribution_ok": validation["defect_counts"] == EXPECTED_DEFECT_COUNTS,
        "defect_type_distribution": validation["defect_counts"],
        "each_split_has_both_defect_types": all(
            all(count > 0 for count in values.values()) for values in validation["split_defect_counts"].values()
        ),
        "split_defect_counts": validation["split_defect_counts"],
        "angle_distribution_ok": set(float(value) for value in angles[defect_types == "rotated_rect"].tolist()) == EXPECTED_ROTATED_ANGLES,
        "angle_distribution": {
            "rectangular_notch": {"0.0": int(np.sum(defect_types == "rectangular_notch"))},
            "rotated_rect": {
                float(angle): int(np.sum((defect_types == "rotated_rect") & (np.isclose(angles, angle))))
                for angle in sorted(EXPECTED_ROTATED_ANGLES)
            },
        },
        "angle_by_split": validation["angle_by_split"],
        "mask_angle_variation_ok": all(validation["rotated_visible"]),
        "geometry_mask_ious_summary": {
            "min": float(np.min(validation["geometry_mask_ious"])),
            "max": float(np.max(validation["geometry_mask_ious"])),
            "mean": float(np.mean(validation["geometry_mask_ious"])),
        },
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "delta_matches": validation["delta_matches"],
        "scan_lines_different": validation["max_line_diff"] > 1e-12,
        "coords_valid": True,
        "train_mean_shape": tuple(train_mean.shape),
        "train_std_shape": tuple(train_std.shape),
        "normalization_train_only": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "threshold": selected_threshold,
        "threshold_scores": threshold_scores,
        "best_epoch": best_epoch,
        "train_loop_ok": True,
        "train_loss_decreased": train_loss_decreased,
        "initial_train_loss": float(initial_train_loss if initial_train_loss is not None else float("nan")),
        "final_train_loss": float(final_train_loss if final_train_loss is not None else float("nan")),
        "can_fit_train_samples": can_fit_train_samples,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "defect_type_summary_path": str(defect_type_summary_path),
        "angle_summary_path": str(angle_summary_path),
        "per_angle_issue": bool(per_angle_issue),
        "mixed_type_imbalance_issue": mixed_type_imbalance_issue,
        "has_empty_prediction": any(int(row["pred_area_zero"]) == 1 for row in metric_rows),
        "has_full_prediction": any(int(row["pred_area"]) >= full_area for row in metric_rows),
        "has_nan": any(not np.isfinite(float(row["total_loss"])) for row in metric_rows),
        "preview_generated": True,
        "preview_dir": str(preview_dir),
        "preview_sample_ids": [tiny.as_text(sample_ids[index]) for index in selected_preview_indices],
        "next_step_recommendation": next_step,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(context), encoding="utf-8")
    print(json.dumps(context, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

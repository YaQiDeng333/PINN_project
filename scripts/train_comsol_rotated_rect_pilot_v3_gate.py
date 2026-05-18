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
    / "comsol_single_defect_multiline_forward_pack_v1_pilot_v3_rotated_rect.npz"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "summaries"
    / "comsol_rotated_rect_pilot_v3_training_gate_summary.txt"
)
DEFAULT_METRICS = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_rotated_rect_pilot_v3_training_gate_metrics.csv"
)
DEFAULT_EPOCH_LOG = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_rotated_rect_pilot_v3_training_gate_epoch_log.csv"
)
DEFAULT_ANGLE_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_rotated_rect_pilot_v3_angle_summary.csv"
)
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results" / "previews" / "comsol_rotated_rect_pilot_v3_gate"

THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPECTED_ANGLES = {-30.0, -20.0, -10.0, 10.0, 20.0, 30.0}

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

ANGLE_FIELDS = [
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
    parser = argparse.ArgumentParser(description="Run a rotated_rect COMSOL pilot_v3 training gate.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--angle-summary", type=Path, default=DEFAULT_ANGLE_SUMMARY)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
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


def rasterize_rotated_rect(geometry: dict[str, Any], mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    angle_rad = float(geometry.get("angle_rad", np.deg2rad(float(geometry["angle_deg"]))))
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    xx, yy = np.meshgrid(mask_x, mask_y, indexing="xy")
    dx = xx - float(geometry["center_x_m"])
    dy = yy - float(geometry["center_y_m"])
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return (
        (np.abs(local_x) <= float(geometry["width_m"]) / 2.0)
        & (np.abs(local_y) <= float(geometry["length_m"]) / 2.0)
    )


def mask_rotation_visible(mask: np.ndarray, geometry: dict[str, Any]) -> bool:
    if abs(float(geometry["angle_deg"])) < 1e-6:
        return False
    if abs(float(geometry["width_m"]) - float(geometry["length_m"])) < 1e-9:
        return False
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


def validate_rotated_rect_npz(npz_path: Path) -> dict[str, Any]:
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
    geometry_params = tiny.load_json_array(data["geometry_params"])

    if delta_bz.shape != (48, 3, 201):
        raise RuntimeError(f"unexpected delta_bz shape: {delta_bz.shape}")
    if bz_defect.shape != delta_bz.shape or bz_no_defect.shape != delta_bz.shape:
        raise RuntimeError("bz_defect / bz_no_defect shape mismatch with delta_bz")
    if masks.shape != (48, 64, 128):
        raise RuntimeError(f"unexpected masks shape: {masks.shape}")
    if sensor_x.shape != (201,) or scan_line_y.shape != (3,):
        raise RuntimeError("sensor_x / scan_line_y shape mismatch")
    if mask_x.shape != (128,) or mask_y.shape != (64,):
        raise RuntimeError("mask_x / mask_y shape mismatch")
    if len(set(sample_ids.tolist())) != 48:
        raise RuntimeError("sample_id values are not unique")
    split_counts = {name: int(np.sum(split_values == name)) for name in ("train", "val", "test")}
    if split_counts != {"train": 32, "val": 8, "test": 8}:
        raise RuntimeError(f"unexpected split counts: {split_counts}")
    if set(defect_types.tolist()) != {"rotated_rect"}:
        raise RuntimeError(f"unexpected defect types: {sorted(set(defect_types.tolist()))}")

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

    geometry_mask_ious: list[float] = []
    visible_rotation: list[bool] = []
    angles: list[float] = []
    for index, geometry in enumerate(geometry_params):
        if "angle_deg" not in geometry:
            raise RuntimeError(f"geometry_params missing angle_deg for sample {index}")
        expected = rasterize_rotated_rect(geometry, mask_x, mask_y)
        stored = masks[index].astype(bool)
        union = np.logical_or(expected, stored).sum()
        iou = 1.0 if union == 0 else float(np.logical_and(expected, stored).sum() / union)
        geometry_mask_ious.append(iou)
        visible_rotation.append(mask_rotation_visible(stored, geometry))
        angles.append(float(geometry["angle_deg"]))
    if min(geometry_mask_ious) < 0.999:
        raise RuntimeError(f"geometry_params do not explain rotated masks: min IoU={min(geometry_mask_ious):.6f}")
    if not all(visible_rotation):
        raise RuntimeError("one or more masks do not visibly express angle variation")
    if set(angles) != EXPECTED_ANGLES:
        raise RuntimeError(f"unexpected angle values: {sorted(set(angles))}")

    angle_by_split = {
        name: sorted({angles[index] for index in np.where(split_values == name)[0].tolist()})
        for name in ("train", "val", "test")
    }
    if any(len(values) < 3 for values in angle_by_split.values()):
        raise RuntimeError(f"insufficient angle coverage by split: {angle_by_split}")

    geometry_ranges: dict[str, dict[str, list[float]]] = {}
    for split_name in ("train", "val", "test"):
        indices = np.where(split_values == split_name)[0].tolist()
        geometry_ranges[split_name] = {}
        for key in ("width_m", "length_m", "depth_m", "center_x_m", "center_y_m", "angle_deg"):
            values = [float(geometry_params[index][key]) for index in indices]
            geometry_ranges[split_name][key] = [float(min(values)), float(max(values))]

    splits = {
        "train": np.where(split_values == "train")[0].tolist(),
        "val": np.where(split_values == "val")[0].tolist(),
        "test": np.where(split_values == "test")[0].tolist(),
    }
    return {
        "data": data,
        "missing": missing,
        "finite": finite,
        "delta_matches": delta_matches,
        "max_line_diff": max_line_diff,
        "geometry_mask_ious": geometry_mask_ious,
        "visible_rotation": visible_rotation,
        "angles": np.array(angles, dtype=np.float32),
        "angle_by_split": angle_by_split,
        "geometry_ranges": geometry_ranges,
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
                    "notes": "rotated_rect_pilot_v3_training_gate_only",
                }
            )
    return rows, probs


def summarize_split_metrics(metric_rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    selected = [row for row in metric_rows if row["split"] == split]
    result: dict[str, float] = {}
    for key in ("iou", "dice", "area_error", "center_error", "pred_area", "true_area", "pred_area_zero", "bce_loss", "dice_loss", "total_loss"):
        values = [float(row[key]) for row in selected if str(row[key]).lower() != "nan"]
        result[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
    return result


def angle_summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        for angle in sorted({float(row["angle_deg"]) for row in source}):
            selected = [row for row in source if float(row["angle_deg"]) == angle]
            rows.append(
                {
                    "angle_deg": angle,
                    "split": split_name,
                    "sample_count": len(selected),
                    "iou_mean": float(np.mean([float(row["iou"]) for row in selected])),
                    "dice_mean": float(np.mean([float(row["dice"]) for row in selected])),
                    "area_error_mean": float(np.mean([float(row["area_error"]) for row in selected])),
                    "center_error_mean": float(
                        np.mean([float(row["center_error"]) for row in selected if str(row["center_error"]).lower() != "nan"])
                    ),
                    "pred_area_zero_sum": int(sum(int(row["pred_area_zero"]) for row in selected)),
                }
            )
    return rows


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []
    by_angle: dict[float, list[dict[str, Any]]] = {}
    for row in metric_rows:
        if row["split"] in {"val", "test"}:
            by_angle.setdefault(float(row["angle_deg"]), []).append(row)
    for angle in sorted(by_angle):
        row = sorted(by_angle[angle], key=lambda item: float(item["dice"]), reverse=True)[0]
        index = int(row["source_index"])
        if index not in selected:
            selected.append(index)

    def add_rows(rows: list[dict[str, Any]], count: int, reverse: bool) -> None:
        ordered = sorted(rows, key=lambda item: float(item["dice"]), reverse=reverse)
        for row in ordered[:count]:
            index = int(row["source_index"])
            if index not in selected:
                selected.append(index)

    val_test = [row for row in metric_rows if row["split"] in {"val", "test"}]
    train_rows = [row for row in metric_rows if row["split"] == "train"]
    add_rows(val_test, 5, True)
    add_rows(val_test, 5, False)
    add_rows(train_rows, 3, True)
    add_rows(train_rows, 3, False)
    return selected[:16]


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
        axes[0, 1].set_title("true rotated mask")
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
        fig.savefig(preview_dir / f"{row['sample_id']}_{row['split']}_angle_{float(row['angle_deg']):+04.0f}.png", dpi=140)
        plt.close(fig)


def build_summary(context: dict[str, Any]) -> str:
    lines = [
        "# 第 20.15 COMSOL rotated_rect pilot_v3 training gate",
        "",
        "## 1. NPZ / schema / angle 检查",
        "",
        f"- rotated_rect pilot_v3 NPZ 是否可读：{context['npz_readable']}",
        f"- schema 是否完整：{context['schema_complete']}",
        f"- split 是否为 32 / 8 / 8：{context['split_is_32_8_8']}",
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
        "- normalization 方法：只使用 train split 的 delta_bz 统计量，按 `(n_lines)` 通道计算 mean/std；val/test 只复用 train mean/std。",
        f"- train mean shape：{context['train_mean_shape']}",
        f"- train std shape：{context['train_std_shape']}",
        f"- normalization 是否只使用 train split：{context['normalization_train_only']}",
        "",
        "## 3. 训练 gate",
        "",
        "- 模型：轻量 Conv1d Bz encoder 处理 `(3, 201)`，latent 投影到 `4x8` feature map，再用 ConvTranspose2d 上采样到 `(64, 128)` mask logits。",
        "- loss：BCEWithLogits + soft Dice；未使用 angle / geometry 参数监督。",
        f"- epochs：{context['epochs']}",
        f"- batch_size：{context['batch_size']}",
        "- checkpoint selection：validation score = IoU + Dice - area_error，test 只用于最终 smoke evaluation。",
        f"- selected threshold：{context['threshold']}",
        f"- best validation epoch：{context['best_epoch']}",
        f"- train loop 是否跑通：{context['train_loop_ok']}",
        f"- train loss 是否下降：{context['train_loss_decreased']}，initial={context['initial_train_loss']:.6f}, final={context['final_train_loss']:.6f}",
        f"- 是否能拟合 32 个 train samples：{context['can_fit_train_samples']}",
        "",
        "## 4. pilot_v3 metrics",
        "",
        f"- train：{context['train_metrics']}",
        f"- val：{context['val_metrics']}",
        f"- test：{context['test_metrics']}",
        f"- per-angle 指标是否有明显问题：{context['per_angle_issue']}",
        f"- per-angle 摘要文件：{context['angle_summary_path']}",
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
        "- 该结果只说明 rotated_rect pilot_v3 数据包的读取、angle/mask 一致性检查、dataset loader、train-only normalization、训练、validation threshold selection、test smoke evaluation 和 preview 链路可以跑通。",
        "- 48-sample rotated_rect pack 可以支持下一阶段 defect_type 多样性训练准备，但样本数仍小，且 defect_type 单一，不能作为正式泛化结论，也不更新 CURRENT_BASELINE。",
        "- 当前限制：只包含 rotated_rect / angled notch；未与 rectangular_notch 混合训练；未包含 polygon / multi_defect；split 仍是 pilot_v3 级别。",
        "- 下一步优先级：先合并 rectangular_notch + rotated_rect，再扩展 rotated_rect 样本数和加入 polygon；loader/schema/normalization 当前没有明显 blocker。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200 for this pilot_v3 gate.")
    tiny.set_seed(args.seed)

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    angle_summary_path = resolve(args.angle_summary)
    preview_dir = resolve(args.preview_dir)

    validation = validate_rotated_rect_npz(npz_path)
    data = validation["data"]
    splits = validation["splits"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    angles = validation["angles"]
    scan_line_y = data["scan_line_y"].astype(np.float64)
    sensor_x = data["sensor_x"].astype(np.float64)

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
    angle_rows = angle_summary_rows(metric_rows)
    write_csv(angle_summary_path, angle_rows, ANGLE_FIELDS)

    selected_preview_indices = choose_preview_indices(metric_rows)
    all_rows, all_probs = evaluate_model(model, all_dataset, device, selected_threshold, sample_ids, defect_types, angles)
    split_by_index = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_by_index[index] = split_name
    for index, row in enumerate(all_rows):
        row["split"] = split_by_index[index]
    selected_probs = {index: all_probs[index] for index in selected_preview_indices}
    make_previews(preview_dir, selected_probs, masks, delta_bz, sensor_x, scan_line_y, all_rows, selected_threshold)

    train_metrics = summarize_split_metrics(metric_rows, "train")
    val_metrics = summarize_split_metrics(metric_rows, "val")
    test_metrics = summarize_split_metrics(metric_rows, "test")
    train_loss_decreased = bool(
        final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss
    )
    can_fit_train_samples = bool(
        train_loss_decreased
        and train_metrics.get("dice_mean", 0.0) > 0.65
        and train_metrics.get("iou_mean", 0.0) > 0.45
    )
    full_area = masks.shape[1] * masks.shape[2]
    per_angle_issue = any(
        row["split"] == "all" and (float(row["dice_mean"]) < 0.35 or int(row["pred_area_zero_sum"]) > 0)
        for row in angle_rows
    )
    context = {
        "npz_readable": True,
        "schema_complete": len(validation["missing"]) == 0,
        "split_is_32_8_8": {name: len(indices) for name, indices in splits.items()} == {"train": 32, "val": 8, "test": 8},
        "angle_distribution_ok": set(float(value) for value in angles.tolist()) == EXPECTED_ANGLES,
        "angle_distribution": {float(angle): int(np.sum(angles == angle)) for angle in sorted(set(angles.tolist()))},
        "angle_by_split": validation["angle_by_split"],
        "mask_angle_variation_ok": all(validation["visible_rotation"]),
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
        "per_angle_issue": bool(per_angle_issue),
        "angle_summary_path": str(angle_summary_path),
        "has_empty_prediction": any(int(row["pred_area_zero"]) == 1 for row in metric_rows),
        "has_full_prediction": any(int(row["pred_area"]) >= full_area for row in metric_rows),
        "has_nan": any(not np.isfinite(float(row["total_loss"])) for row in metric_rows),
        "preview_generated": True,
        "preview_dir": str(preview_dir),
        "preview_sample_ids": [f"sample_{index + 1:03d}" for index in selected_preview_indices],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(context), encoding="utf-8")
    print(json.dumps(context, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

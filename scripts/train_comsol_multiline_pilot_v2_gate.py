from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

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
    / "comsol_single_defect_multiline_forward_pack_v1_pilot_v2.npz"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "summaries"
    / "comsol_multiline_pilot_v2_training_gate_summary.txt"
)
DEFAULT_METRICS = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_multiline_pilot_v2_training_gate_metrics.csv"
)
DEFAULT_EPOCH_LOG = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_multiline_pilot_v2_training_gate_epoch_log.csv"
)
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results" / "previews" / "comsol_multiline_pilot_v2_gate"
THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded pilot_v2 training gate on the COMSOL multi-line pilot_v2 pack."
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def split_indices_from_required_split(data: np.lib.npyio.NpzFile) -> dict[str, list[int]]:
    if "split" not in data.files:
        raise RuntimeError("pilot_v2 requires an explicit split field; refusing to infer a fallback split.")
    split_values = np.array([tiny.as_text(item) for item in data["split"].tolist()])
    result = {
        "train": np.where(split_values == "train")[0].tolist(),
        "val": np.where(split_values == "val")[0].tolist(),
        "test": np.where(split_values == "test")[0].tolist(),
    }
    expected = {"train": 80, "val": 20, "test": 20}
    counts = {name: len(indices) for name, indices in result.items()}
    if counts != expected:
        raise RuntimeError(f"unexpected pilot_v2 split counts: {counts}, expected {expected}")
    return result


def geometry_ranges(geometry_params: list[dict[str, Any]], indices: list[int]) -> dict[str, list[float]]:
    ranges: dict[str, list[float]] = {}
    for key in ("width_m", "length_m", "depth_m", "center_x_m", "center_y_m"):
        values = [float(geometry_params[index][key]) for index in indices if key in geometry_params[index]]
        ranges[key] = [float(min(values)), float(max(values))] if values else []
    return ranges


def validate_pilot_v2_npz(npz_path: Path) -> dict[str, Any]:
    validation = tiny.validate_npz(npz_path)
    data = validation["data"]
    splits = split_indices_from_required_split(data)
    delta_shape = tuple(data["delta_bz"].shape)
    mask_shape = tuple(data["masks"].shape)
    if delta_shape != (120, 3, 201):
        raise RuntimeError(f"unexpected delta_bz shape: {delta_shape}")
    if mask_shape != (120, 64, 128):
        raise RuntimeError(f"unexpected masks shape: {mask_shape}")
    defect_types = [tiny.as_text(item) for item in data["defect_types"].tolist()]
    if set(defect_types) != {"rectangular_notch"}:
        raise RuntimeError(f"unexpected defect types: {sorted(set(defect_types))}")
    geometry_params = tiny.load_json_array(data["geometry_params"])
    unique_geometry = {
        json.dumps(
            {
                key: geometry.get(key)
                for key in ("width_m", "length_m", "depth_m", "center_x_m", "center_y_m", "angle_rad")
            },
            sort_keys=True,
        )
        for geometry in geometry_params
    }
    if len(unique_geometry) != len(geometry_params):
        raise RuntimeError("geometry_params contain duplicated geometry combinations")
    split_ranges = {name: geometry_ranges(geometry_params, indices) for name, indices in splits.items()}
    return {
        "validation": validation,
        "splits": splits,
        "geometry_params": geometry_params,
        "split_geometry_ranges": split_ranges,
        "unique_geometry_count": len(unique_geometry),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []

    def add_rows(rows: list[dict[str, Any]], count: int, reverse: bool) -> None:
        ordered = sorted(rows, key=lambda item: float(item["dice"]), reverse=reverse)
        for row in ordered[:count]:
            index = int(row["sample_id"].split("_")[-1]) - 1
            if index not in selected:
                selected.append(index)

    val_rows = [row for row in metric_rows if row["split"] == "val"]
    test_rows = [row for row in metric_rows if row["split"] == "test"]
    train_rows = [row for row in metric_rows if row["split"] == "train"]
    add_rows(val_rows, 3, True)
    add_rows(val_rows, 3, False)
    add_rows(test_rows, 3, True)
    add_rows(test_rows, 3, False)
    add_rows(train_rows, 2, True)
    add_rows(train_rows, 2, False)
    return selected[:16]


def summarize_split_metrics(metric_rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    base = tiny.aggregate(metric_rows, split)
    selected = [row for row in metric_rows if row["split"] == split]
    for key in ("pred_area", "true_area", "bce_loss", "dice_loss"):
        values = [float(row[key]) for row in selected]
        base[f"{split}_{key}_mean"] = float(np.mean(values)) if values else float("nan")
    return base


def build_summary(context: dict[str, Any]) -> str:
    lines = [
        "# 第 20.13 COMSOL multiline pilot_v2 training gate",
        "",
        "## 1. 数据读取与 schema",
        "",
        f"- pilot_v2 NPZ 是否可读：{context['npz_readable']}",
        f"- schema 是否完整：{context['schema_complete']}",
        f"- split 是否为 80 / 20 / 20：{context['split_is_80_20_20']}",
        f"- delta_bz 输入 shape：{context['delta_bz_shape']}",
        f"- mask 输出 shape：{context['masks_shape']}",
        f"- defect_type 分布：{context['defect_type_distribution']}",
        "- pilot_v2 limitation：当前全部为 rectangular_notch，尚未包含 rotated_rect / polygon / multi_defect。",
        f"- delta_bz 是否等于 bz_defect - bz_no_defect：{context['delta_matches']}",
        f"- 三条 scan line 是否不同：{context['scan_lines_different']}",
        f"- sample_id 是否唯一：{context['sample_ids_unique']}",
        f"- geometry_params 是否解释 mask：{context['geometry_mask_ious_summary']}",
        f"- train/val/test 几何范围覆盖：{context['split_geometry_ranges']}",
        "",
        "## 2. Normalization / loader",
        "",
        "- normalization 方法：只使用 train split 的 delta_bz 统计量，按 `(n_lines)` 通道计算 mean/std；val/test 只复用 train mean/std。",
        f"- train mean shape：{context['train_mean_shape']}",
        f"- train std shape：{context['train_std_shape']}",
        f"- normalization 是否只使用 train split：{context['normalization_train_only']}",
        "",
        "## 3. pilot_v2 model / train loop",
        "",
        "- 模型：轻量 Conv1d Bz encoder 处理 `(3, 201)`，latent 投影到 `4x8` feature map，再用 ConvTranspose2d 上采样到 `(64, 128)` mask logits。",
        "- loss：BCEWithLogits + soft Dice。",
        f"- epochs：{context['epochs']}",
        f"- batch_size：{context['batch_size']}",
        f"- checkpoint selection：validation score = IoU + Dice - area_error，按 threshold=0.50 评估。",
        f"- selected threshold：{context['threshold']}",
        f"- best validation epoch：{context['best_epoch']}",
        f"- train loop 是否跑通：{context['train_loop_ok']}",
        f"- train loss 是否下降：{context['train_loss_decreased']}，initial={context['initial_train_loss']:.6f}, final={context['final_train_loss']:.6f}",
        f"- 是否能拟合 80 个 train samples：{context['can_fit_train_samples']}",
        "",
        "## 4. pilot_v2 metrics",
        "",
        f"- train：{context['train_metrics']}",
        f"- val：{context['val_metrics']}",
        f"- test：{context['test_metrics']}",
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
        "- 该结果只说明 COMSOL pilot_v2 数据的读取、dataset loader、train-only normalization、训练/验证/测试循环、validation selection、threshold selection 和 preview 链路可以跑通。",
        "- 120-sample rectangular_notch pilot_v2 pack 可以支持下一阶段更大样本训练准备，但仍不能作为正式泛化结论，也不更新 CURRENT_BASELINE。",
        "- 当前限制：defect_type 单一，只包含 rectangular_notch；未包含 rotated_rect / polygon / multi_defect；split 仍是 pilot_v2 级别；scan lines 仍为 3 条。",
        "- 下一步优先级：先扩展 defect_type 到 rotated_rect / polygon，再扩大样本数；loader/schema/normalization 当前没有明显 blocker。增加 scan lines 可放到 defect_type 多样性之后。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200 for this pilot_v2 gate.")
    tiny.set_seed(args.seed)

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    preview_dir = resolve(args.preview_dir)

    pilot = validate_pilot_v2_npz(npz_path)
    validation = pilot["validation"]
    data = validation["data"]
    splits = pilot["splits"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    scan_line_y = data["scan_line_y"].astype(np.float64)
    sensor_x = data["sensor_x"].astype(np.float64)

    # Normalization is deliberately train-only to avoid validation/test leakage.
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
        val_rows_at_half, _ = tiny.evaluate_model(model, val_dataset, device, 0.5, sample_ids, defect_types)
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
    selected_threshold, threshold_scores = tiny.select_threshold(
        model, val_dataset, device, THRESHOLD_CANDIDATES
    )

    metric_rows: list[dict[str, Any]] = []
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        rows, _ = tiny.evaluate_model(model, dataset, device, selected_threshold, sample_ids, defect_types)
        for row in rows:
            row["split"] = split_name
            row["notes"] = "pilot_v2_training_gate_only"
        metric_rows.extend(rows)

    tiny.write_csv(metrics_path, metric_rows, tiny.METRIC_FIELDS)
    tiny.write_csv(epoch_log_path, epoch_rows, tiny.EPOCH_FIELDS)

    selected_preview_indices = choose_preview_indices(metric_rows)
    all_rows, all_preview_probs = tiny.evaluate_model(
        model, all_dataset, device, selected_threshold, sample_ids, defect_types
    )
    split_by_index = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_by_index[index] = split_name
    for index, row in enumerate(all_rows):
        row["split"] = split_by_index[index]
    selected_probs = {index: all_preview_probs[index] for index in selected_preview_indices}
    tiny.make_previews(
        preview_dir,
        selected_probs,
        masks,
        delta_bz,
        sensor_x,
        scan_line_y,
        all_rows,
        selected_threshold,
    )

    train_metrics = summarize_split_metrics(metric_rows, "train")
    val_metrics = summarize_split_metrics(metric_rows, "val")
    test_metrics = summarize_split_metrics(metric_rows, "test")
    train_loss_decreased = bool(
        final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss
    )
    can_fit_train_samples = bool(
        train_loss_decreased
        and train_metrics.get("train_dice_mean", 0.0) > 0.75
        and train_metrics.get("train_iou_mean", 0.0) > 0.55
    )
    full_area = masks.shape[1] * masks.shape[2]
    geometry_mask_ious = np.array(validation["geometry_mask_ious"], dtype=np.float64)
    context = {
        "npz_readable": True,
        "schema_complete": len(validation["missing"]) == 0,
        "split_is_80_20_20": {name: len(indices) for name, indices in splits.items()} == {"train": 80, "val": 20, "test": 20},
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "defect_type_distribution": {name: int(np.sum(defect_types == name)) for name in sorted(set(defect_types.tolist()))},
        "delta_matches": validation["delta_matches"],
        "scan_lines_different": validation["max_line_diff"] > 1e-12,
        "sample_ids_unique": len(set(sample_ids.tolist())) == len(sample_ids),
        "geometry_mask_ious_summary": {
            "min": float(np.min(geometry_mask_ious)),
            "max": float(np.max(geometry_mask_ious)),
            "mean": float(np.mean(geometry_mask_ious)),
        },
        "split_geometry_ranges": pilot["split_geometry_ranges"],
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
        "has_empty_prediction": any(int(row["pred_area_zero"]) == 1 for row in metric_rows),
        "has_full_prediction": any(int(row["pred_area"]) >= full_area for row in metric_rows),
        "has_nan": any(not np.isfinite(float(row["total_loss"])) for row in metric_rows),
        "preview_generated": True,
        "preview_dir": str(preview_dir),
        "preview_sample_ids": [f"sample_{index + 1:03d}" for index in selected_preview_indices],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(context), encoding="utf-8-sig")
    print(json.dumps(context, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

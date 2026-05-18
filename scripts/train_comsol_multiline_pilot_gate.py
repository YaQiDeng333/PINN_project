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
    / "comsol_single_defect_multiline_forward_pack_v1_pilot.npz"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "summaries"
    / "comsol_multiline_pilot_training_gate_summary.txt"
)
DEFAULT_METRICS = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_multiline_pilot_training_gate_metrics.csv"
)
DEFAULT_EPOCH_LOG = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_multiline_pilot_training_gate_epoch_log.csv"
)
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results" / "previews" / "comsol_multiline_pilot_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded pilot training gate on the COMSOL multi-line pilot pack."
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def split_indices_from_npz(data: np.lib.npyio.NpzFile) -> tuple[dict[str, list[int]], str]:
    split_source = "split" if "split" in data.files else "suggested_split" if "suggested_split" in data.files else ""
    if split_source:
        split_values = np.array([tiny.as_text(item) for item in data[split_source].tolist()])
        result = {
            "train": np.where(split_values == "train")[0].tolist(),
            "val": np.where(split_values == "val")[0].tolist(),
            "test": np.where(split_values == "test")[0].tolist(),
        }
        if result["train"] and result["val"] and result["test"]:
            return result, split_source
    sample_count = data["delta_bz"].shape[0]
    if sample_count != 36:
        raise RuntimeError("fallback pilot split expects exactly 36 samples")
    return {"train": list(range(24)), "val": list(range(24, 30)), "test": list(range(30, 36))}, "sample_order_fallback"


def validate_pilot_npz(npz_path: Path) -> dict[str, Any]:
    validation = tiny.validate_npz(npz_path)
    data = validation["data"]
    splits, split_source = split_indices_from_npz(data)
    expected_counts = {"train": 24, "val": 6, "test": 6}
    split_counts = {name: len(indices) for name, indices in splits.items()}
    if split_counts != expected_counts:
        raise RuntimeError(f"unexpected pilot split counts: {split_counts}")
    defect_types = [tiny.as_text(item) for item in data["defect_types"].tolist()]
    if set(defect_types) != {"rectangular_notch"}:
        raise RuntimeError(f"unexpected defect types: {sorted(set(defect_types))}")
    return {"validation": validation, "splits": splits, "split_source": split_source}


def choose_preview_indices(rows: list[dict[str, Any]]) -> list[int]:
    by_sample = {row["sample_id"]: row for row in rows}
    selected: list[int] = []
    # With 6 val + 6 test, prioritize all validation and test samples as requested.
    for row in rows:
        if row["split"] in {"val", "test"}:
            selected.append(int(row["sample_id"].split("_")[-1]) - 1)
    if len(selected) >= 12:
        return selected[:12]
    train_rows = [row for row in rows if row["split"] == "train"]
    train_sorted = sorted(train_rows, key=lambda item: float(item["dice"]))
    candidates = train_sorted[:3] + train_sorted[-3:]
    for row in candidates:
        index = int(row["sample_id"].split("_")[-1]) - 1
        if index not in selected:
            selected.append(index)
        if len(selected) >= 12:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(context: dict[str, Any]) -> str:
    lines = [
        "# 第 20.11 COMSOL multiline pilot training gate",
        "",
        "## 1. 数据读取与 schema",
        "",
        f"- pilot NPZ 是否可读：{context['npz_readable']}",
        f"- schema 是否完整：{context['schema_complete']}",
        f"- delta_bz 输入 shape：{context['delta_bz_shape']}",
        f"- masks shape：{context['masks_shape']}",
        f"- split 构造：{context['split_summary']}",
        f"- split 来源：{context['split_source']}",
        f"- defect_type 分布：{context['defect_type_distribution']}",
        f"- pilot limitation：当前全部为 rectangular_notch，尚不包含 rotated_rect / polygon / multi_defect。",
        f"- delta_bz 是否等于 bz_defect - bz_no_defect：{context['delta_matches']}",
        f"- 三条 scan line 是否不同：{context['scan_lines_different']}",
        f"- geometry_params 是否解释 mask：{context['geometry_mask_ious_summary']}",
        "",
        "## 2. pilot model / train loop",
        "",
        "- 模型：轻量 Conv1d Bz encoder 处理 `(3, 201)`，latent 投影到 `4x8` feature map，再用 ConvTranspose2d 上采样到 `(64, 128)` mask logits。",
        "- loss：BCEWithLogits + soft Dice。",
        f"- epochs：{context['epochs']}",
        f"- batch_size：{context['batch_size']}",
        f"- selected threshold：{context['threshold']}",
        f"- best validation epoch：{context['best_epoch']}",
        f"- train loop 是否跑通：{context['train_loop_ok']}",
        f"- train loss 是否下降：{context['train_loss_decreased']}，initial={context['initial_train_loss']:.6f}, final={context['final_train_loss']:.6f}",
        f"- 是否能 overfit 24 个 train samples：{context['can_overfit_train_samples']}",
        "",
        "## 3. pilot smoke metrics",
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
        f"- preview 样本：{context['preview_sample_ids']}",
        "",
        "## 5. 结论",
        "",
        "- 该结果只能说明 COMSOL pilot pack 的读取、dataset loader、normalization、训练/验证/测试循环、validation selection、threshold selection 和 preview 链路可以跑通。",
        "- 36 个 rectangular_notch 样本可以支持下一阶段 pilot training gate，但仍不足以得出正式泛化或 v3_complex 反演性能结论。",
        "- 当前限制是样本数仍小、defect_type 单一、只包含 rectangular_notch、几何范围有限、val/test split 各 6 个样本。",
        "- 下一步应优先扩展 COMSOL 样本到 100+，并增加 rotated_rect / polygon 等 defect_type；loader/schema 当前没有阻塞。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200 for this pilot gate.")
    tiny.set_seed(args.seed)

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    preview_dir = resolve(args.preview_dir)

    pilot = validate_pilot_npz(npz_path)
    validation = pilot["validation"]
    data = validation["data"]
    splits = pilot["splits"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
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
            total, bce, dice = tiny.loss_components(logits, target)
            total.backward()
            optimizer.step()
            batch_totals.append(float(total.item()))
            batch_bces.append(float(bce.item()))
            batch_dices.append(float(dice.item()))

        train_loss = float(np.mean(batch_totals))
        train_bce = float(np.mean(batch_bces))
        train_dice = float(np.mean(batch_dices))
        val_loss, val_bce, val_dice = tiny.evaluate_loss(model, val_dataset, device)
        val_rows, _ = tiny.evaluate_model(model, val_dataset, device, 0.5, sample_ids, defect_types)
        val_iou = float(np.mean([row["iou"] for row in val_rows]))
        val_dice_metric = float(np.mean([row["dice"] for row in val_rows]))
        val_area_error = float(np.mean([row["area_error"] for row in val_rows]))
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
    selected_threshold, threshold_scores = tiny.select_threshold(model, val_dataset, device, candidates)

    metric_rows: list[dict[str, Any]] = []
    all_probs: dict[int, np.ndarray] = {}
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        rows, probs = tiny.evaluate_model(model, dataset, device, selected_threshold, sample_ids, defect_types)
        for row in rows:
            row["split"] = split_name
        metric_rows.extend(rows)
        all_probs.update(probs)

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

    train_metrics = tiny.aggregate(metric_rows, "train")
    val_metrics = tiny.aggregate(metric_rows, "val")
    test_metrics = tiny.aggregate(metric_rows, "test")
    train_loss_decreased = bool(final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss)
    can_overfit_train_samples = bool(
        train_loss_decreased
        and train_metrics.get("train_dice_mean", 0.0) > 0.75
        and train_metrics.get("train_iou_mean", 0.0) > 0.55
    )
    full_area = masks.shape[1] * masks.shape[2]
    context = {
        "npz_readable": True,
        "schema_complete": len(validation["missing"]) == 0,
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "split_summary": {name: [tiny.as_text(sample_ids[index]) for index in indices] for name, indices in splits.items()},
        "split_source": pilot["split_source"],
        "defect_type_distribution": {name: int(np.sum(defect_types == name)) for name in sorted(set(defect_types.tolist()))},
        "delta_matches": validation["delta_matches"],
        "scan_lines_different": validation["max_line_diff"] > 1e-12,
        "geometry_mask_ious_summary": {
            "min": float(np.min(validation["geometry_mask_ious"])),
            "max": float(np.max(validation["geometry_mask_ious"])),
            "mean": float(np.mean(validation["geometry_mask_ious"])),
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "threshold": selected_threshold,
        "best_epoch": best_epoch,
        "train_loop_ok": True,
        "train_loss_decreased": train_loss_decreased,
        "initial_train_loss": float(initial_train_loss if initial_train_loss is not None else float("nan")),
        "final_train_loss": float(final_train_loss if final_train_loss is not None else float("nan")),
        "can_overfit_train_samples": can_overfit_train_samples,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "has_empty_prediction": any(int(row["pred_area_zero"]) == 1 for row in metric_rows),
        "has_full_prediction": any(int(row["pred_area"]) >= full_area for row in metric_rows),
        "has_nan": any(not np.isfinite(float(row["total_loss"])) for row in metric_rows),
        "preview_generated": True,
        "preview_dir": str(preview_dir),
        "preview_sample_ids": [f"sample_{index + 1:03d}" for index in selected_preview_indices],
        "threshold_scores": threshold_scores,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(context), encoding="utf-8-sig")
    print(json.dumps(context, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

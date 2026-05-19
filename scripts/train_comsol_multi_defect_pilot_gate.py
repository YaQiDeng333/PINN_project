from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_multiline_tiny_smoke as tiny  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_multi_defect_multiline_forward_pack_v1_pilot.npz"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multi_defect_pilot_training_gate_summary.txt"
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multi_defect_pilot_failure_audit_summary.txt"
DEFAULT_PLAN = PROJECT_ROOT / "results/summaries/comsol_multi_defect_pilot_v2_expansion_plan.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_training_gate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_training_gate_epoch_log.csv"
DEFAULT_COMBO_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_component_combination_summary.csv"
DEFAULT_CONNECTED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_connected_component_summary.csv"
DEFAULT_FAILURE_CASES = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_failure_audit_cases.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_multi_defect_pilot_gate"

THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPECTED_SPLITS = {"train": 16, "val": 4, "test": 4}
EXPECTED_COMBOS = {
    "rectangular_notch+rectangular_notch",
    "rectangular_notch+rotated_rect",
    "rotated_rect+rotated_rect",
}

METRIC_FIELDS = [
    "source_index",
    "sample_id",
    "split",
    "component_types",
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
    "true_connected_component_count",
    "pred_connected_component_count",
    "component_count_error",
    "predicted_component_count_is_2",
    "missed_component_flag",
    "merged_component_flag",
    "split_component_flag",
    "largest_component_area_ratio",
    "second_largest_component_area_ratio",
    "component_recall_heuristic",
    "notes",
]

EPOCH_FIELDS = tiny.EPOCH_FIELDS + [
    "best_val_threshold",
    "best_val_iou",
    "best_val_dice",
    "best_val_area_error",
    "best_val_score",
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
    "pred_connected_component_count_mean",
    "predicted_component_count_is_2_rate",
    "missed_component_rate",
    "merged_component_rate",
    "split_component_rate",
    "component_recall_mean",
]

FAILURE_FIELDS = [
    "sample_id",
    "split",
    "component_types",
    "true_connected_component_count",
    "pred_connected_component_count",
    "IoU",
    "Dice",
    "area_error",
    "center_error",
    "pred_area",
    "true_area",
    "failure_category",
    "short_note",
    "preview_path",
]


class MultiDefectValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COMSOL true multi_defect pilot ingest + training gate.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--expansion-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--component-summary", type=Path, default=DEFAULT_COMBO_SUMMARY)
    parser.add_argument("--connected-summary", type=Path, default=DEFAULT_CONNECTED_SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE_CASES)
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


def as_json(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return json.loads(tiny.as_text(value))


def connected_components(mask: np.ndarray) -> tuple[int, list[int], np.ndarray]:
    binary = mask.astype(bool)
    labels = np.zeros(binary.shape, dtype=np.int32)
    areas: list[int] = []
    label = 0
    h, w = binary.shape
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label
                        stack.append((ny, nx))
            areas.append(area)
    return label, areas, labels


def rasterize_component(component: dict[str, Any], mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
    dx = xx - float(component["center_x_m"])
    dy = yy - float(component["center_y_m"])
    angle = float(component.get("angle_rad", math.radians(float(component.get("angle_deg", 0.0)))))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    mask = (np.abs(local_x) <= float(component["width_m"]) / 2.0) & (
        np.abs(local_y) <= float(component["length_m"]) / 2.0
    )
    return mask.astype(np.uint8)


def component_recall(pred: np.ndarray, components: list[dict[str, Any]], mask_x: np.ndarray, mask_y: np.ndarray) -> float:
    hits = 0
    for component in components:
        component_mask = rasterize_component(component, mask_x, mask_y).astype(bool)
        area = int(component_mask.sum())
        if area == 0:
            continue
        overlap_fraction = float(np.logical_and(pred, component_mask).sum() / area)
        if overlap_fraction >= 0.20:
            hits += 1
    return float(hits / max(1, len(components)))


def validate_multi_defect_npz(npz_path: Path) -> dict[str, Any]:
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
        "components_json",
        "component_counts",
        "component_types",
        "split",
        "metadata",
    ]
    data = np.load(npz_path, allow_pickle=True)
    missing = [field for field in required if field not in data.files]
    if missing:
        raise MultiDefectValidationError(f"missing NPZ fields: {missing}")

    delta_bz = data["delta_bz"]
    bz_defect = data["bz_defect"]
    bz_no_defect = data["bz_no_defect"]
    masks = data["masks"]
    sensor_x = data["sensor_x"]
    scan_line_y = data["scan_line_y"]
    mask_x = data["mask_x"]
    mask_y = data["mask_y"]
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    splits = np.array([tiny.as_text(item) for item in data["split"].tolist()])
    component_counts = data["component_counts"].astype(np.int64)
    components = [as_json(item) for item in data["components_json"].tolist()]
    component_types = [as_json(item) for item in data["component_types"].tolist()]

    if delta_bz.shape != (24, 3, 201):
        raise MultiDefectValidationError(f"unexpected delta_bz shape: {delta_bz.shape}")
    if bz_defect.shape != delta_bz.shape or bz_no_defect.shape != delta_bz.shape:
        raise MultiDefectValidationError("bz_defect / bz_no_defect shape mismatch")
    if masks.shape != (24, 64, 128):
        raise MultiDefectValidationError(f"unexpected masks shape: {masks.shape}")
    if sensor_x.shape != (201,) or scan_line_y.shape != (3,) or mask_x.shape != (128,) or mask_y.shape != (64,):
        raise MultiDefectValidationError("coordinate shape mismatch")
    if not (np.all(np.diff(sensor_x) > 0) and np.all(np.diff(scan_line_y) > 0) and np.all(np.diff(mask_x) > 0) and np.all(np.diff(mask_y) > 0)):
        raise MultiDefectValidationError("coordinates are not strictly increasing")
    finite = all(np.isfinite(array).all() for array in (delta_bz, bz_defect, bz_no_defect, masks, sensor_x, scan_line_y, mask_x, mask_y))
    if not finite:
        raise MultiDefectValidationError("non-finite arrays found")
    if not np.allclose(delta_bz, bz_defect - bz_no_defect, rtol=1e-9, atol=1e-12):
        raise MultiDefectValidationError("delta_bz mismatch")
    if float(delta_bz.std()) <= 0.0:
        raise MultiDefectValidationError("delta_bz is zero")
    if np.max(np.abs(delta_bz[:, 1:, :] - delta_bz[:, :1, :])) <= 1e-12:
        raise MultiDefectValidationError("scan lines are identical")
    if not np.all(masks.reshape(masks.shape[0], -1).sum(axis=1) > 0):
        raise MultiDefectValidationError("empty mask found")
    if set(defect_types.tolist()) != {"multi_defect"}:
        raise MultiDefectValidationError(f"unexpected defect_types: {Counter(defect_types.tolist())}")
    if not np.all(component_counts == 2):
        raise MultiDefectValidationError(f"component_counts not all 2: {Counter(component_counts.tolist())}")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise MultiDefectValidationError("sample_ids are not unique")
    split_counts = Counter(splits.tolist())
    if dict(split_counts) != EXPECTED_SPLITS:
        raise MultiDefectValidationError(f"unexpected split counts: {dict(split_counts)}")

    true_cc: list[int] = []
    geometry_mask_ious: list[float] = []
    component_type_combos: list[str] = []
    min_distances: list[float] = []
    for index, component_list in enumerate(components):
        if not isinstance(component_list, list) or len(component_list) != 2:
            raise MultiDefectValidationError(f"components_json index {index} is not a 2-component list")
        combo = "+".join(tiny.as_text(component["component_type"]) for component in component_list)
        component_type_combos.append(combo)
        if combo not in EXPECTED_COMBOS:
            raise MultiDefectValidationError(f"unexpected component combo: {combo}")
        union = np.zeros(masks[index].shape, dtype=bool)
        centers = []
        for component in component_list:
            component_mask = rasterize_component(component, mask_x, mask_y).astype(bool)
            if int(component_mask.sum()) <= 0:
                raise MultiDefectValidationError(f"empty component mask at sample {index}")
            union |= component_mask
            centers.append((float(component["center_x_m"]), float(component["center_y_m"])))
        stored = masks[index].astype(bool)
        denom = int(np.logical_or(union, stored).sum())
        geometry_mask_ious.append(1.0 if denom == 0 else float(np.logical_and(union, stored).sum() / denom))
        cc, _, _ = connected_components(stored)
        true_cc.append(cc)
        min_distances.append(float(math.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1])))
    if min(geometry_mask_ious) < 0.999:
        raise MultiDefectValidationError(f"components_json does not explain masks: min IoU {min(geometry_mask_ious)}")
    if set(true_cc) != {2}:
        raise MultiDefectValidationError(f"true connected component count is not all 2: {Counter(true_cc)}")

    return {
        "data": data,
        "sample_ids": sample_ids,
        "splits_array": splits,
        "splits": {name: np.where(splits == name)[0].tolist() for name in ("train", "val", "test")},
        "components": components,
        "component_types_raw": component_types,
        "component_type_combos": np.array(component_type_combos),
        "true_connected_counts": np.array(true_cc),
        "geometry_mask_ious": geometry_mask_ious,
        "min_distances": np.array(min_distances),
        "split_counts": dict(split_counts),
        "combo_counts": dict(Counter(component_type_combos)),
    }


def extended_sample_metrics(
    prob: np.ndarray,
    target: np.ndarray,
    threshold: float,
    components: list[dict[str, Any]],
    mask_x: np.ndarray,
    mask_y: np.ndarray,
) -> dict[str, Any]:
    base = tiny.sample_metrics(prob, target, threshold)
    pred = prob >= threshold
    true = target >= 0.5
    true_cc, _, _ = connected_components(true)
    pred_cc, pred_areas, _ = connected_components(pred)
    sorted_areas = sorted(pred_areas, reverse=True)
    pred_area = max(1, int(pred.sum()))
    largest_ratio = float(sorted_areas[0] / pred_area) if sorted_areas else 0.0
    second_ratio = float(sorted_areas[1] / pred_area) if len(sorted_areas) > 1 else 0.0
    recall = component_recall(pred, components, mask_x, mask_y)
    merged = int(pred_cc == 1 and recall >= 1.0)
    missed = int(recall < 1.0)
    split = int(pred_cc > true_cc)
    return {
        **base,
        "true_connected_component_count": true_cc,
        "pred_connected_component_count": pred_cc,
        "component_count_error": abs(pred_cc - true_cc),
        "predicted_component_count_is_2": int(pred_cc == 2),
        "missed_component_flag": missed,
        "merged_component_flag": merged,
        "split_component_flag": split,
        "largest_component_area_ratio": largest_ratio,
        "second_largest_component_area_ratio": second_ratio,
        "component_recall_heuristic": recall,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    threshold: float,
    sample_ids: np.ndarray,
    splits_array: np.ndarray,
    component_combos: np.ndarray,
    components: list[list[dict[str, Any]]],
    mask_x: np.ndarray,
    mask_y: np.ndarray,
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
            metrics = extended_sample_metrics(prob, target, threshold, components[index], mask_x, mask_y)
            rows.append(
                {
                    "source_index": index,
                    "sample_id": tiny.as_text(sample_ids[index]),
                    "split": tiny.as_text(splits_array[index]),
                    "component_types": tiny.as_text(component_combos[index]),
                    "threshold": threshold,
                    **metrics,
                    "bce_loss": float(bce.item()),
                    "dice_loss": float(dice.item()),
                    "total_loss": float(total.item()),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                    "notes": "multi_defect_pilot_gate_only",
                }
            )
    return rows, probs


def select_threshold_for_epoch(model: torch.nn.Module, val_dataset: tiny.ComsolSmokeDataset, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        signals, masks, _ = next(iter(DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)))
        prob = torch.sigmoid(model(signals.to(device))).cpu().numpy()
        target = masks.numpy()
    best = {"threshold": THRESHOLD_CANDIDATES[0], "iou": 0.0, "dice": 0.0, "area_error": float("inf"), "score": -float("inf")}
    for threshold in THRESHOLD_CANDIDATES:
        metrics = [tiny.sample_metrics(prob[index], target[index], threshold) for index in range(prob.shape[0])]
        iou = float(np.mean([row["iou"] for row in metrics]))
        dice = float(np.mean([row["dice"] for row in metrics]))
        area_error = float(np.mean([row["area_error"] for row in metrics]))
        score = iou + dice - area_error
        if score > best["score"]:
            best = {"threshold": threshold, "iou": iou, "dice": dice, "area_error": area_error, "score": score}
    return best


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
        "pred_connected_component_count_mean": mean_or_nan([float(row["pred_connected_component_count"]) for row in rows]),
        "predicted_component_count_is_2_rate": mean_or_nan([float(row["predicted_component_count_is_2"]) for row in rows]),
        "missed_component_rate": mean_or_nan([float(row["missed_component_flag"]) for row in rows]),
        "merged_component_rate": mean_or_nan([float(row["merged_component_flag"]) for row in rows]),
        "split_component_rate": mean_or_nan([float(row["split_component_flag"]) for row in rows]),
        "component_recall_mean": mean_or_nan([float(row["component_recall_heuristic"]) for row in rows]),
    }


def group_summary(metric_rows: list[dict[str, Any]], key: str, values: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        for value in values:
            selected = [row for row in source if row[key] == value]
            rows.append(summarize_rows(selected, f"{key}={value}", split_name))
    return rows


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected:
            selected.append(index)

    val_test = [row for row in metric_rows if row["split"] in {"val", "test"}]
    all_rows = metric_rows
    for combo in sorted(EXPECTED_COMBOS):
        rows = [row for row in val_test if row["component_types"] == combo]
        if rows:
            add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
            add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for pred_cc in (1, 2, 3):
        rows = [row for row in all_rows if int(row["pred_connected_component_count"]) == pred_cc]
        if rows:
            add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for reverse in (True, False):
        for row in sorted(val_test, key=lambda item: float(item["dice"]), reverse=reverse)[:6]:
            add(int(row["source_index"]))
    return selected[:12]


def make_previews(
    preview_dir: Path,
    probs: dict[int, np.ndarray],
    masks: np.ndarray,
    signals: np.ndarray,
    sensor_x: np.ndarray,
    scan_line_y: np.ndarray,
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows_by_index = {int(row["source_index"]): row for row in rows}
    paths: dict[str, str] = {}
    for index, prob in probs.items():
        row = rows_by_index[index]
        pred = prob >= threshold
        true = masks[index] >= 0.5
        overlay = np.zeros((*true.shape, 3), dtype=np.float32)
        overlay[..., 1] = true.astype(np.float32)
        overlay[..., 0] = pred.astype(np.float32)
        _, _, pred_labels = connected_components(pred)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for line_index, y_value in enumerate(scan_line_y):
            axes[0, 0].plot(sensor_x, signals[index, line_index], label=f"y={y_value:.4g} m")
        axes[0, 0].set_title("delta_bz scan lines")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].imshow(true, cmap="gray")
        axes[0, 1].set_title("true union mask")
        axes[0, 2].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[0, 2].set_title("predicted probability")
        axes[1, 0].imshow(pred, cmap="gray")
        axes[1, 0].set_title(f"pred mask @ {threshold:.2f}")
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("overlay red=pred green=true")
        axes[1, 2].imshow(pred_labels, cmap="tab20", vmin=0)
        axes[1, 2].set_title(f"pred cc={row['pred_connected_component_count']}")
        axes[1, 2].text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"id: {row['sample_id']}",
                    f"split: {row['split']}",
                    f"types: {row['component_types']}",
                    f"IoU: {float(row['iou']):.4f}",
                    f"Dice: {float(row['dice']):.4f}",
                    f"area_error: {float(row['area_error']):.4f}",
                ]
            ),
            va="top",
            transform=axes[1, 2].transAxes,
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
        )
        for ax in axes.flat:
            if ax is not axes[0, 0]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        path = preview_dir / f"{row['sample_id']}_{row['split']}_{row['component_types'].replace('+', '_')}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths[tiny.as_text(row["sample_id"])] = str(path)
    return paths


def failure_category(row: dict[str, Any]) -> tuple[str, str]:
    if int(row["pred_area_zero"]) == 1:
        return "empty prediction", "predicted mask is empty."
    if int(row["missed_component_flag"]) == 1:
        return "missed second component", "component recall heuristic is below 1.0."
    if int(row["merged_component_flag"]) == 1:
        return "component merge", "prediction has one connected component but overlaps both true components."
    if int(row["split_component_flag"]) == 1:
        return "extra fragments", "prediction has more connected components than the true union mask."
    if float(row["area_error"]) > 0.35:
        return "area error", "predicted union mask area is biased."
    if not math.isnan(float(row["center_error"])) and float(row["center_error"]) > 5.0:
        return "localization", "center error is high."
    return "boundary/shape smoothing", "mask is non-empty but boundary or union shape is imprecise."


def build_failure_cases(metric_rows: list[dict[str, Any]], preview_paths: dict[str, str]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(metric_rows, key=lambda item: float(item["dice"]))[:10]:
        selected[tiny.as_text(row["sample_id"])] = row
    for row in sorted(metric_rows, key=lambda item: float(item["area_error"]), reverse=True)[:10]:
        selected[tiny.as_text(row["sample_id"])] = row
    for row in sorted(metric_rows, key=lambda item: float(item["component_count_error"]), reverse=True)[:10]:
        selected[tiny.as_text(row["sample_id"])] = row
    for combo in sorted(EXPECTED_COMBOS):
        rows = [row for row in metric_rows if row["component_types"] == combo]
        if rows:
            selected[tiny.as_text(min(rows, key=lambda item: float(item["dice"]))["sample_id"])] = min(rows, key=lambda item: float(item["dice"]))
    output: list[dict[str, Any]] = []
    for row in sorted(selected.values(), key=lambda item: (item["split"], float(item["dice"]))):
        category, note = failure_category(row)
        output.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "component_types": row["component_types"],
                "true_connected_component_count": row["true_connected_component_count"],
                "pred_connected_component_count": row["pred_connected_component_count"],
                "IoU": row["iou"],
                "Dice": row["dice"],
                "area_error": row["area_error"],
                "center_error": row["center_error"],
                "pred_area": row["pred_area"],
                "true_area": row["true_area"],
                "failure_category": category,
                "short_note": note,
                "preview_path": preview_paths.get(tiny.as_text(row["sample_id"]), ""),
            }
        )
    return output


def split_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: summarize_rows([row for row in metric_rows if row["split"] == split_name], "all", split_name)
        for split_name in ("train", "val", "test")
    }


def build_training_summary(context: dict[str, Any]) -> str:
    lines = [
        "# COMSOL true multi_defect pilot training gate",
        "",
        "## Schema",
        "",
        f"- multi_defect NPZ readable: {context['npz_readable']}",
        f"- schema complete: {context['schema_complete']}",
        f"- split is 16 / 4 / 4: {context['split_ok']} ({context['split_counts']})",
        f"- component_count all 2: {context['component_count_all_2']}",
        f"- true connected_component_count all 2: {context['true_cc_all_2']}",
        f"- component_type combination distribution: {context['combo_counts']}",
        f"- delta_bz input shape: {context['delta_bz_shape']}",
        f"- mask output shape: {context['masks_shape']}",
        f"- components_json/mask IoU min/mean/max: {context['geometry_iou_summary']}",
        "",
        "## Normalization And Model",
        "",
        "- Normalization: per-channel delta_bz mean/std computed only from train split.",
        f"- train mean shape: {context['train_mean_shape']}",
        f"- train std shape: {context['train_std_shape']}",
        "- Model: lightweight mask-only Conv1d encoder for `(3, 201)` delta_bz and ConvTranspose2d decoder to `(64, 128)` union-mask logits.",
        "- Loss: BCEWithLogits + soft Dice. No component_count, component_type, geometry, or bz_defect/bz_no_defect input.",
        "",
        "## Training Gate",
        "",
        f"- seed: {context['seed']}",
        f"- epochs: {context['epochs']}",
        f"- batch_size: {context['batch_size']}",
        "- Checkpoint selection: each epoch scans validation thresholds and uses best validation IoU + Dice - area_error.",
        f"- selected threshold: {context['threshold']}",
        f"- best epoch: {context['best_epoch']}",
        f"- train loop ok: {context['train_loop_ok']}",
        f"- train loss decreased: {context['train_loss_decreased']} (initial={context['initial_train_loss']:.6f}, final={context['final_train_loss']:.6f})",
        f"- can fit 16 train samples: {context['can_fit_train_samples']}",
        "",
        "## Pilot Metrics",
        "",
        f"- train: {context['train_metrics']}",
        f"- val: {context['val_metrics']}",
        f"- test: {context['test_metrics']}",
        f"- predicted connected component count summary: {context['connected_summary_path']}",
        f"- component combination summary: {context['component_summary_path']}",
        f"- missed_component / merged_component / extra_fragment: {context['component_error_summary']}",
        f"- hardest component_type combination: {context['hardest_combo']}",
        f"- empty predictions: {context['has_empty_prediction']}",
        f"- full-image predictions: {context['has_full_prediction']}",
        f"- NaN detected: {context['has_nan']}",
        "",
        "## Preview And Audit",
        "",
        f"- preview generated: {context['preview_generated']}",
        f"- preview dir: {context['preview_dir']}",
        f"- preview sample ids: {context['preview_sample_ids']}",
        f"- failure audit summary: {context['audit_summary_path']}",
        f"- failure cases CSV: {context['failure_cases_path']}",
        f"- failure audit main conclusion: {context['failure_main']}",
        "",
        "## Conclusion",
        "",
        "- This is a multi_defect pilot gate only, not a baseline and not a v3_complex comparison.",
        f"- recommend expanding multi_defect data: {context['recommend_expand']}",
        f"- recommended next step: {context['next_step']}",
    ]
    return "\n".join(lines) + "\n"


def build_audit_summary(context: dict[str, Any]) -> str:
    lines = [
        "# COMSOL true multi_defect pilot failure audit",
        "",
        f"- model predicts non-empty union masks: {not context['has_empty_prediction']}",
        f"- missed second component count: {context['missed_count']}",
        f"- merged one-blob count: {context['merged_count']}",
        f"- extra fragments count: {context['split_count']}",
        f"- hardest component combination: {context['hardest_combo']}",
        f"- dominant failure category: {context['dominant_failure_category']}",
        f"- schema / mask / components_json issue found: {context['schema_issue_found']}",
        f"- preview dir: {context['preview_dir']}",
        "",
        "## Interpretation",
        "",
        context["interpretation"],
        "",
        "## Recommendation",
        "",
        context["recommendation"],
    ]
    return "\n".join(lines) + "\n"


def build_expansion_plan() -> str:
    lines = [
        "# COMSOL multi_defect pilot_v2 expansion plan",
        "",
        "- Recommended scale: 120 samples as the next bounded pilot_v2; use 80 train / 20 val / 20 test.",
        "- Component count: keep `component_count=2` for pilot_v2 before attempting 3+ components.",
        "- Component combinations: balance rectangular_notch+rectangular_notch, rectangular_notch+rotated_rect, rotated_rect+rotated_rect, and add a small rectangular_notch+polygon group only if COMSOL polygon components remain stable.",
        "- Polygon components: introduce cautiously after a 1-sample smoke and a 12-sample mini-pack; do not block pilot_v2 on polygon if it destabilizes geometry.",
        "- Component distance: explicitly stratify near / medium / far center distances, while preventing touching or overlapping masks.",
        "- Relative size/depth: include balanced small+large, shallow+deep, and same-size pairs so one component is not always dominant.",
        "- Split design: stratify by component combination, distance bin, size ratio, and depth ratio; validation and test must contain every major combination.",
        "- Metrics to keep: predicted connected component count, missed component flag, merge flag, extra-fragment flag, largest/second component area ratios, component recall heuristic, and per-combination summaries.",
        "- Single-defect mixing: do not merge with single-defect data until the multi_defect-only pilot_v2 ingest/training gate passes.",
        "- Acceptance condition: schema complete, train-only normalization, no empty predictions, connected-component diagnostics are meaningful, and val/test both contain all planned combinations.",
        "- Stop condition: COMSOL geometry instability, components_json cannot explain masks, or systematic second-component miss on val/test.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200.")
    tiny.set_seed(args.seed)

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    audit_summary_path = resolve(args.audit_summary)
    plan_path = resolve(args.expansion_plan)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    component_summary_path = resolve(args.component_summary)
    connected_summary_path = resolve(args.connected_summary)
    failure_cases_path = resolve(args.failure_cases)
    preview_dir = resolve(args.preview_dir)

    validation = validate_multi_defect_npz(npz_path)
    data = validation["data"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = validation["sample_ids"]
    splits_array = validation["splits_array"]
    splits = validation["splits"]
    components = validation["components"]
    component_combos = validation["component_type_combos"]
    mask_x = data["mask_x"].astype(np.float64)
    mask_y = data["mask_y"].astype(np.float64)
    sensor_x = data["sensor_x"].astype(np.float64)
    scan_line_y = data["scan_line_y"].astype(np.float64)

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

    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
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
        threshold_result = select_threshold_for_epoch(model, val_dataset, device)
        if threshold_result["score"] > best_score:
            best_score = float(threshold_result["score"])
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
                "val_iou_at_0_5": float("nan"),
                "val_dice_at_0_5": float("nan"),
                "val_area_error_at_0_5": float("nan"),
                "val_score_at_0_5": float("nan"),
                "best_val_threshold": threshold_result["threshold"],
                "best_val_iou": threshold_result["iou"],
                "best_val_dice": threshold_result["dice"],
                "best_val_area_error": threshold_result["area_error"],
                "best_val_score": threshold_result["score"],
            }
        )

    model.load_state_dict(best_state)
    selected_threshold, threshold_scores = tiny.select_threshold(model, val_dataset, device, THRESHOLD_CANDIDATES)
    metric_rows: list[dict[str, Any]] = []
    for dataset in (train_dataset, val_dataset, test_dataset):
        rows, _ = evaluate_model(
            model, dataset, device, selected_threshold, sample_ids, splits_array, component_combos, components, mask_x, mask_y
        )
        metric_rows.extend(rows)
    all_rows, all_probs = evaluate_model(
        model, all_dataset, device, selected_threshold, sample_ids, splits_array, component_combos, components, mask_x, mask_y
    )

    component_rows = group_summary(metric_rows, "component_types", sorted(EXPECTED_COMBOS))
    connected_values = sorted({int(row["pred_connected_component_count"]) for row in metric_rows})
    connected_rows = group_summary(metric_rows, "pred_connected_component_count", connected_values)
    selected_preview_indices = choose_preview_indices(metric_rows)
    selected_probs = {index: all_probs[index] for index in selected_preview_indices}
    preview_paths = make_previews(preview_dir, selected_probs, masks, delta_bz, sensor_x, scan_line_y, all_rows, selected_threshold)
    failure_rows = build_failure_cases(metric_rows, preview_paths)

    train_loss_decreased = bool(final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss)
    split_summaries = split_metrics(metric_rows)
    can_fit_train_samples = bool(
        train_loss_decreased
        and float(split_summaries["train"]["dice_mean"]) > 0.70
        and float(split_summaries["train"]["iou_mean"]) > 0.50
    )
    has_empty = any(int(row["pred_area_zero"]) for row in metric_rows)
    has_full = any(float(row["pred_area"]) > 0.95 * masks.shape[1] * masks.shape[2] for row in metric_rows)
    has_nan = any(not np.isfinite(float(row["total_loss"])) for row in metric_rows)
    missed_count = sum(int(row["missed_component_flag"]) for row in metric_rows)
    merged_count = sum(int(row["merged_component_flag"]) for row in metric_rows)
    split_count = sum(int(row["split_component_flag"]) for row in metric_rows)
    combo_all_rows = [row for row in component_rows if row["split"] == "all" and int(row["sample_count"]) > 0]
    hardest_combo = min(combo_all_rows, key=lambda row: float(row["dice_mean"]))["group"] if combo_all_rows else ""
    category_counts = Counter(row["failure_category"] for row in failure_rows)
    dominant_failure = category_counts.most_common(1)[0][0] if category_counts else ""
    component_error_summary = {
        "missed_component_count": missed_count,
        "merged_component_count": merged_count,
        "extra_fragment_count": split_count,
    }
    failure_main = (
        "The pilot chain works, but with 24 samples the dominant errors are "
        f"{dominant_failure}; component diagnostics should be treated as pilot-only."
    )
    recommend_expand = not has_nan and not has_empty

    write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    write_csv(epoch_log_path, epoch_rows, EPOCH_FIELDS)
    write_csv(component_summary_path, component_rows, GROUP_FIELDS)
    write_csv(connected_summary_path, connected_rows, GROUP_FIELDS)
    write_csv(failure_cases_path, failure_rows, FAILURE_FIELDS)

    common_context = {
        "npz_readable": True,
        "schema_complete": True,
        "split_ok": validation["split_counts"] == EXPECTED_SPLITS,
        "split_counts": validation["split_counts"],
        "component_count_all_2": True,
        "true_cc_all_2": set(validation["true_connected_counts"].tolist()) == {2},
        "combo_counts": validation["combo_counts"],
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "geometry_iou_summary": [
            float(min(validation["geometry_mask_ious"])),
            float(np.mean(validation["geometry_mask_ious"])),
            float(max(validation["geometry_mask_ious"])),
        ],
        "train_mean_shape": tuple(train_mean.shape),
        "train_std_shape": tuple(train_std.shape),
        "seed": args.seed,
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
        "train_metrics": split_summaries["train"],
        "val_metrics": split_summaries["val"],
        "test_metrics": split_summaries["test"],
        "connected_summary_path": str(connected_summary_path),
        "component_summary_path": str(component_summary_path),
        "component_error_summary": component_error_summary,
        "hardest_combo": hardest_combo,
        "has_empty_prediction": has_empty,
        "has_full_prediction": has_full,
        "has_nan": has_nan,
        "preview_generated": True,
        "preview_dir": str(preview_dir),
        "preview_sample_ids": [tiny.as_text(sample_ids[index]) for index in selected_preview_indices],
        "audit_summary_path": str(audit_summary_path),
        "failure_cases_path": str(failure_cases_path),
        "failure_main": failure_main,
        "recommend_expand": recommend_expand,
        "next_step": "Expand multi_defect pilot_v2 to 120 samples before changing model capacity.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_training_summary(common_context), encoding="utf-8")
    audit_context = {
        "has_empty_prediction": has_empty,
        "missed_count": missed_count,
        "merged_count": merged_count,
        "split_count": split_count,
        "hardest_combo": hardest_combo,
        "dominant_failure_category": dominant_failure,
        "schema_issue_found": False,
        "preview_dir": str(preview_dir),
        "interpretation": failure_main,
        "recommendation": "Proceed with a bounded multi_defect pilot_v2 expansion; do not update any baseline from this 24-sample pilot.",
    }
    audit_summary_path.parent.mkdir(parents=True, exist_ok=True)
    audit_summary_path.write_text(build_audit_summary(audit_context), encoding="utf-8")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(build_expansion_plan(), encoding="utf-8")

    print(
        json.dumps(
            {
                "selected_threshold": selected_threshold,
                "best_epoch": best_epoch,
                "train": split_summaries["train"],
                "val": split_summaries["val"],
                "test": split_summaries["test"],
                "missed_count": missed_count,
                "merged_count": merged_count,
                "extra_fragment_count": split_count,
                "hardest_combo": hardest_combo,
                "summary": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

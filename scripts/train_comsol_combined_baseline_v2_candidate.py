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
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_multiline_tiny_smoke as tiny  # noqa: E402
import train_comsol_multi_defect_pilot_gate as pilot_v1  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_combined_single_multi_defect_baseline_v2_candidate.npz"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_combined_baseline_v2_training_summary.txt"
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_combined_baseline_v2_failure_audit_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_epoch_log.csv"
DEFAULT_SEED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_seed_summary.csv"
DEFAULT_DEFECT_GROUP_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_defect_group_summary.csv"
DEFAULT_DEFECT_TYPE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_defect_type_summary.csv"
DEFAULT_CONNECTED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_connected_component_summary.csv"
DEFAULT_COMPONENT_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_component_combination_summary.csv"
DEFAULT_SOURCE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_source_summary.csv"
DEFAULT_FAILURE_CASES = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_failure_cases.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_combined_baseline_v2_candidate"

SEEDS = [42, 123, 2026]
EXPECTED_SPLITS = {"train": 562, "val": 139, "test": 139}
THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SINGLE_BASELINE_TEST = {"iou": 0.6515, "dice": 0.7861}
MULTI_BASELINE_TEST = {"iou": 0.6118, "dice": 0.7573, "pred_cc_is_2": 1.0}
SUBSTANTIAL_DROP = 0.03
_BASE_RASTERIZE_COMPONENT = pilot_v1.rasterize_component

METRIC_FIELDS = [
    "seed",
    "source_index",
    "sample_id",
    "split",
    "defect_group",
    "defect_type",
    "component_types",
    "component_type_combination",
    "source_dataset",
    "source_pack",
    "angle_deg",
    "vertex_count",
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
    "predicted_component_count_correct",
    "predicted_component_count_is_1",
    "predicted_component_count_is_2",
    "component_count_error",
    "missed_component_flag",
    "merged_component_flag",
    "split_component_flag",
    "largest_component_area_ratio",
    "second_largest_component_area_ratio",
    "component_recall_heuristic",
    "polygon_component_present",
    "notes",
]

EPOCH_FIELDS = [
    "seed",
    *tiny.EPOCH_FIELDS,
    "best_val_threshold",
    "best_val_iou",
    "best_val_dice",
    "best_val_area_error",
    "best_val_score",
]

SUMMARY_FIELDS = [
    "seed",
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
    "predicted_component_count_correct_rate",
    "predicted_component_count_is_1_rate",
    "predicted_component_count_is_2_rate",
    "missed_component_rate",
    "merged_component_rate",
    "split_component_rate",
    "component_recall_mean",
]

SEED_SUMMARY_FIELDS = [
    "seed",
    "split",
    "best_epoch",
    "selected_threshold",
    "best_val_score",
    "initial_train_loss",
    "final_train_loss",
    "train_loss_decreased",
    "sample_count",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_mean",
    "pred_area_zero_sum",
    "total_loss_mean",
    "pred_connected_component_count_mean",
    "predicted_component_count_correct_rate",
    "predicted_component_count_is_1_rate",
    "predicted_component_count_is_2_rate",
    "missed_component_rate",
    "merged_component_rate",
    "split_component_rate",
    "component_recall_mean",
]

FAILURE_FIELDS = [
    "seed",
    "sample_id",
    "split",
    "defect_group",
    "defect_type",
    "component_type_combination",
    "source_dataset",
    "source_pack",
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


class CombinedTrainingError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train COMSOL combined single + multi-defect baseline v2 candidate.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--seed-summary", type=Path, default=DEFAULT_SEED_SUMMARY)
    parser.add_argument("--defect-group-summary", type=Path, default=DEFAULT_DEFECT_GROUP_SUMMARY)
    parser.add_argument("--defect-type-summary", type=Path, default=DEFAULT_DEFECT_TYPE_SUMMARY)
    parser.add_argument("--connected-summary", type=Path, default=DEFAULT_CONNECTED_SUMMARY)
    parser.add_argument("--component-summary", type=Path, default=DEFAULT_COMPONENT_SUMMARY)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE_CASES)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return str(value)


def parse_json(value: Any) -> Any:
    return json.loads(as_text(value))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rasterize_component(component: dict[str, Any], mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    if as_text(component.get("component_type")) != "polygon":
        return _BASE_RASTERIZE_COMPONENT(component, mask_x, mask_y)
    vertices = np.asarray(component.get("polygon_vertices", []), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        return np.zeros((mask_y.shape[0], mask_x.shape[0]), dtype=np.uint8)
    yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
    inside = np.zeros(xx.shape, dtype=bool)
    xj = vertices[-1, 0]
    yj = vertices[-1, 1]
    for xi, yi in vertices:
        crosses = ((yi > yy) != (yj > yy)) & (xx < (xj - xi) * (yy - yi) / ((yj - yi) + 1e-30) + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside.astype(np.uint8)


def connected_components(mask: np.ndarray) -> tuple[int, list[int], np.ndarray]:
    return pilot_v1.connected_components(mask)


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
    return {
        **base,
        "true_connected_component_count": true_cc,
        "pred_connected_component_count": pred_cc,
        "predicted_component_count_correct": int(pred_cc == true_cc),
        "predicted_component_count_is_1": int(pred_cc == 1),
        "predicted_component_count_is_2": int(pred_cc == 2),
        "component_count_error": abs(pred_cc - true_cc),
        "missed_component_flag": int(recall < 1.0),
        "merged_component_flag": int(true_cc >= 2 and pred_cc == 1 and recall >= 1.0),
        "split_component_flag": int(pred_cc > true_cc),
        "largest_component_area_ratio": largest_ratio,
        "second_largest_component_area_ratio": second_ratio,
        "component_recall_heuristic": recall,
    }


def validate_npz(npz_path: Path) -> dict[str, Any]:
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
        "defect_group",
        "sample_ids",
        "split",
        "geometry_params",
        "components_json",
        "component_counts",
        "component_types",
        "connected_component_counts",
        "source_dataset",
        "metadata",
    ]
    data = np.load(npz_path, allow_pickle=True)
    missing = [field for field in required if field not in data.files]
    if missing:
        raise CombinedTrainingError(f"missing fields: {missing}")
    delta = data["delta_bz"]
    masks = data["masks"]
    if delta.shape != (840, 3, 201):
        raise CombinedTrainingError(f"unexpected delta_bz shape: {delta.shape}")
    if data["bz_defect"].shape != delta.shape or data["bz_no_defect"].shape != delta.shape:
        raise CombinedTrainingError("bz_defect / bz_no_defect shape mismatch")
    if masks.shape != (840, 64, 128):
        raise CombinedTrainingError(f"unexpected masks shape: {masks.shape}")
    for name, shape in {"sensor_x": (201,), "scan_line_y": (3,), "mask_x": (128,), "mask_y": (64,)}.items():
        if data[name].shape != shape:
            raise CombinedTrainingError(f"{name} shape mismatch: {data[name].shape}")
        if not np.all(np.diff(data[name]) > 0):
            raise CombinedTrainingError(f"{name} is not strictly increasing")
    if not all(np.isfinite(data[name]).all() for name in ("delta_bz", "bz_defect", "bz_no_defect", "masks")):
        raise CombinedTrainingError("non-finite arrays")
    if not np.allclose(delta, data["bz_defect"] - data["bz_no_defect"], rtol=1e-9, atol=1e-12):
        raise CombinedTrainingError("delta_bz mismatch")
    if not np.all(masks.reshape(masks.shape[0], -1).sum(axis=1) > 0):
        raise CombinedTrainingError("empty mask found")
    sample_ids = np.array([as_text(item) for item in data["sample_ids"].tolist()])
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise CombinedTrainingError("sample_ids are not unique")
    splits = np.array([as_text(item) for item in data["split"].tolist()])
    split_counts = dict(Counter(splits.tolist()))
    if split_counts != EXPECTED_SPLITS:
        raise CombinedTrainingError(f"unexpected split counts: {split_counts}")
    defect_group = np.array([as_text(item) for item in data["defect_group"].tolist()])
    defect_types = np.array([as_text(item) for item in data["defect_types"].tolist()])
    source_dataset = np.array([as_text(item) for item in data["source_dataset"].tolist()])
    component_counts = data["component_counts"].astype(np.int64)
    connected_counts = data["connected_component_counts"].astype(np.int64)
    components = [parse_json(item) for item in data["components_json"].tolist()]
    component_types_json = [parse_json(item) for item in data["component_types"].tolist()]
    geometries = [parse_json(item) for item in data["geometry_params"].tolist()]
    if dict(Counter(defect_group.tolist())) != {"single_defect": 600, "multi_defect": 240}:
        raise CombinedTrainingError(f"unexpected defect_group distribution: {Counter(defect_group.tolist())}")
    multi_idx = np.where(defect_group == "multi_defect")[0]
    if not np.all(component_counts[multi_idx] == 2) or not np.all(connected_counts[multi_idx] == 2):
        raise CombinedTrainingError("multi_defect component/connected counts are not all 2")
    component_combos = np.array(["+".join(as_text(value) for value in values) for values in component_types_json])
    source_pack: list[str] = []
    angles: list[Any] = []
    vertex_counts: list[Any] = []
    polygon_present: list[str] = []
    for geometry, combo in zip(geometries, component_combos):
        source_pack.append(as_text(geometry.get("source_pack", geometry.get("source_dataset", ""))))
        angles.append(geometry.get("angle_deg", geometry.get("angle", "")))
        vertex_counts.append(geometry.get("vertex_count", ""))
        polygon_present.append("yes" if "polygon" in combo else "no")
    return {
        "data": data,
        "sample_ids": sample_ids,
        "splits_array": splits,
        "splits": {name: np.where(splits == name)[0].tolist() for name in ("train", "val", "test")},
        "defect_group": defect_group,
        "defect_types": defect_types,
        "source_dataset": source_dataset,
        "source_pack": np.array(source_pack),
        "components": components,
        "component_combos": component_combos,
        "component_counts": component_counts,
        "connected_counts": connected_counts,
        "angles": np.array(angles, dtype=object),
        "vertex_counts": np.array(vertex_counts, dtype=object),
        "polygon_component_present": np.array(polygon_present),
        "split_counts": split_counts,
        "defect_group_counts": dict(Counter(defect_group.tolist())),
        "defect_type_counts": dict(Counter(defect_types.tolist())),
        "component_combo_counts": dict(Counter(component_combos.tolist())),
    }


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


def evaluate_model(
    model: torch.nn.Module,
    dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    threshold: float,
    validation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    data = validation["data"]
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
            metrics = extended_sample_metrics(
                prob,
                target,
                threshold,
                validation["components"][index],
                data["mask_x"].astype(np.float64),
                data["mask_y"].astype(np.float64),
            )
            rows.append(
                {
                    "source_index": index,
                    "sample_id": as_text(validation["sample_ids"][index]),
                    "split": as_text(validation["splits_array"][index]),
                    "defect_group": as_text(validation["defect_group"][index]),
                    "defect_type": as_text(validation["defect_types"][index]),
                    "component_types": as_text(validation["component_combos"][index]),
                    "component_type_combination": as_text(validation["component_combos"][index]),
                    "source_dataset": as_text(validation["source_dataset"][index]),
                    "source_pack": as_text(validation["source_pack"][index]),
                    "angle_deg": as_text(validation["angles"][index]),
                    "vertex_count": as_text(validation["vertex_counts"][index]),
                    "threshold": threshold,
                    **metrics,
                    "bce_loss": float(bce.item()),
                    "dice_loss": float(dice.item()),
                    "total_loss": float(total.item()),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                    "polygon_component_present": as_text(validation["polygon_component_present"][index]),
                    "notes": "combined_baseline_v2_candidate_eval",
                }
            )
    return rows, probs


def train_one_seed(seed: int, args: argparse.Namespace, validation: dict[str, Any], device: torch.device) -> dict[str, Any]:
    tiny.set_seed(seed)
    data = validation["data"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    splits = validation["splits"]
    train_mean = delta_bz[splits["train"]].mean(axis=(0, 2), keepdims=True)
    train_std = np.maximum(delta_bz[splits["train"]].std(axis=(0, 2), keepdims=True), 1e-8)
    normalized = (delta_bz - train_mean) / train_std
    train_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["train"])
    val_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["val"])
    all_dataset = tiny.ComsolSmokeDataset(normalized, masks, list(range(delta_bz.shape[0])))
    model = tiny.TinyComsolMaskDecoder(delta_bz.shape[1], masks.shape[1], masks.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_totals: list[float] = []
        batch_bces: list[float] = []
        batch_dices: list[float] = []
        for signals, targets, _ in train_loader:
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            total, bce, dice = tiny.loss_components(logits, targets)
            total.backward()
            optimizer.step()
            batch_totals.append(float(total.item()))
            batch_bces.append(float(bce.item()))
            batch_dices.append(float(dice.item()))
        train_loss = {
            "total_loss": float(np.mean(batch_totals)),
            "bce_loss": float(np.mean(batch_bces)),
            "dice_loss": float(np.mean(batch_dices)),
        }
        val_total, val_bce, val_dice = tiny.evaluate_loss(model, val_dataset, device)
        best_val = select_threshold_for_epoch(model, val_dataset, device)
        epoch_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss["total_loss"],
                "train_bce": train_loss["bce_loss"],
                "train_dice_loss": train_loss["dice_loss"],
                "val_loss": float(val_total),
                "val_bce": float(val_bce),
                "val_dice_loss": float(val_dice),
                "best_val_threshold": best_val["threshold"],
                "best_val_iou": best_val["iou"],
                "best_val_dice": best_val["dice"],
                "best_val_area_error": best_val["area_error"],
                "best_val_score": best_val["score"],
            }
        )
        if best_val["score"] > best_score:
            best_score = best_val["score"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected = select_threshold_for_epoch(model, val_dataset, device)
    metric_rows, all_probs = evaluate_model(model, all_dataset, device, float(selected["threshold"]), validation)
    for row in metric_rows:
        row["seed"] = seed
    preview_probs = {}
    if seed == SEEDS[0]:
        preview_indices = choose_preview_indices(metric_rows)
        preview_probs = {index: all_probs[index] for index in preview_indices if index in all_probs}
    split_summaries = {
        split_name: summarize([row for row in metric_rows if row["split"] == split_name], "all", split_name, seed)
        for split_name in ("train", "val", "test")
    }
    return {
        "seed": seed,
        "model": model,
        "threshold": float(selected["threshold"]),
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "epoch_rows": epoch_rows,
        "metric_rows": metric_rows,
        "selected_probs": preview_probs,
        "split_summaries": split_summaries,
        "initial_train_loss": float(epoch_rows[0]["train_loss"]),
        "final_train_loss": float(epoch_rows[-1]["train_loss"]),
        "train_loss_decreased": float(epoch_rows[-1]["train_loss"]) < float(epoch_rows[0]["train_loss"]),
        "train_mean_shape": tuple(train_mean.shape),
        "train_std_shape": tuple(train_std.shape),
    }


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def summarize(rows: list[dict[str, Any]], group: str, split: str, seed: Any = "all") -> dict[str, Any]:
    return {
        "seed": seed,
        "group": group,
        "split": split,
        "sample_count": len(rows),
        "iou_mean": mean_or_nan([float(row["iou"]) for row in rows]),
        "dice_mean": mean_or_nan([float(row["dice"]) for row in rows]),
        "area_error_mean": mean_or_nan([float(row["area_error"]) for row in rows]),
        "center_error_mean": mean_or_nan([float(row["center_error"]) for row in rows if as_text(row["center_error"]).lower() != "nan"]),
        "pred_area_mean": mean_or_nan([float(row["pred_area"]) for row in rows]),
        "true_area_mean": mean_or_nan([float(row["true_area"]) for row in rows]),
        "pred_area_zero_sum": int(sum(int(row["pred_area_zero"]) for row in rows)),
        "total_loss_mean": mean_or_nan([float(row["total_loss"]) for row in rows]),
        "pred_connected_component_count_mean": mean_or_nan([float(row["pred_connected_component_count"]) for row in rows]),
        "predicted_component_count_correct_rate": mean_or_nan([float(row["predicted_component_count_correct"]) for row in rows]),
        "predicted_component_count_is_1_rate": mean_or_nan([float(row["predicted_component_count_is_1"]) for row in rows]),
        "predicted_component_count_is_2_rate": mean_or_nan([float(row["predicted_component_count_is_2"]) for row in rows]),
        "missed_component_rate": mean_or_nan([float(row["missed_component_flag"]) for row in rows]),
        "merged_component_rate": mean_or_nan([float(row["merged_component_flag"]) for row in rows]),
        "split_component_rate": mean_or_nan([float(row["split_component_flag"]) for row in rows]),
        "component_recall_mean": mean_or_nan([float(row["component_recall_heuristic"]) for row in rows]),
    }


def grouped_summary(metric_rows: list[dict[str, Any]], key: str, values: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        for value in values:
            selected = [row for row in source if row[key] == value]
            rows.append(summarize(selected, f"{key}={value}", split_name))
    return rows


def mean_std(seed_results: list[dict[str, Any]], split: str, field: str) -> tuple[float, float]:
    values = [float(result["split_summaries"][split][field]) for result in seed_results]
    return float(np.mean(values)), float(np.std(values, ddof=0))


def format_mean_std(seed_results: list[dict[str, Any]], split: str) -> str:
    parts = []
    for field, label in (
        ("iou_mean", "IoU"),
        ("dice_mean", "Dice"),
        ("area_error_mean", "area_error"),
        ("center_error_mean", "center_error"),
        ("predicted_component_count_correct_rate", "pred_cc_correct"),
        ("predicted_component_count_is_2_rate", "pred_cc_is_2"),
        ("missed_component_rate", "missed"),
        ("merged_component_rate", "merged"),
        ("split_component_rate", "split"),
    ):
        mean, std = mean_std(seed_results, split, field)
        parts.append(f"{label}={mean:.4f}+/-{std:.4f}")
    return ", ".join(parts)


def seed_summary_rows(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in seed_results:
        for split_name in ("train", "val", "test"):
            summary = result["split_summaries"][split_name]
            rows.append(
                {
                    "seed": result["seed"],
                    "split": split_name,
                    "best_epoch": result["best_epoch"],
                    "selected_threshold": result["threshold"],
                    "best_val_score": result["best_val_score"],
                    "initial_train_loss": result["initial_train_loss"],
                    "final_train_loss": result["final_train_loss"],
                    "train_loss_decreased": result["train_loss_decreased"],
                    **{key: value for key, value in summary.items() if key not in {"seed", "group", "split"}},
                }
            )
    return rows


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected:
            selected.append(index)

    val_test = [row for row in metric_rows if row["split"] in {"val", "test"}]
    for key, values in (
        ("defect_group", ["single_defect", "multi_defect"]),
        ("defect_type", ["rectangular_notch", "rotated_rect", "polygon", "multi_defect"]),
    ):
        for value in values:
            rows = [row for row in val_test if row[key] == value]
            if rows:
                add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
                add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    cc_mismatch = [row for row in val_test if int(row["predicted_component_count_correct"]) == 0]
    for row in sorted(cc_mismatch, key=lambda item: float(item["dice"]))[:8]:
        add(int(row["source_index"]))
    for reverse in (True, False):
        for row in sorted(val_test, key=lambda item: float(item["dice"]), reverse=reverse)[:12]:
            add(int(row["source_index"]))
    return selected[:32]


def make_previews(
    preview_dir: Path,
    probs: dict[int, np.ndarray],
    masks: np.ndarray,
    delta_bz: np.ndarray,
    sensor_x: np.ndarray,
    scan_line_y: np.ndarray,
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, str]:
    return pilot_v1.make_previews(preview_dir, probs, masks, delta_bz, sensor_x, scan_line_y, rows, threshold)


def failure_category(row: dict[str, Any]) -> tuple[str, str]:
    if int(row["pred_area_zero"]):
        return "empty_prediction", "predicted area is zero"
    if int(row["predicted_component_count_correct"]) == 0:
        return "component_count_mismatch", f"pred_cc={row['pred_connected_component_count']}, true_cc={row['true_connected_component_count']}"
    if float(row["area_error"]) > 0.35:
        return "area_error", "large area mismatch"
    if float(row["center_error"]) > 8.0:
        return "localization", "large center error"
    return "boundary/shape smoothing", "mask area and component count are plausible but boundary remains coarse"


def build_failure_cases(metric_rows: list[dict[str, Any]], preview_paths: dict[str, str]) -> list[dict[str, Any]]:
    seed_rows = [row for row in metric_rows if int(row["seed"]) == SEEDS[0] and row["split"] in {"val", "test"}]
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any]) -> None:
        if not any(existing["sample_id"] == row["sample_id"] for existing in selected):
            selected.append(row)

    for row in sorted(seed_rows, key=lambda item: float(item["dice"]))[:12]:
        add(row)
    for row in sorted(seed_rows, key=lambda item: float(item["area_error"]), reverse=True)[:8]:
        add(row)
    for row in sorted(seed_rows, key=lambda item: float(item["center_error"]), reverse=True)[:8]:
        add(row)
    for row in [row for row in seed_rows if int(row["predicted_component_count_correct"]) == 0][:8]:
        add(row)
    cases: list[dict[str, Any]] = []
    for row in selected[:32]:
        category, note = failure_category(row)
        cases.append(
            {
                "seed": row["seed"],
                "sample_id": row["sample_id"],
                "split": row["split"],
                "defect_group": row["defect_group"],
                "defect_type": row["defect_type"],
                "component_type_combination": row["component_type_combination"],
                "source_dataset": row["source_dataset"],
                "source_pack": row["source_pack"],
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
                "preview_path": preview_paths.get(row["sample_id"], ""),
            }
        )
    return cases


def summary_lookup(rows: list[dict[str, Any]], group: str, split: str) -> dict[str, Any]:
    matches = [row for row in rows if row["group"] == group and row["split"] == split]
    return matches[0] if matches else summarize([], group, split)


def build_training_summary(context: dict[str, Any]) -> str:
    lines = [
        "# COMSOL combined single + multi-defect baseline v2 candidate training summary",
        "",
        f"- combined NPZ readable: {context['npz_readable']}",
        f"- schema complete: {context['schema_complete']}",
        f"- split distribution: {context['split_counts']}",
        f"- input shape: {context['input_shape']}",
        f"- output shape: {context['output_shape']}",
        f"- normalization train-only: {context['normalization_train_only']}",
        f"- model: mask-only Conv1d/BzEncoder + grid decoder, delta_bz input only",
        f"- seeds completed: {context['seeds_completed']}",
        f"- best epoch / threshold per seed: {context['seed_best']}",
        f"- train mean+/-std: {context['train_mean_std']}",
        f"- val mean+/-std: {context['val_mean_std']}",
        f"- test mean+/-std: {context['test_mean_std']}",
        "",
        "## Group Metrics",
        "",
        f"- single_defect test: {context['single_test']}",
        f"- multi_defect test: {context['multi_test']}",
        f"- defect_type test summary: {context['defect_type_test']}",
        f"- multi_defect connected-component behavior: {context['multi_cc_test']}",
        f"- missed / merged / split behavior: {context['multi_error_test']}",
        f"- single_defect substantial degradation vs COMSOL_DATA_BASELINE: {context['single_degraded']}",
        f"- multi_defect substantial degradation vs COMSOL_MULTI_DEFECT_DATA_BASELINE: {context['multi_degraded']}",
        f"- preview generated: {context['preview_generated']}",
        f"- preview dir: {context['preview_dir']}",
        f"- should document as COMSOL_DATA_BASELINE_V2: {context['document_baseline_v2']}",
        "",
        "## Limitations",
        "",
        "- COMSOL data-domain only; not v3_complex and not CURRENT_BASELINE.",
        "- Controlled synthetic pilot data only.",
        "- Multi_defect component_count remains fixed to 2.",
        "- No real experimental data.",
    ]
    return "\n".join(lines) + "\n"


def build_audit_summary(context: dict[str, Any]) -> str:
    lines = [
        "# COMSOL combined baseline v2 candidate failure audit",
        "",
        f"- single_defect degradation vs COMSOL_DATA_BASELINE: {context['single_degraded']}",
        f"- multi_defect degradation vs COMSOL_MULTI_DEFECT_DATA_BASELINE: {context['multi_degraded']}",
        f"- model still predicts 2 components for multi_defect: {context['multi_pred_cc_is_2_ok']}",
        f"- hardest group: {context['hardest_group']}",
        f"- dominant failure mode: {context['dominant_failure']}",
        f"- source_dataset/source_pack issue: {context['source_issue']}",
        f"- combined baseline stable enough to document: {context['document_baseline_v2']}",
        "",
        "## Interpretation",
        "",
        context["interpretation"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    paths = {
        "npz": resolve(args.npz),
        "summary": resolve(args.summary),
        "audit_summary": resolve(args.audit_summary),
        "metrics": resolve(args.metrics),
        "epoch_log": resolve(args.epoch_log),
        "seed_summary": resolve(args.seed_summary),
        "defect_group_summary": resolve(args.defect_group_summary),
        "defect_type_summary": resolve(args.defect_type_summary),
        "connected_summary": resolve(args.connected_summary),
        "component_summary": resolve(args.component_summary),
        "source_summary": resolve(args.source_summary),
        "failure_cases": resolve(args.failure_cases),
        "preview_dir": resolve(args.preview_dir),
    }
    validation = validate_npz(paths["npz"])
    pilot_v1.rasterize_component = rasterize_component
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = [train_one_seed(seed, args, validation, device) for seed in SEEDS]
    all_metric_rows = [row for result in seed_results for row in result["metric_rows"]]
    all_epoch_rows = [row for result in seed_results for row in result["epoch_rows"]]

    defect_group_rows = grouped_summary(all_metric_rows, "defect_group", ["single_defect", "multi_defect"])
    defect_type_rows = grouped_summary(all_metric_rows, "defect_type", ["rectangular_notch", "rotated_rect", "polygon", "multi_defect"])
    connected_rows = grouped_summary(
        all_metric_rows,
        "pred_connected_component_count",
        sorted({int(row["pred_connected_component_count"]) for row in all_metric_rows}),
    )
    component_rows = grouped_summary(
        all_metric_rows,
        "component_type_combination",
        sorted({row["component_type_combination"] for row in all_metric_rows}),
    )
    component_rows.extend(grouped_summary(all_metric_rows, "polygon_component_present", ["no", "yes"]))
    source_rows = grouped_summary(all_metric_rows, "source_dataset", sorted(set(row["source_dataset"] for row in all_metric_rows)))
    source_rows.extend(grouped_summary(all_metric_rows, "source_pack", sorted(set(row["source_pack"] for row in all_metric_rows))))

    preview_result = seed_results[0]
    preview_indices = choose_preview_indices(preview_result["metric_rows"])
    preview_probs = {index: preview_result["selected_probs"].get(index) for index in preview_indices if index in preview_result["selected_probs"]}
    preview_rows = [row for row in preview_result["metric_rows"] if int(row["source_index"]) in preview_probs]
    data = validation["data"]
    preview_paths = make_previews(
        paths["preview_dir"],
        preview_probs,
        data["masks"].astype(np.float32),
        data["delta_bz"].astype(np.float32),
        data["sensor_x"].astype(np.float64),
        data["scan_line_y"].astype(np.float64),
        preview_rows,
        preview_result["threshold"],
    )
    failure_cases = build_failure_cases(all_metric_rows, preview_paths)

    write_csv(paths["metrics"], all_metric_rows, METRIC_FIELDS)
    write_csv(paths["epoch_log"], all_epoch_rows, EPOCH_FIELDS)
    write_csv(paths["seed_summary"], seed_summary_rows(seed_results), SEED_SUMMARY_FIELDS)
    write_csv(paths["defect_group_summary"], defect_group_rows, SUMMARY_FIELDS)
    write_csv(paths["defect_type_summary"], defect_type_rows, SUMMARY_FIELDS)
    write_csv(paths["connected_summary"], connected_rows, SUMMARY_FIELDS)
    write_csv(paths["component_summary"], component_rows, SUMMARY_FIELDS)
    write_csv(paths["source_summary"], source_rows, SUMMARY_FIELDS)
    write_csv(paths["failure_cases"], failure_cases, FAILURE_FIELDS)

    single_test = summary_lookup(defect_group_rows, "defect_group=single_defect", "test")
    multi_test = summary_lookup(defect_group_rows, "defect_group=multi_defect", "test")
    single_degraded = (
        SINGLE_BASELINE_TEST["iou"] - float(single_test["iou_mean"]) > SUBSTANTIAL_DROP
        or SINGLE_BASELINE_TEST["dice"] - float(single_test["dice_mean"]) > SUBSTANTIAL_DROP
    )
    multi_degraded = (
        MULTI_BASELINE_TEST["iou"] - float(multi_test["iou_mean"]) > SUBSTANTIAL_DROP
        or MULTI_BASELINE_TEST["dice"] - float(multi_test["dice_mean"]) > SUBSTANTIAL_DROP
    )
    multi_rows = [row for row in all_metric_rows if row["split"] == "test" and row["defect_group"] == "multi_defect"]
    multi_pred_cc2 = float(np.mean([float(row["predicted_component_count_is_2"]) for row in multi_rows])) if multi_rows else float("nan")
    multi_errors = {
        "missed": float(np.mean([float(row["missed_component_flag"]) for row in multi_rows])) if multi_rows else float("nan"),
        "merged": float(np.mean([float(row["merged_component_flag"]) for row in multi_rows])) if multi_rows else float("nan"),
        "split": float(np.mean([float(row["split_component_flag"]) for row in multi_rows])) if multi_rows else float("nan"),
    }
    test_group_rows = [row for row in [*defect_type_rows, *component_rows] if row["split"] == "test" and int(row["sample_count"]) > 0]
    hardest = min(test_group_rows, key=lambda row: float(row["dice_mean"])) if test_group_rows else {"group": "n/a", "dice_mean": "nan"}
    failure_counts = Counter(row["failure_category"] for row in failure_cases)
    dominant_failure = failure_counts.most_common(1)[0][0] if failure_counts else "none"
    source_test_rows = [row for row in source_rows if row["split"] == "test" and int(row["sample_count"]) > 0]
    source_dices = [float(row["dice_mean"]) for row in source_test_rows]
    source_issue = bool(source_dices and max(source_dices) - min(source_dices) > 0.08)
    document_baseline_v2 = not single_degraded and not multi_degraded and multi_pred_cc2 >= 0.95
    context = {
        "npz_readable": True,
        "schema_complete": True,
        "split_counts": validation["split_counts"],
        "input_shape": tuple(data["delta_bz"].shape),
        "output_shape": tuple(data["masks"].shape),
        "normalization_train_only": True,
        "seeds_completed": len(seed_results) == len(SEEDS),
        "seed_best": {
            result["seed"]: {
                "best_epoch": result["best_epoch"],
                "selected_threshold": result["threshold"],
                "best_val_score": result["best_val_score"],
            }
            for result in seed_results
        },
        "train_mean_std": format_mean_std(seed_results, "train"),
        "val_mean_std": format_mean_std(seed_results, "val"),
        "test_mean_std": format_mean_std(seed_results, "test"),
        "single_test": {key: single_test[key] for key in ("iou_mean", "dice_mean", "area_error_mean", "sample_count")},
        "multi_test": {key: multi_test[key] for key in ("iou_mean", "dice_mean", "area_error_mean", "sample_count")},
        "defect_type_test": {
            row["group"]: {"iou": row["iou_mean"], "dice": row["dice_mean"], "n": row["sample_count"]}
            for row in defect_type_rows
            if row["split"] == "test"
        },
        "multi_cc_test": {"pred_cc_is_2": multi_pred_cc2, "pred_cc_correct": float(np.mean([float(row["predicted_component_count_correct"]) for row in multi_rows])) if multi_rows else float("nan")},
        "multi_error_test": multi_errors,
        "single_degraded": single_degraded,
        "multi_degraded": multi_degraded,
        "preview_generated": bool(preview_paths),
        "preview_dir": str(paths["preview_dir"]),
        "document_baseline_v2": document_baseline_v2,
        "multi_pred_cc_is_2_ok": multi_pred_cc2 >= 0.95,
        "hardest_group": f"{hardest['group']} dice={float(hardest['dice_mean']):.4f}" if hardest["group"] != "n/a" else "n/a",
        "dominant_failure": dominant_failure,
        "source_issue": source_issue,
        "interpretation": (
            "Combined training is acceptable as a baseline v2 candidate because it preserves multi_defect component behavior "
            "and does not show substantial degradation against the single-defect or multi_defect data-domain baselines."
            if document_baseline_v2
            else "Combined training should not be documented as baseline v2 yet because one or more degradation/component criteria failed."
        ),
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["audit_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(build_training_summary(context), encoding="utf-8")
    paths["audit_summary"].write_text(build_audit_summary(context), encoding="utf-8")
    print(
        json.dumps(
            {
                "seeds_completed": context["seeds_completed"],
                "seed_best": context["seed_best"],
                "train": context["train_mean_std"],
                "val": context["val_mean_std"],
                "test": context["test_mean_std"],
                "single_test": context["single_test"],
                "multi_test": context["multi_test"],
                "multi_cc_test": context["multi_cc_test"],
                "document_baseline_v2": context["document_baseline_v2"],
                "summary": str(paths["summary"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

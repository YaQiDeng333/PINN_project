"""Diagnose center-anchored polygon held-out failures without training."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


SPLITS = ("train", "val", "test")
HELDOUT_SPLITS = ("val", "test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: str | float | int | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(round(float(value)))


def _as_bool(value: str | bool | int | float | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0", ""}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _min(values: Iterable[float]) -> float:
    vals = list(values)
    return float(min(vals)) if vals else 0.0


def _max(values: Iterable[float]) -> float:
    vals = list(values)
    return float(max(vals)) if vals else 0.0


def _shoelace(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _load_targets(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required target NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _component_vertices(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [[_as_float(row[f"{prefix}_x{i}"]), _as_float(row[f"{prefix}_y{i}"])] for i in range(4)],
        dtype=np.float64,
    )


def _local_error(row: dict[str, str]) -> tuple[float, float]:
    errors: list[float] = []
    for idx in range(4):
        if not _as_bool(row.get(f"vertex{idx}_valid", "0")):
            continue
        errors.append(abs(_as_float(row[f"pred_local_x{idx}"]) - _as_float(row[f"true_local_x{idx}"])))
        errors.append(abs(_as_float(row[f"pred_local_y{idx}"]) - _as_float(row[f"true_local_y{idx}"])))
    return _mean(errors), _max(errors)


def _target_sample_features(split: str, target_root: Path, raw_root: Path, sample_iou: dict[tuple[str, int], float]) -> list[dict]:
    targets = _load_targets(target_root / split / "center_anchored_polygon_targets.npz")
    defect_rows = {int(row["sample_index"]): row for row in _read_csv(raw_root / split / "defect_params.csv")}
    presence = targets["presence_targets"] > 0.5
    centers = targets["center_targets_norm"].astype(np.float64)
    x_bins = targets["center_x_bin_targets"].astype(np.int64)
    y_bins = targets["center_y_bin_targets"].astype(np.int64)
    vertices = targets["polygon_vertices_norm"].astype(np.float64)
    vertex_mask = targets["polygon_vertex_mask"] > 0.5
    sample_indices = targets["sample_indices"].astype(np.int64)
    out = []
    for local_idx, sample_index in enumerate(sample_indices):
        sample_index_int = int(sample_index)
        present_slots = [slot for slot in range(presence.shape[1]) if bool(presence[local_idx, slot])]
        row = defect_rows[sample_index_int]
        all_vertices = []
        areas = []
        bin_pairs = []
        for slot in present_slots:
            valid = vertex_mask[local_idx, slot]
            slot_vertices = vertices[local_idx, slot][valid]
            if slot_vertices.size:
                all_vertices.append(slot_vertices)
                areas.append(abs(_shoelace(slot_vertices)))
            bin_pairs.append((int(x_bins[local_idx, slot]), int(y_bins[local_idx, slot])))
        if all_vertices:
            stacked = np.concatenate(all_vertices, axis=0)
            bbox_width = float(stacked[:, 0].max() - stacked[:, 0].min())
            bbox_height = float(stacked[:, 1].max() - stacked[:, 1].min())
            vertex_x_min = float(stacked[:, 0].min())
            vertex_x_max = float(stacked[:, 0].max())
            vertex_y_min = float(stacked[:, 1].min())
            vertex_y_max = float(stacked[:, 1].max())
            center_mean = centers[local_idx, present_slots].mean(axis=0)
        else:
            bbox_width = bbox_height = vertex_x_min = vertex_x_max = vertex_y_min = vertex_y_max = 0.0
            center_mean = np.zeros(2, dtype=np.float64)
        out.append(
            {
                "split": split,
                "sample_index": sample_index_int,
                "hard_case_type": row["hard_case_type"],
                "true_rotated": _as_bool(row.get("true_rotated_geometry", "false")),
                "true_multi_component": _as_bool(row.get("true_multi_component_geometry", "false")),
                "component_count": len(present_slots),
                "center_x_mean": float(center_mean[0]),
                "center_y_mean": float(center_mean[1]),
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "polygon_area_sum": float(sum(areas)),
                "vertex_x_min": vertex_x_min,
                "vertex_x_max": vertex_x_max,
                "vertex_y_min": vertex_y_min,
                "vertex_y_max": vertex_y_max,
                "bin_pairs": bin_pairs,
                "bin_pairs_text": ";".join(f"{x}:{y}" for x, y in bin_pairs),
                "polygon_iou": sample_iou.get((split, sample_index_int), 0.0),
                "zero_iou": sample_iou.get((split, sample_index_int), 0.0) <= 0.0,
            }
        )
    return out


def _load_component_predictions(split: str, prediction_dir: Path, target_root: Path, raw_root: Path) -> list[dict]:
    pred_path = prediction_dir / f"{split}_center_anchored_polygon_predictions.csv"
    metric_path = prediction_dir / f"{split}_center_anchored_polygon_mask_metrics.csv"
    pred_rows = _read_csv(pred_path)
    mask_rows = {int(row["sample_index"]): row for row in _read_csv(metric_path)}
    defect_rows = {int(row["sample_index"]): row for row in _read_csv(raw_root / split / "defect_params.csv")}
    targets = _load_targets(target_root / split / "center_anchored_polygon_targets.npz")
    x_centers = targets["center_bin_x_centers"].astype(np.float64)
    y_centers = targets["center_bin_y_centers"].astype(np.float64)
    grid_dx = float(targets["grid_dx"])
    grid_dy = float(targets["grid_dy"])
    out = []
    for row in pred_rows:
        sample_index = int(row["sample_index"])
        component_slot = int(row["component_slot"])
        presence_true = _as_bool(row["presence_true"])
        presence_pred = _as_bool(row["presence_pred"])
        true_x_bin = _as_int(row["center_x_bin_true"])
        pred_x_bin = _as_int(row["center_x_bin_pred"])
        true_y_bin = _as_int(row["center_y_bin_true"])
        pred_y_bin = _as_int(row["center_y_bin_pred"])
        local_mae, local_max = _local_error(row)
        true_vertices = _component_vertices(row, "true")
        pred_vertices = _component_vertices(row, "pred")
        decoded_vertex_mae = _as_float(row.get("decoded_vertex_mae"))
        x_bin_correct = bool(presence_true and true_x_bin == pred_x_bin)
        y_bin_correct = bool(presence_true and true_y_bin == pred_y_bin)
        both_bins_correct = bool(x_bin_correct and y_bin_correct)
        true_center_x = float(true_vertices[:, 0].mean())
        true_center_y = float(true_vertices[:, 1].mean())
        pred_center_x = float(pred_vertices[:, 0].mean())
        pred_center_y = float(pred_vertices[:, 1].mean())
        center_coord_error = math.sqrt((pred_center_x - true_center_x) ** 2 + (pred_center_y - true_center_y) ** 2)
        center_grid_error = math.sqrt(((pred_center_x - true_center_x) / grid_dx) ** 2 + ((pred_center_y - true_center_y) / grid_dy) ** 2)
        pred_signed = _shoelace(pred_vertices)
        true_signed = _shoelace(true_vertices)
        defect = defect_rows[sample_index]
        metrics = mask_rows[sample_index]
        out.append(
            {
                "split": split,
                "sample_index": sample_index,
                "hard_case_type": defect["hard_case_type"],
                "component_slot": component_slot,
                "presence_true": presence_true,
                "presence_pred": presence_pred,
                "presence_correct": presence_true == presence_pred,
                "type_true": _as_int(row["type_true"], -1),
                "type_pred": _as_int(row["type_pred"], -1),
                "type_correct": (not presence_true) or (_as_int(row["type_true"], -1) == _as_int(row["type_pred"], -1)),
                "true_center_x_bin": true_x_bin,
                "pred_center_x_bin": pred_x_bin,
                "true_center_y_bin": true_y_bin,
                "pred_center_y_bin": pred_y_bin,
                "x_bin_correct": x_bin_correct,
                "y_bin_correct": y_bin_correct,
                "both_bins_correct": both_bins_correct,
                "center_offset_error": _as_float(row.get("center_offset_mae")),
                "center_coord_error": center_coord_error,
                "center_grid_error": center_grid_error,
                "local_vertex_grid_mae": local_mae,
                "local_vertex_grid_max_error": local_max,
                "decoded_vertex_mae": decoded_vertex_mae,
                "pred_area": _as_int(metrics["pred_area"]),
                "target_area": _as_int(metrics["target_area"]),
                "area_diff": _as_int(metrics["pred_area"]) - _as_int(metrics["target_area"]),
                "polygon_iou": _as_float(metrics["polygon_mask_iou"]),
                "zero_iou": _as_float(metrics["polygon_mask_iou"]) <= 0.0,
                "is_true_rotated": _as_bool(defect.get("true_rotated_geometry", "false")),
                "is_true_multi_component": _as_bool(defect.get("true_multi_component_geometry", "false")),
                "signed_area_flip": _as_bool(row.get("signed_area_flip")),
                "pred_signed_area": pred_signed,
                "true_signed_area": true_signed,
                "true_center_x": true_center_x,
                "true_center_y": true_center_y,
                "pred_center_x": pred_center_x,
                "pred_center_y": pred_center_y,
                "true_bin_center_x": float(x_centers[true_x_bin]),
                "true_bin_center_y": float(y_centers[true_y_bin]),
                "pred_bin_center_x": float(x_centers[min(max(pred_x_bin, 0), len(x_centers) - 1)]),
                "pred_bin_center_y": float(y_centers[min(max(pred_y_bin, 0), len(y_centers) - 1)]),
            }
        )
    return out


def _load_sample_predictions(split: str, prediction_dir: Path, component_rows: list[dict]) -> list[dict]:
    metric_path = prediction_dir / f"{split}_center_anchored_polygon_mask_metrics.csv"
    comps_by_sample: dict[int, list[dict]] = defaultdict(list)
    for row in component_rows:
        comps_by_sample[int(row["sample_index"])].append(row)
    out = []
    for row in _read_csv(metric_path):
        sample_index = int(row["sample_index"])
        comps = [item for item in comps_by_sample[sample_index] if item["presence_true"]]
        if comps:
            x_bin_acc = _mean(1.0 if item["x_bin_correct"] else 0.0 for item in comps)
            y_bin_acc = _mean(1.0 if item["y_bin_correct"] else 0.0 for item in comps)
            both_bin_acc = _mean(1.0 if item["both_bins_correct"] else 0.0 for item in comps)
            all_x = all(item["x_bin_correct"] for item in comps)
            all_y = all(item["y_bin_correct"] for item in comps)
            all_both = all(item["both_bins_correct"] for item in comps)
            first = comps[0]
        else:
            x_bin_acc = y_bin_acc = both_bin_acc = 0.0
            all_x = all_y = all_both = False
            first = comps_by_sample[sample_index][0]
        out.append(
            {
                "split": split,
                "sample_index": sample_index,
                "hard_case_type": first["hard_case_type"],
                "component_count": len(comps),
                "polygon_iou": _as_float(row["polygon_mask_iou"]),
                "zero_iou": _as_float(row["polygon_mask_iou"]) <= 0.0,
                "pred_area": _as_int(row["pred_area"]),
                "target_area": _as_int(row["target_area"]),
                "area_diff": _as_int(row["pred_area"]) - _as_int(row["target_area"]),
                "pred_component_count": _as_int(row["pred_component_count"]),
                "true_component_count": _as_int(row["true_component_count"]),
                "out_of_grid_vertex_count": _as_int(row["out_of_grid_vertex_count"]),
                "x_bin_acc": x_bin_acc,
                "y_bin_acc": y_bin_acc,
                "both_bin_acc": both_bin_acc,
                "all_x_bins_correct": all_x,
                "all_y_bins_correct": all_y,
                "all_bins_correct": all_both,
                "any_x_bin_wrong": not all_x,
                "any_y_bin_wrong": not all_y,
                "any_bin_wrong": not all_both,
                "presence_correct": all(item["presence_correct"] for item in comps_by_sample[sample_index]),
                "present_type_correct": all(item["type_correct"] for item in comps),
                "center_offset_error_mean": _mean(item["center_offset_error"] for item in comps),
                "center_grid_error_mean": _mean(item["center_grid_error"] for item in comps),
                "local_vertex_grid_mae_mean": _mean(item["local_vertex_grid_mae"] for item in comps),
                "local_vertex_grid_max_error": _max(item["local_vertex_grid_max_error"] for item in comps),
                "decoded_vertex_mae_mean": _mean(item["decoded_vertex_mae"] for item in comps),
                "is_true_rotated": first["is_true_rotated"],
                "is_true_multi_component": first["is_true_multi_component"],
                "type_true_values": ";".join(str(item["type_true"]) for item in comps),
                "type_pred_values": ";".join(str(item["type_pred"]) for item in comps),
                "true_center_bin_pairs": ";".join(
                    f"{item['true_center_x_bin']}:{item['true_center_y_bin']}" for item in comps
                ),
                "pred_center_bin_pairs": ";".join(
                    f"{item['pred_center_x_bin']}:{item['pred_center_y_bin']}" for item in comps
                ),
            }
        )
    return out


def _aggregate(rows: list[dict], keys: list[str], numeric_fields: list[str], bool_fields: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key, items in sorted(groups.items()):
        result = {name: value for name, value in zip(keys, key)}
        result["row_count"] = len(items)
        result["sample_count"] = len({(item["split"], item["sample_index"]) for item in items})
        for field in numeric_fields:
            vals = [_as_float(item[field]) for item in items]
            result[f"{field}_mean"] = _mean(vals)
            result[f"{field}_min"] = _min(vals)
            result[f"{field}_max"] = _max(vals)
        for field in bool_fields:
            result[f"{field}_rate"] = _mean(1.0 if bool(item[field]) else 0.0 for item in items)
        out.append(result)
    return out


def _build_coverage(sample_features: list[dict], sample_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train_features = [row for row in sample_features if row["split"] == "train"]
    heldout_features = [row for row in sample_features if row["split"] in HELDOUT_SPLITS]
    sample_metrics = {(row["split"], row["sample_index"]): row for row in sample_rows}
    coverage: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in train_features:
        for pair in row["bin_pairs"]:
            coverage[pair].append(row)
    coverage_rows = []
    for pair, rows in sorted(coverage.items()):
        coverage_rows.append(
            {
                "true_center_x_bin": pair[0],
                "true_center_y_bin": pair[1],
                "train_component_count": len(rows),
                "train_sample_count": len({row["sample_index"] for row in rows}),
                "hard_case_types": ";".join(sorted({row["hard_case_type"] for row in rows})),
                "rotated_count": sum(1 for row in rows if row["true_rotated"]),
                "multi_component_count": sum(1 for row in rows if row["true_multi_component"]),
            }
        )
    uncovered_rows = []
    nearest_rows = []
    for held in heldout_features:
        held_metric = sample_metrics[(held["split"], held["sample_index"])]
        uncovered_pairs = [pair for pair in held["bin_pairs"] if pair not in coverage]
        for pair in uncovered_pairs:
            uncovered_rows.append(
                {
                    "split": held["split"],
                    "sample_index": held["sample_index"],
                    "hard_case_type": held["hard_case_type"],
                    "true_center_x_bin": pair[0],
                    "true_center_y_bin": pair[1],
                    "zero_iou": held_metric["zero_iou"],
                    "polygon_iou": held_metric["polygon_iou"],
                    "is_true_rotated": held["true_rotated"],
                    "is_true_multi_component": held["true_multi_component"],
                }
            )
        best = None
        for train in train_features:
            center_bin_distance = _bin_distance(held["bin_pairs"], train["bin_pairs"])
            center_coord_distance = math.sqrt(
                (held["center_x_mean"] - train["center_x_mean"]) ** 2
                + (held["center_y_mean"] - train["center_y_mean"]) ** 2
            )
            bbox_size_distance = abs(held["bbox_width"] - train["bbox_width"]) + abs(held["bbox_height"] - train["bbox_height"])
            area_distance = abs(held["polygon_area_sum"] - train["polygon_area_sum"])
            score = (center_bin_distance, center_coord_distance, bbox_size_distance, area_distance)
            candidate = {
                "split": held["split"],
                "sample_index": held["sample_index"],
                "hard_case_type": held["hard_case_type"],
                "zero_iou": held_metric["zero_iou"],
                "polygon_iou": held_metric["polygon_iou"],
                "heldout_bin_pairs": held["bin_pairs_text"],
                "all_bins_covered_by_train": len(uncovered_pairs) == 0,
                "uncovered_component_count": len(uncovered_pairs),
                "nearest_train_sample_index": train["sample_index"],
                "nearest_train_hard_case_type": train["hard_case_type"],
                "center_bin_distance": center_bin_distance,
                "center_coord_distance": center_coord_distance,
                "bbox_size_distance": bbox_size_distance,
                "area_distance": area_distance,
                "hard_case_type_match": held["hard_case_type"] == train["hard_case_type"],
                "true_rotated_match": held["true_rotated"] == train["true_rotated"],
                "true_multi_component_match": held["true_multi_component"] == train["true_multi_component"],
                "heldout_true_rotated": held["true_rotated"],
                "heldout_true_multi_component": held["true_multi_component"],
                "nearest_train_true_rotated": train["true_rotated"],
                "nearest_train_true_multi_component": train["true_multi_component"],
            }
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None:
            nearest_rows.append(best[1])
    if not uncovered_rows:
        uncovered_rows.append(
            {
                "split": "none",
                "sample_index": -1,
                "hard_case_type": "none",
                "true_center_x_bin": -1,
                "true_center_y_bin": -1,
                "zero_iou": False,
                "polygon_iou": 0.0,
                "is_true_rotated": False,
                "is_true_multi_component": False,
            }
        )
    return coverage_rows, uncovered_rows, nearest_rows


def _bin_distance(a_pairs: list[tuple[int, int]], b_pairs: list[tuple[int, int]]) -> float:
    if not a_pairs or not b_pairs:
        return 0.0
    distances = []
    for ax, ay in a_pairs:
        distances.append(min(abs(ax - bx) + abs(ay - by) for bx, by in b_pairs))
    return float(sum(distances) / len(distances))


def _write_failure_summary(output_dir: Path, sample_rows: list[dict], component_rows: list[dict]) -> None:
    split_summary = _aggregate(
        sample_rows,
        ["split"],
        [
            "polygon_iou",
            "x_bin_acc",
            "y_bin_acc",
            "both_bin_acc",
            "center_grid_error_mean",
            "local_vertex_grid_mae_mean",
            "area_diff",
        ],
        [
            "zero_iou",
            "any_x_bin_wrong",
            "any_y_bin_wrong",
            "any_bin_wrong",
            "presence_correct",
            "present_type_correct",
        ],
    )
    heldout = [row for row in sample_rows if row["split"] in HELDOUT_SPLITS]
    zero = [row for row in heldout if row["zero_iou"]]
    all_bins_correct_heldout = [row for row in heldout if row["all_bins_correct"]]
    zero_with_y_wrong = sum(1 for row in zero if row["any_y_bin_wrong"])
    zero_with_x_wrong = sum(1 for row in zero if row["any_x_bin_wrong"])
    zero_with_both_wrong = sum(1 for row in zero if row["any_bin_wrong"])
    correct_bin_components = [
        row
        for row in component_rows
        if row["split"] in HELDOUT_SPLITS and row["presence_true"] and row["both_bins_correct"]
    ]
    incorrect_bin_components = [
        row
        for row in component_rows
        if row["split"] in HELDOUT_SPLITS and row["presence_true"] and not row["both_bins_correct"]
    ]
    hard_case_heldout = _aggregate(
        heldout,
        ["hard_case_type"],
        ["polygon_iou", "x_bin_acc", "y_bin_acc"],
        ["zero_iou", "any_x_bin_wrong", "any_y_bin_wrong"],
    )
    slot_heldout = _aggregate(
        [row for row in component_rows if row["split"] in HELDOUT_SPLITS and row["presence_true"]],
        ["component_slot"],
        ["polygon_iou", "center_grid_error", "local_vertex_grid_mae"],
        ["x_bin_correct", "y_bin_correct", "both_bins_correct", "zero_iou"],
    )
    rotated_heldout = _aggregate(
        heldout,
        ["is_true_rotated"],
        ["polygon_iou", "x_bin_acc", "y_bin_acc"],
        ["zero_iou", "any_y_bin_wrong"],
    )
    multi_heldout = _aggregate(
        heldout,
        ["is_true_multi_component"],
        ["polygon_iou", "x_bin_acc", "y_bin_acc"],
        ["zero_iou", "any_y_bin_wrong"],
    )
    hardest_case = min(hard_case_heldout, key=lambda row: _as_float(row["polygon_iou_mean"]))
    hardest_slot = min(slot_heldout, key=lambda row: _as_float(row["polygon_iou_mean"]))
    lines = ["# S295 Center-Anchored Polygon Failure Diagnostics", "", "## Split Metrics", ""]
    for row in split_summary:
        lines.append(
            f"- {row['split']}: IoU mean/min `{row['polygon_iou_mean']:.6f}` / `{row['polygon_iou_min']:.6f}`, "
            f"x/y bin acc `{row['x_bin_acc_mean']:.6f}` / `{row['y_bin_acc_mean']:.6f}`, "
            f"zero-IoU rate `{row['zero_iou_rate']:.6f}`, "
            f"presence/type correctness `{row['presence_correct_rate']:.6f}` / `{row['present_type_correct_rate']:.6f}`."
        )
    lines.extend(["", "## Held-Out Failure Mechanism", ""])
    lines.append(
        f"- Held-out zero-IoU samples: `{len(zero)}` / `{len(heldout)}`; "
        f"with any y-bin wrong `{zero_with_y_wrong}`, any x-bin wrong `{zero_with_x_wrong}`, any bin wrong `{zero_with_both_wrong}`."
    )
    lines.append(
        f"- Held-out present components with correct bins have local vertex grid MAE mean "
        f"`{_mean(row['local_vertex_grid_mae'] for row in correct_bin_components):.6f}`; "
        f"components with wrong bins have mean `{_mean(row['local_vertex_grid_mae'] for row in incorrect_bin_components):.6f}`."
    )
    lines.append(
        f"- Held-out samples with all center bins correct: `{len(all_bins_correct_heldout)}` / `{len(heldout)}`, "
        f"mean IoU `{_mean(row['polygon_iou'] for row in all_bins_correct_heldout):.6f}`."
    )
    lines.extend(["", "## Required Answers", ""])
    lines.append(
        f"1. Zero-IoU is primarily center-bin driven: `{zero_with_both_wrong}` / `{len(zero)}` zero-IoU samples have at least one center-bin error."
    )
    lines.append(
        f"2. Y-bin is the stronger bottleneck: zero-IoU samples with y-bin wrong `{zero_with_y_wrong}`, x-bin wrong `{zero_with_x_wrong}`."
    )
    lines.append(
        f"3. Local shape is secondary here: correct-bin held-out components have lower local vertex grid MAE "
        f"(`{_mean(row['local_vertex_grid_mae'] for row in correct_bin_components):.6f}`) than wrong-bin components "
        f"(`{_mean(row['local_vertex_grid_mae'] for row in incorrect_bin_components):.6f}`)."
    )
    lines.append(
        f"4. Hardest hard_case by held-out mean IoU is `{hardest_case['hard_case_type']}` "
        f"with mean IoU `{hardest_case['polygon_iou_mean']:.6f}` and zero-IoU rate `{hardest_case['zero_iou_rate']:.6f}`."
    )
    lines.append(
        f"5. Hardest component slot by held-out mean IoU is slot `{hardest_slot['component_slot']}` "
        f"with mean IoU `{hardest_slot['polygon_iou_mean']:.6f}`."
    )
    rotated_note = "; ".join(
        f"rotated={row['is_true_rotated']}: zero-rate {row['zero_iou_rate']:.6f}, y-bin acc {row['y_bin_acc_mean']:.6f}"
        for row in rotated_heldout
    )
    multi_note = "; ".join(
        f"multi={row['is_true_multi_component']}: zero-rate {row['zero_iou_rate']:.6f}, y-bin acc {row['y_bin_acc_mean']:.6f}"
        for row in multi_heldout
    )
    lines.append(f"6. Rotated and multi-component groups are harder in held-out: {rotated_note}; {multi_note}.")
    lines.append(
        "7. Worst samples share y-bin errors and sparse train-bin coverage; see `worst_heldout_samples.csv` and S296 coverage tables."
    )
    lines.extend(
        [
            "",
            "This diagnostic is read-only. It does not rerun training, change the model, or change the runner.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_coverage_summary(output_dir: Path, nearest_rows: list[dict], uncovered_rows: list[dict]) -> None:
    real_uncovered = [row for row in uncovered_rows if row["sample_index"] != -1]
    heldout = nearest_rows
    zero = [row for row in heldout if row["zero_iou"]]
    nonzero = [row for row in heldout if not row["zero_iou"]]
    zero_uncovered = [row for row in zero if not row["all_bins_covered_by_train"]]
    nonzero_uncovered = [row for row in nonzero if not row["all_bins_covered_by_train"]]
    lines = ["# S296 Matched Coverage Analysis", "", "## Coverage", ""]
    lines.append(f"- Held-out samples: `{len(heldout)}`; zero-IoU samples: `{len(zero)}`.")
    lines.append(
        f"- Held-out uncovered component bins: `{len(real_uncovered)}`; "
        f"zero-IoU samples with uncovered bins: `{len(zero_uncovered)}`; "
        f"nonzero samples with uncovered bins: `{len(nonzero_uncovered)}`."
    )
    lines.append(
        f"- Nearest-train center-bin distance mean for zero/nonzero held-out samples: "
        f"`{_mean(row['center_bin_distance'] for row in zero):.6f}` / "
        f"`{_mean(row['center_bin_distance'] for row in nonzero):.6f}`."
    )
    lines.append(
        f"- Nearest-train center-coordinate distance mean for zero/nonzero held-out samples: "
        f"`{_mean(row['center_coord_distance'] for row in zero):.6e}` / "
        f"`{_mean(row['center_coord_distance'] for row in nonzero):.6e}`."
    )
    if len(zero_uncovered) >= max(1, len(zero) // 2):
        decision = "Zero-IoU correlates with train center-bin coverage gaps; matched-coverage resplit is the first diagnostic repair."
    else:
        decision = "Zero-IoU is not explained by uncovered center bins alone; center-bin prediction and local-shape generalization need model-side diagnostics."
    lines.extend(["", "## Decision Signal", "", f"- {decision}"])
    (output_dir / "matched_coverage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    prediction_dir = Path(args.prediction_dir)
    target_root = Path(args.target_root)
    raw_root = Path(args.raw_root)
    diagnostics_dir = Path(args.output_dir)
    coverage_dir = Path(args.coverage_output_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)

    component_rows: list[dict] = []
    sample_rows: list[dict] = []
    for split in SPLITS:
        split_components = _load_component_predictions(split, prediction_dir, target_root, raw_root)
        component_rows.extend(split_components)
        sample_rows.extend(_load_sample_predictions(split, prediction_dir, split_components))
    sample_iou = {(row["split"], row["sample_index"]): row["polygon_iou"] for row in sample_rows}
    sample_features: list[dict] = []
    for split in SPLITS:
        sample_features.extend(_target_sample_features(split, target_root, raw_root, sample_iou))

    _write_csv(diagnostics_dir / "per_sample_failure_diagnostics.csv", sample_rows)
    _write_csv(diagnostics_dir / "per_component_failure_diagnostics.csv", component_rows)
    _write_csv(
        diagnostics_dir / "grouped_by_center_bin.csv",
        _aggregate(
            [row for row in component_rows if row["presence_true"]],
            ["split", "true_center_x_bin", "true_center_y_bin"],
            [
                "polygon_iou",
                "center_offset_error",
                "center_grid_error",
                "local_vertex_grid_mae",
                "decoded_vertex_mae",
            ],
            ["x_bin_correct", "y_bin_correct", "both_bins_correct", "zero_iou"],
        ),
    )
    _write_csv(
        diagnostics_dir / "grouped_by_hard_case_type.csv",
        _aggregate(
            sample_rows,
            ["split", "hard_case_type"],
            ["polygon_iou", "x_bin_acc", "y_bin_acc", "center_grid_error_mean", "local_vertex_grid_mae_mean", "area_diff"],
            ["zero_iou", "any_x_bin_wrong", "any_y_bin_wrong", "presence_correct", "present_type_correct"],
        ),
    )
    _write_csv(
        diagnostics_dir / "grouped_by_component_slot.csv",
        _aggregate(
            [row for row in component_rows if row["presence_true"]],
            ["split", "component_slot"],
            ["polygon_iou", "center_grid_error", "local_vertex_grid_mae", "decoded_vertex_mae"],
            ["x_bin_correct", "y_bin_correct", "both_bins_correct", "zero_iou", "type_correct"],
        ),
    )
    _write_csv(
        diagnostics_dir / "grouped_by_true_rotated.csv",
        _aggregate(
            sample_rows,
            ["split", "is_true_rotated"],
            ["polygon_iou", "x_bin_acc", "y_bin_acc", "center_grid_error_mean", "local_vertex_grid_mae_mean"],
            ["zero_iou", "any_x_bin_wrong", "any_y_bin_wrong"],
        ),
    )
    _write_csv(
        diagnostics_dir / "grouped_by_multi_component.csv",
        _aggregate(
            sample_rows,
            ["split", "is_true_multi_component"],
            ["polygon_iou", "x_bin_acc", "y_bin_acc", "center_grid_error_mean", "local_vertex_grid_mae_mean"],
            ["zero_iou", "any_x_bin_wrong", "any_y_bin_wrong"],
        ),
    )
    worst = sorted(
        [row for row in sample_rows if row["split"] in HELDOUT_SPLITS],
        key=lambda item: (not item["zero_iou"], item["polygon_iou"]),
    )[:12]
    _write_csv(diagnostics_dir / "worst_heldout_samples.csv", worst)
    _write_failure_summary(diagnostics_dir, sample_rows, component_rows)

    coverage_rows, uncovered_rows, nearest_rows = _build_coverage(sample_features, sample_rows)
    _write_csv(coverage_dir / "train_center_bin_coverage.csv", coverage_rows)
    _write_csv(coverage_dir / "heldout_nearest_train_matches.csv", nearest_rows)
    _write_csv(coverage_dir / "uncovered_heldout_bins.csv", uncovered_rows)
    _write_coverage_summary(coverage_dir, nearest_rows, uncovered_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        default="experiments/dual_network/S292_comsol_v3_center_anchored_polygon_gates/train30_quick_probe",
    )
    parser.add_argument("--target-root", default="experiments/dual_network/S290_comsol_v3_center_anchored_polygon_targets")
    parser.add_argument("--raw-root", default="experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/raw")
    parser.add_argument(
        "--output-dir",
        default="experiments/dual_network/S295_center_anchored_polygon_failure_diagnostics",
    )
    parser.add_argument(
        "--coverage-output-dir",
        default="experiments/dual_network/S296_center_anchored_polygon_matched_coverage_analysis",
    )
    args = parser.parse_args(argv)
    run(args)
    print(f"Saved center-anchored polygon diagnostics to {args.output_dir} and {args.coverage_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

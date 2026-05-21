from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402
import train_comsol_rect_rot_strong_dense_initializer as dense_init  # noqa: E402

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_improved_proposal_extraction_summary.txt"
DEFAULT_CANDIDATES = PROJECT_ROOT / "results/metrics/comsol_rect_rot_proposal_extraction_candidates.csv"
DEFAULT_SELECTED = PROJECT_ROOT / "results/metrics/comsol_rect_rot_proposal_extraction_selected_geometry.csv"
DEFAULT_GROUP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_proposal_extraction_group_summary.csv"

MAX_ANGLE_DEG = 35.0
GEOMETRY_THRESHOLD = 0.50

CANDIDATE_FIELDS = [
    "method",
    "extraction_threshold",
    "split",
    "sample_count",
    "geometry_iou_mean",
    "geometry_dice_mean",
    "geometry_area_error_mean",
    "geometry_center_error_px_mean",
    "angle_mae_deg",
    "empty_count",
    "fallback_count",
    "score",
    "available",
    "notes",
]

SELECTED_FIELDS = [
    "sample_id",
    "source_index",
    "split",
    "defect_type",
    "source_pack",
    "method",
    "threshold",
    "dense_threshold",
    "geometry_threshold",
    "extraction_threshold",
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
    "dense_center_error_px",
    "dense_pred_area",
    "true_area",
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

GROUP_FIELDS = [
    "split",
    "group_name",
    "group_value",
    "sample_count",
    "dense_iou_mean",
    "dense_dice_mean",
    "dense_area_error_mean",
    "geometry_iou_mean",
    "geometry_dice_mean",
    "geometry_area_error_mean",
    "angle_mae_deg",
    "empty_count",
    "fallback_count",
]


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


def normalize_angle_deg(angle_deg: float) -> float:
    angle = (angle_deg + 90.0) % 180.0 - 90.0
    if angle > 45.0:
        angle -= 90.0
    if angle < -45.0:
        angle += 90.0
    return float(np.clip(angle, -MAX_ANGLE_DEG, MAX_ANGLE_DEG))


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.percentile(values, percentile))
    cutoff = percentile / 100.0 * cumulative[-1]
    return float(values[np.searchsorted(cumulative, cutoff, side="left").clip(0, values.size - 1)])


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
    count = max(16, int(round(flat.size * 0.035)))
    cutoff = np.partition(flat, -count)[-count]
    return prob >= cutoff


def component_points(prob: np.ndarray, threshold: float, arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    raw_mask = prob >= threshold
    empty = not bool(raw_mask.any())
    fallback = False
    if empty:
        raw_mask = top_region(prob)
        fallback = True
    component, component_empty = largest_component(raw_mask)
    fallback = fallback or component_empty
    if not component.any():
        component = top_region(prob)
        fallback = True
    ys, xs = np.argwhere(component).T
    coords = np.stack([arrays["mask_x"][xs], arrays["mask_y"][ys]], axis=1).astype(np.float64)
    weights = prob[ys, xs].astype(np.float64)
    weights = np.maximum(weights, 1e-6)
    return coords, weights, empty, fallback


def finalize_geometry(
    center: np.ndarray,
    width: float,
    length: float,
    angle_deg: float,
    train_median_depth: float,
    arrays: dict[str, Any],
    empty: bool,
    fallback: bool,
    component_area_px: float,
    method: str,
) -> dict[str, Any]:
    angle = normalize_angle_deg(angle_deg)
    if length > width:
        width, length = length, width
        angle = normalize_angle_deg(angle + 90.0)
    width = float(np.clip(width, 0.001, 0.025))
    length = float(np.clip(length, 0.001, 0.020))
    center_x = float(np.clip(center[0], float(arrays["mask_x"].min()), float(arrays["mask_x"].max())))
    center_y = float(np.clip(center[1], float(arrays["mask_y"].min()), float(arrays["mask_y"].max())))
    aspect = width / max(length, 1e-6)
    p_rot = 1.0 / (1.0 + math.exp(-(abs(angle) - 6.0) / 3.0))
    if aspect < 1.25:
        p_rot *= 0.85
    p_rot = float(np.clip(p_rot, 0.05, 0.95))
    return {
        "center_x": center_x,
        "center_y": center_y,
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
        "component_area_px": float(component_area_px),
        "method": method,
    }


def pca_geometry(prob: np.ndarray, threshold: float, arrays: dict[str, Any], train_median_depth: float, method: str) -> dict[str, Any]:
    coords, weights, empty, fallback = component_points(prob, threshold, arrays)
    center = coords.mean(axis=0)
    if coords.shape[0] < 3:
        return finalize_geometry(center, 0.002, 0.002, 0.0, train_median_depth, arrays, empty, True, coords.shape[0], method)
    centered = coords - center
    cov = centered.T @ centered / max(coords.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    theta = math.radians(normalize_angle_deg(angle))
    axis_major = np.array([math.cos(theta), math.sin(theta)])
    axis_minor = np.array([-math.sin(theta), math.cos(theta)])
    proj_major = centered @ axis_major
    proj_minor = centered @ axis_minor
    width = float(proj_major.max() - proj_major.min())
    length = float(proj_minor.max() - proj_minor.min())
    return finalize_geometry(center, width, length, angle, train_median_depth, arrays, empty, fallback, coords.shape[0], method)


def weighted_pca_geometry(prob: np.ndarray, threshold: float, arrays: dict[str, Any], train_median_depth: float) -> dict[str, Any]:
    low = max(0.05, threshold - 0.25)
    mask = prob >= low
    empty = not bool(mask.any())
    fallback = False
    if empty:
        mask = top_region(prob)
        fallback = True
    ys, xs = np.argwhere(mask).T
    coords = np.stack([arrays["mask_x"][xs], arrays["mask_y"][ys]], axis=1).astype(np.float64)
    weights = np.maximum(prob[ys, xs].astype(np.float64) - low, 1e-6)
    center = np.average(coords, axis=0, weights=weights)
    centered = coords - center
    cov = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1e-6)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = normalize_angle_deg(math.degrees(math.atan2(float(major[1]), float(major[0]))))
    theta = math.radians(angle)
    axis_major = np.array([math.cos(theta), math.sin(theta)])
    axis_minor = np.array([-math.sin(theta), math.cos(theta)])
    proj_major = centered @ axis_major
    proj_minor = centered @ axis_minor
    width = weighted_percentile(proj_major, weights, 96.0) - weighted_percentile(proj_major, weights, 4.0)
    length = weighted_percentile(proj_minor, weights, 96.0) - weighted_percentile(proj_minor, weights, 4.0)
    return finalize_geometry(center, width, length, angle, train_median_depth, arrays, empty, fallback, float(mask.sum()), "probability_weighted_pca")


def min_area_rect_geometry(prob: np.ndarray, threshold: float, arrays: dict[str, Any], train_median_depth: float) -> dict[str, Any]:
    if cv2 is None:
        geom = pca_geometry(prob, threshold, arrays, train_median_depth, "min_area_rect_if_available")
        geom["fallback_used"] = 1.0
        return geom
    coords, _weights, empty, fallback = component_points(prob, threshold, arrays)
    if coords.shape[0] < 3:
        return finalize_geometry(coords.mean(axis=0), 0.002, 0.002, 0.0, train_median_depth, arrays, empty, True, coords.shape[0], "min_area_rect_if_available")
    rect = cv2.minAreaRect(coords.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    return finalize_geometry(
        np.array([cx, cy], dtype=np.float64),
        float(w),
        float(h),
        float(angle),
        train_median_depth,
        arrays,
        empty,
        fallback,
        coords.shape[0],
        "min_area_rect_if_available",
    )


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


def extract_one(prob: np.ndarray, method: str, threshold: float, arrays: dict[str, Any], train_median_depth: float) -> dict[str, Any]:
    if method == "largest_component_pca":
        return pca_geometry(prob, threshold, arrays, train_median_depth, method)
    if method == "probability_weighted_pca":
        return weighted_pca_geometry(prob, threshold, arrays, train_median_depth)
    if method == "min_area_rect_if_available":
        return min_area_rect_geometry(prob, threshold, arrays, train_median_depth)
    if method.startswith("hybrid_threshold_sweep"):
        return pca_geometry(prob, threshold, arrays, train_median_depth, method)
    raise ValueError(f"Unknown extraction method: {method}")


def dense_predictions(bundle: dense_init.DenseInitializerBundle, batch_size: int) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        ds = dense_init.DenseMaskDataset(bundle.arrays["split_indices"][split], bundle.arrays)
        out[split] = dense_init.predict(bundle.model, ds, bundle.device, batch_size)
    return out


def candidate_specs(selected_threshold: float) -> list[tuple[str, float]]:
    thresholds = sorted({float(np.clip(selected_threshold + delta, 0.30, 0.90)) for delta in (-0.10, 0.0, 0.10)})
    specs = [
        ("largest_component_pca", selected_threshold),
        ("probability_weighted_pca", selected_threshold),
        ("min_area_rect_if_available", selected_threshold),
    ]
    specs.extend((f"hybrid_threshold_sweep_t{threshold:.2f}", threshold) for threshold in thresholds)
    return specs


def rows_for_spec(
    split: str,
    pred: dict[str, np.ndarray],
    arrays: dict[str, Any],
    method: str,
    extraction_threshold: float,
    train_median_depth: float,
    dense_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_method = "hybrid_threshold_sweep" if method.startswith("hybrid_threshold_sweep") else method
    for order, local_idx_raw in enumerate(pred["indices"]):
        local_idx = int(local_idx_raw)
        prob = pred["prob"][order]
        dense_metric = base.mask_metric(prob, arrays["masks"][local_idx], dense_threshold)
        geom = extract_one(prob, base_method, extraction_threshold, arrays, train_median_depth)
        geom["method"] = method
        geom_prob = geometry_mask(geom, arrays)
        geom_metric = base.mask_metric(geom_prob, arrays["masks"][local_idx], GEOMETRY_THRESHOLD)
        true_geom = arrays["raw_geom"][local_idx]
        true_angle = math.degrees(math.atan2(float(arrays["angle_targets"][local_idx, 0]), float(arrays["angle_targets"][local_idx, 1])))
        angle_error = (
            base.circular_angle_error_deg(geom["angle_deg"], true_angle)
            if str(arrays["defect_types"][local_idx]) == "rotated_rect"
            else math.nan
        )
        rows.append(
            {
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": split,
                "defect_type": str(arrays["defect_types"][local_idx]),
                "source_pack": str(arrays["source_packs"][local_idx]),
                "method": method,
                "threshold": GEOMETRY_THRESHOLD,
                "dense_threshold": dense_threshold,
                "geometry_threshold": GEOMETRY_THRESHOLD,
                "extraction_threshold": extraction_threshold,
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
                "dense_center_error_px": dense_metric["center_error_px"],
                "dense_pred_area": dense_metric["pred_area"],
                "true_area": dense_metric["true_area"],
                "geometry_iou": geom_metric["iou"],
                "geometry_dice": geom_metric["dice"],
                "geometry_area_error": geom_metric["area_error"],
                "center_abs_error_m": float(math.hypot(geom["center_x"] - true_geom[0], geom["center_y"] - true_geom[1])),
                "width_abs_error_m": abs(geom["width"] - true_geom[2]),
                "length_abs_error_m": abs(geom["length"] - true_geom[3]),
                "depth_abs_error_m": abs(geom["depth"] - true_geom[4]),
                "angle_abs_error_deg": angle_error,
                "empty_prediction": geom["empty_prediction"],
                "fallback_used": geom["fallback_used"],
                "component_area_px": geom["component_area_px"],
                "depth_source": "train_median_depth",
                "type_probability_source": "angle_aspect_heuristic_no_true_type",
                "notes": "" if cv2 is not None or not method.startswith("min_area") else "cv2 unavailable; PCA fallback",
            }
        )
    return rows


def summarize_candidate(rows: list[dict[str, Any]], method: str, extraction_threshold: float, split: str, available: bool) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split]
    score = safe_mean(subset, "geometry_iou") + safe_mean(subset, "geometry_dice") - safe_mean(subset, "geometry_area_error")
    return {
        "method": method,
        "extraction_threshold": extraction_threshold,
        "split": split,
        "sample_count": len(subset),
        "geometry_iou_mean": safe_mean(subset, "geometry_iou"),
        "geometry_dice_mean": safe_mean(subset, "geometry_dice"),
        "geometry_area_error_mean": safe_mean(subset, "geometry_area_error"),
        "geometry_center_error_px_mean": math.nan,
        "angle_mae_deg": safe_mean([row for row in subset if row["defect_type"] == "rotated_rect"], "angle_abs_error_deg"),
        "empty_count": int(sum(float(row["empty_prediction"]) > 0 for row in subset)),
        "fallback_count": int(sum(float(row["fallback_used"]) > 0 for row in subset)),
        "score": score,
        "available": available,
        "notes": "" if available else "optional dependency unavailable; fallback used",
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
                        "dense_iou_mean": safe_mean(subset, "dense_iou"),
                        "dense_dice_mean": safe_mean(subset, "dense_dice"),
                        "dense_area_error_mean": safe_mean(subset, "dense_area_error"),
                        "geometry_iou_mean": safe_mean(subset, "geometry_iou"),
                        "geometry_dice_mean": safe_mean(subset, "geometry_dice"),
                        "geometry_area_error_mean": safe_mean(subset, "geometry_area_error"),
                        "angle_mae_deg": safe_mean([row for row in subset if row["defect_type"] == "rotated_rect"], "angle_abs_error_deg"),
                        "empty_count": int(sum(float(row["empty_prediction"]) > 0 for row in subset)),
                        "fallback_count": int(sum(float(row["fallback_used"]) > 0 for row in subset)),
                    }
                )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    dense_args = argparse.Namespace(
        npz=args.npz,
        labels=args.labels,
        summary=args.dense_summary,
        metrics=args.dense_metrics,
        epoch_log=args.dense_epoch_log,
        group_summary=args.dense_group_summary,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cpu=args.cpu,
    )
    bundle, _dense_rows, _epoch_rows = dense_init.train_dense_initializer(dense_args, write_outputs=False)
    predictions = dense_predictions(bundle, args.batch_size)
    train_median_depth = float(np.median(bundle.arrays["raw_geom"][bundle.arrays["split_indices"]["train"], 4]))
    all_candidate_rows: list[dict[str, Any]] = []
    candidate_summary_rows: list[dict[str, Any]] = []
    by_spec: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for method, threshold in candidate_specs(bundle.selected_threshold):
        rows: list[dict[str, Any]] = []
        for split in ["train", "val", "test"]:
            rows.extend(
                rows_for_spec(
                    split,
                    predictions[split],
                    bundle.arrays,
                    method,
                    threshold,
                    train_median_depth,
                    bundle.selected_threshold,
                )
            )
        by_spec[(method, threshold)] = rows
        available = not (method.startswith("min_area") and cv2 is None)
        for split in ["train", "val", "test"]:
            candidate_summary_rows.append(summarize_candidate(rows, method, threshold, split, available))
        all_candidate_rows.extend(rows)

    val_rows = [row for row in candidate_summary_rows if row["split"] == "val"]
    selected = sorted(
        val_rows,
        key=lambda row: (
            float(row["score"]),
            -float(row["fallback_count"]),
            -float(row["angle_mae_deg"]) if np.isfinite(float(row["angle_mae_deg"])) else -999.0,
        ),
        reverse=True,
    )[0]
    selected_key = (str(selected["method"]), float(selected["extraction_threshold"]))
    selected_rows = by_spec[selected_key]
    group_rows = build_group_rows(selected_rows)
    write_csv(args.candidates, candidate_summary_rows, CANDIDATE_FIELDS)
    write_csv(args.selected_geometry, selected_rows, SELECTED_FIELDS)
    write_csv(args.group_summary, group_rows, GROUP_FIELDS)
    write_summary(args, bundle, candidate_summary_rows, selected, selected_rows)
    return {
        "selected_method": selected_key[0],
        "selected_threshold": selected_key[1],
        "selected_val_score": float(selected["score"]),
    }


def split_stats(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split]
    return {
        "dense_iou": safe_mean(subset, "dense_iou"),
        "dense_dice": safe_mean(subset, "dense_dice"),
        "dense_area_error": safe_mean(subset, "dense_area_error"),
        "geometry_iou": safe_mean(subset, "geometry_iou"),
        "geometry_dice": safe_mean(subset, "geometry_dice"),
        "geometry_area_error": safe_mean(subset, "geometry_area_error"),
        "angle_mae": safe_mean([row for row in subset if row["defect_type"] == "rotated_rect"], "angle_abs_error_deg"),
        "empty": int(sum(float(row["empty_prediction"]) > 0 for row in subset)),
        "fallback": int(sum(float(row["fallback_used"]) > 0 for row in subset)),
    }


def write_summary(
    args: argparse.Namespace,
    bundle: dense_init.DenseInitializerBundle,
    candidate_summary_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    selected_rows: list[dict[str, Any]],
) -> None:
    stats = {split: split_stats(selected_rows, split) for split in ["train", "val", "test"]}
    test = stats["test"]
    improves_2053 = test["geometry_iou"] >= 0.5652 + 0.02 or test["geometry_dice"] >= 0.7169 + 0.015
    area_ok = test["geometry_area_error"] <= 0.3804 + 1e-6
    top_val = sorted(
        [row for row in candidate_summary_rows if row["split"] == "val"],
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:5]
    lines = [
        "COMSOL rect/rot improved dense-to-geometry proposal extraction summary",
        "",
        "Dense predictions source: in-memory strong dense initializer from scripts/train_comsol_rect_rot_strong_dense_initializer.py.",
        "No checkpoint or prediction tensor is written.",
        "Extraction inputs: dense probability map, validation-selected dense threshold, mask grid coordinates, and train-only median depth.",
        "True mask / true geometry are used only for validation method selection and final metrics.",
        "",
        f"Dense selected threshold: {bundle.selected_threshold}",
        f"Dense best epoch / val score: {bundle.best_epoch} / {bundle.best_val['score']:.6f}",
        f"cv2 minAreaRect available: {cv2 is not None}",
        "",
        "Top validation extraction candidates:",
    ]
    for row in top_val:
        lines.append(
            f"- {row['method']} @ {float(row['extraction_threshold']):.2f}: "
            f"IoU/Dice/area/score = {float(row['geometry_iou_mean']):.4f} / "
            f"{float(row['geometry_dice_mean']):.4f} / {float(row['geometry_area_error_mean']):.4f} / "
            f"{float(row['score']):.4f}"
        )
    lines.extend(
        [
            "",
            f"Selected method: {selected['method']} @ threshold {float(selected['extraction_threshold']):.2f}",
            f"Selected by validation score IoU+Dice-area_error = {float(selected['score']):.6f}",
            "",
            "Selected method train/val/test metrics:",
        ]
    )
    for split in ["train", "val", "test"]:
        s = stats[split]
        lines.append(
            f"- {split} dense IoU/Dice/area = {s['dense_iou']:.4f} / {s['dense_dice']:.4f} / {s['dense_area_error']:.4f}; "
            f"geometry IoU/Dice/area = {s['geometry_iou']:.4f} / {s['geometry_dice']:.4f} / {s['geometry_area_error']:.4f}; "
            f"angle MAE = {s['angle_mae']:.4f}; empty/fallback = {s['empty']} / {s['fallback']}"
        )
    lines.extend(
        [
            "",
            "Stage C acceptance check:",
            "- 20.53 extracted geometry reference test IoU/Dice/area = 0.5652 / 0.7169 / 0.3804",
            f"- improved over 20.53 by required margin: {improves_2053}",
            f"- area_error not worse than 20.53: {area_ok}",
            f"- usable for refinement: {improves_2053 and area_ok}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=base.DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=base.DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--selected-geometry", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--dense-summary", type=Path, default=dense_init.DEFAULT_SUMMARY)
    parser.add_argument("--dense-metrics", type=Path, default=dense_init.DEFAULT_METRICS)
    parser.add_argument("--dense-epoch-log", type=Path, default=dense_init.DEFAULT_EPOCH_LOG)
    parser.add_argument("--dense-group-summary", type=Path, default=dense_init.DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

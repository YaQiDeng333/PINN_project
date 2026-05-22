from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402
import train_comsol_rect_rot_strong_dense_initializer as dense_init  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = base.DEFAULT_NPZ
DEFAULT_LABELS = base.DEFAULT_LABELS
DEFAULT_INPUT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_basis_input_check_summary.txt"
DEFAULT_INPUT_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_input_check.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_basis_extraction_summary.txt"
DEFAULT_CANDIDATES = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_extraction_candidates.csv"
DEFAULT_SELECTED = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_selected_profiles.csv"
DEFAULT_GROUP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_extraction_group_summary.csv"

K_STATIONS = 8
TEMPERATURE_M = 5.0e-4
MAIN_TYPES = {"rectangular_notch", "rotated_rect"}
METHODS = ["P1_hardmask_profile", "P2_prob_weighted_profile", "P3_hybrid_profile"]

INPUT_FIELDS = ["check", "status", "value", "notes"]
CANDIDATE_FIELDS = [
    "method",
    "split",
    "sample_count",
    "dense_iou_mean",
    "dense_dice_mean",
    "dense_area_error_mean",
    "profile_iou_mean",
    "profile_dice_mean",
    "profile_area_error_mean",
    "roughness_penalty",
    "component_count_mean",
    "fallback_rate",
    "score",
    "available",
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
    "profile_iou_mean",
    "profile_dice_mean",
    "profile_area_error_mean",
    "roughness_penalty",
    "component_count_mean",
    "fallback_rate",
]


def profile_fields() -> list[str]:
    fields = [
        "sample_id",
        "source_index",
        "split",
        "defect_type",
        "source_pack",
        "method",
        "selected_method",
        "dense_threshold",
        "profile_threshold",
        "k_stations",
        "temperature_m",
        "center_x",
        "center_y",
        "angle_rad",
        "angle_deg",
        "length",
        "depth_proxy",
        "dense_iou",
        "dense_dice",
        "dense_area_error",
        "dense_center_error_px",
        "dense_pred_area",
        "true_area",
        "profile_iou",
        "profile_dice",
        "profile_area_error",
        "profile_center_error_px",
        "profile_pred_area",
        "component_count",
        "fallback_used",
        "roughness_penalty",
        "area_from_profile_params",
        "notes",
    ]
    fields += [f"u_station_{i}" for i in range(K_STATIONS)]
    fields += [f"half_width_{i}" for i in range(K_STATIONS)]
    fields += [f"center_offset_{i}" for i in range(K_STATIONS)]
    fields += [f"occupancy_{i}" for i in range(K_STATIONS)]
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract K=8 profile/basis masks from COMSOL rect/rot dense predictions.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT_SUMMARY)
    parser.add_argument("--input-check", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            v = float(row[key])
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else math.nan


def metric(prob: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = prob >= threshold
    true = true_mask > 0
    inter = int(np.logical_and(pred, true).sum())
    union = int(np.logical_or(pred, true).sum())
    pred_area = int(pred.sum())
    true_area = int(true.sum())
    iou = inter / union if union else 1.0
    dice = 2.0 * inter / (pred_area + true_area) if (pred_area + true_area) else 1.0
    area_error = abs(pred_area - true_area) / max(true_area, 1)
    if pred_area and true_area:
        py, px = np.where(pred)
        ty, tx = np.where(true)
        center_error = math.hypot(float(px.mean() - tx.mean()), float(py.mean() - ty.mean()))
    else:
        center_error = math.nan
    return {
        "iou": float(iou),
        "dice": float(dice),
        "area_error": float(area_error),
        "center_error_px": float(center_error),
        "pred_area": float(pred_area),
        "true_area": float(true_area),
    }


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, bool]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask, 0, True
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    component_count = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            component_count += 1
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out, component_count, False


def top_region(prob: np.ndarray) -> np.ndarray:
    flat = prob.reshape(-1)
    count = max(16, int(round(flat.size * 0.035)))
    cutoff = np.partition(flat, -count)[-count]
    return prob >= cutoff


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 1.0e-9)
    cdf = np.cumsum(w)
    cutoff = percentile / 100.0 * cdf[-1]
    return float(v[min(np.searchsorted(cdf, cutoff, side="left"), v.size - 1)])


def pca_frame(coords: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    if weights is None:
        center = coords.mean(axis=0)
        centered = coords - center
        cov = centered.T @ centered / max(coords.shape[0] - 1, 1)
    else:
        weights = np.maximum(weights.astype(np.float64), 1.0e-9)
        center = np.average(coords, axis=0, weights=weights)
        centered = coords - center
        cov = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1.0e-9)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.atan2(float(major[1]), float(major[0]))
    angle = (angle + math.pi / 2.0) % math.pi - math.pi / 2.0
    return center.astype(np.float64), float(angle)


def collect_points(
    prob: np.ndarray,
    threshold: float,
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    if mode == "hard":
        support = prob >= threshold
    elif mode == "prob":
        support = prob >= max(0.05, threshold - 0.25)
    else:
        hard, _count, empty = largest_component(prob >= threshold)
        if empty:
            hard = top_region(prob)
        support = hard | (prob >= max(0.05, threshold - 0.20))
    fallback = False
    if not support.any():
        support = top_region(prob)
        fallback = True
    component, count, empty = largest_component(support)
    if empty:
        component = top_region(prob)
        count = 1
        fallback = True
    ys, xs = np.where(component)
    coords = np.stack([mask_x[xs], mask_y[ys]], axis=1).astype(np.float64)
    if mode == "hard":
        weights = np.ones(coords.shape[0], dtype=np.float64)
    else:
        weights = np.maximum(prob[ys, xs].astype(np.float64), 1.0e-6)
    return coords, weights, count, fallback


def station_profile(
    coords: np.ndarray,
    weights: np.ndarray,
    center: np.ndarray,
    angle: float,
    method: str,
) -> dict[str, np.ndarray | float]:
    axis_u = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    axis_v = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    rel = coords - center
    u = rel @ axis_u
    v = rel @ axis_v
    if method == "P2_prob_weighted_profile":
        u_min = weighted_percentile(u, weights, 3.0)
        u_max = weighted_percentile(u, weights, 97.0)
    else:
        u_min = float(np.percentile(u, 2.0))
        u_max = float(np.percentile(u, 98.0))
    if u_max <= u_min + 1.0e-6:
        u_min, u_max = float(u.min()), float(u.max() + 1.0e-6)
    stations = np.linspace(u_min, u_max, K_STATIONS)
    edges = np.linspace(u_min, u_max, K_STATIONS + 1)
    half_widths = np.zeros(K_STATIONS, dtype=np.float64)
    offsets = np.zeros(K_STATIONS, dtype=np.float64)
    occupancy = np.zeros(K_STATIONS, dtype=np.float64)
    for i in range(K_STATIONS):
        if i == K_STATIONS - 1:
            sel = (u >= edges[i]) & (u <= edges[i + 1])
        else:
            sel = (u >= edges[i]) & (u < edges[i + 1])
        if not np.any(sel):
            half_widths[i] = np.nan
            offsets[i] = np.nan
            occupancy[i] = 0.0
            continue
        vv = v[sel]
        ww = weights[sel]
        occupancy[i] = float(np.clip(ww.mean(), 0.0, 1.0))
        offsets[i] = weighted_percentile(vv, ww, 50.0)
        centered_v = vv - offsets[i]
        if method == "P1_hardmask_profile":
            hw = float(np.percentile(np.abs(centered_v), 98.0))
        else:
            hw = weighted_percentile(np.abs(centered_v), ww, 92.0)
        half_widths[i] = max(hw, 2.5e-4)
    finite = np.isfinite(half_widths)
    if not finite.any():
        half_widths[:] = 5.0e-4
        offsets[:] = 0.0
    else:
        half_widths[~finite] = np.interp(stations[~finite], stations[finite], half_widths[finite])
        off_finite = np.isfinite(offsets)
        offsets[~off_finite] = np.interp(stations[~off_finite], stations[off_finite], offsets[off_finite]) if off_finite.any() else 0.0
    half_widths = np.clip(half_widths, 2.5e-4, 0.015)
    offsets = np.clip(offsets, -0.004, 0.004)
    length = float(u_max - u_min)
    roughness = float(np.mean(np.diff(half_widths, n=2) ** 2)) if K_STATIONS >= 3 else 0.0
    return {
        "u_stations": stations.astype(np.float64),
        "half_widths": half_widths.astype(np.float64),
        "center_offsets": offsets.astype(np.float64),
        "occupancy": occupancy.astype(np.float64),
        "length": max(length, 1.0e-3),
        "roughness": roughness,
    }


def rasterize_profile_np(
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    center_x: float,
    center_y: float,
    angle_rad: float,
    u_stations: np.ndarray,
    half_widths: np.ndarray,
    center_offsets: np.ndarray,
    temperature: float = TEMPERATURE_M,
) -> np.ndarray:
    xg, yg = np.meshgrid(mask_x, mask_y)
    dx = xg - center_x
    dy = yg - center_y
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    hw = np.interp(u, u_stations, half_widths, left=half_widths[0], right=half_widths[-1])
    off = np.interp(u, u_stations, center_offsets, left=center_offsets[0], right=center_offsets[-1])
    length_gate = np.minimum(u - u_stations[0], u_stations[-1] - u)
    width_gate = hw - np.abs(v - off)
    logits = np.minimum(length_gate, width_gate) / temperature
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def extract_one(
    prob: np.ndarray,
    method: str,
    threshold: float,
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    depth_proxy: float,
) -> dict[str, Any]:
    mode = "hard" if method == "P1_hardmask_profile" else "prob" if method == "P2_prob_weighted_profile" else "hybrid"
    coords, weights, component_count, fallback = collect_points(prob, threshold, mask_x, mask_y, mode)
    use_weights = weights if method != "P1_hardmask_profile" else None
    center, angle = pca_frame(coords, use_weights)
    prof = station_profile(coords, weights, center, angle, method)
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "angle_rad": float(angle),
        "angle_deg": math.degrees(float(angle)),
        "length": float(prof["length"]),
        "depth_proxy": float(depth_proxy),
        "u_stations": prof["u_stations"],
        "half_widths": prof["half_widths"],
        "center_offsets": prof["center_offsets"],
        "occupancy": prof["occupancy"],
        "component_count": int(component_count),
        "fallback_used": float(fallback),
        "roughness_penalty": float(prof["roughness"]),
        "area_from_profile_params": float(np.trapz(2.0 * prof["half_widths"], prof["u_stations"])),
    }


def dense_predictions(bundle: dense_init.DenseInitializerBundle, args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        ds = dense_init.DenseMaskDataset(bundle.arrays["split_indices"][split], bundle.arrays)
        out[split] = dense_init.predict(bundle.model, ds, bundle.device, args.batch_size)
    return out


def profile_rows_for_method(
    method: str,
    bundle: dense_init.DenseInitializerBundle,
    preds: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    arrays = bundle.arrays
    mask_x = arrays["mask_x"].astype(np.float64)
    mask_y = arrays["mask_y"].astype(np.float64)
    threshold = float(bundle.selected_threshold)
    # A single train-derived physical-depth proxy is used only for the optional
    # profile-to-forward-surrogate summary path. It is not fitted per sample and
    # is not used for profile mask rasterization.
    train_depths = arrays["raw_geom"][arrays["split_indices"]["train"], 4]
    depth_proxy = float(np.median(train_depths))
    rows: list[dict[str, Any]] = []
    for split, pred in preds.items():
        for order, local_idx_raw in enumerate(pred["indices"]):
            idx = int(local_idx_raw)
            prob = pred["prob"][order]
            dense_m = metric(prob, arrays["masks"][idx], threshold)
            profile = extract_one(prob, method, threshold, mask_x, mask_y, depth_proxy)
            prof_prob = rasterize_profile_np(
                mask_x,
                mask_y,
                profile["center_x"],
                profile["center_y"],
                profile["angle_rad"],
                profile["u_stations"],
                profile["half_widths"],
                profile["center_offsets"],
            )
            prof_m = metric(prof_prob, arrays["masks"][idx], 0.5)
            row: dict[str, Any] = {
                "sample_id": str(arrays["sample_ids"][idx]),
                "source_index": int(arrays["source_indices"][idx]),
                "split": split,
                "defect_type": str(arrays["defect_types"][idx]),
                "source_pack": str(arrays["source_packs"][idx]),
                "method": method,
                "selected_method": "",
                "dense_threshold": threshold,
                "profile_threshold": 0.5,
                "k_stations": K_STATIONS,
                "temperature_m": TEMPERATURE_M,
                "center_x": profile["center_x"],
                "center_y": profile["center_y"],
                "angle_rad": profile["angle_rad"],
                "angle_deg": profile["angle_deg"],
                "length": profile["length"],
                "depth_proxy": profile["depth_proxy"],
                "dense_iou": dense_m["iou"],
                "dense_dice": dense_m["dice"],
                "dense_area_error": dense_m["area_error"],
                "dense_center_error_px": dense_m["center_error_px"],
                "dense_pred_area": dense_m["pred_area"],
                "true_area": dense_m["true_area"],
                "profile_iou": prof_m["iou"],
                "profile_dice": prof_m["dice"],
                "profile_area_error": prof_m["area_error"],
                "profile_center_error_px": prof_m["center_error_px"],
                "profile_pred_area": prof_m["pred_area"],
                "component_count": profile["component_count"],
                "fallback_used": profile["fallback_used"],
                "roughness_penalty": profile["roughness_penalty"],
                "area_from_profile_params": profile["area_from_profile_params"],
                "notes": "",
            }
            for i in range(K_STATIONS):
                row[f"u_station_{i}"] = float(profile["u_stations"][i])
                row[f"half_width_{i}"] = float(profile["half_widths"][i])
                row[f"center_offset_{i}"] = float(profile["center_offsets"][i])
                row[f"occupancy_{i}"] = float(profile["occupancy"][i])
            rows.append(row)
    return rows


def candidate_summary(rows: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
    subset = [row for row in rows if row["method"] == method and row["split"] == split]
    fallback_rate = safe_mean(subset, "fallback_used")
    rough = safe_mean(subset, "roughness_penalty")
    profile_iou = safe_mean(subset, "profile_iou")
    profile_dice = safe_mean(subset, "profile_dice")
    area_error = safe_mean(subset, "profile_area_error")
    score = profile_iou + profile_dice - area_error - 0.05 * fallback_rate - 0.02 * rough
    return {
        "method": method,
        "split": split,
        "sample_count": len(subset),
        "dense_iou_mean": safe_mean(subset, "dense_iou"),
        "dense_dice_mean": safe_mean(subset, "dense_dice"),
        "dense_area_error_mean": safe_mean(subset, "dense_area_error"),
        "profile_iou_mean": profile_iou,
        "profile_dice_mean": profile_dice,
        "profile_area_error_mean": area_error,
        "roughness_penalty": rough,
        "component_count_mean": safe_mean(subset, "component_count"),
        "fallback_rate": fallback_rate,
        "score": score,
        "available": True,
        "notes": "",
    }


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        for group_name, values in [
            ("overall", ["rect_rot"]),
            ("defect_type", sorted({row["defect_type"] for row in split_rows})),
        ]:
            for value in values:
                subset = split_rows if group_name == "overall" else [row for row in split_rows if row[group_name] == value]
                out.append(
                    {
                        "split": split,
                        "group_name": group_name,
                        "group_value": value,
                        "sample_count": len(subset),
                        "dense_iou_mean": safe_mean(subset, "dense_iou"),
                        "dense_dice_mean": safe_mean(subset, "dense_dice"),
                        "dense_area_error_mean": safe_mean(subset, "dense_area_error"),
                        "profile_iou_mean": safe_mean(subset, "profile_iou"),
                        "profile_dice_mean": safe_mean(subset, "profile_dice"),
                        "profile_area_error_mean": safe_mean(subset, "profile_area_error"),
                        "roughness_penalty": safe_mean(subset, "roughness_penalty"),
                        "component_count_mean": safe_mean(subset, "component_count"),
                        "fallback_rate": safe_mean(subset, "fallback_used"),
                    }
                )
    return out


def input_check(args: argparse.Namespace, bundle: dense_init.DenseInitializerBundle, metric_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    diagnostics = bundle.diagnostics
    stats = {split: dense_init.split_stats(metric_rows, split) for split in ["train", "val", "test"]}
    rows.extend(
        [
            {
                "check": "rect_rot_subset_count",
                "status": "pass" if diagnostics["n_rect_rot"] == 400 else "fail",
                "value": diagnostics["n_rect_rot"],
                "notes": "",
            },
            {
                "check": "rect_rot_split",
                "status": "pass" if diagnostics["split_counts"] == {"train": 268, "val": 66, "test": 66} else "warn",
                "value": diagnostics["split_counts"],
                "notes": "",
            },
            {
                "check": "dense_initializer_test_iou",
                "status": "pass" if stats["test"]["iou"] >= 0.60 else "fail",
                "value": stats["test"]["iou"],
                "notes": "Dense initializer is a proposal generator only, not a baseline.",
            },
            {
                "check": "polygon_excluded",
                "status": "pass" if set(diagnostics["type_counts"].keys()) <= MAIN_TYPES else "fail",
                "value": diagnostics["type_counts"],
                "notes": "",
            },
        ]
    )
    write_csv(args.input_check, rows, INPUT_FIELDS)
    failed = [row for row in rows if row["status"] == "fail"]
    lines = [
        "COMSOL rect/rot profile basis input check summary",
        "",
        "No COMSOL run and no new data generation. Dense initializer is retrained in memory using 20.54 protocol only as proposal generator.",
        f"Input NPZ: {args.npz}",
        f"Rect+rot subset / split: {diagnostics['n_rect_rot']} / {diagnostics['split_counts']}",
        f"Dense initializer selected threshold: {bundle.selected_threshold}",
        f"Dense train/val/test IoU/Dice/area_error: "
        f"{stats['train']['iou']:.4f}/{stats['train']['dice']:.4f}/{stats['train']['area_error']:.4f}; "
        f"{stats['val']['iou']:.4f}/{stats['val']['dice']:.4f}/{stats['val']['area_error']:.4f}; "
        f"{stats['test']['iou']:.4f}/{stats['test']['dice']:.4f}/{stats['test']['area_error']:.4f}",
        f"Input check passed: {not failed}",
    ]
    args.input_summary.parent.mkdir(parents=True, exist_ok=True)
    args.input_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"Input check failed: {[row['check'] for row in failed]}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dense_args = argparse.Namespace(
        npz=args.npz,
        labels=args.labels,
        summary=PROJECT_ROOT / "results/summaries/_profile_basis_internal_dense_summary.txt",
        metrics=PROJECT_ROOT / "results/metrics/_profile_basis_internal_dense_metrics.csv",
        epoch_log=PROJECT_ROOT / "results/metrics/_profile_basis_internal_dense_epoch_log.csv",
        group_summary=PROJECT_ROOT / "results/metrics/_profile_basis_internal_dense_group.csv",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cpu=args.cpu,
    )
    bundle, dense_metric_rows, _epoch_rows = dense_init.train_dense_initializer(dense_args, write_outputs=False)
    input_check(args, bundle, dense_metric_rows)
    preds = dense_predictions(bundle, args)
    all_rows: list[dict[str, Any]] = []
    for method in METHODS:
        all_rows.extend(profile_rows_for_method(method, bundle, preds))
    candidate_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for split in ["train", "val", "test"]:
            candidate_rows.append(candidate_summary(all_rows, method, split))
    val_rows = [row for row in candidate_rows if row["split"] == "val" and row["available"]]
    selected_method = max(val_rows, key=lambda row: float(row["score"]))["method"]
    selected_rows = [row for row in all_rows if row["method"] == selected_method]
    for row in selected_rows:
        row["selected_method"] = selected_method
    group_rows = group_summary(selected_rows)
    write_csv(args.candidates, candidate_rows, CANDIDATE_FIELDS)
    write_csv(args.selected, selected_rows, profile_fields())
    write_csv(args.group_summary, group_rows, GROUP_FIELDS)

    test_group = next(row for row in group_rows if row["split"] == "test" and row["group_name"] == "overall")
    dense_test = (test_group["dense_iou_mean"], test_group["dense_dice_mean"], test_group["dense_area_error_mean"])
    prof_test = (test_group["profile_iou_mean"], test_group["profile_dice_mean"], test_group["profile_area_error_mean"])
    depth_proxy = float(np.median(bundle.arrays["raw_geom"][bundle.arrays["split_indices"]["train"], 4]))
    usable = (
        prof_test[0] >= dense_test[0] - 0.02
        and prof_test[1] >= dense_test[1] - 0.015
        and prof_test[2] <= dense_test[2] + 0.08
        and test_group["fallback_rate"] < 0.10
    )
    lines = [
        "COMSOL rect/rot profile basis extraction summary",
        "",
        "Scope: profile/basis proposal extraction from predicted dense mask/probability only; true mask is used only for validation selection and final metrics.",
        f"Dense initializer source: in-memory 20.54 strong dense initializer protocol; selected threshold={bundle.selected_threshold}; no checkpoint written.",
        f"Depth proxy: train-only raw-geometry median physical depth = {depth_proxy:.6g} m; used only for profile-to-forward-surrogate compatibility, not mask extraction.",
        f"Candidate methods: {METHODS}",
        f"Selected method by validation score: {selected_method}",
        "",
        "Candidate validation summary:",
    ]
    for row in val_rows:
        lines.append(
            f"- {row['method']}: IoU/Dice/area_error={row['profile_iou_mean']:.4f}/{row['profile_dice_mean']:.4f}/{row['profile_area_error_mean']:.4f}, "
            f"fallback={row['fallback_rate']:.4f}, score={row['score']:.4f}"
        )
    lines.extend(
        [
            "",
            f"Selected train dense/profile IoU/Dice/area_error: "
            f"{next(r for r in group_rows if r['split']=='train' and r['group_name']=='overall')['dense_iou_mean']:.4f}/"
            f"{next(r for r in group_rows if r['split']=='train' and r['group_name']=='overall')['dense_dice_mean']:.4f}/"
            f"{next(r for r in group_rows if r['split']=='train' and r['group_name']=='overall')['dense_area_error_mean']:.4f} -> "
            f"{next(r for r in group_rows if r['split']=='train' and r['group_name']=='overall')['profile_iou_mean']:.4f}/"
            f"{next(r for r in group_rows if r['split']=='train' and r['group_name']=='overall')['profile_dice_mean']:.4f}/"
            f"{next(r for r in group_rows if r['split']=='train' and r['group_name']=='overall')['profile_area_error_mean']:.4f}",
            f"Selected val dense/profile IoU/Dice/area_error: "
            f"{next(r for r in group_rows if r['split']=='val' and r['group_name']=='overall')['dense_iou_mean']:.4f}/"
            f"{next(r for r in group_rows if r['split']=='val' and r['group_name']=='overall')['dense_dice_mean']:.4f}/"
            f"{next(r for r in group_rows if r['split']=='val' and r['group_name']=='overall')['dense_area_error_mean']:.4f} -> "
            f"{next(r for r in group_rows if r['split']=='val' and r['group_name']=='overall')['profile_iou_mean']:.4f}/"
            f"{next(r for r in group_rows if r['split']=='val' and r['group_name']=='overall')['profile_dice_mean']:.4f}/"
            f"{next(r for r in group_rows if r['split']=='val' and r['group_name']=='overall')['profile_area_error_mean']:.4f}",
            f"Selected test dense/profile IoU/Dice/area_error: {dense_test[0]:.4f}/{dense_test[1]:.4f}/{dense_test[2]:.4f} -> {prof_test[0]:.4f}/{prof_test[1]:.4f}/{prof_test[2]:.4f}",
            f"Profile extraction usable for refinement gate: {usable}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not usable:
        raise RuntimeError("Selected profile projection degrades dense mask too much; stop before refinement.")
    return {"selected_method": selected_method, "group_rows": group_rows}


def main() -> None:
    result = run(parse_args())
    print(f"Selected profile method: {result['selected_method']}")


if __name__ == "__main__":
    main()

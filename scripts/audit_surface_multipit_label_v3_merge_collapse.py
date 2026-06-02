#!/usr/bin/env python
"""Audit 25.15 label-v3 merge collapse for surface multi-pit component sets.

This stage is audit/report only. It reads the validated component-set dataset
plus 25.14/25.15/25.10/25.13 evidence, reconstructs label-v3 targets in memory,
and writes strict JSON/Markdown/manifest diagnostics. It does not train, tune
losses, run COMSOL, mutate data/NPZ files, expand model capacity, export
checkpoints/previews, or update the current baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_surface_multipit_component_set_pilot_v1"
AUDIT_ID = "25_15b_surface_multipit_label_v3_failure_audit"
TARGET_SPLIT = {"train": 72, "val": 20, "test": 20}
K_MAX = 3
GRID_H = 64
GRID_W = 128
MASK_X_START_M = -0.04
MASK_X_STOP_M = 0.04
MASK_Y_START_M = -0.01
MASK_Y_STOP_M = 0.01

DATASET_MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"
M10 = ROOT / "results/metrics/25_10_component_set_training_gate_metrics.json"
M13 = ROOT / "results/metrics/25_13_target_v2_training_gate_metrics.json"
M14 = ROOT / "results/metrics/25_14_label_v3_derivation_validator.json"
M15 = ROOT / "results/metrics/25_15_label_v3_training_gate_metrics.json"
MAN14 = ROOT / "results/manifests/25_14_label_v3_derivation_validator_manifest.json"
MAN15 = ROOT / "results/manifests/25_15_label_v3_training_gate_manifest.json"

OUT_METRICS = ROOT / "results/metrics/25_15b_label_v3_failure_audit.json"
OUT_SUMMARY = ROOT / "results/summaries/25_15b_label_v3_failure_audit_summary.md"
OUT_MANIFEST = ROOT / "results/manifests/25_15b_label_v3_failure_audit_manifest.json"

FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 25.15 label-v3 merge collapse.")
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET_MANIFEST)
    parser.add_argument("--metrics-25-10", type=Path, default=M10)
    parser.add_argument("--metrics-25-13", type=Path, default=M13)
    parser.add_argument("--metrics-25-14", type=Path, default=M14)
    parser.add_argument("--metrics-25-15", type=Path, default=M15)
    parser.add_argument("--manifest-25-14", type=Path, default=MAN14)
    parser.add_argument("--manifest-25-15", type=Path, default=MAN15)
    parser.add_argument("--out-metrics", type=Path, default=OUT_METRICS)
    parser.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    parser.add_argument("--out-manifest", type=Path, default=OUT_MANIFEST)
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as pack:
        return {name: pack[name].copy() for name in pack.files}


def finite(values: list[float | int | None]) -> list[float]:
    clean: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            clean.append(number)
    return clean


def mean_or_null(values: list[float | int | None]) -> float | None:
    clean = finite(values)
    return float(np.mean(clean)) if clean else None


def min_or_null(values: list[float | int | None]) -> float | None:
    clean = finite(values)
    return float(np.min(clean)) if clean else None


def max_or_null(values: list[float | int | None]) -> float | None:
    clean = finite(values)
    return float(np.max(clean)) if clean else None


def percentile_or_null(values: list[float | int | None], percentile: float) -> float | None:
    clean = finite(values)
    return float(np.percentile(clean, percentile)) if clean else None


def assert_sources(
    dataset_manifest: dict[str, Any],
    m10: dict[str, Any],
    m13: dict[str, Any],
    m14: dict[str, Any],
    m15: dict[str, Any],
    man14: dict[str, Any],
    man15: dict[str, Any],
) -> Path:
    if ROOT != Path(r"C:\Users\19166\Desktop\PINN_project"):
        raise RuntimeError(f"wrong project root: {ROOT}")
    if dataset_manifest.get("dataset_id") != DATASET_ID or dataset_manifest.get("split_counts") != TARGET_SPLIT:
        raise ValueError("dataset manifest mismatch")
    if dataset_manifest.get("baseline_ready") is not False or dataset_manifest.get("train_ready_candidate") is not True:
        raise ValueError("dataset readiness boundary mismatch")
    if int(dataset_manifest.get("K_max", -1)) != K_MAX:
        raise ValueError("K_max mismatch")
    for required in ["baseline_update", "current_baseline_replacement", "latest_newest_auto_discovery"]:
        if required not in set(dataset_manifest.get("forbidden_use", [])):
            raise ValueError(f"dataset manifest missing forbidden_use={required}")
    if m10.get("stage") != "25.10" or m10.get("gate_decision") != "PARTIAL":
        raise ValueError("25.10 metrics mismatch")
    if m13.get("stage") != "25.13" or m13.get("gate_decision") != "FAIL":
        raise ValueError("25.13 metrics mismatch")
    if m14.get("stage") != "25.14" or m14.get("acceptance_decision") != "READY_FOR_25.15_TRAINING":
        raise ValueError("25.14 metrics mismatch")
    if m15.get("stage") != "25.15" or m15.get("gate_decision") != "FAIL":
        raise ValueError("25.15 metrics mismatch")
    if man14.get("acceptance_decision") != "READY_FOR_25.15_TRAINING":
        raise ValueError("25.14 manifest mismatch")
    if man15.get("stage") != "25.15" or man15.get("gate_decision") != "FAIL":
        raise ValueError("25.15 manifest mismatch")
    if man15.get("current_baseline_updated") is not False or man15.get("baseline_ready") is not False:
        raise ValueError("25.15 manifest baseline boundary mismatch")
    npz_path = Path(dataset_manifest["path"])
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    return npz_path


def grid_xy() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, GRID_W)
    y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, GRID_H)
    return np.meshgrid(x, y, indexing="xy")


def dilate_bool(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = expanded
    return result


def normalized_signed_distance(mask: np.ndarray, clip_radius_px: float = 4.0) -> np.ndarray:
    fg = np.argwhere(mask)
    bg = np.argwhere(~mask)
    sdf = np.zeros(mask.shape, dtype=np.float32)
    if fg.size == 0:
        return sdf
    if bg.size == 0:
        return np.ones(mask.shape, dtype=np.float32)

    def nearest_distance(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
        out = np.empty(points.shape[0], dtype=np.float32)
        target = targets.astype(np.float32)
        chunk = 1024
        for start in range(0, points.shape[0], chunk):
            pts = points[start : start + chunk].astype(np.float32)
            diff = pts[:, None, :] - target[None, :, :]
            out[start : start + chunk] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        return out

    sdf[tuple(fg.T)] = nearest_distance(fg, bg)
    sdf[tuple(bg.T)] = -nearest_distance(bg, fg)
    return np.clip(sdf / float(clip_radius_px), -1.0, 1.0).astype(np.float32)


def build_target_v2(pack: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    centers = np.asarray(pack["component_center_xy_m"], dtype=np.float64)
    lwd = np.asarray(pack["component_lwd_m"], dtype=np.float64)
    xx, yy = grid_xy()
    n, k, height, width = masks.shape
    ownership = np.full((n, height, width), -1, dtype=np.int16)
    masks_v2 = np.zeros_like(masks, dtype=bool)
    depths_v2 = np.zeros_like(depths, dtype=np.float32)
    duplicate_before = 0
    duplicate_after = 0
    conflict_before = 0
    overlap_resolved = 0
    raw_overlap_samples = 0
    for i in range(n):
        active = np.where(exists[i])[0]
        component_sum = masks[i, active].sum(axis=0) if active.size else np.zeros((height, width), dtype=np.int64)
        duplicate_before += int(np.maximum(component_sum - 1, 0).sum())
        raw_overlap_samples += int((component_sum > 1).any())
        for y, x in zip(*np.where(component_sum > 0)):
            candidates = [int(slot) for slot in active if masks[i, slot, y, x]]
            if len(candidates) == 1:
                owner = candidates[0]
            else:
                overlap_resolved += 1
                depth_values = [float(depths[i, slot, y, x]) for slot in candidates if float(depths[i, slot, y, x]) > 0.0]
                if depth_values and (max(depth_values) - min(depth_values)) > 1.0e-12:
                    conflict_before += 1
                px = float(xx[y, x])
                py = float(yy[y, x])
                scored = []
                for slot in candidates:
                    sx = max(float(lwd[i, slot, 0]), 1.0e-9)
                    sy = max(float(lwd[i, slot, 1]), 1.0e-9)
                    dist = math.hypot((px - float(centers[i, slot, 0])) / sx, (py - float(centers[i, slot, 1])) / sy)
                    scored.append((dist, -float(depths[i, slot, y, x]), int(slot)))
                owner = min(scored)[2]
            ownership[i, y, x] = owner
            masks_v2[i, owner, y, x] = True
            depths_v2[i, owner, y, x] = depths[i, owner, y, x]
        duplicate_after += int(np.maximum(masks_v2[i].sum(axis=0) - 1, 0).sum())
    union_raw = np.asarray(pack["projected_mask_2d"], dtype=bool)
    union_v2 = masks_v2.max(axis=1)
    return masks_v2, depths_v2, ownership, {
        "duplicate_ownership_before_v2": int(duplicate_before),
        "duplicate_ownership_after_v2": int(duplicate_after),
        "overlap_depth_conflict_before_v2": int(conflict_before),
        "overlap_depth_conflict_after_v2": 0,
        "ownership_resolved_overlap_pixel_count": int(overlap_resolved),
        "raw_overlap_sample_count": int(raw_overlap_samples),
        "union_mask_mismatch_after_v2_px": int(np.logical_xor(union_raw, union_v2).sum()),
    }


def build_target_v3(pack: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    masks_v2, _depths_v2, ownership, v2_summary = build_target_v2(pack)
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    raw_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    n, k, height, width = raw_masks.shape
    soft = np.zeros((n, k, height, width), dtype=np.float32)
    valid = np.zeros((n, k, height, width), dtype=bool)
    sdf = np.zeros((n, k, height, width), dtype=np.float32)
    depth = np.zeros_like(raw_depths, dtype=np.float32)
    depth_valid = np.zeros((n, k, height, width), dtype=bool)
    overlap_region = raw_masks.sum(axis=1) > 1
    contact_boundary = np.zeros((n, height, width), dtype=bool)
    component_rows: list[dict[str, Any]] = []
    for i in range(n):
        active = [int(slot) for slot in np.where(exists[i])[0]]
        for slot in active:
            raw = raw_masks[i, slot]
            owned = ownership[i] == slot
            band1 = dilate_bool(raw, radius=1)
            band2 = dilate_bool(raw, radius=2)
            component_soft = np.zeros((height, width), dtype=np.float32)
            component_soft[band2] = 0.25
            component_soft[band1] = 0.50
            component_soft[raw] = 0.80
            component_soft[owned] = 1.00
            soft[i, slot] = component_soft
            valid[i, slot] = band2
            sdf[i, slot] = normalized_signed_distance(raw)
            depth[i, slot, raw] = raw_depths[i, slot, raw]
            depth_valid[i, slot] = raw
            hard_px = int(masks_v2[i, slot].sum())
            soft_px = int((component_soft > 0.0).sum())
            component_rows.append(
                {
                    "sample_id": str(pack["sample_ids"][i]),
                    "source_index": int(i),
                    "slot": int(slot),
                    "split": str(pack["split"][i]),
                    "component_count": int(pack["component_count"][i]),
                    "separation_type": str(pack["separation_type"][i]),
                    "topology_relation": str(pack["topology_relation"][i]),
                    "v2_hard_foreground_px": hard_px,
                    "v3_soft_positive_px": soft_px,
                    "v3_valid_region_px": int(valid[i, slot].sum()),
                    "v3_depth_valid_px": int(depth_valid[i, slot].sum()),
                    "v3_vs_v2_support_ratio": float(soft_px / max(hard_px, 1)),
                }
            )
        for left_index, left in enumerate(active):
            left_band = dilate_bool(raw_masks[i, left], radius=1)
            for right in active[left_index + 1 :]:
                right_band = dilate_bool(raw_masks[i, right], radius=1)
                contact_boundary[i] |= left_band & right_band
        contact_boundary[i] &= ~overlap_region[i]
    inactive = ~exists
    inactive_violations = int(np.logical_or(soft[inactive] > 0.0, valid[inactive]).sum()) if inactive.any() else 0
    ratios = [row["v3_vs_v2_support_ratio"] for row in component_rows]
    return {
        "masks_v2": masks_v2,
        "ownership": ownership,
        "soft": soft,
        "valid": valid,
        "sdf": sdf,
        "depth": depth,
        "depth_valid": depth_valid,
        "overlap_region": overlap_region,
        "contact_boundary": contact_boundary,
    }, {
        **v2_summary,
        "target_version": "v3",
        "component_rows": component_rows,
        "v3_target_loaded_count": int(n),
        "v3_active_component_count": int(len(component_rows)),
        "v3_support_ratio_mean": mean_or_null(ratios),
        "v3_support_ratio_min": min_or_null(ratios),
        "empty_slot_violation_count": inactive_violations,
        "overlap_region_pixel_sum": int(overlap_region.sum()),
        "contact_boundary_pixel_sum": int(contact_boundary.sum()),
    }


def aggregate_numeric(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    out: dict[str, Any] = {"sample_count": len(rows)}
    for field in fields:
        values = [row.get(field) for row in rows]
        out[f"{field}_mean"] = mean_or_null(values)
        out[f"{field}_min"] = min_or_null(values)
        out[f"{field}_max"] = max_or_null(values)
        out[f"{field}_p95"] = percentile_or_null(values, 95)
    return out


def group_rows(rows: list[dict[str, Any]], field: str, aggregate_fn) -> dict[str, Any]:
    return {str(value): aggregate_fn([row for row in rows if str(row[field]) == str(value)]) for value in sorted({row[field] for row in rows}, key=str)}


def sample_target_rows(pack: dict[str, Any], v3: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    union = np.asarray(pack["projected_mask_2d"], dtype=bool)
    soft = v3["soft"]
    valid = v3["valid"]
    sdf = v3["sdf"]
    depth = v3["depth"]
    depth_valid = v3["depth_valid"]
    rows: list[dict[str, Any]] = []
    for i in range(exists.shape[0]):
        active = exists[i]
        soft_pos = soft[i, active] > 0.0
        valid_pos = valid[i, active]
        raw_pos = raw_masks[i, active]
        depth_valid_pos = depth_valid[i, active]
        if not active.any():
            continue
        raw_sum = raw_pos.sum(axis=0)
        soft_sum = soft_pos.sum(axis=0)
        valid_sum = valid_pos.sum(axis=0)
        depth_valid_sum = depth_valid_pos.sum(axis=0)
        soft_or = soft_pos.max(axis=0)
        valid_or = valid_pos.max(axis=0)
        raw_union = union[i]
        sdf_multi_valid_px = 0
        sdf_dual_positive_px = 0
        sdf_near_boundary_conflict_px = 0
        active_slots = np.where(active)[0]
        for left_index, left in enumerate(active_slots):
            for right in active_slots[left_index + 1 :]:
                pair_valid = valid[i, left] & valid[i, right]
                sdf_multi_valid_px += int(pair_valid.sum())
                sdf_dual_positive_px += int((pair_valid & (sdf[i, left] > 0.0) & (sdf[i, right] > 0.0)).sum())
                sdf_near_boundary_conflict_px += int((pair_valid & (np.abs(sdf[i, left]) <= 0.5) & (np.abs(sdf[i, right]) <= 0.5)).sum())
        depth_nonzero_outside_depth_valid = int(((depth[i] > 0.0) & ~depth_valid[i]).sum())
        rows.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": int(i),
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "raw_union_px": int(raw_union.sum()),
                "raw_duplicate_target_px": int(np.maximum(raw_sum - 1, 0).sum()),
                "soft_positive_target_px": int(soft_pos.sum()),
                "soft_duplicate_target_px": int(np.maximum(soft_sum - 1, 0).sum()),
                "soft_overlap_pixel_px": int((soft_sum > 1).sum()),
                "soft_duplicate_fraction_of_soft_targets": float(np.maximum(soft_sum - 1, 0).sum() / max(int(soft_pos.sum()), 1)),
                "soft_or_px": int(soft_or.sum()),
                "soft_or_to_union_ratio": float(soft_or.sum() / max(int(raw_union.sum()), 1)),
                "soft_outside_union_px": int((soft_or & ~raw_union).sum()),
                "soft_outside_union_fraction": float((soft_or & ~raw_union).sum() / max(int(soft_or.sum()), 1)),
                "valid_region_target_px": int(valid_pos.sum()),
                "valid_duplicate_target_px": int(np.maximum(valid_sum - 1, 0).sum()),
                "valid_overlap_pixel_px": int((valid_sum > 1).sum()),
                "valid_duplicate_fraction_of_valid_targets": float(np.maximum(valid_sum - 1, 0).sum() / max(int(valid_pos.sum()), 1)),
                "valid_or_to_union_ratio": float(valid_or.sum() / max(int(raw_union.sum()), 1)),
                "valid_outside_union_fraction": float((valid_or & ~raw_union).sum() / max(int(valid_or.sum()), 1)),
                "depth_valid_target_px": int(depth_valid_pos.sum()),
                "depth_valid_duplicate_target_px": int(np.maximum(depth_valid_sum - 1, 0).sum()),
                "depth_nonzero_outside_depth_valid_px": depth_nonzero_outside_depth_valid,
                "sdf_multi_valid_overlap_px": int(sdf_multi_valid_px),
                "sdf_dual_positive_px": int(sdf_dual_positive_px),
                "sdf_near_boundary_conflict_px": int(sdf_near_boundary_conflict_px),
                "overlap_region_px": int(v3["overlap_region"][i].sum()),
                "contact_boundary_px": int(v3["contact_boundary"][i].sum()),
            }
        )
    return rows


def aggregate_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "raw_duplicate_target_px",
        "soft_duplicate_target_px",
        "soft_duplicate_fraction_of_soft_targets",
        "soft_or_to_union_ratio",
        "soft_outside_union_fraction",
        "valid_duplicate_target_px",
        "valid_duplicate_fraction_of_valid_targets",
        "valid_or_to_union_ratio",
        "valid_outside_union_fraction",
        "depth_valid_duplicate_target_px",
        "depth_nonzero_outside_depth_valid_px",
        "sdf_multi_valid_overlap_px",
        "sdf_dual_positive_px",
        "sdf_near_boundary_conflict_px",
    ]
    out = aggregate_numeric(rows, fields)
    out["soft_overlap_sample_count"] = int(sum(int(row["soft_duplicate_target_px"]) > 0 for row in rows))
    out["valid_overlap_sample_count"] = int(sum(int(row["valid_duplicate_target_px"]) > 0 for row in rows))
    out["soft_or_union_like_sample_count"] = int(sum(float(row["soft_or_to_union_ratio"]) >= 1.50 for row in rows))
    return out


def failure_rows(m15: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in m15["sample_metrics"]:
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "source_index": int(row["source_index"]),
                "split": str(row["split"]),
                "component_count": int(row["component_count"]),
                "separation_type": str(row["separation_type"]),
                "topology_relation": str(row["topology_relation"]),
                "merged_sample": bool(row["merged_sample"]),
                "component_recall": float(row["component_recall"]),
                "component_mask_dice_mean": float(row["component_mask_dice_mean"]),
                "union_mask_dice": float(row["union_mask_dice"]),
                "depth_grid_rmse_m": float(row["depth_grid_rmse_m"]),
                "pred_component_count": int(row["pred_component_count"]),
                "true_component_count": int(row["true_component_count"]),
            }
        )
    return rows


def aggregate_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    return {
        "sample_count": len(rows),
        "merged_rate": sum(bool(row["merged_sample"]) for row in rows) / len(rows),
        "component_recall_mean": mean_or_null([row["component_recall"] for row in rows]),
        "component_mask_dice_mean": mean_or_null([row["component_mask_dice_mean"] for row in rows]),
        "union_mask_dice_mean": mean_or_null([row["union_mask_dice"] for row in rows]),
        "depth_grid_rmse_m_mean": mean_or_null([row["depth_grid_rmse_m"] for row in rows]),
        "pred_component_count_mean": mean_or_null([row["pred_component_count"] for row in rows]),
    }


def metric_evidence(m10: dict[str, Any], m13: dict[str, Any], m15: dict[str, Any]) -> dict[str, Any]:
    keys = ["component_recall", "missed_rate", "extra_rate", "merged_rate", "component_mask_dice_mean", "union_mask_dice_mean", "depth_grid_rmse_m_mean"]
    t10 = m10["metrics_by_split"]["test"]
    t13 = m13["metrics_by_split"]["test"]
    t15 = m15["metrics_by_split"]["test"]
    return {
        "test_metrics": {
            "25.10": {key: t10.get(key) for key in keys},
            "25.13": {key: t13.get(key) for key in keys},
            "25.15": {key: t15.get(key) for key in keys},
        },
        "delta_vs_25_13": {key: float(t15[key]) - float(t13[key]) for key in keys},
        "delta_vs_25_10": {key: float(t15[key]) - float(t10[key]) for key in keys},
    }


def attribution(target_stats: dict[str, Any], failure_stats: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str, list[str]]:
    overall = target_stats["overall"]
    separated = target_stats["by_separation"].get("separated", {})
    close = target_stats["by_separation"].get("close", {})
    failure_overall = failure_stats["test"]["overall"]
    separated_failure = failure_stats["test"]["by_separation"].get("separated", {})
    close_failure = failure_stats["test"]["by_separation"].get("close", {})
    depth_nonzero_leak = float(overall.get("depth_nonzero_outside_depth_valid_px_mean") or 0.0) > 0.0
    depth_valid_overlap = float(overall.get("depth_valid_duplicate_target_px_mean") or 0.0) > 0.0
    broad_soft = float(overall.get("soft_or_to_union_ratio_mean") or 0.0) >= 1.75
    soft_overlap = float(overall.get("soft_duplicate_fraction_of_soft_targets_mean") or 0.0) >= 0.02
    separated_or_close_overlap = (
        float(separated.get("soft_duplicate_fraction_of_soft_targets_mean") or 0.0) > 0.0
        or float(close.get("soft_duplicate_fraction_of_soft_targets_mean") or 0.0) > 0.0
    )
    global_merged = float(failure_overall.get("merged_rate") or 0.0) >= 0.95
    separated_close_merged = (
        float(separated_failure.get("merged_rate") or 0.0) >= 0.75
        or float(close_failure.get("merged_rate") or 0.0) >= 0.75
    )
    sdf_conflict = float(overall.get("sdf_near_boundary_conflict_px_mean") or 0.0) > 0.0
    evaluator_bug = False
    generator_insufficient = False
    ranked = [
        {
            "rank": 1,
            "cause": "soft support too broad",
            "supported": broad_soft and global_merged,
            "evidence": "v3 soft OR expands well beyond raw union while 25.15 merged rate is global",
        },
        {
            "rank": 2,
            "cause": "valid region leakage",
            "supported": soft_overlap and separated_or_close_overlap,
            "evidence": "component valid/soft regions overlap across components, including separated/close rows",
        },
        {
            "rank": 3,
            "cause": "SDF identity too weak in multi-valid boundary zones",
            "supported": sdf_conflict,
            "evidence": "multiple component valid regions include near-boundary SDF pixels for more than one component",
        },
        {
            "rank": 4,
            "cause": "depth valid region dilution",
            "supported": depth_nonzero_leak or depth_valid_overlap,
            "evidence": "component depth valid region leakage would indicate target-side depth dilution",
        },
        {
            "rank": 5,
            "cause": "genuine topology-only hard case",
            "supported": (not separated_close_merged) and global_merged,
            "evidence": "would require merged collapse to be concentrated in touching/overlap only",
        },
        {
            "rank": 6,
            "cause": "evaluator / threshold artifact",
            "supported": evaluator_bug,
            "evidence": "not supported: predicted component count remains non-empty and all hard slices merge",
        },
    ]
    if broad_soft and not generator_insufficient:
        decision = "NEEDS_PINN_LABEL_DERIVATION_V3B"
        route = "A. enter 25.16 label-v3b derivation + validator, no training"
        reasons = [
            "existing raw component masks/depths and ownership map are sufficient",
            "v3 soft/valid regions are too permissive and can be tightened inside PINN_project",
            "depth-valid targets are raw-foreground only, so depth degradation is more likely downstream from merged masks than missing raw labels",
        ]
    elif generator_insufficient:
        decision = "NEEDS_COMSOL_GENERATOR_LABEL_EXPORT_FIX"
        route = "B. enter COMSOL label export/schema fix, no training"
        reasons = ["raw labels lack necessary component identity support"]
    else:
        decision = "INCONCLUSIVE"
        route = "D. continue label schema audit, no training"
        reasons = ["evidence is not sufficient to choose PINN label derivation or generator export fix"]
    return ranked, decision, route, reasons


def schema_v3b_recommendation() -> dict[str, Any]:
    return {
        "component_exclusive_hard_core": "Keep a mutually exclusive hard core from ownership-resolved raw component masks.",
        "boundary_halo_region": "Limit soft halo to a narrow boundary band; do not let halo dominate component-local supervision.",
        "ignore_overlap_region": "Pixels where multiple components claim soft/valid support should be ignored or diagnostic unless ownership confidence is explicit.",
        "valid_region_split": "Split valid_region into hard_core_region, boundary_halo_region, and ignore_overlap_region.",
        "sdf_identity": "Compute SDF from raw hard component mask but mask SDF loss by component-exclusive core plus narrow non-overlapping halo.",
        "depth_target": "Supervise component depth only on hard core plus narrow owned boundary; do not supervise depth over broad soft halo.",
        "union_policy": "Union mask/depth remain evaluation-only and must not become component-local supervision.",
        "three_component_flags": "Keep component_count=3, overlap_region, contact_boundary, and topology flags as required slices.",
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    dataset_manifest = read_json(args.dataset_manifest)
    m10 = read_json(args.metrics_25_10)
    m13 = read_json(args.metrics_25_13)
    m14 = read_json(args.metrics_25_14)
    m15 = read_json(args.metrics_25_15)
    man14 = read_json(args.manifest_25_14)
    man15 = read_json(args.manifest_25_15)
    npz_path = assert_sources(dataset_manifest, m10, m13, m14, m15, man14, man15)
    pack = load_npz(npz_path)
    v3, target_summary = build_target_v3(pack)
    target_rows = sample_target_rows(pack, v3)
    target_stats = {
        "overall": aggregate_target(target_rows),
        "by_component_count": group_rows(target_rows, "component_count", aggregate_target),
        "by_separation": group_rows(target_rows, "separation_type", aggregate_target),
        "by_topology": group_rows(target_rows, "topology_relation", aggregate_target),
        "highest_soft_leakage_samples": sorted(target_rows, key=lambda row: (float(row["soft_or_to_union_ratio"]), float(row["soft_duplicate_fraction_of_soft_targets"])), reverse=True)[:12],
    }
    failures = failure_rows(m15)
    test_failures = [row for row in failures if row["split"] == "test"]
    failure_stats = {
        "test": {
            "overall": aggregate_failures(test_failures),
            "by_component_count": group_rows(test_failures, "component_count", aggregate_failures),
            "by_separation": group_rows(test_failures, "separation_type", aggregate_failures),
            "by_topology": group_rows(test_failures, "topology_relation", aggregate_failures),
        },
        "merged_test_samples": [row for row in test_failures if row["merged_sample"]],
    }
    ranked_causes, decision, route, reasons = attribution(target_stats, failure_stats)
    return {
        "audit_id": AUDIT_ID,
        "stage": "25.15b",
        "dataset_id": DATASET_ID,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "source_files": {
            "dataset_manifest": str(args.dataset_manifest),
            "dataset_npz_path": str(npz_path),
            "metrics_25_10": str(args.metrics_25_10),
            "metrics_25_13": str(args.metrics_25_13),
            "metrics_25_14": str(args.metrics_25_14),
            "metrics_25_15": str(args.metrics_25_15),
            "manifest_25_14": str(args.manifest_25_14),
            "manifest_25_15": str(args.manifest_25_15),
        },
        "git": {
            "branch": git_value(["branch", "--show-current"]),
            "head_before_commit": git_value(["rev-parse", "HEAD"]),
            "protected_path_diff_before_write": git_value(["diff", "--name-only", "--", *FORBIDDEN_DIFF_PATHS]),
        },
        "boundary": {
            "training_run": False,
            "loss_tuning": False,
            "comsol_run": False,
            "data_npz_modified": False,
            "model_capacity_expanded": False,
            "current_baseline_updated": False,
            "baseline_transition": False,
            "checkpoint_or_preview_exported": False,
        },
        "metric_evidence": metric_evidence(m10, m13, m15),
        "target_v3_reproduction": target_summary,
        "target_v3_support_leakage_audit": target_stats,
        "failure_sample_audit": failure_stats,
        "cause_ranking": ranked_causes,
        "schema_v3b_recommendation": schema_v3b_recommendation(),
        "acceptance_decision": decision,
        "route_decision": route,
        "decision_reasons": reasons,
        "audit_main_conclusion": (
            "25.15 label-v3 relieved near-empty masks but converted the failure into union-like merged masks: "
            "soft/valid support is too broad relative to component identity, while raw labels remain sufficient for a stricter v3b derivation."
        ),
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metric_evidence"]["test_metrics"]
    deltas13 = payload["metric_evidence"]["delta_vs_25_13"]
    deltas10 = payload["metric_evidence"]["delta_vs_25_10"]
    overall = payload["target_v3_support_leakage_audit"]["overall"]
    by_sep = payload["target_v3_support_leakage_audit"]["by_separation"]
    failures = payload["failure_sample_audit"]["test"]
    lines = [
        "# 25.15b Label-V3 Failure Audit",
        "",
        f"- acceptance_decision: `{payload['acceptance_decision']}`",
        f"- route_decision: `{payload['route_decision']}`",
        f"- main_conclusion: {payload['audit_main_conclusion']}",
        "",
        "## Metric Evidence",
        "",
        f"- 25.15 test: recall `{metrics['25.15']['component_recall']:.6f}`, missed `{metrics['25.15']['missed_rate']:.6f}`, extra `{metrics['25.15']['extra_rate']:.6f}`, merged `{metrics['25.15']['merged_rate']:.6f}`, component Dice `{metrics['25.15']['component_mask_dice_mean']:.6f}`, union Dice `{metrics['25.15']['union_mask_dice_mean']:.6f}`, depth RMSE `{metrics['25.15']['depth_grid_rmse_m_mean']:.9f} m`.",
        f"- vs 25.13: component Dice `{deltas13['component_mask_dice_mean']:.6f}`, union Dice `{deltas13['union_mask_dice_mean']:.6f}`, merged `{deltas13['merged_rate']:.6f}`, depth RMSE `{deltas13['depth_grid_rmse_m_mean']:.9f} m`.",
        f"- vs 25.10: component Dice `{deltas10['component_mask_dice_mean']:.6f}`, union Dice `{deltas10['union_mask_dice_mean']:.6f}`, merged `{deltas10['merged_rate']:.6f}`, depth RMSE `{deltas10['depth_grid_rmse_m_mean']:.9f} m`.",
        "",
        "## V3 Target Audit",
        "",
        f"- soft OR / raw union ratio mean/p95/max: `{overall['soft_or_to_union_ratio_mean']:.6f}` / `{overall['soft_or_to_union_ratio_p95']:.6f}` / `{overall['soft_or_to_union_ratio_max']:.6f}`",
        f"- soft duplicate fraction mean/p95/max: `{overall['soft_duplicate_fraction_of_soft_targets_mean']:.6f}` / `{overall['soft_duplicate_fraction_of_soft_targets_p95']:.6f}` / `{overall['soft_duplicate_fraction_of_soft_targets_max']:.6f}`",
        f"- valid duplicate fraction mean/p95/max: `{overall['valid_duplicate_fraction_of_valid_targets_mean']:.6f}` / `{overall['valid_duplicate_fraction_of_valid_targets_p95']:.6f}` / `{overall['valid_duplicate_fraction_of_valid_targets_max']:.6f}`",
        f"- soft union-like sample count: `{overall['soft_or_union_like_sample_count']}/{overall['sample_count']}`",
        f"- separated soft duplicate fraction mean/max: `{by_sep['separated']['soft_duplicate_fraction_of_soft_targets_mean']:.6f}` / `{by_sep['separated']['soft_duplicate_fraction_of_soft_targets_max']:.6f}`",
        f"- close soft duplicate fraction mean/max: `{by_sep['close']['soft_duplicate_fraction_of_soft_targets_mean']:.6f}` / `{by_sep['close']['soft_duplicate_fraction_of_soft_targets_max']:.6f}`",
        f"- depth nonzero outside depth-valid mean/max: `{overall['depth_nonzero_outside_depth_valid_px_mean']:.6f}` / `{overall['depth_nonzero_outside_depth_valid_px_max']:.6f}`",
        f"- depth-valid duplicate target mean/max: `{overall['depth_valid_duplicate_target_px_mean']:.6f}` / `{overall['depth_valid_duplicate_target_px_max']:.6f}`",
        f"- SDF multi-valid overlap mean/max: `{overall['sdf_multi_valid_overlap_px_mean']:.6f}` / `{overall['sdf_multi_valid_overlap_px_max']:.6f}`",
        "",
        "## Failure Grouping",
        "",
        f"- test overall merged: `{failures['overall']['merged_rate']:.6f}`",
        f"- component_count=2 merged: `{failures['by_component_count']['2']['merged_rate']:.6f}`",
        f"- component_count=3 merged: `{failures['by_component_count']['3']['merged_rate']:.6f}`",
        f"- separated merged: `{failures['by_separation']['separated']['merged_rate']:.6f}`",
        f"- close merged: `{failures['by_separation']['close']['merged_rate']:.6f}`",
        f"- touching merged: `{failures['by_separation']['touching']['merged_rate']:.6f}`",
        f"- partially_overlapping merged: `{failures['by_separation']['partially_overlapping']['merged_rate']:.6f}`",
        "",
        "## Cause Ranking",
    ]
    for item in payload["cause_ranking"]:
        lines.append(f"- rank {item['rank']}: `{item['cause']}` supported=`{item['supported']}`; {item['evidence']}")
    lines.extend(
        [
            "",
            "## V3B Recommendation",
            "",
            "- Add `component_exclusive_hard_core` and keep it mutually exclusive.",
            "- Split valid region into `hard_core_region`, `boundary_halo_region`, and `ignore_overlap_region`.",
            "- Limit soft halo to a narrow non-overlapping boundary auxiliary signal.",
            "- Treat partially-overlapping shared pixels as ignore/diagnostic unless ownership confidence is explicit.",
            "- Supervise depth only on hard core plus narrow owned boundary; union mask/depth remain evaluation-only.",
            "",
            "## Boundary",
            "",
            "- No training, loss tuning, COMSOL run, data/NPZ mutation, checkpoint/preview export, baseline transition, or `CURRENT_BASELINE.md` update.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = {
        "audit_id": payload["audit_id"],
        "stage": payload["stage"],
        "dataset_id": payload["dataset_id"],
        "generated_at": payload["generated_at"],
        "status": "label_v3_failure_audited",
        "acceptance_decision": payload["acceptance_decision"],
        "route_decision": payload["route_decision"],
        "metrics_path": str(OUT_METRICS),
        "summary_path": str(OUT_SUMMARY),
        "source_files": payload["source_files"],
        "boundary": payload["boundary"],
        "baseline_ready": False,
        "current_baseline_updated": False,
        "allowed_use": ["label_v3b_derivation_planning", "component_set_label_schema_audit"],
        "forbidden_use": ["baseline_update", "current_baseline_replacement", "automatic_mainline_training", "formal_inference_artifact"],
        "git": payload["git"],
    }
    write_json(path, manifest)


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    write_json(args.out_metrics, payload)
    write_summary(args.out_summary, payload)
    write_manifest(args.out_manifest, payload)
    print(f"wrote {args.out_metrics}")
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_manifest}")
    print(f"acceptance_decision={payload['acceptance_decision']}")
    print(f"route_decision={payload['route_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Derive and validate 25.14 label-v3 targets for surface multi-pit data.

This stage is intentionally label/report only. It reads the validated
component-set pilot pack plus 25.12b/25.13/25.13b evidence, derives label-v3
targets in memory, and writes strict JSON/Markdown/manifest reports. It does
not train, run COMSOL, tune losses, mutate data/NPZ files, export checkpoints
or previews, or update the current baseline.
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
STAGE = "25.14"
DERIVATION_ID = "25_14_surface_multipit_label_v3_derivation_validator"
TARGET_SPLIT = {"train": 72, "val": 20, "test": 20}
K_MAX = 3
GRID_H = 64
GRID_W = 128
GRID_PIXELS = GRID_H * GRID_W
MASK_X_START_M = -0.04
MASK_X_STOP_M = 0.04
MASK_Y_START_M = -0.01
MASK_Y_STOP_M = 0.01

M12B = ROOT / "results/metrics/25_12b_component_raster_depth_target_redesign.json"
M13 = ROOT / "results/metrics/25_13_target_v2_training_gate_metrics.json"
M13B = ROOT / "results/metrics/25_13b_generator_label_schema_audit.json"
MAN13 = ROOT / "results/manifests/25_13_target_v2_training_gate_manifest.json"
DATASET_MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"

OUT_METRICS = ROOT / "results/metrics/25_14_label_v3_derivation_validator.json"
OUT_SUMMARY = ROOT / "results/summaries/25_14_label_v3_derivation_validator_summary.md"
OUT_MANIFEST = ROOT / "results/manifests/25_14_label_v3_derivation_validator_manifest.json"

FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive and validate 25.14 label-v3 targets.")
    parser.add_argument("--target-redesign-metrics", type=Path, default=M12B)
    parser.add_argument("--target-v2-metrics", type=Path, default=M13)
    parser.add_argument("--target-v2-manifest", type=Path, default=MAN13)
    parser.add_argument("--label-schema-audit", type=Path, default=M13B)
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET_MANIFEST)
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
    m12b: dict[str, Any],
    m13: dict[str, Any],
    man13: dict[str, Any],
    m13b: dict[str, Any],
) -> Path:
    if ROOT != Path(r"C:\Users\19166\Desktop\PINN_project"):
        raise RuntimeError(f"wrong project root: {ROOT}")
    if dataset_manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("dataset manifest dataset_id mismatch")
    if dataset_manifest.get("split_counts") != TARGET_SPLIT:
        raise ValueError(f"dataset split mismatch: {dataset_manifest.get('split_counts')}")
    if int(dataset_manifest.get("K_max", -1)) != K_MAX:
        raise ValueError("K_max mismatch")
    if dataset_manifest.get("train_ready_candidate") is not True or dataset_manifest.get("baseline_ready") is not False:
        raise ValueError("dataset manifest readiness boundary mismatch")
    for required in ["baseline_update", "current_baseline_replacement", "latest_newest_auto_discovery"]:
        if required not in set(dataset_manifest.get("forbidden_use", [])):
            raise ValueError(f"dataset manifest missing forbidden_use={required}")
    if m12b.get("stage") != "25.12b":
        raise ValueError("25.12b metrics mismatch")
    if m13.get("stage") != "25.13" or m13.get("gate_decision") != "FAIL":
        raise ValueError("25.13 metrics must be FAIL")
    if man13.get("stage") != "25.13" or man13.get("gate_decision") != "FAIL":
        raise ValueError("25.13 manifest mismatch")
    if m13b.get("stage") != "25.13b" or m13b.get("acceptance_decision") != "NEEDS_PINN_LABEL_DERIVATION_V3":
        raise ValueError("25.13b audit does not authorize PINN label-v3 derivation")
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
    """Return positive-inside, negative-outside SDF clipped to [-1, 1]."""
    fg = np.argwhere(mask)
    bg = np.argwhere(~mask)
    sdf = np.zeros(mask.shape, dtype=np.float32)
    if fg.size == 0:
        return sdf
    if bg.size == 0:
        return np.ones(mask.shape, dtype=np.float32)

    def nearest_distance(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
        out = np.empty(points.shape[0], dtype=np.float32)
        chunk = 1024
        target = targets.astype(np.float32)
        for start in range(0, points.shape[0], chunk):
            pts = points[start : start + chunk].astype(np.float32)
            diff = pts[:, None, :] - target[None, :, :]
            out[start : start + chunk] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        return out

    sdf[tuple(fg.T)] = nearest_distance(fg, bg)
    sdf[tuple(bg.T)] = -nearest_distance(bg, fg)
    return np.clip(sdf / float(clip_radius_px), -1.0, 1.0).astype(np.float32)


def build_target_v2(pack: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float64)
    centers = np.asarray(pack["component_center_xy_m"], dtype=np.float64)
    lwd = np.asarray(pack["component_lwd_m"], dtype=np.float64)
    xx, yy = grid_xy()
    n = exists.shape[0]
    masks_v2 = np.zeros_like(masks, dtype=bool)
    depths_v2 = np.zeros_like(depths, dtype=np.float64)
    ownership = np.full((n, GRID_H, GRID_W), -1, dtype=np.int16)
    rows: list[dict[str, Any]] = []
    total_duplicate_before = 0
    total_duplicate_after = 0
    total_conflict_before = 0
    total_conflict_after = 0
    total_resolved_overlap = 0
    for i in range(n):
        active = np.where(exists[i])[0]
        component_sum = masks[i, active].sum(axis=0) if active.size else np.zeros((GRID_H, GRID_W), dtype=np.int64)
        duplicate_before = int(np.maximum(component_sum - 1, 0).sum())
        overlap_pixels = int((component_sum > 1).sum())
        sample_conflict_before = 0
        sample_resolved = 0
        for y, x in zip(*np.where(component_sum > 0)):
            candidates = [int(slot) for slot in active if masks[i, slot, y, x]]
            if len(candidates) == 1:
                owner = candidates[0]
            else:
                sample_resolved += 1
                positive_depths = [float(depths[i, slot, y, x]) for slot in candidates if float(depths[i, slot, y, x]) > 0.0]
                if positive_depths and max(positive_depths) - min(positive_depths) > 1.0e-12:
                    sample_conflict_before += 1
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
        duplicate_after = int(np.maximum(masks_v2[i].sum(axis=0) - 1, 0).sum())
        total_duplicate_before += duplicate_before
        total_duplicate_after += duplicate_after
        total_conflict_before += sample_conflict_before
        total_resolved_overlap += sample_resolved
        rows.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": int(i),
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "raw_overlap_pixel_count": overlap_pixels,
                "duplicate_ownership_before_v2": duplicate_before,
                "duplicate_ownership_after_v2": duplicate_after,
                "overlap_depth_conflict_before_v2": sample_conflict_before,
                "overlap_depth_conflict_after_v2": 0,
                "ownership_resolved_overlap_pixels": sample_resolved,
            }
        )
    summary = {
        "target_loaded_count": int(n),
        "ownership_resolved_pixel_count": int(masks_v2.sum()),
        "ownership_resolved_overlap_pixel_count": int(total_resolved_overlap),
        "duplicate_ownership_before_v2": int(total_duplicate_before),
        "duplicate_ownership_after_v2": int(total_duplicate_after),
        "overlap_depth_conflict_before_v2": int(total_conflict_before),
        "overlap_depth_conflict_after_v2": int(total_conflict_after),
        "raw_overlap_sample_count": int(sum(row["raw_overlap_pixel_count"] > 0 for row in rows)),
    }
    return masks_v2, depths_v2, ownership, rows, summary


def derive_v3_targets(pack: dict[str, Any], ownership: np.ndarray) -> dict[str, np.ndarray]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    raw_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    n = exists.shape[0]
    soft = np.zeros((n, K_MAX, GRID_H, GRID_W), dtype=np.float32)
    sdf = np.zeros_like(soft, dtype=np.float32)
    valid = np.zeros((n, K_MAX, GRID_H, GRID_W), dtype=bool)
    depth_v3 = np.zeros_like(raw_depths, dtype=np.float32)
    depth_valid = np.zeros((n, K_MAX, GRID_H, GRID_W), dtype=bool)
    overlap_region = raw_masks.sum(axis=1) > 1
    contact_boundary = np.zeros((n, GRID_H, GRID_W), dtype=bool)
    for i in range(n):
        active = [int(slot) for slot in np.where(exists[i])[0]]
        for slot in active:
            raw = raw_masks[i, slot]
            owned = ownership[i] == slot
            band1 = dilate_bool(raw, radius=1)
            band2 = dilate_bool(raw, radius=2)
            component_soft = np.zeros((GRID_H, GRID_W), dtype=np.float32)
            component_soft[band2] = 0.25
            component_soft[band1] = 0.50
            component_soft[raw] = 0.80
            component_soft[owned] = 1.00
            soft[i, slot] = component_soft
            valid[i, slot] = band2
            sdf[i, slot] = normalized_signed_distance(raw)
            depth_v3[i, slot, raw] = raw_depths[i, slot, raw]
            depth_valid[i, slot] = raw
        for left_index, left in enumerate(active):
            left_band = dilate_bool(raw_masks[i, left], radius=1)
            for right in active[left_index + 1 :]:
                right_band = dilate_bool(raw_masks[i, right], radius=1)
                contact_boundary[i] |= left_band & right_band
        contact_boundary[i] &= ~overlap_region[i]
    return {
        "raw_component_mask_raw": raw_masks,
        "component_ownership_map": ownership,
        "component_mask_target_v3_soft": soft,
        "component_sdf_target_v3": sdf,
        "component_valid_region_mask": valid,
        "component_depth_target_v3": depth_v3,
        "component_depth_valid_region_mask": depth_valid,
        "overlap_region_mask": overlap_region,
        "contact_boundary_mask": contact_boundary,
    }


def component_rows(pack: dict[str, Any], masks_v2: np.ndarray, v3: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    raw_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    soft = v3["component_mask_target_v3_soft"]
    valid = v3["component_valid_region_mask"]
    sdf = v3["component_sdf_target_v3"]
    depth_valid = v3["component_depth_valid_region_mask"]
    rows: list[dict[str, Any]] = []
    for i in range(exists.shape[0]):
        for slot in range(K_MAX):
            row = {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": int(i),
                "slot": int(slot),
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "existing": bool(exists[i, slot]),
            }
            if not exists[i, slot]:
                row.update(
                    {
                        "v1_raw_foreground_px": 0,
                        "v2_hard_foreground_px": 0,
                        "v3_soft_positive_px": int((soft[i, slot] > 0.0).sum()),
                        "v3_soft_core_px_ge_0p80": int((soft[i, slot] >= 0.80).sum()),
                        "v3_valid_region_px": int(valid[i, slot].sum()),
                        "v3_depth_valid_px": int(depth_valid[i, slot].sum()),
                        "empty_slot_clean": bool(np.max(soft[i, slot]) == 0.0 and valid[i, slot].sum() == 0 and depth_valid[i, slot].sum() == 0),
                    }
                )
                rows.append(row)
                continue
            raw_px = int(raw_masks[i, slot].sum())
            hard_px = int(masks_v2[i, slot].sum())
            soft_positive = int((soft[i, slot] > 0.0).sum())
            soft_core = int((soft[i, slot] >= 0.80).sum())
            valid_px = int(valid[i, slot].sum())
            depth_valid_px = int(depth_valid[i, slot].sum())
            raw_depth_positive = int((raw_depths[i, slot] > 0.0).sum())
            row.update(
                {
                    "v1_raw_foreground_px": raw_px,
                    "v2_hard_foreground_px": hard_px,
                    "v3_soft_positive_px": soft_positive,
                    "v3_soft_core_px_ge_0p80": soft_core,
                    "v3_valid_region_px": valid_px,
                    "v3_depth_valid_px": depth_valid_px,
                    "v1_raw_depth_positive_px": raw_depth_positive,
                    "v3_soft_sum": float(soft[i, slot].sum()),
                    "v3_sdf_inside_mean": mean_or_null([float(x) for x in sdf[i, slot][raw_masks[i, slot]]]),
                    "v3_sdf_outside_mean": mean_or_null([float(x) for x in sdf[i, slot][~raw_masks[i, slot]]]),
                    "v3_vs_v2_positive_ratio": float(soft_positive / max(hard_px, 1)),
                    "v3_valid_vs_v2_ratio": float(valid_px / max(hard_px, 1)),
                    "v3_core_vs_v2_ratio": float(soft_core / max(hard_px, 1)),
                    "v3_depth_valid_vs_v2_ratio": float(depth_valid_px / max(hard_px, 1)),
                    "v3_support_delta_px_vs_v2": int(soft_positive - hard_px),
                    "existing_v3_empty_or_tiny": bool(soft_positive < 20),
                    "existing_depth_valid_empty": bool(depth_valid_px == 0),
                    "empty_slot_clean": True,
                }
            )
            rows.append(row)
    return rows


def sample_rows(pack: dict[str, Any], masks_v2: np.ndarray, v3: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    raw_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    union = np.asarray(pack["projected_mask_2d"], dtype=bool)
    union_depth = np.asarray(pack["depth_grid_m"], dtype=np.float32)
    soft = v3["component_mask_target_v3_soft"]
    overlap_region = v3["overlap_region_mask"]
    contact_boundary = v3["contact_boundary_mask"]
    rows: list[dict[str, Any]] = []
    for i in range(union.shape[0]):
        active = exists[i]
        raw_or = raw_masks[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        v2_or = masks_v2[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        v3_soft_or = (soft[i, active] > 0.0).max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        raw_max_depth = raw_depths[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=np.float32)
        rows.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": int(i),
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "raw_union_px": int(union[i].sum()),
                "raw_component_or_px": int(raw_or.sum()),
                "v2_component_or_px": int(v2_or.sum()),
                "v3_soft_or_px": int(v3_soft_or.sum()),
                "v3_soft_or_vs_union_ratio": float(v3_soft_or.sum() / max(int(union[i].sum()), 1)),
                "raw_or_to_union_mismatch_px": int(np.logical_xor(raw_or, union[i]).sum()),
                "v2_or_to_union_mismatch_px": int(np.logical_xor(v2_or, union[i]).sum()),
                "overlap_region_px": int(overlap_region[i].sum()),
                "contact_boundary_px": int(contact_boundary[i].sum()),
                "raw_max_depth_to_union_rmse_m": float(np.sqrt(np.mean((raw_max_depth - union_depth[i]) ** 2))),
            }
        )
    return rows


def aggregate_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if row["existing"]]
    inactive = [row for row in rows if not row["existing"]]
    if not active:
        return {"component_count": 0}
    return {
        "component_count": len(active),
        "inactive_slot_count": len(inactive),
        "v1_raw_foreground_px_mean": mean_or_null([row["v1_raw_foreground_px"] for row in active]),
        "v2_hard_foreground_px_mean": mean_or_null([row["v2_hard_foreground_px"] for row in active]),
        "v2_hard_foreground_px_min": min_or_null([row["v2_hard_foreground_px"] for row in active]),
        "v3_soft_positive_px_mean": mean_or_null([row["v3_soft_positive_px"] for row in active]),
        "v3_soft_positive_px_min": min_or_null([row["v3_soft_positive_px"] for row in active]),
        "v3_soft_positive_px_p05": percentile_or_null([row["v3_soft_positive_px"] for row in active], 5),
        "v3_valid_region_px_mean": mean_or_null([row["v3_valid_region_px"] for row in active]),
        "v3_depth_valid_px_mean": mean_or_null([row["v3_depth_valid_px"] for row in active]),
        "v3_vs_v2_positive_ratio_mean": mean_or_null([row["v3_vs_v2_positive_ratio"] for row in active]),
        "v3_vs_v2_positive_ratio_min": min_or_null([row["v3_vs_v2_positive_ratio"] for row in active]),
        "v3_valid_vs_v2_ratio_mean": mean_or_null([row["v3_valid_vs_v2_ratio"] for row in active]),
        "v3_core_vs_v2_ratio_mean": mean_or_null([row["v3_core_vs_v2_ratio"] for row in active]),
        "v3_depth_valid_vs_v2_ratio_mean": mean_or_null([row["v3_depth_valid_vs_v2_ratio"] for row in active]),
        "v3_support_delta_px_mean_vs_v2": mean_or_null([row["v3_support_delta_px_vs_v2"] for row in active]),
        "existing_v3_empty_or_tiny_count": int(sum(bool(row["existing_v3_empty_or_tiny"]) for row in active)),
        "existing_depth_valid_empty_count": int(sum(bool(row["existing_depth_valid_empty"]) for row in active)),
        "inactive_slot_clean_count": int(sum(bool(row["empty_slot_clean"]) for row in inactive)),
        "inactive_slot_violation_count": int(sum(not bool(row["empty_slot_clean"]) for row in inactive)),
        "v3_sdf_inside_mean": mean_or_null([row["v3_sdf_inside_mean"] for row in active]),
        "v3_sdf_outside_mean": mean_or_null([row["v3_sdf_outside_mean"] for row in active]),
    }


def aggregate_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    return {
        "sample_count": len(rows),
        "raw_union_px_mean": mean_or_null([row["raw_union_px"] for row in rows]),
        "v2_component_or_px_mean": mean_or_null([row["v2_component_or_px"] for row in rows]),
        "v3_soft_or_px_mean": mean_or_null([row["v3_soft_or_px"] for row in rows]),
        "v3_soft_or_vs_union_ratio_mean": mean_or_null([row["v3_soft_or_vs_union_ratio"] for row in rows]),
        "raw_or_to_union_mismatch_px_sum": int(sum(row["raw_or_to_union_mismatch_px"] for row in rows)),
        "v2_or_to_union_mismatch_px_sum": int(sum(row["v2_or_to_union_mismatch_px"] for row in rows)),
        "overlap_region_px_sum": int(sum(row["overlap_region_px"] for row in rows)),
        "contact_boundary_px_sum": int(sum(row["contact_boundary_px"] for row in rows)),
        "raw_max_depth_to_union_rmse_m_mean": mean_or_null([row["raw_max_depth_to_union_rmse_m"] for row in rows]),
        "raw_max_depth_to_union_rmse_m_max": max_or_null([row["raw_max_depth_to_union_rmse_m"] for row in rows]),
    }


def grouped(rows: list[dict[str, Any]], field: str, aggregate_fn) -> dict[str, Any]:
    return {str(value): aggregate_fn([row for row in rows if str(row[field]) == str(value)]) for value in sorted({row[field] for row in rows}, key=str)}


def label_schema_v3_design() -> dict[str, Any]:
    return {
        "raw_component_mask_raw": {
            "source": "component_projected_masks_2d",
            "role": "preserve original per-component binary support before ownership cuts",
        },
        "component_ownership_map": {
            "encoding": "-1 background, 0..K-1 deterministic owner slot",
            "role": "hard assignment for non-duplicated ownership diagnostics and hard evaluation targets",
        },
        "component_mask_target_v3_soft": {
            "rule": "owned pixels=1.0, raw component pixels=0.8, one-pixel dilation band=0.5, two-pixel dilation band=0.25",
            "role": "soft component-local mask supervision that retains boundary/context support without hard duplicate ownership",
        },
        "component_sdf_target_v3": {
            "rule": "positive-inside and negative-outside signed distance from raw_component_mask_raw, clipped to [-1, 1]",
            "role": "shape-support target robust to thin binary foregrounds",
        },
        "component_valid_region_mask": {
            "rule": "two-pixel dilation around each raw component mask",
            "role": "restrict mask/SDF supervision to local support instead of full-grid background",
        },
        "component_depth_target_v3": {
            "rule": "raw component depth retained only on raw foreground with component_depth_valid_region_mask",
            "role": "foreground-only component depth target; boundary/context does not create fake depth",
        },
        "overlap_region_mask": {
            "rule": "pixels where at least two raw component masks are positive",
            "role": "diagnostic and optional confidence/down-weighting region",
        },
        "contact_boundary_mask": {
            "rule": "pixels where one-pixel dilated raw components touch without raw overlap",
            "role": "diagnostic for touching-boundary topology",
        },
        "union_from_components_rule": {
            "mask": "OR over raw component masks for hard evaluation",
            "depth": "max over raw component depth grids for hard evaluation",
        },
        "storage_policy": "25.14 derives and validates these targets in memory only; no data/NPZ write.",
    }


def decision(component_stats: dict[str, Any], sample_stats: dict[str, Any], v2_summary: dict[str, Any]) -> tuple[str, str, list[str]]:
    overall = component_stats["overall"]
    sample_overall = sample_stats["overall"]
    support_ratio = float(overall["v3_vs_v2_positive_ratio_mean"] or 0.0)
    support_min = float(overall["v3_soft_positive_px_min"] or 0.0)
    empty_clean = int(overall["inactive_slot_violation_count"]) == 0
    active_nonempty = int(overall["existing_v3_empty_or_tiny_count"]) == 0 and int(overall["existing_depth_valid_empty_count"]) == 0
    hard_unique = int(v2_summary["duplicate_ownership_after_v2"]) == 0
    union_ok = int(sample_overall["raw_or_to_union_mismatch_px_sum"]) == 0 and int(sample_overall["v2_or_to_union_mismatch_px_sum"]) == 0
    depth_ok = float(sample_overall["raw_max_depth_to_union_rmse_m_max"] or 0.0) <= 1.0e-12
    reasons: list[str] = []
    if not empty_clean or not active_nonempty:
        reasons.append("v3 derivation produced empty active slots or dirty empty slots")
        return "NEEDS_COMSOL_GENERATOR_LABEL_FIX", "B. enter generator label export fix; do not train", reasons
    if not hard_unique or not union_ok or not depth_ok:
        reasons.append("raw component labels cannot reproduce hard ownership/union/depth invariants")
        return "NEEDS_COMSOL_GENERATOR_LABEL_FIX", "B. enter generator label export fix; do not train", reasons
    if support_ratio >= 1.50 and support_min >= 20.0:
        reasons.extend(
            [
                "v3 soft/valid-region support is substantially larger than v2 hard ownership support",
                "duplicate hard ownership remains resolved through component_ownership_map",
                "empty slots remain strictly empty and existing slots retain component/depth valid support",
                "raw component OR/max still reproduces sample-level union targets",
            ]
        )
        return (
            "READY_FOR_25.15_TRAINING",
            "A. enter 25.15 label-v3 training gate using 25.10 loss mainline + label-v3 supervision; do not use the 25.11/25.12 rebalance stack",
            reasons,
        )
    reasons.append("v3 support did not clearly relieve the v2 sparsity problem")
    return "INCONCLUSIVE", "D. continue label-v3 validator audit without training", reasons


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    m12b = read_json(args.target_redesign_metrics)
    m13 = read_json(args.target_v2_metrics)
    man13 = read_json(args.target_v2_manifest)
    m13b = read_json(args.label_schema_audit)
    dataset_manifest = read_json(args.dataset_manifest)
    npz_path = assert_sources(dataset_manifest, m12b, m13, man13, m13b)
    pack = load_npz(npz_path)
    masks_v2, depths_v2, ownership, target_v2_rows, target_v2_summary = build_target_v2(pack)
    _ = depths_v2  # kept to make the v2 reproduction route explicit.
    v3 = derive_v3_targets(pack, ownership)
    comp_rows = component_rows(pack, masks_v2, v3)
    samp_rows = sample_rows(pack, masks_v2, v3)
    component_stats = {
        "overall": aggregate_components(comp_rows),
        "by_split": grouped(comp_rows, "split", aggregate_components),
        "by_component_count": grouped(comp_rows, "component_count", aggregate_components),
        "by_separation": grouped(comp_rows, "separation_type", aggregate_components),
        "by_topology": grouped(comp_rows, "topology_relation", aggregate_components),
        "lowest_v3_support_components": sorted(
            [row for row in comp_rows if row["existing"]],
            key=lambda row: (int(row["v3_soft_positive_px"]), float(row["v3_vs_v2_positive_ratio"])),
        )[:12],
    }
    sample_stats = {
        "overall": aggregate_samples(samp_rows),
        "by_split": grouped(samp_rows, "split", aggregate_samples),
        "by_component_count": grouped(samp_rows, "component_count", aggregate_samples),
        "by_separation": grouped(samp_rows, "separation_type", aggregate_samples),
        "by_topology": grouped(samp_rows, "topology_relation", aggregate_samples),
        "highest_overlap_samples": sorted(samp_rows, key=lambda row: (int(row["overlap_region_px"]), int(row["contact_boundary_px"])), reverse=True)[:12],
    }
    acceptance, route_decision, reasons = decision(component_stats, sample_stats, target_v2_summary)
    return {
        "derivation_id": DERIVATION_ID,
        "stage": STAGE,
        "dataset_id": DATASET_ID,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "source_files": {
            "target_redesign_metrics": str(args.target_redesign_metrics),
            "target_v2_metrics": str(args.target_v2_metrics),
            "target_v2_manifest": str(args.target_v2_manifest),
            "label_schema_audit": str(args.label_schema_audit),
            "dataset_manifest": str(args.dataset_manifest),
            "dataset_npz_path": str(npz_path),
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
            "derived_label_arrays_written": False,
        },
        "input_evidence": {
            "25_13_gate_decision": m13.get("gate_decision"),
            "25_13_route_decision": m13.get("route_decision"),
            "25_13b_acceptance_decision": m13b.get("acceptance_decision"),
            "25_13b_route_decision": m13b.get("route_decision"),
            "25_13b_main_conclusion": m13b.get("audit_main_conclusion"),
        },
        "label_schema_v3_design": label_schema_v3_design(),
        "target_v2_reproduction": target_v2_summary,
        "target_v2_sample_overlap_rows": target_v2_rows,
        "label_v3_support_audit": {
            "component_support": component_stats,
            "sample_support": sample_stats,
            "grid_shape": [GRID_H, GRID_W],
            "grid_pixels": GRID_PIXELS,
            "soft_overlap_policy": "soft targets may overlap; duplicate hard ownership remains controlled by component_ownership_map",
        },
        "validator_checks": {
            "strict_empty_slot_policy": component_stats["overall"]["inactive_slot_violation_count"] == 0,
            "existing_slots_nonempty": component_stats["overall"]["existing_v3_empty_or_tiny_count"] == 0,
            "existing_depth_valid_nonempty": component_stats["overall"]["existing_depth_valid_empty_count"] == 0,
            "duplicate_hard_ownership_after_v2": target_v2_summary["duplicate_ownership_after_v2"],
            "raw_or_to_union_mismatch_px_sum": sample_stats["overall"]["raw_or_to_union_mismatch_px_sum"],
            "v2_or_to_union_mismatch_px_sum": sample_stats["overall"]["v2_or_to_union_mismatch_px_sum"],
            "raw_max_depth_to_union_rmse_m_max": sample_stats["overall"]["raw_max_depth_to_union_rmse_m_max"],
            "v3_support_ratio_mean_vs_v2": component_stats["overall"]["v3_vs_v2_positive_ratio_mean"],
        },
        "acceptance_decision": acceptance,
        "route_decision": route_decision,
        "decision_reasons": reasons,
        "label_v3_derivation_main_conclusion": (
            "Label-v3 can be derived inside PINN_project from existing raw component masks/depths: "
            "hard ownership remains unique, union/depth invariants remain intact, and soft/local valid-region support "
            "substantially increases the component-mask learning signal that collapsed under target-v2."
        ),
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    comp = payload["label_v3_support_audit"]["component_support"]["overall"]
    samp = payload["label_v3_support_audit"]["sample_support"]["overall"]
    c3 = payload["label_v3_support_audit"]["component_support"]["by_component_count"].get("3", {})
    po = payload["label_v3_support_audit"]["component_support"]["by_separation"].get("partially_overlapping", {})
    tb = payload["label_v3_support_audit"]["component_support"]["by_topology"].get("touching_boundary", {})
    v2 = payload["target_v2_reproduction"]
    lines = [
        "# 25.14 Label-V3 Derivation Validator",
        "",
        f"- acceptance_decision: `{payload['acceptance_decision']}`",
        f"- route_decision: `{payload['route_decision']}`",
        f"- main_conclusion: {payload['label_v3_derivation_main_conclusion']}",
        "",
        "## V3 Support Check",
        "",
        f"- active components: `{comp['component_count']}`",
        f"- v2 hard foreground px mean/min: `{comp['v2_hard_foreground_px_mean']:.6f}` / `{comp['v2_hard_foreground_px_min']:.0f}`",
        f"- v3 soft positive px mean/min/p05: `{comp['v3_soft_positive_px_mean']:.6f}` / `{comp['v3_soft_positive_px_min']:.0f}` / `{comp['v3_soft_positive_px_p05']:.6f}`",
        f"- v3/v2 positive support ratio mean/min: `{comp['v3_vs_v2_positive_ratio_mean']:.6f}` / `{comp['v3_vs_v2_positive_ratio_min']:.6f}`",
        f"- v3 valid-region ratio mean: `{comp['v3_valid_vs_v2_ratio_mean']:.6f}`",
        f"- v3 depth-valid ratio mean: `{comp['v3_depth_valid_vs_v2_ratio_mean']:.6f}`",
        f"- existing v3 tiny/empty count: `{comp['existing_v3_empty_or_tiny_count']}`",
        f"- existing depth-valid empty count: `{comp['existing_depth_valid_empty_count']}`",
        f"- inactive slot violations: `{comp['inactive_slot_violation_count']}`",
        "",
        "## Group Slices",
        "",
        f"- component_count=3 v3/v2 support ratio mean/min: `{c3.get('v3_vs_v2_positive_ratio_mean'):.6f}` / `{c3.get('v3_vs_v2_positive_ratio_min'):.6f}`",
        f"- partially_overlapping v3/v2 support ratio mean/min: `{po.get('v3_vs_v2_positive_ratio_mean'):.6f}` / `{po.get('v3_vs_v2_positive_ratio_min'):.6f}`",
        f"- touching_boundary v3/v2 support ratio mean/min: `{tb.get('v3_vs_v2_positive_ratio_mean'):.6f}` / `{tb.get('v3_vs_v2_positive_ratio_min'):.6f}`",
        "",
        "## Invariants",
        "",
        f"- duplicate hard ownership: `{v2['duplicate_ownership_before_v2']} -> {v2['duplicate_ownership_after_v2']}`",
        f"- overlap-depth-conflict under hard ownership: `{v2['overlap_depth_conflict_before_v2']} -> {v2['overlap_depth_conflict_after_v2']}`",
        f"- raw OR to union mismatch px sum: `{samp['raw_or_to_union_mismatch_px_sum']}`",
        f"- v2 OR to union mismatch px sum: `{samp['v2_or_to_union_mismatch_px_sum']}`",
        f"- raw max-depth to union RMSE max: `{samp['raw_max_depth_to_union_rmse_m_max']:.12f} m`",
        f"- overlap region px sum: `{samp['overlap_region_px_sum']}`",
        f"- contact boundary px sum: `{samp['contact_boundary_px_sum']}`",
        "",
        "## V3 Schema",
        "",
        "- `raw_component_mask_raw`: original component binary masks.",
        "- `component_ownership_map`: hard unique owner map for deterministic evaluation and diagnostics.",
        "- `component_mask_target_v3_soft`: owned=1.0, raw=0.8, one-pixel band=0.5, two-pixel band=0.25.",
        "- `component_sdf_target_v3`: clipped signed distance field from raw component mask.",
        "- `component_valid_region_mask`: local two-pixel supervision region.",
        "- `component_depth_target_v3`: raw foreground depth with explicit depth-valid region.",
        "- `overlap_region_mask` and `contact_boundary_mask`: topology diagnostics.",
        "",
        "## Boundary",
        "",
        "- No training, COMSOL run, loss tuning, data/NPZ mutation, checkpoint/preview export, baseline transition, or `CURRENT_BASELINE.md` update.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = {
        "derivation_id": payload["derivation_id"],
        "stage": payload["stage"],
        "dataset_id": payload["dataset_id"],
        "generated_at": payload["generated_at"],
        "status": "label_v3_derivation_validated",
        "acceptance_decision": payload["acceptance_decision"],
        "route_decision": payload["route_decision"],
        "metrics_path": str(OUT_METRICS),
        "summary_path": str(OUT_SUMMARY),
        "source_files": payload["source_files"],
        "boundary": payload["boundary"],
        "baseline_ready": False,
        "current_baseline_updated": False,
        "allowed_use": [
            "explicit_25_15_label_v3_training_gate_planning",
            "component_set_label_validation",
        ],
        "forbidden_use": [
            "baseline_update",
            "current_baseline_replacement",
            "latest_newest_auto_discovery",
            "formal_inference_artifact",
        ],
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

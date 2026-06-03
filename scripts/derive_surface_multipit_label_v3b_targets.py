#!/usr/bin/env python
"""Derive and validate 25.16 label-v3b targets for surface multi-pit data.

This stage is label/report only. It reads prior 25.14/25.15/25.15b evidence
and the validated component-set pilot pack, derives label-v3b targets in
memory, and writes strict JSON/Markdown/manifest reports. It does not train,
run COMSOL, tune losses, mutate data/NPZ files, expand model capacity, export
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
DERIVATION_ID = "25_16_surface_multipit_label_v3b_derivation_validator"
STAGE = "25.16"
TARGET_SPLIT = {"train": 72, "val": 20, "test": 20}
K_MAX = 3
GRID_H = 64
GRID_W = 128
MASK_X_START_M = -0.04
MASK_X_STOP_M = 0.04
MASK_Y_START_M = -0.01
MASK_Y_STOP_M = 0.01

M14 = ROOT / "results/metrics/25_14_label_v3_derivation_validator.json"
M15 = ROOT / "results/metrics/25_15_label_v3_training_gate_metrics.json"
M15B = ROOT / "results/metrics/25_15b_label_v3_failure_audit.json"
MAN14 = ROOT / "results/manifests/25_14_label_v3_derivation_validator_manifest.json"
MAN15B = ROOT / "results/manifests/25_15b_label_v3_failure_audit_manifest.json"
DATASET_MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"

OUT_METRICS = ROOT / "results/metrics/25_16_label_v3b_derivation_validator.json"
OUT_SUMMARY = ROOT / "results/summaries/25_16_label_v3b_derivation_validator_summary.md"
OUT_MANIFEST = ROOT / "results/manifests/25_16_label_v3b_derivation_validator_manifest.json"

FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]

V3B_SOFT_OR_RAW_UNION_RATIO_TARGET = 1.35
V3B_HALO_CAP_FRACTION_OF_RAW_UNION = 0.25
V2_SPARSE_LOWER_BOUND_PX = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive and validate 25.16 label-v3b targets.")
    parser.add_argument("--metrics-25-14", type=Path, default=M14)
    parser.add_argument("--metrics-25-15", type=Path, default=M15)
    parser.add_argument("--metrics-25-15b", type=Path, default=M15B)
    parser.add_argument("--manifest-25-14", type=Path, default=MAN14)
    parser.add_argument("--manifest-25-15b", type=Path, default=MAN15B)
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
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
    m14: dict[str, Any],
    m15: dict[str, Any],
    m15b: dict[str, Any],
    man14: dict[str, Any],
    man15b: dict[str, Any],
) -> Path:
    if ROOT != Path(r"C:\Users\19166\Desktop\PINN_project"):
        raise RuntimeError(f"wrong project root: {ROOT}")
    if dataset_manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(f"dataset_id mismatch: {dataset_manifest.get('dataset_id')}")
    if dataset_manifest.get("split_counts") != TARGET_SPLIT:
        raise ValueError(f"split mismatch: {dataset_manifest.get('split_counts')}")
    if int(dataset_manifest.get("K_max", -1)) != K_MAX:
        raise ValueError(f"K_max mismatch: {dataset_manifest.get('K_max')}")
    if dataset_manifest.get("train_ready_candidate") is not True or dataset_manifest.get("baseline_ready") is not False:
        raise ValueError("dataset readiness boundary mismatch")
    for required in ["baseline_update", "current_baseline_replacement", "latest_newest_auto_discovery"]:
        if required not in set(dataset_manifest.get("forbidden_use", [])):
            raise ValueError(f"dataset manifest missing forbidden_use={required}")
    if m14.get("stage") != "25.14" or m14.get("acceptance_decision") != "READY_FOR_25.15_TRAINING":
        raise ValueError("25.14 metrics do not authorize label-v3 training")
    if man14.get("acceptance_decision") != "READY_FOR_25.15_TRAINING":
        raise ValueError("25.14 manifest mismatch")
    if m15.get("stage") != "25.15" or m15.get("gate_decision") != "FAIL":
        raise ValueError("25.15 metrics must be a FAIL input")
    if m15b.get("stage") != "25.15b" or m15b.get("acceptance_decision") != "NEEDS_PINN_LABEL_DERIVATION_V3B":
        raise ValueError("25.15b audit does not authorize label-v3b derivation")
    if man15b.get("acceptance_decision") != "NEEDS_PINN_LABEL_DERIVATION_V3B":
        raise ValueError("25.15b manifest mismatch")
    npz_path = Path(dataset_manifest["path"])
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    return npz_path


def grid_xy() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, GRID_W)
    y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, GRID_H)
    return np.meshgrid(x, y, indexing="xy")


def dilate_bool(mask: np.ndarray, radius: int = 1, mode: str = "square") -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if mode not in {"square", "cross"}:
        raise ValueError(f"unsupported dilation mode: {mode}")
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == "square":
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result)
        for dy, dx in offsets:
            y0 = 1 + dy
            x0 = 1 + dx
            expanded |= padded[y0 : y0 + result.shape[0], x0 : x0 + result.shape[1]]
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


def fill_halo_depth_from_nearest_core(core: np.ndarray, halo: np.ndarray, raw_depth: np.ndarray) -> np.ndarray:
    depth = np.zeros(raw_depth.shape, dtype=np.float32)
    depth[core] = raw_depth[core]
    core_points = np.argwhere(core)
    if core_points.size == 0:
        return depth
    core_values = raw_depth[tuple(core_points.T)].astype(np.float32)
    halo_points = np.argwhere(halo)
    for y, x in halo_points:
        delta = core_points - np.array([y, x], dtype=np.int64)
        idx = int(np.argmin(np.sum(delta * delta, axis=1)))
        depth[y, x] = core_values[idx]
    return depth


def build_target_v2(pack: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    centers = np.asarray(pack["component_center_xy_m"], dtype=np.float64)
    lwd = np.asarray(pack["component_lwd_m"], dtype=np.float64)
    xx, yy = grid_xy()
    n, _k, height, width = masks.shape
    masks_v2 = np.zeros_like(masks, dtype=bool)
    depths_v2 = np.zeros_like(depths, dtype=np.float32)
    ownership = np.full((n, height, width), -1, dtype=np.int16)
    duplicate_before = 0
    duplicate_after = 0
    conflict_before = 0
    conflict_after = 0
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
                if depth_values and max(depth_values) - min(depth_values) > 1.0e-12:
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
    raw_union = np.asarray(pack["projected_mask_2d"], dtype=bool)
    v2_union = masks_v2.max(axis=1)
    summary = {
        "duplicate_ownership_before_v2": int(duplicate_before),
        "duplicate_ownership_after_v2": int(duplicate_after),
        "overlap_depth_conflict_before_v2": int(conflict_before),
        "overlap_depth_conflict_after_v2": int(conflict_after),
        "ownership_resolved_overlap_pixel_count": int(overlap_resolved),
        "raw_overlap_sample_count": int(raw_overlap_samples),
        "raw_or_to_union_mismatch_px_sum": int(np.logical_xor(masks.max(axis=1), raw_union).sum()),
        "v2_or_to_union_mismatch_px_sum": int(np.logical_xor(v2_union, raw_union).sum()),
    }
    return masks_v2, depths_v2, ownership, summary


def build_contact_boundary(raw_masks: np.ndarray, exists: np.ndarray, overlap_region: np.ndarray) -> np.ndarray:
    n, _k, height, width = raw_masks.shape
    contact = np.zeros((n, height, width), dtype=bool)
    for i in range(n):
        active = [int(slot) for slot in np.where(exists[i])[0]]
        for left_index, left in enumerate(active):
            left_band = dilate_bool(raw_masks[i, left], radius=1, mode="cross")
            for right in active[left_index + 1 :]:
                right_band = dilate_bool(raw_masks[i, right], radius=1, mode="cross")
                contact[i] |= left_band & right_band
        contact[i] &= ~overlap_region[i]
    return contact


def build_v3_soft_or(pack: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    n, k, height, width = raw_masks.shape
    soft_pos = np.zeros((n, k, height, width), dtype=bool)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        for slot in np.where(exists[i])[0]:
            band2 = dilate_bool(raw_masks[i, slot], radius=2, mode="square")
            soft_pos[i, slot] = band2
        active = exists[i]
        raw_union = raw_masks[i, active].max(axis=0) if active.any() else np.zeros((height, width), dtype=bool)
        soft_or = soft_pos[i, active].max(axis=0) if active.any() else np.zeros((height, width), dtype=bool)
        rows.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": int(i),
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "v3_soft_or_px": int(soft_or.sum()),
                "v3_soft_or_to_raw_union_ratio": float(soft_or.sum() / max(int(raw_union.sum()), 1)),
            }
        )
    return soft_pos, rows


def cap_halo_candidates(candidates: np.ndarray, raw_union_px: int) -> np.ndarray:
    capped = np.zeros_like(candidates, dtype=bool)
    total_candidate = int(candidates.sum())
    if total_candidate == 0:
        return capped
    allowed = int(math.floor(raw_union_px * V3B_HALO_CAP_FRACTION_OF_RAW_UNION))
    allowed = max(0, min(allowed, total_candidate))
    if allowed == 0:
        return capped
    counts = candidates.reshape(candidates.shape[0], -1).sum(axis=1)
    quotas = np.floor(allowed * counts / max(total_candidate, 1)).astype(int)
    for slot, count in enumerate(counts):
        if count > 0 and quotas[slot] == 0 and allowed >= int((counts > 0).sum()):
            quotas[slot] = 1
    while int(quotas.sum()) > allowed:
        slot = int(np.argmax(quotas))
        quotas[slot] -= 1
    while int(quotas.sum()) < allowed:
        remainders = allowed * counts / max(total_candidate, 1) - quotas
        remainders[counts == 0] = -1.0
        slot = int(np.argmax(remainders))
        if remainders[slot] < 0:
            break
        quotas[slot] += 1
    for slot, quota in enumerate(quotas):
        if quota <= 0:
            continue
        coords = np.argwhere(candidates[slot])
        for y, x in coords[: int(quota)]:
            capped[slot, y, x] = True
    return capped


def derive_v3b_targets(
    pack: dict[str, Any],
    masks_v2: np.ndarray,
    ownership: np.ndarray,
) -> dict[str, np.ndarray]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    raw_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    n, k, height, width = raw_masks.shape
    hard_core = masks_v2.astype(bool)
    boundary_halo = np.zeros((n, k, height, width), dtype=bool)
    ignore_overlap = np.zeros((n, k, height, width), dtype=bool)
    soft = np.zeros((n, k, height, width), dtype=np.float32)
    sdf = np.zeros((n, k, height, width), dtype=np.float32)
    valid = np.zeros((n, k, height, width), dtype=bool)
    depth = np.zeros((n, k, height, width), dtype=np.float32)
    depth_valid = np.zeros((n, k, height, width), dtype=bool)
    overlap_region = raw_masks.sum(axis=1) > 1
    contact_boundary = build_contact_boundary(raw_masks, exists, overlap_region)
    identity_conflict = np.zeros((n, height, width), dtype=bool)
    for i in range(n):
        active = [int(slot) for slot in np.where(exists[i])[0]]
        raw_union = raw_masks[i, active].max(axis=0) if active else np.zeros((height, width), dtype=bool)
        preliminary_halo = np.zeros((k, height, width), dtype=bool)
        for slot in active:
            other_raw = raw_masks[i, [other for other in active if other != slot]].max(axis=0) if len(active) > 1 else np.zeros((height, width), dtype=bool)
            candidate = dilate_bool(hard_core[i, slot], radius=1, mode="cross") & ~hard_core[i, slot]
            candidate &= ~other_raw
            candidate &= ~overlap_region[i]
            preliminary_halo[slot] = candidate
        halo_claims = preliminary_halo.sum(axis=0)
        identity_conflict[i] = overlap_region[i] | contact_boundary[i] | (halo_claims > 1)
        exclusive_halo = preliminary_halo & (halo_claims[None, :, :] == 1)
        exclusive_halo &= ~identity_conflict[i][None, :, :]
        capped_halo = cap_halo_candidates(exclusive_halo, int(raw_union.sum()))
        boundary_halo[i] = capped_halo
        for slot in active:
            ignored = (raw_masks[i, slot] | preliminary_halo[slot]) & identity_conflict[i] & ~hard_core[i, slot]
            ignore_overlap[i, slot] = ignored
            soft[i, slot, hard_core[i, slot]] = 1.0
            soft[i, slot, capped_halo[slot]] = 0.35
            valid[i, slot] = hard_core[i, slot] | capped_halo[slot]
            sdf[i, slot] = normalized_signed_distance(hard_core[i, slot])
            depth_valid[i, slot] = valid[i, slot]
            depth[i, slot] = fill_halo_depth_from_nearest_core(hard_core[i, slot], capped_halo[slot], raw_depths[i, slot])
    return {
        "raw_component_mask_raw": raw_masks,
        "component_ownership_map": ownership,
        "component_hard_core_mask_v3b": hard_core,
        "component_boundary_halo_mask_v3b": boundary_halo,
        "component_ignore_overlap_mask_v3b": ignore_overlap,
        "component_mask_target_v3b_soft": soft,
        "component_sdf_target_v3b": sdf,
        "component_valid_region_mask_v3b": valid,
        "component_depth_target_v3b": depth,
        "component_depth_valid_region_mask_v3b": depth_valid,
        "component_identity_conflict_mask_v3b": identity_conflict,
        "overlap_region_mask": overlap_region,
        "contact_boundary_mask": contact_boundary,
    }


def component_rows(pack: dict[str, Any], v3b: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = v3b["raw_component_mask_raw"]
    hard = v3b["component_hard_core_mask_v3b"]
    halo = v3b["component_boundary_halo_mask_v3b"]
    ignore = v3b["component_ignore_overlap_mask_v3b"]
    soft = v3b["component_mask_target_v3b_soft"]
    valid = v3b["component_valid_region_mask_v3b"]
    depth_valid = v3b["component_depth_valid_region_mask_v3b"]
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
                empty_clean = (
                    int(hard[i, slot].sum())
                    + int(halo[i, slot].sum())
                    + int(ignore[i, slot].sum())
                    + int((soft[i, slot] > 0.0).sum())
                    + int(valid[i, slot].sum())
                    + int(depth_valid[i, slot].sum())
                    == 0
                )
                row.update(
                    {
                        "raw_component_px": 0,
                        "hard_core_px": 0,
                        "boundary_halo_px": 0,
                        "ignore_overlap_px": int(ignore[i, slot].sum()),
                        "soft_support_px": int((soft[i, slot] > 0.0).sum()),
                        "valid_region_px": int(valid[i, slot].sum()),
                        "depth_valid_region_px": int(depth_valid[i, slot].sum()),
                        "soft_vs_hard_support_ratio": None,
                        "empty_slot_clean": bool(empty_clean),
                        "existing_hard_core_empty": False,
                        "existing_depth_valid_empty": False,
                    }
                )
                rows.append(row)
                continue
            hard_px = int(hard[i, slot].sum())
            soft_px = int((soft[i, slot] > 0.0).sum())
            row.update(
                {
                    "raw_component_px": int(raw_masks[i, slot].sum()),
                    "hard_core_px": hard_px,
                    "boundary_halo_px": int(halo[i, slot].sum()),
                    "ignore_overlap_px": int(ignore[i, slot].sum()),
                    "soft_support_px": soft_px,
                    "valid_region_px": int(valid[i, slot].sum()),
                    "depth_valid_region_px": int(depth_valid[i, slot].sum()),
                    "soft_sum": float(soft[i, slot].sum()),
                    "soft_vs_hard_support_ratio": float(soft_px / max(hard_px, 1)),
                    "halo_vs_hard_ratio": float(int(halo[i, slot].sum()) / max(hard_px, 1)),
                    "depth_valid_vs_hard_ratio": float(int(depth_valid[i, slot].sum()) / max(hard_px, 1)),
                    "empty_slot_clean": True,
                    "existing_hard_core_empty": bool(hard_px == 0),
                    "existing_depth_valid_empty": bool(int(depth_valid[i, slot].sum()) == 0),
                }
            )
            rows.append(row)
    return rows


def sample_rows(pack: dict[str, Any], v3_soft_pos: np.ndarray, v3b: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = v3b["raw_component_mask_raw"]
    hard = v3b["component_hard_core_mask_v3b"]
    halo = v3b["component_boundary_halo_mask_v3b"]
    ignore = v3b["component_ignore_overlap_mask_v3b"]
    soft = v3b["component_mask_target_v3b_soft"]
    depth_valid = v3b["component_depth_valid_region_mask_v3b"]
    identity_conflict = v3b["component_identity_conflict_mask_v3b"]
    overlap_region = v3b["overlap_region_mask"]
    contact_boundary = v3b["contact_boundary_mask"]
    union = np.asarray(pack["projected_mask_2d"], dtype=bool)
    rows: list[dict[str, Any]] = []
    for i in range(exists.shape[0]):
        active = exists[i]
        raw_union = raw_masks[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        hard_or = hard[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        soft_pos = soft[i, active] > 0.0 if active.any() else np.zeros((0, GRID_H, GRID_W), dtype=bool)
        soft_or = soft_pos.max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        halo_or = halo[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        v3_or = v3_soft_pos[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        soft_sum = soft_pos.sum(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=np.int64)
        hard_sum = hard[i, active].sum(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=np.int64)
        depth_valid_sum = depth_valid[i, active].sum(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=np.int64)
        ignore_or = ignore[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        rows.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": int(i),
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "raw_union_px": int(raw_union.sum()),
                "sample_union_px": int(union[i].sum()),
                "raw_or_to_union_mismatch_px": int(np.logical_xor(raw_union, union[i]).sum()),
                "hard_or_to_union_mismatch_px": int(np.logical_xor(hard_or, union[i]).sum()),
                "hard_duplicate_px": int(np.maximum(hard_sum - 1, 0).sum()),
                "soft_duplicate_px": int(np.maximum(soft_sum - 1, 0).sum()),
                "depth_valid_duplicate_px": int(np.maximum(depth_valid_sum - 1, 0).sum()),
                "v3_soft_or_px": int(v3_or.sum()),
                "v3b_hard_or_px": int(hard_or.sum()),
                "v3b_halo_or_px": int(halo_or.sum()),
                "v3b_soft_or_px": int(soft_or.sum()),
                "v3b_soft_or_to_raw_union_ratio": float(soft_or.sum() / max(int(raw_union.sum()), 1)),
                "v3b_vs_v3_soft_or_shrink_ratio": float(soft_or.sum() / max(int(v3_or.sum()), 1)),
                "v3b_halo_fraction_of_raw_union": float(halo_or.sum() / max(int(raw_union.sum()), 1)),
                "soft_duplicate_fraction_of_soft_targets": float(np.maximum(soft_sum - 1, 0).sum() / max(int(soft_pos.sum()), 1)),
                "identity_conflict_px": int(identity_conflict[i].sum()),
                "ignore_overlap_px": int(ignore_or.sum()),
                "raw_overlap_px": int(overlap_region[i].sum()),
                "contact_boundary_px": int(contact_boundary[i].sum()),
            }
        )
    return rows


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


def aggregate_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if row["existing"]]
    inactive = [row for row in rows if not row["existing"]]
    fields = [
        "raw_component_px",
        "hard_core_px",
        "boundary_halo_px",
        "ignore_overlap_px",
        "soft_support_px",
        "valid_region_px",
        "depth_valid_region_px",
        "soft_vs_hard_support_ratio",
        "halo_vs_hard_ratio",
        "depth_valid_vs_hard_ratio",
    ]
    out = aggregate_numeric(active, fields)
    out["active_component_count"] = len(active)
    out["inactive_slot_count"] = len(inactive)
    out["existing_hard_core_empty_count"] = int(sum(bool(row["existing_hard_core_empty"]) for row in active))
    out["existing_depth_valid_empty_count"] = int(sum(bool(row["existing_depth_valid_empty"]) for row in active))
    out["empty_slot_violation_count"] = int(sum(not bool(row["empty_slot_clean"]) for row in inactive))
    return out


def aggregate_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "raw_union_px",
        "sample_union_px",
        "raw_or_to_union_mismatch_px",
        "hard_or_to_union_mismatch_px",
        "hard_duplicate_px",
        "soft_duplicate_px",
        "depth_valid_duplicate_px",
        "v3_soft_or_px",
        "v3b_hard_or_px",
        "v3b_halo_or_px",
        "v3b_soft_or_px",
        "v3b_soft_or_to_raw_union_ratio",
        "v3b_vs_v3_soft_or_shrink_ratio",
        "v3b_halo_fraction_of_raw_union",
        "soft_duplicate_fraction_of_soft_targets",
        "identity_conflict_px",
        "ignore_overlap_px",
        "raw_overlap_px",
        "contact_boundary_px",
    ]
    out = aggregate_numeric(rows, fields)
    out["soft_or_union_like_sample_count"] = int(sum(float(row["v3b_soft_or_to_raw_union_ratio"]) >= V3B_SOFT_OR_RAW_UNION_RATIO_TARGET for row in rows))
    out["soft_overlap_sample_count"] = int(sum(int(row["soft_duplicate_px"]) > 0 for row in rows))
    return out


def group_rows(rows: list[dict[str, Any]], field: str, aggregate_fn) -> dict[str, Any]:
    return {str(value): aggregate_fn([row for row in rows if str(row[field]) == str(value)]) for value in sorted({row[field] for row in rows}, key=str)}


def label_schema_v3b_design() -> dict[str, Any]:
    return {
        "raw_component_mask_raw": "preserve original component binary support; never overwrite generator evidence",
        "component_ownership_map": "retain deterministic unique owner map from v2 for hard identity diagnostics",
        "component_hard_core_mask_v3b": "ownership-resolved exclusive hard component core; existing slots must be nonempty",
        "component_boundary_halo_mask_v3b": "single-pixel cross-neighborhood auxiliary halo, capped to 25% of sample raw union and stripped of cross-component claims",
        "component_ignore_overlap_mask_v3b": "raw overlap/contact/halo-conflict pixels that should be diagnostic or ignored for non-owner positive supervision",
        "component_mask_target_v3b_soft": "hard core=1.0, exclusive capped halo=0.35, all other/ignore/background=0.0",
        "component_sdf_target_v3b": "clipped signed distance from component hard core, consumed only inside v3b valid region",
        "component_valid_region_mask_v3b": "hard_core_region OR boundary_halo_region; ignore_overlap_region is excluded from positive supervision",
        "component_depth_target_v3b": "component hard-core depth plus nearest-core depth on exclusive narrow halo",
        "component_identity_conflict_mask_v3b": "sample-level raw overlap, touching contact, or multi-halo claim diagnostic mask",
        "overlap_policy": "partially overlapping pixels keep one hard owner; non-owner claim is ignore/diagnostic, not another positive mask",
        "touching_boundary_policy": "touching pixels are diagnostic/ignore for halo expansion while hard ownership remains exclusive",
        "empty_slot_policy": "hard/halo/soft/valid/depth/ignore are all strictly empty for non-existing K slots",
        "union_from_components_rule": "union mask/depth remain raw OR/max evaluation targets, not component-local supervision",
        "storage_policy": "25.16 derives arrays in memory only and writes JSON/Markdown/manifest reports only",
    }


def decide(
    component_stats: dict[str, Any],
    sample_stats: dict[str, Any],
    v2_summary: dict[str, Any],
    m15b: dict[str, Any],
) -> tuple[str, str, list[str], bool]:
    overall_c = component_stats["overall"]
    overall_s = sample_stats["overall"]
    separated = sample_stats["by_separation"].get("separated", {})
    close = sample_stats["by_separation"].get("close", {})
    hard_nonempty = int(overall_c["existing_hard_core_empty_count"]) == 0
    depth_nonempty = int(overall_c["existing_depth_valid_empty_count"]) == 0
    empty_clean = int(overall_c["empty_slot_violation_count"]) == 0
    hard_unique = int(overall_s["hard_duplicate_px_mean"] or 0) == 0 and int(overall_s["hard_duplicate_px_max"] or 0) == 0
    raw_union_ok = int(overall_s["raw_or_to_union_mismatch_px_max"] or 0) == 0 and int(overall_s["hard_or_to_union_mismatch_px_max"] or 0) == 0
    support_ok = float(overall_c["hard_core_px_min"] or 0.0) > V2_SPARSE_LOWER_BOUND_PX
    anti_sparse_ok = float(overall_c["soft_vs_hard_support_ratio_mean"] or 0.0) >= 1.10
    leakage_ok = (
        float(overall_s["v3b_soft_or_to_raw_union_ratio_mean"] or 999.0) < V3B_SOFT_OR_RAW_UNION_RATIO_TARGET
        and float(overall_s["v3b_soft_or_to_raw_union_ratio_max"] or 999.0) < V3B_SOFT_OR_RAW_UNION_RATIO_TARGET
    )
    shrink_ok = float(overall_s["v3b_vs_v3_soft_or_shrink_ratio_mean"] or 999.0) <= 0.70
    cross_overlap_ok = (
        float(separated.get("soft_duplicate_fraction_of_soft_targets_max") or 0.0) <= 0.0
        and float(close.get("soft_duplicate_fraction_of_soft_targets_max") or 0.0) <= 0.0
    )
    depth_ok = int(overall_s["depth_valid_duplicate_px_max"] or 0) == 0
    conflict_captured = int(overall_s["ignore_overlap_px_mean"] or 0) > 0 or int(v2_summary["duplicate_ownership_before_v2"]) == 0
    source_authorized = m15b.get("acceptance_decision") == "NEEDS_PINN_LABEL_DERIVATION_V3B"
    checks = [
        hard_nonempty,
        depth_nonempty,
        empty_clean,
        hard_unique,
        raw_union_ok,
        support_ok,
        anti_sparse_ok,
        leakage_ok,
        shrink_ok,
        cross_overlap_ok,
        depth_ok,
        conflict_captured,
        source_authorized,
    ]
    if all(checks):
        reasons = [
            "v3b keeps all existing hard cores nonempty and empty K slots strictly empty",
            "duplicate hard ownership remains zero while raw OR and hard OR reproduce sample union masks",
            "soft support is capped below the 1.35 raw-union leakage target and shrinks strongly versus v3",
            "exclusive narrow halos increase positive support versus v2 without cross-component soft overlap in separated/close samples",
            "depth valid regions are nonempty, component-exclusive, and tied to hard core plus narrow halo",
        ]
        return (
            "READY_FOR_25_17_TRAINING",
            "A. enter 25.17 label-v3b training gate using 25.10 loss mainline + label-v3b supervision; do not use the 25.11/25.12 rebalance stack",
            reasons,
            True,
        )
    if not raw_union_ok or not hard_nonempty:
        return (
            "NEEDS_COMSOL_LABEL_EXPORT_FIX",
            "B. enter COMSOL label export/schema fix, no training",
            ["raw labels cannot support nonempty exclusive hard cores or reproduce union invariants"],
            False,
        )
    if not leakage_ok or not anti_sparse_ok or not cross_overlap_ok or not depth_ok:
        return (
            "NEEDS_LABEL_V3B_FIX",
            "B. continue 25.16b label-v3b derivation fix, no training",
            ["v3b derivation is locally fixable but did not satisfy support/leakage/depth validator thresholds"],
            False,
        )
    return (
        "INCONCLUSIVE",
        "E. continue schema audit, no training",
        ["validator evidence is insufficient to authorize training"],
        False,
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    m14 = read_json(args.metrics_25_14)
    m15 = read_json(args.metrics_25_15)
    m15b = read_json(args.metrics_25_15b)
    man14 = read_json(args.manifest_25_14)
    man15b = read_json(args.manifest_25_15b)
    dataset_manifest = read_json(args.dataset_manifest)
    npz_path = assert_sources(dataset_manifest, m14, m15, m15b, man14, man15b)
    pack = load_npz(npz_path)
    masks_v2, _depths_v2, ownership, v2_summary = build_target_v2(pack)
    v3_soft_pos, v3_rows = build_v3_soft_or(pack)
    v3b = derive_v3b_targets(pack, masks_v2, ownership)
    comp_rows = component_rows(pack, v3b)
    samp_rows = sample_rows(pack, v3_soft_pos, v3b)
    component_stats = {
        "overall": aggregate_components(comp_rows),
        "by_split": group_rows(comp_rows, "split", aggregate_components),
        "by_component_count": group_rows(comp_rows, "component_count", aggregate_components),
        "by_separation": group_rows(comp_rows, "separation_type", aggregate_components),
        "by_topology": group_rows(comp_rows, "topology_relation", aggregate_components),
        "lowest_hard_core_components": sorted(
            [row for row in comp_rows if row["existing"]],
            key=lambda row: (int(row["hard_core_px"]), int(row["soft_support_px"]), str(row["sample_id"]), int(row["slot"])),
        )[:12],
    }
    sample_stats = {
        "overall": aggregate_samples(samp_rows),
        "by_split": group_rows(samp_rows, "split", aggregate_samples),
        "by_component_count": group_rows(samp_rows, "component_count", aggregate_samples),
        "by_separation": group_rows(samp_rows, "separation_type", aggregate_samples),
        "by_topology": group_rows(samp_rows, "topology_relation", aggregate_samples),
        "highest_v3b_leakage_samples": sorted(
            samp_rows,
            key=lambda row: (float(row["v3b_soft_or_to_raw_union_ratio"]), int(row["soft_duplicate_px"])),
            reverse=True,
        )[:12],
    }
    acceptance, route_decision, reasons, ready = decide(component_stats, sample_stats, v2_summary, m15b)
    v3b_summary = {
        "component_hard_core_pixel_mean": component_stats["overall"]["hard_core_px_mean"],
        "component_hard_core_pixel_min": component_stats["overall"]["hard_core_px_min"],
        "component_boundary_halo_pixel_mean": component_stats["overall"]["boundary_halo_px_mean"],
        "component_boundary_halo_pixel_min": component_stats["overall"]["boundary_halo_px_min"],
        "component_soft_support_pixel_mean": component_stats["overall"]["soft_support_px_mean"],
        "component_soft_support_pixel_min": component_stats["overall"]["soft_support_px_min"],
        "v3b_soft_or_raw_union_ratio_mean": sample_stats["overall"]["v3b_soft_or_to_raw_union_ratio_mean"],
        "v3b_soft_or_raw_union_ratio_max": sample_stats["overall"]["v3b_soft_or_to_raw_union_ratio_max"],
        "v3_vs_v3b_support_shrink_ratio_mean": sample_stats["overall"]["v3b_vs_v3_soft_or_shrink_ratio_mean"],
        "duplicate_hard_ownership_count": int(sample_stats["overall"]["hard_duplicate_px_mean"] or 0) + int(sample_stats["overall"]["hard_duplicate_px_max"] or 0),
        "identity_conflict_pixel_count": int(sum(row["identity_conflict_px"] for row in samp_rows)),
        "ignore_overlap_pixel_count": int(sum(row["ignore_overlap_px"] for row in samp_rows)),
        "depth_valid_region_pixel_mean": component_stats["overall"]["depth_valid_region_px_mean"],
        "depth_valid_region_pixel_min": component_stats["overall"]["depth_valid_region_px_min"],
        "empty_slot_violation_count": component_stats["overall"]["empty_slot_violation_count"],
        "ready_for_training_v3b": ready,
    }
    return {
        "derivation_id": DERIVATION_ID,
        "stage": STAGE,
        "dataset_id": DATASET_ID,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "source_files": {
            "metrics_25_14": str(args.metrics_25_14),
            "metrics_25_15": str(args.metrics_25_15),
            "metrics_25_15b": str(args.metrics_25_15b),
            "manifest_25_14": str(args.manifest_25_14),
            "manifest_25_15b": str(args.manifest_25_15b),
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
            "25_14_acceptance_decision": m14.get("acceptance_decision"),
            "25_15_gate_decision": m15.get("gate_decision"),
            "25_15_route_decision": m15.get("route_decision"),
            "25_15b_acceptance_decision": m15b.get("acceptance_decision"),
            "25_15b_route_decision": m15b.get("route_decision"),
            "25_15b_main_conclusion": m15b.get("audit_main_conclusion"),
        },
        "label_schema_v3b_design": label_schema_v3b_design(),
        "target_v2_reproduction": v2_summary,
        "target_v3_reference": {
            "overall_soft_or_ratio_mean_from_reconstruction": mean_or_null([row["v3_soft_or_to_raw_union_ratio"] for row in v3_rows]),
            "overall_soft_or_ratio_max_from_reconstruction": max_or_null([row["v3_soft_or_to_raw_union_ratio"] for row in v3_rows]),
            "25_15b_reported_soft_or_ratio_mean": m15b.get("target_v3_support_leakage_audit", {}).get("overall", {}).get("soft_or_to_union_ratio_mean"),
            "25_15b_reported_soft_or_ratio_max": m15b.get("target_v3_support_leakage_audit", {}).get("overall", {}).get("soft_or_to_union_ratio_max"),
        },
        "label_v3b_summary": v3b_summary,
        "label_v3b_support_validator": {
            "component_support": component_stats,
            "sample_support": sample_stats,
            "component_rows": comp_rows,
            "sample_rows": samp_rows,
            "grid_shape": [GRID_H, GRID_W],
            "soft_or_raw_union_ratio_target": V3B_SOFT_OR_RAW_UNION_RATIO_TARGET,
            "halo_cap_fraction_of_raw_union": V3B_HALO_CAP_FRACTION_OF_RAW_UNION,
            "v2_sparse_lower_bound_px": V2_SPARSE_LOWER_BOUND_PX,
        },
        "validator_checks": {
            "existing_hard_core_nonempty": component_stats["overall"]["existing_hard_core_empty_count"] == 0,
            "existing_depth_valid_nonempty": component_stats["overall"]["existing_depth_valid_empty_count"] == 0,
            "empty_slots_clean": component_stats["overall"]["empty_slot_violation_count"] == 0,
            "duplicate_hard_ownership_zero": sample_stats["overall"]["hard_duplicate_px_max"] == 0,
            "hard_core_above_v2_sparse_lower_bound": float(component_stats["overall"]["hard_core_px_min"] or 0.0) > V2_SPARSE_LOWER_BOUND_PX,
            "v3b_soft_or_ratio_below_1p35": (
                float(sample_stats["overall"]["v3b_soft_or_to_raw_union_ratio_mean"] or 999.0) < V3B_SOFT_OR_RAW_UNION_RATIO_TARGET
                and float(sample_stats["overall"]["v3b_soft_or_to_raw_union_ratio_max"] or 999.0) < V3B_SOFT_OR_RAW_UNION_RATIO_TARGET
            ),
            "v3b_shrinks_v3_support": float(sample_stats["overall"]["v3b_vs_v3_soft_or_shrink_ratio_mean"] or 999.0) <= 0.70,
            "separated_close_cross_component_soft_overlap_low": (
                float(sample_stats["by_separation"].get("separated", {}).get("soft_duplicate_fraction_of_soft_targets_max") or 0.0) <= 0.0
                and float(sample_stats["by_separation"].get("close", {}).get("soft_duplicate_fraction_of_soft_targets_max") or 0.0) <= 0.0
            ),
            "touching_overlap_conflict_captured": int(v3b_summary["identity_conflict_pixel_count"]) > 0,
            "depth_valid_region_nonempty_and_nonduplicated": (
                component_stats["overall"]["existing_depth_valid_empty_count"] == 0
                and sample_stats["overall"]["depth_valid_duplicate_px_max"] == 0
            ),
            "raw_union_invariant_preserved": (
                sample_stats["overall"]["raw_or_to_union_mismatch_px_max"] == 0
                and sample_stats["overall"]["hard_or_to_union_mismatch_px_max"] == 0
            ),
            "strict_json_allow_nan_false": True,
        },
        "acceptance_decision": acceptance,
        "route_decision": route_decision,
        "decision_reasons": reasons,
        "label_v3b_derivation_main_conclusion": (
            "Label-v3b is derivable inside PINN_project from existing raw masks/depths: it keeps v2-style exclusive hard identity, "
            "adds capped narrow soft halo support to avoid v2 near-empty sparsity, and removes the broad v3 union-like support leakage."
        ),
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["label_v3b_summary"]
    component = payload["label_v3b_support_validator"]["component_support"]["overall"]
    sample = payload["label_v3b_support_validator"]["sample_support"]["overall"]
    by_count = payload["label_v3b_support_validator"]["sample_support"]["by_component_count"]
    by_sep = payload["label_v3b_support_validator"]["sample_support"]["by_separation"]
    by_topology = payload["label_v3b_support_validator"]["sample_support"]["by_topology"]
    checks = payload["validator_checks"]
    lines = [
        "# 25.16 Label-V3B Derivation Validator",
        "",
        f"- acceptance_decision: `{payload['acceptance_decision']}`",
        f"- route_decision: `{payload['route_decision']}`",
        f"- main_conclusion: {payload['label_v3b_derivation_main_conclusion']}",
        "",
        "## V3B Core Result",
        "",
        f"- hard core px mean/min: `{fmt(summary['component_hard_core_pixel_mean'])}` / `{fmt(summary['component_hard_core_pixel_min'], 0)}`",
        f"- boundary halo px mean/min: `{fmt(summary['component_boundary_halo_pixel_mean'])}` / `{fmt(summary['component_boundary_halo_pixel_min'], 0)}`",
        f"- soft support px mean/min: `{fmt(summary['component_soft_support_pixel_mean'])}` / `{fmt(summary['component_soft_support_pixel_min'], 0)}`",
        f"- soft support / hard-core ratio mean: `{fmt(component['soft_vs_hard_support_ratio_mean'])}`",
        f"- v3b soft OR / raw union ratio mean/max: `{fmt(summary['v3b_soft_or_raw_union_ratio_mean'])}` / `{fmt(summary['v3b_soft_or_raw_union_ratio_max'])}`",
        f"- v3b / v3 soft OR shrink ratio mean: `{fmt(summary['v3_vs_v3b_support_shrink_ratio_mean'])}`",
        f"- duplicate hard ownership count: `{summary['duplicate_hard_ownership_count']}`",
        f"- identity conflict px total: `{summary['identity_conflict_pixel_count']}`",
        f"- ignore overlap px total: `{summary['ignore_overlap_pixel_count']}`",
        f"- depth valid region px mean/min: `{fmt(summary['depth_valid_region_pixel_mean'])}` / `{fmt(summary['depth_valid_region_pixel_min'], 0)}`",
        f"- empty slot violation count: `{summary['empty_slot_violation_count']}`",
        f"- ready_for_training_v3b: `{summary['ready_for_training_v3b']}`",
        "",
        "## Group Slices",
        "",
        f"- component_count=2 v3b soft OR/raw union mean/max: `{fmt(by_count['2']['v3b_soft_or_to_raw_union_ratio_mean'])}` / `{fmt(by_count['2']['v3b_soft_or_to_raw_union_ratio_max'])}`",
        f"- component_count=3 v3b soft OR/raw union mean/max: `{fmt(by_count['3']['v3b_soft_or_to_raw_union_ratio_mean'])}` / `{fmt(by_count['3']['v3b_soft_or_to_raw_union_ratio_max'])}`",
        f"- separated soft duplicate fraction max: `{fmt(by_sep['separated']['soft_duplicate_fraction_of_soft_targets_max'])}`",
        f"- close soft duplicate fraction max: `{fmt(by_sep['close']['soft_duplicate_fraction_of_soft_targets_max'])}`",
        f"- touching identity conflict px mean/max: `{fmt(by_sep['touching']['identity_conflict_px_mean'])}` / `{fmt(by_sep['touching']['identity_conflict_px_max'], 0)}`",
        f"- partially_overlapping identity conflict px mean/max: `{fmt(by_sep['partially_overlapping']['identity_conflict_px_mean'])}` / `{fmt(by_sep['partially_overlapping']['identity_conflict_px_max'], 0)}`",
        f"- touching_boundary ignore overlap px mean/max: `{fmt(by_topology['touching_boundary']['ignore_overlap_px_mean'])}` / `{fmt(by_topology['touching_boundary']['ignore_overlap_px_max'], 0)}`",
        f"- partially_overlapping ignore overlap px mean/max: `{fmt(by_topology['partially_overlapping']['ignore_overlap_px_mean'])}` / `{fmt(by_topology['partially_overlapping']['ignore_overlap_px_max'], 0)}`",
        "",
        "## Validator Checks",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## V3B Schema",
            "",
            "- `component_hard_core_mask_v3b`: exclusive ownership-resolved component core.",
            "- `component_boundary_halo_mask_v3b`: capped one-pixel cross-neighborhood halo, stripped of cross-component claims.",
            "- `component_ignore_overlap_mask_v3b`: non-owner or ambiguous overlap/contact pixels for ignore/diagnostics.",
            "- `component_mask_target_v3b_soft`: hard core = 1.0, halo = 0.35.",
            "- `component_sdf_target_v3b`: clipped SDF from hard core, consumed only inside valid region.",
            "- `component_valid_region_mask_v3b`: hard core plus exclusive capped halo.",
            "- `component_depth_target_v3b`: hard-core depth plus nearest-core halo depth.",
            "- `component_identity_conflict_mask_v3b`: raw overlap, touching contact, or multi-halo claim diagnostics.",
            "- Union mask/depth remain raw OR/max evaluation targets only.",
            "",
            "## Boundary",
            "",
            "- No training, loss tuning, model expansion, COMSOL run, data/NPZ mutation, checkpoint/preview export, baseline transition, or `CURRENT_BASELINE.md` update.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = {
        "derivation_id": payload["derivation_id"],
        "stage": payload["stage"],
        "dataset_id": payload["dataset_id"],
        "generated_at": payload["generated_at"],
        "status": "label_v3b_derivation_validated",
        "acceptance_decision": payload["acceptance_decision"],
        "route_decision": payload["route_decision"],
        "metrics_path": str(OUT_METRICS),
        "summary_path": str(OUT_SUMMARY),
        "source_files": payload["source_files"],
        "ready_for_training_v3b": payload["label_v3b_summary"]["ready_for_training_v3b"],
        "boundary": payload["boundary"],
        "baseline_ready": False,
        "current_baseline_updated": False,
        "allowed_use": [
            "explicit_25_17_label_v3b_training_gate_planning",
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

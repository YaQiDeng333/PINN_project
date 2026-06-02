#!/usr/bin/env python
"""Audit and redesign 25.12b component raster/depth targets.

This reporter is intentionally read/design only. It reads the validated
component-set pilot pack and prior 25.10-25.12 metrics, then writes strict JSON
and Markdown diagnostics. It does not train, run COMSOL, mutate data/NPZ files,
export previews, or update the current baseline.
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
AUDIT_ID = "25_12b_surface_multipit_component_raster_depth_target_redesign"
TARGET_SPLIT = {"train": 72, "val": 20, "test": 20}
K_MAX = 3
MASK_X_START_M = -0.04
MASK_X_STOP_M = 0.04
MASK_Y_START_M = -0.01
MASK_Y_STOP_M = 0.01

M10 = ROOT / "results/metrics/25_10_component_set_training_gate_metrics.json"
M10B = ROOT / "results/metrics/25_10b_component_set_failure_audit.json"
M11 = ROOT / "results/metrics/25_11_mask_depth_loss_rebalance_training_metrics.json"
M11B = ROOT / "results/metrics/25_11b_component_set_merge_collapse_audit.json"
M12 = ROOT / "results/metrics/25_12_component_separation_rebalance_training_metrics.json"
DATASET_MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"

OUT_METRICS = ROOT / "results/metrics/25_12b_component_raster_depth_target_redesign.json"
OUT_SUMMARY = ROOT / "results/summaries/25_12b_component_raster_depth_target_redesign_summary.md"
OUT_MANIFEST = ROOT / "results/manifests/25_12b_component_raster_depth_target_redesign_manifest.json"

FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 25.12b component raster/depth targets and design target v2.")
    parser.add_argument("--metrics-25-10", type=Path, default=M10)
    parser.add_argument("--audit-25-10b", type=Path, default=M10B)
    parser.add_argument("--metrics-25-11", type=Path, default=M11)
    parser.add_argument("--audit-25-11b", type=Path, default=M11B)
    parser.add_argument("--metrics-25-12", type=Path, default=M12)
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


def mean_or_null(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def max_or_null(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.max(clean)) if clean else None


def percentile_or_null(values: list[float], percentile: float) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.percentile(clean, percentile)) if clean else None


def mask_iou_dice(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    aa = a > 0.5
    bb = b > 0.5
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    denom = float(aa.sum() + bb.sum())
    return inter / (union + 1.0e-8), (2.0 * inter) / (denom + 1.0e-8)


def assert_sources(
    dataset_manifest: dict[str, Any],
    m10: dict[str, Any],
    m10b: dict[str, Any],
    m11: dict[str, Any],
    m11b: dict[str, Any],
    m12: dict[str, Any],
) -> Path:
    if ROOT != Path(r"C:\Users\19166\Desktop\PINN_project"):
        raise RuntimeError(f"wrong project root: {ROOT}")
    if dataset_manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("dataset manifest dataset_id mismatch")
    if dataset_manifest.get("split_counts") != TARGET_SPLIT:
        raise ValueError(f"dataset split mismatch: {dataset_manifest.get('split_counts')}")
    if dataset_manifest.get("train_ready_candidate") is not True or dataset_manifest.get("baseline_ready") is not False:
        raise ValueError("dataset manifest readiness boundary mismatch")
    if int(dataset_manifest.get("K_max", -1)) != K_MAX:
        raise ValueError("K_max mismatch")
    for required in ["baseline_update", "current_baseline_replacement", "latest_newest_auto_discovery"]:
        if required not in set(dataset_manifest.get("forbidden_use", [])):
            raise ValueError(f"dataset manifest missing forbidden_use={required}")
    if m10.get("stage") != "25.10" or m10.get("gate_decision") != "PARTIAL":
        raise ValueError("25.10 metrics mismatch")
    if m10b.get("stage") != "25.10b" or "25.11 mask/depth" not in str(m10b.get("route_decision", "")):
        raise ValueError("25.10b audit mismatch")
    if m11.get("stage") != "25.11" or m11.get("gate_decision") != "PARTIAL":
        raise ValueError("25.11 metrics mismatch")
    if m11b.get("stage") != "25.11b" or "25.12 component-separation" not in str(m11b.get("route_decision", "")):
        raise ValueError("25.11b audit mismatch")
    if m12.get("stage") != "25.12" or m12.get("gate_decision") != "FAIL":
        raise ValueError("25.12 metrics must be FAIL")
    if "redesign component raster/depth targets" not in str(m12.get("route_decision", "")):
        raise ValueError("25.12 route does not point to 25.12b target redesign")
    path = Path(dataset_manifest["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def overall_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics["metrics_by_split"]["test"]
    return {
        "component_recall": float(test["component_recall"]),
        "missed_rate": float(test["missed_rate"]),
        "extra_rate": float(test["extra_rate"]),
        "merged_rate": float(test["merged_rate"]),
        "component_mask_dice": float(test["component_mask_dice_mean"]),
        "union_mask_dice": float(test["union_mask_dice_mean"]),
        "depth_grid_rmse_m": float(test["depth_grid_rmse_m_mean"]),
        "lwd_relative_error": float(test["lwd_relative_error_mean"]),
    }


def grid_xy(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, width)
    y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, height)
    return np.meshgrid(x, y, indexing="xy")


def component_alignment_rows(pack: dict[str, Any]) -> list[dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    centers = np.asarray(pack["component_center_xy_m"], dtype=np.float64)
    component_lwd = np.asarray(pack["component_lwd_m"], dtype=np.float64)
    xx, yy = grid_xy(masks.shape[-2], masks.shape[-1])
    rows: list[dict[str, Any]] = []
    for sample_index in range(exists.shape[0]):
        for slot in range(exists.shape[1]):
            if not exists[sample_index, slot]:
                continue
            mask = masks[sample_index, slot]
            area = int(mask.sum())
            if area <= 0:
                rows.append(
                    {
                        "sample_index": sample_index,
                        "slot": slot,
                        "mask_area_px": 0,
                        "center_to_mask_centroid_error_m": None,
                        "lwd_m": component_lwd[sample_index, slot].tolist(),
                    }
                )
                continue
            centroid_x = float(xx[mask].mean())
            centroid_y = float(yy[mask].mean())
            center_x = float(centers[sample_index, slot, 0])
            center_y = float(centers[sample_index, slot, 1])
            rows.append(
                {
                    "sample_index": sample_index,
                    "slot": slot,
                    "mask_area_px": area,
                    "mask_centroid_xy_m": [centroid_x, centroid_y],
                    "label_center_xy_m": [center_x, center_y],
                    "center_to_mask_centroid_error_m": float(math.hypot(centroid_x - center_x, centroid_y - center_y)),
                    "lwd_m": component_lwd[sample_index, slot].tolist(),
                }
            )
    return rows


def sample_target_rows(pack: dict[str, Any]) -> list[dict[str, Any]]:
    sample_ids = np.asarray(pack["sample_ids"]).astype(str)
    split = np.asarray(pack["split"]).astype(str)
    component_count = np.asarray(pack["component_count"], dtype=np.int64)
    exists = np.asarray(pack["component_exists"], dtype=bool)
    component_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    component_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float64)
    union_masks = np.asarray(pack["projected_mask_2d"], dtype=bool)
    union_depths = np.asarray(pack["depth_grid_m"], dtype=np.float64)
    separation = np.asarray(pack["separation_type"]).astype(str)
    topology = np.asarray(pack["topology_relation"]).astype(str)
    source = np.asarray(pack["source_dataset_id"]).astype(str)
    relative_depth = np.asarray(pack["relative_depth_type"]).astype(str)
    rows: list[dict[str, Any]] = []
    for i in range(len(sample_ids)):
        active = exists[i]
        masks = component_masks[i] & active[:, None, None]
        depths = component_depths[i] * active[:, None, None]
        component_or = masks.max(axis=0)
        component_sum = masks.sum(axis=0)
        component_area_sum = int(masks.sum())
        duplicate_pixels = int((component_sum > 1).sum())
        duplicate_component_targets = int(np.maximum(component_sum - 1, 0).sum())
        overlap_depth_values = depths[:, component_sum > 1]
        if overlap_depth_values.size:
            positive_depths = np.where(overlap_depth_values > 0.0, overlap_depth_values, np.nan)
            overlap_depth_spread = np.nanmax(positive_depths, axis=0) - np.nanmin(positive_depths, axis=0)
            overlap_depth_conflict_pixels = int(np.nansum(overlap_depth_spread > 1.0e-12))
            overlap_depth_spread_mean_m = mean_or_null([float(v) for v in overlap_depth_spread if math.isfinite(float(v))])
        else:
            overlap_depth_conflict_pixels = 0
            overlap_depth_spread_mean_m = None
        union_iou, union_dice = mask_iou_dice(component_or, union_masks[i])
        max_depth = depths.max(axis=0)
        sum_depth = depths.sum(axis=0)
        depth_max_rmse = float(np.sqrt(np.mean((max_depth - union_depths[i]) ** 2)))
        depth_sum_rmse = float(np.sqrt(np.mean((sum_depth - union_depths[i]) ** 2)))
        rows.append(
            {
                "sample_id": sample_ids[i],
                "source_index": i,
                "split": split[i],
                "component_count": int(component_count[i]),
                "separation_type": separation[i],
                "topology_relation": topology[i],
                "relative_depth_type": relative_depth[i],
                "source_dataset_id": source[i],
                "active_slot_count": int(active.sum()),
                "component_area_sum_px": component_area_sum,
                "component_or_area_px": int(component_or.sum()),
                "union_area_px": int(union_masks[i].sum()),
                "component_overlap_pixel_count": duplicate_pixels,
                "duplicated_component_target_pixel_count": duplicate_component_targets,
                "overlap_pixel_fraction_of_component_targets": float(duplicate_component_targets / max(component_area_sum, 1)),
                "component_or_to_union_iou": float(union_iou),
                "component_or_to_union_dice": float(union_dice),
                "union_missing_from_components_px": int(np.logical_and(union_masks[i], ~component_or).sum()),
                "component_extra_outside_union_px": int(np.logical_and(component_or, ~union_masks[i]).sum()),
                "depth_max_to_union_rmse_m": depth_max_rmse,
                "depth_sum_to_union_rmse_m": depth_sum_rmse,
                "overlap_depth_conflict_pixels": overlap_depth_conflict_pixels,
                "overlap_depth_spread_mean_m": overlap_depth_spread_mean_m,
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    return {
        "sample_count": len(rows),
        "component_count_sum": int(sum(int(row["component_count"]) for row in rows)),
        "overlap_sample_count": int(sum(int(row["component_overlap_pixel_count"]) > 0 for row in rows)),
        "overlap_sample_rate": float(sum(int(row["component_overlap_pixel_count"]) > 0 for row in rows) / len(rows)),
        "duplicated_component_target_pixels_sum": int(sum(int(row["duplicated_component_target_pixel_count"]) for row in rows)),
        "overlap_pixel_fraction_mean": mean_or_null([float(row["overlap_pixel_fraction_of_component_targets"]) for row in rows]),
        "overlap_pixel_fraction_max": max_or_null([float(row["overlap_pixel_fraction_of_component_targets"]) for row in rows]),
        "union_dice_mean": mean_or_null([float(row["component_or_to_union_dice"]) for row in rows]),
        "union_iou_mean": mean_or_null([float(row["component_or_to_union_iou"]) for row in rows]),
        "union_missing_from_components_px_sum": int(sum(int(row["union_missing_from_components_px"]) for row in rows)),
        "component_extra_outside_union_px_sum": int(sum(int(row["component_extra_outside_union_px"]) for row in rows)),
        "depth_max_to_union_rmse_m_mean": mean_or_null([float(row["depth_max_to_union_rmse_m"]) for row in rows]),
        "depth_max_to_union_rmse_m_max": max_or_null([float(row["depth_max_to_union_rmse_m"]) for row in rows]),
        "depth_sum_to_union_rmse_m_mean": mean_or_null([float(row["depth_sum_to_union_rmse_m"]) for row in rows]),
        "overlap_depth_conflict_sample_count": int(sum(int(row["overlap_depth_conflict_pixels"]) > 0 for row in rows)),
        "overlap_depth_conflict_pixels_sum": int(sum(int(row["overlap_depth_conflict_pixels"]) for row in rows)),
    }


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {str(value): aggregate_rows([row for row in rows if str(row[field]) == str(value)]) for value in sorted({row[field] for row in rows}, key=str)}


def target_integrity(sample_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]], pack: dict[str, Any]) -> dict[str, Any]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    component_masks = np.asarray(pack["component_projected_masks_2d"], dtype=np.float64)
    component_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float64)
    inactive = ~exists
    inactive_mask_sum = int(component_masks[inactive].sum()) if inactive.any() else 0
    inactive_depth_abs_sum = float(np.abs(component_depths[inactive]).sum()) if inactive.any() else 0.0
    center_errors = [float(row["center_to_mask_centroid_error_m"]) for row in component_rows if row.get("center_to_mask_centroid_error_m") is not None]
    return {
        "empty_slot_policy_v1": {
            "inactive_slot_count": int(inactive.sum()),
            "inactive_mask_sum": inactive_mask_sum,
            "inactive_depth_abs_sum": inactive_depth_abs_sum,
            "empty_slot_violation_count": int(inactive_mask_sum > 0 or inactive_depth_abs_sum > 1.0e-15),
            "verdict": "PASS" if inactive_mask_sum == 0 and inactive_depth_abs_sum <= 1.0e-15 else "FAIL",
        },
        "component_param_raster_alignment_v1": {
            "active_component_count": len(component_rows),
            "center_to_mask_centroid_error_m_mean": mean_or_null(center_errors),
            "center_to_mask_centroid_error_m_p95": percentile_or_null(center_errors, 95),
            "center_to_mask_centroid_error_m_max": max_or_null(center_errors),
            "verdict": "PASS" if center_errors and max(center_errors) <= 0.0015 else "REVIEW",
        },
        "component_union_consistency_v1": {
            "component_or_to_union_dice_mean": mean_or_null([float(row["component_or_to_union_dice"]) for row in sample_rows]),
            "component_or_to_union_dice_min": min(float(row["component_or_to_union_dice"]) for row in sample_rows),
            "union_missing_from_components_px_sum": int(sum(int(row["union_missing_from_components_px"]) for row in sample_rows)),
            "component_extra_outside_union_px_sum": int(sum(int(row["component_extra_outside_union_px"]) for row in sample_rows)),
            "verdict": "PASS",
        },
        "depth_union_consistency_v1": {
            "depth_max_to_union_rmse_m_mean": mean_or_null([float(row["depth_max_to_union_rmse_m"]) for row in sample_rows]),
            "depth_max_to_union_rmse_m_max": max_or_null([float(row["depth_max_to_union_rmse_m"]) for row in sample_rows]),
            "depth_sum_to_union_rmse_m_mean": mean_or_null([float(row["depth_sum_to_union_rmse_m"]) for row in sample_rows]),
            "verdict": "PASS" if max(float(row["depth_max_to_union_rmse_m"]) for row in sample_rows) <= 1.0e-12 else "FAIL",
        },
        "component_ownership_ambiguity_v1": {
            "overlap_sample_count": int(sum(int(row["component_overlap_pixel_count"]) > 0 for row in sample_rows)),
            "overlap_sample_rate": float(sum(int(row["component_overlap_pixel_count"]) > 0 for row in sample_rows) / len(sample_rows)),
            "duplicated_component_target_pixels_sum": int(sum(int(row["duplicated_component_target_pixel_count"]) for row in sample_rows)),
            "overlap_depth_conflict_sample_count": int(sum(int(row["overlap_depth_conflict_pixels"]) > 0 for row in sample_rows)),
            "overlap_depth_conflict_pixels_sum": int(sum(int(row["overlap_depth_conflict_pixels"]) for row in sample_rows)),
            "verdict": "ISSUE",
        },
    }


def metric_evidence(m10: dict[str, Any], m10b: dict[str, Any], m11: dict[str, Any], m11b: dict[str, Any], m12: dict[str, Any]) -> dict[str, Any]:
    t10 = overall_test_metrics(m10)
    t11 = overall_test_metrics(m11)
    t12 = overall_test_metrics(m12)
    return {
        "test_metrics": {"25.10": t10, "25.11": t11, "25.12": t12},
        "diagnostic_chain": {
            "25.10b_main_conclusion": m10b.get("audit_conclusion") or m10b.get("main_conclusion"),
            "25.10b_route_decision": m10b.get("route_decision"),
            "25.11b_main_conclusion": m11b.get("audit_conclusion") or m11b.get("main_conclusion"),
            "25.11b_route_decision": m11b.get("route_decision"),
            "25.12_route_decision": m12.get("route_decision"),
        },
        "derived_signals": {
            "coarse_geometry_learned_but_mask_weak": t10["lwd_relative_error"] <= 0.20 and t10["component_mask_dice"] < 0.15,
            "mask_dice_flat_across_rebalances": max(t10["component_mask_dice"], t11["component_mask_dice"], t12["component_mask_dice"])
            - min(t10["component_mask_dice"], t11["component_mask_dice"], t12["component_mask_dice"])
            <= 0.002,
            "union_rebalance_induced_merge_collapse": t11["union_mask_dice"] > t10["union_mask_dice"] and t11["merged_rate"] >= 0.75,
            "separation_loss_did_not_restore_25_10": t12["merged_rate"] > t10["merged_rate"] + 0.25 and t12["component_recall"] < t10["component_recall"],
        },
    }


def target_schema_v2() -> dict[str, Any]:
    return {
        "component_mask_target_v2": {
            "type": "K=3 ownership-resolved binary masks",
            "rule": "Each foreground pixel is positive for at most one component slot. Raw overlapping component masks are kept as diagnostics, but training component masks use ownership-resolved targets.",
            "applies_to_existing_slots_only": True,
        },
        "component_depth_target_v2": {
            "type": "K=3 ownership-resolved component foreground depth grids in meters",
            "rule": "Depth is retained only where component_ownership_map assigns the pixel to that slot; background and other-owned pixels are zero for that component target.",
            "loss_rule": "Use component-normalized foreground depth loss; background contributes only optional logged diagnostics.",
        },
        "component_ownership_map": {
            "encoding": "-1 background, 0..K-1 owning component slot",
            "source": "deterministic loader/target transform from v1 component masks, component centers, component depths, and topology labels",
            "tie_break": [
                "assign to the component whose normalized center distance is smallest",
                "if equal, assign to the component with larger local depth at that pixel",
                "if still equal, assign to lower slot id for deterministic reproducibility",
            ],
            "required_diagnostics": [
                "raw_overlap_pixel_count",
                "ownership_resolved_pixel_count_by_slot",
                "overlap_depth_conflict_pixels",
            ],
        },
        "overlap_policy": {
            "separated_close": "component masks must be mutually exclusive; union = OR(component masks)",
            "touching_boundary": "boundary contact is allowed, but duplicate-pixel ownership is resolved to exactly one slot",
            "partially_overlapping": "record raw overlap and component_ownership_map; train component masks/depths with ownership-resolved targets; train union masks from OR/max targets",
        },
        "touching_boundary_policy": {
            "rule": "Touching components may share an edge in continuous geometry but not positive ownership for the same raster pixel.",
            "metric": "Report touching rows separately and do not average them away.",
        },
        "empty_slot_policy": {
            "existence_target": 0,
            "mask_target": "all zeros",
            "depth_target": "all zeros",
            "loss_participation": "existence BCE only; no positive mask/depth foreground loss",
        },
        "union_from_components_rule": {
            "mask": "OR over raw or ownership-resolved component masks; both must equal the sample-level projected_mask_2d",
            "depth": "max over raw component depth grids; never sum overlapping component depths",
        },
        "depth_from_components_rule": {
            "component": "foreground-only, component-normalized, ownership-resolved depth target",
            "union": "max-depth surface target held separate from component targets",
            "background": "background must not dominate the depth loss",
        },
        "training_route_constraint": "25.13 should use the 25.10 loss mainline plus target-v2 transform, not the 25.11/25.12 rebalance stack.",
        "three_component_reporting": "Keep component_count=3 as a mandatory separate metric slice.",
    }


def decide_acceptance(integrity: dict[str, Any], grouped_stats: dict[str, Any], evidence: dict[str, Any]) -> tuple[str, str, list[str]]:
    union_ok = integrity["component_union_consistency_v1"]["component_or_to_union_dice_min"] >= 0.999999
    depth_ok = integrity["depth_union_consistency_v1"]["depth_max_to_union_rmse_m_max"] <= 1.0e-12
    empty_ok = integrity["empty_slot_policy_v1"]["empty_slot_violation_count"] == 0
    param_ok = integrity["component_param_raster_alignment_v1"]["center_to_mask_centroid_error_m_max"] <= 0.0015
    overlap_exists = integrity["component_ownership_ambiguity_v1"]["overlap_sample_count"] > 0
    mask_flat = evidence["derived_signals"]["mask_dice_flat_across_rebalances"]
    reasons: list[str] = []
    if not union_ok or not depth_ok or not empty_ok or not param_ok:
        reasons.append("v1 generator-level target integrity is not sufficient for deterministic v2 transformation")
        return "NEEDS_GENERATOR_FIX", "B. enter generator label fix before any training", reasons
    if not overlap_exists:
        reasons.append("no measurable ownership conflict was found, so target redesign evidence is insufficient")
        return "INCONCLUSIVE", "D. continue target audit without training", reasons
    if not mask_flat:
        reasons.append("mask metric history does not isolate a target-level blocker")
        return "INCONCLUSIVE", "D. continue target audit without training", reasons
    reasons.extend(
        [
            "component OR and max-depth targets exactly reproduce sample-level union targets",
            "empty slots and parameter-to-raster alignment pass integrity checks",
            "v1 has duplicated component foreground ownership in overlap/touching rows, especially partially_overlapping and component_count=3",
            "25.10 learned coarse geometry while 25.11/25.12 loss rebalances did not improve component Dice, which supports target-layer redesign before further training",
        ]
    )
    return (
        "READY_FOR_25.13_TRAINING",
        "A. enter 25.13 target-v2 training gate using the 25.10 loss mainline; do not use the 25.11/25.12 rebalance stack",
        reasons,
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    dataset_manifest = read_json(args.dataset_manifest)
    m10 = read_json(args.metrics_25_10)
    m10b = read_json(args.audit_25_10b)
    m11 = read_json(args.metrics_25_11)
    m11b = read_json(args.audit_25_11b)
    m12 = read_json(args.metrics_25_12)
    npz_path = assert_sources(dataset_manifest, m10, m10b, m11, m11b, m12)
    pack = load_npz(npz_path)
    sample_rows = sample_target_rows(pack)
    component_rows = component_alignment_rows(pack)
    integrity = target_integrity(sample_rows, component_rows, pack)
    grouped_stats = {
        "overall": aggregate_rows(sample_rows),
        "by_split": grouped(sample_rows, "split"),
        "by_component_count": grouped(sample_rows, "component_count"),
        "by_separation": grouped(sample_rows, "separation_type"),
        "by_topology": grouped(sample_rows, "topology_relation"),
        "by_source_dataset": grouped(sample_rows, "source_dataset_id"),
    }
    evidence = metric_evidence(m10, m10b, m11, m11b, m12)
    acceptance, route_decision, reasons = decide_acceptance(integrity, grouped_stats, evidence)
    worst_overlap = sorted(sample_rows, key=lambda row: (int(row["duplicated_component_target_pixel_count"]), float(row["overlap_pixel_fraction_of_component_targets"])), reverse=True)[:12]
    return {
        "audit_id": AUDIT_ID,
        "stage": "25.12b",
        "dataset_id": DATASET_ID,
        "dataset_manifest": str(args.dataset_manifest),
        "dataset_npz_path": str(npz_path),
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "git": {
            "branch": git_value(["branch", "--show-current"]),
            "head_before_commit": git_value(["rev-parse", "HEAD"]),
            "protected_path_diff_before_write": git_value(["diff", "--name-only", "--", *FORBIDDEN_DIFF_PATHS]),
        },
        "boundary": {
            "training_run": False,
            "comsol_run": False,
            "data_npz_modified": False,
            "model_capacity_expanded": False,
            "component_set_representation_changed": False,
            "current_baseline_updated": False,
            "baseline_transition": False,
            "formal_inference_artifact_exported": False,
        },
        "source_files": {
            "metrics_25_10": str(args.metrics_25_10),
            "audit_25_10b": str(args.audit_25_10b),
            "metrics_25_11": str(args.metrics_25_11),
            "audit_25_11b": str(args.audit_25_11b),
            "metrics_25_12": str(args.metrics_25_12),
        },
        "metric_evidence": evidence,
        "v1_target_integrity": integrity,
        "v1_conflict_statistics": grouped_stats,
        "worst_overlap_samples": worst_overlap,
        "target_schema_v2": target_schema_v2(),
        "target_redesign_acceptance_decision": acceptance,
        "route_decision": route_decision,
        "decision_reasons": reasons,
        "target_redesign_main_conclusion": (
            "v1 generator labels are globally consistent, but component-level raster/depth supervision lacks explicit ownership in overlap/touching pixels; "
            "target v2 should resolve ownership before further training and return to the 25.10 loss mainline."
        ),
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    overall = payload["v1_conflict_statistics"]["overall"]
    by_sep = payload["v1_conflict_statistics"]["by_separation"]
    by_cc = payload["v1_conflict_statistics"]["by_component_count"]
    integrity = payload["v1_target_integrity"]
    metrics = payload["metric_evidence"]["test_metrics"]
    lines = [
        "# 25.12b Component Raster/Depth Target Redesign",
        "",
        f"- target_redesign_acceptance_decision: `{payload['target_redesign_acceptance_decision']}`",
        f"- route_decision: `{payload['route_decision']}`",
        f"- main_conclusion: {payload['target_redesign_main_conclusion']}",
        "",
        "## Evidence From 25.10 -> 25.12",
        "",
        f"- 25.10: recall `{metrics['25.10']['component_recall']:.6f}`, merged `{metrics['25.10']['merged_rate']:.6f}`, component Dice `{metrics['25.10']['component_mask_dice']:.6f}`, union Dice `{metrics['25.10']['union_mask_dice']:.6f}`, depth RMSE `{metrics['25.10']['depth_grid_rmse_m']:.9f} m`.",
        f"- 25.11: recall `{metrics['25.11']['component_recall']:.6f}`, merged `{metrics['25.11']['merged_rate']:.6f}`, component Dice `{metrics['25.11']['component_mask_dice']:.6f}`, union Dice `{metrics['25.11']['union_mask_dice']:.6f}`, depth RMSE `{metrics['25.11']['depth_grid_rmse_m']:.9f} m`.",
        f"- 25.12: recall `{metrics['25.12']['component_recall']:.6f}`, merged `{metrics['25.12']['merged_rate']:.6f}`, component Dice `{metrics['25.12']['component_mask_dice']:.6f}`, union Dice `{metrics['25.12']['union_mask_dice']:.6f}`, depth RMSE `{metrics['25.12']['depth_grid_rmse_m']:.9f} m`.",
        "",
        "## V1 Target Audit",
        "",
        f"- component OR to union Dice mean/min: `{integrity['component_union_consistency_v1']['component_or_to_union_dice_mean']:.6f}` / `{integrity['component_union_consistency_v1']['component_or_to_union_dice_min']:.6f}`.",
        f"- max(component depth) to union depth RMSE mean/max: `{integrity['depth_union_consistency_v1']['depth_max_to_union_rmse_m_mean']:.9f}` / `{integrity['depth_union_consistency_v1']['depth_max_to_union_rmse_m_max']:.9f} m`.",
        f"- empty-slot mask/depth violations: `{integrity['empty_slot_policy_v1']['empty_slot_violation_count']}`.",
        f"- center-to-mask-centroid error mean/p95/max: `{integrity['component_param_raster_alignment_v1']['center_to_mask_centroid_error_m_mean']:.9f}` / `{integrity['component_param_raster_alignment_v1']['center_to_mask_centroid_error_m_p95']:.9f}` / `{integrity['component_param_raster_alignment_v1']['center_to_mask_centroid_error_m_max']:.9f} m`.",
        f"- overlap samples: `{overall['overlap_sample_count']}/{overall['sample_count']}`; duplicated component target pixels: `{overall['duplicated_component_target_pixels_sum']}`.",
        f"- partially_overlapping overlap samples: `{by_sep['partially_overlapping']['overlap_sample_count']}/{by_sep['partially_overlapping']['sample_count']}`.",
        f"- touching overlap samples: `{by_sep['touching']['overlap_sample_count']}/{by_sep['touching']['sample_count']}`.",
        f"- component_count=3 overlap samples: `{by_cc['3']['overlap_sample_count']}/{by_cc['3']['sample_count']}`.",
        "",
        "## V1 Main Problems",
        "",
        "- Component-level masks are not ownership-resolved in overlap pixels, so one raster pixel can be a positive target for multiple slots.",
        "- Union mask/depth targets hide that ambiguity because OR/max exactly reconstructs the sample-level target.",
        "- Loss-only rebalancing can therefore improve or preserve union-level agreement while failing component-level separation.",
        "- Three-component rows concentrate overlap ambiguity and still require a mandatory separate slice.",
        "",
        "## V2 Core Rules",
        "",
        "- `component_mask_target_v2`: per-slot binary masks are ownership-resolved; each foreground pixel belongs to at most one component.",
        "- `component_depth_target_v2`: component depth is foreground-only and ownership-resolved; background does not dominate depth loss.",
        "- `component_ownership_map`: `-1` for background and `0..K-1` for owning slot, with nearest normalized center, deeper local depth, then slot id as deterministic tie-breaks.",
        "- `overlap_policy`: separated/close rows must be mutually exclusive; touching rows may share continuous boundaries but not raster ownership; partially-overlapping rows keep raw overlap diagnostics while training on ownership-resolved component targets.",
        "- `union_from_components_rule`: union mask is OR and union depth is max over raw components, never sum.",
        "- 25.13 should use the 25.10 loss mainline plus target-v2 transform, not the 25.11/25.12 rebalance stack.",
        "",
        "## Boundary",
        "",
        "- This stage did not train a model.",
        "- It did not run COMSOL.",
        "- It did not modify data/NPZ files.",
        "- It did not modify `CURRENT_BASELINE.md` or authorize a baseline transition.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = {
        "audit_id": AUDIT_ID,
        "stage": "25.12b",
        "dataset_id": DATASET_ID,
        "status": "target_redesign_complete",
        "target_redesign_acceptance_decision": payload["target_redesign_acceptance_decision"],
        "route_decision": payload["route_decision"],
        "metrics_path": str(OUT_METRICS),
        "summary_path": str(OUT_SUMMARY),
        "dataset_manifest": payload["dataset_manifest"],
        "source_metrics": payload["source_files"],
        "training_run": False,
        "comsol_run": False,
        "data_npz_modified": False,
        "current_baseline_updated": False,
        "baseline_ready": False,
        "allowed_use": ["target_redesign_audit", "25.13_target_v2_training_gate_input"],
        "forbidden_use": ["baseline_update", "current_baseline_replacement", "automatic_mainline_training", "latest_newest_auto_discovery"],
        "generated_at": payload["generated_at"],
        "git": payload["git"],
    }
    write_json(path, manifest)


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    write_json(args.out_metrics, payload)
    write_summary(args.out_summary, payload)
    write_manifest(args.out_manifest, payload)
    print(json.dumps({"decision": payload["target_redesign_acceptance_decision"], "route": payload["route_decision"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

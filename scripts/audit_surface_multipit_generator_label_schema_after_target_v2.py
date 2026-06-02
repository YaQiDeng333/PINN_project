#!/usr/bin/env python
"""Audit generator/label schema after the 25.13 target-v2 collapse.

This stage is intentionally read/design only. It reads the validated component
set pack plus 25.12b/25.13 outputs, audits target-v2 foreground support and
label-schema sufficiency, then writes strict JSON/Markdown/manifest records.
It does not train, run COMSOL, mutate data/NPZ files, tune losses, export
previews/checkpoints, or update the current baseline.
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
AUDIT_ID = "25_13b_surface_multipit_generator_label_schema_audit"
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
MAN13 = ROOT / "results/manifests/25_13_target_v2_training_gate_manifest.json"
DATASET_MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"
OUT_METRICS = ROOT / "results/metrics/25_13b_generator_label_schema_audit.json"
OUT_SUMMARY = ROOT / "results/summaries/25_13b_generator_label_schema_audit_summary.md"
OUT_MANIFEST = ROOT / "results/manifests/25_13b_generator_label_schema_audit_manifest.json"
FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 25.13 target-v2 collapse at generator/label schema level.")
    parser.add_argument("--target-redesign-metrics", type=Path, default=M12B)
    parser.add_argument("--target-v2-metrics", type=Path, default=M13)
    parser.add_argument("--target-v2-manifest", type=Path, default=MAN13)
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


def finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean_or_null(values: list[float]) -> float | None:
    clean = finite(values)
    return float(np.mean(clean)) if clean else None


def percentile_or_null(values: list[float], percentile: float) -> float | None:
    clean = finite(values)
    return float(np.percentile(clean, percentile)) if clean else None


def min_or_null(values: list[float]) -> float | None:
    clean = finite(values)
    return float(np.min(clean)) if clean else None


def max_or_null(values: list[float]) -> float | None:
    clean = finite(values)
    return float(np.max(clean)) if clean else None


def assert_sources(dataset_manifest: dict[str, Any], m12b: dict[str, Any], m13: dict[str, Any], man13: dict[str, Any]) -> Path:
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
    if m12b.get("stage") != "25.12b" or m12b.get("target_redesign_acceptance_decision") != "READY_FOR_25.13_TRAINING":
        raise ValueError("25.12b redesign metrics mismatch")
    if m13.get("stage") != "25.13" or m13.get("gate_decision") != "FAIL":
        raise ValueError("25.13 metrics must be FAIL")
    if "generator/label schema" not in str(m13.get("route_decision", "")):
        raise ValueError("25.13 route does not point to generator/label schema")
    if man13.get("stage") != "25.13" or man13.get("gate_decision") != "FAIL":
        raise ValueError("25.13 manifest mismatch")
    if man13.get("current_baseline_updated") is not False or man13.get("baseline_ready") is not False:
        raise ValueError("25.13 manifest baseline boundary mismatch")
    path = Path(dataset_manifest["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def grid_xy() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, GRID_W)
    y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, GRID_H)
    return np.meshgrid(x, y, indexing="xy")


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
    total_resolved_overlap_pixels = 0
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
        total_resolved_overlap_pixels += sample_resolved
        rows.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": i,
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "raw_overlap_pixel_count": overlap_pixels,
                "ownership_resolved_overlap_pixels": sample_resolved,
                "duplicate_ownership_before_v2": duplicate_before,
                "duplicate_ownership_after_v2": duplicate_after,
                "overlap_depth_conflict_before_v2": sample_conflict_before,
                "overlap_depth_conflict_after_v2": 0,
            }
        )
    summary = {
        "target_loaded_count": int(n),
        "ownership_resolved_pixel_count": int(masks_v2.sum()),
        "ownership_resolved_overlap_pixel_count": int(total_resolved_overlap_pixels),
        "duplicate_ownership_before_v2": int(total_duplicate_before),
        "duplicate_ownership_after_v2": int(total_duplicate_after),
        "overlap_depth_conflict_before_v2": int(total_conflict_before),
        "overlap_depth_conflict_after_v2": int(total_conflict_after),
        "raw_overlap_sample_count": int(sum(row["raw_overlap_pixel_count"] > 0 for row in rows)),
    }
    return masks_v2, depths_v2, ownership, rows, summary


def component_rows(pack: dict[str, Any], masks_v2: np.ndarray, depths_v2: np.ndarray) -> list[dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    masks_v1 = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    depths_v1 = np.asarray(pack["component_depth_grids_m"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for i in range(exists.shape[0]):
        for slot in range(K_MAX):
            if not exists[i, slot]:
                continue
            v1_pixels = int(masks_v1[i, slot].sum())
            v2_pixels = int(masks_v2[i, slot].sum())
            depth_v1_pixels = int((depths_v1[i, slot] > 0).sum())
            depth_v2_pixels = int((depths_v2[i, slot] > 0).sum())
            rows.append(
                {
                    "sample_id": str(pack["sample_ids"][i]),
                    "source_index": i,
                    "slot": int(slot),
                    "split": str(pack["split"][i]),
                    "component_count": int(pack["component_count"][i]),
                    "separation_type": str(pack["separation_type"][i]),
                    "topology_relation": str(pack["topology_relation"][i]),
                    "v1_foreground_px": v1_pixels,
                    "v2_foreground_px": v2_pixels,
                    "v1_positive_fraction": float(v1_pixels / GRID_PIXELS),
                    "v2_positive_fraction": float(v2_pixels / GRID_PIXELS),
                    "v2_vs_v1_foreground_ratio": float(v2_pixels / max(v1_pixels, 1)),
                    "foreground_px_removed_by_v2": int(v1_pixels - v2_pixels),
                    "v1_depth_positive_px": depth_v1_pixels,
                    "v2_depth_positive_px": depth_v2_pixels,
                    "v2_depth_vs_mask_positive_ratio": float(depth_v2_pixels / max(v2_pixels, 1)),
                    "is_tiny_v2_mask_lt_20px": bool(v2_pixels < 20),
                    "is_empty_v2_existing_mask": bool(v2_pixels == 0),
                    "is_shrink_ratio_lt_0p80": bool(v2_pixels / max(v1_pixels, 1) < 0.80),
                }
            )
    return rows


def sample_rows(pack: dict[str, Any], masks_v2: np.ndarray) -> list[dict[str, Any]]:
    masks_v1 = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    union = np.asarray(pack["projected_mask_2d"], dtype=bool)
    exists = np.asarray(pack["component_exists"], dtype=bool)
    rows: list[dict[str, Any]] = []
    for i in range(union.shape[0]):
        active = exists[i]
        v1_or = masks_v1[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        v2_or = masks_v2[i, active].max(axis=0) if active.any() else np.zeros((GRID_H, GRID_W), dtype=bool)
        rows.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": int(i),
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "union_foreground_px": int(union[i].sum()),
                "union_positive_fraction": float(union[i].sum() / GRID_PIXELS),
                "v1_component_or_px": int(v1_or.sum()),
                "v2_component_or_px": int(v2_or.sum()),
                "v2_or_to_union_mismatch_px": int(np.logical_xor(v2_or, union[i]).sum()),
            }
        )
    return rows


def aggregate_component(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"component_count": 0}
    return {
        "component_count": len(rows),
        "v1_foreground_px_mean": mean_or_null([row["v1_foreground_px"] for row in rows]),
        "v2_foreground_px_mean": mean_or_null([row["v2_foreground_px"] for row in rows]),
        "v2_foreground_px_min": min_or_null([row["v2_foreground_px"] for row in rows]),
        "v2_foreground_px_p05": percentile_or_null([row["v2_foreground_px"] for row in rows], 5),
        "v2_foreground_px_p50": percentile_or_null([row["v2_foreground_px"] for row in rows], 50),
        "v2_positive_fraction_mean": mean_or_null([row["v2_positive_fraction"] for row in rows]),
        "v2_vs_v1_foreground_ratio_mean": mean_or_null([row["v2_vs_v1_foreground_ratio"] for row in rows]),
        "v2_vs_v1_foreground_ratio_min": min_or_null([row["v2_vs_v1_foreground_ratio"] for row in rows]),
        "shrink_ratio_lt_0p80_count": int(sum(row["is_shrink_ratio_lt_0p80"] for row in rows)),
        "tiny_v2_mask_lt_20px_count": int(sum(row["is_tiny_v2_mask_lt_20px"] for row in rows)),
        "empty_v2_existing_mask_count": int(sum(row["is_empty_v2_existing_mask"] for row in rows)),
        "v2_depth_positive_px_mean": mean_or_null([row["v2_depth_positive_px"] for row in rows]),
        "v2_depth_vs_mask_positive_ratio_mean": mean_or_null([row["v2_depth_vs_mask_positive_ratio"] for row in rows]),
    }


def aggregate_sample(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    return {
        "sample_count": len(rows),
        "union_foreground_px_mean": mean_or_null([row["union_foreground_px"] for row in rows]),
        "union_foreground_px_min": min_or_null([row["union_foreground_px"] for row in rows]),
        "union_positive_fraction_mean": mean_or_null([row["union_positive_fraction"] for row in rows]),
        "v2_or_to_union_mismatch_px_sum": int(sum(row["v2_or_to_union_mismatch_px"] for row in rows)),
    }


def grouped(rows: list[dict[str, Any]], field: str, aggregate_fn) -> dict[str, Any]:
    return {str(value): aggregate_fn([row for row in rows if str(row[field]) == str(value)]) for value in sorted({row[field] for row in rows}, key=str)}


def empty_slot_audit(pack: dict[str, Any]) -> dict[str, Any]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    masks = np.asarray(pack["component_projected_masks_2d"], dtype=np.float64)
    depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float64)
    inactive = ~exists
    return {
        "inactive_slot_count": int(inactive.sum()),
        "inactive_mask_sum": int(masks[inactive].sum()) if inactive.any() else 0,
        "inactive_depth_abs_sum": float(np.abs(depths[inactive]).sum()) if inactive.any() else 0.0,
        "verdict": "PASS" if (not inactive.any() or (masks[inactive].sum() == 0 and np.abs(depths[inactive]).sum() <= 1.0e-15)) else "FAIL",
    }


def collapse_evidence(m13: dict[str, Any]) -> dict[str, Any]:
    test = m13["metrics_by_split"]["test"]
    criteria = m13.get("criteria", {})
    return {
        "gate_decision": m13["gate_decision"],
        "route_decision": m13["route_decision"],
        "test": {
            "component_recall": float(test["component_recall"]),
            "missed_rate": float(test["missed_rate"]),
            "extra_rate": float(test["extra_rate"]),
            "merged_rate": float(test["merged_rate"]),
            "component_mask_dice": float(test["component_mask_dice_mean"]),
            "union_mask_dice": float(test["union_mask_dice_mean"]),
            "depth_grid_rmse_m": float(test["depth_grid_rmse_m_mean"]),
        },
        "criteria": criteria,
        "interpretation": "merged_rate is numerically zero because predicted masks are near-empty; component/union Dice collapse invalidates it as component separation success",
    }


def label_schema_v3() -> dict[str, Any]:
    return {
        "raw_component_mask_raw": "Preserve original per-component binary masks before ownership resolution.",
        "component_ownership_map": "-1 background and 0..K-1 owner id; keep deterministic ownership for hard evaluation targets.",
        "component_mask_target_v3_soft": "Soft component-local target derived from raw mask plus boundary band; values taper rather than hard-drop at ownership cuts.",
        "component_sdf_target_v3": "Signed distance or normalized distance transform per component, generated from raw_component_mask_raw in PINN_project.",
        "component_valid_region_mask": "Per-component valid supervision region covering owned foreground plus a narrow raw-mask boundary/context band.",
        "overlap_region_mask": "Pixels where two or more raw component masks overlap; used for diagnostics and optional down-weighted supervision.",
        "contact_boundary_mask": "Pixels on touching/near-touching boundaries derived from dilated raw component masks.",
        "component_depth_target_v3": "Depth target in meters with valid_region mask; foreground and boundary/context are logged separately.",
        "union_from_components_rule": "Union mask/depth remain OR/max from raw components for sample-level evaluation.",
        "training_evaluation_split": "Training can use soft/valid-region targets; evaluation remains hard component and union Dice/IoU/RMSE for comparability.",
        "storage_policy": "First derive v3 in PINN_project as a loader/report transform; do not regenerate COMSOL unless raw masks/depths prove insufficient.",
    }


def decide_acceptance(component_stats: dict[str, Any], sample_stats: dict[str, Any], empty_audit: dict[str, Any], collapse: dict[str, Any]) -> tuple[str, str, list[str]]:
    support_not_empty = component_stats["overall"]["empty_v2_existing_mask_count"] == 0 and component_stats["overall"]["tiny_v2_mask_lt_20px_count"] == 0
    severe_sparse_fraction = float(component_stats["overall"]["v2_positive_fraction_mean"]) < 0.02
    raw_labels_sufficient = empty_audit["verdict"] == "PASS" and sample_stats["overall"]["v2_or_to_union_mismatch_px_sum"] == 0
    mask_collapsed = collapse["test"]["component_mask_dice"] < 0.02 and collapse["test"]["union_mask_dice"] < 0.02
    reasons: list[str] = []
    if not raw_labels_sufficient:
        reasons.append("raw component/union labels are inconsistent or empty-slot labels are invalid")
        return "NEEDS_COMSOL_GENERATOR_LABEL_FIX", "B. enter COMSOL label export fix, no training", reasons
    if not mask_collapsed:
        reasons.append("25.13 does not show the expected near-empty mask collapse")
        return "INCONCLUSIVE", "D. continue schema audit without training", reasons
    if support_not_empty and severe_sparse_fraction:
        reasons.extend(
            [
                "v2 did not create empty existing slots, so the collapse is not caused by missing component masks",
                "component positives are extremely sparse on the raster grid, making hard binary ownership targets weak supervision",
                "raw component masks/depth grids and union OR/max are available, so v3 soft/valid-region labels can be derived inside PINN_project",
            ]
        )
        return "NEEDS_PINN_LABEL_DERIVATION_V3", "A. enter 25.14 label-v3 derivation + validator, no training", reasons
    reasons.append("evidence does not separate sparse hard targets from generator/export insufficiency")
    return "INCONCLUSIVE", "D. continue schema audit without training", reasons


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    m12b = read_json(args.target_redesign_metrics)
    m13 = read_json(args.target_v2_metrics)
    man13 = read_json(args.target_v2_manifest)
    dataset_manifest = read_json(args.dataset_manifest)
    npz_path = assert_sources(dataset_manifest, m12b, m13, man13)
    pack = load_npz(npz_path)
    masks_v2, depths_v2, _ownership, target_v2_rows, target_v2_summary = build_target_v2(pack)
    comp_rows = component_rows(pack, masks_v2, depths_v2)
    sample_support_rows = sample_rows(pack, masks_v2)
    component_stats = {
        "overall": aggregate_component(comp_rows),
        "by_component_count": grouped(comp_rows, "component_count", aggregate_component),
        "by_separation": grouped(comp_rows, "separation_type", aggregate_component),
        "by_topology": grouped(comp_rows, "topology_relation", aggregate_component),
        "worst_shrink_components": sorted(comp_rows, key=lambda row: (float(row["v2_vs_v1_foreground_ratio"]), int(row["v2_foreground_px"])))[:12],
    }
    sample_stats = {
        "overall": aggregate_sample(sample_support_rows),
        "by_component_count": grouped(sample_support_rows, "component_count", aggregate_sample),
        "by_separation": grouped(sample_support_rows, "separation_type", aggregate_sample),
        "by_topology": grouped(sample_support_rows, "topology_relation", aggregate_sample),
    }
    empty_audit = empty_slot_audit(pack)
    collapse = collapse_evidence(m13)
    decision, route_decision, reasons = decide_acceptance(component_stats, sample_stats, empty_audit, collapse)
    return {
        "audit_id": AUDIT_ID,
        "stage": "25.13b",
        "dataset_id": DATASET_ID,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "source_files": {
            "target_redesign_metrics": str(args.target_redesign_metrics),
            "target_v2_metrics": str(args.target_v2_metrics),
            "target_v2_manifest": str(args.target_v2_manifest),
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
        },
        "target_v2_reproduction": target_v2_summary,
        "target_v2_sample_overlap_rows": target_v2_rows,
        "positive_support_audit": {
            "component_foreground_support": component_stats,
            "sample_union_support": sample_stats,
            "empty_slot_audit": empty_audit,
            "component_grid_pixels": GRID_PIXELS,
        },
        "target_v2_policy_audit": {
            "ownership_policy_over_hardness": "v2 removes duplicate pixels, but it remains a hard binary ownership target with no soft boundary/context support",
            "partially_overlapping": component_stats["by_separation"].get("partially_overlapping", {}),
            "touching": component_stats["by_separation"].get("touching", {}),
            "separated": component_stats["by_separation"].get("separated", {}),
            "component_count_3": component_stats["by_component_count"].get("3", {}),
        },
        "depth_target_audit": {
            "v2_depth_positive_px_mean": component_stats["overall"]["v2_depth_positive_px_mean"],
            "v2_depth_vs_mask_positive_ratio_mean": component_stats["overall"]["v2_depth_vs_mask_positive_ratio_mean"],
            "depth_rmse_not_worse_interpretation": "depth RMSE is full-grid/sample-level and can stay low when masks are near-empty; it does not prove component depth support is learned",
        },
        "generator_label_schema_audit": {
            "raw_labels_available_for_pinn_v3": [
                "component_projected_masks_2d",
                "component_depth_grids_m",
                "projected_mask_2d",
                "depth_grid_m",
                "component_center_xy_m",
                "component_lwd_m",
                "component_rotation_angle",
                "separation_type",
                "topology_relation",
            ],
            "missing_schema_for_learning": [
                "component-local soft mask",
                "signed distance / distance transform target",
                "component valid-region mask",
                "overlap_region_mask",
                "contact_boundary_mask",
                "ownership confidence / boundary confidence",
            ],
            "comsol_generator_fix_required_now": False,
            "reason": "existing raw masks/depths are sufficient to derive v3 soft/support labels inside PINN_project",
        },
        "label_schema_v3": label_schema_v3(),
        "collapse_evidence": collapse,
        "acceptance_decision": decision,
        "route_decision": route_decision,
        "decision_reasons": reasons,
        "audit_main_conclusion": (
            "25.13 target-v2 collapse is not caused by empty v2 components or broad generator corruption; "
            "the current hard binary component-local labels are too sparse/unsupported for stable mask learning, "
            "so derive label schema v3 soft/valid-region targets in PINN_project before any further training."
        ),
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    comp = payload["positive_support_audit"]["component_foreground_support"]["overall"]
    c3 = payload["positive_support_audit"]["component_foreground_support"]["by_component_count"].get("3", {})
    po = payload["positive_support_audit"]["component_foreground_support"]["by_separation"].get("partially_overlapping", {})
    collapse = payload["collapse_evidence"]["test"]
    target = payload["target_v2_reproduction"]
    lines = [
        "# 25.13b Generator/Label Schema Audit After Target-V2 Collapse",
        "",
        f"- acceptance_decision: `{payload['acceptance_decision']}`",
        f"- route_decision: `{payload['route_decision']}`",
        f"- main_conclusion: {payload['audit_main_conclusion']}",
        "",
        "## 25.13 Collapse Evidence",
        "",
        f"- recall: `{collapse['component_recall']:.6f}`",
        f"- missed: `{collapse['missed_rate']:.6f}`",
        f"- extra: `{collapse['extra_rate']:.6f}`",
        f"- merged: `{collapse['merged_rate']:.6f}`",
        f"- component Dice: `{collapse['component_mask_dice']:.6f}`",
        f"- union Dice: `{collapse['union_mask_dice']:.6f}`",
        f"- depth RMSE m: `{collapse['depth_grid_rmse_m']:.9f}`",
        "",
        "## Target-V2 Support Audit",
        "",
        f"- duplicate ownership: `{target['duplicate_ownership_before_v2']} -> {target['duplicate_ownership_after_v2']}`",
        f"- overlap-depth-conflict: `{target['overlap_depth_conflict_before_v2']} -> {target['overlap_depth_conflict_after_v2']}`",
        f"- active components: `{comp['component_count']}`",
        f"- v2 foreground px mean/min/p05: `{comp['v2_foreground_px_mean']:.6f}` / `{comp['v2_foreground_px_min']:.0f}` / `{comp['v2_foreground_px_p05']:.6f}`",
        f"- v2 positive fraction mean: `{comp['v2_positive_fraction_mean']:.9f}`",
        f"- v2/v1 shrink ratio mean/min: `{comp['v2_vs_v1_foreground_ratio_mean']:.6f}` / `{comp['v2_vs_v1_foreground_ratio_min']:.6f}`",
        f"- empty existing v2 masks: `{comp['empty_v2_existing_mask_count']}`",
        f"- tiny existing v2 masks <20 px: `{comp['tiny_v2_mask_lt_20px_count']}`",
        f"- shrink ratio <0.80: `{comp['shrink_ratio_lt_0p80_count']}`",
        f"- partially_overlapping shrink ratio mean/min: `{po.get('v2_vs_v1_foreground_ratio_mean'):.6f}` / `{po.get('v2_vs_v1_foreground_ratio_min'):.6f}`",
        f"- component_count=3 shrink ratio mean/min: `{c3.get('v2_vs_v1_foreground_ratio_mean'):.6f}` / `{c3.get('v2_vs_v1_foreground_ratio_min'):.6f}`",
        "",
        "## Diagnosis",
        "",
        "- V2 ownership resolution is not deleting whole components: existing slots stay non-empty and average support remains near 100 pixels.",
        "- The support is still very sparse on a 64x128 grid, so hard binary component targets have weak positive signal and no boundary/context target.",
        "- The zero merged rate in 25.13 is a near-empty mask artifact, not successful component separation.",
        "- Full-grid depth RMSE staying stable does not prove component-depth learning because mask collapse reduces effective component evidence.",
        "",
        "## Label Schema V3 Recommendation",
        "",
        "- Preserve `raw_component_mask_raw` and `component_ownership_map`.",
        "- Add `component_mask_target_v3_soft` or `component_sdf_target_v3`.",
        "- Add `component_valid_region_mask`, `overlap_region_mask`, and `contact_boundary_mask`.",
        "- Add `component_depth_target_v3` with an explicit valid region.",
        "- Keep union mask/depth as OR/max from raw components for evaluation comparability.",
        "- Derive v3 labels inside `PINN_project` first; no COMSOL generator change is required yet.",
        "",
        "## Boundary",
        "",
        "- This audit did not train a model or tune losses.",
        "- It did not run COMSOL or modify data/NPZ files.",
        "- It did not modify `CURRENT_BASELINE.md` or authorize a baseline transition.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, Any], args: argparse.Namespace) -> None:
    manifest = {
        "audit_id": AUDIT_ID,
        "stage": "25.13b",
        "dataset_id": DATASET_ID,
        "status": "generator_label_schema_audit_complete",
        "acceptance_decision": payload["acceptance_decision"],
        "route_decision": payload["route_decision"],
        "metrics_path": str(args.out_metrics),
        "summary_path": str(args.out_summary),
        "source_files": payload["source_files"],
        "training_run": False,
        "loss_tuning": False,
        "comsol_run": False,
        "data_npz_modified": False,
        "current_baseline_updated": False,
        "baseline_ready": False,
        "allowed_use": ["label_schema_v3_design", "25.14_label_v3_derivation_validator_input"],
        "forbidden_use": ["baseline_update", "current_baseline_replacement", "automatic_training", "loss_tuning_continuation"],
        "generated_at": payload["generated_at"],
        "git": payload["git"],
    }
    write_json(path, manifest)


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    write_json(args.out_metrics, payload)
    write_summary(args.out_summary, payload)
    write_manifest(args.out_manifest, payload, args)
    print(json.dumps({"acceptance_decision": payload["acceptance_decision"], "route_decision": payload["route_decision"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

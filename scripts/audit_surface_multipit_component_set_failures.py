#!/usr/bin/env python
"""Audit 25.10 surface multi-pit component-set training failures.

The audit consumes the 25.10 gate metrics and the explicit component-set
dataset manifest. It does not train, infer, export previews, or update the
current baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GATE_METRICS = ROOT / "results/metrics/25_10_component_set_training_gate_metrics.json"
GATE_MANIFEST = ROOT / "results/manifests/25_10_component_set_training_gate_manifest.json"
GATE_SUMMARY = ROOT / "results/summaries/25_10_component_set_training_gate_summary.md"
DATASET_MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"
OUT_METRICS = ROOT / "results/metrics/25_10b_component_set_failure_audit.json"
OUT_SUMMARY = ROOT / "results/summaries/25_10b_component_set_failure_audit_summary.md"
OUT_MANIFEST = ROOT / "results/manifests/25_10b_component_set_failure_audit_manifest.json"
TRAINING_SCRIPT = ROOT / "scripts/train_surface_multipit_component_set_gate.py"

DATASET_ID = "comsol_surface_multipit_component_set_pilot_v1"
AUDIT_ID = "25_10b_surface_multipit_component_set_failure_audit"
MASK_X_START_M = -0.04
MASK_X_STOP_M = 0.04
MASK_Y_START_M = -0.01
MASK_Y_STOP_M = 0.01
FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 25.10 component-set failures.")
    parser.add_argument("--gate-metrics", type=Path, default=GATE_METRICS)
    parser.add_argument("--gate-manifest", type=Path, default=GATE_MANIFEST)
    parser.add_argument("--gate-summary", type=Path, default=GATE_SUMMARY)
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET_MANIFEST)
    parser.add_argument("--out-metrics", type=Path, default=OUT_METRICS)
    parser.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    parser.add_argument("--out-manifest", type=Path, default=OUT_MANIFEST)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as pack:
        return {name: pack[name].copy() for name in pack.files}


def finite_mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def finite_max(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.max(clean)) if clean else None


def finite_corr(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(float(x)) and np.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(x.std()) <= 1.0e-12 or float(y.std()) <= 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def iou_dice(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    aa = a > 0.5
    bb = b > 0.5
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    denom = float(aa.sum() + bb.sum())
    return inter / (union + 1.0e-8), (2.0 * inter) / (denom + 1.0e-8)


def add_failure_flags(row: dict[str, Any], split_depth_p75: float | None) -> dict[str, Any]:
    out = dict(row)
    out["missed_failure"] = int(row["missed_components"]) > 0
    out["merged_failure"] = bool(row["merged_sample"])
    out["extra_failure"] = int(row["extra_components"]) > 0
    out["low_component_dice_failure"] = float(row["component_mask_dice_mean"]) < 0.20
    out["low_union_dice_failure"] = float(row["union_mask_dice"]) < 0.20
    out["high_depth_rmse_failure"] = split_depth_p75 is not None and float(row["depth_grid_rmse_m"]) >= split_depth_p75
    out["single_component_collapse_failure"] = bool(row["single_component_collapse_sample"])
    failures = [
        name
        for name in [
            "missed_failure",
            "merged_failure",
            "extra_failure",
            "low_component_dice_failure",
            "low_union_dice_failure",
            "high_depth_rmse_failure",
            "single_component_collapse_failure",
        ]
        if out[name]
    ]
    out["failure_tags"] = failures
    out["failure_count"] = len(failures)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    true_components = sum(int(row["true_component_count"]) for row in rows)
    pred_components = sum(int(row["pred_component_count"]) for row in rows)
    matched = sum(int(row["matched_components"]) for row in rows)
    missed = sum(int(row["missed_components"]) for row in rows)
    extra = sum(int(row["extra_components"]) for row in rows)
    return {
        "sample_count": len(rows),
        "true_components": true_components,
        "pred_components": pred_components,
        "matched_components": matched,
        "component_recall": matched / max(true_components, 1),
        "component_precision": matched / max(pred_components, 1) if pred_components else 0.0,
        "missed_rate": missed / max(true_components, 1),
        "merged_rate": sum(bool(row["merged_sample"]) for row in rows) / len(rows),
        "extra_rate": extra / max(pred_components, 1) if pred_components else 0.0,
        "low_component_dice_rate": sum(bool(row["low_component_dice_failure"]) for row in rows) / len(rows),
        "low_union_dice_rate": sum(bool(row["low_union_dice_failure"]) for row in rows) / len(rows),
        "high_depth_rmse_rate": sum(bool(row["high_depth_rmse_failure"]) for row in rows) / len(rows),
        "single_component_collapse_rate": sum(bool(row["single_component_collapse_sample"]) for row in rows) / len(rows),
        "pred_component_count_mean": finite_mean([row["pred_component_count"] for row in rows]),
        "center_error_m_mean": finite_mean([row["center_error_m_mean"] for row in rows]),
        "lwd_relative_error_mean": finite_mean([row["lwd_relative_error_mean"] for row in rows]),
        "rotation_error_rad_mean": finite_mean([row["rotation_error_rad_mean"] for row in rows]),
        "component_mask_dice_mean": finite_mean([row["component_mask_dice_mean"] for row in rows]),
        "union_mask_dice_mean": finite_mean([row["union_mask_dice"] for row in rows]),
        "depth_grid_rmse_m_mean": finite_mean([row["depth_grid_rmse_m"] for row in rows]),
    }


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(row[field]) for row in rows})
    return {value: aggregate([row for row in rows if str(row[field]) == value]) for value in values}


def split_depth_thresholds(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        values = [float(row["depth_grid_rmse_m"]) for row in rows if str(row["split"]) == split]
        out[split] = float(np.percentile(values, 75)) if values else None
    return out


def build_failure_rows(metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    base_rows = list(metrics["sample_metrics"])
    thresholds = split_depth_thresholds(base_rows)
    rows = [add_failure_flags(row, thresholds[str(row["split"])]) for row in base_rows]
    return rows, thresholds


def failure_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        out[split] = {
            "overall": aggregate(split_rows),
            "by_component_count": grouped(split_rows, "component_count"),
            "by_separation": grouped(split_rows, "separation_type"),
            "by_topology": grouped(split_rows, "topology_relation"),
            "worst_low_dice": sorted(
                split_rows,
                key=lambda row: (float(row["union_mask_dice"]), float(row["component_mask_dice_mean"])),
            )[:8],
            "worst_high_depth_rmse": sorted(split_rows, key=lambda row: float(row["depth_grid_rmse_m"]), reverse=True)[:8],
            "missed_samples": [row for row in split_rows if row["missed_failure"]],
            "merged_samples": [row for row in split_rows if row["merged_failure"]],
            "extra_samples": [row for row in split_rows if row["extra_failure"]],
        }
    return out


def three_component_audit(rows: list[dict[str, Any]], pack: dict[str, Any]) -> dict[str, Any]:
    all_three = [row for row in rows if int(row["component_count"]) == 3]
    by_split = {split: aggregate([row for row in all_three if row["split"] == split]) for split in ["train", "val", "test"]}
    test_rows = [row for row in all_three if row["split"] == "test"]
    test_source_indices = [int(row["source_index"]) for row in test_rows]
    true_context = []
    for idx in test_source_indices:
        true_context.append(
            {
                "sample_id": str(pack["sample_ids"][idx]),
                "separation_type": str(pack["separation_type"][idx]),
                "topology_relation": str(pack["topology_relation"][idx]),
                "orientation_type": str(pack["orientation_type"][idx]),
                "component_lwd_m": np.asarray(pack["component_lwd_m"][idx]).round(9).tolist(),
                "component_exists": np.asarray(pack["component_exists"][idx]).astype(bool).tolist(),
            }
        )
    pred_count_counter = Counter(int(row["pred_component_count"]) for row in test_rows)
    return {
        "total_three_component_samples": len(all_three),
        "split_counts": dict(Counter(str(row["split"]) for row in all_three)),
        "by_split": by_split,
        "test_rows": test_rows,
        "test_pred_component_count_counts": dict(pred_count_counter),
        "test_true_context": true_context,
        "interpretation": (
            "three-component evidence is underdetermined by sample count and shows predicted slot under-count/merge behavior"
            if len(test_rows) <= 3 and by_split["test"].get("merged_rate", 0.0) >= 0.75
            else "three-component failure is present but not solely sample-count limited"
        ),
    }


def training_script_static_checks() -> dict[str, Any]:
    text = TRAINING_SCRIPT.read_text(encoding="utf-8")
    checks = {
        "uses_all_slot_permutations": "PERMS = list(permutations(range(K_MAX)))" in text,
        "uses_min_over_permutations": "losses.min(dim=1)" in text,
        "param_loss_masked_by_exists": "param_raw * exists" in text,
        "shape_loss_masked_by_exists": "shape_flat * exists" in text,
        "mask_loss_masked_by_exists": "mask_bce + mask_dice" in text and "mask_loss = (" in text and "* exists" in text,
        "depth_loss_masked_by_exists": "depth_mse * exists" in text,
        "existence_bce_includes_empty_slots": "binary_cross_entropy_with_logits(pred[\"exist_logits\"], exists" in text,
        "checkpoint_saved_false": "\"checkpoint_saved\": False" in text,
        "current_baseline_updated_false": "\"current_baseline_updated\": False" in text,
    }
    verdict = all(checks.values())
    return {
        "verdict": verdict,
        "checks": checks,
        "interpretation": "slot permutation and empty-slot treatment look structurally correct" if verdict else "training script static checks found a possible matching/empty-slot issue",
    }


def target_integrity_audit(pack: dict[str, Any]) -> dict[str, Any]:
    n = int(pack["sample_ids"].shape[0])
    x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, int(pack["projected_mask_2d"].shape[-1]))
    y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, int(pack["projected_mask_2d"].shape[-2]))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    union_ious: list[float] = []
    union_dices: list[float] = []
    union_depth_rmses: list[float] = []
    centroid_errors: list[float] = []
    depth_max_errors: list[float] = []
    bbox_l_rel_errors: list[float] = []
    bbox_w_rel_errors: list[float] = []
    empty_slot_mask_sum = 0.0
    empty_slot_depth_sum = 0.0
    active_components = 0
    for i in range(n):
        exists = np.asarray(pack["component_exists"][i]).astype(bool)
        comp_masks = np.asarray(pack["component_projected_masks_2d"][i], dtype=np.float64)
        comp_depths = np.asarray(pack["component_depth_grids_m"][i], dtype=np.float64)
        active_components += int(exists.sum())
        empty_slot_mask_sum += float(comp_masks[~exists].sum())
        empty_slot_depth_sum += float(comp_depths[~exists].sum())
        union_mask_from_components = np.max(comp_masks[exists], axis=0) if exists.any() else np.zeros_like(pack["projected_mask_2d"][i], dtype=np.float64)
        union_depth_from_components = np.max(comp_depths[exists], axis=0) if exists.any() else np.zeros_like(pack["depth_grid_m"][i], dtype=np.float64)
        iou, dice = iou_dice(union_mask_from_components, pack["projected_mask_2d"][i])
        union_ious.append(iou)
        union_dices.append(dice)
        union_depth_rmses.append(float(np.sqrt(np.mean((union_depth_from_components - pack["depth_grid_m"][i]) ** 2))))
        for slot in np.where(exists)[0]:
            mask = comp_masks[slot] > 0.5
            if not mask.any():
                continue
            weights = mask.astype(np.float64)
            cx = float((xx * weights).sum() / weights.sum())
            cy = float((yy * weights).sum() / weights.sum())
            target_center = np.asarray(pack["component_center_xy_m"][i, slot], dtype=np.float64)
            centroid_errors.append(float(np.linalg.norm(np.asarray([cx, cy]) - target_center)))
            d_target = float(pack["component_lwd_m"][i, slot, 2])
            depth_max_errors.append(abs(float(comp_depths[slot].max()) - d_target))
            ys, xs = np.where(mask)
            bbox_l = float(x[xs.max()] - x[xs.min()]) if xs.size else 0.0
            bbox_w = float(y[ys.max()] - y[ys.min()]) if ys.size else 0.0
            l_target = max(float(pack["component_lwd_m"][i, slot, 0]), 1.0e-9)
            w_target = max(float(pack["component_lwd_m"][i, slot, 1]), 1.0e-9)
            bbox_l_rel_errors.append(abs(bbox_l - l_target) / l_target)
            bbox_w_rel_errors.append(abs(bbox_w - w_target) / w_target)
    union_iou_mean = finite_mean(union_ious)
    centroid_error_mean = finite_mean(centroid_errors)
    return {
        "active_component_count": active_components,
        "union_mask_iou_mean": union_iou_mean,
        "union_mask_iou_min": float(np.min(union_ious)),
        "union_mask_dice_mean": finite_mean(union_dices),
        "component_union_depth_rmse_mean": finite_mean(union_depth_rmses),
        "component_union_depth_rmse_max": finite_max(union_depth_rmses),
        "component_center_to_mask_centroid_error_m_mean": finite_mean(centroid_errors),
        "component_center_to_mask_centroid_error_m_max": finite_max(centroid_errors),
        "component_depth_max_vs_D_abs_error_m_mean": finite_mean(depth_max_errors),
        "component_depth_max_vs_D_abs_error_m_max": finite_max(depth_max_errors),
        "component_bbox_L_relative_error_mean": finite_mean(bbox_l_rel_errors),
        "component_bbox_W_relative_error_mean": finite_mean(bbox_w_rel_errors),
        "empty_slot_mask_sum": empty_slot_mask_sum,
        "empty_slot_depth_sum": empty_slot_depth_sum,
        "coordinate_bug_likely": bool(
            (union_iou_mean is not None and union_iou_mean < 0.90)
            or (centroid_error_mean is not None and centroid_error_mean > 0.003)
            or empty_slot_mask_sum > 0.0
            or empty_slot_depth_sum > 0.0
        ),
    }


def threshold_audit(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = metrics["selection"]["rows"]
    compact = [
        {
            "threshold": row["threshold"],
            "selection_score": row["selection_score"],
            "component_recall": row["component_recall"],
            "missed_rate": row["missed_rate"],
            "extra_rate": row["extra_rate"],
            "pred_component_count_mean": row["pred_component_count_mean"],
            "union_mask_dice_mean": row["union_mask_dice_mean"],
        }
        for row in rows
    ]
    selected = metrics["selection"]["selected_threshold"]
    low_threshold_selected = float(selected) <= 0.25
    return {
        "selected_threshold": selected,
        "validation_threshold_rows": compact,
        "interpretation": (
            "selected low threshold favors recall and reduces missed components, while extra components remain moderate"
            if low_threshold_selected
            else "selected threshold does not indicate a low-threshold recall bias"
        ),
    }


def rotation_audit(rows: list[dict[str, Any]], pack: dict[str, Any]) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["source_index"])
        exists = np.asarray(pack["component_exists"][idx]).astype(bool)
        lwd = np.asarray(pack["component_lwd_m"][idx, exists], dtype=np.float64)
        aspect = np.maximum(lwd[:, 0] / np.maximum(lwd[:, 1], 1.0e-9), lwd[:, 1] / np.maximum(lwd[:, 0], 1.0e-9))
        enriched.append({**row, "near_circular_mean": bool(float(aspect.mean()) <= 1.35), "mean_component_aspect_ratio": float(aspect.mean())})
    test_rows = [row for row in enriched if row["split"] == "test"]
    near = [row for row in test_rows if row["near_circular_mean"]]
    non_near = [row for row in test_rows if not row["near_circular_mean"]]
    by_sep = {
        sep: {
            "sample_count": len(subset),
            "rotation_error_rad_mean": finite_mean([row["rotation_error_rad_mean"] for row in subset]),
            "mean_component_aspect_ratio": finite_mean([row["mean_component_aspect_ratio"] for row in subset]),
        }
        for sep, subset in ((sep, [row for row in test_rows if row["separation_type"] == sep]) for sep in sorted({row["separation_type"] for row in test_rows}))
    }
    return {
        "test_near_circular": {
            "sample_count": len(near),
            "rotation_error_rad_mean": finite_mean([row["rotation_error_rad_mean"] for row in near]),
            "mean_component_aspect_ratio": finite_mean([row["mean_component_aspect_ratio"] for row in near]),
        },
        "test_non_near_circular": {
            "sample_count": len(non_near),
            "rotation_error_rad_mean": finite_mean([row["rotation_error_rad_mean"] for row in non_near]),
            "mean_component_aspect_ratio": finite_mean([row["mean_component_aspect_ratio"] for row in non_near]),
        },
        "test_by_separation": by_sep,
        "rotation_error_vs_aspect_corr": finite_corr(
            [row["rotation_error_rad_mean"] for row in test_rows],
            [row["mean_component_aspect_ratio"] for row in test_rows],
        ),
        "interpretation": "rotation error should be audited with aspect/shape confidence; near-circular and topology cases can make angle weakly identifiable",
    }


def taxonomy_decision(
    summaries: dict[str, Any],
    three: dict[str, Any],
    static: dict[str, Any],
    target_integrity: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    test = summaries["test"]["overall"]
    val = summaries["val"]["overall"]
    empty = metrics["degenerate_baselines"]["empty"]["test"]
    one_slot = metrics["degenerate_baselines"]["one_slot_prior"]["test"]
    ranking: list[dict[str, Any]] = []
    loss_imbalance_score = 0
    if test["component_recall"] > one_slot["component_recall"] + 0.50:
        loss_imbalance_score += 2
    if test["component_mask_dice_mean"] < 0.20 and test["union_mask_dice_mean"] < 0.20:
        loss_imbalance_score += 3
    if test["depth_grid_rmse_m_mean"] >= 0.98 * one_slot["depth_grid_rmse_m_mean"]:
        loss_imbalance_score += 2
    ranking.append(
        {
            "failure_type": "loss imbalance",
            "rank_score": loss_imbalance_score,
            "evidence": "existence/coarse geometry learned, but component/union Dice remain <0.20 and depth RMSE barely beats degenerate baselines",
        }
    )
    three_score = 0
    if three["split_counts"].get("test", 0) <= 3:
        three_score += 2
    if three["by_split"]["test"].get("merged_rate", 0.0) >= 0.75:
        three_score += 3
    ranking.append(
        {
            "failure_type": "data scarcity failure",
            "rank_score": three_score,
            "evidence": f"three-component rows are {three['split_counts']} and test merged_rate={three['by_split']['test'].get('merged_rate')}",
        }
    )
    hard_topology_score = 0
    touching = summaries["test"]["by_topology"].get("touching_boundary", {})
    overlap = summaries["test"]["by_topology"].get("partially_overlapping", {})
    if touching.get("merged_rate", 0.0) >= 0.50:
        hard_topology_score += 2
    if overlap.get("missed_rate", 0.0) >= 0.20:
        hard_topology_score += 1
    ranking.append(
        {
            "failure_type": "genuine hard topology failure",
            "rank_score": hard_topology_score,
            "evidence": f"touching_boundary merged_rate={touching.get('merged_rate')}; partially_overlapping missed_rate={overlap.get('missed_rate')}",
        }
    )
    matching_score = 0
    if not static["verdict"]:
        matching_score += 3
    if test["component_recall"] >= 0.80 and test["single_component_collapse_rate"] == 0.0:
        matching_score -= 1
    ranking.append(
        {
            "failure_type": "matching/statistics failure",
            "rank_score": matching_score,
            "evidence": static["interpretation"],
        }
    )
    raster_bug_score = 0
    if target_integrity["coordinate_bug_likely"]:
        raster_bug_score += 4
    if target_integrity["union_mask_iou_mean"] >= 0.98 and target_integrity["empty_slot_mask_sum"] == 0.0:
        raster_bug_score -= 2
    ranking.append(
        {
            "failure_type": "raster/coordinate bug",
            "rank_score": raster_bug_score,
            "evidence": f"target union IoU mean={target_integrity['union_mask_iou_mean']:.6f}, centroid error mean={target_integrity['component_center_to_mask_centroid_error_m_mean']:.6f}, empty_slot_mask_sum={target_integrity['empty_slot_mask_sum']}",
        }
    )
    representation_score = 0
    if test["component_recall"] < 0.30:
        representation_score += 3
    if test["component_recall"] >= 0.80:
        representation_score -= 2
    ranking.append(
        {
            "failure_type": "representation limit",
            "rank_score": representation_score,
            "evidence": "component-set representation has clear learning signal, so representation is not the primary blocker yet",
        }
    )
    ranking = sorted(ranking, key=lambda item: item["rank_score"], reverse=True)
    top = ranking[0]["failure_type"]
    if top == "raster/coordinate bug":
        route = "A. run 25.10c raster/statistics bugfix + re-evaluate before new training"
    elif top == "loss imbalance":
        route = "B. enter 25.11 mask/depth loss rebalance training"
    elif top == "data scarcity failure":
        route = "C. enter 25.11 data-aware split/top-up or three-component focused training gate"
    elif top == "genuine hard topology failure":
        route = "D. enter 25.11 component interaction / union-consistency redesign"
    else:
        route = "E. return to representation/loss design before expanding training"
    conclusion = (
        "Primary failure is loss imbalance: component existence and coarse geometry learned, but raster/depth supervision did not translate into aligned masks."
        if route.startswith("B.")
        else f"Primary failure is {top}."
    )
    return ranking, route, conclusion


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    test = payload["failure_summaries"]["test"]["overall"]
    three = payload["three_component_audit"]
    target = payload["target_integrity_audit"]
    lines = [
        "# 25.10b Component-Set Failure Audit",
        "",
        f"- audit_conclusion: {payload['audit_conclusion']}",
        f"- next_route: `{payload['route_decision']}`",
        f"- primary_failure: `{payload['failure_taxonomy_ranked'][0]['failure_type']}`",
        "",
        "## Test Failure Snapshot",
        "",
        f"- component_recall: `{test['component_recall']:.6f}`",
        f"- missed_rate: `{test['missed_rate']:.6f}`",
        f"- merged_rate: `{test['merged_rate']:.6f}`",
        f"- extra_rate: `{test['extra_rate']:.6f}`",
        f"- component_mask_dice: `{test['component_mask_dice_mean']:.6f}`",
        f"- union_mask_dice: `{test['union_mask_dice_mean']:.6f}`",
        f"- depth_grid_RMSE_m: `{test['depth_grid_rmse_m_mean']:.9f}`",
        "",
        "## Three-Component Finding",
        "",
        f"- split_counts: `{three['split_counts']}`",
        f"- test_pred_component_count_counts: `{three['test_pred_component_count_counts']}`",
        f"- test_merged_rate: `{three['by_split']['test'].get('merged_rate')}`",
        "",
        "## Raster Target Integrity",
        "",
        f"- target_union_mask_iou_mean: `{target['union_mask_iou_mean']:.6f}`",
        f"- center_to_mask_centroid_error_mean_m: `{target['component_center_to_mask_centroid_error_m_mean']:.9f}`",
        f"- empty_slot_mask_sum: `{target['empty_slot_mask_sum']}`",
        f"- coordinate_bug_likely: `{target['coordinate_bug_likely']}`",
        "",
        "## Boundary",
        "",
        "- This audit did not train a new model.",
        "- It did not modify `CURRENT_BASELINE.md`.",
        "- It does not authorize a baseline transition.",
        "- It does not recommend simply increasing model size without addressing the audited failure mode.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    metrics = read_json(args.gate_metrics)
    gate_manifest = read_json(args.gate_manifest)
    dataset_manifest = read_json(args.dataset_manifest)
    if metrics.get("gate_decision") != "PARTIAL" or gate_manifest.get("gate_decision") != "PARTIAL":
        raise ValueError("25.10b expects the 25.10 gate decision to be PARTIAL")
    if dataset_manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(f"dataset mismatch: {dataset_manifest.get('dataset_id')}")
    pack = load_npz(Path(dataset_manifest["path"]))
    rows, depth_thresholds = build_failure_rows(metrics)
    summaries = failure_summaries(rows)
    three = three_component_audit(rows, pack)
    static = training_script_static_checks()
    target_integrity = target_integrity_audit(pack)
    threshold = threshold_audit(metrics)
    rotation = rotation_audit(rows, pack)
    taxonomy, route, conclusion = taxonomy_decision(summaries, three, static, target_integrity, metrics)
    payload = {
        "audit_id": AUDIT_ID,
        "stage": "25.10b",
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "dataset_id": DATASET_ID,
        "source_gate_commit": "2d2245e56eee52a8be215e70efdaa60423923001",
        "source_paths": {
            "gate_metrics": str(args.gate_metrics),
            "gate_manifest": str(args.gate_manifest),
            "gate_summary": str(args.gate_summary),
            "dataset_manifest": str(args.dataset_manifest),
        },
        "audit_conclusion": conclusion,
        "route_decision": route,
        "failure_taxonomy_ranked": taxonomy,
        "failure_depth_rmse_p75_thresholds": depth_thresholds,
        "failure_summaries": summaries,
        "three_component_audit": three,
        "matching_and_empty_slot_audit": static,
        "target_integrity_audit": target_integrity,
        "threshold_audit": threshold,
        "rotation_audit": rotation,
        "boundary": {
            "new_training_run": False,
            "current_baseline_updated": False,
            "baseline_transition": False,
            "comsol_modified": False,
            "checkpoint_or_preview_created": False,
        },
        "git": {
            "branch": git_value(["branch", "--show-current"]),
            "head": git_value(["rev-parse", "HEAD"]),
            "protected_path_diff": git_value(["diff", "--name-only", "--", *FORBIDDEN_DIFF_PATHS]),
        },
    }
    write_json(args.out_metrics, payload)
    audit_manifest = {
        "audit_id": AUDIT_ID,
        "stage": "25.10b",
        "dataset_id": DATASET_ID,
        "source_gate_metrics": str(args.gate_metrics),
        "metrics_path": str(args.out_metrics),
        "summary_path": str(args.out_summary),
        "gate_decision": "failure_audit_complete",
        "audit_conclusion": conclusion,
        "route_decision": route,
        "new_training_run": False,
        "baseline_ready": False,
        "current_baseline_updated": False,
        "allowed_use": ["failure_audit_input", "25_11_route_selection"],
        "forbidden_use": ["baseline_update", "current_baseline_replacement", "automatic_training_expansion"],
    }
    write_json(args.out_manifest, audit_manifest)
    write_summary(args.out_summary, payload)
    print(json.dumps({"audit_conclusion": conclusion, "route_decision": route, "metrics": str(args.out_metrics)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

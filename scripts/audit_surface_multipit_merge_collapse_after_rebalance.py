#!/usr/bin/env python
"""Audit merge collapse after the 25.11 mask/depth rebalance gate.

This script only reads existing metrics/manifests and writes audit records.
It does not train, run COMSOL, export previews, or update the current baseline.
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
AUDIT_ID = "25_11b_surface_multipit_merge_collapse_audit"
M10 = ROOT / "results/metrics/25_10_component_set_training_gate_metrics.json"
M10B = ROOT / "results/metrics/25_10b_component_set_failure_audit.json"
M11 = ROOT / "results/metrics/25_11_mask_depth_loss_rebalance_training_metrics.json"
MAN10 = ROOT / "results/manifests/25_10_component_set_training_gate_manifest.json"
MAN10B = ROOT / "results/manifests/25_10b_component_set_failure_audit_manifest.json"
MAN11 = ROOT / "results/manifests/25_11_mask_depth_loss_rebalance_training_manifest.json"
OUT_METRICS = ROOT / "results/metrics/25_11b_component_set_merge_collapse_audit.json"
OUT_SUMMARY = ROOT / "results/summaries/25_11b_component_set_merge_collapse_audit_summary.md"
OUT_MANIFEST = ROOT / "results/manifests/25_11b_component_set_merge_collapse_audit_manifest.json"
FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 25.11 merge collapse after loss rebalance.")
    parser.add_argument("--metrics-25-10", type=Path, default=M10)
    parser.add_argument("--audit-25-10b", type=Path, default=M10B)
    parser.add_argument("--metrics-25-11", type=Path, default=M11)
    parser.add_argument("--manifest-25-10", type=Path, default=MAN10)
    parser.add_argument("--manifest-25-10b", type=Path, default=MAN10B)
    parser.add_argument("--manifest-25-11", type=Path, default=MAN11)
    parser.add_argument("--out-metrics", type=Path, default=OUT_METRICS)
    parser.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    parser.add_argument("--out-manifest", type=Path, default=OUT_MANIFEST)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            values.append(value_f)
    return values


def mean_or_null(values: list[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    true_components = sum(int(row["true_component_count_25_11"]) for row in rows)
    pred_components = sum(int(row["pred_component_count_25_11"]) for row in rows)
    matched = sum(int(row["matched_components_25_11"]) for row in rows)
    return {
        "sample_count": len(rows),
        "true_components": true_components,
        "pred_components": pred_components,
        "matched_components": matched,
        "component_recall_25_11": matched / max(true_components, 1),
        "merged_rate_25_10": sum(bool(row["merged_25_10"]) for row in rows) / len(rows),
        "merged_rate_25_11": sum(bool(row["merged_25_11"]) for row in rows) / len(rows),
        "newly_merged_rate": sum(bool(row["newly_merged"]) for row in rows) / len(rows),
        "union_over_component_collapse_rate": sum(bool(row["union_over_component_collapse"]) for row in rows) / len(rows),
        "component_count_ambiguity_rate": sum(bool(row["component_count_ambiguity"]) for row in rows) / len(rows),
        "depth_supervision_dilution_rate": sum(bool(row["depth_supervision_dilution"]) for row in rows) / len(rows),
        "union_mask_dice_delta_mean": mean_or_null(finite_values(rows, "union_mask_dice_delta")),
        "component_mask_dice_delta_mean": mean_or_null(finite_values(rows, "component_mask_dice_delta")),
        "depth_rmse_delta_m_mean": mean_or_null(finite_values(rows, "depth_grid_rmse_delta_m")),
        "pred_component_count_delta_mean": mean_or_null(finite_values(rows, "pred_component_count_delta")),
        "component_recall_delta_mean": mean_or_null(finite_values(rows, "component_recall_delta")),
    }


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {str(value): aggregate_rows([row for row in rows if str(row[field]) == str(value)]) for value in sorted({row[field] for row in rows}, key=str)}


def assert_sources(
    m10: dict[str, Any],
    m10b: dict[str, Any],
    m11: dict[str, Any],
    man10: dict[str, Any],
    man10b: dict[str, Any],
    man11: dict[str, Any],
) -> None:
    if ROOT != Path(r"C:\Users\19166\Desktop\PINN_project"):
        raise RuntimeError(f"wrong project root: {ROOT}")
    manifests = [man10, man10b, man11]
    if m10.get("dataset_id") != DATASET_ID or m11.get("dataset_id") != DATASET_ID or any(item.get("dataset_id") != DATASET_ID for item in manifests):
        raise ValueError("dataset_id mismatch")
    if m10.get("gate_decision") != "PARTIAL":
        raise ValueError("25.10 gate must be PARTIAL")
    if man10.get("stage") != "25.10" or man10.get("gate_decision") != "PARTIAL":
        raise ValueError("25.10 manifest mismatch")
    if m10b.get("route_decision") != "B. enter 25.11 mask/depth loss rebalance training":
        raise ValueError("25.10b route does not point to 25.11")
    if man10b.get("stage") != "25.10b" or man10b.get("route_decision") != "B. enter 25.11 mask/depth loss rebalance training":
        raise ValueError("25.10b manifest mismatch")
    if m11.get("stage") != "25.11" or m11.get("gate_decision") != "PARTIAL":
        raise ValueError("25.11 metrics must be PARTIAL")
    if man11.get("current_baseline_updated") is not False or man11.get("baseline_ready") is not False:
        raise ValueError("25.11 manifest baseline boundary mismatch")
    if man11.get("checkpoint_saved") is not False or man11.get("inference_artifact_exported") is not False:
        raise ValueError("25.11 manifest artifact boundary mismatch")
    for name, manifest in [("25.10", man10), ("25.10b", man10b), ("25.11", man11)]:
        if manifest.get("baseline_ready") is not False or manifest.get("current_baseline_updated") is not False:
            raise ValueError(f"{name} manifest baseline boundary mismatch")


def compare_samples(m10: dict[str, Any], m11: dict[str, Any]) -> list[dict[str, Any]]:
    old = {str(row["sample_id"]): row for row in m10["sample_metrics"]}
    new = {str(row["sample_id"]): row for row in m11["sample_metrics"]}
    if set(old) != set(new):
        raise ValueError("25.10 and 25.11 sample_id sets differ")
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(old):
        r10 = old[sample_id]
        r11 = new[sample_id]
        union_delta = float(r11["union_mask_dice"]) - float(r10["union_mask_dice"])
        comp_delta = float(r11["component_mask_dice_mean"]) - float(r10["component_mask_dice_mean"])
        depth_delta = float(r11["depth_grid_rmse_m"]) - float(r10["depth_grid_rmse_m"])
        recall_delta = float(r11["component_recall"]) - float(r10["component_recall"])
        pred_delta = int(r11["pred_component_count"]) - int(r10["pred_component_count"])
        merged10 = bool(r10["merged_sample"])
        merged11 = bool(r11["merged_sample"])
        newly_merged = (not merged10) and merged11
        row = {
            "sample_id": sample_id,
            "source_index": int(r11["source_index"]),
            "split": str(r11["split"]),
            "component_count": int(r11["component_count"]),
            "separation_type": str(r11["separation_type"]),
            "topology_relation": str(r11["topology_relation"]),
            "orientation_type": str(r11["orientation_type"]),
            "true_component_count_25_11": int(r11["true_component_count"]),
            "pred_component_count_25_10": int(r10["pred_component_count"]),
            "pred_component_count_25_11": int(r11["pred_component_count"]),
            "pred_component_count_delta": int(pred_delta),
            "matched_components_25_10": int(r10["matched_components"]),
            "matched_components_25_11": int(r11["matched_components"]),
            "component_recall_25_10": float(r10["component_recall"]),
            "component_recall_25_11": float(r11["component_recall"]),
            "component_recall_delta": recall_delta,
            "merged_25_10": merged10,
            "merged_25_11": merged11,
            "newly_merged": newly_merged,
            "missed_components_25_10": int(r10["missed_components"]),
            "missed_components_25_11": int(r11["missed_components"]),
            "extra_components_25_10": int(r10["extra_components"]),
            "extra_components_25_11": int(r11["extra_components"]),
            "component_mask_dice_25_10": float(r10["component_mask_dice_mean"]),
            "component_mask_dice_25_11": float(r11["component_mask_dice_mean"]),
            "component_mask_dice_delta": comp_delta,
            "union_mask_dice_25_10": float(r10["union_mask_dice"]),
            "union_mask_dice_25_11": float(r11["union_mask_dice"]),
            "union_mask_dice_delta": union_delta,
            "depth_grid_rmse_25_10_m": float(r10["depth_grid_rmse_m"]),
            "depth_grid_rmse_25_11_m": float(r11["depth_grid_rmse_m"]),
            "depth_grid_rmse_delta_m": depth_delta,
            "union_over_component_collapse": merged11 and union_delta > 0.0 and (union_delta - comp_delta) >= 0.01,
            "component_count_ambiguity": recall_delta >= 0.0 and (newly_merged or (merged11 and pred_delta <= 0)),
            "depth_supervision_dilution": depth_delta > 0.00010 and union_delta > 0.02,
        }
        tags = []
        for key in ["newly_merged", "union_over_component_collapse", "component_count_ambiguity", "depth_supervision_dilution"]:
            if row[key]:
                tags.append(key)
        row["audit_tags"] = tags
        rows.append(row)
    return rows


def loss_term_audit(m11: dict[str, Any]) -> dict[str, Any]:
    history = m11["training"]["history"]
    weights = m11["model"]["loss_weights"]
    epochs = {1, int(m11["training"]["best_epoch"]), len(history)}
    snapshots = {str(row["epoch"]): row for row in history if int(row["epoch"]) in epochs}

    def ratios(row: dict[str, Any], prefix: str) -> dict[str, Any]:
        weighted = row[f"{prefix}_weighted_terms"]
        unweighted = row[f"{prefix}_unweighted_terms"]
        component_mask = float(weighted["component_mask"])
        union_mask = float(weighted["union_mask"])
        component_depth = float(weighted["component_depth"])
        union_depth = float(weighted["union_depth"])
        component_total = component_mask + component_depth
        union_total = union_mask + union_depth
        mask_depth_total = component_total + union_total
        total = float(sum(float(v) for v in weighted.values()))
        return {
            "loss": float(row[f"{prefix}_loss"]) if f"{prefix}_loss" in row else float(row["train_loss" if prefix == "train" else "val_loss"]),
            "unweighted_terms": unweighted,
            "weighted_terms": weighted,
            "mask_depth_weighted_ratio": float(row[f"{prefix}_mask_depth_weighted_ratio"]),
            "component_mask_to_param_weighted_ratio": component_mask / max(float(weighted["param"]), 1.0e-12),
            "union_mask_to_component_mask_weighted_ratio": union_mask / max(component_mask, 1.0e-12),
            "union_depth_to_component_depth_weighted_ratio": union_depth / max(component_depth, 1.0e-12),
            "union_to_component_mask_depth_weighted_ratio": union_total / max(component_total, 1.0e-12),
            "union_fraction_of_mask_depth_weighted": union_total / max(mask_depth_total, 1.0e-12),
            "total_weighted_sum": total,
        }

    final = history[-1]
    best = history[int(m11["training"]["best_epoch"]) - 1]
    last20 = history[-20:]
    last20_train_ratios = [float(row["train_mask_depth_weighted_ratio"]) for row in last20]
    last20_val_ratios = [float(row["val_mask_depth_weighted_ratio"]) for row in last20]
    return {
        "loss_config": m11["model"]["loss_config"],
        "loss_weights": weights,
        "mask_supervision": m11["model"]["mask_supervision"],
        "depth_supervision": m11["model"]["depth_supervision"],
        "snapshots": {
            epoch: {
                "train": ratios(row, "train"),
                "val": ratios(row, "val"),
            }
            for epoch, row in snapshots.items()
        },
        "best_epoch": {
            "epoch": int(best["epoch"]),
            "train": ratios(best, "train"),
            "val": ratios(best, "val"),
        },
        "final_epoch": {
            "epoch": int(final["epoch"]),
            "train": ratios(final, "train"),
            "val": ratios(final, "val"),
        },
        "last20": {
            "train_mask_depth_weighted_ratio_mean": mean_or_null(last20_train_ratios),
            "val_mask_depth_weighted_ratio_mean": mean_or_null(last20_val_ratios),
            "train_mask_depth_weighted_ratio_max": float(max(last20_train_ratios)),
            "val_mask_depth_weighted_ratio_max": float(max(last20_val_ratios)),
        },
        "interpretation": (
            "mask/depth terms dominate the objective; final train ratio is "
            f"{float(final['train_mask_depth_weighted_ratio']):.6f} and final val ratio is "
            f"{float(final['val_mask_depth_weighted_ratio']):.6f}"
        ),
    }


def threshold_audit(m10: dict[str, Any], m11: dict[str, Any]) -> dict[str, Any]:
    def rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return payload.get("selection", {}).get("rows", [])

    return {
        "selected_threshold_25_10": m10.get("selection", {}).get("selected_threshold"),
        "selected_threshold_25_11": m11.get("selection", {}).get("selected_threshold"),
        "validation_rows_25_10": rows(m10),
        "validation_rows_25_11": rows(m11),
        "interpretation": (
            "25.11 threshold moved from 0.25 to 0.35, but validation merged_rate remains high across candidate rows; "
            "the merge collapse is not just a selected-threshold artifact."
        ),
    }


def grouped_merge_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        out[split] = {
            "overall": aggregate_rows(split_rows),
            "by_component_count": grouped(split_rows, "component_count"),
            "by_separation": grouped(split_rows, "separation_type"),
            "by_topology": grouped(split_rows, "topology_relation"),
            "newly_merged_samples": [row for row in split_rows if row["newly_merged"]],
            "union_over_component_collapse_samples": [row for row in split_rows if row["union_over_component_collapse"]],
            "depth_supervision_dilution_samples": [row for row in split_rows if row["depth_supervision_dilution"]],
        }
    return out


def design_rebalance(loss_audit: dict[str, Any], grouped_audit: dict[str, Any]) -> dict[str, Any]:
    test = grouped_audit["test"]["overall"]
    separated = grouped_audit["test"]["by_separation"].get("separated", {"newly_merged_rate": 0.0, "merged_rate_25_11": 0.0})
    topology_touch = grouped_audit["test"]["by_topology"].get("touching_boundary", {"merged_rate_25_11": 0.0})
    topology_overlap = grouped_audit["test"]["by_topology"].get("partially_overlapping", {"merged_rate_25_11": 0.0})
    final_train = loss_audit["final_epoch"]["train"]
    final_val = loss_audit["final_epoch"]["val"]
    evidence = {
        "test_newly_merged_rate": test["newly_merged_rate"],
        "test_union_over_component_collapse_rate": test["union_over_component_collapse_rate"],
        "test_depth_supervision_dilution_rate": test["depth_supervision_dilution_rate"],
        "separated_newly_merged_rate": separated["newly_merged_rate"],
        "separated_merged_rate_25_11": separated["merged_rate_25_11"],
        "touching_boundary_merged_rate_25_11": topology_touch["merged_rate_25_11"],
        "partially_overlapping_merged_rate_25_11": topology_overlap["merged_rate_25_11"],
        "final_train_mask_depth_weighted_ratio": final_train["mask_depth_weighted_ratio"],
        "final_val_mask_depth_weighted_ratio": final_val["mask_depth_weighted_ratio"],
        "final_val_union_fraction_of_mask_depth_weighted": final_val["union_fraction_of_mask_depth_weighted"],
    }
    design = [
        {
            "item": "lower_or_schedule_union_mask_loss",
            "action": "start with component-mask-only warmup, then introduce union mask with a capped weight after component Dice improves",
            "reason": "union Dice improved while merged rate and component Dice did not; union agreement can reward merged blobs",
        },
        {
            "item": "add_component_separation_regularizer",
            "action": "penalize predicted component-mask overlap for separated/close samples and require slot-specific local masks after Hungarian matching",
            "reason": "newly merged samples are not confined to touching/overlap, so component separation needs an explicit term",
        },
        {
            "item": "add_topology_aware_merge_penalty",
            "action": "report and weight touching/overlap separately; apply pairwise center/separation consistency and merge penalty by topology label",
            "reason": "touching_boundary and partially_overlapping remain high-risk and three-component rows still merge",
        },
        {
            "item": "redesign_depth_loss",
            "action": "stage depth after masks stabilize, normalize per component foreground, and compute overlap/touching depth with conflict-aware masks",
            "reason": "depth RMSE worsened despite union Dice improvement, matching depth-supervision dilution",
        },
        {
            "item": "keep_threshold_audit_slice",
            "action": "evaluate merge/missed/extra across thresholds, not only selected threshold, before claiming component-count gains",
            "reason": "existence threshold improves missed/extra but does not explain the persistent high merged rate",
        },
    ]
    return {
        "evidence": evidence,
        "targeted_rebalance_design": design,
    }


def decide_route(grouped_audit: dict[str, Any], loss_audit: dict[str, Any]) -> dict[str, Any]:
    test = grouped_audit["test"]["overall"]
    separated = grouped_audit["test"]["by_separation"].get("separated", {"newly_merged_rate": 0.0})
    depth_dilution = float(test["depth_supervision_dilution_rate"])
    union_collapse = float(test["union_over_component_collapse_rate"])
    newly_merged = float(test["newly_merged_rate"])
    separated_new_merge = float(separated["newly_merged_rate"])
    val_ratio = float(loss_audit["final_epoch"]["val"]["mask_depth_weighted_ratio"])

    scores = {
        "union-over-component collapse": 0,
        "topology/touching dominated collapse": 0,
        "depth target/loss region design": 0,
        "evaluation/threshold bug": 0,
    }
    evidence: dict[str, str] = {}

    if union_collapse >= 0.40 or newly_merged >= 0.40:
        scores["union-over-component collapse"] += 4
        evidence["union-over-component collapse"] = f"test newly_merged_rate={newly_merged:.3f}, union_over_component_collapse_rate={union_collapse:.3f}"
    if separated_new_merge >= 0.40:
        scores["union-over-component collapse"] += 2
        evidence["union-over-component collapse"] += f"; separated_newly_merged_rate={separated_new_merge:.3f}"
        scores["topology/touching dominated collapse"] -= 1
    if val_ratio >= 0.85:
        scores["union-over-component collapse"] += 1
        evidence["union-over-component collapse"] += f"; final_val_mask_depth_weighted_ratio={val_ratio:.3f}"
    if depth_dilution >= 0.40:
        scores["depth target/loss region design"] += 2
        evidence["depth target/loss region design"] = f"depth_supervision_dilution_rate={depth_dilution:.3f}"

    touch = grouped_audit["test"]["by_topology"].get("touching_boundary", {"merged_rate_25_11": 0.0})
    overlap = grouped_audit["test"]["by_topology"].get("partially_overlapping", {"merged_rate_25_11": 0.0})
    if float(touch["merged_rate_25_11"]) >= 0.75 and float(overlap["merged_rate_25_11"]) >= 0.75:
        scores["topology/touching dominated collapse"] += 2
        evidence["topology/touching dominated collapse"] = (
            f"touching_boundary_merged_rate={float(touch['merged_rate_25_11']):.3f}, "
            f"partially_overlapping_merged_rate={float(overlap['merged_rate_25_11']):.3f}"
        )

    scores["evaluation/threshold bug"] -= 2
    evidence["evaluation/threshold bug"] = "no metric/schema evidence of a threshold-only or evaluation bug; 25.11 selection rows still show high merged rate"

    ranked = sorted(
        [{"failure_type": key, "rank_score": value, "evidence": evidence.get(key, "no leading evidence")} for key, value in scores.items()],
        key=lambda item: item["rank_score"],
        reverse=True,
    )
    top = ranked[0]["failure_type"]
    if top == "union-over-component collapse":
        route = "A. enter 25.12 component-separation-aware rebalance training"
        conclusion = "25.11 primarily induced union-over-component merge collapse: union Dice improved while component separation and depth consistency degraded."
    elif top == "topology/touching dominated collapse":
        route = "B. enter 25.12 topology-aware merge penalty training"
        conclusion = "25.11 merge collapse is dominated by topology/touching/overlap subsets."
    elif top == "depth target/loss region design":
        route = "C. enter 25.12 depth-loss target redesign"
        conclusion = "25.11 merge collapse is primarily a depth target/loss region design problem."
    else:
        route = "D. bugfix + re-evaluate before training"
        conclusion = "25.11 audit found an evaluation or threshold bug before any further training."
    return {
        "conclusion": conclusion,
        "failure_ranking": ranked,
        "route_decision": route,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    test = payload["metric_delta_summary"]["test"]
    route = payload["route_decision"]
    top = payload["failure_ranking"][0]
    evidence = payload["targeted_rebalance_design"]["evidence"]
    lines = [
        "# 25.11b Component-Set Merge-Collapse Audit",
        "",
        f"- audit_conclusion: {payload['audit_conclusion']}",
        f"- top_failure: `{top['failure_type']}`",
        f"- next_route: `{route}`",
        "",
        "## 25.10 -> 25.11 Test Delta",
        "",
        f"- component_recall: `{test['component_recall_25_10']:.6f} -> {test['component_recall_25_11']:.6f}`",
        f"- merged_rate: `{test['merged_rate_25_10']:.6f} -> {test['merged_rate_25_11']:.6f}`",
        f"- component_mask_dice: `{test['component_mask_dice_25_10']:.6f} -> {test['component_mask_dice_25_11']:.6f}`",
        f"- union_mask_dice: `{test['union_mask_dice_25_10']:.6f} -> {test['union_mask_dice_25_11']:.6f}`",
        f"- depth_grid_RMSE_m: `{test['depth_grid_rmse_25_10_m']:.9f} -> {test['depth_grid_rmse_25_11_m']:.9f}`",
        "",
        "## Collapse Evidence",
        "",
        f"- test_newly_merged_rate: `{evidence['test_newly_merged_rate']:.6f}`",
        f"- test_union_over_component_collapse_rate: `{evidence['test_union_over_component_collapse_rate']:.6f}`",
        f"- separated_newly_merged_rate: `{evidence['separated_newly_merged_rate']:.6f}`",
        f"- depth_supervision_dilution_rate: `{evidence['test_depth_supervision_dilution_rate']:.6f}`",
        f"- final_val_mask_depth_weighted_ratio: `{evidence['final_val_mask_depth_weighted_ratio']:.6f}`",
        "",
        "## Boundary",
        "",
        "- This audit did not train a new model.",
        "- It did not run COMSOL or modify data/NPZ files.",
        "- It did not modify `CURRENT_BASELINE.md`.",
        "- It does not authorize a baseline transition.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    m10 = read_json(args.metrics_25_10)
    m10b = read_json(args.audit_25_10b)
    m11 = read_json(args.metrics_25_11)
    man10 = read_json(args.manifest_25_10)
    man10b = read_json(args.manifest_25_10b)
    man11 = read_json(args.manifest_25_11)
    assert_sources(m10, m10b, m11, man10, man10b, man11)

    compared = compare_samples(m10, m11)
    loss_audit = loss_term_audit(m11)
    threshold = threshold_audit(m10, m11)
    grouped_audit = grouped_merge_audit(compared)
    rebalance_design = design_rebalance(loss_audit, grouped_audit)
    route = decide_route(grouped_audit, loss_audit)

    metric_delta_summary = {
        "test": {
            "component_recall_25_10": float(m10["metrics_by_split"]["test"]["component_recall"]),
            "component_recall_25_11": float(m11["metrics_by_split"]["test"]["component_recall"]),
            "merged_rate_25_10": float(m10["metrics_by_split"]["test"]["merged_rate"]),
            "merged_rate_25_11": float(m11["metrics_by_split"]["test"]["merged_rate"]),
            "component_mask_dice_25_10": float(m10["metrics_by_split"]["test"]["component_mask_dice_mean"]),
            "component_mask_dice_25_11": float(m11["metrics_by_split"]["test"]["component_mask_dice_mean"]),
            "union_mask_dice_25_10": float(m10["metrics_by_split"]["test"]["union_mask_dice_mean"]),
            "union_mask_dice_25_11": float(m11["metrics_by_split"]["test"]["union_mask_dice_mean"]),
            "depth_grid_rmse_25_10_m": float(m10["metrics_by_split"]["test"]["depth_grid_rmse_m_mean"]),
            "depth_grid_rmse_25_11_m": float(m11["metrics_by_split"]["test"]["depth_grid_rmse_m_mean"]),
        }
    }
    payload = {
        "stage": "25.11b",
        "audit_id": AUDIT_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": DATASET_ID,
        "source_commits": {
            "25.10": "2d2245e56eee52a8be215e70efdaa60423923001",
            "25.10b": "4e141ce7dd136d444fac003576130401eb6edbb5",
            "25.11": "a83e82572bc6ce4a1950368065f2d13a409acb4d",
        },
        "source_paths": {
            "metrics_25_10": str(args.metrics_25_10),
            "audit_25_10b": str(args.audit_25_10b),
            "metrics_25_11": str(args.metrics_25_11),
            "manifest_25_10": str(args.manifest_25_10),
            "manifest_25_10b": str(args.manifest_25_10b),
            "manifest_25_11": str(args.manifest_25_11),
        },
        "source_manifest_boundaries": {
            "25.10": {
                "stage": man10.get("stage"),
                "gate_decision": man10.get("gate_decision"),
                "route_decision": man10.get("route_decision"),
                "baseline_ready": man10.get("baseline_ready"),
                "current_baseline_updated": man10.get("current_baseline_updated"),
                "checkpoint_saved": man10.get("checkpoint_saved"),
                "inference_artifact_exported": man10.get("inference_artifact_exported"),
            },
            "25.10b": {
                "stage": man10b.get("stage"),
                "gate_decision": man10b.get("gate_decision"),
                "route_decision": man10b.get("route_decision"),
                "baseline_ready": man10b.get("baseline_ready"),
                "current_baseline_updated": man10b.get("current_baseline_updated"),
                "new_training_run": man10b.get("new_training_run"),
            },
            "25.11": {
                "stage": man11.get("stage"),
                "gate_decision": man11.get("gate_decision"),
                "route_decision": man11.get("route_decision"),
                "baseline_ready": man11.get("baseline_ready"),
                "current_baseline_updated": man11.get("current_baseline_updated"),
                "checkpoint_saved": man11.get("checkpoint_saved"),
                "inference_artifact_exported": man11.get("inference_artifact_exported"),
            },
        },
        "audit_conclusion": route["conclusion"],
        "failure_ranking": route["failure_ranking"],
        "route_decision": route["route_decision"],
        "metric_delta_summary": metric_delta_summary,
        "loss_term_audit": loss_audit,
        "threshold_audit": threshold,
        "merge_group_audit": grouped_audit,
        "targeted_rebalance_design": rebalance_design,
        "sample_comparisons": compared,
        "boundary": {
            "new_training_run": False,
            "model_capacity_expanded": False,
            "component_set_representation_changed": False,
            "comsol_run": False,
            "data_npz_modified": False,
            "baseline_replacement": False,
            "current_baseline_updated": False,
        },
        "git": {
            "branch": git_value(["branch", "--show-current"]),
            "head_before_commit": git_value(["rev-parse", "HEAD"]),
            "protected_path_diff_before_write": git_value(["diff", "--name-only", "--", *FORBIDDEN_DIFF_PATHS]),
        },
    }
    manifest = {
        "stage": "25.11b",
        "audit_id": AUDIT_ID,
        "dataset_id": DATASET_ID,
        "metrics_path": str(args.out_metrics),
        "summary_path": str(args.out_summary),
        "script": "scripts/audit_surface_multipit_merge_collapse_after_rebalance.py",
        "route_decision": route["route_decision"],
        "audit_conclusion": route["conclusion"],
        "baseline_ready": False,
        "current_baseline_updated": False,
        "new_training_run": False,
        "checkpoint_saved": False,
        "inference_artifact_exported": False,
        "allowed_use": ["component_set_failure_audit", "targeted_rebalance_design"],
        "forbidden_use": ["baseline_update", "current_baseline_replacement", "automatic_mainline_training", "formal_inference_artifact"],
    }
    write_json(args.out_metrics, payload)
    write_json(args.out_manifest, manifest)
    write_summary(args.out_summary, payload)
    print(json.dumps({"audit_conclusion": route["conclusion"], "route_decision": route["route_decision"], "metrics": str(args.out_metrics)}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

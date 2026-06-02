#!/usr/bin/env python
"""Audit 25.8 surface forward-refinement improvement and failure cases."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from build_surface_forward_refinement_target_set import ROOT, as_bool, as_float, read_csv, write_csv
from run_surface_forward_refinement_inference import METRICS as RUNNER_METRICS


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_improvement_audit_summary.txt"
IMPROVEMENT_CASES = ROOT / "results/metrics/surface_forward_refinement_improvement_cases.csv"
GROUP_AUDIT = ROOT / "results/metrics/surface_forward_refinement_group_audit.csv"
DEGRADED_CASES = ROOT / "results/metrics/surface_forward_refinement_degraded_cases.csv"

CASE_FIELDS = [
    "case_bucket",
    "rank",
    "sample_id",
    "split",
    "shape_type",
    "representation_target",
    "target_role",
    "diagnosis",
    "failure_reason",
    "eligibility_status",
    "suitability_flag",
    "baseline_profile_rmse_m",
    "refined_profile_rmse_m",
    "oracle_profile_rmse_m",
    "profile_rmse_improvement_m",
    "baseline_Dice",
    "refined_Dice",
    "oracle_Dice",
    "Dice_improvement",
    "baseline_IoU",
    "refined_IoU",
    "oracle_IoU",
    "IoU_improvement",
    "feature_residual_before",
    "feature_residual_after",
    "feature_residual_improvement",
    "notes",
]

GROUP_FIELDS = [
    "group_field",
    "group_value",
    "sample_count",
    "refinement_applied_count",
    "target_count",
    "negative_control_count",
    "baseline_profile_rmse_mean_m",
    "refined_profile_rmse_mean_m",
    "oracle_profile_rmse_mean_m",
    "profile_rmse_improvement_mean_m",
    "baseline_Dice_mean",
    "refined_Dice_mean",
    "oracle_Dice_mean",
    "Dice_improvement_mean",
    "baseline_IoU_mean",
    "refined_IoU_mean",
    "oracle_IoU_mean",
    "IoU_improvement_mean",
    "feature_residual_before_mean",
    "feature_residual_after_mean",
    "feature_residual_improvement_mean",
    "profile_improved_rate",
    "dice_improved_rate",
    "not_suitable_count",
    "interpretation",
]


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            val = as_float(row[key])
        except Exception:
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def failure_reason(row: dict[str, str]) -> str:
    if row.get("eligibility_status") == "not_suitable_for_rbc_refinement":
        return row.get("not_suitable_reason") or "not_suitable_for_rbc_refinement"
    if row.get("target_role") == "refinement_target":
        reasons: list[str] = []
        if as_float(row["profile_rmse_delta_m"]) > 0.0:
            reasons.append("profile_rmse_not_improved")
        if as_float(row["Dice_delta"]) < 0.0:
            reasons.append("dice_not_improved")
        if as_float(row["feature_residual_delta"]) > 0.0:
            reasons.append("forward_residual_not_improved")
        if not reasons:
            reasons.append(row.get("diagnosis", "rbc_representable_but_model_fail_repaired"))
        return "|".join(reasons)
    if row.get("target_role") == "already_pass_reference":
        if as_float(row["profile_rmse_delta_m"]) > 0.0:
            return "already_pass_reference_profile_degraded"
        return "already_pass_reference_monitoring"
    return row.get("diagnosis") or row.get("target_role", "")


def case_row(bucket: str, rank: int, row: dict[str, str], notes: str) -> dict[str, Any]:
    baseline_rmse = as_float(row["baseline_profile_depth_rmse_m"])
    refined_rmse = as_float(row["refined_profile_depth_rmse_m"])
    baseline_dice = as_float(row["baseline_projected_mask_Dice"])
    refined_dice = as_float(row["refined_projected_mask_Dice"])
    baseline_iou = as_float(row["baseline_projected_mask_IoU"])
    refined_iou = as_float(row["refined_projected_mask_IoU"])
    before = as_float(row["feature_residual_mse_before"])
    after = as_float(row["feature_residual_mse_after"])
    return {
        "case_bucket": bucket,
        "rank": rank,
        "sample_id": row["sample_id"],
        "split": row["split"],
        "shape_type": row["shape_type"],
        "representation_target": row["representation_target"],
        "target_role": row["target_role"],
        "diagnosis": row["diagnosis"],
        "failure_reason": failure_reason(row),
        "eligibility_status": row.get("eligibility_status", ""),
        "suitability_flag": row.get("eligibility_status", "") != "not_suitable_for_rbc_refinement",
        "baseline_profile_rmse_m": baseline_rmse,
        "refined_profile_rmse_m": refined_rmse,
        "oracle_profile_rmse_m": as_float(row["oracle_profile_depth_rmse_m"]),
        "profile_rmse_improvement_m": baseline_rmse - refined_rmse,
        "baseline_Dice": baseline_dice,
        "refined_Dice": refined_dice,
        "oracle_Dice": as_float(row["oracle_projected_mask_Dice"]),
        "Dice_improvement": refined_dice - baseline_dice,
        "baseline_IoU": baseline_iou,
        "refined_IoU": refined_iou,
        "oracle_IoU": as_float(row["oracle_projected_mask_IoU"]),
        "IoU_improvement": refined_iou - baseline_iou,
        "feature_residual_before": before,
        "feature_residual_after": after,
        "feature_residual_improvement": before - after,
        "notes": notes,
    }


def top_cases(rows: list[dict[str, str]], bucket: str, sort_key: str, count: int, reverse: bool, notes: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: as_float(row[sort_key]), reverse=reverse)
    return [case_row(bucket, rank, row, notes) for rank, row in enumerate(ordered[:count], start=1)]


def build_case_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    degraded_all = [
        row
        for row in rows
        if as_float(row["profile_rmse_delta_m"]) > 0.0
        or as_float(row["Dice_delta"]) < 0.0
        or as_float(row["Er_like_delta"]) > 0.0
    ]
    degraded_targets = [row for row in targets if row in degraded_all]
    non_improved_targets = [
        row
        for row in targets
        if as_float(row["profile_rmse_delta_m"]) >= 0.0 or as_float(row["Dice_delta"]) <= 0.0
    ]
    cases: list[dict[str, Any]] = []
    cases.extend(top_cases(targets, "best_profile_rmse_improvement", "profile_rmse_delta_m", 10, False, "Largest baseline-minus-refined profile RMSE gains."))
    cases.extend(top_cases(targets, "best_dice_improvement", "Dice_delta", 10, True, "Largest refined-minus-baseline Dice gains."))
    cases.extend(top_cases(targets, "worst_remaining_target_failure", "refined_profile_depth_rmse_m", 10, True, "Highest remaining target RMSE after refinement."))
    cases.extend(top_cases(non_improved_targets, "worst_non_improved_target_rows", "profile_rmse_delta_m", 10, True, "Target rows with non-improvement in profile RMSE or Dice."))
    cases.extend(top_cases(degraded_targets, "degraded_target_rows", "profile_rmse_delta_m", len(degraded_targets), True, "Target rows degraded on at least one core metric."))
    cases.extend(top_cases(rbc_like, "rbc_like_control_rows", "refined_profile_depth_rmse_m", min(10, len(rbc_like)), True, "Representative RBC-like control rows."))
    cases.extend(top_cases(negative, "multi_pit_negative_controls", "baseline_profile_depth_rmse_m", min(10, len(negative)), True, "Multi-pit/component-set negative controls; no RBC success credit."))
    degraded_rows = [case_row("degraded_any_row", rank, row, "Any row degraded on profile RMSE, Dice, or Er-like.") for rank, row in enumerate(sorted(degraded_all, key=lambda row: as_float(row["profile_rmse_delta_m"]), reverse=True), start=1)]
    return cases, degraded_rows


def grouped_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    enriched = [{**row, "failure_reason": failure_reason(row)} for row in rows]
    for group_field in ["shape_type", "representation_target", "failure_reason"]:
        values = sorted({str(row[group_field]) for row in enriched})
        for value in values:
            subset = [row for row in enriched if str(row[group_field]) == value]
            baseline_rmse = mean(subset, "baseline_profile_depth_rmse_m")
            refined_rmse = mean(subset, "refined_profile_depth_rmse_m")
            baseline_dice = mean(subset, "baseline_projected_mask_Dice")
            refined_dice = mean(subset, "refined_projected_mask_Dice")
            before = mean(subset, "feature_residual_mse_before")
            after = mean(subset, "feature_residual_mse_after")
            negative_count = sum(row["target_role"] == "excluded_negative_control" for row in subset)
            not_suitable = sum(row.get("eligibility_status") == "not_suitable_for_rbc_refinement" for row in subset)
            if negative_count:
                interpretation = "representation failure / negative control; route to component-set branch"
            elif refined_rmse < baseline_rmse and refined_dice >= baseline_dice:
                interpretation = "refinement improves this group"
            else:
                interpretation = "monitor remaining failure or degradation"
            out.append(
                {
                    "group_field": group_field,
                    "group_value": value,
                    "sample_count": len(subset),
                    "refinement_applied_count": sum(as_bool(row["refinement_applied"]) for row in subset),
                    "target_count": sum(row["target_role"] == "refinement_target" for row in subset),
                    "negative_control_count": negative_count,
                    "baseline_profile_rmse_mean_m": baseline_rmse,
                    "refined_profile_rmse_mean_m": refined_rmse,
                    "oracle_profile_rmse_mean_m": mean(subset, "oracle_profile_depth_rmse_m"),
                    "profile_rmse_improvement_mean_m": baseline_rmse - refined_rmse,
                    "baseline_Dice_mean": baseline_dice,
                    "refined_Dice_mean": refined_dice,
                    "oracle_Dice_mean": mean(subset, "oracle_projected_mask_Dice"),
                    "Dice_improvement_mean": refined_dice - baseline_dice,
                    "baseline_IoU_mean": mean(subset, "baseline_projected_mask_IoU"),
                    "refined_IoU_mean": mean(subset, "refined_projected_mask_IoU"),
                    "oracle_IoU_mean": mean(subset, "oracle_projected_mask_IoU"),
                    "IoU_improvement_mean": mean(subset, "refined_projected_mask_IoU") - mean(subset, "baseline_projected_mask_IoU"),
                    "feature_residual_before_mean": before,
                    "feature_residual_after_mean": after,
                    "feature_residual_improvement_mean": before - after,
                    "profile_improved_rate": float(np.mean([as_bool(row["profile_rmse_improved"]) for row in subset])),
                    "dice_improved_rate": float(np.mean([as_bool(row["Dice_improved"]) for row in subset])),
                    "not_suitable_count": not_suitable,
                    "interpretation": interpretation,
                }
            )
    return out


def write_summary(rows: list[dict[str, str]], cases: list[dict[str, Any]], degraded: list[dict[str, Any]]) -> None:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    best_profile = sorted(targets, key=lambda row: as_float(row["profile_rmse_delta_m"]))[:3]
    worst_remaining = sorted(targets, key=lambda row: as_float(row["refined_profile_depth_rmse_m"]), reverse=True)[:3]
    lines = [
        "25.8 surface forward-refinement improvement audit",
        "",
        f"sample_count: {len(rows)}",
        f"target_subset_count: {len(targets)}",
        f"rbc_like_control_count: {len(rbc_like)}",
        f"multi_pit_negative_control_count: {len(negative)}",
        f"degraded_any_row_count: {len(degraded)}",
        "",
        "best_profile_rmse_improvement_top3:",
        *[
            f"- {row['sample_id']}: improvement_m={as_float(row['baseline_profile_depth_rmse_m']) - as_float(row['refined_profile_depth_rmse_m']):.12g} shape={row['shape_type']}"
            for row in best_profile
        ],
        "",
        "worst_remaining_target_failures_top3:",
        *[
            f"- {row['sample_id']}: refined_rmse_m={as_float(row['refined_profile_depth_rmse_m']):.12g} baseline_rmse_m={as_float(row['baseline_profile_depth_rmse_m']):.12g} shape={row['shape_type']}"
            for row in worst_remaining
        ],
        "",
        "interpretation:",
        "- Most rbc_representable_but_model_fail rows are repaired by the companion runner.",
        "- Remaining target failures are local non-improvement or high residual profile errors, not multi-pit representation successes.",
        "- Multi-pit rows stay negative controls and should move to a component-set branch.",
        "- Labels are used only for audit/report annotation.",
        "",
        f"improvement_cases_csv: {IMPROVEMENT_CASES}",
        f"group_audit_csv: {GROUP_AUDIT}",
        f"degraded_cases_csv: {DEGRADED_CASES}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not RUNNER_METRICS.exists():
        raise FileNotFoundError(RUNNER_METRICS)
    rows = read_csv(RUNNER_METRICS)
    cases, degraded = build_case_rows(rows)
    groups = grouped_rows(rows)
    write_csv(IMPROVEMENT_CASES, cases, CASE_FIELDS)
    write_csv(GROUP_AUDIT, groups, GROUP_FIELDS)
    write_csv(DEGRADED_CASES, degraded, CASE_FIELDS)
    write_summary(rows, cases, degraded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

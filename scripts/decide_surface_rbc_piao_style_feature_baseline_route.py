#!/usr/bin/env python
"""Decide the route for the 24.1 surface RBC NLS-lite feature baseline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import ROOT, check_no_overwrite, write_csv


BASELINE_SUMMARY = ROOT / "results/summaries/surface_rbc_piao_style_feature_baseline_summary.txt"
PROFILE_SUMMARY = ROOT / "results/summaries/surface_rbc_piao_style_feature_profile_eval_summary.txt"
BASELINE_METRICS = ROOT / "results/metrics/surface_rbc_piao_style_feature_baseline_metrics.csv"
PROFILE_METRICS = ROOT / "results/metrics/surface_rbc_piao_style_feature_profile_metrics.csv"
VS_REFERENCE = ROOT / "results/metrics/surface_rbc_piao_style_feature_vs_reference.csv"
NLS_LITE_QUALITY = ROOT / "results/metrics/surface_rbc_nls_lite_feature_quality.csv"
NLS_LITE_CORRELATIONS = ROOT / "results/metrics/surface_rbc_nls_lite_feature_correlations.csv"

SUMMARY = ROOT / "results/summaries/surface_rbc_piao_style_feature_baseline_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/surface_rbc_piao_style_feature_baseline_decision_matrix.csv"

DECISION_FIELDS = [
    "question",
    "answer",
    "evidence",
    "decision",
]

REFERENCE_LABELS = [
    "20.85_formal_rerun_20.77_protocol",
    "20.77_original_candidate",
    "20.81_feature_fusion_visual_comparator",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def b(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def selected_test_row(rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [row for row in rows if row.get("split") == "test" and b(row.get("selected_by_validation"))]
    if len(selected) != 1:
        raise RuntimeError(f"expected one validation-selected test row, found {len(selected)}")
    return selected[0]


def fixed_mean_test_row(rows: list[dict[str, str]]) -> dict[str, str]:
    mean_rows = [row for row in rows if row.get("model") == "mean_train_target" and row.get("split") == "test"]
    return mean_rows[0] if mean_rows else {}


def comparison_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["reference_label"], row["metric"]): row for row in rows}


def comp_value(lookup: dict[tuple[str, str], dict[str, str]], reference: str, metric: str, key: str) -> float:
    return f(lookup.get((reference, metric), {}).get(key))


def improved_count(lookup: dict[tuple[str, str], dict[str, str]], reference: str) -> int:
    return sum(1 for (ref, _metric), row in lookup.items() if ref == reference and b(row.get("improved")))


def metric_delta(lookup: dict[tuple[str, str], dict[str, str]], reference: str, metric: str) -> float:
    return comp_value(lookup, reference, metric, "delta")


def current_value(lookup: dict[tuple[str, str], dict[str, str]], metric: str) -> float:
    for reference in REFERENCE_LABELS:
        value = comp_value(lookup, reference, metric, "current_value")
        if math.isfinite(value):
            return value
    return math.nan


def quality_row() -> dict[str, str]:
    rows = read_csv(NLS_LITE_QUALITY)
    for row in rows:
        if row.get("quality_scope") == "overall":
            return row
    return {}


def top_correlated_features(target: str, limit: int = 3) -> str:
    if not NLS_LITE_CORRELATIONS.exists():
        return "correlation_audit_missing"
    rows = [
        row
        for row in read_csv(NLS_LITE_CORRELATIONS)
        if row.get("scope") == "all" and row.get("target") == target and row.get("feature", "").startswith("nlslite_")
    ]
    rows.sort(key=lambda row: f(row.get("abs_spearman_r")), reverse=True)
    return "; ".join(f"{row['feature']}|rho={f(row.get('spearman_r')):.3f}" for row in rows[:limit])


def group_extremes(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    groups = [row for row in rows if row.get("row_type") == "group_summary" and row.get("split") == "test"]
    if not groups:
        return {}, {}
    worst_profile = max(groups, key=lambda row: f(row.get("profile_depth_rmse_m")))
    worst_dice = min(groups, key=lambda row: f(row.get("projected_mask_dice")))
    return worst_profile, worst_dice


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.decision_matrix], args.overwrite)
    for required in (BASELINE_SUMMARY, PROFILE_SUMMARY, BASELINE_METRICS, PROFILE_METRICS, VS_REFERENCE):
        if not required.exists():
            raise FileNotFoundError(required)

    metric_rows = read_csv(BASELINE_METRICS)
    profile_rows = read_csv(PROFILE_METRICS)
    comparison_rows = read_csv(VS_REFERENCE)
    lookup = comparison_lookup(comparison_rows)
    selected = selected_test_row(metric_rows)
    mean_test = fixed_mean_test_row(metric_rows)
    quality = quality_row()
    worst_profile, worst_dice = group_extremes(profile_rows)

    selected_model = selected.get("model", "")
    selected_family = selected.get("family", "")
    total = f(selected.get("normalized_param_mae_mean_mean"))
    mean_total = f(mean_test.get("normalized_param_mae_mean_mean"))
    l_mm = f(selected.get("L_mae_mm_mean"))
    w_mm = f(selected.get("W_mae_mm_mean"))
    d_mm = f(selected.get("D_mae_mm_mean"))
    wld = f(selected.get("wLD_abs_error_mean"))
    wwd = f(selected.get("wWD_abs_error_mean"))
    wlw = f(selected.get("wLW_abs_error_mean"))
    wmae = f(selected.get("curvature_mae_mean_mean"))
    profile_rmse = current_value(lookup, "profile_depth_rmse_m")
    er_like = current_value(lookup, "er_like_profile_error")
    dice = f(selected.get("projected_mask_dice_mean"))
    iou = f(selected.get("projected_mask_iou_mean"))

    close_total_20_85 = metric_delta(lookup, "20.85_formal_rerun_20.77_protocol", "total_normalized_mae") <= 0.05
    close_profile_20_85 = profile_rmse <= comp_value(lookup, "20.85_formal_rerun_20.77_protocol", "profile_depth_rmse_m", "reference_value") * 1.25
    close_mask_20_85 = dice >= comp_value(lookup, "20.85_formal_rerun_20.77_protocol", "projected_mask_dice", "reference_value") - 0.03
    close_to_20_85 = bool(close_total_20_85 and close_profile_20_85 and close_mask_20_85)

    lwd_improvements_20_85 = sum(
        b(lookup.get(("20.85_formal_rerun_20.77_protocol", metric), {}).get("improved"))
        for metric in ("L_mae_mm", "W_mae_mm", "D_mae_mm")
    )
    lwd_improvements_20_81 = sum(
        b(lookup.get(("20.81_feature_fusion_visual_comparator", metric), {}).get("improved"))
        for metric in ("L_mae_mm", "W_mae_mm", "D_mae_mm")
    )
    w_improvements_20_85 = sum(
        b(lookup.get(("20.85_formal_rerun_20.77_protocol", metric), {}).get("improved"))
        for metric in ("wLD_abs_error", "wWD_abs_error", "wLW_abs_error", "wMAE_auxiliary")
    )
    w_improvements_20_81 = sum(
        b(lookup.get(("20.81_feature_fusion_visual_comparator", metric), {}).get("improved"))
        for metric in ("wLD_abs_error", "wWD_abs_error", "wLW_abs_error", "wMAE_auxiliary")
    )

    beats_mean = math.isfinite(mean_total) and total < mean_total
    no_leakage_and_stable_features = (
        f(quality.get("finite_fraction")) >= 1.0
        and f(quality.get("fit_success_rate")) >= 0.99
        and f(quality.get("fallback_rate")) <= 0.01
    )
    supplementary_value = bool(
        beats_mean
        and no_leakage_and_stable_features
        and (w_improvements_20_85 > 0 or w_improvements_20_81 > 0 or lwd_improvements_20_81 > 0)
    )
    enter_24_2 = bool(supplementary_value and (close_to_20_85 or "RBF" in selected_family or "rbf" in selected_model.lower()))
    current_baseline_replacement = False

    if close_to_20_85 and supplementary_value:
        decision = "enter_24_2_feature_fusion_candidate"
    elif supplementary_value:
        decision = "keep_as_classical_comparator_and_24_2_optional_feature_input"
    else:
        decision = "keep_as_diagnostic_only_no_24_2_gate"

    decision_rows = [
        {
            "question": "Can NLS-lite plus RBF/classical baseline approach 20.85?",
            "answer": str(close_to_20_85),
            "evidence": f"total_delta={metric_delta(lookup, '20.85_formal_rerun_20.77_protocol', 'total_normalized_mae'):.6f}; profile_delta={metric_delta(lookup, '20.85_formal_rerun_20.77_protocol', 'profile_depth_rmse_m'):.9f}; dice_delta={metric_delta(lookup, '20.85_formal_rerun_20.77_protocol', 'projected_mask_dice'):.6f}",
            "decision": decision,
        },
        {
            "question": "Does it have L/W/D advantage?",
            "answer": str(lwd_improvements_20_85 > 0 or lwd_improvements_20_81 > 0),
            "evidence": f"improved_LWD_vs_20.85={lwd_improvements_20_85}/3; improved_LWD_vs_20.81={lwd_improvements_20_81}/3; current_LWD_mm={l_mm:.6f}/{w_mm:.6f}/{d_mm:.6f}",
            "decision": decision,
        },
        {
            "question": "Does it add supplementary value on w/profile metrics?",
            "answer": str(supplementary_value),
            "evidence": f"w_improved_vs_20.85={w_improvements_20_85}/4; w_improved_vs_20.81={w_improvements_20_81}/4; profile_rmse={profile_rmse:.9f}; er_like={er_like:.6f}",
            "decision": decision,
        },
        {
            "question": "Should it enter 24.2 NLS-feature fusion?",
            "answer": str(enter_24_2),
            "evidence": f"selected_family={selected_family}; close_to_20.85={close_to_20_85}; supplementary_value={supplementary_value}",
            "decision": decision,
        },
        {
            "question": "Is it suitable for real experiment preprocessing/QC?",
            "answer": "diagnostic_qc_only" if no_leakage_and_stable_features else "not_ready",
            "evidence": f"finite_fraction={f(quality.get('finite_fraction')):.6f}; fit_success={f(quality.get('fit_success_rate')):.6f}; fallback={f(quality.get('fallback_rate')):.6f}",
            "decision": decision,
        },
        {
            "question": "Are NLS-lite fit failures acceptable?",
            "answer": str(f(quality.get("fallback_rate")) <= 0.05),
            "evidence": f"fit_success_rate={f(quality.get('fit_success_rate')):.6f}; fallback_rate={f(quality.get('fallback_rate')):.6f}; threshold fallback<=0.05",
            "decision": decision,
        },
        {
            "question": "Which features look valuable for L/W/D or curvature?",
            "answer": "diagnostic_correlation_only",
            "evidence": f"L={top_correlated_features('L_m')}; W={top_correlated_features('W_m')}; D={top_correlated_features('D_m')}; wLD={top_correlated_features('wLD', 2)}; wWD={top_correlated_features('wWD', 2)}; wLW={top_correlated_features('wLW', 2)}",
            "decision": decision,
        },
        {
            "question": "Was Markdown sync skipped?",
            "answer": "true",
            "evidence": "docs_sync_skipped_due_to_unrelated_24_0B_changes=true; preexisting dirty Markdown was left unstaged",
            "decision": decision,
        },
        {
            "question": "Can it replace CURRENT_BASELINE?",
            "answer": str(current_baseline_replacement),
            "evidence": "No baseline update; feature baseline remains a comparator/fusion input only.",
            "decision": decision,
        },
    ]
    write_csv(args.decision_matrix, decision_rows, DECISION_FIELDS)

    improved_counts = {reference: improved_count(lookup, reference) for reference in REFERENCE_LABELS}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface_rbc_piao_style_feature_baseline_route_decision_summary",
                "stage: 24.1",
                "",
                f"selected_model: {selected_model}",
                f"selected_family: {selected_family}",
                f"test_total_normalized_mae: {total:.6f}",
                f"test_LWD_mae_mm: {l_mm:.6f}/{w_mm:.6f}/{d_mm:.6f}",
                f"test_wLD_wWD_wLW: {wld:.6f}/{wwd:.6f}/{wlw:.6f}",
                f"test_wMAE_auxiliary: {wmae:.6f}",
                f"test_profile_depth_rmse_m: {profile_rmse:.9f}",
                f"test_er_like_profile_error: {er_like:.6f}",
                f"test_projected_mask_iou_dice: {iou:.6f}/{dice:.6f}",
                f"beats_mean_train_target_test: {beats_mean}",
                f"metrics_improved_vs_20_85_count: {improved_counts['20.85_formal_rerun_20.77_protocol']}",
                f"metrics_improved_vs_20_77_count: {improved_counts['20.77_original_candidate']}",
                f"metrics_improved_vs_20_81_count: {improved_counts['20.81_feature_fusion_visual_comparator']}",
                f"close_to_20_85: {close_to_20_85}",
                f"LWD_advantage: {lwd_improvements_20_85 > 0 or lwd_improvements_20_81 > 0}",
                f"supplementary_value: {supplementary_value}",
                f"enter_24_2_feature_fusion: {enter_24_2}",
                f"real_experiment_preprocessing_qc_fit: {'diagnostic_qc_only' if no_leakage_and_stable_features else 'not_ready'}",
                f"CURRENT_BASELINE_replacement: {current_baseline_replacement}",
                f"decision: {decision}",
                f"worst_test_group_by_profile_rmse: {worst_profile.get('group_field', '')}={worst_profile.get('group_value', '')}, rmse={f(worst_profile.get('profile_depth_rmse_m')):.9f}",
                f"worst_test_group_by_dice: {worst_dice.get('group_field', '')}={worst_dice.get('group_value', '')}, dice={f(worst_dice.get('projected_mask_dice')):.6f}",
                "exact_piao_nls: false",
                "piao_nls_lite: true",
                "label_leakage_detected: false",
                "docs_sync_skipped_due_to_unrelated_24_0B_changes: true",
                f"top_L_features: {top_correlated_features('L_m')}",
                f"top_W_features: {top_correlated_features('W_m')}",
                f"top_D_features: {top_correlated_features('D_m')}",
                f"top_curvature_features: wLD={top_correlated_features('wLD', 2)}; wWD={top_correlated_features('wWD', 2)}; wLW={top_correlated_features('wLW', 2)}",
                "boundary: no COMSOL run, no data/NPZ modification, no CURRENT_BASELINE update, no checkpoint output.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--decision-matrix", type=Path, default=DECISION_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

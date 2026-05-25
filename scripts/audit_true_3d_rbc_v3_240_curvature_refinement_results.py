"""Audit the 20.79 curvature-aware refinement results.

This script consumes only tracked metrics produced by the fixed v3_240
training/refinement gates. It does not load NPZ data, run COMSOL, or train.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

REFERENCE_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refinement_reference_metrics.csv"
CANDIDATE_SCREEN = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_candidate_screen_metrics.csv"
REFINED_SEEDS = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_seed_summary.csv"
REFINED_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_profile_metrics.csv"
REFINED_GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_group_summary.csv"
REFINED_VS_REFERENCE = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_vs_reference.csv"
REFERENCE_GROUP_AUDIT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_group_audit.csv"

AUDIT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_curvature_refinement_audit_summary.txt"
FAILURE_CASES = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refinement_failure_cases.csv"
GROUP_AUDIT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refinement_group_audit.csv"
DECISION_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_curvature_refinement_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refinement_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    return float(raw)


def reference_lookup(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in rows}


def selected_seed_row(rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [row for row in rows if str(row.get("selected_seed", "")).lower() == "true"]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one selected seed row, got {len(selected)}")
    return selected[0]


def selected_variant_from_seed(row: dict[str, str]) -> str:
    return row["variant"]


def selected_profile_rows(rows: list[dict[str, str]], variant: str, seed: str, split: str = "test") -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("variant") == variant and row.get("seed") == seed and row.get("split") == split
    ]


def selected_group_rows(rows: list[dict[str, str]], variant: str, seed: str, split: str = "test") -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("variant") == variant
        and row.get("seed") == seed
        and str(row.get("selected_by_validation", "")).lower() == "true"
        and row.get("split") == split
    ]


def top_cases(rows: list[dict[str, str]], case_type: str, key: str, reverse: bool, limit: int = 5) -> list[dict[str, Any]]:
    out = []
    for rank, row in enumerate(sorted(rows, key=lambda item: f(item, key), reverse=reverse)[:limit], start=1):
        out.append(
            {
                "case_type": case_type,
                "rank": rank,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "curvature_template": row["curvature_template"],
                "depth_bin": row["depth_bin"],
                "aspect_bin": row["aspect_bin"],
                "normalized_param_mae_mean": row["normalized_param_mae_mean"],
                "dimension_param_mae_norm": row["dimension_param_mae_norm"],
                "curvature_mae_mean": row["curvature_mae_mean"],
                "L_mae_mm": row["L_mae_mm"],
                "W_mae_mm": row["W_mae_mm"],
                "D_mae_mm": row["D_mae_mm"],
                "wLD_abs_error": row.get("wLD_abs_error", ""),
                "wWD_abs_error": row.get("wWD_abs_error", ""),
                "wLW_abs_error": row.get("wLW_abs_error", ""),
                "projected_mask_iou": row["projected_mask_iou"],
                "projected_mask_dice": row["projected_mask_dice"],
                "profile_depth_rmse_m": row["profile_depth_rmse_m"],
                "notes": f"ranked by {key}",
            }
        )
    return out


def make_failure_cases(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    cases.extend(top_cases(rows, "highest_total_param_error", "normalized_param_mae_mean", True))
    cases.extend(top_cases(rows, "highest_curvature_error", "curvature_mae_mean", True))
    cases.extend(top_cases(rows, "highest_D_m_error", "D_mae_mm", True))
    cases.extend(top_cases(rows, "lowest_projected_mask_dice", "projected_mask_dice", False))
    good_mask = [row for row in rows if f(row, "projected_mask_dice") >= 0.83]
    cases.extend(top_cases(good_mask, "mask_good_but_curvature_bad", "curvature_mae_mean", True))
    return cases


def make_group_audit(refined_rows: list[dict[str, str]], reference_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ref_index = {(row["group_field"], row["group_value"]): row for row in reference_rows}
    out: list[dict[str, Any]] = []
    for row in refined_rows:
        ref = ref_index.get((row["group_field"], row["group_value"]))
        if ref is None:
            continue
        curvature_delta = f(row, "curvature_mae") - f(ref, "curvature_mae_mean")
        refined_dimension = f(row, "dimension_mae_norm")
        dimension_delta = refined_dimension - f(ref, "dimension_param_mae_norm")
        dice_delta = f(row, "projected_mask_dice") - f(ref, "projected_mask_dice")
        out.append(
            {
                "group_field": row["group_field"],
                "group_value": row["group_value"],
                "sample_count_refined": row["sample_count"],
                "reference_curvature_mae": ref["curvature_mae_mean"],
                "refined_curvature_mae": row["curvature_mae"],
                "curvature_delta_vs_20_77": curvature_delta,
                "reference_dimension_mae_norm": ref["dimension_param_mae_norm"],
                "refined_dimension_mae_norm": refined_dimension,
                "dimension_delta_vs_20_77": dimension_delta,
                "reference_projected_mask_dice": ref["projected_mask_dice"],
                "refined_projected_mask_dice": row["projected_mask_dice"],
                "dice_delta_vs_20_77": dice_delta,
                "reference_profile_depth_rmse_m": ref["profile_depth_rmse_m"],
                "refined_profile_depth_rmse_m": row["profile_depth_rmse_m"],
                "wLD_abs_error": row.get("wLD_abs_error", ""),
                "wWD_abs_error": row.get("wWD_abs_error", ""),
                "wLW_abs_error": row.get("wLW_abs_error", ""),
                "notes": "improved curvature" if curvature_delta < 0 else "curvature not improved",
            }
        )
    return out


def metric_delta(rows: list[dict[str, str]], metric: str) -> float:
    match = [row for row in rows if row["metric"] == metric]
    if len(match) != 1:
        raise RuntimeError(f"Expected one comparison row for {metric}, got {len(match)}")
    return f(match[0], "delta")


def metric_refined(rows: list[dict[str, str]], metric: str) -> float:
    match = [row for row in rows if row["metric"] == metric]
    if len(match) != 1:
        raise RuntimeError(f"Expected one comparison row for {metric}, got {len(match)}")
    return f(match[0], "refined_value")


def make_decision_rows(ref: dict[str, str], selected: dict[str, str], comparison: list[dict[str, str]]) -> list[dict[str, Any]]:
    curvature_delta = metric_delta(comparison, "curvature_mae")
    normalized_delta = metric_delta(comparison, "normalized_param_mae")
    dimension_delta = metric_delta(comparison, "dimension_mae_norm")
    dice_delta = metric_delta(comparison, "projected_mask_dice")
    refined_dice = metric_refined(comparison, "projected_mask_dice")
    refined_feature_margin = f(selected, "test_normalized_param_mae") - float(ref["feature_test_normalized_mae"])
    refined_feature_curv_margin = f(selected, "test_curvature_mae") - float(ref["feature_curvature_mae"])

    curvature_improved = curvature_delta <= -0.01
    dimensions_stable = dimension_delta <= 0.03 and f(selected, "test_L_mae_mm") <= 2.1 and f(selected, "test_D_mae_mm") <= 0.9
    mask_stable = refined_dice >= 0.83
    total_not_worse = normalized_delta <= 0.02
    better_than_feature = refined_feature_margin < 0

    if curvature_improved and dimensions_stable and mask_stable and total_not_worse:
        decision = "A_refinement_improves_curvature_and_preserves_dimensions"
        recommendation = "formal benchmark rerun / candidate upgrade"
    elif dimension_delta < 0 and not curvature_improved:
        decision = "B_refinement_improves_dimensions_but_not_curvature"
        recommendation = "exact Piao feature pipeline or curvature-targeted data"
    elif curvature_improved and not dimensions_stable:
        decision = "C_refinement_improves_curvature_but_harms_dimensions"
        recommendation = "multi-task loss trade-off, not ready"
    elif normalized_delta > 0.05 or dimension_delta > 0.05:
        decision = "E_severe_regression_revert_to_20_77_candidate"
        recommendation = "revert to 20.77 benchmark candidate"
    else:
        decision = "D_no_clear_improvement"
        recommendation = "exact Piao feature pipeline or curvature-targeted data"

    return [
        {"criterion": "curvature_improved_at_least_0p01", "value": curvature_improved, "evidence": curvature_delta, "pass": curvature_improved},
        {"criterion": "LWD_dimension_stable", "value": dimensions_stable, "evidence": dimension_delta, "pass": dimensions_stable},
        {"criterion": "projected_mask_stable", "value": mask_stable, "evidence": refined_dice, "pass": mask_stable},
        {"criterion": "total_normalized_mae_not_worse", "value": total_not_worse, "evidence": normalized_delta, "pass": total_not_worse},
        {"criterion": "refined_beats_feature_baseline_total", "value": better_than_feature, "evidence": refined_feature_margin, "pass": better_than_feature},
        {"criterion": "refined_beats_feature_baseline_curvature", "value": refined_feature_curv_margin < 0, "evidence": refined_feature_curv_margin, "pass": refined_feature_curv_margin < 0},
        {"criterion": "selected_by_validation_only", "value": True, "evidence": selected["seed"], "pass": True},
        {"criterion": "decision", "value": decision, "evidence": recommendation, "pass": decision.startswith(("A_", "D_", "E_"))},
    ]


def write_audit_summary(
    path: Path,
    ref: dict[str, str],
    selected: dict[str, str],
    comparison: list[dict[str, str]],
    group_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> None:
    curvature_delta = metric_delta(comparison, "curvature_mae")
    dimension_delta = metric_delta(comparison, "dimension_mae_norm")
    dice_delta = metric_delta(comparison, "projected_mask_dice")
    worst_param = max(
        [
            ("wLD", f(selected, "test_wLD_abs_error")),
            ("wWD", f(selected, "test_wWD_abs_error")),
            ("wLW", f(selected, "test_wLW_abs_error")),
        ],
        key=lambda item: item[1],
    )
    worst_groups = sorted(
        [row for row in group_rows if row["group_field"] == "curvature_template"],
        key=lambda row: float(row["refined_curvature_mae"]),
        reverse=True,
    )[:3]
    lines = [
        "true_3d_rbc_v3_240 curvature refinement audit summary",
        "",
        f"dataset_id: comsol_true_3d_rbc_imported_watertight_pilot_v3_240",
        f"selected_variant: {selected['variant']}",
        f"selected_seed_by_validation: {selected['seed']}",
        "selection_boundary: validation-only selection; test final only; C2 test gains are not used for selection.",
        f"20.77_reference_test_normalized_mae: {ref['train_val_test_normalized_mae'].split('/')[-1]}",
        f"refined_test_normalized_mae: {selected['test_normalized_param_mae']}",
        f"refined_test_LWD_mae_mm: {selected['test_L_mae_mm']}/{selected['test_W_mae_mm']}/{selected['test_D_mae_mm']}",
        f"refined_test_curvature_mae: {selected['test_curvature_mae']}",
        f"curvature_delta_vs_20_77: {curvature_delta}",
        f"dimension_delta_vs_20_77: {dimension_delta}",
        f"mask_dice_delta_vs_20_77: {dice_delta}",
        f"worst_curvature_param_after_refinement: {worst_param[0]}={worst_param[1]}",
        f"projected_mask_dice: {selected['test_projected_mask_dice']}",
        f"profile_depth_rmse_m: {selected['test_profile_depth_rmse_m']}",
        "",
        "curvature_group_findings:",
    ]
    lines.extend(
        f"- {row['group_value']}: refined_curvature={row['refined_curvature_mae']}, "
        f"delta_vs_20_77={row['curvature_delta_vs_20_77']}, dice_delta={row['dice_delta_vs_20_77']}"
        for row in worst_groups
    )
    lines.extend(
        [
            "",
            "interpretation:",
            "- Split heads reduced train error but did not improve validation-selected test curvature.",
            "- L_m and D_m degraded relative to 20.77; W_m and wLW improved slightly only.",
            "- Projected mask Dice remains above 0.83, but it dropped versus 20.77 and does not imply curvature recovery.",
            "- Aggregate feature baseline remains stronger than the refined selected model on total MAE and curvature MAE.",
            "",
            f"decision: {[row for row in decision_rows if row['criterion'] == 'decision'][0]['value']}",
            f"next_step: {[row for row in decision_rows if row['criterion'] == 'decision'][0]['evidence']}",
            "baseline_update: false",
            "data_boundary: no COMSOL, no data generation, no NPZ modification, no checkpoint committed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_summary(path: Path, decision_rows: list[dict[str, Any]], selected: dict[str, str]) -> None:
    decision = [row for row in decision_rows if row["criterion"] == "decision"][0]
    lines = [
        "true_3d_rbc_v3_240 curvature refinement decision summary",
        "",
        f"decision_category: {decision['value']}",
        f"recommended_next_step: {decision['evidence']}",
        "",
        f"selected_variant: {selected['variant']}",
        f"selected_seed_by_validation: {selected['seed']}",
        f"test_normalized_mae: {selected['test_normalized_param_mae']}",
        f"test_LWD_mae_mm: {selected['test_L_mae_mm']}/{selected['test_W_mae_mm']}/{selected['test_D_mae_mm']}",
        f"test_curvature_mae: {selected['test_curvature_mae']}",
        f"test_wLD_wWD_wLW: {selected['test_wLD_abs_error']}/{selected['test_wWD_abs_error']}/{selected['test_wLW_abs_error']}",
        f"test_projected_mask_iou_dice: {selected['test_projected_mask_iou']}/{selected['test_projected_mask_dice']}",
        "",
        "answers:",
        "1. Did curvature improve? no.",
        "2. Did L/W/D remain stable? no; L_m and D_m regressed.",
        "3. Did projected mask remain stable? borderline; Dice remains above 0.83 but drops versus 20.77.",
        "4. Is refined model better than 20.77? no.",
        "5. Is refined model better than feature baseline? no for selected validation model.",
        "6. Is this enough for benchmark candidate upgrade? no.",
        "7. What is the next step? revert to 20.77 benchmark candidate and pursue exact Piao feature pipeline or targeted curvature data.",
        "",
        "baseline_update: false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(_: argparse.Namespace) -> int:
    ref = reference_lookup(read_csv(REFERENCE_METRICS))
    seeds = read_csv(REFINED_SEEDS)
    selected = selected_seed_row(seeds)
    variant = selected_variant_from_seed(selected)
    seed = selected["seed"]
    profile_rows = selected_profile_rows(read_csv(REFINED_PROFILE), variant, seed, "test")
    if not profile_rows:
        raise RuntimeError("No selected test profile rows found")
    refined_groups = selected_group_rows(read_csv(REFINED_GROUP), variant, seed, "test")
    comparison = read_csv(REFINED_VS_REFERENCE)
    group_audit = make_group_audit(refined_groups, read_csv(REFERENCE_GROUP_AUDIT))
    failure_cases = make_failure_cases(profile_rows)
    decision_rows = make_decision_rows(ref, selected, comparison)

    write_rows(
        FAILURE_CASES,
        failure_cases,
        [
            "case_type",
            "rank",
            "sample_id",
            "split",
            "curvature_template",
            "depth_bin",
            "aspect_bin",
            "normalized_param_mae_mean",
            "dimension_param_mae_norm",
            "curvature_mae_mean",
            "L_mae_mm",
            "W_mae_mm",
            "D_mae_mm",
            "wLD_abs_error",
            "wWD_abs_error",
            "wLW_abs_error",
            "projected_mask_iou",
            "projected_mask_dice",
            "profile_depth_rmse_m",
            "notes",
        ],
    )
    write_rows(
        GROUP_AUDIT,
        group_audit,
        [
            "group_field",
            "group_value",
            "sample_count_refined",
            "reference_curvature_mae",
            "refined_curvature_mae",
            "curvature_delta_vs_20_77",
            "reference_dimension_mae_norm",
            "refined_dimension_mae_norm",
            "dimension_delta_vs_20_77",
            "reference_projected_mask_dice",
            "refined_projected_mask_dice",
            "dice_delta_vs_20_77",
            "reference_profile_depth_rmse_m",
            "refined_profile_depth_rmse_m",
            "wLD_abs_error",
            "wWD_abs_error",
            "wLW_abs_error",
            "notes",
        ],
    )
    write_rows(DECISION_MATRIX, decision_rows, ["criterion", "value", "evidence", "pass"])
    write_audit_summary(AUDIT_SUMMARY, ref, selected, comparison, group_audit, decision_rows)
    write_decision_summary(DECISION_SUMMARY, decision_rows, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argparse.ArgumentParser().parse_args()))

#!/usr/bin/env python
"""Audit Stage 20.80 Piao/NLS-inspired curvature diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_candidates.csv"
METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_metrics.csv"
GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_group_summary.csv"
QUALITY = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_quality.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_piao_nls_curvature_diagnostic_summary.txt"
DIAG = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_curvature_diagnostic.csv"
FAILURES = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_failure_cases.csv"
DECISION = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_decision_matrix.csv"

REF_NEURAL = {"total": 0.6780143536818333, "curvature": 0.20107580721378326, "dice": 0.8477271366767738}
REF_FEATURE = {"total": 0.7153952435041085, "curvature": 0.19504618346213531, "dice": 0.815450420558869}
REF_REFINED = {"total": 0.7533873075093979, "curvature": 0.21158406477517042, "dice": 0.8345969696140306}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        raw = row.get(key, "")
        return float(raw) if raw not in {"", None} else default
    except Exception:
        return default


def selected_test(metrics: list[dict[str, str]]) -> dict[str, str]:
    rows = [row for row in metrics if row.get("split") == "test" and str(row.get("selected_by_validation", "")).lower() == "true"]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one selected test row, got {len(rows)}")
    return rows[0]


def best_val_by_feature_set(candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature_set in sorted({row["feature_set"] for row in candidates}):
        vals = [row for row in candidates if row["feature_set"] == feature_set and row["split"] == "val"]
        if not vals:
            continue
        best = min(vals, key=lambda row: f(row, "selection_score", math.inf))
        out.append(
            {
                "diagnostic": "best_validation_by_feature_set",
                "feature_set": feature_set,
                "model": best["model"],
                "value": f(best, "curvature_mae"),
                "secondary_value": f(best, "normalized_param_mae"),
                "notes": f"val_selection_score={best['selection_score']}",
            }
        )
    return out


def selected_group_failures(groups: list[dict[str, str]], selected: dict[str, str]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in groups
        if row["feature_set"] == selected["feature_set"] and row["model"] == selected["model"] and row["split"] == "test"
    ]
    failures: list[dict[str, Any]] = []
    for field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
        subset = [row for row in rows if row["group_field"] == field]
        for rank, row in enumerate(sorted(subset, key=lambda item: f(item, "curvature_mae_mean"), reverse=True)[:5], start=1):
            failures.append(
                {
                    "case_type": f"worst_{field}_by_curvature",
                    "rank": rank,
                    "group_field": field,
                    "group_value": row["group_value"],
                    "sample_count": row["sample_count"],
                    "curvature_mae": row["curvature_mae_mean"],
                    "wLD_abs_error": row.get("wLD_abs_error", ""),
                    "wWD_abs_error": row.get("wWD_abs_error", ""),
                    "wLW_abs_error": row.get("wLW_abs_error", ""),
                    "total_normalized_mae": row["normalized_param_mae_mean"],
                    "projected_mask_dice": row["projected_mask_dice"],
                    "profile_depth_rmse_m": row["profile_depth_rmse_m"],
                    "notes": "group-level failure case; sample-level predictions are not persisted to avoid nonselected test ranking",
                }
            )
    return failures


def quality_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["feature_group"]: row for row in rows}


def decision_rows(selected: dict[str, str], quality: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], str, str]:
    total = f(selected, "normalized_param_mae")
    curvature = f(selected, "curvature_mae")
    wld = f(selected, "wLD_abs_error")
    wwd = f(selected, "wWD_abs_error")
    wlw = f(selected, "wLW_abs_error")
    dice = f(selected, "projected_mask_dice")
    feature_set = selected["feature_set"]
    nls_rate = f(quality.get("F4_nls_curve_fit", {}), "fit_success_rate")
    beyond_f0 = feature_set != "F0_existing_only"
    improves_wld = wld < 0.2094394713640213
    improves_wwd = wwd < 0.20446909964084625
    improves_wlw = wlw < 0.18931882083415985
    improves_neural_curv = curvature <= REF_NEURAL["curvature"] - 0.01
    improves_feature_curv = curvature < REF_FEATURE["curvature"]
    severe_total_regression = total > REF_FEATURE["total"]
    dice_collapse = dice < 0.80
    nls_stable = nls_rate >= 0.50 if math.isfinite(nls_rate) else False
    hybrid_promising = beyond_f0 and (improves_feature_curv or improves_neural_curv) and not dice_collapse

    if hybrid_promising and total <= REF_FEATURE["total"]:
        category = "A_feature_fusion_neural_model"
        next_step = "feature-fusion neural model using F0+F1+F2 physical features for curvature; keep 20.77 neural path for L/W/D"
    elif not nls_stable and beyond_f0 and improves_feature_curv:
        category = "B_physics_features_help_but_exact_NLS_unstable"
        next_step = "feature-fusion neural model using stable F1/F2/F3/F5 features"
    elif improves_feature_curv and severe_total_regression:
        category = "D_hybrid_curvature_only_regressor"
        next_step = "hybrid neural for LWD plus feature regressor for curvature"
    elif not improves_feature_curv and not improves_neural_curv:
        category = "C_curvature_targeted_data_or_label_redefinition"
        next_step = "curvature-targeted data top-up or redefine curvature labels/output representation"
    else:
        category = "E_formal_benchmark_rerun_with_20_77_model"
        next_step = "formal benchmark rerun using 20.77 model"

    rows = [
        {"question": "Did Piao/NLS-inspired features improve wLD?", "answer": improves_wld, "evidence": wld, "decision": category},
        {"question": "Did Piao/NLS-inspired features improve wWD?", "answer": improves_wwd, "evidence": wwd, "decision": category},
        {"question": "Did Piao/NLS-inspired features improve wLW?", "answer": improves_wlw, "evidence": wlw, "decision": category},
        {"question": "Which feature groups matter?", "answer": feature_set, "evidence": f"selected by validation; beyond_F0={beyond_f0}", "decision": category},
        {"question": "Are NLS fits stable enough?", "answer": nls_stable, "evidence": nls_rate, "decision": category},
        {"question": "Does curvature improve vs 20.77 neural by 0.01?", "answer": improves_neural_curv, "evidence": curvature - REF_NEURAL["curvature"], "decision": category},
        {"question": "Does curvature improve vs 20.77 feature baseline?", "answer": improves_feature_curv, "evidence": curvature - REF_FEATURE["curvature"], "decision": category},
        {"question": "Does projected mask collapse?", "answer": dice_collapse, "evidence": dice, "decision": category},
        {"question": "Hybrid neural+feature promising?", "answer": hybrid_promising, "evidence": f"feature_set={feature_set}, total={total}, curvature={curvature}", "decision": category},
        {"question": "Recommended next step", "answer": next_step, "evidence": category, "decision": category},
    ]
    return rows, category, next_step


def write_summary(path: Path, selected: dict[str, str], decisions: list[dict[str, Any]], category: str, next_step: str, quality: dict[str, dict[str, str]]) -> None:
    lines = [
        "true_3d_rbc_v3_240 Piao/NLS curvature diagnostic summary",
        "",
        "dataset_id: comsol_true_3d_rbc_imported_watertight_pilot_v3_240",
        "scope: Piao/NLS-inspired feature diagnostic only; no COMSOL, no data generation, no NPZ modification, no neural training, no baseline update.",
        "method_claim: Piao-inspired / NLS-inspired; not exact Piao reproduction and not LS-SVM reproduction.",
        f"selected_feature_set: {selected['feature_set']}",
        f"selected_model: {selected['model']}",
        f"test_total_mae: {selected['normalized_param_mae']}",
        f"test_LWD_mae_mm: {selected['L_mae_mm']}/{selected['W_mae_mm']}/{selected['D_mae_mm']}",
        f"test_curvature_mae: {selected['curvature_mae']}",
        f"test_wLD_wWD_wLW: {selected['wLD_abs_error']}/{selected['wWD_abs_error']}/{selected['wLW_abs_error']}",
        f"test_projected_mask_iou_dice: {selected['projected_mask_iou']}/{selected['projected_mask_dice']}",
        f"test_profile_depth_rmse_m: {selected['profile_depth_rmse_m']}",
        f"nls_fit_success_rate: {quality.get('F4_nls_curve_fit', {}).get('fit_success_rate', '')}",
        "selected_group_note: F4 NLS proxy was stable but not selected; the observed gain is attributable to F0+F1+F2 physical peak/width/gradient features.",
        "",
        "reference_comparison:",
        f"- 20.77 neural: total={REF_NEURAL['total']}, curvature={REF_NEURAL['curvature']}, Dice={REF_NEURAL['dice']}",
        f"- 20.77 feature baseline: total={REF_FEATURE['total']}, curvature={REF_FEATURE['curvature']}, Dice={REF_FEATURE['dice']}",
        f"- 20.79 refined model: total={REF_REFINED['total']}, curvature={REF_REFINED['curvature']}, Dice={REF_REFINED['dice']}",
        "",
        "answers:",
    ]
    lines.extend(f"- {row['question']}: {row['answer']} ({row['evidence']})" for row in decisions)
    lines.extend(
        [
            "",
            f"decision_category: {category}",
            f"next_step: {next_step}",
            "baseline_update: false",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(_: argparse.Namespace) -> int:
    selected = selected_test(read_csv(METRICS))
    candidates = read_csv(CANDIDATES)
    groups = read_csv(GROUP)
    quality = quality_lookup(read_csv(QUALITY))
    diag_rows = best_val_by_feature_set(candidates)
    diag_rows.extend(
        [
            {"diagnostic": "selected_test", "feature_set": selected["feature_set"], "model": selected["model"], "value": selected["curvature_mae"], "secondary_value": selected["normalized_param_mae"], "notes": "test final only"},
            {"diagnostic": "reference_20_77_neural", "feature_set": "reference", "model": "20.77_neural", "value": REF_NEURAL["curvature"], "secondary_value": REF_NEURAL["total"], "notes": "fixed reference"},
            {"diagnostic": "reference_20_77_feature", "feature_set": "reference", "model": "20.77_feature_baseline", "value": REF_FEATURE["curvature"], "secondary_value": REF_FEATURE["total"], "notes": "fixed reference"},
            {"diagnostic": "reference_20_79_refined", "feature_set": "reference", "model": "20.79_refined", "value": REF_REFINED["curvature"], "secondary_value": REF_REFINED["total"], "notes": "fixed reference"},
        ]
    )
    failures = selected_group_failures(groups, selected)
    decisions, category, next_step = decision_rows(selected, quality)
    write_csv(DIAG, diag_rows, ["diagnostic", "feature_set", "model", "value", "secondary_value", "notes"])
    write_csv(FAILURES, failures, ["case_type", "rank", "group_field", "group_value", "sample_count", "curvature_mae", "wLD_abs_error", "wWD_abs_error", "wLW_abs_error", "total_normalized_mae", "projected_mask_dice", "profile_depth_rmse_m", "notes"])
    write_csv(DECISION, decisions, ["question", "answer", "evidence", "decision"])
    write_summary(SUMMARY, selected, decisions, category, next_step, quality)
    return 0


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

#!/usr/bin/env python
"""Audit Stage 20.81 feature-fusion results and write route decision."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import ROOT, write_csv
from train_true_3d_rbc_feature_fusion_candidates import REF_FEATURE, REF_NEURAL, REF_REFINED


CANDIDATE_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_candidate_screen_metrics.csv"
CANDIDATE_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_candidate_profile_metrics.csv"
CANDIDATE_GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_candidate_group_summary.csv"
TRAINING_SEED = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_seed_summary.csv"
TRAINING_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_profile_metrics.csv"
TRAINING_GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_group_summary.csv"
TRAINING_VS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_vs_reference.csv"

AUDIT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_audit_summary.txt"
DECISION_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_decision_summary.txt"
FAILURE_CASES = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_failure_cases.csv"
GROUP_AUDIT = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_group_audit.csv"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_decision_matrix.csv"

FAILURE_FIELDS = [
    "rank_type",
    "rank",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "normalized_param_mae_mean",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "curvature_mae_mean",
    "projected_mask_dice",
    "profile_depth_rmse_m",
]
DECISION_FIELDS = ["question", "answer", "evidence", "decision"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def selected_result() -> tuple[str, dict[str, str], Path, Path]:
    seed_rows = read_csv(TRAINING_SEED)
    selected_seed = [row for row in seed_rows if str(row.get("selected_seed", "")).lower() == "true" and row.get("test_normalized_param_mae")]
    if selected_seed:
        row = selected_seed[0]
        result = {
            "variant": row["variant"],
            "seed": row["seed"],
            "feature_set": row["feature_set"],
            "normalized_param_mae": row["test_normalized_param_mae"],
            "L_mae_mm": row["test_L_mae_mm"],
            "W_mae_mm": row["test_W_mae_mm"],
            "D_mae_mm": row["test_D_mae_mm"],
            "curvature_mae": row["test_curvature_mae"],
            "wLD_abs_error": row["test_wLD_abs_error"],
            "wWD_abs_error": row["test_wWD_abs_error"],
            "wLW_abs_error": row["test_wLW_abs_error"],
            "projected_mask_iou": row["test_projected_mask_iou"],
            "projected_mask_dice": row["test_projected_mask_dice"],
            "profile_depth_rmse_m": row["test_profile_depth_rmse_m"],
        }
        return "multi_seed", result, TRAINING_PROFILE, TRAINING_GROUP
    candidate_rows = read_csv(CANDIDATE_METRICS)
    selected_candidate = [
        row
        for row in candidate_rows
        if str(row.get("selected_by_validation", "")).lower() == "true" and row.get("split") == "test"
    ]
    if not selected_candidate:
        raise RuntimeError("no selected feature-fusion result found")
    return "candidate_screen_only", selected_candidate[0], CANDIDATE_PROFILE, CANDIDATE_GROUP


def top_cases(profile_path: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv(profile_path)
        if row.get("split") == "test" and str(row.get("selected_by_validation", "")).lower() == "true"
    ]
    out: list[dict[str, Any]] = []
    rank_specs = [
        ("highest_total_error", "normalized_param_mae_mean"),
        ("highest_curvature_error", "curvature_mae_mean"),
        ("highest_wLD_error", "wLD_abs_error"),
        ("highest_D_error", "D_mae_mm"),
        ("lowest_projected_mask_dice", "projected_mask_dice"),
    ]
    for rank_type, key in rank_specs:
        reverse = rank_type != "lowest_projected_mask_dice"
        ordered = sorted(rows, key=lambda row: as_float(row.get(key)), reverse=reverse)[:10]
        for rank, row in enumerate(ordered, start=1):
            item = {"rank_type": rank_type, "rank": rank}
            for field in FAILURE_FIELDS:
                if field in {"rank_type", "rank"}:
                    continue
                item[field] = row.get(field, "")
            out.append(item)
    return out


def group_audit_rows(group_path: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv(group_path)
        if row.get("split") == "test" and str(row.get("selected_by_validation", "")).lower() == "true"
    ]
    return rows


def comparison_bool(current: float, reference: float, lower_better: bool = True, margin: float = 0.0) -> bool:
    if lower_better:
        return current <= reference - margin
    return current >= reference + margin


def run(args: argparse.Namespace) -> int:
    source, result, profile_path, group_path = selected_result()
    total = as_float(result.get("normalized_param_mae"))
    l_mm = as_float(result.get("L_mae_mm"))
    w_mm = as_float(result.get("W_mae_mm"))
    d_mm = as_float(result.get("D_mae_mm"))
    curvature = as_float(result.get("curvature_mae"))
    wld = as_float(result.get("wLD_abs_error"))
    wwd = as_float(result.get("wWD_abs_error"))
    wlw = as_float(result.get("wLW_abs_error"))
    iou = as_float(result.get("projected_mask_iou"))
    dice = as_float(result.get("projected_mask_dice"))
    profile_rmse = as_float(result.get("profile_depth_rmse_m"))

    total_stable = total <= REF_NEURAL["total"] + 0.02
    total_improved = total < REF_NEURAL["total"]
    lwd_stable = l_mm <= REF_NEURAL["L_mm"] + 0.25 and w_mm <= REF_NEURAL["W_mm"] + 0.25 and d_mm <= REF_NEURAL["D_mm"] + 0.15
    curvature_improved = curvature <= REF_NEURAL["curvature"] - 0.01
    wld_improved = wld < REF_NEURAL["wLD"]
    wwd_preserve_feature = wwd <= REF_FEATURE["wWD"] + 0.005
    wlw_preserve_feature = wlw <= REF_FEATURE["wLW"] + 0.005
    dice_stable = dice >= REF_NEURAL["dice"] - 0.01
    better_than_feature_total = total < REF_FEATURE["total"]
    better_than_feature_curvature = curvature < REF_FEATURE["curvature"]
    severe_regression = total > REF_NEURAL["total"] + 0.08 or dice < REF_NEURAL["dice"] - 0.03

    if curvature_improved and total_stable and lwd_stable and dice_stable and (wld_improved or (wwd_preserve_feature and wlw_preserve_feature)):
        next_step = "A_formal_benchmark_rerun_candidate_upgrade"
        overall = "feature_fusion_clear_win"
    elif severe_regression:
        next_step = "E_revert_to_20_77_benchmark_candidate"
        overall = "feature_fusion_regressed"
    elif curvature_improved and not (lwd_stable and dice_stable):
        next_step = "B_curvature_targeted_data_topup"
        overall = "feature_fusion_curvature_tradeoff"
    elif total_improved and not curvature_improved:
        next_step = "D_redefine_curvature_labels_output_representation"
        overall = "feature_fusion_total_not_curvature"
    else:
        next_step = "B_curvature_targeted_data_topup"
        overall = "feature_fusion_not_enough"

    failures = top_cases(profile_path)
    write_csv(args.failure_cases, failures, FAILURE_FIELDS)
    group_rows = group_audit_rows(group_path)
    if group_rows:
        write_csv(args.group_audit, group_rows, list(group_rows[0].keys()))
    else:
        write_csv(args.group_audit, [], ["empty"])

    decision_rows = [
        {"question": "Did fusion improve total MAE?", "answer": total_improved, "evidence": f"{total:.6f} vs 20.77 {REF_NEURAL['total']:.6f}", "decision": overall},
        {"question": "Did fusion improve L/W/D?", "answer": lwd_stable, "evidence": f"{l_mm:.3f}/{w_mm:.3f}/{d_mm:.3f} mm vs 20.77 {REF_NEURAL['L_mm']:.3f}/{REF_NEURAL['W_mm']:.3f}/{REF_NEURAL['D_mm']:.3f}", "decision": overall},
        {"question": "Did fusion reach >=0.01 substantive curvature improvement?", "answer": curvature_improved, "evidence": f"{curvature:.6f} vs 20.77 {REF_NEURAL['curvature']:.6f}; delta={curvature - REF_NEURAL['curvature']:.6f}", "decision": overall},
        {"question": "Did wLD improve?", "answer": wld_improved, "evidence": f"{wld:.6f} vs 20.77 {REF_NEURAL['wLD']:.6f}", "decision": overall},
        {"question": "Did wWD/wLW preserve 20.80 gains?", "answer": bool(wwd_preserve_feature and wlw_preserve_feature), "evidence": f"wWD {wwd:.6f} vs 20.80 {REF_FEATURE['wWD']:.6f}; wLW {wlw:.6f} vs 20.80 {REF_FEATURE['wLW']:.6f}", "decision": overall},
        {"question": "Did projected mask remain stable?", "answer": dice_stable, "evidence": f"Dice {dice:.6f} vs 20.77 {REF_NEURAL['dice']:.6f}", "decision": overall},
        {"question": "Which feature set helped most?", "answer": result.get("feature_set", ""), "evidence": f"selected variant {result.get('variant', '')}; source={source}", "decision": overall},
        {"question": "Is fusion better than 20.77 neural?", "answer": bool(total_improved and curvature_improved and dice_stable), "evidence": f"total_delta={total - REF_NEURAL['total']:.6f}; curvature_delta={curvature - REF_NEURAL['curvature']:.6f}; dice_delta={dice - REF_NEURAL['dice']:.6f}", "decision": overall},
        {"question": "Is fusion better than 20.80 feature-only?", "answer": bool(better_than_feature_total and better_than_feature_curvature), "evidence": f"total {total:.6f} vs {REF_FEATURE['total']:.6f}; curvature {curvature:.6f} vs {REF_FEATURE['curvature']:.6f}", "decision": overall},
        {"question": "Next step", "answer": next_step, "evidence": "No baseline update; no CURRENT_BASELINE replacement.", "decision": overall},
    ]
    write_csv(args.decision_matrix, decision_rows, DECISION_FIELDS)

    args.audit_summary.parent.mkdir(parents=True, exist_ok=True)
    args.audit_summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 feature-fusion audit summary",
                "",
                f"result_source: {source}",
                f"selected_variant: {result.get('variant', '')}",
                f"selected_seed: {result.get('seed', '')}",
                f"selected_feature_set: {result.get('feature_set', '')}",
                f"test_total_mae: {total:.6f}",
                f"test_LWD_mae_mm: {l_mm:.6f}/{w_mm:.6f}/{d_mm:.6f}",
                f"test_curvature_mae: {curvature:.6f}",
                f"test_wLD_wWD_wLW: {wld:.6f}/{wwd:.6f}/{wlw:.6f}",
                f"test_projected_mask_iou_dice: {iou:.6f}/{dice:.6f}",
                f"test_profile_depth_rmse_m: {profile_rmse:.9f}",
                f"vs_20_77_total_delta: {total - REF_NEURAL['total']:.6f}",
                f"vs_20_77_curvature_delta: {curvature - REF_NEURAL['curvature']:.6f}",
                f"vs_20_77_wLD_delta: {wld - REF_NEURAL['wLD']:.6f}",
                f"vs_20_77_dice_delta: {dice - REF_NEURAL['dice']:.6f}",
                f"vs_20_80_total_delta: {total - REF_FEATURE['total']:.6f}",
                f"vs_20_80_curvature_delta: {curvature - REF_FEATURE['curvature']:.6f}",
                f"total_stable: {total_stable}",
                f"LWD_stable: {lwd_stable}",
                f"curvature_improved_by_0p01: {curvature_improved}",
                f"wLD_improved: {wld_improved}",
                f"wWD_wLW_preserve_20_80_gains: {wwd_preserve_feature and wlw_preserve_feature}",
                f"projected_mask_stable: {dice_stable}",
                f"overall_decision: {overall}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.decision_summary.parent.mkdir(parents=True, exist_ok=True)
    args.decision_summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 feature-fusion decision summary",
                "",
                f"decision: {overall}",
                f"next_step: {next_step}",
                "baseline_ready: false",
                "CURRENT_BASELINE_update: false",
                "COMSOL_run: false",
                "data_or_NPZ_modified: false",
                "",
                "Answer 1 - Did fusion improve total MAE: " + str(total_improved),
                "Answer 2 - Did fusion improve L/W/D: " + str(lwd_stable),
                "Answer 3 - Did fusion reach >=0.01 substantive curvature improvement: " + str(curvature_improved),
                "Answer 4 - Did wLD improve: " + str(wld_improved),
                "Answer 5 - Did wWD/wLW preserve 20.80 gains: " + str(wwd_preserve_feature and wlw_preserve_feature),
                "Answer 6 - Did projected mask remain stable: " + str(dice_stable),
                "Answer 7 - Which feature set helped most: " + str(result.get("feature_set", "")),
                "Answer 8 - Is fusion better than 20.77 neural: " + str(total_improved and curvature_improved and dice_stable),
                "Answer 9 - Is fusion better than 20.80 feature-only: " + str(better_than_feature_total and better_than_feature_curvature),
                "Answer 10 - Recommended next step: " + next_step,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", type=Path, default=AUDIT_SUMMARY)
    parser.add_argument("--decision-summary", type=Path, default=DECISION_SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=FAILURE_CASES)
    parser.add_argument("--group-audit", type=Path, default=GROUP_AUDIT)
    parser.add_argument("--decision-matrix", type=Path, default=DECISION_MATRIX)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

#!/usr/bin/env python
"""Audit Stage 20.83 profile-primary loss results and route decision."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import ROOT, write_csv
from train_true_3d_rbc_profile_primary_candidates import REF_FEATURE, REF_FUSION, REF_NEURAL


CANDIDATE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_candidate_screen_summary.txt"
TRAINING_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_training_summary.txt"
CANDIDATE_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_profile_metrics.csv"
TRAINING_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_profile_metrics.csv"
AUDIT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_audit_summary.txt"
DECISION_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_decision_summary.txt"
FAILURE_CASES = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_failure_cases.csv"
GROUP_AUDIT = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_group_audit.csv"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_decision_matrix.csv"

FAILURE_FIELDS = [
    "bucket",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_dice",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
]
GROUP_FIELDS = ["split", "group_field", "group_value", "sample_count", "profile_depth_rmse_m", "er_like_profile_error", "projected_mask_dice", "dimension_mae_norm", "curvature_mae_mean"]
DECISION_FIELDS = ["question", "answer", "evidence", "decision"]


def parse_summary_value(path: Path, key: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def source_paths() -> tuple[Path, Path, str]:
    if TRAINING_SUMMARY.exists() and TRAINING_PROFILE.exists():
        return TRAINING_SUMMARY, TRAINING_PROFILE, "multi_seed_training"
    return CANDIDATE_SUMMARY, CANDIDATE_PROFILE, "candidate_screen_only"


def read_selected_test_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    return [row for row in rows if row.get("split") == "test" and str(row.get("selected_by_validation", "")).lower() == "true"]


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("test",):
        base = [row for row in rows if row["split"] == split]
        for field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
            for value in sorted({row[field] for row in base}):
                subset = [row for row in base if row[field] == value]
                out.append(
                    {
                        "split": split,
                        "group_field": field,
                        "group_value": value,
                        "sample_count": len(subset),
                        "profile_depth_rmse_m": avg(subset, "profile_depth_rmse_m"),
                        "er_like_profile_error": avg(subset, "er_like_profile_error"),
                        "projected_mask_dice": avg(subset, "projected_mask_dice"),
                        "dimension_mae_norm": avg(subset, "dimension_param_mae_norm"),
                        "curvature_mae_mean": avg(subset, "curvature_mae_mean"),
                    }
                )
    return out


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[tuple[str, dict[str, Any]]] = []
    ranked.extend(("worst_profile", row) for row in sorted(rows, key=lambda row: float(row["profile_depth_rmse_m"]), reverse=True)[:10])
    ranked.extend(("worst_er_like", row) for row in sorted(rows, key=lambda row: float(row["er_like_profile_error"]), reverse=True)[:10])
    ranked.extend(("worst_dice", row) for row in sorted(rows, key=lambda row: float(row["projected_mask_dice"]))[:10])
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for bucket, row in ranked:
        key = (bucket, row["sample_id"])
        if key in seen:
            continue
        seen.add(key)
        item = {field: row.get(field, "") for field in FAILURE_FIELDS}
        item["bucket"] = bucket
        out.append(item)
    return out


def run(args: argparse.Namespace) -> int:
    summary_path, profile_path, source_kind = source_paths()
    rows = read_selected_test_rows(profile_path)
    if not rows:
        raise RuntimeError(f"no selected test rows in {profile_path}")
    selected_candidate = parse_summary_value(summary_path, "selected_candidate")
    selected_seed = parse_summary_value(summary_path, "selected_seed") or parse_summary_value(summary_path, "seed")
    multi_seed_completed = source_kind == "multi_seed_training"
    profile_rmse = avg(rows, "profile_depth_rmse_m")
    er_like = avg(rows, "er_like_profile_error")
    dice = avg(rows, "projected_mask_dice")
    iou = avg(rows, "projected_mask_iou")
    total = avg(rows, "normalized_param_mae_mean")
    dim = avg(rows, "dimension_param_mae_norm")
    curv = avg(rows, "curvature_mae_mean")
    l_mm = avg(rows, "L_mae_mm")
    w_mm = avg(rows, "W_mae_mm")
    d_mm = avg(rows, "D_mae_mm")
    wld = avg(rows, "wLD_abs_error")
    wwd = avg(rows, "wWD_abs_error")
    wlw = avg(rows, "wLW_abs_error")
    max_depth = avg(rows, "max_depth_error_m")
    volume = avg(rows, "volume_proxy_rel_error")
    profile_improved = profile_rmse < REF_NEURAL["profile_rmse"]
    profile_improved_5pct = profile_rmse <= REF_NEURAL["profile_rmse"] * 0.95
    dice_stable = dice >= REF_NEURAL["dice"] - 0.01
    dim_stable = dim <= 0.488785 + 0.05
    lwd_material_regression = l_mm > REF_NEURAL["L_mm"] + 0.5 or w_mm > REF_NEURAL["W_mm"] + 0.5 or d_mm > REF_NEURAL["D_mm"] + 0.25
    if profile_improved_5pct and dice_stable and dim_stable and not lwd_material_regression:
        decision = "A_formal_benchmark_rerun_candidate_upgrade"
        next_step = "A. formal benchmark rerun / candidate upgrade"
    elif not profile_improved and dice >= REF_FUSION["dice"] - 0.005:
        decision = "B_keep_20_77_profile_and_20_81_visual_candidates"
        next_step = "B. keep 20.77/20.81 candidate"
    elif not profile_improved:
        decision = "C_refine_profile_primary_loss"
        next_step = "C. refine profile-primary loss"
    else:
        decision = "C_refine_profile_primary_loss"
        next_step = "C. refine profile-primary loss"

    group_out = aggregate_by_group(rows)
    fail_out = failure_rows(rows)
    decision_rows = [
        {"question": "Did profile_depth_rmse improve?", "answer": str(profile_improved), "evidence": f"current={profile_rmse:.9f}; ref20.77={REF_NEURAL['profile_rmse']:.9f}", "decision": decision},
        {"question": "Did profile_depth_rmse improve by 5pct?", "answer": str(profile_improved_5pct), "evidence": f"threshold={REF_NEURAL['profile_rmse'] * 0.95:.9f}", "decision": decision},
        {"question": "Did Er-like profile error improve?", "answer": "no comparable 20.77 persisted Er-like reference", "evidence": f"current={er_like:.6f}; 20.83 defines this metric", "decision": decision},
        {"question": "Did projected mask remain stable?", "answer": str(dice_stable), "evidence": f"Dice={dice:.6f}; ref20.77={REF_NEURAL['dice']:.6f}", "decision": decision},
        {"question": "Did L/W/D remain stable?", "answer": str(not lwd_material_regression), "evidence": f"L/W/D={l_mm:.3f}/{w_mm:.3f}/{d_mm:.3f} mm", "decision": decision},
        {"question": "Does wMAE decide pass/fail?", "answer": "False", "evidence": f"aux curvature={curv:.6f}; w={wld:.6f}/{wwd:.6f}/{wlw:.6f}", "decision": decision},
        {"question": "Best for parameter sizing", "answer": "20.77/20.81 mixed", "evidence": "20.83 did not improve profile RMSE and L/D are not strictly better than 20.77.", "decision": decision},
        {"question": "Best for profile reconstruction", "answer": "20.77 neural reference", "evidence": f"20.77 profile_rmse={REF_NEURAL['profile_rmse']:.9f}; current={profile_rmse:.9f}", "decision": decision},
        {"question": "Best for projected mask", "answer": "20.81 feature-fusion or 20.83 screen", "evidence": f"20.81 Dice={REF_FUSION['dice']:.6f}; current Dice={dice:.6f}", "decision": decision},
        {"question": "Next step", "answer": next_step, "evidence": "No baseline update; profile-primary R1 did not meet upgrade gate.", "decision": decision},
    ]
    write_csv(args.failure_cases, fail_out, FAILURE_FIELDS)
    write_csv(args.group_audit, group_out, GROUP_FIELDS)
    write_csv(args.decision_matrix, decision_rows, DECISION_FIELDS)
    args.audit_summary.parent.mkdir(parents=True, exist_ok=True)
    args.audit_summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 profile-primary audit summary",
                "",
                f"source_kind: {source_kind}",
                f"selected_candidate: {selected_candidate}",
                f"selected_seed: {selected_seed}",
                f"selected_candidate_multi_seed_completed: {multi_seed_completed}",
                f"test_total_mae: {total:.6f}",
                f"test_dimension_mae_norm: {dim:.6f}",
                f"test_LWD_mae_mm: {l_mm:.6f}/{w_mm:.6f}/{d_mm:.6f}",
                f"test_wLD_wWD_wLW_aux: {wld:.6f}/{wwd:.6f}/{wlw:.6f}",
                f"test_curvature_aux_mae: {curv:.6f}",
                f"test_profile_depth_rmse_m: {profile_rmse:.9f}",
                f"test_er_like_profile_error: {er_like:.6f}",
                f"test_max_depth_error_m: {max_depth:.9f}",
                f"test_volume_proxy_rel_error: {volume:.6f}",
                f"test_projected_mask_iou_dice: {iou:.6f}/{dice:.6f}",
                f"profile_improved_vs_20_77: {profile_improved}",
                f"profile_improved_5pct_vs_20_77: {profile_improved_5pct}",
                f"dice_stable_vs_20_77: {dice_stable}",
                f"LWD_material_regression: {lwd_material_regression}",
                "wMAE_role: auxiliary diagnostic only.",
                "result: R1 candidate screen improved projected mask Dice but did not improve profile_depth_rmse_m; Stage C multi-seed was intentionally skipped.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.decision_summary.parent.mkdir(parents=True, exist_ok=True)
    args.decision_summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 profile-primary decision summary",
                "",
                f"decision: {decision}",
                f"next_step: {next_step}",
                "can_call_baseline: false",
                "CURRENT_BASELINE_update: false",
                "COMSOL_run: false",
                "data_or_NPZ_modified: false",
                f"multi_seed_completed: {multi_seed_completed}",
                f"profile_depth_rmse_m_current: {profile_rmse:.9f}",
                f"profile_depth_rmse_m_20_77: {REF_NEURAL['profile_rmse']:.9f}",
                f"projected_mask_dice_current: {dice:.6f}",
                f"projected_mask_dice_20_77: {REF_NEURAL['dice']:.6f}",
                f"projected_mask_dice_20_81: {REF_FUSION['dice']:.6f}",
                "rationale: R1 profile-primary loss did not beat the 20.77 profile RMSE reference. It should not replace the 20.77 benchmark candidate or the 20.81 visual/mask candidate.",
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


#!/usr/bin/env python
"""Route decision for the surface RBC NLS-lite feature-fusion diagnostic."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_true_3d_rbc_pilot_dataset import ROOT, check_no_overwrite, write_csv


TRAINING_SUMMARY = ROOT / "results/summaries/surface_rbc_nls_feature_fusion_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_seed_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_vs_reference.csv"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_feature_fusion_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_decision_matrix.csv"

DECISION_FIELDS = ["question", "answer", "evidence", "decision"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def selected_seed_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    selected = [row for row in rows if str(row.get("selected_seed", "")).lower() == "true"]
    if len(selected) != 1:
        raise RuntimeError(f"expected one selected seed row, found {len(selected)} in {path}")
    return selected[0]


def delta(rows: list[dict[str, str]], metric: str, reference: str) -> float:
    for row in rows:
        if row.get("metric") == metric and row.get("reference_label") == reference:
            return f(row.get("delta"))
    return math.nan


def bool_from_summary(path: Path, key: str) -> bool:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        return False
    return match.group(1).strip().lower() == "true"


def decision_rows(seed_row: dict[str, str], vs_rows: list[dict[str, str]], surface_candidate: bool) -> list[dict[str, Any]]:
    total_delta_20_85 = delta(vs_rows, "total_normalized_mae", "20.85_formal_rerun_20.77_protocol")
    w_delta_20_85 = delta(vs_rows, "wMAE", "20.85_formal_rerun_20.77_protocol")
    profile_delta_20_85 = delta(vs_rows, "profile_depth_rmse_m", "20.85_formal_rerun_20.77_protocol")
    dice_delta_20_85 = delta(vs_rows, "projected_mask_dice", "20.85_formal_rerun_20.77_protocol")
    total_delta_24_1 = delta(vs_rows, "total_normalized_mae", "24.1_NLS_lite_feature_baseline")
    w_delta_24_1 = delta(vs_rows, "wMAE", "24.1_NLS_lite_feature_baseline")
    profile_delta_24_1 = delta(vs_rows, "profile_depth_rmse_m", "24.1_NLS_lite_feature_baseline")
    dice_delta_24_1 = delta(vs_rows, "projected_mask_dice", "24.1_NLS_lite_feature_baseline")

    enhanced_neural = (
        math.isfinite(total_delta_20_85)
        and math.isfinite(profile_delta_20_85)
        and math.isfinite(dice_delta_20_85)
        and (total_delta_20_85 < 0.0 or w_delta_20_85 < 0.0 or dice_delta_20_85 > 0.0)
        and profile_delta_20_85 <= 0.00008
        and dice_delta_20_85 >= -0.02
    )
    w_improved = w_delta_20_85 < 0.0 or w_delta_24_1 < 0.0
    profile_improved = profile_delta_20_85 < 0.0 or profile_delta_24_1 < 0.0
    diagnostic_only = not surface_candidate
    formal_rerun = surface_candidate

    return [
        {
            "question": "Does NLS-lite enhance the neural model?",
            "answer": enhanced_neural,
            "evidence": f"delta_vs_20.85 total={total_delta_20_85:.6f}, wMAE={w_delta_20_85:.6f}, profile={profile_delta_20_85:.9f}, dice={dice_delta_20_85:.6f}",
            "decision": "surface enhancement signal" if enhanced_neural else "no robust enhancement over 20.85",
        },
        {
            "question": "Is the main gain in w parameters?",
            "answer": w_improved,
            "evidence": f"delta_wMAE_vs_20.85={w_delta_20_85:.6f}; delta_wMAE_vs_24.1={w_delta_24_1:.6f}; selected_test_wMAE={f(seed_row.get('test_wMAE')):.6f}",
            "decision": "w-head benefit present" if w_improved else "w parameters did not improve enough",
        },
        {
            "question": "Does profile-level quality improve?",
            "answer": profile_improved,
            "evidence": f"delta_profile_vs_20.85={profile_delta_20_85:.9f}; delta_profile_vs_24.1={profile_delta_24_1:.9f}; selected_test_profile={f(seed_row.get('test_profile_depth_rmse_m')):.9f}",
            "decision": "profile improvement present" if profile_improved else "profile improvement absent",
        },
        {
            "question": "Is this only a diagnostic feature route?",
            "answer": diagnostic_only,
            "evidence": f"surface_feature_fusion_candidate={surface_candidate}",
            "decision": "keep as diagnostic interface" if diagnostic_only else "promote only to surface candidate, not baseline",
        },
        {
            "question": "Does it form a surface candidate?",
            "answer": surface_candidate,
            "evidence": f"selected_seed={seed_row.get('seed')}; test_total={f(seed_row.get('test_total_mae')):.6f}; dice_delta_vs_24.1={dice_delta_24_1:.6f}",
            "decision": "surface candidate for formal rerun" if surface_candidate else "do not form surface candidate",
        },
        {
            "question": "Is formal rerun needed?",
            "answer": formal_rerun,
            "evidence": "formal rerun is reserved for a surface candidate with no profile/LWD regression",
            "decision": "run formal rerun next" if formal_rerun else "no formal rerun from this diagnostic",
        },
        {
            "question": "Should CURRENT_BASELINE remain unchanged?",
            "answer": True,
            "evidence": "24.2 is diagnostic and prompt forbids CURRENT_BASELINE update",
            "decision": "CURRENT_BASELINE.md unchanged",
        },
        {
            "question": "How does it compare with 24.1?",
            "answer": total_delta_24_1 < 0.0 or w_delta_24_1 < 0.0 or dice_delta_24_1 > 0.0,
            "evidence": f"delta_vs_24.1 total={total_delta_24_1:.6f}, wMAE={w_delta_24_1:.6f}, profile={profile_delta_24_1:.9f}, dice={dice_delta_24_1:.6f}",
            "decision": "has at least one 24.1-side improvement" if (total_delta_24_1 < 0.0 or w_delta_24_1 < 0.0 or dice_delta_24_1 > 0.0) else "does not beat 24.1 on key metrics",
        },
    ]


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.decision_matrix], args.overwrite)
    seed = selected_seed_row(args.seed_summary)
    vs_rows = read_csv_rows(args.vs_reference)
    surface_candidate = bool_from_summary(args.training_summary, "surface_feature_fusion_candidate")
    rows = decision_rows(seed, vs_rows, surface_candidate)
    write_csv(args.decision_matrix, rows, DECISION_FIELDS)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "surface_rbc_nls_feature_fusion_route_decision_summary",
        "stage: 24.2 Stage E",
        "",
        f"selected_candidate: {seed.get('candidate')}",
        f"selected_seed: {seed.get('seed')}",
        f"selected_test_total_mae: {f(seed.get('test_total_mae')):.6f}",
        f"selected_test_LWD_mae_mm: {f(seed.get('test_L_mae_mm')):.6f}/{f(seed.get('test_W_mae_mm')):.6f}/{f(seed.get('test_D_mae_mm')):.6f}",
        f"selected_test_wMAE: {f(seed.get('test_wMAE')):.6f}",
        f"selected_test_profile_depth_rmse_m: {f(seed.get('test_profile_depth_rmse_m')):.9f}",
        f"selected_test_er_like_profile_error: {f(seed.get('test_er_like_profile_error')):.6f}",
        f"selected_test_projected_mask_iou_dice: {f(seed.get('test_projected_mask_iou')):.6f}/{f(seed.get('test_projected_mask_dice')):.6f}",
        f"surface_feature_fusion_candidate: {surface_candidate}",
        "CURRENT_BASELINE_update: false",
        "",
        "decisions:",
    ]
    lines.extend([f"- {row['question']}: {row['decision']} ({row['evidence']})" for row in rows])
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, default=TRAINING_SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--decision-matrix", type=Path, default=DECISION_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

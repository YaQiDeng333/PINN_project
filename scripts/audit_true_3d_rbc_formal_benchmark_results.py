#!/usr/bin/env python
"""Audit 20.85 formal benchmark rerun against 20.77/20.81/20.83."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/true_3d_rbc_formal_benchmark_audit_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_comparison_matrix.csv"
FAILURES = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_failure_cases.csv"

FORMAL_SEED = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_seed_summary.csv"
FORMAL_PROFILE = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_profile_metrics.csv"
ORIG_SEED = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_seed_summary.csv"
ORIG_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_profile_metrics.csv"
FUSION_SEED = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_seed_summary.csv"
FUSION_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_profile_metrics.csv"
PP_SCREEN = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_screen_metrics.csv"
PP_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_profile_metrics.csv"

FIELDS = [
    "candidate_id",
    "role",
    "selected_seed",
    "selected_variant",
    "test_total_mae",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_wMAE_auxiliary",
    "test_wLD_abs_error",
    "test_wWD_abs_error",
    "test_wLW_abs_error",
    "test_profile_depth_rmse_m",
    "test_er_like_profile_error",
    "test_projected_mask_iou",
    "test_projected_mask_dice",
    "test_max_depth_error_m",
    "test_volume_proxy_rel_error",
    "best_profile_depth",
    "best_projected_mask",
    "benchmark_candidate",
    "baseline_ready",
    "notes",
]

FAILURE_FIELDS = [
    "case_type",
    "rank",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "size_bin",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_dice",
    "projected_mask_iou",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "curvature_mae_mean",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def selected_seed_row(path: Path) -> dict[str, str]:
    for row in read_csv(path):
        if str(row.get("selected_seed", "")).lower() == "true":
            return row
    raise RuntimeError(f"selected seed not found: {path}")


def selected_test_profile(path: Path, selected_key: str = "selected_seed") -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(path)
        if row.get("split") == "test" and str(row.get(selected_key, "")).lower() == "true"
    ]
    if not rows:
        raise RuntimeError(f"selected test profile rows not found: {path}")
    return rows


def avg(rows: list[dict[str, str]], key: str) -> float:
    vals = [f(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def pp_selected_row() -> dict[str, str]:
    for row in read_csv(PP_SCREEN):
        if row.get("split") == "test" and row.get("selected_by_validation", "").lower() == "true":
            return row
    raise RuntimeError(f"20.83 selected test row not found: {PP_SCREEN}")


def candidate_row(candidate_id: str, role: str, seed: dict[str, str], profile_rows: list[dict[str, str]], variant: str = "") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "role": role,
        "selected_seed": seed.get("seed", ""),
        "selected_variant": variant or seed.get("variant", "conv1d_rbc_param_gate"),
        "test_total_mae": seed.get("test_normalized_param_mae", avg(profile_rows, "normalized_param_mae_mean")),
        "test_L_mae_mm": seed.get("test_L_mae_mm", avg(profile_rows, "L_mae_mm")),
        "test_W_mae_mm": seed.get("test_W_mae_mm", avg(profile_rows, "W_mae_mm")),
        "test_D_mae_mm": seed.get("test_D_mae_mm", avg(profile_rows, "D_mae_mm")),
        "test_wMAE_auxiliary": seed.get("test_wMAE_auxiliary", seed.get("test_curvature_mae", avg(profile_rows, "curvature_mae_mean"))),
        "test_wLD_abs_error": seed.get("test_wLD_abs_error", avg(profile_rows, "wLD_abs_error")),
        "test_wWD_abs_error": seed.get("test_wWD_abs_error", avg(profile_rows, "wWD_abs_error")),
        "test_wLW_abs_error": seed.get("test_wLW_abs_error", avg(profile_rows, "wLW_abs_error")),
        "test_profile_depth_rmse_m": seed.get("test_profile_depth_rmse_m", avg(profile_rows, "profile_depth_rmse_m")),
        "test_er_like_profile_error": seed.get("test_er_like_profile_error", avg(profile_rows, "er_like_profile_error")),
        "test_projected_mask_iou": seed.get("test_projected_mask_iou", avg(profile_rows, "projected_mask_iou")),
        "test_projected_mask_dice": seed.get("test_projected_mask_dice", avg(profile_rows, "projected_mask_dice")),
        "test_max_depth_error_m": seed.get("test_max_depth_error_m", avg(profile_rows, "max_depth_error_m")),
        "test_volume_proxy_rel_error": seed.get("test_volume_proxy_rel_error", avg(profile_rows, "volume_proxy_rel_error")),
        "best_profile_depth": "false",
        "best_projected_mask": "false",
        "benchmark_candidate": "false",
        "baseline_ready": "false",
        "notes": "",
    }


def run(_: argparse.Namespace) -> int:
    formal = candidate_row("20.85_formal_rerun_20.77_protocol", "formal_profile_depth_candidate", selected_seed_row(FORMAL_SEED), selected_test_profile(FORMAL_PROFILE))
    original = candidate_row("20.77_original_candidate", "original_profile_depth_reference", selected_seed_row(ORIG_SEED), selected_test_profile(ORIG_PROFILE))
    fusion = candidate_row("20.81_feature_fusion", "projected_mask_visual_comparator", selected_seed_row(FUSION_SEED), selected_test_profile(FUSION_PROFILE, "selected_by_validation"))
    pp_seed = pp_selected_row()
    pp_profile = selected_test_profile(PP_PROFILE, "selected_by_validation")
    profile_primary = candidate_row("20.83_profile_primary_negative_gate", "negative_gate", pp_seed, pp_profile, pp_seed.get("variant", ""))

    rows = [formal, original, fusion, profile_primary]
    min_profile = min(f(row["test_profile_depth_rmse_m"]) for row in rows if math.isfinite(f(row["test_profile_depth_rmse_m"])))
    non_negative_rows = [row for row in rows if row["role"] != "negative_gate"]
    max_dice = max(f(row["test_projected_mask_dice"]) for row in non_negative_rows if math.isfinite(f(row["test_projected_mask_dice"])))
    for row in rows:
        row["best_profile_depth"] = str(abs(f(row["test_profile_depth_rmse_m"]) - min_profile) <= 1.0e-12).lower()
        row["best_projected_mask"] = str(abs(f(row["test_projected_mask_dice"]) - max_dice) <= 1.0e-12).lower()
        row["benchmark_candidate"] = "true" if row["candidate_id"].startswith("20.85") and f(row["test_profile_depth_rmse_m"]) <= max(f(original["test_profile_depth_rmse_m"]) * 1.20, min_profile + 1.0e-12) else "false"
        row["notes"] = "Benchmark candidate only; baseline_ready remains false." if row["benchmark_candidate"] == "true" else row["notes"]
    original["notes"] = "Original 20.77 profile/depth reference."
    fusion["notes"] = "Visual/mask comparator; Dice is strong but profile RMSE is not the primary winner."
    profile_primary["notes"] = "Negative gate: numerically high Dice did not translate into best profile depth RMSE, so it is not the visual comparator role."
    write_csv(MATRIX, rows, FIELDS)

    formal_profile = selected_test_profile(FORMAL_PROFILE)
    cases: list[dict[str, Any]] = []
    case_specs = [
        ("highest_profile_rmse", "profile_depth_rmse_m", True),
        ("highest_er_like", "er_like_profile_error", True),
        ("lowest_dice", "projected_mask_dice", False),
        ("highest_curvature_aux_error", "curvature_mae_mean", True),
        ("highest_D_error", "D_mae_mm", True),
    ]
    seen: set[tuple[str, str]] = set()
    for case_type, key, reverse in case_specs:
        ranked = sorted(formal_profile, key=lambda row: f(row.get(key)), reverse=reverse)[:8]
        for rank, row in enumerate(ranked, 1):
            case_key = (case_type, row["sample_id"])
            if case_key in seen:
                continue
            seen.add(case_key)
            cases.append(
                {
                    "case_type": case_type,
                    "rank": rank,
                    "sample_id": row.get("sample_id", ""),
                    "split": row.get("split", ""),
                    "curvature_template": row.get("curvature_template", ""),
                    "depth_bin": row.get("depth_bin", ""),
                    "aspect_bin": row.get("aspect_bin", ""),
                    "size_bin": row.get("size_bin", ""),
                    "profile_depth_rmse_m": row.get("profile_depth_rmse_m", ""),
                    "er_like_profile_error": row.get("er_like_profile_error", ""),
                    "projected_mask_dice": row.get("projected_mask_dice", ""),
                    "projected_mask_iou": row.get("projected_mask_iou", ""),
                    "L_mae_mm": row.get("L_mae_mm", ""),
                    "W_mae_mm": row.get("W_mae_mm", ""),
                    "D_mae_mm": row.get("D_mae_mm", ""),
                    "wLD_abs_error": row.get("wLD_abs_error", ""),
                    "wWD_abs_error": row.get("wWD_abs_error", ""),
                    "wLW_abs_error": row.get("wLW_abs_error", ""),
                    "curvature_mae_mean": row.get("curvature_mae_mean", ""),
                    "notes": "selected formal rerun test split; labels used only for metrics",
                }
            )
    write_csv(FAILURES, cases, FAILURE_FIELDS)

    profile_stable = f(formal["test_profile_depth_rmse_m"]) <= f(original["test_profile_depth_rmse_m"]) * 1.20
    beats_visual_profile = f(formal["test_profile_depth_rmse_m"]) < f(fusion["test_profile_depth_rmse_m"])
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.85 true 3D RBC formal benchmark audit summary",
                "",
                f"formal_selected_seed: {formal['selected_seed']}",
                f"formal_test_profile_depth_rmse_m: {formal['test_profile_depth_rmse_m']}",
                f"original_20_77_profile_depth_rmse_m: {original['test_profile_depth_rmse_m']}",
                f"20_81_profile_depth_rmse_m: {fusion['test_profile_depth_rmse_m']}",
                f"20_83_profile_depth_rmse_m: {profile_primary['test_profile_depth_rmse_m']}",
                f"formal_profile_stable_vs_original_20_77: {profile_stable}",
                f"formal_beats_20_81_profile_rmse: {beats_visual_profile}",
                f"formal_test_dice: {formal['test_projected_mask_dice']}",
                f"best_projected_mask_candidate_excluding_negative_gate: {[row['candidate_id'] for row in rows if row['best_projected_mask'] == 'true'][0]}",
                f"highest_raw_dice_including_negative_gate: {profile_primary['candidate_id']} ({profile_primary['test_projected_mask_dice']})",
                "baseline_ready: false",
                "conclusion: formal rerun is evaluated as a benchmark candidate only; profile/depth remains the primary axis, Dice is a visual comparator.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

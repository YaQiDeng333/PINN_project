#!/usr/bin/env python
"""Consolidate Stage 20.77/20.81/20.83 true-3D RBC candidate roles.

This is a metadata/results audit only. It reads persisted summaries and metrics;
it does not load latest/newest datasets, train models, run COMSOL, or regenerate
prediction artifacts.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_candidate_consolidation_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_candidate_consolidation_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_candidate_consolidation_matrix.csv"

NEURAL_SEED = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_seed_summary.csv"
NEURAL_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_profile_metrics.csv"
NEURAL_DECISION = ROOT / "results/summaries/true_3d_rbc_v3_240_training_gate_decision_summary.txt"

FUSION_SEED = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_seed_summary.csv"
FUSION_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_profile_metrics.csv"
FUSION_DECISION = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_decision_summary.txt"

PROFILE_PRIMARY_SCREEN = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_screen_metrics.csv"
PROFILE_PRIMARY_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_profile_metrics.csv"
PROFILE_PRIMARY_DECISION = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_decision_summary.txt"

GALLERY_INDEX = ROOT / "results/metrics/true_3d_rbc_profile_primary_loss_gallery_index.csv"
GALLERY_SAMPLE_METRICS = ROOT / "results/metrics/true_3d_rbc_profile_primary_loss_gallery_sample_metrics.csv"
GALLERY_DIR = ROOT / "results/previews/true_3d_rbc_profile_primary_loss_gallery"

FIELDS = [
    "candidate_id",
    "stage",
    "role",
    "source",
    "selected_seed",
    "selected_variant",
    "test_total_mae",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_wLD_abs_error",
    "test_wWD_abs_error",
    "test_wLW_abs_error",
    "test_wMAE_auxiliary",
    "test_profile_depth_rmse_m",
    "test_er_like_profile_error",
    "test_projected_mask_iou",
    "test_projected_mask_dice",
    "test_max_depth_error_m",
    "test_volume_proxy_rel_error",
    "best_for_profile_depth",
    "best_for_projected_mask_visual",
    "negative_gate",
    "baseline_ready",
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


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [as_float(row.get(key)) for row in rows]
    vals = [val for val in vals if math.isfinite(val)]
    return float(sum(vals) / len(vals)) if vals else math.nan


def fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def selected_row(path: Path, flag: str) -> dict[str, str]:
    rows = read_csv(path)
    for row in rows:
        if row.get(flag, "").lower() == "true":
            return row
    raise RuntimeError(f"no selected row in {path}")


def selected_profile_rows(path: Path, flag: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    selected = [row for row in rows if row.get("split") == "test" and row.get(flag, "").lower() == "true"]
    if not selected:
        raise RuntimeError(f"no selected test profile rows in {path}")
    return selected


def profile_primary_selected() -> tuple[dict[str, str], list[dict[str, str]]]:
    rows = read_csv(PROFILE_PRIMARY_SCREEN)
    selected_test = [
        row
        for row in rows
        if row.get("split") == "test" and row.get("selected_by_validation", "").lower() == "true"
    ]
    if not selected_test:
        raise RuntimeError(f"no selected 20.83 test row in {PROFILE_PRIMARY_SCREEN}")
    profile_rows = selected_profile_rows(PROFILE_PRIMARY_PROFILE, "selected_by_validation")
    return selected_test[0], profile_rows


def metric_from_profile(rows: list[dict[str, str]], key: str) -> str:
    value = mean(rows, key)
    return fmt(value)


def write_preflight() -> None:
    required = [
        NEURAL_SEED,
        NEURAL_PROFILE,
        NEURAL_DECISION,
        FUSION_SEED,
        FUSION_PROFILE,
        FUSION_DECISION,
        PROFILE_PRIMARY_SCREEN,
        PROFILE_PRIMARY_PROFILE,
        PROFILE_PRIMARY_DECISION,
        GALLERY_INDEX,
        GALLERY_SAMPLE_METRICS,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    png_count = len(list(GALLERY_DIR.glob("*.png"))) if GALLERY_DIR.exists() else 0
    ignored_preview_path = "results/previews/" in GALLERY_DIR.as_posix()
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text(
        "\n".join(
            [
                "20.84 true 3D RBC candidate consolidation preflight",
                "",
                f"repository_root: {ROOT}",
                "stage_type: result audit only",
                "training_run: false",
                "COMSOL_run: false",
                "new_data_generated: false",
                "NPZ_modified: false",
                "latest_newest_NPZ_scan: false",
                f"missing_required_inputs: {', '.join(missing) if missing else 'none'}",
                f"gallery_dir: {GALLERY_DIR}",
                f"gallery_png_count_existing: {png_count}",
                f"gallery_path_ignored_by_policy: {ignored_preview_path}",
                "critical_gate: pass" if not missing and ignored_preview_path else "critical_gate: blocker",
                "",
                "candidate_inputs:",
                f"- 20.77: {NEURAL_SEED.relative_to(ROOT)} / {NEURAL_PROFILE.relative_to(ROOT)}",
                f"- 20.81: {FUSION_SEED.relative_to(ROOT)} / {FUSION_PROFILE.relative_to(ROOT)}",
                f"- 20.83: {PROFILE_PRIMARY_SCREEN.relative_to(ROOT)} / {PROFILE_PRIMARY_PROFILE.relative_to(ROOT)}",
                "",
                "forbidden_artifacts: data/, NPZ, checkpoint, preview PNG, notes, baseline docs, CURRENT_BASELINE.md, scripts/visualize_current_baseline.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if missing:
        raise RuntimeError(f"missing required inputs: {missing}")


def build_rows() -> list[dict[str, str]]:
    neural_seed = selected_row(NEURAL_SEED, "selected_seed")
    neural_profile = selected_profile_rows(NEURAL_PROFILE, "selected_seed")
    fusion_seed = selected_row(FUSION_SEED, "selected_seed")
    fusion_profile = selected_profile_rows(FUSION_PROFILE, "selected_by_validation")
    profile_primary_screen, profile_primary_profile = profile_primary_selected()

    rows = [
        {
            "candidate_id": "20.77_neural_reference",
            "stage": "20.77",
            "role": "profile_depth_main_candidate",
            "source": "selected_seed_profile_metrics",
            "selected_seed": neural_seed.get("seed", ""),
            "selected_variant": "conv1d_rbc_param_gate",
            "test_total_mae": neural_seed.get("test_normalized_param_mae", metric_from_profile(neural_profile, "normalized_param_mae_mean")),
            "test_L_mae_mm": neural_seed.get("test_L_mae_mm", metric_from_profile(neural_profile, "L_mae_mm")),
            "test_W_mae_mm": neural_seed.get("test_W_mae_mm", metric_from_profile(neural_profile, "W_mae_mm")),
            "test_D_mae_mm": neural_seed.get("test_D_mae_mm", metric_from_profile(neural_profile, "D_mae_mm")),
            "test_wLD_abs_error": metric_from_profile(neural_profile, "wLD_abs_error"),
            "test_wWD_abs_error": metric_from_profile(neural_profile, "wWD_abs_error"),
            "test_wLW_abs_error": metric_from_profile(neural_profile, "wLW_abs_error"),
            "test_wMAE_auxiliary": neural_seed.get("test_curvature_mae", metric_from_profile(neural_profile, "curvature_mae_mean")),
            "test_profile_depth_rmse_m": neural_seed.get("test_profile_depth_rmse_m", metric_from_profile(neural_profile, "profile_depth_rmse_m")),
            "test_er_like_profile_error": "",
            "test_projected_mask_iou": neural_seed.get("test_projected_mask_iou", metric_from_profile(neural_profile, "projected_mask_iou")),
            "test_projected_mask_dice": neural_seed.get("test_projected_mask_dice", metric_from_profile(neural_profile, "projected_mask_dice")),
            "test_max_depth_error_m": "",
            "test_volume_proxy_rel_error": metric_from_profile(neural_profile, "volume_proxy_rel_error"),
            "best_for_profile_depth": "true",
            "best_for_projected_mask_visual": "false",
            "negative_gate": "false",
            "baseline_ready": "false",
            "notes": "Best persisted profile_depth_rmse_m among 20.77/20.81/20.83; benchmark candidate only, not baseline.",
        },
        {
            "candidate_id": "20.81_feature_fusion",
            "stage": "20.81",
            "role": "projected_mask_visual_reference",
            "source": "selected_seed_feature_fusion_metrics",
            "selected_seed": fusion_seed.get("seed", ""),
            "selected_variant": fusion_seed.get("variant", ""),
            "test_total_mae": fusion_seed.get("test_normalized_param_mae", metric_from_profile(fusion_profile, "normalized_param_mae_mean")),
            "test_L_mae_mm": fusion_seed.get("test_L_mae_mm", metric_from_profile(fusion_profile, "L_mae_mm")),
            "test_W_mae_mm": fusion_seed.get("test_W_mae_mm", metric_from_profile(fusion_profile, "W_mae_mm")),
            "test_D_mae_mm": fusion_seed.get("test_D_mae_mm", metric_from_profile(fusion_profile, "D_mae_mm")),
            "test_wLD_abs_error": fusion_seed.get("test_wLD_abs_error", metric_from_profile(fusion_profile, "wLD_abs_error")),
            "test_wWD_abs_error": fusion_seed.get("test_wWD_abs_error", metric_from_profile(fusion_profile, "wWD_abs_error")),
            "test_wLW_abs_error": fusion_seed.get("test_wLW_abs_error", metric_from_profile(fusion_profile, "wLW_abs_error")),
            "test_wMAE_auxiliary": fusion_seed.get("test_curvature_mae", metric_from_profile(fusion_profile, "curvature_mae_mean")),
            "test_profile_depth_rmse_m": fusion_seed.get("test_profile_depth_rmse_m", metric_from_profile(fusion_profile, "profile_depth_rmse_m")),
            "test_er_like_profile_error": "",
            "test_projected_mask_iou": fusion_seed.get("test_projected_mask_iou", metric_from_profile(fusion_profile, "projected_mask_iou")),
            "test_projected_mask_dice": fusion_seed.get("test_projected_mask_dice", metric_from_profile(fusion_profile, "projected_mask_dice")),
            "test_max_depth_error_m": "",
            "test_volume_proxy_rel_error": "",
            "best_for_profile_depth": "false",
            "best_for_projected_mask_visual": "false",
            "negative_gate": "false",
            "baseline_ready": "false",
            "notes": "Improves total MAE and Dice versus 20.77, but profile_depth_rmse_m remains worse than 20.77.",
        },
        {
            "candidate_id": "20.83_profile_primary_loss",
            "stage": "20.83",
            "role": "negative_gate_profile_primary_loss",
            "source": "selected_candidate_screen_metrics",
            "selected_seed": profile_primary_screen.get("seed", ""),
            "selected_variant": profile_primary_screen.get("variant", ""),
            "test_total_mae": profile_primary_screen.get("normalized_param_mae", metric_from_profile(profile_primary_profile, "normalized_param_mae_mean")),
            "test_L_mae_mm": profile_primary_screen.get("L_mae_mm", metric_from_profile(profile_primary_profile, "L_mae_mm")),
            "test_W_mae_mm": profile_primary_screen.get("W_mae_mm", metric_from_profile(profile_primary_profile, "W_mae_mm")),
            "test_D_mae_mm": profile_primary_screen.get("D_mae_mm", metric_from_profile(profile_primary_profile, "D_mae_mm")),
            "test_wLD_abs_error": profile_primary_screen.get("wLD_abs_error", metric_from_profile(profile_primary_profile, "wLD_abs_error")),
            "test_wWD_abs_error": profile_primary_screen.get("wWD_abs_error", metric_from_profile(profile_primary_profile, "wWD_abs_error")),
            "test_wLW_abs_error": profile_primary_screen.get("wLW_abs_error", metric_from_profile(profile_primary_profile, "wLW_abs_error")),
            "test_wMAE_auxiliary": profile_primary_screen.get("curvature_mae", metric_from_profile(profile_primary_profile, "curvature_mae_mean")),
            "test_profile_depth_rmse_m": profile_primary_screen.get("profile_depth_rmse_m", metric_from_profile(profile_primary_profile, "profile_depth_rmse_m")),
            "test_er_like_profile_error": profile_primary_screen.get("er_like_profile_error", metric_from_profile(profile_primary_profile, "er_like_profile_error")),
            "test_projected_mask_iou": profile_primary_screen.get("projected_mask_iou", metric_from_profile(profile_primary_profile, "projected_mask_iou")),
            "test_projected_mask_dice": profile_primary_screen.get("projected_mask_dice", metric_from_profile(profile_primary_profile, "projected_mask_dice")),
            "test_max_depth_error_m": profile_primary_screen.get("max_depth_error_m", metric_from_profile(profile_primary_profile, "max_depth_error_m")),
            "test_volume_proxy_rel_error": profile_primary_screen.get("volume_proxy_rel_error", metric_from_profile(profile_primary_profile, "volume_proxy_rel_error")),
            "best_for_profile_depth": "false",
            "best_for_projected_mask_visual": "true",
            "negative_gate": "true",
            "baseline_ready": "false",
            "notes": "Best Dice in this three-way comparison, but profile_depth_rmse_m worsens versus 20.77; not eligible to replace 20.77/20.81.",
        },
    ]

    best_profile = min(rows, key=lambda row: as_float(row["test_profile_depth_rmse_m"]))
    eligible_visual_rows = [row for row in rows if row["negative_gate"] != "true"]
    best_dice = max(eligible_visual_rows, key=lambda row: as_float(row["test_projected_mask_dice"]))
    for row in rows:
        row["best_for_profile_depth"] = str(row["candidate_id"] == best_profile["candidate_id"]).lower()
        row["best_for_projected_mask_visual"] = str(row["candidate_id"] == best_dice["candidate_id"]).lower()
    by_id = {row["candidate_id"]: row for row in rows}
    if as_float(by_id["20.83_profile_primary_loss"]["test_projected_mask_dice"]) > as_float(by_id["20.81_feature_fusion"]["test_projected_mask_dice"]):
        by_id["20.83_profile_primary_loss"]["notes"] += " It has the numerically highest Dice, but is not selected as the visual reference because it is a negative profile-depth gate."
    return rows


def write_summary(rows: list[dict[str, str]]) -> None:
    by_id = {row["candidate_id"]: row for row in rows}
    n77 = by_id["20.77_neural_reference"]
    n81 = by_id["20.81_feature_fusion"]
    n83 = by_id["20.83_profile_primary_loss"]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.84 true 3D RBC candidate consolidation summary",
                "",
                "training_run: false",
                "COMSOL_run: false",
                "new_data_generated: false",
                "NPZ_modified: false",
                "baseline_ready: false",
                "CURRENT_BASELINE_update: false",
                "",
                "candidate_roles:",
                f"- 20.77 neural reference: profile/depth main candidate; profile_depth_rmse_m={n77['test_profile_depth_rmse_m']}; Dice={n77['test_projected_mask_dice']}.",
                f"- 20.81 feature-fusion: projected-mask/visual reference candidate; profile_depth_rmse_m={n81['test_profile_depth_rmse_m']}; Dice={n81['test_projected_mask_dice']}.",
                f"- 20.83 profile-primary loss: negative gate; profile_depth_rmse_m={n83['test_profile_depth_rmse_m']}; Dice={n83['test_projected_mask_dice']}.",
                "",
                "core_judgment:",
                "20.77 remains the profile/depth main candidate because it has the best profile_depth_rmse_m.",
                "20.81 remains the non-negative visual/mask reference because it improves projected footprint quality without becoming the profile-depth winner.",
                "20.83 is negative evidence for the current R1 profile-primary loss setup: it improves projected mask Dice but worsens profile RMSE, so it cannot replace 20.77 or 20.81.",
                "wLD/wWD/wLW errors remain auxiliary diagnostics; they are not the main replacement criterion in this consolidation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    write_preflight()
    rows = build_rows()
    write_csv(MATRIX, rows, FIELDS)
    write_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

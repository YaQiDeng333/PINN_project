#!/usr/bin/env python
"""Audit the 20.92 nominal/non-nominal liftoff trade-off.

This stage is read-only. It consumes existing 20.90/20.91/20.92 summaries and
metrics, writes audit tables, and does not run COMSOL, train, or touch data/NPZ.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_tradeoff_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_tradeoff_audit_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_tradeoff_matrix.csv"
GROUP_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_tradeoff_group_summary.csv"
FAILURE_CASES = ROOT / "results/metrics/true_3d_rbc_liftoff_tradeoff_failure_cases.csv"

TRAINING_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_training_metrics.csv"
BY_LIFTOFF = ROOT / "results/metrics/true_3d_rbc_liftoff_training_by_liftoff.csv"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_training_seed_summary.csv"
VS_BASELINE = ROOT / "results/metrics/true_3d_rbc_liftoff_training_vs_baseline.csv"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_training_route_decision_matrix.csv"
DIAG_20_90 = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_group_summary.csv"
PACK_20_91 = ROOT / "results/metrics/true_3d_rbc_liftoff_aug_pack_group_summary.csv"
REQUIRED = [
    ROOT / "results/summaries/true_3d_rbc_liftoff_sensor_offset_route_decision_summary.txt",
    ROOT / "results/summaries/true_3d_rbc_liftoff_aug_pack_validation_summary.txt",
    ROOT / "results/summaries/true_3d_rbc_liftoff_training_summary.txt",
    ROOT / "results/summaries/true_3d_rbc_liftoff_training_route_decision_summary.txt",
    TRAINING_METRICS,
    BY_LIFTOFF,
    SEED_SUMMARY,
    VS_BASELINE,
    DECISION_MATRIX,
    DIAG_20_90,
    PACK_20_91,
]

METRICS = [
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "normalized_param_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wMAE_auxiliary",
    "projected_mask_iou",
    "projected_mask_dice",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pct(current: float, reference: float) -> float:
    return 0.0 if abs(reference) < 1.0e-20 else 100.0 * (current - reference) / reference


def fnum(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value in {"", None} else float(value)


def git_staged_names() -> list[str]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def write_preflight() -> None:
    missing = [path for path in REQUIRED if not path.exists()]
    staged = git_staged_names()
    forbidden = [
        path
        for path in staged
        if path.startswith("data/")
        or path.startswith("notes/")
        or path.lower().endswith((".npz", ".pt", ".pth", ".png", ".mph"))
        or path == "CURRENT_BASELINE.md"
        or path == "scripts/visualize_current_baseline.py"
    ]
    lines = [
        "20.93 true 3D RBC liftoff trade-off audit preflight",
        "",
        f"required_inputs_present: {not missing}",
        f"missing_inputs: {[str(path.relative_to(ROOT)) for path in missing]}",
        "analysis_source: existing 20.90/20.91/20.92 summaries and metrics only",
        "COMSOL_run: false",
        "training_run: false",
        "data_or_npz_write: false",
        "CURRENT_BASELINE_update: false",
        f"forbidden_staged_files: {forbidden}",
        "artifact_boundary: 20.92 has aggregate training metrics, but no per-sample C1/C2 prediction artifact or group-level C1/C2 output; group concentration is limited accordingly.",
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing or forbidden:
        raise RuntimeError(f"20.93 preflight failed; missing={missing}; forbidden_staged={forbidden}")


def candidate_metric_rows(training: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    refs = {
        (row["split"], row["liftoff_subset"]): row
        for row in training
        if row["candidate"] == "C0_reference_20_85_baseline"
    }
    for row in training:
        if row["split"] != "test":
            continue
        ref = refs.get((row["split"], row["liftoff_subset"]))
        for metric in METRICS:
            current = fnum(row, metric)
            reference = fnum(ref, metric) if ref else math.nan
            lower_better = metric not in {"projected_mask_iou", "projected_mask_dice"}
            improved = current < reference if lower_better else current > reference
            rows.append(
                {
                    "analysis_type": "candidate_subset_metric",
                    "candidate": row["candidate"],
                    "seed": row.get("seed", ""),
                    "selected_seed": row.get("selected_seed", ""),
                    "split": row["split"],
                    "liftoff_subset": row["liftoff_subset"],
                    "sensor_z_m": "",
                    "metric": metric,
                    "value": current,
                    "reference_candidate": "C0_reference_20_85_baseline" if ref else "",
                    "reference_value": reference,
                    "delta_vs_reference": current - reference if ref else "",
                    "relative_change_pct_vs_reference": pct(current, reference) if ref else "",
                    "improved_vs_reference": improved if ref else "",
                    "notes": "test-final reporting only; not used for selection",
                }
            )
    return rows


def liftoff_level_rows(by_liftoff: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    refs = {
        (row["split"], row["liftoff_subset"]): row
        for row in by_liftoff
        if row["candidate"] == "C0_reference_20_85_baseline"
    }
    for row in by_liftoff:
        if row["split"] != "test":
            continue
        ref = refs.get((row["split"], row["liftoff_subset"]))
        sensor_z = row["liftoff_subset"].replace("sensor_z_", "")
        for metric in METRICS:
            current = fnum(row, metric)
            reference = fnum(ref, metric) if ref else math.nan
            lower_better = metric not in {"projected_mask_iou", "projected_mask_dice"}
            rows.append(
                {
                    "analysis_type": "per_liftoff_level_metric",
                    "candidate": row["candidate"],
                    "seed": row.get("seed", ""),
                    "selected_seed": row.get("selected_seed", ""),
                    "split": row["split"],
                    "liftoff_subset": row["liftoff_subset"],
                    "sensor_z_m": sensor_z,
                    "metric": metric,
                    "value": current,
                    "reference_candidate": "C0_reference_20_85_baseline" if ref else "",
                    "reference_value": reference,
                    "delta_vs_reference": current - reference if ref else "",
                    "relative_change_pct_vs_reference": pct(current, reference) if ref else "",
                    "improved_vs_reference": (current < reference if lower_better else current > reference) if ref else "",
                    "notes": "per-liftoff aggregate from 20.92 metrics",
                }
            )
    return rows


def validation_selection_rows(training: list[dict[str, str]], seeds: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in seeds:
        if row["candidate"] in {"C0_reference_20_85_baseline", "C3_calibrated_input_conditioned"}:
            continue
        val_nom = next(
            r for r in training if r["candidate"] == row["candidate"] and r["seed"] == row["seed"] and r["split"] == "val" and r["liftoff_subset"] == "nominal_0p008"
        )
        val_non = next(
            r for r in training if r["candidate"] == row["candidate"] and r["seed"] == row["seed"] and r["split"] == "val" and r["liftoff_subset"] == "non_nominal"
        )
        test_nom = next(
            r for r in training if r["candidate"] == row["candidate"] and r["seed"] == row["seed"] and r["split"] == "test" and r["liftoff_subset"] == "nominal_0p008"
        )
        test_non = next(
            r for r in training if r["candidate"] == row["candidate"] and r["seed"] == row["seed"] and r["split"] == "test" and r["liftoff_subset"] == "non_nominal"
        )
        rows.append(
            {
                "analysis_type": "validation_selection_diagnostic",
                "candidate": row["candidate"],
                "seed": row["seed"],
                "selected_seed": row["selected_robustness_candidate"],
                "best_val_selection_metric": row.get("best_val_selection_metric", ""),
                "val_nominal_profile_depth_rmse_m": val_nom["profile_depth_rmse_m"],
                "val_non_nominal_profile_depth_rmse_m": val_non["profile_depth_rmse_m"],
                "val_nominal_normalized_param_mae": val_nom["normalized_param_mae"],
                "val_non_nominal_normalized_param_mae": val_non["normalized_param_mae"],
                "test_nominal_profile_depth_rmse_m": test_nom["profile_depth_rmse_m"],
                "test_non_nominal_profile_depth_rmse_m": test_non["profile_depth_rmse_m"],
                "notes": "Selection used validation param metric over all rows; nominal rows are 25pct of validation rows.",
            }
        )
    return rows


def group_rows_from_existing_sources() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if DIAG_20_90.exists():
        for row in read_csv(DIAG_20_90):
            if row.get("split") != "test" or row.get("factor_group") != "liftoff":
                continue
            out.append(
                {
                    "source": "20.90_liftoff_sensor_offset_diagnostic",
                    "candidate": "20.85_fixed_baseline_raw_or_calibrated",
                    "input_mode": row.get("input_mode", ""),
                    "group_field": row.get("group_field", ""),
                    "group_value": row.get("group_value", ""),
                    "variant_name": row.get("variant_name", ""),
                    "sample_count": row.get("sample_count", ""),
                    "profile_depth_rmse_m": row.get("profile_depth_rmse_m", ""),
                    "profile_depth_rmse_degradation_pct": row.get("profile_depth_rmse_degradation_pct_vs_nominal", ""),
                    "projected_mask_dice": row.get("projected_mask_dice", ""),
                    "status_band": row.get("status_band", ""),
                    "notes": "Available group evidence is from 20.90 diagnostic, not C1/C2 20.92 training outputs.",
                }
            )
    for field in ["curvature_template", "depth_bin", "aspect_bin"]:
        out.append(
            {
                "source": "20.92_training_gate",
                "candidate": "C1/C2",
                "input_mode": "aggregate_metrics_only",
                "group_field": field,
                "group_value": "not_available",
                "sample_count": "",
                "profile_depth_rmse_m": "",
                "profile_depth_rmse_degradation_pct": "",
                "projected_mask_dice": "",
                "status_band": "artifact_limited",
                "notes": "20.92 did not persist per-sample or group-level C1/C2 prediction metrics; do not infer group concentration.",
            }
        )
    return out


def failure_rows(training: list[dict[str, str]], by_liftoff: list[dict[str, str]], seeds: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = next(row for row in seeds if row.get("selected_robustness_candidate", "").lower() == "true")
    c0_nom = next(row for row in training if row["candidate"] == "C0_reference_20_85_baseline" and row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008")
    c1_nom = next(row for row in training if row["candidate"] == selected["candidate"] and row["seed"] == selected["seed"] and row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008")
    c0_non = next(row for row in training if row["candidate"] == "C0_reference_20_85_baseline" and row["split"] == "test" and row["liftoff_subset"] == "non_nominal")
    c1_non = next(row for row in training if row["candidate"] == selected["candidate"] and row["seed"] == selected["seed"] and row["split"] == "test" and row["liftoff_subset"] == "non_nominal")
    rows.append(
        {
            "failure_type": "nominal_forgetting",
            "candidate": selected["candidate"],
            "seed": selected["seed"],
            "split": "test",
            "scope": "nominal_0p008",
            "reference_profile_depth_rmse_m": c0_nom["profile_depth_rmse_m"],
            "current_profile_depth_rmse_m": c1_nom["profile_depth_rmse_m"],
            "relative_change_pct": pct(fnum(c1_nom, "profile_depth_rmse_m"), fnum(c0_nom, "profile_depth_rmse_m")),
            "profile_depth_rmse_relative_change_pct": pct(fnum(c1_nom, "profile_depth_rmse_m"), fnum(c0_nom, "profile_depth_rmse_m")),
            "projected_mask_dice_delta": fnum(c1_nom, "projected_mask_dice") - fnum(c0_nom, "projected_mask_dice"),
            "notes": "Main blocker: non-nominal robustness came with severe nominal profile regression.",
        }
    )
    rows.append(
        {
            "failure_type": "curvature_auxiliary_worse",
            "candidate": selected["candidate"],
            "seed": selected["seed"],
            "split": "test",
            "scope": "non_nominal",
            "reference_profile_depth_rmse_m": c0_non["profile_depth_rmse_m"],
            "current_profile_depth_rmse_m": c1_non["profile_depth_rmse_m"],
            "relative_change_pct": pct(fnum(c1_non, "profile_depth_rmse_m"), fnum(c0_non, "profile_depth_rmse_m")),
            "profile_depth_rmse_relative_change_pct": pct(fnum(c1_non, "profile_depth_rmse_m"), fnum(c0_non, "profile_depth_rmse_m")),
            "wmae_relative_change_pct": pct(fnum(c1_non, "wMAE_auxiliary"), fnum(c0_non, "wMAE_auxiliary")),
            "projected_mask_dice_delta": fnum(c1_non, "projected_mask_dice") - fnum(c0_non, "projected_mask_dice"),
            "notes": "Profile/Dice improved, but auxiliary wMAE worsened; keep w parameters diagnostic only.",
        }
    )
    best_test = min(
        [row for row in training if row["candidate"].startswith(("C1_", "C2_")) and row["split"] == "test" and row["liftoff_subset"] == "non_nominal"],
        key=lambda row: fnum(row, "profile_depth_rmse_m"),
    )
    rows.append(
        {
            "failure_type": "test_best_not_selected",
            "candidate": best_test["candidate"],
            "seed": best_test["seed"],
            "split": "test",
            "scope": "non_nominal",
            "reference_profile_depth_rmse_m": c0_non["profile_depth_rmse_m"],
            "current_profile_depth_rmse_m": best_test["profile_depth_rmse_m"],
            "relative_change_pct": pct(fnum(best_test, "profile_depth_rmse_m"), fnum(c0_non, "profile_depth_rmse_m")),
            "profile_depth_rmse_relative_change_pct": pct(fnum(best_test, "profile_depth_rmse_m"), fnum(c0_non, "profile_depth_rmse_m")),
            "projected_mask_dice_delta": fnum(best_test, "projected_mask_dice") - fnum(c0_non, "projected_mask_dice"),
            "notes": "Do not test-reselect this seed; it only shows selection score should be redesigned for the next stage.",
        }
    )
    for row in by_liftoff:
        if row["candidate"] == selected["candidate"] and row["seed"] == selected["seed"] and row["split"] == "test":
            ref = next(r for r in by_liftoff if r["candidate"] == "C0_reference_20_85_baseline" and r["split"] == "test" and r["liftoff_subset"] == row["liftoff_subset"])
            rel = pct(fnum(row, "profile_depth_rmse_m"), fnum(ref, "profile_depth_rmse_m"))
            if row["liftoff_subset"] == "sensor_z_0.008" or rel > 0.0:
                rows.append(
                    {
                        "failure_type": "per_liftoff_regression",
                        "candidate": selected["candidate"],
                        "seed": selected["seed"],
                        "split": "test",
                        "scope": row["liftoff_subset"],
                        "reference_profile_depth_rmse_m": ref["profile_depth_rmse_m"],
                        "current_profile_depth_rmse_m": row["profile_depth_rmse_m"],
                        "relative_change_pct": rel,
                        "profile_depth_rmse_relative_change_pct": rel,
                        "projected_mask_dice_delta": fnum(row, "projected_mask_dice") - fnum(ref, "projected_mask_dice"),
                        "notes": "Aggregate liftoff-level regression/nominal risk.",
                    }
                )
    return rows


def write_summary(matrix: list[dict[str, Any]], failures: list[dict[str, Any]], selection: list[dict[str, Any]]) -> None:
    selected = next(row for row in selection if str(row["selected_seed"]).lower() == "true")
    nominal_failure = next(row for row in failures if row["failure_type"] == "nominal_forgetting")
    non_improve = next(
        row
        for row in matrix
        if row["analysis_type"] == "candidate_subset_metric"
        and row["candidate"] == selected["candidate"]
        and str(row["seed"]) == str(selected["seed"])
        and row["liftoff_subset"] == "non_nominal"
        and row["metric"] == "profile_depth_rmse_m"
    )
    lines = [
        "20.93 true 3D RBC liftoff nominal/non-nominal trade-off audit",
        "",
        "Scope: read-only audit of 20.90/20.91/20.92 summaries and metrics; no COMSOL, no training, no data/NPZ mutation, no CURRENT_BASELINE update.",
        "",
        "Main finding:",
        f"- Selected 20.92 model: {selected['candidate']} seed={selected['seed']}.",
        f"- Non-nominal profile RMSE improved by {float(non_improve['relative_change_pct_vs_reference']):.3f}% versus C0.",
        f"- Nominal 0.008m profile RMSE regressed by {float(nominal_failure['relative_change_pct']):.3f}% versus C0.",
        "",
        "Root-cause diagnosis:",
        "- C1 is unconditioned: it sees liftoff-altered signal amplitude/shape but is not told sensor_z_m, so the same geometry across four liftoffs becomes an ambiguous inverse mapping.",
        "- Validation selection used a row-aggregate parameter loss; nominal rows are only one quarter of the validation rows, and there was no explicit nominal-preservation penalty against the fixed 20.85 baseline.",
        "- C1 seed=123 looked best by validation, including validation nominal metrics, but failed to preserve nominal behavior on held-out base geometries; this points to base-level generalization and nominal forgetting, not test-based model selection.",
        "- C2 sensor_z conditioning was not selected because its validation selection metric was worse than C1. Post-hoc test comparisons are diagnostic only and must not be used to select the 20.92 model.",
        "",
        "Artifact limitation:",
        "- 20.92 persisted aggregate metrics, not per-sample C1/C2 predictions. Therefore concentration by depth_bin, curvature_template, or aspect_bin cannot be honestly computed for C1/C2 in this audit.",
        "",
        "Implication:",
        "- Do not continue unconditional C1 augmentation. The next training stage needs nominal-preserving selection/loss and likely a baseline+liftoff correction structure or revised sensor_z-conditioned protocol.",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    _args = parse_args()
    write_preflight()
    training = read_csv(TRAINING_METRICS)
    by_liftoff = read_csv(BY_LIFTOFF)
    seeds = read_csv(SEED_SUMMARY)
    matrix = candidate_metric_rows(training) + liftoff_level_rows(by_liftoff)
    selection = validation_selection_rows(training, seeds)
    matrix.extend(selection)
    groups = group_rows_from_existing_sources()
    failures = failure_rows(training, by_liftoff, seeds)
    write_csv(MATRIX, matrix)
    write_csv(GROUP_SUMMARY, groups)
    write_csv(FAILURE_CASES, failures)
    write_summary(matrix, failures, selection)
    print(f"wrote {SUMMARY}")
    print(f"wrote {MATRIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

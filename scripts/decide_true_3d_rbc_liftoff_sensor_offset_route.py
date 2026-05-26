#!/usr/bin/env python
"""Route decision for the 20.90 true-3D RBC liftoff/sensor-offset diagnostic."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import ROOT, write_csv


ROBUSTNESS_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_robustness_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_group_summary.csv"
VALIDATION_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_validation_metrics.csv"
FORMAL_PROFILE = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_profile_metrics.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_sensor_offset_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_decision_matrix.csv"

FIELDS = [
    "decision_item",
    "status",
    "evidence_metric",
    "raw_value",
    "calibrated_value",
    "threshold",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide 20.90 true-3D RBC liftoff/sensor-offset route.")
    parser.add_argument("--metrics", type=Path, default=ROBUSTNESS_METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--validation-metrics", type=Path, default=VALIDATION_METRICS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--decision-matrix", type=Path, default=DECISION_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def group_row(rows: list[dict[str, str]], input_mode: str, factor_group: str, variant_name: str | None = None) -> list[dict[str, str]]:
    out = [
        row
        for row in rows
        if row.get("input_mode") == input_mode
        and row.get("factor_group") == factor_group
        and row.get("split") == "test"
        and row.get("group_field") == "all"
    ]
    if variant_name is not None:
        out = [row for row in out if row.get("variant_name") == variant_name]
    return out


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [f(row, key) for row in rows if not math.isnan(f(row, key))]
    return float(np.mean(vals)) if vals else math.nan


def status_from(degradation: float, dice_drop: float) -> str:
    if degradation <= 10.0 and dice_drop <= 0.02:
        return "green"
    if degradation <= 25.0 and dice_drop <= 0.05:
        return "warning"
    return "fail"


def add(rows: list[dict[str, Any]], item: str, status: str, metric: str, raw: Any, calibrated: Any, threshold: str, decision: str, notes: str) -> None:
    rows.append(
        {
            "decision_item": item,
            "status": status,
            "evidence_metric": metric,
            "raw_value": raw,
            "calibrated_value": calibrated,
            "threshold": threshold,
            "decision": decision,
            "notes": notes,
        }
    )


def run(args: argparse.Namespace) -> int:
    check_overwrite([args.summary, args.decision_matrix], args.overwrite)
    metrics = read_csv(args.metrics)
    groups = read_csv(args.group_summary)
    validation = read_csv(args.validation_metrics)
    formal_profile = [row for row in read_csv(FORMAL_PROFILE) if row.get("selected_seed", "").lower() == "true"]
    formal_by_sample = {row["sample_id"]: row for row in formal_profile}
    validation_pass = all(str(row.get("pass", "")).lower() == "true" for row in validation)

    decisions: list[dict[str, Any]] = []
    raw_nominal_rows = [row for row in metrics if row.get("input_mode") == "raw" and row.get("variant_name") == "nominal"]
    calibrated_nominal_rows = [row for row in metrics if row.get("input_mode") == "calibrated" and row.get("variant_name") == "nominal"]
    clean_nominal_rows = [formal_by_sample[row["base_sample_id"]] for row in raw_nominal_rows if row.get("base_sample_id") in formal_by_sample]
    raw_nominal_mean = mean(raw_nominal_rows, "profile_depth_rmse_m")
    calibrated_nominal_mean = mean(calibrated_nominal_rows, "profile_depth_rmse_m")
    clean_nominal_mean = mean(clean_nominal_rows, "profile_depth_rmse_m")
    raw_nominal_degradation = 0.0 if clean_nominal_mean <= 1.0e-20 else 100.0 * (raw_nominal_mean - clean_nominal_mean) / clean_nominal_mean
    calibrated_nominal_degradation = 0.0 if clean_nominal_mean <= 1.0e-20 else 100.0 * (calibrated_nominal_mean - clean_nominal_mean) / clean_nominal_mean
    raw_nominal_dice_drop = mean(clean_nominal_rows, "projected_mask_dice") - mean(raw_nominal_rows, "projected_mask_dice")
    nominal_status = (
        "green"
        if validation_pass
        and len(clean_nominal_rows) == len(raw_nominal_rows) == 12
        and abs(raw_nominal_degradation) <= 5.0
        and raw_nominal_dice_drop <= 0.02
        else "fail"
    )
    add(
        decisions,
        "nominal_replay",
        nominal_status,
        "regenerated COMSOL nominal vs 20.88a clean prediction artifact on same 12 base samples",
        f"{raw_nominal_degradation:.6f}% profile / {raw_nominal_dice_drop:.6f} Dice drop",
        f"{calibrated_nominal_degradation:.6f}% profile",
        "raw nominal profile <=5% and Dice drop <=0.02 vs clean artifact",
        "factor conclusions trusted" if nominal_status == "green" else "block factor conclusions",
        "not a self-comparison: compares regenerated nominal rows with original 20.88a/20.85 predictions for the same base_sample_id set",
    )

    for factor in ["liftoff", "scan_line_offset", "source_amplitude", "axis_misalignment_postprocess"]:
        raw = group_row(groups, "raw", factor)
        calibrated = group_row(groups, "calibrated", factor)
        raw_deg = mean(raw, "profile_depth_rmse_degradation_pct_vs_nominal")
        raw_dice = mean(raw, "projected_mask_dice_drop_vs_nominal")
        cal_deg = mean(calibrated, "profile_depth_rmse_degradation_pct_vs_nominal")
        cal_dice = mean(calibrated, "projected_mask_dice_drop_vs_nominal")
        raw_status = status_from(raw_deg, raw_dice)
        cal_status = status_from(cal_deg, cal_dice)
        if factor == "source_amplitude" and raw_status == "fail" and cal_status in {"green", "warning"}:
            decision = "amplitude calibration remains acquisition blocker; do not replace baseline"
        elif factor == "axis_misalignment_postprocess" and raw_status == "fail":
            decision = "require sensor alignment/resampling protocol before real-data claims"
        elif factor in {"liftoff", "scan_line_offset"} and raw_status == "fail" and cal_status == "fail":
            decision = "generate dedicated COMSOL robustness/augmentation data before internal-defect work"
        elif factor in {"liftoff", "scan_line_offset"} and raw_status == "fail":
            decision = "calibration may help but factor still needs COMSOL augmentation design"
        else:
            decision = "diagnostic acceptable; keep monitoring in robustness pack"
        add(
            decisions,
            factor,
            raw_status if raw_status == cal_status else f"raw_{raw_status}_calibrated_{cal_status}",
            "mean_test_profile_rmse_degradation_pct_vs_nominal / dice_drop",
            f"{raw_deg:.3f}% / {raw_dice:.6f}",
            f"{cal_deg:.3f}% / {cal_dice:.6f}",
            "green <=10% and Dice drop <=0.02; fail >25% or Dice drop >0.05",
            decision,
            "calibrated input is diagnostic only, selected from 20.89 validation protocol",
        )

    factor_failures = [row for row in decisions if "fail" in str(row["status"]) and row["decision_item"] != "nominal_replay"]
    liftoff_or_offset_fail = any(row["decision_item"] in {"liftoff", "scan_line_offset"} for row in factor_failures)
    misalignment_fail = any(row["decision_item"] == "axis_misalignment_postprocess" for row in factor_failures)
    source_fail = any(row["decision_item"] == "source_amplitude" for row in factor_failures)
    if not validation_pass or nominal_status == "fail":
        recommendation = "blocker: fix diagnostic pack or nominal replay before route decision"
    elif liftoff_or_offset_fail:
        recommendation = "20.90 should feed dedicated COMSOL robustness augmentation data before internal defects"
    elif misalignment_fail:
        recommendation = "define sensor alignment/resampling protocol before real-data claims"
    elif source_fail:
        recommendation = "treat amplitude calibration as acquisition blocker; keep baseline unchanged"
    else:
        recommendation = "proceed to 20.91 internal-defect feasibility design while preserving amplitude-calibration caveat"

    add(
        decisions,
        "overall_route",
        "go" if validation_pass and nominal_status != "fail" else "blocker",
        "factor decision aggregation",
        "",
        "",
        "no baseline update allowed",
        recommendation,
        "CURRENT_BASELINE remains unchanged",
    )
    write_csv(args.decision_matrix, decisions, FIELDS)

    worst = max(
        [row for row in groups if row.get("input_mode") == "raw" and row.get("split") == "test" and row.get("group_field") == "all"],
        key=lambda r: f(r, "profile_depth_rmse_degradation_pct_vs_nominal"),
    )
    lines = [
        "20.90 true 3D RBC liftoff / sensor-offset route decision",
        "",
        f"validation_pass: {validation_pass}",
        f"nominal_replay_status: {nominal_status}",
        f"nominal_replay_basis: regenerated COMSOL nominal raw rows vs 20.88a clean prediction artifact for the same 12 base_sample_id rows",
        f"nominal_replay_profile_degradation_pct_vs_clean_artifact: {raw_nominal_degradation:.6f}",
        f"nominal_replay_dice_drop_vs_clean_artifact: {raw_nominal_dice_drop:.6f}",
        f"most_sensitive_raw_factor: {worst.get('factor_group')} / {worst.get('variant_name')} degradation={f(worst, 'profile_depth_rmse_degradation_pct_vs_nominal'):.3f}% dice_drop={f(worst, 'projected_mask_dice_drop_vs_nominal'):.6f}",
        f"calibration_protocol: per_axis_rms_train_stats; diagnostic caveat only",
        f"overall_recommendation: {recommendation}",
        "baseline_update: false",
        "training_run: false",
        "COMSOL_data_committed: false",
        "",
        "Decision matrix:",
    ]
    lines.extend(
        f"- {row['decision_item']}: status={row['status']}; decision={row['decision']}; raw={row['raw_value']}; calibrated={row['calibrated_value']}"
        for row in decisions
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not validation_pass or nominal_status == "fail":
        raise RuntimeError("route decision blocked by validation/nominal replay failure")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

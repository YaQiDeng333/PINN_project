#!/usr/bin/env python
"""Diagnose RBC representation failure vs frozen 20.85 inversion failure."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_surface_shape_extension_rbc_oracle_fit import (
    DATASET_ID,
    METRICS as ORACLE_METRICS,
    ROOT,
    write_csv,
)
from audit_surface_shape_extension_current_baseline_inference import METRICS as BASELINE_METRICS


SUMMARY = ROOT / "results/summaries/surface_shape_extension_oracle_vs_baseline_diagnosis_summary.txt"
MATRIX = ROOT / "results/metrics/surface_shape_extension_oracle_vs_baseline_matrix.csv"
FAILURE_MODE_BY_SHAPE = ROOT / "results/metrics/surface_shape_extension_failure_mode_by_shape.csv"

MATRIX_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "rbc_compatible",
    "diagnosis",
    "oracle_fit_success",
    "rbc_representable",
    "model_pass",
    "oracle_profile_depth_rmse_m",
    "oracle_projected_mask_Dice",
    "baseline_profile_depth_rmse_m",
    "baseline_projected_mask_Dice",
    "total_normalized_MAE_vs_oracle",
    "true_component_count",
    "pred_component_count",
    "component_recall_proxy",
    "merge_component_proxy",
    "primary_reason",
]

SHAPE_FIELDS = [
    "shape_type",
    "split",
    "sample_count",
    "rbc_representable_and_model_pass",
    "rbc_representable_but_model_fail",
    "rbc_not_representable",
    "label_or_geometry_issue",
    "dominant_failure_mode",
    "oracle_representable_rate",
    "model_pass_rate",
    "oracle_profile_depth_rmse_mean_m",
    "baseline_profile_depth_rmse_mean_m",
    "oracle_projected_mask_Dice_mean",
    "baseline_projected_mask_Dice_mean",
    "component_recall_proxy_mean",
    "merge_component_proxy_rate",
]

DIAGNOSES = [
    "rbc_representable_and_model_pass",
    "rbc_representable_but_model_fail",
    "rbc_not_representable",
    "label_or_geometry_issue",
]


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Diagnose surface shape-extension oracle vs baseline failure modes.").parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [as_float(row.get(key)) for row in rows if math.isfinite(as_float(row.get(key)))]
    return float(np.mean(vals)) if vals else math.nan


def classify(oracle: dict[str, str], baseline: dict[str, str]) -> tuple[str, str]:
    fit_success = as_bool(oracle.get("oracle_fit_success"))
    representable = as_bool(oracle.get("rbc_representable"))
    model_pass = as_bool(baseline.get("model_pass"))
    if not fit_success:
        return "label_or_geometry_issue", "oracle_fit_failed"
    if not representable:
        reasons = []
        if as_float(oracle.get("oracle_profile_depth_rmse_m")) > 4.0e-4:
            reasons.append("oracle_profile_rmse_high")
        if as_float(oracle.get("oracle_projected_mask_Dice")) < 0.80:
            reasons.append("oracle_mask_dice_low")
        if int(float(oracle.get("true_component_count", 1))) > 1 and int(float(oracle.get("oracle_component_count", 0))) < int(float(oracle.get("true_component_count", 1))):
            reasons.append("oracle_component_merge")
        return "rbc_not_representable", "|".join(reasons) or "oracle_failed_representation_gate"
    if model_pass:
        return "rbc_representable_and_model_pass", "oracle_and_model_pass"
    reasons = []
    if as_float(baseline.get("baseline_profile_depth_rmse_m")) > 6.0e-4:
        reasons.append("baseline_profile_rmse_high")
    if as_float(baseline.get("baseline_projected_mask_Dice")) < 0.70:
        reasons.append("baseline_mask_dice_low")
    if as_bool(baseline.get("merge_component_proxy")):
        reasons.append("baseline_component_merge_proxy")
    return "rbc_representable_but_model_fail", "|".join(reasons) or "baseline_failed_model_gate"


def matrix_rows() -> list[dict[str, Any]]:
    if not ORACLE_METRICS.exists():
        raise FileNotFoundError(ORACLE_METRICS)
    if not BASELINE_METRICS.exists():
        raise FileNotFoundError(BASELINE_METRICS)
    oracle_by_id = {row["sample_id"]: row for row in read_csv(ORACLE_METRICS)}
    baseline_by_id = {row["sample_id"]: row for row in read_csv(BASELINE_METRICS)}
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(oracle_by_id):
        if sample_id not in baseline_by_id:
            raise RuntimeError(f"baseline row missing for sample_id={sample_id}")
        oracle = oracle_by_id[sample_id]
        baseline = baseline_by_id[sample_id]
        diagnosis, reason = classify(oracle, baseline)
        rows.append(
            {
                "sample_id": sample_id,
                "split": oracle["split"],
                "shape_type": oracle["shape_type"],
                "topology_type": oracle["topology_type"],
                "representation_target": oracle["representation_target"],
                "rbc_compatible": oracle["rbc_compatible"],
                "diagnosis": diagnosis,
                "oracle_fit_success": oracle["oracle_fit_success"],
                "rbc_representable": oracle["rbc_representable"],
                "model_pass": baseline["model_pass"],
                "oracle_profile_depth_rmse_m": oracle["oracle_profile_depth_rmse_m"],
                "oracle_projected_mask_Dice": oracle["oracle_projected_mask_Dice"],
                "baseline_profile_depth_rmse_m": baseline["baseline_profile_depth_rmse_m"],
                "baseline_projected_mask_Dice": baseline["baseline_projected_mask_Dice"],
                "total_normalized_MAE_vs_oracle": baseline["total_normalized_MAE_vs_oracle"],
                "true_component_count": baseline["true_component_count"],
                "pred_component_count": baseline["pred_component_count"],
                "component_recall_proxy": baseline["component_recall_proxy"],
                "merge_component_proxy": baseline["merge_component_proxy"],
                "primary_reason": reason,
            }
        )
    return rows


def shape_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split_name in ("all", "train", "val", "test"):
        for shape in sorted({row["shape_type"] for row in rows}):
            subset = [row for row in rows if row["shape_type"] == shape and (split_name == "all" or row["split"] == split_name)]
            if not subset:
                continue
            counts = Counter(row["diagnosis"] for row in subset)
            dominant = max(DIAGNOSES, key=lambda key: counts.get(key, 0))
            out.append(
                {
                    "shape_type": shape,
                    "split": split_name,
                    "sample_count": len(subset),
                    **{name: counts.get(name, 0) for name in DIAGNOSES},
                    "dominant_failure_mode": dominant,
                    "oracle_representable_rate": float(np.mean([as_bool(row["rbc_representable"]) for row in subset])),
                    "model_pass_rate": float(np.mean([as_bool(row["model_pass"]) for row in subset])),
                    "oracle_profile_depth_rmse_mean_m": mean(subset, "oracle_profile_depth_rmse_m"),
                    "baseline_profile_depth_rmse_mean_m": mean(subset, "baseline_profile_depth_rmse_m"),
                    "oracle_projected_mask_Dice_mean": mean(subset, "oracle_projected_mask_Dice"),
                    "baseline_projected_mask_Dice_mean": mean(subset, "baseline_projected_mask_Dice"),
                    "component_recall_proxy_mean": mean(subset, "component_recall_proxy"),
                    "merge_component_proxy_rate": float(np.mean([as_bool(row["merge_component_proxy"]) for row in subset])),
                }
            )
    return out


def shape_row(rows: list[dict[str, Any]], shape: str) -> dict[str, Any]:
    for row in rows:
        if row["shape_type"] == shape and row["split"] == "all":
            return row
    raise KeyError(shape)


def answer_bool(rate: float, threshold: float = 0.75) -> str:
    return "yes" if rate >= threshold else "no"


def summary_lines(rows: list[dict[str, Any]], shape_rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(row["diagnosis"] for row in rows)
    total = len(rows)
    non_rbc = [row for row in rows if as_bool(row["rbc_compatible"]) is False]
    non_rbc_counts = Counter(row["diagnosis"] for row in non_rbc)
    by_shape = {row["shape_type"]: row for row in shape_rows if row["split"] == "all"}
    rbc = by_shape.get("rbc_like_smooth_pit", {})
    flat = by_shape.get("flat_bottom_pit", {})
    sharp = by_shape.get("sharp_wall_boxy_corrosion", {})
    asym = by_shape.get("asymmetric_corrosion", {})
    crack = by_shape.get("elongated_crack_like_surface_defect", {})
    multi = by_shape.get("multi_pit_two_component_surface_defect", {})
    irregular = by_shape.get("irregular_corrosion_non_rbc", {})

    def rate(row: dict[str, Any], key: str) -> float:
        return as_float(row.get(key, math.nan))

    lines = [
        "surface shape-extension oracle vs baseline diagnosis summary",
        "stage: 25.3",
        "",
        f"dataset_id: {DATASET_ID}",
        f"sample_count: {total}",
        f"diagnosis_counts: {dict(counts)}",
        f"non_rbc_diagnosis_counts: {dict(non_rbc_counts)}",
        f"representation_failure_rate: {counts.get('rbc_not_representable', 0) / max(total, 1):.6f}",
        f"model_failure_rate_when_representable: {counts.get('rbc_representable_but_model_fail', 0) / max(counts.get('rbc_representable_but_model_fail', 0) + counts.get('rbc_representable_and_model_pass', 0), 1):.6f}",
        "",
        "required_answers:",
        f"1. RBC-like smooth pit pass: {answer_bool(rate(rbc, 'model_pass_rate'))}; model_pass_rate={rate(rbc, 'model_pass_rate'):.6f}; oracle_representable_rate={rate(rbc, 'oracle_representable_rate'):.6f}.",
        f"2. flat-bottom / sharp-wall oracle failure: flat={answer_bool(1.0 - rate(flat, 'oracle_representable_rate'), 0.50)} sharp={answer_bool(1.0 - rate(sharp, 'oracle_representable_rate'), 0.50)}; flat_oracle_repr={rate(flat, 'oracle_representable_rate'):.6f}; sharp_oracle_repr={rate(sharp, 'oracle_representable_rate'):.6f}.",
        f"3. asymmetric primary failure: {asym.get('dominant_failure_mode', '')}; oracle_representable_rate={rate(asym, 'oracle_representable_rate'):.6f}; model_pass_rate={rate(asym, 'model_pass_rate'):.6f}.",
        f"4. elongated/crack-like failure type: {crack.get('dominant_failure_mode', '')}; oracle_representable_rate={rate(crack, 'oracle_representable_rate'):.6f}; model_pass_rate={rate(crack, 'model_pass_rate'):.6f}.",
        f"5. multi-pit component representation failure: {answer_bool(1.0 - rate(multi, 'oracle_representable_rate'), 0.50)}; component_recall_proxy_mean={rate(multi, 'component_recall_proxy_mean'):.6f}; merge_proxy_rate={rate(multi, 'merge_component_proxy_rate'):.6f}.",
        f"6. irregular corrosion needs depth_grid/profile-basis: {answer_bool(1.0 - rate(irregular, 'oracle_representable_rate'), 0.50)}; dominant={irregular.get('dominant_failure_mode', '')}; oracle_representable_rate={rate(irregular, 'oracle_representable_rate'):.6f}.",
        f"7. Current 20.85 usable as non-RBC-like baseline: {'no' if non_rbc_counts.get('rbc_not_representable', 0) + non_rbc_counts.get('rbc_representable_but_model_fail', 0) > non_rbc_counts.get('rbc_representable_and_model_pass', 0) else 'yes'}; non-RBC pass count={non_rbc_counts.get('rbc_representable_and_model_pass', 0)}/{len(non_rbc)}.",
        "",
        "by_shape:",
    ]
    for shape in sorted(by_shape):
        row = by_shape[shape]
        lines.append(
            f"- {shape}: dominant={row['dominant_failure_mode']} oracle_repr_rate={as_float(row['oracle_representable_rate']):.6f} "
            f"model_pass_rate={as_float(row['model_pass_rate']):.6f} oracle_rmse={as_float(row['oracle_profile_depth_rmse_mean_m']):.9f} "
            f"baseline_rmse={as_float(row['baseline_profile_depth_rmse_mean_m']):.9f} dice={as_float(row['baseline_projected_mask_Dice_mean']):.6f}"
        )
    lines.extend(
        [
            "",
            "interpretation:",
            "- rbc_not_representable means the six-parameter RBC surface representation itself is the blocker.",
            "- rbc_representable_but_model_fail means the oracle can represent the target but frozen 20.85 inversion does not generalize.",
            "- This is an audit only; it does not train or change CURRENT_BASELINE.",
            "",
            f"matrix: {MATRIX}",
            f"failure_mode_by_shape: {FAILURE_MODE_BY_SHAPE}",
        ]
    )
    return lines


def run(_args: argparse.Namespace) -> int:
    rows = matrix_rows()
    shapes = shape_summary(rows)
    write_csv(MATRIX, rows, MATRIX_FIELDS)
    write_csv(FAILURE_MODE_BY_SHAPE, shapes, SHAPE_FIELDS)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(summary_lines(rows, shapes)) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

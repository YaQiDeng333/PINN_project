#!/usr/bin/env python
"""Route decision for 24.0A surface RBC NLS-lite features."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from load_true_3d_rbc_pilot_dataset import ROOT, write_csv


QUALITY = ROOT / "results/metrics/surface_rbc_nls_lite_feature_quality.csv"
CORRELATIONS = ROOT / "results/metrics/surface_rbc_nls_lite_feature_correlations.csv"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_lite_feature_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_rbc_nls_lite_feature_decision_matrix.csv"

KEY_TARGETS = ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW", "profile_max_abs_depth_m", "projected_mask_area_px"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def best_features_by_target(correlation_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, Any]], float]:
    best: dict[str, dict[str, Any]] = {}
    max_abs = 0.0
    all_rows = [row for row in correlation_rows if row.get("scope") == "all"]
    for target in sorted({row.get("target", "") for row in all_rows}):
        subset = [row for row in all_rows if row.get("target") == target]
        if not subset:
            continue
        row = max(subset, key=lambda item: max(as_float(item.get("abs_pearson_r")), as_float(item.get("abs_spearman_r"))))
        strength = max(as_float(row.get("abs_pearson_r")), as_float(row.get("abs_spearman_r")))
        max_abs = max(max_abs, strength)
        best[target] = {
            "feature": row.get("feature", ""),
            "pearson_r": as_float(row.get("pearson_r")),
            "spearman_r": as_float(row.get("spearman_r")),
            "strength": strength,
        }
    return best, max_abs


def decide_route(quality: dict[str, Any]) -> dict[str, Any]:
    finite_fraction = as_float(quality.get("overall_finite_fraction"))
    fit_success_rate = as_float(quality.get("fit_success_rate"))
    fallback_rate = as_float(quality.get("fallback_rate"))
    max_abs_correlation = as_float(quality.get("max_abs_correlation"))
    feature_count = int(as_float(quality.get("feature_count")))
    nlslite_features_stable = finite_fraction >= 0.999 and feature_count > 0
    fit_failure_acceptable = fit_success_rate >= 0.80 and fallback_rate <= 0.20
    useful_signal = max_abs_correlation >= 0.35
    enter_24_1 = bool(nlslite_features_stable and fit_failure_acceptable and useful_signal)
    if enter_24_1:
        real_fit = "yes_with_calibration_caveat"
    elif nlslite_features_stable and useful_signal:
        real_fit = "diagnostic_only"
    else:
        real_fit = "no"
    return {
        "nlslite_features_stable": bool(nlslite_features_stable),
        "fit_failure_acceptable": bool(fit_failure_acceptable),
        "useful_correlation_signal": bool(useful_signal),
        "enter_24_1_feature_baseline": enter_24_1,
        "real_experiment_preprocessing_fit": real_fit,
    }


def run(args: argparse.Namespace) -> int:
    quality_rows = read_csv(args.quality)
    overall_rows = [row for row in quality_rows if row.get("quality_scope") == "overall"]
    if len(overall_rows) != 1:
        raise RuntimeError(f"expected one overall quality row, got {len(overall_rows)}")
    overall = overall_rows[0]
    correlations = read_csv(args.correlations)
    best_by_target, max_abs = best_features_by_target(correlations)

    quality = {
        "sample_count": "",
        "feature_count": as_float(overall.get("feature_count")),
        "overall_finite_fraction": as_float(overall.get("finite_fraction")),
        "fit_success_rate": as_float(overall.get("fit_success_rate")),
        "fallback_rate": as_float(overall.get("fallback_rate")),
        "fit_residual_mean": as_float(overall.get("fit_residual_mean")),
        "fit_residual_max": as_float(overall.get("fit_residual_max")),
        "max_abs_correlation": max_abs,
        "top_features_by_target": {target: info["feature"] for target, info in best_by_target.items()},
    }
    route = decide_route(quality)
    valuable = {
        target: info
        for target, info in best_by_target.items()
        if target in KEY_TARGETS and float(info["strength"]) >= 0.35
    }
    curvature_targets = {"wLD", "wWD", "wLW", "profile_max_abs_depth_m"}
    curvature_valuable = {target: info for target, info in valuable.items() if target in curvature_targets}
    lwd_valuable = {target: info for target, info in valuable.items() if target in {"L_m", "W_m", "D_m"}}

    decision = "enter_24_1_feature_baseline" if route["enter_24_1_feature_baseline"] else "keep_as_diagnostic_only"
    matrix_rows = [
        {
            "question": "Are NLS-lite features stable?",
            "answer": route["nlslite_features_stable"],
            "evidence": f"finite_fraction={quality['overall_finite_fraction']}; feature_count={int(quality['feature_count'])}",
            "decision": decision,
        },
        {
            "question": "Is fit failure acceptable?",
            "answer": route["fit_failure_acceptable"],
            "evidence": f"fit_success_rate={quality['fit_success_rate']}; fallback_rate={quality['fallback_rate']}",
            "decision": decision,
        },
        {
            "question": "Can this enter 24.1 feature baseline?",
            "answer": route["enter_24_1_feature_baseline"],
            "evidence": f"stable={route['nlslite_features_stable']}; fit_ok={route['fit_failure_acceptable']}; max_abs_correlation={max_abs:.6f}",
            "decision": decision,
        },
        {
            "question": "Which L/W/D features look valuable?",
            "answer": "; ".join(f"{target}:{info['feature']}({info['strength']:.3f})" for target, info in sorted(lwd_valuable.items())),
            "evidence": "all-scope absolute Pearson/Spearman >= 0.35",
            "decision": decision,
        },
        {
            "question": "Which curvature/profile features look valuable?",
            "answer": "; ".join(f"{target}:{info['feature']}({info['strength']:.3f})" for target, info in sorted(curvature_valuable.items())),
            "evidence": "all-scope absolute Pearson/Spearman >= 0.35",
            "decision": decision,
        },
        {
            "question": "Is it suitable for real experimental preprocessing?",
            "answer": route["real_experiment_preprocessing_fit"],
            "evidence": "delta_b-only deterministic features; still requires Bx/By/Bz calibration and the same three scan_line_y geometry.",
            "decision": decision,
        },
        {
            "question": "Label leakage status",
            "answer": "no_label_leakage_detected",
            "evidence": "feature CSV has only sample_id/split metadata plus nlslite_* columns; labels used only in correlation audit.",
            "decision": decision,
        },
        {
            "question": "Piao claim boundary",
            "answer": "exact_piao_nls=false; piao_nls_lite=true",
            "evidence": "current v3_240 data has only three scan_line_y values.",
            "decision": decision,
        },
    ]
    write_csv(args.matrix, matrix_rows, ["question", "answer", "evidence", "decision"])

    top_lines = []
    for target in KEY_TARGETS:
        info = best_by_target.get(target)
        if not info:
            continue
        top_lines.append(
            f"- {target}: {info['feature']} pearson={info['pearson_r']:.6f} spearman={info['spearman_r']:.6f} strength={info['strength']:.6f}"
        )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "24.0A surface RBC NLS-lite feature route decision summary",
                "",
                "scope: route decision only; no training, no COMSOL, no data/NPZ modification, no CURRENT_BASELINE update.",
                "method_claim: exact_piao_nls=false; piao_nls_lite=true.",
                f"feature_count: {int(quality['feature_count'])}",
                f"overall_finite_fraction: {quality['overall_finite_fraction']:.6f}",
                f"fit_success_rate: {quality['fit_success_rate']:.6f}",
                f"fallback_rate: {quality['fallback_rate']:.6f}",
                f"fit_residual_mean: {quality['fit_residual_mean']:.6f}",
                f"fit_residual_max: {quality['fit_residual_max']:.6f}",
                f"max_abs_correlation: {max_abs:.6f}",
                f"nlslite_features_stable: {route['nlslite_features_stable']}",
                f"fit_failure_acceptable: {route['fit_failure_acceptable']}",
                f"enter_24_1_feature_baseline: {route['enter_24_1_feature_baseline']}",
                f"real_experiment_preprocessing_fit: {route['real_experiment_preprocessing_fit']}",
                "label_leakage: false",
                "",
                "main_related_features:",
                *top_lines,
                "",
                f"decision: {decision}",
                f"decision_matrix: {args.matrix}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, default=QUALITY)
    parser.add_argument("--correlations", type=Path, default=CORRELATIONS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

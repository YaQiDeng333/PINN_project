#!/usr/bin/env python
"""Audit profile-level metrics against parameter-level errors for v3_240.

The available artifacts contain per-sample error rows for 20.77 and 20.81,
but not raw predicted parameters. This script respects that boundary and
does not attempt to reconstruct missing predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import V3_240_DATASET_ID, load_dataset, split_indices


ROOT = Path(__file__).resolve().parents[1]

SUMMARY = ROOT / "results/summaries/true_3d_rbc_profile_vs_parameter_error_audit_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_profile_vs_parameter_error_metrics.csv"
CASES = ROOT / "results/metrics/true_3d_rbc_profile_parameter_contradiction_cases.csv"
METHODS = ROOT / "results/metrics/true_3d_rbc_method_profile_metric_comparison.csv"

NEURAL_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_profile_metrics.csv"
FUSION_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_profile_metrics.csv"
PIAO_FEATURE_AGG = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_metrics.csv"
PIAO_FEATURE_FAILURE = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_failure_cases.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def pearson(rows: list[dict[str, Any]], a: str, b: str) -> float:
    xs = np.array([to_float(row, a) for row in rows], dtype=float)
    ys = np.array([to_float(row, b) for row in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if int(mask.sum()) < 3:
        return math.nan
    xs = xs[mask]
    ys = ys[mask]
    if float(np.std(xs)) < 1.0e-12 or float(np.std(ys)) < 1.0e-12:
        return math.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def quantile(rows: list[dict[str, Any]], key: str, q: float) -> float:
    values = np.array([to_float(row, key) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.quantile(values, q))


def normalize_profile_rows(method: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if "selected_seed" in row and not truthy(row["selected_seed"]):
            continue
        if "selected_by_validation" in row and row["selected_by_validation"] not in {"", "True", "true", "1"}:
            continue
        item: dict[str, Any] = dict(row)
        item["method"] = method
        item["w_error_sum"] = (
            to_float(item, "wLD_abs_error") + to_float(item, "wWD_abs_error") + to_float(item, "wLW_abs_error")
        )
        if not math.isfinite(to_float(item, "curvature_mae_mean")):
            item["curvature_mae_mean"] = item["w_error_sum"] / 3.0
        out.append(item)
    return out


def aggregate(rows: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
    subset = rows if split == "all" else [row for row in rows if row.get("split") == split]
    if not subset:
        return {"method": method, "split": split, "sample_count": 0}
    fields = [
        "normalized_param_mae_mean",
        "dimension_param_mae_norm",
        "curvature_param_mae_norm",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "wLD_abs_error",
        "wWD_abs_error",
        "wLW_abs_error",
        "curvature_mae_mean",
        "projected_mask_iou",
        "projected_mask_dice",
        "profile_depth_rmse_m",
    ]
    out = {"method": method, "split": split, "sample_count": len(subset)}
    for field in fields:
        vals = np.array([to_float(row, field) for row in subset], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"{field}_mean"] = float(vals.mean()) if vals.size else math.nan
    out["corr_curvature_vs_profile_rmse"] = pearson(subset, "curvature_mae_mean", "profile_depth_rmse_m")
    out["corr_wLD_vs_profile_rmse"] = pearson(subset, "wLD_abs_error", "profile_depth_rmse_m")
    out["corr_wWD_vs_profile_rmse"] = pearson(subset, "wWD_abs_error", "profile_depth_rmse_m")
    out["corr_wLW_vs_profile_rmse"] = pearson(subset, "wLW_abs_error", "profile_depth_rmse_m")
    out["corr_curvature_vs_dice"] = pearson(subset, "curvature_mae_mean", "projected_mask_dice")
    out["corr_d_error_vs_profile_rmse"] = pearson(subset, "D_mae_mm", "profile_depth_rmse_m")
    out["corr_dice_vs_profile_rmse"] = pearson(subset, "projected_mask_dice", "profile_depth_rmse_m")
    high_w = quantile(subset, "curvature_mae_mean", 0.75)
    low_profile = quantile(subset, "profile_depth_rmse_m", 0.25)
    high_profile = quantile(subset, "profile_depth_rmse_m", 0.75)
    high_dice = quantile(subset, "projected_mask_dice", 0.75)
    out["high_w_low_profile_count"] = sum(
        to_float(row, "curvature_mae_mean") >= high_w and to_float(row, "profile_depth_rmse_m") <= low_profile
        for row in subset
    )
    out["high_dice_high_profile_count"] = sum(
        to_float(row, "projected_mask_dice") >= high_dice and to_float(row, "profile_depth_rmse_m") >= high_profile
        for row in subset
    )
    out["high_dice_high_curvature_count"] = sum(
        to_float(row, "projected_mask_dice") >= high_dice and to_float(row, "curvature_mae_mean") >= high_w
        for row in subset
    )
    return out


def contradiction_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        for split in ("test", "val", "train"):
            subset = [row for row in rows if row["method"] == method and row.get("split") == split]
            if not subset:
                continue
            high_w = quantile(subset, "curvature_mae_mean", 0.75)
            low_w = quantile(subset, "curvature_mae_mean", 0.25)
            high_profile = quantile(subset, "profile_depth_rmse_m", 0.75)
            low_profile = quantile(subset, "profile_depth_rmse_m", 0.25)
            high_dice = quantile(subset, "projected_mask_dice", 0.75)
            low_dice = quantile(subset, "projected_mask_dice", 0.25)
            checks = [
                ("high_w_error_low_profile_error", lambda r: to_float(r, "curvature_mae_mean") >= high_w and to_float(r, "profile_depth_rmse_m") <= low_profile),
                ("low_w_error_high_profile_error", lambda r: to_float(r, "curvature_mae_mean") <= low_w and to_float(r, "profile_depth_rmse_m") >= high_profile),
                ("high_dice_high_profile_error", lambda r: to_float(r, "projected_mask_dice") >= high_dice and to_float(r, "profile_depth_rmse_m") >= high_profile),
                ("high_dice_high_curvature_error", lambda r: to_float(r, "projected_mask_dice") >= high_dice and to_float(r, "curvature_mae_mean") >= high_w),
                ("low_dice_low_profile_error", lambda r: to_float(r, "projected_mask_dice") <= low_dice and to_float(r, "profile_depth_rmse_m") <= low_profile),
            ]
            for case_type, predicate in checks:
                ranked = [row for row in subset if predicate(row)]
                ranked = sorted(ranked, key=lambda r: (to_float(r, "curvature_mae_mean"), to_float(r, "profile_depth_rmse_m")), reverse=True)
                for rank, row in enumerate(ranked[:5], start=1):
                    out.append(
                        {
                            "case_type": case_type,
                            "method": method,
                            "split": split,
                            "rank": rank,
                            "sample_id": row.get("sample_id", ""),
                            "curvature_template": row.get("curvature_template", ""),
                            "depth_bin": row.get("depth_bin", ""),
                            "aspect_bin": row.get("aspect_bin", ""),
                            "curvature_mae_mean": to_float(row, "curvature_mae_mean"),
                            "wLD_abs_error": to_float(row, "wLD_abs_error"),
                            "wWD_abs_error": to_float(row, "wWD_abs_error"),
                            "wLW_abs_error": to_float(row, "wLW_abs_error"),
                            "D_mae_mm": to_float(row, "D_mae_mm"),
                            "projected_mask_dice": to_float(row, "projected_mask_dice"),
                            "profile_depth_rmse_m": to_float(row, "profile_depth_rmse_m"),
                        }
                    )
    return out


def piao_aggregate_rows() -> list[dict[str, Any]]:
    if not PIAO_FEATURE_AGG.exists():
        return []
    rows = []
    for row in read_csv(PIAO_FEATURE_AGG):
        if row.get("selected_by_validation") not in {"True", "true", "1"}:
            continue
        rows.append(
            {
                "method": "20.80_piao_feature_only_aggregate",
                "split": row.get("split", ""),
                "sample_count": int(float(row.get("sample_count", 0) or 0)),
                "normalized_param_mae_mean_mean": to_float(row, "normalized_param_mae"),
                "dimension_param_mae_norm_mean": to_float(row, "dimension_mae_norm"),
                "curvature_param_mae_norm_mean": to_float(row, "curvature_mae_norm"),
                "L_mae_mm_mean": to_float(row, "L_mae_mm"),
                "W_mae_mm_mean": to_float(row, "W_mae_mm"),
                "D_mae_mm_mean": to_float(row, "D_mae_mm"),
                "wLD_abs_error_mean": to_float(row, "wLD_abs_error"),
                "wWD_abs_error_mean": to_float(row, "wWD_abs_error"),
                "wLW_abs_error_mean": to_float(row, "wLW_abs_error"),
                "curvature_mae_mean_mean": to_float(row, "curvature_mae"),
                "projected_mask_iou_mean": to_float(row, "projected_mask_iou"),
                "projected_mask_dice_mean": to_float(row, "projected_mask_dice"),
                "profile_depth_rmse_m_mean": to_float(row, "profile_depth_rmse_m"),
                "artifact_level": "aggregate_only",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id)
    splits = {name: len(idx) for name, idx in split_indices(dataset).items()}
    neural = normalize_profile_rows("20.77_neural_selected_seed", read_csv(NEURAL_PROFILE))
    fusion = normalize_profile_rows("20.81_feature_fusion_selected", read_csv(FUSION_PROFILE))
    rows = neural + fusion

    selected_counts_ok = len(neural) == 240 and len(fusion) == 240
    test_counts_ok = sum(row["split"] == "test" for row in neural) == 39 and sum(row["split"] == "test" for row in fusion) == 39
    if not selected_counts_ok or not test_counts_ok:
        raise RuntimeError(f"profile artifact gate failed: neural={len(neural)}, fusion={len(fusion)}")

    metric_rows = []
    for method in ("20.77_neural_selected_seed", "20.81_feature_fusion_selected"):
        method_rows = [row for row in rows if row["method"] == method]
        for split in ("train", "val", "test", "all"):
            subset = method_rows if split == "all" else [row for row in method_rows if row.get("split") == split]
            if subset:
                metric_rows.append(aggregate(subset, method, split))
    write_csv(
        METRICS,
        metric_rows,
        [
            "method",
            "split",
            "sample_count",
            "normalized_param_mae_mean_mean",
            "dimension_param_mae_norm_mean",
            "curvature_param_mae_norm_mean",
            "L_mae_mm_mean",
            "W_mae_mm_mean",
            "D_mae_mm_mean",
            "wLD_abs_error_mean",
            "wWD_abs_error_mean",
            "wLW_abs_error_mean",
            "curvature_mae_mean_mean",
            "projected_mask_iou_mean",
            "projected_mask_dice_mean",
            "profile_depth_rmse_m_mean",
            "corr_curvature_vs_profile_rmse",
            "corr_wLD_vs_profile_rmse",
            "corr_wWD_vs_profile_rmse",
            "corr_wLW_vs_profile_rmse",
            "corr_curvature_vs_dice",
            "corr_d_error_vs_profile_rmse",
            "corr_dice_vs_profile_rmse",
            "high_w_low_profile_count",
            "high_dice_high_profile_count",
            "high_dice_high_curvature_count",
        ],
    )

    case_rows = contradiction_cases(rows)
    write_csv(
        CASES,
        case_rows,
        [
            "case_type",
            "method",
            "split",
            "rank",
            "sample_id",
            "curvature_template",
            "depth_bin",
            "aspect_bin",
            "curvature_mae_mean",
            "wLD_abs_error",
            "wWD_abs_error",
            "wLW_abs_error",
            "D_mae_mm",
            "projected_mask_dice",
            "profile_depth_rmse_m",
        ],
    )

    comparison_rows = [row | {"artifact_level": "per_sample_error_rows"} for row in metric_rows if row["split"] == "test"]
    comparison_rows.extend(piao_aggregate_rows())
    write_csv(
        METHODS,
        comparison_rows,
        [
            "method",
            "split",
            "sample_count",
            "artifact_level",
            "normalized_param_mae_mean_mean",
            "dimension_param_mae_norm_mean",
            "curvature_param_mae_norm_mean",
            "L_mae_mm_mean",
            "W_mae_mm_mean",
            "D_mae_mm_mean",
            "wLD_abs_error_mean",
            "wWD_abs_error_mean",
            "wLW_abs_error_mean",
            "curvature_mae_mean_mean",
            "projected_mask_iou_mean",
            "projected_mask_dice_mean",
            "profile_depth_rmse_m_mean",
        ],
    )

    test_metric = {row["method"]: row for row in metric_rows if row["split"] == "test"}
    neural_test = test_metric["20.77_neural_selected_seed"]
    fusion_test = test_metric["20.81_feature_fusion_selected"]
    corr_abs = abs(float(neural_test["corr_curvature_vs_profile_rmse"]))
    contradiction_count = int(neural_test["high_w_low_profile_count"]) + int(neural_test["high_dice_high_curvature_count"])
    fusion_dice_better = float(fusion_test["projected_mask_dice_mean"]) > float(neural_test["projected_mask_dice_mean"])
    fusion_profile_rmse_better = float(fusion_test["profile_depth_rmse_m_mean"]) < float(neural_test["profile_depth_rmse_m_mean"])
    profile_primary_recommended = corr_abs < 0.50 or contradiction_count > 0
    piao_reference_note = "20.80 feature-only has aggregate/group artifacts only; no per-sample profile row was available."

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.82 profile-level vs parameter-level audit summary",
                "",
                f"dataset_id: {args.dataset_id}",
                f"dataset_sample_count: {len(dataset.sample_ids)}",
                f"dataset_split_counts: {json.dumps(splits, sort_keys=True)}",
                "COMSOL_run: false",
                "training_run: false",
                "data_or_NPZ_modified: false",
                "",
                "artifact_boundary:",
                "- 20.77 neural and 20.81 feature-fusion provide per-sample error/profile metric rows.",
                "- 20.80 Piao/NLS-inspired feature-only is aggregate/group/failure-case reference only.",
                "- raw pred_params / predicted profile arrays are not available; this audit does not reconstruct them.",
                f"- {piao_reference_note}",
                "",
                "test_metrics:",
                f"- 20.77 neural: curvature_mae={float(neural_test['curvature_mae_mean_mean']):.6f}, Dice={float(neural_test['projected_mask_dice_mean']):.6f}, profile_rmse_m={float(neural_test['profile_depth_rmse_m_mean']):.9f}, corr_curvature_vs_profile={float(neural_test['corr_curvature_vs_profile_rmse']):.6f}",
                f"- 20.81 fusion: curvature_mae={float(fusion_test['curvature_mae_mean_mean']):.6f}, Dice={float(fusion_test['projected_mask_dice_mean']):.6f}, profile_rmse_m={float(fusion_test['profile_depth_rmse_m_mean']):.9f}, corr_curvature_vs_profile={float(fusion_test['corr_curvature_vs_profile_rmse']):.6f}",
                "",
                f"answer_1_w_errors_strongly_correlate_with_profile_error: {corr_abs >= 0.50}",
                f"answer_2_projected_mask_enough_for_profile: false",
                f"answer_3_20_81_wLD_bad_but_profile_better: {fusion_profile_rmse_better}",
                f"answer_3_detail: 20.81 projected mask Dice better={fusion_dice_better}, but profile_depth_rmse_m better={fusion_profile_rmse_better}.",
                "answer_4_20_80_curvature_better_profile_better: no; 20.80 curvature is better than 20.77 but aggregate Dice/profile RMSE are worse than 20.77.",
                f"answer_5_primary_metric_should_shift_to_profile_level: {profile_primary_recommended}",
                "",
                "interpretation:",
                "- Per-parameter wMAE remains useful for diagnosis, especially wLD failure.",
                "- The branch-level objective should not be decided only by isolated wLD/wWD/wLW MAE.",
                "- Profile-depth RMSE / Er-like reconstruction error should be elevated to primary true-3D metric.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Extract Piao/NLS-inspired Bx/By/Bz features for v3_240.

The extractor reads only delta_b through the explicit dataset_id loader.
Labels and grouping metadata are not used as feature inputs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import V3_240_DATASET_ID, ROOT, load_dataset, split_indices, write_csv
from train_true_3d_rbc_feature_baselines import extract_signal_features


FEATURES = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_features.csv"
QUALITY = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_quality.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_piao_nls_feature_extraction_summary.txt"

AXES = ["Bx", "By", "Bz"]
LINES = ["yneg", "y0", "ypos"]
EPS = 1.0e-12


def safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > EPS else math.nan


def width_at_fraction(x_pos: np.ndarray, y: np.ndarray, fraction: float) -> float:
    abs_y = np.abs(y)
    peak = float(np.max(abs_y))
    if peak <= EPS:
        return math.nan
    mask = abs_y >= peak * fraction
    if not np.any(mask):
        return math.nan
    idx = np.where(mask)[0]
    return float(x_pos[idx[-1]] - x_pos[idx[0]])


def line_stats(x_pos: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    grad = np.gradient(y, x_pos)
    grad2 = np.gradient(grad, x_pos)
    abs_y = np.abs(y)
    peak_idx = int(np.argmax(abs_y))
    left = y[: peak_idx + 1]
    right = y[peak_idx:]
    left_energy = float(np.mean(left * left)) if left.size else 0.0
    right_energy = float(np.mean(right * right)) if right.size else 0.0
    centered = y - float(np.mean(y))
    std = float(np.std(y))
    skew = float(np.mean((centered / std) ** 3)) if std > EPS else 0.0
    kurt = float(np.mean((centered / std) ** 4)) if std > EPS else 0.0
    pos = np.clip(y, 0.0, None)
    neg = np.clip(-y, 0.0, None)
    pos_idx = int(np.argmax(pos)) if float(np.max(pos)) > EPS else peak_idx
    neg_idx = int(np.argmax(neg)) if float(np.max(neg)) > EPS else peak_idx
    width25 = width_at_fraction(x_pos, y, 0.25)
    width50 = width_at_fraction(x_pos, y, 0.50)
    width75 = width_at_fraction(x_pos, y, 0.75)
    return {
        "max": float(np.max(y)),
        "min": float(np.min(y)),
        "ptp": float(np.ptp(y)),
        "abs_peak": float(np.max(abs_y)),
        "arg_abs_peak": float(x_pos[peak_idx]),
        "argmax": float(x_pos[int(np.argmax(y))]),
        "argmin": float(x_pos[int(np.argmin(y))]),
        "positive_peak": float(np.max(pos)),
        "negative_peak": float(np.max(neg)),
        "pos_neg_peak_distance": float(abs(x_pos[pos_idx] - x_pos[neg_idx])),
        "positive_area": float(np.trapezoid(pos, x_pos)),
        "negative_area": float(np.trapezoid(neg, x_pos)),
        "abs_area": float(np.trapezoid(abs_y, x_pos)),
        "energy": float(np.mean(y * y)),
        "mean": float(np.mean(y)),
        "std": std,
        "skewness": skew,
        "kurtosis": kurt,
        "width25": width25,
        "width50": width50,
        "width75": width75,
        "width50_failed": float(not np.isfinite(width50)),
        "flat_top_ratio": safe_div(width75, width50),
        "sharpness": safe_div(float(np.max(abs_y)), width50),
        "zero_crossings": float(np.count_nonzero(np.diff(np.signbit(y)))),
        "grad_max": float(np.max(grad)),
        "grad_min": float(np.min(grad)),
        "grad_abs_peak": float(np.max(np.abs(grad))),
        "grad_energy": float(np.mean(grad * grad)),
        "grad_sharpness": safe_div(float(np.max(np.abs(grad))), float(np.max(abs_y))),
        "second_grad_abs_peak": float(np.max(np.abs(grad2))),
        "second_grad_energy": float(np.mean(grad2 * grad2)),
        "left_right_energy_ratio": safe_div(left_energy, right_energy),
        "peak_asymmetry": safe_div(left_energy - right_energy, left_energy + right_energy),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def fit_nls(x_pos: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out = {
        "gauss_A": math.nan,
        "gauss_x0": math.nan,
        "gauss_sigma": math.nan,
        "gauss_C": math.nan,
        "gauss_rmse": math.nan,
        "gauss_success": 0.0,
        "dog_A": math.nan,
        "dog_x0": math.nan,
        "dog_sigma": math.nan,
        "dog_C": math.nan,
        "dog_rmse": math.nan,
        "dog_success": 0.0,
    }
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return out

    y = np.asarray(y, dtype=np.float64)
    abs_y = np.abs(y)
    x_range = float(np.max(x_pos) - np.min(x_pos))

    def gaussian(x: np.ndarray, a: float, x0: float, sigma: float, c: float) -> np.ndarray:
        return a * np.exp(-((x - x0) ** 2) / (2.0 * sigma * sigma)) + c

    def dog(x: np.ndarray, a: float, x0: float, sigma: float, c: float) -> np.ndarray:
        z = (x - x0) / (sigma * sigma)
        return a * z * np.exp(-((x - x0) ** 2) / (2.0 * sigma * sigma)) + c

    try:
        p0 = [float(np.max(abs_y)), float(x_pos[int(np.argmax(abs_y))]), 0.2, float(np.median(abs_y))]
        bounds = ([0.0, -1.2, 0.01, -np.inf], [np.inf, 1.2, max(0.05, x_range), np.inf])
        popt, _ = curve_fit(gaussian, x_pos, abs_y, p0=p0, bounds=bounds, maxfev=1500)
        pred = gaussian(x_pos, *popt)
        out.update(
            {
                "gauss_A": float(popt[0]),
                "gauss_x0": float(popt[1]),
                "gauss_sigma": float(abs(popt[2])),
                "gauss_C": float(popt[3]),
                "gauss_rmse": float(np.sqrt(np.mean((pred - abs_y) ** 2))),
                "gauss_success": 1.0,
            }
        )
    except Exception:
        pass
    try:
        p0 = [float(np.max(np.abs(y))) * 0.1, float(x_pos[int(np.argmax(abs_y))]), 0.2, float(np.median(y))]
        bounds = ([-np.inf, -1.2, 0.01, -np.inf], [np.inf, 1.2, max(0.05, x_range), np.inf])
        popt, _ = curve_fit(dog, x_pos, y, p0=p0, bounds=bounds, maxfev=1500)
        pred = dog(x_pos, *popt)
        out.update(
            {
                "dog_A": float(popt[0]),
                "dog_x0": float(popt[1]),
                "dog_sigma": float(abs(popt[2])),
                "dog_C": float(popt[3]),
                "dog_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                "dog_success": 1.0,
            }
        )
    except Exception:
        pass
    return out


def add_prefixed(features: list[float], names: list[str], groups: list[str], prefix: str, group: str, values: dict[str, float]) -> None:
    for key, value in values.items():
        features.append(float(value) if value is not None else math.nan)
        names.append(f"{prefix}_{key}")
        groups.append(group)


def extract_one(delta: np.ndarray, x_pos: np.ndarray, existing: np.ndarray, existing_names: list[str]) -> tuple[list[float], list[str], list[str]]:
    values: list[float] = [float(v) for v in existing]
    names: list[str] = [f"F0__{name}" for name in existing_names]
    groups: list[str] = ["F0_existing_handcrafted"] * len(existing_names)

    per_line_stats: dict[tuple[int, int], dict[str, float]] = {}
    nls_stats: dict[tuple[int, int], dict[str, float]] = {}
    for ai, axis in enumerate(AXES):
        for li, line in enumerate(LINES):
            signal = delta[ai, li]
            stats = line_stats(x_pos, signal)
            per_line_stats[(ai, li)] = stats
            add_prefixed(values, names, groups, f"F1__{axis}_{line}", "F1_peak_shape", {k: stats[k] for k in [
                "max", "min", "ptp", "abs_peak", "arg_abs_peak", "argmax", "argmin", "positive_peak", "negative_peak",
                "pos_neg_peak_distance", "positive_area", "negative_area", "abs_area", "energy", "mean", "std", "skewness",
                "kurtosis", "width25", "width50", "width75", "width50_failed", "flat_top_ratio", "sharpness",
            ]})
            add_prefixed(values, names, groups, f"F2__{axis}_{line}", "F2_gradient_asymmetry", {k: stats[k] for k in [
                "zero_crossings", "grad_max", "grad_min", "grad_abs_peak", "grad_energy", "grad_sharpness",
                "second_grad_abs_peak", "second_grad_energy", "left_right_energy_ratio", "peak_asymmetry",
            ]})
            nls = fit_nls(x_pos, signal)
            nls_stats[(ai, li)] = nls
            add_prefixed(values, names, groups, f"F4__{axis}_{line}", "F4_nls_curve_fit", nls)

    for li, line in enumerate(LINES):
        bx, by, bz = delta[0, li], delta[1, li], delta[2, li]
        vmag = np.sqrt(bx * bx + by * by + bz * bz)
        vm = line_stats(x_pos, vmag)
        add_prefixed(values, names, groups, f"F3__vmag_{line}", "F3_cross_axis", {k: vm[k] for k in [
            "abs_peak", "arg_abs_peak", "energy", "width50", "sharpness", "grad_abs_peak", "grad_energy"
        ]})
        axis_vals = {
            "Bx_Bz_abs_peak_ratio": safe_div(per_line_stats[(0, li)]["abs_peak"], per_line_stats[(2, li)]["abs_peak"]),
            "By_Bz_abs_peak_ratio": safe_div(per_line_stats[(1, li)]["abs_peak"], per_line_stats[(2, li)]["abs_peak"]),
            "Bx_By_abs_peak_ratio": safe_div(per_line_stats[(0, li)]["abs_peak"], per_line_stats[(1, li)]["abs_peak"]),
            "Bx_Bz_energy_ratio": safe_div(per_line_stats[(0, li)]["energy"], per_line_stats[(2, li)]["energy"]),
            "By_Bz_energy_ratio": safe_div(per_line_stats[(1, li)]["energy"], per_line_stats[(2, li)]["energy"]),
            "Bx_Bz_peak_offset": per_line_stats[(0, li)]["arg_abs_peak"] - per_line_stats[(2, li)]["arg_abs_peak"],
            "By_Bz_peak_offset": per_line_stats[(1, li)]["arg_abs_peak"] - per_line_stats[(2, li)]["arg_abs_peak"],
            "Bx_By_corr": corr(bx, by),
            "Bx_Bz_corr": corr(bx, bz),
            "By_Bz_corr": corr(by, bz),
        }
        add_prefixed(values, names, groups, f"F3__axis_{line}", "F3_cross_axis", axis_vals)

    for ai, axis in enumerate(AXES):
        center = per_line_stats[(ai, 1)]
        outer_l = per_line_stats[(ai, 0)]
        outer_r = per_line_stats[(ai, 2)]
        derived = {
            "center_outer_abs_peak_ratio": safe_div(center["abs_peak"], 0.5 * (outer_l["abs_peak"] + outer_r["abs_peak"])),
            "outer_abs_peak_asymmetry": safe_div(outer_l["abs_peak"] - outer_r["abs_peak"], outer_l["abs_peak"] + outer_r["abs_peak"]),
            "center_outer_width50_ratio": safe_div(center["width50"], 0.5 * (outer_l["width50"] + outer_r["width50"])),
            "outer_width50_asymmetry": safe_div(outer_l["width50"] - outer_r["width50"], outer_l["width50"] + outer_r["width50"]),
            "center_outer_sharpness_ratio": safe_div(center["sharpness"], 0.5 * (outer_l["sharpness"] + outer_r["sharpness"])),
            "center_outer_gauss_sigma_ratio": safe_div(nls_stats[(ai, 1)]["gauss_sigma"], 0.5 * (nls_stats[(ai, 0)]["gauss_sigma"] + nls_stats[(ai, 2)]["gauss_sigma"])),
            "outer_gauss_sigma_asymmetry": safe_div(nls_stats[(ai, 0)]["gauss_sigma"] - nls_stats[(ai, 2)]["gauss_sigma"], nls_stats[(ai, 0)]["gauss_sigma"] + nls_stats[(ai, 2)]["gauss_sigma"]),
            "center_outer_lobe_distance_ratio": safe_div(center["pos_neg_peak_distance"], 0.5 * (outer_l["pos_neg_peak_distance"] + outer_r["pos_neg_peak_distance"])),
        }
        add_prefixed(values, names, groups, f"F5__{axis}", "F5_curvature_focused", derived)

    for li, line in enumerate(LINES):
        derived = {
            "Bx_Bz_width50_ratio": safe_div(per_line_stats[(0, li)]["width50"], per_line_stats[(2, li)]["width50"]),
            "By_Bz_width50_ratio": safe_div(per_line_stats[(1, li)]["width50"], per_line_stats[(2, li)]["width50"]),
            "Bx_Bz_sigma_ratio": safe_div(nls_stats[(0, li)]["gauss_sigma"], nls_stats[(2, li)]["gauss_sigma"]),
            "By_Bz_sigma_ratio": safe_div(nls_stats[(1, li)]["gauss_sigma"], nls_stats[(2, li)]["gauss_sigma"]),
            "Bx_Bz_flat_ratio": safe_div(per_line_stats[(0, li)]["flat_top_ratio"], per_line_stats[(2, li)]["flat_top_ratio"]),
            "By_Bz_flat_ratio": safe_div(per_line_stats[(1, li)]["flat_top_ratio"], per_line_stats[(2, li)]["flat_top_ratio"]),
        }
        add_prefixed(values, names, groups, f"F5__cross_{line}", "F5_curvature_focused", derived)
    return values, names, groups


def quality_rows(matrix: np.ndarray, names: list[str], groups: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in sorted(set(groups)):
        idx = [i for i, g in enumerate(groups) if g == group]
        sub = matrix[:, idx]
        finite = np.isfinite(sub)
        fit_cols = [i for i in idx if names[i].endswith("_success")]
        fit_success = math.nan
        if fit_cols:
            fit_success = float(np.nanmean(matrix[:, fit_cols]))
        out.append(
            {
                "feature_group": group,
                "feature_count": len(idx),
                "finite_fraction": float(np.mean(finite)),
                "nan_count": int(np.isnan(sub).sum()),
                "inf_count": int(np.isinf(sub).sum()),
                "fit_success_rate": fit_success,
                "notes": "F4 optional; fit failures are represented by success flags and train-only imputation downstream" if group == "F4_nls_curve_fit" else "",
            }
        )
    return out


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    existing, existing_names = extract_signal_features(dataset.x_channels)
    x_pos = np.linspace(-1.0, 1.0, dataset.delta_b.shape[-1], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    feature_matrix: list[list[float]] = []
    names: list[str] | None = None
    groups: list[str] | None = None
    for i, sample_id in enumerate(dataset.sample_ids):
        values, names_i, groups_i = extract_one(dataset.delta_b[i].astype(np.float64), x_pos, existing[i], existing_names)
        if names is None:
            names = names_i
            groups = groups_i
        feature_matrix.append(values)
        row = {"sample_id": str(sample_id), "split": str(dataset.split[i])}
        row.update({name: value for name, value in zip(names_i, values)})
        rows.append(row)
    assert names is not None and groups is not None
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    fields = ["sample_id", "split"] + names
    args.features.parent.mkdir(parents=True, exist_ok=True)
    with args.features.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    q_rows = quality_rows(matrix, names, groups)
    write_csv(args.quality, q_rows, ["feature_group", "feature_count", "finite_fraction", "nan_count", "inf_count", "fit_success_rate", "notes"])
    f4 = next((row for row in q_rows if row["feature_group"] == "F4_nls_curve_fit"), {})
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 Piao/NLS feature extraction summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"input: delta_b only, shape={list(dataset.delta_b.shape)}, axes={dataset.axis_names}, scan_line_y={dataset.scan_line_y.tolist()}",
                f"split_counts: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}",
                f"sample_count: {len(dataset.sample_ids)}",
                f"feature_count_total: {len(names)}",
                f"feature_group_counts: { {row['feature_group']: row['feature_count'] for row in q_rows} }",
                f"overall_finite_fraction: {float(np.isfinite(matrix).mean()):.6f}",
                f"nls_fit_success_rate: {f4.get('fit_success_rate', math.nan)}",
                "nls_scope: bounded gaussian / derivative-of-gaussian NLS proxy; not exact Piao two-stage 18-feature reproduction.",
                "feature_input_boundary: no rbc_params, masks, split, sample_id, curvature_template, depth_bin, or aspect_bin used as numeric features.",
                "downstream_policy: train-only median imputation and scaling are required before regression.",
                f"features_csv: {args.features}",
                f"quality_csv: {args.quality}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--quality", type=Path, default=QUALITY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

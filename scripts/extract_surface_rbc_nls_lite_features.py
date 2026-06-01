#!/usr/bin/env python
"""Extract stable NLS-lite surface RBC features from v3_240 delta_b.

This is an NLS-lite diagnostic feature extractor, not an exact Piao
18-feature reproduction. Formal feature columns are derived only from
delta_b Bx/By/Bz signals. Target labels and profile/mask metadata are not
written to the feature matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    V3_240_DATASET_ID,
    load_dataset,
    split_indices,
    write_csv,
)


FEATURES = ROOT / "results/metrics/surface_rbc_nls_lite_features.csv"
QUALITY = ROOT / "results/metrics/surface_rbc_nls_lite_feature_quality.csv"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_lite_feature_summary.txt"
MANIFEST = ROOT / "results/manifests/surface_rbc_nls_lite_feature_manifest.json"

AXES = ["Bx", "By", "Bz"]
LINES = ["yneg", "y0", "ypos"]
EXPECTED_SHAPE_SUFFIX = (3, 3, 201)
EPS = 1.0e-12


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def safe_div(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= EPS:
        return 0.0
    return safe_float(float(numerator) / float(denominator))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2 or float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return 0.0
    return safe_float(np.corrcoef(a, b)[0, 1])


def width_at_half_peak(sensor_x: np.ndarray, signal: np.ndarray) -> float:
    abs_signal = np.abs(signal)
    peak = float(np.max(abs_signal))
    if peak <= EPS:
        return 0.0
    mask = abs_signal >= 0.5 * peak
    if not np.any(mask):
        return 0.0
    idx = np.where(mask)[0]
    return safe_float(sensor_x[idx[-1]] - sensor_x[idx[0]])


def width_proxy(sensor_x: np.ndarray, signal: np.ndarray) -> float:
    weights = np.abs(signal)
    total = float(np.sum(weights))
    if total <= EPS:
        return 0.0
    center = float(np.sum(sensor_x * weights) / total)
    variance = float(np.sum(((sensor_x - center) ** 2) * weights) / total)
    return safe_float(math.sqrt(max(variance, 0.0)))


def decay_proxy(signal: np.ndarray) -> float:
    abs_signal = np.abs(signal)
    peak = float(np.max(abs_signal))
    if peak <= EPS:
        return 0.0
    edge = max(1, int(round(0.10 * signal.size)))
    edge_abs = float(np.mean(np.concatenate([abs_signal[:edge], abs_signal[-edge:]])))
    return safe_float(1.0 - safe_div(edge_abs, peak))


def line_stats(sensor_x: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    grad = np.gradient(signal, sensor_x)
    abs_signal = np.abs(signal)
    abs_peak_idx = int(np.argmax(abs_signal))
    pos_idx = int(np.argmax(signal))
    neg_idx = int(np.argmin(signal))
    pos_peak = max(float(signal[pos_idx]), 0.0)
    neg_peak_abs = max(float(-signal[neg_idx]), 0.0)
    peak_pos = float(sensor_x[abs_peak_idx])
    left_energy = float(np.mean(signal[: abs_peak_idx + 1] ** 2)) if abs_peak_idx >= 0 else 0.0
    right_energy = float(np.mean(signal[abs_peak_idx:] ** 2)) if abs_peak_idx < signal.size else 0.0
    return {
        "positive_peak": safe_float(pos_peak),
        "negative_peak_abs": safe_float(neg_peak_abs),
        "peak_to_peak": safe_float(float(np.ptp(signal))),
        "abs_peak": safe_float(float(np.max(abs_signal))),
        "abs_peak_position_m": safe_float(peak_pos),
        "positive_peak_position_m": safe_float(float(sensor_x[pos_idx])),
        "negative_peak_position_m": safe_float(float(sensor_x[neg_idx])),
        "half_peak_width_m": width_at_half_peak(sensor_x, signal),
        "width_proxy_m": width_proxy(sensor_x, signal),
        "energy": safe_float(float(np.trapezoid(signal * signal, sensor_x))),
        "mean_abs": safe_float(float(np.mean(abs_signal))),
        "mean_signed": safe_float(float(np.mean(signal))),
        "gradient_energy": safe_float(float(np.trapezoid(grad * grad, sensor_x))),
        "gradient_abs_peak": safe_float(float(np.max(np.abs(grad)))),
        "left_right_asymmetry": safe_div(left_energy - right_energy, left_energy + right_energy),
        "decay_proxy": decay_proxy(signal),
        "zero_crossings": safe_float(float(np.count_nonzero(np.diff(np.signbit(signal))))),
    }


def fallback_fit(stats: dict[str, float]) -> dict[str, float]:
    residual = safe_float(stats["energy"] / max(1.0, abs(stats["peak_to_peak"])))
    return {
        "fit_amplitude": stats["abs_peak"],
        "fit_center_m": stats["abs_peak_position_m"],
        "fit_width_m": stats["half_peak_width_m"] if stats["half_peak_width_m"] > EPS else stats["width_proxy_m"],
        "fit_residual": residual,
        "fit_success": 0.0,
        "fallback_used": 1.0,
    }


def fit_abs_gaussian(sensor_x: np.ndarray, signal: np.ndarray, stats: dict[str, float]) -> dict[str, float]:
    abs_signal = np.abs(np.asarray(signal, dtype=np.float64))
    if float(np.max(abs_signal)) <= EPS:
        out = fallback_fit(stats)
        out["fit_residual"] = 0.0
        return out
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return fallback_fit(stats)

    def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset

    try:
        x_range = float(np.max(sensor_x) - np.min(sensor_x))
        p0 = [
            float(np.max(abs_signal)),
            float(stats["abs_peak_position_m"]),
            max(float(stats["half_peak_width_m"]) / 2.355, 0.002),
            float(np.percentile(abs_signal, 10.0)),
        ]
        bounds = ([0.0, float(np.min(sensor_x)), 1.0e-5, 0.0], [np.inf, float(np.max(sensor_x)), max(x_range, 1.0e-5), np.inf])
        popt, _ = curve_fit(gaussian, sensor_x, abs_signal, p0=p0, bounds=bounds, maxfev=1500)
        pred = gaussian(sensor_x, *popt)
        rmse = float(np.sqrt(np.mean((pred - abs_signal) ** 2)))
        denom = max(float(np.max(abs_signal)), EPS)
        return {
            "fit_amplitude": safe_float(popt[0]),
            "fit_center_m": safe_float(popt[1]),
            "fit_width_m": safe_float(abs(popt[2])),
            "fit_residual": safe_float(rmse / denom),
            "fit_success": 1.0,
            "fallback_used": 0.0,
        }
    except Exception:
        return fallback_fit(stats)


def add_features(out: dict[str, float], prefix: str, values: dict[str, float]) -> None:
    for key, value in values.items():
        out[f"nlslite_{prefix}_{key}"] = safe_float(value)


def extract_sample_features(delta: np.ndarray, sensor_x: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    delta = np.asarray(delta, dtype=np.float64)
    sensor_x = np.asarray(sensor_x, dtype=np.float64)
    if tuple(delta.shape) != EXPECTED_SHAPE_SUFFIX:
        raise ValueError(f"expected sample delta_b shape {EXPECTED_SHAPE_SUFFIX}, got {tuple(delta.shape)}")
    if sensor_x.shape != (EXPECTED_SHAPE_SUFFIX[-1],):
        raise ValueError(f"expected sensor_x shape {(EXPECTED_SHAPE_SUFFIX[-1],)}, got {tuple(sensor_x.shape)}")

    row: dict[str, float] = {}
    stats_by_axis_line: dict[tuple[int, int], dict[str, float]] = {}
    fit_success: list[float] = []
    fallback_used: list[float] = []
    residuals: list[float] = []

    for axis_i, axis in enumerate(AXES):
        for line_i, line in enumerate(LINES):
            signal = delta[axis_i, line_i]
            stats = line_stats(sensor_x, signal)
            fit = fit_abs_gaussian(sensor_x, signal, stats)
            combined = {**stats, **fit}
            stats_by_axis_line[(axis_i, line_i)] = combined
            add_features(row, f"{axis}_{line}", combined)
            fit_success.append(combined["fit_success"])
            fallback_used.append(combined["fallback_used"])
            residuals.append(combined["fit_residual"])

    for line_i, line in enumerate(LINES):
        bx = delta[0, line_i]
        by = delta[1, line_i]
        bz = delta[2, line_i]
        vmag = np.sqrt(bx * bx + by * by + bz * bz)
        vmag_stats = line_stats(sensor_x, vmag)
        add_features(
            row,
            f"vmag_{line}",
            {
                "abs_peak": vmag_stats["abs_peak"],
                "abs_peak_position_m": vmag_stats["abs_peak_position_m"],
                "half_peak_width_m": vmag_stats["half_peak_width_m"],
                "energy": vmag_stats["energy"],
                "gradient_energy": vmag_stats["gradient_energy"],
                "decay_proxy": vmag_stats["decay_proxy"],
            },
        )
        bx_s = stats_by_axis_line[(0, line_i)]
        by_s = stats_by_axis_line[(1, line_i)]
        bz_s = stats_by_axis_line[(2, line_i)]
        add_features(
            row,
            line,
            {
                "Bx_to_Bz_abs_peak_ratio": safe_div(bx_s["abs_peak"], bz_s["abs_peak"]),
                "By_to_Bz_abs_peak_ratio": safe_div(by_s["abs_peak"], bz_s["abs_peak"]),
                "Bz_to_Bx_abs_peak_ratio": safe_div(bz_s["abs_peak"], bx_s["abs_peak"]),
                "Bx_to_By_abs_peak_ratio": safe_div(bx_s["abs_peak"], by_s["abs_peak"]),
                "Bx_to_Bz_energy_ratio": safe_div(bx_s["energy"], bz_s["energy"]),
                "By_to_Bz_energy_ratio": safe_div(by_s["energy"], bz_s["energy"]),
                "Bz_to_Bx_energy_ratio": safe_div(bz_s["energy"], bx_s["energy"]),
                "Bx_Bz_peak_shift_m": safe_float(bx_s["abs_peak_position_m"] - bz_s["abs_peak_position_m"]),
                "By_Bz_peak_shift_m": safe_float(by_s["abs_peak_position_m"] - bz_s["abs_peak_position_m"]),
                "Bx_By_correlation": corr(bx, by),
                "Bx_Bz_correlation": corr(bx, bz),
                "By_Bz_correlation": corr(by, bz),
            },
        )

    for axis_i, axis in enumerate(AXES):
        axis_stats = [stats_by_axis_line[(axis_i, line_i)] for line_i in range(3)]
        abs_peaks = np.asarray([item["abs_peak"] for item in axis_stats], dtype=np.float64)
        peak_positions = np.asarray([item["abs_peak_position_m"] for item in axis_stats], dtype=np.float64)
        widths = np.asarray([item["half_peak_width_m"] for item in axis_stats], dtype=np.float64)
        energies = np.asarray([item["energy"] for item in axis_stats], dtype=np.float64)
        center = axis_stats[1]
        outer_mean_peak = 0.5 * (axis_stats[0]["abs_peak"] + axis_stats[2]["abs_peak"])
        add_features(
            row,
            axis,
            {
                "line_to_line_amplitude_spread": safe_div(float(np.max(abs_peaks) - np.min(abs_peaks)), float(np.mean(abs_peaks))),
                "line_to_line_peak_shift_m": safe_float(float(np.max(peak_positions) - np.min(peak_positions))),
                "line_to_line_peak_position_std_m": safe_float(float(np.std(peak_positions))),
                "line_to_line_width_spread": safe_div(float(np.max(widths) - np.min(widths)), float(np.mean(widths))),
                "line_to_line_energy_spread": safe_div(float(np.max(energies) - np.min(energies)), float(np.mean(energies))),
                "center_to_outer_abs_peak_ratio": safe_div(center["abs_peak"], outer_mean_peak),
                "outer_abs_peak_asymmetry": safe_div(axis_stats[0]["abs_peak"] - axis_stats[2]["abs_peak"], axis_stats[0]["abs_peak"] + axis_stats[2]["abs_peak"]),
                "outer_peak_shift_m": safe_float(axis_stats[0]["abs_peak_position_m"] - axis_stats[2]["abs_peak_position_m"]),
                "fit_residual_mean": safe_float(float(np.mean([item["fit_residual"] for item in axis_stats]))),
                "fit_residual_max": safe_float(float(np.max([item["fit_residual"] for item in axis_stats]))),
            },
        )

    quality = {
        "fit_success_rate": safe_float(float(np.mean(fit_success))),
        "fallback_rate": safe_float(float(np.mean(fallback_used))),
        "fit_residual_mean": safe_float(float(np.mean(residuals))),
        "fit_residual_max": safe_float(float(np.max(residuals))),
    }
    return row, quality


def feature_group(name: str) -> str:
    if "_fit_" in name or name.endswith("_fit_success") or name.endswith("_fallback_used"):
        return "fit_quality"
    if "_line_to_line_" in name or "_center_to_outer_" in name or "_outer_" in name:
        return "line_to_line"
    if "_to_" in name or "_correlation" in name or "_peak_shift" in name:
        return "cross_axis"
    if "_vmag_" in name:
        return "vector_magnitude"
    if "_gradient_" in name or "_asymmetry" in name or "_decay_" in name:
        return "gradient_asymmetry_decay"
    return "peak_width_energy"


def quality_rows(matrix: np.ndarray, feature_names: list[str], sample_qualities: list[dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in sorted({feature_group(name) for name in feature_names}):
        idx = [i for i, name in enumerate(feature_names) if feature_group(name) == group]
        sub = matrix[:, idx]
        rows.append(
            {
                "quality_scope": group,
                "feature_count": len(idx),
                "finite_fraction": safe_float(float(np.isfinite(sub).mean())),
                "nan_count": int(np.isnan(sub).sum()),
                "inf_count": int(np.isinf(sub).sum()),
                "fit_success_rate": "",
                "fallback_rate": "",
                "fit_residual_mean": "",
                "fit_residual_max": "",
                "notes": "feature group quality",
            }
        )
    rows.append(
        {
            "quality_scope": "overall",
            "feature_count": len(feature_names),
            "finite_fraction": safe_float(float(np.isfinite(matrix).mean())),
            "nan_count": int(np.isnan(matrix).sum()),
            "inf_count": int(np.isinf(matrix).sum()),
            "fit_success_rate": safe_float(float(np.mean([q["fit_success_rate"] for q in sample_qualities]))),
            "fallback_rate": safe_float(float(np.mean([q["fallback_rate"] for q in sample_qualities]))),
            "fit_residual_mean": safe_float(float(np.mean([q["fit_residual_mean"] for q in sample_qualities]))),
            "fit_residual_max": safe_float(float(np.max([q["fit_residual_max"] for q in sample_qualities]))),
            "notes": "sample-level fit quality averaged over 3 axes x 3 lines",
        }
    )
    return rows


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset_id)
    if tuple(dataset.delta_b.shape[1:]) != EXPECTED_SHAPE_SUFFIX:
        raise RuntimeError(f"expected delta_b suffix {EXPECTED_SHAPE_SUFFIX}, got {tuple(dataset.delta_b.shape)}")
    if dataset.axis_names != AXES:
        raise RuntimeError(f"expected axis order {AXES}, got {dataset.axis_names}")
    if len(dataset.scan_line_y) != 3:
        raise RuntimeError(f"expected exactly 3 scan_line_y values, got {len(dataset.scan_line_y)}")

    splits = split_indices(dataset)
    rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    matrix_rows: list[list[float]] = []
    sample_qualities: list[dict[str, float]] = []
    for i, sample_id in enumerate(dataset.sample_ids):
        feature_row, quality = extract_sample_features(dataset.delta_b[i], dataset.sensor_x)
        if feature_names is None:
            feature_names = list(feature_row.keys())
        row = {"sample_id": str(sample_id), "split": str(dataset.split[i])}
        row.update(feature_row)
        rows.append(row)
        matrix_rows.append([feature_row[name] for name in feature_names])
        sample_qualities.append(quality)

    if feature_names is None:
        raise RuntimeError("no samples found")
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    fields = ["sample_id", "split", *feature_names]
    args.features.parent.mkdir(parents=True, exist_ok=True)
    with args.features.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    q_rows = quality_rows(matrix, feature_names, sample_qualities)
    write_csv(
        args.quality,
        q_rows,
        [
            "quality_scope",
            "feature_count",
            "finite_fraction",
            "nan_count",
            "inf_count",
            "fit_success_rate",
            "fallback_rate",
            "fit_residual_mean",
            "fit_residual_max",
            "notes",
        ],
    )
    overall = next(row for row in q_rows if row["quality_scope"] == "overall")

    manifest = {
        "artifact_id": "surface_rbc_nls_lite_feature_extractor",
        "stage": "24.0A",
        "dataset_id": dataset.dataset_id,
        "source_manifest_path": dataset.manifest.get("manifest_path"),
        "source_npz_path": str(dataset.npz_path),
        "feature_csv": str(args.features),
        "quality_csv": str(args.quality),
        "summary": str(args.summary),
        "feature_manifest": str(args.manifest),
        "feature_count": len(feature_names),
        "sample_count": len(rows),
        "formal_feature_prefix": "nlslite_",
        "formal_feature_inputs": ["delta_b/Bx", "delta_b/By", "delta_b/Bz"],
        "metadata_columns": ["sample_id", "split"],
        "target_labels_in_feature_csv": False,
        "excluded_numeric_feature_inputs": ["rbc_params", "profile", "mask", "split", "sample_id", "curvature_template", "depth_bin", "aspect_bin", "size_bin"],
        "axis_order": AXES,
        "delta_b_shape": list(dataset.delta_b.shape),
        "scan_line_y": [float(x) for x in dataset.scan_line_y.tolist()],
        "exact_piao_nls": False,
        "piao_nls_lite": True,
        "claim_boundary": "stable NLS-lite physical features from the current three scan_line_y dataset; not exact Piao 18-feature reproduction",
        "fit_success_rate": overall["fit_success_rate"],
        "fallback_rate": overall["fallback_rate"],
        "overall_finite_fraction": overall["finite_fraction"],
        "COMSOL_run": False,
        "training_run": False,
        "data_or_NPZ_modified": False,
        "CURRENT_BASELINE_update": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_commit_before_run": git_value(["rev-parse", "HEAD"]),
    }
    write_json(args.manifest, manifest)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "24.0A surface RBC NLS-lite feature extraction summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                "scope: feature extraction only; no COMSOL, no training, no data/NPZ modification, no CURRENT_BASELINE update.",
                "loading_policy: explicit COMSOL_DATA_REGISTRY.md plus manifest; latest/newest NPZ scan is not used.",
                "method_claim: NLS-lite stable physical feature extractor; exact_piao_nls=false; piao_nls_lite=true.",
                f"input_delta_b_shape: {tuple(dataset.delta_b.shape)}",
                f"axis_order: {dataset.axis_names}",
                f"scan_line_y_m: {dataset.scan_line_y.tolist()}",
                f"split_counts: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}",
                f"sample_count: {len(rows)}",
                f"feature_count: {len(feature_names)}",
                f"overall_finite_fraction: {overall['finite_fraction']}",
                f"fit_success_rate: {overall['fit_success_rate']}",
                f"fallback_rate: {overall['fallback_rate']}",
                f"fit_residual_mean: {overall['fit_residual_mean']}",
                f"fit_residual_max: {overall['fit_residual_max']}",
                "formal_feature_csv_labels: false",
                "feature_input_boundary: formal numeric features come only from delta_b Bx/By/Bz; rbc_params/profile/mask/split/sample_id are not numeric feature inputs.",
                "current_data_boundary: only three scan_line_y values are available, so this is not an exact Piao 18-feature reproduction.",
                f"features_csv: {args.features}",
                f"quality_csv: {args.quality}",
                f"feature_manifest: {args.manifest}",
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
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

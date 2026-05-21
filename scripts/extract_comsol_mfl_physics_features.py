from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover - runtime capability is summarized
    curve_fit = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_FEATURES = PROJECT_ROOT / "results/metrics/comsol_mfl_physics_features.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_mfl_physics_feature_extraction_summary.txt"
TAU_FALLBACK = 0.004


def _safe_float(value: float) -> float:
    if np.isfinite(value):
        return float(value)
    return 0.0


def _skewness(values: np.ndarray) -> float:
    std = float(values.std())
    if std <= 0:
        return 0.0
    centered = values - float(values.mean())
    return _safe_float(float(np.mean((centered / std) ** 3)))


def _zero_crossings(values: np.ndarray) -> int:
    signs = np.sign(values)
    signs[signs == 0] = 1
    return int(np.sum(signs[1:] != signs[:-1]))


def _width_at_half_absmax(x: np.ndarray, values: np.ndarray) -> float:
    abs_values = np.abs(values)
    peak = float(abs_values.max())
    if peak <= 0:
        return 0.0
    active = np.flatnonzero(abs_values >= 0.5 * peak)
    if active.size == 0:
        return 0.0
    return float(x[active[-1]] - x[active[0]])


def _line_features(x: np.ndarray, values: np.ndarray, prefix: str) -> dict[str, float]:
    grad = np.gradient(values, x)
    pos_idx = int(np.argmax(values))
    neg_idx = int(np.argmin(values))
    abs_idx = int(np.argmax(np.abs(values)))
    left = values[: len(values) // 2]
    right = values[len(values) // 2 + 1 :]
    left_abs = float(np.trapz(np.abs(left), x[: len(left)])) if left.size else 0.0
    right_abs = float(np.trapz(np.abs(right), x[-len(right) :])) if right.size else 0.0
    asym = (right_abs - left_abs) / (right_abs + left_abs + 1e-12)
    return {
        f"{prefix}_max": _safe_float(float(values.max())),
        f"{prefix}_min": _safe_float(float(values.min())),
        f"{prefix}_ptp": _safe_float(float(values.max() - values.min())),
        f"{prefix}_pos_peak_x": _safe_float(float(x[pos_idx])),
        f"{prefix}_neg_peak_x": _safe_float(float(x[neg_idx])),
        f"{prefix}_abs_peak_x": _safe_float(float(x[abs_idx])),
        f"{prefix}_peak_distance": _safe_float(abs(float(x[pos_idx] - x[neg_idx]))),
        f"{prefix}_integral": _safe_float(float(np.trapz(values, x))),
        f"{prefix}_abs_integral": _safe_float(float(np.trapz(np.abs(values), x))),
        f"{prefix}_energy": _safe_float(float(np.trapz(values**2, x))),
        f"{prefix}_mean": _safe_float(float(values.mean())),
        f"{prefix}_std": _safe_float(float(values.std())),
        f"{prefix}_skewness": _safe_float(_skewness(values)),
        f"{prefix}_grad_max": _safe_float(float(grad.max())),
        f"{prefix}_grad_min": _safe_float(float(grad.min())),
        f"{prefix}_grad_ptp": _safe_float(float(grad.max() - grad.min())),
        f"{prefix}_zero_crossing_count": float(_zero_crossings(values)),
        f"{prefix}_width_half_absmax": _safe_float(_width_at_half_absmax(x, values)),
        f"{prefix}_left_right_asymmetry": _safe_float(asym),
    }


def _exp_abs_model(x: np.ndarray, c: float, a: float, x0: float, tau: float) -> np.ndarray:
    return c + a * np.exp(-np.abs(x - x0) / np.maximum(tau, 1e-9))


def _fallback_exp_fit(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], bool]:
    baseline = float(np.median(y))
    idx = int(np.argmax(y))
    peak = float(y[idx])
    amplitude = max(peak - baseline, 0.0)
    x0 = float(x[idx])
    tau = TAU_FALLBACK
    pred = _exp_abs_model(x, baseline, amplitude, x0, tau)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    denom = float(np.sum((y - float(y.mean())) ** 2))
    r2 = 1.0 - float(np.sum((pred - y) ** 2)) / denom if denom > 0 else 0.0
    return {"c": baseline, "A": amplitude, "x0": x0, "tau": tau, "rmse": rmse, "r2": r2}, True


def _fit_exponential_lobe(x: np.ndarray, y: np.ndarray, preferred_idx: int | None = None) -> tuple[dict[str, float], bool]:
    y = np.asarray(y, dtype=float)
    if y.size != x.size or y.size < 5 or not np.isfinite(y).all():
        return _fallback_exp_fit(x, np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))
    y = np.maximum(y, 0.0)
    baseline = float(np.percentile(y, 20))
    idx = int(np.argmax(y) if preferred_idx is None else preferred_idx)
    idx = max(0, min(idx, y.size - 1))
    amplitude = max(float(y[idx]) - baseline, 1e-12)
    x0 = float(x[idx])
    p0 = [baseline, amplitude, x0, TAU_FALLBACK]
    bounds = ([0.0, 0.0, float(x.min()), 1e-5], [float(max(y.max() * 2, 1e-9)), float(max(y.max() * 3, 1e-9)), float(x.max()), 0.05])
    if curve_fit is None:
        return _fallback_exp_fit(x, y)
    try:
        popt, _ = curve_fit(_exp_abs_model, x, y, p0=p0, bounds=bounds, maxfev=4000)
        pred = _exp_abs_model(x, *popt)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        denom = float(np.sum((y - float(y.mean())) ** 2))
        r2 = 1.0 - float(np.sum((pred - y) ** 2)) / denom if denom > 0 else 0.0
        out = {"c": float(popt[0]), "A": float(popt[1]), "x0": float(popt[2]), "tau": float(popt[3]), "rmse": rmse, "r2": r2}
        return out, False
    except Exception:
        return _fallback_exp_fit(x, y)


def _nls_style_features(x: np.ndarray, values: np.ndarray, prefix: str) -> dict[str, float]:
    # Bz-only weak adaptation of Piao's NLS feature idea: fit exponential lobes
    # to abs, positive, and negative Bz responses. This is not tri-axis NLS.
    abs_y = np.abs(values)
    pos_y = np.maximum(values, 0.0)
    neg_y = np.maximum(-values, 0.0)
    abs_fit, abs_failed = _fit_exponential_lobe(x, abs_y)
    pos_fit, pos_failed = _fit_exponential_lobe(x, pos_y, int(np.argmax(pos_y)))
    neg_fit, neg_failed = _fit_exponential_lobe(x, neg_y, int(np.argmax(neg_y)))
    eps = 1e-12
    return {
        f"{prefix}_nls_abs_A": _safe_float(abs_fit["A"]),
        f"{prefix}_nls_abs_x0": _safe_float(abs_fit["x0"]),
        f"{prefix}_nls_abs_tau": _safe_float(abs_fit["tau"]),
        f"{prefix}_nls_abs_c": _safe_float(abs_fit["c"]),
        f"{prefix}_nls_abs_rmse": _safe_float(abs_fit["rmse"]),
        f"{prefix}_nls_abs_r2": _safe_float(abs_fit["r2"]),
        f"{prefix}_nls_abs_fit_failed": float(abs_failed),
        f"{prefix}_nls_pos_A": _safe_float(pos_fit["A"]),
        f"{prefix}_nls_pos_x0": _safe_float(pos_fit["x0"]),
        f"{prefix}_nls_pos_tau": _safe_float(pos_fit["tau"]),
        f"{prefix}_nls_pos_rmse": _safe_float(pos_fit["rmse"]),
        f"{prefix}_nls_pos_fit_failed": float(pos_failed),
        f"{prefix}_nls_neg_A": _safe_float(neg_fit["A"]),
        f"{prefix}_nls_neg_x0": _safe_float(neg_fit["x0"]),
        f"{prefix}_nls_neg_tau": _safe_float(neg_fit["tau"]),
        f"{prefix}_nls_neg_rmse": _safe_float(neg_fit["rmse"]),
        f"{prefix}_nls_neg_fit_failed": float(neg_failed),
        f"{prefix}_nls_pos_neg_x_distance": _safe_float(pos_fit["x0"] - neg_fit["x0"]),
        f"{prefix}_nls_pos_neg_abs_x_distance": _safe_float(abs(pos_fit["x0"] - neg_fit["x0"])),
        f"{prefix}_nls_pos_neg_tau_ratio": _safe_float(pos_fit["tau"] / (neg_fit["tau"] + eps)),
        f"{prefix}_nls_pos_neg_A_ratio": _safe_float(pos_fit["A"] / (neg_fit["A"] + eps)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 0 or b_std <= 0:
        return 0.0
    return _safe_float(float(np.corrcoef(a, b)[0, 1]))


def _aggregate_features(features: dict[str, float], prefixes: list[str], suffixes: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for suffix in suffixes:
        values = np.array([features[f"{prefix}_{suffix}"] for prefix in prefixes], dtype=float)
        out[f"lines_{suffix}_mean"] = _safe_float(float(values.mean()))
        out[f"lines_{suffix}_std"] = _safe_float(float(values.std()))
        out[f"lines_{suffix}_min"] = _safe_float(float(values.min()))
        out[f"lines_{suffix}_max"] = _safe_float(float(values.max()))
        out[f"lines_{suffix}_range"] = _safe_float(float(values.max() - values.min()))
        center = features[f"{prefixes[1]}_{suffix}"]
        out[f"lines_{suffix}_center_minus_outer_mean"] = _safe_float(
            float(center - 0.5 * (features[f"{prefixes[0]}_{suffix}"] + features[f"{prefixes[2]}_{suffix}"]))
        )
    return out


def extract_features(npz_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    required = ["delta_bz", "sensor_x", "scan_line_y", "sample_ids", "split"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing required NPZ keys: {missing}")

    delta = data["delta_bz"].astype(float)
    sensor_x = data["sensor_x"].astype(float)
    scan_line_y = data["scan_line_y"].astype(float)
    sample_ids = data["sample_ids"].astype(str)
    splits = data["split"].astype(str)

    rows: list[dict[str, Any]] = []
    prefixes = [f"line{i}" for i in range(delta.shape[1])]
    aggregate_suffixes = [
        "max",
        "min",
        "ptp",
        "peak_distance",
        "integral",
        "abs_integral",
        "energy",
        "mean",
        "std",
        "grad_ptp",
        "width_half_absmax",
        "left_right_asymmetry",
    ]
    for idx, sample_id in enumerate(sample_ids):
        row: dict[str, Any] = {
            "sample_index": idx,
            "sample_id": sample_id,
            "split": str(splits[idx]),
        }
        line_feature_map: dict[str, float] = {}
        for line_idx, prefix in enumerate(prefixes):
            values = delta[idx, line_idx]
            line_feature_map.update(_line_features(sensor_x, values, prefix))
            line_feature_map.update(_nls_style_features(sensor_x, values, prefix))
            row[f"{prefix}_scan_y"] = float(scan_line_y[line_idx])
        line_feature_map["line0_corr_center"] = _corr(delta[idx, 0], delta[idx, 1])
        line_feature_map["line2_corr_center"] = _corr(delta[idx, 2], delta[idx, 1])
        line_feature_map["outer_corr"] = _corr(delta[idx, 0], delta[idx, 2])
        line_feature_map["center_line_abs_dominance"] = _safe_float(
            line_feature_map["line1_abs_integral"]
            / (
                0.5 * (line_feature_map["line0_abs_integral"] + line_feature_map["line2_abs_integral"])
                + 1e-12
            )
        )
        line_feature_map["outer_abs_integral_difference"] = _safe_float(
            line_feature_map["line2_abs_integral"] - line_feature_map["line0_abs_integral"]
        )
        line_feature_map["outer_energy_difference"] = _safe_float(
            line_feature_map["line2_energy"] - line_feature_map["line0_energy"]
        )
        for suffix in [
            "nls_abs_A",
            "nls_abs_tau",
            "nls_abs_rmse",
            "nls_abs_r2",
            "nls_pos_A",
            "nls_neg_A",
            "nls_pos_tau",
            "nls_neg_tau",
            "nls_pos_neg_abs_x_distance",
            "nls_pos_neg_tau_ratio",
            "nls_pos_neg_A_ratio",
        ]:
            values = np.array([line_feature_map[f"{prefix}_{suffix}"] for prefix in prefixes], dtype=float)
            line_feature_map[f"lines_{suffix}_mean"] = _safe_float(float(values.mean()))
            line_feature_map[f"lines_{suffix}_std"] = _safe_float(float(values.std()))
            line_feature_map[f"lines_{suffix}_range"] = _safe_float(float(values.max() - values.min()))
        line_feature_map.update(_aggregate_features(line_feature_map, prefixes, aggregate_suffixes))
        row.update(line_feature_map)
        rows.append(row)

    feature_fields = [
        key
        for key in rows[0].keys()
        if key not in {"sample_index", "sample_id", "split"} and not key.endswith("_scan_y")
    ]
    train_rows = [row for row in rows if row["split"] == "train"]
    stats: dict[str, tuple[float, float]] = {}
    for field in feature_fields:
        values = np.array([float(row[field]) for row in train_rows], dtype=float)
        mean = float(values.mean())
        std = float(values.std())
        if std <= 0:
            std = 1.0
        stats[field] = (mean, std)
    for row in rows:
        for field in feature_fields:
            mean, std = stats[field]
            row[f"z_{field}"] = (float(row[field]) - mean) / std

    finite = all(np.isfinite(float(row[field])) for row in rows for field in feature_fields)
    nls_fields = [field for field in feature_fields if "_nls_" in field or field.startswith("lines_nls_")]
    fit_failed_fields = [field for field in feature_fields if field.endswith("_fit_failed")]
    fit_fail_values = [float(row[field]) for row in rows for field in fit_failed_fields]
    fit_failure_rate = float(np.mean(fit_fail_values)) if fit_fail_values else 0.0
    diagnostics = {
        "n": len(rows),
        "shape": tuple(delta.shape),
        "split_counts": Counter(row["split"] for row in rows),
        "feature_count_raw": len(feature_fields),
        "nls_style_feature_count_raw": len(nls_fields),
        "feature_count_with_z": len(feature_fields) * 2,
        "all_raw_features_finite": finite,
        "curve_fit_available": curve_fit is not None,
        "nls_fit_failure_rate": fit_failure_rate,
        "train_scaler_stats": stats,
        "sensor_x_monotonic": bool(np.all(np.diff(sensor_x) > 0)),
        "scan_line_y": scan_line_y.tolist(),
    }
    return rows, diagnostics


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out_path: Path, npz_path: Path, features_path: Path, diagnostics: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "COMSOL MFL physics-inspired feature extraction summary",
        "",
        f"Input NPZ: {npz_path}",
        f"Output features CSV: {features_path}",
        f"delta_bz shape: {diagnostics['shape']}",
        f"N: {diagnostics['n']}",
        f"Split counts: {dict(diagnostics['split_counts'])}",
        f"Raw feature count: {diagnostics['feature_count_raw']}",
        f"Raw Bz-only NLS-style feature count: {diagnostics['nls_style_feature_count_raw']}",
        f"Raw + train-scaled feature count: {diagnostics['feature_count_with_z']}",
        f"curve_fit available: {diagnostics['curve_fit_available']}",
        f"Bz-only NLS-style fit failure rate: {diagnostics['nls_fit_failure_rate']:.6f}",
        f"All raw features finite: {diagnostics['all_raw_features_finite']}",
        f"sensor_x monotonic: {diagnostics['sensor_x_monotonic']}",
        f"scan_line_y: {diagnostics['scan_line_y']}",
        "",
        "Feature policy:",
        "- Features use only delta_bz, sensor_x, scan_line_y, and split for train-only scaling.",
        "- No mask, geometry label, defect type, source_pack, or metadata is used as an input feature.",
        "- Feature scaler mean/std was fit on train split only; val/test reuse train statistics.",
        "- Extracted generic line-wise peak, integral, energy, gradient, zero-crossing, width, asymmetry, and cross-line correlation features.",
        "- Added Bz-only NLS-style exponential fitting features for abs, positive-lobe, and negative-lobe delta_bz on each scan line.",
        "- This is not the tri-axis Piao 2019 NLS implementation: no Bx/By, no full axial/tangential matrices, and no 18-parameter paper-faithful feature vector.",
        "",
        "Scaler preview:",
        json.dumps(
            {
                key: {"mean": value[0], "std": value[1]}
                for key, value in list(diagnostics["train_scaler_stats"].items())[:12]
            },
            indent=2,
            sort_keys=True,
        ),
        "",
        "Quality gate passed: "
        + str(
            diagnostics["n"] == 600
            and diagnostics["shape"] == (600, 3, 201)
            and diagnostics["all_raw_features_finite"]
        ),
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--out", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows, diagnostics = extract_features(args.npz)
    write_csv(rows, args.out)
    write_summary(args.summary, args.npz, args.out, diagnostics)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Extract Piao NLS full-compatible features with explicit degraded mode.

This script reads existing delta_b arrays only. It does not train, does not run
COMSOL, and does not create or modify data/NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "nls_full_compatible_v1"
AXES = ["Bx", "By", "Bz"]
FULL_MIN_LINE_COUNT = 5
FULL_CANDIDATE_LINE_COUNT = 9
MIN_SENSOR_X_COUNT = 5
EPS = 1.0e-12

DEFAULT_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
FEATURES = ROOT / "results/metrics/surface_rbc_nls_full_compatible_features.csv"
QUALITY = ROOT / "results/metrics/surface_rbc_nls_full_compatible_quality.csv"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_full_compatible_summary.txt"
MANIFEST = ROOT / "results/manifests/surface_rbc_nls_full_compatible_feature_manifest.json"

METADATA_FIELDS = [
    "sample_id",
    "split",
    "dataset_id",
    "feature_schema_version",
    "scan_line_count",
    "sensor_x_count",
    "piao_full_compatible",
    "exact_piao_full",
    "full_feature_ready",
    "full_candidate_mode",
    "degraded_mode",
    "degraded_mode_reason",
]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > EPS else math.nan


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def normalise_delta(delta_b: np.ndarray) -> np.ndarray:
    arr = np.asarray(delta_b, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr[None, :, :, :]
    if arr.ndim != 4:
        raise ValueError(f"delta_b must have shape (S,3,M,K) or (3,M,K), got {arr.shape}")
    if arr.shape[1] != 3:
        raise ValueError(f"delta_b axis dimension must be 3 for Bx/By/Bz, got {arr.shape}")
    return arr


def center_line_index(scan_line_y_m: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.asarray(scan_line_y_m, dtype=np.float64))))


def width_at_fraction(position: np.ndarray, values: np.ndarray, fraction: float) -> float:
    pos = np.asarray(position, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if pos.size != y.size or pos.size < 2 or not np.all(np.isfinite(y)):
        return math.nan
    peak = float(np.max(np.abs(y)))
    if peak <= EPS:
        return math.nan
    mask = np.abs(y) >= peak * float(fraction)
    if not np.any(mask):
        return math.nan
    idx = np.where(mask)[0]
    return float(pos[idx[-1]] - pos[idx[0]])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size != bb.size or aa.size < 2 or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return math.nan
    if float(np.std(aa)) <= EPS or float(np.std(bb)) <= EPS:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def line_stats(sensor_x_m: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    x = np.asarray(sensor_x_m, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    if x.size != y.size or x.size < 2 or not np.all(np.isfinite(y)):
        return {
            "abs_peak": math.nan,
            "abs_peak_x_m": math.nan,
            "signed_peak": math.nan,
            "signed_peak_x_m": math.nan,
            "min": math.nan,
            "min_x_m": math.nan,
            "width50_x_m": math.nan,
            "energy": math.nan,
            "grad_abs_peak": math.nan,
            "grad_energy": math.nan,
            "left_right_energy_ratio": math.nan,
            "peak_asymmetry": math.nan,
        }
    abs_y = np.abs(y)
    abs_idx = int(np.argmax(abs_y))
    max_idx = int(np.argmax(y))
    min_idx = int(np.argmin(y))
    grad = np.gradient(y, x)
    left = y[: abs_idx + 1]
    right = y[abs_idx:]
    left_energy = float(np.mean(left * left)) if left.size else 0.0
    right_energy = float(np.mean(right * right)) if right.size else 0.0
    return {
        "abs_peak": float(abs_y[abs_idx]),
        "abs_peak_x_m": float(x[abs_idx]),
        "signed_peak": float(y[max_idx]),
        "signed_peak_x_m": float(x[max_idx]),
        "min": float(y[min_idx]),
        "min_x_m": float(x[min_idx]),
        "width50_x_m": width_at_fraction(x, y, 0.50),
        "energy": float(np.mean(y * y)),
        "grad_abs_peak": float(np.max(np.abs(grad))),
        "grad_energy": float(np.mean(grad * grad)),
        "left_right_energy_ratio": safe_div(left_energy, right_energy),
        "peak_asymmetry": safe_div(left_energy - right_energy, left_energy + right_energy),
    }


def fit_gaussian_envelope(y_pos: np.ndarray, envelope: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_pos, dtype=np.float64)
    env = np.asarray(envelope, dtype=np.float64)
    base = {
        "A": math.nan,
        "y0_m": math.nan,
        "sigma_m": math.nan,
        "rmse": math.nan,
        "nrmse": math.nan,
        "residual_peak_ratio": math.nan,
        "fit_success": 0.0,
        "failure_reason": "",
    }
    if y.size != env.size or y.size < FULL_MIN_LINE_COUNT:
        base["failure_reason"] = "scan_line_count_lt_5"
        return base
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(env)):
        base["failure_reason"] = "non_finite_envelope"
        return base
    amplitude = float(np.max(env) - np.min(env))
    peak = float(np.max(env))
    if amplitude <= EPS or peak <= EPS:
        base["failure_reason"] = "zero_or_constant_envelope"
        return base

    offset = float(np.min(env))
    weights = np.clip(env - offset, 0.0, None)
    weight_sum = float(np.sum(weights))
    if weight_sum <= EPS:
        base["failure_reason"] = "zero_or_constant_envelope"
        return base

    y0 = float(np.sum(y * weights) / weight_sum)
    sigma = float(np.sqrt(max(np.sum(((y - y0) ** 2) * weights) / weight_sum, EPS)))
    A = amplitude

    try:
        from scipy.optimize import curve_fit

        def gaussian(arg_y: np.ndarray, a: float, center: float, width: float, c: float) -> np.ndarray:
            return a * np.exp(-((arg_y - center) ** 2) / (2.0 * width * width)) + c

        min_step = float(np.min(np.diff(np.sort(y)))) if y.size > 1 else 1.0e-6
        span = float(np.max(y) - np.min(y))
        lower = [0.0, float(np.min(y)) - abs(min_step), max(abs(min_step) * 0.25, 1.0e-9), -np.inf]
        upper = [np.inf, float(np.max(y)) + abs(min_step), max(span * 2.0, abs(min_step)), np.inf]
        popt, _ = curve_fit(gaussian, y, env, p0=[A, y0, sigma, offset], bounds=(lower, upper), maxfev=2000)
        A, y0, sigma, offset = float(popt[0]), float(popt[1]), abs(float(popt[2])), float(popt[3])
    except Exception as exc:
        base["failure_reason"] = f"curve_fit_failed:{type(exc).__name__}"
        return base

    pred = A * np.exp(-((y - y0) ** 2) / (2.0 * sigma * sigma)) + offset
    residual = pred - env
    rmse = float(np.sqrt(np.mean(residual * residual)))
    residual_peak = float(np.max(np.abs(residual)))
    base.update(
        {
            "A": A,
            "y0_m": y0,
            "sigma_m": sigma,
            "rmse": rmse,
            "nrmse": safe_div(rmse, peak),
            "residual_peak_ratio": safe_div(residual_peak, peak),
            "fit_success": 1.0,
            "failure_reason": "",
        }
    )
    return base


def _feature_defs() -> list[tuple[str, str]]:
    defs: list[tuple[str, str]] = []
    for axis in AXES:
        for key in [
            "center_abs_peak",
            "center_peak_x_m",
            "center_signed_peak",
            "center_signed_peak_x_m",
            "center_min",
            "center_min_x_m",
            "center_width50_x_m",
            "center_energy",
        ]:
            defs.append((f"A1__{axis}_{key}", "axial_local_features"))
        for key in [
            "center_grad_abs_peak",
            "center_grad_energy",
            "center_left_right_energy_ratio",
            "center_peak_asymmetry",
            "outer_peak_decay_ratio",
            "outer_width_decay_ratio",
        ]:
            defs.append((f"A2__{axis}_{key}", "axial_decay_features"))
        for key in [
            "env_abs_peak",
            "env_peak_y_m",
            "env_width50_y_m",
            "env_gauss_A",
            "env_gauss_y0_m",
            "env_gauss_sigma_m",
            "env_gauss_rmse",
            "env_gauss_fit_success",
        ]:
            defs.append((f"T1__{axis}_{key}", "tangential_envelope_features"))
        for key in [
            "tangential_gauss_nrmse",
            "tangential_fit_residual_peak_ratio",
        ]:
            defs.append((f"R1__{axis}_{key}", "fit_residuals"))
    for key in [
        "center_Bx_Bz_abs_peak_ratio",
        "center_By_Bz_abs_peak_ratio",
        "center_Bx_By_abs_peak_ratio",
        "center_Bx_Bz_corr",
        "center_By_Bz_corr",
        "center_Bx_By_corr",
        "center_vmag_abs_peak",
        "center_vmag_peak_x_m",
        "center_vmag_width50_x_m",
        "center_vmag_energy",
    ]:
        defs.append((f"X1__{key}", "cross_axis_features"))
    return defs


FEATURE_DEFS = _feature_defs()
FEATURE_GROUP_BY_NAME = {name: group for name, group in FEATURE_DEFS}


def feature_columns() -> list[str]:
    return [name for name, _ in FEATURE_DEFS]


def validity_columns() -> list[str]:
    return [f"valid__{name}" for name in feature_columns()]


def input_adequacy(
    delta_b: np.ndarray,
    sensor_x_m: np.ndarray,
    scan_line_y_m: np.ndarray,
    axis_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    arr = normalise_delta(delta_b)
    x = np.asarray(sensor_x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(scan_line_y_m, dtype=np.float64).reshape(-1)
    axis_order_ok = list(axis_names) == AXES
    scan_line_count = int(arr.shape[2])
    sensor_x_count = int(arr.shape[3])
    y_matches_shape = y.size == scan_line_count
    x_matches_shape = x.size == sensor_x_count
    y_finite = bool(y.size > 0 and np.all(np.isfinite(y)))
    x_finite = bool(x.size > 0 and np.all(np.isfinite(x)))
    y_strict = bool(y.size < 2 or np.all(np.diff(y) > 0.0))
    y_uniform = True
    if y.size >= 3:
        diffs = np.diff(y)
        y_uniform = bool(np.max(np.abs(diffs - float(np.mean(diffs)))) <= max(1.0e-12, abs(float(np.mean(diffs))) * 0.05))
    missing_values = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    reasons: list[str] = []
    if not axis_order_ok:
        reasons.append("axis_order_invalid")
    if scan_line_count < FULL_MIN_LINE_COUNT:
        reasons.append("scan_line_count_lt_5")
    if sensor_x_count < MIN_SENSOR_X_COUNT:
        reasons.append("sensor_x_count_lt_5")
    if not x_matches_shape:
        reasons.append("sensor_x_count_mismatch")
    if not y_matches_shape:
        reasons.append("scan_line_y_count_mismatch")
    if not y_finite or not x_finite or not y_strict or not y_uniform:
        reasons.append("axis_coordinate_invalid")
    if missing_values:
        reasons.append("missing_values")
    full_feature_ready = len(reasons) == 0
    return {
        "axis_order_ok": axis_order_ok,
        "axis_names": list(axis_names),
        "shape": list(arr.shape),
        "sample_count": int(arr.shape[0]),
        "scan_line_count": scan_line_count,
        "sensor_x_count": sensor_x_count,
        "sensor_x_matches_shape": x_matches_shape,
        "scan_line_y_matches_shape": y_matches_shape,
        "y_line_spacing_ok": bool(y_matches_shape and y_finite and y_strict and y_uniform),
        "missing_values": missing_values,
        "fit_feasibility_by_geometry": bool(scan_line_count >= FULL_MIN_LINE_COUNT and sensor_x_count >= MIN_SENSOR_X_COUNT),
        "full_feature_ready": full_feature_ready,
        "full_candidate_mode": bool(full_feature_ready and scan_line_count >= FULL_CANDIDATE_LINE_COUNT),
        "degraded_mode": not full_feature_ready,
        "degraded_mode_reason": ";".join(reasons) if reasons else "",
    }


def set_feature(row: dict[str, Any], name: str, value: Any, valid: bool | None = None) -> None:
    numeric = safe_float(value)
    row[name] = numeric
    row[f"valid__{name}"] = bool(np.isfinite(numeric)) if valid is None else bool(valid)


def initialise_feature_row() -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name in feature_columns():
        row[name] = math.nan
        row[f"valid__{name}"] = False
    return row


def extract_one(
    sample: np.ndarray,
    sensor_x_m: np.ndarray,
    scan_line_y_m: np.ndarray,
    adequacy: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    row = initialise_feature_row()
    failures: list[str] = []
    x = np.asarray(sensor_x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(scan_line_y_m, dtype=np.float64).reshape(-1)
    ci = center_line_index(y)
    per_axis_center: dict[str, dict[str, float]] = {}
    per_axis_outer: dict[str, tuple[dict[str, float], dict[str, float]] | None] = {}

    for ai, axis in enumerate(AXES):
        center_signal = sample[ai, ci]
        stats = line_stats(x, center_signal)
        per_axis_center[axis] = stats
        mapping = {
            f"A1__{axis}_center_abs_peak": stats["abs_peak"],
            f"A1__{axis}_center_peak_x_m": stats["abs_peak_x_m"],
            f"A1__{axis}_center_signed_peak": stats["signed_peak"],
            f"A1__{axis}_center_signed_peak_x_m": stats["signed_peak_x_m"],
            f"A1__{axis}_center_min": stats["min"],
            f"A1__{axis}_center_min_x_m": stats["min_x_m"],
            f"A1__{axis}_center_width50_x_m": stats["width50_x_m"],
            f"A1__{axis}_center_energy": stats["energy"],
            f"A2__{axis}_center_grad_abs_peak": stats["grad_abs_peak"],
            f"A2__{axis}_center_grad_energy": stats["grad_energy"],
            f"A2__{axis}_center_left_right_energy_ratio": stats["left_right_energy_ratio"],
            f"A2__{axis}_center_peak_asymmetry": stats["peak_asymmetry"],
        }
        for name, value in mapping.items():
            set_feature(row, name, value)

        if sample.shape[1] >= 3:
            left_stats = line_stats(x, sample[ai, 0])
            right_stats = line_stats(x, sample[ai, -1])
            per_axis_outer[axis] = (left_stats, right_stats)
            outer_peak = 0.5 * (left_stats["abs_peak"] + right_stats["abs_peak"])
            outer_width = 0.5 * (left_stats["width50_x_m"] + right_stats["width50_x_m"])
            set_feature(row, f"A2__{axis}_outer_peak_decay_ratio", safe_div(stats["abs_peak"], outer_peak))
            set_feature(row, f"A2__{axis}_outer_width_decay_ratio", safe_div(stats["width50_x_m"], outer_width))
        else:
            per_axis_outer[axis] = None

        envelope = np.max(np.abs(sample[ai]), axis=1)
        env_peak_idx = int(np.argmax(envelope)) if envelope.size else 0
        if envelope.size == y.size and np.all(np.isfinite(envelope)) and envelope.size:
            set_feature(row, f"T1__{axis}_env_abs_peak", float(np.max(envelope)), valid=bool(np.max(envelope) > EPS))
            set_feature(row, f"T1__{axis}_env_peak_y_m", float(y[env_peak_idx]), valid=True)
            set_feature(row, f"T1__{axis}_env_width50_y_m", width_at_fraction(y, envelope, 0.50))
        if bool(adequacy["fit_feasibility_by_geometry"]):
            fit = fit_gaussian_envelope(y, envelope)
            if fit["fit_success"]:
                set_feature(row, f"T1__{axis}_env_gauss_A", fit["A"])
                set_feature(row, f"T1__{axis}_env_gauss_y0_m", fit["y0_m"])
                set_feature(row, f"T1__{axis}_env_gauss_sigma_m", fit["sigma_m"])
                set_feature(row, f"T1__{axis}_env_gauss_rmse", fit["rmse"])
                set_feature(row, f"R1__{axis}_tangential_gauss_nrmse", fit["nrmse"])
                set_feature(row, f"R1__{axis}_tangential_fit_residual_peak_ratio", fit["residual_peak_ratio"])
            else:
                failures.append(f"{axis}:{fit['failure_reason']}")
            set_feature(row, f"T1__{axis}_env_gauss_fit_success", fit["fit_success"], valid=True)
        else:
            set_feature(row, f"T1__{axis}_env_gauss_fit_success", 0.0, valid=True)

    bx = sample[0, ci]
    by = sample[1, ci]
    bz = sample[2, ci]
    bx_stats = per_axis_center["Bx"]
    by_stats = per_axis_center["By"]
    bz_stats = per_axis_center["Bz"]
    set_feature(row, "X1__center_Bx_Bz_abs_peak_ratio", safe_div(bx_stats["abs_peak"], bz_stats["abs_peak"]))
    set_feature(row, "X1__center_By_Bz_abs_peak_ratio", safe_div(by_stats["abs_peak"], bz_stats["abs_peak"]))
    set_feature(row, "X1__center_Bx_By_abs_peak_ratio", safe_div(bx_stats["abs_peak"], by_stats["abs_peak"]))
    set_feature(row, "X1__center_Bx_Bz_corr", corr(bx, bz))
    set_feature(row, "X1__center_By_Bz_corr", corr(by, bz))
    set_feature(row, "X1__center_Bx_By_corr", corr(bx, by))
    vmag = np.sqrt(bx * bx + by * by + bz * bz)
    vmag_stats = line_stats(x, vmag)
    set_feature(row, "X1__center_vmag_abs_peak", vmag_stats["abs_peak"])
    set_feature(row, "X1__center_vmag_peak_x_m", vmag_stats["abs_peak_x_m"])
    set_feature(row, "X1__center_vmag_width50_x_m", vmag_stats["width50_x_m"])
    set_feature(row, "X1__center_vmag_energy", vmag_stats["energy"])
    return row, failures


def quality_rows(rows: list[dict[str, Any]], adequacy: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    row_count = max(len(rows), 1)
    for group in [
        "axial_local_features",
        "axial_decay_features",
        "tangential_envelope_features",
        "cross_axis_features",
        "fit_residuals",
    ]:
        names = [name for name, feature_group in FEATURE_DEFS if feature_group == group]
        finite_total = 0
        valid_total = 0
        for row in rows:
            finite_total += sum(1 for name in names if np.isfinite(safe_float(row.get(name))))
            valid_total += sum(1 for name in names if bool(row.get(f"valid__{name}", False)))
        total = row_count * len(names)
        fit_attempts = 0
        fit_success = 0
        fit_failures = 0
        if group in {"tangential_envelope_features", "fit_residuals"}:
            fit_attempts = int(len(rows) * len(AXES)) if bool(adequacy["fit_feasibility_by_geometry"]) else 0
            fit_success = int(
                sum(
                    int(safe_float(row.get(f"T1__{axis}_env_gauss_fit_success", 0.0)) == 1.0)
                    for row in rows
                    for axis in AXES
                )
            )
            fit_failures = max(fit_attempts - fit_success, 0)
        out.append(
            {
                "feature_group": group,
                "feature_count": len(names),
                "finite_fraction": float(finite_total / total) if total else math.nan,
                "valid_fraction": float(valid_total / total) if total else math.nan,
                "fit_attempt_count": fit_attempts,
                "fit_success_count": fit_success,
                "fit_failure_count": fit_failures,
                "notes": "degraded mode: tangential fits not attempted" if group in {"tangential_envelope_features", "fit_residuals"} and fit_attempts == 0 else "",
            }
        )
    out.append(
        {
            "feature_group": "feature_validity_flags",
            "feature_count": len(validity_columns()),
            "finite_fraction": 1.0,
            "valid_fraction": 1.0,
            "fit_attempt_count": 0,
            "fit_success_count": 0,
            "fit_failure_count": 0,
            "notes": "one validity flag column per feature column",
        }
    )
    return out


def extract_feature_table(
    delta_b: np.ndarray,
    sensor_x_m: np.ndarray,
    scan_line_y_m: np.ndarray,
    axis_names: list[str] | tuple[str, ...],
    sample_ids: list[str] | np.ndarray | None = None,
    split: list[str] | np.ndarray | None = None,
    dataset_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    arr = normalise_delta(delta_b)
    x = np.asarray(sensor_x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(scan_line_y_m, dtype=np.float64).reshape(-1)
    adequacy = input_adequacy(arr, x, y, axis_names)
    if not bool(adequacy["axis_order_ok"]):
        raise ValueError(f"axis order must be {AXES}, got {list(axis_names)}")
    if not bool(adequacy["sensor_x_matches_shape"]) or not bool(adequacy["scan_line_y_matches_shape"]):
        raise ValueError(f"metadata coordinate lengths do not match delta_b shape: {adequacy}")

    sample_ids_list = [str(x) for x in sample_ids] if sample_ids is not None else [f"sample_{i:04d}" for i in range(arr.shape[0])]
    split_list = [str(x) for x in split] if split is not None else [""] * arr.shape[0]
    if len(sample_ids_list) != arr.shape[0] or len(split_list) != arr.shape[0]:
        raise ValueError("sample_ids and split must match sample count")

    rows: list[dict[str, Any]] = []
    all_failures: list[str] = []
    for i in range(arr.shape[0]):
        feature_row, failures = extract_one(arr[i], x, y, adequacy)
        all_failures.extend(failures)
        meta = {
            "sample_id": sample_ids_list[i],
            "split": split_list[i],
            "dataset_id": dataset_id,
            "feature_schema_version": SCHEMA_VERSION,
            "scan_line_count": int(arr.shape[2]),
            "sensor_x_count": int(arr.shape[3]),
            "piao_full_compatible": True,
            "exact_piao_full": False,
            "full_feature_ready": bool(adequacy["full_feature_ready"]),
            "full_candidate_mode": bool(adequacy["full_candidate_mode"]),
            "degraded_mode": bool(adequacy["degraded_mode"]),
            "degraded_mode_reason": str(adequacy["degraded_mode_reason"]),
        }
        row = {**meta, **feature_row, "fit_failure_reasons": ";".join(failures)}
        rows.append(row)

    q_rows = quality_rows(rows, adequacy)
    manifest = {
        "framework": "surface_rbc_nls_full_compatible_features",
        "feature_schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_shape": list(arr.shape),
        "axis_names": list(axis_names),
        "scan_line_count": int(arr.shape[2]),
        "sensor_x_count": int(arr.shape[3]),
        "full_feature_ready": bool(adequacy["full_feature_ready"]),
        "full_candidate_mode": bool(adequacy["full_candidate_mode"]),
        "degraded_mode": bool(adequacy["degraded_mode"]),
        "degraded_mode_reason": str(adequacy["degraded_mode_reason"]),
        "piao_full_compatible": True,
        "exact_piao_full": False,
        "feature_count": len(feature_columns()),
        "validity_flag_count": len(validity_columns()),
        "fit_failure_count": len(all_failures),
        "fit_failure_reasons": sorted(set(all_failures)),
        "outputs": {
            "features_csv": str(FEATURES),
            "quality_csv": str(QUALITY),
            "summary": str(SUMMARY),
            "manifest": str(MANIFEST),
        },
        "forbidden_actions": {
            "comsol_run": False,
            "training_run": False,
            "data_npz_written": False,
            "current_baseline_updated": False,
        },
    }
    return rows, q_rows, manifest


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    features_path: Path,
    quality_path: Path,
    summary_path: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
    q_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    fields = METADATA_FIELDS + feature_columns() + validity_columns() + ["fit_failure_reasons"]
    write_csv(features_path, rows, fields)
    write_csv(
        quality_path,
        q_rows,
        [
            "feature_group",
            "feature_count",
            "finite_fraction",
            "valid_fraction",
            "fit_attempt_count",
            "fit_success_count",
            "fit_failure_count",
            "notes",
        ],
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["outputs"] = {
        "features_csv": str(features_path),
        "quality_csv": str(quality_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "surface RBC NLS full-compatible feature extraction summary",
                "",
                f"dataset_id: {manifest.get('dataset_id')}",
                f"feature_schema_version: {SCHEMA_VERSION}",
                f"input_shape: {manifest.get('input_shape')}",
                f"axis_names: {manifest.get('axis_names')}",
                f"scan_line_count: {manifest.get('scan_line_count')}",
                f"sensor_x_count: {manifest.get('sensor_x_count')}",
                f"piao_full_compatible: {bool_text(True)}",
                f"exact_piao_full: {bool_text(False)}",
                f"full_feature_ready: {bool_text(bool(manifest.get('full_feature_ready')))}",
                f"full_candidate_mode: {bool_text(bool(manifest.get('full_candidate_mode')))}",
                f"degraded_mode: {bool_text(bool(manifest.get('degraded_mode')))}",
                f"degraded_mode_reason: {manifest.get('degraded_mode_reason')}",
                f"feature_count: {manifest.get('feature_count')}",
                f"validity_flag_count: {manifest.get('validity_flag_count')}",
                f"fit_failure_count: {manifest.get('fit_failure_count')}",
                "",
                "mode_boundary: M<5 is degraded-compatible; M>=5 may attempt full-compatible tangential envelope fitting; M>=9 may be full-candidate.",
                "claim_boundary: this output is not exact Piao full NLS.",
                "actions: no COMSOL, no training, no data/NPZ writing, no CURRENT_BASELINE update.",
                f"features_csv: {features_path}",
                f"quality_csv: {quality_path}",
                f"manifest_json: {manifest_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--quality", type=Path, default=QUALITY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    from load_true_3d_rbc_pilot_dataset import load_dataset

    dataset = load_dataset(args.dataset_id)
    rows, q_rows, manifest = extract_feature_table(
        delta_b=dataset.delta_b,
        sensor_x_m=dataset.sensor_x,
        scan_line_y_m=dataset.scan_line_y,
        axis_names=dataset.axis_names,
        sample_ids=dataset.sample_ids,
        split=dataset.split,
        dataset_id=dataset.dataset_id,
    )
    write_outputs(args.features, args.quality, args.summary, args.manifest, rows, q_rows, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

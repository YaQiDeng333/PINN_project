#!/usr/bin/env python
"""Self-tests for the surface RBC NLS-lite feature scripts.

These tests use synthetic delta_b arrays only. They do not load COMSOL data,
train models, write data/NPZ files, or update baseline documents.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def synthetic_delta() -> tuple[np.ndarray, np.ndarray]:
    sensor_x = np.linspace(-0.04, 0.04, 201, dtype=np.float64)
    delta = np.zeros((3, 3, sensor_x.size), dtype=np.float64)
    base = np.exp(-0.5 * ((sensor_x - 0.006) / 0.006) ** 2)
    for axis_i, axis_scale in enumerate((1.0, 0.45, 1.8)):
        for line_i, line_scale in enumerate((0.75, 1.0, 0.65)):
            signal = axis_scale * line_scale * base
            signal -= 0.25 * axis_scale * line_scale * np.exp(-0.5 * ((sensor_x + 0.010) / 0.004) ** 2)
            delta[axis_i, line_i] = signal
    return delta, sensor_x


def test_extract_sample_features_are_prefixed_and_delta_only() -> None:
    from extract_surface_rbc_nls_lite_features import extract_sample_features

    delta, sensor_x = synthetic_delta()
    row, quality = extract_sample_features(delta, sensor_x)
    assert row
    assert all(name.startswith("nlslite_") for name in row)
    forbidden_tokens = ("L_m", "W_m", "D_m", "wLD", "wWD", "wLW", "rbc_params", "profile", "mask", "sample_id", "split")
    assert not any(token in name for token in forbidden_tokens for name in row)
    assert all(math.isfinite(float(value)) for value in row.values())
    assert 0.0 <= quality["fit_success_rate"] <= 1.0
    assert 0.0 <= quality["fallback_rate"] <= 1.0
    assert row["nlslite_Bx_y0_positive_peak"] > 0.9
    assert row["nlslite_Bx_y0_negative_peak_abs"] > 0.20
    assert row["nlslite_Bx_y0_peak_to_peak"] > 1.1
    assert abs(row["nlslite_Bx_y0_abs_peak_position_m"] - 0.006) < 0.002
    assert row["nlslite_Bx_line_to_line_amplitude_spread"] > 0.0
    assert row["nlslite_y0_Bz_to_Bx_abs_peak_ratio"] > 1.5


def test_zero_signal_falls_back_without_nan() -> None:
    from extract_surface_rbc_nls_lite_features import extract_sample_features

    sensor_x = np.linspace(-0.04, 0.04, 201, dtype=np.float64)
    row, quality = extract_sample_features(np.zeros((3, 3, 201), dtype=np.float64), sensor_x)
    assert all(math.isfinite(float(value)) for value in row.values())
    assert quality["fallback_rate"] == 1.0
    assert quality["fit_success_rate"] == 0.0
    assert row["nlslite_Bx_y0_fallback_used"] == 1.0
    assert row["nlslite_Bx_y0_fit_residual"] == 0.0


def test_correlation_audit_uses_labels_only_as_targets() -> None:
    from audit_surface_rbc_nls_lite_feature_correlations import compute_correlations

    features = {
        "nlslite_signal": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "nlslite_constant": np.ones(4, dtype=np.float64),
    }
    targets = {"L_m": np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)}
    rows = compute_correlations(features, targets)
    signal = next(row for row in rows if row["feature"] == "nlslite_signal")
    constant = next(row for row in rows if row["feature"] == "nlslite_constant")
    assert signal["target"] == "L_m"
    assert signal["pearson_r"] > 0.999
    assert constant["pearson_r"] == 0.0


def test_route_decision_promotes_stable_useful_features() -> None:
    from decide_surface_rbc_nls_lite_feature_route import decide_route

    quality = {
        "sample_count": 4,
        "feature_count": 12,
        "overall_finite_fraction": 1.0,
        "fit_success_rate": 0.95,
        "fallback_rate": 0.05,
        "max_abs_correlation": 0.72,
        "top_features_by_target": {"L_m": "nlslite_signal"},
    }
    decision = decide_route(quality)
    assert decision["nlslite_features_stable"] is True
    assert decision["fit_failure_acceptable"] is True
    assert decision["enter_24_1_feature_baseline"] is True
    assert decision["real_experiment_preprocessing_fit"] in {"yes", "yes_with_calibration_caveat"}


def main() -> int:
    tests = [
        test_extract_sample_features_are_prefixed_and_delta_only,
        test_zero_signal_falls_back_without_nan,
        test_correlation_audit_uses_labels_only_as_targets,
        test_route_decision_promotes_stable_useful_features,
    ]
    for test in tests:
        test()
    print(f"surface_rbc_nls_lite_self_tests: {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

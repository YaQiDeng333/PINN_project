#!/usr/bin/env python
"""Validate input adequacy for the NLS full-compatible feature framework."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

import extract_surface_rbc_nls_full_compatible_features as extractor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_full_compatible_input_validation_summary.txt"
METRICS = ROOT / "results/metrics/surface_rbc_nls_full_compatible_input_validation.csv"

FIELDS = ["check_name", "pass", "observed", "expected", "severity", "notes"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any, severity: str, notes: str = "") -> None:
    rows.append(
        {
            "check_name": name,
            "pass": bool(passed),
            "observed": observed,
            "expected": expected,
            "severity": severity,
            "notes": notes,
        }
    )


def validate_arrays(
    delta_b: np.ndarray,
    sensor_x_m: np.ndarray,
    scan_line_y_m: np.ndarray,
    axis_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adequacy = extractor.input_adequacy(delta_b, sensor_x_m, scan_line_y_m, axis_names)
    x = np.asarray(sensor_x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(scan_line_y_m, dtype=np.float64).reshape(-1)
    rows: list[dict[str, Any]] = []
    add(rows, "axis_order", bool(adequacy["axis_order_ok"]), axis_names, extractor.AXES, "blocker")
    add(rows, "scan_line_count_min_for_degraded", int(adequacy["scan_line_count"]) >= 3, adequacy["scan_line_count"], ">=3", "warning")
    add(rows, "scan_line_count_min_for_full", int(adequacy["scan_line_count"]) >= extractor.FULL_MIN_LINE_COUNT, adequacy["scan_line_count"], ">=5", "degraded")
    add(rows, "scan_line_count_recommended_full", int(adequacy["scan_line_count"]) >= extractor.FULL_CANDIDATE_LINE_COUNT, adequacy["scan_line_count"], ">=9", "candidate")
    add(rows, "sensor_x_count", bool(adequacy["sensor_x_matches_shape"]) and int(adequacy["sensor_x_count"]) >= extractor.MIN_SENSOR_X_COUNT, adequacy["sensor_x_count"], f">={extractor.MIN_SENSOR_X_COUNT}", "blocker")
    y_spacing = "n/a"
    if y.size >= 2:
        y_spacing = [float(v) for v in np.diff(y).tolist()]
    add(rows, "y_line_spacing", bool(adequacy["y_line_spacing_ok"]), y_spacing, "strictly increasing and approximately uniform", "blocker")
    add(rows, "missing_values", int(adequacy["missing_values"]) == 0, adequacy["missing_values"], 0, "blocker")
    add(rows, "fit_feasibility", bool(adequacy["fit_feasibility_by_geometry"]), f"M={adequacy['scan_line_count']}; K={adequacy['sensor_x_count']}", "M>=5 and K>=5", "degraded")
    add(rows, "full_feature_ready", bool(adequacy["full_feature_ready"]), adequacy["full_feature_ready"], True, "degraded", "False is acceptable only in degraded-compatible mode.")
    add(rows, "degraded_mode_reason", not bool(adequacy["degraded_mode"]) or bool(adequacy["degraded_mode_reason"]), adequacy["degraded_mode_reason"], "reason present when degraded", "info")
    return rows, adequacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    from load_true_3d_rbc_pilot_dataset import load_dataset

    dataset = load_dataset(args.dataset_id)
    rows, adequacy = validate_arrays(dataset.delta_b, dataset.sensor_x, dataset.scan_line_y, dataset.axis_names)
    write_csv(args.metrics, rows)
    status = "full-ready" if bool(adequacy["full_feature_ready"]) else "degraded-compatible"
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface RBC NLS full-compatible input validation summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"input_shape: {list(dataset.delta_b.shape)}",
                f"axis_order: {dataset.axis_names}",
                f"scan_line_count: {adequacy['scan_line_count']}",
                f"sensor_x_count: {adequacy['sensor_x_count']}",
                f"scan_line_y_m: {[float(v) for v in np.asarray(dataset.scan_line_y).reshape(-1).tolist()]}",
                f"missing_values: {adequacy['missing_values']}",
                f"fit_feasibility: {str(bool(adequacy['fit_feasibility_by_geometry'])).lower()}",
                f"full_feature_ready: {str(bool(adequacy['full_feature_ready'])).lower()}",
                f"full_candidate_mode: {str(bool(adequacy['full_candidate_mode'])).lower()}",
                f"degraded_mode: {str(bool(adequacy['degraded_mode'])).lower()}",
                f"degraded_mode_reason: {adequacy['degraded_mode_reason']}",
                f"status: {status}",
                "",
                "decision: current v3_240 is not full-ready because scan_line_count=3 < 5.",
                "actions: no COMSOL, no training, no data/NPZ writing, no CURRENT_BASELINE update.",
                f"metrics_csv: {args.metrics}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

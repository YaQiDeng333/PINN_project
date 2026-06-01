#!/usr/bin/env python
"""Synthetic tests for the NLS full-compatible feature framework.

The tests intentionally use only synthetic arrays. They do not read real data,
do not write NPZ files, do not train, and do not call COMSOL.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_full_compatible_test_summary.txt"
RESULTS = ROOT / "results/metrics/surface_rbc_nls_full_compatible_test_results.csv"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["test_name", "pass", "observed", "expected", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def synthetic_roi(scan_line_count: int, *, flat: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor_x = np.linspace(-0.02, 0.02, 41, dtype=np.float64)
    scan_y = np.linspace(-0.004, 0.004, scan_line_count, dtype=np.float64)
    delta = np.zeros((2, 3, scan_line_count, sensor_x.size), dtype=np.float64)
    for si in range(delta.shape[0]):
        for ai in range(3):
            for yi, y in enumerate(scan_y):
                if flat:
                    delta[si, ai, yi] = 0.0
                    continue
                x0 = (si - 0.5) * 0.001 + ai * 0.0003
                sigma_x = 0.004 + ai * 0.0008
                envelope = math.exp(-((float(y) - 0.0004 * ai) ** 2) / (2.0 * (0.0018 + ai * 0.0002) ** 2))
                base = np.exp(-((sensor_x - x0) ** 2) / (2.0 * sigma_x**2))
                if ai == 1:
                    base = ((sensor_x - x0) / sigma_x) * base
                delta[si, ai, yi] = (1.0 + 0.2 * si + 0.1 * ai) * envelope * base
    return delta, sensor_x, scan_y


def add_result(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append(
        {
            "test_name": name,
            "pass": bool(passed),
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def run(_: Any = None) -> int:
    rows: list[dict[str, Any]] = []
    try:
        import extract_surface_rbc_nls_full_compatible_features as extractor
    except Exception as exc:
        add_result(rows, "import_extractor", False, repr(exc), "module imports", "Extractor module is required.")
        write_csv(RESULTS, rows)
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(
            "\n".join(
                [
                    "surface RBC NLS full-compatible synthetic test summary",
                    "status: fail",
                    "reason: extractor import failed",
                    f"error: {exc!r}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 1

    feature_names: list[str] | None = None
    for line_count, expected_ready, expected_candidate in [(3, False, False), (5, True, False), (9, True, True)]:
        delta, sensor_x, scan_y = synthetic_roi(line_count)
        feature_rows, quality_rows, manifest = extractor.extract_feature_table(
            delta_b=delta,
            sensor_x_m=sensor_x,
            scan_line_y_m=scan_y,
            axis_names=["Bx", "By", "Bz"],
            sample_ids=[f"synthetic_{line_count}_a", f"synthetic_{line_count}_b"],
            split=["synthetic", "synthetic"],
            dataset_id=f"synthetic_{line_count}line",
        )
        current_feature_names = extractor.feature_columns()
        if feature_names is None:
            feature_names = current_feature_names
        add_result(
            rows,
            f"{line_count}_line_full_feature_ready",
            bool(manifest["full_feature_ready"]) is expected_ready,
            manifest["full_feature_ready"],
            expected_ready,
        )
        add_result(
            rows,
            f"{line_count}_line_full_candidate_mode",
            bool(manifest["full_candidate_mode"]) is expected_candidate,
            manifest["full_candidate_mode"],
            expected_candidate,
        )
        add_result(
            rows,
            f"{line_count}_line_validity_flags_present",
            all(f"valid__{name}" in feature_rows[0] for name in current_feature_names),
            sum(1 for name in current_feature_names if f"valid__{name}" in feature_rows[0]),
            len(current_feature_names),
        )
        add_result(
            rows,
            f"{line_count}_line_feature_name_stability",
            current_feature_names == feature_names,
            len(current_feature_names),
            len(feature_names),
        )
        add_result(
            rows,
            f"{line_count}_line_quality_rows_present",
            len(quality_rows) >= 5,
            len(quality_rows),
            ">=5",
        )

    flat_delta, flat_x, flat_y = synthetic_roi(5, flat=True)
    flat_rows, flat_quality, flat_manifest = extractor.extract_feature_table(
        delta_b=flat_delta,
        sensor_x_m=flat_x,
        scan_line_y_m=flat_y,
        axis_names=["Bx", "By", "Bz"],
        sample_ids=["flat_a", "flat_b"],
        split=["synthetic", "synthetic"],
        dataset_id="synthetic_failed_fit",
    )
    failure_reasons = " ".join(str(row.get("fit_failure_reasons", "")) for row in flat_rows)
    fit_quality = {row["feature_group"]: row for row in flat_quality}
    add_result(
        rows,
        "failed_fit_not_silent",
        "zero_or_constant_envelope" in failure_reasons
        and int(fit_quality["tangential_envelope_features"]["fit_failure_count"]) > 0,
        failure_reasons,
        "fit failure reason and aggregate failure count",
    )
    add_result(
        rows,
        "failed_fit_keeps_compatibility_gate",
        bool(flat_manifest["full_feature_ready"]) is True,
        flat_manifest["full_feature_ready"],
        True,
        "Input can be adequate even when per-sample fits fail.",
    )
    direct_failure = extractor.fit_gaussian_envelope(np.zeros(5, dtype=np.float64), np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    add_result(
        rows,
        "curve_fit_failure_not_silent",
        direct_failure["fit_success"] == 0.0 and str(direct_failure["failure_reason"]).startswith("curve_fit_failed:"),
        direct_failure,
        "fit_success=0 with explicit curve_fit_failed reason",
    )

    passed = all(bool(row["pass"]) for row in rows)
    write_csv(RESULTS, rows)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "surface RBC NLS full-compatible synthetic test summary",
                f"status: {'pass' if passed else 'fail'}",
                f"test_count: {len(rows)}",
                f"passed_count: {sum(1 for row in rows if bool(row['pass']))}",
                "real_data_used: false",
                "npz_used: false",
                "training_run: false",
                "comsol_run: false",
                f"results_csv: {RESULTS}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())

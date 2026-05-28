#!/usr/bin/env python
"""Build the 20.97 real-data preprocessing plan artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/true_3d_rbc_real_data_preprocessing_plan.md"
STEPS_CSV = ROOT / "results/metrics/true_3d_rbc_real_data_preprocessing_steps.csv"


STEPS: list[dict[str, Any]] = [
    {"step": 1, "name": "read_raw_bxyz", "input": "raw Bx/By/Bz scans or prepared delta_b", "output": "array with axis metadata", "blocker_if_missing": "Bx/By/Bz tri-axis data"},
    {"step": 2, "name": "spatial_align_three_axes", "input": "Bx/By/Bz channels", "output": "aligned tri-axis field", "blocker_if_missing": "sensor_alignment_status"},
    {"step": 3, "name": "convert_units_to_tesla", "input": "field array and unit metadata", "output": "Tesla-valued field array", "blocker_if_missing": "known unit"},
    {"step": 4, "name": "match_no_defect_reference", "input": "defect scan and no-defect reference", "output": "matched reference pair", "blocker_if_missing": "no_defect_reference_id"},
    {"step": 5, "name": "compute_delta_b", "input": "b_defect and b_no_defect", "output": "delta_b=b_defect-b_no_defect", "blocker_if_missing": "trusted delta_b or raw pair"},
    {"step": 6, "name": "resample_sensor_x", "input": "sensor_x coordinate and signal", "output": "201 x-samples", "blocker_if_missing": "resampling map to 201"},
    {"step": 7, "name": "map_scan_line_y", "input": "scan_line_y metadata", "output": "three scan lines matching [-0.001,0,0.001] convention", "blocker_if_missing": "three-line y mapping"},
    {"step": 8, "name": "write_sensor_z_metadata", "input": "measured liftoff", "output": "sensor_z_m per sample", "blocker_if_missing": "sensor_z_m"},
    {"step": 9, "name": "record_gain_calibration", "input": "gain/amplitude calibration info", "output": "diagnostic calibration flag", "blocker_if_missing": "not a blocker, but warning if unknown"},
    {"step": 10, "name": "call_20_96_inference_runner", "input": "delta_b + sensor_z_m + metadata", "output": "RBC params, profile/depth, projected mask", "blocker_if_missing": "validated intake manifest"},
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    write_csv(STEPS_CSV, STEPS)
    lines = [
        "# True 3D RBC Real-Data Preprocessing Plan",
        "",
        "This plan prepares real experimental MFL observations for the 20.96 liftoff-conditioned inference runner. It does not train a model, run COMSOL, write NPZ data, or change CURRENT_BASELINE.md.",
        "",
        "Main chain: raw Bx/By/Bz -> alignment -> Tesla units -> matched no-defect reference -> delta_b -> 201-sample x grid -> three scan lines -> sensor_z_m metadata -> 20.96 inference runner.",
        "",
        "Gain/amplitude calibration remains diagnostic. It may be recorded and compared, but it does not replace the baseline or A2 companion routing contract.",
        "",
        "## Steps",
    ]
    for row in STEPS:
        lines.append(f"{row['step']}. `{row['name']}`: {row['input']} -> {row['output']}. Blocker if missing: {row['blocker_if_missing']}.")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

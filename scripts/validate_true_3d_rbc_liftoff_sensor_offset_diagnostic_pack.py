#!/usr/bin/env python
"""Validate the 20.90 true-3D RBC liftoff/sensor-offset diagnostic pack.

This validation only reads the explicit 20.90 plan CSV and diagnostic NPZ
generated from that plan. It does not discover newest/latest data, train, run
COMSOL, or modify baseline files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    V3_240_DATASET_ID,
    gate_manifest,
    load_dataset,
    resolve_dataset,
    write_csv,
)


PLAN_CSV = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_plan.csv"
DIAGNOSTIC_NPZ = (
    ROOT
    / "data/comsol_mfl/generated/true_3d_rbc_liftoff_sensor_offset_diagnostic_pack/true_3d_rbc_liftoff_sensor_offset_diagnostic_pack.npz"
)
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_sensor_offset_validation_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_validation_metrics.csv"

FIELDS = ["check_name", "pass", "observed", "expected", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 20.90 liftoff/sensor-offset diagnostic pack.")
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--diagnostic-npz", type=Path, default=DIAGNOSTIC_NPZ)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def run(args: argparse.Namespace) -> int:
    check_overwrite([args.summary, args.metrics], args.overwrite)
    checks: list[dict[str, Any]] = []

    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    gate_checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    dataset = load_dataset(args.dataset_id)
    failed_gate = [row for row in gate_checks if not row["pass"]]
    add(checks, "registry_manifest_gate", not failed_gate, len(failed_gate), 0, "explicit v3_240 registry/manifest gate")
    add(checks, "plan_csv_exists", args.plan_csv.exists(), str(args.plan_csv), "existing 20.90 plan")
    add(checks, "diagnostic_npz_exists", args.diagnostic_npz.exists(), str(args.diagnostic_npz), "generated data path is ignored and uncommitted")

    plan_rows = read_csv(args.plan_csv) if args.plan_csv.exists() else []
    comsol_plan = [row for row in plan_rows if row.get("requires_comsol", "").lower() == "true"]
    post_plan = [row for row in plan_rows if row.get("requires_comsol", "").lower() == "false"]
    add(checks, "plan_row_count", len(plan_rows) == 132, len(plan_rows), 132)
    add(checks, "plan_comsol_row_count", len(comsol_plan) == 96, len(comsol_plan), 96)
    add(checks, "plan_postprocess_row_count", len(post_plan) == 36, len(post_plan), 36)
    add(checks, "plan_base_count", len({row.get("base_sample_id", "") for row in plan_rows}) == 12, len({row.get("base_sample_id", "") for row in plan_rows}), 12)

    arrays = load_npz_arrays(args.diagnostic_npz)
    required = [
        "delta_b",
        "b_defect",
        "b_no_defect",
        "sample_ids",
        "base_sample_ids",
        "variant_name",
        "factor_group",
        "split",
        "rbc_params",
        "profile_pose",
        "profile_depth_grid_m",
        "profile_depth_map_xy_m",
        "projected_mask_2d",
        "sensor_x",
        "scan_line_y",
    ]
    missing = [key for key in required if key not in arrays]
    add(checks, "diagnostic_npz_required_fields", not missing, ",".join(missing) if missing else "none", "all required fields")

    if arrays:
        n = int(arrays.get("delta_b", np.empty((0,))).shape[0])
        add(checks, "diagnostic_success_rows", n >= 92, n, ">=92 of 96 COMSOL rows", "95% success threshold for factor conclusions")
        add(checks, "delta_shape", arrays["delta_b"].shape[1:] == (3, 3, 201), list(arrays["delta_b"].shape), "(n,3,3,201)")
        add(checks, "finite_delta_b", bool(np.isfinite(arrays["delta_b"]).all()), bool(np.isfinite(arrays["delta_b"]).all()), True)
        delta_error = float(np.max(np.abs(arrays["delta_b"] - (arrays["b_defect"] - arrays["b_no_defect"])))) if n else float("nan")
        add(checks, "delta_recompute_error", delta_error <= 1.0e-8, f"{delta_error:.6e}", "<=1e-8", "float32 diagnostic NPZ storage")
        nominal = [str(v) for v in arrays.get("variant_name", []) if str(v) == "nominal"]
        add(checks, "nominal_rows", len(nominal) == 12, len(nominal), 12)
        base_ids = {str(sid) for sid in dataset.sample_ids}
        joined = {str(sid) for sid in arrays.get("base_sample_ids", [])}.issubset(base_ids)
        add(checks, "labels_join_dataset", joined, len({str(sid) for sid in arrays.get("base_sample_ids", [])} & base_ids), "all diagnostic base_sample_ids in v3_240")

    passed = all(bool(row["pass"]) for row in checks)
    write_csv(args.metrics, checks, FIELDS)
    lines = [
        "20.90 true 3D RBC liftoff / sensor-offset diagnostic validation summary",
        "",
        f"dataset_id: {args.dataset_id}",
        f"registry_npz_path: {npz_path}",
        f"plan_csv: {args.plan_csv}",
        f"diagnostic_npz: {args.diagnostic_npz}",
        f"validation_pass: {passed}",
        f"plan_rows: {len(plan_rows)}",
        f"comsol_plan_rows: {len(comsol_plan)}",
        f"postprocess_plan_rows: {len(post_plan)}",
        f"successful_comsol_rows_in_npz: {int(arrays.get('delta_b', np.empty((0,))).shape[0]) if arrays else 0}",
        "latest_newest_npz_scan: false",
        "COMSOL_run_by_this_script: false",
        "training_run: false",
        "baseline_update: false",
        "",
        "Failed checks:",
    ]
    lines.extend(f"- {row['check_name']}: observed={row['observed']} expected={row['expected']} notes={row['notes']}" for row in checks if not row["pass"])
    if passed:
        lines.append("- none")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("diagnostic pack validation failed; see validation summary")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

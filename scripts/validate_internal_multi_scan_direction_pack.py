#!/usr/bin/env python
"""验证 23.2b internal multi-scan-direction y_scan top-up pack。

本脚本只读取显式路径，不扫描 latest/newest；不训练、不运行 COMSOL、不写 data/NPZ、
不更新 CURRENT_BASELINE.md。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLAN_CSV = ROOT / "results/metrics/internal_multi_scan_direction_diagnostic_pack_plan.csv"
Y_SCAN_NPZ = ROOT / "data/comsol_mfl/generated/internal_multi_scan_direction_pack/internal_multi_scan_direction_y_scan_pack_v1.npz"
COMSOL_INVENTORY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_internal_multi_scan_direction_pack.csv")
COMSOL_SUMMARY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\internal_multi_scan_direction_pack_summary.txt")
SUMMARY = ROOT / "results/summaries/internal_multi_scan_direction_pack_validation_summary.txt"
METRICS = ROOT / "results/metrics/internal_multi_scan_direction_pack_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_multi_scan_direction_pack_group_summary.csv"

EXPECTED_VARIANTS = {"D1_y_scan_5line_z0p008", "D2_y_scan_9line_z0p008"}
EXPECTED_LINE_COUNTS = {5, 9}
EXPECTED_SENSOR_Z = 0.008


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 23.2b y_scan top-up pack。")
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--y-scan-npz", type=Path, default=Y_SCAN_NPZ)
    parser.add_argument("--comsol-inventory", type=Path, default=COMSOL_INVENTORY)
    parser.add_argument("--comsol-summary", type=Path, default=COMSOL_SUMMARY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def strings(arr: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(arr).reshape(-1).tolist()]


def git_lines(args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append(
        {
            "check_name": name,
            "pass": bool(passed),
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def no_forbidden_staged() -> tuple[bool, str]:
    staged = git_lines(["diff", "--cached", "--name-only"])
    forbidden = [
        path
        for path in staged
        if path.startswith("data/")
        or path.endswith(".npz")
        or path.endswith(".mph")
        or path.startswith("checkpoints/")
        or path.startswith("results/previews/")
        or path.startswith("notes/")
        or path == "CURRENT_BASELINE.md"
        or path == "scripts/visualize_current_baseline.py"
    ]
    return not forbidden, "; ".join(forbidden)


def protected_workdirs_clean() -> tuple[bool, str]:
    lines = git_lines(["status", "--short", "--", "data", "checkpoints", "results/previews", "notes", "CURRENT_BASELINE.md", "scripts/visualize_current_baseline.py"])
    return len(lines) == 0, "; ".join(lines)


def complete_bases(arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    variants_by_base: dict[str, set[str]] = defaultdict(set)
    for base, variant in zip(strings(arrays.get("base_group_id", np.asarray([]))), strings(arrays.get("observation_variant", np.asarray([]))), strict=False):
        variants_by_base[base].add(variant)
    complete = sum(1 for variants in variants_by_base.values() if EXPECTED_VARIANTS.issubset(variants))
    return len(variants_by_base), complete


def coordinate_check(arrays: dict[str, np.ndarray]) -> tuple[bool, float, str]:
    required = {"sensor_point_x_m", "sensor_point_y_m", "path_coordinate_m", "line_coordinate_m", "scan_line_mask"}
    if not required.issubset(arrays):
        return False, float("nan"), "缺少方向化坐标字段"
    x_grid = arrays["sensor_point_x_m"].astype(float)
    y_grid = arrays["sensor_point_y_m"].astype(float)
    path = arrays["path_coordinate_m"].astype(float)
    lines = arrays["line_coordinate_m"].astype(float)
    mask = arrays["scan_line_mask"].astype(bool)
    max_err = 0.0
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            if not mask[i, j]:
                continue
            max_err = max(max_err, float(np.max(np.abs(y_grid[i, j, :] - path[i]))))
            max_err = max(max_err, float(np.max(np.abs(x_grid[i, j, :] - lines[i, j]))))
            if np.ptp(y_grid[i, j, :]) <= 0:
                return False, max_err, "y_scan 路径没有沿 y 方向变化"
            if np.ptp(x_grid[i, j, :]) > 1e-12:
                return False, max_err, "y_scan 单条线内 x offset 不恒定"
    return max_err < 1e-9, max_err, "y_scan=(x_line, y_path, sensor_z_m)"


def group_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(len(arrays.get("sample_ids", [])))
    base = strings(arrays["base_group_id"]) if "base_group_id" in arrays else [""] * n
    for field in ["observation_variant", "scan_line_count", "shape_type", "burial_depth_level", "size_level", "aspect_bin", "source_split"]:
        if field not in arrays:
            continue
        buckets: dict[str, list[int]] = defaultdict(list)
        for idx, value in enumerate(np.asarray(arrays[field]).reshape(-1).tolist()):
            buckets[str(value)].append(idx)
        for value, indices in sorted(buckets.items()):
            rows.append({"group_type": field, "group_value": value, "row_count": len(indices), "base_count": len({base[i] for i in indices})})
    return rows


def run(args: argparse.Namespace) -> int:
    plan = read_csv(args.plan_csv)
    inventory = read_csv(args.comsol_inventory)
    arrays = load_npz(args.y_scan_npz)
    checks: list[dict[str, Any]] = []

    success_inventory = [row for row in inventory if row.get("status") == "success"]
    n = int(len(arrays.get("sample_ids", []))) if arrays else 0
    base_count, complete_base_count = complete_bases(arrays) if arrays else (0, 0)

    add(checks, "plan_rows_60", len(plan) == 60, len(plan), 60)
    add(checks, "y_scan_npz_exists", args.y_scan_npz.exists(), args.y_scan_npz, "NPZ exists")
    add(checks, "comsol_inventory_exists", args.comsol_inventory.exists(), args.comsol_inventory, "COMSOL inventory")
    add(checks, "comsol_summary_exists", args.comsol_summary.exists(), args.comsol_summary, "COMSOL summary")
    add(checks, "success_rows_target_or_fallback", n >= 48, n, ">=48")
    add(checks, "success_rows_match_inventory", n == len(success_inventory), f"npz={n}; inventory={len(success_inventory)}", "match")
    add(checks, "base_count_target_or_fallback", base_count >= 24, base_count, ">=24")
    add(checks, "d1_d2_paired_complete", complete_base_count >= 24, complete_base_count, ">=24 complete bases")

    if arrays:
        shape_ok = arrays.get("delta_b", np.empty(())).shape == (n, 3, 9, 201)
        add(checks, "delta_shape_n_3_9_201", shape_ok, arrays.get("delta_b", np.empty(())).shape, "(N,3,9,201)")
        for key in ["delta_b", "b_defect", "b_no_defect"]:
            arr = arrays.get(key)
            add(checks, f"{key}_finite", arr is not None and np.isfinite(arr).all(), "" if arr is None else bool(np.isfinite(arr).all()), "finite")
        if {"delta_b", "b_defect", "b_no_defect"}.issubset(arrays):
            max_delta_err = float(np.max(np.abs(arrays["delta_b"] - (arrays["b_defect"] - arrays["b_no_defect"])))) if n else float("nan")
        else:
            max_delta_err = float("nan")
        add(checks, "delta_b_matches_difference", np.isfinite(max_delta_err) and max_delta_err < 1e-7, max_delta_err, "<1e-7")
        variants = set(strings(arrays.get("observation_variant", np.asarray([]))))
        add(checks, "variants_d1_d2_only", variants == EXPECTED_VARIANTS, variants, EXPECTED_VARIANTS)
        line_counts = {int(x) for x in np.asarray(arrays.get("scan_line_count", [])).reshape(-1).tolist()}
        add(checks, "line_counts_5_and_9", line_counts == EXPECTED_LINE_COUNTS, line_counts, EXPECTED_LINE_COUNTS)
        sensor_z = {round(float(x), 6) for x in np.asarray(arrays.get("sensor_z_m", [])).reshape(-1).tolist()}
        add(checks, "sensor_z_nominal", sensor_z == {EXPECTED_SENSOR_Z}, sensor_z, {EXPECTED_SENSOR_Z})
        directions = set(strings(arrays.get("scan_direction", np.asarray([]))))
        add(checks, "scan_direction_y_only", directions == {"y_scan"}, directions, {"y_scan"})
        path_axis = set(strings(arrays.get("path_coordinate_axis", np.asarray([]))))
        line_axis = set(strings(arrays.get("line_coordinate_axis", np.asarray([]))))
        add(checks, "path_axis_y", path_axis == {"y"}, path_axis, {"y"})
        add(checks, "line_axis_x", line_axis == {"x"}, line_axis, {"x"})
        coord_ok, coord_err, coord_note = coordinate_check(arrays)
        add(checks, "direction_aware_y_scan_coordinate_check", coord_ok, coord_err, "<1e-9", coord_note)
        sample_ids = strings(arrays.get("sample_ids", np.asarray([])))
        add(checks, "sample_id_unique", len(sample_ids) == len(set(sample_ids)), f"{len(sample_ids)} rows/{len(set(sample_ids))} unique", "unique")
        required_labels = {"shape_type", "L_m", "W_m", "D_m", "burial_depth_m", "depth_to_surface_m", "defect_center_xyz_m", "cavity_internal"}
        add(checks, "required_labels_present", required_labels.issubset(arrays), sorted(required_labels.intersection(arrays)), sorted(required_labels))
        if "cavity_internal" in arrays:
            add(checks, "cavity_internal_true", bool(np.asarray(arrays["cavity_internal"]).astype(bool).all()), bool(np.asarray(arrays["cavity_internal"]).astype(bool).all()), "all true")
    else:
        add(checks, "npz_loadable", False, "missing", "loadable NPZ")
        max_delta_err = float("nan")

    staged_ok, staged_notes = no_forbidden_staged()
    protected_ok, protected_notes = protected_workdirs_clean()
    add(checks, "no_forbidden_staged", staged_ok, staged_notes, "no forbidden staged files")
    add(checks, "protected_artifact_paths_clean", protected_ok, protected_notes, "no data/checkpoint/preview/notes/CURRENT_BASELINE status")

    validation_passed = all(bool(row["pass"]) for row in checks)
    write_csv(args.metrics, checks, ["check_name", "pass", "observed", "expected", "notes"])
    write_csv(args.group_summary, group_rows(arrays), ["group_type", "group_value", "row_count", "base_count"])

    lines = [
        "23.2b internal multi-scan-direction y_scan pack validation summary",
        "",
        f"planned_rows: {len(plan)}",
        f"success_rows: {n}",
        f"base_count: {base_count}",
        f"d1_d2_complete_base_count: {complete_base_count}",
        f"validation_passed: {str(validation_passed).lower()}",
        f"max_delta_error: {max_delta_err}",
        "direction_aware_coordinate: y_scan 使用 path_coordinate_axis=y，line_coordinate_axis=x；采样点为 (x_line, y_path, sensor_z_m)。",
        "training_run: false",
        "comsol_run_by_this_script: false",
        "data_npz_modified_by_this_script: false",
        "current_baseline_updated: false",
        "",
        "结论: y_scan top-up 只作为 23.2b dual-direction diagnostic 输入；不是训练集，不是 baseline。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if validation_passed else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

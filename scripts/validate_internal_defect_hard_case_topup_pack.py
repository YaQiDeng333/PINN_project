#!/usr/bin/env python
"""验证 22.2b internal defect hard-case top-up pack。

只读取显式 plan / inventory / NPZ 路径，不扫描 latest/newest，不训练，
不运行 COMSOL，不更新 CURRENT_BASELINE。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_hard_case_topup_pack_v1"
PLAN_CSV = ROOT / "results/metrics/internal_defect_hard_case_topup_plan.csv"
PACK_NPZ = ROOT / "data/comsol_mfl/generated/internal_defect_hard_case_topup_pack/internal_defect_hard_case_topup_pack_v1.npz"
COMSOL_INVENTORY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_internal_defect_hard_case_topup_pack.csv")
COMSOL_SUMMARY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\internal_defect_hard_case_topup_pack_summary.txt")
SUMMARY = ROOT / "results/summaries/internal_defect_hard_case_topup_pack_validation_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_hard_case_topup_pack_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_hard_case_topup_pack_group_summary.csv"

AXIS_NAMES = ["Bx", "By", "Bz"]
BURIALS = ["shallow", "medium", "deep", "deep_plus"]
SIZES = ["small", "medium", "large"]
ASPECTS = ["compact", "elongated_x", "elongated_y"]
SUCCESS_MINIMUM = 72
PLANNED_ROWS = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 internal defect hard-case top-up pack。")
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--pack-npz", type=Path, default=PACK_NPZ)
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


def strings(value: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(value).reshape(-1).tolist()]


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def git_lines(args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def no_forbidden_staged() -> tuple[bool, str]:
    staged = git_lines(["diff", "--cached", "--name-only"])
    forbidden = [
        path
        for path in staged
        if path.startswith("data/")
        or path.endswith(".npz")
        or path.endswith(".mph")
        or path.endswith(".pt")
        or path.endswith(".pth")
        or path.startswith("checkpoints/")
        or path.startswith("results/previews/")
        or path.startswith("notes/")
        or path == "CURRENT_BASELINE.md"
        or path == "scripts/visualize_current_baseline.py"
    ]
    return not forbidden, ",".join(forbidden)


def no_forbidden_worktree_visible() -> tuple[bool, str]:
    status = git_lines(["status", "--short", "--untracked-files=all"])
    paths: list[str] = []
    for line in status:
        raw = line[3:] if len(line) > 3 else line
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw.replace("\\", "/"))
    forbidden = [
        path
        for path in paths
        if path.startswith("data/")
        or path.endswith(".npz")
        or path.endswith(".mph")
        or path.endswith(".pt")
        or path.endswith(".pth")
        or path.startswith("checkpoints/")
        or path.startswith("results/previews/")
        or path.startswith("notes/")
        or path == "CURRENT_BASELINE.md"
        or path == "scripts/visualize_current_baseline.py"
    ]
    return not forbidden, ",".join(forbidden)


def plan_by_id(plan_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in plan_rows:
        sample_id = row.get("planned_sample_id") or row.get("sample_id")
        if sample_id:
            out[sample_id] = row
    return out


def count(values: list[str]) -> dict[str, int]:
    return dict(Counter(values))


def group_rows(plan_rows: list[dict[str, str]], inventory_rows: list[dict[str, str]], arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    plan_lookup = plan_by_id(plan_rows)
    success_ids = set(strings(arrays.get("sample_ids", np.asarray([], dtype=str)))) if arrays else set()
    rows: list[dict[str, Any]] = []
    plan_success = [plan_lookup[sid] for sid in success_ids if sid in plan_lookup]
    for field in [
        "shape_type",
        "burial_depth_level",
        "size_level",
        "aspect_bin",
        "target_id",
        "target_reason",
        "neighbor_strategy",
        "center_region",
        "split_hint",
    ]:
        for value, row_count in sorted(Counter(row.get(field, "") for row in plan_success).items()):
            rows.append({"group_field": field, "group_value": value, "success_count": row_count})
    for row in inventory_rows:
        rows.append(
            {
                "group_field": "inventory_status",
                "group_value": row.get("status", ""),
                "success_count": 1 if row.get("status") == "success" else 0,
                "sample_id": row.get("sample_id", ""),
                "failure_reason": row.get("failure_reason", ""),
            }
        )
    return rows


def run(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    plan_rows = read_csv(args.plan_csv)
    inventory_rows = read_csv(args.comsol_inventory)
    arrays = load_npz(args.pack_npz)
    plan_lookup = plan_by_id(plan_rows)
    success_inventory = [row for row in inventory_rows if row.get("status") == "success"]
    n = int(len(arrays.get("sample_ids", []))) if arrays else 0

    add(checks, "plan_csv_exists", args.plan_csv.exists(), str(args.plan_csv), "22.2 hard-case plan")
    add(checks, "planned_rows_120", len(plan_rows) == PLANNED_ROWS, len(plan_rows), PLANNED_ROWS)
    add(checks, "comsol_inventory_exists", args.comsol_inventory.exists(), str(args.comsol_inventory), "COMSOL inventory")
    add(checks, "comsol_summary_exists", args.comsol_summary.exists(), str(args.comsol_summary), "COMSOL summary")
    add(checks, "npz_exists", args.pack_npz.exists(), str(args.pack_npz), "ignored generated NPZ")
    add(checks, "inventory_success_minimum", len(success_inventory) >= SUCCESS_MINIMUM, len(success_inventory), f">={SUCCESS_MINIMUM}")
    add(checks, "npz_rows_match_success", n == len(success_inventory), n, len(success_inventory))

    if arrays:
        sample_ids = strings(arrays["sample_ids"])
        add(checks, "dataset_id_explicit", str(np.asarray(arrays["dataset_id"]).item()) == DATASET_ID, str(np.asarray(arrays["dataset_id"]).item()), DATASET_ID)
        add(checks, "no_duplicate_sample_id", len(sample_ids) == len(set(sample_ids)), len(sample_ids) - len(set(sample_ids)), 0)
        add(checks, "delta_shape", tuple(arrays["delta_b"].shape[1:]) == (3, 3, 201), tuple(arrays["delta_b"].shape), "(N,3,3,201)")
        add(checks, "b_defect_shape", tuple(arrays["b_defect"].shape) == tuple(arrays["delta_b"].shape), tuple(arrays["b_defect"].shape), tuple(arrays["delta_b"].shape))
        add(checks, "b_no_defect_shape", tuple(arrays["b_no_defect"].shape) == tuple(arrays["delta_b"].shape), tuple(arrays["b_no_defect"].shape), tuple(arrays["delta_b"].shape))
        finite = bool(np.isfinite(arrays["delta_b"]).all() and np.isfinite(arrays["b_defect"]).all() and np.isfinite(arrays["b_no_defect"]).all())
        add(checks, "signals_finite", finite, finite, True)
        delta_err = float(np.max(np.abs(arrays["delta_b"] - (arrays["b_defect"] - arrays["b_no_defect"])))) if n else float("nan")
        add(checks, "delta_check", delta_err <= 1e-7, delta_err, "<=1e-7")
        add(checks, "axis_names", strings(arrays["axis_names"]) == AXIS_NAMES, strings(arrays["axis_names"]), AXIS_NAMES)
        add(checks, "sensor_x_count", len(np.asarray(arrays["sensor_x"]).reshape(-1)) == 201, len(np.asarray(arrays["sensor_x"]).reshape(-1)), 201)
        for field in ["shape_type", "L_m", "W_m", "D_m", "D_m_or_cavity_size_m", "burial_depth_m", "depth_to_surface_m", "defect_center_xyz_m", "ground_truth_method", "cavity_internal"]:
            add(checks, f"required_field_{field}", field in arrays, field, "present")

        plan_success = [plan_lookup[sid] for sid in sample_ids if sid in plan_lookup]
        target_reasons = set(row.get("target_reason", "") for row in plan_success)
        neighbor_strategies = set(row.get("neighbor_strategy", "") for row in plan_success)
        burial_values = set(row.get("burial_depth_level", "") for row in plan_success)
        size_values = set(row.get("size_level", "") for row in plan_success)
        aspect_values = set(row.get("aspect_bin", "") for row in plan_success)
        shapes = set(row.get("shape_type", "") for row in plan_success)
        add(checks, "target_strata_has_confusion", any("confusion" in value for value in target_reasons), sorted(target_reasons), "contains confusion target")
        add(checks, "target_strata_has_center_neighbor", any("center" in value for value in neighbor_strategies | target_reasons), sorted(neighbor_strategies | target_reasons), "contains center neighbor target")
        add(checks, "shape_cuboid_ellipsoid_covered", {"internal_cuboid", "internal_ellipsoid"}.issubset(shapes), sorted(shapes), "cuboid and ellipsoid")
        add(checks, "burial_shallow_deep_plus_covered", {"shallow", "deep_plus"}.issubset(burial_values), sorted(burial_values), "shallow and deep_plus")
        add(checks, "size_medium_large_covered", {"medium", "large"}.issubset(size_values), sorted(size_values), "medium and large")
        add(checks, "aspect_compact_covered", "compact" in aspect_values, sorted(aspect_values), "compact")

    clean_staging, forbidden = no_forbidden_staged()
    add(checks, "no_forbidden_staged_artifacts", clean_staging, forbidden, "no data/NPZ/checkpoint/preview/CURRENT_BASELINE staged")
    clean_worktree, forbidden_worktree = no_forbidden_worktree_visible()
    add(
        checks,
        "no_forbidden_worktree_artifacts",
        clean_worktree,
        forbidden_worktree,
        "no non-ignored data/NPZ/checkpoint/preview/CURRENT_BASELINE worktree changes",
    )

    group = group_rows(plan_rows, inventory_rows, arrays)
    write_csv(args.metrics, checks, ["check_name", "pass", "observed", "expected", "notes"])
    write_csv(args.group_summary, group, ["group_field", "group_value", "success_count", "sample_id", "failure_reason"])
    failed = [row for row in checks if not row["pass"]]
    status = "pass" if not failed else "blocked"
    lines = [
        "22.2b internal defect hard-case top-up pack validation summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"status: {status}",
        f"planned_rows: {len(plan_rows)}",
        f"success_rows: {len(success_inventory)}",
        f"npz_rows: {n}",
        f"npz_path_ignored: {args.pack_npz}",
        f"forbidden_staged: {forbidden or 'none'}",
        f"forbidden_worktree: {forbidden_worktree or 'none'}",
        "",
        "结论：hard-case top-up pack 只用于 internal branch schema/training gate；不是 baseline，不更新 CURRENT_BASELINE。",
    ]
    if failed:
        lines.extend(["", "failed_checks:"])
        for row in failed:
            lines.append(f"- {row['check_name']}: observed={row['observed']} expected={row['expected']}")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if not failed else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""23.2 first multi-scan-direction diagnostic pack decision.

This script creates the plan CSV for a future 23.2b COMSOL top-up. It does not
run COMSOL and does not create data, NPZ, registry, or manifest artifacts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results/metrics"
SUMMARY_DIR = ROOT / "results/summaries"

SOURCE_PLAN = METRICS_DIR / "internal_richer_observation_diagnostic_pack_plan.csv"
SOURCE_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json"
PACK_PLAN = METRICS_DIR / "internal_multi_scan_direction_diagnostic_pack_plan.csv"
DECISION_MATRIX = METRICS_DIR / "internal_multi_scan_direction_decision_matrix.csv"
SUMMARY_PATH = SUMMARY_DIR / "internal_multi_scan_direction_diagnostic_pack_decision_summary.txt"

Y_SCAN_5LINE_OFFSETS = [-0.024, -0.012, 0.0, 0.012, 0.024]
Y_SCAN_9LINE_OFFSETS = [-0.032, -0.024, -0.016, -0.008, 0.0, 0.008, 0.016, 0.024, 0.032]
Y_SCAN_PATH_RANGE = [-0.010, 0.010]
SENSOR_Z_M = 0.008


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def selected_bases(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_base: dict[str, dict[str, str]] = {}
    for row in source_rows:
        if row.get("observation_variant") == "R0_3line_z0p008":
            by_base[row["base_group_id"]] = row
    bases = [by_base[key] for key in sorted(by_base)]
    if len(bases) < 24:
        raise RuntimeError(f"not enough bases for fallback multi-scan plan: {len(bases)}")
    return bases[:30]


def build_plan_rows(bases: list[dict[str, str]]) -> list[dict[str, Any]]:
    variants = [
        {
            "observation_variant": "D1_y_scan_5line_z0p008",
            "observation_family": "D1",
            "paired_existing_x_variant": "R1_5line_z0p008",
            "line_offsets": Y_SCAN_5LINE_OFFSETS,
            "line_count": 5,
            "notes": "future 23.2b y_scan 5-line top-up; pair with existing x_scan R1_5line_z0p008",
        },
        {
            "observation_variant": "D2_y_scan_9line_z0p008",
            "observation_family": "D2",
            "paired_existing_x_variant": "R1_9line_z0p008",
            "line_offsets": Y_SCAN_9LINE_OFFSETS,
            "line_count": 9,
            "notes": "future 23.2b y_scan 9-line top-up; pair with existing x_scan R1_9line_z0p008",
        },
    ]
    rows: list[dict[str, Any]] = []
    for base in bases:
        for variant in variants:
            rows.append(
                {
                    "planned_row_id": f"{base['base_group_id']}_{variant['observation_variant']}",
                    "base_group_id": base["base_group_id"],
                    "base_sample_id": base["base_sample_id"],
                    "base_role": base.get("base_role", ""),
                    "source_split": base.get("source_split", ""),
                    "source_subset": base.get("source_subset", ""),
                    "true_shape_type": base.get("true_shape_type", ""),
                    "pred_shape_type": base.get("pred_shape_type", ""),
                    "burial_depth_level": base.get("burial_depth_level", ""),
                    "size_level": base.get("size_level", ""),
                    "aspect_bin": base.get("aspect_bin", ""),
                    "hardcase_target_id": base.get("hardcase_target_id", ""),
                    "center_xyz_error_mm": base.get("center_xyz_error_mm", ""),
                    "burial_depth_error_mm": base.get("burial_depth_error_mm", ""),
                    "risk_score": base.get("risk_score", ""),
                    "failure_tags": base.get("failure_tags", ""),
                    "paired_existing_x_variant": variant["paired_existing_x_variant"],
                    "observation_variant": variant["observation_variant"],
                    "observation_family": variant["observation_family"],
                    "generation_action": "generate_y_scan_only",
                    "reuse_existing_x_scan": True,
                    "sensor_z_m": f"{SENSOR_Z_M:.3f}",
                    "scan_direction": "y_scan",
                    "path_coordinate_axis": "y",
                    "path_coordinate_start_m": Y_SCAN_PATH_RANGE[0],
                    "path_coordinate_stop_m": Y_SCAN_PATH_RANGE[1],
                    "path_point_count": 201,
                    "line_coordinate_axis": "x",
                    "line_coordinate_offsets_m": dumps(variant["line_offsets"]),
                    "line_count": variant["line_count"],
                    "axis_order": "Bx,By,Bz",
                    "direction_names_after_assembly": dumps(["x_scan", "y_scan"]),
                    "expected_y_scan_delta_shape": f"(3,{variant['line_count']},201)",
                    "expected_assembled_delta_shape": "(3,2,9,201)",
                    "requires_comsol": True,
                    "requires_direction_aware_sensor_points": True,
                    "requires_new_manifest_now": False,
                    "source_richer_manifest": str(SOURCE_MANIFEST),
                    "notes": variant["notes"],
                }
            )
    return rows


def write_outputs(rows: list[dict[str, Any]], bases: list[dict[str, str]]) -> None:
    fields = [
        "planned_row_id",
        "base_group_id",
        "base_sample_id",
        "base_role",
        "source_split",
        "source_subset",
        "true_shape_type",
        "pred_shape_type",
        "burial_depth_level",
        "size_level",
        "aspect_bin",
        "hardcase_target_id",
        "center_xyz_error_mm",
        "burial_depth_error_mm",
        "risk_score",
        "failure_tags",
        "paired_existing_x_variant",
        "observation_variant",
        "observation_family",
        "generation_action",
        "reuse_existing_x_scan",
        "sensor_z_m",
        "scan_direction",
        "path_coordinate_axis",
        "path_coordinate_start_m",
        "path_coordinate_stop_m",
        "path_point_count",
        "line_coordinate_axis",
        "line_coordinate_offsets_m",
        "line_count",
        "axis_order",
        "direction_names_after_assembly",
        "expected_y_scan_delta_shape",
        "expected_assembled_delta_shape",
        "requires_comsol",
        "requires_direction_aware_sensor_points",
        "requires_new_manifest_now",
        "source_richer_manifest",
        "notes",
    ]
    write_csv(PACK_PLAN, rows, fields)
    row_count = len(rows)
    base_count = len({row["base_group_id"] for row in rows})
    variant_counts = Counter(row["observation_variant"] for row in rows)
    shape_counts = Counter(base.get("true_shape_type", "") for base in bases)
    burial_counts = Counter(base.get("burial_depth_level", "") for base in bases)
    decision_rows = [
        {"decision_item": "reuse_22_9_base_geometries", "status": "yes", "value": base_count, "rationale": "保持与 R1/R2 richer-observation 失败样本 paired comparison"},
        {"decision_item": "only_generate_y_scan_topup", "status": "yes", "value": True, "rationale": "既有 x_scan R1_5/R1_9 可复用；23.2b 只补正交方向"},
        {"decision_item": "target_base_count", "status": "pass", "value": base_count, "rationale": "target=30; fallback=24"},
        {"decision_item": "target_planned_rows", "status": "pass", "value": row_count, "rationale": "30 bases x 2 y_scan variants = 60"},
        {"decision_item": "fallback_rows", "status": "pass", "value": 48, "rationale": "24 bases x 2 y_scan variants"},
        {"decision_item": "paired_completeness_required", "status": "yes", "value": "D1 and D2 per selected base", "rationale": "缺任一 y_scan variant 的 base 不能进入完整 dual-direction comparison"},
        {"decision_item": "input_tensor_contract", "status": "selected", "value": "(N,3,2,9,201)", "rationale": "axes, directions, padded lines, path points"},
        {"decision_item": "include_multi_liftoff_now", "status": "no", "value": False, "rationale": "避免 D3 成本膨胀；先验证 direction 信息"},
        {"decision_item": "include_multi_magnetization_now", "status": "no", "value": False, "rationale": "R4 高成本且 evidence 不足"},
        {"decision_item": "unique_next_step", "status": "selected", "value": "23.2b_internal_multi_scan_direction_generation", "rationale": "执行 y_scan top-up COMSOL generation；训练暂缓到 23.3"},
    ]
    write_csv(DECISION_MATRIX, decision_rows, ["decision_item", "status", "value", "rationale"])
    lines = [
        "# 23.2 internal multi-scan-direction diagnostic pack decision",
        "",
        f"- target_base_count: {base_count}",
        f"- planned_rows: {row_count}",
        "- fallback: 24 bases / 48 rows",
        f"- variant_counts: {dict(variant_counts)}",
        f"- shape_counts: {dict(shape_counts)}",
        f"- burial_counts: {dict(burial_counts)}",
        "- reuse_22_9_base_geometries: true",
        "- generation_scope: y_scan only; do not regenerate existing x_scan unless 23.2b preflight proves pairing impossible.",
        "- y_scan path coordinate: y in [-0.010, 0.010] with 201 points.",
        "- y_scan line coordinate: x offsets for D1/D2.",
        "- assembled_tensor_contract: delta_b shape `(N,3,2,9,201)` with direction and scan-line masks.",
        "- 23.2b requirement: implement real direction-aware sensor points; metadata-only scan_direction is insufficient.",
        "- training: deferred to 23.3.",
        "- CURRENT_BASELINE update: false.",
        "",
        "唯一下一步：`23.2b_internal_multi_scan_direction_generation`。",
    ]
    write_text(SUMMARY_PATH, "\n".join(lines) + "\n")


def main() -> int:
    if not SOURCE_PLAN.exists():
        raise FileNotFoundError(f"missing source 22.9 plan: {SOURCE_PLAN}")
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(f"missing source 22.9 manifest: {SOURCE_MANIFEST}")
    source_rows = read_csv(SOURCE_PLAN)
    bases = selected_bases(source_rows)
    rows = build_plan_rows(bases)
    write_outputs(rows, bases)
    print(json.dumps({"base_count": len(bases), "planned_rows": len(rows), "plan": str(PACK_PLAN)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

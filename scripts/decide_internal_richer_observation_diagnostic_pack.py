#!/usr/bin/env python
"""22.8 richer-observation diagnostic pack decision and plan."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results/metrics"
SUMMARY_DIR = ROOT / "results/summaries"

FAILURE_CSV = METRICS_DIR / "internal_defect_inference_abstention_failure_cases.csv"
ABSTAINED_CSV = METRICS_DIR / "internal_defect_inference_abstention_abstained_subset.csv"
ACCEPTED_CSV = METRICS_DIR / "internal_defect_inference_abstention_accepted_subset.csv"
PLAN_CSV = METRICS_DIR / "internal_richer_observation_diagnostic_pack_plan.csv"
DECISION_CSV = METRICS_DIR / "internal_richer_observation_decision_matrix.csv"
SUMMARY_PATH = SUMMARY_DIR / "internal_richer_observation_diagnostic_pack_decision_summary.txt"

VARIANTS = [
    ("R0_3line_z0p008", [-0.001, 0.0, 0.001], 0.008, "current reference"),
    ("R1_5line_z0p008", [-0.002, -0.001, 0.0, 0.001, 0.002], 0.008, "more y-lines"),
    ("R1_9line_z0p008", [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004], 0.008, "dense y-lines"),
    ("R2_5line_z0p006", [-0.002, -0.001, 0.0, 0.001, 0.002], 0.006, "multi-liftoff low"),
    ("R2_5line_z0p010", [-0.002, -0.001, 0.0, 0.001, 0.002], 0.010, "multi-liftoff high"),
    ("R2_5line_z0p012", [-0.002, -0.001, 0.0, 0.001, 0.002], 0.012, "multi-liftoff high"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def role_for(row: dict[str, str]) -> str:
    if as_bool(row.get("geometry_branch_failure")):
        return "geometry_branch_failure"
    if as_bool(row.get("catastrophic_failure")):
        return "catastrophic_failure"
    if as_bool(row.get("center_outlier")):
        return "center_outlier"
    if as_bool(row.get("burial_outlier")):
        return "burial_outlier"
    if row.get("inference_status") == "accepted_prediction":
        return "accepted_representative"
    return "high_risk_representative"


def row_score(row: dict[str, str]) -> float:
    score = safe_float(row.get("center_xyz_error_mm")) + 2.0 * safe_float(row.get("burial_depth_error_mm"))
    if as_bool(row.get("geometry_branch_failure")):
        score += 100.0
    if as_bool(row.get("catastrophic_failure")):
        score += 50.0
    if row.get("inference_status") == "accepted_prediction":
        score = -score
    return score


def select_bases() -> list[dict[str, str]]:
    failures = read_csv(FAILURE_CSV)
    abstained = read_csv(ABSTAINED_CSV)
    accepted = read_csv(ACCEPTED_CSV)
    by_id: dict[str, dict[str, str]] = {}
    for row in sorted(failures + abstained, key=row_score, reverse=True):
        by_id.setdefault(row["sample_id"], row)
    # 保留少量 accepted representative，避免 diagnostic pack 只看失败样本。
    accepted_sorted = sorted(accepted, key=lambda r: (r.get("true_shape_type", ""), r.get("burial_depth_level", ""), safe_float(r.get("total_abs_normalized_error"))))
    for row in accepted_sorted[:8]:
        by_id.setdefault(row["sample_id"], row)
    selected = list(by_id.values())
    # 优先 tail / hard-case，补足到 30。
    selected = sorted(selected, key=lambda r: (role_rank(role_for(r)), -row_score(r), r["sample_id"]))
    return selected[:30]


def role_rank(role: str) -> int:
    order = {
        "geometry_branch_failure": 0,
        "catastrophic_failure": 1,
        "center_outlier": 2,
        "burial_outlier": 3,
        "high_risk_representative": 4,
        "accepted_representative": 5,
    }
    return order.get(role, 9)


def build_plan_rows(bases: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_idx, base in enumerate(bases, start=1):
        group = f"richer_obs_base_{base_idx:03d}"
        for variant_idx, (variant, y_lines, sensor_z, notes) in enumerate(VARIANTS, start=1):
            rows.append(
                {
                    "planned_row_id": f"{group}_{variant_idx:02d}_{variant}",
                    "base_group_id": group,
                    "base_sample_id": base["sample_id"],
                    "base_role": role_for(base),
                    "source_split": base.get("split", "test"),
                    "source_subset": base.get("subset", ""),
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
                    "observation_variant": variant,
                    "observation_family": variant.split("_")[0],
                    "sensor_z_m": f"{sensor_z:.3f}",
                    "scan_line_y_m": json.dumps(y_lines),
                    "y_line_count": len(y_lines),
                    "sensor_x_count": 201,
                    "axis_order": "Bx,By,Bz",
                    "scan_direction": "x_direction",
                    "magnetization_direction": "nominal_existing_protocol",
                    "expected_delta_b_shape": f"(3,{len(y_lines)},201)",
                    "requires_comsol": True,
                    "notes": notes,
                }
            )
    return rows


def main() -> int:
    bases = select_bases()
    if len(bases) < 24:
        raise RuntimeError(f"not enough bases for fallback diagnostic pack: {len(bases)}")
    rows = build_plan_rows(bases)
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
        "observation_variant",
        "observation_family",
        "sensor_z_m",
        "scan_line_y_m",
        "y_line_count",
        "sensor_x_count",
        "axis_order",
        "scan_direction",
        "magnetization_direction",
        "expected_delta_b_shape",
        "requires_comsol",
        "notes",
    ]
    write_csv(PLAN_CSV, rows, fields)
    decision_rows = [
        {"decision_item": "first_round_candidate", "result": "R0_vs_R1_vs_R2", "evidence": "R1/R2 use existing scan_line_y_m and sensor_z_m generator hooks"},
        {"decision_item": "target_base_count", "result": 30, "evidence": "30 bases selected from failure, abstained, and accepted representative samples"},
        {"decision_item": "target_rows", "result": len(rows), "evidence": "30 bases x 6 variants"},
        {"decision_item": "fallback_rows", "result": 144, "evidence": "24 bases x 6 variants"},
        {"decision_item": "include_R3_now", "result": False, "evidence": "multi-scan-direction requires new direction-aware schema; second priority"},
        {"decision_item": "include_R4_now", "result": False, "evidence": "multi-magnetization is high cost and not supported by current failure evidence"},
        {"decision_item": "unique_next_step", "result": "22.9_richer_observation_COMSOL_diagnostic_pack_generation", "evidence": "plan-only stage complete; next stage may generate diagnostic pack"},
    ]
    write_csv(DECISION_CSV, decision_rows, ["decision_item", "result", "evidence"])
    role_counts: dict[str, int] = {}
    for base in bases:
        role_counts[role_for(base)] = role_counts.get(role_for(base), 0) + 1
    lines = [
        "22.8 internal richer-observation diagnostic pack decision",
        "",
        f"- target bases: {len(bases)}; target rows: {len(rows)}; fallback: 24 bases / 144 rows.",
        f"- base role counts: {role_counts}",
        "- variants per base: R0_3line_z0p008, R1_5line_z0p008, R1_9line_z0p008, R2_5line_z0p006, R2_5line_z0p010, R2_5line_z0p012.",
        "- first-round priority: R1_more_y_lines and R2_multi_liftoff.",
        "- R3 multi_scan_direction: second-priority feasibility probe, not included in first-round rows.",
        "- R4 multi_magnetization_direction: deferred.",
        "",
        "唯一下一步：执行 22.9 richer-observation COMSOL diagnostic pack generation；仍不进入训练或真实样品推理。",
    ]
    write_text(SUMMARY_PATH, "\n".join(lines) + "\n")
    print(json.dumps({"bases": len(bases), "rows": len(rows), "plan": str(PLAN_CSV)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

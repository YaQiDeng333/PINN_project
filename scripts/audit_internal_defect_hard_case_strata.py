#!/usr/bin/env python
"""Audit 22.0/22.1 internal defect tail failures for hard-case top-up planning.

This is a plan-only script. It reads tracked metrics and writes summary/CSV
planning artifacts only. It does not run COMSOL, train, or touch data/NPZ.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import ROOT, write_csv


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json"
B2_FAILURE_CASES = ROOT / "results/metrics/internal_defect_b2_failure_cases.csv"
B2_GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_b2_failure_group_summary.csv"
B2_TAIL_SUMMARY = ROOT / "results/metrics/internal_defect_b2_tail_error_summary.csv"
T3_TAIL_METRICS = ROOT / "results/metrics/internal_defect_shape_conditioned_tail_metrics.csv"
T3_VS_B2 = ROOT / "results/metrics/internal_defect_shape_conditioned_vs_b2.csv"
SCHEMA = ROOT / "INTERNAL_DEFECT_SCHEMA.md"

SUMMARY_OUT = ROOT / "results/summaries/internal_defect_hard_case_strata_audit_summary.txt"
AUDIT_OUT = ROOT / "results/metrics/internal_defect_hard_case_strata_audit.csv"
TARGETS_OUT = ROOT / "results/metrics/internal_defect_hard_case_topup_targets.csv"

GROUP_FIELDS = [
    ("shape", ["true_shape_type"]),
    ("burial_depth", ["burial_depth_level"]),
    ("size", ["size_level"]),
    ("aspect", ["aspect_bin"]),
    ("center_region", ["center_region"]),
    ("shape_burial", ["true_shape_type", "burial_depth_level"]),
    ("shape_aspect", ["true_shape_type", "aspect_bin"]),
    ("shape_size", ["true_shape_type", "size_level"]),
    ("shape_burial_size_aspect", ["true_shape_type", "burial_depth_level", "size_level", "aspect_bin"]),
]

AUDIT_FIELDS = [
    "dataset_id",
    "evidence_source",
    "group_field",
    "group_value",
    "sample_count",
    "catastrophic_failure_count",
    "geometry_branch_failure_count",
    "center_outlier_count",
    "burial_outlier_count",
    "dimension_outlier_count",
    "shape_misclassified_count",
    "mean_total_error",
    "mean_burial_depth_error_mm",
    "mean_center_xyz_error_mm",
    "max_burial_depth_error_mm",
    "max_center_xyz_error_mm",
    "priority_score",
    "topup_need",
    "notes",
]

TARGET_FIELDS = [
    "target_id",
    "target_reason",
    "source_sample_ids",
    "shape_focus",
    "burial_focus",
    "size_focus",
    "aspect_focus",
    "center_region_focus",
    "catastrophic_failure_count",
    "geometry_branch_failure_count",
    "worst_center_error_mm",
    "worst_burial_depth_error_mm",
    "neighbor_strategies",
    "recommended_rows",
    "minimum_rows",
    "topup_priority",
    "evidence",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit internal hard-case strata for top-up planning.")
    parser.add_argument("--failure-cases", type=Path, default=B2_FAILURE_CASES)
    parser.add_argument("--summary", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_OUT)
    parser.add_argument("--targets-csv", type=Path, default=TARGETS_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def has_tag(row: dict[str, Any], tag: str) -> bool:
    return tag in str(row.get("failure_tags", "")).split("|")


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def group_key(row: dict[str, Any], fields: list[str]) -> str:
    return "|".join(str(row.get(field, "")) for field in fields)


def summarize_bucket(group_name: str, group_value: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    total = [safe_float(row.get("total_abs_normalized_error")) for row in rows]
    burial = [safe_float(row.get("burial_depth_error_mm")) for row in rows]
    center = [safe_float(row.get("center_xyz_error_mm")) for row in rows]
    catastrophic = sum(bool_value(row.get("is_catastrophic_failure")) for row in rows)
    branch = sum(bool_value(row.get("is_geometry_branch_failure")) for row in rows)
    center_outliers = sum(has_tag(row, "center_outlier") for row in rows)
    burial_outliers = sum(has_tag(row, "burial_outlier") for row in rows)
    dimension_outliers = sum(has_tag(row, "dimension_outlier") for row in rows)
    shape_errors = sum(not bool_value(row.get("shape_correct")) for row in rows)
    priority = (
        catastrophic * 8.0
        + branch * 12.0
        + center_outliers * 2.0
        + burial_outliers * 2.0
        + dimension_outliers
        + float(np.mean(center) if center else 0.0) / 2.0
        + float(np.mean(burial) if burial else 0.0) * 2.0
        + shape_errors * 4.0
    )
    topup_need = (
        branch > 0
        or catastrophic >= 2
        or center_outliers >= 3
        or burial_outliers >= 2
        or (center and float(np.mean(center)) > 3.0)
        or (burial and float(np.mean(burial)) > 0.50)
    )
    notes: list[str] = []
    if branch:
        notes.append("包含 geometry branch failure")
    if catastrophic:
        notes.append("包含 center+burial full-shift failure")
    if group_value in {"internal_cuboid", "internal_ellipsoid", "compact", "medium", "large", "shallow", "deep_plus"}:
        notes.append("属于 22.2 指定关注 strata")
    return {
        "dataset_id": DATASET_ID,
        "evidence_source": "22.0_b2_failure_cases",
        "group_field": group_name,
        "group_value": group_value,
        "sample_count": len(rows),
        "catastrophic_failure_count": catastrophic,
        "geometry_branch_failure_count": branch,
        "center_outlier_count": center_outliers,
        "burial_outlier_count": burial_outliers,
        "dimension_outlier_count": dimension_outliers,
        "shape_misclassified_count": shape_errors,
        "mean_total_error": float(np.mean(total)) if total else 0.0,
        "mean_burial_depth_error_mm": float(np.mean(burial)) if burial else 0.0,
        "mean_center_xyz_error_mm": float(np.mean(center)) if center else 0.0,
        "max_burial_depth_error_mm": max(burial) if burial else 0.0,
        "max_center_xyz_error_mm": max(center) if center else 0.0,
        "priority_score": priority,
        "topup_need": topup_need,
        "notes": "; ".join(notes) or "诊断记录",
    }


def audit_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_name, fields in GROUP_FIELDS:
        buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            buckets[group_key(row, fields)].append(row)
        for value, bucket in sorted(buckets.items()):
            out.append(summarize_bucket(group_name, value, bucket))
    out.sort(key=lambda row: safe_float(row["priority_score"]), reverse=True)
    return out


def sample_ids(rows: list[dict[str, str]], limit: int = 8) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            bool_value(row.get("is_geometry_branch_failure")),
            bool_value(row.get("is_catastrophic_failure")),
            safe_float(row.get("center_xyz_error_mm")),
            safe_float(row.get("burial_depth_error_mm")),
        ),
        reverse=True,
    )
    return "|".join(row.get("sample_id", "") for row in ordered[:limit])


def target_row(
    target_id: str,
    reason: str,
    rows: list[dict[str, str]],
    shape_focus: str,
    burial_focus: str,
    size_focus: str,
    aspect_focus: str,
    center_focus: str,
    strategies: list[str],
    recommended_rows: int,
    minimum_rows: int,
    priority: str,
    notes: str,
) -> dict[str, Any]:
    worst_center = max([safe_float(row.get("center_xyz_error_mm")) for row in rows] or [0.0])
    worst_burial = max([safe_float(row.get("burial_depth_error_mm")) for row in rows] or [0.0])
    catastrophic = sum(bool_value(row.get("is_catastrophic_failure")) for row in rows)
    branch = sum(bool_value(row.get("is_geometry_branch_failure")) for row in rows)
    evidence_bits = []
    if rows:
        pair_counts = Counter(f"{row.get('true_shape_type')}->{row.get('pred_shape_type')}" for row in rows if row.get("true_shape_type") != row.get("pred_shape_type"))
        if pair_counts:
            evidence_bits.append("shape_confusion=" + ";".join(f"{k}:{v}" for k, v in sorted(pair_counts.items())))
        evidence_bits.append(f"max_center={worst_center:.3f}mm")
        evidence_bits.append(f"max_burial={worst_burial:.3f}mm")
    return {
        "target_id": target_id,
        "target_reason": reason,
        "source_sample_ids": sample_ids(rows),
        "shape_focus": shape_focus,
        "burial_focus": burial_focus,
        "size_focus": size_focus,
        "aspect_focus": aspect_focus,
        "center_region_focus": center_focus,
        "catastrophic_failure_count": catastrophic,
        "geometry_branch_failure_count": branch,
        "worst_center_error_mm": worst_center,
        "worst_burial_depth_error_mm": worst_burial,
        "neighbor_strategies": "|".join(strategies),
        "recommended_rows": recommended_rows,
        "minimum_rows": minimum_rows,
        "topup_priority": priority,
        "evidence": "; ".join(evidence_bits) or "group-level risk",
        "notes": notes,
    }


def build_targets(rows: list[dict[str, str]], audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    branch_rows = [row for row in rows if bool_value(row.get("is_geometry_branch_failure"))]
    catastrophic_rows = [row for row in rows if bool_value(row.get("is_catastrophic_failure"))]
    worst_center = sorted(rows, key=lambda row: safe_float(row.get("center_xyz_error_mm")), reverse=True)[:5]
    worst_burial = sorted(rows, key=lambda row: safe_float(row.get("burial_depth_error_mm")), reverse=True)[:5]
    cuboid_ellipsoid = [
        row
        for row in rows
        if {row.get("true_shape_type"), row.get("pred_shape_type")} <= {"internal_cuboid", "internal_ellipsoid"}
        and row.get("true_shape_type") != row.get("pred_shape_type")
    ]
    compact_rows = [row for row in rows if row.get("aspect_bin") == "compact" and bool_value(row.get("is_catastrophic_failure"))]
    deep_plus_rows = [row for row in rows if row.get("burial_depth_level") == "deep_plus" and (bool_value(row.get("is_catastrophic_failure")) or bool_value(row.get("is_geometry_branch_failure")))]
    shallow_rows = [row for row in rows if row.get("burial_depth_level") == "shallow" and (bool_value(row.get("is_catastrophic_failure")) or has_tag(row, "burial_outlier"))]
    medium_large_rows = [row for row in rows if row.get("size_level") in {"medium", "large"} and (bool_value(row.get("is_catastrophic_failure")) or has_tag(row, "center_outlier"))]
    x_pos_y_pos = [row for row in rows if row.get("center_region") == "x_pos_y_pos" and has_tag(row, "center_outlier")]
    x_neg_y_pos = [row for row in rows if row.get("center_region") == "x_neg_y_pos" and (bool_value(row.get("is_catastrophic_failure")) or has_tag(row, "burial_outlier"))]

    targets = [
        target_row(
            "target_01_geometry_branch_cuboid_ellipsoid",
            "cuboid/ellipsoid branch confusion",
            branch_rows or cuboid_ellipsoid,
            "internal_cuboid|internal_ellipsoid",
            "deep_plus|shallow",
            "medium|large",
            "compact|elongated_y",
            "x_pos_y_neg|x_pos_y_pos",
            ["cuboid_ellipsoid_pair", "same_burial_varied_shape", "same_size_aspect_varied_center"],
            24,
            12,
            "P0",
            "唯一 geometry_branch_failure 来自 cuboid->ellipsoid，必须成对补 cuboid/ellipsoid 邻域。",
        ),
        target_row(
            "target_02_full_shift_catastrophic",
            "center and burial full-shift failures",
            catastrophic_rows,
            "internal_cuboid|internal_ellipsoid|internal_sphere",
            "shallow|deep_plus",
            "medium|large",
            "compact|elongated_y",
            "x_pos_y_pos|x_neg_y_pos|x_pos_y_neg",
            ["same_shape_varied_burial", "center_burial_tradeoff"],
            20,
            12,
            "P0",
            "5/40 catastrophic failures 说明 hard-case 数据覆盖不足，真实 internal smoke 继续暂缓。",
        ),
        target_row(
            "target_03_worst_center_regions",
            "worst center_xyz tail",
            worst_center,
            "internal_cuboid|internal_ellipsoid",
            "shallow|deep_plus",
            "medium|large",
            "compact|elongated_y",
            "x_pos_y_pos|x_neg_y_pos|x_pos_y_neg",
            ["same_size_aspect_varied_center", "center_burial_tradeoff"],
            18,
            10,
            "P0",
            "center max 仍接近厘米级，需围绕失败区域做 lateral neighbor samples。",
        ),
        target_row(
            "target_04_worst_burial_depth",
            "worst burial_depth tail",
            worst_burial,
            "internal_cuboid|internal_ellipsoid|internal_sphere",
            "shallow|deep_plus",
            "medium|large",
            "compact",
            "x_pos_y_neg|x_neg_y_pos",
            ["same_shape_varied_burial", "same_burial_varied_shape"],
            16,
            10,
            "P0",
            "burial p95/max 在 B2/T3 均未解决，补 matched burial ladder。",
        ),
        target_row(
            "target_05_compact_medium_large",
            "compact medium/large hard cases",
            compact_rows or medium_large_rows,
            "internal_cuboid|internal_ellipsoid",
            "shallow|deep_plus",
            "medium|large",
            "compact",
            "all_failure_regions",
            ["cuboid_ellipsoid_pair", "same_size_aspect_varied_center"],
            14,
            8,
            "P1",
            "compact 是 catastrophic 和 branch failure 的共同出现点，需要加密。",
        ),
        target_row(
            "target_06_shallow_edge",
            "shallow edge ambiguity",
            shallow_rows,
            "internal_cuboid|internal_ellipsoid|internal_sphere",
            "shallow",
            "medium|large",
            "compact|elongated_y",
            "x_neg_y_pos|x_pos_y_pos",
            ["same_burial_varied_shape", "center_burial_tradeoff"],
            10,
            6,
            "P1",
            "shallow 仍有 center/burial full-shift，需同 burial 变 shape。",
        ),
        target_row(
            "target_07_deep_plus_edge",
            "deep_plus edge ambiguity",
            deep_plus_rows,
            "internal_cuboid|internal_ellipsoid|internal_sphere",
            "deep_plus",
            "medium|large",
            "compact|elongated_y",
            "x_pos_y_pos|x_pos_y_neg",
            ["same_burial_varied_shape", "center_burial_tradeoff"],
            10,
            6,
            "P1",
            "deep_plus 同时包含 branch failure 与高 center tail。",
        ),
        target_row(
            "target_08_x_pos_y_pos_region",
            "x_pos_y_pos high center tail",
            x_pos_y_pos,
            "internal_cuboid|internal_ellipsoid",
            "shallow|deep_plus",
            "large|medium",
            "elongated_y|compact",
            "x_pos_y_pos",
            ["same_size_aspect_varied_center"],
            4,
            4,
            "P2",
            "x_pos_y_pos 的 mean center error 最高，作为 lateral-region sentinel。",
        ),
        target_row(
            "target_09_x_neg_y_pos_region",
            "x_neg_y_pos full-shift support",
            x_neg_y_pos,
            "internal_cuboid|internal_ellipsoid|internal_sphere",
            "shallow|deep",
            "medium|large",
            "compact",
            "x_neg_y_pos",
            ["same_size_aspect_varied_center"],
            4,
            4,
            "P2",
            "x_neg_y_pos 有 full-shift 和 burial outlier，保留为补充区域。",
        ),
    ]
    if sum(int(row["recommended_rows"]) for row in targets) != 120:
        raise RuntimeError("target recommended_rows must sum to 120")
    return targets


def summarize_tail_metric(path: Path, metric: str, split: str = "test") -> str:
    if not path.exists():
        return "missing"
    rows = [row for row in read_csv(path) if row.get("split") == split and row.get("metric") == metric]
    if not rows:
        return "missing"
    row = rows[0]
    return f"mean={safe_float(row.get('mean')):.3f}, median={safe_float(row.get('median')):.3f}, p95={safe_float(row.get('p95')):.3f}, max={safe_float(row.get('max')):.3f}"


def summarize_shape_conditioned_tail(path: Path, metric_prefix: str) -> str:
    """Read 22.1 tail metrics, whose schema is candidate-row based."""
    if not path.exists():
        return "missing"
    rows = [
        row
        for row in read_csv(path)
        if row.get("split") == "test"
        and row.get("candidate") == "T3_shape_specific_heads"
        and str(row.get("selected_candidate", "")).strip().lower() == "true"
    ]
    if not rows:
        return "missing"
    row = rows[0]
    return (
        f"mean={safe_float(row.get(metric_prefix + '_mean_mm')):.3f}, "
        f"median={safe_float(row.get(metric_prefix + '_median_mm')):.3f}, "
        f"p95={safe_float(row.get(metric_prefix + '_p95_mm')):.3f}, "
        f"max={safe_float(row.get(metric_prefix + '_max_mm')):.3f}"
    )


def main() -> int:
    args = parse_args()
    for path in [MANIFEST, args.failure_cases, B2_TAIL_SUMMARY, T3_TAIL_METRICS, SCHEMA]:
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')}")
    rows = [row for row in read_csv(args.failure_cases) if row.get("split") == "test"]
    if len(rows) != 40:
        raise RuntimeError(f"expected 40 test failure rows, got {len(rows)}")

    audit = audit_rows(rows)
    targets = build_targets(rows, audit)
    write_csv(args.audit_csv, audit, AUDIT_FIELDS)
    write_csv(args.targets_csv, targets, TARGET_FIELDS)

    catastrophic = [row for row in rows if bool_value(row.get("is_catastrophic_failure"))]
    branch = [row for row in rows if bool_value(row.get("is_geometry_branch_failure"))]
    worst_center = max(rows, key=lambda row: safe_float(row.get("center_xyz_error_mm")))
    worst_burial = max(rows, key=lambda row: safe_float(row.get("burial_depth_error_mm")))
    shape_confusions = Counter(
        f"{row.get('true_shape_type')}->{row.get('pred_shape_type')}"
        for row in rows
        if row.get("true_shape_type") != row.get("pred_shape_type")
    )
    target_focus = Counter()
    for target in targets:
        for field in ["shape_focus", "burial_focus", "size_focus", "aspect_focus", "center_region_focus"]:
            target_focus.update(str(target[field]).split("|"))

    summary = [
        "22.2 内部/埋藏缺陷 hard-case strata audit",
        f"dataset_id: {DATASET_ID}",
        "stage_scope: plan_only; no_COMSOL=true; no_training=true; no_data_or_npz_mutation=true; current_baseline_update=false",
        f"manifest_path: {MANIFEST}",
        f"test_failure_rows: {len(rows)}",
        f"catastrophic_failure_count: {len(catastrophic)} / {len(rows)}",
        f"geometry_branch_failure_count: {len(branch)} / {len(rows)}",
        f"B2_total_error_tail: {summarize_tail_metric(B2_TAIL_SUMMARY, 'total_abs_normalized_error')}",
        f"B2_burial_depth_tail_mm: {summarize_tail_metric(B2_TAIL_SUMMARY, 'burial_depth_error_mm')}",
        f"B2_center_xyz_tail_mm: {summarize_tail_metric(B2_TAIL_SUMMARY, 'center_xyz_error_mm')}",
        f"T3_burial_depth_tail_mm: {summarize_shape_conditioned_tail(T3_TAIL_METRICS, 'burial_depth_error')}",
        f"T3_center_xyz_tail_mm: {summarize_shape_conditioned_tail(T3_TAIL_METRICS, 'center_xyz_error')}",
        f"worst_center_sample: {worst_center['sample_id']} true={worst_center['true_shape_type']} pred={worst_center['pred_shape_type']} center={safe_float(worst_center['center_xyz_error_mm']):.3f}mm burial={safe_float(worst_center['burial_depth_error_mm']):.3f}mm tags={worst_center.get('failure_tags')}",
        f"worst_burial_sample: {worst_burial['sample_id']} true={worst_burial['true_shape_type']} pred={worst_burial['pred_shape_type']} burial={safe_float(worst_burial['burial_depth_error_mm']):.3f}mm center={safe_float(worst_burial['center_xyz_error_mm']):.3f}mm tags={worst_burial.get('failure_tags')}",
        "shape_confusion_pairs: " + ("; ".join(f"{key}:{value}" for key, value in sorted(shape_confusions.items())) or "none"),
        "hard_case_strata: internal_cuboid/internal_ellipsoid confusion; compact; medium/large; shallow/deep_plus; x_pos_y_pos/x_neg_y_pos/x_pos_y_neg center regions",
        "model_status: B2 和 22.1 shape-conditioned/T3 都不是 stable inference model；继续调模型收益有限，下一步应补 hard-case COMSOL 数据。",
        f"topup_targets_csv: {args.targets_csv}",
        f"strata_audit_csv: {args.audit_csv}",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

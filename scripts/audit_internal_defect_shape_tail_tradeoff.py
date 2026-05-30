#!/usr/bin/env python
"""22.4 shape/tail trade-off audit for the internal hard-case model."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import ROOT, write_csv


V3_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
CURRENT_BASELINE = ROOT / "CURRENT_BASELINE.md"
SELECTED_PREDICTIONS = ROOT / "results/metrics/internal_defect_hardcase_selected_predictions.csv"
B2_PREDICTIONS = ROOT / "results/metrics/internal_defect_hardcase_b2_reference_predictions.csv"
VS_B2 = ROOT / "results/metrics/internal_defect_hardcase_vs_b2_reference.csv"
HARDCASE_FAILURES = ROOT / "results/metrics/internal_defect_hardcase_failure_cases.csv"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_hardcase_seed_summary.csv"
SHAPE_CONDITIONED_TAIL = ROOT / "results/metrics/internal_defect_shape_conditioned_tail_metrics.csv"
B2_FAILURE_22_0 = ROOT / "results/metrics/internal_defect_b2_failure_cases.csv"

PREFLIGHT = ROOT / "results/summaries/internal_defect_shape_preserving_tail_strategy_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/internal_defect_shape_tail_tradeoff_audit_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_shape_tail_tradeoff_matrix.csv"
FAILURES = ROOT / "results/metrics/internal_defect_shape_tail_failure_cases.csv"

MATRIX_FIELDS = ["audit_item", "value", "evidence", "interpretation", "recommended_action"]
FAILURE_FIELDS = [
    "sample_id",
    "subset",
    "true_shape_type",
    "pred_shape_type",
    "shape_correct",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
    "center_region",
    "center_xyz_error_mm",
    "burial_depth_error_mm",
    "total_abs_normalized_error",
    "failure_tags",
    "old_b2_center_xyz_error_mm",
    "old_b2_burial_depth_error_mm",
    "center_delta_vs_old_b2_mm",
    "burial_delta_vs_old_b2_mm",
    "shape_tail_diagnosis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 22.3 shape/tail trade-off.")
    parser.add_argument("--predictions", type=Path, default=SELECTED_PREDICTIONS)
    parser.add_argument("--b2-predictions", type=Path, default=B2_PREDICTIONS)
    parser.add_argument("--vs-b2", type=Path, default=VS_B2)
    parser.add_argument("--failures", type=Path, default=HARDCASE_FAILURES)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--failure-cases", type=Path, default=FAILURES)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def git_status(paths: list[str]) -> str:
    cmd = ["git", "status", "--short", "--", *paths]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip()


def selected_seed(path: Path) -> dict[str, str]:
    for row in read_csv(path):
        if bool_value(row.get("selected_model")):
            return row
    return {}


def write_preflight(args: argparse.Namespace) -> None:
    critical = {
        "v3_hardcase_manifest": V3_MANIFEST,
        "COMSOL_DATA_REGISTRY.md": REGISTRY,
        "CURRENT_BASELINE.md": CURRENT_BASELINE,
        "22.3_selected_predictions": args.predictions,
        "22.3_vs_old_b2": args.vs_b2,
        "22.3_failure_cases": args.failures,
        "22.3_seed_summary": args.seed_summary,
    }
    optional = {
        "22.0_b2_failure_cases": B2_FAILURE_22_0,
        "22.1_shape_conditioned_tail": SHAPE_CONDITIONED_TAIL,
        "old_b2_predictions_on_v3": args.b2_predictions,
    }
    missing_critical = [name for name, path in critical.items() if not path.exists()]
    missing_optional = [name for name, path in optional.items() if not path.exists()]
    forbidden = git_status(["data", "checkpoints", "results\\previews", "notes", "CURRENT_BASELINE.md", "scripts\\visualize_current_baseline.py"])
    lines = [
        "22.4 shape-preserving tail strategy preflight",
        "scope: analysis artifact generator；只读取既有 metrics/manifest 并写 summary/metrics，不训练，不运行 COMSOL，不生成或修改 data/NPZ，不更新 CURRENT_BASELINE.md。",
        f"repo_root: {ROOT}",
        f"critical_files_present: {len(missing_critical) == 0}",
        f"missing_critical: {missing_critical}",
        f"missing_optional: {missing_optional}",
        f"forbidden_artifact_status_empty: {forbidden == ''}",
        f"forbidden_artifact_status: {forbidden if forbidden else 'clean'}",
        "baseline_status: internal branch 不是 CURRENT_BASELINE；CURRENT_BASELINE.md 只读检查。",
    ]
    args.preflight.parent.mkdir(parents=True, exist_ok=True)
    args.preflight.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing_critical:
        raise RuntimeError(f"preflight blocker: missing {missing_critical}")


def metric_lookup(path: Path) -> dict[str, dict[str, str]]:
    return {row["metric"]: row for row in read_csv(path)}


def count_by(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(Counter(row.get(field, "") for row in rows))


def group_tail(rows: list[dict[str, str]], field: str) -> list[str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(field, "")].append(row)
    parts: list[str] = []
    for key, vals in sorted(grouped.items()):
        center = [safe_float(row.get("center_xyz_error_mm")) for row in vals]
        burial = [safe_float(row.get("burial_depth_error_mm")) for row in vals]
        shape_err = sum(not bool_value(row.get("shape_correct")) for row in vals)
        cat = sum("full_shift_failure" in row.get("failure_tags", "") for row in vals)
        parts.append(f"{key}:n={len(vals)},shape_err={shape_err},cat={cat},center_p95={pct(center,95):.3f},burial_p95={pct(burial,95):.3f}")
    return parts


def build_failure_rows(h2_rows: list[dict[str, str]], old_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    test_rows = [row for row in h2_rows if row.get("split") == "test"]
    selected = [
        row
        for row in test_rows
        if not bool_value(row.get("shape_correct"))
        or "full_shift_failure" in row.get("failure_tags", "")
        or "geometry_branch_failure" in row.get("failure_tags", "")
        or safe_float(row.get("center_xyz_error_mm")) > 8.0
        or safe_float(row.get("burial_depth_error_mm")) > 1.5
    ]
    selected = sorted(
        selected,
        key=lambda row: (
            "geometry_branch_failure" in row.get("failure_tags", ""),
            "full_shift_failure" in row.get("failure_tags", ""),
            not bool_value(row.get("shape_correct")),
            safe_float(row.get("center_xyz_error_mm")) + safe_float(row.get("burial_depth_error_mm")),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in selected:
        old = old_by_id.get(row["sample_id"], {})
        old_center = safe_float(old.get("center_xyz_error_mm"), np.nan)
        old_burial = safe_float(old.get("burial_depth_error_mm"), np.nan)
        center = safe_float(row.get("center_xyz_error_mm"))
        burial = safe_float(row.get("burial_depth_error_mm"))
        shape_correct = bool_value(row.get("shape_correct"))
        tags = row.get("failure_tags", "")
        if "geometry_branch_failure" in tags:
            diagnosis = "shape 错分与 center/burial 同时失稳，是真正的 geometry branch failure。"
        elif not shape_correct and center > 3.0:
            diagnosis = "shape 错分伴随 center outlier，说明 shape branch 退化会放大定位风险。"
        elif center < old_center and shape_correct:
            diagnosis = "center tail 被 H2 改善，但该样本不能证明 shape 分支稳定。"
        elif burial > old_burial:
            diagnosis = "burial tail 相比旧 B2 退化，需要冻结 shape 后单独约束 burial head。"
        else:
            diagnosis = "tail case 保留为策略设计证据。"
        out.append(
            {
                "sample_id": row.get("sample_id", ""),
                "subset": row.get("subset", ""),
                "true_shape_type": row.get("true_shape_type", ""),
                "pred_shape_type": row.get("pred_shape_type", ""),
                "shape_correct": row.get("shape_correct", ""),
                "burial_depth_level": row.get("burial_depth_level", ""),
                "size_level": row.get("size_level", ""),
                "aspect_bin": row.get("aspect_bin", ""),
                "center_region": row.get("center_region", ""),
                "center_xyz_error_mm": center,
                "burial_depth_error_mm": burial,
                "total_abs_normalized_error": safe_float(row.get("total_abs_normalized_error")),
                "failure_tags": tags,
                "old_b2_center_xyz_error_mm": old_center,
                "old_b2_burial_depth_error_mm": old_burial,
                "center_delta_vs_old_b2_mm": center - old_center if not np.isnan(old_center) else "",
                "burial_delta_vs_old_b2_mm": burial - old_burial if not np.isnan(old_burial) else "",
                "shape_tail_diagnosis": diagnosis,
            }
        )
    return out


def main() -> int:
    args = parse_args()
    write_preflight(args)

    h2_rows = read_csv(args.predictions)
    old_rows = read_csv(args.b2_predictions)
    old_by_id = {row["sample_id"]: row for row in old_rows}
    test = [row for row in h2_rows if row.get("split") == "test"]
    selected = selected_seed(args.seed_summary)
    vs = metric_lookup(args.vs_b2)

    shape_errors = [row for row in test if not bool_value(row.get("shape_correct"))]
    catastrophic = [row for row in test if "full_shift_failure" in row.get("failure_tags", "")]
    geometry = [row for row in test if "geometry_branch_failure" in row.get("failure_tags", "")]
    hardcase = [row for row in test if row.get("subset") == "hardcase_topup"]
    source = [row for row in test if row.get("subset") == "source_v2"]

    paired = [(row, old_by_id[row["sample_id"]]) for row in test if row.get("sample_id") in old_by_id]
    center_improved = [row for row, old in paired if safe_float(row["center_xyz_error_mm"]) < safe_float(old["center_xyz_error_mm"])]
    burial_worse = [row for row, old in paired if safe_float(row["burial_depth_error_mm"]) > safe_float(old["burial_depth_error_mm"])]
    old_shape_acc = safe_float(vs.get("shape_accuracy", {}).get("old_B2_reference"))
    h2_shape_acc = safe_float(vs.get("shape_accuracy", {}).get("hardcase_augmented"))
    h2_shape_f1 = safe_float(selected.get("test_shape_macro_f1"))
    old_b2_shape_f1 = 0.841143

    matrix = [
        {
            "audit_item": "selected_h2_summary",
            "value": "H2_B2_hardcase_tail_weighted seed=42",
            "evidence": (
                f"total={safe_float(selected.get('test_total_normalized_mae')):.6f}; "
                f"shape_acc/f1={h2_shape_acc:.6f}/{h2_shape_f1:.6f}; "
                f"catastrophic={selected.get('test_catastrophic_failure_count')}/60; "
                f"geometry={selected.get('test_geometry_branch_failure_count')}/60"
            ),
            "interpretation": "H2 是 22.3 validation-only 选择结果，但不是 stable inference model。",
            "recommended_action": "不要把 H2 写成 baseline；进入 shape-preserving tail 策略。",
        },
        {
            "audit_item": "shape_branch_regression",
            "value": f"old_B2_shape_acc={old_shape_acc:.6f}; H2_shape_acc={h2_shape_acc:.6f}; old_B2_shape_F1={old_b2_shape_f1:.6f}; H2_shape_F1={h2_shape_f1:.6f}",
            "evidence": f"test shape error count={len(shape_errors)}/{len(test)}; confusion={count_by(shape_errors, 'true_shape_type')}",
            "interpretation": "tail weighting 改善回归尾部时牺牲了 shape branch，shape F1 下降是主 blocker。",
            "recommended_action": "下一步冻结或强化 shape classifier，再训练 center/burial tail heads。",
        },
        {
            "audit_item": "center_tail_improvement",
            "value": f"paired_center_improved={len(center_improved)}/{len(paired)}",
            "evidence": f"center p95/max old->H2: {safe_float(vs['center_xyz_error_p95_mm']['old_B2_reference']):.3f}/{safe_float(vs['center_xyz_error_max_mm']['old_B2_reference']):.3f} -> {safe_float(vs['center_xyz_error_p95_mm']['hardcase_augmented']):.3f}/{safe_float(vs['center_xyz_error_max_mm']['hardcase_augmented']):.3f} mm",
            "interpretation": "H2 的主要收益在 center tail，但收益不足以抵消 shape branch 风险。",
            "recommended_action": "保留 hard-case 数据价值，但改变训练策略，不继续直接加权 H2。",
        },
        {
            "audit_item": "burial_tail_regression",
            "value": f"paired_burial_worse={len(burial_worse)}/{len(paired)}",
            "evidence": f"burial p95/max old->H2: {safe_float(vs['burial_depth_error_p95_mm']['old_B2_reference']):.3f}/{safe_float(vs['burial_depth_error_max_mm']['old_B2_reference']):.3f} -> {safe_float(vs['burial_depth_error_p95_mm']['hardcase_augmented']):.3f}/{safe_float(vs['burial_depth_error_max_mm']['hardcase_augmented']):.3f} mm",
            "interpretation": "burial p95 小幅改善但 max 明显退化，说明 tail weighting 对最坏 burial case 不稳定。",
            "recommended_action": "冻结 shape 后给 burial head 单独 tail loss，而不是共享端到端加权。",
        },
        {
            "audit_item": "failure_concentration",
            "value": f"catastrophic_by_shape={count_by(catastrophic, 'true_shape_type')}; catastrophic_by_burial={count_by(catastrophic, 'burial_depth_level')}",
            "evidence": f"by_size={count_by(catastrophic, 'size_level')}; by_aspect={count_by(catastrophic, 'aspect_bin')}; geometry_cases={[row['sample_id'] for row in geometry]}",
            "interpretation": "failure 有 deep_plus/compact/medium-large 倾向，但不是单一 strata；盲目第二轮 top-up 不是首选。",
            "recommended_action": "优先做 freeze-shape then tail-regression；若仍失败，再定向补样。",
        },
        {
            "audit_item": "hardcase_test_difficulty",
            "value": f"test_source_rows={len(source)}; test_hardcase_rows={len(hardcase)}",
            "evidence": f"hardcase_shape_errors={sum(not bool_value(r.get('shape_correct')) for r in hardcase)}; source_shape_errors={sum(not bool_value(r.get('shape_correct')) for r in source)}",
            "interpretation": "v3_hardcase test 明确更偏 hard-case 诊断；旧 v2 平均指标不能直接当作稳定推理证据。",
            "recommended_action": "后续 acceptance 必须同时报告 source_v2 与 hardcase_topup 子集。",
        },
        {
            "audit_item": "shape_confidence_proxy",
            "value": "no_logits_available; proxy=shape_error_or_total_error_p75_or_cuboid_ellipsoid_confusion",
            "evidence": "22.3 prediction artifact 只含 pred_shape_type，不含 shape logits/probability。",
            "interpretation": "当前只能做 error-based proxy，下一轮模型应导出 shape probability 用于 router/abstention。",
            "recommended_action": "把 shape-confidence 输出作为下一阶段接口要求，但不替代 freeze-shape training。",
        },
    ]
    failures = build_failure_rows(h2_rows, old_by_id)
    write_csv(args.matrix, matrix, MATRIX_FIELDS)
    write_csv(args.failure_cases, failures, FAILURE_FIELDS)

    summary_lines = [
        "22.4 internal defect shape/tail trade-off audit",
        "scope: analysis artifact generator；只读取既有 metrics，写出审计 summary/CSV；没有训练，没有 COMSOL，没有 data/NPZ mutation，没有 CURRENT_BASELINE 更新。",
        f"selected_model: {selected.get('model')} seed={selected.get('seed')} best_epoch={selected.get('best_epoch')}",
        f"h2_total_mae: {safe_float(selected.get('test_total_normalized_mae')):.6f}",
        f"h2_shape_acc_f1: {h2_shape_acc:.6f} / {h2_shape_f1:.6f}",
        f"old_b2_shape_acc_f1: {old_shape_acc:.6f} / {old_b2_shape_f1:.6f}",
        f"h2_catastrophic_geometry: {selected.get('test_catastrophic_failure_count')}/60 / {selected.get('test_geometry_branch_failure_count')}/60",
        f"h2_center_p95_max_mm: {safe_float(selected.get('test_center_p95_mm')):.3f} / {safe_float(selected.get('test_center_max_mm')):.3f}",
        f"h2_burial_p95_max_mm: {safe_float(selected.get('test_burial_p95_mm')):.3f} / {safe_float(selected.get('test_burial_max_mm')):.3f}",
        f"shape_error_by_true_shape: {count_by(shape_errors, 'true_shape_type')}",
        f"catastrophic_by_shape: {count_by(catastrophic, 'true_shape_type')}",
        f"catastrophic_by_burial: {count_by(catastrophic, 'burial_depth_level')}",
        f"catastrophic_by_size: {count_by(catastrophic, 'size_level')}",
        f"catastrophic_by_aspect: {count_by(catastrophic, 'aspect_bin')}",
        "group_tail_by_shape: " + " | ".join(group_tail(test, "true_shape_type")),
        "group_tail_by_burial: " + " | ".join(group_tail(test, "burial_depth_level")),
        "diagnosis: H2 的 hard-case weighting 降低了部分 center tail，但破坏 shape branch，并使 burial max 退化。",
        "strategy_implication: 下一步应先保护 shape classifier/shared encoder，再做 center/burial tail regression；不要继续直接 H2 tail weighting。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

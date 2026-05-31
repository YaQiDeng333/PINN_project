#!/usr/bin/env python
"""23.5 route decision for internal multi-magnetization evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from audit_internal_multi_magnetization_pairs import ROOT


PROBE_METRICS = ROOT / "results/metrics/internal_multi_magnetization_diagnostic_probe_metrics.csv"
SEPARABILITY = ROOT / "results/metrics/internal_multi_magnetization_feature_separability_metrics.csv"
PAIR_METRICS = ROOT / "results/metrics/internal_multi_magnetization_pair_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_evaluation_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_multi_magnetization_evaluation_decision_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide route after 23.5 multi-magnetization evaluation.")
    parser.add_argument("--probe-metrics", type=Path, default=PROBE_METRICS)
    parser.add_argument("--separability", type=Path, default=SEPARABILITY)
    parser.add_argument("--pair-metrics", type=Path, default=PAIR_METRICS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
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


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def selected_test_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("selected_model", "").lower() == "true" and row.get("split") == "test":
            out[row["observation_config"]] = row
    return out


def score(row: dict[str, str]) -> float:
    return (
        f(row, "total_normalized_mae")
        + 0.10 * f(row, "burial_depth_mae_mm")
        + 0.05 * f(row, "center_xyz_component_mae_mm")
        + 0.50 * f(row, "catastrophic_failure_rate")
        + 0.35 * f(row, "geometry_branch_failure_rate")
        + 0.10 * (1.0 - f(row, "shape_macro_f1"))
    )


def compare(a: dict[str, str], b: dict[str, str]) -> dict[str, float]:
    """Return b-a deltas; negative is better for error metrics."""
    return {
        "score_delta": score(b) - score(a),
        "shape_f1_delta": f(b, "shape_macro_f1") - f(a, "shape_macro_f1"),
        "shape_accuracy_delta": f(b, "shape_accuracy") - f(a, "shape_accuracy"),
        "center_p95_delta_mm": f(b, "center_xyz_error_p95_mm") - f(a, "center_xyz_error_p95_mm"),
        "center_max_delta_mm": f(b, "center_xyz_error_max_mm") - f(a, "center_xyz_error_max_mm"),
        "burial_p95_delta_mm": f(b, "burial_depth_error_p95_mm") - f(a, "burial_depth_error_p95_mm"),
        "burial_max_delta_mm": f(b, "burial_depth_error_max_mm") - f(a, "burial_depth_error_max_mm"),
        "catastrophic_delta": f(b, "catastrophic_failure_count") - f(a, "catastrophic_failure_count"),
        "geometry_delta": f(b, "geometry_branch_failure_count") - f(a, "geometry_branch_failure_count"),
        "cuboid_ellipsoid_confusion_delta": f(b, "cuboid_ellipsoid_confusion_rate") - f(a, "cuboid_ellipsoid_confusion_rate"),
    }


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def mixed_signal(deltas: list[float], epsilon: float = 1e-9) -> str:
    improved = any(v < -epsilon for v in deltas)
    worsened = any(v > epsilon for v in deltas)
    if improved and worsened:
        return "mixed"
    if improved:
        return "yes"
    return "no"


def run(args: argparse.Namespace) -> int:
    probe = selected_test_rows(read_csv(args.probe_metrics))
    sep_rows = {row["observation_config"]: row for row in read_csv(args.separability)}
    required = ["mag_x_5line_only", "dual_mag_xy_5line", "mag_x_9line_only", "dual_mag_xy_9line"]
    missing = [name for name in required if name not in probe]
    if missing:
        raise RuntimeError(f"missing selected test probe rows: {missing}")
    missing_sep = [name for name in required if name not in sep_rows]
    if missing_sep:
        raise RuntimeError(f"missing separability rows: {missing_sep}")

    c5 = compare(probe["mag_x_5line_only"], probe["dual_mag_xy_5line"])
    c9 = compare(probe["mag_x_9line_only"], probe["dual_mag_xy_9line"])
    d9_vs_d5 = compare(probe["dual_mag_xy_5line"], probe["dual_mag_xy_9line"])
    sep5 = {
        "shape_delta": f(sep_rows["dual_mag_xy_5line"], "shape_nn_consistency") - f(sep_rows["mag_x_5line_only"], "shape_nn_consistency"),
        "ambiguous_delta": f(sep_rows["dual_mag_xy_5line"], "ambiguous_neighbor_rate") - f(sep_rows["mag_x_5line_only"], "ambiguous_neighbor_rate"),
        "cuboid_ellipsoid_delta": f(sep_rows["dual_mag_xy_5line"], "cuboid_ellipsoid_cross_nn_rate") - f(sep_rows["mag_x_5line_only"], "cuboid_ellipsoid_cross_nn_rate"),
    }
    sep9 = {
        "shape_delta": f(sep_rows["dual_mag_xy_9line"], "shape_nn_consistency") - f(sep_rows["mag_x_9line_only"], "shape_nn_consistency"),
        "ambiguous_delta": f(sep_rows["dual_mag_xy_9line"], "ambiguous_neighbor_rate") - f(sep_rows["mag_x_9line_only"], "ambiguous_neighbor_rate"),
        "cuboid_ellipsoid_delta": f(sep_rows["dual_mag_xy_9line"], "cuboid_ellipsoid_cross_nn_rate") - f(sep_rows["mag_x_9line_only"], "cuboid_ellipsoid_cross_nn_rate"),
    }

    dual5_better = c5["score_delta"] < -0.02 and c5["shape_f1_delta"] >= -0.05
    dual9_better = c9["score_delta"] < -0.02 and c9["shape_f1_delta"] >= -0.05
    nine_better = d9_vs_d5["score_delta"] < -0.02 and d9_vs_d5["shape_f1_delta"] >= -0.05
    confusion_signal = mixed_signal(
        [
            c5["cuboid_ellipsoid_confusion_delta"],
            c9["cuboid_ellipsoid_confusion_delta"],
            sep5["cuboid_ellipsoid_delta"],
            sep9["cuboid_ellipsoid_delta"],
        ]
    )
    tail_signal = mixed_signal(
        [
            c5["center_p95_delta_mm"],
            c9["center_p95_delta_mm"],
            c5["burial_p95_delta_mm"],
            c9["burial_p95_delta_mm"],
        ]
    )

    optional = [row for name, row in probe.items() if name.endswith("plus_mag_features")]
    optional_best = min(optional, key=score) if optional else None
    best_main = min([probe[name] for name in required], key=score)
    if optional_best and score(optional_best) + 0.02 < score(best_main):
        next_step = "C_train_dual_mag_xy_plus_richer_feature_fusion_model"
        next_note = "mag_x/mag_y 差异特征在 diagnostic probe 中优于 plain dual/single configs"
    elif dual9_better and (nine_better or score(probe["dual_mag_xy_9line"]) <= score(probe["dual_mag_xy_5line"])):
        next_step = "B_train_dual_mag_xy_9line_internal_model"
        next_note = "dual_mag_xy_9line 有最强 diagnostic 证据"
    elif dual5_better:
        next_step = "A_train_dual_mag_xy_5line_internal_model"
        next_note = "dual_mag_xy_5line 相对 mag_x_5line 有改善，采集成本低于 9line"
    elif not (dual5_better or dual9_better):
        next_step = "E_keep_abstention_only_route_and_pause_internal_refinement"
        next_note = "dual magnetization 没有清晰优于 single mag；暂不进入正式训练"
    else:
        next_step = "D_revise_internal_output_labels"
        next_note = "信号混合且 tail/confusion 未稳定改善，需要重新审计输出定义"

    rows = [
        {
            "question": "dual_mag_xy_5line 是否优于 mag_x_5line",
            "answer": yes_no(dual5_better),
            "evidence": f"score_delta={c5['score_delta']:.6f}; shape_f1_delta={c5['shape_f1_delta']:.6f}; center_p95_delta_mm={c5['center_p95_delta_mm']:.3f}; burial_p95_delta_mm={c5['burial_p95_delta_mm']:.3f}",
            "decision": "支持 5line dual 进入训练" if dual5_better else "5line dual 不足以单独进入训练",
        },
        {
            "question": "dual_mag_xy_9line 是否优于 mag_x_9line",
            "answer": yes_no(dual9_better),
            "evidence": f"score_delta={c9['score_delta']:.6f}; shape_f1_delta={c9['shape_f1_delta']:.6f}; center_p95_delta_mm={c9['center_p95_delta_mm']:.3f}; burial_p95_delta_mm={c9['burial_p95_delta_mm']:.3f}",
            "decision": "支持 9line dual 进入训练" if dual9_better else "9line dual 不足以单独进入训练",
        },
        {
            "question": "9line 是否明显优于 5line",
            "answer": yes_no(nine_better),
            "evidence": f"dual_9_vs_dual_5_score_delta={d9_vs_d5['score_delta']:.6f}; center_p95_delta_mm={d9_vs_d5['center_p95_delta_mm']:.3f}; burial_p95_delta_mm={d9_vs_d5['burial_p95_delta_mm']:.3f}",
            "decision": "9line 成本可能值得" if nine_better else "9line 成本优势未被证明",
        },
        {
            "question": "dual magnetization 是否改善 shape separability",
            "answer": "yes" if (sep5["shape_delta"] > 0 or sep9["shape_delta"] > 0) else "no",
            "evidence": f"shape_nn_delta_5={sep5['shape_delta']:.6f}; shape_nn_delta_9={sep9['shape_delta']:.6f}; ambiguous_delta_5={sep5['ambiguous_delta']:.6f}; ambiguous_delta_9={sep9['ambiguous_delta']:.6f}",
            "decision": "可分性有改善信号" if (sep5["shape_delta"] > 0 or sep9["shape_delta"] > 0) else "可分性没有改善信号",
        },
        {
            "question": "dual magnetization 是否降低 cuboid/ellipsoid confusion",
            "answer": confusion_signal,
            "evidence": f"probe5_delta={c5['cuboid_ellipsoid_confusion_delta']:.6f}; probe9_delta={c9['cuboid_ellipsoid_confusion_delta']:.6f}; nn5_delta={sep5['cuboid_ellipsoid_delta']:.6f}; nn9_delta={sep9['cuboid_ellipsoid_delta']:.6f}",
            "decision": "混合信号" if confusion_signal == "mixed" else ("几何分支混淆有改善信号" if confusion_signal == "yes" else "cuboid/ellipsoid 分支仍不充分"),
        },
        {
            "question": "dual magnetization 是否降低 center/burial tail",
            "answer": tail_signal,
            "evidence": f"center_p95_delta_5={c5['center_p95_delta_mm']:.3f}; center_p95_delta_9={c9['center_p95_delta_mm']:.3f}; burial_p95_delta_5={c5['burial_p95_delta_mm']:.3f}; burial_p95_delta_9={c9['burial_p95_delta_mm']:.3f}",
            "decision": "混合信号" if tail_signal == "mixed" else ("tail 有改善信号" if tail_signal == "yes" else "tail 没有改善信号"),
        },
        {
            "question": "是否进入 23.6 正式训练",
            "answer": "yes" if next_step.startswith(("A_", "B_", "C_")) else "no",
            "evidence": next_note,
            "decision": next_step,
        },
        {
            "question": "是否可更新 baseline",
            "answer": "no",
            "evidence": "diagnostic evaluation only; internal branch is not CURRENT_BASELINE",
            "decision": "CURRENT_BASELINE 保持不变",
        },
    ]
    write_csv(args.matrix, rows, ["question", "answer", "evidence", "decision"])

    lines = [
        "23.5 internal multi-magnetization evaluation route decision summary",
        "",
        f"dual_mag_xy_5line_better_than_mag_x_5line: {str(dual5_better).lower()}",
        f"dual_mag_xy_9line_better_than_mag_x_9line: {str(dual9_better).lower()}",
        f"dual_mag_xy_9line_better_than_dual_mag_xy_5line: {str(nine_better).lower()}",
        f"shape_separability_delta_5line: {sep5['shape_delta']:.6f}",
        f"shape_separability_delta_9line: {sep9['shape_delta']:.6f}",
        f"cuboid_ellipsoid_confusion_signal: {confusion_signal}",
        f"center_or_burial_tail_signal: {tail_signal}",
        f"recommended_next_step: {next_step}",
        f"recommendation_reason: {next_note}",
        "baseline_update: false",
        "formal_training_in_23_5: false",
        "",
        "结论：23.5 只用于 multi-magnetization 观测诊断，不得写成正式模型候选或 baseline。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

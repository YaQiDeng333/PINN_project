#!/usr/bin/env python
"""Decide the route after the 21.2 internal defect training gate."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, load_dataset, split_indices, write_csv


NEURAL_METRICS = ROOT / "results/metrics/internal_defect_neural_metrics.csv"
FEATURE_METRICS = ROOT / "results/metrics/internal_defect_feature_baseline_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_training_gate_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_training_gate_decision_matrix.csv"

FIELDS = ["question", "answer", "evidence", "decision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide internal defect training gate route.")
    parser.add_argument("--neural-metrics", type=Path, default=NEURAL_METRICS)
    parser.add_argument("--feature-metrics", type=Path, default=FEATURE_METRICS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def score(row: dict[str, str]) -> float:
    return f(row, "total_normalized_mae") + 0.35 * (1.0 - f(row, "shape_accuracy", 0.0))


def select_neural(rows: list[dict[str, str]], split: str) -> dict[str, str]:
    candidates = [row for row in rows if row.get("split") == split and is_true(row.get("selected_seed", ""))]
    if not candidates:
        raise RuntimeError(f"selected neural {split} row not found")
    return candidates[0]


def select_feature(rows: list[dict[str, str]], split: str, selected: bool) -> dict[str, str] | None:
    for row in rows:
        if row.get("split") != split:
            continue
        if selected and row.get("selected_model"):
            return row
        if (not selected) and row.get("model") == "mean_baseline":
            return row
    return None


def rel_improved(new: float, ref: float, margin: float = 0.0) -> bool:
    return new <= ref * (1.0 - margin)


def main() -> int:
    args = parse_args()
    neural_rows = read_csv(args.neural_metrics)
    feature_rows = read_csv(args.feature_metrics)
    neural_train = select_neural(neural_rows, "train")
    neural_val = select_neural(neural_rows, "val")
    neural_test = select_neural(neural_rows, "test")
    mean_test = select_feature(feature_rows, "test", selected=False)
    feat_test = select_feature(feature_rows, "test", selected=True)
    if mean_test is None or feat_test is None:
        raise RuntimeError("mean or selected feature baseline test row not found")

    neural_score = score(neural_test)
    mean_score = score(mean_test)
    feat_score = score(feat_test)
    dataset = load_dataset()
    splits = split_indices(dataset.split)
    split_shape_counts = {name: {str(k): int(v) for k, v in Counter(dataset.shape_type[idx]).items()} for name, idx in splits.items()}
    split_burial_counts = {name: {str(k): int(v) for k, v in Counter(dataset.burial_depth_level[idx]).items()} for name, idx in splits.items()}
    split_shape_complete = all(set(counts) == {"internal_sphere", "internal_ellipsoid", "internal_cuboid"} for counts in split_shape_counts.values())
    split_burial_complete = all(set(counts) == {"shallow", "medium", "deep", "deep_plus"} for counts in split_burial_counts.values())
    train_fit = f(neural_train, "total_normalized_mae") <= 0.35 and f(neural_train, "shape_accuracy") >= 0.75
    beat_mean = neural_score < mean_score and f(neural_test, "shape_accuracy") >= f(mean_test, "shape_accuracy")
    beat_feature_score = neural_score < feat_score and f(neural_test, "shape_accuracy") >= f(feat_test, "shape_accuracy") - 0.05
    beat_feature_regression = f(neural_test, "total_normalized_mae") < f(feat_test, "total_normalized_mae")
    beat_feature = beat_feature_score and beat_feature_regression
    lwd_learnable = all(
        rel_improved(f(neural_test, key), f(mean_test, key), margin=0.05)
        for key in ["L_mae_mm", "W_mae_mm", "D_mae_mm"]
    )
    burial_learnable = rel_improved(f(neural_test, "burial_depth_mae_mm"), f(mean_test, "burial_depth_mae_mm"), margin=0.05)
    center_learnable = rel_improved(f(neural_test, "center_xyz_mae_mm"), f(mean_test, "center_xyz_mae_mm"), margin=0.05)
    shape_learnable = f(neural_test, "shape_accuracy") >= 0.50 and f(neural_test, "shape_accuracy") > f(mean_test, "shape_accuracy") + 0.05

    if not split_shape_complete or not split_burial_complete:
        next_step = "A_expand_internal_defect_dataset"
        next_reason = "val/test split 覆盖不完整，当前 gate 只能证明有可学习信号，不能支撑稳健 shape/depth 结论。"
        n96_sufficient = "不足；val/test shape 或 burial 覆盖不完整，需要扩展或重做分层 split。"
    elif beat_feature and lwd_learnable and burial_learnable and center_learnable and shape_learnable:
        next_step = "A_expand_internal_defect_dataset"
        next_reason = "neural gate 已超过 mean/feature baseline，下一步应扩展 internal pilot 数据到 formal training pack。"
        n96_sufficient = "作为训练门控足够；作为 formal benchmark 不足。"
    elif beat_mean and not beat_feature:
        next_step = "C_add_shape_conditioned_model"
        next_reason = "neural 能超过 mean baseline 但未超过 delta_b feature baseline，先做 shape-conditioned/internal architecture。"
        n96_sufficient = "足够暴露可学习性，但不足以支撑稳健模型选择。"
    elif not train_fit:
        next_step = "B_improve_internal_model"
        next_reason = "当前网络连 train fit 都不充分，应先修模型容量、loss 或归一化。"
        n96_sufficient = "不足；当前结果无法证明模型容量或数据覆盖。"
    else:
        next_step = "A_expand_internal_defect_dataset"
        next_reason = "已有可学习信号但泛化/分组稳定性有限，扩展数据比继续小修更关键。"
        n96_sufficient = "只够 training gate，不够 baseline 或 formal benchmark。"

    rows = [
        {
            "question": "模型是否能 fit train",
            "answer": "是" if train_fit else "否",
            "evidence": f"train_total_normalized_mae={f(neural_train, 'total_normalized_mae'):.6f}; train_shape_accuracy={f(neural_train, 'shape_accuracy'):.6f}",
            "decision": "train_fit_check",
        },
        {
            "question": "neural 是否 beat mean baseline",
            "answer": "是" if beat_mean else "否",
            "evidence": f"neural_score={neural_score:.6f}; mean_score={mean_score:.6f}; neural_shape_acc={f(neural_test, 'shape_accuracy'):.6f}; mean_shape_acc={f(mean_test, 'shape_accuracy'):.6f}",
            "decision": "mean_baseline_comparison",
        },
        {
            "question": "neural 是否 beat feature baseline",
            "answer": "综合 score 是；纯回归 MAE 否" if beat_feature_score and not beat_feature_regression else ("是" if beat_feature else "否"),
            "evidence": f"neural_score={neural_score:.6f}; feature_score={feat_score:.6f}; neural_total_mae={f(neural_test, 'total_normalized_mae'):.6f}; feature_total_mae={f(feat_test, 'total_normalized_mae'):.6f}; feature_model={feat_test.get('model')}",
            "decision": "feature_baseline_comparison",
        },
        {
            "question": "L/W/D 是否可学习",
            "answer": "是" if lwd_learnable else "风险",
            "evidence": f"neural={f(neural_test, 'L_mae_mm'):.3f}/{f(neural_test, 'W_mae_mm'):.3f}/{f(neural_test, 'D_mae_mm'):.3f} mm; mean={f(mean_test, 'L_mae_mm'):.3f}/{f(mean_test, 'W_mae_mm'):.3f}/{f(mean_test, 'D_mae_mm'):.3f} mm",
            "decision": "dimension_learnability",
        },
        {
            "question": "burial_depth 是否可学习",
            "answer": "是" if burial_learnable else "风险",
            "evidence": f"neural={f(neural_test, 'burial_depth_mae_mm'):.3f} mm; mean={f(mean_test, 'burial_depth_mae_mm'):.3f} mm",
            "decision": "burial_depth_learnability",
        },
        {
            "question": "center_xyz 是否可学习",
            "answer": "是" if center_learnable else "风险",
            "evidence": f"neural={f(neural_test, 'center_xyz_mae_mm'):.3f} mm; mean={f(mean_test, 'center_xyz_mae_mm'):.3f} mm",
            "decision": "center_learnability",
        },
        {
            "question": "shape_type 是否可分类",
            "answer": "局部可分，但评价不完整" if shape_learnable and not split_shape_complete else ("是" if shape_learnable else "风险"),
            "evidence": f"neural_acc={f(neural_test, 'shape_accuracy'):.6f}; mean_acc={f(mean_test, 'shape_accuracy'):.6f}; feature_acc={f(feat_test, 'shape_accuracy'):.6f}; split_shape_counts={split_shape_counts}",
            "decision": "shape_classification",
        },
        {
            "question": "split 覆盖是否足够",
            "answer": "否" if (not split_shape_complete or not split_burial_complete) else "是",
            "evidence": f"split_shape_counts={split_shape_counts}; split_burial_counts={split_burial_counts}",
            "decision": "split_coverage",
        },
        {
            "question": "N=96 是否足够",
            "answer": n96_sufficient,
            "evidence": "21.1 pilot pack 覆盖 3 shape、4 burial、3 size，但每组 test 样本很少。",
            "decision": "dataset_scale",
        },
        {
            "question": "下一步唯一建议",
            "answer": next_step,
            "evidence": next_reason,
            "decision": "route",
        },
    ]
    write_csv(args.matrix, rows, FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.2 内部/埋藏缺陷 training gate 决策摘要",
                "",
                "CURRENT_BASELINE: 未修改，仍为 surface/near-surface true 3D RBC baseline。",
                "internal_branch_status: 独立分支 training gate，不是 baseline replacement。",
                f"selected_neural_test_total_normalized_mae: {f(neural_test, 'total_normalized_mae'):.6f}",
                f"selected_neural_test_LWD_mae_mm: {f(neural_test, 'L_mae_mm'):.3f} / {f(neural_test, 'W_mae_mm'):.3f} / {f(neural_test, 'D_mae_mm'):.3f}",
                f"selected_neural_test_burial_depth_mae_mm: {f(neural_test, 'burial_depth_mae_mm'):.3f}",
                f"selected_neural_test_center_xyz_mae_mm: {f(neural_test, 'center_xyz_mae_mm'):.3f}",
                f"selected_neural_test_shape_accuracy: {f(neural_test, 'shape_accuracy'):.6f}",
                f"beat_mean_baseline: {beat_mean}",
                f"beat_feature_baseline_composite_score: {beat_feature_score}",
                f"beat_feature_baseline_regression_mae: {beat_feature_regression}",
                f"feature_selected_model: {feat_test.get('model')}",
                f"feature_test_total_normalized_mae: {f(feat_test, 'total_normalized_mae'):.6f}",
                f"feature_test_burial_depth_mae_mm: {f(feat_test, 'burial_depth_mae_mm'):.3f}",
                f"LWD_learnable: {lwd_learnable}",
                f"burial_depth_learnable: {burial_learnable}",
                f"center_xyz_learnable: {center_learnable}",
                f"shape_type_classifiable: {shape_learnable}",
                f"split_shape_complete: {split_shape_complete}",
                f"split_burial_complete: {split_burial_complete}",
                f"split_shape_counts: {split_shape_counts}",
                f"split_burial_counts: {split_burial_counts}",
                f"N96_sufficiency: {n96_sufficient}",
                f"next_step: {next_step}",
                f"reason: {next_reason}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

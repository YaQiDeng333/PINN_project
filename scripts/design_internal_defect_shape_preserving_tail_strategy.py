#!/usr/bin/env python
"""22.4 strategy design for shape-preserving tail reduction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY = ROOT / "results/summaries/internal_defect_shape_preserving_tail_strategy_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_shape_preserving_tail_strategy_matrix.csv"
AUDIT = ROOT / "results/metrics/internal_defect_shape_tail_tradeoff_matrix.csv"

FIELDS = [
    "rank",
    "strategy_id",
    "strategy_name",
    "requires_training",
    "requires_comsol",
    "requires_new_data",
    "protects_shape_f1",
    "expected_tail_reduction",
    "risk",
    "acceptance_criteria",
    "decision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design shape-preserving internal tail strategy.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = [
        {
            "rank": 1,
            "strategy_id": "S3_two_stage_freeze_shape_then_tail_regress",
            "strategy_name": "先训练/复用 shape classifier 和 shared encoder，冻结 shape 相关参数，再训练 center/burial tail heads",
            "requires_training": True,
            "requires_comsol": False,
            "requires_new_data": False,
            "protects_shape_f1": "高；训练阶段显式禁止 tail loss 反向破坏 shape head",
            "expected_tail_reduction": "高；直接针对 center/burial tail，同时保留 shape branch",
            "risk": "如果 frozen representation 对 hard-case center/burial 不够表达，tail heads 改善有限",
            "acceptance_criteria": "shape F1 >= 旧 B2 0.841 或至少不低于 0.82；catastrophic < 9/60 且目标 <=5%；geometry_branch=0；center p95/max 和 burial p95/max 均不比 H2 退化",
            "decision": "recommended_next_step",
        },
        {
            "rank": 2,
            "strategy_id": "S2_shape_confidence_router",
            "strategy_name": "导出 shape probability；高置信 shape 走对应 regression path，低置信输出 unstable/abstain",
            "requires_training": True,
            "requires_comsol": False,
            "requires_new_data": False,
            "protects_shape_f1": "中高；依赖 shape confidence calibration",
            "expected_tail_reduction": "中；主要降低不可信样本的误用风险，而不是直接修正所有回归错误",
            "risk": "会增加 abstain/unstable 输出；如果概率未校准，router 可能误放行 geometry branch failure",
            "acceptance_criteria": "shape probability 可用；geometry_branch=0 或被 unstable 标记；stable 子集 catastrophic <=5%；全量指标不隐瞒失败",
            "decision": "secondary_safety_layer",
        },
        {
            "rank": 3,
            "strategy_id": "S1_fixed_shape_classifier_plus_regressors",
            "strategy_name": "提高 shape CE 权重或单独预训练 shape head，再训练共享 regressor",
            "requires_training": True,
            "requires_comsol": False,
            "requires_new_data": False,
            "protects_shape_f1": "中；比 H2 好，但仍可能被共享 regression loss 拉偏",
            "expected_tail_reduction": "中；能缓解 shape 退化，但对 center/burial tail 的隔离不如 S3",
            "risk": "shape 权重过高会牺牲 L/W/D 或 burial/center 平均误差",
            "acceptance_criteria": "shape F1 不低于旧 B2；total MAE 不明显退化；catastrophic 低于 H2",
            "decision": "fallback_ablation",
        },
        {
            "rank": 4,
            "strategy_id": "S4_hardcase_topup_second_round",
            "strategy_name": "仅在 failure 明确集中到少数 strata 时追加第二轮 hard-case COMSOL top-up",
            "requires_training": False,
            "requires_comsol": True,
            "requires_new_data": True,
            "protects_shape_f1": "低；数据增加不能保证不破坏 shape head",
            "expected_tail_reduction": "不确定；22.3 已证明 top-up 有帮助但不够稳定",
            "risk": "failure 不是单一 strata，继续盲目 top-up 会扩大成本且仍可能牺牲 shape",
            "acceptance_criteria": "只有当 22.4 audit 证明 failure 集中在明确 shape/burial/size/aspect strata，才进入 COMSOL top-up",
            "decision": "defer_until_after_S3",
        },
        {
            "rank": 5,
            "strategy_id": "S5_output_uncertainty_or_abstention",
            "strategy_name": "为可能 geometry_branch_failure 的样本输出 warning / unstable，不强行声称稳定预测",
            "requires_training": False,
            "requires_comsol": False,
            "requires_new_data": False,
            "protects_shape_f1": "不直接保护；属于推理输出安全层",
            "expected_tail_reduction": "不降低真实误差，但降低错误预测被误用的风险",
            "risk": "如果没有 shape probability 或 calibrated uncertainty，只能做粗糙规则",
            "acceptance_criteria": "必须基于模型输出概率或明确 tail flags；不能用 true labels；报告 full-set 与 stable-subset 两套指标",
            "decision": "future_inference_contract",
        },
    ]
    write_csv(args.matrix, rows, FIELDS)
    lines = [
        "22.4 shape-preserving tail reduction strategy design",
        "scope：analysis artifact generator；只读取 22.4 audit 证据并写出策略 summary/CSV，不训练，不运行 COMSOL，不生成或修改 data/NPZ。",
        "核心判断：H2 的 hard-case sample weighting 降低部分 center tail，但明显牺牲 shape branch，并使 burial max 退化。",
        "唯一推荐下一步：S3_two_stage_freeze_shape_then_tail_regress。",
        "执行含义：下一训练阶段先保护 shape classifier/shared encoder，再只训练 center/burial tail heads；不要继续直接 H2 tail weighting。",
        "shape-confidence router：作为后续推理安全层或次级 ablation，不作为下一步主训练路线。",
        "新 COMSOL：本策略不需要；只有 S3 后仍有集中 strata failure 时，才考虑第二轮 hard-case top-up。",
        "baseline_status：internal defect 仍是独立 branch，不是 CURRENT_BASELINE，不混入 surface/near-surface RBC baseline。",
        "acceptance：shape F1 不低于旧 B2 过多；catastrophic failure 低于 9/60 且目标 <=5%；geometry_branch_failure 目标 0；center/burial p95/max 不退化。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

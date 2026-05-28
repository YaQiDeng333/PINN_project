#!/usr/bin/env python
"""20.99 internal / buried defect feasibility 路线判定。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "INTERNAL_DEFECT_SCHEMA.md"
PLAN_CSV = ROOT / "results/metrics/internal_defect_comsol_smoke_pack_plan.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_feasibility_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_feasibility_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not SCHEMA.exists():
        raise SystemExit(f"缺少 schema: {SCHEMA}")
    rows = read_csv(PLAN_CSV)
    shape_types = sorted({row["shape_type"] for row in rows})
    burial_bins = sorted({row["burial_depth_bin"] for row in rows})
    decisions = [
        {
            "question": "can_transfer_from_surface_rbc_baseline",
            "answer": False,
            "evidence": "internal/buried defect needs burial_depth_m and volumetric labels",
            "decision": "不能直接从当前 surface RBC baseline 迁移，只能借用三轴输入规范和部分 preprocessing 约定",
        },
        {
            "question": "need_independent_comsol_generator",
            "answer": True,
            "evidence": "internal cavity geometry and no-defect reference semantics differ from surface RBC",
            "decision": "需要独立 internal defect COMSOL generator / smoke pack",
        },
        {
            "question": "first_model_family_if_smoke_passes",
            "answer": "shape_type_conditioned_sizing",
            "evidence": f"planned shape_types={shape_types}",
            "decision": "先做 shape_type + L/W/D + burial_depth 的 sizing，不直接做自由 3D occupancy",
        },
        {
            "question": "need_bx_by_bz",
            "answer": True,
            "evidence": "current true 3D MFL route and 20.98 blockers require tri-axis availability",
            "decision": "默认主线需要 Bx/By/Bz；Bz-only 不作为主线",
        },
        {
            "question": "bz_only_feasible",
            "answer": "limited_diagnostic_only",
            "evidence": "Bz-only loses lateral/vector information and was a 20.98 blocker for current route",
            "decision": "Bz-only 可以做低能力 feasibility 对照，但不能替代三轴主线",
        },
        {
            "question": "next_unique_step",
            "answer": "generate internal COMSOL smoke pack",
            "evidence": f"planned_rows={len(rows)}, burial_bins={burial_bins}",
            "decision": "下一步进入 6-12 sample internal COMSOL smoke pack；本轮不执行",
        },
    ]
    write_csv(MATRIX, decisions)
    lines = [
        "20.99 internal / buried defect feasibility route decision",
        "",
        "surface_rbc_direct_use: false",
        "need_independent_schema: true",
        "need_independent_comsol_generator: true",
        "recommended_first_output: shape_type + L/W/D + burial_depth + center_xyz",
        "recommended_smoke_pack_size: 12 samples",
        "minimum_smoke_pack_size: 6 samples",
        "need_Bx_By_Bz: true",
        "Bz_only_status: limited diagnostic branch only",
        "",
        "why:",
        "- internal / buried defect 的核心标签是 `burial_depth_m` / `depth_to_surface_m`，当前 surface RBC 六参数没有这个语义。",
        "- 如果直接使用 surface RBC baseline，模型会把埋深变化误解释成 surface profile / curvature 变化。",
        "- 三轴 `Bx/By/Bz` 是默认主线；只有 `Bz` 时可做低能力诊断，但不能宣称进入 true 3D internal baseline。",
        "",
        "next_unique_step: 生成 internal COMSOL smoke pack 之前先确认 schema 字段和实验可采 metadata；确认后再运行 COMSOL。",
        "COMSOL_run_this_stage: false",
        "training_run_this_stage: false",
        "data_npz_generation_this_stage: false",
        "CURRENT_BASELINE_update: false",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

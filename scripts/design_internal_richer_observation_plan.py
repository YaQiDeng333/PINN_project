#!/usr/bin/env python
"""22.8 richer-observation candidate plan generator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/summaries/internal_richer_observation_candidate_plan_summary.txt"
MATRIX_PATH = ROOT / "results/metrics/internal_richer_observation_candidate_matrix.csv"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "R0_current_reference",
            "observation_config": "Bx/By/Bz; 3 y-lines [-0.001,0,0.001]; 201 x-points; sensor_z=0.008m; single x-direction scan; nominal magnetization",
            "comsol_difficulty": "low",
            "real_experiment_difficulty": "low",
            "relative_data_multiplier_per_base": 1,
            "input_shape_change": "(3,3,201), current reference",
            "loader_model_changes": "none; reference replay only",
            "target_failures": "baseline comparator for all richer-observation variants",
            "expected_benefit": "none by itself; required paired reference",
            "fit_for_22_9": True,
            "priority": 0,
        },
        {
            "candidate_id": "R1_more_y_lines",
            "observation_config": "Bx/By/Bz; 5 or 9 y-lines; 201 x-points; sensor_z=0.008m; single x-direction scan",
            "comsol_difficulty": "low",
            "real_experiment_difficulty": "medium",
            "relative_data_multiplier_per_base": 2,
            "input_shape_change": "(3,5,201) and (3,9,201); loader must preserve variable y-line count",
            "loader_model_changes": "new diagnostic loader or pad/resample y-lines; models need y-line-aware input adapter before training",
            "target_failures": "center_y / lateral extent tail, compact large high-risk, first-pass shape confusion",
            "expected_benefit": "directly tests whether sparse 3-line lateral sampling causes high abstention",
            "fit_for_22_9": True,
            "priority": 1,
        },
        {
            "candidate_id": "R2_multi_liftoff",
            "observation_config": "Bx/By/Bz; 5 y-lines; liftoff 0.006/0.008/0.010/0.012m; single x-direction scan",
            "comsol_difficulty": "medium",
            "real_experiment_difficulty": "medium_high",
            "relative_data_multiplier_per_base": 4,
            "input_shape_change": "paired liftoff stack, e.g. (4,3,5,201) or flattened observation set",
            "loader_model_changes": "manifest must encode sensor_z_m per observation; model needs liftoff-conditioned or stacked input",
            "target_failures": "burial_depth / size attenuation ambiguity, deep_plus tail, center/burial tradeoff",
            "expected_benefit": "strongest diagnostic for depth attenuation because the same geometry is observed at multiple liftoffs",
            "fit_for_22_9": True,
            "priority": 2,
        },
        {
            "candidate_id": "R3_multi_scan_direction",
            "observation_config": "x-direction scan plus y-direction scan; 3 or 5 y-lines each; nominal liftoff",
            "comsol_difficulty": "medium_high",
            "real_experiment_difficulty": "high",
            "relative_data_multiplier_per_base": 2,
            "input_shape_change": "direction-stacked observations with direction metadata and axis-order convention",
            "loader_model_changes": "new direction-aware schema, no-defect cache per direction, model direction fusion",
            "target_failures": "cuboid/ellipsoid confusion, elongated_x/y aspect ambiguity, geometry branch failure",
            "expected_benefit": "likely helpful for shape branch, but protocol cost is higher than R1/R2",
            "fit_for_22_9": False,
            "priority": 3,
        },
        {
            "candidate_id": "R4_multi_magnetization_direction",
            "observation_config": "multiple source/magnetization directions via Je variants",
            "comsol_difficulty": "high",
            "real_experiment_difficulty": "very_high",
            "relative_data_multiplier_per_base": 2,
            "input_shape_change": "magnetization-direction stack with source metadata",
            "loader_model_changes": "new source-direction schema and physics validation; model needs magnetization fusion",
            "target_failures": "residual shape discriminability after R1/R2/R3",
            "expected_benefit": "possible but not first-order evidence from current failures",
            "fit_for_22_9": False,
            "priority": 5,
        },
    ]


def main() -> int:
    rows = candidate_rows()
    fields = [
        "candidate_id",
        "observation_config",
        "comsol_difficulty",
        "real_experiment_difficulty",
        "relative_data_multiplier_per_base",
        "input_shape_change",
        "loader_model_changes",
        "target_failures",
        "expected_benefit",
        "fit_for_22_9",
        "priority",
    ]
    write_csv(MATRIX_PATH, rows, fields)
    lines = [
        "22.8 internal richer-observation candidate plan",
        "",
        "候选结论：R1_more_y_lines 与 R2_multi_liftoff 是 22.9 第一轮默认候选；R3 multi-scan-direction 记录为第二优先级；R4 multi-magnetization 暂缓。",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['candidate_id']}: priority={row['priority']}, fit_for_22_9={row['fit_for_22_9']}, target={row['target_failures']}")
    lines.extend(
        [
            "",
            "实现边界：本阶段只规划，不创建 COMSOL generator、不生成 data/NPZ、不更新 registry/manifest。",
        ]
    )
    write_text(SUMMARY_PATH, "\n".join(lines) + "\n")
    print(json.dumps({"candidate_rows": len(rows), "matrix": str(MATRIX_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

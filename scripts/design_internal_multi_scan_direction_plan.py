#!/usr/bin/env python
"""23.2 multi-scan-direction candidate design.

This script writes plan-only candidate matrices for internal dual-direction
diagnostics. It does not run COMSOL, train models, or create data/NPZ files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/summaries/internal_multi_scan_direction_candidate_plan_summary.txt"
MATRIX_PATH = ROOT / "results/metrics/internal_multi_scan_direction_candidate_matrix.csv"


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


def candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "D0_reference_single_direction",
            "description": "复用 22.9 既有 x_scan R1/R2 reference，不重新生成。",
            "comsol_cost": "none",
            "real_experiment_cost": "none",
            "new_rows_per_base": 0,
            "total_new_rows_target": 0,
            "input_tensor_shape": "(N,3,1,9,201) after adapter or existing (N,3,9,201)",
            "metadata_changes": "读取 existing scan_direction=x_direction；不新增 manifest。",
            "loader_model_changes": "23.3 中作为 single-direction baseline comparator。",
            "target_failures": "reference only",
            "fit_for_23_2b": False,
            "priority": 0,
            "decision": "reuse_existing",
        },
        {
            "candidate_id": "D1_dual_direction_5line_z0p008",
            "description": "x_scan 5-line + y_scan 5-line，sensor_z_m=0.008。",
            "comsol_cost": "medium_low",
            "real_experiment_cost": "medium",
            "new_rows_per_base": 1,
            "total_new_rows_target": 30,
            "input_tensor_shape": "(N,3,2,9,201) padded; direction_mask=[true,true]; scan_line_mask has 5 valid lines per direction",
            "metadata_changes": "新增 direction_names、direction_mask、path_coordinate_axis、line_coordinate_axis、path_coordinate_m、line_coordinate_m。",
            "loader_model_changes": "23.3 loader 需要 direction dimension；model 需要 direction fusion 或 flatten 后保留 direction mask。",
            "target_failures": "cuboid/ellipsoid confusion, center_xyz tail, compact/large shape ambiguity",
            "fit_for_23_2b": True,
            "priority": 1,
            "decision": "default_generate_y_scan",
        },
        {
            "candidate_id": "D2_dual_direction_9line_z0p008",
            "description": "x_scan 9-line + y_scan 9-line，sensor_z_m=0.008。",
            "comsol_cost": "medium",
            "real_experiment_cost": "medium_high",
            "new_rows_per_base": 1,
            "total_new_rows_target": 30,
            "input_tensor_shape": "(N,3,2,9,201) no line padding needed for D2, but keep mask for shared schema",
            "metadata_changes": "同 D1；line offsets 更密集。",
            "loader_model_changes": "23.3 中直接比较 D1 vs D2 是否有增益。",
            "target_failures": "elongated_x/elongated_y asymmetry, geometry_branch_failure, lateral extent ambiguity",
            "fit_for_23_2b": True,
            "priority": 1,
            "decision": "default_generate_y_scan",
        },
        {
            "candidate_id": "D3_dual_direction_5line_multi_liftoff",
            "description": "双方向 5-line 再叠加 0.006/0.008/0.010/0.012m liftoff。",
            "comsol_cost": "high",
            "real_experiment_cost": "high",
            "new_rows_per_base": 6,
            "total_new_rows_target": 180,
            "input_tensor_shape": "(N,3,2,4,5,201) or staged observation stack; schema cost high",
            "metadata_changes": "需要同时管理 direction 和 liftoff dimensions。",
            "loader_model_changes": "需要新的 multi-axis observation fusion，不进入 23.2b。",
            "target_failures": "burial/size ambiguity if D1/D2 improves shape but burial tail remains",
            "fit_for_23_2b": False,
            "priority": 3,
            "decision": "defer_until_D1_D2_result",
        },
        {
            "candidate_id": "D4_multi_magnetization_direction",
            "description": "改变磁化/source 方向以增强 shape discriminability。",
            "comsol_cost": "very_high",
            "real_experiment_cost": "very_high",
            "new_rows_per_base": 2,
            "total_new_rows_target": 60,
            "input_tensor_shape": "(N,3,magnetization,observations,lines,201)",
            "metadata_changes": "需要 source/magnetization protocol 和物理一致性校验。",
            "loader_model_changes": "需要 source-direction fusion；当前 evidence 不足。",
            "target_failures": "residual shape ambiguity after multi-scan direction fails",
            "fit_for_23_2b": False,
            "priority": 5,
            "decision": "defer",
        },
    ]


def main() -> int:
    rows = candidate_rows()
    fields = [
        "candidate_id",
        "description",
        "comsol_cost",
        "real_experiment_cost",
        "new_rows_per_base",
        "total_new_rows_target",
        "input_tensor_shape",
        "metadata_changes",
        "loader_model_changes",
        "target_failures",
        "fit_for_23_2b",
        "priority",
        "decision",
    ]
    write_csv(MATRIX_PATH, rows, fields)
    lines = [
        "# 23.2 multi-scan-direction candidate plan",
        "",
        "候选结论：D1 和 D2 是 23.2b 默认执行候选；D0 只作既有 x_scan reference；D3/D4 暂缓。",
        "",
        "建议 assembled tensor contract:",
        "- delta_b shape: `(N,3,2,9,201)`",
        "- axis 1: `[Bx, By, Bz]`",
        "- axis 2: `direction_names=[x_scan,y_scan]`",
        "- axis 3: padded scan lines",
        "- axis 4: path samples",
        "",
        f"direction metadata: {dumps(['direction_mask','scan_line_mask','direction_names','path_coordinate_axis','line_coordinate_axis','path_coordinate_m','line_coordinate_m','sensor_z_m','observation_variant','base_group_id'])}",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['candidate_id']}: fit_for_23_2b={row['fit_for_23_2b']}, rows/base={row['new_rows_per_base']}, decision={row['decision']}")
    lines.extend(
        [
            "",
            "实现约束：23.2b 必须新增真正的 direction-aware sensor point builder；不能只写 `scan_direction=y_scan` metadata。",
            "本阶段不创建 manifest / registry / data / NPZ。",
        ]
    )
    write_text(SUMMARY_PATH, "\n".join(lines) + "\n")
    print(json.dumps({"candidate_rows": len(rows), "matrix": str(MATRIX_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

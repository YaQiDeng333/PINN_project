#!/usr/bin/env python
"""20.98 内部缺陷铁块 dry-run manifest 路线判定。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/real_data_internal_block_dry_run_manifest.json"
VALIDATION = ROOT / "results/metrics/true_3d_rbc_real_data_manifest_dry_run_validation.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_real_data_manifest_dry_run_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_real_data_manifest_dry_run_decision_matrix.csv"


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
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = read_csv(VALIDATION)
    blockers = [row for row in rows if row["severity"] == "blocker" and row["pass"] == "False"]
    blocker_fields = sorted({f"{row['item']}::{row['field']}" for row in blockers})
    internal_blocker = any(row["field"] == "defect_location_type" and row["pass"] == "False" for row in blockers)
    decisions = [
        {
            "question": "ready_for_inference",
            "answer": False,
            "evidence": f"hard_blocker_count={len(blockers)}",
            "decision": "不要对这个 dry-run manifest 运行 20.96 推理",
        },
        {
            "question": "internal_or_buried_defect",
            "answer": bool(internal_blocker),
            "evidence": manifest.get("defect_location_type"),
            "decision": "当前 surface/near-surface RBC baseline 不是正确分支",
        },
        {
            "question": "need_separate_internal_schema",
            "answer": True,
            "evidence": manifest.get("schema_branch_recommendation"),
            "decision": "真实数据接入前先创建 internal defect feasibility schema",
        },
        {
            "question": "minimum_metadata_if_future_surface_capture",
            "answer": "sensor_z_m, matched no-defect reference, tri-axis Bx/By/Bz, axis_order, scan_line_y_m, sensor_x_m=201, Tesla unit, coordinate_system, alignment status, gain status, magnetization setup, specimen geometry",
            "evidence": "20.97 schema contract",
            "decision": "任何 surface/RBC-compatible 真实推理前必须补齐这些字段",
        },
        {
            "question": "unique_next_step",
            "answer": "C. create internal defect feasibility schema",
            "evidence": "internal defect iron block does not match current surface RBC baseline",
            "decision": "实验设置澄清前不要进入 Bx/By/Bz surface baseline 推理",
        },
    ]
    write_csv(MATRIX, decisions)
    lines = [
        "20.98 true 3D RBC 真实数据 manifest dry-run 路线判定",
        "",
        f"manifest: {MANIFEST.relative_to(ROOT)}",
        "ready_for_inference: false",
        f"hard_blocker_count: {len(blockers)}",
        "internal_defect_requires_separate_schema: true",
        "current_surface_rbc_baseline_fit: false",
        "",
        "hard blockers:",
        *[f"- {field}" for field in blocker_fields],
        "",
        "如果未来要采集真实数据，最少需要补齐：",
        "- 实测 `sensor_z_m`，单位 m",
        "- 匹配的 `no_defect_reference_id` 和 no-defect reference 方法",
        "- 三轴 `Bx/By/Bz`；只有 Bz 仍是 blocker",
        "- `axis_order=[Bx,By,Bz]`",
        "- 三条 `scan_line_y_m`，以及重采样到 201 点的 `sensor_x_m`",
        "- Tesla 单位和坐标系定义",
        "- 传感器对齐状态和 gain calibration 状态",
        "- 试件几何、材料、励磁设置；如果是内部缺陷，还需要 burial depth / depth-to-surface 标签",
        "",
        "unique_next_step: C. create internal defect feasibility schema",
        "COMSOL_run: false",
        "training_run: false",
        "data_npz_generation: false",
        "CURRENT_BASELINE_update: false",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

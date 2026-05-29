#!/usr/bin/env python
"""21.8 internal defect report route decision."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VS_REFERENCE = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_vs_reference.csv"
VISUAL_ASSETS = ROOT / "results/metrics/internal_defect_benchmark_visual_assets_index.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_report_route_decision_summary.txt"
MATRIX_OUT = ROOT / "results/metrics/internal_defect_report_route_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def main() -> int:
    vs_rows = read_csv(VS_REFERENCE)
    b2 = next(row for row in vs_rows if row["source"] == "21.7_B2_formal_rerun")
    b0 = next(row for row in vs_rows if row["source"] == "21.4_neural_reference")
    feature = next(row for row in vs_rows if row["source"] == "21.4_feature_baseline")
    visual_rows = read_csv(VISUAL_ASSETS) if VISUAL_ASSETS.exists() else []
    has_visual_artifacts = any(str(row.get("exists", "")).lower() == "true" for row in visual_rows)

    b2_candidate = (
        safe_float(b2["total_normalized_mae"]) < safe_float(b0["total_normalized_mae"])
        and safe_float(b2["total_normalized_mae"]) < safe_float(feature["total_normalized_mae"])
        and safe_float(b2["burial_depth_mae_mm"]) < safe_float(feature["burial_depth_mae_mm"])
    )
    rows = [
        {
            "option": "A_internal_real_data_schema_alignment",
            "decision": "recommended",
            "reason": "B2 已形成 internal benchmark candidate；下一步应把真实 internal 样本 metadata/schema 对齐到该分支，而不是继续盲目扩数据。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
        {
            "option": "B_internal_real_sample_metadata_dry_run",
            "decision": "next_substep_after_schema_alignment",
            "reason": "如果用户已有 internal block 元数据，可用 dry run 先查 sensor_z、reference、axis、unit、coordinate、ground truth blockers。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
        {
            "option": "C_internal_shape_extension_dataset",
            "decision": "defer",
            "reason": "当前短板不是数据量或 shape 扩展；B2 已稳定优于 reference，先做真实样本对齐。",
            "requires_training": False,
            "requires_comsol": True,
            "updates_current_baseline": False,
        },
        {
            "option": "D_internal_inference_smoke",
            "decision": "blocked_until_artifact_recovery",
            "reason": "21.7 没有可加载的 B2 inference artifact；需要先 recover/export artifact 才能做 inference smoke 或 gallery。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
        {
            "option": "E_pause_internal_branch",
            "decision": "reject",
            "reason": "B2 candidate 有稳定正结果，暂停没有必要。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
    ]
    write_csv(
        MATRIX_OUT,
        rows,
        ["option", "decision", "reason", "requires_training", "requires_comsol", "updates_current_baseline"],
    )
    summary = [
        "21.8 internal defect report route decision",
        f"B2_candidate_confirmed: {str(b2_candidate).lower()}",
        f"visual_artifacts_available: {str(has_visual_artifacts).lower()}",
        "unique_next_step: A_internal_real_data_schema_alignment",
        "reason: B2 已经是 internal benchmark candidate；下一步应对齐真实 internal 样本的 schema/metadata，而不是继续盲目扩数据或把 internal 写成 CURRENT_BASELINE。",
        "artifact_note: internal inference smoke / gallery 需要先恢复 B2 inference artifact。",
        "baseline_decision: not baseline; no CURRENT_BASELINE update; internal branch remains independent.",
    ]
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

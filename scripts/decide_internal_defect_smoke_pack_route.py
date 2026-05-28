#!/usr/bin/env python
"""Decide the next route after the 21.0 internal defect smoke pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_smoke_pack_v1.manifest.json"
VALIDATION = ROOT / "results/metrics/internal_defect_smoke_pack_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_smoke_pack_group_summary.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_smoke_pack_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_smoke_pack_route_decision_matrix.csv"

FIELDS = ["decision_option", "recommended", "status", "evidence", "next_action", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide internal defect smoke pack route.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def bool_text(value: str) -> bool:
    return str(value).strip().lower() == "true"


def main() -> int:
    args = parse_args()
    check_no_overwrite([args.summary, args.matrix], args.overwrite)
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = read_csv(args.validation)
    groups = read_csv(args.group_summary)
    status = str(manifest.get("status", "missing"))
    n_samples = int(manifest.get("n_samples", 0))
    failed_checks = [row["check_name"] for row in checks if not bool_text(row.get("pass", ""))]
    shape_groups = [row for row in groups if row.get("group_field") == "shape_type" and int(row.get("success_count", "0") or 0) > 0]
    burial_groups = [
        row for row in groups if row.get("group_field") == "burial_depth_level" and int(row.get("success_count", "0") or 0) > 0
    ]
    full_coverage = len(shape_groups) == 3 and len(burial_groups) >= 3 and n_samples == 12 and not failed_checks
    partial_coverage = n_samples >= 6 and len(shape_groups) >= 2 and len(burial_groups) >= 2

    rows = [
        {
            "decision_option": "A_enter_21_1_internal_pilot_pack",
            "recommended": bool(full_coverage),
            "status": "recommended" if full_coverage else "not_ready",
            "evidence": f"status={status}; n_samples={n_samples}; shape_groups={len(shape_groups)}; burial_groups={len(burial_groups)}",
            "next_action": "设计 21.1 internal pilot pack" if full_coverage else "等待 full smoke 通过",
            "notes": "只有 full smoke_generated 才进入 pilot。",
        },
        {
            "decision_option": "B_fix_or_top_up_smoke_pack",
            "recommended": (not full_coverage) and partial_coverage,
            "status": "recommended" if (not full_coverage) and partial_coverage else "not_selected",
            "evidence": f"partial_coverage={partial_coverage}; failed_checks={failed_checks}",
            "next_action": "修复失败 shape 或补齐 burial 覆盖",
            "notes": "partial smoke 只能作为 feasibility diagnostic。",
        },
        {
            "decision_option": "C_block_internal_route",
            "recommended": not full_coverage and not partial_coverage,
            "status": "recommended" if not full_coverage and not partial_coverage else "not_selected",
            "evidence": f"status={status}; n_samples={n_samples}; failed_checks={failed_checks}",
            "next_action": "先修复 Boolean/mesh/solve/schema blocker",
            "notes": "不可训练，不可声明 internal baseline。",
        },
        {
            "decision_option": "D_real_data_deferred",
            "recommended": True,
            "status": "active_constraint",
            "evidence": "internal schema 仍是 COMSOL feasibility 阶段",
            "next_action": "真实实验继续暂缓",
            "notes": "真实 internal block 需要独立 schema 和 ground truth。",
        },
    ]
    write_csv(args.matrix, rows)
    recommended = [row for row in rows if row["recommended"] and row["decision_option"] != "D_real_data_deferred"]
    next_step = recommended[0]["next_action"] if recommended else "先修复 internal smoke blocker"
    lines = [
        "21.0 internal defect smoke pack route decision",
        "",
        f"dataset_id: {manifest.get('dataset_id')}",
        f"status: {status}",
        f"successful_samples: {n_samples}",
        f"shape_groups_with_success: {len(shape_groups)}",
        f"burial_depth_groups_with_success: {len(burial_groups)}",
        f"failed_checks: {failed_checks if failed_checks else 'none'}",
        "",
        f"unique_next_step: {next_step}",
        "current_baseline_update: false",
        "surface_rbc_mixed: false",
        "real_data_alignment: deferred",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

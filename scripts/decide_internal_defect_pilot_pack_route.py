#!/usr/bin/env python
"""Decide the next route after the 21.1 internal defect pilot pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v1.manifest.json"
VALIDATION = ROOT / "results/metrics/internal_defect_pilot_pack_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_pilot_pack_group_summary.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_pilot_pack_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_pilot_pack_route_decision_matrix.csv"

FIELDS = ["decision_option", "recommended", "status", "evidence", "next_action", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide internal defect pilot pack route.")
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
    train_ready = bool(manifest.get("train_ready_candidate", False))
    failed_checks = [row["check_name"] for row in checks if not bool_text(row.get("pass", ""))]
    shape_groups = [row for row in groups if row.get("group_field") == "shape_type" and int(row.get("success_count", "0") or 0) > 0]
    burial_groups = [row for row in groups if row.get("group_field") == "burial_depth_level" and int(row.get("success_count", "0") or 0) > 0]
    size_groups = [row for row in groups if row.get("group_field") == "size_level" and int(row.get("success_count", "0") or 0) > 0]
    split_groups = {row.get("group_value"): int(row.get("success_count", "0") or 0) for row in groups if row.get("group_field") == "split"}
    full_ready = status == "pilot_generated" and n_samples == 96 and train_ready and not failed_checks
    partial_ready = (
        status == "partial_pilot_generated"
        and n_samples >= 72
        and train_ready
        and len(shape_groups) == 3
        and len(burial_groups) == 4
        and len(size_groups) == 3
        and split_groups.get("train", 0) >= 48
        and split_groups.get("val", 0) >= 12
        and split_groups.get("test", 0) >= 12
        and not failed_checks
    )

    rows = [
        {
            "decision_option": "A_enter_21_2_internal_training_gate",
            "recommended": full_ready or partial_ready,
            "status": "recommended" if (full_ready or partial_ready) else "not_ready",
            "evidence": f"status={status}; n_samples={n_samples}; train_ready={train_ready}; split={split_groups}",
            "next_action": "进入 21.2 internal defect training gate" if full_ready else ("进入 21.2，但保留 pilot top-up 建议" if partial_ready else "等待 pilot pack validation 通过"),
            "notes": "21.2 仍是显式 internal training gate，不是 baseline replacement。",
        },
        {
            "decision_option": "B_top_up_or_fix_pilot_pack",
            "recommended": (not full_ready) and partial_ready,
            "status": "recommended" if (not full_ready) and partial_ready else "not_selected",
            "evidence": f"status={status}; n_samples={n_samples}; failed_checks={failed_checks}",
            "next_action": "补齐到 96 行或修复失败 shape/burial/size 组合",
            "notes": "partial pilot 可用于初步训练 gate，但正式 benchmark 前应 top-up。",
        },
        {
            "decision_option": "C_block_internal_training",
            "recommended": not full_ready and not partial_ready,
            "status": "recommended" if not full_ready and not partial_ready else "not_selected",
            "evidence": f"status={status}; n_samples={n_samples}; failed_checks={failed_checks}",
            "next_action": "先修复 Boolean/mesh/solve/schema blocker",
            "notes": "不可训练，不可声明 internal baseline。",
        },
        {
            "decision_option": "D_real_data_deferred",
            "recommended": True,
            "status": "active_constraint",
            "evidence": "internal branch 仍处于 COMSOL pilot / training-gate 前阶段",
            "next_action": "真实实验数据继续暂缓",
            "notes": "真实 internal block 需要独立 schema、ground truth 和采集 metadata。",
        },
    ]
    write_csv(args.matrix, rows)
    primary = [row for row in rows if row["recommended"] and row["decision_option"] != "D_real_data_deferred"][0]
    lines = [
        "21.1 internal defect pilot pack route decision",
        "",
        f"dataset_id: {manifest.get('dataset_id')}",
        f"status: {status}",
        f"successful_samples: {n_samples}",
        f"train_ready_candidate: {str(train_ready).lower()}",
        f"shape_groups_with_success: {len(shape_groups)}",
        f"burial_depth_groups_with_success: {len(burial_groups)}",
        f"size_groups_with_success: {len(size_groups)}",
        f"split_counts: {split_groups}",
        f"failed_checks: {failed_checks if failed_checks else 'none'}",
        "",
        f"unique_next_step: {primary['next_action']}",
        "current_baseline_update: false",
        "surface_rbc_mixed: false",
        "real_data_alignment: deferred",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""判定 22.9 richer-observation diagnostic pack 是否可进入 23.0 评估。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json"
VALIDATION = ROOT / "results/metrics/internal_richer_observation_pack_validation_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_richer_observation_pack_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_richer_observation_pack_route_decision_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="判定 richer-observation pack route。")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
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


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def run(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = {row["check_name"]: as_bool(row["pass"]) for row in read_csv(args.validation)}
    status = str(manifest.get("status", ""))
    success_rows = int(manifest.get("success_rows", 0))
    complete_bases = int(manifest.get("complete_base_count", 0))
    full_pack = status == "diagnostic_pack_generated" and success_rows == 180 and complete_bases == 30
    partial_pack = status == "partial_diagnostic_pack_generated" and success_rows >= 144 and complete_bases >= 24
    validation_passed = bool(manifest.get("validation_passed", False))
    paired_complete = checks.get("complete_base_count_min_24", False)
    can_enter_23 = validation_passed and (full_pack or partial_pack)
    needs_topup = not can_enter_23
    rows = [
        {
            "question": "pack 是否 full/partial 可用",
            "answer": "full" if full_pack else ("partial" if partial_pack else "blocked"),
            "evidence": f"status={status}; success_rows={success_rows}; complete_base_count={complete_bases}",
            "decision": "可用" if (full_pack or partial_pack) else "需要 top-up/fix",
        },
        {
            "question": "paired richer-observation variants 是否完整",
            "answer": "yes" if paired_complete else "no",
            "evidence": f"complete_base_count={complete_bases}; expected variants={manifest.get('observation_variants')}",
            "decision": "满足 23.0 paired diagnostic" if paired_complete else "先补齐 paired variants",
        },
        {
            "question": "scan_line / liftoff coverage 是否达标",
            "answer": "yes"
            if checks.get("scan_line_count_coverage", False) and checks.get("liftoff_level_coverage", False)
            else "no",
            "evidence": f"scan_line_counts={manifest.get('scan_line_counts')}; sensor_z={manifest.get('sensor_z_levels_m')}",
            "decision": "R0/R1/R2 可比较" if checks.get("scan_line_count_coverage", False) else "观测配置不完整",
        },
        {
            "question": "是否可进入 23.0 richer-observation evaluation",
            "answer": "yes" if can_enter_23 else "no",
            "evidence": f"validation_passed={validation_passed}; failed_blockers={manifest.get('failed_blockers')}",
            "decision": "进入 23.0 评估，不训练" if can_enter_23 else "先修复 pack",
        },
        {
            "question": "是否需要 top-up",
            "answer": "no" if not needs_topup else "yes",
            "evidence": f"full_pack={full_pack}; partial_pack={partial_pack}",
            "decision": "无需 top-up" if not needs_topup else "执行 richer-observation top-up 或重跑失败行",
        },
        {
            "question": "是否可训练或更新 baseline",
            "answer": "no",
            "evidence": "train_ready_candidate=false; baseline_ready=false",
            "decision": "训练和 CURRENT_BASELINE 更新继续暂缓",
        },
    ]
    write_csv(args.matrix, rows, ["question", "answer", "evidence", "decision"])
    next_step = "进入 23.0 richer-observation evaluation gate" if can_enter_23 else "先做 22.9 top-up/fix，不进入 23.0"
    lines = [
        "22.9 internal richer-observation pack route decision summary",
        f"dataset_id: {manifest.get('dataset_id')}",
        f"status: {status}",
        f"success_rows: {success_rows}",
        f"complete_base_count: {complete_bases}",
        f"observation_variants: {manifest.get('observation_variants')}",
        f"scan_line_counts: {manifest.get('scan_line_counts')}",
        f"sensor_z_levels_m: {manifest.get('sensor_z_levels_m')}",
        f"validation_passed: {str(validation_passed).lower()}",
        f"can_enter_23_0: {str(can_enter_23).lower()}",
        f"needs_topup: {str(needs_topup).lower()}",
        f"next_step: {next_step}",
        "",
        "判断：本 pack 只支持 richer-observation diagnostic/evaluation；训练、真实样品推理和 CURRENT_BASELINE 更新继续暂缓。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

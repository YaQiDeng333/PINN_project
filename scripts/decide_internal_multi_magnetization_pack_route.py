#!/usr/bin/env python
"""判定 23.4 internal multi-magnetization diagnostic pack 是否可进入 23.5。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_multi_magnetization_pack_v1.manifest.json"
VALIDATION = ROOT / "results/metrics/internal_multi_magnetization_pack_validation_metrics.csv"
ASSEMBLY = ROOT / "results/metrics/internal_multi_magnetization_pack_assembly_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_pack_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_multi_magnetization_pack_decision_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="判定 internal multi-magnetization diagnostic pack route。")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--assembly", type=Path, default=ASSEMBLY)
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
    validation_checks = {row["check_name"]: as_bool(row["pass"]) for row in read_csv(args.validation)}
    assembly_checks = {row["check_name"]: as_bool(row["pass"]) for row in read_csv(args.assembly)}

    status = str(manifest.get("status", ""))
    n = int(manifest.get("n_samples", 0))
    base_count = int(manifest.get("base_count", 0))
    complete = int(manifest.get("complete_base_count", 0))
    shape = manifest.get("assembled_delta_shape", [])
    source_je_changed = bool(manifest.get("source_je_changed", False))
    validation_passed = all(validation_checks.values()) if validation_checks else False
    assembly_passed = all(assembly_checks.values()) if assembly_checks else False
    full_pack = status == "diagnostic_pack_generated" and n == 60 and complete == 30 and shape == [60, 3, 2, 9, 201]
    partial_pack = status == "partial_diagnostic_pack_generated" and n >= 48 and complete >= 24
    can_enter_23_5 = validation_passed and assembly_passed and source_je_changed and (full_pack or partial_pack)

    rows = [
        {
            "question": "multi-magnetization COMSOL pack 是否生成成功",
            "answer": "yes" if (full_pack or partial_pack) else "no",
            "evidence": f"status={status}; n={n}; base_count={base_count}; complete_base_count={complete}",
            "decision": "可用" if (full_pack or partial_pack) else "先修复 COMSOL generation",
        },
        {
            "question": "magnetization/source 方向是否真实改变",
            "answer": "yes" if source_je_changed and validation_checks.get("source_je_not_metadata_only", False) else "no",
            "evidence": f"source_je_changed={source_je_changed}; validation={validation_checks.get('source_je_not_metadata_only')}",
            "decision": "满足 23.4 核心 gate" if source_je_changed else "不能进入 23.5",
        },
        {
            "question": "M1/M2 paired completeness 是否达标",
            "answer": "yes" if complete >= 24 else "no",
            "evidence": f"complete_base_count={complete}; target=30; fallback=24",
            "decision": "可配对 mag_x/mag_y" if complete >= 24 else "需要补齐 pack",
        },
        {
            "question": "assembled tensor shape 是否正确",
            "answer": "yes" if shape == [n, 3, 2, 9, 201] else "no",
            "evidence": f"assembled_delta_shape={shape}",
            "decision": "可供 23.5 loader/eval 使用" if shape == [n, 3, 2, 9, 201] else "修复 assembly",
        },
        {
            "question": "是否可进入 23.5 diagnostic evaluation",
            "answer": "yes" if can_enter_23_5 else "no",
            "evidence": f"validation_passed={validation_passed}; assembly_passed={assembly_passed}; full={full_pack}; partial={partial_pack}",
            "decision": "进入 23.5 multi-magnetization diagnostic evaluation" if can_enter_23_5 else "先修复 23.4 pack",
        },
        {
            "question": "是否可训练或更新 baseline",
            "answer": "no",
            "evidence": "train_ready_candidate=false; baseline_ready=false; CURRENT_BASELINE unchanged",
            "decision": "训练、真实 internal smoke 和 baseline 更新继续暂缓",
        },
    ]
    write_csv(args.matrix, rows, ["question", "answer", "evidence", "decision"])

    next_step = "进入 23.5 internal multi-magnetization diagnostic evaluation" if can_enter_23_5 else "先修复 23.4 multi-magnetization pack"
    lines = [
        "23.4 internal multi-magnetization route decision summary",
        "",
        f"dataset_id: {manifest.get('dataset_id')}",
        f"status: {status}",
        f"n_samples: {n}",
        f"base_count: {base_count}",
        f"complete_base_count: {complete}",
        f"assembled_delta_shape: {shape}",
        f"source_je_changed: {str(source_je_changed).lower()}",
        f"validation_passed: {str(validation_passed).lower()}",
        f"assembly_passed: {str(assembly_passed).lower()}",
        f"can_enter_23_5: {str(can_enter_23_5).lower()}",
        "train_ready_candidate: false",
        "baseline_ready: false",
        f"next_step: {next_step}",
        "",
        "判断：23.4 只完成 multi-magnetization diagnostic pack gate；internal branch 仍不是 baseline，下一步只进入 23.5 diagnostic evaluation。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if can_enter_23_5 else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

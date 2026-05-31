#!/usr/bin/env python
"""判定 23.2b dual-direction diagnostic pack 是否可进入 23.3。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_multi_scan_direction_pack_v1.manifest.json"
VALIDATION = ROOT / "results/metrics/internal_multi_scan_direction_pack_validation_metrics.csv"
ASSEMBLY = ROOT / "results/metrics/internal_multi_scan_direction_pack_assembly_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_multi_scan_direction_pack_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_multi_scan_direction_pack_decision_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="判定 internal multi-scan-direction diagnostic pack route。")
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
    checks = {row["check_name"]: as_bool(row["pass"]) for row in read_csv(args.validation)}
    status = str(manifest.get("status", ""))
    n = int(manifest.get("n_samples", 0))
    base_count = int(manifest.get("base_count", 0))
    complete = int(manifest.get("complete_base_count", 0))
    shape = manifest.get("assembled_delta_shape", [])
    full_pack = status == "diagnostic_pack_generated" and n == 60 and complete == 30 and shape == [60, 3, 2, 9, 201]
    partial_pack = status == "partial_diagnostic_pack_generated" and n >= 48 and complete >= 24
    validation_passed = all(checks.values()) if checks else False
    direction_ok = checks.get("direction_aware_y_scan_coordinate_check", False)
    can_enter_23_3 = validation_passed and direction_ok and (full_pack or partial_pack)
    needs_topup = not can_enter_23_3

    rows = [
        {
            "question": "top-up 是否生成成功",
            "answer": "yes" if (full_pack or partial_pack) else "no",
            "evidence": f"status={status}; n={n}; base_count={base_count}; complete_base_count={complete}",
            "decision": "可用" if (full_pack or partial_pack) else "先修复 y_scan generation",
        },
        {
            "question": "y_scan 是否是真正方向化坐标",
            "answer": "yes" if direction_ok else "no",
            "evidence": f"direction_check={direction_ok}; path_axis_y={checks.get('path_axis_y')}; line_axis_x={checks.get('line_axis_x')}",
            "decision": "满足 23.2b 核心 gate" if direction_ok else "不能进入 23.3，需修 generator",
        },
        {
            "question": "D1/D2 paired completeness 是否达标",
            "answer": "yes" if complete >= 24 else "no",
            "evidence": f"complete_base_count={complete}; target=30; fallback=24",
            "decision": "可配对 existing x_scan" if complete >= 24 else "需要 top-up",
        },
        {
            "question": "assembled tensor shape 是否正确",
            "answer": "yes" if shape == [n, 3, 2, 9, 201] else "no",
            "evidence": f"assembled_delta_shape={shape}",
            "decision": "可供 23.3 loader/eval 使用" if shape == [n, 3, 2, 9, 201] else "修 assembly",
        },
        {
            "question": "是否可进入 23.3 diagnostic evaluation",
            "answer": "yes" if can_enter_23_3 else "no",
            "evidence": f"validation_passed={validation_passed}; direction_ok={direction_ok}; full={full_pack}; partial={partial_pack}",
            "decision": "进入 23.3，不训练" if can_enter_23_3 else "先修复/补齐 pack",
        },
        {
            "question": "是否可训练或更新 baseline",
            "answer": "no",
            "evidence": "train_ready_candidate=false; baseline_ready=false; CURRENT_BASELINE unchanged",
            "decision": "训练、真实 internal smoke 和 baseline 更新继续暂缓",
        },
    ]
    write_csv(args.matrix, rows, ["question", "answer", "evidence", "decision"])
    next_step = "进入 23.3 internal multi-scan-direction diagnostic evaluation" if can_enter_23_3 else "先修复 23.2b pack，不进入 23.3"
    lines = [
        "23.2b internal multi-scan-direction route decision summary",
        "",
        f"dataset_id: {manifest.get('dataset_id')}",
        f"status: {status}",
        f"n_samples: {n}",
        f"base_count: {base_count}",
        f"complete_base_count: {complete}",
        f"assembled_delta_shape: {shape}",
        f"validation_passed: {str(validation_passed).lower()}",
        f"direction_aware_coordinate_ok: {str(direction_ok).lower()}",
        f"can_enter_23_3: {str(can_enter_23_3).lower()}",
        f"needs_topup: {str(needs_topup).lower()}",
        f"next_step: {next_step}",
        "",
        "判断: 23.2b 只完成 dual-direction diagnostic pack gate；internal branch 仍不是 baseline。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if can_enter_23_3 else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

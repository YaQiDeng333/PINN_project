#!/usr/bin/env python
"""判定 21.3b internal defect v2_240 是否可进入 21.4 training gate。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json"
VALIDATION = ROOT / "results/metrics/internal_defect_dataset_v2_validation_metrics.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_dataset_v2_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_dataset_v2_route_decision_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="判定 internal defect v2_240 route。")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--overwrite", action="store_true")
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
    n_samples = int(manifest.get("n_samples", 0))
    status = str(manifest.get("status", ""))
    train_ready = bool(manifest.get("train_ready_candidate", False))
    rows = [
        {
            "question": "top-up 是否成功",
            "answer": "yes" if checks.get("topup_success_target", False) else "no",
            "evidence": f"topup_success_target={checks.get('topup_success_target', False)}",
            "decision": "可用" if checks.get("topup_success_target", False) else "需要 second top-up",
        },
        {
            "question": "assembled N / split 是否达标",
            "answer": "yes" if checks.get("assembled_n_240", False) and checks.get("split_counts_160_40_40", False) else "no",
            "evidence": f"n={n_samples}, split_counts={manifest.get('split_counts')}",
            "decision": "达标" if n_samples == 240 else "未达标",
        },
        {
            "question": "shape/burial/size/aspect coverage 是否达标",
            "answer": "yes"
            if all(
                checks.get(name, False)
                for name in [
                    "shape_counts_balanced",
                    "burial_counts_balanced",
                    "size_counts_balanced",
                    "each_split_has_all_shapes",
                    "each_split_has_all_burials",
                    "each_split_has_all_sizes",
                    "ellipsoid_cuboid_aspect_each_split",
                ]
            )
            else "no",
            "evidence": f"shape={manifest.get('shape_counts')}; burial={manifest.get('burial_depth_counts')}; size={manifest.get('size_counts')}; aspect={manifest.get('aspect_counts')}",
            "decision": "解决 21.2 split blocker" if train_ready else "coverage 仍需修复",
        },
        {
            "question": "是否 train_ready_candidate",
            "answer": "yes" if train_ready else "no",
            "evidence": f"status={status}, train_ready_candidate={train_ready}, baseline_ready={manifest.get('baseline_ready')}",
            "decision": "进入 21.4 internal v2_240 training gate" if train_ready else "先 top-up/修复数据包",
        },
        {
            "question": "是否需要 second top-up",
            "answer": "no" if train_ready else "yes",
            "evidence": f"failed_blockers={manifest.get('failed_blockers', [])}",
            "decision": "不需要 second top-up" if train_ready else "需要 second top-up 或修复 blocker",
        },
    ]
    write_csv(args.matrix, rows, ["question", "answer", "evidence", "decision"])
    recommendation = "进入 21.4 internal v2_240 training gate" if train_ready else "先执行 second top-up / blocker fix，不进入训练"
    lines = [
        "21.3b internal defect v2_240 route decision summary",
        f"dataset_id: {manifest.get('dataset_id')}",
        f"status: {status}",
        f"train_ready_candidate: {str(train_ready).lower()}",
        f"n_samples: {n_samples}",
        f"split_counts: {manifest.get('split_counts')}",
        f"shape_counts: {manifest.get('shape_counts')}",
        f"burial_depth_counts: {manifest.get('burial_depth_counts')}",
        f"size_counts: {manifest.get('size_counts')}",
        f"aspect_counts: {manifest.get('aspect_counts')}",
        f"next_step: {recommendation}",
        "",
        "判断：v2_240 只属于 internal/buried defect 独立分支，baseline_ready=false，不替换 CURRENT_BASELINE，也不混入 surface RBC baseline。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

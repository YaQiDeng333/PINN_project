#!/usr/bin/env python
"""Decide the route for the surface RBC NLS full-compatible framework."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_MANIFEST = ROOT / "results/manifests/surface_rbc_nls_full_compatible_feature_manifest.json"
INPUT_VALIDATION = ROOT / "results/metrics/surface_rbc_nls_full_compatible_input_validation.csv"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_full_compatible_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_rbc_nls_full_compatible_decision_matrix.csv"

FIELDS = ["question", "answer", "evidence", "decision"]


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


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=FEATURE_MANIFEST)
    parser.add_argument("--input-validation", type=Path, default=INPUT_VALIDATION)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation_rows = {row["check_name"]: row for row in read_csv(args.input_validation)}
    scan_line_count = int(manifest.get("scan_line_count", 0))
    full_ready = bool(manifest.get("full_feature_ready", False))
    degraded_reason = str(manifest.get("degraded_mode_reason", ""))
    full_candidate = bool(manifest.get("full_candidate_mode", False))
    exact_piao = bool(manifest.get("exact_piao_full", False))
    compatible = bool(manifest.get("piao_full_compatible", False))
    sensor_x_count = int(manifest.get("sensor_x_count", 0))

    rows = [
        {
            "question": "Is current v3_240 full-ready?",
            "answer": "no" if not full_ready else "yes",
            "evidence": f"scan_line_count={scan_line_count}; full_feature_ready={full_ready}; reason={degraded_reason}",
            "decision": "current v3_240 can only be degraded-compatible" if not full_ready else "may run full-compatible mode",
        },
        {
            "question": "What data is required for true full-compatible mode?",
            "answer": "Bx/By/Bz ROI matrices with M>=5, recommended M>=9",
            "evidence": f"current axes={manifest.get('axis_names')}; current M={scan_line_count}; current K={sensor_x_count}",
            "decision": "collect or generate surface RBC y-line ROI pack before claiming full mode",
        },
        {
            "question": "Is a surface richer y-line pack needed?",
            "answer": "yes",
            "evidence": "internal richer-observation pack has 5/9-line references, but it is not surface RBC full data",
            "decision": "use internal richer pack as interface reference only; create/validate surface RBC richer pack separately",
        },
        {
            "question": "Is NLS-full-compatible only a future interface for now?",
            "answer": "yes" if not full_ready else "partial",
            "evidence": f"exact_piao_full={exact_piao}; piao_full_compatible={compatible}; full_candidate_mode={full_candidate}",
            "decision": "keep as schema/extractor/validator framework until full ROI and equations are validated",
        },
        {
            "question": "Does NLS-full-compatible replace NLS-lite?",
            "answer": "no",
            "evidence": "degraded-compatible features coexist with current 3-line NLS-lite/Piao-inspired branch",
            "decision": "run in parallel; do not replace NLS-lite or CURRENT_BASELINE",
        },
    ]
    full_check = validation_rows.get("scan_line_count_min_for_full", {})
    rows.append(
        {
            "question": "Input adequacy gate present?",
            "answer": "yes",
            "evidence": f"scan_line_count_min_for_full={full_check.get('pass')}; metrics={args.input_validation}",
            "decision": "gate blocks full-ready claim on 3-line input",
        }
    )
    write_csv(args.matrix, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface RBC NLS full-compatible route decision summary",
                "",
                f"dataset_id: {manifest.get('dataset_id')}",
                f"feature_schema_version: {manifest.get('feature_schema_version')}",
                f"scan_line_count: {scan_line_count}",
                f"sensor_x_count: {sensor_x_count}",
                f"full_feature_ready: {str(full_ready).lower()}",
                f"full_candidate_mode: {str(full_candidate).lower()}",
                f"degraded_mode: {str(bool(manifest.get('degraded_mode'))).lower()}",
                f"degraded_mode_reason: {degraded_reason}",
                f"piao_full_compatible: {str(compatible).lower()}",
                f"exact_piao_full: {str(exact_piao).lower()}",
                "",
                "decision_1_current_v3_240: degraded-compatible only; M=3 is below the M>=5 full-compatible minimum.",
                "decision_2_true_full_data_needed: Bx/By/Bz surface RBC ROI matrices with validated axis order, sensor_x, scan_line_y, no missing values, M>=5 minimum and M>=9 recommended.",
                "decision_3_richer_y_line_pack: needed for surface RBC; existing internal richer-observation 5/9-line pack is only an interface reference.",
                "decision_4_framework_role: NLS-full-compatible remains a future real-experiment/richer-observation interface until full ROI and exact equations are validated.",
                "decision_5_nls_lite_relation: keep parallel with NLS-lite/Piao-inspired 3-line branch; do not replace it.",
                "",
                "actions: no COMSOL, no training, no data/NPZ writing, no CURRENT_BASELINE update.",
                f"decision_matrix: {args.matrix}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

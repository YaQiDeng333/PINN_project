#!/usr/bin/env python
"""Summarize the 21.7 internal defect benchmark candidate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, load_dataset, split_indices, write_csv


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SUMMARY = ROOT / "results/summaries/internal_defect_benchmark_candidate_summary.txt"
COMPARISON = ROOT / "results/metrics/internal_defect_benchmark_candidate_comparison.csv"
RERUN_VS = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_vs_reference.csv"
RERUN_GROUP = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_group_summary.csv"


FIELDS = [
    "model",
    "source",
    "selected",
    "role",
    "split",
    "sample_count",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize internal defect benchmark candidate.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    parser.add_argument("--rerun-vs", type=Path, default=RERUN_VS)
    parser.add_argument("--rerun-group", type=Path, default=RERUN_GROUP)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def comparison_rows(vs_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    role_map = {
        "21.4_neural_reference": "previous neural benchmark candidate",
        "21.4_feature_baseline": "delta_b-derived feature comparator",
        "21.7_B2_formal_rerun": "21.7 B2 benchmark candidate rerun",
    }
    rows: list[dict[str, Any]] = []
    for row in vs_rows:
        rows.append(
            {
                "model": row.get("model", ""),
                "source": row.get("source", ""),
                "selected": row.get("selected", ""),
                "role": role_map.get(row.get("source", ""), ""),
                "split": row.get("split", ""),
                "sample_count": row.get("sample_count", ""),
                "total_normalized_mae": row.get("total_normalized_mae", ""),
                "L_mae_mm": row.get("L_mae_mm", ""),
                "W_mae_mm": row.get("W_mae_mm", ""),
                "D_mae_mm": row.get("D_mae_mm", ""),
                "burial_depth_mae_mm": row.get("burial_depth_mae_mm", ""),
                "center_xyz_mae_mm": row.get("center_xyz_mae_mm", ""),
                "shape_accuracy": row.get("shape_accuracy", ""),
                "shape_macro_f1": row.get("shape_macro_f1", ""),
                "notes": "not CURRENT_BASELINE; internal branch only",
            }
        )
    return rows


def selected_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("selected") == "True":
            return row
    return {}


def group_highlights(group_rows: list[dict[str, str]], group_field: str) -> str:
    rows = [row for row in group_rows if row.get("split") == "test" and row.get("group_field") == group_field]
    if not rows:
        return "not available"
    parts = []
    for row in sorted(rows, key=lambda item: item.get("group_value", "")):
        parts.append(
            f"{row.get('group_value')} total={safe_float(row.get('total_normalized_mae')):.3f}, "
            f"burial={safe_float(row.get('burial_depth_mae_mm')):.3f}mm, "
            f"shape_acc={safe_float(row.get('shape_accuracy')):.3f}"
        )
    return "; ".join(parts)


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    vs_rows = read_csv(args.rerun_vs)
    group_rows = read_csv(args.rerun_group)
    rows = comparison_rows(vs_rows)
    write_csv(args.comparison, rows, FIELDS)
    selected = selected_row(vs_rows)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.7 internal defect benchmark candidate report",
                f"dataset_id: {args.dataset_id}",
                f"dataset_identity: N={dataset.delta_b.shape[0]}; split={splits['train'].size}/{splits['val'].size}/{splits['test'].size}; "
                f"shape_counts={dataset.manifest.get('shape_counts')}; burial_counts={dataset.manifest.get('burial_depth_counts')}; "
                f"size_counts={dataset.manifest.get('size_counts')}; aspect_counts={dataset.manifest.get('aspect_counts')}",
                "task_definition: Bx/By/Bz delta_b -> internal cavity L/W/D, burial_depth, center_xyz, and shape_type.",
                "model_chain: delta_b (N,3,3,201) -> Conv1D encoder (N,9,201) plus delta_b-derived feature MLP -> B2 multitask heads.",
                "input_boundary: no true shape_type, burial bin, size/aspect, split, or sample_id as model input.",
                f"feature_baseline: total={safe_float(vs_rows[1].get('total_normalized_mae')):.6f}; "
                f"L/W/D={safe_float(vs_rows[1].get('L_mae_mm')):.3f}/{safe_float(vs_rows[1].get('W_mae_mm')):.3f}/{safe_float(vs_rows[1].get('D_mae_mm')):.3f} mm; "
                f"burial={safe_float(vs_rows[1].get('burial_depth_mae_mm')):.3f} mm; center={safe_float(vs_rows[1].get('center_xyz_mae_mm')):.3f} mm; "
                f"shape acc/F1={safe_float(vs_rows[1].get('shape_accuracy')):.6f}/{safe_float(vs_rows[1].get('shape_macro_f1')):.6f}.",
                f"21.4_neural_reference: total={safe_float(vs_rows[0].get('total_normalized_mae')):.6f}; "
                f"L/W/D={safe_float(vs_rows[0].get('L_mae_mm')):.3f}/{safe_float(vs_rows[0].get('W_mae_mm')):.3f}/{safe_float(vs_rows[0].get('D_mae_mm')):.3f} mm; "
                f"burial={safe_float(vs_rows[0].get('burial_depth_mae_mm')):.3f} mm; center={safe_float(vs_rows[0].get('center_xyz_mae_mm')):.3f} mm; "
                f"shape acc/F1={safe_float(vs_rows[0].get('shape_accuracy')):.6f}/{safe_float(vs_rows[0].get('shape_macro_f1')):.6f}.",
                f"21.7_B2_candidate: total={safe_float(selected.get('total_normalized_mae')):.6f}; "
                f"L/W/D={safe_float(selected.get('L_mae_mm')):.3f}/{safe_float(selected.get('W_mae_mm')):.3f}/{safe_float(selected.get('D_mae_mm')):.3f} mm; "
                f"burial={safe_float(selected.get('burial_depth_mae_mm')):.3f} mm; center={safe_float(selected.get('center_xyz_mae_mm')):.3f} mm; "
                f"shape acc/F1={safe_float(selected.get('shape_accuracy')):.6f}/{safe_float(selected.get('shape_macro_f1')):.6f}.",
                f"by_shape_type: {group_highlights(group_rows, 'shape_type')}",
                f"by_burial_depth: {group_highlights(group_rows, 'burial_depth_level')}",
                f"by_size: {group_highlights(group_rows, 'size_level')}",
                f"by_aspect: {group_highlights(group_rows, 'aspect_bin')}",
                "limitations: COMSOL simulation domain only; internal shapes currently limited to sphere/ellipsoid/cuboid; no real experimental validation; not CURRENT_BASELINE.",
                "candidate_status: internal benchmark candidate, not baseline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

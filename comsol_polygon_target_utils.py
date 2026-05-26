"""Validate and export polygon targets already embedded in COMSOL V3 NPZ files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_NPZ_FIELDS = [
    "masks",
    "x",
    "y",
    "polygon_vertices_raw",
    "polygon_vertices_norm",
    "polygon_vertex_mask",
    "polygon_presence",
    "type_targets",
    "polygon_type_vocab",
    "component_counts",
]


def _usage() -> str:
    return (
        "Usage: python comsol_polygon_target_utils.py --npz-path converted.npz "
        "--polygon-params-csv polygon_params.csv --output-dir out"
    )


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_polygon_params(rows: list[dict], n_samples: int, presence: np.ndarray) -> dict:
    if not rows:
        raise ValueError("polygon_params.csv contains no rows.")
    required = {
        "sample_index",
        "split",
        "component_index",
        "presence",
        "hard_case_type",
        "component_type",
        "vertex_ordering",
        "source_geometry_type",
        "is_true_rotated",
        "is_true_multi_component",
    }
    for i in range(4):
        required.update({f"raw_x{i}", f"raw_y{i}", f"norm_x{i}", f"norm_y{i}"})
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"polygon_params.csv missing columns: {sorted(missing)}")
    by_sample = Counter(int(float(row["sample_index"])) for row in rows if float(row.get("presence", 1.0)) > 0.5)
    if sorted(by_sample) != list(range(n_samples)):
        raise ValueError("polygon_params.csv does not cover every sample_index 0..N-1.")
    expected_by_sample = {idx: int((presence[idx] > 0.5).sum()) for idx in range(n_samples)}
    for sample_idx, expected_count in expected_by_sample.items():
        if by_sample[sample_idx] != expected_count:
            raise ValueError(
                f"sample={sample_idx} polygon row count {by_sample[sample_idx]} does not match presence {expected_count}."
            )
    return {
        "true_rotated_rows": sum(str(row["is_true_rotated"]).lower() == "true" for row in rows),
        "true_multi_rows": sum(str(row["is_true_multi_component"]).lower() == "true" for row in rows),
        "component_rows": len(rows),
    }


def export_polygon_targets(npz_path: Path, polygon_params_csv: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_NPZ_FIELDS if name not in data.files]
        if missing:
            raise ValueError(f"NPZ missing required polygon fields: {missing}")
        masks = data["masks"].astype(np.float32)
        vertices_raw = data["polygon_vertices_raw"].astype(np.float32)
        vertices_norm = data["polygon_vertices_norm"].astype(np.float32)
        vertex_mask = data["polygon_vertex_mask"].astype(np.float32)
        presence = data["polygon_presence"].astype(np.float32)
        type_targets = data["type_targets"].astype(np.int64)
        type_vocab = data["polygon_type_vocab"]
        component_counts = data["component_counts"].astype(np.int64)
        x_norm = data["x"].astype(np.float32)
        y_norm = data["y"].astype(np.float32)
    if vertices_norm.shape != vertices_raw.shape or vertices_norm.ndim != 4 or vertices_norm.shape[-1] != 2:
        raise ValueError("polygon vertex arrays must have matching shape [N,K,V,2].")
    if vertex_mask.shape != vertices_norm.shape[:3]:
        raise ValueError("polygon_vertex_mask shape does not match vertices.")
    if presence.shape != vertices_norm.shape[:2]:
        raise ValueError("polygon_presence shape does not match vertices.")
    if masks.shape[0] != vertices_norm.shape[0]:
        raise ValueError("mask sample count does not match polygon target sample count.")
    rows = _read_csv(polygon_params_csv)
    param_stats = _validate_polygon_params(rows, masks.shape[0], presence)
    sample_indices = np.arange(masks.shape[0], dtype=np.int64)
    np.savez_compressed(
        output_dir / "polygon_targets.npz",
        polygon_vertices_raw=vertices_raw,
        polygon_vertices_norm=vertices_norm,
        polygon_vertex_mask=vertex_mask,
        presence_targets=presence,
        type_targets=type_targets,
        type_vocab=type_vocab,
        component_counts=component_counts,
        sample_indices=sample_indices,
        x_norm=x_norm,
        y_norm=y_norm,
        vertex_ordering=np.array("clockwise_top_left", dtype="U64"),
        max_components=np.array(vertices_norm.shape[1], dtype=np.int64),
        max_vertices=np.array(vertices_norm.shape[2], dtype=np.int64),
    )
    counts = Counter(int(x) for x in component_counts)
    summary = [
        "# COMSOL polygon embedded target summary",
        "",
        f"- samples: `{masks.shape[0]}`",
        f"- max_components: `{vertices_norm.shape[1]}`",
        f"- max_vertices: `{vertices_norm.shape[2]}`",
        f"- type_vocab: `{', '.join(str(x) for x in type_vocab)}`",
        f"- polygon_params component rows: `{param_stats['component_rows']}`",
        f"- true rotated polygon rows: `{param_stats['true_rotated_rows']}`",
        f"- true multi-component polygon rows: `{param_stats['true_multi_rows']}`",
        "",
        "## Component count distribution",
        "",
    ]
    summary.extend(f"- `{count}` components: `{num}` samples" for count, num in sorted(counts.items()))
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {
        "samples": int(masks.shape[0]),
        "max_components": int(vertices_norm.shape[1]),
        "max_vertices": int(vertices_norm.shape[2]),
        "component_rows": int(param_stats["component_rows"]),
        "true_rotated_rows": int(param_stats["true_rotated_rows"]),
        "true_multi_rows": int(param_stats["true_multi_rows"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path")
    parser.add_argument("--polygon-params-csv")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if not args.npz_path or not args.polygon_params_csv or not args.output_dir:
        print(_usage())
        return 0
    stats = export_polygon_targets(Path(args.npz_path), Path(args.polygon_params_csv), Path(args.output_dir))
    print(f"Saved embedded polygon targets to {args.output_dir}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

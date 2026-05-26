"""Build fixed-corner polygon targets for COMSOL V3 geometry."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


POLYGON_COLUMNS = [
    "sample_index",
    "split",
    "component_slot",
    "component_id",
    "component_type",
    "vertex_index",
    "x_raw",
    "y_raw",
    "x_norm",
    "y_norm",
    "ordering",
    "geometry_feature_tag",
    "selection_name",
    "hard_case_type",
    "component_count",
    "union_selection_name",
    "true_rotated_geometry",
    "true_multi_component_geometry",
]
AUX_SCHEMA = [
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "depth_or_shape_param",
    "rotation_angle",
]


def _usage() -> str:
    return (
        "Usage: python comsol_polygon_targets.py --npz-path data.npz "
        "--polygon-params-csv polygon_params.csv --output-dir out "
        "[--max-components 3 --max-vertices 4]"
    )


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _signed_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _canonical_order(raw_vertices: np.ndarray, norm_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = norm_vertices.mean(axis=0)
    angles = np.arctan2(norm_vertices[:, 1] - center[1], norm_vertices[:, 0] - center[0])
    order = np.argsort(-angles)
    raw = raw_vertices[order]
    norm = norm_vertices[order]
    if _signed_area(norm) > 0:
        raw = raw[::-1]
        norm = norm[::-1]
    start = min(range(norm.shape[0]), key=lambda idx: (float(norm[idx, 1]), float(norm[idx, 0])))
    raw = np.concatenate([raw[start:], raw[:start]], axis=0)
    norm = np.concatenate([norm[start:], norm[:start]], axis=0)
    return raw, norm


def _component_key(row: dict) -> tuple[int, int]:
    return int(float(row["sample_index"])), int(float(row["component_slot"]))


def _load_components(rows: list[dict], max_vertices: int) -> dict[int, list[dict]]:
    grouped_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[_component_key(row)].append(row)
    components_by_sample: dict[int, list[dict]] = defaultdict(list)
    for (sample_index, source_slot), items in grouped_rows.items():
        if len(items) != max_vertices:
            raise ValueError(
                f"sample={sample_index} component_slot={source_slot} has {len(items)} vertices; expected {max_vertices}."
            )
        items = sorted(items, key=lambda row: int(float(row["vertex_index"])))
        raw = np.array([[float(row["x_raw"]), float(row["y_raw"])] for row in items], dtype=np.float32)
        norm = np.array([[float(row["x_norm"]), float(row["y_norm"])] for row in items], dtype=np.float32)
        raw, norm = _canonical_order(raw, norm)
        centroid = norm.mean(axis=0)
        components_by_sample[sample_index].append(
            {
                "source_component_slot": source_slot,
                "component_id": items[0]["component_id"],
                "component_type": items[0]["component_type"],
                "hard_case_type": items[0]["hard_case_type"],
                "component_count": int(float(items[0]["component_count"])),
                "union_selection_name": items[0]["union_selection_name"],
                "geometry_feature_tag": items[0]["geometry_feature_tag"],
                "selection_name": items[0]["selection_name"],
                "true_rotated_geometry": items[0]["true_rotated_geometry"],
                "true_multi_component_geometry": items[0]["true_multi_component_geometry"],
                "raw": raw,
                "norm": norm,
                "centroid": centroid,
            }
        )
    for sample_index in list(components_by_sample):
        components_by_sample[sample_index] = sorted(
            components_by_sample[sample_index],
            key=lambda comp: (float(comp["centroid"][0]), float(comp["centroid"][1])),
        )
    return components_by_sample


def _build_targets(rows: list[dict], max_components: int, max_vertices: int, x_raw, y_raw) -> dict:
    components_by_sample = _load_components(rows, max_vertices)
    sample_indices = np.array(sorted(components_by_sample), dtype=np.int64)
    type_vocab = np.array(
        sorted({comp["component_type"] for comps in components_by_sample.values() for comp in comps}),
        dtype="U128",
    )
    type_to_index = {str(name): idx for idx, name in enumerate(type_vocab)}
    n = len(sample_indices)
    polygon_raw = np.zeros((n, max_components, max_vertices, 2), dtype=np.float32)
    polygon_norm = np.zeros_like(polygon_raw)
    vertex_mask = np.zeros((n, max_components, max_vertices), dtype=np.float32)
    polygon_valid = np.zeros((n, max_components), dtype=np.float32)
    presence = np.zeros((n, max_components), dtype=np.float32)
    type_targets = np.full((n, max_components), -1, dtype=np.int64)
    source_component_slots = np.full((n, max_components), -1, dtype=np.int64)
    component_counts = np.zeros(n, dtype=np.int64)
    aux = np.zeros((n, max_components, len(AUX_SCHEMA)), dtype=np.float32)
    hard_case_type = np.array(["" for _ in range(n)], dtype="U96")
    for i, sample_index in enumerate(sample_indices):
        comps = components_by_sample[int(sample_index)]
        component_counts[i] = len(comps)
        if len(comps) > max_components:
            raise ValueError(f"sample={sample_index} has {len(comps)} components; max_components={max_components}.")
        if comps:
            hard_case_type[i] = comps[0]["hard_case_type"]
        for slot, comp in enumerate(comps):
            polygon_raw[i, slot] = comp["raw"]
            polygon_norm[i, slot] = comp["norm"]
            vertex_mask[i, slot] = 1.0
            polygon_valid[i, slot] = 1.0
            presence[i, slot] = 1.0
            type_targets[i, slot] = type_to_index[comp["component_type"]]
            source_component_slots[i, slot] = comp["source_component_slot"]
            xmin, ymin = comp["norm"].min(axis=0)
            xmax, ymax = comp["norm"].max(axis=0)
            center = comp["norm"].mean(axis=0)
            aux[i, slot] = [
                center[0],
                center[1],
                xmax - xmin,
                ymax - ymin,
                0.0,
                0.0,
            ]
    x_norm = (x_raw.astype(np.float64) - 2250.0) * (0.08 / 4500.0)
    y_norm = (y_raw.astype(np.float64) - 1500.0) * (0.02 / 3000.0)
    return {
        "sample_indices": sample_indices,
        "polygon_vertices_raw": polygon_raw,
        "polygon_vertices_norm": polygon_norm,
        "polygon_vertex_mask": vertex_mask,
        "polygon_valid": polygon_valid,
        "presence_targets": presence,
        "type_targets": type_targets,
        "type_vocab": type_vocab,
        "component_counts": component_counts,
        "source_component_slots": source_component_slots,
        "aux_continuous_targets_norm": aux,
        "aux_target_schema": np.array(AUX_SCHEMA, dtype="U64"),
        "polygon_schema": np.array(["x", "y"], dtype="U16"),
        "max_vertices": np.array(max_vertices, dtype=np.int64),
        "vertex_ordering": np.array("clockwise_start_min_y_then_min_x_in_normalized_space", dtype="U96"),
        "hard_case_type": hard_case_type,
        "x_norm": x_norm.astype(np.float32),
        "y_norm": y_norm.astype(np.float32),
        "normalization_metadata_json": np.array(
            '{"x_scale":1.7777777777777777e-05,"y_scale":6.666666666666667e-06,"x_origin":2250.0,"y_origin":1500.0}',
            dtype="U160",
        ),
    }


def _write_preview(path: Path, targets: dict) -> None:
    rows = []
    for i, sample_index in enumerate(targets["sample_indices"]):
        for slot in range(targets["presence_targets"].shape[1]):
            if targets["presence_targets"][i, slot] <= 0.5:
                continue
            for vertex in range(targets["polygon_vertices_norm"].shape[2]):
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "component_slot": slot,
                        "vertex_index": vertex,
                        "x_norm": float(targets["polygon_vertices_norm"][i, slot, vertex, 0]),
                        "y_norm": float(targets["polygon_vertices_norm"][i, slot, vertex, 1]),
                        "x_raw": float(targets["polygon_vertices_raw"][i, slot, vertex, 0]),
                        "y_raw": float(targets["polygon_vertices_raw"][i, slot, vertex, 1]),
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(Path(args.polygon_params_csv))
    with np.load(args.npz_path, allow_pickle=True) as data:
        x_raw = data["x"].astype(np.float32)
        y_raw = data["y"].astype(np.float32)
        n_samples = int(data["masks"].shape[0])
    targets = _build_targets(rows, args.max_components, args.max_vertices, x_raw, y_raw)
    if len(targets["sample_indices"]) != n_samples:
        raise ValueError("polygon target samples do not match NPZ masks.")
    np.savez_compressed(output_dir / "polygon_targets.npz", **targets)
    _write_preview(output_dir / "polygon_targets_preview.csv", targets)
    counts = Counter(int(x) for x in targets["component_counts"])
    summary = [
        "# COMSOL polygon target summary",
        "",
        f"- samples: `{len(targets['sample_indices'])}`",
        f"- max_components: `{args.max_components}`",
        f"- max_vertices: `{args.max_vertices}`",
        f"- type_vocab: `{', '.join(str(x) for x in targets['type_vocab'])}`",
        f"- vertex_ordering: `{str(targets['vertex_ordering'])}`",
        "- raw vertices are preserved for COMSOL audit.",
        "- normalized vertices are the primary polygon target for V2-compatible route.",
        "",
        "## Component count distribution",
        "",
    ]
    summary.extend(f"- `{count}` components: `{num}` samples" for count, num in sorted(counts.items()))
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Saved polygon targets to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--npz-path")
    parser.add_argument("--polygon-params-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--max-vertices", type=int, default=4)
    args = parser.parse_args()
    if not args.npz_path or not args.polygon_params_csv or not args.output_dir:
        print(_usage())
        return
    run(args)


if __name__ == "__main__":
    main()

"""Smoke test for embedded COMSOL polygon target export utility."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components
from comsol_polygon_target_utils import export_polygon_targets


def _write_polygon_params(path: Path) -> None:
    columns = [
        "sample_index",
        "split",
        "component_index",
        "presence",
        "hard_case_type",
        "component_type",
        "vertex_ordering",
        "raw_x0",
        "raw_y0",
        "raw_x1",
        "raw_y1",
        "raw_x2",
        "raw_y2",
        "raw_x3",
        "raw_y3",
        "norm_x0",
        "norm_y0",
        "norm_x1",
        "norm_y1",
        "norm_x2",
        "norm_y2",
        "norm_x3",
        "norm_y3",
        "source_geometry_type",
        "is_true_rotated",
        "is_true_multi_component",
    ]
    row = {
        "sample_index": 0,
        "split": "smoke",
        "component_index": 0,
        "presence": 1,
        "hard_case_type": "mock",
        "component_type": "rotated_rect",
        "vertex_ordering": "clockwise_top_left",
        "raw_x0": -1,
        "raw_y0": -1,
        "raw_x1": 1,
        "raw_y1": -1,
        "raw_x2": 1,
        "raw_y2": 1,
        "raw_x3": -1,
        "raw_y3": 1,
        "norm_x0": -1,
        "norm_y0": -1,
        "norm_x1": 1,
        "norm_y1": -1,
        "norm_x2": 1,
        "norm_y2": 1,
        "norm_x3": -1,
        "norm_y3": 1,
        "source_geometry_type": "rotated_rect",
        "is_true_rotated": "true",
        "is_true_multi_component": "false",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        x = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
        y = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
        vertices = np.zeros((1, 3, 4, 2), dtype=np.float32)
        vertices[0, 0] = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
        vertex_mask = np.zeros((1, 3, 4), dtype=np.float32)
        vertex_mask[0, 0] = 1.0
        presence = np.zeros((1, 3), dtype=np.float32)
        presence[0, 0] = 1.0
        masks = rasterize_polygon_components(vertices, vertex_mask, presence, x, y).astype(np.float32)
        npz_path = root / "converted.npz"
        np.savez_compressed(
            npz_path,
            masks=masks,
            mu_maps=np.where(masks > 0.5, 1.0, 1000.0).astype(np.float32),
            x=x,
            y=y,
            polygon_vertices_raw=vertices,
            polygon_vertices_norm=vertices,
            polygon_vertex_mask=vertex_mask,
            polygon_presence=presence,
            type_targets=np.array([[0, -1, -1]], dtype=np.int64),
            polygon_type_vocab=np.array(["rotated_rect"]),
            component_counts=np.array([1], dtype=np.int64),
        )
        csv_path = root / "polygon_params.csv"
        _write_polygon_params(csv_path)
        out_dir = root / "targets"
        export_polygon_targets(npz_path, csv_path, out_dir)
        with np.load(out_dir / "polygon_targets.npz", allow_pickle=False) as data:
            pred = rasterize_polygon_components(
                data["polygon_vertices_norm"],
                data["polygon_vertex_mask"],
                data["presence_targets"],
                data["x_norm"],
                data["y_norm"],
            )
        iou, _dice = mask_iou_dice(pred, masks)
        assert float(iou[0]) == 1.0
        assert (out_dir / "summary.md").exists()
    print("COMSOL polygon target utils smoke test passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_polygon_rasterizer import rasterize_polygon_components, mask_iou_dice
from comsol_polygon_targets import run as build_polygon_targets


class Args:
    pass


def _write_polygon_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rect_vertices(cx, cy, ax, ay, deg):
    theta = np.deg2rad(deg)
    corners = np.array([[-0.5 * ax, 0.5 * ay], [0.5 * ax, 0.5 * ay], [0.5 * ax, -0.5 * ay], [-0.5 * ax, -0.5 * ay]])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return corners @ rot.T + np.array([cx, cy])


def _norm(vertices):
    out = vertices.copy()
    out[:, 0] = (out[:, 0] - 2250.0) * (0.08 / 4500.0)
    out[:, 1] = (out[:, 1] - 1500.0) * (0.02 / 3000.0)
    return out


def _rows_for(sample, slot, vertices, component_type="rotated_rect"):
    norm = _norm(vertices)
    rows = []
    for i, (raw_v, norm_v) in enumerate(zip(vertices, norm)):
        rows.append(
            {
                "sample_index": sample,
                "split": "smoke",
                "component_slot": slot,
                "component_id": f"s{sample}_c{slot}",
                "component_type": component_type,
                "vertex_index": i,
                "x_raw": raw_v[0],
                "y_raw": raw_v[1],
                "x_norm": norm_v[0],
                "y_norm": norm_v[1],
                "ordering": "clockwise_top_left",
                "geometry_feature_tag": f"blk{slot + 1}",
                "selection_name": "geom1_blk1_dom",
                "hard_case_type": "mock",
                "component_count": 2 if sample == 1 else 1,
                "union_selection_name": "geom1_uni1_dom" if sample == 1 else "",
                "true_rotated_geometry": "true" if component_type == "rotated_rect" else "false",
                "true_multi_component_geometry": "true" if sample == 1 else "false",
            }
        )
    return rows


def main() -> None:
    x_raw = np.linspace(0.0, 4500.0, 200, dtype=np.float32)
    y_raw = np.linspace(0.0, 3000.0, 100, dtype=np.float32)
    x_norm = (x_raw - 2250.0) * (0.08 / 4500.0)
    y_norm = (y_raw - 1500.0) * (0.02 / 3000.0)
    rows = []
    sample0 = _rect_vertices(2250.0, 1500.0, 700.0, 220.0, 30.0)
    sample1a = _rect_vertices(1650.0, 1350.0, 420.0, 120.0, -25.0)
    sample1b = _rect_vertices(2150.0, 1650.0, 360.0, 100.0, 0.0)
    rows.extend(_rows_for(0, 0, sample0))
    rows.extend(_rows_for(1, 0, sample1a))
    rows.extend(_rows_for(1, 1, sample1b, "rectangular_notch"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        polygon_csv = tmp_path / "polygon_params.csv"
        _write_polygon_csv(polygon_csv, rows)
        masks = np.zeros((2, len(y_raw), len(x_raw)), dtype=np.float32)
        direct_vertices = np.zeros((2, 3, 4, 2), dtype=np.float32)
        direct_mask = np.zeros((2, 3, 4), dtype=np.float32)
        presence = np.zeros((2, 3), dtype=np.float32)
        direct_vertices[0, 0] = _norm(sample0)
        direct_mask[0, 0] = 1.0
        presence[0, 0] = 1.0
        direct_vertices[1, 0] = _norm(sample1a)
        direct_vertices[1, 1] = _norm(sample1b)
        direct_mask[1, 0] = 1.0
        direct_mask[1, 1] = 1.0
        presence[1, 0:2] = 1.0
        masks[:] = rasterize_polygon_components(direct_vertices, direct_mask, presence, x_norm, y_norm)
        npz_path = tmp_path / "mock.npz"
        np.savez_compressed(npz_path, masks=masks, mu_maps=np.where(masks > 0.5, 1.0, 1000.0), x=x_raw, y=y_raw)
        args = Args()
        args.npz_path = str(npz_path)
        args.polygon_params_csv = str(polygon_csv)
        args.output_dir = str(tmp_path / "targets")
        args.max_components = 3
        args.max_vertices = 4
        build_polygon_targets(args)
        with np.load(tmp_path / "targets" / "polygon_targets.npz", allow_pickle=True) as data:
            pred = rasterize_polygon_components(
                data["polygon_vertices_norm"],
                data["polygon_vertex_mask"],
                data["presence_targets"],
                data["x_norm"],
                data["y_norm"],
            )
        iou, _dice = mask_iou_dice(pred, masks)
        assert float(iou.min()) > 0.99, iou
    print("COMSOL polygon rasterizer smoke test passed.")


if __name__ == "__main__":
    main()

"""Smoke tests for center-anchored polygon targets."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from comsol_center_anchored_polygon_targets import build_center_anchored_targets, decode_center_anchored_vertices, run
from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components


def _fixture() -> tuple[dict, np.ndarray, np.ndarray]:
    x = np.linspace(-0.04, 0.04, 200, dtype=np.float32)
    y = np.linspace(-0.01, 0.01, 100, dtype=np.float32)
    vertices = np.zeros((2, 3, 4, 2), dtype=np.float32)
    vertices[0, 0] = np.array([[-0.01, -0.003], [0.006, -0.002], [0.008, 0.003], [-0.012, 0.002]], dtype=np.float32)
    vertices[1, 0] = np.array([[-0.03, -0.006], [-0.021, -0.006], [-0.021, -0.001], [-0.03, -0.001]], dtype=np.float32)
    vertices[1, 1] = np.array([[0.015, 0.002], [0.026, 0.002], [0.026, 0.007], [0.015, 0.007]], dtype=np.float32)
    vertex_mask = np.zeros((2, 3, 4), dtype=np.float32)
    vertex_mask[0, 0] = 1.0
    vertex_mask[1, 0:2] = 1.0
    presence = np.zeros((2, 3), dtype=np.float32)
    presence[0, 0] = 1.0
    presence[1, 0:2] = 1.0
    polygon_targets = {
        "polygon_vertices_norm": vertices,
        "polygon_vertices_raw": vertices,
        "polygon_vertex_mask": vertex_mask,
        "presence_targets": presence,
        "type_targets": np.array([[0, -1, -1], [0, 1, -1]], dtype=np.int64),
        "type_vocab": np.array(["rectangular_notch", "rotated_rect"], dtype="U64"),
        "component_counts": np.array([1, 2], dtype=np.int64),
        "sample_indices": np.array([0, 1], dtype=np.int64),
        "x_norm": x,
        "y_norm": y,
        "vertex_ordering": np.array("clockwise_top_left", dtype="U64"),
        "max_components": np.array(3, dtype=np.int64),
        "max_vertices": np.array(4, dtype=np.int64),
    }
    return polygon_targets, x, y


def test_encode_decode_round_trip() -> None:
    polygon_targets, x, y = _fixture()
    targets = build_center_anchored_targets(polygon_targets, x, y, center_bin_size_cells=8)
    decoded = decode_center_anchored_vertices(targets)
    valid = (targets["presence_targets"][..., None] * targets["polygon_vertex_mask"]) > 0.5
    max_error = float(np.max(np.abs(decoded - polygon_targets["polygon_vertices_norm"])[valid]))
    assert max_error < 1.0e-8
    pred = rasterize_polygon_components(decoded, targets["polygon_vertex_mask"], targets["presence_targets"], x, y)
    truth = rasterize_polygon_components(
        polygon_targets["polygon_vertices_norm"],
        polygon_targets["polygon_vertex_mask"],
        polygon_targets["presence_targets"],
        x,
        y,
    )
    ious, _dices = mask_iou_dice(pred, truth)
    assert float(np.min(ious)) == 1.0
    assert int(targets["center_bin_size_cells"]) == 8
    assert np.max(np.abs(targets["center_offset_targets"][targets["presence_targets"] > 0.5])) <= 0.5001
    alias_targets = dict(polygon_targets)
    alias_targets["vertex_ordering"] = np.array("clockwise_start_min_y_then_min_x_in_normalized_space", dtype="U96")
    alias_out = build_center_anchored_targets(alias_targets, x, y, center_bin_size_cells=8)
    assert str(alias_out["vertex_ordering"]) == "clockwise_top_left"


def test_cli_writes_package() -> None:
    polygon_targets, x, y = _fixture()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        npz_path = base / "data.npz"
        targets_path = base / "polygon_targets.npz"
        output_dir = base / "out"
        masks = rasterize_polygon_components(
            polygon_targets["polygon_vertices_norm"],
            polygon_targets["polygon_vertex_mask"],
            polygon_targets["presence_targets"],
            x,
            y,
        ).astype(np.float32)
        np.savez_compressed(npz_path, signals=np.zeros((2, 3, 200), dtype=np.float32), masks=masks, x=x, y=y)
        np.savez_compressed(targets_path, **polygon_targets)
        run(
            type(
                "Args",
                (),
                {
                    "npz_path": str(npz_path),
                    "polygon_targets": str(targets_path),
                    "output_dir": str(output_dir),
                    "center_bin_size_cells": 8,
                },
            )()
        )
        assert (output_dir / "center_anchored_polygon_targets.npz").exists()
        assert (output_dir / "summary.md").exists()


if __name__ == "__main__":
    test_encode_decode_round_trip()
    test_cli_writes_package()
    print("center-anchored polygon target smoke passed")

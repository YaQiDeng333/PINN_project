"""Smoke test for component-query polygon raster sensitivity diagnostics."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from component_query_polygon_raster_sensitivity_diagnostics import run_diagnostics
from comsol_polygon_rasterizer import rasterize_polygon_components


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(pred_vertices: np.ndarray, target_vertices: np.ndarray, local: np.ndarray, center: np.ndarray) -> list[dict]:
    rows = []
    for slot in range(3):
        row = {
            "sample_index": 0,
            "component_slot": slot,
            "presence_pred": 1.0 if slot == 0 else 0.0,
            "type_pred": 1,
            "hard_center_x_pred": center[0] if slot == 0 else 0.0,
            "hard_center_y_pred": center[1] if slot == 0 else 0.0,
        }
        for vertex in range(4):
            row[f"vertex{vertex}_valid"] = 1.0 if slot == 0 else 0.0
            row[f"pred_x{vertex}"] = float(pred_vertices[vertex, 0]) if slot == 0 else 0.0
            row[f"pred_y{vertex}"] = float(pred_vertices[vertex, 1]) if slot == 0 else 0.0
            row[f"true_x{vertex}"] = float(target_vertices[vertex, 0]) if slot == 0 else 0.0
            row[f"true_y{vertex}"] = float(target_vertices[vertex, 1]) if slot == 0 else 0.0
            row[f"pred_local_x{vertex}"] = float(local[vertex, 0]) if slot == 0 else 0.0
            row[f"pred_local_y{vertex}"] = float(local[vertex, 1]) if slot == 0 else 0.0
        rows.append(row)
    return rows


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        x = np.arange(10, dtype=np.float32)
        y = np.arange(10, dtype=np.float32)
        target_vertices = np.array([[[[2.0, 2.0], [2.0, 5.0], [5.0, 5.0], [5.0, 2.0]], [[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]]], dtype=np.float32)
        pred_vertices = target_vertices.copy()
        pred_vertices[0, 0, :, 0] += np.array([-0.1, -0.1, 0.2, 0.2], dtype=np.float32)
        vertex_mask = np.zeros((1, 3, 4), dtype=np.float32)
        vertex_mask[0, 0] = 1.0
        presence = np.zeros((1, 3), dtype=np.float32)
        presence[0, 0] = 1.0
        target_masks = rasterize_polygon_components(target_vertices, vertex_mask, presence, x, y).astype(np.float32)
        pred_masks = rasterize_polygon_components(pred_vertices, vertex_mask, presence, x, y)
        intersection = int(np.logical_and(pred_masks[0], target_masks[0] > 0.5).sum())
        union = int(np.logical_or(pred_masks[0], target_masks[0] > 0.5).sum())
        pred_area = int(pred_masks[0].sum())
        center = target_vertices[0, 0].mean(axis=0)
        local = pred_vertices[0, 0] - center
        np.savez_compressed(root / "data.npz", masks=target_masks, x=x, y=y)
        center_targets = np.zeros((1, 3, 2), dtype=np.float32)
        center_targets[0, 0] = center
        local_targets = np.zeros_like(target_vertices)
        local_targets[0, 0] = target_vertices[0, 0] - center
        np.savez_compressed(
            root / "targets.npz",
            sample_indices=np.array([0], dtype=np.int64),
            polygon_vertices_norm=target_vertices,
            polygon_vertex_mask=vertex_mask,
            presence_targets=presence,
            center_targets_norm=center_targets,
            local_vertices_grid=local_targets,
            grid_dx=np.array(1.0, dtype=np.float32),
            grid_dy=np.array(1.0, dtype=np.float32),
        )
        _write_csv(root / "predictions.csv", _prediction_rows(pred_vertices[0, 0], target_vertices[0, 0], local, center))
        _write_csv(
            root / "mask_metrics.csv",
            [
                {
                    "sample_index": 0,
                    "polygon_mask_iou": intersection / union,
                    "polygon_dice": 2 * intersection / (pred_area + int(target_masks[0].sum())),
                    "target_area": int(target_masks[0].sum()),
                    "pred_area": pred_area,
                }
            ],
        )
        out = root / "out"
        result = run_diagnostics(root / "predictions.csv", root / "mask_metrics.csv", root / "data.npz", root / "targets.npz", out, 0)
        assert (out / "raster_sensitivity_variants.csv").exists()
        assert (out / "per_vertex_errors.csv").exists()
        assert (out / "mask_diff_summary.csv").exists()
        assert (out / "summary.md").exists()
        variants = {row["variant"]: row for row in result["variant_rows"]}
        assert "pred_polygon" in variants
        assert "gt_polygon" in variants
        assert float(variants["gt_polygon"]["iou"]) == 1.0
    print("component-query raster sensitivity diagnostics smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

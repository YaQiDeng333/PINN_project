"""Smoke test for center-anchored polygon oracle ablation."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from center_anchored_polygon_oracle_ablation import run_ablation
from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _component_row(sample_index: int, slot: int, present: float, pred_vertices: np.ndarray, true_vertices: np.ndarray) -> dict:
    row = {
        "sample_index": sample_index,
        "component_slot": slot,
        "presence_true": present,
        "presence_prob": present,
        "presence_pred": present,
        "type_true": 0,
        "type_pred": 0,
        "center_x_bin_true": 0,
        "center_x_bin_pred": 1 if present else 0,
        "center_y_bin_true": 0,
        "center_y_bin_pred": 0,
        "center_offset_mae": 0.0,
        "signed_area_flip": 0,
        "decoded_vertex_mae": 0.0,
        "local_vertex_mae_grid": 0.0,
    }
    true_local = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float64)
    pred_local = true_local.copy()
    for vertex_idx in range(4):
        row[f"vertex{vertex_idx}_valid"] = present
        row[f"pred_x{vertex_idx}"] = float(pred_vertices[vertex_idx, 0]) if present else 0.0
        row[f"pred_y{vertex_idx}"] = float(pred_vertices[vertex_idx, 1]) if present else 0.0
        row[f"true_x{vertex_idx}"] = float(true_vertices[vertex_idx, 0]) if present else 0.0
        row[f"true_y{vertex_idx}"] = float(true_vertices[vertex_idx, 1]) if present else 0.0
        row[f"pred_local_x{vertex_idx}"] = float(pred_local[vertex_idx, 0]) if present else 0.0
        row[f"pred_local_y{vertex_idx}"] = float(pred_local[vertex_idx, 1]) if present else 0.0
        row[f"pred_local_raw_x{vertex_idx}"] = row[f"pred_local_x{vertex_idx}"]
        row[f"pred_local_raw_y{vertex_idx}"] = row[f"pred_local_y{vertex_idx}"]
        row[f"true_local_x{vertex_idx}"] = float(true_local[vertex_idx, 0]) if present else 0.0
        row[f"true_local_y{vertex_idx}"] = float(true_local[vertex_idx, 1]) if present else 0.0
    return row


def test_oracle_ablation_smoke() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        split_dir = root / "matched" / "val"
        pred_dir = root / "pred"
        out_dir = root / "out"
        split_dir.mkdir(parents=True)
        pred_dir.mkdir()

        x = np.arange(4, dtype=np.float32)
        y = np.arange(4, dtype=np.float32)
        true_vertices = np.zeros((1, 3, 4, 2), dtype=np.float32)
        pred_vertices = np.zeros_like(true_vertices)
        true_vertices[0, 0] = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]], dtype=np.float32)
        pred_vertices[0, 0] = np.array([[2.5, 0.5], [2.5, 1.5], [3.5, 1.5], [3.5, 0.5]], dtype=np.float32)
        vertex_mask = np.zeros((1, 3, 4), dtype=np.float32)
        vertex_mask[0, 0] = 1.0
        presence = np.zeros((1, 3), dtype=np.float32)
        presence[0, 0] = 1.0
        true_masks = rasterize_polygon_components(true_vertices, vertex_mask, presence, x, y).astype(np.float32)
        pred_masks = rasterize_polygon_components(pred_vertices, np.ones_like(vertex_mask), presence, x, y)
        pred_iou, pred_dice = mask_iou_dice(pred_masks, true_masks)

        np.savez_compressed(split_dir / "comsol_v3_polygon_matched_coverage.npz", masks=true_masks)
        np.savez_compressed(
            split_dir / "center_anchored_polygon_targets.npz",
            polygon_vertices_norm=true_vertices,
            polygon_vertex_mask=vertex_mask,
            presence_targets=presence,
            type_targets=np.zeros((1, 3), dtype=np.int64),
            sample_indices=np.array([0], dtype=np.int64),
            center_x_bin_targets=np.zeros((1, 3), dtype=np.int64),
            center_y_bin_targets=np.zeros((1, 3), dtype=np.int64),
            center_offset_targets=np.zeros((1, 3, 2), dtype=np.float32),
            local_vertices_grid=np.array(
                [[[[ -0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], [[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]]],
                dtype=np.float32,
            ),
            center_bin_x_centers=np.array([1.0, 3.0], dtype=np.float32),
            center_bin_y_centers=np.array([1.0, 3.0], dtype=np.float32),
            center_bin_width_x=np.array(2.0, dtype=np.float32),
            center_bin_width_y=np.array(2.0, dtype=np.float32),
            grid_dx=np.array(1.0, dtype=np.float32),
            grid_dy=np.array(1.0, dtype=np.float32),
            x_norm=x,
            y_norm=y,
        )
        component_rows = [
            _component_row(0, 0, 1.0, pred_vertices[0, 0], true_vertices[0, 0]),
            _component_row(0, 1, 0.0, pred_vertices[0, 1], true_vertices[0, 1]),
            _component_row(0, 2, 0.0, pred_vertices[0, 2], true_vertices[0, 2]),
        ]
        mask_rows = [
            {
                "sample_index": 0,
                "polygon_mask_iou": float(pred_iou[0]),
                "polygon_dice": float(pred_dice[0]),
                "target_area": int(true_masks[0].sum()),
                "pred_area": int(pred_masks[0].sum()),
                "true_component_count": 1,
                "pred_component_count": 1,
                "out_of_grid_vertex_count": 0,
            }
        ]
        for split in ("train", "test"):
            other = root / "matched" / split
            other.mkdir(parents=True)
            for name in ("comsol_v3_polygon_matched_coverage.npz", "center_anchored_polygon_targets.npz"):
                (other / name).write_bytes((split_dir / name).read_bytes())
            _write_csv(pred_dir / f"{split}_center_anchored_polygon_predictions.csv", component_rows)
            _write_csv(pred_dir / f"{split}_center_anchored_polygon_mask_metrics.csv", mask_rows)
        _write_csv(pred_dir / "val_center_anchored_polygon_predictions.csv", component_rows)
        _write_csv(pred_dir / "val_center_anchored_polygon_mask_metrics.csv", mask_rows)

        result = run_ablation(pred_dir, root / "matched", out_dir, "smoke")
        rows = result["summary_rows"]
        lookup = {(row["split"], row["variant"]): row for row in rows}
        assert lookup[("val", "pred_all")]["polygon_iou_mean"] == float(pred_iou[0])
        assert lookup[("val", "gt_center_bin")]["polygon_iou_mean"] == 1.0
        assert lookup[("val", "gt_center_bin_offset_local")]["polygon_iou_mean"] == 1.0


if __name__ == "__main__":
    test_oracle_ablation_smoke()
    print("center-anchored polygon oracle ablation smoke passed")

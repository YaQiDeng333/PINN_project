"""Smoke test for polygon vertex-to-raster sensitivity diagnostics."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_polygon_raster_sensitivity_diagnostics import main
from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        x = np.linspace(-1.0, 1.0, 21, dtype=np.float32)
        y = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
        signals = np.ones((1, 3, 21), dtype=np.float32)
        vertices = np.zeros((1, 3, 4, 2), dtype=np.float32)
        vertex_mask = np.zeros((1, 3, 4), dtype=np.float32)
        presence = np.zeros((1, 3), dtype=np.float32)
        type_targets = np.full((1, 3), -1, dtype=np.int64)
        vertices[0, 0] = np.array(
            [[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]],
            dtype=np.float32,
        )
        vertex_mask[0, 0] = 1.0
        presence[0, 0] = 1.0
        type_targets[0, 0] = 0
        masks = rasterize_polygon_components(vertices, vertex_mask, presence, x, y).astype(np.float32)
        npz_path = base / "sample.npz"
        targets_path = base / "polygon_targets.npz"
        np.savez_compressed(npz_path, signals=signals, masks=masks, x=x, y=y)
        np.savez_compressed(
            targets_path,
            polygon_vertices_norm=vertices,
            polygon_vertex_mask=vertex_mask,
            presence_targets=presence,
            type_targets=type_targets,
            sample_indices=np.array([0], dtype=np.int64),
        )
        pred_vertices = vertices.copy()
        pred_vertices[0, 0, :, 0] += 0.05
        pred_presence = presence.copy()
        pred_masks = rasterize_polygon_components(pred_vertices, np.ones_like(vertex_mask), pred_presence, x, y)
        ious, dices = mask_iou_dice(pred_masks, masks)
        prediction_rows = []
        for slot in range(3):
            row = {
                "sample_index": 0,
                "component_slot": slot,
                "presence_true": float(presence[0, slot]),
                "presence_prob": float(pred_presence[0, slot]),
                "presence_pred": float(pred_presence[0, slot]),
                "type_true": int(type_targets[0, slot]),
                "type_pred": 0,
                "vertex_mae": float(np.abs(pred_vertices[0, slot] - vertices[0, slot]).mean()) if presence[0, slot] else 0.0,
            }
            for vertex_idx in range(4):
                row[f"vertex{vertex_idx}_valid"] = float(vertex_mask[0, slot, vertex_idx])
                row[f"pred_x{vertex_idx}"] = float(pred_vertices[0, slot, vertex_idx, 0])
                row[f"pred_y{vertex_idx}"] = float(pred_vertices[0, slot, vertex_idx, 1])
                row[f"true_x{vertex_idx}"] = float(vertices[0, slot, vertex_idx, 0])
                row[f"true_y{vertex_idx}"] = float(vertices[0, slot, vertex_idx, 1])
            prediction_rows.append(row)
        predictions_csv = base / "train_polygon_predictions.csv"
        mask_metrics_csv = base / "train_polygon_mask_metrics.csv"
        _write_csv(predictions_csv, prediction_rows)
        _write_csv(
            mask_metrics_csv,
            [
                {
                    "sample_index": 0,
                    "polygon_mask_iou": float(ious[0]),
                    "polygon_dice": float(dices[0]),
                    "target_area": int(masks[0].sum()),
                    "pred_area": int(pred_masks[0].sum()),
                    "true_component_count": 1,
                    "pred_component_count": 1,
                }
            ],
        )
        out = base / "out"
        rc = main(
            [
                "--predictions-csv",
                str(predictions_csv),
                "--mask-metrics-csv",
                str(mask_metrics_csv),
                "--npz-path",
                str(npz_path),
                "--polygon-targets",
                str(targets_path),
                "--output-dir",
                str(out),
                "--label",
                "smoke",
            ]
        )
        assert rc == 0
        for name in [
            "vertex_raster_sensitivity.csv",
            "component_raster_sensitivity.csv",
            "sample_raster_sensitivity.csv",
            "summary.md",
        ]:
            assert (out / name).exists(), name
        summary = (out / "summary.md").read_text(encoding="utf-8")
        assert "vertex-to-raster sensitivity" in summary
        assert "mean oracle IoU" in summary
    print("COMSOL polygon raster sensitivity diagnostics smoke test passed.")


if __name__ == "__main__":
    main_test()

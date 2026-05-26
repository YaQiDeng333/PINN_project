"""Smoke test for center-anchored center decode diagnostics."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from center_anchored_center_decode_diagnostics import main as diagnostics_main
from comsol_center_anchored_polygon_targets import build_center_anchored_targets
from smoke_test_comsol_center_anchored_polygon_targets import _fixture


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_center_decode_diagnostics_smoke() -> None:
    polygon_targets, x, y = _fixture()
    targets = build_center_anchored_targets(polygon_targets, x, y, center_bin_size_cells=8)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        resplit_root = base / "resplit"
        prediction_dir = base / "predictions"
        for split in ["train", "val", "test"]:
            split_dir = resplit_root / split
            split_dir.mkdir(parents=True)
            np.savez_compressed(split_dir / "center_anchored_polygon_targets.npz", **targets)
            _write_csv(
                split_dir / "defect_params.csv",
                [
                    {"sample_index": 0, "hard_case_type": "x_bin_wrong_like"},
                    {"sample_index": 1, "hard_case_type": "rare_y_bin_wrong"},
                ],
            )
            prediction_rows = []
            for sample_index in range(targets["presence_targets"].shape[0]):
                for slot in range(3):
                    presence = float(targets["presence_targets"][sample_index, slot])
                    row = {
                        "sample_index": sample_index,
                        "component_slot": slot,
                        "presence_true": presence,
                        "presence_pred": presence,
                        "center_x_bin_true": int(targets["center_x_bin_targets"][sample_index, slot]),
                        "center_x_bin_pred": int(targets["center_x_bin_targets"][sample_index, slot]),
                        "center_y_bin_true": int(targets["center_y_bin_targets"][sample_index, slot]),
                        "center_y_bin_pred": int(targets["center_y_bin_targets"][sample_index, slot]),
                        "center_offset_mae": 0.0,
                        "center_offset_x_true": float(targets["center_offset_targets"][sample_index, slot, 0]),
                        "center_offset_x_pred": float(targets["center_offset_targets"][sample_index, slot, 0]),
                        "center_offset_y_true": float(targets["center_offset_targets"][sample_index, slot, 1]),
                        "center_offset_y_pred": float(targets["center_offset_targets"][sample_index, slot, 1]),
                        "center_x_bin_prob_top1": 0.9,
                        "center_x_bin_prob_top2": 0.1,
                        "center_x_bin_prob_margin": 0.8,
                        "center_y_bin_prob_top1": 0.8,
                        "center_y_bin_prob_top2": 0.2,
                        "center_y_bin_prob_margin": 0.6,
                        "center_x_true": float(targets["center_targets_norm"][sample_index, slot, 0]),
                        "center_y_true": float(targets["center_targets_norm"][sample_index, slot, 1]),
                        "hard_center_x_pred": float(targets["center_targets_norm"][sample_index, slot, 0]),
                        "hard_center_y_pred": float(targets["center_targets_norm"][sample_index, slot, 1]),
                        "soft_center_x_pred": float(targets["center_targets_norm"][sample_index, slot, 0]),
                        "soft_center_y_pred": float(targets["center_targets_norm"][sample_index, slot, 1]),
                        "local_vertex_mae_grid": 0.0,
                        "decoded_vertex_mae": 0.0,
                        "signed_area_flip": 0,
                    }
                    for vertex_idx in range(4):
                        row[f"vertex{vertex_idx}_valid"] = float(targets["polygon_vertex_mask"][sample_index, slot, vertex_idx])
                        row[f"pred_x{vertex_idx}"] = float(targets["polygon_vertices_norm"][sample_index, slot, vertex_idx, 0])
                        row[f"pred_y{vertex_idx}"] = float(targets["polygon_vertices_norm"][sample_index, slot, vertex_idx, 1])
                        row[f"true_x{vertex_idx}"] = float(targets["polygon_vertices_norm"][sample_index, slot, vertex_idx, 0])
                        row[f"true_y{vertex_idx}"] = float(targets["polygon_vertices_norm"][sample_index, slot, vertex_idx, 1])
                        row[f"pred_local_x{vertex_idx}"] = float(targets["local_vertices_grid"][sample_index, slot, vertex_idx, 0])
                        row[f"pred_local_y{vertex_idx}"] = float(targets["local_vertices_grid"][sample_index, slot, vertex_idx, 1])
                    prediction_rows.append(row)
            _write_csv(prediction_dir / f"{split}_center_anchored_polygon_predictions.csv", prediction_rows)
            _write_csv(
                prediction_dir / f"{split}_center_anchored_polygon_mask_metrics.csv",
                [
                    {"sample_index": 0, "polygon_mask_iou": 1.0, "target_area": 5, "pred_area": 5},
                    {"sample_index": 1, "polygon_mask_iou": 1.0, "target_area": 5, "pred_area": 5},
                ],
            )
        output_dir = base / "diagnostics"
        rc = diagnostics_main(
            [
                "--prediction-dir",
                str(prediction_dir),
                "--resplit-root",
                str(resplit_root),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "smoke",
            ]
        )
        assert rc == 0
        with (output_dir / "center_decode_summary.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert all(float(row["x_bin_acc"]) == 1.0 for row in rows)
        assert all(float(row["y_bin_acc"]) == 1.0 for row in rows)
        assert all(float(row["hard_center_l2_error_grid"]) == 0.0 for row in rows)
        assert all(float(row["soft_center_l2_error_grid"]) == 0.0 for row in rows)
        assert (output_dir / "center_decode_per_component.csv").exists()
        assert (output_dir / "center_decode_per_sample.csv").exists()
        assert (output_dir / "summary.md").exists()


if __name__ == "__main__":
    test_center_decode_diagnostics_smoke()
    print("center-anchored center decode diagnostics smoke passed")

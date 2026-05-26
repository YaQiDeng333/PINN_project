"""Smoke test for center_anchored_y_bin_diagnostics.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from center_anchored_y_bin_diagnostics import build_diagnostics


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _target_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.zeros((1, 3, 4, 2), dtype=np.float32)
    vertices[0, 0, :, :] = np.array(
        [[-0.01, -0.001], [-0.01, 0.001], [0.01, 0.001], [0.01, -0.001]],
        dtype=np.float32,
    )
    vertex_mask = np.zeros((1, 3, 4), dtype=np.float32)
    vertex_mask[0, 0, :] = 1.0
    np.savez_compressed(
        path,
        sample_indices=np.array([0], dtype=np.int64),
        grid_dy=np.array(0.0002, dtype=np.float32),
        presence_targets=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        polygon_vertices_norm=vertices,
        polygon_vertex_mask=vertex_mask,
        center_offset_targets=np.zeros((1, 3, 2), dtype=np.float32),
    )


def test_y_bin_diagnostics_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        resplit = base / "resplit"
        preds = base / "preds"
        out = base / "out"
        coverage_rows = []
        for split in ["train", "val", "test"]:
            _target_npz(resplit / split / "center_anchored_polygon_targets.npz")
            _write_csv(
                resplit / split / "defect_params.csv",
                [
                    {
                        "sample_index": 0,
                        "hard_case_type": "x_bin_wrong_like",
                        "defect_axis_y": 0.001,
                        "rotation_angle": 0.0,
                        "true_rotated_geometry": "false",
                        "true_multi_component_geometry": "false",
                    }
                ],
            )
            _write_csv(
                resplit / split / "polygon_params.csv",
                [
                    {
                        "sample_index": 0,
                        "component_index": 0,
                        "presence": 1,
                        "hard_case_type": "x_bin_wrong_like",
                        "component_type": "rectangular_notch",
                        "is_true_rotated": "false",
                        "is_true_multi_component": "false",
                    }
                ],
            )
            _write_csv(
                preds / f"{split}_center_anchored_polygon_predictions.csv",
                [
                    {
                        "sample_index": 0,
                        "component_slot": 0,
                        "type_true": 0,
                        "type_pred": 0,
                        "center_x_bin_true": 2,
                        "center_x_bin_pred": 2,
                        "center_y_bin_true": 3,
                        "center_y_bin_pred": 4 if split != "train" else 3,
                        "center_offset_mae": 0.1,
                        "local_vertex_mae_grid": 0.2,
                        "decoded_vertex_mae": 0.001,
                    }
                ],
            )
            _write_csv(
                preds / f"{split}_center_anchored_polygon_mask_metrics.csv",
                [
                    {
                        "sample_index": 0,
                        "polygon_mask_iou": 0.0 if split != "train" else 1.0,
                        "pred_area": 12,
                        "target_area": 10,
                    }
                ],
            )
            coverage_rows.append(
                {
                    "new_split": split,
                    "new_sample_index": 0,
                    "all_bins_exactly_covered": "true",
                    "all_bins_within_distance1": "true",
                    "max_center_bin_distance_to_train": 0,
                }
            )
        _write_csv(resplit / "coverage_report.csv", coverage_rows)
        result = build_diagnostics(resplit, preds, out)
        assert result["component_count"] == 3
        for name in [
            "per_component_y_bin_diagnostics.csv",
            "per_sample_y_bin_diagnostics.csv",
            "grouped_by_y_bin.csv",
            "grouped_by_component_slot.csv",
            "grouped_by_true_rotated.csv",
            "grouped_by_multi_component.csv",
            "y_bin_confusion.csv",
            "y_bin_error_histogram.csv",
            "summary.md",
        ]:
            assert (out / name).exists(), name


if __name__ == "__main__":
    test_y_bin_diagnostics_smoke()
    print("center-anchored y-bin diagnostics smoke passed")

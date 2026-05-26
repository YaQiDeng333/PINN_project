"""Smoke test for center_anchored_polygon_failure_diagnostics.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from center_anchored_polygon_failure_diagnostics import run


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row(split: str, sample_index: int, iou: float) -> dict:
    return {
        "sample_index": sample_index,
        "component_slot": 0,
        "presence_true": 1.0,
        "presence_prob": 0.99,
        "presence_pred": 1.0,
        "type_true": 0,
        "type_pred": 0,
        "center_x_bin_true": 1,
        "center_x_bin_pred": 1 if split == "train" else 2,
        "center_y_bin_true": 1,
        "center_y_bin_pred": 1 if split != "test" else 2,
        "center_offset_mae": 0.01,
        "signed_area_flip": 0,
        "decoded_vertex_mae": 0.001,
        "local_vertex_mae_grid": 0.1,
        "vertex0_valid": 1,
        "pred_x0": -0.01,
        "pred_y0": -0.001,
        "true_x0": -0.01,
        "true_y0": -0.001,
        "pred_local_x0": -1.0,
        "pred_local_y0": -1.0,
        "true_local_x0": -1.0,
        "true_local_y0": -1.0,
        "vertex1_valid": 1,
        "pred_x1": -0.01,
        "pred_y1": 0.001,
        "true_x1": -0.01,
        "true_y1": 0.001,
        "pred_local_x1": -1.0,
        "pred_local_y1": 1.0,
        "true_local_x1": -1.0,
        "true_local_y1": 1.0,
        "vertex2_valid": 1,
        "pred_x2": 0.01,
        "pred_y2": 0.001,
        "true_x2": 0.01,
        "true_y2": 0.001,
        "pred_local_x2": 1.0,
        "pred_local_y2": 1.0,
        "true_local_x2": 1.0,
        "true_local_y2": 1.0,
        "vertex3_valid": 1,
        "pred_x3": 0.01,
        "pred_y3": -0.001,
        "true_x3": 0.01,
        "true_y3": -0.001,
        "pred_local_x3": 1.0,
        "pred_local_y3": -1.0,
        "true_local_x3": 1.0,
        "true_local_y3": -1.0,
    }


def _make_split(root: Path, split: str, iou: float) -> None:
    raw = root / "raw" / split
    pred = root / "pred"
    target = root / "targets" / split
    raw.mkdir(parents=True, exist_ok=True)
    pred.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    _write_csv(
        raw / "defect_params.csv",
        [
            {
                "sample_index": 0,
                "split": split,
                "hard_case_type": "x_bin_wrong_like",
                "defect_type": "rectangular_notch",
                "defect_center_x": 0.0,
                "defect_center_y": 0.0,
                "true_rotated_geometry": "false",
                "true_multi_component_geometry": "false",
            }
        ],
    )
    _write_csv(pred / f"{split}_center_anchored_polygon_predictions.csv", [_prediction_row(split, 0, iou)])
    _write_csv(
        pred / f"{split}_center_anchored_polygon_mask_metrics.csv",
        [
            {
                "sample_index": 0,
                "polygon_mask_iou": iou,
                "polygon_dice": iou,
                "target_area": 10,
                "pred_area": 10 if iou > 0 else 0,
                "true_component_count": 1,
                "pred_component_count": 1,
                "out_of_grid_vertex_count": 0,
            }
        ],
    )
    vertices = np.asarray([[[[-0.01, -0.001], [-0.01, 0.001], [0.01, 0.001], [0.01, -0.001]], [[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]]], dtype=np.float32)
    presence = np.asarray([[1, 0, 0]], dtype=np.float32)
    vertex_mask = np.asarray([[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.float32)
    np.savez_compressed(
        target / "center_anchored_polygon_targets.npz",
        polygon_vertices_norm=vertices,
        polygon_vertex_mask=vertex_mask,
        presence_targets=presence,
        type_targets=np.asarray([[0, 0, 0]], dtype=np.int64),
        sample_indices=np.asarray([0], dtype=np.int64),
        center_x_bin_targets=np.asarray([[1, 0, 0]], dtype=np.int64),
        center_y_bin_targets=np.asarray([[1, 0, 0]], dtype=np.int64),
        center_offset_targets=np.zeros((1, 3, 2), dtype=np.float32),
        center_targets_norm=np.asarray([[[0.0, 0.0], [0, 0], [0, 0]]], dtype=np.float32),
        local_vertices_grid=np.zeros((1, 3, 4, 2), dtype=np.float32),
        center_bin_x_centers=np.asarray([-0.02, 0.0, 0.02], dtype=np.float32),
        center_bin_y_centers=np.asarray([-0.004, 0.0, 0.004], dtype=np.float32),
        grid_dx=np.asarray(0.001, dtype=np.float32),
        grid_dy=np.asarray(0.001, dtype=np.float32),
    )


class Args:
    prediction_dir: str
    target_root: str
    raw_root: str
    output_dir: str
    coverage_output_dir: str


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_split(root, "train", 1.0)
        _make_split(root, "val", 0.0)
        _make_split(root, "test", 0.5)
        args = Args()
        args.prediction_dir = str(root / "pred")
        args.target_root = str(root / "targets")
        args.raw_root = str(root / "raw")
        args.output_dir = str(root / "out" / "diag")
        args.coverage_output_dir = str(root / "out" / "coverage")
        run(args)
        required = [
            root / "out" / "diag" / "per_sample_failure_diagnostics.csv",
            root / "out" / "diag" / "per_component_failure_diagnostics.csv",
            root / "out" / "diag" / "summary.md",
            root / "out" / "coverage" / "heldout_nearest_train_matches.csv",
            root / "out" / "coverage" / "matched_coverage_summary.md",
        ]
        for path in required:
            if not path.exists():
                raise AssertionError(f"missing output: {path}")
        rows = list(csv.DictReader((root / "out" / "diag" / "per_sample_failure_diagnostics.csv").open()))
        if len(rows) != 3:
            raise AssertionError(f"expected 3 sample rows, got {len(rows)}")
        if not any(row["zero_iou"] == "True" for row in rows):
            raise AssertionError("expected at least one zero-IoU row")


if __name__ == "__main__":
    main()

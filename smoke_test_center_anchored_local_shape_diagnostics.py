"""Smoke test for center-anchored local-shape diagnostics."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from center_anchored_local_shape_diagnostics import run


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_split(root: Path, pred_root: Path, split: str, iou: float) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    local = np.asarray(
        [[[[2.0, -1.0], [2.0, 1.0], [-2.0, 1.0], [-2.0, -1.0]], [[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]]],
        dtype=np.float32,
    )
    np.savez_compressed(
        split_dir / "center_anchored_polygon_targets.npz",
        local_vertices_grid=local,
        polygon_vertex_mask=np.asarray([[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.float32),
        presence_targets=np.asarray([[1, 0, 0]], dtype=np.float32),
        sample_indices=np.asarray([0], dtype=np.int64),
    )
    _write_csv(
        split_dir / "defect_params.csv",
        [{"sample_index": 0, "hard_case_type": "x_bin_wrong_like", "is_true_rotated": 1, "is_true_multi_component": 0}],
    )
    _write_csv(
        split_dir / "polygon_params.csv",
        [{"sample_index": 0, "component_index": 0, "component_type": "rotated_rect", "is_true_rotated": 1, "is_true_multi_component": 0}],
    )
    comp_row = {
        "sample_index": 0,
        "component_slot": 0,
        "presence_true": 1,
        "presence_pred": 1,
        "type_true": 0,
        "type_pred": 0,
        "center_x_bin_true": 2,
        "center_x_bin_pred": 2,
        "center_y_bin_true": 3,
        "center_y_bin_pred": 4 if split == "test" else 3,
        "signed_area_flip": 0,
    }
    for idx in range(4):
        comp_row[f"vertex{idx}_valid"] = 1
        comp_row[f"pred_local_x{idx}"] = float(local[0, 0, idx, 0] + 0.1)
        comp_row[f"pred_local_y{idx}"] = float(local[0, 0, idx, 1])
        comp_row[f"pred_local_raw_x{idx}"] = float(local[0, 0, idx, 0] + 0.2)
        comp_row[f"pred_local_raw_y{idx}"] = float(local[0, 0, idx, 1])
        comp_row[f"true_local_x{idx}"] = float(local[0, 0, idx, 0])
        comp_row[f"true_local_y{idx}"] = float(local[0, 0, idx, 1])
    _write_csv(pred_root / f"{split}_center_anchored_polygon_predictions.csv", [comp_row])
    _write_csv(
        pred_root / f"{split}_center_anchored_polygon_mask_metrics.csv",
        [{"sample_index": 0, "polygon_mask_iou": iou, "target_area": 10, "pred_area": 9, "out_of_grid_vertex_count": 0}],
    )


def test_local_shape_diagnostics_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "resplit"
        pred_root = base / "pred"
        out = base / "out"
        for split, iou in [("train", 1.0), ("val", 0.5), ("test", 0.0)]:
            _write_split(root, pred_root, split, iou)
        stats = run(root, pred_root, out)
        assert stats["component_rows"] == 3.0
        assert stats["sample_rows"] == 3.0
        assert (out / "local_shape_target_stats_by_split.csv").exists()
        assert (out / "local_shape_prediction_diagnostics_per_component.csv").exists()
        assert (out / "summary.md").exists()


def test_duplicate_prediction_rows_fail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "resplit"
        pred_root = base / "pred"
        out = base / "out"
        for split, iou in [("train", 1.0), ("val", 0.5), ("test", 0.0)]:
            _write_split(root, pred_root, split, iou)
        duplicate_path = pred_root / "val_center_anchored_polygon_predictions.csv"
        rows = list(csv.DictReader(duplicate_path.open(newline="", encoding="utf-8")))
        _write_csv(duplicate_path, [rows[0], rows[0]])
        try:
            run(root, pred_root, out)
        except ValueError as exc:
            assert "Duplicate prediction row" in str(exc)
        else:
            raise AssertionError("duplicate prediction rows should fail")


if __name__ == "__main__":
    test_local_shape_diagnostics_smoke()
    test_duplicate_prediction_rows_fail()
    print("center-anchored local-shape diagnostics smoke passed")

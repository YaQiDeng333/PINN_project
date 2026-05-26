"""Smoke test for comsol_polygon_generalization_diagnostics.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_polygon_generalization_diagnostics import run


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class Args:
    raw_root: str
    converted_dir: str
    prediction_dir: str
    output_dir: str


def _make_split(root: Path, split: str) -> None:
    raw = root / "raw" / split
    converted = root / "converted"
    pred = root / "pred"
    raw.mkdir(parents=True, exist_ok=True)
    converted.mkdir(parents=True, exist_ok=True)
    pred.mkdir(parents=True, exist_ok=True)
    defect = {
        "sample_index": 0,
        "split": split,
        "source_sample_id": f"{split}_0",
        "source_global_index": 0,
        "hard_case_type": "x_bin_wrong_like",
        "defect_type": "rotated_rect",
        "defect_center_x": 0.0,
        "defect_center_y": 0.0,
        "defect_center_z": 200.0,
        "defect_axis_x": 0.01,
        "defect_axis_y": 0.001,
        "defect_axis_z": 80.0,
        "defect_depth_or_shape_param": 80.0,
        "rotation_angle": 0.5,
        "boundary_irregularity_proxy": 0.5,
        "component_type_combination": "rotated_rect",
        "defect_mu": 1.0,
        "c_magn": 0.25,
        "mur_magn": 3.5,
        "Mr_magn_A_per_m": 60.0,
        "true_rotated_geometry": "true",
        "true_multi_component_geometry": "false",
        "source_component_json": "[]",
    }
    _write_csv(raw / "defect_params.csv", [defect])
    poly = {
        "sample_index": 0,
        "split": split,
        "component_index": 0,
        "presence": 1,
        "hard_case_type": "x_bin_wrong_like",
        "component_type": "rotated_rect",
        "vertex_ordering": "clockwise_top_left",
        "raw_x0": 0,
        "raw_y0": 0,
        "raw_x1": 0,
        "raw_y1": 1,
        "raw_x2": 1,
        "raw_y2": 1,
        "raw_x3": 1,
        "raw_y3": 0,
        "norm_x0": -0.01,
        "norm_y0": -0.001,
        "norm_x1": -0.01,
        "norm_y1": 0.001,
        "norm_x2": 0.01,
        "norm_y2": 0.001,
        "norm_x3": 0.01,
        "norm_y3": -0.001,
        "source_geometry_type": "rotated_rect",
        "is_true_rotated": "true",
        "is_true_multi_component": "false",
        "geometry_feature_tag": "blk1",
        "selection_name": "sel1",
        "union_selection_name": "",
        "component_count": 1,
    }
    _write_csv(raw / "polygon_params.csv", [poly])
    signals = np.asarray([[[0.0, 1.0, 0.0, -1.0], [0.0, 0.8, 0.0, -0.8], [0.0, 0.6, 0.0, -0.6]]], dtype=np.float32)
    np.savez_compressed(converted / f"{split}_comsol_v3_polygon_hard_case.npz", signals=signals, masks=np.zeros((1, 4, 4)), x=np.linspace(-0.04, 0.04, 4), y=np.linspace(-0.01, 0.01, 4))
    _write_csv(
        pred / f"{split}_polygon_mask_metrics.csv",
        [
            {
                "sample_index": 0,
                "polygon_mask_iou": 0.5,
                "polygon_dice": 0.66,
                "target_area": 10,
                "pred_area": 12,
                "true_component_count": 1,
                "pred_component_count": 1,
            }
        ],
    )
    row = {
        "sample_index": 0,
        "component_slot": 0,
        "presence_true": 1.0,
        "presence_prob": 0.9,
        "presence_pred": 1.0,
        "type_true": 0,
        "type_pred": 0,
        "vertex_mae": 0.001,
    }
    for idx, (x, y) in enumerate([(-0.01, -0.001), (-0.01, 0.001), (0.01, 0.001), (0.01, -0.001)]):
        row[f"vertex{idx}_valid"] = 1.0
        row[f"pred_x{idx}"] = x
        row[f"pred_y{idx}"] = y
        row[f"true_x{idx}"] = x
        row[f"true_y{idx}"] = y
    _write_csv(pred / f"{split}_polygon_predictions.csv", [row])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for split in ("train", "val", "test"):
            _make_split(root, split)
        args = Args()
        args.raw_root = str(root / "raw")
        args.converted_dir = str(root / "converted")
        args.prediction_dir = str(root / "pred")
        args.output_dir = str(root / "out")
        run(args)
        for name in [
            "split_geometry_signal_distribution.csv",
            "grouped_geometry_signal_distribution.csv",
            "prediction_failure_per_sample.csv",
            "prediction_failure_per_component.csv",
            "grouped_prediction_failures.csv",
            "worst_val_test_polygon_samples.csv",
            "summary.md",
        ]:
            assert (root / "out" / name).exists(), name
    print("smoke_test_comsol_polygon_generalization_diagnostics passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke test for comsol_parametric_oracle_ablation.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_parametric_oracle_ablation import (
    _degree_raw_to_raster_continuous,
    load_prediction_data,
    load_target_data,
    main,
)
from comsol_parametric_rasterizer import rasterize_components


def _write_prediction_csv(path: Path, continuous_true: np.ndarray, continuous_pred: np.ndarray) -> None:
    type_vocab = ["rectangular_notch", "rotated_rect"]
    fieldnames = [
        "split",
        "sample_index",
        "component_slot",
        "matched_slot",
        "presence_true",
        "presence_prob",
        "presence_pred",
        "type_true",
        "type_pred",
        "type_correct",
        "center_x_true",
        "center_x_pred",
        "center_y_true",
        "center_y_pred",
        "axis_x_true",
        "axis_x_pred",
        "axis_y_true",
        "axis_y_pred",
        "depth_true",
        "depth_pred",
        "rotation_true",
        "rotation_pred",
        "center_error",
        "axis_error",
        "depth_error",
        "rotation_error",
        "target_schema",
        "type_vocab",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index in range(2):
            for slot in range(3):
                present = 1 if slot == 0 else 0
                true_type = 0 if slot == 0 else -1
                pred_type = 1 if sample_index == 0 and slot == 0 else true_type
                row = {
                    "split": "mock",
                    "sample_index": sample_index,
                    "component_slot": slot,
                    "matched_slot": slot,
                    "presence_true": present,
                    "presence_prob": float(present),
                    "presence_pred": present,
                    "type_true": type_vocab[true_type] if true_type >= 0 else "",
                    "type_pred": type_vocab[pred_type] if pred_type >= 0 else type_vocab[0],
                    "type_correct": int(true_type == pred_type),
                    "target_schema": "center_x|center_y|axis_x|axis_y|depth_or_shape_param|rotation_angle",
                    "type_vocab": "|".join(type_vocab),
                    "center_error": 0.0,
                    "axis_error": 0.0,
                    "depth_error": 0.0,
                    "rotation_error": 0.0,
                }
                aliases = [
                    ("center_x", 0),
                    ("center_y", 1),
                    ("axis_x", 2),
                    ("axis_y", 3),
                    ("depth", 4),
                    ("rotation", 5),
                ]
                for name, idx in aliases:
                    row[f"{name}_true"] = float(continuous_true[sample_index, slot, idx])
                    row[f"{name}_pred"] = float(continuous_pred[sample_index, slot, idx])
                writer.writerow(row)


def main_test() -> None:
    x = np.linspace(-1.0, 1.0, 80).astype(np.float32)
    y = np.linspace(-1.0, 1.0, 60).astype(np.float32)
    raw_schema = np.array(
        ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"],
        dtype="U64",
    )
    raster_schema = np.array(
        ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_sin", "rotation_cos"],
        dtype="U64",
    )
    type_vocab = np.array(["rectangular_notch", "rotated_rect"], dtype="U64")
    continuous_true = np.array(
        [
            [[0.0, 0.0, 0.7, 0.25, 0.1, 0.0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0.1, -0.1, 0.6, 0.25, 0.1, 30.0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        ],
        dtype=np.float32,
    )
    continuous_pred = continuous_true.copy()
    continuous_pred[0, 0, 5] = 55.0
    continuous_pred[1, 0, 5] = -20.0
    presence = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    type_targets = np.array([[0, -1, -1], [0, -1, -1]], dtype=np.int64)
    masks = rasterize_components(
        _degree_raw_to_raster_continuous(continuous_true),
        type_targets,
        presence,
        raster_schema,
        type_vocab,
        x,
        y,
    ).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        npz_path = base / "mock.npz"
        targets_path = base / "targets.npz"
        predictions_path = base / "predictions.csv"
        output_dir = base / "out"
        np.savez(npz_path, masks=masks, x=x, y=y)
        np.savez(
            targets_path,
            sample_indices=np.array([0, 1], dtype=np.int64),
            continuous_targets=continuous_true,
            type_targets=type_targets,
            presence_targets=presence,
            target_schema=raw_schema,
            type_vocab=type_vocab,
        )
        _write_prediction_csv(predictions_path, continuous_true, continuous_pred)

        rc = main(
            [
                "--npz-path",
                str(npz_path),
                "--targets-path",
                str(targets_path),
                "--predictions-csv",
                str(predictions_path),
                "--output-dir",
                str(output_dir),
                "--split",
                "mock",
                "--max-components",
                "3",
            ]
        )
        assert rc == 0
        assert (output_dir / "per_sample_oracle_ablation.csv").exists()
        assert (output_dir / "aggregate_oracle_ablation.csv").exists()
        assert (output_dir / "summary.md").exists()

        with (output_dir / "aggregate_oracle_ablation.csv").open("r", newline="", encoding="utf-8") as handle:
            rows = {row["variant"]: row for row in csv.DictReader(handle)}
        pred_iou = float(rows["pred_all"]["avg_mask_iou"])
        assert float(rows["gt_rotation"]["avg_mask_iou"]) > pred_iou
        assert float(rows["gt_all"]["avg_mask_iou"]) >= float(rows["gt_rotation"]["avg_mask_iou"])
        assert np.isclose(float(rows["gt_type"]["avg_mask_iou"]), pred_iou)

        target = load_target_data(npz_path, targets_path, max_components=3)
        bad_csv = base / "bad_predictions.csv"
        bad_csv.write_text("sample_index,component_slot\n0,0\n", encoding="utf-8")
        try:
            load_prediction_data(bad_csv, target, "mock")
        except ValueError as exc:
            assert "missing required fields" in str(exc)
        else:
            raise AssertionError("Expected missing-field prediction CSV to fail.")
    print("COMSOL parametric oracle ablation smoke test passed.")


if __name__ == "__main__":
    main_test()

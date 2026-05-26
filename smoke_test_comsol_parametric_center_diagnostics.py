"""Smoke test for comsol_parametric_center_diagnostics.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_parametric_center_diagnostics import main


FIELDNAMES = [
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


def _write_prediction_csv(path: Path, continuous_true: np.ndarray, continuous_pred: np.ndarray, presence: np.ndarray, type_targets: np.ndarray) -> None:
    type_vocab = ["rectangular_notch", "rotated_rect"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for sample in range(continuous_true.shape[0]):
            for slot in range(continuous_true.shape[1]):
                present = int(presence[sample, slot] > 0.5)
                true_type_id = int(type_targets[sample, slot])
                pred_type_id = 0 if true_type_id < 0 else true_type_id
                true = continuous_true[sample, slot]
                pred = continuous_pred[sample, slot]
                writer.writerow(
                    {
                        "split": "mock",
                        "sample_index": sample,
                        "component_slot": slot,
                        "matched_slot": slot,
                        "presence_true": present,
                        "presence_prob": float(present),
                        "presence_pred": present,
                        "type_true": type_vocab[true_type_id] if true_type_id >= 0 else "",
                        "type_pred": type_vocab[pred_type_id],
                        "type_correct": int(true_type_id == pred_type_id) if present else "",
                        "center_x_true": float(true[0]),
                        "center_x_pred": float(pred[0]),
                        "center_y_true": float(true[1]),
                        "center_y_pred": float(pred[1]),
                        "axis_x_true": float(true[2]),
                        "axis_x_pred": float(pred[2]),
                        "axis_y_true": float(true[3]),
                        "axis_y_pred": float(pred[3]),
                        "depth_true": float(true[4]),
                        "depth_pred": float(pred[4]),
                        "rotation_true": float(true[5]),
                        "rotation_pred": float(pred[5]),
                        "center_error": float(np.linalg.norm(pred[:2] - true[:2])) if present else "",
                        "axis_error": 0.0 if present else "",
                        "depth_error": 0.0 if present else "",
                        "rotation_error": 0.0 if present else "",
                        "target_schema": "center_x|center_y|axis_x|axis_y|depth_or_shape_param|rotation_angle",
                        "type_vocab": "|".join(type_vocab),
                    }
                )


def _write_mask_metrics(path: Path, n: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "sample_index",
                "pred_mask_iou",
                "pred_dice",
                "oracle_mask_iou",
                "oracle_gap",
                "target_area",
                "pred_area",
                "area_diff",
                "type_sequence_true",
                "type_sequence_pred",
            ],
        )
        writer.writeheader()
        for sample in range(n):
            writer.writerow(
                {
                    "split": "mock",
                    "sample_index": sample,
                    "pred_mask_iou": 0.8 - sample * 0.2,
                    "pred_dice": 0.9,
                    "oracle_mask_iou": 0.95,
                    "oracle_gap": 0.15 + sample * 0.2,
                    "target_area": 100 + sample * 20,
                    "pred_area": 100,
                    "area_diff": 0,
                    "type_sequence_true": "rectangular_notch",
                    "type_sequence_pred": "rectangular_notch",
                }
            )


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        x = np.linspace(-0.04, 0.04, 5).astype(np.float32)
        y = np.linspace(-0.01, 0.01, 5).astype(np.float32)
        masks = np.zeros((2, len(y), len(x)), dtype=np.float32)
        continuous_true = np.zeros((2, 3, 6), dtype=np.float32)
        continuous_pred = np.zeros((2, 3, 6), dtype=np.float32)
        presence = np.zeros((2, 3), dtype=np.float32)
        type_targets = np.full((2, 3), -1, dtype=np.int64)
        for sample in range(2):
            presence[sample, 0] = 1.0
            type_targets[sample, 0] = sample % 2
            continuous_true[sample, 0] = [0.0, 0.0, 0.02, 0.01, 0.1, 0.0]
            continuous_pred[sample, 0] = [0.02 * (sample + 1), 0.005, 0.02, 0.01, 0.1, 0.0]
        npz_path = base / "mock.npz"
        targets_path = base / "targets.npz"
        pred_path = base / "predictions.csv"
        mask_metrics = base / "mask_metrics.csv"
        out = base / "out"
        np.savez(npz_path, masks=masks, x=x, y=y, geometry_units=np.array("m"))
        np.savez(
            targets_path,
            sample_indices=np.arange(2),
            continuous_targets=continuous_true,
            type_targets=type_targets,
            presence_targets=presence,
            target_schema=np.array(
                ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"],
                dtype="U64",
            ),
            type_vocab=np.array(["rectangular_notch", "rotated_rect"], dtype="U64"),
        )
        _write_prediction_csv(pred_path, continuous_true, continuous_pred, presence, type_targets)
        _write_mask_metrics(mask_metrics, 2)
        rc = main(
            [
                "--npz-path",
                str(npz_path),
                "--targets-path",
                str(targets_path),
                "--predictions-csv",
                str(pred_path),
                "--mask-metrics-csv",
                str(mask_metrics),
                "--output-dir",
                str(out),
                "--split",
                "mock",
            ]
        )
        assert rc == 0
        for name in [
            "per_component_center_errors.csv",
            "per_sample_center_error_summary.csv",
            "grouped_center_errors.csv",
            "center_error_correlation.csv",
            "summary.md",
        ]:
            assert (out / name).exists(), name
        with (out / "per_component_center_errors.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert np.isclose(float(rows[0]["center_x_error_grid"]), 1.0)
        assert np.isclose(float(rows[0]["center_y_error_grid"]), 1.0)
        assert np.isclose(float(rows[0]["center_x_error_axis_relative"]), 1.0)
        assert np.isclose(float(rows[0]["center_y_error_axis_relative"]), 0.5)
        with (out / "center_error_correlation.csv").open(newline="", encoding="utf-8") as handle:
            corr_rows = list(csv.DictReader(handle))
        assert {"pearson", "spearman"}.issubset(corr_rows[0].keys())
    print("COMSOL parametric center diagnostics smoke test passed.")


if __name__ == "__main__":
    main_test()

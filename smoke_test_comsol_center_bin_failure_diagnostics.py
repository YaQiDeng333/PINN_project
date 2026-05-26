"""Smoke test for center-bin failure diagnostics."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from comsol_center_bin_failure_diagnostics import main


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        pred = base / "pred"
        out = base / "out"
        pred.mkdir()
        (pred / "run_summary.md").write_text(
            "\n".join(
                [
                    "- `x_min`: 0.0",
                    "- `y_min`: 0.0",
                    "- `dx`: 1.0",
                    "- `dy`: 1.0",
                    "- `bin_width_x`: 2.0",
                    "- `bin_width_y`: 2.0",
                    "- `center_x_bins`: 4",
                    "- `center_y_bins`: 4",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        prediction_rows = [
            {
                "split": "val",
                "sample_index": 0,
                "component_slot": 0,
                "matched_slot": 0,
                "presence_true": 1,
                "presence_prob": 1,
                "presence_pred": 1,
                "type_true": "rect",
                "type_pred": "rect",
                "center_x_true": 1.0,
                "center_x_pred": 1.1,
                "center_y_true": 1.0,
                "center_y_pred": 1.1,
                "axis_x_true": 0.2,
                "axis_x_pred": 0.2,
                "axis_y_true": 0.3,
                "axis_y_pred": 0.3,
                "rotation_true": 0,
                "rotation_pred": 0,
                "center_error": 0.1,
                "axis_error": 0.0,
                "rotation_error": 0.0,
            },
            {
                "split": "val",
                "sample_index": 1,
                "component_slot": 0,
                "matched_slot": 0,
                "presence_true": 1,
                "presence_prob": 1,
                "presence_pred": 1,
                "type_true": "rect",
                "type_pred": "rect",
                "center_x_true": 1.0,
                "center_x_pred": 3.1,
                "center_y_true": 1.0,
                "center_y_pred": 1.1,
                "axis_x_true": 0.2,
                "axis_x_pred": 0.2,
                "axis_y_true": 0.3,
                "axis_y_pred": 0.3,
                "rotation_true": 20,
                "rotation_pred": 20,
                "center_error": 2.1,
                "axis_error": 0.0,
                "rotation_error": 0.0,
            },
            {
                "split": "val",
                "sample_index": 2,
                "component_slot": 0,
                "matched_slot": 0,
                "presence_true": 1,
                "presence_prob": 1,
                "presence_pred": 1,
                "type_true": "rot",
                "type_pred": "rot",
                "center_x_true": 1.0,
                "center_x_pred": 1.1,
                "center_y_true": 1.0,
                "center_y_pred": 3.1,
                "axis_x_true": 0.2,
                "axis_x_pred": 0.2,
                "axis_y_true": 0.3,
                "axis_y_pred": 0.3,
                "rotation_true": 60,
                "rotation_pred": 60,
                "center_error": 2.1,
                "axis_error": 0.0,
                "rotation_error": 0.0,
            },
        ]
        metric_rows = [
            {
                "split": "val",
                "sample_index": 0,
                "pred_mask_iou": 0.8,
                "pred_dice": 0.9,
                "oracle_mask_iou": 0.9,
                "oracle_gap": 0.1,
                "target_area": 400,
                "pred_area": 410,
                "area_diff": 10,
                "type_sequence_true": "rect",
                "type_sequence_pred": "rect",
            },
            {
                "split": "val",
                "sample_index": 1,
                "pred_mask_iou": 0.3,
                "pred_dice": 0.4,
                "oracle_mask_iou": 0.9,
                "oracle_gap": 0.6,
                "target_area": 900,
                "pred_area": 700,
                "area_diff": -200,
                "type_sequence_true": "rect",
                "type_sequence_pred": "rect",
            },
            {
                "split": "val",
                "sample_index": 2,
                "pred_mask_iou": 0.4,
                "pred_dice": 0.5,
                "oracle_mask_iou": 0.9,
                "oracle_gap": 0.5,
                "target_area": 1200,
                "pred_area": 1100,
                "area_diff": -100,
                "type_sequence_true": "rot",
                "type_sequence_pred": "rot",
            },
        ]
        for split in ["val", "test"]:
            _write_csv(pred / f"{split}_predictions.csv", prediction_rows)
            _write_csv(pred / f"{split}_prediction_mask_metrics.csv", metric_rows)
        rc = main(["--prediction-dir", str(pred), "--output-dir", str(out), "--label", "mock"])
        assert rc == 0
        for name in [
            "per_component_center_bin_errors.csv",
            "per_sample_center_bin_errors.csv",
            "grouped_center_bin_errors.csv",
            "worst_samples.csv",
            "summary.md",
        ]:
            assert (out / name).exists(), name
        grouped = (out / "grouped_center_bin_errors.csv").read_text(encoding="utf-8")
        assert "x_bin_correct" in grouped
        assert "y_bin_correct" in grouped
    print("COMSOL center-bin failure diagnostics smoke test passed.")


if __name__ == "__main__":
    main_test()

"""Smoke test for comsol_parametric_grouped_diagnostics.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from comsol_parametric_grouped_diagnostics import main


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        pred = base / "pred"
        pred.mkdir()
        for split in ["train", "val", "test"]:
            component_rows = []
            mask_rows = []
            for sample in range(3):
                for slot in range(3):
                    present = 1 if slot <= sample else 0
                    component_rows.append(
                        {
                            "split": split,
                            "sample_index": sample,
                            "component_slot": slot,
                            "presence_true": present,
                            "presence_prob": 0.8 if present else 0.1,
                            "presence_pred": present,
                            "type_true": "rectangular_notch" if slot % 2 == 0 else "rotated_rect",
                            "type_pred": "rectangular_notch",
                            "type_correct": 1 if slot % 2 == 0 else 0,
                            "center_error": 0.01 * slot,
                            "axis_error": 0.02 * slot,
                            "depth_error": 0.001 * slot,
                            "rotation_error": 4 + 10 * slot,
                        }
                    )
                mask_rows.append(
                    {
                        "split": split,
                        "sample_index": sample,
                        "pred_mask_iou": 0.5 - 0.1 * sample,
                        "pred_dice": 0.6 - 0.1 * sample,
                        "oracle_mask_iou": 0.8,
                        "oracle_gap": 0.3 + 0.1 * sample,
                        "target_area": 100 + 50 * sample,
                        "pred_area": 90 + 10 * sample,
                        "area_diff": -10 - 40 * sample,
                        "type_sequence_true": "rectangular_notch|rotated_rect",
                        "type_sequence_pred": "rectangular_notch",
                    }
                )
            _write_csv(pred / f"{split}_predictions.csv", component_rows)
            _write_csv(pred / f"{split}_prediction_mask_metrics.csv", mask_rows)
        out = base / "out"
        rc = main(["--prediction-dir", str(pred), "--output-dir", str(out)])
        assert rc == 0
        for name in [
            "grouped_by_type.csv",
            "grouped_by_slot.csv",
            "grouped_by_rotation_bin.csv",
            "grouped_by_area_bin.csv",
            "worst_samples.csv",
            "summary.md",
        ]:
            assert (out / name).exists(), name
        text = (out / "summary.md").read_text(encoding="utf-8")
        assert "rotation" in text.lower()
    print("COMSOL parametric grouped diagnostics smoke test passed.")


if __name__ == "__main__":
    main_test()

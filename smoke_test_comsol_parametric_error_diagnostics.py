"""Smoke test for comsol_parametric_error_diagnostics.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from comsol_parametric_error_diagnostics import main


def _write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        run_dir = base / "run"
        oracle_dir = base / "oracle"
        metric_row = {
            "split": "train",
            "presence_accuracy": "1.0",
            "type_accuracy_present": "0.75",
            "continuous_mae_mean": "0.2",
            "center_mae": "0.01",
            "axis_mae": "0.02",
            "rotation_mae": "8.0",
            "depth_mae": "0.03",
            "param_mask_iou": "0.4",
            "param_mask_dice": "0.55",
        }
        for name, split in [("metrics.csv", "train"), ("eval_metrics.csv", "val"), ("test_metrics.csv", "test")]:
            row = dict(metric_row)
            row["split"] = split
            _write_csv(run_dir / name, row)
            _write_csv(
                oracle_dir / split / "oracle_parametric_mask_aggregate.csv",
                {
                    "split_or_dataset": split,
                    "samples": "2",
                    "avg_oracle_iou": "0.8",
                    "min_oracle_iou": "0.7",
                    "max_oracle_iou": "0.9",
                    "avg_oracle_dice": "0.88",
                    "avg_target_area": "10",
                    "avg_raster_area": "11",
                    "avg_abs_area_diff": "1",
                },
            )
        out = base / "out"
        rc = main(["--run-dir", str(run_dir), "--oracle-dir", str(oracle_dir), "--output-dir", str(out), "--label", "mock"])
        assert rc == 0
        assert (out / "parametric_error_summary.csv").exists()
        assert (out / "oracle_gap_summary.csv").exists()
        summary = (out / "summary.md").read_text(encoding="utf-8")
        assert "oracle_gap" in summary
        assert "type_acc" in summary
    print("COMSOL parametric error diagnostics smoke test passed.")


if __name__ == "__main__":
    main_test()

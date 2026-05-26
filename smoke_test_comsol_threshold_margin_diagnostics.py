"""Smoke test for threshold-margin diagnostics."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    repo_root = Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_dir = tmp_path / "run"
        output_dir = tmp_path / "diagnostics"
        run_dir.mkdir()

        metric_rows = [
            {
                "sample_index": "0",
                "defect_iou": "0.0",
                "defect_area_pred": "0",
                "defect_area_label": "12",
                "mu_mse": "100.0",
                "mu_mae": "10.0",
            }
        ]
        _write_csv(run_dir / "metrics.csv", metric_rows)
        _write_csv(run_dir / "eval_metrics.csv", metric_rows)
        _write_csv(run_dir / "test_metrics.csv", metric_rows)
        _write_csv(
            run_dir / "training_history.csv",
            [
                {
                    "phase": "finetune",
                    "step": "1",
                    "total_loss": "1.0",
                    "bce_loss": "0.2",
                    "dice_loss": "0.8",
                    "mu_mse_loss": "100.0",
                    "area_loss": "0.01",
                    "batch_iou": "0.0",
                    "batch_area_pred": "0.0",
                    "batch_area_label": "12.0",
                    "pred_area_soft_mean": "7.0",
                    "true_area_mean": "12.0",
                    "mean_mu": "620.0",
                    "min_mu": "510.0",
                    "max_mu": "800.0",
                    "mean_soft_defect": "0.08",
                    "mask_bce_mode": "bce",
                    "point_sampling_mode": "random",
                    "train_point_subsample": "50",
                }
            ],
        )

        cmd = [
            sys.executable,
            str(repo_root / "comsol_threshold_margin_diagnostics.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise AssertionError(f"diagnostics failed with return code {result.returncode}")

        summary_csv = output_dir / "threshold_margin_summary.csv"
        summary_md = output_dir / "summary.md"
        if not summary_csv.exists():
            raise AssertionError("threshold_margin_summary.csv was not created")
        if not summary_md.exists():
            raise AssertionError("summary.md was not created")
        with summary_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows[0]["soft_hard_mismatch"] != "True":
            raise AssertionError("expected soft-hard mismatch")
        if rows[0]["no_threshold_crossing"] != "True":
            raise AssertionError("expected no-threshold-crossing")

    print("COMSOL threshold margin diagnostics smoke test passed.")


if __name__ == "__main__":
    main()

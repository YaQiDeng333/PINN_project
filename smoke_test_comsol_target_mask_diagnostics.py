"""Smoke test for COMSOL target/mask diagnostics."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def main():
    repo_root = Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz_path = tmp_path / "targets.npz"
        output_dir = tmp_path / "diagnostics"
        x = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
        signals = np.zeros((4, 3, 20), dtype=np.float32)
        mu_maps = np.full((4, 10, 20), 1000.0, dtype=np.float32)
        mu_maps[:, 4:6, 8:12] = 1.0
        masks = (mu_maps < 500.0).astype(np.float32)
        masks[3, 0, 0] = 1.0
        masks[3, 4, 8] = 0.0
        np.savez(npz_path, signals=signals, mu_maps=mu_maps, masks=masks, x=x, y=y)

        cmd = [
            sys.executable,
            str(repo_root / "comsol_target_mask_diagnostics.py"),
            "--npz-path",
            str(npz_path),
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise AssertionError(f"diagnostics failed with return code {result.returncode}")

        per_sample_path = output_dir / "per_sample_mask_consistency.csv"
        aggregate_path = output_dir / "aggregate_mask_consistency.csv"
        summary_path = output_dir / "summary.md"
        for path in [per_sample_path, aggregate_path, summary_path]:
            if not path.exists():
                raise AssertionError(f"{path.name} was not created")

        with per_sample_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 4:
            raise AssertionError(f"expected 4 per-sample rows, got {len(rows)}")
        if float(rows[0]["mask_iou"]) != 1.0:
            raise AssertionError("expected a consistent sample with mask_iou=1")
        if int(rows[3]["mismatch_count"]) <= 0:
            raise AssertionError("expected inconsistent sample to have mismatch_count > 0")

    print("COMSOL target/mask diagnostics smoke test passed.")


if __name__ == "__main__":
    main()

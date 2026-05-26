from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from comsol_v3_hard_case_grouped_diagnostics import main


def write_mock_run(root: Path) -> None:
    run = root / "run"
    run.mkdir()
    pd.DataFrame(
        [
            {
                "split": "val",
                "sample_index": 0,
                "component_slot": 0,
                "presence_true": 1,
                "center_x_true": 20.0,
                "center_x_pred": 21.0,
                "center_y_true": 10.0,
                "center_y_pred": 10.0,
                "type_true": "rectangular_notch",
                "type_pred": "rectangular_notch",
            },
            {
                "split": "val",
                "sample_index": 1,
                "component_slot": 0,
                "presence_true": 1,
                "center_x_true": 20.0,
                "center_x_pred": 60.0,
                "center_y_true": 10.0,
                "center_y_pred": 50.0,
                "type_true": "rectangular_notch",
                "type_pred": "rectangular_notch",
            },
        ]
    ).to_csv(run / "val_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "val",
                "sample_index": 0,
                "pred_mask_iou": 0.8,
                "pred_dice": 0.88,
                "oracle_mask_iou": 1.0,
                "oracle_gap": 0.2,
                "target_area": 10,
                "pred_area": 11,
                "area_diff": 1,
            },
            {
                "split": "val",
                "sample_index": 1,
                "pred_mask_iou": 0.1,
                "pred_dice": 0.18,
                "oracle_mask_iou": 1.0,
                "oracle_gap": 0.9,
                "target_area": 12,
                "pred_area": 9,
                "area_diff": -3,
            },
        ]
    ).to_csv(run / "val_prediction_mask_metrics.csv", index=False)

    defect_root = root / "defects" / "val"
    defect_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"sample_index": 0, "hard_case_type": "x_bin_wrong_like"},
            {"sample_index": 1, "hard_case_type": "both_bins_wrong_like"},
        ]
    ).to_csv(defect_root / "defect_params.csv", index=False)

    npz_root = root / "npz"
    npz_root.mkdir()
    np.savez(
        npz_root / "val_mock.npz",
        x=np.linspace(0.0, 100.0, 101, dtype=np.float32),
        y=np.linspace(0.0, 100.0, 101, dtype=np.float32),
    )


def test_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_mock_run(root)
        out = root / "out"
        import sys

        old_argv = sys.argv
        sys.argv = [
            "comsol_v3_hard_case_grouped_diagnostics.py",
            "--run",
            f"mock={root / 'run'}",
            "--defect-root",
            str(root / "defects"),
            "--npz-root",
            str(root / "npz"),
            "--output-dir",
            str(out),
            "--center-bin-size-cells",
            "8",
        ]
        try:
            assert main() == 0
        finally:
            sys.argv = old_argv
        for name in [
            "per_component_v3_hard_case_diagnostics.csv",
            "per_sample_v3_hard_case_diagnostics.csv",
            "grouped_by_hard_case_type.csv",
            "worst_v3_samples.csv",
            "summary.md",
        ]:
            assert (out / name).exists(), name


if __name__ == "__main__":
    test_smoke()
    print("COMSOL V3 hard-case grouped diagnostics smoke test passed.")

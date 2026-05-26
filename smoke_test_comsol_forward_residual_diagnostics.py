"""Smoke test for comsol_forward_residual_diagnostics.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_forward_residual_diagnostics import main


def _write_fixture(base: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(42)
    n = 6
    signals = rng.normal(size=(n, 3, 20)).astype(np.float32)
    continuous = np.zeros((n, 3, 6), dtype=np.float32)
    presence = np.zeros((n, 3), dtype=np.float32)
    type_targets = np.full((n, 3), -1, dtype=np.int64)
    for i in range(n):
        presence[i, 0] = 1.0
        type_targets[i, 0] = i % 2
        continuous[i, 0] = [0.1 * i, 0.0, 0.2, 0.1, 0.05, float(i * 5)]
    npz_path = base / "mock.npz"
    targets_path = base / "targets.npz"
    np.savez(npz_path, signals=signals)
    np.savez(
        targets_path,
        continuous_targets=continuous,
        presence_targets=presence,
        type_targets=type_targets,
        sample_indices=np.arange(n),
        target_schema=np.asarray(
            ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"],
            dtype="U64",
        ),
        type_vocab=np.asarray(["rectangular_notch", "rotated_rect"], dtype="U64"),
    )
    pred_dir = base / "pred"
    pred_dir.mkdir()
    with (pred_dir / "val_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "sample_index",
            "component_slot",
            "presence_pred",
            "type_pred",
            "center_x_pred",
            "center_y_pred",
            "axis_x_pred",
            "axis_y_pred",
            "depth_pred",
            "rotation_pred",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            for slot in range(3):
                writer.writerow(
                    {
                        "split": "val",
                        "sample_index": i,
                        "component_slot": slot,
                        "presence_pred": float(presence[i, slot]),
                        "type_pred": "rectangular_notch" if type_targets[i, slot] <= 0 else "rotated_rect",
                        "center_x_pred": float(continuous[i, slot, 0]),
                        "center_y_pred": float(continuous[i, slot, 1]),
                        "axis_x_pred": float(continuous[i, slot, 2]),
                        "axis_y_pred": float(continuous[i, slot, 3]),
                        "depth_pred": float(continuous[i, slot, 4]),
                        "rotation_pred": float(continuous[i, slot, 5]),
                    }
                )
    return npz_path, targets_path, pred_dir


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        npz_path, targets_path, pred_dir = _write_fixture(base)
        out = base / "out"
        rc = main(
            [
                "--npz-path",
                str(npz_path),
                "--targets-path",
                str(targets_path),
                "--prediction-dir",
                str(pred_dir),
                "--output-dir",
                str(out),
                "--split",
                "val",
                "--forward-steps",
                "5",
            ]
        )
        assert rc == 0
        for name in ["per_sample_forward_residual.csv", "aggregate_forward_residual.csv", "residual_sensitivity_summary.md"]:
            assert (out / name).exists(), name
        text = (out / "aggregate_forward_residual.csv").read_text(encoding="utf-8")
        assert "true_geometry" in text
        assert "rotation_perturbed_geometry" in text
        assert "axis_scaled_geometry" in text
    print("COMSOL forward residual diagnostics smoke test passed.")


if __name__ == "__main__":
    main_test()

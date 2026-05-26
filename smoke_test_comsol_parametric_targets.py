"""Smoke test for comsol_parametric_targets.py."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from comsol_parametric_targets import build_parametric_targets, load_defect_params, main


def _write_mock_csv(path: Path) -> None:
    rows = [
        {
            "sample_index": 0,
            "split": "train",
            "source_component_json": json.dumps(
                [
                    {
                        "component_type": "rotated_rect",
                        "center_x_m": 0.2,
                        "center_y_m": 0.0,
                        "length_m": 0.04,
                        "width_m": 0.02,
                        "depth_m": 0.003,
                        "angle_deg": 15.0,
                    },
                    {
                        "component_type": "rectangular_notch",
                        "center_x_m": 0.1,
                        "center_y_m": -0.1,
                        "length_m": 0.03,
                        "width_m": 0.01,
                        "depth_m": 0.002,
                        "angle_deg": 0.0,
                    },
                ]
            ),
        },
        {
            "sample_index": 1,
            "split": "train",
            "source_component_json": json.dumps(
                [
                    {
                        "component_type": "rectangular_notch",
                        "center_x_m": -0.2,
                        "center_y_m": 0.1,
                        "length_m": 0.05,
                        "width_m": 0.02,
                        "depth_m": 0.004,
                        "angle_deg": 0.0,
                    }
                ]
            ),
        },
        {
            "sample_index": 2,
            "split": "train",
            "source_component_json": json.dumps(
                [
                    {
                        "component_type": "rotated_rect",
                        "center_x_m": -0.1,
                        "center_y_m": 0.0,
                        "length_m": 0.02,
                        "width_m": 0.01,
                        "depth_m": 0.001,
                        "angle_deg": 30.0,
                    },
                    {
                        "component_type": "rotated_rect",
                        "center_x_m": 0.0,
                        "center_y_m": 0.0,
                        "length_m": 0.02,
                        "width_m": 0.01,
                        "depth_m": 0.001,
                        "angle_deg": 30.0,
                    },
                    {
                        "component_type": "rotated_rect",
                        "center_x_m": 0.1,
                        "center_y_m": 0.0,
                        "length_m": 0.02,
                        "width_m": 0.01,
                        "depth_m": 0.001,
                        "angle_deg": 30.0,
                    },
                ]
            ),
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "split", "source_component_json"])
        writer.writeheader()
        writer.writerows(rows)


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "defect_params.csv"
        npz_path = tmp_path / "mock.npz"
        out_dir = tmp_path / "out"
        _write_mock_csv(csv_path)
        np.savez(npz_path, signals=np.zeros((3, 3, 20), dtype=np.float32))

        rows = load_defect_params(npz_path, csv_path)
        targets = build_parametric_targets(rows, max_components=3)
        assert targets["continuous_targets"].shape == (3, 3, 6)
        refined = build_parametric_targets(rows, max_components=3, angle_encoding="sincos")
        assert refined["continuous_targets"].shape == (3, 3, 7)
        assert list(refined["target_schema"])[-2:] == ["rotation_sin", "rotation_cos"]
        assert np.isclose(refined["continuous_targets"][0, 0, -1], 1.0)
        assert targets["presence_targets"].shape == (3, 3)
        assert targets["type_targets"].shape == (3, 3)
        assert list(targets["type_vocab"]) == ["rectangular_notch", "rotated_rect"]
        assert targets["presence_targets"][1].tolist() == [1.0, 0.0, 0.0]
        assert np.isclose(targets["continuous_targets"][0, 0, 0], 0.1)
        assert np.isclose(targets["continuous_targets"][0, 1, 0], 0.2)

        rc = main(
            [
                "--npz-path",
                str(npz_path),
                "--defect-params-csv",
                str(csv_path),
                "--output-dir",
                str(out_dir),
                "--max-components",
                "3",
                "--angle-encoding",
                "sincos",
                "--normalize-continuous",
            ]
        )
        assert rc == 0
        assert (out_dir / "parametric_targets.npz").exists()
        assert (out_dir / "continuous_normalization_stats.npz").exists()
        assert (out_dir / "parametric_target_summary.md").exists()
        assert (out_dir / "parametric_target_preview.csv").exists()
        with np.load(out_dir / "parametric_targets.npz", allow_pickle=True) as data:
            assert bool(data["continuous_targets_normalized"])
            assert data["continuous_targets_raw"].shape == (3, 3, 6)
            assert data["continuous_targets"].shape == (3, 3, 7)

    print("COMSOL parametric targets smoke test passed.")


if __name__ == "__main__":
    main_test()

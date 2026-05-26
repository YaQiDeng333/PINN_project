"""Smoke test for COMSOL V1/V2 target distribution diagnostics."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_v1_v2_target_distribution_diagnostics import main


def _make_npz(path: Path, samples: int, area_scale: int) -> None:
    signals = np.zeros((samples, 3, 20), dtype=np.float32)
    mu_maps = np.full((samples, 10, 20), 1000.0, dtype=np.float32)
    masks = np.zeros((samples, 10, 20), dtype=np.float32)
    for i in range(samples):
        height = 2 + area_scale
        width = 3 + i
        mu_maps[i, 2 : 2 + height, 4 : 4 + width] = 1.0
        masks[i] = (mu_maps[i] < 500.0).astype(np.float32)
    defect_params = np.zeros(
        samples,
        dtype=[
            ("sample_index", "i4"),
            ("defect_type", "U32"),
            ("rotation_angle", "f4"),
            ("boundary_irregularity_level", "U16"),
            ("defect_center_x", "f4"),
            ("defect_center_y", "f4"),
            ("defect_axis_x", "f4"),
            ("defect_axis_y", "f4"),
            ("defect_depth_or_shape_param", "f4"),
            ("defect_mu", "f4"),
        ],
    )
    for i in range(samples):
        defect_params[i]["sample_index"] = i
        defect_params[i]["defect_type"] = "mock_rect" if i % 2 == 0 else "mock_rotated"
        defect_params[i]["rotation_angle"] = float(i * 5)
        defect_params[i]["boundary_irregularity_level"] = "near" if i % 2 == 0 else "far"
        defect_params[i]["defect_center_x"] = float(i)
        defect_params[i]["defect_center_y"] = float(-i)
        defect_params[i]["defect_axis_x"] = 1.0 + i
        defect_params[i]["defect_axis_y"] = 2.0 + i
        defect_params[i]["defect_depth_or_shape_param"] = 0.1 + i
        defect_params[i]["defect_mu"] = 1.0
    np.savez(
        path,
        signals=signals,
        mu_maps=mu_maps,
        masks=masks,
        x=np.linspace(-1, 1, 20, dtype=np.float32),
        y=np.linspace(-1, 1, 10, dtype=np.float32),
        defect_params=defect_params,
    )


def test_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        paths = {}
        for version, scale in [("v1", 1), ("v2", 2)]:
            for split, samples in [("train", 4), ("val", 2), ("test", 2)]:
                path = tmp_path / f"{version}_{split}.npz"
                _make_npz(path, samples=samples, area_scale=scale)
                paths[f"{version}_{split}"] = path
        out = tmp_path / "out"
        rc = main(
            [
                "--v1-train-npz",
                str(paths["v1_train"]),
                "--v1-val-npz",
                str(paths["v1_val"]),
                "--v1-test-npz",
                str(paths["v1_test"]),
                "--v2-train-npz",
                str(paths["v2_train"]),
                "--v2-val-npz",
                str(paths["v2_val"]),
                "--v2-test-npz",
                str(paths["v2_test"]),
                "--output-dir",
                str(out),
            ]
        )
        assert rc == 0
        for name in [
            "per_sample_target_distribution.csv",
            "aggregate_target_distribution.csv",
            "defect_param_distribution.md",
            "summary.md",
        ]:
            assert (out / name).exists()
        with (out / "aggregate_target_distribution.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert {(row["dataset_version"], row["split"]) for row in rows} == {
            ("v1", "train"),
            ("v1", "val"),
            ("v1", "test"),
            ("v2", "train"),
            ("v2", "val"),
            ("v2", "test"),
        }
        assert all(float(row["avg_mask_iou"]) == 1.0 for row in rows)
        assert float([row for row in rows if row["dataset_version"] == "v2" and row["split"] == "train"][0]["mean_label_area_ratio"]) > float(
            [row for row in rows if row["dataset_version"] == "v1" and row["split"] == "train"][0]["mean_label_area_ratio"]
        )


if __name__ == "__main__":
    test_smoke()
    print("COMSOL V1/V2 target distribution diagnostics smoke test passed.")

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    npz_root = root / "npz"
    targets_root = root / "targets"
    defect_root = root / "defects"
    npz_root.mkdir()
    for split in ("train", "val", "test"):
        (targets_root / split).mkdir(parents=True)
        (defect_root / split).mkdir(parents=True)
        x = np.linspace(-0.04, 0.04, 20, dtype=np.float32)
        y = np.linspace(-0.01, 0.01, 10, dtype=np.float32)
        masks = np.zeros((2, len(y), len(x)), dtype=np.uint8)
        masks[:, 3:6, 8:12] = 1
        mu_maps = np.where(masks > 0, 100.0, 1000.0).astype(np.float32)
        signals = np.zeros((2, 3, 20), dtype=np.float32)
        signals[0, :, 8:12] = 1e-7
        signals[1, :, 10:14] = 2e-7
        np.savez_compressed(
            npz_root / f"{split}_fixture.npz",
            signals=signals,
            masks=masks,
            mu_maps=mu_maps,
            x=x,
            y=y,
            source_sample_ids=np.array(["a", "b"]),
            source_global_indices=np.array([0, 1]),
            csv_sample_indices=np.array([0, 1]),
        )
        continuous = np.zeros((2, 3, 6), dtype=np.float32)
        continuous[:, 0, 0] = [float(x[9]), float(x[11])]
        continuous[:, 0, 1] = float(y[4])
        continuous[:, 0, 2] = 0.016
        continuous[:, 0, 3] = 0.006
        continuous[:, 0, 4] = 1.0
        presence = np.zeros((2, 3), dtype=np.float32)
        presence[:, 0] = 1.0
        np.savez_compressed(
            targets_root / split / "parametric_targets.npz",
            sample_indices=np.arange(2),
            continuous_targets=continuous,
            continuous_targets_raw=continuous,
            continuous_targets_unscaled=continuous,
            presence_targets=presence,
            type_targets=np.zeros((2, 3), dtype=np.int64),
            target_schema=np.array(["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"]),
            raw_target_schema=np.array(["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"]),
            type_vocab=np.array(["rectangular_notch"]),
            component_counts=np.ones(2, dtype=np.int64),
            angle_encoding=np.array("raw"),
            continuous_targets_mean=np.zeros(6, dtype=np.float32),
            continuous_targets_std=np.ones(6, dtype=np.float32),
            continuous_targets_normalized=np.array(False),
        )
        pd.DataFrame(
            {
                "sample_index": [0, 1],
                "split": [split, split],
                "hard_case_type": ["a", "b"],
                "defect_center_x": [float(x[9]), float(x[11])],
                "defect_center_y": [float(y[4]), float(y[4])],
                "defect_axis_x": [0.016, 0.016],
                "defect_axis_y": [0.006, 0.006],
                "defect_depth_or_shape_param": [1.0, 1.0],
                "rotation_angle": [0.0, 0.0],
            }
        ).to_csv(defect_root / split / "defect_params.csv", index=False)
    return npz_root, targets_root, defect_root


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        npz_root, targets_root, defect_root = _write_fixture(root)
        out = root / "out"
        subprocess.run(
            [
                sys.executable,
                "comsol_v3_signal_target_sanity.py",
                "--npz-root",
                str(npz_root),
                "--targets-root",
                str(targets_root),
                "--defect-root",
                str(defect_root),
                "--output-dir",
                str(out),
                "--center-bin-size-cells",
                "4",
            ],
            check=True,
        )
        for name in [
            "per_sample_signal_target_sanity.csv",
            "split_signal_target_sanity.csv",
            "sanity_stats.json",
            "summary.md",
        ]:
            assert (out / name).exists(), name
        split = pd.read_csv(out / "split_signal_target_sanity.csv")
        assert bool(split["center_bin_targets_in_range"].all())
        assert int(split["mask_mu_threshold_mismatch_pixels"].sum()) == 0
    print("COMSOL V3 signal-target sanity smoke test passed.")


if __name__ == "__main__":
    main()

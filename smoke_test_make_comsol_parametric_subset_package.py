from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        signals = np.arange(8 * 3 * 5, dtype=np.float32).reshape(8, 3, 5)
        masks = np.zeros((8, 4, 5), dtype=np.uint8)
        mu_maps = np.where(masks > 0, 100.0, 1000.0).astype(np.float32)
        npz_path = root / "input.npz"
        np.savez_compressed(
            npz_path,
            signals=signals,
            masks=masks,
            mu_maps=mu_maps,
            x=np.linspace(-1, 1, 5, dtype=np.float32),
            y=np.linspace(-1, 1, 4, dtype=np.float32),
            source_sample_ids=np.array([f"s{i}" for i in range(8)]),
            source_global_indices=np.arange(100, 108),
            csv_sample_indices=np.arange(8),
            metadata_json=np.array("{}"),
        )
        targets_path = root / "targets.npz"
        np.savez_compressed(
            targets_path,
            sample_indices=np.arange(8),
            continuous_targets=np.ones((8, 3, 6), dtype=np.float32),
            continuous_targets_raw=np.ones((8, 3, 6), dtype=np.float32) * 2,
            continuous_targets_unscaled=np.ones((8, 3, 6), dtype=np.float32) * 3,
            presence_targets=np.ones((8, 3), dtype=np.float32),
            type_targets=np.zeros((8, 3), dtype=np.int64),
            component_counts=np.ones(8, dtype=np.int64),
            target_schema=np.array(["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"]),
            raw_target_schema=np.array(["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"]),
            type_vocab=np.array(["rectangular_notch"]),
            continuous_targets_mean=np.zeros(6, dtype=np.float32),
            continuous_targets_std=np.ones(6, dtype=np.float32),
            continuous_targets_normalized=np.array(False),
        )
        out_npz = root / "subset.npz"
        out_targets = root / "subset_targets.npz"
        subprocess.run(
            [
                sys.executable,
                "make_comsol_parametric_subset_package.py",
                "--npz-path",
                str(npz_path),
                "--targets-path",
                str(targets_path),
                "--indices",
                "0,2,5",
                "--output-npz",
                str(out_npz),
                "--output-targets",
                str(out_targets),
            ],
            check=True,
        )
        with np.load(out_npz, allow_pickle=True) as data:
            assert data["signals"].shape == (3, 3, 5)
            assert data["masks"].shape == (3, 4, 5)
            assert data["x"].shape == (5,)
            assert data["csv_sample_indices"].tolist() == [0, 1, 2]
            assert data["source_sample_ids"].tolist() == ["s0", "s2", "s5"]
        with np.load(out_targets, allow_pickle=True) as data:
            assert data["continuous_targets"].shape == (3, 3, 6)
            assert data["sample_indices"].tolist() == [0, 1, 2]
            assert data["continuous_targets_mean"].shape == (6,)
    print("COMSOL parametric subset package smoke test passed.")


if __name__ == "__main__":
    main()

"""Smoke test for comsol_mfl_physics_features.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from comsol_mfl_physics_features import extract_physics_features, main


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        x = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
        signals = np.zeros((5, 3, 20), dtype=np.float32)
        for sample in range(5):
            center = -0.4 + 0.2 * sample
            width = 0.12 + 0.02 * sample
            peak = np.exp(-((x - center) ** 2) / width)
            valley = -0.5 * np.exp(-((x - center - 0.25) ** 2) / (width * 1.2))
            ch0 = peak + valley
            signals[sample, 0] = ch0
            signals[sample, 1] = 0.5 * ch0
            signals[sample, 2] = 0.25 * ch0
        npz_path = base / "mock.npz"
        output_dir = base / "features"
        np.savez(npz_path, signals=signals, x=x)

        features, feature_names = extract_physics_features(signals, x, "peak_decay_width")
        assert features.shape == (5, len(feature_names))
        assert features.shape[1] > 36
        assert np.isfinite(features).all()
        ratio_name = "peak_abs_ch1_over_ch0"
        ratio_index = feature_names.index(ratio_name)
        assert np.allclose(features[:, ratio_index], 0.5, atol=0.05)

        rc = main(["--npz-path", str(npz_path), "--output-dir", str(output_dir), "--feature-mode", "peak_decay_width"])
        assert rc == 0
        with np.load(output_dir / "physics_features.npz", allow_pickle=True) as data:
            saved_features = data["features"]
            saved_names = [str(x) for x in data["feature_names"]]
        assert saved_features.shape == features.shape
        assert saved_names == feature_names
        assert (output_dir / "physics_features.csv").exists()
        assert (output_dir / "feature_summary.md").exists()
    print("COMSOL MFL physics features smoke test passed.")


if __name__ == "__main__":
    main_test()

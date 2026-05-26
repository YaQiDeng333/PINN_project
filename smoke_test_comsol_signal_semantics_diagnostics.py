"""Smoke test for COMSOL signal semantics diagnostics."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_signal_semantics_diagnostics import main


def _make_npz(path: Path, non_monotonic: bool) -> None:
    x = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    base = np.exp(-x**2 / 0.2).astype(np.float32)
    signals = np.zeros((4, 3, 20), dtype=np.float32)
    for i in range(4):
        signals[i, 0] = base * (1.0 + 0.1 * i)
        signals[i, 1] = base * (0.7 + 0.05 * i)
        signals[i, 2] = base * (0.4 + 0.02 * i)
    if non_monotonic:
        signals[0, 2] = base * 1.5
    np.savez(
        path,
        signals=signals,
        mu_maps=np.ones((4, 10, 20), dtype=np.float32),
        masks=np.zeros((4, 10, 20), dtype=np.float32),
        x=x,
        y=np.linspace(-1.0, 1.0, 10, dtype=np.float32),
        signal_channel_names=np.asarray(["low", "mid", "high"]),
        lift_off_values=np.asarray([0.5, 1.0, 2.0], dtype=np.float32),
        field_components=np.asarray(["Bz", "Bz", "Bz"]),
    )


def test_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        paths = {}
        for version, non_monotonic in [("v1", False), ("v2", True)]:
            for split in ["train", "val", "test"]:
                path = tmp_path / f"{version}_{split}.npz"
                _make_npz(path, non_monotonic=non_monotonic)
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
            "aggregate_signal_semantics.csv",
            "per_channel_signal_semantics.csv",
            "lift_off_decay_diagnostics.csv",
            "summary.md",
        ]:
            assert (out / name).exists()
        with (out / "lift_off_decay_diagnostics.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        v1_train = [row for row in rows if row["dataset_version"] == "v1" and row["split"] == "train"][0]
        v2_train = [row for row in rows if row["dataset_version"] == "v2" and row["split"] == "train"][0]
        assert float(v1_train["monotonic_decay_fraction"]) == 1.0
        assert float(v2_train["monotonic_decay_fraction"]) < 1.0


if __name__ == "__main__":
    test_smoke()
    print("COMSOL signal semantics diagnostics smoke test passed.")

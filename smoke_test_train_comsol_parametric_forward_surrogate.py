"""Smoke test for train_comsol_parametric_forward_surrogate.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from train_comsol_parametric_forward_surrogate import main


def _write_split(base: Path, name: str, n: int, signal_len: int = 20) -> tuple[Path, Path]:
    rng = np.random.default_rng(100 + n)
    signals = rng.normal(size=(n, 3, signal_len)).astype(np.float32)
    continuous = np.zeros((n, 3, 6), dtype=np.float32)
    presence = np.zeros((n, 3), dtype=np.float32)
    type_targets = np.full((n, 3), -1, dtype=np.int64)
    for i in range(n):
        presence[i, 0] = 1.0
        type_targets[i, 0] = i % 2
        continuous[i, 0] = [0.1 * i, 0.0, 0.2, 0.1, 0.05, float((i % 4) * 10)]
        if i % 3 == 0:
            presence[i, 1] = 1.0
            type_targets[i, 1] = 1
            continuous[i, 1] = [-0.1, 0.1, 0.15, 0.08, 0.03, 20.0]
    npz_path = base / f"{name}.npz"
    target_path = base / f"{name}_targets.npz"
    np.savez(npz_path, signals=signals)
    np.savez(
        target_path,
        continuous_targets=continuous,
        type_targets=type_targets,
        presence_targets=presence,
        sample_indices=np.arange(n),
        target_schema=np.array(
            ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"],
            dtype="U64",
        ),
        type_vocab=np.array(["rectangular_notch", "rotated_rect"], dtype="U64"),
    )
    return npz_path, target_path


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        train_npz, train_targets = _write_split(base, "train", 8)
        val_npz, val_targets = _write_split(base, "val", 4)
        test_npz, test_targets = _write_split(base, "test", 4)
        out = base / "out"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--num-layers",
                "2",
                "--max-components",
                "3",
                "--history-interval",
                "1",
            ]
        )
        assert rc == 0
        for name in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv", "training_history.csv", "run_summary.md"]:
            assert (out / name).exists(), name
        forbidden_suffixes = {".pt", ".pth", ".ckpt"}
        for path in out.iterdir():
            assert path.suffix not in forbidden_suffixes, path.name
    print("COMSOL parametric forward surrogate training smoke test passed.")


if __name__ == "__main__":
    main_test()

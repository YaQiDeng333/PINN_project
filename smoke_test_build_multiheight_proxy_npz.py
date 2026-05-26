"""Smoke test for synthetic multi-height proxy .npz builder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from conditional_dual_data_utils import get_conditional_batch, infer_signal_len, load_conditional_npz
from conditional_dual_models import ConditionalDualNet


def _assert_shape(array_or_tensor, expected_shape, name: str) -> None:
    actual = tuple(array_or_tensor.shape)
    if actual != tuple(expected_shape):
        raise AssertionError(f"{name} shape mismatch: expected {expected_shape}, got {actual}")


def _write_single_channel_npz(path: Path) -> None:
    x = np.linspace(-15.0, 15.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 10.0, 10, dtype=np.float32)
    signals = np.random.default_rng(65).normal(size=(4, 20)).astype(np.float32)
    mu_maps = np.full((4, 10, 20), 1000.0, dtype=np.float32)
    mu_maps[:, 4:6, 8:12] = 1.0
    np.savez(path, x=x, y=y, signals=signals, mu_maps=mu_maps)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_npz = tmp_path / "single_channel.npz"
        output_npz = tmp_path / "proxy.npz"
        _write_single_channel_npz(input_npz)

        cmd = [
            sys.executable,
            str(repo_root / "build_multiheight_proxy_npz.py"),
            "--input-npz",
            str(input_npz),
            "--output-npz",
            str(output_npz),
        ]
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise AssertionError(f"build_multiheight_proxy_npz.py failed with return code {result.returncode}")
        if not output_npz.exists():
            raise AssertionError("proxy output npz was not created")

        with np.load(output_npz, allow_pickle=False) as data:
            _assert_shape(data["signals"], (4, 3, 20), "proxy signals")
            _assert_shape(data["mu_maps"], (4, 10, 20), "mu_maps")
            _assert_shape(data["x"], (20,), "x")
            _assert_shape(data["y"], (10,), "y")
            if "source_type" not in data.files and "proxy_warning" not in data.files:
                raise AssertionError("proxy metadata should include source_type or proxy_warning")

        dataset = load_conditional_npz(output_npz)
        batch = get_conditional_batch(dataset, [0, 1, 2])
        _assert_shape(batch["signals"], (3, 60), "batch signals")
        if infer_signal_len(dataset) != 60:
            raise AssertionError("infer_signal_len should return 60 for [N, 3, 20] signals")

        model = ConditionalDualNet(signal_len=60, latent_dim=16, hidden_dim=32, num_layers=2)
        out = model(batch["signals"], batch["coords"])
        _assert_shape(out["latent"], (3, 16), "latent")
        _assert_shape(out["mu"], (3, 200, 1), "mu")
        _assert_shape(out["phi"], (3, 200, 1), "phi")

    print("Synthetic multi-height proxy builder smoke test passed.")


if __name__ == "__main__":
    main()

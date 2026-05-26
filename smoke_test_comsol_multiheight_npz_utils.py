"""Smoke test for COMSOL-style multi-height Bz NPZ utilities."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from comsol_multiheight_npz_utils import validate_comsol_multiheight_npz
from conditional_dual_data_utils import get_conditional_batch, infer_signal_len, load_conditional_npz
from conditional_dual_models import ConditionalDualNet


def _assert_shape(array_or_tensor, expected_shape, name):
    actual = tuple(array_or_tensor.shape)
    if actual != tuple(expected_shape):
        raise AssertionError(f"{name} shape mismatch: expected {expected_shape}, got {actual}")


def _expect_value_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {label}")


def _write_mock_comsol_npz(
    path: Path,
    include_signals: bool = True,
    include_target: bool = True,
    signals_shape=(5, 3, 20),
):
    x = np.linspace(-15.0, 15.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 10.0, 10, dtype=np.float32)
    signals = np.random.default_rng(66).normal(size=signals_shape).astype(np.float32)
    mu_maps = np.full((5, 10, 20), 1000.0, dtype=np.float32)
    mu_maps[:, 4:6, 8:12] = 1.0
    payload = {
        "x": x,
        "y": y,
        "signal_channel_names": np.array(["Bz_liftoff_0p5", "Bz_liftoff_1p0", "Bz_liftoff_2p0"]),
        "lift_off_values": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "field_components": np.array(["Bz", "Bz", "Bz"]),
        "source_type": np.array("mock_comsol_multiheight"),
        "signal_flatten_order": np.array("channels_first"),
    }
    if include_signals:
        payload["signals"] = signals
    if include_target:
        payload["mu_maps"] = mu_maps
    np.savez(path, **payload)


def main():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz_path = tmp_path / "mock_comsol_multiheight.npz"
        _write_mock_comsol_npz(npz_path)

        summary = validate_comsol_multiheight_npz(npz_path)
        if summary["num_samples"] != 5:
            raise AssertionError("num_samples should be 5")
        if summary["num_channels"] != 3:
            raise AssertionError("num_channels should be 3")
        if summary["signal_len"] != 20:
            raise AssertionError("signal_len should be 20")
        if not summary["has_mu_maps"]:
            raise AssertionError("has_mu_maps should be true")
        if not summary["has_x_y"]:
            raise AssertionError("has_x_y should be true")
        if summary["channel_names"][0] != "Bz_liftoff_0p5":
            raise AssertionError("channel_names should be readable")
        if [round(float(v), 1) for v in summary["lift_off_values"]] != [0.5, 1.0, 2.0]:
            raise AssertionError("lift_off_values should be readable")

        dataset = load_conditional_npz(npz_path)
        batch = get_conditional_batch(dataset, [0, 1, 2])
        _assert_shape(batch["signals"], (3, 60), "batch signals")
        if infer_signal_len(dataset) != 60:
            raise AssertionError("infer_signal_len should return 60")
        model = ConditionalDualNet(signal_len=60, latent_dim=16, hidden_dim=32, num_layers=2)
        out = model(batch["signals"], batch["coords"])
        _assert_shape(out["latent"], (3, 16), "latent")
        _assert_shape(out["mu"], (3, 200, 1), "mu")
        _assert_shape(out["phi"], (3, 200, 1), "phi")

        missing_signals = tmp_path / "missing_signals.npz"
        _write_mock_comsol_npz(missing_signals, include_signals=False)
        _expect_value_error(lambda: validate_comsol_multiheight_npz(missing_signals), "missing signals")

        missing_target = tmp_path / "missing_target.npz"
        _write_mock_comsol_npz(missing_target, include_target=False)
        _expect_value_error(lambda: validate_comsol_multiheight_npz(missing_target), "missing mu_maps and masks")

        two_dim_signals = tmp_path / "two_dim_signals.npz"
        _write_mock_comsol_npz(two_dim_signals, signals_shape=(5, 20))
        _expect_value_error(lambda: validate_comsol_multiheight_npz(two_dim_signals), "2D signals")

        one_channel = tmp_path / "one_channel.npz"
        _write_mock_comsol_npz(one_channel, signals_shape=(5, 1, 20))
        _expect_value_error(lambda: validate_comsol_multiheight_npz(one_channel), "C < 2")

    print("COMSOL multi-height NPZ utility smoke test passed.")


if __name__ == "__main__":
    main()

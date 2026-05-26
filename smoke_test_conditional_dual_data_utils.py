"""Smoke tests for conditional dual-network batch data utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from conditional_dual_data_utils import (
    get_conditional_batch,
    infer_signal_len,
    load_conditional_npz,
)
from conditional_dual_models import ConditionalDualNet


def _assert_shape(tensor, expected_shape, name):
    actual = tuple(tensor.shape)
    if actual != tuple(expected_shape):
        raise AssertionError(f"{name} shape mismatch: expected {expected_shape}, got {actual}")


def _expect_value_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {label}")


def _write_npz(path, include_signals=True, include_mu_maps=True, include_coords=True, multi_channel=False, include_masks=False):
    x = np.linspace(-15.0, 15.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 10.0, 10, dtype=np.float32)
    signal_shape = (4, 3, 20) if multi_channel else (4, 20)
    signals = np.random.default_rng(49).normal(size=signal_shape).astype(np.float32)
    mu_maps = np.full((4, 10, 20), 1000.0, dtype=np.float32)
    mu_maps[:, 4:6, 8:12] = 1.0
    masks = (mu_maps < 500.0).astype(np.float32)
    masks[:, 0, 0] = 1.0

    payload = {}
    if include_signals:
        payload["signals"] = signals
    if include_mu_maps:
        payload["mu_maps"] = mu_maps
    if include_masks:
        payload["masks"] = masks
    if include_coords:
        payload["x"] = x
        payload["y"] = y
    np.savez(path, **payload)


def main():
    torch.manual_seed(49)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz_path = tmp_path / "conditional_train.npz"
        _write_npz(npz_path)

        dataset = load_conditional_npz(npz_path)
        batch = get_conditional_batch(dataset, [0, 1, 2])

        _assert_shape(batch["signals"], (3, 20), "signals")
        _assert_shape(batch["coords"], (200, 2), "coords")
        _assert_shape(batch["mu_label"], (3, 200, 1), "mu_label")
        _assert_shape(batch["mask_label"], (3, 200, 1), "mask_label")
        if torch.sum(batch["mask_label"]).item() <= 0:
            raise AssertionError("mask_label should contain nonzero defect points")
        if batch["mask_source"] != "mu_threshold":
            raise AssertionError("default mask_source should be mu_threshold")

        masks_npz_path = tmp_path / "conditional_train_with_masks.npz"
        _write_npz(masks_npz_path, include_masks=True)
        masks_dataset = load_conditional_npz(masks_npz_path)
        threshold_batch = get_conditional_batch(masks_dataset, [0, 1, 2], mask_source="mu_threshold")
        masks_batch = get_conditional_batch(masks_dataset, [0, 1, 2], mask_source="masks")
        _assert_shape(masks_batch["mask_label"], (3, 200, 1), "masks mask_label")
        if masks_batch["mask_source"] != "masks":
            raise AssertionError("mask_source should be recorded as masks")
        if torch.sum(masks_batch["mask_label"]).item() <= torch.sum(threshold_batch["mask_label"]).item():
            raise AssertionError("provided masks should include the extra test mask point")

        if infer_signal_len(dataset) != 20:
            raise AssertionError("infer_signal_len should return 20")

        model = ConditionalDualNet(
            signal_len=20,
            latent_dim=16,
            hidden_dim=32,
            num_layers=2,
        )
        out = model(batch["signals"], batch["coords"])
        _assert_shape(out["latent"], (3, 16), "latent")
        _assert_shape(out["mu"], (3, 200, 1), "mu")
        _assert_shape(out["phi"], (3, 200, 1), "phi")

        multi_npz_path = tmp_path / "conditional_multi_channel_train.npz"
        _write_npz(multi_npz_path, multi_channel=True)
        multi_dataset = load_conditional_npz(multi_npz_path)
        multi_batch = get_conditional_batch(multi_dataset, [0, 1, 2])
        _assert_shape(multi_batch["signals"], (3, 60), "multi_channel signals")
        _assert_shape(multi_batch["coords"], (200, 2), "multi_channel coords")
        _assert_shape(multi_batch["mu_label"], (3, 200, 1), "multi_channel mu_label")
        _assert_shape(multi_batch["mask_label"], (3, 200, 1), "multi_channel mask_label")
        if infer_signal_len(multi_dataset) != 60:
            raise AssertionError("infer_signal_len should return flattened multi-channel length 60")
        if multi_batch["signal_original_shape"] != (3, 20):
            raise AssertionError("multi-channel signal_original_shape should be (3, 20)")
        if multi_batch["signal_channels"] != 3:
            raise AssertionError("multi-channel signal_channels should be 3")
        if multi_batch["signal_length_per_channel"] != 20:
            raise AssertionError("multi-channel signal_length_per_channel should be 20")
        if multi_batch["flattened_signal_length"] != 60:
            raise AssertionError("multi-channel flattened_signal_length should be 60")
        if multi_batch["signal_flatten_order"] != "channels_first":
            raise AssertionError("multi-channel flatten order should be channels_first")

        multi_model = ConditionalDualNet(
            signal_len=60,
            latent_dim=16,
            hidden_dim=32,
            num_layers=2,
        )
        multi_out = multi_model(multi_batch["signals"], multi_batch["coords"])
        _assert_shape(multi_out["latent"], (3, 16), "multi_channel latent")
        _assert_shape(multi_out["mu"], (3, 200, 1), "multi_channel mu")
        _assert_shape(multi_out["phi"], (3, 200, 1), "multi_channel phi")

        missing_signals = tmp_path / "missing_signals.npz"
        _write_npz(missing_signals, include_signals=False)
        _expect_value_error(lambda: load_conditional_npz(missing_signals), "missing signals")

        missing_mu_maps = tmp_path / "missing_mu_maps.npz"
        _write_npz(missing_mu_maps, include_mu_maps=False)
        _expect_value_error(lambda: load_conditional_npz(missing_mu_maps), "missing mu_maps")

        missing_coords = tmp_path / "missing_coords.npz"
        _write_npz(missing_coords, include_coords=False)
        _expect_value_error(lambda: load_conditional_npz(missing_coords), "missing coords and x/y")

        no_masks = tmp_path / "no_masks.npz"
        _write_npz(no_masks)
        no_masks_dataset = load_conditional_npz(no_masks)
        _expect_value_error(lambda: get_conditional_batch(no_masks_dataset, [0], mask_source="masks"), "missing masks")

    print("Conditional dual-network data utility smoke test passed.")


if __name__ == "__main__":
    main()

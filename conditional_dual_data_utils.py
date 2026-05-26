"""Batch data utilities for signal-conditioned dual-network models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def load_conditional_npz(npz_path):
    """Load a conditional dual-network dataset from an .npz file."""
    path = Path(npz_path)
    if not path.exists():
        raise ValueError(f"npz_path does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        if "signals" not in files:
            raise ValueError("conditional dataset must contain a 'signals' field")
        if "mu_maps" not in files:
            raise ValueError("conditional dataset must contain a 'mu_maps' field")
        if "coords" not in files and not {"x", "y"}.issubset(files):
            raise ValueError("conditional dataset must contain either 'coords' or both 'x' and 'y'")
        return {name: data[name] for name in data.files}


def build_coords_from_xy(x, y):
    """Build [ny * nx, 2] coords from 1D x/y arrays in mu_map flatten order."""
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    if x_array.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {x_array.shape}")
    if y_array.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y_array.shape}")

    x_grid, y_grid = np.meshgrid(x_array, y_array, indexing="xy")
    return np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)


def infer_signal_len(dataset):
    """Return flattened signal length accepted by BzEncoder."""
    if "signals" not in dataset:
        raise ValueError("dataset must contain 'signals'")
    signals = np.asarray(dataset["signals"])
    if signals.ndim == 2:
        return int(signals.shape[1])
    if signals.ndim == 3:
        return int(signals.shape[1] * signals.shape[2])
    raise ValueError(f"signals must have shape [num_samples, signal_len] or [num_samples, channels, signal_len], got {signals.shape}")


def _signal_metadata(signals):
    if signals.ndim == 2:
        return {
            "signal_original_shape": (int(signals.shape[1]),),
            "signal_channels": 1,
            "signal_length_per_channel": int(signals.shape[1]),
            "flattened_signal_length": int(signals.shape[1]),
            "signal_flatten_order": "single_channel",
        }
    if signals.ndim == 3:
        channels = int(signals.shape[1])
        signal_len = int(signals.shape[2])
        return {
            "signal_original_shape": (channels, signal_len),
            "signal_channels": channels,
            "signal_length_per_channel": signal_len,
            "flattened_signal_length": channels * signal_len,
            "signal_flatten_order": "channels_first",
        }
    raise ValueError(f"signals must have shape [num_samples, signal_len] or [num_samples, channels, signal_len], got {signals.shape}")


def _flatten_signals(signals, indices):
    selected = signals[indices]
    if selected.ndim == 2:
        return selected
    if selected.ndim == 3:
        return selected.reshape(selected.shape[0], -1)
    raise ValueError(f"signals must have shape [num_samples, signal_len] or [num_samples, channels, signal_len], got {signals.shape}")


def _normalize_sample_indices(sample_indices: Iterable[int], num_samples: int):
    indices = [int(idx) for idx in sample_indices]
    if not indices:
        raise ValueError("sample_indices must not be empty")
    for idx in indices:
        if idx < 0 or idx >= num_samples:
            raise ValueError(f"sample index {idx} is out of range for {num_samples} samples")
    return indices


def _coords_from_dataset(dataset, indices):
    if "coords" in dataset:
        coords = np.asarray(dataset["coords"])
        if coords.ndim == 2 and coords.shape[1] == 2:
            return coords
        if coords.ndim == 3 and coords.shape[2] == 2:
            selected = coords[indices]
            first = selected[0]
            if not np.allclose(selected, first[None, :, :]):
                raise ValueError("batched coords differ across selected samples; expected a shared [N, 2] grid")
            return first
        raise ValueError(f"coords must have shape [N, 2] or [num_samples, N, 2], got {coords.shape}")

    if "x" not in dataset or "y" not in dataset:
        raise ValueError("dataset must contain coords or both x and y")
    return build_coords_from_xy(dataset["x"], dataset["y"])


def _flatten_mu_maps(mu_maps, indices):
    selected = mu_maps[indices]
    if selected.ndim == 2:
        return selected[:, :, None]
    if selected.ndim == 3:
        return selected.reshape(selected.shape[0], -1, 1)
    if selected.ndim == 4 and selected.shape[-1] == 1:
        return selected.reshape(selected.shape[0], -1, 1)
    raise ValueError(
        "mu_maps samples must flatten to [N, 1]; expected [num_samples, N], "
        f"[num_samples, ny, nx], or [num_samples, ny, nx, 1], got {mu_maps.shape}"
    )


def _flatten_masks(masks, indices):
    selected = masks[indices]
    if selected.ndim == 2:
        return selected[:, :, None]
    if selected.ndim == 3:
        return selected.reshape(selected.shape[0], -1, 1)
    if selected.ndim == 4 and selected.shape[-1] == 1:
        return selected.reshape(selected.shape[0], -1, 1)
    raise ValueError(
        "masks samples must flatten to [N, 1]; expected [num_samples, N], "
        f"[num_samples, ny, nx], or [num_samples, ny, nx, 1], got {masks.shape}"
    )


def get_conditional_batch(dataset, sample_indices, device="cpu", mask_source="mu_threshold", mu_threshold=500.0):
    """Build a small conditional model batch without enabling coord gradients."""
    if mask_source not in {"mu_threshold", "masks"}:
        raise ValueError(f"unsupported mask_source: {mask_source}")
    if "signals" not in dataset:
        raise ValueError("dataset must contain 'signals'")
    if "mu_maps" not in dataset:
        raise ValueError("dataset must contain 'mu_maps'; the current runner needs mu_label for diagnostics")

    signals = np.asarray(dataset["signals"])
    mu_maps = np.asarray(dataset["mu_maps"])
    signal_meta = _signal_metadata(signals)
    if mu_maps.ndim < 2:
        raise ValueError(f"mu_maps must have a sample dimension and map dimensions, got {mu_maps.shape}")
    if signals.shape[0] != mu_maps.shape[0]:
        raise ValueError(
            "signals and mu_maps must have the same sample count: "
            f"got {signals.shape[0]} and {mu_maps.shape[0]}"
        )

    indices = _normalize_sample_indices(sample_indices, signals.shape[0])
    coords = _coords_from_dataset(dataset, indices)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape [N, 2], got {coords.shape}")

    mu_label = _flatten_mu_maps(mu_maps, indices)
    if mu_label.shape[1] != coords.shape[0]:
        raise ValueError(
            "coords and mu_maps must describe the same number of points: "
            f"got {coords.shape[0]} coords and {mu_label.shape[1]} mu values"
        )

    device_obj = torch.device(device)
    flattened_signals = _flatten_signals(signals, indices)
    signals_tensor = torch.as_tensor(flattened_signals, dtype=torch.float32, device=device_obj)
    coords_tensor = torch.as_tensor(coords, dtype=torch.float32, device=device_obj)
    mu_label_tensor = torch.as_tensor(mu_label, dtype=torch.float32, device=device_obj)
    if mask_source == "mu_threshold":
        mask_label = (mu_label_tensor < float(mu_threshold)).to(dtype=torch.float32)
    else:
        if "masks" not in dataset:
            raise ValueError("mask_source='masks' requires dataset to contain a 'masks' field")
        masks = np.asarray(dataset["masks"])
        if masks.shape[0] != signals.shape[0]:
            raise ValueError(
                "signals and masks must have the same sample count: "
                f"got {signals.shape[0]} and {masks.shape[0]}"
            )
        mask_label_np = _flatten_masks(masks, indices)
        if mask_label_np.shape[1] != coords.shape[0]:
            raise ValueError(
                "coords and masks must describe the same number of points: "
                f"got {coords.shape[0]} coords and {mask_label_np.shape[1]} mask values"
            )
        mask_label = (torch.as_tensor(mask_label_np, dtype=torch.float32, device=device_obj) > 0.5).to(
            dtype=torch.float32
        )

    x_unique = torch.as_tensor(np.unique(coords[:, 0]), dtype=torch.float32, device=device_obj)
    y_unique = torch.as_tensor(np.unique(coords[:, 1]), dtype=torch.float32, device=device_obj)

    return {
        "signals": signals_tensor,
        "coords": coords_tensor,
        "mu_label": mu_label_tensor,
        "mask_label": mask_label,
        "x_unique": x_unique,
        "y_unique": y_unique,
        "mask_source": mask_source,
        **signal_meta,
    }


def main():
    print("conditional_dual_data_utils.py provides conditional .npz batch helpers.")
    print("It does not read files or train unless imported by a runner.")


if __name__ == "__main__":
    main()

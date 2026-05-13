"""Data utilities for single-sample dual-network branch experiments.

This module does not call train_pinn.py or data_generator_v2.py. It only
provides explicit helpers for loading an existing .npz file and converting one
sample into tensors for future branch prototypes.
"""

import argparse

import numpy as np
import torch


REQUIRED_SAMPLE_FIELDS = ("signals", "mu_maps")


def load_npz_dataset(npz_path):
    """Load an .npz dataset and verify the minimal dual-network fields."""
    data = np.load(npz_path, allow_pickle=False)
    missing = [field for field in REQUIRED_SAMPLE_FIELDS if field not in data.files]
    if missing:
        raise ValueError(
            "Missing required fields in npz dataset: "
            f"{missing}. Required sample fields are {list(REQUIRED_SAMPLE_FIELDS)}."
        )

    has_coords = "coords" in data.files
    has_xy = "x" in data.files and "y" in data.files
    if not has_coords and not has_xy:
        raise ValueError(
            "Dataset must contain either coords or both x and y coordinate fields."
        )
    return {key: data[key] for key in data.files}


def build_coords_from_xy(x, y):
    """Build [ny * nx, 2] coords from 1D x/y arrays in mu_map flatten order."""
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    if x_array.ndim != 1:
        raise ValueError(f"x must be a 1D array, got shape {x_array.shape}")
    if y_array.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape {y_array.shape}")

    x_grid, y_grid = np.meshgrid(x_array, y_array, indexing="xy")
    return np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)


def get_single_sample(dataset, sample_index=0):
    """Return signal, coords, and mu_map for one sample without mutation."""
    signals = dataset["signals"]
    mu_maps = dataset["mu_maps"]

    num_samples = signals.shape[0]
    if sample_index < 0 or sample_index >= num_samples:
        raise ValueError(
            f"sample_index {sample_index} is out of range for {num_samples} samples"
        )

    if mu_maps.shape[0] != num_samples:
        raise ValueError(
            "signals and mu_maps must have the same sample dimension: "
            f"got {num_samples} and {mu_maps.shape[0]}"
        )

    if "coords" in dataset:
        coords = dataset["coords"]
        if coords.ndim == 2 and coords.shape[1] == 2:
            sample_coords = coords
        elif coords.ndim == 3 and coords.shape[0] == num_samples and coords.shape[2] == 2:
            sample_coords = coords[sample_index]
        else:
            raise ValueError(
                "coords must have shape [N, 2] or [num_samples, N, 2], "
                f"got {coords.shape}"
            )
    else:
        if "x" not in dataset or "y" not in dataset:
            raise ValueError("Dataset must contain coords or both x and y fields")
        sample_coords = build_coords_from_xy(dataset["x"], dataset["y"])

    return signals[sample_index], sample_coords, mu_maps[sample_index]


def _reshape_mu_label(mu_map):
    mu_array = np.asarray(mu_map)
    if mu_array.ndim == 2:
        return mu_array.reshape(-1, 1)
    if mu_array.ndim == 1:
        return mu_array.reshape(-1, 1)
    if mu_array.ndim == 2 and mu_array.shape[1] == 1:
        return mu_array
    raise ValueError(f"mu_map must have shape [ny, nx], [N], or [N, 1], got {mu_array.shape}")


def build_dual_inputs(signal, coords, mu_map, device="cpu"):
    """Convert one sample into tensors for branch losses.

    mu_label is kept on the original scale and is intended for diagnostics,
    validation, or an optional prior. It is not the core weak-form training
    target for the branch.
    """
    coords_tensor = torch.as_tensor(coords, dtype=torch.float32, device=device)
    if coords_tensor.dim() != 2 or coords_tensor.shape[1] != 2:
        raise ValueError(
            f"coords must have shape [N, 2], got {tuple(coords_tensor.shape)}"
        )
    coords_tensor = coords_tensor.clone().detach().requires_grad_(True)

    bz_meas = torch.as_tensor(signal, dtype=torch.float32, device=device).reshape(-1, 1)
    mu_label = torch.as_tensor(_reshape_mu_label(mu_map), dtype=torch.float32, device=device)

    if mu_label.shape[0] != coords_tensor.shape[0]:
        raise ValueError(
            "mu_label and coords must use the same number of points: "
            f"got {mu_label.shape[0]} and {coords_tensor.shape[0]}"
        )

    return {
        "coords": coords_tensor,
        "bz_meas": bz_meas,
        "mu_label": mu_label,
    }


def infer_grid_shape(coords):
    """Infer grid dimensions from [N, 2] coordinates."""
    coords_array = np.asarray(coords)
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(f"coords must have shape [N, 2], got {coords_array.shape}")

    x_unique = np.unique(coords_array[:, 0])
    y_unique = np.unique(coords_array[:, 1])
    nx = int(x_unique.size)
    ny = int(y_unique.size)
    if nx * ny != coords_array.shape[0]:
        raise ValueError(
            "coords do not appear to form a complete regular grid: "
            f"nx * ny = {nx * ny}, but N = {coords_array.shape[0]}"
        )
    return {
        "nx": nx,
        "ny": ny,
        "x_unique": x_unique,
        "y_unique": y_unique,
    }


def get_probe_coords_from_grid(x_unique, y_s=10.0, device="cpu"):
    """Build probe-line coordinates at y_s.

    The first branch version uses y_s=10.0 to match the data-audit assumption
    that probe signals align with bz_signal[-1, :].
    """
    x_tensor = torch.as_tensor(x_unique, dtype=torch.float32, device=device).reshape(-1)
    y_tensor = torch.full_like(x_tensor, float(y_s))
    coords_probe = torch.stack([x_tensor, y_tensor], dim=1)
    return coords_probe.clone().detach().requires_grad_(True)


def main():
    parser = argparse.ArgumentParser(
        description="Dual-network data utility module; does not train automatically."
    )
    parser.add_argument("--npz-path", default=None)
    args = parser.parse_args()

    print("dual_network_data_utils.py is a data utility module.")
    print("It does not start training, save files, or generate plots.")
    if args.npz_path is None:
        print("No .npz path provided; no file was read.")
        return

    dataset = load_npz_dataset(args.npz_path)
    print(f"Loaded fields: {sorted(dataset.keys())}")


if __name__ == "__main__":
    main()

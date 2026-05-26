"""Utilities for validating COMSOL-style multi-height Bz .npz datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _optional_array(data, name):
    if name not in data.files:
        return None
    value = data[name]
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def validate_comsol_multiheight_npz(npz_path):
    """Validate a COMSOL-style multi-height Bz dataset and return a summary dict."""
    path = Path(npz_path)
    if not path.exists():
        raise ValueError(f"npz_path does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        if "signals" not in files:
            raise ValueError("COMSOL multi-height dataset must contain 'signals'")
        signals = np.asarray(data["signals"])
        if signals.ndim != 3:
            raise ValueError(f"signals must have shape [num_samples, num_channels, signal_len], got {signals.shape}")
        num_samples, num_channels, signal_len = signals.shape
        if num_channels < 2:
            raise ValueError(f"signals must contain at least two channels, got {num_channels}")

        has_x_y = "x" in files and "y" in files
        has_coords = "coords" in files
        if not has_x_y and not has_coords:
            raise ValueError("dataset must contain either x/y or coords")

        has_mu_maps = "mu_maps" in files
        has_masks = "masks" in files
        if not has_mu_maps and not has_masks:
            raise ValueError("dataset must contain either 'mu_maps' or 'masks'")

        if has_mu_maps:
            mu_maps = np.asarray(data["mu_maps"])
            if mu_maps.shape[0] != num_samples:
                raise ValueError(
                    "mu_maps sample count must match signals: "
                    f"got {mu_maps.shape[0]} and {num_samples}"
                )
        if has_masks:
            masks = np.asarray(data["masks"])
            if masks.shape[0] != num_samples:
                raise ValueError(
                    "masks sample count must match signals: "
                    f"got {masks.shape[0]} and {num_samples}"
                )

        notes = []
        if has_x_y:
            x = np.asarray(data["x"])
            if x.ndim != 1:
                raise ValueError(f"x must be one-dimensional when present, got {x.shape}")
            if len(x) != signal_len:
                notes.append(f"len(x)={len(x)} differs from signal_len={signal_len}")
            y = np.asarray(data["y"])
            if y.ndim != 1:
                raise ValueError(f"y must be one-dimensional when present, got {y.shape}")

        return {
            "num_samples": int(num_samples),
            "num_channels": int(num_channels),
            "signal_len": int(signal_len),
            "has_mu_maps": bool(has_mu_maps),
            "has_masks": bool(has_masks),
            "has_x_y": bool(has_x_y),
            "has_coords": bool(has_coords),
            "channel_names": _optional_array(data, "signal_channel_names"),
            "lift_off_values": _optional_array(data, "lift_off_values"),
            "field_components": _optional_array(data, "field_components"),
            "source_type": _optional_array(data, "source_type"),
            "notes": notes,
        }


def print_npz_summary(npz_path):
    summary = validate_comsol_multiheight_npz(npz_path)
    print("COMSOL-style multi-height Bz NPZ summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.npz_path:
        print("comsol_multiheight_npz_utils.py validates COMSOL-style multi-height Bz .npz files.")
        print("Example: python comsol_multiheight_npz_utils.py --npz-path data/comsol_multiheight.npz")
        return 0
    print_npz_summary(args.npz_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

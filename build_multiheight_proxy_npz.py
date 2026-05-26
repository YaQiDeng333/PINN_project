"""Build synthetic multi-height proxy Bz signals from a single-channel .npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CHANNEL_NAMES = np.array(["raw", "smooth3_decay0.8", "smooth7_decay0.6"])
PROXY_WARNING = "proxy channels are derived from single Bz signal; not physical COMSOL multi-height data"


def moving_average_same(signals: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape [num_samples, signal_len], got {signals.shape}")
    pad = window // 2
    padded = np.pad(signals, ((0, 0), (pad, pad)), mode="edge")
    output = np.empty_like(signals, dtype=np.float32)
    for idx in range(signals.shape[1]):
        output[:, idx] = padded[:, idx : idx + window].mean(axis=1)
    return output


def build_proxy_signals(signals: np.ndarray, mode: str) -> np.ndarray:
    if mode != "smooth_decay_proxy":
        raise ValueError(f"unsupported mode: {mode}")
    if signals.ndim != 2:
        raise ValueError(
            "input signals must be single-channel [num_samples, signal_len]; "
            f"got shape {signals.shape}"
        )
    raw = signals.astype(np.float32, copy=False)
    smooth3 = moving_average_same(raw, window=3) * np.float32(0.8)
    smooth7 = moving_average_same(raw, window=7) * np.float32(0.6)
    return np.stack([raw, smooth3, smooth7], axis=1).astype(np.float32, copy=False)


def convert_npz(input_npz: str | Path, output_npz: str | Path, mode: str = "smooth_decay_proxy") -> None:
    input_path = Path(input_npz)
    output_path = Path(output_npz)
    if not input_path.exists():
        raise ValueError(f"input npz does not exist: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        files = set(data.files)
        if "signals" not in files:
            raise ValueError("input npz must contain 'signals'")
        if "mu_maps" not in files:
            raise ValueError("input npz must contain 'mu_maps'")
        signals = np.asarray(data["signals"])
        if signals.ndim == 3:
            raise ValueError("input signals already have channels; refusing to build proxy twice")
        if signals.ndim != 2:
            raise ValueError(f"input signals must have shape [num_samples, signal_len], got {signals.shape}")

        payload = {name: data[name] for name in data.files if name != "signals"}
        proxy_signals = build_proxy_signals(signals, mode=mode)
        payload["signals"] = proxy_signals
        payload["source_signal_shape"] = np.array(signals.shape, dtype=np.int64)
        payload["proxy_signal_shape"] = np.array(proxy_signals.shape, dtype=np.int64)
        payload["signal_channels"] = np.array(3, dtype=np.int64)
        payload["signal_channel_names"] = CHANNEL_NAMES
        payload["source_type"] = np.array("synthetic_multiheight_proxy")
        payload["proxy_warning"] = np.array(PROXY_WARNING)
        payload["proxy_mode"] = np.array(mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz")
    parser.add_argument("--output-npz")
    parser.add_argument("--mode", default="smooth_decay_proxy")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.input_npz or not args.output_npz:
        print("build_multiheight_proxy_npz.py builds synthetic proxy multi-channel Bz .npz files.")
        print(
            "Example: python build_multiheight_proxy_npz.py "
            "--input-npz data/train.npz --output-npz data/train_proxy.npz"
        )
        return 0
    convert_npz(args.input_npz, args.output_npz, mode=args.mode)
    print(f"Saved synthetic multi-height proxy npz to {args.output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

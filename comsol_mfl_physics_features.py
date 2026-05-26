"""Extract lightweight physics-inspired MFL features from COMSOL multi-height Bz signals."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


EPS = 1e-8


def _as_channels(signals: np.ndarray) -> np.ndarray:
    if signals.ndim == 3:
        return signals.astype(np.float32)
    if signals.ndim == 2:
        return signals[:, None, :].astype(np.float32)
    raise ValueError(f"signals must have shape [N,C,L] or [N,L], got {signals.shape}")


def _load_npz(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(npz_path)
    with np.load(path, allow_pickle=True) as data:
        if "signals" not in data:
            raise ValueError(f"{path} does not contain signals.")
        signals = _as_channels(data["signals"])
        x = data["x"].astype(np.float32) if "x" in data else np.linspace(-1.0, 1.0, signals.shape[-1], dtype=np.float32)
        sample_indices = data["sample_indices"].astype(np.int64) if "sample_indices" in data else np.arange(signals.shape[0])
    if x.ndim != 1 or x.shape[0] != signals.shape[-1]:
        x = np.linspace(-1.0, 1.0, signals.shape[-1], dtype=np.float32)
    return signals, x, sample_indices


def _local_peak_count(values: np.ndarray, positive: bool) -> float:
    if values.size < 3:
        return 0.0
    data = values if positive else -values
    threshold = 0.25 * float(np.max(np.abs(values)) + EPS)
    middle = data[1:-1]
    peaks = (middle > data[:-2]) & (middle >= data[2:]) & (middle > threshold)
    return float(np.count_nonzero(peaks))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) < EPS or float(np.std(b)) < EPS:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(value) else value


def _channel_features(signal: np.ndarray, x: np.ndarray, mode: str) -> tuple[list[float], list[str]]:
    values = signal.astype(np.float64)
    x64 = x.astype(np.float64)
    abs_values = np.abs(values)
    peak_abs = float(abs_values.max())
    base_values = [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        peak_abs,
        float(values.max() - values.min()),
        float(x64[int(values.argmax())]),
        float(x64[int(values.argmin())]),
        float(x64[int(abs_values.argmax())]),
        float(np.mean(values * values)),
        float(abs_values.mean()),
        float(values.mean()),
    ]
    base_names = [
        "mean",
        "std",
        "min",
        "max",
        "peak_abs",
        "peak_to_peak",
        "argmax_x",
        "argmin_x",
        "argmax_abs_x",
        "energy",
        "abs_area",
        "signed_area",
    ]
    if mode == "basic_peak":
        return base_values, base_names
    if mode != "peak_decay_width":
        raise ValueError(f"Unsupported feature_mode: {mode}")
    if x64.size > 1:
        dx = float(np.mean(np.abs(np.diff(x64))))
    else:
        dx = 1.0
    half_mask = abs_values >= 0.5 * peak_abs
    half_abs_width = float(np.count_nonzero(half_mask) * dx)
    abs_sum = float(abs_values.sum())
    center_of_abs_mass = float((abs_values * x64).sum() / (abs_sum + EPS))
    split = x64 < center_of_abs_mass
    left_area = float(abs_values[split].sum())
    right_area = float(abs_values[~split].sum())
    left_right_abs_balance = float((right_area - left_area) / (right_area + left_area + EPS))
    extra_values = [
        _local_peak_count(values, positive=True),
        _local_peak_count(values, positive=False),
        half_abs_width,
        center_of_abs_mass,
        left_right_abs_balance,
    ]
    extra_names = [
        "positive_peak_count",
        "negative_peak_count",
        "half_abs_width",
        "center_of_abs_mass",
        "left_right_abs_balance",
    ]
    return base_values + extra_values, base_names + extra_names


def extract_physics_features(signals: np.ndarray, x: np.ndarray, feature_mode: str = "basic_peak") -> tuple[np.ndarray, list[str]]:
    signals = _as_channels(signals)
    rows: list[list[float]] = []
    feature_names: list[str] = []
    for sample in range(signals.shape[0]):
        row: list[float] = []
        names: list[str] = []
        for channel in range(signals.shape[1]):
            values, channel_names = _channel_features(signals[sample, channel], x, feature_mode)
            row.extend(values)
            names.extend([f"ch{channel}_{name}" for name in channel_names])
        if feature_mode == "peak_decay_width":
            peak_abs = np.max(np.abs(signals[sample]), axis=1).astype(np.float64)
            energy = np.mean(signals[sample].astype(np.float64) ** 2, axis=1)
            for channel in [1, 2]:
                if signals.shape[1] > channel:
                    row.append(float(peak_abs[channel] / (peak_abs[0] + EPS)))
                    names.append(f"peak_abs_ch{channel}_over_ch0")
            for channel in [1, 2]:
                if signals.shape[1] > channel:
                    row.append(float(energy[channel] / (energy[0] + EPS)))
                    names.append(f"energy_ch{channel}_over_ch0")
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                if signals.shape[1] > b:
                    row.append(_safe_corr(signals[sample, a], signals[sample, b]))
                    names.append(f"corr_ch{a}_ch{b}")
        if sample == 0:
            feature_names = names
        elif len(row) != len(feature_names):
            raise ValueError("Feature count changed between samples.")
        rows.append(row)
    features = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Extracted physics features contain NaN or Inf.")
    return features, feature_names


def _write_csv(path: Path, features: np.ndarray, feature_names: list[str], sample_indices: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", *feature_names])
        writer.writeheader()
        for index, row in zip(sample_indices, features):
            writer.writerow({"sample_index": int(index), **{name: float(value) for name, value in zip(feature_names, row)}})


def _write_summary(path: Path, npz_path: str | Path, features: np.ndarray, feature_names: list[str], feature_mode: str) -> None:
    lines = [
        "# COMSOL MFL physics feature summary",
        "",
        f"- npz_path: `{npz_path}`",
        f"- feature_mode: `{feature_mode}`",
        f"- samples: `{features.shape[0]}`",
        f"- feature_count: `{features.shape[1]}`",
        f"- contains_nan_or_inf: `{not np.isfinite(features).all()}`",
        "",
        "## Feature ranges",
        "",
    ]
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    means = features.mean(axis=0)
    for name, lo, hi, mean in zip(feature_names, mins, maxs, means):
        lines.append(f"- `{name}`: min={lo:.6e}, max={hi:.6e}, mean={mean:.6e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals, x, sample_indices = _load_npz(args.npz_path)
    features, feature_names = extract_physics_features(signals, x, args.feature_mode)
    np.savez(
        output_dir / "physics_features.npz",
        features=features,
        feature_names=np.asarray(feature_names, dtype="U128"),
        sample_indices=sample_indices.astype(np.int64),
    )
    _write_csv(output_dir / "physics_features.csv", features, feature_names, sample_indices)
    _write_summary(output_dir / "feature_summary.md", args.npz_path, features, feature_names, args.feature_mode)
    print(f"Saved {features.shape[0]}x{features.shape[1]} physics features to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--feature-mode", choices=["basic_peak", "peak_decay_width"], default="basic_peak")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.npz_path or not args.output_dir:
        parser.print_help()
        print("\nExample: python comsol_mfl_physics_features.py --npz-path train.npz --output-dir features --feature-mode peak_decay_width")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

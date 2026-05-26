"""Convert COMSOL-style long CSV signals plus target NPZ into multi-channel NPZ."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "sample_index",
    "channel_index",
    "channel_name",
    "lift_off",
    "field_component",
    "x_index",
    "x",
    "value",
}


def _read_signal_rows(signals_csv: str | Path):
    path = Path(signals_csv)
    if not path.exists():
        raise ValueError(f"signals CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("signals CSV must contain a header row")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"signals CSV is missing required columns: {sorted(missing)}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "channel_index": int(row["channel_index"]),
                    "channel_name": row["channel_name"],
                    "lift_off": float(row["lift_off"]),
                    "field_component": row["field_component"],
                    "x_index": int(row["x_index"]),
                    "x": float(row["x"]),
                    "value": float(row["value"]),
                }
            )
    if not rows:
        raise ValueError("signals CSV contains no data rows")
    return rows


def _ordered_unique(values):
    return sorted(set(values))


def _build_signals_from_rows(rows):
    sample_indices = _ordered_unique(row["sample_index"] for row in rows)
    channel_indices = _ordered_unique(row["channel_index"] for row in rows)
    x_indices = _ordered_unique(row["x_index"] for row in rows)
    if not sample_indices or not channel_indices or not x_indices:
        raise ValueError("signals CSV must contain sample, channel, and x indices")

    sample_to_pos = {sample: idx for idx, sample in enumerate(sample_indices)}
    channel_to_pos = {channel: idx for idx, channel in enumerate(channel_indices)}
    x_to_pos = {x_idx: idx for idx, x_idx in enumerate(x_indices)}
    expected_count = len(sample_indices) * len(channel_indices) * len(x_indices)
    if len(rows) != expected_count:
        raise ValueError(
            "signals CSV is incomplete or has duplicate rows: "
            f"got {len(rows)} rows, expected {expected_count}"
        )

    signals = np.empty((len(sample_indices), len(channel_indices), len(x_indices)), dtype=np.float32)
    seen = set()
    channel_names = {}
    lift_off_values = {}
    field_components = {}
    x_values = {}

    for row in sorted(rows, key=lambda r: (r["sample_index"], r["channel_index"], r["x_index"])):
        key = (row["sample_index"], row["channel_index"], row["x_index"])
        if key in seen:
            raise ValueError(f"duplicate signal row for {key}")
        seen.add(key)
        sample_pos = sample_to_pos[row["sample_index"]]
        channel_pos = channel_to_pos[row["channel_index"]]
        x_pos = x_to_pos[row["x_index"]]
        signals[sample_pos, channel_pos, x_pos] = row["value"]
        channel_names.setdefault(row["channel_index"], row["channel_name"])
        lift_off_values.setdefault(row["channel_index"], row["lift_off"])
        field_components.setdefault(row["channel_index"], row["field_component"])
        x_values.setdefault(row["x_index"], row["x"])

    if len(seen) != expected_count:
        raise ValueError("signals CSV does not cover every sample/channel/x combination")
    if not np.all(np.isfinite(signals)):
        nonfinite_count = int(np.sum(~np.isfinite(signals)))
        raise ValueError(f"signals array contains {nonfinite_count} non-finite values")
    if set(channel_names) != set(channel_indices):
        raise ValueError("channel metadata is incomplete")
    if set(x_values) != set(x_indices):
        raise ValueError("x metadata is incomplete")

    return {
        "signals": signals,
        "sample_indices": np.array(sample_indices, dtype=np.int64),
        "channel_indices": np.array(channel_indices, dtype=np.int64),
        "x_indices": np.array(x_indices, dtype=np.int64),
        "x_values": np.array([x_values[idx] for idx in x_indices], dtype=np.float32),
        "signal_channel_names": np.array([channel_names[idx] for idx in channel_indices]),
        "lift_off_values": np.array([lift_off_values[idx] for idx in channel_indices], dtype=np.float32),
        "field_components": np.array([field_components[idx] for idx in channel_indices]),
    }


def convert_csv_to_npz(signals_csv: str | Path, target_npz: str | Path, output_npz: str | Path) -> None:
    target_path = Path(target_npz)
    output_path = Path(output_npz)
    if not target_path.exists():
        raise ValueError(f"target NPZ does not exist: {target_path}")

    signal_data = _build_signals_from_rows(_read_signal_rows(signals_csv))
    signals = signal_data["signals"]

    with np.load(target_path, allow_pickle=False) as target:
        files = set(target.files)
        has_mu_maps = "mu_maps" in files
        has_masks = "masks" in files
        if not has_mu_maps and not has_masks:
            raise ValueError("target NPZ must contain either 'mu_maps' or 'masks'")
        if "coords" not in files and not {"x", "y"}.issubset(files):
            raise ValueError("target NPZ must contain either 'coords' or both 'x' and 'y'")

        payload = {name: target[name] for name in target.files}
        target_samples = None
        if has_mu_maps:
            target_samples = int(np.asarray(target["mu_maps"]).shape[0])
        elif has_masks:
            target_samples = int(np.asarray(target["masks"]).shape[0])
        if target_samples != signals.shape[0]:
            raise ValueError(
                "target sample count must match signal CSV sample count: "
                f"got {target_samples} and {signals.shape[0]}"
            )

    payload["signals"] = signals.astype(np.float32, copy=False)
    payload["signal_channel_names"] = signal_data["signal_channel_names"]
    payload["lift_off_values"] = signal_data["lift_off_values"]
    payload["field_components"] = signal_data["field_components"]
    payload["source_type"] = np.array("converted_comsol_multiheight_csv")
    payload["signal_flatten_order"] = np.array("channels_first")
    payload["converter_note"] = np.array("converted from COMSOL-style long CSV plus target NPZ")
    payload["csv_sample_indices"] = signal_data["sample_indices"]
    payload["csv_channel_indices"] = signal_data["channel_indices"]
    payload["csv_x_indices"] = signal_data["x_indices"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-csv")
    parser.add_argument("--target-npz")
    parser.add_argument("--output-npz")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.signals_csv or not args.target_npz or not args.output_npz:
        print("convert_comsol_multiheight_csv_to_npz.py converts COMSOL-style long CSV signals to NPZ.")
        print(
            "Example: python convert_comsol_multiheight_csv_to_npz.py "
            "--signals-csv signals.csv --target-npz targets.npz --output-npz converted.npz"
        )
        return 0
    convert_csv_to_npz(args.signals_csv, args.target_npz, args.output_npz)
    print(f"Saved converted COMSOL-style multi-height NPZ to {args.output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare COMSOL V1/V2 Bz signal scale, lift-off behavior, and offset semantics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_ARGS = [
    ("v1", "train", "v1_train_npz"),
    ("v1", "val", "v1_val_npz"),
    ("v1", "test", "v1_test_npz"),
    ("v2", "train", "v2_train_npz"),
    ("v2", "val", "v2_val_npz"),
    ("v2", "test", "v2_test_npz"),
]


def _optional_list(data: dict[str, np.ndarray], key: str, fallback: list[Any]) -> list[Any]:
    if key not in data:
        return fallback
    value = data[key]
    if value.ndim == 0:
        return [value.item()]
    return [item.item() if hasattr(item, "item") else item for item in value.tolist()]


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"NPZ does not exist: {p}")
    with np.load(p, allow_pickle=True) as data:
        return {name: data[name] for name in data.files}


def _require_signals(version: str, split: str, data: dict[str, np.ndarray]) -> np.ndarray:
    if "signals" not in data:
        raise ValueError(f"{version}/{split} does not contain signals")
    signals = np.asarray(data["signals"], dtype=float)
    if signals.ndim != 3:
        raise ValueError(f"{version}/{split} signals must be [N,C,L], got {signals.shape}")
    if not np.isfinite(signals).all():
        raise ValueError(f"{version}/{split} signals contain NaN or Inf")
    return signals


def _corr_values(signals: np.ndarray) -> dict[str, float]:
    num_samples, num_channels, _ = signals.shape
    pairs: dict[str, list[float]] = {}
    for sample_index in range(num_samples):
        sample = signals[sample_index]
        if num_channels < 2:
            continue
        corr = np.corrcoef(sample)
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                key = f"corr_{i}_{j}"
                value = corr[i, j]
                if np.isfinite(value):
                    pairs.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in pairs.items()}


def _summarize_dataset(version: str, split: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    data = _load_npz(path)
    signals = _require_signals(version, split, data)
    num_samples, num_channels, signal_len = signals.shape
    channel_names = _optional_list(data, "signal_channel_names", [f"channel_{i}" for i in range(num_channels)])
    lift_off_values = _optional_list(data, "lift_off_values", list(range(num_channels)))
    field_components = _optional_list(data, "field_components", [""] * num_channels)

    sample_peak_abs = np.max(np.abs(signals), axis=(1, 2))
    sample_mean_offsets = np.mean(signals, axis=2)
    centered = signals - sample_mean_offsets[:, :, None]
    centered_peak_abs = np.max(np.abs(centered), axis=(1, 2))
    centered_std = np.std(centered, axis=(1, 2))
    offset_abs = np.mean(np.abs(sample_mean_offsets), axis=1)
    peak_ratio = sample_peak_abs / np.maximum(centered_peak_abs, 1e-12)

    aggregate = {
        "dataset_version": version,
        "split": split,
        "npz_path": str(path),
        "num_samples": num_samples,
        "num_channels": num_channels,
        "signal_len": signal_len,
        "signal_min": float(np.min(signals)),
        "signal_max": float(np.max(signals)),
        "signal_mean": float(np.mean(signals)),
        "signal_std": float(np.std(signals)),
        "mean_abs_signal": float(np.mean(np.abs(signals))),
        "mean_peak_abs_signal": float(np.mean(sample_peak_abs)),
        "mean_abs_offset": float(np.mean(offset_abs)),
        "mean_centered_std": float(np.mean(centered_std)),
        "mean_abs_peak_to_centered_peak": float(np.mean(peak_ratio)),
    }

    per_channel = []
    for channel_index in range(num_channels):
        ch = signals[:, channel_index, :]
        ch_mean = np.mean(ch, axis=1)
        ch_centered = ch - ch_mean[:, None]
        per_channel.append(
            {
                "dataset_version": version,
                "split": split,
                "channel_index": channel_index,
                "channel_name": str(channel_names[channel_index]) if channel_index < len(channel_names) else f"channel_{channel_index}",
                "lift_off": "" if channel_index >= len(lift_off_values) else str(lift_off_values[channel_index]),
                "field_component": "" if channel_index >= len(field_components) else str(field_components[channel_index]),
                "min": float(np.min(ch)),
                "max": float(np.max(ch)),
                "mean": float(np.mean(ch)),
                "std": float(np.std(ch)),
                "mean_abs": float(np.mean(np.abs(ch))),
                "mean_peak_abs": float(np.mean(np.max(np.abs(ch), axis=1))),
                "mean_abs_offset": float(np.mean(np.abs(ch_mean))),
                "mean_centered_std": float(np.mean(np.std(ch_centered, axis=1))),
            }
        )

    channel_peaks = np.max(np.abs(signals), axis=2)
    monotonic_flags = np.all(channel_peaks[:, :-1] >= channel_peaks[:, 1:], axis=1)
    higher_liftoff_larger = np.any(channel_peaks[:, 1:] > channel_peaks[:, :-1], axis=1)
    corr = _corr_values(signals)
    decay_row: dict[str, Any] = {
        "dataset_version": version,
        "split": split,
        "num_samples": num_samples,
        "monotonic_decay_fraction": float(np.mean(monotonic_flags)),
        "higher_liftoff_larger_fraction": float(np.mean(higher_liftoff_larger)),
        "mean_channel0_peak_abs": float(np.mean(channel_peaks[:, 0])) if num_channels > 0 else "",
        "mean_channel1_peak_abs": float(np.mean(channel_peaks[:, 1])) if num_channels > 1 else "",
        "mean_channel2_peak_abs": float(np.mean(channel_peaks[:, 2])) if num_channels > 2 else "",
    }
    decay_row.update(corr)
    return aggregate, per_channel, decay_row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_summary(aggregates: list[dict[str, Any]], decay_rows: list[dict[str, Any]]) -> str:
    by_key = {(row["dataset_version"], row["split"]): row for row in aggregates}
    decay_by_key = {(row["dataset_version"], row["split"]): row for row in decay_rows}
    v1 = by_key.get(("v1", "train"))
    v2 = by_key.get(("v2", "train"))
    lines = [
        "# S88 COMSOL V1/V2 signal semantics diagnostics",
        "",
        "## 核心结论",
        "",
    ]
    if v1 and v2:
        mean_abs_ratio = float(v2["mean_abs_signal"]) / max(float(v1["mean_abs_signal"]), 1e-30)
        peak_ratio = float(v2["mean_peak_abs_signal"]) / max(float(v1["mean_peak_abs_signal"]), 1e-30)
        offset_ratio_v1 = float(v1["mean_abs_offset"]) / max(float(v1["mean_peak_abs_signal"]), 1e-30)
        offset_ratio_v2 = float(v2["mean_abs_offset"]) / max(float(v2["mean_peak_abs_signal"]), 1e-30)
        lines.append(f"- train mean_abs_signal: V1 = `{float(v1['mean_abs_signal']):.6e}`, V2 = `{float(v2['mean_abs_signal']):.6e}`, V2/V1 = `{mean_abs_ratio:.3f}`。")
        lines.append(f"- train mean_peak_abs_signal: V1 = `{float(v1['mean_peak_abs_signal']):.6e}`, V2 = `{float(v2['mean_peak_abs_signal']):.6e}`, V2/V1 = `{peak_ratio:.3f}`。")
        lines.append(f"- offset/peak: V1 = `{offset_ratio_v1:.3f}`, V2 = `{offset_ratio_v2:.3f}`。")
        if mean_abs_ratio > 10.0 or mean_abs_ratio < 0.1:
            lines.append("- V1/V2 signal scale 差异超过一个数量级，存在 signal scale / 语义不一致风险。")
        else:
            lines.append("- V1/V2 signal scale 在同一数量级。")
        if offset_ratio_v2 > 0.25:
            lines.append("- V2 存在明显 DC/background offset 风险；不过 S85 使用 `per_sample_zscore`，训练路径已经做均值和尺度归一化。")
        else:
            lines.append("- V2 offset/peak 不高，未显示强 DC/background 主导风险。")
    lines.extend(["", "## lift-off decay", ""])
    lines.append("| dataset | split | monotonic_decay_fraction | higher_liftoff_larger_fraction | ch0_peak | ch1_peak | ch2_peak |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in decay_rows:
        lines.append(
            f"| {row['dataset_version']} | {row['split']} | "
            f"{float(row['monotonic_decay_fraction']):.6e} | {float(row['higher_liftoff_larger_fraction']):.6e} | "
            f"{row.get('mean_channel0_peak_abs', '')} | {row.get('mean_channel1_peak_abs', '')} | {row.get('mean_channel2_peak_abs', '')} |"
        )
    if all(float(row["monotonic_decay_fraction"]) >= 0.8 for row in decay_rows):
        lines.append("\n- lift-off 通道总体符合 peak abs 随 lift-off 增大而衰减的预期。")
    else:
        lines.append("\n- 存在较多样本不满足 lift-off peak abs 单调衰减，需要检查 lift-off/channel 定义。")

    lines.extend(["", "## aggregate signal semantics", ""])
    lines.append("| dataset | split | signal_mean | signal_std | mean_abs_signal | mean_peak_abs_signal | mean_abs_offset | centered_std |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in aggregates:
        lines.append(
            f"| {row['dataset_version']} | {row['split']} | "
            f"{float(row['signal_mean']):.6e} | {float(row['signal_std']):.6e} | "
            f"{float(row['mean_abs_signal']):.6e} | {float(row['mean_peak_abs_signal']):.6e} | "
            f"{float(row['mean_abs_offset']):.6e} | {float(row['mean_centered_std']):.6e} |"
        )

    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- 本诊断不修改数据，只比较 V1/V2 signal scale、lift-off 衰减和 offset。",
            "- 如果 V2 scale 与 V1 同量级、lift-off 衰减合理，并且 S85 已使用 `per_sample_zscore`，则不需要额外执行 center-only S89。",
            "- 如果存在明显 scale 或 lift-off 语义异常，后续应优先修正数据导出或 preprocessing，而不是继续扩大训练。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates: list[dict[str, Any]] = []
    per_channel_rows: list[dict[str, Any]] = []
    decay_rows: list[dict[str, Any]] = []
    for version, split, attr in SPLIT_ARGS:
        aggregate, per_channel, decay = _summarize_dataset(version, split, Path(getattr(args, attr)))
        aggregates.append(aggregate)
        per_channel_rows.extend(per_channel)
        decay_rows.append(decay)
    _write_csv(output_dir / "aggregate_signal_semantics.csv", aggregates)
    _write_csv(output_dir / "per_channel_signal_semantics.csv", per_channel_rows)
    _write_csv(output_dir / "lift_off_decay_diagnostics.csv", decay_rows)
    (output_dir / "summary.md").write_text(_make_summary(aggregates, decay_rows), encoding="utf-8")
    print(f"Saved S88 diagnostics to {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-train-npz")
    parser.add_argument("--v1-val-npz")
    parser.add_argument("--v1-test-npz")
    parser.add_argument("--v2-train-npz")
    parser.add_argument("--v2-val-npz")
    parser.add_argument("--v2-test-npz")
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    required = [attr for _, _, attr in SPLIT_ARGS] + ["output_dir"]
    if not all(getattr(args, attr) for attr in required):
        parser.print_help()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

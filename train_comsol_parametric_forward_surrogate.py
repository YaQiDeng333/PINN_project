"""Train a lightweight COMSOL parametric geometry-to-Bz forward surrogate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comsol_parametric_forward_surrogate import (
    ParametricForwardSurrogate,
    SignalNormalizationStats,
    build_forward_geometry_vector,
    compute_train_zscore_stats,
    denormalize_signals,
    normalize_signals,
)


def _flatten_signals(signals: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    if signals.ndim == 3:
        return signals.reshape(signals.shape[0], -1).astype(np.float32), tuple(signals.shape[1:])
    if signals.ndim == 2:
        return signals.astype(np.float32), (signals.shape[1],)
    raise ValueError(f"signals must have shape [N,C,L] or [N,L], got {signals.shape}")


def load_signal_npz(path: str | Path) -> dict:
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=True) as data:
        if "signals" not in data:
            raise ValueError(f"{npz_path} does not contain signals.")
        signals_flat, signal_shape = _flatten_signals(data["signals"].astype(np.float32))
    if not np.isfinite(signals_flat).all():
        raise ValueError(f"{npz_path} contains NaN or Inf signals.")
    return {"signals_flat": signals_flat, "signal_shape": signal_shape}


def load_targets(path: str | Path, expected_type_vocab: list[str] | None = None) -> dict:
    target_path = Path(path)
    with np.load(target_path, allow_pickle=True) as data:
        continuous = data["continuous_targets"].astype(np.float32)
        presence = data["presence_targets"].astype(np.float32)
        type_targets = data["type_targets"].astype(np.int64)
        sample_indices = data["sample_indices"].astype(np.int64) if "sample_indices" in data else np.arange(continuous.shape[0])
        target_schema = [str(x) for x in data["target_schema"]]
        type_vocab = [str(x) for x in data["type_vocab"]]

    if continuous.ndim != 3:
        raise ValueError(f"continuous_targets must have shape [N,K,P], got {continuous.shape}")
    if presence.shape != continuous.shape[:2]:
        raise ValueError("presence_targets must have shape [N,K].")
    if type_targets.shape != continuous.shape[:2]:
        raise ValueError("type_targets must have shape [N,K].")
    if expected_type_vocab is not None and type_vocab != expected_type_vocab:
        mapping = {name: i for i, name in enumerate(expected_type_vocab)}
        remapped = np.full_like(type_targets, -1)
        for old_index, name in enumerate(type_vocab):
            if name not in mapping:
                raise ValueError(f"Target type {name!r} is absent from train type_vocab.")
            remapped[type_targets == old_index] = mapping[name]
        type_targets = remapped
        type_vocab = list(expected_type_vocab)
    if not np.isfinite(continuous).all():
        raise ValueError(f"{target_path} contains NaN or Inf continuous targets.")
    return {
        "continuous": continuous,
        "presence": presence,
        "type_targets": type_targets,
        "sample_indices": sample_indices,
        "target_schema": target_schema,
        "type_vocab": type_vocab,
    }


def build_split_tensors(
    signals: dict,
    targets: dict,
    signal_stats: SignalNormalizationStats,
    device: torch.device,
) -> dict:
    if signals["signals_flat"].shape[0] != targets["presence"].shape[0]:
        raise ValueError("signals and targets sample counts do not match.")
    signals_norm = normalize_signals(signals["signals_flat"], signal_stats)
    presence = torch.from_numpy(targets["presence"]).to(device)
    type_targets = torch.from_numpy(targets["type_targets"]).to(device)
    continuous = torch.from_numpy(targets["continuous"]).to(device)
    geometry = build_forward_geometry_vector(
        presence=presence,
        type_targets_or_probs=type_targets,
        continuous=continuous,
        num_types=len(targets["type_vocab"]),
        target_schema=targets["target_schema"],
    )
    return {
        "geometry": geometry,
        "signals_norm": torch.from_numpy(signals_norm).to(device),
        "signals_flat": signals["signals_flat"],
        "signals_norm_np": signals_norm,
        "signal_shape": signals["signal_shape"],
        "sample_indices": targets["sample_indices"],
    }


def _corr_flat(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    pred_f = pred.reshape(-1).astype(np.float64)
    true_f = true.reshape(-1).astype(np.float64)
    pred_centered = pred_f - pred_f.mean()
    true_centered = true_f - true_f.mean()
    denom = np.sqrt(np.sum(pred_centered**2) * np.sum(true_centered**2))
    if denom < eps:
        return 0.0
    return float(np.sum(pred_centered * true_centered) / denom)


def evaluate_model(
    model: ParametricForwardSurrogate,
    split: dict,
    signal_stats: SignalNormalizationStats,
    split_name: str,
) -> dict[str, float | str]:
    model.eval()
    with torch.no_grad():
        pred_norm = model(split["geometry"]).detach().cpu().numpy().astype(np.float32)
    true_norm = split["signals_norm_np"]
    true_raw = split["signals_flat"]
    pred_raw = denormalize_signals(pred_norm, signal_stats)

    diff_norm = pred_norm - true_norm
    signal_mse_norm = float(np.mean(diff_norm**2))
    signal_rmse_norm = float(np.sqrt(signal_mse_norm))
    raw_rmse = float(np.sqrt(np.mean((pred_raw - true_raw) ** 2)))
    raw_scale = float(np.sqrt(np.mean(true_raw**2)) + 1e-8)
    signal_nrmse_raw = raw_rmse / raw_scale
    signal_corr = _corr_flat(pred_raw, true_raw)
    peak_true = np.max(np.abs(true_raw), axis=1)
    peak_pred = np.max(np.abs(pred_raw), axis=1)
    peak_abs_nrmse = float(np.mean(np.abs(peak_pred - peak_true)) / (np.mean(peak_true) + 1e-8))

    row: dict[str, float | str] = {
        "split": split_name,
        "signal_mse_norm": signal_mse_norm,
        "signal_rmse_norm": signal_rmse_norm,
        "signal_nrmse_raw": float(signal_nrmse_raw),
        "signal_corr": signal_corr,
        "peak_abs_nrmse": peak_abs_nrmse,
        "mean_abs_signal_true": float(np.mean(np.abs(true_raw))),
        "mean_abs_signal_pred": float(np.mean(np.abs(pred_raw))),
    }
    signal_shape = split["signal_shape"]
    if len(signal_shape) == 2:
        channels, length = signal_shape
        true_ch = true_raw.reshape(true_raw.shape[0], channels, length)
        pred_ch = pred_raw.reshape(pred_raw.shape[0], channels, length)
        for channel in range(channels):
            ch_true = true_ch[:, channel, :]
            ch_pred = pred_ch[:, channel, :]
            ch_rmse = float(np.sqrt(np.mean((ch_pred - ch_true) ** 2)))
            ch_scale = float(np.sqrt(np.mean(ch_true**2)) + 1e-8)
            row[f"ch{channel}_nrmse"] = ch_rmse / ch_scale
            row[f"ch{channel}_corr"] = _corr_flat(ch_pred, ch_true)
    return row


def _write_single_row_csv(path: Path, row: dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric_row(row: dict[str, float | str]) -> str:
    keys = ["signal_nrmse_raw", "signal_corr", "peak_abs_nrmse", "signal_mse_norm"]
    return ", ".join(f"{key}={float(row[key]):.6e}" for key in keys if key in row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", default="")
    parser.add_argument("--train-targets", default="")
    parser.add_argument("--val-npz", default="")
    parser.add_argument("--val-targets", default="")
    parser.add_argument("--test-npz", default="")
    parser.add_argument("--test-targets", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--history-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    required = [
        args.train_npz,
        args.train_targets,
        args.val_npz,
        args.val_targets,
        args.test_npz,
        args.test_targets,
        args.output_dir,
    ]
    if not any(required):
        parse_args(["--help"])
        return 0
    if not all(required):
        raise ValueError("All train/val/test NPZ, target paths, and output-dir are required.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.history_interval <= 0:
        raise ValueError("--history-interval must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_signals = load_signal_npz(args.train_npz)
    val_signals = load_signal_npz(args.val_npz)
    test_signals = load_signal_npz(args.test_npz)
    signal_stats = compute_train_zscore_stats(train_signals["signals_flat"])

    train_targets = load_targets(args.train_targets)
    if train_targets["presence"].shape[1] != args.max_components:
        raise ValueError(
            f"train targets use max_components={train_targets['presence'].shape[1]}, "
            f"but --max-components={args.max_components}."
        )
    val_targets = load_targets(args.val_targets, expected_type_vocab=train_targets["type_vocab"])
    test_targets = load_targets(args.test_targets, expected_type_vocab=train_targets["type_vocab"])
    for split_name, targets in [("val", val_targets), ("test", test_targets)]:
        if targets["target_schema"] != train_targets["target_schema"]:
            raise ValueError(f"{split_name} target_schema differs from train target_schema.")
        if targets["presence"].shape[1] != args.max_components:
            raise ValueError(f"{split_name} max_components differs from --max-components.")

    train_split = build_split_tensors(train_signals, train_targets, signal_stats, device)
    val_split = build_split_tensors(val_signals, val_targets, signal_stats, device)
    test_split = build_split_tensors(test_signals, test_targets, signal_stats, device)

    input_dim = train_split["geometry"].shape[1]
    output_dim = train_split["signals_norm"].shape[1]
    model = ParametricForwardSurrogate(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_split["geometry"])
        loss = F.mse_loss(pred, train_split["signals_norm"])
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.history_interval == 0 or step == args.steps:
            history.append({"step": step, "train_signal_mse_norm": float(loss.detach().cpu())})

    train_metrics = evaluate_model(model, train_split, signal_stats, "train")
    val_metrics = evaluate_model(model, val_split, signal_stats, "val")
    test_metrics = evaluate_model(model, test_split, signal_stats, "test")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_single_row_csv(out / "metrics.csv", train_metrics)
    _write_single_row_csv(out / "eval_metrics.csv", val_metrics)
    _write_single_row_csv(out / "test_metrics.csv", test_metrics)
    _write_history(out / "training_history.csv", history)

    summary = [
        "# COMSOL parametric forward surrogate run summary",
        "",
        f"- steps: {args.steps}",
        f"- lr: {args.lr}",
        f"- hidden_dim: {args.hidden_dim}",
        f"- num_layers: {args.num_layers}",
        f"- max_components: {args.max_components}",
        f"- input_dim: {input_dim}",
        f"- output_dim: {output_dim}",
        f"- target_schema: {', '.join(train_targets['target_schema'])}",
        f"- type_vocab: {', '.join(train_targets['type_vocab'])}",
        "- signal_normalization: train_zscore",
        "- checkpoint_saved: false",
        "- weights_saved: false",
        "",
        "## Metrics",
        "",
        f"- train: {_format_metric_row(train_metrics)}",
        f"- val: {_format_metric_row(val_metrics)}",
        f"- test: {_format_metric_row(test_metrics)}",
    ]
    (out / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

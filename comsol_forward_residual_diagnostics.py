"""Forward surrogate residual sensitivity diagnostics for COMSOL parametric geometry."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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


def _flatten_signals(signals: np.ndarray) -> np.ndarray:
    if signals.ndim == 3:
        return signals.reshape(signals.shape[0], -1).astype(np.float32)
    if signals.ndim == 2:
        return signals.astype(np.float32)
    raise ValueError(f"signals must have shape [N,C,L] or [N,L], got {signals.shape}")


def load_npz_signals(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "signals" not in data:
            raise ValueError(f"{path} does not contain signals.")
        signals = _flatten_signals(data["signals"].astype(np.float32))
    if not np.isfinite(signals).all():
        raise ValueError(f"{path} contains NaN/Inf signals.")
    return signals


def load_targets(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        continuous = data["continuous_targets"].astype(np.float32)
        presence = data["presence_targets"].astype(np.float32)
        type_targets = data["type_targets"].astype(np.int64)
        sample_indices = data["sample_indices"].astype(np.int64) if "sample_indices" in data else np.arange(continuous.shape[0])
        target_schema = [str(x) for x in data["target_schema"]]
        type_vocab = [str(x) for x in data["type_vocab"]]
    return {
        "continuous": continuous,
        "presence": presence,
        "type_targets": type_targets,
        "sample_indices": sample_indices,
        "target_schema": target_schema,
        "type_vocab": type_vocab,
    }


def _train_forward_model(
    signals_flat: np.ndarray,
    targets: dict,
    steps: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
) -> tuple[ParametricForwardSurrogate, SignalNormalizationStats]:
    stats = compute_train_zscore_stats(signals_flat)
    signals_norm = torch.from_numpy(normalize_signals(signals_flat, stats)).to(device)
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
    model = ParametricForwardSurrogate(
        input_dim=geometry.shape[1],
        output_dim=signals_norm.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _step in range(steps):
        model.train()
        optimizer.zero_grad()
        pred = model(geometry)
        loss = F.mse_loss(pred, signals_norm)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, stats


def _load_prediction_geometry(prediction_dir: str | Path, split: str, targets: dict) -> dict | None:
    pred_file = Path(prediction_dir) / f"{split}_predictions.csv"
    if not pred_file.exists():
        return None
    sample_to_pos = {int(sample): i for i, sample in enumerate(targets["sample_indices"])}
    continuous = np.zeros_like(targets["continuous"], dtype=np.float32)
    presence = np.zeros_like(targets["presence"], dtype=np.float32)
    type_targets = np.full_like(targets["type_targets"], -1, dtype=np.int64)
    type_vocab = targets["type_vocab"]
    type_to_id = {name: idx for idx, name in enumerate(type_vocab)}
    with pred_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_index = int(float(row["sample_index"]))
            if sample_index not in sample_to_pos:
                continue
            sample_pos = sample_to_pos[sample_index]
            slot = int(float(row["component_slot"]))
            if slot < 0 or slot >= continuous.shape[1]:
                continue
            presence[sample_pos, slot] = float(row.get("presence_pred", row.get("presence_prob", 0.0)))
            type_name = row.get("type_pred", "")
            type_targets[sample_pos, slot] = type_to_id.get(type_name, -1)
            continuous[sample_pos, slot, 0] = float(row["center_x_pred"])
            continuous[sample_pos, slot, 1] = float(row["center_y_pred"])
            continuous[sample_pos, slot, 2] = float(row["axis_x_pred"])
            continuous[sample_pos, slot, 3] = float(row["axis_y_pred"])
            continuous[sample_pos, slot, 4] = float(row["depth_pred"])
            continuous[sample_pos, slot, 5] = float(row["rotation_pred"])
    return {"continuous": continuous, "presence": presence, "type_targets": type_targets}


def _load_prediction_mask_iou(prediction_dir: str | Path, split: str) -> dict[int, float]:
    path = Path(prediction_dir) / f"{split}_prediction_mask_metrics.csv"
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[int(float(row["sample_index"]))] = float(row["pred_mask_iou"])
    return out


def _variant_geometries(targets: dict, args: argparse.Namespace) -> dict[str, dict]:
    variants: dict[str, dict] = {
        "true_geometry": {
            "continuous": targets["continuous"].copy(),
            "presence": targets["presence"].copy(),
            "type_targets": targets["type_targets"].copy(),
        }
    }
    if len(targets["type_vocab"]) > 1:
        swapped = targets["type_targets"].copy()
        for i in range(swapped.shape[0]):
            present_slots = np.where(targets["presence"][i] > 0.5)[0]
            for slot in present_slots[: args.num_type_swaps]:
                if swapped[i, slot] >= 0:
                    swapped[i, slot] = (swapped[i, slot] + 1) % len(targets["type_vocab"])
        variants["type_swapped_geometry"] = {
            "continuous": targets["continuous"].copy(),
            "presence": targets["presence"].copy(),
            "type_targets": swapped,
        }
    rotated = targets["continuous"].copy()
    if "rotation_angle" in targets["target_schema"]:
        ridx = targets["target_schema"].index("rotation_angle")
        rotated[:, :, ridx] = rotated[:, :, ridx] + args.rotation_perturb_deg * (targets["presence"] > 0.5)
    variants["rotation_perturbed_geometry"] = {
        "continuous": rotated,
        "presence": targets["presence"].copy(),
        "type_targets": targets["type_targets"].copy(),
    }
    scaled = targets["continuous"].copy()
    for name in ["axis_x", "axis_y"]:
        if name in targets["target_schema"]:
            idx = targets["target_schema"].index(name)
            scaled[:, :, idx] = np.where(targets["presence"] > 0.5, scaled[:, :, idx] * args.axis_scale_perturb, scaled[:, :, idx])
    variants["axis_scaled_geometry"] = {
        "continuous": scaled,
        "presence": targets["presence"].copy(),
        "type_targets": targets["type_targets"].copy(),
    }
    return variants


def _corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom < eps:
        return 0.0
    return float(np.sum(aa * bb) / denom)


def _score_variant(
    model: ParametricForwardSurrogate,
    stats: SignalNormalizationStats,
    signals_flat: np.ndarray,
    geometry: dict,
    targets: dict,
    split: str,
    variant: str,
    pred_mask_iou_by_sample: dict[int, float],
) -> list[dict]:
    device = next(model.parameters()).device
    with torch.no_grad():
        geometry_vector = build_forward_geometry_vector(
            presence=torch.from_numpy(geometry["presence"].astype(np.float32)).to(device),
            type_targets_or_probs=torch.from_numpy(geometry["type_targets"].astype(np.int64)).to(device),
            continuous=torch.from_numpy(geometry["continuous"].astype(np.float32)).to(device),
            num_types=len(targets["type_vocab"]),
            target_schema=targets["target_schema"],
        )
        pred_norm = model(geometry_vector).detach().cpu().numpy().astype(np.float32)
    true_norm = normalize_signals(signals_flat, stats)
    pred_raw = denormalize_signals(pred_norm, stats)
    rows = []
    for i, sample_index in enumerate(targets["sample_indices"]):
        mse_norm = float(np.mean((pred_norm[i] - true_norm[i]) ** 2))
        rmse_raw = float(np.sqrt(np.mean((pred_raw[i] - signals_flat[i]) ** 2)))
        scale = float(np.sqrt(np.mean(signals_flat[i] ** 2)) + 1e-8)
        peak_abs_error = float(abs(np.max(np.abs(pred_raw[i])) - np.max(np.abs(signals_flat[i]))))
        row = {
            "split": split,
            "sample_index": int(sample_index),
            "geometry_variant": variant,
            "signal_mse_norm": mse_norm,
            "signal_nrmse": rmse_raw / scale,
            "signal_corr": _corr(pred_raw[i], signals_flat[i]),
            "peak_abs_error": peak_abs_error,
            "pred_mask_iou": pred_mask_iou_by_sample.get(int(sample_index), float("nan")) if variant == "predicted_geometry" else float("nan"),
        }
        rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["geometry_variant"])].append(row)
    agg = []
    for (split, variant), items in sorted(grouped.items()):
        pred_ious = np.asarray([x["pred_mask_iou"] for x in items], dtype=np.float64)
        pred_ious = pred_ious[np.isfinite(pred_ious)]
        agg.append(
            {
                "split": split,
                "geometry_variant": variant,
                "count": len(items),
                "avg_signal_mse_norm": float(np.mean([x["signal_mse_norm"] for x in items])),
                "avg_signal_nrmse": float(np.mean([x["signal_nrmse"] for x in items])),
                "avg_signal_corr": float(np.mean([x["signal_corr"] for x in items])),
                "avg_peak_abs_error": float(np.mean([x["peak_abs_error"] for x in items])),
                "avg_pred_mask_iou": float(np.mean(pred_ious)) if pred_ious.size else float("nan"),
            }
        )
    return agg


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _residual_iou_corr(rows: list[dict]) -> float:
    xs = []
    ys = []
    for row in rows:
        if row["geometry_variant"] == "predicted_geometry" and np.isfinite(row["pred_mask_iou"]):
            xs.append(row["signal_nrmse"])
            ys.append(row["pred_mask_iou"])
    if len(xs) < 2:
        return float("nan")
    return _corr(np.asarray(xs), np.asarray(ys))


def _write_summary(path: Path, split: str, agg: list[dict], rows: list[dict]) -> None:
    by_variant = {row["geometry_variant"]: row for row in agg}
    true_nrmse = by_variant.get("true_geometry", {}).get("avg_signal_nrmse", float("nan"))
    lines = [
        f"# S152 forward residual diagnostic ({split})",
        "",
        "| geometry_variant | avg_signal_nrmse | avg_signal_corr | avg_peak_abs_error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in agg:
        lines.append(
            f"| {row['geometry_variant']} | {row['avg_signal_nrmse']:.6e} | "
            f"{row['avg_signal_corr']:.6e} | {row['avg_peak_abs_error']:.6e} |"
        )
    lines.extend(["", "## 判断", ""])
    for variant in ["type_swapped_geometry", "rotation_perturbed_geometry", "axis_scaled_geometry", "predicted_geometry"]:
        if variant in by_variant and np.isfinite(true_nrmse):
            delta = by_variant[variant]["avg_signal_nrmse"] - true_nrmse
            lines.append(f"- `{variant}` 相对 true_geometry 的 avg_signal_nrmse delta = `{delta:.6e}`。")
    corr = _residual_iou_corr(rows)
    if np.isfinite(corr):
        lines.append(f"- predicted_geometry 的 signal_nrmse 与 pred_mask_iou 相关系数 = `{corr:.6e}`。")
    else:
        lines.append("- predicted_geometry 没有足够 mask IoU 数据用于 residual / mask_iou 相关性判断。")
    lines.append("")
    lines.append("如果扰动 geometry 的 residual 没有明显高于 true_geometry，forward residual 不适合作为强训练 loss；如果只对部分扰动敏感，应作为 diagnostic 或定向约束使用。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--targets-path", default="")
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--num-type-swaps", type=int, default=1)
    parser.add_argument("--rotation-perturb-deg", type=float, default=10.0)
    parser.add_argument("--axis-scale-perturb", type=float, default=1.2)
    parser.add_argument("--forward-steps", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    required = [args.npz_path, args.targets_path, args.output_dir, args.split]
    if not any(required):
        parse_args(["--help"])
        return 0
    if not all(required):
        raise ValueError("npz-path, targets-path, output-dir, and split are required.")
    if args.forward_steps <= 0:
        raise ValueError("forward-steps must be positive.")
    device = torch.device("cpu")
    signals = load_npz_signals(args.npz_path)
    targets = load_targets(args.targets_path)
    if signals.shape[0] != targets["presence"].shape[0]:
        raise ValueError("signals and targets sample counts do not match.")
    model, stats = _train_forward_model(
        signals,
        targets,
        steps=args.forward_steps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
    )
    variants = _variant_geometries(targets, args)
    pred = _load_prediction_geometry(args.prediction_dir, args.split, targets) if args.prediction_dir else None
    if pred is not None:
        variants["predicted_geometry"] = pred
    pred_mask_iou = _load_prediction_mask_iou(args.prediction_dir, args.split) if args.prediction_dir else {}
    rows: list[dict] = []
    for variant, geometry in variants.items():
        rows.extend(_score_variant(model, stats, signals, geometry, targets, args.split, variant, pred_mask_iou))
    agg = _aggregate(rows)
    out = Path(args.output_dir)
    _write_csv(out / "per_sample_forward_residual.csv", rows)
    _write_csv(out / "aggregate_forward_residual.csv", agg)
    _write_summary(out / "residual_sensitivity_summary.md", args.split, agg, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

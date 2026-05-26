"""Train COMSOL parametric inverse model with an in-memory learned forward consistency loss."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from comsol_parametric_forward_surrogate import (
    ParametricForwardSurrogate,
    build_forward_geometry_vector,
    compute_train_zscore_stats,
    denormalize_signals,
    normalize_signals,
)
from comsol_parametric_inverse_models import ParametricInverseNet
from train_comsol_parametric_forward_surrogate import (
    build_split_tensors as build_forward_split_tensors,
    load_signal_npz,
    load_targets as load_forward_targets,
)
from train_comsol_parametric_inverse import (
    build_tensors,
    compute_continuous_norm,
    compute_loss,
    evaluate,
    load_dataset,
    load_targets,
    write_csv,
)


def _corr_flat(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    pred_f = pred.reshape(-1).astype(np.float64)
    true_f = true.reshape(-1).astype(np.float64)
    pred_c = pred_f - pred_f.mean()
    true_c = true_f - true_f.mean()
    denom = np.sqrt(np.sum(pred_c**2) * np.sum(true_c**2))
    if denom < eps:
        return 0.0
    return float(np.sum(pred_c * true_c) / denom)


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_param_loss_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        lambda_presence=1.0,
        lambda_type=1.0,
        lambda_continuous=1.0,
        lambda_center=1.0,
        lambda_axis=1.0,
        lambda_depth=1.0,
        lambda_rotation=1.0,
        component_matching_mode="fixed",
        lambda_raster_bce=0.0,
        lambda_raster_dice=0.0,
        raster_loss_start_step=0,
        raster_softness_cells=1.0,
        raster_target_source="masks",
    )


def _train_forward_surrogate(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ParametricForwardSurrogate, dict, dict, dict, object]:
    train_signals = load_signal_npz(args.train_npz)
    val_signals = load_signal_npz(args.val_npz)
    test_signals = load_signal_npz(args.test_npz)
    signal_stats = compute_train_zscore_stats(train_signals["signals_flat"])
    train_targets = load_forward_targets(args.train_targets)
    val_targets = load_forward_targets(args.val_targets, expected_type_vocab=train_targets["type_vocab"])
    test_targets = load_forward_targets(args.test_targets, expected_type_vocab=train_targets["type_vocab"])
    for split_name, targets in [("val", val_targets), ("test", test_targets)]:
        if targets["target_schema"] != train_targets["target_schema"]:
            raise ValueError(f"{split_name} target_schema differs from train target_schema.")

    train_split = build_forward_split_tensors(train_signals, train_targets, signal_stats, device)
    val_split = build_forward_split_tensors(val_signals, val_targets, signal_stats, device)
    test_split = build_forward_split_tensors(test_signals, test_targets, signal_stats, device)
    train_presence = torch.from_numpy(train_targets["presence"]).to(device)
    train_type_targets = torch.from_numpy(train_targets["type_targets"]).to(device)
    train_continuous = torch.from_numpy(train_targets["continuous"]).to(device)
    num_types = len(train_targets["type_vocab"])
    model = ParametricForwardSurrogate(
        input_dim=train_split["geometry"].shape[1],
        output_dim=train_split["signals_norm"].shape[1],
        hidden_dim=args.forward_hidden_dim,
        num_layers=args.forward_num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _step in range(1, args.forward_pretrain_steps + 1):
        model.train()
        optimizer.zero_grad()
        if args.forward_soft_input_augmentation > 0.0:
            geometry_input = _soft_augmented_geometry(
                train_presence,
                train_type_targets,
                train_continuous,
                num_types,
                train_targets["target_schema"],
                args.forward_soft_input_augmentation,
            )
        else:
            geometry_input = train_split["geometry"]
        pred = model(geometry_input)
        loss = F.mse_loss(pred, train_split["signals_norm"])
        loss.backward()
        optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, train_split, val_split, test_split, signal_stats


def _soft_augmented_geometry(
    presence: torch.Tensor,
    type_targets: torch.Tensor,
    continuous: torch.Tensor,
    num_types: int,
    target_schema: list[str],
    strength: float,
) -> torch.Tensor:
    strength = max(0.0, min(float(strength), 0.45))
    presence_noise = (torch.rand_like(presence) - 0.5) * (2.0 * strength)
    presence_soft = torch.clamp(presence + presence_noise, 0.0, 1.0)
    type_one_hot = torch.zeros((*presence.shape, num_types), dtype=continuous.dtype, device=continuous.device)
    valid = (type_targets >= 0) & (type_targets < num_types)
    if valid.any():
        type_one_hot.scatter_(2, type_targets.clamp(0, num_types - 1).unsqueeze(-1), 1.0)
        type_one_hot = type_one_hot * valid.unsqueeze(-1).to(type_one_hot.dtype)
    uniform = torch.full_like(type_one_hot, 1.0 / float(num_types))
    mix = torch.rand((*presence.shape, 1), dtype=continuous.dtype, device=continuous.device) * strength
    type_soft = torch.where(valid.unsqueeze(-1), (1.0 - mix) * type_one_hot + mix * uniform, type_one_hot)
    return build_forward_geometry_vector(
        presence=presence_soft,
        type_targets_or_probs=type_soft,
        continuous=continuous,
        num_types=num_types,
        target_schema=target_schema,
    )


def _forward_consistency_loss(
    inverse_model: ParametricInverseNet,
    forward_model: ParametricForwardSurrogate,
    tensors: dict,
    target_signal_norm: torch.Tensor,
    num_types: int,
) -> torch.Tensor:
    out = inverse_model(tensors["signals"], tensors.get("features"))
    continuous_unscaled = (
        out["continuous"] * tensors["continuous_std"].detach().view(1, 1, -1)
        + tensors["continuous_mean"].detach().view(1, 1, -1)
    )
    presence_st, type_st = _straight_through_presence_type(out, num_types)
    geometry_vector = build_forward_geometry_vector(
        presence=presence_st,
        type_targets_or_probs=type_st,
        continuous=continuous_unscaled,
        num_types=num_types,
        target_schema=tensors["target_schema"],
    )
    signal_pred_norm = forward_model(geometry_vector)
    return F.mse_loss(signal_pred_norm, target_signal_norm)


def _straight_through_presence_type(out: dict, num_types: int) -> tuple[torch.Tensor, torch.Tensor]:
    presence_prob = out["presence_prob"]
    presence_hard = (presence_prob >= 0.5).to(presence_prob.dtype)
    presence_st = presence_hard + presence_prob - presence_prob.detach()
    type_probs = torch.softmax(out["type_logits"], dim=-1)
    type_ids = type_probs.argmax(dim=-1)
    type_hard = F.one_hot(type_ids, num_classes=num_types).to(type_probs.dtype)
    type_st = type_hard + type_probs - type_probs.detach()
    return presence_st, type_st


def _forward_residual_metrics(
    inverse_model: ParametricInverseNet,
    forward_model: ParametricForwardSurrogate,
    tensors: dict,
    raw_signals_flat: np.ndarray,
    signal_stats,
    num_types: int,
) -> dict[str, float]:
    inverse_model.eval()
    forward_model.eval()
    with torch.no_grad():
        out = inverse_model(tensors["signals"], tensors.get("features"))
        continuous_unscaled = (
            out["continuous"] * tensors["continuous_std"].detach().view(1, 1, -1)
            + tensors["continuous_mean"].detach().view(1, 1, -1)
        )
        presence_st, type_st = _straight_through_presence_type(out, num_types)
        geometry_vector = build_forward_geometry_vector(
            presence=presence_st,
            type_targets_or_probs=type_st,
            continuous=continuous_unscaled,
            num_types=num_types,
            target_schema=tensors["target_schema"],
        )
        pred_norm = forward_model(geometry_vector).detach().cpu().numpy().astype(np.float32)
    pred_raw = denormalize_signals(pred_norm, signal_stats)
    raw_rmse = float(np.sqrt(np.mean((pred_raw - raw_signals_flat) ** 2)))
    raw_scale = float(np.sqrt(np.mean(raw_signals_flat**2)) + 1e-8)
    return {
        "forward_signal_nrmse": raw_rmse / raw_scale,
        "forward_signal_corr": _corr_flat(pred_raw, raw_signals_flat),
    }


def _load_inverse_splits(args: argparse.Namespace, device: torch.device) -> tuple[dict, dict, dict, dict, dict, dict, np.ndarray, np.ndarray]:
    train_dataset = load_dataset(args.train_npz)
    val_dataset = load_dataset(args.val_npz)
    test_dataset = load_dataset(args.test_npz)
    train_targets = load_targets(args.train_targets)
    val_targets = load_targets(args.val_targets, train_targets["type_vocab"])
    test_targets = load_targets(args.test_targets, train_targets["type_vocab"])
    mean, std = compute_continuous_norm(train_targets)
    train_tensors = build_tensors(train_dataset, train_targets, mean, std, device)
    val_tensors = build_tensors(val_dataset, val_targets, mean, std, device)
    test_tensors = build_tensors(test_dataset, test_targets, mean, std, device)
    return train_tensors, val_tensors, test_tensors, train_targets, val_targets, test_targets, mean, std


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forward_model, forward_train, forward_val, forward_test, signal_stats = _train_forward_surrogate(args, device)
    train_tensors, val_tensors, test_tensors, train_targets, val_targets, test_targets, mean, std = _load_inverse_splits(args, device)
    train_signal_norm = forward_train["signals_norm"]
    val_signal_norm = forward_val["signals_norm"]
    test_signal_norm = forward_test["signals_norm"]

    signal_len = int(train_tensors["signals"].shape[1])
    num_types = len(train_targets["type_vocab"])
    num_continuous = len(train_targets["target_schema"])
    inverse_model = ParametricInverseNet(
        signal_len=signal_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_components=args.max_components,
        num_types=num_types,
        num_continuous=num_continuous,
        num_layers=3,
        encoder_type="mlp",
        head_mode="shared",
    ).to(device)
    optimizer = torch.optim.Adam(inverse_model.parameters(), lr=args.lr)
    param_loss_args = _make_param_loss_args(args)
    history: list[dict] = []
    for step in range(1, args.inverse_steps + 1):
        inverse_model.train()
        optimizer.zero_grad()
        param_loss, param_parts = compute_loss(inverse_model, train_tensors, param_loss_args, type_weights=None, step=step)
        forward_loss = _forward_consistency_loss(
            inverse_model,
            forward_model,
            train_tensors,
            train_signal_norm,
            num_types=num_types,
        )
        total_loss = param_loss + args.lambda_forward_consistency * forward_loss
        total_loss.backward()
        optimizer.step()
        if step == 1 or step == args.inverse_steps or step % 250 == 0:
            history.append(
                {
                    "phase": "inverse",
                    "step": step,
                    "param_loss": float(param_loss.detach().cpu()),
                    "forward_consistency_loss": float(forward_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    **param_parts,
                }
            )

    metric_specs = [
        ("train", train_tensors, train_targets, forward_train["signals_flat"], train_signal_norm),
        ("val", val_tensors, val_targets, forward_val["signals_flat"], val_signal_norm),
        ("test", test_tensors, test_targets, forward_test["signals_flat"], test_signal_norm),
    ]
    metric_rows: dict[str, dict] = {}
    for split, tensors, targets, raw_signals, _target_norm in metric_specs:
        row = evaluate(inverse_model, tensors, targets, mean, std, split)
        row.update(_forward_residual_metrics(inverse_model, forward_model, tensors, raw_signals, signal_stats, num_types))
        row["lambda_forward_consistency"] = args.lambda_forward_consistency
        row["forward_pretrain_steps"] = args.forward_pretrain_steps
        row["inverse_steps"] = args.inverse_steps
        metric_rows[split] = row

    write_csv(output_dir / "metrics.csv", [metric_rows["train"]])
    write_csv(output_dir / "eval_metrics.csv", [metric_rows["val"]])
    write_csv(output_dir / "test_metrics.csv", [metric_rows["test"]])
    _write_rows(output_dir / "training_history.csv", history)
    summary = [
        "# COMSOL parametric inverse forward consistency run summary",
        "",
        f"- forward_pretrain_steps: {args.forward_pretrain_steps}",
        f"- inverse_steps: {args.inverse_steps}",
        f"- lambda_forward_consistency: {args.lambda_forward_consistency}",
        f"- forward_hidden_dim: {args.forward_hidden_dim}",
        f"- forward_num_layers: {args.forward_num_layers}",
        f"- forward_soft_input_augmentation: {args.forward_soft_input_augmentation}",
        "- forward_surrogate_saved: false",
        "- inverse_weights_saved: false",
        "- checkpoint_saved: false",
        "",
        "## Metrics",
        "",
    ]
    for split in ["train", "val", "test"]:
        row = metric_rows[split]
        summary.append(
            f"- {split}: mask_iou={row['param_mask_iou']:.6e}, "
            f"type_acc={row['type_accuracy_present']:.6e}, "
            f"rotation_mae={row['rotation_mae']:.6e}, "
            f"forward_signal_nrmse={row['forward_signal_nrmse']:.6e}, "
            f"forward_signal_corr={row['forward_signal_corr']:.6e}"
        )
    (output_dir / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", default="")
    parser.add_argument("--train-targets", default="")
    parser.add_argument("--val-npz", default="")
    parser.add_argument("--val-targets", default="")
    parser.add_argument("--test-npz", default="")
    parser.add_argument("--test-targets", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--forward-pretrain-steps", type=int, default=3000)
    parser.add_argument("--inverse-steps", type=int, default=3000)
    parser.add_argument("--lambda-forward-consistency", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--forward-hidden-dim", type=int, default=256)
    parser.add_argument("--forward-num-layers", type=int, default=4)
    parser.add_argument("--forward-soft-input-augmentation", type=float, default=0.15)
    parser.add_argument("--max-components", type=int, default=3)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    required = [
        args.train_npz,
        args.train_targets,
        args.val_npz,
        args.val_targets,
        args.test_npz,
        args.test_targets,
        args.output_dir,
    ]
    if not all(required):
        parser.print_help()
        print(
            "\nExample: python train_comsol_parametric_inverse_forward_consistency.py "
            "--train-npz train.npz --train-targets train_targets.npz "
            "--val-npz val.npz --val-targets val_targets.npz "
            "--test-npz test.npz --test-targets test_targets.npz --output-dir out"
        )
        return 0
    if args.forward_pretrain_steps <= 0 or args.inverse_steps <= 0:
        raise ValueError("forward-pretrain-steps and inverse-steps must be positive.")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal supervised runner for signal-conditioned dual-network models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from conditional_dual_data_utils import get_conditional_batch, infer_signal_len, load_conditional_npz
from conditional_dual_models import ConditionalDualNet


def parse_sample_indices(text: str) -> list[int]:
    indices = []
    for item in text.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    if not indices:
        raise ValueError("--sample-indices must contain at least one index")
    return indices


def dataset_num_samples(dataset) -> int:
    if "signals" not in dataset:
        raise ValueError("dataset must contain 'signals'")
    return int(dataset["signals"].shape[0])


def parse_eval_sample_indices(text: str, dataset) -> list[int]:
    if text.strip():
        return parse_sample_indices(text)
    return list(range(dataset_num_samples(dataset)))


def soft_dice_loss(pred, target, eps=1.0e-6):
    dims = tuple(range(1, pred.ndim))
    intersection = torch.sum(pred * target, dim=dims)
    denominator = torch.sum(pred, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return torch.mean(1.0 - dice)


def batch_mean_iou_from_mask(pred_mask, mask_label):
    label_mask = mask_label > 0.5
    ious = []
    for pred, label in zip(pred_mask, label_mask):
        intersection = torch.logical_and(pred, label).sum().item()
        union = torch.logical_or(pred, label).sum().item()
        ious.append(0.0 if union == 0 else intersection / union)
    return sum(ious) / len(ious)


def prediction_mask_from_output(output, mask_head_mode: str):
    if mask_head_mode == "mu_threshold":
        return output["mu"] < 500.0
    if mask_head_mode == "direct":
        if "mask_prob" not in output:
            raise ValueError("mask_head_mode=direct requires model output to include mask_prob")
        return output["mask_prob"] >= 0.5
    raise ValueError(f"unsupported mask head mode: {mask_head_mode}")


def soft_defect_from_output(output, mask_head_mode: str, mask_temperature: float):
    if mask_head_mode == "mu_threshold":
        return torch.sigmoid((500.0 - output["mu"]) / mask_temperature)
    if mask_head_mode == "direct":
        if "mask_prob" not in output:
            raise ValueError("mask_head_mode=direct requires model output to include mask_prob")
        return output["mask_prob"]
    raise ValueError(f"unsupported mask head mode: {mask_head_mode}")


def mask_bce_loss(pred_prob, target, mode: str, pos_weight: float, focal_gamma: float, focal_alpha: float):
    target = target.float()
    if mode == "bce":
        return F.binary_cross_entropy(pred_prob, target)
    if mode == "pos_weighted_bce":
        bce = F.binary_cross_entropy(pred_prob, target, reduction="none")
        weight = 1.0 + (pos_weight - 1.0) * target
        return torch.mean(weight * bce)
    if mode == "focal_bce":
        eps = 1.0e-6
        prob = pred_prob.clamp(eps, 1.0 - eps)
        bce = F.binary_cross_entropy(prob, target, reduction="none")
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = focal_alpha * target + (1.0 - focal_alpha) * (1.0 - target)
        return torch.mean(alpha_t * torch.pow(1.0 - pt, focal_gamma) * bce)
    raise ValueError(f"unsupported mask BCE mode: {mode}")


def area_calibration_loss(soft_defect, target_mask_label, mode: str, foreground_floor_ratio: float):
    pred_dims = tuple(range(1, soft_defect.ndim))
    target_dims = tuple(range(1, target_mask_label.ndim))
    num_points = max(1, int(soft_defect[0].numel()))
    target_num_points = max(1, int(target_mask_label[0].numel()))
    pred_area_soft = soft_defect.sum(dim=pred_dims)
    true_area = target_mask_label.float().sum(dim=target_dims)
    pred_ratio = pred_area_soft / float(num_points)
    true_ratio = true_area / float(target_num_points)
    if mode == "none":
        area_loss = soft_defect.new_zeros(())
    elif mode == "batch_ratio_mse":
        area_loss = torch.mean((pred_ratio - true_ratio) ** 2)
    elif mode == "foreground_floor":
        floor_ratio = foreground_floor_ratio * true_ratio
        area_loss = torch.mean(torch.relu(floor_ratio - pred_ratio) ** 2)
    else:
        raise ValueError(f"unsupported area loss mode: {mode}")
    return area_loss, pred_area_soft.mean(), true_area.mean()


def threshold_margin_loss(
    mu,
    target_mask_label,
    mode: str,
    positive_mu_margin: float,
    negative_mu_margin: float,
    mu_threshold: float = 500.0,
):
    positive_mask = target_mask_label > 0.5
    negative_mask = ~positive_mask
    zero = mu.new_zeros(())

    positive_mu = mu[positive_mask]
    negative_mu = mu[negative_mask]
    positive_count = int(positive_mu.numel())
    negative_count = int(negative_mu.numel())
    positive_mu_mean = positive_mu.mean() if positive_count else zero
    negative_mu_mean = negative_mu.mean() if negative_count else zero

    if mode == "none":
        positive_loss = zero
        negative_loss = zero
    elif mode in {"positive_hinge", "bidirectional_hinge"}:
        target_positive_mu = mu_threshold - positive_mu_margin
        positive_loss = (
            torch.mean(torch.relu(positive_mu - target_positive_mu) ** 2)
            if positive_count
            else zero
        )
        if mode == "bidirectional_hinge":
            target_negative_mu = mu_threshold + negative_mu_margin
            negative_loss = (
                torch.mean(torch.relu(target_negative_mu - negative_mu) ** 2)
                if negative_count
                else zero
            )
        else:
            negative_loss = zero
    else:
        raise ValueError(f"unsupported threshold margin mode: {mode}")

    margin_loss = positive_loss + negative_loss
    return {
        "threshold_margin_loss": margin_loss,
        "positive_margin_loss": positive_loss,
        "negative_margin_loss": negative_loss,
        "sampled_positive_count": positive_count,
        "sampled_negative_count": negative_count,
        "sampled_mu_positive_mean": positive_mu_mean,
        "sampled_mu_negative_mean": negative_mu_mean,
    }


def per_sample_metrics(sample_indices, mu, mu_label, mask_label, pred_mask):
    label_mask = mask_label > 0.5
    rows = []
    for row_idx, sample_index in enumerate(sample_indices):
        pred = pred_mask[row_idx]
        label = label_mask[row_idx]
        intersection = torch.logical_and(pred, label).sum().item()
        union = torch.logical_or(pred, label).sum().item()
        iou = 0.0 if union == 0 else intersection / union
        area_pred = int(pred.sum().item())
        area_label = int(label.sum().item())
        diff = mu[row_idx] - mu_label[row_idx]
        rows.append(
            {
                "sample_index": sample_index,
                "defect_iou": iou,
                "defect_area_pred": area_pred,
                "defect_area_label": area_label,
                "mu_mse": torch.mean(diff * diff).item(),
                "mu_mae": torch.mean(torch.abs(diff)).item(),
            }
        )
    return rows


def write_metrics(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "signal_normalization",
                "signal_feature_mode",
                "conditioning_mode",
                "encoder_type",
                "point_signal_mode",
                "mask_head_mode",
                "mask_source",
                "mask_bce_mode",
                "area_loss_mode",
                "lambda_area_loss",
                "threshold_margin_mode",
                "lambda_threshold_margin",
                "defect_iou",
                "defect_area_pred",
                "defect_area_label",
                "mu_mse",
                "mu_mae",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def add_metric_metadata(
    rows,
    signal_normalization: str,
    signal_feature_mode: str,
    conditioning_mode: str,
    encoder_type: str,
    point_signal_mode: str,
    mask_head_mode: str,
    mask_source: str,
    mask_bce_mode: str,
    area_loss_mode: str,
    lambda_area_loss: float,
    threshold_margin_mode: str,
    lambda_threshold_margin: float,
):
    for row in rows:
        row["signal_normalization"] = signal_normalization
        row["signal_feature_mode"] = signal_feature_mode
        row["conditioning_mode"] = conditioning_mode
        row["encoder_type"] = encoder_type
        row["point_signal_mode"] = point_signal_mode
        row["mask_head_mode"] = mask_head_mode
        row["mask_source"] = mask_source
        row["mask_bce_mode"] = mask_bce_mode
        row["area_loss_mode"] = area_loss_mode
        row["lambda_area_loss"] = lambda_area_loss
        row["threshold_margin_mode"] = threshold_margin_mode
        row["lambda_threshold_margin"] = lambda_threshold_margin
    return rows


def normalize_signals(signals, mode: str, stats=None, eps: float = 1.0e-8):
    if mode == "none":
        return signals, stats
    if mode == "train_zscore":
        if stats is None:
            mean = signals.mean()
            std = signals.std(unbiased=False).clamp_min(eps)
            stats = {"mean": mean, "std": std}
        return (signals - stats["mean"]) / stats["std"], stats
    if mode == "per_sample_zscore":
        mean = signals.mean(dim=1, keepdim=True)
        std = signals.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return (signals - mean) / std, stats
    raise ValueError(f"unsupported signal normalization mode: {mode}")


def normalize_batch_signals(batch, mode: str, stats=None):
    normalized_signals, stats = normalize_signals(batch["signals"], mode, stats=stats)
    normalized_batch = dict(batch)
    normalized_batch["signals"] = normalized_signals
    return normalized_batch, stats


def finite_difference_gradient(signals):
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape [B, signal_len], got {tuple(signals.shape)}")
    grad = torch.zeros_like(signals)
    signal_len = signals.shape[1]
    if signal_len == 1:
        return grad
    grad[:, 0] = signals[:, 1] - signals[:, 0]
    grad[:, -1] = signals[:, -1] - signals[:, -2]
    if signal_len > 2:
        grad[:, 1:-1] = 0.5 * (signals[:, 2:] - signals[:, :-2])
    return grad


def build_signal_features(signals, mode: str):
    if mode == "raw":
        return signals
    if mode == "raw_abs_grad":
        grad = finite_difference_gradient(signals)
        return torch.cat([signals, torch.abs(signals), grad], dim=1)
    raise ValueError(f"unsupported signal feature mode: {mode}")


def transform_batch_signals(batch, mode: str):
    transformed_batch = dict(batch)
    transformed_batch["signals"] = build_signal_features(batch["signals"], mode)
    return transformed_batch


def point_signal_feature_dim(mode: str) -> int:
    if mode == "none":
        return 0
    if mode == "local_value":
        return 1
    if mode == "local_value_abs":
        return 2
    raise ValueError(f"unsupported point signal mode: {mode}")


def build_point_signal_features(signals, coords, mode: str):
    feature_dim = point_signal_feature_dim(mode)
    if feature_dim == 0:
        return None
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape [B, signal_len], got {tuple(signals.shape)}")
    if coords.ndim == 2:
        if coords.shape[1] != 2:
            raise ValueError(f"coords with rank 2 must have shape [N, 2], got {tuple(coords.shape)}")
        x_coords = coords[:, 0]
    elif coords.ndim == 3:
        if coords.shape[2] != 2:
            raise ValueError(f"coords with rank 3 must have shape [B, N, 2], got {tuple(coords.shape)}")
        if coords.shape[0] != signals.shape[0]:
            raise ValueError(
                f"coords batch dimension must match signals batch {signals.shape[0]}, got {coords.shape[0]}"
            )
        x_coords = coords[0, :, 0]
    else:
        raise ValueError(f"coords must have shape [N, 2] or [B, N, 2], got {tuple(coords.shape)}")

    signal_len = signals.shape[1]
    x_min = torch.min(x_coords)
    x_max = torch.max(x_coords)
    x_range = x_max - x_min
    if torch.abs(x_range).item() < 1.0e-12:
        indices = torch.zeros_like(x_coords, dtype=torch.long, device=signals.device)
    else:
        scaled = (x_coords - x_min) / x_range * float(signal_len - 1)
        indices = torch.round(scaled).long().clamp(0, signal_len - 1).to(signals.device)
    local_values = signals.index_select(1, indices).unsqueeze(-1)
    if mode == "local_value":
        return local_values
    return torch.cat([local_values, torch.abs(local_values)], dim=-1)


def choose_random_point_indices(num_points: int, sample_count: int, device):
    return torch.randperm(num_points, device=device)[:sample_count]


def choose_positive_balanced_point_indices(mask_label, num_points: int, sample_count: int, positive_fraction: float):
    point_positive = (mask_label > 0.5).squeeze(-1)
    if point_positive.ndim != 2 or point_positive.shape[1] != num_points:
        raise ValueError(
            "mask_label must have shape [B, N, 1] for positive-balanced sampling, "
            f"got {tuple(mask_label.shape)} with num_points={num_points}"
        )
    positive_indices = torch.nonzero(point_positive.any(dim=0), as_tuple=False).flatten()
    negative_indices = torch.nonzero(~point_positive.any(dim=0), as_tuple=False).flatten()
    if positive_indices.numel() == 0 or negative_indices.numel() == 0:
        return choose_random_point_indices(num_points, sample_count, mask_label.device)

    positive_perm = positive_indices[torch.randperm(positive_indices.numel(), device=mask_label.device)]
    negative_perm = negative_indices[torch.randperm(negative_indices.numel(), device=mask_label.device)]
    positive_take = min(int(round(sample_count * positive_fraction)), int(positive_perm.numel()))
    negative_take = min(sample_count - positive_take, int(negative_perm.numel()))
    remaining = sample_count - positive_take - negative_take
    if remaining > 0:
        extra_positive = min(remaining, int(positive_perm.numel()) - positive_take)
        positive_take += extra_positive
        remaining -= extra_positive
    if remaining > 0:
        extra_negative = min(remaining, int(negative_perm.numel()) - negative_take)
        negative_take += extra_negative

    point_indices = torch.cat([positive_perm[:positive_take], negative_perm[:negative_take]], dim=0)
    if point_indices.numel() != sample_count:
        return choose_random_point_indices(num_points, sample_count, mask_label.device)
    shuffle = torch.randperm(point_indices.numel(), device=mask_label.device)
    return point_indices.index_select(0, shuffle)


def sample_training_points(
    coords,
    mu_label,
    mask_label,
    train_point_subsample: int,
    point_sampling_mode: str = "random",
    positive_fraction: float = 0.5,
):
    if train_point_subsample <= 0:
        return coords, mu_label, mask_label
    if coords.ndim == 2:
        num_points = coords.shape[0]
        sample_count = min(train_point_subsample, num_points)
        if point_sampling_mode == "random":
            point_indices = choose_random_point_indices(num_points, sample_count, coords.device)
        elif point_sampling_mode == "positive_balanced":
            point_indices = choose_positive_balanced_point_indices(
                mask_label,
                num_points,
                sample_count,
                positive_fraction,
            )
        else:
            raise ValueError(f"unsupported point sampling mode: {point_sampling_mode}")
        sampled_coords = coords.index_select(0, point_indices)
    elif coords.ndim == 3:
        num_points = coords.shape[1]
        sample_count = min(train_point_subsample, num_points)
        if point_sampling_mode == "random":
            point_indices = choose_random_point_indices(num_points, sample_count, coords.device)
        elif point_sampling_mode == "positive_balanced":
            point_indices = choose_positive_balanced_point_indices(
                mask_label,
                num_points,
                sample_count,
                positive_fraction,
            )
        else:
            raise ValueError(f"unsupported point sampling mode: {point_sampling_mode}")
        sampled_coords = coords.index_select(1, point_indices)
    else:
        raise ValueError(f"coords must have shape [N, 2] or [B, N, 2], got {tuple(coords.shape)}")
    sampled_mu_label = mu_label.index_select(1, point_indices)
    sampled_mask_label = mask_label.index_select(1, point_indices)
    return sampled_coords, sampled_mu_label, sampled_mask_label


def batch_point_count(batch) -> int:
    coords = batch["coords"]
    if coords.ndim == 2:
        return int(coords.shape[0])
    if coords.ndim == 3:
        return int(coords.shape[1])
    raise ValueError(f"coords must have shape [N, 2] or [B, N, 2], got {tuple(coords.shape)}")


def should_log_history(step: int, total_steps: int, history_interval: int) -> bool:
    if history_interval <= 0:
        return False
    return step == 1 or step == total_steps or step % history_interval == 0


def should_run_validation_selection(step: int, total_steps: int, args) -> bool:
    if args.val_selection_metric == "none":
        return False
    if args.val_selection_interval <= 0:
        return False
    return step == total_steps or step % args.val_selection_interval == 0


def clone_state_dict_to_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def batch_mean_area(mask):
    dims = tuple(range(1, mask.ndim))
    return mask.float().sum(dim=dims).mean().item()


def history_row(
    phase: str,
    step: int,
    total_loss,
    bce_loss,
    dice_loss,
    mu_mse_loss,
    area_loss,
    margin_stats,
    pred_area_soft_mean,
    true_area_mean,
    output,
    soft_defect,
    mask_label,
    args,
    eval_iou_at_step=None,
    eval_loss_at_step=None,
    is_best_step=False,
):
    pred_mask = prediction_mask_from_output({key: value.detach() for key, value in output.items()}, args.mask_head_mode)
    return {
        "phase": phase,
        "step": step,
        "total_loss": total_loss.item(),
        "bce_loss": bce_loss.item(),
        "dice_loss": dice_loss.item(),
        "mu_mse_loss": mu_mse_loss.item(),
        "area_loss": area_loss.item(),
        "threshold_margin_loss": margin_stats["threshold_margin_loss"].item(),
        "positive_margin_loss": margin_stats["positive_margin_loss"].item(),
        "negative_margin_loss": margin_stats["negative_margin_loss"].item(),
        "sampled_positive_count": margin_stats["sampled_positive_count"],
        "sampled_negative_count": margin_stats["sampled_negative_count"],
        "sampled_mu_positive_mean": margin_stats["sampled_mu_positive_mean"].item(),
        "sampled_mu_negative_mean": margin_stats["sampled_mu_negative_mean"].item(),
        "batch_iou": batch_mean_iou_from_mask(pred_mask, mask_label),
        "batch_area_pred": batch_mean_area(pred_mask),
        "batch_area_label": batch_mean_area(mask_label > 0.5),
        "pred_area_soft_mean": pred_area_soft_mean.item(),
        "true_area_mean": true_area_mean.item(),
        "mean_mu": output["mu"].detach().mean().item(),
        "min_mu": output["mu"].detach().min().item(),
        "max_mu": output["mu"].detach().max().item(),
        "mean_soft_defect": soft_defect.detach().mean().item(),
        "eval_iou_at_step": "" if eval_iou_at_step is None else eval_iou_at_step,
        "eval_loss_at_step": "" if eval_loss_at_step is None else eval_loss_at_step,
        "is_best_step": is_best_step,
        "mask_bce_mode": args.mask_bce_mode,
        "point_sampling_mode": args.point_sampling_mode,
        "train_point_subsample": args.train_point_subsample,
    }


def write_training_history(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "step",
                "total_loss",
                "bce_loss",
                "dice_loss",
                "mu_mse_loss",
                "area_loss",
                "threshold_margin_loss",
                "positive_margin_loss",
                "negative_margin_loss",
                "sampled_positive_count",
                "sampled_negative_count",
                "sampled_mu_positive_mean",
                "sampled_mu_negative_mean",
                "batch_iou",
                "batch_area_pred",
                "batch_area_label",
                "pred_area_soft_mean",
                "true_area_mean",
                "mean_mu",
                "min_mu",
                "max_mu",
                "mean_soft_defect",
                "eval_iou_at_step",
                "eval_loss_at_step",
                "is_best_step",
                "mask_bce_mode",
                "point_sampling_mode",
                "train_point_subsample",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def compute_eval_loss(model, batch, args, eval_batch_size: int) -> float:
    model.eval()
    sample_count = int(batch["signals"].shape[0])
    total_loss = 0.0
    total_samples = 0
    step_size = eval_batch_size if eval_batch_size > 0 else sample_count
    with torch.no_grad():
        for start in range(0, sample_count, step_size):
            stop = min(sample_count, start + step_size)
            sliced = slice_conditional_batch(batch, start, stop)
            point_features = build_point_signal_features(sliced["signals"], sliced["coords"], args.point_signal_mode)
            out = model(sliced["signals"], sliced["coords"], point_features=point_features, return_phi=False)
            mu = out["mu"]
            soft_defect = soft_defect_from_output(out, args.mask_head_mode, args.mask_temperature)
            bce_loss = mask_bce_loss(
                soft_defect,
                sliced["mask_label"],
                args.mask_bce_mode,
                args.pos_weight,
                args.focal_gamma,
                args.focal_alpha,
            )
            dice_loss = soft_dice_loss(soft_defect, sliced["mask_label"])
            mu_mse_loss = F.mse_loss(mu, sliced["mu_label"])
            area_loss, _, _ = area_calibration_loss(
                soft_defect,
                sliced["mask_label"],
                args.area_loss_mode,
                args.foreground_floor_ratio,
            )
            margin_stats = threshold_margin_loss(
                mu,
                sliced["mask_label"],
                args.threshold_margin_mode,
                args.positive_mu_margin,
                args.negative_mu_margin,
            )
            loss = (
                args.lambda_mask_bce * bce_loss
                + args.lambda_mask_dice * dice_loss
                + args.lambda_mu_mse * mu_mse_loss
                + args.lambda_area_loss * area_loss
                + args.lambda_threshold_margin * margin_stats["threshold_margin_loss"]
            )
            batch_n = stop - start
            total_loss += float(loss.item()) * batch_n
            total_samples += batch_n
    if total_samples == 0:
        raise ValueError("cannot compute eval loss with zero samples")
    return total_loss / total_samples


def maybe_update_best_validation_state(
    model,
    step: int,
    eval_batch,
    eval_indices,
    args,
    selection_state,
):
    if args.val_selection_metric == "none":
        return None, None, False
    eval_iou = None
    eval_loss = None
    if args.val_selection_metric == "eval_iou":
        rows = evaluate_model(
            model,
            eval_batch,
            eval_indices,
            args.point_signal_mode,
            args.mask_head_mode,
            args.eval_batch_size,
        )
        eval_iou = average_metrics(rows)["defect_iou"]
        score = eval_iou
        is_better = selection_state["best_score"] is None or score > selection_state["best_score"]
    elif args.val_selection_metric == "eval_loss":
        eval_loss = compute_eval_loss(model, eval_batch, args, args.eval_batch_size)
        score = eval_loss
        is_better = selection_state["best_score"] is None or score < selection_state["best_score"]
    else:
        raise ValueError(f"unsupported validation selection metric: {args.val_selection_metric}")

    if is_better:
        selection_state["best_score"] = score
        selection_state["best_state"] = clone_state_dict_to_cpu(model)
        selection_state["best_step"] = step
        selection_state["best_eval_iou"] = eval_iou
        selection_state["best_eval_loss"] = eval_loss
    return eval_iou, eval_loss, is_better


def train_phase(
    model,
    optimizer,
    phase: str,
    batch,
    steps: int,
    args,
    history_rows,
    eval_batch=None,
    eval_indices=None,
    selection_state=None,
):
    if steps <= 0:
        return {}

    signals = batch["signals"]
    coords = batch["coords"]
    mu_label = batch["mu_label"]
    mask_label = batch["mask_label"]
    log_interval = max(1, steps // 5)
    last_losses = {}

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_coords, step_mu_label, step_mask_label = sample_training_points(
            coords,
            mu_label,
            mask_label,
            args.train_point_subsample,
            args.point_sampling_mode,
            args.positive_fraction,
        )
        step_point_features = build_point_signal_features(signals, step_coords, args.point_signal_mode)
        out = model(signals, step_coords, point_features=step_point_features, return_phi=False)
        mu = out["mu"]
        soft_defect = soft_defect_from_output(out, args.mask_head_mode, args.mask_temperature)
        bce_loss = mask_bce_loss(
            soft_defect,
            step_mask_label,
            args.mask_bce_mode,
            args.pos_weight,
            args.focal_gamma,
            args.focal_alpha,
        )
        dice_loss = soft_dice_loss(soft_defect, step_mask_label)
        mu_mse_loss = F.mse_loss(mu, step_mu_label)
        area_loss, pred_area_soft_mean, true_area_mean = area_calibration_loss(
            soft_defect,
            step_mask_label,
            args.area_loss_mode,
            args.foreground_floor_ratio,
        )
        margin_stats = threshold_margin_loss(
            mu,
            step_mask_label,
            args.threshold_margin_mode,
            args.positive_mu_margin,
            args.negative_mu_margin,
        )
        loss = (
            args.lambda_mask_bce * bce_loss
            + args.lambda_mask_dice * dice_loss
            + args.lambda_mu_mse * mu_mse_loss
            + args.lambda_area_loss * area_loss
            + args.lambda_threshold_margin * margin_stats["threshold_margin_loss"]
        )
        loss.backward()
        optimizer.step()

        eval_iou_at_step = None
        eval_loss_at_step = None
        is_best_step = False
        if (
            phase == "finetune"
            and eval_batch is not None
            and selection_state is not None
            and should_run_validation_selection(step, steps, args)
        ):
            eval_iou_at_step, eval_loss_at_step, is_best_step = maybe_update_best_validation_state(
                model,
                step,
                eval_batch,
                eval_indices,
                args,
                selection_state,
            )

        last_losses = {
            "total_loss": loss.item(),
            "bce_loss": bce_loss.item(),
            "dice_loss": dice_loss.item(),
            "mu_mse_loss": mu_mse_loss.item(),
            "area_loss": area_loss.item(),
            "threshold_margin_loss": margin_stats["threshold_margin_loss"].item(),
            "positive_margin_loss": margin_stats["positive_margin_loss"].item(),
            "negative_margin_loss": margin_stats["negative_margin_loss"].item(),
            "pred_area_soft_mean": pred_area_soft_mean.item(),
            "true_area_mean": true_area_mean.item(),
            "batch_mean_iou": batch_mean_iou_from_mask(
                prediction_mask_from_output({key: value.detach() for key, value in out.items()}, args.mask_head_mode),
                step_mask_label,
            ),
        }
        if should_log_history(step, steps, args.history_interval):
            history_rows.append(
                history_row(
                    phase,
                    step,
                    loss.detach(),
                    bce_loss.detach(),
                    dice_loss.detach(),
                    mu_mse_loss.detach(),
                    area_loss.detach(),
                    {key: value.detach() if torch.is_tensor(value) else value for key, value in margin_stats.items()},
                    pred_area_soft_mean.detach(),
                    true_area_mean.detach(),
                    {key: value.detach() for key, value in out.items()},
                    soft_defect.detach(),
                    step_mask_label.detach(),
                    args,
                    eval_iou_at_step=eval_iou_at_step,
                    eval_loss_at_step=eval_loss_at_step,
                    is_best_step=is_best_step,
                )
            )
        if step == 1 or step == steps or step % log_interval == 0:
            print(
                f"phase={phase} "
                f"step={step} "
                f"loss={last_losses['total_loss']:.6e} "
                f"bce_loss={last_losses['bce_loss']:.6e} "
                f"dice_loss={last_losses['dice_loss']:.6e} "
                f"mu_mse_loss={last_losses['mu_mse_loss']:.6e} "
                f"area_loss={last_losses['area_loss']:.6e} "
                f"threshold_margin_loss={last_losses['threshold_margin_loss']:.6e} "
                f"batch_mean_iou={last_losses['batch_mean_iou']:.6e}"
            )

    return last_losses


def slice_conditional_batch(batch, start: int, stop: int):
    sliced = dict(batch)
    batch_size = int(batch["signals"].shape[0])
    for key in ("signals", "mu_label", "mask_label"):
        sliced[key] = batch[key][start:stop]
    coords = batch["coords"]
    if coords.ndim == 3 and int(coords.shape[0]) == batch_size:
        sliced["coords"] = coords[start:stop]
    return sliced


def evaluate_model_once(model, batch, sample_indices, point_signal_mode: str, mask_head_mode: str):
    model.eval()
    with torch.no_grad():
        point_features = build_point_signal_features(batch["signals"], batch["coords"], point_signal_mode)
        output = model(batch["signals"], batch["coords"], point_features=point_features, return_phi=False)
        pred_mask = prediction_mask_from_output(output, mask_head_mode)
    return per_sample_metrics(sample_indices, output["mu"], batch["mu_label"], batch["mask_label"], pred_mask)


def evaluate_model(model, batch, sample_indices, point_signal_mode: str, mask_head_mode: str, eval_batch_size: int):
    if eval_batch_size <= 0 or eval_batch_size >= len(sample_indices):
        return evaluate_model_once(model, batch, sample_indices, point_signal_mode, mask_head_mode)

    rows = []
    for start in range(0, len(sample_indices), eval_batch_size):
        stop = min(start + eval_batch_size, len(sample_indices))
        rows.extend(
            evaluate_model_once(
                model,
                slice_conditional_batch(batch, start, stop),
                sample_indices[start:stop],
                point_signal_mode,
                mask_head_mode,
            )
        )
    return rows


def evaluate_model_with_signals(
    model,
    batch,
    sample_indices,
    signals,
    point_signal_mode: str,
    mask_head_mode: str,
    eval_batch_size: int,
):
    ablated_batch = dict(batch)
    ablated_batch["signals"] = signals
    return evaluate_model(model, ablated_batch, sample_indices, point_signal_mode, mask_head_mode, eval_batch_size)


def run_signal_ablation_metrics(
    model,
    output_dir: Path,
    split_name: str,
    batch,
    sample_indices,
    signal_normalization: str,
    signal_feature_mode: str,
    conditioning_mode: str,
    encoder_type: str,
    point_signal_mode: str,
    mask_head_mode: str,
    mask_source: str,
    mask_bce_mode: str,
    area_loss_mode: str,
    lambda_area_loss: float,
    threshold_margin_mode: str,
    lambda_threshold_margin: float,
    eval_batch_size: int,
):
    zero_rows = add_metric_metadata(
        evaluate_model_with_signals(
            model,
            batch,
            sample_indices,
            torch.zeros_like(batch["signals"]),
            point_signal_mode,
            mask_head_mode,
            eval_batch_size,
        ),
        signal_normalization,
        signal_feature_mode,
        conditioning_mode,
        encoder_type,
        point_signal_mode,
        mask_head_mode,
        mask_source,
        mask_bce_mode,
        area_loss_mode,
        lambda_area_loss,
        threshold_margin_mode,
        lambda_threshold_margin,
    )
    shuffled_rows = add_metric_metadata(
        evaluate_model_with_signals(
            model,
            batch,
            sample_indices,
            torch.flip(batch["signals"], dims=[0]),
            point_signal_mode,
            mask_head_mode,
            eval_batch_size,
        ),
        signal_normalization,
        signal_feature_mode,
        conditioning_mode,
        encoder_type,
        point_signal_mode,
        mask_head_mode,
        mask_source,
        mask_bce_mode,
        area_loss_mode,
        lambda_area_loss,
        threshold_margin_mode,
        lambda_threshold_margin,
    )
    write_metrics(output_dir / f"{split_name}_metrics_zero_signal.csv", zero_rows)
    write_metrics(output_dir / f"{split_name}_metrics_shuffled_signal.csv", shuffled_rows)
    return {
        "zero_signal": zero_rows,
        "shuffled_signal": shuffled_rows,
    }


def average_metrics(rows):
    if not rows:
        raise ValueError("cannot average empty metric rows")
    keys = ["defect_iou", "defect_area_pred", "mu_mse", "mu_mae"]
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in keys
    }


def append_average_summary(lines, label, rows):
    avgs = average_metrics(rows)
    lines.append(f"{label} average metrics:")
    for key, value in avgs.items():
        lines.append(f"- {key}: `{value:.6e}`")
    lines.append("")


def append_signal_ablation_summary(lines, label, correct_rows, ablation_rows):
    correct_iou = average_metrics(correct_rows)["defect_iou"]
    zero_iou = average_metrics(ablation_rows["zero_signal"])["defect_iou"]
    shuffled_iou = average_metrics(ablation_rows["shuffled_signal"])["defect_iou"]
    lines.append(f"{label} signal ablation avg defect_iou:")
    lines.append(f"- correct_signal: `{correct_iou:.6e}`")
    lines.append(f"- zero_signal: `{zero_iou:.6e}`")
    lines.append(f"- shuffled_signal: `{shuffled_iou:.6e}`")
    lines.append(f"- correct_minus_zero: `{correct_iou - zero_iou:.6e}`")
    lines.append(f"- correct_minus_shuffled: `{correct_iou - shuffled_iou:.6e}`")
    if correct_iou > zero_iou and correct_iou > shuffled_iou:
        lines.append("- interpretation: correct signals outperform ablated signals in this split.")
    else:
        lines.append("- interpretation: ablated signals are close to or better than correct signals in this split.")
    lines.append("")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--eval-npz-path", default=None)
    parser.add_argument("--test-npz-path", default=None)
    parser.add_argument("--pretrain-npz-path", default=None)
    parser.add_argument("--sample-indices", default="0,1,2")
    parser.add_argument("--eval-sample-indices", default="")
    parser.add_argument("--test-sample-indices", default="")
    parser.add_argument("--pretrain-sample-indices", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lambda-mask-bce", type=float, default=1.0)
    parser.add_argument("--lambda-mask-dice", type=float, default=1.0)
    parser.add_argument("--lambda-mu-mse", type=float, default=0.0)
    parser.add_argument("--lambda-area-loss", type=float, default=0.0)
    parser.add_argument("--lambda-threshold-margin", type=float, default=0.0)
    parser.add_argument("--mask-temperature", type=float, default=50.0)
    parser.add_argument(
        "--mask-bce-mode",
        choices=["bce", "pos_weighted_bce", "focal_bce"],
        default="bce",
    )
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument(
        "--area-loss-mode",
        choices=["none", "batch_ratio_mse", "foreground_floor"],
        default="none",
    )
    parser.add_argument("--foreground-floor-ratio", type=float, default=0.5)
    parser.add_argument(
        "--threshold-margin-mode",
        choices=["none", "positive_hinge", "bidirectional_hinge"],
        default="none",
    )
    parser.add_argument("--positive-mu-margin", type=float, default=50.0)
    parser.add_argument("--negative-mu-margin", type=float, default=50.0)
    parser.add_argument("--signal-ablation", action="store_true")
    parser.add_argument(
        "--signal-normalization",
        choices=["none", "train_zscore", "per_sample_zscore"],
        default="none",
    )
    parser.add_argument(
        "--signal-feature-mode",
        choices=["raw", "raw_abs_grad"],
        default="raw",
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=["concat", "film"],
        default="concat",
    )
    parser.add_argument(
        "--encoder-type",
        choices=["mlp", "cnn"],
        default="mlp",
    )
    parser.add_argument(
        "--point-signal-mode",
        choices=["none", "local_value", "local_value_abs"],
        default="none",
    )
    parser.add_argument(
        "--mask-head-mode",
        choices=["mu_threshold", "direct"],
        default="mu_threshold",
    )
    parser.add_argument(
        "--mask-source",
        choices=["mu_threshold", "masks"],
        default="mu_threshold",
    )
    parser.add_argument("--train-point-subsample", type=int, default=0)
    parser.add_argument(
        "--point-sampling-mode",
        choices=["random", "positive_balanced"],
        default="random",
    )
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--history-interval", type=int, default=0)
    parser.add_argument(
        "--val-selection-metric",
        choices=["none", "eval_iou", "eval_loss"],
        default="none",
    )
    parser.add_argument("--val-selection-interval", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.npz_path or not args.output_dir:
        print("train_conditional_dual.py is a minimal conditional supervised runner.")
        print("It requires both --npz-path and --output-dir to run.")
        print(
            "Example: python train_conditional_dual.py --npz-path data/train.npz "
            "--output-dir experiments/dual_network/conditional_smoke --sample-indices 0,1,2"
        )
        return 0

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.pretrain_steps < 0:
        raise ValueError("--pretrain-steps must be non-negative")
    if args.lambda_area_loss < 0:
        raise ValueError("--lambda-area-loss must be non-negative")
    if args.lambda_threshold_margin < 0:
        raise ValueError("--lambda-threshold-margin must be non-negative")
    if args.mask_temperature <= 0:
        raise ValueError("--mask-temperature must be positive")
    if args.pos_weight <= 0:
        raise ValueError("--pos-weight must be positive")
    if args.focal_gamma < 0:
        raise ValueError("--focal-gamma must be non-negative")
    if not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be between 0 and 1")
    if args.foreground_floor_ratio < 0:
        raise ValueError("--foreground-floor-ratio must be non-negative")
    if args.positive_mu_margin < 0:
        raise ValueError("--positive-mu-margin must be non-negative")
    if args.negative_mu_margin < 0:
        raise ValueError("--negative-mu-margin must be non-negative")
    if args.train_point_subsample < 0:
        raise ValueError("--train-point-subsample must be non-negative")
    if not 0.0 <= args.positive_fraction <= 1.0:
        raise ValueError("--positive-fraction must be between 0 and 1")
    if args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size must be non-negative")
    if args.history_interval < 0:
        raise ValueError("--history-interval must be non-negative")
    if args.val_selection_metric != "none" and not args.eval_npz_path:
        raise ValueError("--eval-npz-path is required when --val-selection-metric is not none")
    if args.val_selection_metric != "none" and args.val_selection_interval <= 0:
        raise ValueError("--val-selection-interval must be positive when validation selection is enabled")
    if args.val_selection_metric == "none" and args.val_selection_interval < 0:
        raise ValueError("--val-selection-interval must be non-negative")

    sample_indices = parse_sample_indices(args.sample_indices)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_conditional_npz(args.npz_path)
    batch = get_conditional_batch(dataset, sample_indices, device=args.device, mask_source=args.mask_source)
    signal_len = infer_signal_len(dataset)
    eval_dataset = None
    eval_batch = None
    eval_indices = None
    test_dataset = None
    test_batch = None
    test_indices = None
    pretrain_dataset = None
    pretrain_batch = None
    pretrain_indices = None
    if args.eval_npz_path:
        eval_dataset = load_conditional_npz(args.eval_npz_path)
        eval_signal_len = infer_signal_len(eval_dataset)
        if eval_signal_len != signal_len:
            raise ValueError(
                "eval signal length must match train signal length: "
                f"got {eval_signal_len} and {signal_len}"
            )
        eval_indices = parse_eval_sample_indices(args.eval_sample_indices, eval_dataset)
        eval_batch = get_conditional_batch(eval_dataset, eval_indices, device=args.device, mask_source=args.mask_source)
    if args.test_npz_path:
        test_dataset = load_conditional_npz(args.test_npz_path)
        test_signal_len = infer_signal_len(test_dataset)
        if test_signal_len != signal_len:
            raise ValueError(
                "test signal length must match train signal length: "
                f"got {test_signal_len} and {signal_len}"
            )
        test_indices = parse_eval_sample_indices(args.test_sample_indices, test_dataset)
        test_batch = get_conditional_batch(test_dataset, test_indices, device=args.device, mask_source=args.mask_source)
    if args.pretrain_npz_path and args.pretrain_steps > 0:
        pretrain_dataset = load_conditional_npz(args.pretrain_npz_path)
        pretrain_signal_len = infer_signal_len(pretrain_dataset)
        if pretrain_signal_len != signal_len:
            raise ValueError(
                "pretrain signal length must match train signal length: "
                f"got {pretrain_signal_len} and {signal_len}"
            )
        pretrain_indices = parse_eval_sample_indices(args.pretrain_sample_indices, pretrain_dataset)
        pretrain_batch = get_conditional_batch(
            pretrain_dataset,
            pretrain_indices,
            device=args.device,
            mask_source=args.mask_source,
        )

    batch, normalization_stats = normalize_batch_signals(batch, args.signal_normalization)
    if eval_batch is not None:
        eval_batch, _ = normalize_batch_signals(eval_batch, args.signal_normalization, stats=normalization_stats)
    if test_batch is not None:
        test_batch, _ = normalize_batch_signals(test_batch, args.signal_normalization, stats=normalization_stats)
    if pretrain_batch is not None:
        pretrain_batch, _ = normalize_batch_signals(pretrain_batch, args.signal_normalization, stats=normalization_stats)
    batch = transform_batch_signals(batch, args.signal_feature_mode)
    if eval_batch is not None:
        eval_batch = transform_batch_signals(eval_batch, args.signal_feature_mode)
    if test_batch is not None:
        test_batch = transform_batch_signals(test_batch, args.signal_feature_mode)
    if pretrain_batch is not None:
        pretrain_batch = transform_batch_signals(pretrain_batch, args.signal_feature_mode)
    encoder_input_len = int(batch["signals"].shape[1])
    if pretrain_batch is not None:
        pretrain_encoder_input_len = int(pretrain_batch["signals"].shape[1])
        if pretrain_encoder_input_len != encoder_input_len:
            raise ValueError(
                "pretrain encoder input length must match train encoder input length: "
                f"got {pretrain_encoder_input_len} and {encoder_input_len}"
            )
        pretrain_point_count = batch_point_count(pretrain_batch)
        train_point_count = batch_point_count(batch)
        if pretrain_point_count != train_point_count:
            raise ValueError(
                "pretrain coords point count must match train coords point count: "
                f"got {pretrain_point_count} and {train_point_count}"
            )
    signal_dataset_shape = tuple(int(dim) for dim in dataset["signals"].shape)
    signal_original_shape = tuple(int(dim) for dim in batch.get("signal_original_shape", (signal_len,)))
    signal_channels = int(batch.get("signal_channels", 1))
    signal_length_per_channel = int(batch.get("signal_length_per_channel", signal_len))
    flattened_signal_length = int(batch.get("flattened_signal_length", signal_len))
    signal_flatten_order = str(batch.get("signal_flatten_order", "single_channel"))

    model = ConditionalDualNet(
        signal_len=encoder_input_len,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        conditioning_mode=args.conditioning_mode,
        encoder_type=args.encoder_type,
        extra_point_dim=point_signal_feature_dim(args.point_signal_mode),
        predict_mask=args.mask_head_mode == "direct",
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # S50 is a conditional supervised runner skeleton. Weak-form / physics
    # losses are intentionally left for later stages; this only validates the
    # signal-conditioned training closure.
    history_rows = []
    selection_state = {
        "best_score": None,
        "best_state": None,
        "best_step": None,
        "best_eval_iou": None,
        "best_eval_loss": None,
    }
    if pretrain_batch is not None:
        train_phase(model, optimizer, "pretrain", pretrain_batch, args.pretrain_steps, args, history_rows)
    last_losses = train_phase(
        model,
        optimizer,
        "finetune",
        batch,
        args.steps,
        args,
        history_rows,
        eval_batch=eval_batch,
        eval_indices=eval_indices,
        selection_state=selection_state if args.val_selection_metric != "none" else None,
    )
    if selection_state["best_state"] is not None:
        model.load_state_dict(selection_state["best_state"])
    if args.history_interval > 0:
        write_training_history(output_dir / "training_history.csv", history_rows)

    rows = add_metric_metadata(
        evaluate_model(model, batch, sample_indices, args.point_signal_mode, args.mask_head_mode, args.eval_batch_size),
        args.signal_normalization,
        args.signal_feature_mode,
        args.conditioning_mode,
        args.encoder_type,
        args.point_signal_mode,
        args.mask_head_mode,
        args.mask_source,
        args.mask_bce_mode,
        args.area_loss_mode,
        args.lambda_area_loss,
        args.threshold_margin_mode,
        args.lambda_threshold_margin,
    )
    write_metrics(output_dir / "metrics.csv", rows)
    eval_rows = None
    if eval_batch is not None:
        eval_rows = add_metric_metadata(
            evaluate_model(
                model,
                eval_batch,
                eval_indices,
                args.point_signal_mode,
                args.mask_head_mode,
                args.eval_batch_size,
            ),
            args.signal_normalization,
            args.signal_feature_mode,
            args.conditioning_mode,
            args.encoder_type,
            args.point_signal_mode,
            args.mask_head_mode,
            args.mask_source,
            args.mask_bce_mode,
            args.area_loss_mode,
            args.lambda_area_loss,
            args.threshold_margin_mode,
            args.lambda_threshold_margin,
        )
        write_metrics(output_dir / "eval_metrics.csv", eval_rows)
    test_rows = None
    if test_batch is not None:
        test_rows = add_metric_metadata(
            evaluate_model(
                model,
                test_batch,
                test_indices,
                args.point_signal_mode,
                args.mask_head_mode,
                args.eval_batch_size,
            ),
            args.signal_normalization,
            args.signal_feature_mode,
            args.conditioning_mode,
            args.encoder_type,
            args.point_signal_mode,
            args.mask_head_mode,
            args.mask_source,
            args.mask_bce_mode,
            args.area_loss_mode,
            args.lambda_area_loss,
            args.threshold_margin_mode,
            args.lambda_threshold_margin,
        )
        write_metrics(output_dir / "test_metrics.csv", test_rows)
    signal_ablation_rows = {}
    if args.signal_ablation:
        signal_ablation_rows["Train"] = run_signal_ablation_metrics(
            model,
            output_dir,
            "train",
            batch,
            sample_indices,
            args.signal_normalization,
            args.signal_feature_mode,
            args.conditioning_mode,
            args.encoder_type,
            args.point_signal_mode,
            args.mask_head_mode,
            args.mask_source,
            args.mask_bce_mode,
            args.area_loss_mode,
            args.lambda_area_loss,
            args.threshold_margin_mode,
            args.lambda_threshold_margin,
            args.eval_batch_size,
        )
        if eval_batch is not None:
            signal_ablation_rows["Eval"] = run_signal_ablation_metrics(
                model,
                output_dir,
                "eval",
                eval_batch,
                eval_indices,
                args.signal_normalization,
                args.signal_feature_mode,
                args.conditioning_mode,
                args.encoder_type,
                args.point_signal_mode,
                args.mask_head_mode,
                args.mask_source,
                args.mask_bce_mode,
                args.area_loss_mode,
                args.lambda_area_loss,
                args.threshold_margin_mode,
                args.lambda_threshold_margin,
                args.eval_batch_size,
            )
        if test_batch is not None:
            signal_ablation_rows["Test"] = run_signal_ablation_metrics(
                model,
                output_dir,
                "test",
                test_batch,
                test_indices,
                args.signal_normalization,
                args.signal_feature_mode,
                args.conditioning_mode,
                args.encoder_type,
                args.point_signal_mode,
                args.mask_head_mode,
                args.mask_source,
                args.mask_bce_mode,
                args.area_loss_mode,
                args.lambda_area_loss,
                args.threshold_margin_mode,
                args.lambda_threshold_margin,
                args.eval_batch_size,
            )

    summary_lines = [
        "# Conditional supervised runner summary",
        "",
        f"- npz_path: `{args.npz_path}`",
        f"- sample_indices: `{args.sample_indices}`",
        f"- eval_npz_path: `{args.eval_npz_path}`",
        f"- eval_sample_indices: `{args.eval_sample_indices}`",
        f"- test_npz_path: `{args.test_npz_path}`",
        f"- test_sample_indices: `{args.test_sample_indices}`",
        f"- pretrain_npz_path: `{args.pretrain_npz_path}`",
        f"- pretrain_steps: `{args.pretrain_steps}`",
        f"- pretrain_sample_indices: `{args.pretrain_sample_indices}`",
        f"- pretrain_sample_indices_count: `{0 if pretrain_indices is None else len(pretrain_indices)}`",
        f"- steps: `{args.steps}`",
        f"- lr: `{args.lr}`",
        f"- hidden_dim: `{args.hidden_dim}`",
        f"- num_layers: `{args.num_layers}`",
        f"- latent_dim: `{args.latent_dim}`",
        f"- original_signals_shape: `{signal_dataset_shape}`",
        f"- per_sample_signal_original_shape: `{signal_original_shape}`",
        f"- flattened_signal_length: `{flattened_signal_length}`",
        f"- signal_channels: `{signal_channels}`",
        f"- signal_length_per_channel: `{signal_length_per_channel}`",
        f"- signal_flatten_order: `{signal_flatten_order}`",
        f"- encoder_input_length: `{encoder_input_len}`",
        f"- conditioning_mode: `{args.conditioning_mode}`",
        f"- encoder_type: `{args.encoder_type}`",
        f"- point_signal_mode: `{args.point_signal_mode}`",
        f"- mask_head_mode: `{args.mask_head_mode}`",
        f"- mask_source: `{args.mask_source}`",
        f"- mask_bce_mode: `{args.mask_bce_mode}`",
        f"- pos_weight: `{args.pos_weight}`",
        f"- focal_gamma: `{args.focal_gamma}`",
        f"- focal_alpha: `{args.focal_alpha}`",
        f"- area_loss_mode: `{args.area_loss_mode}`",
        f"- lambda_area_loss: `{args.lambda_area_loss}`",
        f"- foreground_floor_ratio: `{args.foreground_floor_ratio}`",
        f"- threshold_margin_mode: `{args.threshold_margin_mode}`",
        f"- lambda_threshold_margin: `{args.lambda_threshold_margin}`",
        f"- positive_mu_margin: `{args.positive_mu_margin}`",
        f"- negative_mu_margin: `{args.negative_mu_margin}`",
        f"- val_selection_metric: `{args.val_selection_metric}`",
        f"- val_selection_interval: `{args.val_selection_interval}`",
        f"- best_step: `{selection_state['best_step']}`",
        f"- best_eval_iou: `{selection_state['best_eval_iou']}`",
        f"- best_eval_loss: `{selection_state['best_eval_loss']}`",
        f"- lambda_mask_bce: `{args.lambda_mask_bce}`",
        f"- lambda_mask_dice: `{args.lambda_mask_dice}`",
        f"- lambda_mu_mse: `{args.lambda_mu_mse}`",
        f"- mask_temperature: `{args.mask_temperature}`",
        f"- signal_normalization: `{args.signal_normalization}`",
        f"- signal_feature_mode: `{args.signal_feature_mode}`",
        f"- signal_ablation: `{args.signal_ablation}`",
        f"- train_point_subsample: `{args.train_point_subsample}`",
        f"- point_sampling_mode: `{args.point_sampling_mode}`",
        f"- positive_fraction: `{args.positive_fraction}`",
        f"- history_interval: `{args.history_interval}`",
        f"- eval_batch_size: `{args.eval_batch_size}`",
        "",
        "S50 uses supervised mask losses only. Weak-form / physics losses are not connected in this skeleton.",
        "Signal normalization is applied before signal feature construction; optional signal ablation is applied to the constructed encoder input.",
        "Point signal features are generated from the signals actually passed to the model.",
        "Direct mask head mode trains BCE / Dice on mask probability instead of a mu-threshold-derived soft mask.",
        "",
        "Final losses:",
    ]
    for key, value in last_losses.items():
        summary_lines.append(f"- {key}: `{value:.6e}`")
    summary_lines.append("")
    append_average_summary(summary_lines, "Train", rows)
    if eval_rows is not None:
        append_average_summary(summary_lines, "Eval", eval_rows)
    if test_rows is not None:
        append_average_summary(summary_lines, "Test", test_rows)
    if args.signal_ablation:
        append_signal_ablation_summary(summary_lines, "Train", rows, signal_ablation_rows["Train"])
        if eval_rows is not None:
            append_signal_ablation_summary(summary_lines, "Eval", eval_rows, signal_ablation_rows["Eval"])
        if test_rows is not None:
            append_signal_ablation_summary(summary_lines, "Test", test_rows, signal_ablation_rows["Test"])
    summary_lines.append("No model weights, checkpoints, arrays, or images were saved.")
    (output_dir / "run_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved metrics to {output_dir / 'metrics.csv'}")
    if eval_rows is not None:
        print(f"Saved eval metrics to {output_dir / 'eval_metrics.csv'}")
    if test_rows is not None:
        print(f"Saved test metrics to {output_dir / 'test_metrics.csv'}")
    if args.signal_ablation:
        print(f"Saved signal ablation metrics to {output_dir}")
    if args.history_interval > 0:
        print(f"Saved training history to {output_dir / 'training_history.csv'}")
    print(f"Saved summary to {output_dir / 'run_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

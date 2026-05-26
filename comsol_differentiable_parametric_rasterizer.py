"""Differentiable soft rasterization for COMSOL parametric inverse components."""

from __future__ import annotations

import torch
import torch.nn.functional as F


RAW_FIELDS = [
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "depth_or_shape_param",
    "rotation_angle",
]


def _schema_list(values) -> list[str]:
    return [str(v) for v in values]


def _grid_spacing(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    spacings = []
    if x.numel() > 1:
        spacings.append(torch.mean(torch.abs(x[1:] - x[:-1])))
    if y.numel() > 1:
        spacings.append(torch.mean(torch.abs(y[1:] - y[:-1])))
    if not spacings:
        return torch.as_tensor(1.0, dtype=x.dtype, device=x.device)
    return torch.stack(spacings).mean().clamp_min(eps)


def _positive_axis(values: torch.Tensor, eps: float) -> torch.Tensor:
    # S117 hard rasterizer uses abs(axis) and treats axis_x/y as full width/height.
    return torch.abs(values).clamp_min(eps)


def _rotation_radians(continuous: torch.Tensor, schema: list[str]) -> torch.Tensor:
    if "rotation_angle_rad" in schema:
        return continuous[..., schema.index("rotation_angle_rad")]
    if "rotation_angle_deg" in schema:
        return torch.deg2rad(continuous[..., schema.index("rotation_angle_deg")])
    if "rotation_angle" in schema:
        # Project raw COMSOL targets use degrees; refined targets use sin/cos.
        return torch.deg2rad(continuous[..., schema.index("rotation_angle")])
    if "rotation_sin" in schema and "rotation_cos" in schema:
        sin_v = continuous[..., schema.index("rotation_sin")]
        cos_v = continuous[..., schema.index("rotation_cos")]
        return torch.atan2(sin_v, cos_v)
    return torch.zeros_like(continuous[..., 0])


def _field(continuous: torch.Tensor, schema: list[str], name: str) -> torch.Tensor:
    if name not in schema:
        raise ValueError(f"target_schema missing required field: {name}")
    return continuous[..., schema.index(name)]


def soft_rasterize_components(
    continuous: torch.Tensor,
    presence_prob: torch.Tensor,
    type_logits: torch.Tensor | None,
    x,
    y,
    target_schema,
    type_vocab,
    softness_cells: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rasterize component parameters into a differentiable soft union mask."""

    if continuous.ndim != 3:
        raise ValueError(f"continuous must have shape [B,K,P], got {tuple(continuous.shape)}")
    if presence_prob.shape != continuous.shape[:2]:
        raise ValueError("presence_prob must have shape [B,K] matching continuous.")
    if type_logits is not None and type_logits.shape[:2] != continuous.shape[:2]:
        raise ValueError("type_logits must have shape [B,K,T] matching continuous.")
    if softness_cells <= 0:
        raise ValueError("softness_cells must be positive.")

    schema = _schema_list(target_schema)
    _ = _schema_list(type_vocab)
    device = continuous.device
    dtype = continuous.dtype
    x_t = torch.as_tensor(x, dtype=dtype, device=device)
    y_t = torch.as_tensor(y, dtype=dtype, device=device)
    if x_t.ndim != 1 or y_t.ndim != 1:
        raise ValueError("x and y must be 1D coordinate arrays.")

    grid_y, grid_x = torch.meshgrid(y_t, x_t, indexing="ij")
    grid_x = grid_x.view(1, 1, y_t.numel(), x_t.numel())
    grid_y = grid_y.view(1, 1, y_t.numel(), x_t.numel())

    center_x = _field(continuous, schema, "center_x").view(*continuous.shape[:2], 1, 1)
    center_y = _field(continuous, schema, "center_y").view(*continuous.shape[:2], 1, 1)
    half_x = (_positive_axis(_field(continuous, schema, "axis_x"), eps) * 0.5).view(*continuous.shape[:2], 1, 1)
    half_y = (_positive_axis(_field(continuous, schema, "axis_y"), eps) * 0.5).view(*continuous.shape[:2], 1, 1)
    theta = _rotation_radians(continuous, schema).view(*continuous.shape[:2], 1, 1)
    presence = presence_prob.clamp(0.0, 1.0).view(*continuous.shape[:2], 1, 1)

    dx = grid_x - center_x
    dy = grid_y - center_y
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy

    softness = (float(softness_cells) * _grid_spacing(x_t, y_t, eps)).clamp_min(eps)
    prob_x = torch.sigmoid((half_x - torch.abs(local_x)) / softness)
    prob_y = torch.sigmoid((half_y - torch.abs(local_y)) / softness)
    component_prob = (presence * prob_x * prob_y).clamp(0.0, 1.0)
    # The product-form union is intended for small component counts such as V2 K=3.
    background_prob = torch.prod((1.0 - component_prob).clamp(eps, 1.0), dim=1)
    soft_mask = (1.0 - background_prob).clamp(0.0, 1.0)
    return soft_mask


def soft_bce_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target_mask.to(dtype=pred_mask.dtype, device=pred_mask.device).clamp(0.0, 1.0)
    pred = pred_mask.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(pred, target)


def soft_dice_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target_mask.to(dtype=pred_mask.dtype, device=pred_mask.device).clamp(0.0, 1.0)
    pred = pred_mask.clamp(0.0, 1.0)
    reduce_dims = tuple(range(1, pred.ndim))
    intersection = torch.sum(pred * target, dim=reduce_dims)
    denom = torch.sum(pred, dim=reduce_dims) + torch.sum(target, dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def soft_iou_score(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target_mask.to(dtype=pred_mask.dtype, device=pred_mask.device).clamp(0.0, 1.0)
    pred = pred_mask.clamp(0.0, 1.0)
    reduce_dims = tuple(range(1, pred.ndim))
    intersection = torch.sum(pred * target, dim=reduce_dims)
    union = torch.sum(pred + target - pred * target, dim=reduce_dims)
    return ((intersection + eps) / (union + eps)).mean()


def soft_dice_score(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 1.0 - soft_dice_loss(pred_mask, target_mask, eps=eps)

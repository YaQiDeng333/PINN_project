"""Smoke test for differentiable COMSOL parametric rasterization."""

from __future__ import annotations

import torch

from comsol_differentiable_parametric_rasterizer import (
    soft_bce_loss,
    soft_dice_loss,
    soft_iou_score,
    soft_rasterize_components,
)


def main() -> None:
    batch = 2
    components = 3
    x = torch.linspace(-1.0, 1.0, 30)
    y = torch.linspace(-0.7, 0.7, 20)
    continuous = torch.zeros(batch, components, 6, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        continuous[0, 0] = torch.tensor([-0.25, 0.0, 0.55, 0.25, 0.1, 0.0])
        continuous[0, 1] = torch.tensor([0.35, 0.1, 0.35, 0.2, 0.1, 30.0])
        continuous[1, 0] = torch.tensor([0.0, -0.1, 0.65, 0.3, 0.1, -20.0])
        continuous[1, 1] = torch.tensor([0.5, 0.2, 0.25, 0.25, 0.1, 45.0])
        continuous[1, 2] = torch.tensor([-0.5, 0.2, 0.25, 0.25, 0.1, 0.0])
    presence = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32, requires_grad=True)
    type_logits = torch.zeros(batch, components, 2, dtype=torch.float32, requires_grad=True)
    schema = ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"]
    type_vocab = ["rectangular_notch", "rotated_rect"]
    soft_mask = soft_rasterize_components(
        continuous,
        presence,
        type_logits,
        x,
        y,
        schema,
        type_vocab,
        softness_cells=1.0,
    )
    assert soft_mask.shape == (batch, y.numel(), x.numel())
    assert torch.isfinite(soft_mask).all()
    assert float(soft_mask.min()) >= 0.0
    assert float(soft_mask.max()) <= 1.0

    no_absent = soft_rasterize_components(
        continuous[:, :2],
        presence[:, :2],
        type_logits[:, :2],
        x,
        y,
        schema,
        type_vocab,
        softness_cells=1.0,
    )
    assert torch.allclose(soft_mask, no_absent, atol=1e-6)

    rotated = continuous.detach().clone()
    rotated[:, 0, 5] += 45.0
    rotated_mask = soft_rasterize_components(rotated, presence.detach(), type_logits.detach(), x, y, schema, type_vocab)
    assert not torch.allclose(soft_mask.detach(), rotated_mask, atol=1e-4)

    small_degree = continuous.detach().clone()
    small_degree[:, :, 5] = 3.0
    small_degree_mask = soft_rasterize_components(small_degree, presence.detach(), None, x, y, schema, type_vocab)
    rad_schema = ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle_rad"]
    small_radian = small_degree.detach().clone()
    small_radian[:, :, 5] = torch.deg2rad(torch.full_like(small_radian[:, :, 5], 3.0))
    small_radian_mask = soft_rasterize_components(small_radian, presence.detach(), None, x, y, rad_schema, type_vocab)
    assert torch.allclose(small_degree_mask, small_radian_mask, atol=1e-6)

    sincos_schema = ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_sin", "rotation_cos"]
    sincos = torch.cat(
        [
            continuous.detach()[:, :, :5],
            torch.sin(torch.deg2rad(continuous.detach()[:, :, 5:6])),
            torch.cos(torch.deg2rad(continuous.detach()[:, :, 5:6])),
        ],
        dim=2,
    )
    sincos_mask = soft_rasterize_components(sincos, presence.detach(), None, x, y, sincos_schema, type_vocab)
    assert torch.allclose(soft_mask.detach(), sincos_mask, atol=1e-6)

    zero_presence = torch.zeros_like(presence)
    zero_mask = soft_rasterize_components(continuous.detach(), zero_presence, None, x, y, schema, type_vocab)
    assert torch.allclose(zero_mask, torch.zeros_like(zero_mask), atol=1e-6)

    target = (soft_mask.detach() > 0.5).float()
    loss = soft_dice_loss(soft_mask, target) + 0.25 * soft_bce_loss(soft_mask, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert continuous.grad is not None
    assert torch.isfinite(continuous.grad).all()
    assert presence.grad is not None
    assert soft_iou_score(soft_mask.detach(), target) > 0.2
    print("COMSOL differentiable parametric rasterizer smoke test passed.")


if __name__ == "__main__":
    main()

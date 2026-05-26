"""Smoke tests for center-anchored polygon inverse models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from comsol_center_anchored_polygon_inverse_models import CenterAnchoredPolygonInverseNet


def _check_model(local_shape_conditioning_mode: str) -> None:
    model = CenterAnchoredPolygonInverseNet(
        signal_len=600,
        center_x_bins=25,
        center_y_bins=13,
        hidden_dim=32,
        latent_dim=16,
        max_components=3,
        max_vertices=4,
        num_types=2,
        local_shape_conditioning_mode=local_shape_conditioning_mode,
        local_shape_conditioning_dim=8,
    )
    signals = torch.randn(5, 600)
    out = model(signals)
    assert out["presence_logits"].shape == (5, 3)
    assert out["type_logits"].shape == (5, 3, 2)
    assert out["center_x_bin_logits"].shape == (5, 3, 25)
    assert out["center_y_bin_logits"].shape == (5, 3, 13)
    assert out["center_offset"].shape == (5, 3, 2)
    assert out["local_vertices_grid"].shape == (5, 3, 4, 2)
    presence = torch.ones(5, 3)
    type_targets = torch.zeros(5, 3, dtype=torch.long)
    x_bin = torch.zeros(5, 3, dtype=torch.long)
    y_bin = torch.zeros(5, 3, dtype=torch.long)
    local_vertices = torch.zeros(5, 3, 4, 2)
    loss = F.binary_cross_entropy_with_logits(out["presence_logits"], presence)
    loss = loss + F.cross_entropy(out["type_logits"].reshape(-1, 2), type_targets.reshape(-1))
    loss = loss + F.cross_entropy(out["center_x_bin_logits"].reshape(-1, 25), x_bin.reshape(-1))
    loss = loss + F.cross_entropy(out["center_y_bin_logits"].reshape(-1, 13), y_bin.reshape(-1))
    loss = loss + F.smooth_l1_loss(out["center_offset"], torch.zeros_like(out["center_offset"]))
    loss = loss + F.smooth_l1_loss(out["local_vertices_grid"], local_vertices)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())
    if local_shape_conditioning_mode != "none":
        conditioned_params = [param for name, param in model.named_parameters() if "conditioned_local_vertex_head" in name]
        assert conditioned_params
        assert all(param.grad is not None for param in conditioned_params)
        model.zero_grad(set_to_none=True)
        local_only = F.smooth_l1_loss(model(signals)["local_vertices_grid"], local_vertices)
        local_only.backward()
        detached_head_prefixes = (
            "center_x_bin_head",
            "center_y_bin_head",
            "center_offset_head",
            "type_head",
        )
        for name, param in model.named_parameters():
            if name.startswith(detached_head_prefixes):
                assert param.grad is None, f"{name} received gradient through detached local-shape conditioning"


def test_forward_shapes_and_backward() -> None:
    _check_model("none")
    _check_model("center_bin_slot_type")
    model = CenterAnchoredPolygonInverseNet(
        signal_len=600,
        center_x_bins=25,
        center_y_bins=13,
        hidden_dim=32,
        latent_dim=16,
        max_components=3,
        max_vertices=4,
        num_types=2,
        joint_center_shape_mode="soft_center_scheduled",
    )
    signals = torch.randn(5, 600)
    centers = torch.linspace(-1.0, 1.0, 25)
    y_centers = torch.linspace(-1.0, 1.0, 13)
    out = model(
        signals,
        x_centers=centers,
        y_centers=y_centers,
        bin_width_x=torch.tensor(0.1),
        bin_width_y=torch.tensor(0.1),
        grid_dx=torch.tensor(0.01),
        grid_dy=torch.tensor(0.01),
    )
    assert out["local_vertices_grid"].shape == (5, 3, 4, 2)
    model.zero_grad(set_to_none=True)
    F.smooth_l1_loss(out["local_vertices_grid"], torch.zeros_like(out["local_vertices_grid"])).backward()
    assert any(param.grad is not None for name, param in model.named_parameters() if name.startswith("joint_local_vertex_head"))
    assert any(param.grad is not None for name, param in model.named_parameters() if name.startswith("center_x_bin_head"))
    assert any(param.grad is not None for name, param in model.named_parameters() if name.startswith("center_y_bin_head"))
    assert any(param.grad is not None for name, param in model.named_parameters() if name.startswith("center_offset_head"))
    assert all(param.grad is None for name, param in model.named_parameters() if name.startswith("type_head"))


if __name__ == "__main__":
    test_forward_shapes_and_backward()
    print("center-anchored polygon inverse model smoke passed")

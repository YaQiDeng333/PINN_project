"""Smoke tests for component-query COMSOL polygon inverse models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from comsol_component_query_polygon_inverse_models import ComponentQueryPolygonInverseNet


def test_forward_shapes_and_backward() -> None:
    model = ComponentQueryPolygonInverseNet(
        signal_len=600,
        center_x_bins=25,
        center_y_bins=13,
        hidden_dim=32,
        latent_dim=16,
        max_components=3,
        max_vertices=4,
        num_types=2,
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
    assert model.query_embedding.weight.grad is not None
    for prefix in [
        "presence_head",
        "type_head",
        "center_x_bin_head",
        "center_y_bin_head",
        "center_offset_head",
        "local_vertex_head",
    ]:
        assert any(param.grad is not None for name, param in model.named_parameters() if name.startswith(prefix)), prefix


def test_supported_shape_guards() -> None:
    try:
        ComponentQueryPolygonInverseNet(600, 25, 13, max_components=4)
    except ValueError as exc:
        assert "max_components=3" in str(exc)
    else:
        raise AssertionError("Expected max_components guard.")
    try:
        ComponentQueryPolygonInverseNet(600, 25, 13, max_vertices=5)
    except ValueError as exc:
        assert "max_vertices=4" in str(exc)
    else:
        raise AssertionError("Expected max_vertices guard.")


if __name__ == "__main__":
    test_forward_shapes_and_backward()
    test_supported_shape_guards()
    print("component-query polygon inverse model smoke passed")

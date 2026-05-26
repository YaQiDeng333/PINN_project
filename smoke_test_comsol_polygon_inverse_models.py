"""Smoke test for COMSOL polygon inverse model modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from comsol_polygon_inverse_models import PolygonInverseNet


def main() -> None:
    torch.manual_seed(0)
    model = PolygonInverseNet(
        signal_len=600,
        hidden_dim=32,
        latent_dim=16,
        max_components=3,
        max_vertices=4,
        num_types=2,
        num_layers=2,
    )
    signals = torch.randn(5, 600)
    out = model(signals)
    assert out["presence_logits"].shape == (5, 3)
    assert out["presence_prob"].shape == (5, 3)
    assert out["type_logits"].shape == (5, 3, 2)
    assert out["vertices_norm"].shape == (5, 3, 4, 2)
    presence = torch.zeros(5, 3)
    presence[:, 0] = 1.0
    type_targets = torch.zeros(5, 3, dtype=torch.long)
    vertices = torch.zeros(5, 3, 4, 2)
    vertex_mask = torch.zeros(5, 3, 4)
    vertex_mask[:, 0] = 1.0
    loss = F.binary_cross_entropy_with_logits(out["presence_logits"], presence)
    loss = loss + F.cross_entropy(out["type_logits"][:, 0], type_targets[:, 0])
    vertex_loss = F.smooth_l1_loss(out["vertices_norm"], vertices, reduction="none")
    valid = (presence.unsqueeze(-1) * vertex_mask).unsqueeze(-1)
    loss = loss + (vertex_loss * valid).sum() / valid.sum().clamp_min(1.0)
    assert torch.isfinite(loss)
    loss.backward()
    try:
        PolygonInverseNet(signal_len=600, max_components=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected max_components guard.")
    try:
        model(torch.randn(5, 3, 200))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected flattened signal shape guard.")
    print("COMSOL polygon inverse model smoke test passed.")


if __name__ == "__main__":
    main()

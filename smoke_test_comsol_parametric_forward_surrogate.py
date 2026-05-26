"""Smoke test for comsol_parametric_forward_surrogate.py."""

from __future__ import annotations

import torch

from comsol_parametric_forward_surrogate import (
    ParametricForwardSurrogate,
    build_forward_geometry_vector,
)


def main() -> None:
    batch = 4
    max_components = 3
    num_types = 2
    num_continuous = 6
    schema = ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"]

    presence = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    type_targets = torch.tensor([[0, 1, -1], [1, -1, -1], [0, 1, 1], [0, -1, -1]], dtype=torch.long)
    continuous = torch.randn(batch, max_components, num_continuous, requires_grad=True)

    geometry_vector = build_forward_geometry_vector(
        presence=presence,
        type_targets_or_probs=type_targets,
        continuous=continuous,
        num_types=num_types,
        target_schema=schema,
    )
    expected_dim = max_components * (1 + num_types + num_continuous)
    assert geometry_vector.shape == (batch, expected_dim)

    model = ParametricForwardSurrogate(input_dim=expected_dim, output_dim=600, hidden_dim=32, num_layers=3)
    output = model(geometry_vector)
    assert output.shape == (batch, 600)
    loss = output.square().mean()
    loss.backward()
    assert continuous.grad is not None

    type_probs = torch.softmax(torch.randn(batch, max_components, num_types), dim=-1)
    geometry_vector_probs = build_forward_geometry_vector(
        presence=presence,
        type_targets_or_probs=type_probs,
        continuous=continuous.detach(),
        num_types=num_types,
        target_schema=schema,
    )
    assert geometry_vector_probs.shape == (batch, expected_dim)

    try:
        build_forward_geometry_vector(presence[:, :2], type_targets, continuous, num_types, schema)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch should raise ValueError")

    print("COMSOL parametric forward surrogate smoke test passed.")


if __name__ == "__main__":
    main()

"""Smoke test for compact-support weak-form test-function gradients.

This script validates the first local bump test-gradient generator with
weak_form_loss. It does not read data, save models, save images, create
checkpoints, or call train_pinn.py.
"""

import torch

from dual_network_losses import generate_compact_support_test_grads, weak_form_loss
from dual_network_models import MuNet, PhiNet


def main():
    torch.manual_seed(0)

    x = torch.linspace(-15.0, 15.0, 20)
    y = torch.linspace(0.0, 10.0, 10)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    coords.requires_grad_(True)
    num_points = coords.shape[0]

    centers = torch.tensor(
        [
            [-5.0, 5.0],
            [0.0, 5.0],
            [5.0, 5.0],
        ],
        dtype=coords.dtype,
        device=coords.device,
    )
    test_grads = generate_compact_support_test_grads(coords, centers, radius=5.0)

    expected_shape = (3, num_points, 2)
    if tuple(test_grads.shape) != expected_shape:
        raise AssertionError(f"test_grads shape must be {expected_shape}, got {tuple(test_grads.shape)}")
    if not torch.isfinite(test_grads).all():
        raise AssertionError("test_grads contains NaN or inf")
    if torch.count_nonzero(test_grads).item() == 0:
        raise AssertionError("test_grads must contain nonzero entries")
    if test_grads.requires_grad:
        raise AssertionError("test_grads should not require gradients")

    phi_net = PhiNet(hidden_dim=32, num_layers=2)
    mu_net = MuNet(hidden_dim=32, num_layers=2)
    phi = phi_net(coords)
    mu = mu_net(coords)

    loss = weak_form_loss(mu, phi, coords, test_grads=test_grads)
    if loss.dim() != 0:
        raise AssertionError(f"weak_form_loss must return a scalar, got shape {tuple(loss.shape)}")
    if not torch.isfinite(loss):
        raise AssertionError("weak_form_loss returned a non-finite value")

    loss.backward()
    has_mu_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in mu_net.parameters()
    )
    if not has_mu_grad:
        raise AssertionError("MuNet parameters did not receive finite gradients")
    phi_has_no_grad = all(parameter.grad is None for parameter in phi_net.parameters())
    if not phi_has_no_grad:
        raise AssertionError("PhiNet parameters should not receive gradients from weak_form_loss")

    print("Weak-form test function smoke test passed.")


if __name__ == "__main__":
    main()

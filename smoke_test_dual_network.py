"""Smoke test for the dual-network variational skeleton.

This script validates imports, forward passes, first derivatives, and the
basic loss helpers. It does not read data files, start training, or call
train_pinn.py.
"""

import torch

from dual_network_losses import data_loss, energy_loss, tv_loss, weak_form_loss
from dual_network_models import MuNet, PhiNet


def _assert_scalar(name, value):
    if value.dim() != 0:
        raise AssertionError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}")


def main():
    torch.manual_seed(0)

    x = torch.linspace(-15.0, 15.0, 20)
    y = torch.linspace(0.0, 10.0, 10)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    coords.requires_grad_(True)
    num_points = coords.shape[0]

    phi_net = PhiNet(hidden_dim=32, num_layers=2)
    mu_net = MuNet(hidden_dim=32, num_layers=2, mu_min=1.0, mu_max=1000.0)

    phi = phi_net(coords)
    mu = mu_net(coords)

    expected_shape = (num_points, 1)
    if tuple(phi.shape) != expected_shape:
        raise AssertionError(f"phi shape must be {expected_shape}, got {tuple(phi.shape)}")
    if tuple(mu.shape) != expected_shape:
        raise AssertionError(f"mu shape must be {expected_shape}, got {tuple(mu.shape)}")

    e_loss = energy_loss(phi, mu, coords)
    tv = tv_loss(mu, coords)
    _assert_scalar("energy_loss", e_loss)
    _assert_scalar("tv_loss", tv)
    print(f"energy_loss: {e_loss.item():.6e}")
    print(f"tv_loss: {tv.item():.6e}")

    x_probe = torch.linspace(-15.0, 15.0, 20)
    y_probe = torch.full_like(x_probe, 10.0)
    coords_probe = torch.stack([x_probe, y_probe], dim=1)
    coords_probe.requires_grad_(True)
    phi_probe = phi_net(coords_probe)
    bz_meas = torch.zeros(20)
    d_loss = data_loss(phi_probe, coords_probe, bz_meas)
    _assert_scalar("data_loss", d_loss)
    print(f"data_loss: {d_loss.item():.6e}")

    try:
        weak_form_loss(mu, phi, coords, test_grads=None)
    except NotImplementedError:
        print("weak_form_loss without test_grads raised NotImplementedError as expected.")
    else:
        raise AssertionError("weak_form_loss must raise NotImplementedError when test_grads is None")

    dummy_test_grads = torch.zeros(num_points, 2)
    weak_single = weak_form_loss(mu, phi, coords, test_grads=dummy_test_grads)
    _assert_scalar("weak_form_loss [N, 2]", weak_single)
    print(f"weak_form_loss [N, 2]: {weak_single.item():.6e}")

    dummy_test_grads_q = torch.zeros(3, num_points, 2)
    dummy_test_grads_q[0, :, 0] = 1.0
    dummy_test_grads_q[1, :, 1] = 1.0
    dummy_test_grads_q[2, :, :] = 0.5
    weak_q = weak_form_loss(mu, phi, coords, test_grads=dummy_test_grads_q)
    _assert_scalar("weak_form_loss [Q, N, 2]", weak_q)
    print(f"weak_form_loss [Q, N, 2]: {weak_q.item():.6e}")

    print("Dual-network smoke test passed.")


if __name__ == "__main__":
    main()

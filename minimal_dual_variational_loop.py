"""Minimal alternating loop prototype for the dual-network branch.

This is a code-closure test, not formal training. The synthetic Bz signal is
not real MFL data, and the dummy test gradients are not real weak-form test
functions. Future work must replace both with dataset signals and
compact-support test functions.
"""

import torch

from dual_network_losses import data_loss, energy_loss, tv_loss, weak_form_loss
from dual_network_models import MuNet, PhiNet


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def clear_coord_grads(*tensors):
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def main():
    torch.manual_seed(0)

    x = torch.linspace(-15.0, 15.0, 20)
    y = torch.linspace(0.0, 10.0, 10)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    coords.requires_grad_(True)
    num_points = coords.shape[0]

    x_probe = torch.linspace(-15.0, 15.0, 20)
    y_probe = torch.full_like(x_probe, 10.0)
    coords_probe = torch.stack([x_probe, y_probe], dim=1)
    coords_probe.requires_grad_(True)

    # Synthetic signal for code-loop testing only; this is not real MFL data.
    bz_meas = -0.1 * torch.exp(-torch.pow(x_probe / 5.0, 2)).reshape(-1, 1)

    phi_net = PhiNet(hidden_dim=32, num_layers=2)
    mu_net = MuNet(hidden_dim=32, num_layers=2, mu_min=1.0, mu_max=1000.0)
    phi_optimizer = torch.optim.Adam(phi_net.parameters(), lr=1e-3)
    mu_optimizer = torch.optim.Adam(mu_net.parameters(), lr=1e-3)

    outer_steps = 2
    phi_steps = 3
    mu_steps = 3
    lambda_data = 1.0
    beta_tv = 1e-6

    final_loss_phi = None
    final_loss_mu = None

    for outer_idx in range(outer_steps):
        set_requires_grad(mu_net, False)
        set_requires_grad(phi_net, True)
        for _ in range(phi_steps):
            phi_optimizer.zero_grad()
            clear_coord_grads(coords, coords_probe)
            with torch.no_grad():
                mu = mu_net(coords)
            phi = phi_net(coords)
            phi_probe = phi_net(coords_probe)
            loss_phi = energy_loss(phi, mu, coords) + lambda_data * data_loss(
                phi_probe,
                coords_probe,
                bz_meas,
            )
            loss_phi.backward()
            phi_optimizer.step()
            final_loss_phi = loss_phi.detach()

        set_requires_grad(phi_net, False)
        set_requires_grad(mu_net, True)
        for _ in range(mu_steps):
            mu_optimizer.zero_grad()
            clear_coord_grads(coords)
            phi_fixed = phi_net(coords)
            mu = mu_net(coords)

            # Dummy gradients for code-loop testing only; these are not final
            # compact-support weak-form test functions.
            dummy_test_grads = torch.zeros(1, num_points, 2)
            dummy_test_grads[0, :, 0] = 1.0

            loss_mu = weak_form_loss(
                mu,
                phi_fixed,
                coords,
                test_grads=dummy_test_grads,
            ) + beta_tv * tv_loss(mu, coords)
            loss_mu.backward()
            mu_optimizer.step()
            final_loss_mu = loss_mu.detach()

        with torch.no_grad():
            mu_snapshot = mu_net(coords)
            print(
                "outer "
                f"{outer_idx + 1}/{outer_steps} | "
                f"loss_phi={final_loss_phi.item():.6e} | "
                f"loss_mu={final_loss_mu.item():.6e} | "
                f"mu_min={mu_snapshot.min().item():.6e} | "
                f"mu_max={mu_snapshot.max().item():.6e}"
            )

    print("Minimal dual-network variational loop passed.")


if __name__ == "__main__":
    main()

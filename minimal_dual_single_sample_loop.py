"""Single-sample .npz loop prototype for the dual-network branch.

This script is only a data-interface closure prototype. It does not perform
formal training, does not save checkpoints or images, and does not claim the
resulting fields are physically valid. Dummy weak-form test gradients must be
replaced with compact-support test functions before real experiments.
"""

import argparse

import torch

from dual_network_data_utils import (
    build_dual_inputs,
    get_probe_coords_from_grid,
    get_single_sample,
    infer_grid_shape,
    load_npz_dataset,
)
from dual_network_losses import data_loss, energy_loss, tv_loss, weak_form_loss
from dual_network_models import MuNet, PhiNet


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def clear_coord_grads(*tensors):
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-sample dual-network .npz loop prototype."
    )
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--phi-steps", type=int, default=2)
    parser.add_argument("--mu-steps", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.npz_path is None:
        print("minimal_dual_single_sample_loop.py is a .npz interface prototype.")
        print("No --npz-path was provided, so no file was read and no loop ran.")
        print("Example:")
        print("  python minimal_dual_single_sample_loop.py --npz-path data/sample.npz")
        return

    torch.manual_seed(0)
    device = torch.device(args.device)

    dataset = load_npz_dataset(args.npz_path)
    signal, coords_array, mu_map = get_single_sample(dataset, args.sample_index)
    dual_inputs = build_dual_inputs(signal, coords_array, mu_map, device=device)
    grid_info = infer_grid_shape(coords_array)

    coords = dual_inputs["coords"]
    bz_meas = dual_inputs["bz_meas"]
    mu_label = dual_inputs["mu_label"]
    probe_coords = get_probe_coords_from_grid(
        grid_info["x_unique"],
        y_s=10.0,
        device=device,
    )
    if bz_meas.shape[0] != probe_coords.shape[0]:
        raise ValueError(
            "bz_meas probe length mismatch: "
            f"bz_meas has {bz_meas.shape[0]} points, "
            f"probe_coords has {probe_coords.shape[0]} points, "
            f"nx is {grid_info['nx']}. "
            "signal length must match the number of unique x coordinates."
        )

    phi_net = PhiNet(hidden_dim=32, num_layers=2).to(device)
    mu_net = MuNet(hidden_dim=32, num_layers=2, mu_min=1.0, mu_max=1000.0).to(device)
    phi_optimizer = torch.optim.Adam(phi_net.parameters(), lr=1e-3)
    mu_optimizer = torch.optim.Adam(mu_net.parameters(), lr=1e-3)

    final_loss_phi = None
    final_loss_mu = None

    for outer_idx in range(args.outer_steps):
        set_requires_grad(mu_net, False)
        set_requires_grad(phi_net, True)
        for _ in range(args.phi_steps):
            phi_optimizer.zero_grad()
            clear_coord_grads(coords, probe_coords)
            with torch.no_grad():
                mu = mu_net(coords)
            phi = phi_net(coords)
            phi_probe = phi_net(probe_coords)
            loss_phi = energy_loss(phi, mu, coords) + data_loss(
                phi_probe,
                probe_coords,
                bz_meas,
            )
            loss_phi.backward()
            phi_optimizer.step()
            final_loss_phi = loss_phi.detach()

        set_requires_grad(phi_net, False)
        set_requires_grad(mu_net, True)
        for _ in range(args.mu_steps):
            mu_optimizer.zero_grad()
            clear_coord_grads(coords)

            # Do not use torch.no_grad() here. weak_form_loss still needs
            # d(phi_fixed) / d(coords), even though PhiNet parameters are frozen.
            phi_fixed = phi_net(coords)
            mu = mu_net(coords)

            # Dummy gradients for interface testing only. These are not the
            # final compact-support weak-form test functions.
            dummy_test_grads = torch.zeros(1, coords.shape[0], 2, device=device)
            dummy_test_grads[0, :, 0] = 1.0

            loss_mu = weak_form_loss(
                mu,
                phi_fixed,
                coords,
                test_grads=dummy_test_grads,
            ) + 1e-6 * tv_loss(mu, coords)
            loss_mu.backward()
            mu_optimizer.step()
            final_loss_mu = loss_mu.detach()

        with torch.no_grad():
            mu_pred = mu_net(coords)
            print(
                f"sample={args.sample_index} | "
                f"outer={outer_idx + 1}/{args.outer_steps} | "
                f"loss_phi={final_loss_phi.item():.6e} | "
                f"loss_mu={final_loss_mu.item():.6e} | "
                f"mu_pred_min={mu_pred.min().item():.6e} | "
                f"mu_pred_max={mu_pred.max().item():.6e} | "
                f"mu_label_min={mu_label.min().item():.6e} | "
                f"mu_label_max={mu_label.max().item():.6e}"
            )

    print("Minimal dual-network single-sample .npz loop passed.")


if __name__ == "__main__":
    main()

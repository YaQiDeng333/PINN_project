"""Single-sample .npz loop prototype for the dual-network branch.

This script is only a data-interface closure prototype. It does not perform
formal training, does not save checkpoints or images, and does not claim the
resulting fields are physically valid. The first fixed compact-support test
centers are only for checking that real test gradients can enter the loop.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dual_network_data_utils import (
    build_dual_inputs,
    get_probe_coords_from_grid,
    get_single_sample,
    infer_grid_shape,
    load_npz_dataset,
)
from dual_network_losses import (
    data_loss,
    energy_loss,
    generate_compact_support_test_grads,
    tv_loss,
    weak_form_loss,
)
from dual_network_models import MuNet, PhiNet


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def clear_coord_grads(*tensors):
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def build_test_centers(
    center_mode,
    coords,
    x_peak=None,
    label_centroid=None,
    x_min=-15.0,
    x_max=15.0,
    y_min=0.0,
    y_max=10.0,
):
    if center_mode == "three":
        center_values = [[-5.0, 5.0], [0.0, 5.0], [5.0, 5.0]]
    elif center_mode == "five":
        center_values = [
            [-7.5, 5.0],
            [-3.75, 5.0],
            [0.0, 5.0],
            [3.75, 5.0],
            [7.5, 5.0],
        ]
    elif center_mode == "nine":
        center_values = [
            [x_value, y_value]
            for y_value in [2.5, 5.0, 7.5]
            for x_value in [-7.5, 0.0, 7.5]
        ]
    elif center_mode == "signal_three":
        if x_peak is None:
            raise ValueError("x_peak is required for signal-informed center modes.")
        center_values = [
            [x_peak - 2.5, 5.0],
            [x_peak, 5.0],
            [x_peak + 2.5, 5.0],
        ]
    elif center_mode == "signal_nine":
        if x_peak is None:
            raise ValueError("x_peak is required for signal-informed center modes.")
        center_values = [
            [x_value, y_value]
            for y_value in [2.5, 5.0, 7.5]
            for x_value in [x_peak - 2.5, x_peak, x_peak + 2.5]
        ]
    elif center_mode == "label_three":
        if label_centroid is None:
            raise ValueError(
                "label centroid is required for label-informed center modes."
            )
        label_x, label_y = label_centroid
        center_values = [
            [label_x - 2.5, label_y],
            [label_x, label_y],
            [label_x + 2.5, label_y],
        ]
    elif center_mode == "label_nine":
        if label_centroid is None:
            raise ValueError(
                "label centroid is required for label-informed center modes."
            )
        label_x, label_y = label_centroid
        center_values = [
            [x_value, y_value]
            for y_value in [label_y - 2.5, label_y, label_y + 2.5]
            for x_value in [label_x - 2.5, label_x, label_x + 2.5]
        ]
    else:
        raise ValueError(f"unsupported center_mode: {center_mode}")

    centers = torch.tensor(center_values, dtype=coords.dtype, device=coords.device)
    centers[:, 0] = torch.clamp(centers[:, 0], min=x_min, max=x_max)
    centers[:, 1] = torch.clamp(centers[:, 1], min=y_min, max=y_max)
    return centers


def find_signal_peak_x(bz_meas, probe_coords):
    peak_index = torch.argmax(torch.abs(bz_meas.reshape(-1))).item()
    return float(probe_coords.detach()[peak_index, 0].item())


def compute_mu_diagnostics(mu_pred, mu_label, threshold=500.0):
    diff = mu_pred - mu_label
    pred_mask = mu_pred < threshold
    label_mask = mu_label < threshold
    intersection = torch.logical_and(pred_mask, label_mask).sum()
    union = torch.logical_or(pred_mask, label_mask).sum()

    # If both masks are empty, report 0.0 instead of treating it as a perfect
    # match. The thresholded defect signal is absent in that case.
    if union.item() == 0:
        defect_iou = 0.0
    else:
        defect_iou = (intersection.float() / union.float()).item()

    return {
        "mu_mse": torch.mean(diff.pow(2)).item(),
        "mu_mae": torch.mean(torch.abs(diff)).item(),
        "defect_area_pred": int(pred_mask.sum().item()),
        "defect_area_label": int(label_mask.sum().item()),
        "defect_iou": defect_iou,
    }


def compute_area_prior_loss(mu, target_defect_fraction, temperature):
    soft_defect = torch.sigmoid((500.0 - mu) / temperature)
    pred_defect_fraction = torch.mean(soft_defect)
    area_prior_loss = torch.square(pred_defect_fraction - target_defect_fraction)
    return area_prior_loss, pred_defect_fraction


def compute_mask_prior_loss(mu, mu_label, temperature, eps=1e-8):
    soft_defect = torch.sigmoid((500.0 - mu) / temperature)
    label_mask = (mu_label < 500.0).float()
    intersection = torch.sum(soft_defect * label_mask)
    denominator = torch.sum(soft_defect) + torch.sum(label_mask) + eps
    dice_loss = 1.0 - (2.0 * intersection + eps) / denominator
    return dice_loss


def compute_centroid(coords, mask):
    flat_mask = mask.reshape(-1)
    if not torch.any(flat_mask):
        return None
    selected_coords = coords[flat_mask]
    centroid = torch.mean(selected_coords, dim=0)
    return float(centroid[0].item()), float(centroid[1].item())


def find_label_centroid(coords, mu_label, require=False):
    label_mask = mu_label < 500.0
    centroid = compute_centroid(coords.detach(), label_mask)
    if centroid is None and require:
        raise ValueError(
            "label_mask is empty for label-informed oracle center mode; "
            "mu_label must contain at least one point below 500."
        )
    return centroid


def save_final_diagnostics(
    diagnostics_dir,
    mu_pred,
    mu_label,
    coords,
    grid_info,
    final_loss_phi,
    final_loss_mu,
    final_diagnostics,
):
    output_dir = Path(diagnostics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nx = grid_info["nx"]
    ny = grid_info["ny"]
    mu_pred_map = mu_pred.detach().cpu().numpy().reshape(ny, nx)
    mu_label_map = mu_label.detach().cpu().numpy().reshape(ny, nx)
    pred_mask = (mu_pred.detach() < 500.0)
    label_mask = (mu_label.detach() < 500.0)
    pred_mask_map = pred_mask.cpu().numpy().reshape(ny, nx)
    label_mask_map = label_mask.cpu().numpy().reshape(ny, nx)
    pred_centroid = compute_centroid(coords.detach(), pred_mask)
    label_centroid = compute_centroid(coords.detach(), label_mask)

    np.save(output_dir / "final_mu_pred.npy", mu_pred_map)
    np.save(output_dir / "final_mu_label.npy", mu_label_map)
    np.save(output_dir / "final_pred_mask.npy", pred_mask_map)
    np.save(output_dir / "final_label_mask.npy", label_mask_map)

    with (output_dir / "final_diagnostics.txt").open("w", encoding="utf-8") as file:
        file.write(f"final_loss_phi={final_loss_phi.item():.6e}\n")
        file.write(f"final_loss_mu={final_loss_mu.item():.6e}\n")
        file.write(f"final_mu_mse={final_diagnostics['mu_mse']:.6e}\n")
        file.write(f"final_mu_mae={final_diagnostics['mu_mae']:.6e}\n")
        file.write(
            f"final_defect_area_pred={final_diagnostics['defect_area_pred']}\n"
        )
        file.write(f"defect_area_label={final_diagnostics['defect_area_label']}\n")
        file.write(f"final_defect_iou={final_diagnostics['defect_iou']:.6e}\n")
        file.write(f"pred_centroid={pred_centroid}\n")
        file.write(f"label_centroid={label_centroid}\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_unique = grid_info["x_unique"]
    y_unique = grid_info["y_unique"]
    extent = [
        float(np.min(x_unique)),
        float(np.max(x_unique)),
        float(np.min(y_unique)),
        float(np.max(y_unique)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = [
        (axes[0, 0], mu_pred_map, "final mu_pred", "viridis"),
        (axes[0, 1], mu_label_map, "mu_label", "viridis"),
        (axes[1, 0], pred_mask_map.astype(float), "pred_mask mu<500", "gray_r"),
        (axes[1, 1], label_mask_map.astype(float), "label_mask mu<500", "gray_r"),
    ]
    for axis, values, title, cmap in panels:
        image = axis.imshow(values, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "mu_pred_vs_label.png", dpi=150)
    plt.close(fig)

    print(f"Saved final diagnostics to {output_dir}")


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
    parser.add_argument("--test-radius", type=float, default=5.0)
    parser.add_argument(
        "--center-mode",
        choices=[
            "three",
            "five",
            "nine",
            "signal_three",
            "signal_nine",
            "label_three",
            "label_nine",
        ],
        default="three",
    )
    parser.add_argument("--lambda-area-prior", type=float, default=0.0)
    parser.add_argument("--area-prior-temperature", type=float, default=50.0)
    parser.add_argument("--lambda-mask-prior", type=float, default=0.0)
    parser.add_argument("--mask-prior-temperature", type=float, default=50.0)
    parser.add_argument("--diagnostics-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.npz_path is None:
        print("minimal_dual_single_sample_loop.py is a .npz interface prototype.")
        print("No --npz-path was provided, so no file was read and no loop ran.")
        print("Example:")
        print("  python minimal_dual_single_sample_loop.py --npz-path data/sample.npz")
        return
    if args.area_prior_temperature <= 0.0:
        raise ValueError("--area-prior-temperature must be positive.")
    if args.mask_prior_temperature <= 0.0:
        raise ValueError("--mask-prior-temperature must be positive.")

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

    # First fixed test-function layout for single-sample closure testing only.
    # Later versions should place centers based on Omega, grid density, and the
    # defect region. normalize=True improves numerical stability, but changes
    # residual weights across different support sizes.
    x_peak = find_signal_peak_x(bz_meas, probe_coords)
    label_centroid = find_label_centroid(
        coords,
        mu_label,
        require=args.center_mode in {"label_three", "label_nine"},
    )
    # label_* modes are oracle diagnostics only. They use the true mu_label
    # defect centroid and are not part of the final unsupervised inversion plan.
    centers = build_test_centers(
        args.center_mode,
        coords,
        x_peak=x_peak,
        label_centroid=label_centroid,
    )
    label_centroid_text = (
        "None"
        if label_centroid is None
        else f"({label_centroid[0]:.6e}, {label_centroid[1]:.6e})"
    )
    test_radius = args.test_radius
    target_defect_fraction = torch.mean((mu_label < 500.0).float())
    printed_test_info = False

    phi_net = PhiNet(hidden_dim=32, num_layers=2).to(device)
    mu_net = MuNet(hidden_dim=32, num_layers=2, mu_min=1.0, mu_max=1000.0).to(device)
    phi_optimizer = torch.optim.Adam(phi_net.parameters(), lr=1e-3)
    mu_optimizer = torch.optim.Adam(mu_net.parameters(), lr=1e-3)

    final_loss_phi = None
    final_loss_mu = None
    final_diagnostics = None
    final_area_prior = None
    final_mask_prior = None
    final_mu_pred = None

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

            test_grads = generate_compact_support_test_grads(
                coords=coords,
                centers=centers,
                radius=test_radius,
                normalize=True,
            )
            if not printed_test_info:
                print(
                    f"center_mode={args.center_mode} | "
                    f"x_peak={x_peak:.6e} | "
                    f"label_centroid={label_centroid_text} | "
                    f"test_centers={centers.shape[0]} | "
                    f"test_radius={test_radius:.6e} | "
                    f"test_grads_shape={tuple(test_grads.shape)}"
                )
                printed_test_info = True

            loss_mu = weak_form_loss(
                mu,
                phi_fixed,
                coords,
                test_grads=test_grads,
            ) + 1e-6 * tv_loss(mu, coords)
            area_prior_loss, _ = compute_area_prior_loss(
                mu,
                target_defect_fraction,
                args.area_prior_temperature,
            )
            # mask prior is a supervised branch diagnostic only, not part of
            # the final unsupervised inversion design.
            dice_loss = compute_mask_prior_loss(
                mu,
                mu_label,
                args.mask_prior_temperature,
            )
            loss_mu = loss_mu + args.lambda_area_prior * area_prior_loss
            loss_mu = loss_mu + args.lambda_mask_prior * dice_loss
            loss_mu.backward()
            mu_optimizer.step()
            final_loss_mu = loss_mu.detach()

        with torch.no_grad():
            mu_pred = mu_net(coords)
            final_mu_pred = mu_pred.detach().clone()
            final_diagnostics = compute_mu_diagnostics(mu_pred, mu_label)
            area_prior_loss, pred_defect_fraction = compute_area_prior_loss(
                mu_pred,
                target_defect_fraction,
                args.area_prior_temperature,
            )
            final_area_prior = {
                "area_prior_loss": area_prior_loss.item(),
                "pred_defect_fraction": pred_defect_fraction.item(),
                "target_defect_fraction": target_defect_fraction.item(),
            }
            dice_loss = compute_mask_prior_loss(
                mu_pred,
                mu_label,
                args.mask_prior_temperature,
            )
            final_mask_prior = {
                "dice_loss": dice_loss.item(),
                "lambda_mask_prior": args.lambda_mask_prior,
            }
            print(
                f"sample={args.sample_index} | "
                f"outer={outer_idx + 1}/{args.outer_steps} | "
                f"loss_phi={final_loss_phi.item():.6e} | "
                f"loss_mu={final_loss_mu.item():.6e} | "
                f"mu_pred_min={mu_pred.min().item():.6e} | "
                f"mu_pred_max={mu_pred.max().item():.6e} | "
                f"mu_label_min={mu_label.min().item():.6e} | "
                f"mu_label_max={mu_label.max().item():.6e} | "
                f"mu_mse={final_diagnostics['mu_mse']:.6e} | "
                f"mu_mae={final_diagnostics['mu_mae']:.6e} | "
                f"defect_area_pred={final_diagnostics['defect_area_pred']} | "
                f"defect_area_label={final_diagnostics['defect_area_label']} | "
                f"defect_iou={final_diagnostics['defect_iou']:.6e} | "
                f"area_prior_loss={final_area_prior['area_prior_loss']:.6e} | "
                f"pred_defect_fraction="
                f"{final_area_prior['pred_defect_fraction']:.6e} | "
                f"target_defect_fraction="
                f"{final_area_prior['target_defect_fraction']:.6e} | "
                f"dice_loss={final_mask_prior['dice_loss']:.6e} | "
                f"lambda_mask_prior="
                f"{final_mask_prior['lambda_mask_prior']:.6e}"
            )

    if (
        final_diagnostics is not None
        and final_area_prior is not None
        and final_mask_prior is not None
    ):
        print(
            "final diagnostics summary | "
            f"mu_mse={final_diagnostics['mu_mse']:.6e} | "
            f"mu_mae={final_diagnostics['mu_mae']:.6e} | "
            f"defect_area_pred={final_diagnostics['defect_area_pred']} | "
            f"defect_area_label={final_diagnostics['defect_area_label']} | "
            f"defect_iou={final_diagnostics['defect_iou']:.6e} | "
            f"area_prior_loss={final_area_prior['area_prior_loss']:.6e} | "
            f"pred_defect_fraction="
            f"{final_area_prior['pred_defect_fraction']:.6e} | "
            f"target_defect_fraction="
            f"{final_area_prior['target_defect_fraction']:.6e} | "
            f"dice_loss={final_mask_prior['dice_loss']:.6e} | "
            f"lambda_mask_prior="
            f"{final_mask_prior['lambda_mask_prior']:.6e}"
        )

    if args.diagnostics_dir is not None and final_mu_pred is not None:
        save_final_diagnostics(
            args.diagnostics_dir,
            final_mu_pred,
            mu_label,
            coords,
            grid_info,
            final_loss_phi,
            final_loss_mu,
            final_diagnostics,
        )

    print("Minimal dual-network single-sample loop passed.")


if __name__ == "__main__":
    main()

"""Small-scale experiment runner for the dual-network variational branch.

This script is a branch-local runner for controlled diagnostics. It is not the
main supervised training pipeline, not a large-scale training script, and it
does not save model checkpoints. Mask and BCE priors use mu_label and are
therefore semi-supervised diagnostic upper-bound terms, not an unsupervised
weak-form inversion setting.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


METRIC_COLUMNS = [
    "sample_index",
    "loss_phi_final",
    "loss_mu_final",
    "mu_mse_final",
    "mu_mae_final",
    "defect_area_pred",
    "defect_area_label",
    "defect_iou",
    "pred_centroid_x",
    "pred_centroid_y",
    "label_centroid_x",
    "label_centroid_y",
    "lambda_area_prior",
    "lambda_mask_prior",
    "lambda_mask_bce_prior",
    "center_mode",
    "test_radius",
    "hidden_dim",
    "num_layers",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Branch-local dual-network weak-form experiment runner."
    )
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-indices", default="0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outer-steps", type=int, default=30)
    parser.add_argument("--phi-steps", type=int, default=30)
    parser.add_argument("--mu-steps", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
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
    parser.add_argument("--lambda-area-prior", type=float, default=1.0)
    parser.add_argument("--lambda-mask-prior", type=float, default=1.0)
    parser.add_argument("--lambda-mask-bce-prior", type=float, default=0.0)
    parser.add_argument("--area-prior-temperature", type=float, default=50.0)
    parser.add_argument("--mask-prior-temperature", type=float, default=50.0)
    return parser.parse_args()


def parse_sample_indices(raw_text):
    indices = []
    for item in raw_text.split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    if not indices:
        raise ValueError("--sample-indices must contain at least one integer")
    return indices


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
            raise ValueError("x_peak is required for signal-informed centers")
        center_values = [[x_peak - 2.5, 5.0], [x_peak, 5.0], [x_peak + 2.5, 5.0]]
    elif center_mode == "signal_nine":
        if x_peak is None:
            raise ValueError("x_peak is required for signal-informed centers")
        center_values = [
            [x_value, y_value]
            for y_value in [2.5, 5.0, 7.5]
            for x_value in [x_peak - 2.5, x_peak, x_peak + 2.5]
        ]
    elif center_mode == "label_three":
        if label_centroid is None:
            raise ValueError("label_centroid is required for label-informed centers")
        label_x, label_y = label_centroid
        center_values = [[label_x - 2.5, label_y], [label_x, label_y], [label_x + 2.5, label_y]]
    elif center_mode == "label_nine":
        if label_centroid is None:
            raise ValueError("label_centroid is required for label-informed centers")
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
        raise ValueError("label-informed center mode requires mu_label < 500 points")
    return centroid


def compute_mu_diagnostics(mu_pred, mu_label, coords, threshold=500.0):
    diff = mu_pred - mu_label
    pred_mask = mu_pred < threshold
    label_mask = mu_label < threshold
    intersection = torch.logical_and(pred_mask, label_mask).sum()
    union = torch.logical_or(pred_mask, label_mask).sum()
    defect_iou = 0.0 if union.item() == 0 else (intersection.float() / union.float()).item()
    pred_centroid = compute_centroid(coords.detach(), pred_mask)
    label_centroid = compute_centroid(coords.detach(), label_mask)
    return {
        "mu_mse": torch.mean(diff.pow(2)).item(),
        "mu_mae": torch.mean(torch.abs(diff)).item(),
        "defect_area_pred": int(pred_mask.sum().item()),
        "defect_area_label": int(label_mask.sum().item()),
        "defect_iou": defect_iou,
        "pred_centroid": pred_centroid,
        "label_centroid": label_centroid,
    }


def compute_area_prior_loss(mu, target_defect_fraction, temperature):
    soft_defect = torch.sigmoid((500.0 - mu) / temperature)
    pred_defect_fraction = torch.mean(soft_defect)
    return torch.square(pred_defect_fraction - target_defect_fraction)


def compute_mask_prior_loss(mu, mu_label, temperature, eps=1e-8):
    soft_defect = torch.sigmoid((500.0 - mu) / temperature)
    label_mask = (mu_label < 500.0).float()
    intersection = torch.sum(soft_defect * label_mask)
    denominator = torch.sum(soft_defect) + torch.sum(label_mask) + eps
    return 1.0 - (2.0 * intersection + eps) / denominator


def compute_mask_bce_prior_loss(mu, mu_label, temperature):
    soft_defect = torch.sigmoid((500.0 - mu) / temperature)
    label_mask = (mu_label < 500.0).float()
    return F.binary_cross_entropy(soft_defect, label_mask)


def save_final_outputs(
    sample_dir,
    mu_pred,
    mu_label,
    coords,
    grid_info,
    final_loss_phi,
    final_loss_mu,
    diagnostics,
):
    sample_dir.mkdir(parents=True, exist_ok=True)
    nx = grid_info["nx"]
    ny = grid_info["ny"]
    mu_pred_map = mu_pred.detach().cpu().numpy().reshape(ny, nx)
    mu_label_map = mu_label.detach().cpu().numpy().reshape(ny, nx)
    pred_mask = mu_pred.detach() < 500.0
    label_mask = mu_label.detach() < 500.0
    pred_mask_map = pred_mask.cpu().numpy().reshape(ny, nx)
    label_mask_map = label_mask.cpu().numpy().reshape(ny, nx)

    np.save(sample_dir / "final_mu_pred.npy", mu_pred_map)
    np.save(sample_dir / "final_mu_label.npy", mu_label_map)
    np.save(sample_dir / "final_pred_mask.npy", pred_mask_map)
    np.save(sample_dir / "final_label_mask.npy", label_mask_map)

    with (sample_dir / "final_diagnostics.txt").open("w", encoding="utf-8") as file:
        file.write(f"final_loss_phi={final_loss_phi.item():.6e}\n")
        file.write(f"final_loss_mu={final_loss_mu.item():.6e}\n")
        file.write(f"final_mu_mse={diagnostics['mu_mse']:.6e}\n")
        file.write(f"final_mu_mae={diagnostics['mu_mae']:.6e}\n")
        file.write(f"final_defect_area_pred={diagnostics['defect_area_pred']}\n")
        file.write(f"defect_area_label={diagnostics['defect_area_label']}\n")
        file.write(f"final_defect_iou={diagnostics['defect_iou']:.6e}\n")
        file.write(f"pred_centroid={diagnostics['pred_centroid']}\n")
        file.write(f"label_centroid={diagnostics['label_centroid']}\n")

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
    fig.savefig(sample_dir / "mu_pred_vs_label.png", dpi=150)
    plt.close(fig)


def write_log(log_file, message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


def run_sample(dataset, sample_index, args, output_dir, device):
    sample_dir = output_dir / f"sample_{sample_index}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_path = sample_dir / "run_log.txt"

    torch.manual_seed(0)
    signal, coords_array, mu_map = get_single_sample(dataset, sample_index)
    dual_inputs = build_dual_inputs(signal, coords_array, mu_map, device=device)
    grid_info = infer_grid_shape(coords_array)
    coords = dual_inputs["coords"]
    bz_meas = dual_inputs["bz_meas"]
    mu_label = dual_inputs["mu_label"]
    probe_coords = get_probe_coords_from_grid(grid_info["x_unique"], y_s=10.0, device=device)
    if bz_meas.shape[0] != probe_coords.shape[0]:
        raise ValueError(
            "signal length must match the number of unique x coordinates: "
            f"bz_meas={bz_meas.shape[0]}, probe_coords={probe_coords.shape[0]}"
        )

    x_peak = find_signal_peak_x(bz_meas, probe_coords)
    label_centroid = find_label_centroid(
        coords,
        mu_label,
        require=args.center_mode in {"label_three", "label_nine"},
    )
    centers = build_test_centers(
        args.center_mode,
        coords,
        x_peak=x_peak,
        label_centroid=label_centroid,
    )
    target_defect_fraction = torch.mean((mu_label < 500.0).float())

    phi_net = PhiNet(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    mu_net = MuNet(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        mu_min=1.0,
        mu_max=1000.0,
    ).to(device)
    phi_optimizer = torch.optim.Adam(phi_net.parameters(), lr=1e-3)
    mu_optimizer = torch.optim.Adam(mu_net.parameters(), lr=1e-3)

    final_loss_phi = None
    final_loss_mu = None
    final_mu_pred = None
    final_diagnostics = None

    with log_path.open("w", encoding="utf-8") as log_file:
        write_log(
            log_file,
            "runner config | "
            f"sample={sample_index} | center_mode={args.center_mode} | "
            f"test_centers={centers.shape[0]} | test_radius={args.test_radius:.6e} | "
            f"hidden_dim={args.hidden_dim} | num_layers={args.num_layers} | "
            f"lambda_area_prior={args.lambda_area_prior:.6e} | "
            f"lambda_mask_prior={args.lambda_mask_prior:.6e} | "
            f"lambda_mask_bce_prior={args.lambda_mask_bce_prior:.6e}",
        )

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
                phi_fixed = phi_net(coords)
                mu = mu_net(coords)
                test_grads = generate_compact_support_test_grads(
                    coords=coords,
                    centers=centers,
                    radius=args.test_radius,
                    normalize=True,
                )
                weak_loss = weak_form_loss(mu, phi_fixed, coords, test_grads=test_grads)
                area_prior_loss = compute_area_prior_loss(
                    mu,
                    target_defect_fraction,
                    args.area_prior_temperature,
                )
                dice_loss = compute_mask_prior_loss(
                    mu,
                    mu_label,
                    args.mask_prior_temperature,
                )
                mask_bce_loss = compute_mask_bce_prior_loss(
                    mu,
                    mu_label,
                    args.mask_prior_temperature,
                )
                loss_mu = (
                    weak_loss
                    + 1e-6 * tv_loss(mu, coords)
                    + args.lambda_area_prior * area_prior_loss
                    + args.lambda_mask_prior * dice_loss
                    + args.lambda_mask_bce_prior * mask_bce_loss
                )
                loss_mu.backward()
                mu_optimizer.step()
                final_loss_mu = loss_mu.detach()

            with torch.no_grad():
                mu_pred = mu_net(coords)
                diagnostics = compute_mu_diagnostics(mu_pred, mu_label, coords)
                write_log(
                    log_file,
                    f"sample={sample_index} | outer={outer_idx + 1}/{args.outer_steps} | "
                    f"loss_phi={final_loss_phi.item():.6e} | "
                    f"loss_mu={final_loss_mu.item():.6e} | "
                    f"mu_mse={diagnostics['mu_mse']:.6e} | "
                    f"mu_mae={diagnostics['mu_mae']:.6e} | "
                    f"defect_area_pred={diagnostics['defect_area_pred']} | "
                    f"defect_area_label={diagnostics['defect_area_label']} | "
                    f"defect_iou={diagnostics['defect_iou']:.6e}",
                )
                final_mu_pred = mu_pred.detach().clone()
                final_diagnostics = diagnostics

    save_final_outputs(
        sample_dir,
        final_mu_pred,
        mu_label,
        coords,
        grid_info,
        final_loss_phi,
        final_loss_mu,
        final_diagnostics,
    )

    pred_centroid = final_diagnostics["pred_centroid"]
    label_centroid = final_diagnostics["label_centroid"]
    return {
        "sample_index": sample_index,
        "loss_phi_final": final_loss_phi.item(),
        "loss_mu_final": final_loss_mu.item(),
        "mu_mse_final": final_diagnostics["mu_mse"],
        "mu_mae_final": final_diagnostics["mu_mae"],
        "defect_area_pred": final_diagnostics["defect_area_pred"],
        "defect_area_label": final_diagnostics["defect_area_label"],
        "defect_iou": final_diagnostics["defect_iou"],
        "pred_centroid_x": "" if pred_centroid is None else pred_centroid[0],
        "pred_centroid_y": "" if pred_centroid is None else pred_centroid[1],
        "label_centroid_x": "" if label_centroid is None else label_centroid[0],
        "label_centroid_y": "" if label_centroid is None else label_centroid[1],
        "lambda_area_prior": args.lambda_area_prior,
        "lambda_mask_prior": args.lambda_mask_prior,
        "lambda_mask_bce_prior": args.lambda_mask_bce_prior,
        "center_mode": args.center_mode,
        "test_radius": args.test_radius,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
    }


def write_metrics_csv(output_dir, rows):
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return metrics_path


def main():
    args = parse_args()
    if args.npz_path is None or args.output_dir is None:
        print("train_dual_variational.py is a small branch experiment runner.")
        print("It does not start unless both --npz-path and --output-dir are provided.")
        print("Example:")
        print(
            "  python train_dual_variational.py --npz-path data/train.npz "
            "--output-dir experiments/dual_network/run --sample-indices 0,1,2"
        )
        return
    if args.area_prior_temperature <= 0.0:
        raise ValueError("--area-prior-temperature must be positive")
    if args.mask_prior_temperature <= 0.0:
        raise ValueError("--mask-prior-temperature must be positive")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive")
    if args.num_layers < 1:
        raise ValueError("--num-layers must be at least 1")

    sample_indices = parse_sample_indices(args.sample_indices)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = load_npz_dataset(args.npz_path)

    rows = []
    for sample_index in sample_indices:
        rows.append(run_sample(dataset, sample_index, args, output_dir, device))

    metrics_path = write_metrics_csv(output_dir, rows)
    print(f"Saved metrics to {metrics_path}")
    print("Dual-network train runner completed.")


if __name__ == "__main__":
    main()

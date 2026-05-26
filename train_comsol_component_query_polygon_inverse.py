"""Train a component-query COMSOL center-anchored polygon inverse model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from comsol_component_query_polygon_inverse_models import ComponentQueryPolygonInverseNet
from train_comsol_center_anchored_polygon_inverse import (
    _append_config,
    build_tensors,
    choose_device,
    compute_losses,
    evaluate,
    export_predictions,
    load_dataset,
    load_targets,
    write_csv,
)


def _set_compat_defaults(args: argparse.Namespace) -> None:
    args.lambda_center_aux = 0.0
    args.lambda_box_aux = 0.0
    if not hasattr(args, "lambda_area_aux"):
        args.lambda_area_aux = 0.0
    args.lambda_edge_aux = 0.0
    args.center_consistency_mode = "none"
    args.lambda_center_consistency = 0.0
    args.center_consistency_smoothl1_beta = 0.1
    args.center_y_bin_extra_loss_mode = "none"
    args.lambda_center_y_bin_extra = 0.0
    args.center_y_bin_neighbor_smoothing = 0.0
    args.center_y_bin_distance_sigma = 0.75
    args.local_shape_output_mode = "raw"
    args.local_shape_bound_mode = "fixed_grid"
    args.local_shape_fixed_bound_x_grid = 24.0
    args.local_shape_fixed_bound_y_grid = 8.0
    args.local_shape_bound_x_grid = 24.0
    args.local_shape_bound_y_grid = 8.0
    args.local_shape_train_stats_margin = 1.25
    args.local_shape_conditioning_mode = "none"
    args.local_shape_conditioning_dim = 0
    args.joint_center_shape_mode = "none"
    args.joint_center_teacher_forcing_start = 0.0
    args.joint_center_teacher_forcing_end = 0.0
    args.joint_center_teacher_forcing_steps = 0


def _append_component_query_config(row: dict, args: argparse.Namespace) -> dict:
    row = _append_config(row, args)
    row["inverse_route"] = "component_query"
    row["component_query_count"] = args.max_components
    row["component_query_shared_head"] = 1
    return row


def write_component_query_run_summary(
    output_dir: Path,
    args: argparse.Namespace,
    train_row: dict,
    val_row: dict,
    test_row: dict,
    train_targets: dict,
) -> None:
    lines = [
        "# COMSOL component-query polygon inverse run summary",
        "",
        "This runner keeps the center-anchored target schema and hard argmax decode, but predicts all component outputs from shared fixed-slot query latents.",
        "",
        "## Config",
        "",
        f"- inverse_route: `component_query`",
        f"- steps: `{args.steps}`",
        f"- lr: `{args.lr}`",
        f"- hidden_dim: `{args.hidden_dim}`",
        f"- latent_dim: `{args.latent_dim}`",
        f"- max_components: `{args.max_components}`",
        f"- max_vertices: `{args.max_vertices}`",
        f"- type_vocab: `{', '.join(train_targets['type_vocab'])}`",
        f"- center_bin_size_cells: `{train_targets['center_bin_size_cells']}`",
        f"- center_x_bins: `{len(train_targets['x_centers'])}`",
        f"- center_y_bins: `{len(train_targets['y_centers'])}`",
        f"- lambda_presence: `{args.lambda_presence}`",
        f"- lambda_type: `{args.lambda_type}`",
        f"- lambda_center_bin: `{args.lambda_center_bin}`",
        f"- lambda_center_offset: `{args.lambda_center_offset}`",
        f"- lambda_local_vertex: `{args.lambda_local_vertex}`",
        f"- lambda_area_aux: `{args.lambda_area_aux}`",
        f"- lambda_decoded_center_aux: `{args.lambda_decoded_center_aux}`",
        f"- lambda_polygon_centroid_aux: `{args.lambda_polygon_centroid_aux}`",
        f"- center_centroid_aux_smoothl1_beta: `{args.center_centroid_aux_smoothl1_beta}`",
        f"- export_predictions: `{args.export_predictions}`",
        f"- seed: `{args.seed}`",
        "",
        "## Final Metrics",
        "",
        "| split | polygon_mask_iou | polygon_mask_iou_min | decoded_vertex_mae | local_vertex_mae_grid | hard_center_l2_grid | presence_acc | type_acc | x_bin_acc | y_bin_acc | zero_iou | out_of_grid | signed_flip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [train_row, val_row, test_row]:
        lines.append(
            f"| {row['split']} | `{row['polygon_mask_iou']:.6f}` | `{row['polygon_mask_iou_min']:.6f}` | "
            f"`{row['decoded_vertex_mae']:.6e}` | `{row['local_vertex_mae_grid']:.6e}` | "
            f"`{row['hard_decoded_center_l2_grid']:.6f}` | `{row['presence_acc']:.6f}` | "
            f"`{row['present_type_acc']:.6f}` | `{row['center_x_bin_acc']:.6f}` | `{row['center_y_bin_acc']:.6f}` | "
            f"`{int(row['zero_iou_count'])}` | `{int(row['out_of_grid_vertex_count'])}` | `{int(row['signed_area_flip_count'])}` |"
        )
    lines.extend(["", "No checkpoint or model weights are saved by this runner."])
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz")
    parser.add_argument("--train-targets")
    parser.add_argument("--val-npz")
    parser.add_argument("--val-targets")
    parser.add_argument("--test-npz")
    parser.add_argument("--test-targets")
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--max-vertices", type=int, default=4)
    parser.add_argument("--lambda-presence", type=float, default=1.0)
    parser.add_argument("--lambda-type", type=float, default=1.0)
    parser.add_argument("--lambda-center-bin", type=float, default=1.0)
    parser.add_argument("--lambda-center-offset", type=float, default=10.0)
    parser.add_argument("--lambda-local-vertex", type=float, default=1.0)
    parser.add_argument("--lambda-area-aux", type=float, default=0.0)
    parser.add_argument("--lambda-decoded-center-aux", type=float, default=0.0)
    parser.add_argument("--lambda-polygon-centroid-aux", type=float, default=0.0)
    parser.add_argument("--center-centroid-aux-smoothl1-beta", type=float, default=0.01)
    parser.add_argument("--local-vertex-smoothl1-beta", type=float, default=0.1)
    parser.add_argument("--center-offset-smoothl1-beta", type=float, default=0.01)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--history-interval", type=int, default=100)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    required = [args.train_npz, args.train_targets, args.val_npz, args.val_targets, args.test_npz, args.test_targets, args.output_dir]
    if not all(required):
        parser.print_help()
        return 0
    if args.max_components != 3 or args.max_vertices != 4:
        raise ValueError("Component-query polygon inverse runner supports max_components=3 and max_vertices=4 only.")
    if args.lambda_decoded_center_aux < 0.0 or args.lambda_polygon_centroid_aux < 0.0:
        raise ValueError("--lambda-decoded-center-aux and --lambda-polygon-centroid-aux must be non-negative.")
    if args.lambda_area_aux < 0.0:
        raise ValueError("--lambda-area-aux must be non-negative.")
    if args.center_centroid_aux_smoothl1_beta <= 0.0:
        raise ValueError("--center-centroid-aux-smoothl1-beta must be positive.")
    _set_compat_defaults(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_dataset = load_dataset(args.train_npz)
    val_dataset = load_dataset(args.val_npz)
    test_dataset = load_dataset(args.test_npz)
    train_targets = load_targets(args.train_targets)
    val_targets = load_targets(args.val_targets, train_type_vocab=train_targets["type_vocab"])
    test_targets = load_targets(args.test_targets, train_type_vocab=train_targets["type_vocab"])
    for split, targets in [("val", val_targets), ("test", test_targets)]:
        if len(targets["x_centers"]) != len(train_targets["x_centers"]) or len(targets["y_centers"]) != len(train_targets["y_centers"]):
            raise ValueError(f"{split} center bin counts do not match train targets.")
    train_tensors = build_tensors(train_dataset, train_targets, device)
    val_tensors = build_tensors(val_dataset, val_targets, device)
    test_tensors = build_tensors(test_dataset, test_targets, device)
    model = ComponentQueryPolygonInverseNet(
        signal_len=train_dataset["signals"].shape[1],
        center_x_bins=len(train_targets["x_centers"]),
        center_y_bins=len(train_targets["y_centers"]),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_components=args.max_components,
        max_vertices=args.max_vertices,
        num_types=len(train_targets["type_vocab"]),
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        out = model(train_tensors["signals"])
        loss, values = compute_losses(out, train_tensors, args)
        values["inverse_route"] = "component_query"
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()
        if step == 1 or step == args.steps or (args.history_interval > 0 and step % args.history_interval == 0):
            history.append({"step": step, **values})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if history:
        write_csv(output_dir / "training_history.csv", history)
    train_row = _append_component_query_config(evaluate(model, train_tensors, "train", args), args)
    val_row = _append_component_query_config(evaluate(model, val_tensors, "val", args), args)
    test_row = _append_component_query_config(evaluate(model, test_tensors, "test", args), args)
    write_csv(output_dir / "metrics.csv", [train_row])
    write_csv(output_dir / "eval_metrics.csv", [val_row])
    write_csv(output_dir / "test_metrics.csv", [test_row])
    if args.export_predictions:
        export_predictions(output_dir, "train", model, train_tensors, train_targets, args)
        export_predictions(output_dir, "val", model, val_tensors, val_targets, args)
        export_predictions(output_dir, "test", model, test_tensors, test_targets, args)
    write_component_query_run_summary(output_dir, args, train_row, val_row, test_row, train_targets)
    print(f"Saved component-query polygon inverse metrics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

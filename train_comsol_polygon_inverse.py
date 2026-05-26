"""Train a small COMSOL polygon inverse model from Bz signals to polygon vertices."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comsol_polygon_inverse_models import PolygonInverseNet
from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components


def _flatten_signals(signals: np.ndarray) -> np.ndarray:
    if signals.ndim == 3:
        return signals.reshape(signals.shape[0], -1).astype(np.float32)
    if signals.ndim == 2:
        return signals.astype(np.float32)
    raise ValueError(f"signals must have shape [B,C,L] or [B,L], got {signals.shape}")


def _zscore_per_sample(signals: np.ndarray) -> np.ndarray:
    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((signals - mean) / std).astype(np.float32)


def load_dataset(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        signals = _zscore_per_sample(_flatten_signals(data["signals"]))
        masks = data["masks"].astype(np.float32)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
    return {"signals": signals, "masks": masks, "x": x, "y": y}


def _string_list(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.tolist()]


def load_polygon_targets(path: str | Path, train_type_vocab: list[str] | None = None) -> dict:
    with np.load(path, allow_pickle=True) as data:
        vertices = data["polygon_vertices_norm"].astype(np.float32)
        vertex_mask = data["polygon_vertex_mask"].astype(np.float32)
        presence = data["presence_targets"].astype(np.float32)
        type_targets = data["type_targets"].astype(np.int64)
        type_vocab = _string_list(data["type_vocab"])
        sample_indices = data["sample_indices"].astype(np.int64) if "sample_indices" in data else np.arange(vertices.shape[0])
        component_counts = data["component_counts"].astype(np.int64) if "component_counts" in data else presence.sum(axis=1).astype(np.int64)
        vertex_ordering = str(data["vertex_ordering"]) if "vertex_ordering" in data else "unknown"
    if train_type_vocab is not None and type_vocab != train_type_vocab:
        mapping = {name: idx for idx, name in enumerate(train_type_vocab)}
        remapped = np.full_like(type_targets, -1)
        for old_idx, name in enumerate(type_vocab):
            if name not in mapping:
                raise ValueError(f"Target type {name!r} is absent from train type_vocab.")
            remapped[type_targets == old_idx] = mapping[name]
        type_targets = remapped
        type_vocab = list(train_type_vocab)
    if vertices.ndim != 4 or vertices.shape[-1] != 2:
        raise ValueError(f"polygon_vertices_norm must have shape [N,K,V,2], got {vertices.shape}")
    if vertices.shape[1] != 3 or vertices.shape[2] != 4:
        raise ValueError("First polygon inverse route expects max_components=3 and max_vertices=4.")
    if vertex_mask.shape != vertices.shape[:3]:
        raise ValueError("polygon_vertex_mask shape does not match polygon vertices.")
    if presence.shape != vertices.shape[:2]:
        raise ValueError("presence_targets shape does not match polygon vertices.")
    if type_targets.shape != vertices.shape[:2]:
        raise ValueError("type_targets shape does not match polygon vertices.")
    if vertex_ordering != "clockwise_top_left":
        raise ValueError(f"Unsupported vertex_ordering: {vertex_ordering}")
    expected_counts = (presence > 0.5).sum(axis=1).astype(np.int64)
    if not np.array_equal(component_counts, expected_counts):
        raise ValueError("component_counts must match presence_targets per sample.")
    present_vertex_counts = vertex_mask[presence > 0.5].sum(axis=1)
    if present_vertex_counts.size and np.any(present_vertex_counts < 3):
        raise ValueError("Every present polygon component must have at least 3 valid vertices.")
    if np.any(type_targets[presence <= 0.5] != -1):
        raise ValueError("Absent component slots must use type target -1.")
    return {
        "vertices": vertices,
        "vertex_mask": vertex_mask,
        "presence": presence,
        "type_targets": type_targets,
        "type_vocab": type_vocab,
        "sample_indices": sample_indices,
        "component_counts": component_counts,
        "vertex_ordering": vertex_ordering,
    }


def build_tensors(dataset: dict, targets: dict, device: torch.device) -> dict:
    if dataset["signals"].shape[0] != targets["presence"].shape[0]:
        raise ValueError("Dataset and polygon target sample counts do not match.")
    return {
        "signals": torch.from_numpy(dataset["signals"]).to(device),
        "masks": dataset["masks"],
        "x": dataset["x"],
        "y": dataset["y"],
        "vertices": torch.from_numpy(targets["vertices"]).to(device),
        "vertex_mask": torch.from_numpy(targets["vertex_mask"]).to(device),
        "presence": torch.from_numpy(targets["presence"]).to(device),
        "type_targets": torch.from_numpy(targets["type_targets"]).to(device),
        "sample_indices": targets["sample_indices"],
    }


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _grid_scale(tensors: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = np.asarray(tensors["x"], dtype=np.float64)
    y = np.asarray(tensors["y"], dtype=np.float64)
    if x.size < 2 or y.size < 2:
        raise ValueError("x/y grids must each contain at least two points for grid-space polygon losses.")
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    if dx == 0.0 or dy == 0.0:
        raise ValueError("x/y grid spacing must be non-zero for grid-space polygon losses.")
    return torch.tensor([1.0 / abs(dx), 1.0 / abs(dy)], device=device, dtype=dtype)


def _vertices_for_loss(vertices: torch.Tensor, tensors: dict, args) -> torch.Tensor:
    if args.vertex_loss_space == "norm":
        return vertices
    if args.vertex_loss_space == "grid":
        return vertices * _grid_scale(tensors, vertices.device, vertices.dtype)
    raise ValueError(f"Unsupported vertex_loss_space: {args.vertex_loss_space}")


def _derived_centers(vertices: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
    weights = vertex_mask.unsqueeze(-1)
    denom = weights.sum(dim=2).clamp_min(1.0)
    return (vertices * weights).sum(dim=2) / denom


def _derived_boxes(vertices: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
    valid = vertex_mask.unsqueeze(-1) > 0.5
    mins = torch.where(valid, vertices, torch.full_like(vertices, 1.0e6)).amin(dim=2)
    maxs = torch.where(valid, vertices, torch.full_like(vertices, -1.0e6)).amax(dim=2)
    return torch.cat([mins, maxs], dim=-1)


def _polygon_areas_grid(vertices: torch.Tensor, tensors: dict) -> torch.Tensor:
    vertices_grid = vertices * _grid_scale(tensors, vertices.device, vertices.dtype)
    x = vertices_grid[..., 0]
    y = vertices_grid[..., 1]
    x_next = torch.roll(x, shifts=-1, dims=2)
    y_next = torch.roll(y, shifts=-1, dims=2)
    return 0.5 * torch.abs((x * y_next - y * x_next).sum(dim=2))


def _edge_lengths_grid(vertices: torch.Tensor, tensors: dict) -> torch.Tensor:
    vertices_grid = vertices * _grid_scale(tensors, vertices.device, vertices.dtype)
    edge_vectors = torch.roll(vertices_grid, shifts=-1, dims=2) - vertices_grid
    return torch.linalg.norm(edge_vectors, dim=-1)


def _present_quad_mask(presence: torch.Tensor, vertex_mask: torch.Tensor, loss_name: str) -> torch.Tensor:
    quad_mask = presence * (vertex_mask.sum(dim=2) == vertex_mask.shape[2]).to(presence.dtype)
    if presence.sum() > quad_mask.sum():
        raise ValueError(f"{loss_name} currently requires four valid vertices for every present component.")
    return quad_mask


def compute_losses(out: dict, tensors: dict, args) -> tuple[torch.Tensor, dict[str, float]]:
    presence = tensors["presence"]
    type_targets = tensors["type_targets"]
    vertices = tensors["vertices"]
    vertex_mask = tensors["vertex_mask"]
    present = presence > 0.5
    presence_loss = F.binary_cross_entropy_with_logits(out["presence_logits"], presence)
    if present.any():
        type_loss = F.cross_entropy(out["type_logits"][present], type_targets[present])
    else:
        type_loss = out["type_logits"].sum() * 0.0
    valid_vertex = (presence.unsqueeze(-1) * vertex_mask).unsqueeze(-1).expand_as(vertices)
    pred_vertices_for_loss = _vertices_for_loss(out["vertices_norm"], tensors, args)
    true_vertices_for_loss = _vertices_for_loss(vertices, tensors, args)
    vertex_loss = _masked_smooth_l1(pred_vertices_for_loss, true_vertices_for_loss, valid_vertex, args.vertex_smoothl1_beta)
    center_loss = out["vertices_norm"].sum() * 0.0
    if args.lambda_center_aux != 0.0 and present.any():
        pred_center = _derived_centers(out["vertices_norm"], vertex_mask)
        true_center = _derived_centers(vertices, vertex_mask)
        center_loss = _masked_smooth_l1(pred_center, true_center, presence.unsqueeze(-1).expand_as(pred_center), args.vertex_smoothl1_beta)
    box_loss = out["vertices_norm"].sum() * 0.0
    if args.lambda_box_aux != 0.0 and present.any():
        pred_box = _derived_boxes(out["vertices_norm"], vertex_mask)
        true_box = _derived_boxes(vertices, vertex_mask)
        box_loss = _masked_smooth_l1(pred_box, true_box, presence.unsqueeze(-1).expand_as(pred_box), args.vertex_smoothl1_beta)
    area_loss = out["vertices_norm"].sum() * 0.0
    if args.lambda_area_aux != 0.0 and present.any():
        pred_area = _polygon_areas_grid(out["vertices_norm"], tensors)
        true_area = _polygon_areas_grid(vertices, tensors)
        area_loss = _masked_smooth_l1(
            pred_area,
            true_area,
            _present_quad_mask(presence, vertex_mask, "area auxiliary loss"),
            args.vertex_smoothl1_beta,
        )
    edge_loss = out["vertices_norm"].sum() * 0.0
    if args.lambda_edge_aux != 0.0 and present.any():
        pred_edges = _edge_lengths_grid(out["vertices_norm"], tensors)
        true_edges = _edge_lengths_grid(vertices, tensors)
        edge_mask = _present_quad_mask(presence, vertex_mask, "edge auxiliary loss").unsqueeze(-1).expand_as(pred_edges)
        edge_loss = _masked_smooth_l1(pred_edges, true_edges, edge_mask, args.vertex_smoothl1_beta)
    total = (
        args.lambda_presence * presence_loss
        + args.lambda_type * type_loss
        + args.lambda_vertex * vertex_loss
        + args.lambda_center_aux * center_loss
        + args.lambda_box_aux * box_loss
        + args.lambda_area_aux * area_loss
        + args.lambda_edge_aux * edge_loss
    )
    values = {
        "loss": float(total.detach().cpu()),
        "presence_loss": float(presence_loss.detach().cpu()),
        "type_loss": float(type_loss.detach().cpu()),
        "vertex_loss": float(vertex_loss.detach().cpu()),
        "center_aux_loss": float(center_loss.detach().cpu()),
        "box_aux_loss": float(box_loss.detach().cpu()),
        "area_aux_loss": float(area_loss.detach().cpu()),
        "edge_aux_loss": float(edge_loss.detach().cpu()),
        "weighted_presence_loss": float((args.lambda_presence * presence_loss).detach().cpu()),
        "weighted_type_loss": float((args.lambda_type * type_loss).detach().cpu()),
        "weighted_vertex_loss": float((args.lambda_vertex * vertex_loss).detach().cpu()),
        "weighted_center_aux_loss": float((args.lambda_center_aux * center_loss).detach().cpu()),
        "weighted_box_aux_loss": float((args.lambda_box_aux * box_loss).detach().cpu()),
        "weighted_area_aux_loss": float((args.lambda_area_aux * area_loss).detach().cpu()),
        "weighted_edge_aux_loss": float((args.lambda_edge_aux * edge_loss).detach().cpu()),
    }
    return total, values


def _polygon_metrics(out: dict, tensors: dict, args) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    presence_prob = torch.sigmoid(out["presence_logits"]).detach().cpu().numpy()
    pred_presence = (presence_prob >= args.presence_threshold).astype(np.float32)
    pred_vertices = out["vertices_norm"].detach().cpu().numpy().astype(np.float32)
    true_presence = tensors["presence"].detach().cpu().numpy().astype(np.float32)
    true_vertices = tensors["vertices"].detach().cpu().numpy().astype(np.float32)
    true_vertex_mask = tensors["vertex_mask"].detach().cpu().numpy().astype(np.float32)
    pred_vertex_mask = np.ones_like(true_vertex_mask, dtype=np.float32)
    pred_masks = rasterize_polygon_components(pred_vertices, pred_vertex_mask, pred_presence, tensors["x"], tensors["y"])
    ious, dices = mask_iou_dice(pred_masks, tensors["masks"])
    valid_coord = (true_presence[:, :, None, None] * true_vertex_mask[:, :, :, None]) > 0.5
    vertex_mae = float(np.abs(pred_vertices - true_vertices)[valid_coord.repeat(2, axis=-1)].mean()) if valid_coord.any() else 0.0
    pred_types = out["type_logits"].detach().cpu().numpy().argmax(axis=-1)
    true_types = tensors["type_targets"].detach().cpu().numpy()
    present = true_presence > 0.5
    present_type_acc = float((pred_types[present] == true_types[present]).mean()) if present.any() else 1.0
    presence_acc = float((pred_presence == true_presence).mean())
    metrics = {
        "polygon_mask_iou": float(np.mean(ious)),
        "polygon_mask_iou_min": float(np.min(ious)),
        "polygon_dice": float(np.mean(dices)),
        "polygon_dice_min": float(np.min(dices)),
        "presence_acc": presence_acc,
        "present_type_acc": present_type_acc,
        "vertex_mae": vertex_mae,
        "pred_area_mean": float(pred_masks.sum(axis=(1, 2)).mean()),
        "target_area_mean": float((tensors["masks"] > 0.5).sum(axis=(1, 2)).mean()),
    }
    return metrics, pred_masks, pred_presence, pred_vertices


def evaluate(model: PolygonInverseNet, tensors: dict, split: str, args) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(tensors["signals"])
        _loss, loss_values = compute_losses(out, tensors, args)
        metric_values, _pred_masks, _pred_presence, _pred_vertices = _polygon_metrics(out, tensors, args)
    return {"split": split, **loss_values, **metric_values}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_predictions(output_dir: Path, split: str, model: PolygonInverseNet, tensors: dict, targets: dict, args) -> None:
    model.eval()
    with torch.no_grad():
        out = model(tensors["signals"])
        _metrics, pred_masks, pred_presence, pred_vertices = _polygon_metrics(out, tensors, args)
    presence_prob = torch.sigmoid(out["presence_logits"]).detach().cpu().numpy()
    pred_types = out["type_logits"].detach().cpu().numpy().argmax(axis=-1)
    true_presence = tensors["presence"].detach().cpu().numpy()
    true_vertices = tensors["vertices"].detach().cpu().numpy()
    true_types = tensors["type_targets"].detach().cpu().numpy()
    vertex_mask = tensors["vertex_mask"].detach().cpu().numpy()
    component_rows = []
    for row_idx, sample_index in enumerate(targets["sample_indices"]):
        for slot in range(true_presence.shape[1]):
            row = {
                "sample_index": int(sample_index),
                "component_slot": slot,
                "presence_true": float(true_presence[row_idx, slot]),
                "presence_prob": float(presence_prob[row_idx, slot]),
                "presence_pred": float(pred_presence[row_idx, slot]),
                "type_true": int(true_types[row_idx, slot]),
                "type_pred": int(pred_types[row_idx, slot]),
            }
            valid = vertex_mask[row_idx, slot] > 0.5
            if valid.any():
                row["vertex_mae"] = float(np.abs(pred_vertices[row_idx, slot, valid] - true_vertices[row_idx, slot, valid]).mean())
            else:
                row["vertex_mae"] = 0.0
            for vertex_idx in range(pred_vertices.shape[2]):
                row[f"vertex{vertex_idx}_valid"] = float(vertex_mask[row_idx, slot, vertex_idx])
                row[f"pred_x{vertex_idx}"] = float(pred_vertices[row_idx, slot, vertex_idx, 0])
                row[f"pred_y{vertex_idx}"] = float(pred_vertices[row_idx, slot, vertex_idx, 1])
                row[f"true_x{vertex_idx}"] = float(true_vertices[row_idx, slot, vertex_idx, 0])
                row[f"true_y{vertex_idx}"] = float(true_vertices[row_idx, slot, vertex_idx, 1])
            component_rows.append(row)
    write_csv(output_dir / f"{split}_polygon_predictions.csv", component_rows)
    ious, dices = mask_iou_dice(pred_masks, tensors["masks"])
    sample_rows = []
    for row_idx, sample_index in enumerate(targets["sample_indices"]):
        sample_rows.append(
            {
                "sample_index": int(sample_index),
                "polygon_mask_iou": float(ious[row_idx]),
                "polygon_dice": float(dices[row_idx]),
                "target_area": int((tensors["masks"][row_idx] > 0.5).sum()),
                "pred_area": int(pred_masks[row_idx].sum()),
                "true_component_count": int((true_presence[row_idx] > 0.5).sum()),
                "pred_component_count": int((pred_presence[row_idx] > 0.5).sum()),
            }
        )
    write_csv(output_dir / f"{split}_polygon_mask_metrics.csv", sample_rows)


def _append_config(row: dict, args) -> dict:
    config = {
        "steps": args.steps,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "max_components": args.max_components,
        "max_vertices": args.max_vertices,
        "lambda_presence": args.lambda_presence,
        "lambda_type": args.lambda_type,
        "lambda_vertex": args.lambda_vertex,
        "lambda_center_aux": args.lambda_center_aux,
        "lambda_box_aux": args.lambda_box_aux,
        "lambda_area_aux": args.lambda_area_aux,
        "lambda_edge_aux": args.lambda_edge_aux,
        "vertex_loss_space": args.vertex_loss_space,
        "vertex_smoothl1_beta": args.vertex_smoothl1_beta,
        "presence_threshold": args.presence_threshold,
        "seed": args.seed,
    }
    return {**row, **config}


def write_run_summary(output_dir: Path, args, train_row: dict, val_row: dict, test_row: dict, type_vocab: list[str]) -> None:
    lines = [
        "# COMSOL polygon inverse run summary",
        "",
        "## Config",
        "",
        f"- steps: `{args.steps}`",
        f"- lr: `{args.lr}`",
        f"- hidden_dim: `{args.hidden_dim}`",
        f"- latent_dim: `{args.latent_dim}`",
        f"- max_components: `{args.max_components}`",
        f"- max_vertices: `{args.max_vertices}`",
        f"- type_vocab: `{', '.join(type_vocab)}`",
        f"- lambda_presence: `{args.lambda_presence}`",
        f"- lambda_type: `{args.lambda_type}`",
        f"- lambda_vertex: `{args.lambda_vertex}`",
        f"- lambda_center_aux: `{args.lambda_center_aux}`",
        f"- lambda_box_aux: `{args.lambda_box_aux}`",
        f"- lambda_area_aux: `{args.lambda_area_aux}`",
        f"- lambda_edge_aux: `{args.lambda_edge_aux}`",
        f"- vertex_loss_space: `{args.vertex_loss_space}`",
        f"- vertex_smoothl1_beta: `{args.vertex_smoothl1_beta}`",
        f"- export_predictions: `{args.export_predictions}`",
        f"- seed: `{args.seed}`",
        "",
        "## Final Metrics",
        "",
        "| split | polygon_mask_iou | polygon_mask_iou_min | vertex_mae | presence_acc | present_type_acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [train_row, val_row, test_row]:
        lines.append(
            f"| {row['split']} | `{row['polygon_mask_iou']:.6f}` | `{row['polygon_mask_iou_min']:.6f}` | "
            f"`{row['vertex_mae']:.6e}` | `{row['presence_acc']:.6f}` | `{row['present_type_acc']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "No checkpoint or model weights are saved by this runner.",
        ]
    )
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


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
    parser.add_argument("--lambda-vertex", type=float, default=10.0)
    parser.add_argument("--lambda-center-aux", type=float, default=0.0)
    parser.add_argument("--lambda-box-aux", type=float, default=0.0)
    parser.add_argument("--lambda-area-aux", type=float, default=0.0)
    parser.add_argument("--lambda-edge-aux", type=float, default=0.0)
    parser.add_argument("--vertex-loss-space", choices=["norm", "grid"], default="norm")
    parser.add_argument("--vertex-smoothl1-beta", type=float, default=0.01)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--history-interval", type=int, default=100)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    required = [
        args.train_npz,
        args.train_targets,
        args.val_npz,
        args.val_targets,
        args.test_npz,
        args.test_targets,
        args.output_dir,
    ]
    if not all(required):
        parser.print_help()
        return 0
    if args.max_components != 3 or args.max_vertices != 4:
        raise ValueError("S259-S263 polygon inverse runner supports max_components=3 and max_vertices=4 only.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_dataset = load_dataset(args.train_npz)
    val_dataset = load_dataset(args.val_npz)
    test_dataset = load_dataset(args.test_npz)
    train_targets = load_polygon_targets(args.train_targets)
    val_targets = load_polygon_targets(args.val_targets, train_type_vocab=train_targets["type_vocab"])
    test_targets = load_polygon_targets(args.test_targets, train_type_vocab=train_targets["type_vocab"])
    train_tensors = build_tensors(train_dataset, train_targets, device)
    val_tensors = build_tensors(val_dataset, val_targets, device)
    test_tensors = build_tensors(test_dataset, test_targets, device)
    model = PolygonInverseNet(
        signal_len=train_dataset["signals"].shape[1],
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
    train_row = evaluate(model, train_tensors, "train", args)
    val_row = evaluate(model, val_tensors, "val", args)
    test_row = evaluate(model, test_tensors, "test", args)
    train_row = _append_config(train_row, args)
    val_row = _append_config(val_row, args)
    test_row = _append_config(test_row, args)
    write_csv(output_dir / "metrics.csv", [train_row])
    write_csv(output_dir / "eval_metrics.csv", [val_row])
    write_csv(output_dir / "test_metrics.csv", [test_row])
    if args.export_predictions:
        export_predictions(output_dir, "train", model, train_tensors, train_targets, args)
        export_predictions(output_dir, "val", model, val_tensors, val_targets, args)
        export_predictions(output_dir, "test", model, test_tensors, test_targets, args)
    write_run_summary(output_dir, args, train_row, val_row, test_row, train_targets["type_vocab"])
    print(f"Saved polygon inverse metrics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

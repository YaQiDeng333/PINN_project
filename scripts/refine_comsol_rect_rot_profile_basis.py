from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import extract_comsol_rect_rot_profile_basis_from_dense as profile_extract  # noqa: E402
import train_comsol_rect_rot_strong_dense_initializer as dense_init  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = profile_extract.DEFAULT_NPZ
DEFAULT_LABELS = profile_extract.DEFAULT_LABELS
DEFAULT_PROFILES = profile_extract.DEFAULT_SELECTED
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_basis_no_forward_refinement_summary.txt"
DEFAULT_CONFIG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_no_forward_refinement_config_sweep.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_no_forward_refinement_metrics.csv"

K_STATIONS = profile_extract.K_STATIONS
TEMPERATURE_M = profile_extract.TEMPERATURE_M


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-forward COMSOL rect/rot profile-basis refinement control.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--config-sweep", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-dense", type=float, default=3.0e-3)
    parser.add_argument("--refine-lr", type=float, default=3.0e-3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def soft_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pf = prob.reshape(prob.shape[0], -1)
    tf = target.reshape(target.shape[0], -1)
    inter = (pf * tf).sum(dim=1)
    denom = pf.sum(dim=1) + tf.sum(dim=1)
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def profile_mask_torch(
    mask_x: torch.Tensor,
    mask_y: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    angle: torch.Tensor,
    u_stations: torch.Tensor,
    half_widths: torch.Tensor,
    offsets: torch.Tensor,
    temperature: float = TEMPERATURE_M,
) -> torch.Tensor:
    xg, yg = torch.meshgrid(mask_x, mask_y, indexing="xy")
    xg = xg.unsqueeze(0)
    yg = yg.unsqueeze(0)
    dx = xg - center_x.view(-1, 1, 1)
    dy = yg - center_y.view(-1, 1, 1)
    ca = torch.cos(angle).view(-1, 1, 1)
    sa = torch.sin(angle).view(-1, 1, 1)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    n = u.shape[0]
    h = u.shape[1]
    w = u.shape[2]
    u_flat = u.reshape(n, -1).contiguous()
    idx = torch.searchsorted(u_stations, u_flat).clamp(1, K_STATIONS - 1)
    left = idx - 1
    right = idx
    u0 = torch.gather(u_stations, 1, left).reshape(n, h, w)
    u1 = torch.gather(u_stations, 1, right).reshape(n, h, w)
    denom = (u1 - u0).clamp_min(1.0e-9)
    frac = ((u - u0) / denom).clamp(0.0, 1.0)
    hw0 = torch.gather(half_widths, 1, left).reshape(n, h, w)
    hw1 = torch.gather(half_widths, 1, right).reshape(n, h, w)
    off0 = torch.gather(offsets, 1, left).reshape(n, h, w)
    off1 = torch.gather(offsets, 1, right).reshape(n, h, w)
    hw = hw0 * (1.0 - frac) + hw1 * frac
    off = off0 * (1.0 - frac) + off1 * frac
    length_gate = torch.minimum(u - u_stations[:, :1].view(-1, 1, 1), u_stations[:, -1:].view(-1, 1, 1) - u)
    width_gate = hw - torch.abs(v - off)
    logits = torch.minimum(length_gate, width_gate) / temperature
    return torch.sigmoid(logits)


def dense_context(args: argparse.Namespace) -> tuple[dense_init.DenseInitializerBundle, dict[str, np.ndarray]]:
    dense_args = argparse.Namespace(
        npz=args.npz,
        labels=args.labels,
        summary=PROJECT_ROOT / "results/summaries/_profile_basis_refine_internal_dense_summary.txt",
        metrics=PROJECT_ROOT / "results/metrics/_profile_basis_refine_internal_dense_metrics.csv",
        epoch_log=PROJECT_ROOT / "results/metrics/_profile_basis_refine_internal_dense_epoch.csv",
        group_summary=PROJECT_ROOT / "results/metrics/_profile_basis_refine_internal_dense_group.csv",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr_dense,
        cpu=args.cpu,
    )
    bundle, _rows, _epochs = dense_init.train_dense_initializer(dense_args, write_outputs=False)
    pred_by_sample: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        ds = dense_init.DenseMaskDataset(bundle.arrays["split_indices"][split], bundle.arrays)
        pred = dense_init.predict(bundle.model, ds, bundle.device, args.batch_size)
        for order, idx_raw in enumerate(pred["indices"]):
            idx = int(idx_raw)
            pred_by_sample[str(bundle.arrays["sample_ids"][idx])] = pred["prob"][order]
    return bundle, pred_by_sample


def row_params(rows: list[dict[str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "center_x": torch.tensor([to_float(r["center_x"]) for r in rows], dtype=torch.float32, device=device),
        "center_y": torch.tensor([to_float(r["center_y"]) for r in rows], dtype=torch.float32, device=device),
        "angle": torch.tensor([to_float(r["angle_rad"]) for r in rows], dtype=torch.float32, device=device),
        "u": torch.tensor([[to_float(r[f"u_station_{i}"]) for i in range(K_STATIONS)] for r in rows], dtype=torch.float32, device=device),
        "half_width": torch.tensor([[to_float(r[f"half_width_{i}"]) for i in range(K_STATIONS)] for r in rows], dtype=torch.float32, device=device),
        "offset": torch.tensor([[to_float(r[f"center_offset_{i}"]) for i in range(K_STATIONS)] for r in rows], dtype=torch.float32, device=device),
    }


def params_to_masks(params: dict[str, torch.Tensor], mask_x: torch.Tensor, mask_y: torch.Tensor) -> torch.Tensor:
    return profile_mask_torch(
        mask_x,
        mask_y,
        params["center_x"],
        params["center_y"],
        params["angle"],
        params["u"],
        params["half_width"].clamp_min(2.5e-4),
        params["offset"].clamp(-0.004, 0.004),
    )


def optimize_split(
    rows: list[dict[str, str]],
    dense_probs: dict[str, np.ndarray],
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    lambda_smooth: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    init = row_params(rows, device)
    dense_target = torch.tensor(np.stack([dense_probs[r["sample_id"]] for r in rows]), dtype=torch.float32, device=device)
    params = {
        "center_x": torch.nn.Parameter(init["center_x"].clone()),
        "center_y": torch.nn.Parameter(init["center_y"].clone()),
        "angle": torch.nn.Parameter(init["angle"].clone()),
        "u": init["u"].clone(),
        "half_width": torch.nn.Parameter(init["half_width"].clone()),
        "offset": torch.nn.Parameter(init["offset"].clone()),
    }
    opt = torch.optim.Adam([params["center_x"], params["center_y"], params["angle"], params["half_width"], params["offset"]], lr=args.refine_lr)
    mx = torch.tensor(mask_x, dtype=torch.float32, device=device)
    my = torch.tensor(mask_y, dtype=torch.float32, device=device)
    dense_area = dense_target.sum(dim=(1, 2)).clamp_min(1.0)
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        prob = params_to_masks(params, mx, my)
        bce = F.binary_cross_entropy(prob.clamp(1.0e-6, 1.0 - 1.0e-6), dense_target)
        dice = soft_dice(prob, dense_target)
        smooth = torch.mean((params["half_width"][:, 2:] - 2.0 * params["half_width"][:, 1:-1] + params["half_width"][:, :-2]).square())
        area = torch.mean(torch.abs(prob.sum(dim=(1, 2)) - dense_area) / dense_area)
        bounds = torch.relu(2.5e-4 - params["half_width"]).square().mean() + torch.relu(torch.abs(params["offset"]) - 0.004).square().mean()
        loss = bce + dice + lambda_smooth * smooth + 0.05 * area + 0.05 * bounds
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["center_x"].clamp_(float(mask_x.min()), float(mask_x.max()))
            params["center_y"].clamp_(float(mask_y.min()), float(mask_y.max()))
            params["angle"].clamp_(-math.pi / 2.0, math.pi / 2.0)
            params["half_width"].clamp_(2.5e-4, 0.015)
            params["offset"].clamp_(-0.004, 0.004)
    return {k: v.detach().cpu().numpy() for k, v in params.items()}


def metric_row(row: dict[str, str], stage: str, prob: np.ndarray, true_mask: np.ndarray, lambda_smooth: float) -> dict[str, Any]:
    m = profile_extract.metric(prob, true_mask, 0.5)
    return {
        "sample_id": row["sample_id"],
        "source_index": int(to_float(row["source_index"])),
        "split": row["split"],
        "defect_type": row["defect_type"],
        "stage": stage,
        "lambda_smooth": lambda_smooth,
        "iou": m["iou"],
        "dice": m["dice"],
        "area_error": m["area_error"],
        "center_error_px": m["center_error_px"],
        "pred_area": m["pred_area"],
        "true_area": m["true_area"],
    }


def evaluate_rows(
    rows: list[dict[str, str]],
    params_np: dict[str, np.ndarray] | None,
    bundle: dense_init.DenseInitializerBundle,
    lambda_smooth: float,
) -> list[dict[str, Any]]:
    arrays = bundle.arrays
    id_to_idx = {str(sid): i for i, sid in enumerate(arrays["sample_ids"])}
    out: list[dict[str, Any]] = []
    if params_np is None:
        for row in rows:
            prob = profile_extract.rasterize_profile_np(
                arrays["mask_x"],
                arrays["mask_y"],
                to_float(row["center_x"]),
                to_float(row["center_y"]),
                to_float(row["angle_rad"]),
                np.array([to_float(row[f"u_station_{i}"]) for i in range(K_STATIONS)]),
                np.array([to_float(row[f"half_width_{i}"]) for i in range(K_STATIONS)]),
                np.array([to_float(row[f"center_offset_{i}"]) for i in range(K_STATIONS)]),
            )
            out.append(metric_row(row, "pre_refine", prob, arrays["masks"][id_to_idx[row["sample_id"]]], lambda_smooth))
    else:
        mask_x = torch.tensor(arrays["mask_x"], dtype=torch.float32)
        mask_y = torch.tensor(arrays["mask_y"], dtype=torch.float32)
        with torch.no_grad():
            prob = profile_mask_torch(
                mask_x,
                mask_y,
                torch.tensor(params_np["center_x"], dtype=torch.float32),
                torch.tensor(params_np["center_y"], dtype=torch.float32),
                torch.tensor(params_np["angle"], dtype=torch.float32),
                torch.tensor(params_np["u"], dtype=torch.float32),
                torch.tensor(params_np["half_width"], dtype=torch.float32),
                torch.tensor(params_np["offset"], dtype=torch.float32),
            ).cpu().numpy()
        for i, row in enumerate(rows):
            out.append(metric_row(row, "post_refine", prob[i], arrays["masks"][id_to_idx[row["sample_id"]]], lambda_smooth))
    return out


def aggregate(rows: list[dict[str, Any]], split: str, stage: str) -> dict[str, Any]:
    subset = [r for r in rows if r["split"] == split and r["stage"] == stage]
    return {
        "split": split,
        "stage": stage,
        "sample_count": len(subset),
        "iou": safe_mean(subset, "iou"),
        "dice": safe_mean(subset, "dice"),
        "area_error": safe_mean(subset, "area_error"),
        "center_error_px": safe_mean(subset, "center_error_px"),
    }


def config_summary(pre: list[dict[str, Any]], post: list[dict[str, Any]], lambda_smooth: float) -> dict[str, Any]:
    pre_by = {r["sample_id"]: r for r in pre}
    deltas = []
    for row in post:
        before = pre_by[row["sample_id"]]
        deltas.append(
            {
                "delta_iou": row["iou"] - before["iou"],
                "delta_dice": row["dice"] - before["dice"],
                "delta_area_error": row["area_error"] - before["area_error"],
            }
        )
    diou = safe_mean(deltas, "delta_iou")
    ddice = safe_mean(deltas, "delta_dice")
    darea = safe_mean(deltas, "delta_area_error")
    score = diou + ddice - max(0.0, darea)
    return {
        "lambda_smooth": lambda_smooth,
        "delta_iou": diou,
        "delta_dice": ddice,
        "delta_area_error": darea,
        "val_score": score,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    bundle, dense_probs = dense_context(args)
    rows = read_csv(args.profiles)
    selected = [r for r in rows if r.get("selected_method", r.get("method")) == r["method"] or r.get("selected_method", "") == r["method"]]
    if not selected:
        selected = rows
    by_split = {split: [r for r in selected if r["split"] == split] for split in ["train", "val", "test"]}
    config_rows: list[dict[str, Any]] = []
    for lam in [0.01, 0.05, 0.10]:
        pre = evaluate_rows(by_split["val"], None, bundle, lam)
        refined = optimize_split(by_split["val"], dense_probs, bundle.arrays["mask_x"], bundle.arrays["mask_y"], lam, args, device)
        post = evaluate_rows(by_split["val"], refined, bundle, lam)
        config_rows.append(config_summary(pre, post, lam))
    selected_config = max(config_rows, key=lambda r: float(r["val_score"]))
    for r in config_rows:
        r["selected"] = abs(float(r["lambda_smooth"]) - float(selected_config["lambda_smooth"])) < 1.0e-12
    write_csv(args.config_sweep, config_rows)

    metric_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        lam = float(selected_config["lambda_smooth"])
        pre = evaluate_rows(by_split[split], None, bundle, lam)
        refined = optimize_split(by_split[split], dense_probs, bundle.arrays["mask_x"], bundle.arrays["mask_y"], lam, args, device)
        post = evaluate_rows(by_split[split], refined, bundle, lam)
        metric_rows.extend(pre)
        metric_rows.extend(post)
    write_csv(args.metrics, metric_rows)
    stats = {(split, stage): aggregate(metric_rows, split, stage) for split in ["train", "val", "test"] for stage in ["pre_refine", "post_refine"]}
    test_pre = stats[("test", "pre_refine")]
    test_post = stats[("test", "post_refine")]
    passes = (
        test_post["iou"] >= test_pre["iou"] - 0.02
        and test_post["dice"] >= test_pre["dice"] - 0.015
        and test_post["area_error"] <= test_pre["area_error"] + 0.08
    )
    lines = [
        "COMSOL rect/rot profile basis no-forward refinement summary",
        "",
        "Control objective: optimize profile parameters against dense initializer probability with smoothness and area priors only.",
        "No true mask / true geometry is used in optimization. Validation true masks are used only to choose lambda_smooth.",
        f"Selected lambda_smooth from validation: {selected_config['lambda_smooth']}",
        "",
        "Pre/post profile-raster metrics:",
    ]
    for split in ["train", "val", "test"]:
        pre = stats[(split, "pre_refine")]
        post = stats[(split, "post_refine")]
        lines.append(
            f"- {split}: IoU/Dice/area_error {pre['iou']:.4f}/{pre['dice']:.4f}/{pre['area_error']:.4f} -> "
            f"{post['iou']:.4f}/{post['dice']:.4f}/{post['area_error']:.4f}"
        )
    lines.extend(
        [
            "",
            "Reference comparison:",
            "- 20.54 dense initializer test IoU/Dice/area_error = 0.6689 / 0.7994 / 0.1979.",
            "- 20.54 extracted rotated-box geometry test IoU/Dice/area_error = 0.6726 / 0.8017 / 0.1945.",
            "- 20.57 S1-calibrated rect/rot refinement test IoU/Dice/area_error = 0.6492 / 0.7829 / 0.2417.",
            f"- 20.58 no-forward profile-refined test IoU/Dice/area_error = {test_post['iou']:.4f} / {test_post['dice']:.4f} / {test_post['area_error']:.4f}.",
            "",
            f"No-forward profile refinement gate passed: {passes}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passes:
        raise RuntimeError("No-forward profile refinement failed gate; stop before forward refinement.")
    return {"selected_lambda": selected_config["lambda_smooth"], "stats": stats}


def main() -> None:
    result = run(parse_args())
    print(f"Selected no-forward lambda_smooth: {result['selected_lambda']}")


if __name__ == "__main__":
    main()

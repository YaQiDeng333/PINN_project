from __future__ import annotations

import argparse
import csv
import json
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
import refine_comsol_rect_rot_profile_basis as no_forward  # noqa: E402
import train_comsol_rect_rot_perturbation_calibrated_forward_surrogate as perturb  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = profile_extract.DEFAULT_NPZ
DEFAULT_LABELS = profile_extract.DEFAULT_LABELS
DEFAULT_PROFILES = profile_extract.DEFAULT_SELECTED
PERTURB_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_rect_rot_local_perturbation_forward_pack_v1.npz"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_basis_forward_refinement_summary.txt"
DEFAULT_AUDIT = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_basis_failure_audit_summary.txt"
DEFAULT_REP_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_basis_representation_audit_summary.txt"
DEFAULT_CONFIG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_forward_refinement_config_sweep.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_forward_refinement_metrics.csv"
DEFAULT_GROUP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_forward_group_summary.csv"
DEFAULT_FAILURE = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_forward_failure_cases.csv"
DEFAULT_REP_AUDIT = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_representation_audit.csv"
DEFAULT_PREVIEW = PROJECT_ROOT / "results/previews/comsol_rect_rot_profile_basis_refinement"

K_STATIONS = profile_extract.K_STATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cautious forward-consistent profile-basis refinement.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--perturb-npz", type=Path, default=PERTURB_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--representation-summary", type=Path, default=DEFAULT_REP_SUMMARY)
    parser.add_argument("--config-sweep", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--representation-audit", type=Path, default=DEFAULT_REP_AUDIT)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-epochs", type=int, default=200)
    parser.add_argument("--surrogate-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-dense", type=float, default=3.0e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--preview-count", type=int, default=24)
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


def safe_corr(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or y.size < 2 or x.std() <= 1.0e-12 or y.std() <= 1.0e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def perturb_stats(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    split = data["split"].astype(str)
    train_idx = np.where(split == "train")[0]
    geom_rows = [parse_json(x) for x in data["geometry_params"]]
    geom8 = np.array(
        [
            [
                float(g["center_x_m"]),
                float(g["center_y_m"]),
                float(g["width_m"]),
                float(g["length_m"]),
                float(g["depth_m"]),
                float(g["angle_rad"]),
                math.sin(float(g["angle_rad"])),
                math.cos(float(g["angle_rad"])),
            ]
            for g in geom_rows
        ],
        dtype=np.float32,
    )
    mean = geom8[train_idx].mean(axis=0).astype(np.float32)
    std = geom8[train_idx].std(axis=0).astype(np.float32)
    std = np.where(std <= 1.0e-12, 1.0, std).astype(np.float32)
    target = data["delta_bz"].astype(np.float32)
    target_mean = float(target[train_idx].mean())
    target_std = float(target[train_idx].std()) or 1.0
    return {"geom_mean": mean, "geom_std": std, "target_mean": target_mean, "target_std": target_std}


def recover_s1(args: argparse.Namespace, device: torch.device) -> perturb.Bundle:
    perturb.set_seed(42)
    arrays = perturb.load_arrays(args.perturb_npz)
    train_args = argparse.Namespace(epochs=args.surrogate_epochs, batch_size=16, lr=1.0e-3)
    bundle, _ = perturb.train_candidate("S1_perturb_geom_mlp", arrays, train_args, device)
    for p in bundle.model.parameters():
        p.requires_grad_(False)
    bundle.model.eval()
    return bundle


def pilot_observed(args: argparse.Namespace, target_mean: float, target_std: float) -> dict[str, np.ndarray]:
    data = np.load(args.npz, allow_pickle=True)
    return {
        str(sid): ((data["delta_bz"][i].astype(np.float32) - target_mean) / target_std).astype(np.float32)
        for i, sid in enumerate(data["sample_ids"].astype(str))
    }


def profile_equiv_geom(params: dict[str, torch.Tensor]) -> torch.Tensor:
    center_x = params["center_x"]
    center_y = params["center_y"]
    angle = params["angle"]
    u = params["u"]
    half_width = params["half_width"].clamp_min(2.5e-4)
    length_u = (u[:, -1] - u[:, 0]).clamp_min(1.0e-3)
    mean_width_v = (2.0 * half_width.mean(dim=1)).clamp_min(5.0e-4)
    depth = params["depth"]
    return torch.stack([center_x, center_y, length_u, mean_width_v, depth, angle], dim=1)


def type_prob_from_angle(angle: torch.Tensor) -> torch.Tensor:
    angle_deg = torch.abs(angle) * 180.0 / math.pi
    p_rot = torch.sigmoid((angle_deg - 6.0) / 3.0).clamp(0.05, 0.95)
    return torch.stack([1.0 - p_rot, p_rot], dim=1)


def surrogate_input(geom6: torch.Tensor, type_prob: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    angle = geom6[:, 5]
    geom8 = torch.cat([geom6, torch.sin(angle).unsqueeze(1), torch.cos(angle).unsqueeze(1)], dim=1)
    return torch.cat([type_prob, (geom8 - mean) / std], dim=1)


def waveform_loss(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    grad_p = pred[:, :, 1:] - pred[:, :, :-1]
    grad_o = obs[:, :, 1:] - obs[:, :, :-1]
    return F.mse_loss(pred, obs) + 0.2 * F.l1_loss(pred, obs) + 0.1 * F.mse_loss(grad_p, grad_o)


def forward_metrics(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = pred - obs
    mse = np.mean(diff**2, axis=(1, 2))
    denom = np.std(obs, axis=(1, 2))
    denom = np.where(denom <= 1.0e-12, 1.0, denom)
    nrmse = np.sqrt(mse) / denom
    return mse, nrmse


def make_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dense_args = argparse.Namespace(
        npz=args.npz,
        labels=args.labels,
        summary=PROJECT_ROOT / "results/summaries/_profile_basis_forward_internal_dense_summary.txt",
        metrics=PROJECT_ROOT / "results/metrics/_profile_basis_forward_internal_dense_metrics.csv",
        epoch_log=PROJECT_ROOT / "results/metrics/_profile_basis_forward_internal_dense_epoch.csv",
        group_summary=PROJECT_ROOT / "results/metrics/_profile_basis_forward_internal_dense_group.csv",
        seed=args.seed,
        epochs=args.dense_epochs,
        batch_size=args.batch_size,
        lr=args.lr_dense,
        cpu=args.cpu,
    )
    dense_bundle, _dense_rows, _ = no_forward.dense_init.train_dense_initializer(dense_args, write_outputs=False)
    dense_probs: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        ds = no_forward.dense_init.DenseMaskDataset(dense_bundle.arrays["split_indices"][split], dense_bundle.arrays)
        pred = no_forward.dense_init.predict(dense_bundle.model, ds, dense_bundle.device, args.batch_size)
        for order, idx_raw in enumerate(pred["indices"]):
            idx = int(idx_raw)
            dense_probs[str(dense_bundle.arrays["sample_ids"][idx])] = pred["prob"][order]
    stats = perturb_stats(args.perturb_npz)
    s1 = recover_s1(args, device)
    obs = pilot_observed(args, stats["target_mean"], stats["target_std"])
    return {"dense_bundle": dense_bundle, "dense_probs": dense_probs, "stats": stats, "s1": s1, "observed": obs}


def initial_params(rows: list[dict[str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    params = no_forward.row_params(rows, device)
    params["depth"] = torch.tensor([to_float(r["depth_proxy"]) for r in rows], dtype=torch.float32, device=device)
    return params


def clone_params(init: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "center_x": torch.nn.Parameter(init["center_x"].clone()),
        "center_y": torch.nn.Parameter(init["center_y"].clone()),
        "angle": torch.nn.Parameter(init["angle"].clone()),
        "u": init["u"].clone(),
        "half_width": torch.nn.Parameter(init["half_width"].clone()),
        "offset": torch.nn.Parameter(init["offset"].clone()),
        "depth": torch.nn.Parameter(init["depth"].clone()),
    }


def masks_from_params(params: dict[str, torch.Tensor], arrays: dict[str, Any], device: torch.device) -> torch.Tensor:
    mx = torch.tensor(arrays["mask_x"], dtype=torch.float32, device=device)
    my = torch.tensor(arrays["mask_y"], dtype=torch.float32, device=device)
    return no_forward.profile_mask_torch(
        mx,
        my,
        params["center_x"],
        params["center_y"],
        params["angle"],
        params["u"],
        params["half_width"].clamp_min(2.5e-4),
        params["offset"].clamp(-0.004, 0.004),
    )


def optimize_split(
    rows: list[dict[str, str]],
    ctx: dict[str, Any],
    config: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    arrays = ctx["dense_bundle"].arrays
    init = initial_params(rows, device)
    params = clone_params(init)
    dense_target = torch.tensor(np.stack([ctx["dense_probs"][r["sample_id"]] for r in rows]), dtype=torch.float32, device=device)
    observed = torch.tensor(np.stack([ctx["observed"][r["sample_id"]] for r in rows]), dtype=torch.float32, device=device)
    geom_mean = torch.tensor(ctx["stats"]["geom_mean"], dtype=torch.float32, device=device)
    geom_std = torch.tensor(ctx["stats"]["geom_std"], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(
        [params["center_x"], params["center_y"], params["angle"], params["half_width"], params["offset"], params["depth"]],
        lr=float(config["lr"]),
    )
    dense_area = dense_target.sum(dim=(1, 2)).clamp_min(1.0)
    for _ in range(int(config["steps"])):
        opt.zero_grad(set_to_none=True)
        prob = masks_from_params(params, arrays, device)
        bce = F.binary_cross_entropy(prob.clamp(1.0e-6, 1.0 - 1.0e-6), dense_target)
        dice = no_forward.soft_dice(prob, dense_target)
        smooth = torch.mean((params["half_width"][:, 2:] - 2.0 * params["half_width"][:, 1:-1] + params["half_width"][:, :-2]).square())
        area = torch.mean(torch.abs(prob.sum(dim=(1, 2)) - dense_area) / dense_area)
        geom6 = profile_equiv_geom(params)
        type_prob = type_prob_from_angle(params["angle"])
        pred = ctx["s1"].model(surrogate_input(geom6, type_prob, geom_mean, geom_std))
        fwd = waveform_loss(pred, observed)
        loss = bce + dice + float(config["lambda_smooth"]) * smooth + 0.05 * area + float(config["lambda_forward"]) * fwd
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["center_x"].clamp_(float(arrays["mask_x"].min()), float(arrays["mask_x"].max()))
            params["center_y"].clamp_(float(arrays["mask_y"].min()), float(arrays["mask_y"].max()))
            params["angle"].clamp_(-math.pi / 2.0, math.pi / 2.0)
            params["half_width"].clamp_(2.5e-4, 0.015)
            params["offset"].clamp_(-0.004, 0.004)
            params["depth"].clamp_(5.0e-4, 0.003)
    return {k: v.detach().cpu().numpy() for k, v in params.items()}


def evaluate(
    rows: list[dict[str, str]],
    params_np: dict[str, np.ndarray] | None,
    ctx: dict[str, Any],
    stage: str,
    config: dict[str, float],
    device: torch.device,
) -> list[dict[str, Any]]:
    arrays = ctx["dense_bundle"].arrays
    id_to_idx = {str(sid): i for i, sid in enumerate(arrays["sample_ids"])}
    if params_np is None:
        tmp = initial_params(rows, device)
        params_t = {k: v for k, v in tmp.items()}
    else:
        params_t = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in params_np.items()}
    with torch.no_grad():
        prob = masks_from_params(params_t, arrays, device).cpu().numpy()
        geom_mean = torch.tensor(ctx["stats"]["geom_mean"], dtype=torch.float32, device=device)
        geom_std = torch.tensor(ctx["stats"]["geom_std"], dtype=torch.float32, device=device)
        geom6 = profile_equiv_geom(params_t)
        pred = ctx["s1"].model(surrogate_input(geom6, type_prob_from_angle(params_t["angle"]), geom_mean, geom_std)).cpu().numpy()
    obs = np.stack([ctx["observed"][r["sample_id"]] for r in rows])
    f_mse, f_nrmse = forward_metrics(pred, obs)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        m = profile_extract.metric(prob[i], arrays["masks"][id_to_idx[row["sample_id"]]], 0.5)
        out.append(
            {
                "sample_id": row["sample_id"],
                "source_index": int(to_float(row["source_index"])),
                "split": row["split"],
                "defect_type": row["defect_type"],
                "stage": stage,
                "steps": config.get("steps", ""),
                "lr": config.get("lr", ""),
                "lambda_smooth": config.get("lambda_smooth", ""),
                "lambda_forward": config.get("lambda_forward", ""),
                "iou": m["iou"],
                "dice": m["dice"],
                "area_error": m["area_error"],
                "center_error_px": m["center_error_px"],
                "pred_area": m["pred_area"],
                "true_area": m["true_area"],
                "forward_mse": float(f_mse[i]),
                "forward_nrmse": float(f_nrmse[i]),
            }
        )
    return out


def pair_summary(pre: list[dict[str, Any]], post: list[dict[str, Any]], config: dict[str, float]) -> dict[str, Any]:
    before = {r["sample_id"]: r for r in pre}
    deltas = []
    for row in post:
        b = before[row["sample_id"]]
        fwd_red = b["forward_nrmse"] - row["forward_nrmse"]
        diou = row["iou"] - b["iou"]
        ddice = row["dice"] - b["dice"]
        darea = row["area_error"] - b["area_error"]
        deltas.append(
            {
                "delta_iou": diou,
                "delta_dice": ddice,
                "delta_area_error": darea,
                "forward_nrmse_reduction": fwd_red,
                "mismatch": float(fwd_red > 0 and diou < 0 and ddice < 0),
            }
        )
    diou = no_forward.safe_mean(deltas, "delta_iou")
    ddice = no_forward.safe_mean(deltas, "delta_dice")
    darea = no_forward.safe_mean(deltas, "delta_area_error")
    fred = no_forward.safe_mean(deltas, "forward_nrmse_reduction")
    mismatch = no_forward.safe_mean(deltas, "mismatch")
    score = diou + ddice - max(0.0, darea) + 0.10 * fred - 0.10 * mismatch
    return {
        **config,
        "delta_iou": diou,
        "delta_dice": ddice,
        "delta_area_error": darea,
        "forward_nrmse_reduction": fred,
        "mismatch_flag_rate": mismatch,
        "val_score": score,
    }


def aggregate(rows: list[dict[str, Any]], split: str, stage: str, group: str) -> dict[str, Any]:
    subset = [r for r in rows if r["split"] == split and r["stage"] == stage]
    if group != "all":
        subset = [r for r in subset if r["defect_type"] == group]
    return {
        "split": split,
        "stage": stage,
        "group": group,
        "n": len(subset),
        "iou": no_forward.safe_mean(subset, "iou"),
        "dice": no_forward.safe_mean(subset, "dice"),
        "area_error": no_forward.safe_mean(subset, "area_error"),
        "center_error_px": no_forward.safe_mean(subset, "center_error_px"),
        "forward_nrmse": no_forward.safe_mean(subset, "forward_nrmse"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ctx = make_context(args, device)
    rows_all = read_csv(args.profiles)
    rows = [r for r in rows_all if r.get("selected_method", "") == r["method"] or r.get("selected_method", "") == ""]
    by_split = {s: [r for r in rows if r["split"] == s] for s in ["train", "val", "test"]}
    # Stage C selected lambda.
    no_forward_configs = read_csv(PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_no_forward_refinement_config_sweep.csv")
    selected_nf = next((r for r in no_forward_configs if str(r.get("selected", "")).lower() == "true"), no_forward_configs[0])
    lambda_smooth = float(selected_nf["lambda_smooth"])
    config_rows: list[dict[str, Any]] = []
    for lambda_forward in [0.00, 0.02, 0.05, 0.10]:
        for steps in [20, 50]:
            for lr in [0.003, 0.01]:
                config = {"lambda_smooth": lambda_smooth, "lambda_forward": lambda_forward, "steps": steps, "lr": lr}
                pre = evaluate(by_split["val"], None, ctx, "pre_refine", config, device)
                refined = optimize_split(by_split["val"], ctx, config, args, device)
                post = evaluate(by_split["val"], refined, ctx, "post_refine", config, device)
                config_rows.append(pair_summary(pre, post, config))
    viable = [r for r in config_rows if (r["delta_iou"] >= 0 or r["delta_dice"] >= 0) and r["mismatch_flag_rate"] < 0.50]
    selected = max(viable if viable else config_rows, key=lambda r: float(r["val_score"]))
    for r in config_rows:
        r["selected"] = all(abs(float(r[k]) - float(selected[k])) < 1.0e-12 for k in ["lambda_forward", "steps", "lr", "lambda_smooth"])
    write_csv(args.config_sweep, config_rows)
    metric_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        pre = evaluate(by_split[split], None, ctx, "pre_refine", selected, device)
        refined = optimize_split(by_split[split], ctx, selected, args, device)
        post = evaluate(by_split[split], refined, ctx, "post_refine", selected, device)
        metric_rows.extend(pre)
        metric_rows.extend(post)
        before = {r["sample_id"]: r for r in pre}
        for row in post:
            b = before[row["sample_id"]]
            diou = row["iou"] - b["iou"]
            ddice = row["dice"] - b["dice"]
            fred = b["forward_nrmse"] - row["forward_nrmse"]
            failure_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "defect_type": row["defect_type"],
                    "delta_iou": diou,
                    "delta_dice": ddice,
                    "delta_area_error": row["area_error"] - b["area_error"],
                    "forward_nrmse_reduction": fred,
                    "mismatch_flag": int(fred > 0 and diou < 0 and ddice < 0),
                }
            )
    write_csv(args.metrics, metric_rows)
    group_rows = [
        aggregate(metric_rows, split, stage, group)
        for split in ["train", "val", "test"]
        for stage in ["pre_refine", "post_refine"]
        for group in ["all", "rectangular_notch", "rotated_rect"]
    ]
    write_csv(args.group_summary, group_rows)
    write_csv(args.failure_cases, sorted(failure_rows, key=lambda r: (r["mismatch_flag"], -abs(r["delta_iou"])), reverse=True)[:100])
    rep_rows = []
    for split in ["train", "val", "test"]:
        sub = [r for r in failure_rows if r["split"] == split]
        rep_rows.append(
            {
                "split": split,
                "residual_reduction_vs_delta_iou_corr": safe_corr([r["forward_nrmse_reduction"] for r in sub], [r["delta_iou"] for r in sub]),
                "residual_reduction_vs_delta_dice_corr": safe_corr([r["forward_nrmse_reduction"] for r in sub], [r["delta_dice"] for r in sub]),
                "mismatch_flag_rate": no_forward.safe_mean(sub, "mismatch_flag"),
                "mean_delta_iou": no_forward.safe_mean(sub, "delta_iou"),
                "mean_delta_dice": no_forward.safe_mean(sub, "delta_dice"),
                "mean_forward_nrmse_reduction": no_forward.safe_mean(sub, "forward_nrmse_reduction"),
            }
        )
    write_csv(args.representation_audit, rep_rows)
    return {"selected": selected, "metric_rows": metric_rows, "group_rows": group_rows, "failure_rows": failure_rows, "rep_rows": rep_rows}


def get_group(group_rows: list[dict[str, Any]], split: str, stage: str, group: str = "all") -> dict[str, Any]:
    return next(r for r in group_rows if r["split"] == split and r["stage"] == stage and r["group"] == group)


def profile_rasterizer_consistency(args: argparse.Namespace, metric_rows: list[dict[str, Any]]) -> dict[str, float]:
    profile_rows = read_csv(args.profiles)
    selected_profile = {
        r["sample_id"]: r
        for r in profile_rows
        if r.get("selected_method", "") == r["method"] or r.get("selected_method", "") == ""
    }
    pre_rows = [r for r in metric_rows if r["stage"] == "pre_refine"]
    diffs_iou: list[float] = []
    diffs_dice: list[float] = []
    diffs_area: list[float] = []
    for row in pre_rows:
        prof = selected_profile.get(row["sample_id"])
        if not prof:
            continue
        diffs_iou.append(abs(float(row["iou"]) - to_float(prof["profile_iou"])))
        diffs_dice.append(abs(float(row["dice"]) - to_float(prof["profile_dice"])))
        diffs_area.append(abs(float(row["area_error"]) - to_float(prof["profile_area_error"])))
    return {
        "mean_abs_iou_diff": float(np.mean(diffs_iou)) if diffs_iou else math.nan,
        "mean_abs_dice_diff": float(np.mean(diffs_dice)) if diffs_dice else math.nan,
        "mean_abs_area_error_diff": float(np.mean(diffs_area)) if diffs_area else math.nan,
    }


def write_summaries(args: argparse.Namespace, result: dict[str, Any]) -> None:
    selected = result["selected"]
    group_rows = result["group_rows"]
    pre_test = get_group(group_rows, "test", "pre_refine")
    post_test = get_group(group_rows, "test", "post_refine")
    pre_val = get_group(group_rows, "val", "pre_refine")
    post_val = get_group(group_rows, "val", "post_refine")
    pre_train = get_group(group_rows, "train", "pre_refine")
    post_train = get_group(group_rows, "train", "post_refine")
    rep_test = next(r for r in result["rep_rows"] if r["split"] == "test")
    forward_helped = selected["lambda_forward"] > 0 and post_test["iou"] >= pre_test["iou"] and post_test["dice"] >= pre_test["dice"]
    consistency = profile_rasterizer_consistency(args, result["metric_rows"])
    lines = [
        "COMSOL rect/rot profile basis forward refinement summary",
        "",
        "Forward is used cautiously by mapping profile parameters to an approximate S1-compatible geometry summary.",
        "This mapping is lossy: width = profile length, length = 2 * mean half-width, depth = train-only global physical-depth proxy, type probability is angle heuristic.",
        "True mask / true geometry are not used in optimization.",
        f"Selected config from validation: lambda_forward={selected['lambda_forward']}, steps={selected['steps']}, lr={selected['lr']}, lambda_smooth={selected['lambda_smooth']}",
        "",
        f"train pre/post IoU/Dice/area_error/forward_nrmse: {pre_train['iou']:.4f}/{pre_train['dice']:.4f}/{pre_train['area_error']:.4f}/{pre_train['forward_nrmse']:.4f} -> {post_train['iou']:.4f}/{post_train['dice']:.4f}/{post_train['area_error']:.4f}/{post_train['forward_nrmse']:.4f}",
        f"val pre/post IoU/Dice/area_error/forward_nrmse: {pre_val['iou']:.4f}/{pre_val['dice']:.4f}/{pre_val['area_error']:.4f}/{pre_val['forward_nrmse']:.4f} -> {post_val['iou']:.4f}/{post_val['dice']:.4f}/{post_val['area_error']:.4f}/{post_val['forward_nrmse']:.4f}",
        f"test pre/post IoU/Dice/area_error/forward_nrmse: {pre_test['iou']:.4f}/{pre_test['dice']:.4f}/{pre_test['area_error']:.4f}/{pre_test['forward_nrmse']:.4f} -> {post_test['iou']:.4f}/{post_test['dice']:.4f}/{post_test['area_error']:.4f}/{post_test['forward_nrmse']:.4f}",
        "",
        f"Forward helped without mask degradation: {forward_helped}",
        f"Test residual-reduction vs IoU/Dice delta correlation: {rep_test['residual_reduction_vs_delta_iou_corr']:.4f} / {rep_test['residual_reduction_vs_delta_dice_corr']:.4f}",
        f"Test mismatch rate: {rep_test['mismatch_flag_rate']:.4f}",
        "",
        "Reference comparison:",
        "- 20.54 dense initializer test IoU/Dice/area_error = 0.6689 / 0.7994 / 0.1979.",
        "- 20.54 extracted rotated-box geometry test IoU/Dice/area_error = 0.6726 / 0.8017 / 0.1945.",
        "- 20.57 S1-calibrated rect/rot refinement test IoU/Dice/area_error = 0.6492 / 0.7829 / 0.2417.",
        f"- 20.58 profile forward-refined test IoU/Dice/area_error = {post_test['iou']:.4f} / {post_test['dice']:.4f} / {post_test['area_error']:.4f}.",
        "",
        f"NumPy/Torch profile rasterizer consistency mean abs IoU/Dice/area_error diffs: {consistency['mean_abs_iou_diff']:.6f} / {consistency['mean_abs_dice_diff']:.6f} / {consistency['mean_abs_area_error_diff']:.6f}.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit_lines = [
        "COMSOL rect/rot profile basis failure audit summary",
        "",
        f"Selected lambda_forward: {selected['lambda_forward']}",
        f"Forward helped without mask degradation: {forward_helped}",
        f"20.58 profile forward-refined test IoU/Dice vs 20.57 rect/rot refined: {post_test['iou']:.4f}/{post_test['dice']:.4f} vs 0.6492/0.7829.",
        f"20.58 profile forward-refined test IoU/Dice vs 20.54 extracted rotated-box geometry: {post_test['iou']:.4f}/{post_test['dice']:.4f} vs 0.6726/0.8017.",
        "If lambda_forward=0 wins, current S1 surrogate is not profile-compatible enough.",
        "If lambda_forward>0 wins but mismatch remains high, surrogate mismatch persists in profile space.",
    ]
    args.audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    rep_lines = [
        "COMSOL rect/rot profile basis representation audit summary",
        "",
        "1. Profile basis is more flexible than a single rotated box, and no-forward refinement can recover test IoU/Dice near the dense initializer while remaining low-dimensional.",
        f"2. Test profile forward-refined IoU/Dice: {post_test['iou']:.4f} / {post_test['dice']:.4f}.",
        f"3. Forward residual helped in profile space: {forward_helped}.",
        f"4. Compared with 20.57 rect/rot refinement, profile basis improves test IoU/Dice by {post_test['iou'] - 0.6492:+.4f} / {post_test['dice'] - 0.7829:+.4f}.",
        f"5. Compared with 20.54 extracted rotated-box geometry, profile basis changes test IoU/Dice by {post_test['iou'] - 0.6726:+.4f} / {post_test['dice'] - 0.8017:+.4f}.",
        "6. Remaining failure is attributed to profile-compatible surrogate mismatch if forward does not win, otherwise to area/roughness trade-off.",
    ]
    args.representation_summary.write_text("\n".join(rep_lines) + "\n", encoding="utf-8")


def maybe_preview(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if args.preview_count <= 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = result["metric_rows"]
    by = {(r["sample_id"], r["stage"]): r for r in rows}
    cases = [r for r in result["failure_rows"] if r["split"] in {"val", "test"}]
    cases = sorted(cases, key=lambda r: (r["mismatch_flag"], abs(r["delta_iou"])), reverse=True)[: args.preview_count]
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases):
        pre = by[(case["sample_id"], "pre_refine")]
        post = by[(case["sample_id"], "post_refine")]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["pre IoU", "post IoU", "pre Dice", "post Dice"], [pre["iou"], post["iou"], pre["dice"], post["dice"]])
        ax.set_ylim(0, 1)
        ax.set_title(f"{case['sample_id']} fwd_red={case['forward_nrmse_reduction']:.3f}")
        fig.tight_layout()
        fig.savefig(args.preview_dir / f"{i:02d}_{case['sample_id']}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    result = run(args)
    write_summaries(args, result)
    maybe_preview(args, result)
    selected = result["selected"]
    print(
        f"Selected forward config: lambda_forward={selected['lambda_forward']} "
        f"steps={selected['steps']} lr={selected['lr']}"
    )


if __name__ == "__main__":
    main()

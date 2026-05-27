#!/usr/bin/env python
"""20.92 liftoff-aware true-3D RBC training gate.

Trains liftoff-augmented candidates on comsol_true_3d_rbc_liftoff_aug_pack_v1.
No COMSOL, data/NPZ writes, checkpoints, or baseline updates are performed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import evaluate_true_3d_rbc_liftoff_baseline as baseline_eval
import load_true_3d_rbc_liftoff_aug_dataset as liftoff
import load_true_3d_rbc_pilot_dataset as pilot
import train_true_3d_rbc_neural_parameter_gate as gate
from run_true_3d_rbc_formal_benchmark_20_77_candidate import add_profile_error_rows


ROOT = liftoff.ROOT
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_training_seed_summary.csv"
METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_training_metrics.csv"
BY_LIFTOFF = ROOT / "results/metrics/true_3d_rbc_liftoff_training_by_liftoff.csv"
VS_BASELINE = ROOT / "results/metrics/true_3d_rbc_liftoff_training_vs_baseline.csv"

BASELINE_PROFILE_RMSE = 0.000387737
BASELINE_DICE = 0.847727


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    uses_sensor_z: bool
    calibrated_input: bool = False


class SensorZConditionedRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.z_head = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 8), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(64 * 8 + 8, 96),
            nn.GELU(),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 6),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x).flatten(1)
        z_latent = self.z_head(z)
        return self.head(torch.cat([enc, z_latent], dim=1))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def candidates() -> list[Candidate]:
    return [
        Candidate("C1_unconditioned_liftoff_aug", "delta_b only; train on mixed liftoff rows", False),
        Candidate("C2_sensor_z_conditioned", "delta_b plus train-normalized scalar sensor_z_m fused into latent", True),
        Candidate("C3_calibrated_input_conditioned_skipped", "skipped: calibration remains diagnostic and would change input protocol", True, True),
    ]


def normalized_inputs(dataset: Any, stats: dict[str, np.ndarray], candidate: Candidate) -> tuple[np.ndarray, np.ndarray]:
    x = liftoff.normalize_x(dataset, stats)
    z = liftoff.normalize_sensor_z(dataset, stats)
    if candidate.calibrated_input:
        raise RuntimeError("C3 calibrated input is intentionally skipped in 20.92 to avoid promoting 20.89 calibration into the training protocol.")
    return x, z


def model_for(candidate: Candidate) -> nn.Module:
    return SensorZConditionedRegressor() if candidate.uses_sensor_z else gate.RBCConvRegressor()


def predict_norm(model: nn.Module, x: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(x, dtype=torch.float32)
        if isinstance(model, SensorZConditionedRegressor):
            if z is None:
                raise ValueError("sensor_z input required for conditioned model")
            pred = model(xb, torch.as_tensor(z, dtype=torch.float32))
        else:
            pred = model(xb)
    return pred.cpu().numpy().astype(np.float32)


def selection_components(y_true_norm: np.ndarray, y_pred_norm: np.ndarray) -> dict[str, float]:
    return gate.selection_components(y_true_norm, y_pred_norm)


def train_one(
    candidate: Candidate,
    seed: int,
    dataset: Any,
    stats: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_seed(seed)
    splits = liftoff.split_indices(dataset)
    x_norm, z_norm = normalized_inputs(dataset, stats, candidate)
    y_norm = liftoff.normalize_y(dataset, stats)
    train_idx = splits["train"]
    if candidate.uses_sensor_z:
        train_ds = TensorDataset(
            torch.as_tensor(x_norm[train_idx], dtype=torch.float32),
            torch.as_tensor(z_norm[train_idx], dtype=torch.float32),
            torch.as_tensor(y_norm[train_idx], dtype=torch.float32),
        )
    else:
        train_ds = TensorDataset(
            torch.as_tensor(x_norm[train_idx], dtype=torch.float32),
            torch.as_tensor(y_norm[train_idx], dtype=torch.float32),
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = model_for(candidate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_score = math.inf
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if candidate.uses_sensor_z:
                xb, zb, yb = batch
                pred = model(xb, zb)
            else:
                xb, yb = batch
                pred = model(xb)
            loss = gate.weighted_smooth_l1(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        pred_train = predict_norm(model, x_norm[splits["train"]], z_norm[splits["train"]] if candidate.uses_sensor_z else None)
        pred_val = predict_norm(model, x_norm[splits["val"]], z_norm[splits["val"]] if candidate.uses_sensor_z else None)
        train_comp = selection_components(y_norm[splits["train"]], pred_train)
        val_comp = selection_components(y_norm[splits["val"]], pred_val)
        val_score = gate.selection_metric(val_comp)
        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "candidate": candidate.name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_normalized_param_mae": train_comp["normalized_param_mae"],
                "val_normalized_param_mae": val_comp["normalized_param_mae"],
                "train_dimension_mae_norm": train_comp["dimension_mae_norm"],
                "val_dimension_mae_norm": val_comp["dimension_mae_norm"],
                "train_curvature_mae_norm": train_comp["curvature_mae_norm"],
                "val_curvature_mae_norm": val_comp["curvature_mae_norm"],
                "val_selection_metric": val_score,
            }
        )
    if best_state is None:
        raise RuntimeError(f"no validation checkpoint selected for {candidate.name} seed={seed}")
    model.load_state_dict(best_state)
    pred_norm_all = predict_norm(model, x_norm, z_norm if candidate.uses_sensor_z else None)
    pred_raw_all = liftoff.denormalize_y(pred_norm_all, stats)
    return {
        "candidate": candidate.name,
        "seed": seed,
        "model": model,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "pred_norm": pred_norm_all,
        "pred_raw": pred_raw_all,
        "epoch_rows": epoch_rows,
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None, ""}]
    return float(np.mean(values)) if values else math.nan


def aggregate_profile_rows(rows: list[dict[str, Any]], candidate: str, seed: int | str, split: str, subset_name: str, subset: list[dict[str, Any]], selected: bool, best_epoch: Any = "", best_val_score: Any = "") -> dict[str, Any]:
    return {
        "candidate": candidate,
        "seed": seed,
        "selected_seed": selected,
        "best_epoch": best_epoch,
        "best_val_selection_metric": best_val_score,
        "split": split,
        "liftoff_subset": subset_name,
        "sample_count": len(subset),
        "normalized_param_mae": mean(subset, "normalized_param_mae_mean"),
        "dimension_mae_norm": mean(subset, "dimension_param_mae_norm"),
        "curvature_mae_norm": mean(subset, "curvature_param_mae_norm"),
        "L_mae_mm": mean(subset, "L_mae_mm"),
        "W_mae_mm": mean(subset, "W_mae_mm"),
        "D_mae_mm": mean(subset, "D_mae_mm"),
        "wLD_abs_error": mean(subset, "wLD_abs_error"),
        "wWD_abs_error": mean(subset, "wWD_abs_error"),
        "wLW_abs_error": mean(subset, "wLW_abs_error"),
        "wMAE_auxiliary": mean(subset, "curvature_mae_mean"),
        "projected_mask_iou": mean(subset, "projected_mask_iou"),
        "projected_mask_dice": mean(subset, "projected_mask_dice"),
        "profile_depth_rmse_m": mean(subset, "profile_depth_rmse_m"),
        "er_like_profile_error": mean(subset, "er_like_profile_error"),
        "max_depth_error_m": mean(subset, "max_depth_error_m"),
        "volume_proxy_rel_error": mean(subset, "volume_proxy_rel_error"),
    }


def evaluate_run(dataset: Any, stats: dict[str, np.ndarray], result: dict[str, Any], selected: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = pilot.evaluate_param_predictions(dataset, result["pred_raw"], stats={"y_mean": stats["y_mean"], "y_std": stats["y_std"]})
    profile_rows = add_profile_error_rows(dataset, result["pred_raw"], metric_rows)
    for idx, row in enumerate(profile_rows):
        row["candidate"] = result["candidate"]
        row["seed"] = result["seed"]
        row["selected_seed"] = selected
        row["best_epoch"] = result["best_epoch"]
        row["best_val_selection_metric"] = result["best_val_score"]
        row["base_sample_id"] = str(dataset.base_sample_ids[idx])
        row["sensor_z_m"] = float(dataset.sensor_z_m[idx])
        row["variant_name"] = str(dataset.variant_name[idx])

    aggregate_rows: list[dict[str, Any]] = []
    by_liftoff: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in profile_rows if row["split"] == split]
        aggregate_rows.append(aggregate_profile_rows(profile_rows, result["candidate"], result["seed"], split, "all_liftoff", split_rows, selected, result["best_epoch"], result["best_val_score"]))
        aggregate_rows.append(aggregate_profile_rows(profile_rows, result["candidate"], result["seed"], split, "nominal_0p008", [row for row in split_rows if round(float(row["sensor_z_m"]), 3) == 0.008], selected, result["best_epoch"], result["best_val_score"]))
        aggregate_rows.append(aggregate_profile_rows(profile_rows, result["candidate"], result["seed"], split, "non_nominal", [row for row in split_rows if round(float(row["sensor_z_m"]), 3) != 0.008], selected, result["best_epoch"], result["best_val_score"]))
        for z in sorted({round(float(row["sensor_z_m"]), 3) for row in split_rows}):
            by_liftoff.append(aggregate_profile_rows(profile_rows, result["candidate"], result["seed"], split, f"sensor_z_{z:.3f}", [row for row in split_rows if round(float(row["sensor_z_m"]), 3) == z], selected, result["best_epoch"], result["best_val_score"]))
    return aggregate_rows, by_liftoff


def select_best(results: list[dict[str, Any]]) -> tuple[str, int]:
    scores: list[tuple[float, str, int]] = []
    for result in results:
        scores.append((float(result["best_val_score"]), str(result["candidate"]), int(result["seed"])))
    if not scores:
        raise RuntimeError("no trained liftoff candidate results")
    _, candidate, seed = sorted(scores, key=lambda item: item[0])[0]
    return candidate, seed


def seed_summary_rows(metrics: list[dict[str, Any]], selected_candidate: str, selected_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted({str(row["candidate"]) for row in metrics}):
        for seed in sorted({str(row["seed"]) for row in metrics if row["candidate"] == candidate}):
            subset = [row for row in metrics if row["candidate"] == candidate and str(row["seed"]) == seed]
            test_all = next((row for row in subset if row["split"] == "test" and row["liftoff_subset"] == "all_liftoff"), None)
            test_nom = next((row for row in subset if row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008"), None)
            test_non = next((row for row in subset if row["split"] == "test" and row["liftoff_subset"] == "non_nominal"), None)
            val_all = next((row for row in subset if row["split"] == "val" and row["liftoff_subset"] == "all_liftoff"), None)
            if not test_all:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "selected_robustness_candidate": candidate == selected_candidate and str(seed) == str(selected_seed),
                    "best_epoch": test_all.get("best_epoch", ""),
                    "best_val_selection_metric": test_all.get("best_val_selection_metric", ""),
                    "val_all_profile_depth_rmse_m": val_all.get("profile_depth_rmse_m", "") if val_all else "",
                    "test_all_profile_depth_rmse_m": test_all.get("profile_depth_rmse_m", ""),
                    "test_nominal_profile_depth_rmse_m": test_nom.get("profile_depth_rmse_m", "") if test_nom else "",
                    "test_non_nominal_profile_depth_rmse_m": test_non.get("profile_depth_rmse_m", "") if test_non else "",
                    "test_non_nominal_L_mae_mm": test_non.get("L_mae_mm", "") if test_non else "",
                    "test_non_nominal_W_mae_mm": test_non.get("W_mae_mm", "") if test_non else "",
                    "test_non_nominal_D_mae_mm": test_non.get("D_mae_mm", "") if test_non else "",
                    "test_non_nominal_dice": test_non.get("projected_mask_dice", "") if test_non else "",
                    "test_non_nominal_wMAE_auxiliary": test_non.get("wMAE_auxiliary", "") if test_non else "",
                }
            )
    rows.append(
        {
            "candidate": "C3_calibrated_input_conditioned",
            "seed": "skipped",
            "selected_robustness_candidate": False,
            "best_epoch": "",
            "best_val_selection_metric": "",
            "test_all_profile_depth_rmse_m": "",
            "notes": "Skipped deliberately: 20.89 calibration remains diagnostic caveat and is not promoted into 20.92 training input protocol.",
        }
    )
    return rows


def as_training_metric_rows(rows: list[dict[str, str]], selected: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new = dict(row)
        new["candidate"] = "C0_reference_20_85_baseline"
        new["seed"] = 42
        new["selected_seed"] = selected
        new["best_epoch"] = ""
        new["best_val_selection_metric"] = ""
        out.append(new)
    return out


def vs_baseline_rows(metrics: list[dict[str, Any]], selected_candidate: str, selected_seed: int, baseline_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    selected = [
        row
        for row in metrics
        if row["candidate"] == selected_candidate and str(row["seed"]) == str(selected_seed) and row["split"] == "test"
    ]
    baseline = [row for row in baseline_rows if row.get("split") == "test"]
    for subset in ("all_liftoff", "nominal_0p008", "non_nominal"):
        cur = next((row for row in selected if row["liftoff_subset"] == subset), None)
        ref = next((row for row in baseline if row.get("liftoff_subset") == subset), None)
        if not cur or not ref:
            continue
        for metric in [
            "profile_depth_rmse_m",
            "er_like_profile_error",
            "normalized_param_mae",
            "L_mae_mm",
            "W_mae_mm",
            "D_mae_mm",
            "wMAE_auxiliary",
            "projected_mask_iou",
            "projected_mask_dice",
        ]:
            ref_value = float(ref[metric])
            cur_value = float(cur[metric])
            lower_better = metric not in {"projected_mask_iou", "projected_mask_dice"}
            out.append(
                {
                    "liftoff_subset": subset,
                    "metric": metric,
                    "reference_candidate": "C0_reference_20_85_baseline",
                    "current_candidate": selected_candidate,
                    "current_seed": selected_seed,
                    "reference_value": ref_value,
                    "current_value": cur_value,
                    "delta": cur_value - ref_value,
                    "relative_change_pct": 0.0 if abs(ref_value) < 1.0e-20 else 100.0 * (cur_value - ref_value) / ref_value,
                    "improved": cur_value < ref_value if lower_better else cur_value > ref_value,
                }
            )
    c1 = next((row for row in metrics if row["candidate"] == "C1_unconditioned_liftoff_aug" and row["split"] == "test" and row["liftoff_subset"] == "non_nominal" and str(row["seed"]) == str(selected_seed)), None)
    c2 = next((row for row in metrics if row["candidate"] == "C2_sensor_z_conditioned" and row["split"] == "test" and row["liftoff_subset"] == "non_nominal" and str(row["seed"]) == str(selected_seed)), None)
    if c1 and c2:
        out.append(
            {
                "liftoff_subset": "non_nominal",
                "metric": "C2_vs_C1_profile_depth_rmse_m_same_seed",
                "reference_candidate": "C1_unconditioned_liftoff_aug",
                "current_candidate": "C2_sensor_z_conditioned",
                "current_seed": selected_seed,
                "reference_value": float(c1["profile_depth_rmse_m"]),
                "current_value": float(c2["profile_depth_rmse_m"]),
                "delta": float(c2["profile_depth_rmse_m"]) - float(c1["profile_depth_rmse_m"]),
                "relative_change_pct": 100.0 * (float(c2["profile_depth_rmse_m"]) - float(c1["profile_depth_rmse_m"])) / max(abs(float(c1["profile_depth_rmse_m"])), 1.0e-20),
                "improved": float(c2["profile_depth_rmse_m"]) < float(c1["profile_depth_rmse_m"]),
            }
        )
    return out


def write_summary(metrics: list[dict[str, Any]], selected_candidate: str, selected_seed: int, vs_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    selected_test = [
        row
        for row in metrics
        if row["candidate"] == selected_candidate and str(row["seed"]) == str(selected_seed) and row["split"] == "test"
    ]
    test_all = next(row for row in selected_test if row["liftoff_subset"] == "all_liftoff")
    test_nom = next(row for row in selected_test if row["liftoff_subset"] == "nominal_0p008")
    test_non = next(row for row in selected_test if row["liftoff_subset"] == "non_nominal")
    profile_improvement = next((row for row in vs_rows if row["liftoff_subset"] == "non_nominal" and row["metric"] == "profile_depth_rmse_m"), None)
    dice_change = next((row for row in vs_rows if row["liftoff_subset"] == "non_nominal" and row["metric"] == "projected_mask_dice"), None)
    lines = [
        "20.92 liftoff-aware true 3D RBC training gate",
        "",
        "Scope:",
        "- Dataset: comsol_true_3d_rbc_liftoff_aug_pack_v1 loaded via registry/manifest",
        "- Split: by base_sample_id, train/val/test bases=32/8/8 and rows=128/32/32",
        "- Inputs: C1 uses delta_b only; C2 uses delta_b plus train-normalized sensor_z_m scalar",
        "- C3 skipped because calibration remains a diagnostic caveat, not a baseline/training protocol",
        "- Training: seeds 42/123/2026; validation-only selection; test final only; no checkpoint saved",
        "- COMSOL/data/NPZ/baseline update: false",
        "",
        f"Selected robustness candidate: {selected_candidate}, seed={selected_seed}",
        f"epochs: {args.epochs}",
        f"test_all_profile_depth_rmse_m: {float(test_all['profile_depth_rmse_m']):.9f}",
        f"test_nominal_0p008_profile_depth_rmse_m: {float(test_nom['profile_depth_rmse_m']):.9f}",
        f"test_non_nominal_profile_depth_rmse_m: {float(test_non['profile_depth_rmse_m']):.9f}",
        f"test_non_nominal_er_like_profile_error: {float(test_non['er_like_profile_error']):.6f}",
        f"test_non_nominal_LWD_MAE_mm: {float(test_non['L_mae_mm']):.3f} / {float(test_non['W_mae_mm']):.3f} / {float(test_non['D_mae_mm']):.3f}",
        f"test_non_nominal_projected_mask_IoU_Dice: {float(test_non['projected_mask_iou']):.6f} / {float(test_non['projected_mask_dice']):.6f}",
        f"test_non_nominal_wMAE_auxiliary: {float(test_non['wMAE_auxiliary']):.6f}",
        "",
        f"non_nominal_profile_rmse_change_vs_C0_pct: {float(profile_improvement['relative_change_pct']):.3f}" if profile_improvement else "non_nominal_profile_rmse_change_vs_C0_pct: unavailable",
        f"non_nominal_dice_change_vs_C0: {float(dice_change['delta']):.6f}" if dice_change else "non_nominal_dice_change_vs_C0: unavailable",
        "",
        "Interpretation boundary: this is a robustness candidate gate. CURRENT_BASELINE remains the 20.85 nominal-profile baseline until a separate baseline transition is approved.",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=liftoff.DATASET_ID)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summarize-existing", action="store_true", help="reuse existing C1/C2 metrics and merge C0 baseline without retraining")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = liftoff.load_liftoff_dataset(args.dataset_id)
    if liftoff.base_split_leakage(dataset):
        raise RuntimeError(f"base split leakage detected: {liftoff.base_split_leakage(dataset)}")
    if not liftoff.paired_liftoff_complete(dataset):
        raise RuntimeError("paired liftoff completeness failed")
    stats = liftoff.train_normalization(dataset)

    if not baseline_eval.METRICS.exists():
        baseline_eval.run(args.dataset_id)
    baseline_rows = read_csv(baseline_eval.METRICS)
    baseline_by_liftoff_rows = read_csv(baseline_eval.BY_LIFTOFF)

    if args.summarize_existing:
        existing_metrics = [row for row in read_csv(METRICS) if row.get("candidate") != "C0_reference_20_85_baseline"]
        existing_by_liftoff = [row for row in read_csv(BY_LIFTOFF) if row.get("candidate") != "C0_reference_20_85_baseline"]
        trained_candidates = [row for row in existing_metrics if str(row.get("candidate", "")).startswith(("C1_", "C2_")) and row.get("split") == "val" and row.get("liftoff_subset") == "all_liftoff"]
        if not trained_candidates:
            raise RuntimeError("summarize-existing requested but no C1/C2 validation metrics exist")
        selected_row = min(trained_candidates, key=lambda row: float(row["best_val_selection_metric"]))
        selected_candidate = selected_row["candidate"]
        selected_seed = int(selected_row["seed"])
        for row in existing_metrics + existing_by_liftoff:
            row["selected_seed"] = row.get("candidate") == selected_candidate and str(row.get("seed")) == str(selected_seed)
        metric_rows = as_training_metric_rows(baseline_rows) + existing_metrics
        by_liftoff_rows = as_training_metric_rows(baseline_by_liftoff_rows) + existing_by_liftoff
        seed_rows = seed_summary_rows(metric_rows, selected_candidate, selected_seed)
        vs_rows = vs_baseline_rows(metric_rows, selected_candidate, selected_seed, baseline_rows)
        write_csv(METRICS, metric_rows)
        write_csv(BY_LIFTOFF, by_liftoff_rows)
        write_csv(SEED_SUMMARY, seed_rows)
        write_csv(VS_BASELINE, vs_rows)
        write_summary(metric_rows, selected_candidate, selected_seed, vs_rows, args)
        print(f"selected={selected_candidate} seed={selected_seed}")
        print(f"wrote {SUMMARY}")
        return 0

    trained_results: list[dict[str, Any]] = []
    for candidate in candidates():
        if candidate.calibrated_input:
            continue
        for seed in args.seeds:
            trained_results.append(train_one(candidate, seed, dataset, stats, args))

    selected_candidate, selected_seed = select_best(trained_results)
    metric_rows: list[dict[str, Any]] = as_training_metric_rows(baseline_rows)
    by_liftoff_rows: list[dict[str, Any]] = as_training_metric_rows(baseline_by_liftoff_rows)
    for result in trained_results:
        selected = result["candidate"] == selected_candidate and int(result["seed"]) == int(selected_seed)
        rows, by_rows = evaluate_run(dataset, stats, result, selected)
        metric_rows.extend(rows)
        by_liftoff_rows.extend(by_rows)

    seed_rows = seed_summary_rows(metric_rows, selected_candidate, selected_seed)
    vs_rows = vs_baseline_rows(metric_rows, selected_candidate, selected_seed, baseline_rows)

    write_csv(METRICS, metric_rows)
    write_csv(BY_LIFTOFF, by_liftoff_rows)
    write_csv(SEED_SUMMARY, seed_rows)
    write_csv(VS_BASELINE, vs_rows)
    write_summary(metric_rows, selected_candidate, selected_seed, vs_rows, args)
    print(f"selected={selected_candidate} seed={selected_seed}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

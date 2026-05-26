"""Gain/amplitude augmentation training gate for the true 3D RBC baseline.

This script runs only after calibration-only evaluation does not sufficiently
reduce gain sensitivity. It trains small 20.77-family Conv1D candidates with
in-memory delta_b augmentations and writes metrics only. It does not run COMSOL,
write data/NPZ files, or save checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_true_3d_rbc_gain_calibration_strategies as cal  # noqa: E402
import audit_true_3d_rbc_observation_perturbation_robustness as obs  # noqa: E402
import load_true_3d_rbc_pilot_dataset as loader  # noqa: E402
import train_true_3d_rbc_neural_parameter_gate as gate  # noqa: E402


DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
SUMMARY = ROOT / "results" / "summaries" / "true_3d_rbc_gain_augmented_training_summary.txt"
SEED_SUMMARY = ROOT / "results" / "metrics" / "true_3d_rbc_gain_augmented_seed_summary.csv"
ROBUSTNESS_METRICS = ROOT / "results" / "metrics" / "true_3d_rbc_gain_augmented_robustness_metrics.csv"
VS_REFERENCE = ROOT / "results" / "metrics" / "true_3d_rbc_gain_augmented_vs_reference.csv"
CALIBRATION_METRICS = ROOT / "results" / "metrics" / "true_3d_rbc_gain_calibration_strategy_metrics.csv"

PROFILE_BASELINE_RMSE = 0.000387737
BASELINE_DICE = 0.847727


@dataclass(frozen=True)
class AugCandidate:
    name: str
    description: str
    global_gain_aug: bool
    axis_gain_aug: bool
    bx_attenuation_aug: bool


def candidates() -> List[AugCandidate]:
    return [
        AugCandidate(
            "A1_global_gain_aug",
            "random global gain 0.8-1.2, noise 0-10pct, signed zero drift 0-1pct",
            True,
            False,
            False,
        ),
        AugCandidate(
            "A2_axis_gain_aug",
            "A1 plus per-axis Bx/By/Bz gain 0.9-1.1",
            True,
            True,
            False,
        ),
        AugCandidate(
            "A3_light_bx_attenuation_aug",
            "A2 plus low-probability Bx attenuation 0.7-1.0; no Bx-missing training",
            True,
            True,
            True,
        ),
    ]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def as_axes(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(x.shape[0], 3, 3, x.shape[-1]).copy()


def flatten_axes(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(x.shape[0], 9, x.shape[-1])


def augment_raw_batch(x: np.ndarray, candidate: AugCandidate, rng: np.random.Generator, ctx: Dict[str, object]) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    n = y.shape[0]
    if candidate.global_gain_aug:
        gain = rng.uniform(0.8, 1.2, size=(n, 1, 1)).astype(np.float32)
        y = y * gain
        noise_pct = rng.uniform(0.0, 0.10, size=(n, 1, 1)).astype(np.float32)
        noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32) * noise_pct * float(ctx["train_rms"])
        y = y + noise
        drift_pct = rng.uniform(-0.01, 0.01, size=(n, 1, 1)).astype(np.float32)
        y = y + drift_pct * float(ctx["train_abs_peak"])
    if candidate.axis_gain_aug:
        shaped = as_axes(y)
        axis_gain = rng.uniform(0.9, 1.1, size=(n, 3, 1, 1)).astype(np.float32)
        shaped *= axis_gain
        y = flatten_axes(shaped)
    if candidate.bx_attenuation_aug:
        shaped = as_axes(y)
        mask = rng.random(n) < 0.15
        if np.any(mask):
            attenuation = rng.uniform(0.7, 1.0, size=(int(mask.sum()), 1, 1)).astype(np.float32)
            shaped[mask, 0, :, :] *= attenuation
        y = flatten_axes(shaped)
    return y.astype(np.float32)


def normalize_x_raw(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)


def predict_physical(model: torch.nn.Module, x_raw: np.ndarray, stats: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x_norm = normalize_x_raw(x_raw, stats)
    pred_norm = gate.predict_norm(model, x_norm)
    pred_raw = loader.denormalize_y(pred_norm, stats)
    return pred_norm, pred_raw


def validation_score(
    model: torch.nn.Module,
    dataset: loader.True3DRBCDataset,
    stats: Dict[str, np.ndarray],
    splits: Dict[str, np.ndarray],
    eval_ctx: Dict[str, object],
) -> float:
    y_norm = loader.normalize_y(dataset, stats)
    val_idx = splits["val"]
    selected = {
        "clean": lambda x: x,
        "gain_0.8": lambda x: (x * 0.8).astype(np.float32),
        "gain_1.2": lambda x: (x * 1.2).astype(np.float32),
        "bx_50": None,
    }
    scores: Dict[str, float] = {}
    for name, fn in selected.items():
        if name == "bx_50":
            perturb = next(p for p in cal.selected_perturbations() if p.name == "channel_attenuation_Bx_50pct")
            raw = perturb.apply(dataset.x_channels.copy(), eval_ctx)
        else:
            raw = fn(dataset.x_channels.copy())
        pred_norm, _ = predict_physical(model, raw[val_idx], stats)
        comp = gate.selection_components(y_norm[val_idx], pred_norm)
        scores[name] = gate.selection_metric(comp)
    return scores["clean"] + 0.5 * (scores["gain_0.8"] + scores["gain_1.2"]) + 0.25 * scores["bx_50"]


def train_one_candidate(
    candidate: AugCandidate,
    seed: int,
    dataset: loader.True3DRBCDataset,
    stats: Dict[str, np.ndarray],
    eval_ctx: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    set_seed(seed)
    splits = loader.split_indices(dataset)
    y_norm = loader.normalize_y(dataset, stats)
    train_idx = splits["train"]
    train_ds = TensorDataset(
        torch.as_tensor(dataset.x_channels[train_idx], dtype=torch.float32),
        torch.as_tensor(y_norm[train_idx], dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = gate.RBCConvRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed + 8917)

    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_score = math.inf
    epoch_rows: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for xb_raw, yb in train_loader:
            xb_aug = augment_raw_batch(xb_raw.cpu().numpy(), candidate, rng, eval_ctx)
            xb_norm = torch.as_tensor(normalize_x_raw(xb_aug, stats), dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb_norm)
            loss = gate.weighted_smooth_l1(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_score = validation_score(model, dataset, stats, splits, eval_ctx)
        pred_train_norm, _ = predict_physical(model, dataset.x_channels[train_idx], stats)
        train_comp = gate.selection_components(y_norm[train_idx], pred_train_norm)
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
                "train_dimension_mae_norm": train_comp["dimension_mae_norm"],
                "train_curvature_mae_norm": train_comp["curvature_mae_norm"],
                "val_robust_selection_score": val_score,
            }
        )
    if best_state is None:
        raise RuntimeError(f"no state selected for {candidate.name} seed {seed}")
    model.load_state_dict(best_state)
    return {
        "candidate": candidate,
        "seed": seed,
        "model": model,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "epoch_rows": epoch_rows,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset: loader.True3DRBCDataset,
    stats: Dict[str, np.ndarray],
    eval_ctx: Dict[str, object],
    candidate_name: str,
    seed: int,
    selected: bool,
    best_epoch: int,
    best_val_score: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for perturbation in cal.selected_perturbations():
        raw = perturbation.apply(dataset.x_channels.copy(), eval_ctx)
        _, pred_raw = predict_physical(model, raw, stats)
        metric_rows = cal.split_metric_rows(pred_raw, dataset, stats, candidate_name, perturbation)
        for row in metric_rows:
            row["candidate"] = candidate_name
            row["seed"] = seed
            row["selected_seed"] = selected
            row["best_epoch"] = best_epoch
            row["best_val_score"] = best_val_score
            rows.append(row)
    return rows


def metric_row(rows: Iterable[Dict[str, object]], candidate: str, perturbation: str, split: str = "test") -> Dict[str, object] | None:
    for row in rows:
        if row.get("candidate") == candidate and row.get("perturbation_name") == perturbation and row.get("split") == split:
            return row
    return None


def calibration_reference_rows() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in read_csv(CALIBRATION_METRICS):
        if row.get("strategy") != "no_calibration":
            continue
        new = dict(row)
        new["candidate"] = "A0_reference_20_77"
        new["seed"] = 42
        new["selected_seed"] = True
        new["best_epoch"] = ""
        new["best_val_score"] = ""
        out.append(new)
    return out


def select_best_run(run_rows: List[Dict[str, object]]) -> tuple[str, int]:
    score_rows = []
    for candidate_name in sorted({str(row["candidate"]) for row in run_rows if str(row["candidate"]).startswith("A")}):
        if candidate_name == "A0_reference_20_77":
            continue
        for seed in sorted({int(row["seed"]) for row in run_rows if row["candidate"] == candidate_name}):
            seed_subset = [r for r in run_rows if r["candidate"] == candidate_name and int(r["seed"]) == seed]
            clean = metric_row(seed_subset, candidate_name, "clean", split="val")
            if not clean or clean.get("best_val_score") in {"", None}:
                continue
            # Candidate/seed selection uses the training loop's validation-only
            # robust score captured at checkpoint selection. Test metrics are
            # reported after this selection and never used for choosing.
            score = float(clean["best_val_score"])
            score_rows.append((score, candidate_name, seed))
    if not score_rows:
        raise RuntimeError("no trained candidate rows available")
    _, best_candidate, best_seed = sorted(score_rows, key=lambda item: item[0])[0]
    return best_candidate, best_seed


def seed_summary_rows(rows: List[Dict[str, object]], best_candidate: str, best_seed: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for candidate_name in sorted({str(row["candidate"]) for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows if row["candidate"] == candidate_name and str(row["seed"]).isdigit()}):
            subset = [row for row in rows if row["candidate"] == candidate_name and int(row["seed"]) == seed]
            clean = next((row for row in subset if row["split"] == "test" and row["perturbation_name"] == "clean"), None)
            gain08 = next((row for row in subset if row["split"] == "test" and row["perturbation_name"] == "gain_scaling_0.8x"), None)
            gain12 = next((row for row in subset if row["split"] == "test" and row["perturbation_name"] == "gain_scaling_1.2x"), None)
            bx50 = next((row for row in subset if row["split"] == "test" and row["perturbation_name"] == "channel_attenuation_Bx_50pct"), None)
            if not clean:
                continue
            out.append(
                {
                    "candidate": candidate_name,
                    "seed": seed,
                    "selected_robustness_candidate": candidate_name == best_candidate and seed == best_seed,
                    "best_epoch": clean.get("best_epoch", ""),
                    "best_val_score": clean.get("best_val_score", ""),
                    "test_clean_profile_depth_rmse_m": clean.get("profile_depth_rmse_m", ""),
                    "test_clean_profile_rmse_drop_pct": clean.get("profile_rmse_degradation_pct_vs_20_85", ""),
                    "test_clean_dice": clean.get("projected_mask_dice", ""),
                    "test_gain08_profile_rmse_drop_pct": gain08.get("profile_rmse_degradation_pct_vs_20_85", "") if gain08 else "",
                    "test_gain12_profile_rmse_drop_pct": gain12.get("profile_rmse_degradation_pct_vs_20_85", "") if gain12 else "",
                    "test_bx50_profile_rmse_drop_pct": bx50.get("profile_rmse_degradation_pct_vs_20_85", "") if bx50 else "",
                }
            )
    return out


def vs_reference_rows(rows: List[Dict[str, object]], best_candidate: str, best_seed: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    ref_rows = [row for row in rows if row["candidate"] == "A0_reference_20_77" and row["split"] == "test"]
    best_rows = [
        row
        for row in rows
        if row["candidate"] == best_candidate and int(row["seed"]) == best_seed and row["split"] == "test"
    ]
    for ref in ref_rows:
        perturbation = ref["perturbation_name"]
        current = next((row for row in best_rows if row["perturbation_name"] == perturbation), None)
        if not current:
            continue
        for metric in [
            "profile_depth_rmse_m",
            "profile_rmse_degradation_pct_vs_20_85",
            "er_like_profile_error",
            "L_mae_mm",
            "W_mae_mm",
            "D_mae_mm",
            "curvature_mae_mean",
            "projected_mask_iou",
            "projected_mask_dice",
        ]:
            ref_val = float(ref[metric])
            cur_val = float(current[metric])
            lower_better = metric not in {"projected_mask_iou", "projected_mask_dice"}
            improved = cur_val < ref_val if lower_better else cur_val > ref_val
            out.append(
                {
                    "perturbation_name": perturbation,
                    "metric": metric,
                    "reference_candidate": "A0_reference_20_77",
                    "current_candidate": best_candidate,
                    "current_seed": best_seed,
                    "reference_value": ref_val,
                    "current_value": cur_val,
                    "delta": cur_val - ref_val,
                    "improved": improved,
                }
            )
    return out


def augmentation_effective(vs_rows: List[Dict[str, object]], rows: List[Dict[str, object]], best_candidate: str, best_seed: int) -> bool:
    best_test = [
        row
        for row in rows
        if row["candidate"] == best_candidate and int(row["seed"]) == best_seed and row["split"] == "test"
    ]
    clean = next(row for row in best_test if row["perturbation_name"] == "clean")
    gain08 = next(row for row in best_test if row["perturbation_name"] == "gain_scaling_0.8x")
    gain12 = next(row for row in best_test if row["perturbation_name"] == "gain_scaling_1.2x")
    ref08 = next(row for row in rows if row["candidate"] == "A0_reference_20_77" and row["split"] == "test" and row["perturbation_name"] == "gain_scaling_0.8x")
    ref12 = next(row for row in rows if row["candidate"] == "A0_reference_20_77" and row["split"] == "test" and row["perturbation_name"] == "gain_scaling_1.2x")
    red08 = 100.0 * (float(ref08["profile_rmse_degradation_pct_vs_20_85"]) - float(gain08["profile_rmse_degradation_pct_vs_20_85"])) / max(abs(float(ref08["profile_rmse_degradation_pct_vs_20_85"])), 1e-9)
    red12 = 100.0 * (float(ref12["profile_rmse_degradation_pct_vs_20_85"]) - float(gain12["profile_rmse_degradation_pct_vs_20_85"])) / max(abs(float(ref12["profile_rmse_degradation_pct_vs_20_85"])), 1e-9)
    return (
        float(clean["profile_rmse_degradation_pct_vs_20_85"]) <= 10.0
        and red08 >= 50.0
        and red12 >= 50.0
        and float(clean["projected_mask_dice"]) >= BASELINE_DICE - 0.02
    )


def write_summary(
    rows: List[Dict[str, object]],
    seed_rows: List[Dict[str, object]],
    best_candidate: str,
    best_seed: int,
    effective: bool,
    args: argparse.Namespace,
) -> None:
    best = [row for row in rows if row["candidate"] == best_candidate and int(row["seed"]) == best_seed and row["split"] == "test"]
    def pick(name: str) -> Dict[str, object]:
        return next(row for row in best if row["perturbation_name"] == name)
    clean = pick("clean")
    gain08 = pick("gain_scaling_0.8x")
    gain12 = pick("gain_scaling_1.2x")
    bx50 = pick("channel_attenuation_Bx_50pct")
    lines = [
        "20.89 gain/amplitude augmentation training gate",
        "",
        "Scope:",
        "- Dataset: comsol_true_3d_rbc_imported_watertight_pilot_v3_240 loaded via registry/manifest",
        "- Model family: 20.77 small Conv1D encoder + MLP six-parameter head",
        "- Training: in-memory delta_b augmentation only; no COMSOL, no data/NPZ write, no checkpoint saved",
        f"- Seeds: {list(args.seeds)}",
        f"- Epochs: {args.epochs}",
        "",
        "Candidate variants:",
    ]
    for candidate in candidates():
        lines.append(f"- {candidate.name}: {candidate.description}")
    lines.extend(
        [
            "",
            f"Selected robustness candidate: {best_candidate}, seed={best_seed}",
            f"- clean profile_depth_rmse_m: {float(clean['profile_depth_rmse_m']):.9f}",
            f"- clean profile RMSE drop vs 20.85: {float(clean['profile_rmse_degradation_pct_vs_20_85']):.3f}%",
            f"- clean projected Dice: {float(clean['projected_mask_dice']):.6f}",
            f"- gain 0.8 profile RMSE degradation: {float(gain08['profile_rmse_degradation_pct_vs_20_85']):.3f}%",
            f"- gain 1.2 profile RMSE degradation: {float(gain12['profile_rmse_degradation_pct_vs_20_85']):.3f}%",
            f"- Bx 50pct attenuation degradation: {float(bx50['profile_rmse_degradation_pct_vs_20_85']):.3f}%",
            f"- L/W/D clean MAE mm: {float(clean['L_mae_mm']):.3f} / {float(clean['W_mae_mm']):.3f} / {float(clean['D_mae_mm']):.3f}",
            f"- wMAE auxiliary clean: {float(clean['curvature_mae_mean']):.6f}",
            "",
            f"Augmentation effective by gate: {effective}",
            "",
            "Gate logic:",
            "- Effective requires clean profile RMSE drop <=10%, clean Dice no worse than 0.02 below baseline, and gain 0.8/1.2 degradation reduction >=50% versus A0.",
            "- Bx missing is not trained and remains a diagnostic-only failure mode.",
        ]
    )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summarize-existing", action="store_true", help="reuse existing robustness metrics and only recompute validation-selected summaries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize_existing:
        all_rows = [dict(row) for row in read_csv(ROBUSTNESS_METRICS)]
        if not all_rows:
            raise FileNotFoundError(ROBUSTNESS_METRICS)
        best_candidate, best_seed = select_best_run(all_rows)
        for row in all_rows:
            row["selected_seed"] = bool(row.get("candidate") == best_candidate and str(row.get("seed")) == str(best_seed))
        seed_rows = seed_summary_rows(all_rows, best_candidate, best_seed)
        vs_rows = vs_reference_rows(all_rows, best_candidate, best_seed)
        effective = augmentation_effective(vs_rows, all_rows, best_candidate, best_seed)
        write_csv(ROBUSTNESS_METRICS, all_rows)
        write_csv(SEED_SUMMARY, seed_rows)
        write_csv(VS_REFERENCE, vs_rows)
        write_summary(all_rows, seed_rows, best_candidate, best_seed, effective, args)
        print(f"wrote {SUMMARY}")
        print(f"selected={best_candidate} seed={best_seed} augmentation_effective={effective}")
        return
    if not CALIBRATION_METRICS.exists():
        raise FileNotFoundError(CALIBRATION_METRICS)
    entry, manifest, npz_path = loader.resolve_dataset(DATASET_ID)
    failed = [row for row in loader.gate_manifest(entry, manifest, npz_path, DATASET_ID) if not row["pass"]]
    if failed:
        raise RuntimeError(f"registry/manifest gate failed: {failed}")
    dataset = loader.load_dataset(DATASET_ID)
    artifact_manifest, checkpoint, _reference_model = obs.load_artifact(cal.ARTIFACT_MANIFEST)
    stats = {
        "x_mean": checkpoint["normalization"]["x_mean"],
        "x_std": checkpoint["normalization"]["x_std"],
        "y_mean": checkpoint["normalization"]["y_mean"],
        "y_std": checkpoint["normalization"]["y_std"],
    }
    eval_ctx = obs.make_context(dataset, checkpoint)

    all_rows: List[Dict[str, object]] = calibration_reference_rows()
    epoch_rows: List[Dict[str, object]] = []
    trained_runs: List[Dict[str, object]] = []
    for candidate in candidates():
        for seed in args.seeds:
            result = train_one_candidate(candidate, seed, dataset, stats, eval_ctx, args)
            trained_runs.append(result)
            epoch_rows.extend(result["epoch_rows"])
            metric_rows = evaluate_model(
                result["model"],
                dataset,
                stats,
                eval_ctx,
                candidate.name,
                seed,
                False,
                int(result["best_epoch"]),
                float(result["best_val_score"]),
            )
            all_rows.extend(metric_rows)

    best_candidate, best_seed = select_best_run(all_rows)
    for row in all_rows:
        if row.get("candidate") == best_candidate and str(row.get("seed")) == str(best_seed):
            row["selected_seed"] = True
    seed_rows = seed_summary_rows(all_rows, best_candidate, best_seed)
    vs_rows = vs_reference_rows(all_rows, best_candidate, best_seed)
    effective = augmentation_effective(vs_rows, all_rows, best_candidate, best_seed)

    write_csv(ROBUSTNESS_METRICS, all_rows)
    write_csv(SEED_SUMMARY, seed_rows)
    write_csv(VS_REFERENCE, vs_rows)
    write_summary(all_rows, seed_rows, best_candidate, best_seed, effective, args)
    print(f"wrote {SUMMARY}")
    print(f"selected={best_candidate} seed={best_seed} augmentation_effective={effective}")


if __name__ == "__main__":
    main()

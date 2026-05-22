from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import train_comsol_rect_rot_perturbation_calibrated_forward_surrogate as perturb
from test_comsol_rect_rot_differentiable_rasterizer import TEMPERATURE_M, soft_rect_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
PERTURB_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_rect_rot_local_perturbation_forward_pack_v1.npz"
PROPOSAL_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_proposal_extraction_selected_geometry.csv"
DENSE_METRICS_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_coarse_initializer_metrics.csv"
OLD_REFINEMENT_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_improved_dense_priewald_refinement_metrics.csv"
OLD_REFINEMENT_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_improved_dense_priewald_refinement_summary.txt"
)
PERTURB_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_perturbation_forward_surrogate_summary.txt"
PERTURB_RESIDUAL_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_perturbation_residual_objective_audit_summary.txt"
)
PERTURB_REVIEW = (
    PROJECT_ROOT / "results/summaries/claude_review_comsol_rect_rot_perturbation_forward_calibration.txt"
)
PERTURB_CANDIDATES = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_forward_surrogate_candidates.csv"
)
PERTURB_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_forward_surrogate_metrics.csv"
PERTURB_ORDERING = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_forward_surrogate_ordering_audit.csv"
)
PERTURB_RESIDUAL = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturbation_residual_objective_audit.csv"
)

INPUT_CHECK_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_calibrated_refinement_retry_input_check_summary.txt"
)
INPUT_CHECK_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_calibrated_refinement_retry_input_check.csv"
RECOVERY_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_calibrated_refinement_retry_surrogate_recovery_summary.txt"
)
RECOVERY_CSV = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_calibrated_refinement_retry_surrogate_recovery_metrics.csv"
)
RETRY_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_perturb_calibrated_refinement_retry_summary.txt"
FAILURE_SUMMARY = (
    PROJECT_ROOT
    / "results/summaries/comsol_rect_rot_perturb_calibrated_refinement_retry_failure_audit_summary.txt"
)
INTERPRETABILITY_SUMMARY = (
    PROJECT_ROOT
    / "results/summaries/comsol_rect_rot_perturb_calibrated_residual_interpretability_summary.txt"
)
CONFIG_SWEEP_CSV = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturb_calibrated_refinement_retry_config_sweep.csv"
)
METRICS_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturb_calibrated_refinement_retry_metrics.csv"
GROUP_SUMMARY_CSV = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturb_calibrated_refinement_retry_group_summary.csv"
)
GEOMETRY_SUMMARY_CSV = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturb_calibrated_refinement_retry_geometry_summary.csv"
)
FORWARD_SUMMARY_CSV = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturb_calibrated_refinement_retry_forward_summary.csv"
)
FAILURE_CASES_CSV = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturb_calibrated_refinement_retry_failure_cases.csv"
)
INTERPRETABILITY_CSV = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_perturb_calibrated_residual_interpretability.csv"
)
PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_rect_rot_perturb_calibrated_refinement_retry"

MAIN_TYPES = {"rectangular_notch", "rotated_rect"}
SEED = 42
MAX_ANGLE_RAD = math.radians(35.0)
REFERENCE_2054 = {
    "dense_test_iou": 0.6689,
    "dense_test_dice": 0.7994,
    "proposal_test_iou": 0.6726,
    "proposal_test_dice": 0.8017,
    "old_refined_test_iou": 0.6646,
    "old_refined_test_dice": 0.7958,
    "old_forward_initial_nrmse": 0.4632,
    "old_forward_refined_nrmse": 0.4049,
}
REFERENCE_2056 = {
    "selected": "S1_perturb_geom_mlp",
    "val_nrmse": 0.3666,
    "test_nrmse": 0.4289,
    "val_ordering": 0.7321,
    "test_ordering": 0.8036,
    "val_mismatch": 0.2679,
    "test_mismatch": 0.1964,
    "test_residual_error_corr": -0.0462,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry Priewald-style refinement with the 20.56 perturbation-calibrated S1 forward surrogate."
    )
    parser.add_argument("--pilot-npz", type=Path, default=PILOT_NPZ)
    parser.add_argument("--perturb-npz", type=Path, default=PERTURB_NPZ)
    parser.add_argument("--proposal-csv", type=Path, default=PROPOSAL_CSV)
    parser.add_argument("--surrogate-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--preview-count", type=int, default=24)
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def safe_corr(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or y.size < 2:
        return math.nan
    if float(x.std()) <= 1.0e-12 or float(y.std()) <= 1.0e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def angle_diff_deg(pred_deg: float, true_deg: float) -> float:
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return abs(float(diff))


def geom8_from_geom6(geom: torch.Tensor) -> torch.Tensor:
    angle = geom[:, 5]
    return torch.cat([geom, torch.sin(angle).unsqueeze(1), torch.cos(angle).unsqueeze(1)], dim=1)


def normalized_surrogate_input(
    geom6: torch.Tensor,
    type_prob: torch.Tensor,
    geom_mean: torch.Tensor,
    geom_std: torch.Tensor,
) -> torch.Tensor:
    geom8 = geom8_from_geom6(geom6)
    return torch.cat([type_prob, (geom8 - geom_mean) / geom_std], dim=1)


def waveform_loss(pred_norm: torch.Tensor, observed_norm: torch.Tensor) -> torch.Tensor:
    pred_grad = pred_norm[:, :, 1:] - pred_norm[:, :, :-1]
    obs_grad = observed_norm[:, :, 1:] - observed_norm[:, :, :-1]
    return (
        F.mse_loss(pred_norm, observed_norm)
        + 0.2 * F.l1_loss(pred_norm, observed_norm)
        + 0.1 * F.mse_loss(pred_grad, obs_grad)
    )


def per_sample_forward_metrics(pred_norm: np.ndarray, observed_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = pred_norm - observed_norm
    mse = np.mean(diff**2, axis=(1, 2))
    rmse = np.sqrt(mse)
    denom = np.std(observed_norm, axis=(1, 2))
    denom = np.where(denom <= 1.0e-12, 1.0, denom)
    nrmse = rmse / denom
    corr = np.array([safe_corr(pred_norm[i].reshape(-1), observed_norm[i].reshape(-1)) for i in range(pred_norm.shape[0])])
    return mse, nrmse, corr


@dataclass
class Record:
    sample_id: str
    source_index: int
    split: str
    defect_type: str
    source_pack: str
    true_mask: np.ndarray
    observed_delta: np.ndarray
    true_geom: np.ndarray
    init_geom: np.ndarray
    type_prob: np.ndarray
    dense_iou: float
    dense_dice: float
    dense_area_error: float
    proposal_iou_reference: float
    proposal_dice_reference: float
    proposal_area_error_reference: float
    old_refined_iou: float
    old_refined_dice: float
    old_refined_area_error: float
    old_initial_forward_nrmse: float
    old_refined_forward_nrmse: float


@dataclass
class Context:
    records: list[Record]
    mask_x: np.ndarray
    mask_y: np.ndarray
    observed_mean: float
    observed_std: float
    geom_mean8: np.ndarray
    geom_std8: np.ndarray
    geom_scale6: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    surrogate_bundle: perturb.Bundle
    recovery_selected: dict[str, Any]


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def input_check(args: argparse.Namespace) -> None:
    required = [
        PERTURB_SUMMARY,
        PERTURB_RESIDUAL_SUMMARY,
        PERTURB_REVIEW,
        PERTURB_CANDIDATES,
        PERTURB_METRICS,
        PERTURB_ORDERING,
        PERTURB_RESIDUAL,
        PROPOSAL_CSV,
        DENSE_METRICS_CSV,
        OLD_REFINEMENT_CSV,
        OLD_REFINEMENT_SUMMARY,
        args.pilot_npz,
        args.perturb_npz,
    ]
    rows: list[dict[str, Any]] = []
    for path in required:
        rows.append(
            {
                "check": f"exists:{path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}",
                "status": "pass" if path.exists() else "fail",
                "value": str(path),
                "notes": "",
            }
        )
    missing = [row["value"] for row in rows if row["status"] != "pass"]
    if missing:
        write_csv(INPUT_CHECK_CSV, rows)
        INPUT_CHECK_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        INPUT_CHECK_SUMMARY.write_text(
            "20.57 input check failed.\n\nMissing files:\n" + "\n".join(f"- {x}" for x in missing) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"Missing required inputs: {missing}")

    candidates = read_csv(PERTURB_CANDIDATES)
    selected = max(candidates, key=lambda row: to_float(row.get("selection_score")))
    rows.append(
        {
            "check": "20.56_selected_surrogate",
            "status": "pass" if selected.get("candidate") == "S1_perturb_geom_mlp" else "fail",
            "value": selected.get("candidate"),
            "notes": "Expected S1_perturb_geom_mlp.",
        }
    )
    rows.append(
        {
            "check": "20.56_ordering_improves_20.55",
            "status": "pass" if to_float(selected.get("val_surrogate_mismatch_rate")) < 0.3030 else "fail",
            "value": f"val_mismatch={selected.get('val_surrogate_mismatch_rate')}",
            "notes": "20.55 S2 val mismatch reference is 0.3030.",
        }
    )
    rows.append(
        {
            "check": "20.56_test_correlation_caveat_recorded",
            "status": "pass" if to_float(selected.get("test_surrogate_residual_error_correlation")) < 0.05 else "warn",
            "value": selected.get("test_surrogate_residual_error_correlation"),
            "notes": "Negative/unstable test residual-error correlation should remain a caveat.",
        }
    )

    perturb_data = np.load(args.perturb_npz, allow_pickle=True)
    split = perturb_data["split"].astype(str)
    defect_types = perturb_data["defect_types"].astype(str)
    generated = perturb_data["generated_real_forward"].astype(bool)
    rows.extend(
        [
            {
                "check": "20.56_partial_pack_rows",
                "status": "pass" if len(split) == 96 else "warn",
                "value": len(split),
                "notes": "Expected actual 20.56 partial pack size is 96 rows.",
            },
            {
                "check": "20.56_partial_pack_split",
                "status": "pass" if dict(Counter(split)) == {"train": 64, "val": 16, "test": 16} else "warn",
                "value": dict(Counter(split)),
                "notes": "Split is preserved by base sample.",
            },
            {
                "check": "20.56_partial_pack_type_balance",
                "status": "pass" if dict(Counter(defect_types)) == {"rectangular_notch": 48, "rotated_rect": 48} else "warn",
                "value": dict(Counter(defect_types)),
                "notes": "",
            },
            {
                "check": "20.56_real_forward_count",
                "status": "pass" if int(generated.sum()) == 84 else "warn",
                "value": int(generated.sum()),
                "notes": "True geometry reference rows reuse original pilot NPZ.",
            },
        ]
    )

    pilot = np.load(args.pilot_npz, allow_pickle=True)
    main_mask = np.isin(pilot["defect_types"].astype(str), sorted(MAIN_TYPES))
    main_split = pilot["split"].astype(str)[main_mask]
    proposal_rows = [row for row in read_csv(args.proposal_csv) if row["defect_type"] in MAIN_TYPES]
    rows.extend(
        [
            {
                "check": "rect_rot_subset_count",
                "status": "pass" if int(main_mask.sum()) == 400 else "fail",
                "value": int(main_mask.sum()),
                "notes": "",
            },
            {
                "check": "rect_rot_split",
                "status": "pass" if dict(Counter(main_split)) == {"train": 268, "val": 66, "test": 66} else "fail",
                "value": dict(Counter(main_split)),
                "notes": "",
            },
            {
                "check": "20.54_proposal_rows",
                "status": "pass" if len(proposal_rows) == 400 else "fail",
                "value": len(proposal_rows),
                "notes": "",
            },
        ]
    )

    write_csv(INPUT_CHECK_CSV, rows)
    failed = [row for row in rows if row["status"] == "fail"]
    lines = [
        "20.57 calibrated refinement retry input check summary",
        "",
        "Scope: controlled retry only. No COMSOL run, no new perturbation data, no inverse head or dense baseline training.",
        f"Selected 20.56 surrogate: {selected.get('candidate')}",
        f"20.56 val/test ordering: {selected.get('val_surrogate_ordering_accuracy')} / {selected.get('test_surrogate_ordering_accuracy')}",
        f"20.56 val/test mismatch: {selected.get('val_surrogate_mismatch_rate')} / {selected.get('test_surrogate_mismatch_rate')}",
        f"20.56 test residual-error correlation caveat: {selected.get('test_surrogate_residual_error_correlation')}",
        f"20.56 perturbation pack rows: {len(split)}; split={dict(Counter(split))}; type={dict(Counter(defect_types))}",
        f"20.54 proposal rows available: {len(proposal_rows)}",
        f"Rect+rot pilot subset: {int(main_mask.sum())}; split={dict(Counter(main_split))}",
        "",
        f"Input check passed: {not failed}",
    ]
    if failed:
        lines.extend(["", "Failed checks:"])
        lines.extend(f"- {row['check']}: {row['value']}" for row in failed)
    INPUT_CHECK_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    INPUT_CHECK_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"Input check failed: {[row['check'] for row in failed]}")


def perturb_geom_stats(npz_path: Path) -> dict[str, np.ndarray | float]:
    data = np.load(npz_path, allow_pickle=True)
    split = data["split"].astype(str)
    train_idx = np.where(split == "train")[0]
    geom_rows = [parse_json(raw) for raw in data["geometry_params"]]
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
    geom_mean = geom8[train_idx].mean(axis=0).astype(np.float32)
    geom_std = geom8[train_idx].std(axis=0).astype(np.float32)
    geom_std = np.where(geom_std <= 1.0e-12, 1.0, geom_std).astype(np.float32)
    target = data["delta_bz"].astype(np.float32)
    target_mean = float(target[train_idx].mean())
    target_std = float(target[train_idx].std())
    if target_std <= 1.0e-12:
        target_std = 1.0
    return {"geom_mean8": geom_mean, "geom_std8": geom_std, "target_mean": target_mean, "target_std": target_std}


def recover_s1(args: argparse.Namespace, device: torch.device) -> tuple[perturb.Bundle, dict[str, Any]]:
    perturb.set_seed(SEED)
    arrays = perturb.load_arrays(args.perturb_npz)
    train_args = argparse.Namespace(epochs=args.surrogate_epochs, batch_size=args.batch_size, lr=args.lr)
    bundle, _epoch_rows = perturb.train_candidate("S1_perturb_geom_mlp", arrays, train_args, device)
    waveform_rows, pred_by_split = perturb.evaluate_waveform(bundle, arrays)
    ordering_stats: dict[str, dict[str, float]] = {}
    recovery_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        stats, _ = perturb.ordering_for_split(arrays, split, pred_by_split[split], "S1_perturb_geom_mlp")
        ordering_stats[split] = stats
        wave = next(row for row in waveform_rows if row["split"] == split)
        recovery_rows.append({"candidate": "S1_perturb_geom_mlp", "split": split, "best_epoch": bundle.best_epoch, **wave, **stats})
    selected = perturb.summarize_candidate("S1_perturb_geom_mlp", waveform_rows, ordering_stats, bundle.best_epoch)
    write_csv(RECOVERY_CSV, recovery_rows)

    close_to_2056 = (
        selected["val_surrogate_ordering_accuracy"] >= REFERENCE_2056["val_ordering"] - 0.12
        and selected["test_surrogate_ordering_accuracy"] >= REFERENCE_2056["test_ordering"] - 0.12
        and selected["val_surrogate_mismatch_rate"] <= 0.35
    )
    lines = [
        "20.57 S1 perturbation-calibrated surrogate recovery summary",
        "",
        "No checkpoint is reused or written. S1 is retrained with the 20.56 protocol from the perturbation forward pack.",
        f"Device: {device}",
        f"Best epoch: {bundle.best_epoch}",
        f"Val/test waveform NRMSE: {selected['val_nrmse']:.6f} / {selected['test_nrmse']:.6f}",
        f"Val/test waveform correlation: {selected['val_correlation']:.6f} / {selected['test_correlation']:.6f}",
        f"Val/test ordering accuracy: {selected['val_surrogate_ordering_accuracy']:.6f} / {selected['test_surrogate_ordering_accuracy']:.6f}",
        f"Val/test mismatch rate: {selected['val_surrogate_mismatch_rate']:.6f} / {selected['test_surrogate_mismatch_rate']:.6f}",
        f"Val/test residual-error correlation: {selected['val_surrogate_residual_error_correlation']:.6f} / {selected['test_surrogate_residual_error_correlation']:.6f}",
        f"Close enough to 20.56 gate for controlled retry: {close_to_2056}",
    ]
    RECOVERY_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    RECOVERY_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not close_to_2056:
        raise RuntimeError("Recovered S1 ordering is too far below 20.56; stopping before refinement.")
    return bundle, selected


def load_records(args: argparse.Namespace, target_mean: float, target_std: float) -> tuple[list[Record], np.ndarray, np.ndarray, np.ndarray]:
    pilot = np.load(args.pilot_npz, allow_pickle=True)
    sample_ids = pilot["sample_ids"].astype(str)
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    true_geoms: dict[str, np.ndarray] = {}
    for sample_id, raw in zip(sample_ids, pilot["geometry_params"]):
        g = parse_json(raw)
        true_geoms[sample_id] = np.array(
            [
                float(g["center_x"]),
                float(g["center_y"]),
                float(g["width"]),
                float(g["length"]),
                float(g["depth"]),
                to_float(g.get("angle_rad", g.get("angle", 0.0)), 0.0),
            ],
            dtype=np.float32,
        )

    old_rows = {row["sample_id"]: row for row in read_csv(OLD_REFINEMENT_CSV)}
    proposal_rows = [row for row in read_csv(args.proposal_csv) if row["defect_type"] in MAIN_TYPES]
    records: list[Record] = []
    for row in proposal_rows:
        sample_id = row["sample_id"]
        if sample_id not in id_to_idx:
            continue
        idx = id_to_idx[sample_id]
        init_geom = np.array(
            [
                to_float(row["pred_center_x"]),
                to_float(row["pred_center_y"]),
                max(to_float(row["pred_width"]), 1.0e-6),
                max(to_float(row["pred_length"]), 1.0e-6),
                max(to_float(row["pred_depth"]), 1.0e-6),
                float(np.clip(to_float(row["pred_angle_rad"], 0.0), -MAX_ANGLE_RAD, MAX_ANGLE_RAD)),
            ],
            dtype=np.float32,
        )
        p_rect = to_float(row.get("type_prob_rectangular_notch"), 0.5)
        p_rot = to_float(row.get("type_prob_rotated_rect"), 0.5)
        total = p_rect + p_rot
        if not math.isfinite(total) or total <= 1.0e-8:
            p_rect, p_rot = 0.5, 0.5
        else:
            p_rect, p_rot = p_rect / total, p_rot / total
        old = old_rows.get(sample_id, {})
        records.append(
            Record(
                sample_id=sample_id,
                source_index=int(to_float(row.get("source_index"), idx)),
                split=str(row["split"]),
                defect_type=str(row["defect_type"]),
                source_pack=str(row.get("source_pack", "")),
                true_mask=pilot["masks"][idx].astype(np.uint8),
                observed_delta=((pilot["delta_bz"][idx].astype(np.float32) - target_mean) / target_std).astype(np.float32),
                true_geom=true_geoms[sample_id],
                init_geom=init_geom,
                type_prob=np.array([p_rect, p_rot], dtype=np.float32),
                dense_iou=to_float(row.get("dense_iou")),
                dense_dice=to_float(row.get("dense_dice")),
                dense_area_error=to_float(row.get("dense_area_error")),
                proposal_iou_reference=to_float(row.get("geometry_iou")),
                proposal_dice_reference=to_float(row.get("geometry_dice")),
                proposal_area_error_reference=to_float(row.get("geometry_area_error")),
                old_refined_iou=to_float(old.get("refined_iou")),
                old_refined_dice=to_float(old.get("refined_dice")),
                old_refined_area_error=to_float(old.get("refined_area_error")),
                old_initial_forward_nrmse=to_float(old.get("initial_forward_nrmse")),
                old_refined_forward_nrmse=to_float(old.get("refined_forward_nrmse")),
            )
        )
    if len(records) != 400:
        raise RuntimeError(f"Expected 400 rect+rot proposal records, got {len(records)}")
    mask_x = pilot["mask_x"].astype(np.float32)
    mask_y = pilot["mask_y"].astype(np.float32)
    train_true = np.stack([r.true_geom for r in records if r.split == "train"], axis=0)
    return records, mask_x, mask_y, train_true


def bounds_from_train(train_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.percentile(train_true, 0.5, axis=0)
    upper = np.percentile(train_true, 99.5, axis=0)
    span = np.maximum(upper - lower, 1.0e-6)
    bounds_min = lower - 0.25 * span
    bounds_max = upper + 0.25 * span
    bounds_min[2:5] = np.maximum(bounds_min[2:5], 1.0e-5)
    bounds_max[2:5] = np.maximum(bounds_max[2:5], bounds_min[2:5] + 1.0e-5)
    bounds_min[5] = -MAX_ANGLE_RAD
    bounds_max[5] = MAX_ANGLE_RAD
    scale = np.std(train_true, axis=0).astype(np.float32)
    scale = np.where(scale <= 1.0e-8, 1.0, scale).astype(np.float32)
    return bounds_min.astype(np.float32), bounds_max.astype(np.float32), scale


def build_context(args: argparse.Namespace, device: torch.device) -> Context:
    input_check(args)
    bundle, selected = recover_s1(args, device)
    stats = perturb_geom_stats(args.perturb_npz)
    records, mask_x, mask_y, train_true = load_records(
        args, float(stats["target_mean"]), float(stats["target_std"])
    )
    bounds_min, bounds_max, scale = bounds_from_train(train_true)
    bundle.model.eval()
    for p in bundle.model.parameters():
        p.requires_grad_(False)
    return Context(
        records=records,
        mask_x=mask_x,
        mask_y=mask_y,
        observed_mean=float(stats["target_mean"]),
        observed_std=float(stats["target_std"]),
        geom_mean8=stats["geom_mean8"],
        geom_std8=stats["geom_std8"],
        geom_scale6=scale,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        surrogate_bundle=bundle,
        recovery_selected=selected,
    )


def split_records(records: list[Record], split: str) -> list[Record]:
    return [record for record in records if record.split == split]


def constrain_geom(params: torch.Tensor, bounds_min: torch.Tensor, bounds_max: torch.Tensor) -> torch.Tensor:
    parts = []
    for i in range(params.shape[1]):
        parts.append(params[:, i].clamp(bounds_min[i], bounds_max[i]).unsqueeze(1))
    return torch.cat(parts, dim=1)


def predict_norm(
    model: torch.nn.Module,
    geom: torch.Tensor,
    type_prob: torch.Tensor,
    geom_mean8: torch.Tensor,
    geom_std8: torch.Tensor,
) -> torch.Tensor:
    x = normalized_surrogate_input(geom, type_prob, geom_mean8, geom_std8)
    return model(x)


def rasterize_mixture(
    geom: np.ndarray,
    type_prob: np.ndarray,
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or torch.device("cpu")
    gx = torch.tensor(geom, dtype=torch.float32, device=device)
    tp = torch.tensor(type_prob, dtype=torch.float32, device=device)
    mx = torch.tensor(mask_x, dtype=torch.float32, device=device)
    my = torch.tensor(mask_y, dtype=torch.float32, device=device)
    with torch.no_grad():
        rect = soft_rect_mask(mx, my, gx[:, 0], gx[:, 1], gx[:, 2], gx[:, 3], torch.zeros_like(gx[:, 5]), TEMPERATURE_M)
        rot = soft_rect_mask(mx, my, gx[:, 0], gx[:, 1], gx[:, 2], gx[:, 3], gx[:, 5], TEMPERATURE_M)
        prob = tp[:, 0].view(-1, 1, 1) * rect + tp[:, 1].view(-1, 1, 1) * rot
    return prob.cpu().numpy()


def mask_metrics_from_prob(prob: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    pred = prob >= 0.5
    true = true_mask > 0
    inter = int(np.logical_and(pred, true).sum())
    union = int(np.logical_or(pred, true).sum())
    pred_area = int(pred.sum())
    true_area = int(true.sum())
    iou = inter / union if union else 1.0
    dice = 2.0 * inter / (pred_area + true_area) if (pred_area + true_area) else 1.0
    area_error = abs(pred_area - true_area) / max(true_area, 1)
    if pred_area > 0 and true_area > 0:
        py, px = np.where(pred)
        ty, tx = np.where(true)
        center_error_px = math.hypot(float(px.mean() - tx.mean()), float(py.mean() - ty.mean()))
    else:
        center_error_px = math.nan
    return {
        "iou": float(iou),
        "dice": float(dice),
        "area_error": float(area_error),
        "pred_area": float(pred_area),
        "true_area": float(true_area),
        "center_error_px": float(center_error_px),
    }


def evaluate_geometries(
    ctx: Context,
    records: list[Record],
    geom: np.ndarray,
    stage: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    type_prob = np.stack([r.type_prob for r in records], axis=0).astype(np.float32)
    observed = np.stack([r.observed_delta for r in records], axis=0).astype(np.float32)
    probs = rasterize_mixture(geom, type_prob, ctx.mask_x, ctx.mask_y, device=torch.device("cpu"))
    with torch.no_grad():
        pred_norm = predict_norm(
            ctx.surrogate_bundle.model,
            torch.tensor(geom, dtype=torch.float32, device=device),
            torch.tensor(type_prob, dtype=torch.float32, device=device),
            torch.tensor(ctx.geom_mean8, dtype=torch.float32, device=device),
            torch.tensor(ctx.geom_std8, dtype=torch.float32, device=device),
        ).cpu().numpy()
    forward_mse, forward_nrmse, forward_corr = per_sample_forward_metrics(pred_norm, observed)
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        mm = mask_metrics_from_prob(probs[i], record.true_mask)
        true_angle_deg = math.degrees(float(record.true_geom[5]))
        angle_deg = math.degrees(float(geom[i, 5]))
        angle_error = angle_diff_deg(angle_deg, true_angle_deg) if record.defect_type == "rotated_rect" else math.nan
        drift = geom[i] - record.init_geom
        drift_norm = float(np.linalg.norm(drift / ctx.geom_scale6))
        rows.append(
            {
                "sample_id": record.sample_id,
                "source_index": record.source_index,
                "split": record.split,
                "defect_type": record.defect_type,
                "stage": stage,
                "iou": mm["iou"],
                "dice": mm["dice"],
                "area_error": mm["area_error"],
                "center_error_px": mm["center_error_px"],
                "pred_area": mm["pred_area"],
                "true_area": mm["true_area"],
                "forward_mse": float(forward_mse[i]),
                "forward_nrmse": float(forward_nrmse[i]),
                "forward_correlation": float(forward_corr[i]),
                "true_center_x": float(record.true_geom[0]),
                "true_center_y": float(record.true_geom[1]),
                "center_x": float(geom[i, 0]),
                "center_y": float(geom[i, 1]),
                "center_abs_error_m": float(math.hypot(geom[i, 0] - record.true_geom[0], geom[i, 1] - record.true_geom[1])),
                "true_width": float(record.true_geom[2]),
                "width": float(geom[i, 2]),
                "width_abs_error_m": float(abs(geom[i, 2] - record.true_geom[2])),
                "true_length": float(record.true_geom[3]),
                "length": float(geom[i, 3]),
                "length_abs_error_m": float(abs(geom[i, 3] - record.true_geom[3])),
                "true_depth": float(record.true_geom[4]),
                "depth": float(geom[i, 4]),
                "depth_abs_error_m": float(abs(geom[i, 4] - record.true_geom[4])),
                "true_angle_deg": true_angle_deg,
                "angle_deg": angle_deg,
                "angle_abs_error_deg": angle_error,
                "type_prob_rectangular_notch": float(record.type_prob[0]),
                "type_prob_rotated_rect": float(record.type_prob[1]),
                "parameter_drift_norm": drift_norm,
            }
        )
    return rows


def refine_records(
    ctx: Context,
    records: list[Record],
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    init = np.stack([record.init_geom for record in records], axis=0).astype(np.float32)
    type_prob = np.stack([record.type_prob for record in records], axis=0).astype(np.float32)
    observed = np.stack([record.observed_delta for record in records], axis=0).astype(np.float32)
    bounds_min = torch.tensor(ctx.bounds_min, dtype=torch.float32, device=device)
    bounds_max = torch.tensor(ctx.bounds_max, dtype=torch.float32, device=device)
    geom_scale = torch.tensor(ctx.geom_scale6, dtype=torch.float32, device=device)
    geom_mean8 = torch.tensor(ctx.geom_mean8, dtype=torch.float32, device=device)
    geom_std8 = torch.tensor(ctx.geom_std8, dtype=torch.float32, device=device)
    init_t = torch.tensor(init, dtype=torch.float32, device=device)
    type_t = torch.tensor(type_prob, dtype=torch.float32, device=device)
    observed_t = torch.tensor(observed, dtype=torch.float32, device=device)
    params = torch.nn.Parameter(init_t.clone())
    optimizer = torch.optim.Adam([params], lr=float(config["lr"]))
    for _step in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        geom = constrain_geom(params, bounds_min, bounds_max)
        pred = predict_norm(ctx.surrogate_bundle.model, geom, type_t, geom_mean8, geom_std8)
        forward = waveform_loss(pred, observed_t)
        prior = F.smooth_l1_loss((geom - init_t) / geom_scale, torch.zeros_like(geom))
        drift = torch.sqrt(torch.mean(((geom - init_t) / geom_scale).square(), dim=1) + 1.0e-12)
        ordering_guard = torch.relu(drift - 1.5).square().mean()
        loss = forward + float(config["lambda_prior"]) * prior + 0.05 * ordering_guard
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            params.copy_(constrain_geom(params, bounds_min, bounds_max))
    return constrain_geom(params, bounds_min, bounds_max).detach().cpu().numpy().astype(np.float32)


def aggregate(rows: list[dict[str, Any]], keys: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(keys)
    out["n"] = len(rows)
    for key in [
        "iou",
        "dice",
        "area_error",
        "center_error_px",
        "forward_mse",
        "forward_nrmse",
        "forward_correlation",
        "center_abs_error_m",
        "width_abs_error_m",
        "length_abs_error_m",
        "depth_abs_error_m",
        "angle_abs_error_deg",
        "parameter_drift_norm",
    ]:
        out[key] = safe_mean([to_float(row.get(key)) for row in rows])
    return out


def paired_rows(pre: list[dict[str, Any]], post: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pre_by_id = {row["sample_id"]: row for row in pre}
    pairs: list[dict[str, Any]] = []
    for row in post:
        before = pre_by_id[row["sample_id"]]
        delta_iou = to_float(row["iou"]) - to_float(before["iou"])
        delta_dice = to_float(row["dice"]) - to_float(before["dice"])
        delta_area = to_float(row["area_error"]) - to_float(before["area_error"])
        forward_reduction = to_float(before["forward_nrmse"]) - to_float(row["forward_nrmse"])
        angle_delta = to_float(row["angle_abs_error_deg"]) - to_float(before["angle_abs_error_deg"])
        mismatch = int(forward_reduction > 0 and delta_iou < 0 and delta_dice < 0)
        pairs.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "defect_type": row["defect_type"],
                "delta_iou": delta_iou,
                "delta_dice": delta_dice,
                "delta_area_error": delta_area,
                "forward_nrmse_reduction": forward_reduction,
                "angle_error_delta": angle_delta,
                "parameter_drift_norm": row["parameter_drift_norm"],
                "mismatch_flag": mismatch,
            }
        )
    return pairs


def summarize_pairs(pairs: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    mismatch_rate = safe_mean([to_float(row["mismatch_flag"]) for row in pairs])
    drift_values = [to_float(row["parameter_drift_norm"]) for row in pairs]
    excessive = safe_mean([1.0 if value > 1.5 else 0.0 for value in drift_values if math.isfinite(value)])
    return {
        "steps": config.get("steps") if config else "",
        "lr": config.get("lr") if config else "",
        "lambda_prior": config.get("lambda_prior") if config else "",
        "delta_iou": safe_mean([to_float(row["delta_iou"]) for row in pairs]),
        "delta_dice": safe_mean([to_float(row["delta_dice"]) for row in pairs]),
        "delta_area_error": safe_mean([to_float(row["delta_area_error"]) for row in pairs]),
        "forward_nrmse_reduction": safe_mean([to_float(row["forward_nrmse_reduction"]) for row in pairs]),
        "angle_error_delta": safe_mean([to_float(row["angle_error_delta"]) for row in pairs]),
        "parameter_drift_norm": safe_mean(drift_values),
        "mismatch_flag_rate": mismatch_rate,
        "excessive_parameter_drift_flag": excessive,
    }


def config_score(summary: dict[str, Any]) -> float:
    return (
        to_float(summary["delta_iou"], 0.0)
        + to_float(summary["delta_dice"], 0.0)
        - max(0.0, to_float(summary["delta_area_error"], 0.0))
        + 0.15 * to_float(summary["forward_nrmse_reduction"], 0.0)
        - 0.10 * to_float(summary["excessive_parameter_drift_flag"], 0.0)
        - 0.15 * to_float(summary["mismatch_flag_rate"], 0.0)
    )


def run_refinement(ctx: Context, device: torch.device) -> dict[str, Any]:
    val_records = split_records(ctx.records, "val")
    init_val = np.stack([record.init_geom for record in val_records], axis=0).astype(np.float32)
    val_pre = evaluate_geometries(ctx, val_records, init_val, "pre_refine", device)
    configs = [
        {"steps": steps, "lr": lr, "lambda_prior": prior}
        for steps in [20, 50]
        for lr in [0.003, 0.01]
        for prior in [0.05, 0.10]
    ]
    config_rows: list[dict[str, Any]] = []
    refined_by_config: dict[tuple[int, float, float], np.ndarray] = {}
    for config in configs:
        refined = refine_records(ctx, val_records, config, device)
        refined_by_config[(config["steps"], config["lr"], config["lambda_prior"])] = refined
        val_post = evaluate_geometries(ctx, val_records, refined, "post_refine", device)
        pairs = paired_rows(val_pre, val_post)
        summary = summarize_pairs(pairs, config)
        score = config_score(summary)
        improves_mask = summary["delta_iou"] >= 0 or summary["delta_dice"] >= 0
        row = {
            **summary,
            "val_refinement_score": score,
            "mask_non_degrading": improves_mask,
            "selected": False,
        }
        config_rows.append(row)

    viable = [
        row
        for row in config_rows
        if (str(row["mask_non_degrading"]).lower() == "true" or row["mask_non_degrading"] is True)
        and to_float(row["mismatch_flag_rate"], 1.0) < 0.50
    ]
    if viable:
        selected = max(viable, key=lambda row: to_float(row["val_refinement_score"]))
        selection_note = "Selected from val configs with non-degrading IoU or Dice and mismatch_rate < 0.50."
    else:
        selected = max(config_rows, key=lambda row: to_float(row["val_refinement_score"]))
        selection_note = "All val configs degraded mask or had high mismatch; selected highest score for diagnostic only."
    for row in config_rows:
        row["selected"] = (
            int(row["steps"]) == int(selected["steps"])
            and abs(float(row["lr"]) - float(selected["lr"])) < 1.0e-12
            and abs(float(row["lambda_prior"]) - float(selected["lambda_prior"])) < 1.0e-12
        )
    write_csv(CONFIG_SWEEP_CSV, config_rows)

    selected_config = {
        "steps": int(selected["steps"]),
        "lr": float(selected["lr"]),
        "lambda_prior": float(selected["lambda_prior"]),
    }
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        records = split_records(ctx.records, split)
        init = np.stack([record.init_geom for record in records], axis=0).astype(np.float32)
        pre_rows = evaluate_geometries(ctx, records, init, "pre_refine", device)
        refined = refine_records(ctx, records, selected_config, device)
        post_rows = evaluate_geometries(ctx, records, refined, "post_refine", device)
        for row in pre_rows + post_rows:
            row["selected_steps"] = selected_config["steps"]
            row["selected_lr"] = selected_config["lr"]
            row["selected_lambda_prior"] = selected_config["lambda_prior"]
        metric_rows.extend(pre_rows)
        metric_rows.extend(post_rows)
        pair_rows.extend(paired_rows(pre_rows, post_rows))
    write_csv(METRICS_CSV, metric_rows)
    return {
        "selected_config": selected_config,
        "selection_note": selection_note,
        "metric_rows": metric_rows,
        "pair_rows": pair_rows,
        "config_rows": config_rows,
    }


def summary_tables(ctx: Context, result: dict[str, Any]) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = result["metric_rows"]
    pair_rows: list[dict[str, Any]] = result["pair_rows"]
    group_rows: list[dict[str, Any]] = []
    for stage in ["pre_refine", "post_refine"]:
        for split in ["train", "val", "test"]:
            rows = [row for row in metric_rows if row["stage"] == stage and row["split"] == split]
            group_rows.append(aggregate(rows, {"stage": stage, "split": split, "group": "all"}))
            for defect_type in sorted(MAIN_TYPES):
                group_rows.append(
                    aggregate(
                        [row for row in rows if row["defect_type"] == defect_type],
                        {"stage": stage, "split": split, "group": defect_type},
                    )
                )
    write_csv(GROUP_SUMMARY_CSV, group_rows)

    geometry_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        pairs = [row for row in pair_rows if row["split"] == split]
        geometry_rows.append({"split": split, **summarize_pairs(pairs)})
        forward_rows.append(
            {
                "split": split,
                "forward_nrmse_reduction": safe_mean([to_float(row["forward_nrmse_reduction"]) for row in pairs]),
                "mismatch_flag_rate": safe_mean([to_float(row["mismatch_flag"]) for row in pairs]),
                "residual_reduction_vs_delta_iou_corr": safe_corr(
                    [to_float(row["forward_nrmse_reduction"]) for row in pairs],
                    [to_float(row["delta_iou"]) for row in pairs],
                ),
                "residual_reduction_vs_delta_dice_corr": safe_corr(
                    [to_float(row["forward_nrmse_reduction"]) for row in pairs],
                    [to_float(row["delta_dice"]) for row in pairs],
                ),
                "residual_reduction_vs_delta_area_error_corr": safe_corr(
                    [to_float(row["forward_nrmse_reduction"]) for row in pairs],
                    [to_float(row["delta_area_error"]) for row in pairs],
                ),
            }
        )
    write_csv(GEOMETRY_SUMMARY_CSV, geometry_rows)
    write_csv(FORWARD_SUMMARY_CSV, forward_rows)

    failure_cases = sorted(
        pair_rows,
        key=lambda row: (
            -int(to_float(row["mismatch_flag"], 0)),
            to_float(row["delta_iou"], 0),
            to_float(row["delta_dice"], 0),
        ),
    )[:80]
    write_csv(FAILURE_CASES_CSV, failure_cases)

    interpret_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_pairs = [row for row in pair_rows if row["split"] == split]
        for defect_type in ["all", *sorted(MAIN_TYPES)]:
            rows = split_pairs if defect_type == "all" else [row for row in split_pairs if row["defect_type"] == defect_type]
            interpret_rows.append(
                {
                    "split": split,
                    "defect_type": defect_type,
                    "n": len(rows),
                    "residual_reduction_vs_delta_iou_corr": safe_corr(
                        [to_float(row["forward_nrmse_reduction"]) for row in rows],
                        [to_float(row["delta_iou"]) for row in rows],
                    ),
                    "residual_reduction_vs_delta_dice_corr": safe_corr(
                        [to_float(row["forward_nrmse_reduction"]) for row in rows],
                        [to_float(row["delta_dice"]) for row in rows],
                    ),
                    "residual_reduction_vs_area_error_delta_corr": safe_corr(
                        [to_float(row["forward_nrmse_reduction"]) for row in rows],
                        [to_float(row["delta_area_error"]) for row in rows],
                    ),
                    "residual_reduction_vs_angle_error_delta_corr": safe_corr(
                        [to_float(row["forward_nrmse_reduction"]) for row in rows],
                        [to_float(row["angle_error_delta"]) for row in rows],
                    ),
                    "mean_delta_iou": safe_mean([to_float(row["delta_iou"]) for row in rows]),
                    "mean_delta_dice": safe_mean([to_float(row["delta_dice"]) for row in rows]),
                    "mean_forward_nrmse_reduction": safe_mean(
                        [to_float(row["forward_nrmse_reduction"]) for row in rows]
                    ),
                    "mismatch_flag_rate": safe_mean([to_float(row["mismatch_flag"]) for row in rows]),
                }
            )
    write_csv(INTERPRETABILITY_CSV, interpret_rows)
    return {
        "group_rows": group_rows,
        "geometry_rows": geometry_rows,
        "forward_rows": forward_rows,
        "interpret_rows": interpret_rows,
        "failure_cases": failure_cases,
    }


def get_group(group_rows: list[dict[str, Any]], stage: str, split: str, group: str = "all") -> dict[str, Any]:
    return next(row for row in group_rows if row["stage"] == stage and row["split"] == split and row["group"] == group)


def write_summaries(ctx: Context, result: dict[str, Any], tables: dict[str, Any]) -> None:
    group_rows = tables["group_rows"]
    forward_rows = tables["forward_rows"]
    selected_config = result["selected_config"]
    test_pre = get_group(group_rows, "pre_refine", "test")
    test_post = get_group(group_rows, "post_refine", "test")
    val_pre = get_group(group_rows, "pre_refine", "val")
    val_post = get_group(group_rows, "post_refine", "val")
    train_pre = get_group(group_rows, "pre_refine", "train")
    train_post = get_group(group_rows, "post_refine", "train")
    forward_test = next(row for row in forward_rows if row["split"] == "test")
    forward_val = next(row for row in forward_rows if row["split"] == "val")
    promising = (
        to_float(test_post["iou"]) >= to_float(test_pre["iou"]) - 1.0e-9
        and to_float(test_post["dice"]) >= to_float(test_pre["dice"]) - 1.0e-9
        and to_float(forward_test["forward_nrmse_reduction"]) > 0
        and to_float(forward_val["forward_nrmse_reduction"]) > 0
        and to_float(forward_test["mismatch_flag_rate"]) < 0.50
    )
    summary_lines = [
        "20.57 controlled Priewald refinement retry with perturbation-calibrated surrogate",
        "",
        "Scope: controlled retry only; no COMSOL run, no new perturbation data, no inverse geometry head, no dense baseline.",
        "Initialization: 20.54 improved dense/extracted geometry proposal.",
        "Forward objective: frozen S1_perturb_geom_mlp retrained from the 20.56 perturbation pack protocol.",
        "Type handling: fixed 20.54 proposal type probabilities are used as soft surrogate input; true type is not used for routing or optimization.",
        "Optimization uses observed delta_bz and frozen surrogate residual only. True mask / geometry are used only for validation selection and final metrics.",
        "",
        f"S1 recovery val/test ordering: {ctx.recovery_selected['val_surrogate_ordering_accuracy']:.6f} / {ctx.recovery_selected['test_surrogate_ordering_accuracy']:.6f}",
        f"S1 recovery val/test mismatch: {ctx.recovery_selected['val_surrogate_mismatch_rate']:.6f} / {ctx.recovery_selected['test_surrogate_mismatch_rate']:.6f}",
        f"S1 recovery test residual-error correlation caveat: {ctx.recovery_selected['test_surrogate_residual_error_correlation']:.6f}",
        "",
        f"Selected refinement config: steps={selected_config['steps']}, lr={selected_config['lr']}, lambda_prior={selected_config['lambda_prior']}",
        f"Selection note: {result['selection_note']}",
        "",
        "Geometry-raster mask metrics:",
        f"- train pre/post IoU/Dice/area_error: {train_pre['iou']:.6f}/{train_pre['dice']:.6f}/{train_pre['area_error']:.6f} -> {train_post['iou']:.6f}/{train_post['dice']:.6f}/{train_post['area_error']:.6f}",
        f"- val pre/post IoU/Dice/area_error: {val_pre['iou']:.6f}/{val_pre['dice']:.6f}/{val_pre['area_error']:.6f} -> {val_post['iou']:.6f}/{val_post['dice']:.6f}/{val_post['area_error']:.6f}",
        f"- test pre/post IoU/Dice/area_error: {test_pre['iou']:.6f}/{test_pre['dice']:.6f}/{test_pre['area_error']:.6f} -> {test_post['iou']:.6f}/{test_post['dice']:.6f}/{test_post['area_error']:.6f}",
        "",
        "Forward residual:",
        f"- val mean forward NRMSE reduction: {forward_val['forward_nrmse_reduction']:.6f}; mismatch_rate={forward_val['mismatch_flag_rate']:.6f}",
        f"- test mean forward NRMSE reduction: {forward_test['forward_nrmse_reduction']:.6f}; mismatch_rate={forward_test['mismatch_flag_rate']:.6f}",
        "",
        "20.54 reference:",
        f"- 20.54 extracted proposal test IoU/Dice = {REFERENCE_2054['proposal_test_iou']:.4f} / {REFERENCE_2054['proposal_test_dice']:.4f}",
        f"- 20.54 old refined test IoU/Dice = {REFERENCE_2054['old_refined_test_iou']:.4f} / {REFERENCE_2054['old_refined_test_dice']:.4f}",
        f"- 20.54 old forward test NRMSE = {REFERENCE_2054['old_forward_initial_nrmse']:.4f} -> {REFERENCE_2054['old_forward_refined_nrmse']:.4f}",
        "",
        f"20.57 promising gate passed: {promising}",
    ]
    if promising:
        summary_lines.append("Interpretation: the perturbation-calibrated surrogate made refinement non-degrading on test under this controlled retry, but the 96-row partial pack is still too small for baseline claims.")
    else:
        summary_lines.append("Interpretation: pairwise residual ordering improvement did not reliably transfer into continuous low-dimensional refinement; do not continue tuning this setup blindly.")
    RETRY_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    RETRY_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    fail_lines = [
        "20.57 perturbation-calibrated refinement retry failure audit",
        "",
        f"Promising gate passed: {promising}",
        f"Test IoU delta: {to_float(test_post['iou']) - to_float(test_pre['iou']):.6f}",
        f"Test Dice delta: {to_float(test_post['dice']) - to_float(test_pre['dice']):.6f}",
        f"Test area_error delta: {to_float(test_post['area_error']) - to_float(test_pre['area_error']):.6f}",
        f"Test angle error delta: {to_float(test_post['angle_abs_error_deg']) - to_float(test_pre['angle_abs_error_deg']):.6f}",
        f"Test parameter drift norm: {test_post['parameter_drift_norm']:.6f}",
        f"Test forward NRMSE reduction: {forward_test['forward_nrmse_reduction']:.6f}",
        f"Test mismatch rate: {forward_test['mismatch_flag_rate']:.6f}",
        "",
        "Failure mode judgement:",
    ]
    if to_float(forward_test["forward_nrmse_reduction"]) > 0 and (
        to_float(test_post["iou"]) < to_float(test_pre["iou"]) or to_float(test_post["dice"]) < to_float(test_pre["dice"])
    ):
        fail_lines.append("- Surrogate mismatch / representation limit remains: forward residual improves while mask metric degrades.")
    elif to_float(forward_test["forward_nrmse_reduction"]) <= 0:
        fail_lines.append("- The calibrated residual objective did not consistently reduce forward residual on test.")
    else:
        fail_lines.append("- No hard blocker in this controlled retry, but coverage is limited by the 96-row perturbation pack.")
    fail_lines.append("- True mask / true geometry were not used in optimization; they only support validation selection and final metrics.")
    FAILURE_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    FAILURE_SUMMARY.write_text("\n".join(fail_lines) + "\n", encoding="utf-8")

    interp_test = next(row for row in tables["interpret_rows"] if row["split"] == "test" and row["defect_type"] == "all")
    interp_lines = [
        "20.57 perturbation-calibrated residual interpretability summary",
        "",
        "Question 1: Did perturbation-calibrated residual become more refinement-useful?",
        f"- Test residual reduction vs IoU/Dice delta correlation = {interp_test['residual_reduction_vs_delta_iou_corr']:.6f} / {interp_test['residual_reduction_vs_delta_dice_corr']:.6f}",
        f"- Test mismatch rate = {interp_test['mismatch_flag_rate']:.6f}",
        "",
        "Question 2: Does pairwise ordering improvement transfer to continuous optimization?",
        f"- Test pre/post IoU/Dice = {test_pre['iou']:.6f}/{test_pre['dice']:.6f} -> {test_post['iou']:.6f}/{test_post['dice']:.6f}",
        f"- Test forward NRMSE reduction = {forward_test['forward_nrmse_reduction']:.6f}",
        "",
        "Question 3: Remaining failure mode:",
    ]
    if promising:
        interp_lines.append("- Pairwise ordering partially transferred to refinement, but the conclusion remains limited by partial perturbation coverage.")
    else:
        interp_lines.append("- Pairwise ordering did not robustly transfer; likely residual objective / representation limit or non-identifiability remains.")
    INTERPRETABILITY_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    INTERPRETABILITY_SUMMARY.write_text("\n".join(interp_lines) + "\n", encoding="utf-8")


def maybe_write_previews(ctx: Context, result: dict[str, Any], device: torch.device, preview_count: int) -> None:
    if preview_count <= 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    metric_rows = result["metric_rows"]
    pair_rows = result["pair_rows"]
    post_by_id = {row["sample_id"]: row for row in metric_rows if row["stage"] == "post_refine"}
    pre_by_id = {row["sample_id"]: row for row in metric_rows if row["stage"] == "pre_refine"}
    rec_by_id = {record.sample_id: record for record in ctx.records}
    chosen = sorted(
        [row for row in pair_rows if row["split"] in {"val", "test"}],
        key=lambda row: (int(to_float(row["mismatch_flag"], 0)), abs(to_float(row["delta_iou"], 0))),
        reverse=True,
    )[:preview_count]
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for idx, pair in enumerate(chosen):
        rec = rec_by_id[pair["sample_id"]]
        pre = pre_by_id[pair["sample_id"]]
        post = post_by_id[pair["sample_id"]]
        init_geom = rec.init_geom.reshape(1, 6)
        refined_geom = np.array(
            [[post["center_x"], post["center_y"], post["width"], post["length"], post["depth"], math.radians(post["angle_deg"])]],
            dtype=np.float32,
        )
        type_prob = rec.type_prob.reshape(1, 2)
        pre_prob = rasterize_mixture(init_geom, type_prob, ctx.mask_x, ctx.mask_y)[0]
        post_prob = rasterize_mixture(refined_geom, type_prob, ctx.mask_x, ctx.mask_y)[0]
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes[0, 0].imshow(rec.true_mask, cmap="gray")
        axes[0, 0].set_title("true mask")
        axes[0, 1].imshow(pre_prob >= 0.5, cmap="gray")
        axes[0, 1].set_title(f"initial IoU {pre['iou']:.3f}")
        axes[0, 2].imshow(post_prob >= 0.5, cmap="gray")
        axes[0, 2].set_title(f"refined IoU {post['iou']:.3f}")
        axes[1, 0].imshow(rec.true_mask, cmap="gray")
        axes[1, 0].imshow(pre_prob >= 0.5, cmap="Reds", alpha=0.35)
        axes[1, 0].set_title("initial overlay")
        axes[1, 1].imshow(rec.true_mask, cmap="gray")
        axes[1, 1].imshow(post_prob >= 0.5, cmap="Blues", alpha=0.35)
        axes[1, 1].set_title("refined overlay")
        for line in rec.observed_delta:
            axes[1, 2].plot(line)
        axes[1, 2].set_title(f"dBz; dIoU {pair['delta_iou']:.3f}")
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{rec.sample_id} {rec.split} {rec.defect_type} "
            f"fwd {pair['forward_nrmse_reduction']:.3f} mismatch={pair['mismatch_flag']}"
        )
        fig.tight_layout()
        fig.savefig(PREVIEW_DIR / f"{idx:02d}_{rec.sample_id}.png", dpi=120)
        plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(SEED)
    device = choose_device(args.device)
    ctx = build_context(args, device)
    result = run_refinement(ctx, device)
    tables = summary_tables(ctx, result)
    write_summaries(ctx, result, tables)
    maybe_write_previews(ctx, result, device, args.preview_count)
    return {"ctx": ctx, "result": result, "tables": tables}


def main() -> None:
    output = run(parse_args())
    selected = output["result"]["selected_config"]
    test_pre = get_group(output["tables"]["group_rows"], "pre_refine", "test")
    test_post = get_group(output["tables"]["group_rows"], "post_refine", "test")
    print(
        "Selected config: "
        f"steps={selected['steps']} lr={selected['lr']} lambda_prior={selected['lambda_prior']}"
    )
    print(
        "Test IoU/Dice: "
        f"{test_pre['iou']:.4f}/{test_pre['dice']:.4f} -> "
        f"{test_post['iou']:.4f}/{test_post['dice']:.4f}"
    )


if __name__ == "__main__":
    main()

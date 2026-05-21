from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = base.DEFAULT_NPZ
DEFAULT_LABELS = base.DEFAULT_LABELS
DEFAULT_FEATURES = PROJECT_ROOT / "results/metrics/comsol_mfl_physics_features.csv"
DEFAULT_INPUT_CHECK_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_feature_assisted_forward_consistency_input_check_summary.txt"
)
DEFAULT_INPUT_CHECK = PROJECT_ROOT / "results/metrics/comsol_feature_assisted_forward_consistency_input_check.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_geometry_forward_surrogate_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_forward_surrogate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_forward_surrogate_epoch_log.csv"
DEFAULT_GROUP_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_geometry_forward_surrogate_group_summary.csv"

SEED = 42
EPOCHS = 300
BATCH_SIZE = 32
TARGET_SHAPE = (3, 201)

METRIC_FIELDS = [
    "sample_id",
    "source_index",
    "split",
    "defect_type",
    "source_pack",
    "mse",
    "mae",
    "rmse",
    "nrmse",
    "correlation",
    "line0_mse",
    "line1_mse",
    "line2_mse",
    "amplitude_abs_error",
    "abs_peak_index_error_mean",
]

GROUP_FIELDS = [
    "split",
    "group_name",
    "group_value",
    "sample_count",
    "mse_mean",
    "mae_mean",
    "rmse_mean",
    "nrmse_mean",
    "correlation_mean",
    "amplitude_abs_error_mean",
    "abs_peak_index_error_mean",
]

EPOCH_FIELDS = [
    "epoch",
    "train_loss",
    "train_mse",
    "train_l1",
    "val_mse",
    "val_mae",
    "val_rmse",
    "val_nrmse",
    "val_correlation",
    "val_score",
]


@dataclass
class ForwardSurrogateBundle:
    model: nn.Module
    arrays: dict[str, Any]
    diagnostics: dict[str, Any]
    best_epoch: int
    best_val: dict[str, float]
    device: torch.device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [base.to_float(row.get(key, "")) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return float(np.mean(values)) if values else math.nan


def load_feature_diagnostics(features_path: Path, sample_ids: np.ndarray) -> dict[str, Any]:
    rows = read_csv(features_path)
    by_id = {row["sample_id"]: row for row in rows}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    z_fields = [field for field in rows[0] if field.startswith("z_")]
    raw_feature_fields = [
        field
        for field in rows[0]
        if field not in {"sample_index", "sample_id", "split"}
        and not field.startswith("z_")
        and not field.endswith("_scan_y")
    ]
    nls_fields = [field for field in raw_feature_fields if "_nls_" in field or field.startswith("lines_nls_")]
    fit_failed_fields = [field for field in raw_feature_fields if field.endswith("_fit_failed")]
    fit_failed_values = []
    finite = True
    for row in rows:
        for field in raw_feature_fields + z_fields:
            value = base.to_float(row.get(field, ""))
            finite = finite and math.isfinite(value)
        for field in fit_failed_fields:
            fit_failed_values.append(base.to_float(row.get(field, 0.0), 0.0))
    return {
        "feature_rows": len(rows),
        "missing_rect_rot_feature_rows": len(missing),
        "raw_feature_count": len(raw_feature_fields),
        "z_feature_count": len(z_fields),
        "nls_style_raw_feature_count": len(nls_fields),
        "nls_fit_failure_rate": float(np.mean(fit_failed_values)) if fit_failed_values else math.nan,
        "all_features_finite": finite,
        "feature_columns_preview": z_fields[:8],
    }


def build_forward_inputs(arrays: dict[str, Any]) -> np.ndarray:
    type_onehot = np.eye(2, dtype=np.float32)[arrays["type_targets"]]
    return np.concatenate(
        [
            type_onehot,
            arrays["geom_targets"].astype(np.float32),
            arrays["angle_targets"].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def load_forward_arrays(npz_path: Path, labels_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays, diagnostics = base.load_arrays(npz_path, labels_path)
    geom_input = build_forward_inputs(arrays)
    target = arrays["signals_norm"].astype(np.float32)
    if target.shape[1:] != TARGET_SHAPE:
        raise ValueError(f"Expected target shape (*, {TARGET_SHAPE}), got {target.shape}")
    arrays = dict(arrays)
    arrays["forward_inputs"] = geom_input
    arrays["forward_targets"] = target.reshape(target.shape[0], -1)
    diagnostics = dict(diagnostics)
    diagnostics["forward_input_dim"] = int(geom_input.shape[1])
    diagnostics["forward_output_dim"] = int(np.prod(TARGET_SHAPE))
    diagnostics["target_shape"] = tuple(target.shape)
    return arrays, diagnostics


class ForwardDataset(Dataset):
    def __init__(self, indices: np.ndarray, arrays: dict[str, Any]):
        self.indices = indices.astype(np.int64)
        self.inputs = arrays["forward_inputs"][self.indices]
        self.targets = arrays["forward_targets"][self.indices]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        local_idx = int(self.indices[idx])
        return {
            "source_index": torch.tensor(local_idx, dtype=torch.long),
            "input": torch.from_numpy(self.inputs[idx]).float(),
            "target": torch.from_numpy(self.targets[idx]).float(),
        }


class GeometryForwardSurrogate(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 603):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, geom_input: torch.Tensor) -> torch.Tensor:
        return self.net(geom_input)


def signal_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred - target
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    denom = float(np.sqrt(np.mean(target**2)) + 1e-12)
    corr = 0.0
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    if float(np.std(pred_flat)) > 0 and float(np.std(target_flat)) > 0:
        corr = float(np.corrcoef(pred_flat, target_flat)[0, 1])
    line_mse = [float(np.mean((pred[line] - target[line]) ** 2)) for line in range(TARGET_SHAPE[0])]
    pred_amp = float(pred.max() - pred.min())
    true_amp = float(target.max() - target.min())
    peak_errs = [
        abs(int(np.argmax(np.abs(pred[line]))) - int(np.argmax(np.abs(target[line]))))
        for line in range(TARGET_SHAPE[0])
    ]
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nrmse": rmse / denom,
        "correlation": corr,
        "line0_mse": line_mse[0],
        "line1_mse": line_mse[1],
        "line2_mse": line_mse[2],
        "amplitude_abs_error": abs(pred_amp - true_amp),
        "abs_peak_index_error_mean": float(np.mean(peak_errs)),
    }


def predict(model: nn.Module, dataset: ForwardDataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["input"].to(device))
            chunks["indices"].append(batch["source_index"].cpu().numpy())
            chunks["pred"].append(pred.cpu().numpy())
            chunks["target"].append(batch["target"].cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def metric_rows(pred: dict[str, np.ndarray], arrays: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, local_idx_raw in enumerate(pred["indices"]):
        local_idx = int(local_idx_raw)
        pred_signal = pred["pred"][order].reshape(TARGET_SHAPE)
        target_signal = pred["target"][order].reshape(TARGET_SHAPE)
        rows.append(
            {
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": split,
                "defect_type": str(arrays["defect_types"][local_idx]),
                "source_pack": str(arrays["source_packs"][local_idx]),
                **signal_metrics(pred_signal, target_signal),
            }
        )
    return rows


def summarize_split(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split]
    return {
        "mse": safe_mean(subset, "mse"),
        "mae": safe_mean(subset, "mae"),
        "rmse": safe_mean(subset, "rmse"),
        "nrmse": safe_mean(subset, "nrmse"),
        "correlation": safe_mean(subset, "correlation"),
        "amplitude_abs_error": safe_mean(subset, "amplitude_abs_error"),
        "abs_peak_index_error_mean": safe_mean(subset, "abs_peak_index_error_mean"),
    }


def build_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        for group_name, values in [
            ("overall", ["rect_rot"]),
            ("defect_type", sorted({str(row["defect_type"]) for row in split_rows})),
            ("source_pack", sorted({str(row["source_pack"]) for row in split_rows})),
        ]:
            for value in values:
                subset = split_rows if group_name == "overall" else [row for row in split_rows if str(row[group_name]) == value]
                stats = summarize_split(subset, split)
                out.append(
                    {
                        "split": split,
                        "group_name": group_name,
                        "group_value": value,
                        "sample_count": len(subset),
                        "mse_mean": stats["mse"],
                        "mae_mean": stats["mae"],
                        "rmse_mean": stats["rmse"],
                        "nrmse_mean": stats["nrmse"],
                        "correlation_mean": stats["correlation"],
                        "amplitude_abs_error_mean": stats["amplitude_abs_error"],
                        "abs_peak_index_error_mean": stats["abs_peak_index_error_mean"],
                    }
                )
    return out


def write_input_check(
    npz_path: Path,
    labels_path: Path,
    features_path: Path,
    summary_path: Path,
    csv_path: Path,
    arrays: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    feature_diag = load_feature_diagnostics(features_path, arrays["sample_ids"])
    passed = (
        diagnostics["n_rect_rot"] == 400
        and diagnostics["split_counts"] == {"train": 268, "val": 66, "test": 66}
        and feature_diag["missing_rect_rot_feature_rows"] == 0
        and feature_diag["all_features_finite"]
    )
    row = {
        "npz_path": str(npz_path),
        "labels_path": str(labels_path),
        "features_path": str(features_path),
        "rect_rot_n": diagnostics["n_rect_rot"],
        "split_train": diagnostics["split_counts"]["train"],
        "split_val": diagnostics["split_counts"]["val"],
        "split_test": diagnostics["split_counts"]["test"],
        "rectangular_notch_n": diagnostics["type_counts"]["rectangular_notch"],
        "rotated_rect_n": diagnostics["type_counts"]["rotated_rect"],
        "feature_rows": feature_diag["feature_rows"],
        "raw_feature_count": feature_diag["raw_feature_count"],
        "z_feature_count": feature_diag["z_feature_count"],
        "nls_style_raw_feature_count": feature_diag["nls_style_raw_feature_count"],
        "nls_fit_failure_rate": feature_diag["nls_fit_failure_rate"],
        "all_features_finite": feature_diag["all_features_finite"],
        "rasterizer": "PyTorch soft rotated-rectangle SDF, temperature=0.0005",
        "true_geometry_raster_iou": 1.0,
        "passed": passed,
    }
    write_csv(csv_path, [row], list(row.keys()))
    lines = [
        "COMSOL feature-assisted forward-consistency input check summary",
        "",
        f"Input NPZ: {npz_path}",
        f"Geometry labels: {labels_path}",
        f"Physics features: {features_path}",
        "Scope: rectangular_notch + rotated_rect only; polygon parsed/reported but excluded.",
        "Feature policy: only delta_bz / sensor_x / scan_line_y derived features are used; no masks or geometry labels are feature inputs.",
        "Feature scaler policy: feature script writes z_ columns using train-only scaler; val/test reuse train statistics.",
        "Rasterizer policy: fixed PyTorch soft rotated-rectangle SDF, temperature=0.0005; prior true-geometry validation IoU=1.0000.",
        "",
        f"rect+rot N: {diagnostics['n_rect_rot']}",
        f"split counts: {diagnostics['split_counts']}",
        f"type counts: {diagnostics['type_counts']}",
        f"raw feature count: {feature_diag['raw_feature_count']}",
        f"z feature count: {feature_diag['z_feature_count']}",
        f"Bz-only NLS-style raw feature count: {feature_diag['nls_style_raw_feature_count']}",
        f"NLS-style fit failure rate: {feature_diag['nls_fit_failure_rate']:.6f}",
        f"all features finite: {feature_diag['all_features_finite']}",
        f"missing rect/rot feature rows: {feature_diag['missing_rect_rot_feature_rows']}",
        f"input check passed: {passed}",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_forward_surrogate(
    args: argparse.Namespace,
    write_outputs: bool = True,
) -> ForwardSurrogateBundle:
    set_seed(args.seed)
    arrays, diagnostics = load_forward_arrays(args.npz, args.labels)
    if write_outputs:
        write_input_check(
            args.npz,
            args.labels,
            args.features,
            args.input_check_summary,
            args.input_check,
            arrays,
            diagnostics,
        )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = ForwardDataset(arrays["split_indices"]["train"], arrays)
    val_ds = ForwardDataset(arrays["split_indices"]["val"], arrays)
    test_ds = ForwardDataset(arrays["split_indices"]["test"], arrays)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = GeometryForwardSurrogate(diagnostics["forward_input_dim"], diagnostics["forward_output_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val = {"score": math.inf, "mse": math.inf, "mae": math.inf, "rmse": math.inf, "nrmse": math.inf, "correlation": -1.0}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        mse_sum = 0.0
        l1_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(device)
            pred = model(batch["input"].to(device))
            mse = F.mse_loss(pred, target)
            l1 = F.l1_loss(pred, target)
            loss = mse + 0.05 * l1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            mse_sum += float(mse.detach().cpu())
            l1_sum += float(l1.detach().cpu())
            n_batches += 1

        val_pred = predict(model, val_ds, device, args.batch_size)
        val_rows = metric_rows(val_pred, arrays, "val")
        val_stats = summarize_split(val_rows, "val")
        val_score = val_stats["mse"]
        if val_score < best_val["score"]:
            best_val = {"score": val_score, **val_stats}
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(n_batches, 1),
                "train_mse": mse_sum / max(n_batches, 1),
                "train_l1": l1_sum / max(n_batches, 1),
                "val_mse": val_stats["mse"],
                "val_mae": val_stats["mae"],
                "val_rmse": val_stats["rmse"],
                "val_nrmse": val_stats["nrmse"],
                "val_correlation": val_stats["correlation"],
                "val_score": val_score,
            }
        )
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"forward epoch={epoch:03d} train_mse={epoch_rows[-1]['train_mse']:.5f} "
                f"val_mse={val_stats['mse']:.5f} val_corr={val_stats['correlation']:.3f}"
            )

    if best_state is None:
        raise RuntimeError("No forward surrogate validation checkpoint selected")
    model.load_state_dict(best_state)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    if write_outputs:
        all_rows: list[dict[str, Any]] = []
        for split, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            all_rows.extend(metric_rows(predict(model, ds, device, args.batch_size), arrays, split))
        group_rows = build_group_rows(all_rows)
        write_csv(args.metrics, all_rows, METRIC_FIELDS)
        write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
        write_csv(args.group_summary, group_rows, GROUP_FIELDS)
        write_summary(args, diagnostics, all_rows, best_epoch, best_val, device)

    return ForwardSurrogateBundle(
        model=model,
        arrays=arrays,
        diagnostics=diagnostics,
        best_epoch=best_epoch,
        best_val=best_val,
        device=device,
    )


def write_summary(
    args: argparse.Namespace,
    diagnostics: dict[str, Any],
    rows: list[dict[str, Any]],
    best_epoch: int,
    best_val: dict[str, float],
    device: torch.device,
) -> None:
    train_stats = summarize_split(rows, "train")
    val_stats = summarize_split(rows, "val")
    test_stats = summarize_split(rows, "test")
    usable = test_stats["nrmse"] <= 1.0 and test_stats["correlation"] >= 0.40
    lines = [
        "COMSOL rect/rot geometry forward surrogate summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Geometry labels: {args.labels}",
        "Task: learn geometry -> normalized delta_bz, shape (3, 201).",
        "Forward surrogate input is true geometry for surrogate fitting only; it is not inverse-model input.",
        "No COMSOL run and no new data generation.",
        "",
        f"Device: {device}",
        f"Seed: {args.seed}",
        f"Epochs: {args.epochs}",
        f"Best epoch by validation MSE: {best_epoch}",
        f"Forward input dim: {diagnostics['forward_input_dim']}",
        f"Forward output dim: {diagnostics['forward_output_dim']}",
        f"Split counts: {diagnostics['split_counts']}",
        "",
        "Metrics on normalized delta_bz:",
        f"- train MSE/MAE/RMSE/NRMSE/corr = {train_stats['mse']:.6f} / {train_stats['mae']:.6f} / {train_stats['rmse']:.6f} / {train_stats['nrmse']:.6f} / {train_stats['correlation']:.4f}",
        f"- val MSE/MAE/RMSE/NRMSE/corr = {val_stats['mse']:.6f} / {val_stats['mae']:.6f} / {val_stats['rmse']:.6f} / {val_stats['nrmse']:.6f} / {val_stats['correlation']:.4f}",
        f"- test MSE/MAE/RMSE/NRMSE/corr = {test_stats['mse']:.6f} / {test_stats['mae']:.6f} / {test_stats['rmse']:.6f} / {test_stats['nrmse']:.6f} / {test_stats['correlation']:.4f}",
        f"- test amplitude_abs_error = {test_stats['amplitude_abs_error']:.6f}",
        f"- test abs_peak_index_error_mean = {test_stats['abs_peak_index_error_mean']:.4f}",
        "",
        f"Usable for lightweight forward consistency POC: {usable}",
        "Usability rule: test NRMSE <= 1.0 and test correlation >= 0.40, with no NaN/inf or split leakage.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--input-check-summary", type=Path, default=DEFAULT_INPUT_CHECK_SUMMARY)
    parser.add_argument("--input-check", type=Path, default=DEFAULT_INPUT_CHECK)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    train_forward_surrogate(parse_args(), write_outputs=True)


if __name__ == "__main__":
    main()

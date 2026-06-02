#!/usr/bin/env python
"""Fit the 25.5 lightweight feature-space forward surrogate.

This is a diagnostic surrogate over compact delta_b-derived features. It fits
only on the train split, selects hyperparameters on validation, and uses test
only for final reporting. It does not train/update the main neural baseline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audit_surface_shape_extension_rbc_oracle_fit import DATASET_ID, ROOT, load_surface_dataset
from build_surface_forward_refinement_target_set import (
    PARAM_NAMES,
    REGISTRY,
    TARGET_MATERIALIZED,
    as_bool,
    as_float,
    mean,
    read_csv,
    write_csv,
)
from load_true_3d_rbc_pilot_dataset import depth_grid_from_params


SUMMARY = ROOT / "results/summaries/surface_feature_space_forward_surrogate_summary.txt"
SURROGATE_METRICS = ROOT / "results/metrics/surface_feature_space_forward_surrogate_metrics.csv"
SURROGATE_VALIDATION = ROOT / "results/metrics/surface_feature_space_forward_surrogate_validation.csv"

PARAM_BOUNDS = np.asarray(
    [
        [0.0015, 0.035],
        [0.00075, 0.018],
        [0.00005, 0.0045],
        [0.03, 10.0],
        [0.03, 10.0],
        [0.03, 10.0],
    ],
    dtype=np.float64,
)

VALIDATION_FIELDS = [
    "candidate_id",
    "descriptor_kind",
    "feature_mode",
    "alpha",
    "train_sample_count",
    "val_sample_count",
    "train_feature_mse_norm",
    "val_feature_mse_norm",
    "val_feature_rmse_norm",
    "selected",
    "selection_policy",
]

METRIC_FIELDS = [
    "selected_surrogate",
    "descriptor_kind",
    "feature_mode",
    "alpha",
    "split",
    "subset",
    "sample_count",
    "feature_mse_norm_mean",
    "feature_rmse_norm_mean",
    "feature_mse_norm_p95",
    "target_observed_feature_count",
    "fit_split_only",
    "hyperparameter_selection_split",
    "test_final_only",
]


@dataclass(frozen=True)
class RidgeSurrogate:
    candidate_id: str
    descriptor_kind: str
    feature_mode: str
    alpha: float
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    coef: np.ndarray
    descriptor_names: list[str]
    model_feature_names: list[str]
    observed_feature_names: list[str]

    def design_from_params(self, params: np.ndarray) -> np.ndarray:
        descriptors, _ = descriptor_matrix(params, self.descriptor_kind)
        return model_feature_matrix(descriptors, self.descriptor_names, self.feature_mode)[0]

    def predict_norm(self, params: np.ndarray) -> np.ndarray:
        design = self.design_from_params(np.asarray(params, dtype=np.float64).reshape(-1, 6))
        x_norm = (design - self.x_mean) / self.x_std
        return x_norm @ self.coef

    def observed_norm(self, observed_features: np.ndarray) -> np.ndarray:
        return (np.asarray(observed_features, dtype=np.float64) - self.y_mean) / self.y_std


def ensure_materialized() -> None:
    if not TARGET_MATERIALIZED.exists():
        raise FileNotFoundError(
            f"materialized target set missing; run scripts/build_surface_forward_refinement_target_set.py first: {TARGET_MATERIALIZED}"
        )


def row_indices(rows: list[dict[str, str]], split: str | None = None, representable_only: bool = False) -> np.ndarray:
    indices: list[int] = []
    for i, row in enumerate(rows):
        if split is not None and row["split"] != split:
            continue
        if representable_only and not as_bool(row["rbc_representable"]):
            continue
        if representable_only and not as_bool(row["suitable_for_six_param_refinement"]):
            continue
        indices.append(i)
    return np.asarray(indices, dtype=np.int64)


def params_from_rows(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    return np.asarray([[as_float(row[f"{prefix}_{name}"]) for name in PARAM_NAMES] for row in rows], dtype=np.float64)


def observed_feature_matrix(delta_b: np.ndarray) -> tuple[np.ndarray, list[str]]:
    data = np.asarray(delta_b, dtype=np.float64)
    if data.ndim != 4 or data.shape[1:3] != (3, 3):
        raise RuntimeError(f"expected delta_b shape (N,3,3,X), got {data.shape}")
    n = data.shape[0]
    x_norm = np.linspace(-1.0, 1.0, data.shape[-1], dtype=np.float64)
    features: list[np.ndarray] = []
    names: list[str] = []
    axis_names = ["Bx", "By", "Bz"]
    line_names = ["y_neg", "y_mid", "y_pos"]
    for axis in range(3):
        for line in range(3):
            signal = data[:, axis, line, :]
            abs_signal = np.abs(signal)
            abs_sum = np.maximum(abs_signal.sum(axis=1), 1.0e-30)
            centroid = (abs_signal * x_norm[None, :]).sum(axis=1) / abs_sum
            width = np.sqrt(np.maximum(((x_norm[None, :] - centroid[:, None]) ** 2 * abs_signal).sum(axis=1) / abs_sum, 0.0))
            grad = np.diff(signal, axis=1)
            peak_index = np.argmax(abs_signal, axis=1)
            prefix = f"{axis_names[axis]}_{line_names[line]}"
            stats = {
                "mean": signal.mean(axis=1),
                "std": signal.std(axis=1),
                "min": signal.min(axis=1),
                "max": signal.max(axis=1),
                "p2p": signal.max(axis=1) - signal.min(axis=1),
                "peak_abs": abs_signal.max(axis=1),
                "abs_mean": abs_signal.mean(axis=1),
                "rms": np.sqrt(np.mean(signal * signal, axis=1)),
                "grad_rms": np.sqrt(np.mean(grad * grad, axis=1)),
                "centroid_abs": centroid,
                "width_abs": width,
                "peak_x_norm": x_norm[peak_index],
                "signed_peak_balance": signal.max(axis=1) + signal.min(axis=1),
            }
            for key, value in stats.items():
                features.append(value.reshape(n, 1))
                names.append(f"{prefix}_{key}")
    stacked = np.concatenate(features, axis=1)
    if not np.all(np.isfinite(stacked)):
        raise RuntimeError("non-finite observed delta_b feature detected")
    return stacked, names


def descriptor_matrix(params: np.ndarray, descriptor_kind: str) -> tuple[np.ndarray, list[str]]:
    p = np.asarray(params, dtype=np.float64).reshape(-1, 6)
    clipped = np.clip(p, PARAM_BOUNDS[:, 0][None, :], PARAM_BOUNDS[:, 1][None, :])
    names: list[str] = []
    chunks: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        chunks.append(np.asarray(values, dtype=np.float64).reshape(-1, 1))

    for col, name in enumerate(PARAM_NAMES):
        add(name, clipped[:, col])
    for col, name in enumerate(PARAM_NAMES):
        add(f"log_{name}", np.log(np.maximum(clipped[:, col], 1.0e-12)))
    add("L_over_W", clipped[:, 0] / np.maximum(clipped[:, 1], 1.0e-12))
    add("D_over_L", clipped[:, 2] / np.maximum(clipped[:, 0], 1.0e-12))
    add("D_over_W", clipped[:, 2] / np.maximum(clipped[:, 1], 1.0e-12))
    add("volume_box_proxy", clipped[:, 0] * clipped[:, 1] * clipped[:, 2])
    add("curvature_weight_mean", clipped[:, 3:6].mean(axis=1))
    add("curvature_weight_std", clipped[:, 3:6].std(axis=1))

    if descriptor_kind == "param_only":
        return np.concatenate(chunks, axis=1), names
    if descriptor_kind != "param_profile":
        raise ValueError(f"unknown descriptor_kind: {descriptor_kind}")

    grids = np.asarray([depth_grid_from_params(row) for row in clipped], dtype=np.float64)
    flat = grids.reshape(grids.shape[0], -1)
    nonzero = flat > 1.0e-9
    u = np.linspace(-1.0, 1.0, grids.shape[1])
    v = np.linspace(-1.0, 1.0, grids.shape[2])
    uu, vv = np.meshgrid(u, v, indexing="ij")
    weight_sum = np.maximum(flat.sum(axis=1), 1.0e-30)
    add("canonical_depth_mean", flat.mean(axis=1))
    add("canonical_depth_std", flat.std(axis=1))
    add("canonical_depth_max", flat.max(axis=1))
    add("canonical_depth_p95", np.percentile(flat, 95, axis=1))
    add("canonical_depth_sum", flat.sum(axis=1))
    add("canonical_nonzero_fraction", nonzero.mean(axis=1))
    add("canonical_u_width", np.sqrt(np.maximum((flat * (uu.reshape(1, -1) ** 2)).sum(axis=1) / weight_sum, 0.0)))
    add("canonical_v_width", np.sqrt(np.maximum((flat * (vv.reshape(1, -1) ** 2)).sum(axis=1) / weight_sum, 0.0)))
    return np.concatenate(chunks, axis=1), names


def model_feature_matrix(descriptors: np.ndarray, descriptor_names: list[str], feature_mode: str) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(descriptors, dtype=np.float64)
    names = list(descriptor_names)
    chunks = [x]
    if feature_mode == "linear":
        return x, names
    if feature_mode == "squared":
        chunks.append(x * x)
        names.extend([f"{name}^2" for name in descriptor_names])
        return np.concatenate(chunks, axis=1), names
    raise ValueError(f"unknown feature_mode: {feature_mode}")


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = np.where(x.std(axis=0, keepdims=True) < 1.0e-12, 1.0, x.std(axis=0, keepdims=True))
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = np.where(y.std(axis=0, keepdims=True) < 1.0e-12, 1.0, y.std(axis=0, keepdims=True))
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    xtx = x_norm.T @ x_norm
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    rhs = x_norm.T @ y_norm
    try:
        coef = np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx + reg) @ rhs
    return x_mean, x_std, y_mean, y_std, coef


def residual_mse_norm(surrogate: RidgeSurrogate, params: np.ndarray, observed_features: np.ndarray) -> np.ndarray:
    pred = surrogate.predict_norm(params)
    target = surrogate.observed_norm(observed_features)
    return np.mean((pred - target) ** 2, axis=1)


def candidate_specs() -> list[tuple[str, str, float]]:
    specs: list[tuple[str, str, float]] = []
    for descriptor_kind, feature_mode in [
        ("param_only", "linear"),
        ("param_profile", "linear"),
        ("param_profile", "squared"),
    ]:
        for alpha in [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0]:
            specs.append((descriptor_kind, feature_mode, alpha))
    return specs


def fit_candidate(
    rows: list[dict[str, str]],
    observed_features: np.ndarray,
    train_idx: np.ndarray,
    descriptor_kind: str,
    feature_mode: str,
    alpha: float,
    observed_feature_names: list[str],
) -> RidgeSurrogate:
    oracle_params = params_from_rows(rows, "oracle")
    train_descriptors, descriptor_names = descriptor_matrix(oracle_params[train_idx], descriptor_kind)
    x_train, model_feature_names = model_feature_matrix(train_descriptors, descriptor_names, feature_mode)
    y_train = observed_features[train_idx]
    x_mean, x_std, y_mean, y_std, coef = fit_ridge(x_train, y_train, alpha)
    candidate_id = f"ridge_{descriptor_kind}_{feature_mode}_alpha_{alpha:g}"
    return RidgeSurrogate(
        candidate_id=candidate_id,
        descriptor_kind=descriptor_kind,
        feature_mode=feature_mode,
        alpha=float(alpha),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        coef=coef,
        descriptor_names=descriptor_names,
        model_feature_names=model_feature_names,
        observed_feature_names=observed_feature_names,
    )


def fit_selected_surrogate() -> tuple[RidgeSurrogate, list[dict[str, str]], np.ndarray, list[str], list[dict[str, Any]]]:
    ensure_materialized()
    rows = read_csv(TARGET_MATERIALIZED)
    dataset = load_surface_dataset(DATASET_ID, REGISTRY)
    observed_features, observed_feature_names = observed_feature_matrix(dataset.delta_b)
    train_idx = row_indices(rows, split="train", representable_only=True)
    val_idx = row_indices(rows, split="val", representable_only=True)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("train/val representable rows required for surrogate fit/selection")

    validation_rows: list[dict[str, Any]] = []
    best: tuple[float, RidgeSurrogate] | None = None
    oracle_params = params_from_rows(rows, "oracle")
    for descriptor_kind, feature_mode, alpha in candidate_specs():
        surrogate = fit_candidate(rows, observed_features, train_idx, descriptor_kind, feature_mode, alpha, observed_feature_names)
        train_mse = residual_mse_norm(surrogate, oracle_params[train_idx], observed_features[train_idx])
        val_mse = residual_mse_norm(surrogate, oracle_params[val_idx], observed_features[val_idx])
        val_score = float(np.mean(val_mse))
        if best is None or val_score < best[0]:
            best = (val_score, surrogate)
        validation_rows.append(
            {
                "candidate_id": surrogate.candidate_id,
                "descriptor_kind": descriptor_kind,
                "feature_mode": feature_mode,
                "alpha": alpha,
                "train_sample_count": len(train_idx),
                "val_sample_count": len(val_idx),
                "train_feature_mse_norm": float(np.mean(train_mse)),
                "val_feature_mse_norm": val_score,
                "val_feature_rmse_norm": float(np.sqrt(val_score)),
                "selected": False,
                "selection_policy": "minimum_validation_normalized_feature_mse; train_fit_only; no_test_selection",
            }
        )
    if best is None:
        raise RuntimeError("no surrogate candidate selected")
    selected = best[1]
    for row in validation_rows:
        row["selected"] = row["candidate_id"] == selected.candidate_id
    return selected, rows, observed_features, observed_feature_names, validation_rows


def subset_indices(rows: list[dict[str, str]], split: str, subset: str) -> np.ndarray:
    selected: list[int] = []
    for i, row in enumerate(rows):
        if split != "all" and row["split"] != split:
            continue
        if subset == "all":
            selected.append(i)
        elif subset == "rbc_representable":
            if as_bool(row["rbc_representable"]):
                selected.append(i)
        elif subset == row["target_role"]:
            selected.append(i)
    return np.asarray(selected, dtype=np.int64)


def p95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95)) if values.size else float("nan")


def build_metric_rows(surrogate: RidgeSurrogate, rows: list[dict[str, str]], observed_features: np.ndarray) -> list[dict[str, Any]]:
    oracle_params = params_from_rows(rows, "oracle")
    out: list[dict[str, Any]] = []
    for split in ["train", "val", "test", "all"]:
        for subset in ["rbc_representable", "refinement_target", "already_pass_reference", "excluded_negative_control", "all"]:
            idx = subset_indices(rows, split, subset)
            if idx.size == 0:
                continue
            mse = residual_mse_norm(surrogate, oracle_params[idx], observed_features[idx])
            out.append(
                {
                    "selected_surrogate": surrogate.candidate_id,
                    "descriptor_kind": surrogate.descriptor_kind,
                    "feature_mode": surrogate.feature_mode,
                    "alpha": surrogate.alpha,
                    "split": split,
                    "subset": subset,
                    "sample_count": int(idx.size),
                    "feature_mse_norm_mean": float(np.mean(mse)),
                    "feature_rmse_norm_mean": float(np.sqrt(np.mean(mse))),
                    "feature_mse_norm_p95": p95(mse),
                    "target_observed_feature_count": len(surrogate.observed_feature_names),
                    "fit_split_only": "train",
                    "hyperparameter_selection_split": "val",
                    "test_final_only": split == "test",
                }
            )
    return out


def write_summary(
    surrogate: RidgeSurrogate,
    rows: list[dict[str, str]],
    validation_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> None:
    selected_validation = [row for row in validation_rows if row["selected"]][0]
    target_rows = [row for row in rows if row["target_role"] == "refinement_target"]
    train_count = sum(1 for row in rows if row["split"] == "train" and as_bool(row["rbc_representable"]))
    val_count = sum(1 for row in rows if row["split"] == "val" and as_bool(row["rbc_representable"]))
    test_count = sum(1 for row in rows if row["split"] == "test" and as_bool(row["rbc_representable"]))
    lines = [
        "25.5 surface feature-space forward surrogate",
        "",
        f"selected_surrogate: {surrogate.candidate_id}",
        f"descriptor_kind: {surrogate.descriptor_kind}",
        f"feature_mode: {surrogate.feature_mode}",
        f"alpha: {surrogate.alpha:g}",
        "surrogate_family: closed-form ridge regression on compact delta_b-derived feature vector",
        f"observed_feature_count: {len(surrogate.observed_feature_names)}",
        f"model_feature_count: {len(surrogate.model_feature_names)}",
        "",
        "split_protocol:",
        f"- train_fit_representable_count: {train_count}",
        f"- validation_selection_representable_count: {val_count}",
        f"- test_final_representable_count: {test_count}",
        "- hyperparameter selection criterion: minimum validation normalized feature MSE.",
        "- no test split metric is used for candidate selection.",
        "",
        "input_boundary:",
        "- Surrogate training may use oracle six params on train split as supervised surrogate inputs.",
        "- Test-time refinement will use observed delta_b-derived features and frozen 20.85 predicted six params only.",
        "- Shape labels, true masks, true depth, and oracle params are not test-time refinement inputs.",
        "",
        f"refinement_target_count: {len(target_rows)}",
        f"target_baseline_profile_rmse_mean_m: {mean(target_rows, 'baseline_profile_depth_rmse_m'):.12g}",
        f"selected_train_feature_mse_norm: {float(selected_validation['train_feature_mse_norm']):.12g}",
        f"selected_val_feature_mse_norm: {float(selected_validation['val_feature_mse_norm']):.12g}",
        f"validation_csv: {SURROGATE_VALIDATION}",
        f"metrics_csv: {SURROGATE_METRICS}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    surrogate, rows, observed_features, _feature_names, validation_rows = fit_selected_surrogate()
    metric_rows = build_metric_rows(surrogate, rows, observed_features)
    write_csv(SURROGATE_VALIDATION, validation_rows, VALIDATION_FIELDS)
    write_csv(SURROGATE_METRICS, metric_rows, METRIC_FIELDS)
    write_summary(surrogate, rows, validation_rows, metric_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

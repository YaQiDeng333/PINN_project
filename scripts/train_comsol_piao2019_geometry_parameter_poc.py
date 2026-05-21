from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "results/metrics/comsol_mfl_physics_features.csv"
DEFAULT_LABELS = PROJECT_ROOT / "results/metrics/comsol_single_defect_geometry_labels.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results/metrics/comsol_piao2019_geometry_parameter_prediction_errors.csv"

RAW_TARGETS = ["center_x", "center_y", "width", "length", "depth", "angle_sin", "angle_cos"]
TRAIN_TARGETS = ["center_x_norm", "center_y_norm", "width_norm", "length_norm", "depth_norm", "angle_sin", "angle_cos"]
POSITIVE_TARGETS = {"width", "length", "depth"}
MAIN_GEOMETRY_TYPES = {"rectangular_notch", "rotated_rect"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _angle_deg_from_sincos(sin_value: float, cos_value: float) -> float:
    return math.degrees(math.atan2(sin_value, cos_value))


def _circular_angle_error_deg(true_deg: float, pred_deg: float) -> float:
    diff = (pred_deg - true_deg + 90.0) % 180.0 - 90.0
    return abs(diff)


def _join_rows(features: list[dict[str, str]], labels: list[dict[str, str]]) -> list[dict[str, str]]:
    labels_by_id = {row["sample_id"]: row for row in labels}
    joined: list[dict[str, str]] = []
    for feat in features:
        sample_id = feat["sample_id"]
        if sample_id not in labels_by_id:
            raise KeyError(f"Missing label row for sample_id={sample_id}")
        row = dict(feat)
        for key, value in labels_by_id[sample_id].items():
            if key in row and key not in {"sample_id", "split"}:
                row[f"label_{key}"] = value
            else:
                row[key] = value
        joined.append(row)
    return joined


def _feature_sets(feature_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    all_z = [key for key in feature_rows[0].keys() if key.startswith("z_")]
    if not all_z:
        raise ValueError("No z_* train-scaled feature columns found")
    generic = [key for key in all_z if "_nls_" not in key]
    if len(generic) == len(all_z):
        raise ValueError("No Bz-only NLS-style z_* features found; revised POC should not run")
    return {
        "generic_only": generic,
        "nls_style_main": all_z,
    }


def _matrix(rows: list[dict[str, str]], feature_cols: list[str]) -> np.ndarray:
    return np.array([[_to_float(row[col], 0.0) for col in feature_cols] for row in rows], dtype=float)


def _targets(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([[_to_float(row[field]) for field in TRAIN_TARGETS] for row in rows], dtype=float)


def _target_inverse_stats(train_rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for raw, norm in zip(RAW_TARGETS, TRAIN_TARGETS):
        if norm in {"angle_sin", "angle_cos"}:
            stats[raw] = (0.0, 1.0)
            continue
        values = np.array([_to_float(row[raw]) for row in train_rows], dtype=float)
        mean = float(values.mean())
        std = float(values.std())
        if std <= 0:
            std = 1.0
        stats[raw] = (mean, std)
    return stats


def _inverse_targets(y_pred: np.ndarray, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    out = np.array(y_pred, dtype=float, copy=True)
    for idx, raw in enumerate(RAW_TARGETS):
        mean, std = stats[raw]
        if raw not in {"angle_sin", "angle_cos"}:
            out[:, idx] = out[:, idx] * std + mean
        if raw in POSITIVE_TARGETS:
            out[:, idx] = np.maximum(out[:, idx], 1e-6)
    return out


def _mean_target_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scale = np.maximum(np.std(y_true, axis=0), 1e-9)
    return float(np.mean(np.abs((y_pred - y_true) / scale)))


def _fit_regressor_by_kind(kind: str, params: dict[str, float], x_train: np.ndarray, y_train: np.ndarray) -> Any:
    if kind == "KernelRidge_rbf":
        model = KernelRidge(alpha=params["alpha"], kernel="rbf", gamma=params["gamma"])
    elif kind == "SVR_rbf":
        model = MultiOutputRegressor(SVR(C=params["C"], gamma=params["gamma"], epsilon=params["epsilon"]))
    elif kind == "Ridge_linear":
        model = Ridge(alpha=params["alpha"])
    else:  # pragma: no cover
        raise ValueError(kind)
    model.fit(x_train, y_train)
    return model


def _select_regressor(
    train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], feature_cols: list[str]
) -> tuple[tuple[StandardScaler, Any], dict[str, Any]]:
    x_train_raw = _matrix(train_rows, feature_cols)
    y_train = _targets(train_rows)
    x_val_raw = _matrix(val_rows, feature_cols)
    y_val = _targets(val_rows)
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train_raw)
    x_val = x_scaler.transform(x_val_raw)

    candidates: list[tuple[str, dict[str, float]]] = []
    for alpha in [0.01, 0.1, 1.0]:
        for gamma in [0.01, 0.1, 1.0]:
            candidates.append(("KernelRidge_rbf", {"alpha": alpha, "gamma": gamma}))
    for alpha in [0.1, 1.0]:
        candidates.append(("Ridge_linear", {"alpha": alpha}))
    for gamma in [0.01, 0.1]:
        candidates.append(("SVR_rbf", {"C": 10.0, "gamma": gamma, "epsilon": 0.02}))

    best_model: Any = None
    best_info: dict[str, Any] = {}
    best_score = float("inf")
    for kind, params in candidates:
        model = _fit_regressor_by_kind(kind, params, x_train, y_train)
        y_val_pred = np.asarray(model.predict(x_val), dtype=float)
        score = _mean_target_error(y_val, y_val_pred)
        if score < best_score:
            best_score = score
            best_model = model
            best_info = {"model": kind, "params": params, "val_normalized_target_mae": score}
    assert best_model is not None
    return (x_scaler, best_model), best_info


def _predict_regression(model_bundle: tuple[StandardScaler, Any], rows: list[dict[str, str]], feature_cols: list[str]) -> np.ndarray:
    x_scaler, model = model_bundle
    x = x_scaler.transform(_matrix(rows, feature_cols))
    return np.asarray(model.predict(x), dtype=float)


def _train_classifier(train_rows: list[dict[str, str]], feature_cols: list[str]) -> tuple[tuple[StandardScaler, LogisticRegression], LabelEncoder]:
    x_train_raw = _matrix(train_rows, feature_cols)
    labels = [row["defect_type"] for row in train_rows]
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(labels)
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train_raw)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto", random_state=2026)
    classifier.fit(x_train, y_train)
    return (x_scaler, classifier), encoder


def _classifier_predictions(
    classifier_bundle: tuple[StandardScaler, LogisticRegression],
    encoder: LabelEncoder,
    rows: list[dict[str, str]],
    feature_cols: list[str],
) -> list[str]:
    x_scaler, classifier = classifier_bundle
    pred = classifier.predict(x_scaler.transform(_matrix(rows, feature_cols)))
    return list(encoder.inverse_transform(pred))


def _build_prediction_rows(
    rows: list[dict[str, str]],
    feature_set_name: str,
    feature_cols: list[str],
    classifier_bundle: tuple[StandardScaler, LogisticRegression],
    encoder: LabelEncoder,
    regressor_bundle: tuple[StandardScaler, Any],
    target_stats: dict[str, tuple[float, float]],
    regressor_info: dict[str, Any],
) -> list[dict[str, Any]]:
    pred_types = _classifier_predictions(classifier_bundle, encoder, rows, feature_cols)
    main_rows = [row for row in rows if row["defect_type"] in MAIN_GEOMETRY_TYPES]
    pred_norm = _predict_regression(regressor_bundle, main_rows, feature_cols)
    pred_raw = _inverse_targets(pred_norm, target_stats)
    pred_by_id = {row["sample_id"]: pred for row, pred in zip(main_rows, pred_raw)}

    out_rows: list[dict[str, Any]] = []
    for row, pred_type in zip(rows, pred_types):
        defect_type = row["defect_type"]
        output: dict[str, Any] = {
            "feature_set": feature_set_name,
            "regressor_model": regressor_info["model"],
            "sample_index": row["sample_index"],
            "sample_id": row["sample_id"],
            "split": row["split"],
            "defect_type": defect_type,
            "source_pack": row.get("source_pack", ""),
            "pred_defect_type": pred_type,
            "type_correct": int(pred_type == defect_type),
            "polygon_in_main_result": 0,
            "notes": "",
        }
        for field in RAW_TARGETS:
            output[f"true_{field}"] = _to_float(row[field])
            output[f"pred_{field}"] = ""
            output[f"{field}_abs_error"] = ""
        output["true_angle_deg"] = _to_float(row["angle_deg"])
        output["pred_angle_deg"] = ""
        output["angle_abs_error_deg"] = ""
        output["center_error"] = ""

        if defect_type in MAIN_GEOMETRY_TYPES:
            pred = pred_by_id[row["sample_id"]]
            pred_map = dict(zip(RAW_TARGETS, pred))
            pred_angle_deg = _angle_deg_from_sincos(pred_map["angle_sin"], pred_map["angle_cos"])
            if pred_type == "rectangular_notch":
                pred_angle_deg = 0.0
                pred_map["angle_sin"] = 0.0
                pred_map["angle_cos"] = 1.0
            true_center = np.array([_to_float(row["center_x"]), _to_float(row["center_y"])], dtype=float)
            pred_center = np.array([pred_map["center_x"], pred_map["center_y"]], dtype=float)
            for field in RAW_TARGETS:
                output[f"pred_{field}"] = float(pred_map[field])
                output[f"{field}_abs_error"] = abs(float(pred_map[field]) - _to_float(row[field]))
            output["pred_angle_deg"] = pred_angle_deg
            if defect_type == "rotated_rect":
                output["angle_abs_error_deg"] = _circular_angle_error_deg(_to_float(row["angle_deg"]), pred_angle_deg)
            else:
                output["angle_abs_error_deg"] = abs(pred_angle_deg)
            output["center_error"] = float(np.linalg.norm(pred_center - true_center))
        else:
            output["notes"] = "polygon_excluded_from_geometry_regression_main_result"
        out_rows.append(output)
    return out_rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for feature_set in sorted(set(row["feature_set"] for row in rows)):
        out[feature_set] = {}
        fs_rows = [row for row in rows if row["feature_set"] == feature_set]
        for split in ["train", "val", "test"]:
            split_rows = [row for row in fs_rows if row["split"] == split]
            main_rows = [row for row in split_rows if row["defect_type"] in MAIN_GEOMETRY_TYPES]
            rot_rows = [row for row in main_rows if row["defect_type"] == "rotated_rect"]
            out[feature_set][split] = {
                "n": float(len(split_rows)),
                "main_n": float(len(main_rows)),
                "type_accuracy": float(np.mean([float(row["type_correct"]) for row in split_rows])) if split_rows else math.nan,
                "center_mae": float(np.mean([float(row["center_error"]) for row in main_rows])) if main_rows else math.nan,
                "width_mae": float(np.mean([float(row["width_abs_error"]) for row in main_rows])) if main_rows else math.nan,
                "length_mae": float(np.mean([float(row["length_abs_error"]) for row in main_rows])) if main_rows else math.nan,
                "depth_mae": float(np.mean([float(row["depth_abs_error"]) for row in main_rows])) if main_rows else math.nan,
                "angle_mae_deg_rotated": float(np.mean([float(row["angle_abs_error_deg"]) for row in rot_rows])) if rot_rows else math.nan,
            }
    return out


def run(features_path: Path, labels_path: Path, out_path: Path) -> dict[str, Any]:
    features = _read_csv(features_path)
    labels = _read_csv(labels_path)
    rows = _join_rows(features, labels)
    feature_sets = _feature_sets(features)
    split_counts = Counter(row["split"] for row in rows)
    defect_counts = Counter(row["defect_type"] for row in rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    main_train = [row for row in train_rows if row["defect_type"] in MAIN_GEOMETRY_TYPES]
    main_val = [row for row in val_rows if row["defect_type"] in MAIN_GEOMETRY_TYPES]
    target_stats = _target_inverse_stats(main_train)

    all_output_rows: list[dict[str, Any]] = []
    model_info: dict[str, Any] = {}
    for feature_set_name, feature_cols in feature_sets.items():
        classifier_bundle, type_encoder = _train_classifier(train_rows, feature_cols)
        regressor_bundle, regressor_info = _select_regressor(main_train, main_val, feature_cols)
        output_rows = _build_prediction_rows(
            rows,
            feature_set_name,
            feature_cols,
            classifier_bundle,
            type_encoder,
            regressor_bundle,
            target_stats,
            regressor_info,
        )
        all_output_rows.extend(output_rows)
        model_info[feature_set_name] = {
            "feature_count": len(feature_cols),
            "classifier": "LogisticRegression(StandardScaler, balanced)",
            "regressor": regressor_info,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_output_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_output_rows)

    summary = {
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "predictions_path": str(out_path),
        "split_counts": dict(split_counts),
        "defect_counts": dict(defect_counts),
        "feature_sets": model_info,
        "target_policy": "Regress normalized center/size/depth labels plus raw angle_sin/angle_cos; invert normalized labels with train-only stats.",
        "paper_alignment": "Weak Piao-2019-inspired Bz-only adaptation: not RBC, not tri-axis NLS, not LS-SVM.",
        "polygon_policy": "Polygon participates in the all-3 defect type classifier only; polygon geometry regression and mask reconstruction are excluded from the main POC.",
        "split_metrics": _summarize(all_output_rows),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()
    summary = run(args.features, args.labels, args.out)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

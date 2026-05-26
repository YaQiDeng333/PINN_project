"""Smoke test for train_comsol_parametric_inverse.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from train_comsol_parametric_inverse import main


def _write_split(
    base: Path,
    name: str,
    n: int,
    signal_len: int = 20,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Path, Path, tuple[np.ndarray, np.ndarray]]:
    signals = np.random.default_rng(0).normal(size=(n, 3, signal_len)).astype(np.float32)
    x = np.linspace(-1.0, 1.0, signal_len).astype(np.float32)
    y = np.linspace(-0.5, 0.5, 10).astype(np.float32)
    masks = np.zeros((n, len(y), len(x)), dtype=np.float32)
    continuous_raw = np.zeros((n, 3, 6), dtype=np.float32)
    presence = np.zeros((n, 3), dtype=np.float32)
    type_targets = np.full((n, 3), -1, dtype=np.int64)
    for i in range(n):
        presence[i, 0] = 1.0
        type_targets[i, 0] = i % 2
        continuous_raw[i, 0] = [0.0, 0.0, 0.4, 0.2, 0.1, 0.0 if i % 2 == 0 else 30.0]
        masks[i, 4:6, 8:12] = 1.0
    angle_rad = np.deg2rad(continuous_raw[:, :, 5])
    continuous_unscaled = np.concatenate(
        [continuous_raw[:, :, :5], np.sin(angle_rad)[..., None], np.cos(angle_rad)[..., None]],
        axis=2,
    ).astype(np.float32)
    if stats is None:
        present_values = continuous_unscaled[presence > 0.5]
        mean = present_values.mean(axis=0).astype(np.float32)
        std = present_values.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    else:
        mean, std = stats
    continuous = ((continuous_unscaled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    npz_path = base / f"{name}.npz"
    target_path = base / f"{name}_targets.npz"
    np.savez(npz_path, signals=signals, masks=masks, x=x, y=y)
    np.savez(
        target_path,
        continuous_targets=continuous,
        continuous_targets_raw=continuous_raw,
        continuous_targets_unscaled=continuous_unscaled,
        continuous_targets_mean=mean,
        continuous_targets_std=std,
        continuous_targets_normalized=np.array(True),
        type_targets=type_targets,
        presence_targets=presence,
        sample_indices=np.arange(n),
        target_schema=np.array(
            ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_sin", "rotation_cos"],
            dtype="U64",
        ),
        raw_target_schema=np.array(
            ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"],
            dtype="U64",
        ),
        type_vocab=np.array(["rectangular_notch", "rotated_rect"], dtype="U64"),
        angle_encoding=np.array("sincos", dtype="U16"),
    )
    return npz_path, target_path, (mean, std)


def _write_features(base: Path, name: str, n: int, feature_dim: int = 6) -> Path:
    rng = np.random.default_rng(123 + n)
    features = rng.normal(size=(n, feature_dim)).astype(np.float32)
    path = base / f"{name}_features.npz"
    np.savez(
        path,
        features=features,
        feature_names=np.asarray([f"feature_{i}" for i in range(feature_dim)], dtype="U32"),
        sample_indices=np.arange(n),
    )
    return path


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        train_npz, train_targets, stats = _write_split(base, "train", 8)
        val_npz, val_targets, _ = _write_split(base, "val", 4, stats=stats)
        test_npz, test_targets, _ = _write_split(base, "test", 4, stats=stats)
        train_features = _write_features(base, "train", 8)
        val_features = _write_features(base, "val", 4)
        test_features = _write_features(base, "test", 4)
        out = base / "out"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--type-class-weighting",
                "inverse_freq",
            ]
        )
        assert rc == 0
        for name in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv", "training_history.csv", "run_summary.md"]:
            assert (out / name).exists(), name
        out_cnn = base / "out_cnn"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_cnn),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--type-class-weighting",
                "inverse_freq",
                "--encoder-type",
                "cnn1d",
                "--head-mode",
                "component_specific",
                "--lambda-rotation",
                "3.0",
            ]
        )
        assert rc == 0
        for name in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv", "training_history.csv", "run_summary.md"]:
            assert (out_cnn / name).exists(), name
        summary = (out_cnn / "run_summary.md").read_text(encoding="utf-8")
        assert "encoder_type" in summary
        assert "component_specific" in summary
        out_export = base / "out_export"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_export),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--export-predictions",
            ]
        )
        assert rc == 0
        for name in ["train_predictions.csv", "val_predictions.csv", "test_predictions.csv"]:
            path = out_export / name
            assert path.exists(), name
            text = path.read_text(encoding="utf-8")
            assert "sample_index" in text
            assert "component_slot" in text
            assert "presence_true" in text
            assert "presence_prob" in text
        out_perm = base / "out_perm"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_perm),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--component-matching-mode",
                "permutation_min",
                "--export-predictions",
            ]
        )
        assert rc == 0
        assert (out_perm / "metrics.csv").exists()
        assert (out_perm / "train_predictions.csv").exists()
        perm_summary = (out_perm / "run_summary.md").read_text(encoding="utf-8")
        assert "component_matching_mode" in perm_summary
        assert "permutation_min" in perm_summary
        perm_predictions = (out_perm / "train_predictions.csv").read_text(encoding="utf-8")
        assert "matched_slot" in perm_predictions
        out_raster = base / "out_raster"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_raster),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--lambda-raster-dice",
                "1.0",
                "--lambda-raster-bce",
                "0.5",
                "--raster-softness-cells",
                "1.0",
            ]
        )
        assert rc == 0
        for name in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv", "training_history.csv", "run_summary.md"]:
            assert (out_raster / name).exists(), name
        raster_summary = (out_raster / "run_summary.md").read_text(encoding="utf-8")
        assert "lambda_raster_dice" in raster_summary
        history = (out_raster / "training_history.csv").read_text(encoding="utf-8")
        assert "raster_dice_loss" in history
        out_raster_schedule = base / "out_raster_schedule"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_raster_schedule),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--lambda-raster-dice",
                "1.0",
                "--raster-loss-start-step",
                "3",
            ]
        )
        assert rc == 0
        with (out_raster_schedule / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            schedule_rows = list(csv.DictReader(handle))
        assert "raster_loss_active" in schedule_rows[0]
        active_by_step = {int(row["step"]): float(row["raster_loss_active"]) for row in schedule_rows}
        assert active_by_step[1] == 0.0
        assert active_by_step[5] == 1.0
        out_val_select = base / "out_val_select"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_val_select),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--val-selection-metric",
                "val_mask_iou",
                "--val-selection-interval",
                "2",
            ]
        )
        assert rc == 0
        val_summary = (out_val_select / "run_summary.md").read_text(encoding="utf-8")
        assert "best_step" in val_summary
        val_history = (out_val_select / "training_history.csv").read_text(encoding="utf-8")
        assert "is_best_step" in val_history
        out_feature_fusion = base / "out_feature_fusion"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_feature_fusion),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--feature-fusion-mode",
                "concat_latent",
                "--feature-npz",
                str(train_features),
                "--val-feature-npz",
                str(val_features),
                "--test-feature-npz",
                str(test_features),
            ]
        )
        assert rc == 0
        assert (out_feature_fusion / "metrics.csv").exists()
        feature_summary = (out_feature_fusion / "run_summary.md").read_text(encoding="utf-8")
        assert "feature_fusion_mode" in feature_summary
        assert "concat_latent" in feature_summary
        out_type_rotation = base / "out_type_rotation"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_type_rotation),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--lambda-type-extra",
                "1.0",
                "--lambda-rotation-extra",
                "2.0",
                "--rotation-loss-mode",
                "circular",
            ]
        )
        assert rc == 0
        assert (out_type_rotation / "metrics.csv").exists()
        targeted_summary = (out_type_rotation / "run_summary.md").read_text(encoding="utf-8")
        assert "lambda_type_extra" in targeted_summary
        assert "rotation_loss_mode" in targeted_summary
        out_center = base / "out_center"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_center),
                "--steps",
                "5",
                "--seed",
                "7",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--lambda-center-grid",
                "0.1",
                "--lambda-center-axis-relative",
                "1.0",
            ]
        )
        assert rc == 0
        assert (out_center / "metrics.csv").exists()
        center_summary = (out_center / "run_summary.md").read_text(encoding="utf-8")
        assert "seed" in center_summary
        assert "`7`" in center_summary
        assert "lambda_center_grid" in center_summary
        assert "lambda_center_axis_relative" in center_summary
        with (out_center / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            center_rows = list(csv.DictReader(handle))
        for field in [
            "center_grid_loss",
            "weighted_center_grid_loss",
            "center_axis_relative_loss",
            "weighted_center_axis_relative_loss",
        ]:
            assert field in center_rows[0]
            assert np.isfinite(float(center_rows[-1][field]))
        with (out_center / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            center_metrics = next(csv.DictReader(handle))
        assert center_metrics["seed"] == "7"
        assert "center_grid_mae" in center_metrics
        assert "center_axis_relative_mae" in center_metrics
        out_center_repeat = base / "out_center_repeat"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_center_repeat),
                "--steps",
                "5",
                "--seed",
                "7",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--lambda-center-grid",
                "0.1",
                "--lambda-center-axis-relative",
                "1.0",
            ]
        )
        assert rc == 0
        for name in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv"]:
            with (out_center / name).open(newline="", encoding="utf-8") as handle:
                expected = next(csv.DictReader(handle))
            with (out_center_repeat / name).open(newline="", encoding="utf-8") as handle:
                observed = next(csv.DictReader(handle))
            assert observed == expected
        out_center_seed8 = base / "out_center_seed8"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_center_seed8),
                "--steps",
                "2",
                "--seed",
                "8",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--lambda-center-grid",
                "0.1",
            ]
        )
        assert rc == 0
        with (out_center_seed8 / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            seed8_metrics = next(csv.DictReader(handle))
        assert seed8_metrics["seed"] == "8"
        out_center_bin = base / "out_center_bin"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_center_bin),
                "--steps",
                "5",
                "--seed",
                "9",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--center-representation",
                "bin_offset",
                "--center-bin-size-cells",
                "4",
                "--lambda-center-bin",
                "1.0",
                "--lambda-center-offset",
                "1.0",
                "--lambda-center-grid",
                "0.1",
                "--export-predictions",
            ]
        )
        assert rc == 0
        center_bin_summary = (out_center_bin / "run_summary.md").read_text(encoding="utf-8")
        assert "center_representation" in center_bin_summary
        assert "bin_offset" in center_bin_summary
        assert "center_x_bins" in center_bin_summary
        for name in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv", "training_history.csv", "train_predictions.csv"]:
            assert (out_center_bin / name).exists(), name
        with (out_center_bin / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            center_bin_rows = list(csv.DictReader(handle))
        for field in [
            "center_bin_loss",
            "weighted_center_bin_loss",
            "center_x_bin_loss",
            "center_y_bin_loss",
            "weighted_center_x_bin_loss",
            "weighted_center_y_bin_loss",
            "center_bin_slot_weight_mean",
            "center_offset_loss",
            "weighted_center_offset_loss",
            "center_grid_loss",
        ]:
            assert field in center_bin_rows[0]
            assert np.isfinite(float(center_bin_rows[-1][field]))
        with (out_center_bin / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            center_bin_metrics = next(csv.DictReader(handle))
        assert center_bin_metrics["center_representation"] == "bin_offset"
        assert center_bin_metrics["center_bin_size_cells"] == "4"
        assert center_bin_metrics["center_bin_x_weight"] == "1.0"
        assert center_bin_metrics["center_bin_y_weight"] == "1.0"
        assert "center_x_bin_accuracy" in center_bin_metrics
        assert "center_y_bin_accuracy" in center_bin_metrics
        with (out_center_bin / "train_predictions.csv").open(newline="", encoding="utf-8") as handle:
            decoded_rows = list(csv.DictReader(handle))
        pred_x = np.asarray([float(row["center_x_pred"]) for row in decoded_rows], dtype=np.float32)
        pred_y = np.asarray([float(row["center_y_pred"]) for row in decoded_rows], dtype=np.float32)
        assert np.isfinite(pred_x).all()
        assert np.isfinite(pred_y).all()
        assert pred_x.min() >= -1.25 and pred_x.max() <= 1.25
        assert pred_y.min() >= -0.75 and pred_y.max() <= 0.75
        out_center_bin_weighted = base / "out_center_bin_weighted"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_center_bin_weighted),
                "--steps",
                "5",
                "--seed",
                "11",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--center-representation",
                "bin_offset",
                "--center-bin-size-cells",
                "4",
                "--lambda-center-bin",
                "1.0",
                "--lambda-center-offset",
                "1.0",
                "--lambda-center-grid",
                "0.1",
                "--center-bin-x-weight",
                "1.5",
                "--center-bin-y-weight",
                "1.0",
                "--center-bin-slot-weights",
                "1.5,1.0,1.5",
            ]
        )
        assert rc == 0
        center_bin_weighted_summary = (out_center_bin_weighted / "run_summary.md").read_text(encoding="utf-8")
        assert "center_bin_x_weight" in center_bin_weighted_summary
        assert "1.5,1.0,1.5" in center_bin_weighted_summary
        with (out_center_bin_weighted / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            center_bin_weighted_rows = list(csv.DictReader(handle))
        for field in [
            "center_x_bin_loss",
            "center_y_bin_loss",
            "weighted_center_x_bin_loss",
            "weighted_center_y_bin_loss",
            "center_bin_slot_weight_mean",
            "center_bin_slot_weight_max",
        ]:
            assert field in center_bin_weighted_rows[0]
            assert np.isfinite(float(center_bin_weighted_rows[-1][field]))
        with (out_center_bin_weighted / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            center_bin_weighted_metrics = next(csv.DictReader(handle))
        assert center_bin_weighted_metrics["center_bin_x_weight"] == "1.5"
        assert center_bin_weighted_metrics["center_bin_slot_weights"] == "1.5,1.0,1.5"
        try:
            main(
                [
                    "--train-npz",
                    str(train_npz),
                    "--train-targets",
                    str(train_targets),
                    "--val-npz",
                    str(val_npz),
                    "--val-targets",
                    str(val_targets),
                    "--test-npz",
                    str(test_npz),
                    "--test-targets",
                    str(test_targets),
                    "--output-dir",
                    str(base / "out_invalid_center_slot_weights"),
                    "--steps",
                    "5",
                    "--hidden-dim",
                    "16",
                    "--latent-dim",
                    "8",
                    "--max-components",
                    "3",
                    "--center-representation",
                    "bin_offset",
                    "--center-bin-slot-weights",
                    "1.0,1.0",
                ]
            )
            raise AssertionError("Expected center-bin-slot-weights length mismatch to fail.")
        except ValueError as exc:
            assert "center-bin-slot-weights" in str(exc)
        out_center_aux = base / "out_center_aux"
        rc = main(
            [
                "--train-npz",
                str(train_npz),
                "--train-targets",
                str(train_targets),
                "--val-npz",
                str(val_npz),
                "--val-targets",
                str(val_targets),
                "--test-npz",
                str(test_npz),
                "--test-targets",
                str(test_targets),
                "--output-dir",
                str(out_center_aux),
                "--steps",
                "5",
                "--seed",
                "10",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--max-components",
                "3",
                "--center-representation",
                "bin_offset",
                "--center-bin-size-cells",
                "4",
                "--lambda-center-bin",
                "1.0",
                "--lambda-center-offset",
                "1.0",
                "--lambda-center-grid",
                "0.1",
                "--aux-center-head",
                "--lambda-aux-center-bin",
                "1.0",
                "--lambda-aux-center-offset",
                "1.0",
                "--aux-center-x-weight",
                "1.5",
                "--aux-center-y-weight",
                "1.0",
            ]
        )
        assert rc == 0
        center_aux_summary = (out_center_aux / "run_summary.md").read_text(encoding="utf-8")
        assert "aux_center_head" in center_aux_summary
        assert "lambda_aux_center_bin" in center_aux_summary
        with (out_center_aux / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            center_aux_rows = list(csv.DictReader(handle))
        for field in [
            "aux_center_bin_loss",
            "weighted_aux_center_bin_loss",
            "aux_center_offset_loss",
            "weighted_aux_center_offset_loss",
        ]:
            assert field in center_aux_rows[0]
            assert np.isfinite(float(center_aux_rows[-1][field]))
        with (out_center_aux / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            center_aux_metrics = next(csv.DictReader(handle))
        assert center_aux_metrics["aux_center_head"] == "True"
        assert center_aux_metrics["lambda_aux_center_bin"] == "1.0"
        assert center_aux_metrics["aux_center_x_weight"] == "1.5"
        assert "aux_center_x_bin_accuracy" in center_aux_metrics
        assert "aux_center_y_bin_accuracy" in center_aux_metrics
        try:
            main(
                [
                    "--train-npz",
                    str(train_npz),
                    "--train-targets",
                    str(train_targets),
                    "--val-npz",
                    str(val_npz),
                    "--val-targets",
                    str(val_targets),
                    "--test-npz",
                    str(test_npz),
                    "--test-targets",
                    str(test_targets),
                    "--output-dir",
                    str(base / "out_invalid_val_loss"),
                    "--steps",
                    "5",
                    "--hidden-dim",
                    "16",
                    "--latent-dim",
                    "8",
                    "--max-components",
                    "3",
                    "--lambda-raster-dice",
                    "1.0",
                    "--raster-loss-start-step",
                    "3",
                    "--val-selection-metric",
                    "val_loss",
                    "--val-selection-interval",
                    "2",
                ]
            )
            raise AssertionError("Expected delayed raster loss with val_loss selection to fail.")
        except ValueError as exc:
            assert "val_loss" in str(exc)
        forbidden = list(out.glob("*.pt")) + list(out.glob("*.pth")) + list(out.glob("*.ckpt")) + list(out.glob("*.npy"))
        forbidden += list(out_cnn.glob("*.pt")) + list(out_cnn.glob("*.pth")) + list(out_cnn.glob("*.ckpt")) + list(out_cnn.glob("*.npy"))
        forbidden += list(out_export.glob("*.pt")) + list(out_export.glob("*.pth")) + list(out_export.glob("*.ckpt")) + list(out_export.glob("*.npy"))
        forbidden += list(out_perm.glob("*.pt")) + list(out_perm.glob("*.pth")) + list(out_perm.glob("*.ckpt")) + list(out_perm.glob("*.npy"))
        forbidden += list(out_raster.glob("*.pt")) + list(out_raster.glob("*.pth")) + list(out_raster.glob("*.ckpt")) + list(out_raster.glob("*.npy"))
        forbidden += list(out_raster_schedule.glob("*.pt")) + list(out_raster_schedule.glob("*.pth")) + list(out_raster_schedule.glob("*.ckpt")) + list(out_raster_schedule.glob("*.npy"))
        forbidden += list(out_val_select.glob("*.pt")) + list(out_val_select.glob("*.pth")) + list(out_val_select.glob("*.ckpt")) + list(out_val_select.glob("*.npy"))
        forbidden += list(out_feature_fusion.glob("*.pt")) + list(out_feature_fusion.glob("*.pth")) + list(out_feature_fusion.glob("*.ckpt")) + list(out_feature_fusion.glob("*.npy"))
        forbidden += list(out_type_rotation.glob("*.pt")) + list(out_type_rotation.glob("*.pth")) + list(out_type_rotation.glob("*.ckpt")) + list(out_type_rotation.glob("*.npy"))
        forbidden += list(out_center.glob("*.pt")) + list(out_center.glob("*.pth")) + list(out_center.glob("*.ckpt")) + list(out_center.glob("*.npy"))
        forbidden += list(out_center_repeat.glob("*.pt")) + list(out_center_repeat.glob("*.pth")) + list(out_center_repeat.glob("*.ckpt")) + list(out_center_repeat.glob("*.npy"))
        forbidden += list(out_center_seed8.glob("*.pt")) + list(out_center_seed8.glob("*.pth")) + list(out_center_seed8.glob("*.ckpt")) + list(out_center_seed8.glob("*.npy"))
        forbidden += list(out_center_bin.glob("*.pt")) + list(out_center_bin.glob("*.pth")) + list(out_center_bin.glob("*.ckpt")) + list(out_center_bin.glob("*.npy"))
        forbidden += list(out_center_bin_weighted.glob("*.pt")) + list(out_center_bin_weighted.glob("*.pth")) + list(out_center_bin_weighted.glob("*.ckpt")) + list(out_center_bin_weighted.glob("*.npy"))
        forbidden += list(out_center_aux.glob("*.pt")) + list(out_center_aux.glob("*.pth")) + list(out_center_aux.glob("*.ckpt")) + list(out_center_aux.glob("*.npy"))
        assert not forbidden
    print("COMSOL parametric inverse training smoke test passed.")


if __name__ == "__main__":
    main_test()

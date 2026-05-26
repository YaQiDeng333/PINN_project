"""Smoke test for the center-anchored polygon inverse runner."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_center_anchored_polygon_targets import build_center_anchored_targets
from comsol_polygon_rasterizer import rasterize_polygon_components
from smoke_test_comsol_center_anchored_polygon_targets import _fixture
from train_comsol_center_anchored_polygon_inverse import main as train_main


def test_runner_smoke() -> None:
    polygon_targets, x, y = _fixture()
    center_targets = build_center_anchored_targets(polygon_targets, x, y, center_bin_size_cells=8)
    masks = rasterize_polygon_components(
        polygon_targets["polygon_vertices_norm"],
        polygon_targets["polygon_vertex_mask"],
        polygon_targets["presence_targets"],
        x,
        y,
    ).astype(np.float32)
    signals = np.stack(
        [
            np.tile(np.linspace(-1.0, 1.0, 200, dtype=np.float32), (3, 1)),
            np.tile(np.linspace(1.0, -1.0, 200, dtype=np.float32), (3, 1)),
        ],
        axis=0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        npz_path = base / "data.npz"
        targets_path = base / "center_anchored_polygon_targets.npz"
        output_dir = base / "out"
        extra_output_dir = base / "out_extra"
        bounded_output_dir = base / "out_bounded"
        bounded_stats_output_dir = base / "out_bounded_stats"
        conditioned_output_dir = base / "out_conditioned"
        joint_output_dir = base / "out_joint"
        center_consistency_output_dir = base / "out_center_consistency"
        center_centroid_aux_output_dir = base / "out_center_centroid_aux"
        np.savez_compressed(npz_path, signals=signals, masks=masks, x=x, y=y)
        np.savez_compressed(targets_path, **center_targets)
        base_args = [
            "--train-npz",
            str(npz_path),
            "--train-targets",
            str(targets_path),
            "--val-npz",
            str(npz_path),
            "--val-targets",
            str(targets_path),
            "--test-npz",
            str(npz_path),
            "--test-targets",
            str(targets_path),
            "--steps",
            "3",
            "--hidden-dim",
            "16",
            "--latent-dim",
            "8",
            "--history-interval",
            "1",
            "--export-predictions",
            "--seed",
            "1",
            "--device",
            "cpu",
        ]
        rc = train_main([*base_args, "--output-dir", str(output_dir)])
        assert rc == 0
        rc = train_main(
            [
                *base_args,
                "--output-dir",
                str(extra_output_dir),
                "--center-y-bin-extra-loss-mode",
                "neighbor_soft_ce",
                "--lambda-center-y-bin-extra",
                "0.5",
                "--center-y-bin-neighbor-smoothing",
                "0.2",
            ]
        )
        assert rc == 0
        rc = train_main(
            [
                *base_args,
                "--output-dir",
                str(bounded_output_dir),
                "--local-shape-output-mode",
                "bounded_tanh",
                "--local-shape-bound-mode",
                "fixed_grid",
                "--local-shape-fixed-bound-x-grid",
                "24.0",
                "--local-shape-fixed-bound-y-grid",
                "8.0",
            ]
        )
        assert rc == 0
        rc = train_main(
            [
                *base_args,
                "--output-dir",
                str(bounded_stats_output_dir),
                "--local-shape-output-mode",
                "bounded_tanh",
                "--local-shape-bound-mode",
                "train_stats",
                "--local-shape-train-stats-margin",
                "1.25",
            ]
        )
        assert rc == 0
        rc = train_main(
            [
                *base_args,
                "--output-dir",
                str(conditioned_output_dir),
                "--local-shape-conditioning-mode",
                "center_bin_slot_type",
                "--local-shape-conditioning-dim",
                "8",
            ]
        )
        assert rc == 0
        rc = train_main(
            [
                *base_args,
                "--output-dir",
                str(joint_output_dir),
                "--joint-center-shape-mode",
                "soft_center_scheduled",
                "--joint-center-teacher-forcing-steps",
                "3",
            ]
        )
        assert rc == 0
        rc = train_main(
            [
                *base_args,
                "--output-dir",
                str(center_consistency_output_dir),
                "--center-consistency-mode",
                "soft_decoded_center",
                "--lambda-center-consistency",
                "1.0",
            ]
        )
        assert rc == 0
        rc = train_main(
            [
                *base_args,
                "--output-dir",
                str(center_centroid_aux_output_dir),
                "--lambda-decoded-center-aux",
                "0.05",
                "--lambda-polygon-centroid-aux",
                "0.05",
                "--center-centroid-aux-smoothl1-beta",
                "0.01",
            ]
        )
        assert rc == 0
        for name in [
            "metrics.csv",
            "eval_metrics.csv",
            "test_metrics.csv",
            "training_history.csv",
            "run_summary.md",
            "train_center_anchored_polygon_predictions.csv",
            "train_center_anchored_polygon_mask_metrics.csv",
        ]:
            assert (output_dir / name).exists(), name
        with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        for field in [
            "polygon_mask_iou",
            "decoded_vertex_mae",
            "center_x_bin_acc",
            "center_y_bin_acc",
            "center_y_bin_abs_error",
            "center_y_bin_within1_acc",
            "center_y_bin_extra_loss",
            "weighted_center_y_bin_extra_loss",
            "center_offset_mae",
            "local_vertex_mae_grid",
            "local_shape_raw_abs_max",
            "local_shape_effective_abs_max",
            "local_shape_saturation_frac",
            "local_shape_output_mode",
            "local_shape_bound_x_grid",
            "local_shape_bound_y_grid",
            "local_shape_conditioning_mode",
            "local_shape_conditioning_dim",
            "joint_center_shape_mode",
            "joint_center_teacher_forcing_steps",
            "center_consistency_mode",
            "lambda_center_consistency",
            "center_consistency_loss",
            "weighted_center_consistency_loss",
            "lambda_decoded_center_aux",
            "lambda_polygon_centroid_aux",
            "center_centroid_aux_smoothl1_beta",
            "decoded_center_aux_loss",
            "polygon_centroid_aux_loss",
            "weighted_decoded_center_aux_loss",
            "weighted_polygon_centroid_aux_loss",
            "hard_decoded_center_mae_grid",
            "soft_expected_center_mae_grid",
            "center_x_bin_prob_margin",
            "center_y_bin_prob_margin",
        ]:
            assert field in row
            if field not in {"local_shape_output_mode", "local_shape_conditioning_mode", "joint_center_shape_mode", "center_consistency_mode"}:
                assert np.isfinite(float(row[field]))
        assert row["local_shape_output_mode"] == "raw"
        assert row["local_shape_conditioning_mode"] == "none"
        assert row["joint_center_shape_mode"] == "none"
        assert row["center_consistency_mode"] == "none"
        assert float(row["lambda_center_consistency"]) == 0.0
        assert float(row["lambda_decoded_center_aux"]) == 0.0
        assert float(row["lambda_polygon_centroid_aux"]) == 0.0
        with (output_dir / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            history_row = next(csv.DictReader(handle))
        assert "center_y_bin_extra_loss" in history_row
        assert float(history_row["weighted_center_y_bin_extra_loss"]) == 0.0
        with (extra_output_dir / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            extra_history_row = next(csv.DictReader(handle))
        assert np.isfinite(float(extra_history_row["center_y_bin_extra_loss"]))
        assert float(extra_history_row["weighted_center_y_bin_extra_loss"]) >= 0.0
        with (bounded_output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            bounded_row = next(csv.DictReader(handle))
        assert bounded_row["local_shape_output_mode"] == "bounded_tanh"
        assert float(bounded_row["local_shape_bound_x_grid"]) == 24.0
        assert float(bounded_row["local_shape_bound_y_grid"]) == 8.0
        assert np.isfinite(float(bounded_row["local_shape_saturation_frac"]))
        with (bounded_stats_output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            bounded_stats_row = next(csv.DictReader(handle))
        assert bounded_stats_row["local_shape_output_mode"] == "bounded_tanh"
        assert bounded_stats_row["local_shape_bound_mode"] == "train_stats"
        assert float(bounded_stats_row["local_shape_bound_x_grid"]) > 0.0
        assert float(bounded_stats_row["local_shape_bound_y_grid"]) > 0.0
        with (conditioned_output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            conditioned_row = next(csv.DictReader(handle))
        assert conditioned_row["local_shape_conditioning_mode"] == "center_bin_slot_type"
        assert float(conditioned_row["local_shape_conditioning_dim"]) == 8.0
        with (joint_output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            joint_row = next(csv.DictReader(handle))
        assert joint_row["joint_center_shape_mode"] == "soft_center_scheduled"
        assert float(joint_row["joint_center_teacher_forcing_steps"]) == 3.0
        with (joint_output_dir / "training_history.csv").open(newline="", encoding="utf-8") as handle:
            joint_history = list(csv.DictReader(handle))
        assert "joint_center_teacher_forcing_weight" in joint_history[0]
        assert float(joint_history[0]["joint_center_teacher_forcing_weight"]) == 1.0
        assert float(joint_history[-1]["joint_center_teacher_forcing_weight"]) == 0.0
        with (center_consistency_output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            center_consistency_row = next(csv.DictReader(handle))
        assert center_consistency_row["center_consistency_mode"] == "soft_decoded_center"
        assert float(center_consistency_row["lambda_center_consistency"]) == 1.0
        assert np.isfinite(float(center_consistency_row["center_consistency_loss"]))
        assert float(center_consistency_row["weighted_center_consistency_loss"]) >= 0.0
        with (center_centroid_aux_output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            center_centroid_aux_row = next(csv.DictReader(handle))
        assert float(center_centroid_aux_row["lambda_decoded_center_aux"]) == 0.05
        assert float(center_centroid_aux_row["lambda_polygon_centroid_aux"]) == 0.05
        assert np.isfinite(float(center_centroid_aux_row["decoded_center_aux_loss"]))
        assert np.isfinite(float(center_centroid_aux_row["polygon_centroid_aux_loss"]))
        assert float(center_centroid_aux_row["weighted_decoded_center_aux_loss"]) >= 0.0
        assert float(center_centroid_aux_row["weighted_polygon_centroid_aux_loss"]) >= 0.0
        with (center_consistency_output_dir / "train_center_anchored_polygon_predictions.csv").open(newline="", encoding="utf-8") as handle:
            prediction_row = next(csv.DictReader(handle))
        for field in [
            "center_x_bin_prob_margin",
            "center_y_bin_prob_margin",
            "hard_center_x_error_grid",
            "soft_center_x_error_grid",
        ]:
            assert field in prediction_row
            assert np.isfinite(float(prediction_row[field]))
        forbidden = list(output_dir.glob("*.pt")) + list(output_dir.glob("*.pth")) + list(output_dir.glob("*.ckpt"))
        forbidden += list(output_dir.glob("*.npy"))
        forbidden += list(extra_output_dir.glob("*.pt")) + list(extra_output_dir.glob("*.pth")) + list(extra_output_dir.glob("*.ckpt"))
        forbidden += list(extra_output_dir.glob("*.npy"))
        forbidden += list(bounded_output_dir.glob("*.pt")) + list(bounded_output_dir.glob("*.pth")) + list(bounded_output_dir.glob("*.ckpt"))
        forbidden += list(bounded_output_dir.glob("*.npy"))
        forbidden += list(bounded_stats_output_dir.glob("*.pt")) + list(bounded_stats_output_dir.glob("*.pth")) + list(bounded_stats_output_dir.glob("*.ckpt"))
        forbidden += list(bounded_stats_output_dir.glob("*.npy"))
        forbidden += list(conditioned_output_dir.glob("*.pt")) + list(conditioned_output_dir.glob("*.pth")) + list(conditioned_output_dir.glob("*.ckpt"))
        forbidden += list(conditioned_output_dir.glob("*.npy"))
        forbidden += list(joint_output_dir.glob("*.pt")) + list(joint_output_dir.glob("*.pth")) + list(joint_output_dir.glob("*.ckpt"))
        forbidden += list(joint_output_dir.glob("*.npy"))
        forbidden += list(center_consistency_output_dir.glob("*.pt")) + list(center_consistency_output_dir.glob("*.pth"))
        forbidden += list(center_consistency_output_dir.glob("*.ckpt")) + list(center_consistency_output_dir.glob("*.npy"))
        forbidden += list(center_centroid_aux_output_dir.glob("*.pt")) + list(center_centroid_aux_output_dir.glob("*.pth"))
        forbidden += list(center_centroid_aux_output_dir.glob("*.ckpt")) + list(center_centroid_aux_output_dir.glob("*.npy"))
        assert not forbidden


if __name__ == "__main__":
    test_runner_smoke()
    print("center-anchored polygon inverse runner smoke passed")

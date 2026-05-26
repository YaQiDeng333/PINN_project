"""Smoke test for the component-query polygon inverse runner."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from comsol_center_anchored_polygon_targets import build_center_anchored_targets
from comsol_polygon_rasterizer import rasterize_polygon_components
from smoke_test_comsol_center_anchored_polygon_targets import _fixture
from train_comsol_component_query_polygon_inverse import main as train_main


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
        np.savez_compressed(npz_path, signals=signals, masks=masks, x=x, y=y)
        np.savez_compressed(targets_path, **center_targets)
        rc = train_main(
            [
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
                "--output-dir",
                str(output_dir),
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
            "inverse_route",
            "polygon_mask_iou",
            "decoded_vertex_mae",
            "center_x_bin_acc",
            "center_y_bin_acc",
            "hard_decoded_center_l2_grid",
            "local_vertex_mae_grid",
            "local_shape_output_mode",
            "center_consistency_mode",
            "decoded_center_aux_loss",
            "polygon_centroid_aux_loss",
            "weighted_decoded_center_aux_loss",
            "weighted_polygon_centroid_aux_loss",
            "lambda_decoded_center_aux",
            "lambda_polygon_centroid_aux",
        ]:
            assert field in row
        assert row["inverse_route"] == "component_query"
        assert row["local_shape_output_mode"] == "raw"
        assert row["center_consistency_mode"] == "none"
        assert float(row["lambda_decoded_center_aux"]) == 0.0
        assert float(row["lambda_polygon_centroid_aux"]) == 0.0
        with (output_dir / "train_center_anchored_polygon_predictions.csv").open(newline="", encoding="utf-8") as handle:
            prediction_row = next(csv.DictReader(handle))
        for field in [
            "center_x_bin_true",
            "center_x_bin_pred",
            "center_y_bin_true",
            "center_y_bin_pred",
            "hard_center_x_error_grid",
            "soft_center_x_error_grid",
            "pred_local_x0",
        ]:
            assert field in prediction_row
            assert np.isfinite(float(prediction_row[field]))
        forbidden = []
        for pattern in ("*.pt", "*.pth", "*.ckpt", "*.npy"):
            forbidden.extend(output_dir.glob(pattern))
        assert not forbidden

        aux_output_dir = base / "out_aux"
        rc = train_main(
            [
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
                "--output-dir",
                str(aux_output_dir),
                "--steps",
                "2",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--lambda-decoded-center-aux",
                "0.05",
                "--lambda-polygon-centroid-aux",
                "0.05",
                "--lambda-area-aux",
                "0.001",
                "--center-centroid-aux-smoothl1-beta",
                "0.01",
                "--seed",
                "2",
                "--device",
                "cpu",
            ]
        )
        assert rc == 0
        with (aux_output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            aux_row = next(csv.DictReader(handle))
        assert float(aux_row["lambda_decoded_center_aux"]) == 0.05
        assert float(aux_row["lambda_polygon_centroid_aux"]) == 0.05
        assert float(aux_row["lambda_area_aux"]) == 0.001
        assert "decoded_center_aux_loss" in aux_row
        assert "polygon_centroid_aux_loss" in aux_row
        assert "area_aux_loss" in aux_row


if __name__ == "__main__":
    test_runner_smoke()
    print("component-query polygon inverse runner smoke passed")

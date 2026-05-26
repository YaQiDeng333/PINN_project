"""Smoke test for train_comsol_polygon_inverse.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from comsol_polygon_rasterizer import rasterize_polygon_components
from train_comsol_polygon_inverse import main


def _write_split(base: Path, name: str, n: int, signal_len: int = 20) -> tuple[Path, Path]:
    rng = np.random.default_rng(10 + n)
    signals = rng.normal(size=(n, 3, signal_len)).astype(np.float32)
    x = np.linspace(-1.0, 1.0, signal_len, dtype=np.float32)
    y = np.linspace(-0.5, 0.5, 10, dtype=np.float32)
    vertices = np.zeros((n, 3, 4, 2), dtype=np.float32)
    vertex_mask = np.zeros((n, 3, 4), dtype=np.float32)
    presence = np.zeros((n, 3), dtype=np.float32)
    type_targets = np.full((n, 3), -1, dtype=np.int64)
    for i in range(n):
        cx = -0.25 + 0.1 * (i % 5)
        cy = 0.0
        vertices[i, 0] = np.array(
            [[cx - 0.2, cy - 0.1], [cx + 0.2, cy - 0.1], [cx + 0.2, cy + 0.1], [cx - 0.2, cy + 0.1]],
            dtype=np.float32,
        )
        vertex_mask[i, 0] = 1.0
        presence[i, 0] = 1.0
        type_targets[i, 0] = i % 2
    masks = rasterize_polygon_components(vertices, vertex_mask, presence, x, y).astype(np.float32)
    npz_path = base / f"{name}.npz"
    targets_path = base / f"{name}_polygon_targets.npz"
    np.savez_compressed(npz_path, signals=signals, masks=masks, x=x, y=y, mu_maps=np.where(masks > 0.5, 1.0, 1000.0))
    np.savez_compressed(
        targets_path,
        polygon_vertices_norm=vertices,
        polygon_vertices_raw=vertices,
        polygon_vertex_mask=vertex_mask,
        presence_targets=presence,
        type_targets=type_targets,
        type_vocab=np.array(["rectangular_notch", "rotated_rect"], dtype="U32"),
        component_counts=presence.sum(axis=1).astype(np.int64),
        sample_indices=np.arange(n, dtype=np.int64),
        x_norm=x,
        y_norm=y,
        vertex_ordering=np.array("clockwise_top_left", dtype="U32"),
    )
    return npz_path, targets_path


def main_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        train_npz, train_targets = _write_split(base, "train", 6)
        val_npz, val_targets = _write_split(base, "val", 3)
        test_npz, test_targets = _write_split(base, "test", 3)
        out_default = base / "out_default"
        rc_default = main(
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
                str(out_default),
                "--steps",
                "3",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--history-interval",
                "1",
                "--seed",
                "3",
            ]
        )
        assert rc_default == 0
        default_summary = (out_default / "run_summary.md").read_text(encoding="utf-8")
        assert "vertex_loss_space: `norm`" in default_summary
        assert "lambda_area_aux: `0.0`" in default_summary
        assert "lambda_edge_aux: `0.0`" in default_summary
        out_repair = base / "out_repair"
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
                str(out_repair),
                "--steps",
                "5",
                "--hidden-dim",
                "16",
                "--latent-dim",
                "8",
                "--history-interval",
                "1",
                "--vertex-loss-space",
                "grid",
                "--lambda-area-aux",
                "0.5",
                "--lambda-edge-aux",
                "0.1",
                "--lambda-box-aux",
                "0.1",
                "--export-predictions",
                "--seed",
                "3",
            ]
        )
        assert rc == 0
        for name in [
            "metrics.csv",
            "eval_metrics.csv",
            "test_metrics.csv",
            "training_history.csv",
            "run_summary.md",
            "train_polygon_predictions.csv",
            "train_polygon_mask_metrics.csv",
        ]:
            assert (out_repair / name).exists(), name
        summary = (out_repair / "run_summary.md").read_text(encoding="utf-8")
        assert "COMSOL polygon inverse run summary" in summary
        assert "polygon_mask_iou" in summary
        assert "vertex_loss_space: `grid`" in summary
        assert "lambda_area_aux: `0.5`" in summary
        assert "lambda_edge_aux: `0.1`" in summary
        history = (out_repair / "training_history.csv").read_text(encoding="utf-8")
        assert "area_aux_loss" in history
        assert "edge_aux_loss" in history
    print("COMSOL polygon inverse runner smoke test passed.")


def triangle_aux_guard_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        train_npz, train_targets = _write_split(base, "train", 2)
        with np.load(train_targets, allow_pickle=True) as data:
            target_data = {key: data[key] for key in data.files}
        target_data["polygon_vertex_mask"][0, 0, 3] = 0.0
        triangle_targets = base / "triangle_targets.npz"
        np.savez_compressed(triangle_targets, **target_data)
        out = base / "triangle_out"
        try:
            main(
                [
                    "--train-npz",
                    str(train_npz),
                    "--train-targets",
                    str(triangle_targets),
                    "--val-npz",
                    str(train_npz),
                    "--val-targets",
                    str(triangle_targets),
                    "--test-npz",
                    str(train_npz),
                    "--test-targets",
                    str(triangle_targets),
                    "--output-dir",
                    str(out),
                    "--steps",
                    "1",
                    "--lambda-area-aux",
                    "0.5",
                    "--seed",
                    "4",
                ]
            )
        except ValueError as exc:
            assert "requires four valid vertices" in str(exc)
        else:
            raise AssertionError("area auxiliary loss should reject present 3-vertex polygons")


if __name__ == "__main__":
    main_test()
    triangle_aux_guard_test()

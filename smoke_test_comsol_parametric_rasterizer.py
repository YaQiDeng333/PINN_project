"""Smoke test for comsol_parametric_rasterizer.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from comsol_parametric_rasterizer import mask_iou_dice, rasterize_components, main


def main_test() -> None:
    x = np.linspace(-1.0, 1.0, 80).astype(np.float32)
    y = np.linspace(-1.0, 1.0, 60).astype(np.float32)
    schema = np.array(
        ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param", "rotation_angle"],
        dtype="U64",
    )
    type_vocab = np.array(["rectangular_notch", "rotated_rect"], dtype="U64")
    continuous = np.array(
        [
            [[0.0, 0.0, 0.8, 0.4, 0.1, 0.0], [0.4, 0.2, 0.2, 0.3, 0.1, 30.0], [0, 0, 0, 0, 0, 0]],
            [[0.0, 0.0, 0.8, 0.4, 0.1, 45.0], [0.0, 0.0, 0.8, 0.4, 0.1, 0.0], [0, 0, 0, 0, 0, 0]],
        ],
        dtype=np.float32,
    )
    type_targets = np.array([[0, 1, -1], [1, 1, -1]], dtype=np.int64)
    presence = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32)
    masks = rasterize_components(continuous, type_targets, presence, schema, type_vocab, x, y)
    ious, _dices = mask_iou_dice(masks, masks.astype(np.float32))
    assert np.allclose(ious, 1.0)
    unrotated_presence = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
    unrotated = rasterize_components(continuous, type_targets, unrotated_presence, schema, type_vocab, x, y)
    assert not np.array_equal(masks[1], unrotated[1])
    assert masks[0].sum() > rasterize_components(continuous[:1, :1], type_targets[:1, :1], presence[:1, :1], schema, type_vocab, x, y)[0].sum()

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        npz_path = base / "mock.npz"
        targets_path = base / "targets.npz"
        out = base / "out"
        np.savez(npz_path, masks=masks.astype(np.float32), x=x, y=y)
        np.savez(
            targets_path,
            continuous_targets=continuous,
            type_targets=type_targets,
            presence_targets=presence,
            sample_indices=np.array([0, 1], dtype=np.int64),
            target_schema=schema,
            type_vocab=type_vocab,
        )
        rc = main(["--npz-path", str(npz_path), "--parametric-targets", str(targets_path), "--output-dir", str(out)])
        assert rc == 0
        assert (out / "oracle_parametric_mask_metrics.csv").exists()
        assert (out / "oracle_parametric_mask_aggregate.csv").exists()
        assert (out / "summary.md").exists()
    print("COMSOL parametric rasterizer smoke test passed.")


if __name__ == "__main__":
    main_test()

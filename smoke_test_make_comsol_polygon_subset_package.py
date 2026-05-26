"""Smoke test for COMSOL polygon subset package helper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from make_comsol_polygon_subset_package import create_subset, parse_indices


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        signals = np.arange(6 * 3 * 4, dtype=np.float32).reshape(6, 3, 4)
        masks = np.zeros((6, 2, 4), dtype=np.float32)
        vertices = np.zeros((6, 3, 4, 2), dtype=np.float32)
        vertex_mask = np.zeros((6, 3, 4), dtype=np.float32)
        presence = np.zeros((6, 3), dtype=np.float32)
        presence[:, 0] = 1.0
        vertex_mask[:, 0] = 1.0
        type_targets = np.full((6, 3), -1, dtype=np.int64)
        type_targets[:, 0] = 0
        npz = root / "data.npz"
        targets = root / "targets.npz"
        np.savez_compressed(npz, signals=signals, masks=masks, x=np.arange(4), y=np.arange(2))
        np.savez_compressed(
            targets,
            polygon_vertices_norm=vertices,
            polygon_vertices_raw=vertices,
            polygon_vertex_mask=vertex_mask,
            presence_targets=presence,
            type_targets=type_targets,
            type_vocab=np.array(["rectangular_notch"], dtype="U32"),
            component_counts=np.ones(6, dtype=np.int64),
            sample_indices=np.arange(6, dtype=np.int64),
            vertex_ordering=np.array("clockwise_top_left", dtype="U32"),
        )
        indices = parse_indices("0,2,5", 6)
        out_npz = root / "subset.npz"
        out_targets = root / "subset_targets.npz"
        create_subset(npz, targets, indices, out_npz, out_targets)
        with np.load(out_npz, allow_pickle=True) as data:
            assert data["signals"].shape[0] == 3
            assert data["signals"][1, 0, 0] == signals[2, 0, 0]
        with np.load(out_targets, allow_pickle=True) as data:
            assert data["presence_targets"].shape[0] == 3
            assert np.array_equal(data["sample_indices"], np.arange(3))
            assert "subset_source_indices_json" in data.files
    print("COMSOL polygon subset package smoke test passed.")


if __name__ == "__main__":
    main()

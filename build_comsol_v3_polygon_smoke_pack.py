"""Build local files from COMSOL V3 polygon smoke stdout markers."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from comsol_polygon_rasterizer import rasterize_polygon_components


STDOUT_PATH = Path("ComsolV3PolygonSmokeExport.stdout.txt")
OUT_ROOT = Path("comsol_v3_polygon_geometry_3sample_smoke")
X_RAW = np.linspace(0.0, 4500.0, 200, dtype=np.float32)
Y_RAW = np.linspace(0.0, 3000.0, 100, dtype=np.float32)


def _extract(text: str, name: str) -> str:
    start_marker = f"BEGIN_{name}"
    end_marker = f"END_{name}"
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    return text[start:end].strip("\r\n")


def _read_csv(block: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(block)))


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    columns = columns or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in columns} for row in rows])


def _build_masks(polygon_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 3
    k = 3
    v = 4
    vertices = np.zeros((n, k, v, 2), dtype=np.float32)
    vertex_mask = np.zeros((n, k, v), dtype=np.float32)
    presence = np.zeros((n, k), dtype=np.float32)
    for row in polygon_rows:
        sample = int(row["sample_index"])
        slot = int(row["component_slot"])
        vertex = int(row["vertex_index"])
        vertices[sample, slot, vertex, 0] = float(row["x_raw"])
        vertices[sample, slot, vertex, 1] = float(row["y_raw"])
        vertex_mask[sample, slot, vertex] = 1.0
        presence[sample, slot] = 1.0
    masks = rasterize_polygon_components(vertices, vertex_mask, presence, X_RAW, Y_RAW)
    return masks.astype(np.float32), vertices, presence


def main() -> None:
    if OUT_ROOT.exists():
        raise SystemExit(f"Output root already exists: {OUT_ROOT}")
    text = STDOUT_PATH.read_text(encoding="utf-16")
    defect_rows = _read_csv(_extract(text, "COMSOL_V3_POLYGON_SMOKE_DEFECT_PARAMS_CSV"))
    polygon_rows = _read_csv(_extract(text, "COMSOL_V3_POLYGON_SMOKE_POLYGON_PARAMS_CSV"))
    signal_rows = _read_csv(_extract(text, "COMSOL_V3_POLYGON_SMOKE_SIGNALS_CSV"))
    solver_rows = _read_csv(_extract(text, "COMSOL_V3_POLYGON_SMOKE_SOLVER_CSV"))
    masks, _vertices, _presence = _build_masks(polygon_rows)
    mu_maps = np.where(masks > 0.5, 1.0, 1000.0).astype(np.float32)
    OUT_ROOT.mkdir()
    _write_csv(OUT_ROOT / "defect_params.csv", defect_rows)
    _write_csv(OUT_ROOT / "polygon_params.csv", polygon_rows)
    _write_csv(OUT_ROOT / "signals_multiheight.csv", signal_rows)
    _write_csv(OUT_ROOT / "solver_summary.csv", solver_rows)
    np.savez_compressed(
        OUT_ROOT / "targets.npz",
        mu_maps=mu_maps,
        masks=masks,
        x=X_RAW,
        y=Y_RAW,
        source_sample_ids=np.array([row["source_sample_id"] for row in defect_rows]),
        source_global_indices=np.array([int(row["source_global_index"]) for row in defect_rows], dtype=np.int32),
        signal_channel_names=np.array(["Bz_liftoff_0p5", "Bz_liftoff_1p0", "Bz_liftoff_2p0"]),
        lift_off_values=np.array([0.5, 1.0, 2.0], dtype=np.float32),
        field_components=np.array(["Bz", "Bz", "Bz"]),
        source_type=np.array("comsol_v3_polygon_3sample_smoke"),
        signal_flatten_order=np.array("channel_x"),
        geometry_units=np.array("raw_comsol_coordinate_units"),
        field_units=np.array("T_reduced_anomaly"),
        metadata_json=np.array('{"true_polygon_geometry":true,"smoke_only":true}'),
    )
    values = np.array([float(row["value"]) for row in signal_rows], dtype=np.float64)
    mismatch = int(np.count_nonzero((mu_maps < 500.0) != (masks > 0.5)))
    readme = f"""# COMSOL V3 polygon geometry 3-sample smoke

This smoke was generated from real COMSOL solves with true rotated and multi-component geometry.
It uses the repaired near-defect reduced/anomaly Bz route (`mfnc.redBz`). It is not a full pack
and must not be used as a training set.

- samples: 3
- rows: {len(signal_rows)}
- value finite: {bool(np.isfinite(values).all())}
- value range: [{values.min():.12e}, {values.max():.12e}]
- masks == (mu_maps < 500) mismatch: {mismatch}
- polygon_params.csv contains raw and normalized 4-corner vertices.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Built {OUT_ROOT}")


if __name__ == "__main__":
    main()

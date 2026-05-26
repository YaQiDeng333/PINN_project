from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from normalize_comsol_v3_geometry_convention import main


def test_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        npz_path = root / "input.npz"
        csv_path = root / "defect_params.csv"
        out_npz = root / "out.npz"
        out_csv = root / "out.csv"
        dtype = [
            ("sample_index", "i4"),
            ("defect_center_x", "f8"),
            ("defect_center_y", "f8"),
            ("defect_axis_x", "f8"),
            ("defect_axis_y", "f8"),
            ("defect_depth_or_shape_param", "f8"),
        ]
        defect_params = np.array([(0, 2250.0, 1500.0, 450.0, 300.0, 70.0)], dtype=dtype)
        np.savez(
            npz_path,
            x=np.array([0.0, 2250.0, 4500.0], dtype=np.float32),
            y=np.array([0.0, 1500.0, 3000.0], dtype=np.float32),
            signals=np.zeros((1, 3, 3), dtype=np.float32),
            mu_maps=np.ones((1, 3, 3), dtype=np.float32) * 1000,
            masks=np.zeros((1, 3, 3), dtype=np.uint8),
            defect_params=defect_params,
            geometry_units=np.array("m"),
            field_units=np.array("T"),
            metadata_json=np.array("{}"),
        )
        pd.DataFrame(
            [
                {
                    "sample_index": 0,
                    "defect_center_x": 2250.0,
                    "defect_center_y": 1500.0,
                    "defect_axis_x": 450.0,
                    "defect_axis_y": 300.0,
                    "defect_depth_or_shape_param": 70.0,
                }
            ]
        ).to_csv(csv_path, index=False)

        import sys

        old_argv = sys.argv
        sys.argv = [
            "normalize_comsol_v3_geometry_convention.py",
            "--npz-path",
            str(npz_path),
            "--defect-params-csv",
            str(csv_path),
            "--output-npz",
            str(out_npz),
            "--output-defect-params-csv",
            str(out_csv),
            "--split",
            "smoke",
        ]
        try:
            assert main() == 0
        finally:
            sys.argv = old_argv

        with np.load(out_npz, allow_pickle=True) as out:
            df = pd.read_csv(out_csv)
            assert np.allclose(out["x"], [-0.04, 0.0, 0.04])
            assert np.allclose(out["y"], [-0.01, 0.0, 0.01])
            assert abs(float(df.loc[0, "defect_center_x"])) < 1e-12
            assert abs(float(df.loc[0, "defect_center_y"])) < 1e-12
            assert np.isclose(float(df.loc[0, "defect_axis_x"]), 0.008)
            assert np.isclose(float(df.loc[0, "defect_axis_y"]), 0.002)
            assert float(df.loc[0, "defect_depth_or_shape_param"]) == 70.0
            assert "v3_geometry_normalization" in out.files


if __name__ == "__main__":
    test_smoke()
    print("COMSOL V3 geometry normalization smoke test passed.")

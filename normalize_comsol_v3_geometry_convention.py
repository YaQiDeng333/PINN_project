"""Normalize COMSOL V3 hard-case geometry coordinates to V2 convention.

The V3 hard-case pack is generated in the COMSOL example model coordinate
domain, where x/y are roughly [0, 4500] / [0, 3000]. The V2 parametric route
uses centered meter-scale geometry, x/y roughly [-0.04, 0.04] / [-0.01, 0.01].
This script writes a normalized copy; it never edits the source pack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_X_MIN = 0.0
RAW_X_MAX = 4500.0
RAW_Y_MIN = 0.0
RAW_Y_MAX = 3000.0
TARGET_X_MIN = -0.04
TARGET_X_MAX = 0.04
TARGET_Y_MIN = -0.01
TARGET_Y_MAX = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize COMSOL V3 hard-case x/y geometry to V2 meter-scale convention."
    )
    parser.add_argument("--npz-path")
    parser.add_argument("--defect-params-csv")
    parser.add_argument("--output-npz")
    parser.add_argument("--output-defect-params-csv")
    parser.add_argument("--split", default="")
    return parser.parse_args()


def _usage_and_exit() -> int:
    print(
        "Usage: python normalize_comsol_v3_geometry_convention.py "
        "--npz-path INPUT.npz --defect-params-csv defect_params.csv "
        "--output-npz OUTPUT.npz --output-defect-params-csv OUTPUT.csv"
    )
    return 0


def scale_x(values) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - 0.5 * (RAW_X_MIN + RAW_X_MAX)) * (
        (TARGET_X_MAX - TARGET_X_MIN) / (RAW_X_MAX - RAW_X_MIN)
    )


def scale_y(values) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - 0.5 * (RAW_Y_MIN + RAW_Y_MAX)) * (
        (TARGET_Y_MAX - TARGET_Y_MIN) / (RAW_Y_MAX - RAW_Y_MIN)
    )


def scale_len_x(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * ((TARGET_X_MAX - TARGET_X_MIN) / (RAW_X_MAX - RAW_X_MIN))


def scale_len_y(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * ((TARGET_Y_MAX - TARGET_Y_MIN) / (RAW_Y_MAX - RAW_Y_MIN))


def normalize_defect_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "defect_center_x" in out:
        out["defect_center_x"] = scale_x(out["defect_center_x"])
    if "defect_center_y" in out:
        out["defect_center_y"] = scale_y(out["defect_center_y"])
    if "defect_axis_x" in out:
        out["defect_axis_x"] = scale_len_x(out["defect_axis_x"])
    if "defect_axis_y" in out:
        out["defect_axis_y"] = scale_len_y(out["defect_axis_y"])
    return out


def normalize_structured_defect_params(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    names = out.dtype.names or ()
    if "defect_center_x" in names:
        out["defect_center_x"] = scale_x(out["defect_center_x"])
    if "defect_center_y" in names:
        out["defect_center_y"] = scale_y(out["defect_center_y"])
    if "defect_axis_x" in names:
        out["defect_axis_x"] = scale_len_x(out["defect_axis_x"])
    if "defect_axis_y" in names:
        out["defect_axis_y"] = scale_len_y(out["defect_axis_y"])
    return out


def _json_scalar(value) -> str:
    if isinstance(value, np.ndarray):
        return str(value.item())
    return str(value)


def normalize_npz(npz_path: Path, defect_csv: Path, output_npz: Path, output_csv: Path, split: str = "") -> None:
    npz = np.load(npz_path, allow_pickle=True)
    payload = {name: npz[name] for name in npz.files}
    raw_x = np.asarray(payload["x"], dtype=np.float64)
    raw_y = np.asarray(payload["y"], dtype=np.float64)
    payload["x"] = scale_x(raw_x).astype(np.float32)
    payload["y"] = scale_y(raw_y).astype(np.float32)
    payload["geometry_units"] = np.array("m")
    payload["v3_geometry_normalization"] = np.array(
        json.dumps(
            {
                "source_convention": "COMSOL V3 raw model coordinates",
                "target_convention": "V2-compatible centered meter-scale geometry",
                "x_raw_range": [RAW_X_MIN, RAW_X_MAX],
                "x_normalized_range": [TARGET_X_MIN, TARGET_X_MAX],
                "y_raw_range": [RAW_Y_MIN, RAW_Y_MAX],
                "y_normalized_range": [TARGET_Y_MIN, TARGET_Y_MAX],
                "x_formula": "(x_raw - 2250.0) * (0.08 / 4500.0)",
                "y_formula": "(y_raw - 1500.0) * (0.02 / 3000.0)",
                "depth_z_scaling": "not applied; raw z/depth retained",
            }
        )
    )
    if "metadata_json" in payload:
        try:
            metadata = json.loads(_json_scalar(payload["metadata_json"]))
        except json.JSONDecodeError:
            metadata = {"previous_metadata_json": _json_scalar(payload["metadata_json"])}
        metadata["geometry_normalization"] = json.loads(str(payload["v3_geometry_normalization"]))
        metadata["geometry_units"] = "m"
        metadata["normalized_split"] = split
        payload["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    if "defect_params" in payload and getattr(payload["defect_params"], "dtype", None) is not None:
        if payload["defect_params"].dtype.names:
            payload["defect_params"] = normalize_structured_defect_params(payload["defect_params"])

    df = pd.read_csv(defect_csv)
    out_df = normalize_defect_frame(df)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    out_df.to_csv(output_csv, index=False)


def main() -> int:
    args = parse_args()
    if not args.npz_path or not args.defect_params_csv or not args.output_npz or not args.output_defect_params_csv:
        return _usage_and_exit()
    normalize_npz(
        Path(args.npz_path),
        Path(args.defect_params_csv),
        Path(args.output_npz),
        Path(args.output_defect_params_csv),
        args.split,
    )
    print(f"Saved normalized V3 geometry NPZ to {args.output_npz}")
    print(f"Saved normalized V3 defect params CSV to {args.output_defect_params_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

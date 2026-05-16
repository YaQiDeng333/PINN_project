from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "comsol_mfl" / "rectangular_sweep_small"
DEFAULT_OUTPUT_NAME = "comsol_rectangular_sweep_small.npz"
SENSOR_FIELDS = ("x", "y", "z")
SIGNAL_FIELDS = (
    "Bz_no_defect",
    "Bz_defect",
    "delta_Bz",
    "normB_no_defect",
    "normB_defect",
    "delta_normB",
)
LABEL_PARAM_FIELDS = (
    "width_mm",
    "depth_mm",
    "length_mm",
    "center_x_mm",
    "center_y_mm",
    "center_z_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the small COMSOL MFL rectangular sweep CSV files into a unified intermediate NPZ."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing metadata.csv and sample_XXX.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path. Defaults to <dataset-dir>/processed/comsol_rectangular_sweep_small.npz.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Output summary JSON path. Defaults to <output-dir>/processed_summary.json.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def parse_float(value: str, field_name: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name}: field {field_name} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path.name}: field {field_name} is not finite: {value!r}")
    return parsed


def read_numeric_column(rows: list[dict[str, str]], field_name: str, path: Path) -> np.ndarray:
    return np.array(
        [parse_float(row[field_name], field_name, path) for row in rows],
        dtype=np.float64,
    )


def build_default_paths(dataset_dir: Path, output: Path | None, summary_output: Path | None) -> tuple[Path, Path]:
    if output is None:
        output = dataset_dir / "processed" / DEFAULT_OUTPUT_NAME
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()

    if summary_output is None:
        summary_output = output.parent / "processed_summary.json"
    if not summary_output.is_absolute():
        summary_output = (PROJECT_ROOT / summary_output).resolve()

    return output, summary_output


def prepare_dataset(dataset_dir: Path, output_path: Path, summary_path: Path) -> dict[str, object]:
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata.csv: {metadata_path}")

    metadata_fields, metadata_rows = load_csv_rows(metadata_path)
    required_metadata = {"sample_id", "max_abs_delta_Bz", "x_at_max_abs_delta_Bz", *LABEL_PARAM_FIELDS}
    missing_metadata = sorted(required_metadata - set(metadata_fields))
    if missing_metadata:
        raise ValueError(f"metadata.csv missing fields: {', '.join(missing_metadata)}")
    if not metadata_rows:
        raise ValueError("metadata.csv has no samples")

    metadata_rows = sorted(metadata_rows, key=lambda row: row["sample_id"])
    sample_ids = np.array([row["sample_id"] for row in metadata_rows], dtype="<U32")

    sensor_reference: dict[str, np.ndarray] | None = None
    all_same_sensor_grid = True
    signal_arrays = {field: [] for field in SIGNAL_FIELDS}
    csv_max_abs_delta_bz: list[float] = []

    for metadata_row in metadata_rows:
        sample_id = metadata_row["sample_id"]
        sample_path = dataset_dir / f"{sample_id}.csv"
        if not sample_path.exists():
            raise FileNotFoundError(f"missing sample CSV for {sample_id}: {sample_path}")

        fieldnames, rows = load_csv_rows(sample_path)
        required_sample_fields = set(SENSOR_FIELDS) | set(SIGNAL_FIELDS)
        missing_sample_fields = sorted(required_sample_fields - set(fieldnames))
        if missing_sample_fields:
            raise ValueError(f"{sample_path.name} missing fields: {', '.join(missing_sample_fields)}")
        if not rows:
            raise ValueError(f"{sample_path.name} has no rows")

        sensor_current = {
            field: read_numeric_column(rows, field, sample_path)
            for field in SENSOR_FIELDS
        }
        if sensor_reference is None:
            sensor_reference = sensor_current
        else:
            for field in SENSOR_FIELDS:
                if not np.array_equal(sensor_reference[field], sensor_current[field]):
                    all_same_sensor_grid = False

        for field in SIGNAL_FIELDS:
            signal_arrays[field].append(read_numeric_column(rows, field, sample_path))

        delta_bz = signal_arrays["delta_Bz"][-1]
        csv_max = float(np.max(np.abs(delta_bz)))
        csv_max_abs_delta_bz.append(csv_max)
        metadata_max = parse_float(metadata_row["max_abs_delta_Bz"], "max_abs_delta_Bz", metadata_path)
        if not math.isclose(csv_max, metadata_max, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{sample_id}: metadata max_abs_delta_Bz={metadata_max:.16g} "
                f"does not match CSV recomputed value={csv_max:.16g}"
            )

    assert sensor_reference is not None
    stacked_signals = {
        field: np.stack(values, axis=0)
        for field, values in signal_arrays.items()
    }
    label_params_mm = np.array(
        [
            [parse_float(row[field], field, metadata_path) for field in LABEL_PARAM_FIELDS]
            for row in metadata_rows
        ],
        dtype=np.float64,
    )
    max_abs_delta_bz = np.array(
        [parse_float(row["max_abs_delta_Bz"], "max_abs_delta_Bz", metadata_path) for row in metadata_rows],
        dtype=np.float64,
    )
    x_at_max_abs_delta_bz = np.array(
        [
            parse_float(row["x_at_max_abs_delta_Bz"], "x_at_max_abs_delta_Bz", metadata_path)
            for row in metadata_rows
        ],
        dtype=np.float64,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_ids=sample_ids,
        sensor_x=sensor_reference["x"],
        sensor_y=sensor_reference["y"],
        sensor_z=sensor_reference["z"],
        Bz_no_defect=stacked_signals["Bz_no_defect"],
        Bz_defect=stacked_signals["Bz_defect"],
        delta_Bz=stacked_signals["delta_Bz"],
        normB_no_defect=stacked_signals["normB_no_defect"],
        normB_defect=stacked_signals["normB_defect"],
        delta_normB=stacked_signals["delta_normB"],
        label_params_mm=label_params_mm,
        label_param_names=np.array(LABEL_PARAM_FIELDS, dtype="<U32"),
        max_abs_delta_Bz=max_abs_delta_bz,
        x_at_max_abs_delta_Bz=x_at_max_abs_delta_bz,
    )

    arrays = {
        "sample_ids": sample_ids,
        "sensor_x": sensor_reference["x"],
        "sensor_y": sensor_reference["y"],
        "sensor_z": sensor_reference["z"],
        **stacked_signals,
        "label_params_mm": label_params_mm,
        "label_param_names": np.array(LABEL_PARAM_FIELDS, dtype="<U32"),
        "max_abs_delta_Bz": max_abs_delta_bz,
        "x_at_max_abs_delta_Bz": x_at_max_abs_delta_bz,
    }
    summary = {
        "source_dataset_dir": str(dataset_dir),
        "output_npz": str(output_path),
        "sample_count": int(len(sample_ids)),
        "sensor_point_count": int(sensor_reference["x"].shape[0]),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "delta_Bz_min": float(np.min(stacked_signals["delta_Bz"])),
        "delta_Bz_max": float(np.max(stacked_signals["delta_Bz"])),
        "per_sample_max_abs_delta_Bz": {
            str(sample_id): float(value)
            for sample_id, value in zip(sample_ids, max_abs_delta_bz)
        },
        "csv_recomputed_max_abs_delta_Bz": {
            str(sample_id): float(value)
            for sample_id, value in zip(sample_ids, csv_max_abs_delta_bz)
        },
        "all_samples_same_sensor_grid": bool(all_same_sensor_grid),
        "sample_ids": [str(sample_id) for sample_id in sample_ids],
        "label_param_names": list(LABEL_PARAM_FIELDS),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return summary


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_path, summary_path = build_default_paths(dataset_dir, args.output, args.summary_output)
    summary = prepare_dataset(dataset_dir, output_path, summary_path)

    print(f"Saved NPZ: {output_path}")
    print(f"Saved summary: {summary_path}")
    print(f"sample_count: {summary['sample_count']}")
    print(f"sensor_point_count: {summary['sensor_point_count']}")
    print(f"all_samples_same_sensor_grid: {summary['all_samples_same_sensor_grid']}")
    print(f"delta_Bz_min: {summary['delta_Bz_min']:.12g}")
    print(f"delta_Bz_max: {summary['delta_Bz_max']:.12g}")
    for name, shape in summary["array_shapes"].items():
        print(f"{name}: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

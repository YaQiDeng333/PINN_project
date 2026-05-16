from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "comsol_mfl" / "rectangular_sweep_small"
DEFAULT_PROCESSED_NPZ = (
    DEFAULT_DATASET_DIR / "processed" / "comsol_rectangular_sweep_small.npz"
)
EXPECTED_SAMPLE_IDS = [f"sample_{index:03d}" for index in range(1, 6)]
REQUIRED_SAMPLE_FIELDS = [
    "x",
    "y",
    "z",
    "Bz_no_defect",
    "Bz_defect",
    "delta_Bz",
    "normB_no_defect",
    "normB_defect",
    "delta_normB",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the local COMSOL MFL rectangular sweep intake files."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing metadata.csv and sample_XXX.csv files.",
    )
    parser.add_argument(
        "--expected-points",
        type=int,
        default=201,
        help="Expected fixed sensor line point count for each sample CSV.",
    )
    parser.add_argument(
        "--processed-npz",
        type=Path,
        default=DEFAULT_PROCESSED_NPZ,
        help="Processed intermediate NPZ path to validate.",
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Only validate raw CSV files and metadata.csv.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def parse_float(value: str, field_name: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name}: field {field_name} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path.name}: field {field_name} is not finite: {value!r}")
    return parsed


def inspect_sample(
    dataset_dir: Path,
    metadata_row: dict[str, str],
    expected_points: int,
) -> tuple[str, float, list[str]]:
    sample_id = metadata_row.get("sample_id", "").strip()
    sample_path = dataset_dir / f"{sample_id}.csv"
    errors: list[str] = []

    if not sample_id:
        return "<missing sample_id>", math.nan, ["metadata row has empty sample_id"]
    if not sample_path.exists():
        return sample_id, math.nan, [f"missing sample file: {sample_path}"]

    fieldnames, rows = load_csv_rows(sample_path)
    missing_fields = [field for field in REQUIRED_SAMPLE_FIELDS if field not in fieldnames]
    if missing_fields:
        errors.append(f"missing required fields: {', '.join(missing_fields)}")

    if len(rows) != expected_points:
        errors.append(f"expected {expected_points} rows, found {len(rows)}")

    max_abs_delta_bz = math.nan
    if "delta_Bz" in fieldnames:
        try:
            delta_values = [
                abs(parse_float(row["delta_Bz"], "delta_Bz", sample_path))
                for row in rows
            ]
            max_abs_delta_bz = max(delta_values) if delta_values else math.nan
        except ValueError as exc:
            errors.append(str(exc))

    for field in ("x", "y", "z", "Bz_no_defect", "Bz_defect", "normB_no_defect", "normB_defect"):
        if field not in fieldnames:
            continue
        try:
            for row in rows:
                parse_float(row[field], field, sample_path)
        except ValueError as exc:
            errors.append(str(exc))
            break

    metadata_points = metadata_row.get("points", "").strip()
    if metadata_points:
        try:
            if int(float(metadata_points)) != expected_points:
                errors.append(
                    f"metadata points={metadata_points}, expected {expected_points}"
                )
        except ValueError:
            errors.append(f"metadata points is not numeric: {metadata_points!r}")

    metadata_max_raw = metadata_row.get("max_abs_delta_Bz", "").strip()
    if metadata_max_raw:
        try:
            metadata_max = float(metadata_max_raw)
            if math.isfinite(max_abs_delta_bz) and not math.isclose(
                max_abs_delta_bz, metadata_max, rel_tol=1e-9, abs_tol=1e-12
            ):
                errors.append(
                    "metadata max_abs_delta_Bz mismatch: "
                    f"metadata={metadata_max:.16g}, recomputed={max_abs_delta_bz:.16g}"
                )
        except ValueError:
            errors.append(f"metadata max_abs_delta_Bz is not numeric: {metadata_max_raw!r}")
    else:
        errors.append("metadata max_abs_delta_Bz is empty")

    return sample_id, max_abs_delta_bz, errors


def inspect_processed_npz(
    npz_path: Path,
    expected_points: int,
    recomputed_max_by_sample: dict[str, float],
) -> list[str]:
    errors: list[str] = []
    if not npz_path.exists():
        return [f"missing processed NPZ: {npz_path}"]

    data = np.load(npz_path, allow_pickle=False)
    expected_shapes = {
        "sample_ids": (len(EXPECTED_SAMPLE_IDS),),
        "sensor_x": (expected_points,),
        "sensor_y": (expected_points,),
        "sensor_z": (expected_points,),
        "Bz_no_defect": (len(EXPECTED_SAMPLE_IDS), expected_points),
        "Bz_defect": (len(EXPECTED_SAMPLE_IDS), expected_points),
        "delta_Bz": (len(EXPECTED_SAMPLE_IDS), expected_points),
        "normB_no_defect": (len(EXPECTED_SAMPLE_IDS), expected_points),
        "normB_defect": (len(EXPECTED_SAMPLE_IDS), expected_points),
        "delta_normB": (len(EXPECTED_SAMPLE_IDS), expected_points),
        "label_params_mm": (len(EXPECTED_SAMPLE_IDS), 6),
        "max_abs_delta_Bz": (len(EXPECTED_SAMPLE_IDS),),
        "x_at_max_abs_delta_Bz": (len(EXPECTED_SAMPLE_IDS),),
    }

    for key, expected_shape in expected_shapes.items():
        if key not in data.files:
            errors.append(f"processed NPZ missing array: {key}")
            continue
        if tuple(data[key].shape) != expected_shape:
            errors.append(
                f"processed NPZ {key} shape={tuple(data[key].shape)}, expected={expected_shape}"
            )

    if errors:
        return errors

    sample_ids = [str(value) for value in data["sample_ids"]]
    if sample_ids != EXPECTED_SAMPLE_IDS:
        errors.append(f"processed NPZ sample_ids={sample_ids}, expected={EXPECTED_SAMPLE_IDS}")

    for key in ("sensor_x", "sensor_y", "sensor_z", "delta_Bz", "max_abs_delta_Bz"):
        if not np.isfinite(data[key]).all():
            errors.append(f"processed NPZ {key} contains NaN or Inf")

    for index, sample_id in enumerate(sample_ids):
        csv_max = recomputed_max_by_sample.get(sample_id)
        if csv_max is None:
            errors.append(f"processed NPZ sample {sample_id} has no CSV recomputed max")
            continue
        npz_max = float(data["max_abs_delta_Bz"][index])
        signal_max = float(np.max(np.abs(data["delta_Bz"][index])))
        if not math.isclose(npz_max, csv_max, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(
                f"{sample_id}: NPZ max_abs_delta_Bz={npz_max:.16g}, CSV recomputed={csv_max:.16g}"
            )
        if not math.isclose(signal_max, csv_max, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(
                f"{sample_id}: NPZ delta_Bz max={signal_max:.16g}, CSV recomputed={csv_max:.16g}"
            )

    return errors


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    processed_npz = args.processed_npz.resolve()
    metadata_path = dataset_dir / "metadata.csv"

    errors: list[str] = []
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory does not exist: {dataset_dir}")
        return 1
    if not metadata_path.exists():
        print(f"ERROR: missing metadata.csv: {metadata_path}")
        return 1

    metadata_fields, metadata_rows = load_csv_rows(metadata_path)
    metadata_by_sample = {
        row.get("sample_id", "").strip(): row
        for row in metadata_rows
        if row.get("sample_id", "").strip()
    }

    required_metadata_fields = {
        "sample_id",
        "defect_type",
        "width_mm",
        "depth_mm",
        "length_mm",
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
        "sensor_z_mm",
        "max_abs_delta_Bz",
        "x_at_max_abs_delta_Bz",
        "solve_status",
        "output_csv",
    }
    missing_metadata_fields = sorted(required_metadata_fields - set(metadata_fields))
    if missing_metadata_fields:
        errors.append(f"metadata.csv missing fields: {', '.join(missing_metadata_fields)}")

    print(f"Dataset: {dataset_dir}")
    print(f"Metadata rows: {len(metadata_rows)}")

    passed_samples = 0
    recomputed_max_by_sample: dict[str, float] = {}
    for sample_id in EXPECTED_SAMPLE_IDS:
        row = metadata_by_sample.get(sample_id)
        if row is None:
            print(f"{sample_id}: ERROR missing metadata row")
            errors.append(f"{sample_id}: missing metadata row")
            continue

        inspected_id, max_abs_delta_bz, sample_errors = inspect_sample(
            dataset_dir=dataset_dir,
            metadata_row=row,
            expected_points=args.expected_points,
        )
        solve_status = row.get("solve_status", "").strip()
        if sample_errors:
            print(f"{inspected_id}: ERROR {'; '.join(sample_errors)}")
            errors.extend(f"{inspected_id}: {error}" for error in sample_errors)
            continue

        passed_samples += 1
        recomputed_max_by_sample[inspected_id] = max_abs_delta_bz
        print(
            f"{inspected_id}: rows={args.expected_points}, "
            f"solve_status={solve_status}, "
            f"max_abs_delta_Bz={max_abs_delta_bz:.12g}"
        )

    extra_samples = sorted(set(metadata_by_sample) - set(EXPECTED_SAMPLE_IDS))
    if extra_samples:
        errors.append(f"unexpected metadata sample_id values: {', '.join(extra_samples)}")

    if not args.skip_processed and not errors:
        processed_errors = inspect_processed_npz(
            npz_path=processed_npz,
            expected_points=args.expected_points,
            recomputed_max_by_sample=recomputed_max_by_sample,
        )
        if processed_errors:
            errors.extend(processed_errors)
        else:
            print(f"processed_npz: OK {processed_npz}")

    if errors:
        print(f"Summary: FAILED, passed_samples={passed_samples}/{len(EXPECTED_SAMPLE_IDS)}")
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        "Summary: OK, "
        f"passed_samples={passed_samples}/{len(EXPECTED_SAMPLE_IDS)}, "
        f"all sample CSV files have {args.expected_points} fixed sensor points, "
        "metadata max_abs_delta_Bz matches recomputed values, "
        "processed NPZ matches raw CSV values."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

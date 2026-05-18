from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "generated"
    / "comsol_single_defect_multiline_forward_pack_v1"
)
DEFAULT_NPZ_PATH = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "prepared"
    / "comsol_single_defect_multiline_forward_pack_v1_smoke.npz"
)
INVENTORY_FIELDS = [
    "sample_id",
    "generated_real_data",
    "defect_type",
    "geometry_params_summary",
    "has_mask",
    "has_mask_coords",
    "has_sensor_coords",
    "has_multiline",
    "n_lines",
    "has_bz_defect",
    "has_bz_reference",
    "has_delta_bz",
    "signal_shape",
    "mask_shape",
    "mask_area",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a generated COMSOL single-defect multi-line smoke directory."
    )
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument("--inventory-output", type=Path, default=None)
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def summarize_geometry(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False, sort_keys=True)


def row_from_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        item = json.load(file)

    mask_shape = item.get("mask_shape", "")
    signal_shape = item.get("signal_shape", "")
    scan_line_y = item.get("scan_line_y", [])
    n_lines = len(scan_line_y) if isinstance(scan_line_y, list) else ""
    return {
        "sample_id": item.get("sample_id", path.stem),
        "generated_real_data": bool(item.get("generated_real_data", False)),
        "defect_type": item.get("defect_type", ""),
        "geometry_params_summary": summarize_geometry(item.get("geometry_params")),
        "has_mask": bool(item.get("has_mask", False)),
        "has_mask_coords": bool(item.get("has_mask_coords", False)),
        "has_sensor_coords": bool(item.get("has_sensor_coords", False)),
        "has_multiline": bool(item.get("has_multiline", False)),
        "n_lines": n_lines,
        "has_bz_defect": bool(item.get("has_bz_defect", False)),
        "has_bz_reference": bool(item.get("has_bz_reference", False)),
        "has_delta_bz": bool(item.get("has_delta_bz", False)),
        "signal_shape": signal_shape,
        "mask_shape": mask_shape,
        "mask_area": item.get("mask_area", ""),
        "notes": item.get("notes", ""),
    }


def inspect_npz(npz_path: Path) -> dict[str, Any]:
    if not npz_path.exists():
        return {"exists": False, "path": str(npz_path)}

    data = np.load(npz_path, allow_pickle=True)
    arrays = {name: data[name] for name in data.files}
    required = [
        "masks",
        "sensor_x",
        "scan_line_y",
        "mask_x",
        "mask_y",
        "sample_ids",
        "defect_types",
    ]
    has_signal = any(name in arrays for name in ("signals", "delta_bz", "delta_Bz"))
    missing = [name for name in required if name not in arrays]
    if not has_signal:
        missing.append("signals_or_delta_bz")

    report: dict[str, Any] = {
        "exists": True,
        "path": str(npz_path),
        "files": list(data.files),
        "missing_required": missing,
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "finite_numeric_arrays": {},
        "train_ready_minimal": not missing,
    }
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number):
            report["finite_numeric_arrays"][name] = bool(np.isfinite(value).all())
    return report


def write_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    generated_dir = normalize_path(args.generated_dir)
    npz_path = normalize_path(args.npz)
    manifest_paths = sorted(generated_dir.glob("sample_*.json")) if generated_dir.exists() else []
    rows = [row_from_manifest(path) for path in manifest_paths]
    npz_report = inspect_npz(npz_path)

    if args.inventory_output is not None:
        write_inventory(normalize_path(args.inventory_output), rows)

    print(
        json.dumps(
            {
                "generated_dir": str(generated_dir),
                "manifest_count": len(manifest_paths),
                "inventory_rows": len(rows),
                "npz": npz_report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if npz_report.get("train_ready_minimal", False) or not npz_report["exists"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_local_perturbation_plan.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_local_perturbation_plan_summary.txt"

MAIN_TYPES = ("rectangular_notch", "rotated_rect")
TARGET_BASE_COUNTS = {
    ("train", "rectangular_notch"): 8,
    ("train", "rotated_rect"): 8,
    ("val", "rectangular_notch"): 2,
    ("val", "rotated_rect"): 2,
    ("test", "rectangular_notch"): 2,
    ("test", "rotated_rect"): 2,
}

VARIANTS = [
    "true_geometry_reference",
    "center_shift_small",
    "center_shift_large",
    "size_scale_small",
    "size_scale_large",
    "angle_shift_small",
    "angle_shift_large",
    "mixed_center_size_angle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Design a small rectangular/rotated COMSOL local geometry perturbation plan. "
            "The script does not run COMSOL and does not write NPZ/data artifacts."
        )
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        return float((lo + hi) / 2.0)
    return float(min(max(value, lo), hi))


def circular_angle_delta_deg(a: float, b: float) -> float:
    return abs(float((a - b + 90.0) % 180.0 - 90.0))


def rasterize_rect(
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    length: float,
    angle_rad: float,
) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(mask_x, mask_y)
    dx0 = x_grid - center_x
    dy0 = y_grid - center_y
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x_rot = dx0 * cos_a + dy0 * sin_a
    y_rot = -dx0 * sin_a + dy0 * cos_a
    return ((np.abs(x_rot) <= width / 2.0) & (np.abs(y_rot) <= length / 2.0)).astype(np.uint8)


def mask_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred_b = pred > 0
    true_b = true > 0
    inter = int(np.logical_and(pred_b, true_b).sum())
    union = int(np.logical_or(pred_b, true_b).sum())
    pred_area = int(pred_b.sum())
    true_area = int(true_b.sum())
    iou = inter / union if union else 1.0
    dice = 2.0 * inter / (pred_area + true_area) if (pred_area + true_area) else 1.0
    area_error = abs(pred_area - true_area) / max(true_area, 1)
    return {
        "mask_iou": float(iou),
        "mask_dice": float(dice),
        "area_error": float(area_error),
        "pred_area_px": float(pred_area),
        "true_area_px": float(true_area),
    }


def select_coverage_indices(records: list[dict[str, Any]], count: int) -> list[int]:
    if len(records) < count:
        raise RuntimeError(f"Need {count} records but only found {len(records)}")
    if count <= 0:
        return []
    features = np.array(
        [
            [
                float(r["center_x"]),
                float(r["center_y"]),
                float(r["width"]),
                float(r["length"]),
                float(r["depth"]),
                float(r["angle_deg"]),
                float(r["mask_area"]),
            ]
            for r in records
        ],
        dtype=np.float64,
    )
    std = features.std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    z = (features - features.mean(axis=0)) / std
    score = z[:, 0] * 0.41 + z[:, 1] * 0.37 + z[:, 2] * 0.31 + z[:, 3] * 0.29 + z[:, 4] * 0.23 + z[:, 5] * 0.53
    order = np.argsort(score, kind="mergesort")
    if count == 1:
        return [int(order[len(order) // 2])]
    positions = np.linspace(0, len(order) - 1, count)
    selected: list[int] = []
    used: set[int] = set()
    for pos in positions:
        base = int(round(float(pos)))
        for radius in range(len(order)):
            for cand_pos in (base - radius, base + radius):
                if 0 <= cand_pos < len(order):
                    idx = int(order[cand_pos])
                    if idx not in used:
                        selected.append(idx)
                        used.add(idx)
                        break
            if len(selected) == len(used) and selected[-1] in used:
                break
    return selected[:count]


def geometry_bounds(records: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for key in ["width", "length", "depth"]:
        vals = np.array([float(r[key]) for r in records], dtype=np.float64)
        lo = max(float(vals.min() * 0.70), 1.0e-5)
        hi = float(vals.max() * 1.30)
        bounds[key] = (lo, hi)
    return bounds


def clamp_geometry(geom: dict[str, float], bounds: dict[str, tuple[float, float]], mask_x: np.ndarray, mask_y: np.ndarray) -> dict[str, float]:
    out = dict(geom)
    out["width"] = clamp(float(out["width"]), *bounds["width"])
    out["length"] = clamp(float(out["length"]), *bounds["length"])
    out["depth"] = clamp(float(out["depth"]), *bounds["depth"])
    out["angle_deg"] = clamp(float(out["angle_deg"]), -35.0, 35.0)
    out["angle_rad"] = math.radians(out["angle_deg"])
    radius = math.sqrt((out["width"] / 2.0) ** 2 + (out["length"] / 2.0) ** 2) + 3.0e-4
    out["center_x"] = clamp(float(out["center_x"]), float(mask_x.min() + radius), float(mask_x.max() - radius))
    out["center_y"] = clamp(float(out["center_y"]), float(mask_y.min() + radius), float(mask_y.max() - radius))
    return out


def perturb_geometry(base: dict[str, Any], variant: str, order_index: int, bounds: dict[str, tuple[float, float]], mask_x: np.ndarray, mask_y: np.ndarray) -> tuple[dict[str, float], dict[str, float], int]:
    geom = {
        "center_x": float(base["center_x"]),
        "center_y": float(base["center_y"]),
        "width": float(base["width"]),
        "length": float(base["length"]),
        "depth": float(base["depth"]),
        "angle_deg": float(base["angle_deg"]),
        "angle_rad": float(base["angle_rad"]),
    }
    deltas = {
        "delta_center_x": 0.0,
        "delta_center_y": 0.0,
        "width_scale": 1.0,
        "length_scale": 1.0,
        "depth_scale": 1.0,
        "delta_angle_deg": 0.0,
    }
    quality_rank = 0
    sx = -1.0 if order_index % 2 else 1.0
    sy = -1.0 if (order_index // 2) % 2 else 1.0
    angle_sign = -1.0 if (order_index // 3) % 2 else 1.0

    if variant == "true_geometry_reference":
        quality_rank = 0
    elif variant == "center_shift_small":
        deltas["delta_center_x"] = sx * 0.00075
        deltas["delta_center_y"] = sy * 0.00050
        quality_rank = 1
    elif variant == "center_shift_large":
        deltas["delta_center_x"] = sx * 0.00200
        deltas["delta_center_y"] = sy * 0.00140
        quality_rank = 2
    elif variant == "size_scale_small":
        deltas["width_scale"] = 1.08 if sx > 0 else 0.92
        deltas["length_scale"] = 0.93 if sy > 0 else 1.07
        deltas["depth_scale"] = 1.06
        quality_rank = 1
    elif variant == "size_scale_large":
        deltas["width_scale"] = 1.22 if sx > 0 else 0.80
        deltas["length_scale"] = 0.82 if sy > 0 else 1.20
        deltas["depth_scale"] = 0.85
        quality_rank = 2
    elif variant == "angle_shift_small":
        deltas["delta_angle_deg"] = angle_sign * 5.0
        quality_rank = 1
    elif variant == "angle_shift_large":
        deltas["delta_angle_deg"] = angle_sign * 15.0
        quality_rank = 2
    elif variant == "mixed_center_size_angle":
        deltas["delta_center_x"] = sx * 0.00150
        deltas["delta_center_y"] = sy * 0.00100
        deltas["width_scale"] = 1.15 if sx > 0 else 0.86
        deltas["length_scale"] = 0.88 if sy > 0 else 1.14
        deltas["depth_scale"] = 1.10 if angle_sign > 0 else 0.90
        deltas["delta_angle_deg"] = angle_sign * 10.0
        quality_rank = 3
    else:
        raise ValueError(f"Unknown variant: {variant}")

    geom["center_x"] += deltas["delta_center_x"]
    geom["center_y"] += deltas["delta_center_y"]
    geom["width"] *= deltas["width_scale"]
    geom["length"] *= deltas["length_scale"]
    geom["depth"] *= deltas["depth_scale"]
    geom["angle_deg"] += deltas["delta_angle_deg"]
    geom = clamp_geometry(geom, bounds, mask_x, mask_y)
    deltas["actual_delta_center_x"] = geom["center_x"] - float(base["center_x"])
    deltas["actual_delta_center_y"] = geom["center_y"] - float(base["center_y"])
    deltas["actual_delta_angle_deg"] = geom["angle_deg"] - float(base["angle_deg"])
    deltas["actual_width_scale"] = geom["width"] / max(float(base["width"]), 1.0e-12)
    deltas["actual_length_scale"] = geom["length"] / max(float(base["length"]), 1.0e-12)
    deltas["actual_depth_scale"] = geom["depth"] / max(float(base["depth"]), 1.0e-12)
    return geom, deltas, quality_rank


def build_records(data: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_ids = data["sample_ids"].astype(str)
    split = data["split"].astype(str)
    defect_types = data["defect_types"].astype(str)
    geometry_params = data["geometry_params"]
    masks = data["masks"]
    for idx, sample_id in enumerate(sample_ids):
        defect_type = str(defect_types[idx])
        if defect_type not in MAIN_TYPES:
            continue
        geom = parse_json(geometry_params[idx])
        angle_deg = as_float(geom.get("angle_deg"), 0.0)
        angle_rad = as_float(geom.get("angle_rad"), math.radians(angle_deg))
        if defect_type == "rectangular_notch":
            angle_deg = 0.0
            angle_rad = 0.0
        records.append(
            {
                "source_index": idx,
                "sample_id": str(sample_id),
                "split": str(split[idx]),
                "defect_type": defect_type,
                "source_pack": str(geom.get("source_pack", "")),
                "source_sample_id": str(geom.get("source_sample_id", "")),
                "center_x": as_float(geom.get("center_x")),
                "center_y": as_float(geom.get("center_y")),
                "width": as_float(geom.get("width")),
                "length": as_float(geom.get("length")),
                "depth": as_float(geom.get("depth")),
                "angle_deg": angle_deg,
                "angle_rad": angle_rad,
                "mask_area": int((masks[idx] > 0).sum()),
            }
        )
    return records


def design_plan(npz_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    required = [
        "sample_ids",
        "split",
        "defect_types",
        "geometry_params",
        "masks",
        "delta_bz",
        "sensor_x",
        "scan_line_y",
        "mask_x",
        "mask_y",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing required NPZ keys: {missing}")
    mask_x = data["mask_x"].astype(np.float64)
    mask_y = data["mask_y"].astype(np.float64)
    masks = data["masks"].astype(np.uint8)
    records = build_records(data)
    bounds = geometry_bounds(records)

    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[(record["split"], record["defect_type"])].append(record)

    base_records: list[dict[str, Any]] = []
    for group, count in TARGET_BASE_COUNTS.items():
        group_records = by_group[group]
        selected_indices = select_coverage_indices(group_records, count)
        for local_idx in selected_indices:
            base_records.append(group_records[local_idx])

    rows: list[dict[str, Any]] = []
    for base_ordinal, base in enumerate(base_records):
        true_mask = masks[int(base["source_index"])]
        true_geom = {
            "center_x": float(base["center_x"]),
            "center_y": float(base["center_y"]),
            "width": float(base["width"]),
            "length": float(base["length"]),
            "depth": float(base["depth"]),
            "angle_deg": float(base["angle_deg"]),
            "angle_rad": float(base["angle_rad"]),
        }
        for variant_ordinal, variant in enumerate(VARIANTS):
            geom, deltas, quality_rank = perturb_geometry(base, variant, base_ordinal + variant_ordinal, bounds, mask_x, mask_y)
            pred_mask = rasterize_rect(
                mask_x,
                mask_y,
                geom["center_x"],
                geom["center_y"],
                geom["width"],
                geom["length"],
                geom["angle_rad"],
            )
            metrics = mask_metrics(pred_mask, true_mask)
            center_delta = math.sqrt(
                (geom["center_x"] - true_geom["center_x"]) ** 2 + (geom["center_y"] - true_geom["center_y"]) ** 2
            )
            size_delta_norm = (
                abs(geom["width"] - true_geom["width"]) / max(true_geom["width"], 1.0e-12)
                + abs(geom["length"] - true_geom["length"]) / max(true_geom["length"], 1.0e-12)
            ) / 2.0
            angle_delta_abs = circular_angle_delta_deg(geom["angle_deg"], true_geom["angle_deg"])
            forward_geometry_type = "rectangular_notch" if abs(geom["angle_deg"]) < 1.0e-9 else "rotated_rect"
            geometry_params = {
                "center_x": geom["center_x"],
                "center_y": geom["center_y"],
                "width": geom["width"],
                "length": geom["length"],
                "depth": geom["depth"],
                "angle_deg": geom["angle_deg"],
                "angle_rad": geom["angle_rad"],
                "angle_sin": math.sin(geom["angle_rad"]),
                "angle_cos": math.cos(geom["angle_rad"]),
                "base_defect_type": base["defect_type"],
                "forward_geometry_type": forward_geometry_type,
                "units": "coordinates=m, Bz=T",
            }
            quality = {
                **metrics,
                "center_delta_m": float(center_delta),
                "angle_delta_abs_deg": float(angle_delta_abs),
                "size_delta_norm": float(size_delta_norm),
                "depth_delta_abs_m": float(abs(geom["depth"] - true_geom["depth"])),
            }
            perturb_sample_id = f"{base['sample_id']}__perturb_{variant_ordinal:02d}_{variant}"
            rows.append(
                {
                    "plan_row_index": len(rows),
                    "base_ordinal": base_ordinal,
                    "variant_ordinal": variant_ordinal,
                    "base_sample_id": base["sample_id"],
                    "base_source_index": base["source_index"],
                    "perturb_sample_id": perturb_sample_id,
                    "split": base["split"],
                    "defect_type": base["defect_type"],
                    "forward_geometry_type": forward_geometry_type,
                    "source_pack": base["source_pack"],
                    "source_sample_id": base["source_sample_id"],
                    "variant_type": variant,
                    "perturb_level": (
                        "reference"
                        if quality_rank == 0
                        else "small"
                        if quality_rank == 1
                        else "large"
                        if quality_rank == 2
                        else "mixed"
                    ),
                    "expected_quality_rank": quality_rank,
                    "requires_comsol_forward": str(variant != "true_geometry_reference").lower(),
                    "reference_delta_bz_source": f"{npz_path.name}:{base['sample_id']}",
                    "center_x": geom["center_x"],
                    "center_y": geom["center_y"],
                    "width": geom["width"],
                    "length": geom["length"],
                    "depth": geom["depth"],
                    "angle_rad": geom["angle_rad"],
                    "angle_deg": geom["angle_deg"],
                    "angle_sin": math.sin(geom["angle_rad"]),
                    "angle_cos": math.cos(geom["angle_rad"]),
                    "base_center_x": true_geom["center_x"],
                    "base_center_y": true_geom["center_y"],
                    "base_width": true_geom["width"],
                    "base_length": true_geom["length"],
                    "base_depth": true_geom["depth"],
                    "base_angle_rad": true_geom["angle_rad"],
                    "base_angle_deg": true_geom["angle_deg"],
                    "geometry_mask_iou_vs_true": metrics["mask_iou"],
                    "geometry_mask_dice_vs_true": metrics["mask_dice"],
                    "area_error_vs_true": metrics["area_error"],
                    "pred_mask_area_px": metrics["pred_area_px"],
                    "true_mask_area_px": metrics["true_area_px"],
                    "center_delta_m": center_delta,
                    "angle_delta_abs_deg": angle_delta_abs,
                    "size_delta_norm": size_delta_norm,
                    "depth_delta_abs_m": abs(geom["depth"] - true_geom["depth"]),
                    "perturbation_deltas": json_dumps(deltas),
                    "geometry_params": json_dumps(geometry_params),
                    "geometry_quality_to_true": json_dumps(quality),
                    "notes": (
                        "true_geometry_reference_reuses_original_delta_bz"
                        if variant == "true_geometry_reference"
                        else "requires_real_comsol_forward"
                    ),
                }
            )

    diagnostics = {
        "n_rows": len(rows),
        "n_base": len(base_records),
        "row_split_counts": dict(Counter(row["split"] for row in rows)),
        "base_split_counts": dict(Counter(row["split"] for row in base_records)),
        "row_type_counts": dict(Counter(row["defect_type"] for row in rows)),
        "base_type_counts": dict(Counter(row["defect_type"] for row in base_records)),
        "variant_counts": dict(Counter(row["variant_type"] for row in rows)),
        "forward_geometry_type_counts": dict(Counter(row["forward_geometry_type"] for row in rows)),
        "non_empty_masks": int(sum(float(row["pred_mask_area_px"]) > 0 for row in rows)),
        "bounds": {k: list(v) for k, v in bounds.items()},
        "selected_base_samples": [
            {
                "sample_id": r["sample_id"],
                "split": r["split"],
                "defect_type": r["defect_type"],
                "width": r["width"],
                "length": r["length"],
                "depth": r["depth"],
                "angle_deg": r["angle_deg"],
                "mask_area": r["mask_area"],
            }
            for r in base_records
        ],
    }
    expected_ok = (
        diagnostics["n_base"] == 24
        and diagnostics["n_rows"] == 192
        and diagnostics["row_split_counts"] == {"train": 128, "val": 32, "test": 32}
        and diagnostics["row_type_counts"] == {"rectangular_notch": 96, "rotated_rect": 96}
        and diagnostics["non_empty_masks"] == 192
    )
    diagnostics["passed"] = expected_ok
    if not expected_ok:
        raise RuntimeError(f"Perturbation plan checks failed: {diagnostics}")
    return rows, diagnostics


def write_outputs(rows: list[dict[str, Any]], diagnostics: dict[str, Any], csv_path: Path, summary_path: Path, npz_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    quality_by_variant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        quality_by_variant[row["variant_type"]].append(float(row["geometry_mask_iou_vs_true"]))
    variant_lines = [
        f"- {variant}: n={len(values)}, mean_geometry_iou={float(np.mean(values)):.6f}"
        for variant, values in sorted(quality_by_variant.items())
    ]
    base_lines = [
        f"- {r['split']} / {r['defect_type']} / {r['sample_id']}: width={r['width']:.6g}, "
        f"length={r['length']:.6g}, depth={r['depth']:.6g}, angle_deg={r['angle_deg']:.3f}, "
        f"mask_area={r['mask_area']}"
        for r in diagnostics["selected_base_samples"]
    ]
    lines = [
        "COMSOL rect/rot local perturbation plan summary",
        "",
        f"Input NPZ: {npz_path}",
        f"Plan CSV: {csv_path}",
        "Scope: rectangular_notch + rotated_rect only; polygon excluded.",
        "Purpose: small local perturbation forward-calibration pack for residual ordering audit.",
        "This script does not run COMSOL and does not write data/NPZ/checkpoint/preview artifacts.",
        "",
        f"Base samples: {diagnostics['n_base']}",
        f"Perturbation rows: {diagnostics['n_rows']}",
        f"Base split counts: {diagnostics['base_split_counts']}",
        f"Row split counts: {diagnostics['row_split_counts']}",
        f"Base type counts: {diagnostics['base_type_counts']}",
        f"Row type counts: {diagnostics['row_type_counts']}",
        f"Variant counts: {diagnostics['variant_counts']}",
        f"Forward geometry type counts: {diagnostics['forward_geometry_type_counts']}",
        f"All masks non-empty: {diagnostics['non_empty_masks'] == diagnostics['n_rows']}",
        f"Geometry bounds used for clipping: {diagnostics['bounds']}",
        "",
        "Variant quality check by raster IoU vs original true mask:",
        *variant_lines,
        "",
        "Selected base samples:",
        *base_lines,
        "",
        "Reference policy:",
        "- true_geometry_reference rows may reuse the original NPZ delta_bz and are explicitly marked requires_comsol_forward=false.",
        "- all other perturbation rows are marked requires_comsol_forward=true and must be solved by the COMSOL generation stage.",
        "",
        f"Stage A self-check passed: {diagnostics['passed']}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, diagnostics = design_plan(args.npz)
    write_outputs(rows, diagnostics, args.output_csv, args.summary, args.npz)
    print(f"Wrote {len(rows)} perturbation rows to {args.output_csv}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()

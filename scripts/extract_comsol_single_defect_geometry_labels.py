from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_LABELS = PROJECT_ROOT / "results/metrics/comsol_single_defect_geometry_labels.csv"
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_single_defect_geometry_label_extraction_summary.txt"
)

EXPECTED_N = 600
EXPECTED_SPLITS = {"train": 402, "val": 99, "test": 99}
EXPECTED_DEFECTS = {"rectangular_notch": 200, "rotated_rect": 200, "polygon": 200}
MAX_VERTICES = 6


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def _as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _bbox_area(width: float, length: float) -> float:
    if math.isnan(width) or math.isnan(length):
        return math.nan
    return float(width * length)


def _pad_vertices(vertices: Any) -> tuple[list[list[float]], list[int], int]:
    if not vertices:
        return [[math.nan, math.nan] for _ in range(MAX_VERTICES)], [0] * MAX_VERTICES, 0
    clean: list[list[float]] = []
    for item in vertices:
        if isinstance(item, dict):
            clean.append([_as_float(item.get("x")), _as_float(item.get("y"))])
        else:
            clean.append([_as_float(item[0]), _as_float(item[1])])
    count = len(clean)
    padded = clean[:MAX_VERTICES] + [[math.nan, math.nan] for _ in range(max(0, MAX_VERTICES - len(clean)))]
    mask = [1] * min(count, MAX_VERTICES) + [0] * max(0, MAX_VERTICES - count)
    return padded[:MAX_VERTICES], mask[:MAX_VERTICES], count


def _compute_train_stats(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    train_rows = [row for row in rows if row["split"] == "train"]
    for field in fields:
        values = np.array([row[field] for row in train_rows if not math.isnan(float(row[field]))], dtype=float)
        if values.size == 0:
            stats[field] = (math.nan, math.nan)
            continue
        mean = float(values.mean())
        std = float(values.std())
        if std <= 0:
            std = 1.0
        stats[field] = (mean, std)
    return stats


def extract_labels(npz_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    required = ["masks", "defect_types", "sample_ids", "geometry_params", "split"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing required NPZ keys: {missing}")

    masks = data["masks"]
    defect_types = data["defect_types"].astype(str)
    sample_ids = data["sample_ids"].astype(str)
    splits = data["split"].astype(str)
    geometry_params = data["geometry_params"]

    rows: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    for idx, sample_id in enumerate(sample_ids):
        defect_type = str(defect_types[idx])
        split = str(splits[idx])
        try:
            geom = _parse_json(geometry_params[idx])
        except Exception as exc:  # pragma: no cover - summarized for audit
            parse_failures.append(f"{sample_id}: {exc}")
            geom = {}

        center_x = _as_float(geom.get("center_x"))
        center_y = _as_float(geom.get("center_y"))
        width = _as_float(geom.get("width"))
        length = _as_float(geom.get("length"))
        depth = _as_float(geom.get("depth"))
        source_pack = str(geom.get("source_pack", ""))

        angle_rad = _as_float(geom.get("angle_rad"))
        angle_deg = _as_float(geom.get("angle_deg"))
        polygon_vertices = geom.get("polygon_vertices")
        vertex_count = int(_as_float(geom.get("vertex_count"), 0.0))

        notes: list[str] = []
        if defect_type == "rectangular_notch":
            geometry_type = "axis_aligned_rectangle"
            angle_rad = 0.0
            angle_deg = 0.0
            vertex_count = 0
            polygon_vertices = None
        elif defect_type == "rotated_rect":
            geometry_type = "rotated_rectangle"
            if math.isnan(angle_rad) and not math.isnan(angle_deg):
                angle_rad = math.radians(angle_deg)
            if math.isnan(angle_deg) and not math.isnan(angle_rad):
                angle_deg = math.degrees(angle_rad)
            vertex_count = 0
            polygon_vertices = None
        elif defect_type == "polygon":
            geometry_type = "polygon"
            angle_rad = math.nan
            angle_deg = math.nan
            if not polygon_vertices:
                notes.append("polygon_vertices_missing")
            padded, vertex_mask, inferred_count = _pad_vertices(polygon_vertices)
            if vertex_count <= 0:
                vertex_count = inferred_count
        else:
            geometry_type = "unknown"
            notes.append("unknown_defect_type")

        if defect_type != "polygon":
            padded, vertex_mask, _ = _pad_vertices(None)
        else:
            padded, vertex_mask, _ = _pad_vertices(polygon_vertices)

        mask = masks[idx] > 0
        mask_area = int(mask.sum())
        row = {
            "sample_index": idx,
            "sample_id": sample_id,
            "split": split,
            "defect_type": defect_type,
            "geometry_type": geometry_type,
            "include_in_rect_rot_poc": str(defect_type in {"rectangular_notch", "rotated_rect"}).lower(),
            "center_x": center_x,
            "center_y": center_y,
            "width": width,
            "length": length,
            "depth": depth,
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "angle_sin": math.sin(angle_rad) if not math.isnan(angle_rad) else math.nan,
            "angle_cos": math.cos(angle_rad) if not math.isnan(angle_rad) else math.nan,
            "vertex_count": vertex_count,
            "polygon_vertices": _json_or_empty(polygon_vertices),
            "polygon_vertices_padded": _json_or_empty(padded),
            "polygon_vertex_mask": _json_or_empty(vertex_mask),
            "mask_area": mask_area,
            "bbox_area": _bbox_area(width, length),
            "source_pack": source_pack,
            "notes": ";".join(notes),
        }
        rows.append(row)

    norm_fields = [
        "center_x",
        "center_y",
        "width",
        "length",
        "depth",
        "angle_rad",
        "angle_deg",
        "angle_sin",
        "angle_cos",
        "mask_area",
        "bbox_area",
    ]
    stats = _compute_train_stats(rows, norm_fields)
    for row in rows:
        for field in norm_fields:
            mean, std = stats[field]
            value = float(row[field])
            if math.isnan(value) or math.isnan(mean):
                row[f"{field}_norm"] = math.nan
            else:
                row[f"{field}_norm"] = (value - mean) / std

    diagnostics = {
        "n": len(rows),
        "split_counts": Counter(row["split"] for row in rows),
        "defect_counts": Counter(row["defect_type"] for row in rows),
        "geometry_counts": Counter(row["geometry_type"] for row in rows),
        "sample_id_unique": len(set(sample_ids)) == len(sample_ids),
        "parse_failures": parse_failures,
        "train_normalization_stats": stats,
        "polygon_vertex_counts": Counter(str(row["vertex_count"]) for row in rows if row["defect_type"] == "polygon"),
        "rect_rot_poc_counts": Counter(
            row["split"] for row in rows if row["include_in_rect_rot_poc"] == "true"
        ),
    }
    return rows, diagnostics


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key, value in row.items():
                if isinstance(value, float) and math.isnan(value):
                    clean[key] = ""
                else:
                    clean[key] = value
            writer.writerow(clean)


def write_summary(out_path: Path, npz_path: Path, labels_path: Path, diagnostics: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = (
        diagnostics["n"] == EXPECTED_N
        and dict(diagnostics["split_counts"]) == EXPECTED_SPLITS
        and dict(diagnostics["defect_counts"]) == EXPECTED_DEFECTS
        and diagnostics["sample_id_unique"]
        and not diagnostics["parse_failures"]
    )
    lines = [
        "COMSOL single-defect geometry label extraction summary",
        "",
        f"Input NPZ: {npz_path}",
        f"Output labels CSV: {labels_path}",
        f"N: {diagnostics['n']}",
        f"Expected N met: {diagnostics['n'] == EXPECTED_N}",
        f"Split counts: {dict(diagnostics['split_counts'])}",
        f"Defect counts: {dict(diagnostics['defect_counts'])}",
        f"Geometry type counts: {dict(diagnostics['geometry_counts'])}",
        f"Polygon vertex counts: {dict(diagnostics['polygon_vertex_counts'])}",
        f"Rect+rotated POC split counts: {dict(diagnostics['rect_rot_poc_counts'])}",
        f"Sample IDs unique: {diagnostics['sample_id_unique']}",
        f"geometry_params parse failures: {len(diagnostics['parse_failures'])}",
        "",
        "Label policy:",
        "- Labels come from geometry_params / metadata stored in the NPZ, not from predicted masks.",
        "- rectangular_notch uses angle_rad=0 and angle_deg=0.",
        "- rotated_rect preserves true angle_rad / angle_deg from geometry_params.",
        "- polygon preserves polygon_vertices and pads them to max_vertices=6; polygon is not coerced into rotated_rect.",
        "- include_in_rect_rot_poc is true only for rectangular_notch and rotated_rect; polygon is parsed for reporting only.",
        "- Continuous label normalization was fit on train split only and written as *_norm columns.",
        "",
        "Train normalization fields:",
        json.dumps(
            {
                key: {"mean": value[0], "std": value[1]}
                for key, value in diagnostics["train_normalization_stats"].items()
            },
            indent=2,
            sort_keys=True,
        ),
        "",
        f"Quality gate passed: {status}",
    ]
    if diagnostics["parse_failures"]:
        lines.extend(["", "Parse failures:", *diagnostics["parse_failures"][:20]])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--out", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows, diagnostics = extract_labels(args.npz)
    write_csv(rows, args.out)
    write_summary(args.summary, args.npz, args.out, diagnostics)


if __name__ == "__main__":
    main()

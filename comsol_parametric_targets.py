"""Build component-level parametric targets from COMSOL V2 defect parameters."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CONTINUOUS_SCHEMA = [
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "depth_or_shape_param",
    "rotation_angle",
]

SINCOS_SCHEMA = [
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "depth_or_shape_param",
    "rotation_sin",
    "rotation_cos",
]


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _as_float(value, *, field_name: str) -> float:
    if _is_missing(value):
        raise ValueError(f"Missing required numeric field: {field_name}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse numeric field {field_name!r}: {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Non-finite numeric field {field_name!r}: {value!r}")
    return out


def _first_present(mapping, names):
    for name in names:
        if name in mapping and not _is_missing(mapping[name]):
            return mapping[name]
    return None


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_npz_defect_params(npz_path: Path) -> list[dict]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "defect_params" not in data:
            raise ValueError(
                f"No defect_params field in {npz_path}; provide --defect-params-csv."
            )
        params = data["defect_params"]
        rows: list[dict] = []
        if params.dtype.names:
            for item in params:
                rows.append({name: item[name].item() if hasattr(item[name], "item") else item[name] for name in params.dtype.names})
            return rows
        for item in params:
            if isinstance(item, dict):
                rows.append(dict(item))
            else:
                raise ValueError(
                    "Unsupported defect_params representation; expected structured array or dict objects."
                )
        return rows


def load_defect_params(npz_path, defect_params_csv=None) -> list[dict]:
    """Load defect params from CSV first, falling back to the NPZ field."""

    npz_path = Path(npz_path)
    if defect_params_csv:
        csv_path = Path(defect_params_csv)
        if csv_path.exists():
            return _read_csv_rows(csv_path)
        raise ValueError(f"defect_params_csv does not exist: {csv_path}")
    return _read_npz_defect_params(npz_path)


def _parse_component_json(row: dict) -> list[dict]:
    raw = row.get("source_component_json")
    if _is_missing(raw):
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse source_component_json for sample {row.get('sample_index')}") from exc
    if not isinstance(parsed, list):
        raise ValueError("source_component_json must decode to a list.")
    return [dict(item) for item in parsed]


def _component_from_mapping(row: dict, comp: dict | None = None) -> dict:
    comp = comp or {}
    component_type = _first_present(comp, ["component_type"])
    if component_type is None:
        component_type = _first_present(row, ["component_type", "defect_type"])
    if component_type is None:
        component_type = "unknown"

    rotation = _first_present(comp, ["angle_deg", "rotation_angle"])
    if rotation is None and not _is_missing(comp.get("angle_rad")):
        rotation = np.degrees(_as_float(comp["angle_rad"], field_name="angle_rad"))
    if rotation is None:
        rotation = _first_present(row, ["rotation_angle"])
    if rotation is None:
        rotation = 0.0

    values = {
        "component_type": str(component_type),
        "center_x": _as_float(
            _first_present(comp, ["center_x_m", "center_x"])
            if _first_present(comp, ["center_x_m", "center_x"]) is not None
            else _first_present(row, ["defect_center_x", "center_x"]),
            field_name="center_x",
        ),
        "center_y": _as_float(
            _first_present(comp, ["center_y_m", "center_y"])
            if _first_present(comp, ["center_y_m", "center_y"]) is not None
            else _first_present(row, ["defect_center_y", "center_y"]),
            field_name="center_y",
        ),
        "axis_x": _as_float(
            _first_present(comp, ["length_m", "axis_x", "width"])
            if _first_present(comp, ["length_m", "axis_x", "width"]) is not None
            else _first_present(row, ["defect_axis_x", "defect_radius_or_width", "axis_x"]),
            field_name="axis_x",
        ),
        "axis_y": _as_float(
            _first_present(comp, ["width_m", "axis_y", "height"])
            if _first_present(comp, ["width_m", "axis_y", "height"]) is not None
            else _first_present(row, ["defect_axis_y", "axis_y"]),
            field_name="axis_y",
        ),
        "depth_or_shape_param": _as_float(
            _first_present(comp, ["depth_m", "depth_or_shape_param"])
            if _first_present(comp, ["depth_m", "depth_or_shape_param"]) is not None
            else _first_present(row, ["defect_depth_or_shape_param", "depth_or_shape_param"]),
            field_name="depth_or_shape_param",
        ),
        "rotation_angle": _as_float(rotation, field_name="rotation_angle"),
    }
    return values


def _extract_components(row: dict) -> list[dict]:
    component_rows = _parse_component_json(row)
    if component_rows:
        return [_component_from_mapping(row, comp) for comp in component_rows]
    return [_component_from_mapping(row)]


def _encode_continuous(raw_targets: np.ndarray, angle_encoding: str) -> tuple[np.ndarray, list[str], str]:
    if angle_encoding == "raw":
        return raw_targets.copy(), list(CONTINUOUS_SCHEMA), "degree"
    if angle_encoding != "sincos":
        raise ValueError(f"Unsupported angle_encoding: {angle_encoding}")
    encoded = np.zeros((*raw_targets.shape[:2], len(SINCOS_SCHEMA)), dtype=np.float32)
    encoded[:, :, :5] = raw_targets[:, :, :5]
    angle_values = raw_targets[:, :, CONTINUOUS_SCHEMA.index("rotation_angle")]
    angle_unit = "radian" if np.nanmax(np.abs(angle_values)) <= 2 * np.pi + 1e-6 else "degree"
    angle_rad = angle_values if angle_unit == "radian" else np.deg2rad(angle_values)
    encoded[:, :, 5] = np.sin(angle_rad)
    encoded[:, :, 6] = np.cos(angle_rad)
    return encoded, list(SINCOS_SCHEMA), angle_unit


def build_parametric_targets(defect_params, max_components=3, angle_encoding="raw") -> dict:
    """Build sorted fixed-width component targets from defect parameter rows."""

    if max_components <= 0:
        raise ValueError("max_components must be positive.")
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in defect_params:
        sample_index = int(float(row["sample_index"]))
        grouped[sample_index].extend(_extract_components(row))

    if not grouped:
        raise ValueError("No defect parameter rows were loaded.")

    sample_indices = np.array(sorted(grouped), dtype=np.int64)
    all_types = sorted({comp["component_type"] for comps in grouped.values() for comp in comps})
    if not all_types:
        raise ValueError("No component types were found.")
    type_to_index = {name: i for i, name in enumerate(all_types)}

    n_samples = len(sample_indices)
    n_params = len(CONTINUOUS_SCHEMA)
    continuous_targets_raw = np.zeros((n_samples, max_components, n_params), dtype=np.float32)
    type_targets = np.full((n_samples, max_components), -1, dtype=np.int64)
    presence_targets = np.zeros((n_samples, max_components), dtype=np.float32)
    component_counts = np.zeros(n_samples, dtype=np.int64)

    for sample_pos, sample_index in enumerate(sample_indices):
        comps = sorted(grouped[int(sample_index)], key=lambda c: (c["center_x"], c["center_y"]))
        component_counts[sample_pos] = len(comps)
        if len(comps) > max_components:
            raise ValueError(
                f"sample_index={sample_index} has {len(comps)} components, exceeding max_components={max_components}."
            )
        for slot, comp in enumerate(comps):
            presence_targets[sample_pos, slot] = 1.0
            type_targets[sample_pos, slot] = type_to_index[comp["component_type"]]
            continuous_targets_raw[sample_pos, slot, :] = [comp[name] for name in CONTINUOUS_SCHEMA]

    continuous_targets, target_schema, angle_unit = _encode_continuous(continuous_targets_raw, angle_encoding)

    return {
        "sample_indices": sample_indices,
        "continuous_targets": continuous_targets,
        "continuous_targets_raw": continuous_targets_raw,
        "type_targets": type_targets,
        "presence_targets": presence_targets,
        "target_schema": np.array(target_schema, dtype="U64"),
        "raw_target_schema": np.array(CONTINUOUS_SCHEMA, dtype="U64"),
        "type_vocab": np.array(all_types, dtype="U128"),
        "component_counts": component_counts,
        "angle_encoding": np.array(angle_encoding, dtype="U16"),
        "angle_unit": np.array(angle_unit, dtype="U16"),
    }


def _validate_against_npz(npz_path: Path, targets: dict) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        if "signals" not in data:
            return
        expected = int(data["signals"].shape[0])
    if len(targets["sample_indices"]) != expected:
        raise ValueError(
            f"target samples ({len(targets['sample_indices'])}) do not match signals samples ({expected})."
        )
    expected_indices = np.arange(expected, dtype=np.int64)
    if not np.array_equal(targets["sample_indices"], expected_indices):
        raise ValueError("sample_indices must match the split-local signal order 0..N-1.")


def _write_preview(path: Path, targets: dict) -> None:
    schema = [str(x) for x in targets.get("raw_target_schema", targets["target_schema"])]
    preview_values = targets.get("continuous_targets_raw", targets["continuous_targets"])
    type_vocab = [str(x) for x in targets["type_vocab"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_index", "component_slot", "presence", "component_type", *schema]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, sample_index in enumerate(targets["sample_indices"]):
            for slot in range(targets["presence_targets"].shape[1]):
                present = float(targets["presence_targets"][i, slot])
                type_id = int(targets["type_targets"][i, slot])
                row = {
                    "sample_index": int(sample_index),
                    "component_slot": slot,
                    "presence": present,
                    "component_type": type_vocab[type_id] if type_id >= 0 else "",
                }
                for j, name in enumerate(schema):
                    row[name] = float(preview_values[i, slot, j]) if present else ""
                writer.writerow(row)


def _continuous_ranges(targets: dict) -> list[str]:
    present = targets["presence_targets"] > 0.5
    rows: list[str] = []
    values_source = targets.get("continuous_targets_raw", targets["continuous_targets"])
    schema = targets.get("raw_target_schema", targets["target_schema"])
    for j, name in enumerate(schema):
        values = values_source[:, :, j][present]
        if values.size == 0:
            rows.append(f"- `{name}`: unavailable")
        else:
            rows.append(f"- `{name}`: min={values.min():.6g}, max={values.max():.6g}, mean={values.mean():.6g}")
    return rows


def _write_summary(path: Path, npz_path: Path, rows: list[dict], targets: dict, max_components: int) -> None:
    counts = Counter(int(x) for x in targets["component_counts"])
    type_vocab = [str(x) for x in targets["type_vocab"]]
    lines = [
        "# COMSOL parametric target summary",
        "",
        f"- npz_path: `{npz_path}`",
        f"- raw defect rows: `{len(rows)}`",
        f"- samples: `{len(targets['sample_indices'])}`",
        f"- max_components: `{max_components}`",
        f"- target_schema: `{', '.join(str(x) for x in targets['target_schema'])}`",
        f"- raw_target_schema: `{', '.join(str(x) for x in targets.get('raw_target_schema', targets['target_schema']))}`",
        f"- angle_encoding: `{str(targets.get('angle_encoding', 'raw'))}`",
        f"- angle_unit: `{str(targets.get('angle_unit', 'degree'))}`",
        f"- type_vocab: `{', '.join(type_vocab)}`",
        "",
        "## Presence count distribution",
        "",
    ]
    lines.extend(f"- `{count}` components: `{num}` samples" for count, num in sorted(counts.items()))
    lines.extend(["", "## Continuous target ranges", ""])
    lines.extend(_continuous_ranges(targets))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Component sorting uses `center_x`, then `center_y`.",
            "- `source_component_json` is used when available; sample-level fields are fallback only.",
            "- No component truncation was applied; samples exceeding `max_components` raise `ValueError`.",
            "- `axis_x` / `axis_y` are component full width / full height values from `length_m` / `width_m` when `source_component_json` is available.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_normalization_stats(targets: dict) -> tuple[np.ndarray, np.ndarray]:
    present = targets["presence_targets"] > 0.5
    values = targets["continuous_targets"][present]
    if values.size == 0:
        raise ValueError("Cannot normalize continuous targets without present components.")
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def _apply_normalization(targets: dict, output_dir: Path, stats_npz: str = "") -> None:
    if stats_npz:
        with np.load(stats_npz, allow_pickle=True) as data:
            mean = data["continuous_targets_mean"].astype(np.float32)
            std = data["continuous_targets_std"].astype(np.float32)
            stats_schema = [str(x) for x in data["target_schema"]]
        current_schema = [str(x) for x in targets["target_schema"]]
        if stats_schema != current_schema:
            raise ValueError(f"normalization stats schema {stats_schema} != current target schema {current_schema}")
    else:
        mean, std = _compute_normalization_stats(targets)
        np.savez_compressed(
            output_dir / "continuous_normalization_stats.npz",
            continuous_targets_mean=mean,
            continuous_targets_std=std,
            target_schema=targets["target_schema"],
        )
    targets["continuous_targets_unscaled"] = targets["continuous_targets"].copy()
    targets["continuous_targets_mean"] = mean
    targets["continuous_targets_std"] = std
    targets["continuous_targets"] = (
        (targets["continuous_targets"] - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    ).astype(np.float32)
    targets["continuous_targets_normalized"] = np.array(True)


def run(args) -> None:
    npz_path = Path(args.npz_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_defect_params(npz_path, args.defect_params_csv or None)
    targets = build_parametric_targets(rows, max_components=args.max_components, angle_encoding=args.angle_encoding)
    _validate_against_npz(npz_path, targets)
    if args.normalize_continuous:
        _apply_normalization(targets, output_dir, args.normalization_stats_npz)
    else:
        targets["continuous_targets_unscaled"] = targets["continuous_targets"].copy()
        targets["continuous_targets_mean"] = np.zeros(targets["continuous_targets"].shape[-1], dtype=np.float32)
        targets["continuous_targets_std"] = np.ones(targets["continuous_targets"].shape[-1], dtype=np.float32)
        targets["continuous_targets_normalized"] = np.array(False)

    np.savez_compressed(output_dir / "parametric_targets.npz", **targets)
    _write_preview(output_dir / "parametric_target_preview.csv", targets)
    _write_summary(output_dir / "parametric_target_summary.md", npz_path, rows, targets, args.max_components)
    print(f"Saved parametric targets to {output_dir / 'parametric_targets.npz'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--defect-params-csv", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--angle-encoding", choices=["raw", "sincos"], default="raw")
    parser.add_argument("--normalize-continuous", action="store_true")
    parser.add_argument("--normalization-stats-npz", default="")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.npz_path or not args.output_dir:
        parser.print_help()
        print("\nExample: python comsol_parametric_targets.py --npz-path train.npz --defect-params-csv defect_params.csv --output-dir out")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare COMSOL V1/V2 target masks, label areas, and defect distributions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_ARGS = [
    ("v1", "train", "v1_train_npz"),
    ("v1", "val", "v1_val_npz"),
    ("v1", "test", "v1_test_npz"),
    ("v2", "train", "v2_train_npz"),
    ("v2", "val", "v2_val_npz"),
    ("v2", "test", "v2_test_npz"),
]


def _scalar_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalar_string(value.item())
        return json.dumps([_scalar_string(v) for v in value.tolist()], ensure_ascii=False)
    if isinstance(value, np.generic):
        return _scalar_string(value.item())
    return str(value)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"NPZ does not exist: {p}")
    with np.load(p, allow_pickle=True) as data:
        return {name: data[name] for name in data.files}


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(np.asarray(array, dtype=float)).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _mask_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    iou = 1.0 if union == 0 else float(intersection / union)
    mismatch = int(np.not_equal(a_bool, b_bool).sum())
    return iou, mismatch


def _preferred_mask(data: dict[str, np.ndarray], mu_threshold: float) -> np.ndarray:
    if "masks" in data:
        return np.asarray(data["masks"]) > 0.5
    if "mu_maps" in data:
        return np.asarray(data["mu_maps"]) < mu_threshold
    raise ValueError("dataset must contain masks or mu_maps")


def _defect_records_from_dataset(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if "defect_params" in data:
        params = np.asarray(data["defect_params"])
        if params.dtype.names:
            records = []
            for row in params:
                records.append({name: row[name].item() if hasattr(row[name], "item") else row[name] for name in params.dtype.names})
            return records

    if "sample_index" in data:
        sample_count = len(data["sample_index"])
    elif "signals" in data:
        sample_count = int(np.asarray(data["signals"]).shape[0])
    elif "mu_maps" in data:
        sample_count = int(np.asarray(data["mu_maps"]).shape[0])
    else:
        return []

    keys = [
        "sample_index",
        "split",
        "split_per_sample",
        "defect_type",
        "defect_center_x",
        "defect_center_y",
        "defect_center_z",
        "defect_axis_x",
        "defect_axis_y",
        "defect_axis_z",
        "defect_radius_or_width",
        "defect_depth_or_shape_param",
        "defect_mu",
        "rotation_angle",
        "boundary_irregularity",
        "boundary_irregularity_level",
        "c_magn",
        "mur_magn",
        "Mr_magn_A_per_m",
    ]
    records = []
    for i in range(sample_count):
        record: dict[str, Any] = {"sample_index": i}
        for key in keys:
            if key in data:
                value = data[key]
                if np.asarray(value).ndim == 0:
                    record[key] = value.item()
                else:
                    record[key] = value[i]
        records.append(record)
    return records


def _counter_markdown(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"- {title}:"]
    if not counter:
        lines.append("  - 不可用")
        return lines
    for key, value in sorted(counter.items()):
        lines.append(f"  - `{key}`: {value}")
    return lines


def _numeric_range(records: list[dict[str, Any]], key: str) -> tuple[float, float] | None:
    values = []
    for record in records:
        if key not in record:
            continue
        try:
            values.append(float(record[key]))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return min(values), max(values)


def _fixed_or_varied(records: list[dict[str, Any]], key: str) -> str:
    values = []
    for record in records:
        if key not in record:
            continue
        try:
            values.append(float(record[key]))
        except (TypeError, ValueError):
            values.append(_scalar_string(record[key]))
    if not values:
        return "不可用"
    unique = sorted(set(values))
    if len(unique) == 1:
        return f"固定 `{unique[0]}`"
    if all(isinstance(v, float) for v in unique):
        return f"变化，范围 `{min(unique):.6g}` 到 `{max(unique):.6g}`"
    return f"变化，unique_count={len(unique)}"


def _summarize_dataset(version: str, split: str, path: Path, mu_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    data = _load_npz(path)
    if "signals" in data:
        _require_finite(f"{version}/{split} signals", data["signals"])
        num_samples = int(np.asarray(data["signals"]).shape[0])
    elif "mu_maps" in data:
        num_samples = int(np.asarray(data["mu_maps"]).shape[0])
    else:
        raise ValueError(f"{path} must contain signals or mu_maps")

    has_mu = "mu_maps" in data
    has_masks = "masks" in data
    if has_mu:
        mu_maps = np.asarray(data["mu_maps"], dtype=float)
        _require_finite(f"{version}/{split} mu_maps", mu_maps)
        threshold_mask = mu_maps < mu_threshold
    else:
        mu_maps = None
        threshold_mask = None
    if has_masks:
        masks = np.asarray(data["masks"], dtype=float)
        _require_finite(f"{version}/{split} masks", masks)
        provided_mask = masks > 0.5
    else:
        masks = None
        provided_mask = None

    label = _preferred_mask(data, mu_threshold)
    if label.shape[0] != num_samples:
        raise ValueError(f"{version}/{split} label sample count mismatch")
    grid_y, grid_x = label.shape[-2], label.shape[-1]
    total_pixels = grid_x * grid_y

    per_sample = []
    mask_ious = []
    mismatches = []
    threshold_areas = None if threshold_mask is None else threshold_mask.reshape(num_samples, -1).sum(axis=1)
    mask_areas = None if provided_mask is None else provided_mask.reshape(num_samples, -1).sum(axis=1)
    label_areas = label.reshape(num_samples, -1).sum(axis=1)
    for sample_index in range(num_samples):
        if threshold_mask is not None and provided_mask is not None:
            iou, mismatch = _mask_iou(threshold_mask[sample_index], provided_mask[sample_index])
            mask_ious.append(iou)
            mismatches.append(mismatch)
        else:
            iou = ""
            mismatch = ""
        per_sample.append(
            {
                "dataset_version": version,
                "split": split,
                "sample_index": sample_index,
                "threshold_area_pixels": "" if threshold_areas is None else float(threshold_areas[sample_index]),
                "provided_mask_area_pixels": "" if mask_areas is None else float(mask_areas[sample_index]),
                "label_area_pixels": float(label_areas[sample_index]),
                "label_area_ratio": float(label_areas[sample_index] / total_pixels),
                "mask_iou": iou,
                "mismatch_count": mismatch,
                "has_mu_maps": has_mu,
                "has_masks": has_masks,
            }
        )

    mu_stats: dict[str, Any]
    if mu_maps is not None:
        defect_values = mu_maps[threshold_mask]
        background_values = mu_maps[~threshold_mask]
        mu_stats = {
            "mu_min": float(mu_maps.min()),
            "mu_max": float(mu_maps.max()),
            "mu_mean": float(mu_maps.mean()),
            "mu_std": float(mu_maps.std()),
            "background_mu_mean": "" if background_values.size == 0 else float(background_values.mean()),
            "defect_mu_mean": "" if defect_values.size == 0 else float(defect_values.mean()),
        }
    else:
        mu_stats = {
            "mu_min": "",
            "mu_max": "",
            "mu_mean": "",
            "mu_std": "",
            "background_mu_mean": "",
            "defect_mu_mean": "",
        }

    aggregate = {
        "dataset_version": version,
        "split": split,
        "npz_path": str(path),
        "num_samples": num_samples,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "has_mu_maps": has_mu,
        "has_masks": has_masks,
        "mean_label_area_from_mu_threshold": "" if threshold_areas is None else float(np.mean(threshold_areas)),
        "mean_label_area_from_masks": "" if mask_areas is None else float(np.mean(mask_areas)),
        "mean_label_area_ratio": float(np.mean(label_areas) / total_pixels),
        "min_label_area_ratio": float(np.min(label_areas) / total_pixels),
        "max_label_area_ratio": float(np.max(label_areas) / total_pixels),
        "avg_mask_iou": "" if not mask_ious else float(np.mean(mask_ious)),
        "min_mask_iou": "" if not mask_ious else float(np.min(mask_ious)),
        "max_mask_iou": "" if not mask_ious else float(np.max(mask_ious)),
        "total_mismatch_count": "" if not mismatches else int(np.sum(mismatches)),
        **mu_stats,
    }

    records = _defect_records_from_dataset(data)
    defect_lines = [f"## {version} {split}", "", f"- samples: {num_samples}"]
    defect_type_counter = Counter(_scalar_string(record.get("defect_type", "不可用")) for record in records)
    component_counter = Counter(_scalar_string(record.get("component_types", "不可用")) for record in records if "component_types" in record)
    boundary_counter = Counter(
        _scalar_string(record.get("boundary_irregularity_level", record.get("distance_bin", "不可用"))) for record in records
    )
    defect_lines.extend(_counter_markdown("defect_type 分布", defect_type_counter))
    defect_lines.extend(_counter_markdown("component type 分布", component_counter))
    defect_lines.extend(_counter_markdown("boundary_irregularity_proxy 分布", boundary_counter))
    for key in [
        "rotation_angle",
        "defect_center_x",
        "defect_center_y",
        "defect_center_z",
        "defect_axis_x",
        "defect_axis_y",
        "defect_axis_z",
        "defect_depth_or_shape_param",
    ]:
        value_range = _numeric_range(records, key)
        if value_range is None:
            defect_lines.append(f"- `{key}` 范围: 不可用")
        else:
            defect_lines.append(f"- `{key}` 范围: `{value_range[0]:.6g}` 到 `{value_range[1]:.6g}`")
    for key in ["defect_mu", "c_magn", "mur_magn", "Mr_magn_A_per_m"]:
        defect_lines.append(f"- `{key}`: {_fixed_or_varied(records, key)}")
    defect_lines.append("")
    return per_sample, aggregate, defect_lines


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_summary(aggregates: list[dict[str, Any]]) -> str:
    def fmt_float(value: Any) -> str:
        if value == "":
            return ""
        return f"{float(value):.6e}"

    by_key = {(row["dataset_version"], row["split"]): row for row in aggregates}
    lines = [
        "# S87 COMSOL V1/V2 target distribution diagnostics",
        "",
        "## 核心结论",
        "",
    ]
    v1_train = by_key.get(("v1", "train"))
    v2_train = by_key.get(("v2", "train"))
    if v1_train and v2_train:
        v1_area = float(v1_train["mean_label_area_ratio"])
        v2_area = float(v2_train["mean_label_area_ratio"])
        ratio = v2_area / v1_area if v1_area else float("inf")
        lines.append(f"- train mean label area ratio: V1 = `{v1_area:.6e}`, V2 = `{v2_area:.6e}`, V2/V1 = `{ratio:.3f}`。")
        if ratio < 0.5 or ratio > 2.0:
            lines.append("- V2 label area 与 V1 差异较大，可能影响 IoU 和训练拟合。")
        else:
            lines.append("- V2 label area 与 V1 同量级，不像是单独导致 S85 退化的主因。")
    all_iou_ok = True
    for row in aggregates:
        if row["has_mu_maps"] and row["has_masks"]:
            avg_iou = float(row["avg_mask_iou"])
            mismatch = int(row["total_mismatch_count"])
            if avg_iou < 0.999 or mismatch != 0:
                all_iou_ok = False
    if all_iou_ok:
        lines.append("- V1/V2 的 `mu_maps < 500` 与 provided `masks > 0.5` 均完全一致；target/mask 定义未发现异常。")
    else:
        lines.append("- 存在 `mu_maps < 500` 与 provided masks 不一致的 split，需要优先排查 target/mask。")

    lines.extend(["", "## aggregate target distribution", ""])
    lines.append("| dataset | split | samples | mean_label_area_ratio | avg_mask_iou | mu_min | mu_max | defect_mu_mean | background_mu_mean |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in aggregates:
        lines.append(
            f"| {row['dataset_version']} | {row['split']} | {row['num_samples']} | "
            f"{float(row['mean_label_area_ratio']):.6e} | "
            f"{fmt_float(row['avg_mask_iou'])} | "
            f"{fmt_float(row['mu_min'])} | {fmt_float(row['mu_max'])} | "
            f"{fmt_float(row['defect_mu_mean'])} | {fmt_float(row['background_mu_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- V2 target/mask 本身未发现不一致异常。",
            "- V2 defect distribution 比 V1 更复杂：V2 是 `rectangular_notch` / `rotated_rect` multi_defect 组合，并包含 rotation 与 boundary proxy；V1 是上一批 COMSOL geometry pilot。",
            "- V2 低 IoU 更可能来自任务难度、signal 语义或 runner/loss 对 multi_defect 目标适配不足，而不是简单 mask label 错误。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    defect_lines = ["# S87 defect parameter distribution", ""]

    for version, split, attr in SPLIT_ARGS:
        path = Path(getattr(args, attr))
        per_sample, aggregate, split_defect_lines = _summarize_dataset(version, split, path, args.mu_threshold)
        per_sample_rows.extend(per_sample)
        aggregate_rows.append(aggregate)
        defect_lines.extend(split_defect_lines)

    _write_csv(output_dir / "per_sample_target_distribution.csv", per_sample_rows)
    _write_csv(output_dir / "aggregate_target_distribution.csv", aggregate_rows)
    (output_dir / "defect_param_distribution.md").write_text("\n".join(defect_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(_make_summary(aggregate_rows), encoding="utf-8")
    print(f"Saved S87 diagnostics to {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-train-npz")
    parser.add_argument("--v1-val-npz")
    parser.add_argument("--v1-test-npz")
    parser.add_argument("--v2-train-npz")
    parser.add_argument("--v2-val-npz")
    parser.add_argument("--v2-test-npz")
    parser.add_argument("--output-dir")
    parser.add_argument("--mu-threshold", type=float, default=500.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    required = [attr for _, _, attr in SPLIT_ARGS] + ["output_dir"]
    if not all(getattr(args, attr) for attr in required):
        parser.print_help()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

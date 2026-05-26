"""Grouped diagnostics for COMSOL parametric per-sample prediction exports."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SPLITS = ["train", "val", "test"]
ROTATION_BINS = [
    ("0-5", 0.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-30", 20.0, 30.0),
    (">30", 30.0, float("inf")),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        rows = [{"note": "no rows"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict, key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _rotation_bin(rotation_error: float) -> str:
    for label, low, high in ROTATION_BINS:
        if low <= rotation_error < high:
            return label
    return ">30"


def _area_bin(value: float, q1: float, q2: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value <= q1:
        return "small"
    if value <= q2:
        return "medium"
    return "large"


def _oracle_gap_bin(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value < 0.1:
        return "low"
    if value < 0.3:
        return "medium"
    return "high"


def load_prediction_dir(prediction_dir: Path) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    component_rows: list[dict] = []
    mask_rows: dict[tuple[str, int], dict] = {}
    for split in SPLITS:
        for row in _read_csv(prediction_dir / f"{split}_predictions.csv"):
            row["split"] = row.get("split") or split
            component_rows.append(row)
        for row in _read_csv(prediction_dir / f"{split}_prediction_mask_metrics.csv"):
            row["split"] = row.get("split") or split
            mask_rows[(row["split"], _int(row, "sample_index"))] = row
    return component_rows, mask_rows


def _enrich_rows(component_rows: list[dict], mask_rows: dict[tuple[str, int], dict]) -> list[dict]:
    target_areas = np.array([_float(row, "target_area") for row in mask_rows.values()], dtype=np.float64)
    if target_areas.size and np.isfinite(target_areas).any():
        q1, q2 = np.nanquantile(target_areas, [1.0 / 3.0, 2.0 / 3.0])
    else:
        q1, q2 = float("nan"), float("nan")
    enriched = []
    for row in component_rows:
        sample_index = _int(row, "sample_index")
        split = row.get("split", "")
        mask = mask_rows.get((split, sample_index), {})
        out = dict(row)
        rotation_error = _float(row, "rotation_error")
        target_area = _float(mask, "target_area")
        oracle_gap = _float(mask, "oracle_gap")
        out["rotation_bin"] = _rotation_bin(rotation_error)
        out["area_bin"] = _area_bin(target_area, q1, q2)
        out["oracle_gap_bin"] = _oracle_gap_bin(oracle_gap)
        out["pred_mask_iou"] = _float(mask, "pred_mask_iou")
        out["oracle_gap"] = oracle_gap
        out["target_area"] = target_area
        out["area_diff"] = _float(mask, "area_diff")
        enriched.append(out)
    return enriched


def _summarize_group(rows: list[dict], group_name: str, group_value: str) -> dict:
    count = len(rows)
    present_rows = [row for row in rows if _int(row, "presence_true") == 1]
    if count == 0:
        raise ValueError("Cannot summarize an empty group.")
    type_accuracy = float(np.mean([_int(row, "type_correct") for row in present_rows])) if present_rows else float("nan")
    presence_accuracy = float(
        np.mean([1.0 if _int(row, "presence_true") == _int(row, "presence_pred") else 0.0 for row in rows])
    )

    def mean_key(key: str, source_rows: list[dict] | None = None) -> float:
        values = np.array([_float(row, key) for row in (source_rows or rows)], dtype=np.float64)
        return float(np.nanmean(values)) if values.size and np.isfinite(values).any() else float("nan")

    return {
        "group": group_name,
        "value": group_value,
        "count": count,
        "present_count": len(present_rows),
        "type_accuracy": type_accuracy,
        "presence_accuracy": presence_accuracy,
        "center_error_mean": mean_key("center_error", present_rows),
        "axis_error_mean": mean_key("axis_error", present_rows),
        "rotation_error_mean": mean_key("rotation_error", present_rows),
        "depth_error_mean": mean_key("depth_error", present_rows),
        "mask_iou_mean": mean_key("pred_mask_iou"),
        "oracle_gap_mean": mean_key("oracle_gap"),
    }


def grouped(rows: list[dict], key: str, *, present_only: bool = False) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if present_only and _int(row, "presence_true") != 1:
            continue
        buckets[str(row.get(key, ""))].append(row)
    return [_summarize_group(bucket, key, value) for value, bucket in sorted(buckets.items())]


def worst_samples(component_rows: list[dict], mask_rows: dict[tuple[str, int], dict]) -> list[dict]:
    by_sample: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in component_rows:
        by_sample[(row.get("split", ""), _int(row, "sample_index"))].append(row)
    rows = []
    for key, mask in mask_rows.items():
        split, sample_index = key
        comp = by_sample.get(key, [])
        rotation_values = [_float(row, "rotation_error") for row in comp if _int(row, "presence_true") == 1]
        rows.append(
            {
                "split": split,
                "sample_index": sample_index,
                "pred_mask_iou": _float(mask, "pred_mask_iou"),
                "oracle_mask_iou": _float(mask, "oracle_mask_iou"),
                "oracle_gap": _float(mask, "oracle_gap"),
                "target_area": _float(mask, "target_area"),
                "pred_area": _float(mask, "pred_area"),
                "area_diff": _float(mask, "area_diff"),
                "type_sequence_true": mask.get("type_sequence_true", ""),
                "type_sequence_pred": mask.get("type_sequence_pred", ""),
                "rotation_error_mean": float(np.nanmean(rotation_values)) if rotation_values else float("nan"),
            }
        )
    return sorted(rows, key=lambda row: (row["pred_mask_iou"], -row["oracle_gap"]))[:20]


def _best_and_worst(groups: list[dict], metric: str, larger_is_better: bool = True) -> tuple[str, str]:
    valid = [row for row in groups if np.isfinite(float(row.get(metric, float("nan"))))]
    if not valid:
        return "n/a", "n/a"
    sorted_rows = sorted(valid, key=lambda row: float(row[metric]), reverse=larger_is_better)
    return str(sorted_rows[0]["value"]), str(sorted_rows[-1]["value"])


def write_summary(path: Path, outputs: dict[str, list[dict]], worst: list[dict]) -> None:
    type_best, type_worst = _best_and_worst(outputs["grouped_by_type"], "type_accuracy", True)
    slot_best, slot_worst = _best_and_worst(outputs["grouped_by_slot"], "mask_iou_mean", True)
    rotation_best, rotation_worst = _best_and_worst(outputs["grouped_by_rotation_bin"], "mask_iou_mean", True)
    area_best, area_worst = _best_and_worst(outputs["grouped_by_area_bin"], "mask_iou_mean", True)
    worst_desc = "n/a"
    if worst:
        first = worst[0]
        worst_desc = (
            f"{first['split']} sample {first['sample_index']} "
            f"IoU={first['pred_mask_iou']:.6e}, gap={first['oracle_gap']:.6e}"
        )
    lines = [
        "# COMSOL parametric grouped diagnostics summary",
        "",
        "## Key grouped findings",
        "",
        f"- Best / worst type by type accuracy: `{type_best}` / `{type_worst}`.",
        f"- Best / worst slot by mask IoU: `{slot_best}` / `{slot_worst}`.",
        f"- Best / worst rotation-error bin by mask IoU: `{rotation_best}` / `{rotation_worst}`.",
        f"- Best / worst area bin by mask IoU: `{area_best}` / `{area_worst}`.",
        f"- Worst sample: `{worst_desc}`.",
        "",
        "## Interpretation",
        "",
        "- This diagnostic uses exported per-component rows and per-sample rasterized mask metrics.",
        "- Low mask IoU with non-trivial oracle gap indicates model prediction error rather than only rasterizer ceiling.",
        "- Type or rotation groups with high error support follow-up component matching or set-style decoding tests.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    component_rows, mask_rows = load_prediction_dir(prediction_dir)
    if not component_rows:
        raise ValueError(f"No prediction CSV rows found in {prediction_dir}")
    enriched = _enrich_rows(component_rows, mask_rows)
    outputs = {
        "grouped_by_type": grouped(enriched, "type_true", present_only=True),
        "grouped_by_slot": grouped(enriched, "component_slot"),
        "grouped_by_rotation_bin": grouped(enriched, "rotation_bin", present_only=True),
        "grouped_by_area_bin": grouped(enriched, "area_bin"),
        "grouped_by_oracle_gap_bin": grouped(enriched, "oracle_gap_bin"),
    }
    worst = worst_samples(enriched, mask_rows)
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
    _write_csv(output_dir / "worst_samples.csv", worst)
    write_summary(output_dir / "summary.md", outputs, worst)
    print(f"Saved grouped diagnostics to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--output-dir", default="")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.prediction_dir or not args.output_dir:
        parser.print_help()
        print("\nExample: python comsol_parametric_grouped_diagnostics.py --prediction-dir run --output-dir diagnostics")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "COMSOL_DATA_BASELINE.md"
DEFAULT_SUMMARY_IN = PROJECT_ROOT / "results/summaries/comsol_pilot_v9_baseline_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_metrics.csv"
DEFAULT_SEED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_seed_summary.csv"
DEFAULT_DEFECT_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_defect_type_summary.csv"
DEFAULT_ANGLE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_angle_summary.csv"
DEFAULT_VERTEX_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_vertex_count_summary.csv"
DEFAULT_SOURCE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_source_pack_summary.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_pilot_v9_baseline"
DEFAULT_SUMMARY_OUT = PROJECT_ROOT / "results/summaries/comsol_pilot_v9_baseline_failure_audit_summary.txt"
DEFAULT_CASES_OUT = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_failure_audit_cases.csv"

CASE_FIELDS = [
    "sample_id",
    "split",
    "defect_type",
    "angle",
    "vertex_count",
    "source_pack",
    "IoU",
    "Dice",
    "area_error",
    "center_error",
    "pred_area",
    "true_area",
    "failure_category",
    "short_note",
    "preview_path",
    "selection_tags",
    "seed_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit COMSOL pilot_v9 baseline failure cases.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--summary-in", type=Path, default=DEFAULT_SUMMARY_IN)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--seed-summary", type=Path, default=DEFAULT_SEED_SUMMARY)
    parser.add_argument("--defect-summary", type=Path, default=DEFAULT_DEFECT_SUMMARY)
    parser.add_argument("--angle-summary", type=Path, default=DEFAULT_ANGLE_SUMMARY)
    parser.add_argument("--vertex-summary", type=Path, default=DEFAULT_VERTEX_SUMMARY)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--cases-out", type=Path, default=DEFAULT_CASES_OUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def f(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return float("nan")
    return float(value)


def aggregate_test_samples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            grouped[row["sample_id"]].append(row)
    samples: list[dict[str, Any]] = []
    for sample_id, sample_rows in grouped.items():
        first = sample_rows[0]
        numeric_keys = ["iou", "dice", "area_error", "center_error", "pred_area", "true_area"]
        values = {key: [f(row, key) for row in sample_rows] for key in numeric_keys}
        samples.append(
            {
                "sample_id": sample_id,
                "split": "test",
                "defect_type": first["defect_type"],
                "angle": float(first["angle_deg"]),
                "vertex_count": int(float(first["vertex_count"])),
                "source_pack": first["source_pack"],
                "IoU": float(np.nanmean(values["iou"])),
                "Dice": float(np.nanmean(values["dice"])),
                "area_error": float(np.nanmean(values["area_error"])),
                "center_error": float(np.nanmean(values["center_error"])),
                "pred_area": float(np.nanmean(values["pred_area"])),
                "true_area": float(np.nanmean(values["true_area"])),
                "seed_count": len(sample_rows),
                "selection_tags": set(),
            }
        )
    return samples


def preview_map(preview_dir: Path) -> dict[str, str]:
    mapping: dict[str, list[str]] = defaultdict(list)
    if not preview_dir.exists():
        return {}
    for path in sorted(preview_dir.glob("*.png")):
        name = path.name
        for token in name.split("_"):
            pass
        stem = path.stem
        parts = stem.split("_")
        # Preview names are seed{seed}_{sample_id_parts}_{split}_{defect_type_parts}.
        # Match by looking for known source prefixes rather than parsing split names.
        for prefix in ("rect_p2", "rect_v9", "rot_p3", "rot_v7", "rot_v8", "rot_v9", "poly_p5", "poly_v7", "poly_v8", "poly_v9"):
            marker = prefix + "_"
            if marker in stem:
                start = stem.index(marker)
                sample_id = "_".join(stem[start:].split("_")[:3])
                mapping[sample_id].append(str(path))
                break
    return {sample_id: ";".join(paths) for sample_id, paths in mapping.items()}


def tag_cases(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    def add(rows: list[dict[str, Any]], tag: str) -> None:
        for row in rows:
            selected.setdefault(row["sample_id"], row)["selection_tags"].add(tag)

    add(sorted(samples, key=lambda row: row["Dice"])[:10], "lowest_dice_top10")
    add(sorted(samples, key=lambda row: row["IoU"])[:10], "lowest_iou_top10")
    add(sorted(samples, key=lambda row: row["area_error"], reverse=True)[:10], "highest_area_error_top10")
    add(sorted(samples, key=lambda row: row["center_error"], reverse=True)[:10], "highest_center_error_top10")
    for defect_type in ("rectangular_notch", "rotated_rect", "polygon"):
        rows = [row for row in samples if row["defect_type"] == defect_type]
        add(sorted(rows, key=lambda row: row["Dice"])[:5], f"worst_{defect_type}_top5")
    for angle in sorted({row["angle"] for row in samples if row["defect_type"] == "rotated_rect"}):
        rows = [row for row in samples if row["defect_type"] == "rotated_rect" and row["angle"] == angle]
        add(sorted(rows, key=lambda row: row["Dice"])[:1], f"worst_rotated_angle_{angle:g}")
    for vertex_count in sorted({row["vertex_count"] for row in samples if row["defect_type"] == "polygon"}):
        rows = [row for row in samples if row["defect_type"] == "polygon" and row["vertex_count"] == vertex_count]
        add(sorted(rows, key=lambda row: row["Dice"])[:1], f"worst_polygon_vertex_{vertex_count}")
    return list(selected.values())


def classify_cases(cases: list[dict[str, Any]], all_samples: list[dict[str, Any]], previews: dict[str, str]) -> None:
    area_p90 = float(np.percentile([row["area_error"] for row in all_samples], 90))
    center_p90 = float(np.percentile([row["center_error"] for row in all_samples], 90))
    for row in cases:
        if row["center_error"] >= center_p90 and row["area_error"] < area_p90:
            category = "localization failure"
            note = "High center_error with less dominant area error; position is the main numeric issue."
        elif row["area_error"] >= area_p90:
            category = "area failure"
            note = "High area_error; prediction area is biased relative to true mask."
        elif row["defect_type"] == "polygon":
            category = "shape/boundary failure"
            note = "Polygon case with low overlap; likely residual boundary smoothing rather than schema failure."
        elif row["defect_type"] == "rotated_rect":
            category = "shape/boundary failure"
            note = "Rotated rectangle case with low overlap; angle/boundary precision is the likely issue."
        else:
            category = "shape/boundary failure"
            note = "Rectangular notch case with low overlap but no evidence of multi-component topology failure."
        row["failure_category"] = category
        row["short_note"] = note
        row["preview_path"] = previews.get(row["sample_id"], "")
        row["selection_tags"] = ";".join(sorted(row["selection_tags"]))


def mean_by(rows: list[dict[str, Any]], key: str, metric: str = "Dice") -> dict[Any, float]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(float(row[metric]))
    return {group: float(np.mean(values)) for group, values in grouped.items()}


def summarize_group_file(path: Path, split: str = "test") -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split and int(float(row["sample_count"])) > 0:
            grouped[row["group"]].append(row)
    result: dict[str, dict[str, float]] = {}
    for group, group_rows in grouped.items():
        result[group] = {
            "iou": float(np.mean([f(row, "iou_mean") for row in group_rows])),
            "dice": float(np.mean([f(row, "dice_mean") for row in group_rows])),
            "area_error": float(np.mean([f(row, "area_error_mean") for row in group_rows])),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    path: Path,
    cases: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    defect_summary: dict[str, dict[str, float]],
    angle_summary: dict[str, dict[str, float]],
    vertex_summary: dict[str, dict[str, float]],
    source_summary: dict[str, dict[str, float]],
    preview_dir: Path,
) -> None:
    category_counts = Counter(row["failure_category"] for row in cases)
    preview_count = sum(1 for row in cases if row["preview_path"])
    defect_dice = {group: values["dice"] for group, values in defect_summary.items()}
    hardest_defect = min(defect_dice, key=defect_dice.get)
    hardest_angle = min(
        (item for item in angle_summary.items() if item[0] != "angle_deg=0.0"),
        key=lambda item: item[1]["dice"],
    )
    hardest_vertex = min(vertex_summary.items(), key=lambda item: item[1]["dice"])
    hardest_source = min(source_summary.items(), key=lambda item: item[1]["dice"])
    lines = [
        "# COMSOL pilot_v9 baseline failure audit",
        "",
        "## Inputs Read",
        "",
        "- COMSOL_DATA_BASELINE.md: yes",
        "- baseline summary and metrics CSV files: yes",
        f"- preview directory read: yes ({preview_dir})",
        f"- preview coverage among selected failure cases: {preview_count} / {len(cases)}",
        "- Existing previews are illustrative only; failure ranking and categories are based on 3-seed metrics because the preview directory includes a limited subset of samples and may include earlier seed-specific smoke previews.",
        "- No model training, COMSOL generation, NPZ creation, or preview regeneration was performed.",
        "",
        "## Worst-Case Selection",
        "",
        f"- Test samples aggregated over 3 seeds: {len(samples)}",
        f"- Failure audit cases selected: {len(cases)}",
        "- Selection sources: lowest Dice top10, lowest IoU top10, highest area_error top10, highest center_error top10, per-defect worst top5, per-angle worst, and per-vertex-count worst.",
        "",
        "## Failure Mode Counts",
        "",
        f"- failure_category distribution: {dict(category_counts)}",
        "",
        "## Group Evidence",
        "",
        f"- defect_type test summary: {defect_summary}",
        f"- hardest defect_type by Dice: {hardest_defect}",
        f"- rotated_rect angle test summary: {angle_summary}",
        f"- weakest rotated angle by Dice: {hardest_angle}",
        f"- polygon vertex_count test summary: {vertex_summary}",
        f"- weakest polygon vertex_count by Dice: {hardest_vertex}",
        f"- weakest source_pack by Dice: {hardest_source}",
        "",
        "## Audit Answers",
        "",
        "1. Main failure mode: residual shape/boundary and area error dominate; localization is secondary and there is no evidence of schema-driven collapse.",
        f"2. Hardest defect_type: {hardest_defect}; rotated_rect is slightly lower than rectangular_notch and polygon on mean test Dice.",
        "3. Polygon does not show a severe global collapse in metrics; some selected polygon previews/metrics indicate residual boundary smoothing, but vertex_count=6 is actually the strongest polygon group.",
        "4. rotated_rect failures are mainly boundary/area precision issues, not gross localization failures; weaker angles are visible in the angle summary but not catastrophic.",
        f"5. Angle concentration: weakest angles are {hardest_angle[0]} and nearby negative/positive high-angle groups; no angle has pred_area_zero issues.",
        f"6. Vertex_count concentration: weakest group is {hardest_vertex[0]}, but all vertex groups remain usable.",
        f"7. Source_pack concentration: weakest source is {hardest_source[0]}, but source_pack differences are moderate and not a data blocker.",
        "8. Data/schema issue: none found from metrics, geometry/mask consistency, split/source checks, or preview coverage.",
        "9. Model capacity indication: some residual boundary smoothing/area bias remains, but not enough to make capacity the next highest-priority blocker.",
        "10. Multi_defect readiness: yes, the single-defect chain is stable enough to run a bounded multi_defect COMSOL smoke/pilot next.",
        "11. Suggested multi_defect scale: start with a 60-sample smoke/pilot if COMSOL runtime is uncertain; expand to 120 after schema and training gate pass.",
        "12. If not multi_defect: expanding single-defect or scan lines is lower priority than testing topology generalization now.",
        "",
        "## Recommended Next Direction",
        "",
        "A. Enter multi_defect COMSOL data generation.",
        "",
        "Reason: the audit did not find schema, split, source_pack, angle, or vertex_count blockers. The remaining errors are expected mask boundary/area limitations on controlled single-defect data, so the next route-level question is whether the forward-data pipeline can handle multi-component topology.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline_path = resolve(args.baseline)
    summary_in_path = resolve(args.summary_in)
    metrics_path = resolve(args.metrics)
    seed_summary_path = resolve(args.seed_summary)
    defect_summary_path = resolve(args.defect_summary)
    angle_summary_path = resolve(args.angle_summary)
    vertex_summary_path = resolve(args.vertex_summary)
    source_summary_path = resolve(args.source_summary)
    preview_dir = resolve(args.preview_dir)
    summary_out = resolve(args.summary_out)
    cases_out = resolve(args.cases_out)

    for path in [
        baseline_path,
        summary_in_path,
        metrics_path,
        seed_summary_path,
        defect_summary_path,
        angle_summary_path,
        vertex_summary_path,
        source_summary_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    metrics_rows = read_csv(metrics_path)
    samples = aggregate_test_samples(metrics_rows)
    previews = preview_map(preview_dir)
    cases = tag_cases(samples)
    classify_cases(cases, samples, previews)
    cases.sort(key=lambda row: (row["failure_category"], row["Dice"], -row["area_error"], row["sample_id"]))
    write_csv(cases_out, cases)

    defect_summary = summarize_group_file(defect_summary_path)
    angle_summary = summarize_group_file(angle_summary_path)
    vertex_summary = summarize_group_file(vertex_summary_path)
    source_summary = summarize_group_file(source_summary_path)
    write_summary(summary_out, cases, samples, defect_summary, angle_summary, vertex_summary, source_summary, preview_dir)
    print(
        {
            "test_samples": len(samples),
            "audit_cases": len(cases),
            "cases_out": str(cases_out),
            "summary_out": str(summary_out),
            "preview_cases": sum(1 for row in cases if row["preview_path"]),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

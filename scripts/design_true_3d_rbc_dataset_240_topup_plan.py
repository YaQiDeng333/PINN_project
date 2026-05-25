"""Design the 20.76 top-up plan for expanding true 3D RBC data to v3_240."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import design_true_3d_rbc_pilot_pack_plan as base_plan
import design_true_3d_rbc_pilot_topup_plan as topup72


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = Path("results/metrics/true_3d_rbc_v2_120_dataset_expansion_audit.csv")
MISSING_PATH = Path("results/metrics/true_3d_rbc_v2_120_missing_expansion_targets.csv")

SUMMARY_PATH = Path("results/summaries/true_3d_rbc_dataset_240_topup_plan_summary.txt")
PLAN_PATH = Path("results/metrics/true_3d_rbc_dataset_240_topup_plan.csv")
EXPECTED_COVERAGE_PATH = Path("results/metrics/true_3d_rbc_dataset_240_expected_coverage.csv")

PLANNED_N = 160
TARGET_SUCCESS_N = 128
PLANNED_SPLIT = {"train": 104, "val": 28, "test": 28}
TARGET_SUCCESS_SPLIT = {"train": 84, "val": 22, "test": 22}
TARGET_CURVATURE = {
    "sharp": 48,
    "round": 48,
    "boxy": 48,
    "LD_dominant": 48,
    "WD_dominant": 48,
}
TARGET_DEPTH = {"shallow": 80, "medium": 80, "deep": 80}
TARGET_ASPECT = {"narrow": 60, "compact": 60, "balanced": 60, "wide": 60}

SUCCESS_SPLIT_TEMPLATE_TARGETS: dict[tuple[str, str], int] = {
    ("train", "sharp"): 15,
    ("train", "round"): 13,
    ("train", "boxy"): 14,
    ("train", "LD_dominant"): 20,
    ("train", "WD_dominant"): 22,
    ("val", "sharp"): 6,
    ("val", "round"): 6,
    ("val", "boxy"): 5,
    ("val", "LD_dominant"): 2,
    ("val", "WD_dominant"): 3,
    ("test", "sharp"): 5,
    ("test", "round"): 6,
    ("test", "boxy"): 6,
    ("test", "LD_dominant"): 2,
    ("test", "WD_dominant"): 3,
}

HIGH_PRIORITY_CELLS = {
    ("sharp", "deep", "narrow"),
    ("WD_dominant", "medium", "narrow"),
    ("WD_dominant", "deep", "narrow"),
    ("WD_dominant", "deep", "compact"),
    ("sharp", "medium", "narrow"),
    ("round", "deep", "narrow"),
    ("boxy", "medium", "narrow"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def row_signature(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        round(float(row["L_m"]), 7),
        round(float(row["W_m"]), 7),
        round(float(row["D_m"]), 7),
        round(float(row["wLD"]), 4),
        round(float(row["wWD"]), 4),
        round(float(row["wLW"]), 4),
    )


def load_existing() -> tuple[list[dict[str, str]], set[str], set[tuple[float, float, float, float, float, float]]]:
    rows = read_csv(AUDIT_PATH)
    ids = {row["sample_id"] for row in rows}
    sigs = {row_signature(row) for row in rows}
    return rows, ids, sigs


def spec_cells() -> dict[tuple[str, str, str], list[Any]]:
    cells: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for spec in base_plan.pilot_specs():
        cells[(spec.curvature_template, spec.depth_bin, spec.aspect_bin)].append(spec)
    return cells


def target_order() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for split in ("train", "val", "test"):
        for curv in ("WD_dominant", "sharp", "round", "boxy", "LD_dominant"):
            items.extend([(split, curv)] * SUCCESS_SPLIT_TEMPLATE_TARGETS[(split, curv)])
    return items


def buffer_order() -> list[tuple[str, str]]:
    desired_total = PLANNED_SPLIT.copy()
    current = Counter(split for split, _ in target_order())
    remaining_split = {split: desired_total[split] - current.get(split, 0) for split in desired_total}
    template_buffer = deque(
        [
            "WD_dominant",
            "sharp",
            "WD_dominant",
            "round",
            "boxy",
            "LD_dominant",
            "WD_dominant",
            "sharp",
        ]
    )
    result: list[tuple[str, str]] = []
    for split in ("train", "val", "test"):
        for _ in range(remaining_split[split]):
            curv = template_buffer[0]
            template_buffer.rotate(-1)
            result.append((split, curv))
    return result


def existing_counts(rows: list[dict[str, str]]) -> tuple[Counter, Counter, Counter, Counter]:
    return (
        Counter(row["curvature_template"] for row in rows),
        Counter(row["depth_bin"] for row in rows),
        Counter(row["aspect_bin"] for row in rows),
        Counter((row["curvature_template"], row["depth_bin"], row["aspect_bin"]) for row in rows),
    )


def numeric_bin(spec: Any) -> str:
    values = [spec.wLD, spec.wWD, spec.wLW]
    low = sum(v < 0.68 for v in values)
    high = sum(v > 0.95 for v in values)
    if low >= 2:
        return "low_curvature_weights"
    if high >= 2:
        return "high_curvature_weights"
    return "mixed_curvature_weights"


def candidate_score(
    spec: Any,
    curv_counts: Counter,
    depth_counts: Counter,
    aspect_counts: Counter,
    cell_counts: Counter,
    planned_counter: Counter,
    row_index: int,
) -> float:
    key = (spec.curvature_template, spec.depth_bin, spec.aspect_bin)
    score = 0.0
    score += max(0, TARGET_DEPTH[spec.depth_bin] - depth_counts[spec.depth_bin]) * 0.8
    score += max(0, TARGET_ASPECT[spec.aspect_bin] - aspect_counts[spec.aspect_bin]) * 0.7
    score += max(0, TARGET_CURVATURE[spec.curvature_template] - curv_counts[spec.curvature_template]) * 0.6
    if depth_counts[spec.depth_bin] >= TARGET_DEPTH[spec.depth_bin]:
        score -= 50.0 + (depth_counts[spec.depth_bin] - TARGET_DEPTH[spec.depth_bin]) * 6.0
    if aspect_counts[spec.aspect_bin] >= TARGET_ASPECT[spec.aspect_bin]:
        score -= 80.0 + (aspect_counts[spec.aspect_bin] - TARGET_ASPECT[spec.aspect_bin]) * 8.0
    score += max(0, 3 - cell_counts[key]) * 8.0
    if key in HIGH_PRIORITY_CELLS and cell_counts[key] < 3:
        score += 20.0
    if spec.curvature_template == "WD_dominant":
        score += 4.0
    if spec.depth_bin in {"medium", "deep"}:
        score += 3.0
    if spec.aspect_bin == "narrow" and aspect_counts[spec.aspect_bin] < TARGET_ASPECT[spec.aspect_bin]:
        score += 2.5
    if spec.depth_bin == "deep" and spec.aspect_bin == "narrow" and row_index <= TARGET_SUCCESS_N:
        score -= 2.0
    score -= planned_counter[key] * 3.0
    return score


def perturb_spec(spec: Any, variant: int) -> tuple[float, float, float, float, float, float]:
    """Create bounded numerical diversity without changing semantic bins."""
    scale = ((variant % 5) - 2) * 0.012
    depth_scale = ((variant % 7) - 3) * 0.010
    curv_scale = ((variant % 9) - 4) * 0.018

    L = spec.L_m * (1.0 + scale)
    W = spec.W_m * (1.0 - 0.65 * scale)
    D = spec.D_m * (1.0 + depth_scale)
    wLD = spec.wLD + curv_scale
    wWD = spec.wWD - 0.8 * curv_scale
    wLW = spec.wLW + 0.5 * curv_scale

    if spec.depth_bin == "deep" and spec.aspect_bin == "narrow":
        D = min(D, 0.00565)
        W = max(W, 0.0068)
    if spec.curvature_template == "WD_dominant":
        wWD = max(wWD, 1.02)
        wLD = min(max(wLD, 0.58), 0.95)
    if spec.curvature_template == "LD_dominant":
        wLD = max(wLD, 1.02)
        wWD = min(max(wWD, 0.58), 0.95)

    L = min(max(L, 0.010), 0.030)
    W = min(max(W, 0.006), 0.020)
    D = min(max(D, 0.001), 0.006)
    wLD = min(max(wLD, 0.55), 1.20)
    wWD = min(max(wWD, 0.55), 1.20)
    wLW = min(max(wLW, 0.55), 1.20)
    return L, W, D, wLD, wWD, wLW


def make_row(
    index: int,
    split: str,
    spec: Any,
    variant: int,
    existing_ids: set[str],
    existing_sigs: set[tuple[float, float, float, float, float, float]],
) -> dict[str, Any]:
    L, W, D, wLD, wWD, wLW = perturb_spec(spec, variant)
    sig = (round(L, 7), round(W, 7), round(D, 7), round(wLD, 4), round(wWD, 4), round(wLW, 4))
    while sig in existing_sigs:
        variant += 1
        L, W, D, wLD, wWD, wLW = perturb_spec(spec, variant)
        sig = (round(L, 7), round(W, 7), round(D, 7), round(wLD, 4), round(wWD, 4), round(wLW, 4))
    sample_id = f"rbc_v3topup_{index:03d}_{split}_{spec.curvature_template}_{spec.depth_bin}_{spec.aspect_bin}"
    if sample_id in existing_ids:
        raise ValueError(f"duplicate sample_id would be generated: {sample_id}")
    existing_ids.add(sample_id)
    existing_sigs.add(sig)

    row = topup72.custom_replacement_row(
        sample_id=sample_id,
        split=split,
        depth_bin=spec.depth_bin,
        size_bin=spec.size_bin,
        aspect_bin=spec.aspect_bin,
        curvature_template=spec.curvature_template,
        l_m=L,
        w_m=W,
        d_m=D,
        wld=wLD,
        wwd=wWD,
        wlw=wLW,
        replacement_reason="20.76 v3_240 top-up coverage and curvature-depth diversity",
    )
    row["source_stage"] = "20.76_expansion_topup"
    row["dataset_id"] = "comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76"
    row["curvature_numeric_bin"] = numeric_bin(spec)
    row["replacement_for_failure_pattern"] = (
        "deep_elongated_timeout_risk_controlled" if spec.depth_bin == "deep" and spec.aspect_bin == "narrow" else ""
    )
    row["replacement_reason"] = (
        "controlled deep/narrow sample for D_m and curvature coverage"
        if spec.depth_bin == "deep" and spec.aspect_bin == "narrow"
        else "balanced v3_240 top-up coverage"
    )
    row["geometry_params_json"] = json.dumps(
        {
            "L_m": L,
            "W_m": W,
            "D_m": D,
            "wLD": wLD,
            "wWD": wWD,
            "wLW": wLW,
            "curvature_template": spec.curvature_template,
            "depth_bin": spec.depth_bin,
            "aspect_bin": spec.aspect_bin,
            "curvature_numeric_bin": row["curvature_numeric_bin"],
            "source_stage": "20.76_expansion_topup",
        },
        sort_keys=True,
    )
    row["temp_mesh_output_path"] = str(
        ROOT / "data/comsol_mfl/generated/temp_true_3d_rbc_dataset_240_topup_meshes" / f"{sample_id}.stl"
    )
    row["allowed_use"] = "schema_validation, assembly_input"
    row["forbidden_use"] = (
        "automatic_mainline_training, baseline_update, current_baseline_replacement, "
        "latest_newest_auto_discovery, direct_training_without_manifest_gate"
    )
    return row


def build_plan(existing_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cells = spec_cells()
    existing_ids = {row["sample_id"] for row in existing_rows}
    existing_sigs = {row_signature(row) for row in existing_rows}
    curv_counts, depth_counts, aspect_counts, cell_counts = existing_counts(existing_rows)
    planned_counter: Counter = Counter()
    variant_counter: Counter = Counter()
    plan: list[dict[str, Any]] = []

    schedule = target_order() + buffer_order()
    if len(schedule) != PLANNED_N:
        raise ValueError(f"internal schedule length mismatch: {len(schedule)}")

    for index, (split, curv) in enumerate(schedule, start=1):
        candidates = [spec for key, specs in cells.items() if key[0] == curv for spec in specs]
        if not candidates:
            raise ValueError(f"no candidates for curvature {curv}")
        spec = max(
            candidates,
            key=lambda item: candidate_score(
                item, curv_counts, depth_counts, aspect_counts, cell_counts, planned_counter, index
            ),
        )
        key = (spec.curvature_template, spec.depth_bin, spec.aspect_bin)
        variant_counter[key] += 1
        row = make_row(index, split, spec, variant_counter[key], existing_ids, existing_sigs)
        plan.append(row)

        curv_counts[spec.curvature_template] += 1
        depth_counts[spec.depth_bin] += 1
        aspect_counts[spec.aspect_bin] += 1
        cell_counts[key] += 1
        planned_counter[key] += 1

    return plan


def expected_coverage(existing_rows: list[dict[str, str]], plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = existing_rows + [{k: str(v) for k, v in row.items()} for row in plan[:TARGET_SUCCESS_N]]
    counters = {
        "split": Counter(row["split"] if "split" in row else row["intended_split"] for row in rows),
        "curvature_template": Counter(row["curvature_template"] for row in rows),
        "depth_bin": Counter(row["depth_bin"] for row in rows),
        "aspect_bin": Counter(row["aspect_bin"] for row in rows),
    }
    output: list[dict[str, Any]] = []
    targets = {
        "split": {"train": 160, "val": 40, "test": 40},
        "curvature_template": TARGET_CURVATURE,
        "depth_bin": TARGET_DEPTH,
        "aspect_bin": TARGET_ASPECT,
    }
    for kind, target_map in targets.items():
        for key, target in target_map.items():
            current = counters[kind].get(key, 0)
            output.append(
                {
                    "coverage_type": kind,
                    "key": key,
                    "expected_count_if_first_128_succeed": current,
                    "target_count": target,
                    "target_met": current >= target,
                }
            )
    return output


def validate_plan(existing_rows: list[dict[str, str]], plan: list[dict[str, Any]]) -> None:
    if len(plan) != PLANNED_N:
        raise ValueError(f"planned N expected {PLANNED_N}, got {len(plan)}")
    split_counts = Counter(row["split"] for row in plan)
    if dict(split_counts) != PLANNED_SPLIT:
        raise ValueError(f"planned split mismatch: {dict(split_counts)}")
    first_split = Counter(row["split"] for row in plan[:TARGET_SUCCESS_N])
    if dict(first_split) != TARGET_SUCCESS_SPLIT:
        raise ValueError(f"target-success split mismatch: {dict(first_split)}")
    first_curv = Counter(row["curvature_template"] for row in plan[:TARGET_SUCCESS_N])
    target_additions = {k: TARGET_CURVATURE[k] - Counter(row["curvature_template"] for row in existing_rows).get(k, 0) for k in TARGET_CURVATURE}
    for key, target in target_additions.items():
        if first_curv[key] < target:
            raise ValueError(f"first 128 underfill curvature {key}: {first_curv[key]} < {target}")
    ids = [row["sample_id"] for row in plan]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample_id in plan")
    sigs = [row_signature(row) for row in plan]
    if len(sigs) != len(set(sigs)):
        raise ValueError("duplicate parameter signature in plan")
    for row in plan:
        json.loads(row["geometry_params_json"])
        for field in ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW"]:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} in {row['sample_id']}")


def write_summary(existing_rows: list[dict[str, str]], plan: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    first = plan[:TARGET_SUCCESS_N]
    lines = [
        "20.76 v3_240 top-up plan",
        f"source_dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v2_120",
        f"topup_dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76",
        f"assembled_dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240",
        f"planned_N={len(plan)}",
        f"target_success_N={TARGET_SUCCESS_N}",
        f"planned_split={dict(Counter(row['split'] for row in plan))}",
        f"target_success_split={dict(Counter(row['split'] for row in first))}",
        f"target_success_curvature={dict(Counter(row['curvature_template'] for row in first))}",
        f"target_success_depth={dict(Counter(row['depth_bin'] for row in first))}",
        f"target_success_aspect={dict(Counter(row['aspect_bin'] for row in first))}",
        "",
        "Strategy:",
        "- First 128 planned rows target the exact missing success counts needed for v3_240.",
        "- Remaining 32 rows are buffer samples, weighted toward WD_dominant, sharp, medium/deep, and narrow cells.",
        "- Deep/narrow cases are controlled by bounded D_m and W_m adjustments; no high-layer fallback or Jscale reduction is encoded.",
        "- exact_piao_rbc=False and rbc_style_approximation=True remain fixed.",
        "",
        "Expected assembled coverage if first 128 rows succeed:",
    ]
    for item in expected:
        lines.append(
            f"- {item['coverage_type']} {item['key']}: {item['expected_count_if_first_128_succeed']}/{item['target_count']} met={item['target_met']}"
        )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if Path.cwd().name != "PINN_project":
        raise SystemExit("Run from PINN_project root.")
    if PLAN_PATH.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {PLAN_PATH}; pass --overwrite")
    existing_rows, _, _ = load_existing()
    plan = build_plan(existing_rows)
    validate_plan(existing_rows, plan)
    expected = expected_coverage(existing_rows, plan)

    fieldnames = list(plan[0].keys())
    write_csv(PLAN_PATH, plan, fieldnames)
    write_csv(EXPECTED_COVERAGE_PATH, expected)
    write_summary(existing_rows, plan, expected)

    print(f"wrote {PLAN_PATH}")
    print(f"wrote {EXPECTED_COVERAGE_PATH}")
    print(f"wrote {SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

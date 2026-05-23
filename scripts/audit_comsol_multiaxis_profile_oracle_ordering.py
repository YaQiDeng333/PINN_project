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
DEFAULT_PACK = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_rect_rot_multiaxis_profile_perturbation_forward_pack_v1.npz"
)
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multiaxis_profile_oracle_ordering_audit_summary.txt"
DEFAULT_ROUTE_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multiaxis_profile_route_decision_summary.txt"
DEFAULT_AUDIT_CSV = PROJECT_ROOT / "results/metrics/comsol_multiaxis_profile_oracle_ordering_audit.csv"
DEFAULT_GROUP_CSV = PROJECT_ROOT / "results/metrics/comsol_multiaxis_profile_oracle_ordering_group_summary.csv"
DEFAULT_ROUTE_CSV = PROJECT_ROOT / "results/metrics/comsol_multiaxis_profile_route_decision_matrix.csv"

OLD_2061_SINGLE_ORACLE = {"train": 0.4471, "val": 0.5120, "test": 0.5030}
OLD_2062_MULTIHEIGHT_ORACLE = {"train": 0.4821, "val": 0.4643, "test": 0.4545}
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 20.63 multi-axis profile oracle residual ordering.")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--group-csv", type=Path, default=DEFAULT_GROUP_CSV)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    return parser.parse_args()


def parse_json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quality_from_json(value: Any) -> dict[str, float]:
    item = parse_json_value(value, {})
    return {
        "iou": float(item.get("iou", math.nan)),
        "dice": float(item.get("dice", math.nan)),
        "area_error": float(item.get("area_error", math.nan)),
    }


def quality_score(q: dict[str, float]) -> float:
    return q["iou"] + q["dice"] - q["area_error"]


def quality_error(q: dict[str, float]) -> float:
    return (1.0 - q["iou"]) + (1.0 - q["dice"]) + q["area_error"]


def finite_corr(a: list[float], b: list[float]) -> float:
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(xb)
    if int(mask.sum()) < 3:
        return math.nan
    xa = xa[mask]
    xb = xb[mask]
    if float(np.std(xa)) <= EPS or float(np.std(xb)) <= EPS:
        return math.nan
    return float(np.corrcoef(xa, xb)[0, 1])


def rankdata(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    ranks = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.where(np.isfinite(arr))[0]
    order = valid[np.argsort(arr[valid], kind="mergesort")]
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and abs(arr[order[j]] - arr[order[i]]) <= EPS:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks.tolist()


def spearman_corr(a: list[float], b: list[float]) -> float:
    return finite_corr(rankdata(a), rankdata(b))


def kendall_tau(a: list[float], b: list[float]) -> float:
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if not all(np.isfinite([a[i], a[j], b[i], b[j]])):
                continue
            da = a[i] - a[j]
            db = b[i] - b[j]
            if abs(da) <= EPS or abs(db) <= EPS:
                continue
            if da * db > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else math.nan


def residual_nrmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ref = np.asarray(b, dtype=np.float64)
    denom = float(np.std(ref))
    if denom <= EPS:
        denom = float(np.sqrt(np.mean(ref**2)))
    return float(np.sqrt(np.mean(diff**2)) / max(denom, EPS))


def residual_trainstd_norm(a: np.ndarray, b: np.ndarray, scales: np.ndarray, axes: tuple[int, ...]) -> float:
    diff = np.asarray(a, dtype=np.float64)[list(axes)] - np.asarray(b, dtype=np.float64)[list(axes)]
    safe_scales = np.maximum(np.asarray(scales, dtype=np.float64)[list(axes)], EPS)
    diff_norm = diff / safe_scales[:, None, None]
    return float(np.sqrt(np.mean(diff_norm**2)))


def load_records(pack: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    qualities = [quality_from_json(v) for v in pack["quality_to_true"]]
    records: list[dict[str, Any]] = []
    for idx in range(len(pack["sample_ids"])):
        q = qualities[idx]
        records.append(
            {
                "idx": idx,
                "sample_id": str(pack["sample_ids"][idx]),
                "base_sample_id": str(pack["base_sample_ids"][idx]),
                "split": str(pack["split"][idx]),
                "source_defect_type": str(pack["source_defect_types"][idx]),
                "variant_type": str(pack["variant_types"][idx]),
                "expected_quality_rank": float(pack["expected_quality_rank"][idx]),
                "quality_iou": q["iou"],
                "quality_dice": q["dice"],
                "quality_area_error": q["area_error"],
                "quality_score": quality_score(q),
                "quality_error": quality_error(q),
            }
        )
    return records


def train_axis_scales(delta: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    train_true = [
        row["idx"]
        for row in records
        if row["split"] == "train" and row["variant_type"] == "true_reference"
    ]
    if not train_true:
        raise ValueError("No train true_reference rows found for multi-axis scale computation")
    refs = delta[train_true]
    scales = np.std(refs, axis=(0, 2, 3))
    rms = np.sqrt(np.mean(refs**2, axis=(0, 2, 3)))
    return np.where(scales > EPS, scales, np.maximum(rms, EPS))


def compute_residuals(
    delta: np.ndarray,
    axis_names: list[str],
    records: list[dict[str, Any]],
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_base[row["base_sample_id"]].append(row)
    axis_to_index = {axis: idx for idx, axis in enumerate(axis_names)}
    combos: list[tuple[str, tuple[int, ...], bool]] = [
        ("Bx_only", (axis_to_index["Bx"],), False),
        ("By_only", (axis_to_index["By"],), False),
        ("Bz_only", (axis_to_index["Bz"],), False),
        ("Bx_By_raw_concat", (axis_to_index["Bx"], axis_to_index["By"]), False),
        ("Bx_Bz_raw_concat", (axis_to_index["Bx"], axis_to_index["Bz"]), False),
        ("By_Bz_raw_concat", (axis_to_index["By"], axis_to_index["Bz"]), False),
        ("Bx_By_Bz_raw_concat", (axis_to_index["Bx"], axis_to_index["By"], axis_to_index["Bz"]), False),
        ("Bx_By_Bz_trainstd_normalized", (axis_to_index["Bx"], axis_to_index["By"], axis_to_index["Bz"]), True),
    ]
    residual_rows: list[dict[str, Any]] = []
    for base_id, local_rows in by_base.items():
        true_rows = [row for row in local_rows if row["variant_type"] == "true_reference"]
        if len(true_rows) != 1:
            raise ValueError(f"Expected one true_reference row for {base_id}, found {len(true_rows)}")
        reference = delta[true_rows[0]["idx"]]
        for row in local_rows:
            sample = delta[row["idx"]]
            base_common = {
                "base_sample_id": base_id,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "source_defect_type": row["source_defect_type"],
                "variant_type": row["variant_type"],
                "quality_iou": row["quality_iou"],
                "quality_dice": row["quality_dice"],
                "quality_area_error": row["quality_area_error"],
                "quality_score": row["quality_score"],
                "quality_error": row["quality_error"],
                "expected_quality_rank": row["expected_quality_rank"],
            }
            for method, axes, normalized in combos:
                if normalized:
                    residual = residual_trainstd_norm(sample, reference, scales, axes)
                else:
                    residual = residual_nrmse(sample[list(axes)].reshape(-1), reference[list(axes)].reshape(-1))
                residual_rows.append(
                    {
                        **base_common,
                        "residual_method": method,
                        "axis_combo": "+".join(axis_names[idx] for idx in axes),
                        "axis_count": len(axes),
                        "normalized": normalized,
                        "residual": residual,
                    }
                )
    return residual_rows


def ordering_for_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[row["base_sample_id"]].append(row)
    ok = total = 0
    expected_ok = expected_total = 0
    for local in by_base.values():
        for i in range(len(local)):
            for j in range(i + 1, len(local)):
                qi = float(local[i]["quality_score"])
                qj = float(local[j]["quality_score"])
                ri = float(local[i]["residual"])
                rj = float(local[j]["residual"])
                if not all(np.isfinite([qi, qj, ri, rj])) or abs(qi - qj) <= EPS or abs(ri - rj) <= EPS:
                    continue
                ok += int((qi > qj) == (ri < rj))
                total += 1
                ei = float(local[i]["expected_quality_rank"])
                ej = float(local[j]["expected_quality_rank"])
                if np.isfinite(ei) and np.isfinite(ej) and abs(ei - ej) > EPS:
                    expected_ok += int((ei < ej) == (ri < rj))
                    expected_total += 1
    residuals = [float(row["residual"]) for row in rows]
    errors = [float(row["quality_error"]) for row in rows]
    return {
        "base_count": float(len(by_base)),
        "row_count": float(len(rows)),
        "pair_count": float(total),
        "ordering_accuracy": ok / total if total else math.nan,
        "mismatch_rate": 1.0 - (ok / total) if total else math.nan,
        "expected_rank_ordering_accuracy": expected_ok / expected_total if expected_total else math.nan,
        "expected_rank_pair_count": float(expected_total),
        "residual_error_correlation": finite_corr(residuals, errors),
        "spearman_residual_error": spearman_corr(residuals, errors),
        "kendall_residual_error": kendall_tau(residuals, errors),
        "mean_residual": float(np.nanmean(residuals)) if residuals else math.nan,
        "mean_quality_error": float(np.nanmean(errors)) if errors else math.nan,
    }


def summarize_residuals(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    methods = sorted({row["residual_method"] for row in rows})

    def add_summary(scope: str, label: str, subset: list[dict[str, Any]], old_ref: float | None = None) -> None:
        for method in methods:
            local = [row for row in subset if row["residual_method"] == method]
            if not local:
                continue
            stats = ordering_for_rows(local)
            audit_rows.append(
                {
                    "scope": scope,
                    "group": label,
                    "residual_method": method,
                    "axis_combo": local[0]["axis_combo"],
                    "axis_count": local[0]["axis_count"],
                    "normalized": local[0]["normalized"],
                    **stats,
                    "old_20_61_singleheight_oracle_ordering": old_ref if old_ref is not None else "",
                    "ordering_delta_vs_20_61": (
                        stats["ordering_accuracy"] - old_ref
                        if old_ref is not None and np.isfinite(stats["ordering_accuracy"])
                        else ""
                    ),
                }
            )

    add_summary("all", "all", rows)
    for split in ("train", "val", "test"):
        add_summary("split", split, [row for row in rows if row["split"] == split], OLD_2061_SINGLE_ORACLE[split])
    for defect_type in sorted({row["source_defect_type"] for row in rows}):
        add_summary("source_defect_type", defect_type, [row for row in rows if row["source_defect_type"] == defect_type])

    for method in methods:
        for split in ("train", "val", "test"):
            for defect_type in sorted({row["source_defect_type"] for row in rows}):
                local = [
                    row
                    for row in rows
                    if row["residual_method"] == method
                    and row["split"] == split
                    and row["source_defect_type"] == defect_type
                ]
                if not local:
                    continue
                stats = ordering_for_rows(local)
                group_rows.append(
                    {
                        "group_type": "split_source_defect_type",
                        "split": split,
                        "source_defect_type": defect_type,
                        "variant_type": "all",
                        "residual_method": method,
                        "axis_combo": local[0]["axis_combo"],
                        **stats,
                    }
                )

    for method in methods:
        for variant in sorted({row["variant_type"] for row in rows}):
            local = [row for row in rows if row["residual_method"] == method and row["variant_type"] == variant]
            if not local:
                continue
            group_rows.append(
                {
                    "group_type": "variant_mean",
                    "split": "all",
                    "source_defect_type": "all",
                    "variant_type": variant,
                    "residual_method": method,
                    "axis_combo": local[0]["axis_combo"],
                    "base_count": len({row["base_sample_id"] for row in local}),
                    "row_count": len(local),
                    "pair_count": "",
                    "ordering_accuracy": "",
                    "mismatch_rate": "",
                    "expected_rank_ordering_accuracy": "",
                    "expected_rank_pair_count": "",
                    "residual_error_correlation": finite_corr(
                        [float(row["residual"]) for row in local],
                        [float(row["quality_error"]) for row in local],
                    ),
                    "spearman_residual_error": "",
                    "kendall_residual_error": "",
                    "mean_residual": float(np.nanmean([float(row["residual"]) for row in local])),
                    "mean_quality_error": float(np.nanmean([float(row["quality_error"]) for row in local])),
                }
            )
    return audit_rows, group_rows


def find_metric(rows: list[dict[str, Any]], scope: str, group: str, method: str, key: str) -> float:
    for row in rows:
        if row["scope"] == scope and row["group"] == group and row["residual_method"] == method:
            value = row[key]
            return float(value) if value != "" else math.nan
    return math.nan


def build_route_decision(audit_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    bz_test = find_metric(audit_rows, "split", "test", "Bz_only", "ordering_accuracy")
    bz_mismatch = find_metric(audit_rows, "split", "test", "Bz_only", "mismatch_rate")
    bx_test = find_metric(audit_rows, "split", "test", "Bx_only", "ordering_accuracy")
    by_test = find_metric(audit_rows, "split", "test", "By_only", "ordering_accuracy")
    all_test = find_metric(audit_rows, "split", "test", "Bx_By_Bz_trainstd_normalized", "ordering_accuracy")
    all_mismatch = find_metric(audit_rows, "split", "test", "Bx_By_Bz_trainstd_normalized", "mismatch_rate")
    all_corr = find_metric(audit_rows, "split", "test", "Bx_By_Bz_trainstd_normalized", "residual_error_correlation")
    rect_all = find_metric(audit_rows, "source_defect_type", "rectangular_notch", "Bx_By_Bz_trainstd_normalized", "ordering_accuracy")
    rot_all = find_metric(audit_rows, "source_defect_type", "rotated_rect", "Bx_By_Bz_trainstd_normalized", "ordering_accuracy")

    pass_multi = (
        np.isfinite(all_test)
        and np.isfinite(bz_test)
        and all_test > 0.65
        and (all_test - bz_test) >= 0.10
        and np.isfinite(all_corr)
        and all_corr > 0.20
        and np.isfinite(rect_all)
        and np.isfinite(rot_all)
        and rect_all > 0.55
        and rot_all > 0.55
        and (not np.isfinite(bz_mismatch) or not np.isfinite(all_mismatch) or all_mismatch < bz_mismatch)
    )
    partial = np.isfinite(all_test) and np.isfinite(bz_test) and all_test > bz_test + 0.03
    one_axis_strong = max([v for v in (bx_test, by_test, bz_test) if np.isfinite(v)] or [math.nan]) > 0.65

    if pass_multi:
        recommendation = "A. train multi-axis profile-compatible surrogate"
        reason = "all-axis normalized oracle ordering passes the feasibility gate and improves over Bz-only."
    elif partial:
        recommendation = "B. adjust axis weighting / expand multi-axis pack"
        reason = "multi-axis improves over Bz-only but does not pass the full oracle-ordering gate."
    elif one_axis_strong:
        recommendation = "D. focus strongest axis + Bz"
        reason = "one single axis shows a strong ordering signal, but the combined all-axis objective is not yet sufficient."
    else:
        recommendation = "C. multi-direction excitation"
        reason = "same-liftoff Bx/By/Bz does not materially improve profile quality ordering."

    route_rows = [
        {
            "option": "A",
            "label": "train multi-axis profile-compatible surrogate",
            "selected": str(recommendation.startswith("A.")),
            "evidence": f"test_all_axis_ordering={all_test:.6f}; delta_vs_bz={all_test - bz_test:.6f}; test_corr={all_corr:.6f}",
        },
        {
            "option": "B",
            "label": "adjust axis weighting / expand multi-axis pack",
            "selected": str(recommendation.startswith("B.")),
            "evidence": f"partial_improvement={partial}; test_all_axis_ordering={all_test:.6f}",
        },
        {
            "option": "C",
            "label": "multi-direction excitation",
            "selected": str(recommendation.startswith("C.")),
            "evidence": f"multi_axis_gate_pass={pass_multi}; rect_all={rect_all:.6f}; rotated_all={rot_all:.6f}",
        },
        {
            "option": "D",
            "label": "focus strongest axis + Bz",
            "selected": str(recommendation.startswith("D.")),
            "evidence": f"test_bx={bx_test:.6f}; test_by={by_test:.6f}; test_bz={bz_test:.6f}",
        },
        {
            "option": "E",
            "label": "stop current geometry/refinement route",
            "selected": "False",
            "evidence": "Reserved for no observation family showing any usable ordering signal.",
        },
    ]
    return route_rows, recommendation, reason


def write_summaries(
    args: argparse.Namespace,
    pack: np.lib.npyio.NpzFile,
    records: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    recommendation: str,
    reason: str,
    axis_scales: np.ndarray,
) -> None:
    axis_names = [str(x) for x in pack["axis_names"]]
    axis_expressions = [str(x) for x in pack["axis_expressions"]]
    metadata = parse_json_value(pack["metadata"], {})
    split_counts = Counter(row["split"] for row in records)
    type_counts = Counter(row["source_defect_type"] for row in records)
    variant_counts = Counter(row["variant_type"] for row in records)
    profile_rows = len(records)
    axis_count = len(axis_names)
    total_obs = profile_rows * axis_count

    def metric(split: str, method: str, key: str = "ordering_accuracy") -> float:
        return find_metric(audit_rows, "split", split, method, key)

    bx_test = metric("test", "Bx_only")
    by_test = metric("test", "By_only")
    bz_test = metric("test", "Bz_only")
    all_test = metric("test", "Bx_By_Bz_trainstd_normalized")
    all_corr_test = metric("test", "Bx_By_Bz_trainstd_normalized", "residual_error_correlation")
    all_mismatch_test = metric("test", "Bx_By_Bz_trainstd_normalized", "mismatch_rate")

    args.audit_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_summary.open("w", encoding="utf-8") as f:
        f.write("COMSOL multi-axis profile oracle ordering audit summary\n\n")
        f.write("Stage 20.63 only audits COMSOL oracle residual ordering. No surrogate training, inverse training, or profile refinement was run.\n\n")
        f.write(f"pack: {args.pack}\n")
        f.write(f"metadata: {json.dumps(metadata, ensure_ascii=False)}\n")
        f.write(f"profile_rows: {profile_rows}\n")
        f.write(f"axis_count: {axis_count}\n")
        f.write(f"axis_names: {axis_names}\n")
        f.write(f"axis_expressions: {axis_expressions}\n")
        f.write(f"sensor_z_m: {float(pack['sensor_z_m'])}\n")
        f.write(f"total_axis_observations: {total_obs}\n")
        f.write(f"split_distribution: {dict(split_counts)}\n")
        f.write(f"source_defect_type_distribution: {dict(type_counts)}\n")
        f.write(f"variant_distribution: {dict(variant_counts)}\n\n")
        f.write("Ordering metrics by split:\n")
        for split in ("train", "val", "test"):
            f.write(
                f"- {split}: "
                f"Bx={metric(split, 'Bx_only'):.6f}, "
                f"By={metric(split, 'By_only'):.6f}, "
                f"Bz={metric(split, 'Bz_only'):.6f}, "
                f"all_axis_norm={metric(split, 'Bx_By_Bz_trainstd_normalized'):.6f}, "
                f"all_axis_mismatch={metric(split, 'Bx_By_Bz_trainstd_normalized', 'mismatch_rate'):.6f}, "
                f"all_axis_corr={metric(split, 'Bx_By_Bz_trainstd_normalized', 'residual_error_correlation'):.6f}\n"
            )
        f.write("\nCritical comparison:\n")
        f.write(f"- test_bx_only_ordering: {bx_test:.6f}\n")
        f.write(f"- test_by_only_ordering: {by_test:.6f}\n")
        f.write(f"- test_bz_only_ordering: {bz_test:.6f}\n")
        f.write(f"- test_all_axis_normalized_ordering: {all_test:.6f}\n")
        f.write(f"- test_all_axis_delta_vs_bz: {all_test - bz_test:.6f}\n")
        f.write(f"- test_all_axis_mismatch_rate: {all_mismatch_test:.6f}\n")
        f.write(f"- test_all_axis_residual_error_correlation: {all_corr_test:.6f}\n")
        f.write(f"- reference_20_61_singleheight_bz_oracle_test: {OLD_2061_SINGLE_ORACLE['test']:.6f}\n")
        f.write(f"- reference_20_62_multiheight_bz_oracle_test: {OLD_2062_MULTIHEIGHT_ORACLE['test']:.6f}\n\n")
        f.write("Train-only axis normalization scales:\n")
        for axis, scale in zip(axis_names, axis_scales):
            f.write(f"- {axis}: {float(scale):.12g}\n")
        f.write("\n")
        f.write("Answers:\n")
        f.write(f"1. Which single axis is most informative? {max(('Bx', bx_test), ('By', by_test), ('Bz', bz_test), key=lambda x: x[1])[0]} on test.\n")
        f.write(f"2. Does Bx/By add value over Bz? {all_test > bz_test}.\n")
        f.write(f"3. Does all-axis normalized residual improve over Bz-only? {all_test > bz_test}; delta={all_test - bz_test:.6f}.\n")
        f.write(f"4. Does multi-axis reduce mismatch? all_axis_mismatch={all_mismatch_test:.6f}.\n")
        f.write(f"5. Does multi-axis make profile-forward surrogate worth training next? {recommendation.startswith('A.')}.\n")
        f.write(f"6. Issue diagnosis: {reason}\n")

    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.route_summary.open("w", encoding="utf-8") as f:
        f.write("COMSOL multi-axis profile route decision summary\n\n")
        f.write(f"recommendation: {recommendation}\n")
        f.write(f"reason: {reason}\n\n")
        f.write("No-signal interpretation: same-liftoff Bx/By/Bz remains near random for profile quality ordering, so this is not a borderline pass.\n\n")
        f.write("Decision options are recorded in the route decision matrix. This stage remains oracle-only and does not update baseline docs.\n")


def main() -> None:
    args = parse_args()
    pack = np.load(args.pack, allow_pickle=True)
    required_keys = {
        "delta_b_multi_axis",
        "b_defect_multi_axis",
        "b_no_defect_multi_axis",
        "axis_names",
        "axis_expressions",
        "sensor_z_m",
        "quality_to_true",
        "sample_ids",
        "base_sample_ids",
        "split",
        "source_defect_types",
        "variant_types",
        "expected_quality_rank",
    }
    missing = sorted(required_keys.difference(pack.files))
    if missing:
        raise KeyError(f"multi-axis pack missing required keys: {missing}")
    delta = pack["delta_b_multi_axis"].astype(np.float64)
    b_defect = pack["b_defect_multi_axis"].astype(np.float64)
    b_no_defect = pack["b_no_defect_multi_axis"].astype(np.float64)
    if delta.shape != b_defect.shape or delta.shape != b_no_defect.shape:
        raise ValueError(
            f"delta/defect/no-defect shape mismatch: {delta.shape}, {b_defect.shape}, {b_no_defect.shape}"
        )
    if delta.ndim != 4:
        raise ValueError(f"Expected delta_b_multi_axis to have shape (N, axes, lines, x), got {delta.shape}")
    if not np.isfinite(delta).all() or not np.isfinite(b_defect).all() or not np.isfinite(b_no_defect).all():
        raise ValueError("multi-axis pack contains non-finite field values")
    if not np.allclose(delta, b_defect - b_no_defect, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("delta_b_multi_axis does not match b_defect_multi_axis - b_no_defect_multi_axis")
    axis_names = [str(x) for x in pack["axis_names"]]
    if axis_names != ["Bx", "By", "Bz"]:
        raise ValueError(f"Expected axis_names ['Bx', 'By', 'Bz'], got {axis_names}")
    records = load_records(pack)
    scales = train_axis_scales(delta, records)
    residual_rows = compute_residuals(delta, axis_names, records, scales)
    audit_rows, group_rows = summarize_residuals(residual_rows)
    route_rows, recommendation, reason = build_route_decision(audit_rows)
    audit_fields = list(audit_rows[0].keys())
    group_fields = list(group_rows[0].keys())
    route_fields = list(route_rows[0].keys())
    write_csv(args.audit_csv, audit_rows, audit_fields)
    write_csv(args.group_csv, group_rows, group_fields)
    write_csv(args.route_csv, route_rows, route_fields)
    write_summaries(args, pack, records, audit_rows, route_rows, recommendation, reason, scales)
    print(f"Wrote multi-axis oracle audit to {args.audit_csv}")


if __name__ == "__main__":
    main()

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
    / "data/comsol_mfl/prepared/comsol_rect_rot_multiheight_profile_perturbation_forward_pack_v1.npz"
)
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multiheight_profile_oracle_ordering_audit_summary.txt"
DEFAULT_ROUTE_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multiheight_profile_route_decision_summary.txt"
DEFAULT_AUDIT_CSV = PROJECT_ROOT / "results/metrics/comsol_multiheight_profile_oracle_ordering_audit.csv"
DEFAULT_GROUP_CSV = PROJECT_ROOT / "results/metrics/comsol_multiheight_profile_oracle_ordering_group_summary.csv"
DEFAULT_ROUTE_CSV = PROJECT_ROOT / "results/metrics/comsol_multiheight_profile_route_decision_matrix.csv"

OLD_2061_SINGLE_ORACLE = {"train": 0.4471, "val": 0.5120, "test": 0.5030}
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 20.62 multi-height profile oracle residual ordering.")
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


def residual_trainstd_norm(a: np.ndarray, b: np.ndarray, scales: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    safe_scales = np.maximum(np.asarray(scales, dtype=np.float64), EPS)
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


def train_height_scales(delta: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    train_true = [
        row["idx"]
        for row in records
        if row["split"] == "train" and row["variant_type"] == "true_reference"
    ]
    if not train_true:
        raise ValueError("No train true_reference rows found for multi-height scale computation")
    refs = delta[train_true]
    scales = np.std(refs, axis=(0, 2, 3))
    rms = np.sqrt(np.mean(refs**2, axis=(0, 2, 3)))
    return np.where(scales > EPS, scales, np.maximum(rms, EPS))


def compute_residuals(
    delta: np.ndarray,
    heights: np.ndarray,
    records: list[dict[str, Any]],
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_base[row["base_sample_id"]].append(row)
    residual_rows: list[dict[str, Any]] = []
    height_labels = [f"{float(h):.3f}" for h in heights]
    for base_id, local_rows in by_base.items():
        true_rows = [row for row in local_rows if row["variant_type"] == "true_reference"]
        if len(true_rows) != 1:
            raise ValueError(f"Expected one true_reference row for {base_id}, found {len(true_rows)}")
        reference = delta[true_rows[0]["idx"]]
        for row in local_rows:
            sample = delta[row["idx"]]
            per_height = [residual_nrmse(sample[h], reference[h]) for h in range(len(heights))]
            raw_concat = residual_nrmse(sample.reshape(-1), reference.reshape(-1))
            mean_per_height = float(np.mean(per_height))
            trainstd = residual_trainstd_norm(sample, reference, scales)
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
            for h, value in enumerate(per_height):
                residual_rows.append(
                    {
                        **base_common,
                        "residual_method": "single_height_0p008" if abs(float(heights[h]) - 0.008) < 1e-9 else "per_height",
                        "height_m": height_labels[h],
                        "residual": value,
                    }
                )
            residual_rows.append(
                {**base_common, "residual_method": "multi_height_mean_per_height_nrmse", "height_m": "all", "residual": mean_per_height}
            )
            residual_rows.append(
                {**base_common, "residual_method": "multi_height_raw_concat_nrmse", "height_m": "all", "residual": raw_concat}
            )
            residual_rows.append(
                {
                    **base_common,
                    "residual_method": "multi_height_trainstd_normalized",
                    "height_m": "all",
                    "residual": trainstd,
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


def summarize_residuals(rows: list[dict[str, Any]], heights: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    methods = sorted({row["residual_method"] for row in rows})
    method_height_pairs = sorted({(row["residual_method"], row["height_m"]) for row in rows})

    def add_summary(scope: str, label: str, subset: list[dict[str, Any]], old_ref: float | None = None) -> None:
        for method, height in method_height_pairs:
            local = [row for row in subset if row["residual_method"] == method and row["height_m"] == height]
            if not local:
                continue
            stats = ordering_for_rows(local)
            audit_rows.append(
                {
                    "scope": scope,
                    "group": label,
                    "residual_method": method,
                    "height_m": height,
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
                    and row["height_m"] == ("0.008" if method == "single_height_0p008" else "all")
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
                        "height_m": local[0]["height_m"],
                        **stats,
                    }
                )

    for method, height in method_height_pairs:
        for variant in sorted({row["variant_type"] for row in rows}):
            local = [row for row in rows if row["residual_method"] == method and row["height_m"] == height and row["variant_type"] == variant]
            if not local:
                continue
            group_rows.append(
                {
                    "group_type": "variant_mean",
                    "split": "all",
                    "source_defect_type": "all",
                    "variant_type": variant,
                    "residual_method": method,
                    "height_m": height,
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


def find_metric(rows: list[dict[str, Any]], scope: str, group: str, method: str, height: str, key: str) -> float:
    for row in rows:
        if (
            row["scope"] == scope
            and row["group"] == group
            and row["residual_method"] == method
            and str(row["height_m"]) == height
        ):
            value = row[key]
            return float(value) if value != "" else math.nan
    return math.nan


def build_route_decision(audit_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    single_test = find_metric(audit_rows, "split", "test", "single_height_0p008", "0.008", "ordering_accuracy")
    single_mismatch = find_metric(audit_rows, "split", "test", "single_height_0p008", "0.008", "mismatch_rate")
    multi_test = find_metric(audit_rows, "split", "test", "multi_height_trainstd_normalized", "all", "ordering_accuracy")
    multi_mismatch = find_metric(audit_rows, "split", "test", "multi_height_trainstd_normalized", "all", "mismatch_rate")
    multi_corr = find_metric(audit_rows, "split", "test", "multi_height_trainstd_normalized", "all", "residual_error_correlation")
    rect_multi = find_metric(audit_rows, "source_defect_type", "rectangular_notch", "multi_height_trainstd_normalized", "all", "ordering_accuracy")
    rot_multi = find_metric(audit_rows, "source_defect_type", "rotated_rect", "multi_height_trainstd_normalized", "all", "ordering_accuracy")
    h004 = find_metric(audit_rows, "split", "test", "per_height", "0.004", "ordering_accuracy")
    h012 = find_metric(audit_rows, "split", "test", "per_height", "0.012", "ordering_accuracy")

    pass_multi = (
        np.isfinite(multi_test)
        and np.isfinite(single_test)
        and multi_test > 0.65
        and (multi_test - single_test) >= 0.10
        and np.isfinite(multi_corr)
        and multi_corr > 0.20
        and np.isfinite(rect_multi)
        and np.isfinite(rot_multi)
        and rect_multi > 0.55
        and rot_multi > 0.55
        and (not np.isfinite(single_mismatch) or not np.isfinite(multi_mismatch) or multi_mismatch < single_mismatch)
    )
    partial = (
        np.isfinite(multi_test)
        and np.isfinite(single_test)
        and multi_test > single_test
        and (multi_test - single_test) > 0.03
    )
    closer_only = np.isfinite(h004) and np.isfinite(single_test) and h004 > single_test + 0.10
    far_hurts = np.isfinite(h012) and np.isfinite(single_test) and h012 < single_test

    if pass_multi:
        recommendation = "A. train multi-height profile-compatible surrogate"
        reason = "multi-height normalized oracle ordering passes the feasibility gate and improves over 0.008m single-height."
    elif closer_only and far_hurts:
        recommendation = "D. focus closer liftoff"
        reason = "0.004m improves ordering, while the farther 0.012m height does not support the combined objective."
    elif partial:
        recommendation = "B. expand/adjust multi-height pack"
        reason = "multi-height improves over 0.008m but does not pass the full oracle-ordering gate."
    else:
        recommendation = "C. multi-axis / multi-direction observation"
        reason = "multi-height lift-off alone does not materially improve profile quality ordering."

    rows = [
        {
            "option": "A",
            "label": "train multi-height profile-compatible surrogate",
            "selected": str(recommendation.startswith("A.")),
            "evidence": f"test_multi_height_ordering={multi_test:.6f}; delta_vs_0p008={multi_test - single_test:.6f}; test_corr={multi_corr:.6f}",
        },
        {
            "option": "B",
            "label": "expand/adjust multi-height pack",
            "selected": str(recommendation.startswith("B.")),
            "evidence": f"partial_improvement={partial}; test_multi_height_ordering={multi_test:.6f}",
        },
        {
            "option": "C",
            "label": "multi-axis / multi-direction observation",
            "selected": str(recommendation.startswith("C.")),
            "evidence": f"multi_height_gate_pass={pass_multi}; rect_multi={rect_multi:.6f}; rotated_multi={rot_multi:.6f}",
        },
        {
            "option": "D",
            "label": "focus closer liftoff",
            "selected": str(recommendation.startswith("D.")),
            "evidence": f"test_0p004={h004:.6f}; test_0p008={single_test:.6f}; test_0p012={h012:.6f}",
        },
        {
            "option": "E",
            "label": "stop current geometry/refinement route",
            "selected": "False",
            "evidence": "Reserved for no observation family showing any usable ordering signal.",
        },
    ]
    return rows, recommendation, reason


def write_summaries(
    args: argparse.Namespace,
    pack: np.lib.npyio.NpzFile,
    records: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    recommendation: str,
    reason: str,
) -> None:
    heights = [float(x) for x in pack["sensor_z_heights_m"]]
    metadata = parse_json_value(pack["metadata"], {})
    split_counts = Counter(row["split"] for row in records)
    type_counts = Counter(row["source_defect_type"] for row in records)
    variant_counts = Counter(row["variant_type"] for row in records)
    profile_rows = len(records)
    height_count = len(heights)
    total_obs = profile_rows * height_count

    def metric(split: str, method: str, height: str, key: str = "ordering_accuracy") -> float:
        return find_metric(audit_rows, "split", split, method, height, key)

    single_test = metric("test", "single_height_0p008", "0.008")
    h004_test = metric("test", "per_height", "0.004")
    h012_test = metric("test", "per_height", "0.012")
    multi_test = metric("test", "multi_height_trainstd_normalized", "all")
    multi_corr_test = metric("test", "multi_height_trainstd_normalized", "all", "residual_error_correlation")
    multi_mismatch_test = metric("test", "multi_height_trainstd_normalized", "all", "mismatch_rate")

    args.audit_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_summary.open("w", encoding="utf-8") as f:
        f.write("COMSOL multi-height profile oracle ordering audit summary\n\n")
        f.write("Stage 20.62 only audits COMSOL oracle residual ordering. No surrogate training, inverse training, or profile refinement was run.\n\n")
        f.write(f"pack: {args.pack}\n")
        f.write(f"metadata: {json.dumps(metadata, ensure_ascii=False)}\n")
        f.write(f"profile_rows: {profile_rows}\n")
        f.write(f"height_count: {height_count}\n")
        f.write(f"sensor_z_heights_m: {heights}\n")
        f.write(f"total_height_observations: {total_obs}\n")
        f.write(f"split_distribution: {dict(split_counts)}\n")
        f.write(f"source_defect_type_distribution: {dict(type_counts)}\n")
        f.write(f"variant_distribution: {dict(variant_counts)}\n\n")
        f.write("Ordering metrics by split:\n")
        for split in ("train", "val", "test"):
            f.write(
                f"- {split}: old_20.61_single={OLD_2061_SINGLE_ORACLE[split]:.4f}, "
                f"single_0.008={metric(split, 'single_height_0p008', '0.008'):.6f}, "
                f"h0.004={metric(split, 'per_height', '0.004'):.6f}, "
                f"h0.012={metric(split, 'per_height', '0.012'):.6f}, "
                f"multi_norm={metric(split, 'multi_height_trainstd_normalized', 'all'):.6f}, "
                f"multi_mismatch={metric(split, 'multi_height_trainstd_normalized', 'all', 'mismatch_rate'):.6f}, "
                f"multi_residual_error_corr={metric(split, 'multi_height_trainstd_normalized', 'all', 'residual_error_correlation'):.6f}\n"
            )
        f.write("\nRequired questions:\n")
        f.write(
            f"1. Closer height improves ordering? {'yes' if h004_test > single_test else 'no'} "
            f"(test h0.004={h004_test:.6f}, single_0.008={single_test:.6f}).\n"
        )
        f.write(
            f"2. Farther height improves robustness? {'yes' if h012_test > single_test else 'no'} "
            f"(test h0.012={h012_test:.6f}).\n"
        )
        f.write(
            f"3. Combined multi-height improves over single-height? {'yes' if multi_test > single_test else 'no'} "
            f"(test multi_norm={multi_test:.6f}, delta={multi_test - single_test:.6f}).\n"
        )
        f.write(
            f"4. Multi-height reduces mismatch? {'yes' if multi_mismatch_test < metric('test', 'single_height_0p008', '0.008', 'mismatch_rate') else 'no'} "
            f"(test multi mismatch={multi_mismatch_test:.6f}).\n"
        )
        f.write(
            f"5. Multi-height makes profile-forward surrogate worth training next? {'yes' if recommendation.startswith('A.') else 'no'}.\n"
        )
        f.write(f"6. Failure mode / route reason: {reason}\n\n")
        f.write("Success gate:\n")
        f.write("- test multi-height ordering > 0.65; improvement over 0.008m >= +0.10; test residual-error correlation > 0.20; improvement appears in rect and rotated groups.\n")
        f.write(f"gate_recommendation: {recommendation}\n")

    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.route_summary.open("w", encoding="utf-8") as f:
        f.write("COMSOL multi-height profile route decision summary\n\n")
        f.write("No surrogate or refinement was run in Stage 20.62. Decision is based only on real COMSOL oracle residual ordering.\n\n")
        f.write(f"recommendation: {recommendation}\n")
        f.write(f"reason: {reason}\n\n")
        f.write("Decision matrix:\n")
        for row in route_rows:
            f.write(f"- {row['option']}: {row['label']}; selected={row['selected']}; evidence={row['evidence']}\n")


def main() -> None:
    args = parse_args()
    pack = np.load(args.pack, allow_pickle=True)
    delta = np.asarray(pack["delta_bz_multi_height"], dtype=np.float64)
    bz_defect = np.asarray(pack["bz_defect_multi_height"], dtype=np.float64)
    bz_no_defect = np.asarray(pack["bz_no_defect_multi_height"], dtype=np.float64)
    if delta.shape != bz_defect.shape or delta.shape != bz_no_defect.shape:
        raise ValueError("Multi-height Bz arrays do not have matching shapes")
    if float(np.max(np.abs(delta - (bz_defect - bz_no_defect)))) > 1.0e-8:
        raise ValueError("delta_bz_multi_height does not match bz_defect - bz_no_defect")
    heights = np.asarray(pack["sensor_z_heights_m"], dtype=np.float64)
    if delta.shape[1] != len(heights):
        raise ValueError("Height axis does not match sensor_z_heights_m")
    if not np.any(np.isclose(heights, 0.008)):
        raise ValueError("0.008m reference height is missing")

    records = load_records(pack)
    scales = train_height_scales(delta, records)
    residual_rows = compute_residuals(delta, heights, records, scales)
    audit_rows, group_rows = summarize_residuals(residual_rows, heights)
    route_rows, recommendation, reason = build_route_decision(audit_rows)

    write_csv(args.audit_csv, audit_rows)
    write_csv(args.group_csv, group_rows)
    write_csv(args.route_csv, route_rows)
    write_summaries(args, pack, records, audit_rows, route_rows, recommendation, reason)
    print(f"Wrote audit: {args.audit_csv}")
    print(f"Wrote group summary: {args.group_csv}")
    print(f"Wrote route decision: {args.route_csv}")
    print(f"recommendation: {recommendation}")


if __name__ == "__main__":
    main()

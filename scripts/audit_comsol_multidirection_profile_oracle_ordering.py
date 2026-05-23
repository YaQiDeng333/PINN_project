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
    / "data/comsol_mfl/prepared/comsol_rect_rot_multidirection_profile_perturbation_forward_pack_v1.npz"
)
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multidirection_profile_oracle_ordering_audit_summary.txt"
DEFAULT_ROUTE_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multidirection_profile_route_decision_summary.txt"
DEFAULT_AUDIT_CSV = PROJECT_ROOT / "results/metrics/comsol_multidirection_profile_oracle_ordering_audit.csv"
DEFAULT_GROUP_CSV = PROJECT_ROOT / "results/metrics/comsol_multidirection_profile_oracle_ordering_group_summary.csv"
DEFAULT_ROUTE_CSV = PROJECT_ROOT / "results/metrics/comsol_multidirection_profile_route_decision_matrix.csv"

HISTORICAL_REFERENCES = {
    "20.61_single_height_bz_test": 0.5030,
    "20.62_multi_height_bz_test": 0.4545,
    "20.63_same_direction_multiaxis_test": 0.4505,
}
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 20.64 multi-direction oracle residual ordering.")
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
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
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


def residual_trainstd_norm(a: np.ndarray, b: np.ndarray, scales: np.ndarray, dirs: tuple[int, ...], axes: tuple[int, ...]) -> float:
    diff = np.asarray(a, dtype=np.float64)[np.ix_(dirs, axes)]
    ref = np.asarray(b, dtype=np.float64)[np.ix_(dirs, axes)]
    scale = np.maximum(scales[np.ix_(dirs, axes)], EPS)
    diff_norm = (diff - ref) / scale[:, :, None, None]
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


def train_direction_axis_scales(delta: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    train_true = [
        row["idx"]
        for row in records
        if row["split"] == "train" and row["variant_type"] == "true_reference"
    ]
    if not train_true:
        raise ValueError("No train true_reference rows found for scale computation")
    refs = delta[train_true]
    scales = np.std(refs, axis=(0, 3, 4))
    rms = np.sqrt(np.mean(refs**2, axis=(0, 3, 4)))
    return np.where(scales > EPS, scales, np.maximum(rms, EPS))


def residual_methods(direction_names: list[str], axis_names: list[str]) -> list[dict[str, Any]]:
    d = {name: idx for idx, name in enumerate(direction_names)}
    a = {name: idx for idx, name in enumerate(axis_names)}
    methods: list[dict[str, Any]] = []
    if "Bz" not in a or "direction_0" not in d or "direction_90" not in d:
        raise ValueError("Pack must include direction_0, direction_90, and Bz")
    all_axes = tuple(range(len(axis_names)))
    methods.append({"name": "direction_0_Bz_only", "dirs": (d["direction_0"],), "axes": (a["Bz"],), "normalized": False})
    if len(axis_names) > 1:
        methods.append(
            {
                "name": "direction_0_all_axis_trainstd_normalized",
                "dirs": (d["direction_0"],),
                "axes": all_axes,
                "normalized": True,
            }
        )
    methods.append({"name": "direction_90_Bz_only", "dirs": (d["direction_90"],), "axes": (a["Bz"],), "normalized": False})
    if len(axis_names) > 1:
        methods.append(
            {
                "name": "direction_90_all_axis_trainstd_normalized",
                "dirs": (d["direction_90"],),
                "axes": all_axes,
                "normalized": True,
            }
        )
    if "direction_45" in d:
        methods.append({"name": "direction_45_Bz_only", "dirs": (d["direction_45"],), "axes": (a["Bz"],), "normalized": False})
        if len(axis_names) > 1:
            methods.append(
                {
                    "name": "direction_45_all_axis_trainstd_normalized",
                    "dirs": (d["direction_45"],),
                    "axes": all_axes,
                    "normalized": True,
                }
            )
    methods.append(
        {
            "name": "multi_direction_Bz_trainstd_normalized",
            "dirs": tuple(range(len(direction_names))),
            "axes": (a["Bz"],),
            "normalized": True,
        }
    )
    if len(axis_names) > 1:
        methods.append(
            {
                "name": "multi_direction_all_axis_trainstd_normalized",
                "dirs": tuple(range(len(direction_names))),
                "axes": all_axes,
                "normalized": True,
            }
        )
    return methods


def compute_residuals(
    delta: np.ndarray,
    direction_names: list[str],
    axis_names: list[str],
    records: list[dict[str, Any]],
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_base[row["base_sample_id"]].append(row)
    methods = residual_methods(direction_names, axis_names)
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
            for method in methods:
                dirs = method["dirs"]
                axes = method["axes"]
                if method["normalized"]:
                    residual = residual_trainstd_norm(sample, reference, scales, dirs, axes)
                else:
                    residual = residual_nrmse(sample[np.ix_(dirs, axes)].reshape(-1), reference[np.ix_(dirs, axes)].reshape(-1))
                residual_rows.append(
                    {
                        **base_common,
                        "residual_method": method["name"],
                        "direction_combo": "+".join(direction_names[idx] for idx in dirs),
                        "axis_combo": "+".join(axis_names[idx] for idx in axes),
                        "direction_count": len(dirs),
                        "axis_count": len(axes),
                        "normalized": bool(method["normalized"]),
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
                if abs(qi - qj) > EPS and abs(ri - rj) > EPS:
                    ok += int((qi > qj) == (ri < rj))
                    total += 1
                rank_i = float(local[i]["expected_quality_rank"])
                rank_j = float(local[j]["expected_quality_rank"])
                if abs(rank_i - rank_j) > EPS and abs(ri - rj) > EPS:
                    expected_ok += int((rank_i < rank_j) == (ri < rj))
                    expected_total += 1
    residuals = [float(row["residual"]) for row in rows]
    errors = [float(row["quality_error"]) for row in rows]
    return {
        "base_count": float(len(by_base)),
        "row_count": float(len(rows)),
        "pair_count": float(total),
        "ordering_accuracy": ok / total if total else math.nan,
        "mismatch_rate": 1.0 - ok / total if total else math.nan,
        "expected_rank_ordering_accuracy": expected_ok / expected_total if expected_total else math.nan,
        "expected_rank_pair_count": float(expected_total),
        "residual_error_correlation": finite_corr(residuals, errors),
        "spearman_residual_error": spearman_corr(residuals, errors),
        "kendall_residual_error": kendall_tau(residuals, errors),
        "mean_residual": float(np.mean(residuals)) if residuals else math.nan,
        "mean_quality_error": float(np.mean(errors)) if errors else math.nan,
    }


def summarize(residual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in residual_rows:
        groups[("all", "all", row["residual_method"])].append(row)
        groups[("split", row["split"], row["residual_method"])].append(row)
        groups[("source_defect_type", row["source_defect_type"], row["residual_method"])].append(row)
    for (scope, group, method), local in sorted(groups.items()):
        metric = ordering_for_rows(local)
        exemplar = local[0]
        rows.append(
            {
                "scope": scope,
                "group": group,
                "residual_method": method,
                "direction_combo": exemplar["direction_combo"],
                "axis_combo": exemplar["axis_combo"],
                "direction_count": exemplar["direction_count"],
                "axis_count": exemplar["axis_count"],
                "normalized": exemplar["normalized"],
                **metric,
            }
        )
    return rows


def metric_lookup(summary_rows: list[dict[str, Any]], scope: str, group: str, method: str) -> dict[str, Any]:
    for row in summary_rows:
        if row["scope"] == scope and row["group"] == group and row["residual_method"] == method:
            return row
    raise KeyError((scope, group, method))


def build_route_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_default = metric_lookup(summary_rows, "split", "test", "direction_0_Bz_only")
    val_default = metric_lookup(summary_rows, "split", "val", "direction_0_Bz_only")
    multi_all_method = (
        "multi_direction_all_axis_trainstd_normalized"
        if any(row["residual_method"] == "multi_direction_all_axis_trainstd_normalized" for row in summary_rows)
        else "multi_direction_Bz_trainstd_normalized"
    )
    test_multi_all = metric_lookup(summary_rows, "split", "test", multi_all_method)
    val_multi_all = metric_lookup(summary_rows, "split", "val", multi_all_method)
    test_multi_bz = metric_lookup(summary_rows, "split", "test", "multi_direction_Bz_trainstd_normalized")
    val_multi_bz = metric_lookup(summary_rows, "split", "val", "multi_direction_Bz_trainstd_normalized")
    rect_multi = metric_lookup(summary_rows, "source_defect_type", "rectangular_notch", multi_all_method)
    rot_multi = metric_lookup(summary_rows, "source_defect_type", "rotated_rect", multi_all_method)
    delta_all = float(test_multi_all["ordering_accuracy"]) - float(test_default["ordering_accuracy"])
    delta_bz = float(test_multi_bz["ordering_accuracy"]) - float(test_default["ordering_accuracy"])
    val_delta_bz = float(val_multi_bz["ordering_accuracy"]) - float(val_default["ordering_accuracy"])
    gate_pass = (
        float(test_multi_all["ordering_accuracy"]) > 0.65
        and delta_all >= 0.10
        and float(test_multi_all["residual_error_correlation"]) > 0.20
        and float(test_multi_all["mismatch_rate"]) < float(test_default["mismatch_rate"])
        and float(rect_multi["ordering_accuracy"]) > 0.55
        and float(rot_multi["ordering_accuracy"]) > 0.55
    )
    partial_bz_only = (
        (not gate_pass)
        and delta_bz > 0.0
        and val_delta_bz > 0.0
        and float(test_multi_bz["residual_error_correlation"]) > 0.0
    )
    return [
        {
            "option": "A. train multi-direction profile-compatible surrogate",
            "selected": bool(gate_pass),
            "evidence": (
                f"test_multi_all={float(test_multi_all['ordering_accuracy']):.6f}; "
                f"delta_all_vs_samepack_direction0_bz={delta_all:.6f}; "
                f"test_corr={float(test_multi_all['residual_error_correlation']):.6f}"
            ),
        },
        {
            "option": "B. expand/tune multi-direction pack",
            "selected": bool(partial_bz_only),
            "evidence": (
                f"test_bz_delta={delta_bz:.6f}; val_bz_delta={val_delta_bz:.6f}; "
                f"test_bz_corr={float(test_multi_bz['residual_error_correlation']):.6f}; gate_pass={gate_pass}"
            ),
        },
        {
            "option": "C. true 3D profile / Piao-style route",
            "selected": bool((not gate_pass) and (not partial_bz_only)),
            "evidence": (
                f"same-pack gate failed; test_multi_all={float(test_multi_all['ordering_accuracy']):.6f}; "
                f"test_multi_bz={float(test_multi_bz['ordering_accuracy']):.6f}"
            ),
        },
        {
            "option": "D. pause geometry-forward route",
            "selected": False,
            "evidence": (
                f"fallback if true 3D/Piao-style is out of scope; "
                f"multi_all_mismatch={float(test_multi_all['mismatch_rate']):.6f}"
            ),
        },
    ]


def write_audit_summary(
    path: Path,
    pack_path: Path,
    metadata: dict[str, Any],
    direction_names: list[str],
    axis_names: list[str],
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    scales: np.ndarray,
) -> None:
    split_counts = Counter(row["split"] for row in records)
    type_counts = Counter(row["source_defect_type"] for row in records)
    variant_counts = Counter(row["variant_type"] for row in records)
    default_test = metric_lookup(summary_rows, "split", "test", "direction_0_Bz_only")
    dir90_test = metric_lookup(summary_rows, "split", "test", "direction_90_Bz_only")
    multi_all_method = (
        "multi_direction_all_axis_trainstd_normalized"
        if any(row["residual_method"] == "multi_direction_all_axis_trainstd_normalized" for row in summary_rows)
        else "multi_direction_Bz_trainstd_normalized"
    )
    multi_all_test = metric_lookup(summary_rows, "split", "test", multi_all_method)
    multi_bz_test = metric_lookup(summary_rows, "split", "test", "multi_direction_Bz_trainstd_normalized")
    delta_all_vs_default = float(multi_all_test["ordering_accuracy"]) - float(default_test["ordering_accuracy"])
    delta_bz_vs_default = float(multi_bz_test["ordering_accuracy"]) - float(default_test["ordering_accuracy"])
    gate_pass = bool(route_rows[0]["selected"])
    lines = [
        "COMSOL multi-direction profile oracle ordering audit summary",
        "",
        "Stage 20.64 only audits COMSOL oracle residual ordering. No surrogate training, inverse training, or profile refinement was run.",
        "",
        f"pack: {pack_path}",
        f"metadata: {json.dumps(metadata, sort_keys=True)}",
        f"profile_rows: {len(records)}",
        f"direction_count: {len(direction_names)}",
        f"direction_names: {direction_names}",
        f"axis_count: {len(axis_names)}",
        f"axis_names: {axis_names}",
        f"split_distribution: {dict(split_counts)}",
        f"source_defect_type_distribution: {dict(type_counts)}",
        f"variant_distribution: {dict(variant_counts)}",
        "",
        "Same-pack test comparison:",
        f"- direction_0_Bz_only_ordering: {float(default_test['ordering_accuracy']):.6f}",
        f"- direction_90_Bz_only_ordering: {float(dir90_test['ordering_accuracy']):.6f}",
        f"- multi_direction_Bz_trainstd_normalized_ordering: {float(multi_bz_test['ordering_accuracy']):.6f}",
        f"- {multi_all_method}_ordering: {float(multi_all_test['ordering_accuracy']):.6f}",
        f"- multi_direction_Bz_delta_vs_samepack_direction0_Bz: {delta_bz_vs_default:.6f}",
        f"- multi_direction_all_axis_delta_vs_samepack_direction0_Bz: {delta_all_vs_default:.6f}",
        f"- multi_direction_all_axis_mismatch_rate: {float(multi_all_test['mismatch_rate']):.6f}",
        f"- multi_direction_all_axis_residual_error_correlation: {float(multi_all_test['residual_error_correlation']):.6f}",
        "",
        "Historical references only:",
    ]
    for key, value in HISTORICAL_REFERENCES.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(
        [
            "",
            "Train-only direction-axis normalization scales:",
        ]
    )
    for d_idx, direction in enumerate(direction_names):
        for a_idx, axis in enumerate(axis_names):
            lines.append(f"- {direction}:{axis}: {float(scales[d_idx, a_idx]):.12g}")
    lines.extend(
        [
            "",
            "Answers:",
            f"1. Does orthogonal excitation improve ordering? {float(dir90_test['ordering_accuracy']) > float(default_test['ordering_accuracy'])}.",
            "2. Does direction_45 help if available? See CSV rows for direction_45 methods.",
            f"3. Does multi-direction Bz improve over default Bz? {delta_bz_vs_default > 0.0}; delta={delta_bz_vs_default:.6f}.",
            f"4. Does multi-direction all-axis improve over default Bz? {delta_all_vs_default > 0.0}; delta={delta_all_vs_default:.6f}.",
            f"5. Does multi-direction all-axis reduce mismatch? {float(multi_all_test['mismatch_rate']) < float(default_test['mismatch_rate'])}.",
            f"6. Does multi-direction make surrogate/refinement worth training next? {gate_pass}.",
            "7. Diagnosis: use same-pack direction_0 as the baseline; historical 20.61-20.63 values are not used for the gate.",
            "",
            "Success criteria:",
            "- test multi-direction normalized ordering > 0.65; improvement over same-pack direction_0 Bz >= +0.10; test residual-error correlation > 0.20; mismatch decreases; rect and rotated groups both improve.",
            f"gate_passed: {gate_pass}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_route_summary(path: Path, route_rows: list[dict[str, Any]]) -> None:
    selected = [row for row in route_rows if row["selected"]]
    recommendation = selected[0]["option"] if selected else "C. true 3D profile / Piao-style route"
    lines = [
        "COMSOL multi-direction profile route decision summary",
        "",
        f"recommendation: {recommendation}",
        "reason: decision is based on same-pack direction_0 Bz baseline vs multi-direction normalized residual.",
        "",
        "Decision matrix:",
    ]
    for row in route_rows:
        lines.append(f"- {row['option']}; selected={row['selected']}; evidence={row['evidence']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pack = np.load(args.pack, allow_pickle=True)
    delta = pack["delta_b_multidirection"].astype(np.float64)
    direction_names = pack["direction_names"].astype(str).tolist()
    axis_names = pack["axis_names"].astype(str).tolist()
    metadata = parse_json_value(pack["metadata"], {})
    records = load_records(pack)
    scales = train_direction_axis_scales(delta, records)
    residual_rows = compute_residuals(delta, direction_names, axis_names, records, scales)
    summary_rows = summarize(residual_rows)
    route_rows = build_route_rows(summary_rows)

    write_csv(args.audit_csv, summary_rows)
    write_csv(args.group_csv, [row for row in summary_rows if row["scope"] != "all"])
    write_csv(args.route_csv, route_rows)
    write_audit_summary(args.audit_summary, args.pack, metadata, direction_names, axis_names, records, summary_rows, route_rows, scales)
    write_route_summary(args.route_summary, route_rows)
    print(f"Wrote multi-direction audit to {args.audit_csv}")


if __name__ == "__main__":
    main()

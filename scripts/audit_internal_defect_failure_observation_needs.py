#!/usr/bin/env python
"""22.8 internal richer-observation failure-to-observation audit.

只读 22.0-22.7 结果和 v3_hardcase manifest，生成观测需求矩阵。
不运行 COMSOL，不训练，不读取/写入 data/NPZ。
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")
SUMMARY_DIR = ROOT / "results/summaries"
METRICS_DIR = ROOT / "results/metrics"

V3_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json"
SMOKE_SUMMARY_22_7 = SUMMARY_DIR / "internal_defect_inference_abstention_smoke_summary.txt"
ABSTAINED_CSV = METRICS_DIR / "internal_defect_inference_abstention_abstained_subset.csv"
FAILURE_CSV = METRICS_DIR / "internal_defect_inference_abstention_failure_cases.csv"
METRICS_CSV = METRICS_DIR / "internal_defect_inference_abstention_metrics.csv"

PREFLIGHT_SUMMARY = SUMMARY_DIR / "internal_richer_observation_preflight_summary.txt"
NEED_SUMMARY = SUMMARY_DIR / "internal_richer_observation_need_audit_summary.txt"
NEED_MATRIX = METRICS_DIR / "internal_richer_observation_need_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def git_forbidden_status() -> list[str]:
    paths = ["data", "checkpoints", "results/previews", "notes", "CURRENT_BASELINE.md", "scripts/visualize_current_baseline.py"]
    result = subprocess.run(["git", "status", "--short", "--", *paths], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        return [result.stderr.strip()]
    return [line for line in result.stdout.splitlines() if line.strip()]


def comsol_support_evidence() -> dict[str, Any]:
    pilot = COMSOL_ROOT / "scripts/generate_mfl_internal_defect_pilot_pack.py"
    multidirection = COMSOL_ROOT / "scripts/generate_mfl_multidirection_profile_perturbation_forward_pack.py"
    text = pilot.read_text(encoding="utf-8", errors="replace") if pilot.exists() else ""
    md_text = multidirection.read_text(encoding="utf-8", errors="replace") if multidirection.exists() else ""
    return {
        "pilot_exists": pilot.exists(),
        "supports_scan_line_y_m": "scan_line_y_m" in text and "sensor_points" in text,
        "supports_sensor_z_m": "sensor_z_m" in text,
        "supports_source_scale": "source_scale" in text,
        "has_prior_multidirection_je_protocol": "DIRECTION_JE_TARGET" in md_text and "config_for_direction" in md_text,
        "pilot_path": str(pilot),
        "multidirection_reference_path": str(multidirection),
    }


def load_required() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    missing = [p for p in [V3_MANIFEST, SMOKE_SUMMARY_22_7, ABSTAINED_CSV, FAILURE_CSV, METRICS_CSV] if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required 22.8 inputs: " + "; ".join(str(p) for p in missing))
    manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    return manifest, read_csv(ABSTAINED_CSV), read_csv(FAILURE_CSV), read_csv(METRICS_CSV)


def counts(rows: list[dict[str, str]], key: str) -> str:
    counter = Counter(row.get(key, "") for row in rows)
    return "; ".join(f"{k}={v}" for k, v in sorted(counter.items()))


def metric_scope(metrics: list[dict[str, str]], scope: str) -> dict[str, str]:
    for row in metrics:
        if row.get("metric_scope") == scope:
            return row
    raise RuntimeError(f"missing metric scope: {scope}")


def build_need_matrix(abstained: list[dict[str, str]], failures: list[dict[str, str]]) -> list[dict[str, Any]]:
    center_out = [r for r in failures if as_bool(r.get("center_outlier"))]
    burial_out = [r for r in failures if as_bool(r.get("burial_outlier"))]
    geometry = [r for r in failures if as_bool(r.get("geometry_branch_failure"))]
    shape_mis = [r for r in failures if str(r.get("true_shape_type")) != str(r.get("pred_shape_type"))]
    deep_plus = [r for r in failures if r.get("burial_depth_level") == "deep_plus"]
    elongated = [r for r in failures if r.get("aspect_bin") in {"elongated_x", "elongated_y"}]
    compact_large = [r for r in failures if r.get("aspect_bin") == "compact" and r.get("size_level") == "large"]
    return [
        {
            "failure_mode": "center_xyz_tail_failure",
            "evidence": f"center_outlier={len(center_out)}; abstained={len(abstained)}; shape_counts={counts(center_out, 'true_shape_type')}",
            "likely_observation_gap": "lateral field support too sparse; 3 y-lines undersample center_y and lateral extent",
            "primary_observation_candidate": "R1_more_y_lines",
            "secondary_observation_candidate": "R3_multi_scan_direction",
            "rationale": "5/9 y-lines directly add lateral samples without changing excitation; multi-direction is second-stage if lateral ambiguity remains.",
            "priority": 1,
            "suitable_for_22_9": True,
        },
        {
            "failure_mode": "burial_depth_tail_failure",
            "evidence": f"burial_outlier={len(burial_out)}; deep_plus_failures={len(deep_plus)}; burial_counts={counts(burial_out, 'burial_depth_level')}",
            "likely_observation_gap": "single liftoff cannot separate attenuation due to burial depth from size/amplitude",
            "primary_observation_candidate": "R2_multi_liftoff",
            "secondary_observation_candidate": "R1_more_y_lines",
            "rationale": "paired liftoff response is the most direct diagnostic for depth attenuation; 5 y-lines keeps lateral context stable.",
            "priority": 1,
            "suitable_for_22_9": True,
        },
        {
            "failure_mode": "geometry_branch_failure",
            "evidence": f"geometry_branch_failure={len(geometry)}; shape_misclassified={len(shape_mis)}; shape_pairs={pair_counts(shape_mis)}",
            "likely_observation_gap": "shape branch is underconstrained by one scan orientation and sparse y-lines",
            "primary_observation_candidate": "R1_more_y_lines",
            "secondary_observation_candidate": "R3_multi_scan_direction",
            "rationale": "R1 tests whether denser lateral sampling resolves cuboid/ellipsoid confusion before paying multi-direction cost.",
            "priority": 2,
            "suitable_for_22_9": True,
        },
        {
            "failure_mode": "elongated_aspect_confusion",
            "evidence": f"elongated_failures={len(elongated)}; aspect_counts={counts(failures, 'aspect_bin')}",
            "likely_observation_gap": "single x-direction scan sees elongated_y and compact shapes with partial symmetry",
            "primary_observation_candidate": "R3_multi_scan_direction",
            "secondary_observation_candidate": "R1_more_y_lines",
            "rationale": "R3 is physically aligned with aspect ambiguity, but requires new axis-order/loader protocol; R1 is first-round proxy.",
            "priority": 3,
            "suitable_for_22_9": False,
        },
        {
            "failure_mode": "compact_large_high_risk",
            "evidence": f"compact_large_failures={len(compact_large)}; size_counts={counts(failures, 'size_level')}",
            "likely_observation_gap": "large compact internal cavities can trade off center, burial, and lateral extent",
            "primary_observation_candidate": "R1_more_y_lines",
            "secondary_observation_candidate": "R2_multi_liftoff",
            "rationale": "paired y-line and liftoff observations are needed to separate lateral extent from depth attenuation.",
            "priority": 2,
            "suitable_for_22_9": True,
        },
        {
            "failure_mode": "magnetization_direction_uncertainty",
            "evidence": "no current v3_hardcase evidence proves magnetization direction is the limiting factor",
            "likely_observation_gap": "possible shape discriminability limit, but not first-order evidence yet",
            "primary_observation_candidate": "R4_multi_magnetization_direction",
            "secondary_observation_candidate": "R3_multi_scan_direction",
            "rationale": "high COMSOL and experiment cost; defer until R1/R2/R3 evidence says field direction is limiting.",
            "priority": 5,
            "suitable_for_22_9": False,
        },
    ]


def pair_counts(rows: list[dict[str, str]]) -> str:
    counter = Counter(f"{r.get('true_shape_type')}->{r.get('pred_shape_type')}" for r in rows)
    return "; ".join(f"{k}={v}" for k, v in sorted(counter.items())) or "none"


def write_preflight(manifest: dict[str, Any], metrics: list[dict[str, str]], support: dict[str, Any]) -> None:
    gate = metric_scope(metrics, "abstention_gate_test")
    forbidden = git_forbidden_status()
    text = [
        "22.8 internal richer-observation feasibility preflight",
        "",
        f"- dataset_id: {manifest.get('dataset_id')}",
        f"- v3_hardcase rows: {manifest.get('n_samples')}; split_counts: {manifest.get('split_counts')}",
        f"- 22.7 coverage retained: {safe_float(gate.get('coverage_retained')):.3f}; high-risk count: {gate.get('high_risk_count')}; catastrophic/geometry recall: {gate.get('catastrophic_failure_recall')}/{gate.get('geometry_branch_failure_recall')}",
        f"- COMSOL internal pilot generator: {support['pilot_path']}",
        f"- supports scan_line_y_m: {support['supports_scan_line_y_m']}; supports sensor_z_m: {support['supports_sensor_z_m']}; prior multi-direction Je protocol: {support['has_prior_multidirection_je_protocol']}",
        f"- forbidden artifact status: {'clean' if not forbidden else '; '.join(forbidden)}",
        "- 本阶段为 plan-only：未运行 COMSOL，未训练，未生成或修改 data/NPZ，未更新 CURRENT_BASELINE.md。",
        "",
        "结论：R1_more_y_lines 与 R2_multi_liftoff 可作为 22.9 第一轮 diagnostic pack；R3/R4 只记录为后续候选。",
    ]
    write_text(PREFLIGHT_SUMMARY, "\n".join(text) + "\n")


def write_need_summary(matrix: list[dict[str, Any]], abstained: list[dict[str, str]], failures: list[dict[str, str]]) -> None:
    lines = [
        "22.8 internal richer-observation need audit",
        "",
        f"- abstained samples: {len(abstained)}; failure cases: {len(failures)}.",
        f"- failure shape counts: {counts(failures, 'true_shape_type')}.",
        f"- failure burial counts: {counts(failures, 'burial_depth_level')}.",
        f"- failure size counts: {counts(failures, 'size_level')}.",
        f"- failure aspect counts: {counts(failures, 'aspect_bin')}.",
        "",
        "核心映射：center/lateral tail -> R1_more_y_lines；burial/size tail -> R2_multi_liftoff；geometry/aspect confusion -> R1 first, R3 second.",
        "R4 multi-magnetization 当前没有足够 failure evidence 支撑第一轮执行。",
    ]
    for row in matrix:
        lines.append(f"- {row['failure_mode']}: primary={row['primary_observation_candidate']}, secondary={row['secondary_observation_candidate']}, priority={row['priority']}")
    write_text(NEED_SUMMARY, "\n".join(lines) + "\n")


def main() -> int:
    manifest, abstained, failures, metrics = load_required()
    support = comsol_support_evidence()
    matrix = build_need_matrix(abstained, failures)
    write_preflight(manifest, metrics, support)
    write_need_summary(matrix, abstained, failures)
    write_csv(
        NEED_MATRIX,
        matrix,
        [
            "failure_mode",
            "evidence",
            "likely_observation_gap",
            "primary_observation_candidate",
            "secondary_observation_candidate",
            "rationale",
            "priority",
            "suitable_for_22_9",
        ],
    )
    print(json.dumps({"need_rows": len(matrix), "summary": str(NEED_SUMMARY)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

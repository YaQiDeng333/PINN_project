#!/usr/bin/env python
"""23.2 internal failure-to-scan-direction audit.

This is a plan-only analysis artifact. It reads existing summaries, metrics,
and manifests, then writes 23.2 preflight and scan-direction need outputs.
It does not run COMSOL, train models, or create/modify data or NPZ files.
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
MANIFEST_DIR = ROOT / "results/manifests"

RICHER_MANIFEST = MANIFEST_DIR / "comsol_internal_defect_richer_observation_pack_v1.manifest.json"
V3_MANIFEST = MANIFEST_DIR / "comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json"
TRAIN_SUMMARY_23_1 = SUMMARY_DIR / "internal_richer_observation_training_summary.txt"
TRAIN_ROUTE_23_1 = SUMMARY_DIR / "internal_richer_observation_training_route_decision_summary.txt"
TRAIN_DECISION_23_1 = METRICS_DIR / "internal_richer_observation_training_decision_matrix.csv"
RICHER_PLAN_22_9 = METRICS_DIR / "internal_richer_observation_diagnostic_pack_plan.csv"
B2_FAILURE_22_0 = METRICS_DIR / "internal_defect_b2_failure_cases.csv"
ABSTENTION_FAILURE_22_7 = METRICS_DIR / "internal_defect_inference_abstention_failure_cases.csv"
PRECHECK_OUTPUT = SUMMARY_DIR / "internal_multi_scan_direction_plan_preflight_summary.txt"
NEED_SUMMARY = SUMMARY_DIR / "internal_multi_scan_direction_need_audit_summary.txt"
NEED_MATRIX = METRICS_DIR / "internal_multi_scan_direction_need_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
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


def parse_key_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = line.strip().lstrip("-").strip()
        if ":" in clean:
            key, value = clean.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def require_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def git_status_nonignored_forbidden() -> list[str]:
    paths = ["data", "checkpoints", "results/previews", "notes", "CURRENT_BASELINE.md", "scripts/visualize_current_baseline.py"]
    result = subprocess.run(["git", "status", "--short", "--", *paths], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        return [result.stderr.strip()]
    return [line for line in result.stdout.splitlines() if line.strip()]


def comsol_scan_support() -> dict[str, Any]:
    generator = COMSOL_ROOT / "scripts/generate_mfl_internal_richer_observation_diagnostic_pack.py"
    pilot = COMSOL_ROOT / "scripts/generate_mfl_internal_defect_pilot_pack.py"
    text = generator.read_text(encoding="utf-8", errors="replace") if generator.exists() else ""
    pilot_text = pilot.read_text(encoding="utf-8", errors="replace") if pilot.exists() else ""
    return {
        "generator_exists": generator.exists(),
        "pilot_exists": pilot.exists(),
        "scan_direction_metadata_present": "scan_direction" in text,
        "uses_existing_x_direction_sensor_points": "pilot.solve_no_defect" in text and "pilot.solve_defect" in text,
        "true_direction_aware_points_present": "sensor_points_by_direction" in text or "direction_aware" in text,
        "pilot_sensor_points_fixed_x_path": "points = [[float(x), float(y), float(sensor_z_m)]" in pilot_text,
        "generator_path": str(generator),
    }


def count_by(rows: list[dict[str, str]], field: str) -> str:
    counter = Counter(row.get(field, "") for row in rows)
    return "; ".join(f"{key}={value}" for key, value in sorted(counter.items())) or "none"


def pair_counts(rows: list[dict[str, str]]) -> str:
    counter = Counter(f"{row.get('true_shape_type')}->{row.get('pred_shape_type')}" for row in rows if row.get("true_shape_type") != row.get("pred_shape_type"))
    return "; ".join(f"{key}={value}" for key, value in sorted(counter.items())) or "none"


def build_need_matrix(train_summary: dict[str, str], train_route: dict[str, str], b2_failures: list[dict[str, str]], abstention_failures: list[dict[str, str]], richer_plan: list[dict[str, str]]) -> list[dict[str, Any]]:
    b2_geometry = [row for row in b2_failures if as_bool(row.get("is_geometry_branch_failure"))]
    abstention_geometry = [row for row in abstention_failures if as_bool(row.get("geometry_branch_failure"))]
    center_failures = [row for row in abstention_failures if as_bool(row.get("center_outlier"))]
    burial_failures = [row for row in abstention_failures if as_bool(row.get("burial_outlier"))]
    elongated_failures = [row for row in abstention_failures if row.get("aspect_bin") in {"elongated_x", "elongated_y"}]
    compact_large = [row for row in abstention_failures if row.get("aspect_bin") == "compact" and row.get("size_level") == "large"]
    plan_bases = {row["base_group_id"]: row for row in richer_plan if row.get("observation_variant") == "R0_3line_z0p008"}
    return [
        {
            "failure_mode": "23_1_r1_r2_gate_failure",
            "evidence": f"shape_F1={train_route.get('test_shape_macro_f1')}; catastrophic={train_route.get('test_catastrophic_failure_count')}/5; geometry={train_route.get('test_geometry_branch_failure_count')}/5; selected_config={train_route.get('selected_observation_config')}",
            "single_direction_hypothesis": "更多 y-lines 和 multi-liftoff 仍未提供足够方向性形状证据；单一扫描方向可能把 cuboid/ellipsoid 与 elongated aspect 混为相似响应。",
            "expected_dual_direction_benefit": "y_scan 提供正交投影，能直接检查 shape branch 是否缺方向性观测。",
            "recommended_candidate": "D1_and_D2",
            "priority": 1,
            "plan_action": "进入 23.2b，只补 y_scan 5-line/9-line，不重复 x_scan。",
        },
        {
            "failure_mode": "cuboid_ellipsoid_geometry_branch",
            "evidence": f"22.0 geometry pairs={pair_counts(b2_geometry)}; 22.7 geometry pairs={pair_counts(abstention_geometry)}",
            "single_direction_hypothesis": "单一 x_scan 对沿扫描方向和横向的几何投影不对称，可能让 cuboid edge 与 ellipsoid smooth response 在某些姿态下不可分。",
            "expected_dual_direction_benefit": "x_scan/y_scan 配对能比较两个正交剖面的边缘和宽度响应，降低 cuboid/ellipsoid branch confusion。",
            "recommended_candidate": "D1_dual_direction_5line_z0p008",
            "priority": 1,
            "plan_action": "优先生成 y_scan 5-line，与既有 R1_5line_z0p008 组装。",
        },
        {
            "failure_mode": "elongated_aspect_confusion",
            "evidence": f"elongated_failures={len(elongated_failures)}; aspect_counts={count_by(abstention_failures, 'aspect_bin')}; richer_base_aspect_counts={count_by(list(plan_bases.values()), 'aspect_bin')}",
            "single_direction_hypothesis": "单一方向对 elongated_x / elongated_y 的对称性不足，尤其在 compact/elongated_y 高风险样本上容易混淆。",
            "expected_dual_direction_benefit": "9-line y_scan 与 9-line x_scan 配对后能显式观察长轴方向差异。",
            "recommended_candidate": "D2_dual_direction_9line_z0p008",
            "priority": 1,
            "plan_action": "生成 y_scan 9-line，与既有 R1_9line_z0p008 组装。",
        },
        {
            "failure_mode": "center_xyz_tail",
            "evidence": f"23.1 center_p95={train_route.get('test_center_p95_mm')} mm; 22.7 center_outlier_count={len(center_failures)}; center_region_counts={count_by(center_failures, 'center_region')}",
            "single_direction_hypothesis": "单一方向下 center_x/center_y 的误差可互相补偿，尤其横向定位只靠线间变化。",
            "expected_dual_direction_benefit": "正交扫描把 x 和 y 两个方向都变成 path coordinate，可减少中心偏移的方向性盲区。",
            "recommended_candidate": "D1_and_D2",
            "priority": 2,
            "plan_action": "在 23.2b plan 中保留 center_region / failure_tags，便于 23.3 分组评价。",
        },
        {
            "failure_mode": "burial_depth_center_tradeoff",
            "evidence": f"23.1 burial_p95={train_route.get('test_burial_p95_mm')} mm; 22.7 burial_outlier_count={len(burial_failures)}; burial_counts={count_by(burial_failures, 'burial_depth_level')}",
            "single_direction_hypothesis": "burial/size 混淆在 R2 后仍存在，说明一部分 depth error 可能来自形状/中心错位传导，而非 liftoff 信息不足本身。",
            "expected_dual_direction_benefit": "若 y_scan 改善 shape/center，则 burial tail 应同步下降；否则再考虑 D3 multi-liftoff + dual-direction。",
            "recommended_candidate": "D1_then_D3_if_needed",
            "priority": 3,
            "plan_action": "23.2b 不叠加 multi-liftoff；D3 留作二阶段。",
        },
        {
            "failure_mode": "multi_magnetization_cost",
            "evidence": "22.0-23.1 failure evidence 指向 scan direction 和 shape/center coupling，尚未证明 magnetization direction 是主瓶颈。",
            "single_direction_hypothesis": "磁化方向可能提供额外形状判别，但成本高且会引入新 source 协议变量。",
            "expected_dual_direction_benefit": "先用 D1/D2 验证空间观测是否足够，避免过早引入 R4。",
            "recommended_candidate": "D4_defer",
            "priority": 5,
            "plan_action": "不进入 23.2b。",
        },
    ]


def write_preflight(missing: list[str], richer_manifest: dict[str, Any], v3_manifest: dict[str, Any], support: dict[str, Any], train_route: dict[str, str]) -> None:
    forbidden = git_status_nonignored_forbidden()
    lines = [
        "# 23.2 internal multi-scan-direction plan preflight",
        "",
        "## 输入证据",
        "",
        f"- missing_inputs: {missing if missing else 'none'}",
        f"- richer_observation_dataset: {richer_manifest.get('dataset_id')} rows={richer_manifest.get('n_samples')} complete_base_count={richer_manifest.get('complete_base_count')}",
        f"- v3_hardcase_dataset: {v3_manifest.get('dataset_id')} rows={v3_manifest.get('n_samples')}",
        f"- 23.1 selected_config: {train_route.get('selected_observation_config')}",
        f"- 23.1 selected_model: {train_route.get('selected_model')}",
        f"- 23.1 test total/shapeF1/catastrophic/geometry: {train_route.get('test_total_normalized_mae')} / {train_route.get('test_shape_macro_f1')} / {train_route.get('test_catastrophic_failure_count')} / {train_route.get('test_geometry_branch_failure_count')}",
        "",
        "## COMSOL generator scan metadata",
        "",
        f"- generator: {support['generator_path']}",
        f"- scan_direction_metadata_present: {support['scan_direction_metadata_present']}",
        f"- uses_existing_x_direction_sensor_points: {support['uses_existing_x_direction_sensor_points']}",
        f"- true_direction_aware_points_present: {support['true_direction_aware_points_present']}",
        f"- pilot_sensor_points_fixed_x_path: {support['pilot_sensor_points_fixed_x_path']}",
        "- conclusion: 23.2b 必须新增真正的 direction-aware sensor point builder，不能只写 y_scan metadata。",
        "",
        "## 安全状态",
        "",
        "- no_comsol_run: true",
        "- no_training: true",
        "- no_data_or_npz_mutation: true",
        "- current_baseline_update: false",
        f"- forbidden_artifact_status_nonignored_tracked_paths: {'clean' if not forbidden else '; '.join(forbidden)}",
        "- forbidden_artifact_scope_note: this check uses git status on non-ignored tracked/untracked paths; ignored historical data/checkpoint/preview artifacts are not claimed clean by this line.",
    ]
    write_text(PRECHECK_OUTPUT, "\n".join(lines) + "\n")


def write_need_summary(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 23.2 failure-to-scan-direction need audit",
        "",
        "23.1 的 R1/R2 输入扩展没有形成 stable inference candidate；这说明 y-line/liftoff 不是唯一瓶颈。下一步应验证正交扫描方向是否补足 shape branch 和 center 定位信息。",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['failure_mode']}: 推荐 `{row['recommended_candidate']}`；依据：{row['evidence']}")
    lines.extend(
        [
            "",
            "结论：23.2b 第一轮只生成 y_scan 5-line / 9-line top-up，并与 22.9 既有 x_scan 配对；训练暂缓到 23.3。",
        ]
    )
    write_text(NEED_SUMMARY, "\n".join(lines) + "\n")


def main() -> int:
    required = [
        RICHER_MANIFEST,
        V3_MANIFEST,
        TRAIN_SUMMARY_23_1,
        TRAIN_ROUTE_23_1,
        TRAIN_DECISION_23_1,
        RICHER_PLAN_22_9,
        B2_FAILURE_22_0,
        ABSTENTION_FAILURE_22_7,
    ]
    missing = require_paths(required)
    if missing:
        raise FileNotFoundError("missing required 23.2 planning inputs: " + "; ".join(missing))
    richer_manifest = json.loads(RICHER_MANIFEST.read_text(encoding="utf-8"))
    v3_manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    train_summary = parse_key_summary(TRAIN_SUMMARY_23_1)
    train_route = parse_key_summary(TRAIN_ROUTE_23_1)
    support = comsol_scan_support()
    b2_failures = read_csv(B2_FAILURE_22_0)
    abstention_failures = read_csv(ABSTENTION_FAILURE_22_7)
    richer_plan = read_csv(RICHER_PLAN_22_9)
    rows = build_need_matrix(train_summary, train_route, b2_failures, abstention_failures, richer_plan)
    write_preflight(missing, richer_manifest, v3_manifest, support, train_route)
    write_need_summary(rows)
    write_csv(
        NEED_MATRIX,
        rows,
        [
            "failure_mode",
            "evidence",
            "single_direction_hypothesis",
            "expected_dual_direction_benefit",
            "recommended_candidate",
            "priority",
            "plan_action",
        ],
    )
    print(json.dumps({"need_rows": len(rows), "preflight": str(PRECHECK_OUTPUT), "matrix": str(NEED_MATRIX)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

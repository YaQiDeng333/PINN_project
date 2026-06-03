from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results/metrics/25_18_raster_target_route_reset_metrics.json"
SUMMARY_PATH = ROOT / "results/summaries/25_18_raster_target_route_reset_summary.md"
MANIFEST_PATH = ROOT / "results/manifests/25_18_raster_target_route_reset_manifest.json"

FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]

EVIDENCE_FILES = [
    ("25.10", "training_gate", ROOT / "results/metrics/25_10_component_set_training_gate_metrics.json"),
    ("25.10b", "failure_audit", ROOT / "results/metrics/25_10b_component_set_failure_audit.json"),
    ("25.11", "mask_depth_rebalance_training", ROOT / "results/metrics/25_11_mask_depth_loss_rebalance_training_metrics.json"),
    ("25.11b", "merge_collapse_audit", ROOT / "results/metrics/25_11b_component_set_merge_collapse_audit.json"),
    ("25.12", "component_separation_rebalance_training", ROOT / "results/metrics/25_12_component_separation_rebalance_training_metrics.json"),
    ("25.12b", "target_redesign_audit", ROOT / "results/metrics/25_12b_component_raster_depth_target_redesign.json"),
    ("25.13", "target_v2_training_gate", ROOT / "results/metrics/25_13_target_v2_training_gate_metrics.json"),
    ("25.13b", "generator_label_schema_audit", ROOT / "results/metrics/25_13b_generator_label_schema_audit.json"),
    ("25.14", "label_v3_derivation_validator", ROOT / "results/metrics/25_14_label_v3_derivation_validator.json"),
    ("25.15", "label_v3_training_gate", ROOT / "results/metrics/25_15_label_v3_training_gate_metrics.json"),
    ("25.15b", "label_v3_failure_audit", ROOT / "results/metrics/25_15b_label_v3_failure_audit.json"),
    ("25.16", "label_v3b_derivation_validator", ROOT / "results/metrics/25_16_label_v3b_derivation_validator.json"),
    ("25.17", "label_v3b_training_gate", ROOT / "results/metrics/25_17_label_v3b_training_gate_metrics.json"),
]

TEST_METRIC_KEYS = [
    "component_recall",
    "missed_rate",
    "extra_rate",
    "merged_rate",
    "component_mask_dice_mean",
    "union_mask_dice_mean",
    "depth_grid_rmse_m_mean",
]


def strict_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strict_value(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(strict_value(payload), handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def git_value(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def first_text(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def extract_test_metrics(data: dict[str, Any]) -> dict[str, Any]:
    test = data.get("metrics_by_split", {}).get("test", {})
    return {key: test.get(key) for key in TEST_METRIC_KEYS if key in test}


def evidence_record(stage: str, role: str, path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage,
        "role": role,
        "path": str(path.relative_to(ROOT)),
        "present": path.exists(),
    }
    if not path.exists():
        return record
    data = read_json(path)
    record.update(
        {
            "gate_decision": data.get("gate_decision"),
            "acceptance_decision": data.get("acceptance_decision") or data.get("target_redesign_acceptance_decision"),
            "route_decision": data.get("route_decision"),
            "test_metrics": extract_test_metrics(data),
            "main_conclusion": first_text(
                data,
                [
                    "audit_main_conclusion",
                    "audit_conclusion",
                    "target_redesign_main_conclusion",
                    "label_v3_derivation_main_conclusion",
                    "label_v3b_derivation_main_conclusion",
                ],
            ),
        }
    )
    return record


def stage_record(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    for record in records:
        if record["stage"] == stage:
            return record
    return {}


def route_stop_supported(records: list[dict[str, Any]]) -> bool:
    s13 = stage_record(records, "25.13")
    s15 = stage_record(records, "25.15")
    s17 = stage_record(records, "25.17")
    required_present = all(stage_record(records, stage).get("present") for stage in ["25.10", "25.13", "25.15", "25.17"])
    target_v2_failed = s13.get("gate_decision") == "FAIL"
    target_v3_failed = s15.get("gate_decision") == "FAIL"
    v3b_merged = s17.get("test_metrics", {}).get("merged_rate") == 1.0
    return bool(required_present and target_v2_failed and target_v3_failed and v3b_merged)


def build_payload() -> dict[str, Any]:
    records = [evidence_record(stage, role, path) for stage, role, path in EVIDENCE_FILES]
    missing = [record["path"] for record in records if not record["present"]]
    evidence_sufficient = route_stop_supported(records)
    decision = "STOP_RASTER_TARGET_MAINLINE" if evidence_sufficient else "NEEDS_ROUTE_EVIDENCE_AUDIT"
    route = (
        "A. enter 25.19 geometry-primary component-set design + label derivation plan; no training"
        if evidence_sufficient
        else "B. run route evidence audit before any training"
    )

    stopped_routes = [
        {
            "route": "label-v2 target-v2 training route",
            "decision": "STOP",
            "evidence": "25.13 target-v2 training gate FAIL: ownership cleanup removed duplicate/overlap conflict but produced near-empty mask collapse.",
        },
        {
            "route": "label-v3 soft support training route",
            "decision": "STOP",
            "evidence": "25.15 label-v3 training gate FAIL: soft support relieved sparsity but produced union-like merged collapse.",
        },
        {
            "route": "label-v3b hard-core/halo/SDF raster-supervision route",
            "decision": "STOP",
            "evidence": "25.17 label-v3b training gate PARTIAL: near-empty was partly relieved, but merged_rate remained 1.000000.",
        },
        {
            "route": "loss rebalance / label-v4 raster-target tuning as the mainline",
            "decision": "STOP",
            "evidence": "25.11/25.12 rebalance attempts either caused merge collapse or failed; the route now points away from raster-target main supervision.",
        },
    ]

    preserved_routes = [
        "multi-pit component-set direction",
        "K=3 slot representation",
        "component geometry prediction",
        "future forward consistency",
        "raw labels and COMSOL top-up dataset as evidence/source data",
    ]

    failure_attribution = [
        {
            "rank": 1,
            "cause": "per-component raster ownership main supervision mismatches MFL observability",
            "evidence": "Exclusive ownership collapses toward near-empty masks, while softened support collapses toward union-like merged masks.",
        },
        {
            "rank": 2,
            "cause": "touching / overlap / three-component cases amplify component identity ambiguity",
            "evidence": "25.10-25.17 audits repeatedly isolate touching, overlap, and three-component subsets as unstable.",
        },
        {
            "rank": 3,
            "cause": "loader/loss/matching risk remains worth auditing but is not a reason to continue raster-target tuning",
            "evidence": "25.17 requires a failure audit of hard-core/halo/SDF/depth-valid-region usage, but the route-level pattern spans v2, v3, and v3b.",
        },
    ]

    non_primary_causes = [
        "global COMSOL dataset failure",
        "CURRENT_BASELINE.md or baseline-transition issue",
        "training epochs alone",
        "model capacity alone",
    ]

    geometry_primary_design = {
        "slot_primary_outputs": [
            "existence_prob",
            "center_x_m",
            "center_y_m",
            "L_m",
            "W_m",
            "D_m",
            "rotation_angle",
            "shape_family",
            "compact_shape_parameters",
        ],
        "derived_outputs": [
            "derived_component_mask",
            "derived_union_mask",
            "derived_component_depth",
            "derived_union_depth",
        ],
        "supervision_boundary": "per-component raster targets may remain auxiliary diagnostics or weak supervision, but are no longer the main loss.",
        "matching_priority": ["existence", "center", "L/W/D", "rotation", "optional derived union consistency"],
        "forward_consistency": "geometry slots -> derived profile/mask/depth -> lightweight forward surrogate or feature-space residual -> Bx/By/Bz residual; COMSOL is not placed inside the training loop.",
    }

    roadmap = [
        {"stage": "25.19", "route": "geometry-primary component-set design + label derivation plan", "training": False},
        {"stage": "25.20", "route": "two-component separated/close geometry-primary training gate", "training": True, "scope": "most identifiable subset only"},
        {"stage": "25.21", "route": "geometry-derived mask/depth evaluator + forward-consistency surrogate plan", "training": False},
        {"stage": "25.22", "route": "topology-aware expansion for touching/overlap/three-component", "training": "gate-dependent"},
    ]

    return {
        "stage": "25.18",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_reset_main_conclusion": "Stop the per-component raster-target mainline and move to geometry-primary component-set design.",
        "route_stop_acceptance_decision": decision,
        "route_decision": route,
        "evidence_sufficient": evidence_sufficient,
        "missing_evidence_files": missing,
        "evidence_records": records,
        "stopped_routes": stopped_routes,
        "preserved_routes": preserved_routes,
        "failure_attribution_ranked": failure_attribution,
        "non_primary_causes": non_primary_causes,
        "geometry_primary_component_set_design": geometry_primary_design,
        "roadmap": roadmap,
        "boundary": {
            "training_run": False,
            "loss_tuning": False,
            "model_capacity_expanded": False,
            "comsol_run": False,
            "data_npz_modified": False,
            "current_baseline_updated": False,
            "baseline_transition": False,
            "continues_label_v4_or_loss_v5_raster_training": False,
        },
        "git": {
            "branch": git_value(["branch", "--show-current"]),
            "head_before_commit": git_value(["rev-parse", "HEAD"]),
            "protected_path_diff_before_write": git_value(["diff", "--name-only", "--", *FORBIDDEN_DIFF_PATHS]),
        },
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["evidence_records"]
    lines = [
        "# 25.18 Raster-Target Route Reset",
        "",
        f"- route reset decision: `{payload['route_stop_acceptance_decision']}`",
        f"- next route: `{payload['route_decision']}`",
        f"- evidence sufficient: `{payload['evidence_sufficient']}`",
        f"- missing evidence files: `{len(payload['missing_evidence_files'])}`",
        "",
        "## Main Conclusion",
        "",
        payload["route_reset_main_conclusion"],
        "",
        "The failure pattern is route-level, not a single-run bug: target-v2 becomes near-empty after ownership cleanup, label-v3 becomes union-like after soft support, and label-v3b still has merged_rate `1.000000` after hard-core/halo/SDF supervision.",
        "",
        "## Evidence Snapshot",
        "",
        "| stage | decision | recall | missed | extra | merged | component Dice | union Dice | depth RMSE m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in ["25.10", "25.13", "25.15", "25.17"]:
        record = stage_record(records, stage)
        metrics = record.get("test_metrics", {})
        lines.append(
            "| {stage} | {decision} | {recall} | {missed} | {extra} | {merged} | {component_dice} | {union_dice} | {depth_rmse} |".format(
                stage=stage,
                decision=record.get("gate_decision") or record.get("acceptance_decision"),
                recall=fmt(metrics.get("component_recall")),
                missed=fmt(metrics.get("missed_rate")),
                extra=fmt(metrics.get("extra_rate")),
                merged=fmt(metrics.get("merged_rate")),
                component_dice=fmt(metrics.get("component_mask_dice_mean")),
                union_dice=fmt(metrics.get("union_mask_dice_mean")),
                depth_rmse=fmt(metrics.get("depth_grid_rmse_m_mean"), digits=9),
            )
        )
    lines.extend(
        [
            "",
            "## Stop Routes",
            "",
        ]
    )
    for route in payload["stopped_routes"]:
        lines.append(f"- `{route['route']}`: `{route['decision']}`. {route['evidence']}")
    lines.extend(["", "## Preserve Routes", ""])
    for route in payload["preserved_routes"]:
        lines.append(f"- {route}")
    lines.extend(["", "## Geometry-Primary Route", ""])
    design = payload["geometry_primary_component_set_design"]
    lines.append("- slot primary outputs: `" + "`, `".join(design["slot_primary_outputs"]) + "`")
    lines.append("- derived outputs: `" + "`, `".join(design["derived_outputs"]) + "`")
    lines.append(f"- supervision boundary: {design['supervision_boundary']}")
    lines.append(f"- matching priority: `{', '.join(design['matching_priority'])}`")
    lines.append(f"- forward consistency: {design['forward_consistency']}")
    lines.extend(["", "## Roadmap", ""])
    for item in payload["roadmap"]:
        lines.append(f"- `{item['stage']}`: {item['route']}; training=`{item['training']}`")
    lines.extend(["", "## Boundary", ""])
    for key, value in payload["boundary"].items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    if ROOT != Path(r"C:\Users\19166\Desktop\PINN_project"):
        raise SystemExit(f"Refusing to run outside PINN_project: {ROOT}")
    payload = build_payload()
    manifest = {
        "stage": "25.18",
        "script": "scripts/design_surface_multipit_raster_target_route_reset.py",
        "metrics_path": str(METRICS_PATH.relative_to(ROOT)),
        "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
        "route_stop_acceptance_decision": payload["route_stop_acceptance_decision"],
        "route_decision": payload["route_decision"],
        "training_run": False,
        "loss_tuning": False,
        "model_capacity_expanded": False,
        "current_baseline_updated": False,
        "baseline_transition": False,
        "allowed_use": ["route_reset_record", "geometry_primary_design_input", "25_19_plan_input"],
        "forbidden_use": ["baseline_update", "current_baseline_replacement", "raster_target_training_continuation"],
    }
    write_json(METRICS_PATH, payload)
    write_json(MANIFEST_PATH, manifest)
    write_summary(SUMMARY_PATH, payload)
    print(json.dumps({"decision": payload["route_stop_acceptance_decision"], "next_route": payload["route_decision"]}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

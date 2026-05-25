#!/usr/bin/env python
"""Audit Piao-style RBC evaluation against the project metric boundary.

This is a documentation/metrics audit only. It does not parse COMSOL data,
does not train, and does not modify the dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LITERATURE = Path(r"C:\Users\19166\Desktop\PINN_literature")

SUMMARY = ROOT / "results/summaries/piao_rbc_evaluation_alignment_summary.txt"
MATRIX = ROOT / "results/metrics/piao_rbc_evaluation_alignment_matrix.csv"

SOURCE_PATHS = [
    LITERATURE / "Fast reconstruction of 3-D defect profile from MFL signals using key physics-based parameters and SVM.pdf",
    ROOT / "results/summaries/piao2019_fullpaper_alignment_summary.txt",
    ROOT / "results/summaries/true_3d_piao_style_route_alignment_summary.txt",
    ROOT / "results/metrics/piao2019_method_mapping_matrix.csv",
    ROOT / "results/summaries/true_3d_rbc_v3_240_piao_nls_curvature_diagnostic_summary.txt",
    ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_decision_summary.txt",
]


def read_text(path: Path) -> str:
    if not path.exists() or path.suffix.lower() == ".pdf":
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    source_status = []
    source_text = []
    for path in SOURCE_PATHS:
        status = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "read_as_text": path.exists() and path.suffix.lower() != ".pdf",
        }
        source_status.append(status)
        source_text.append(read_text(path))

    joined = "\n".join(source_text).lower()
    mentions_profile = "profile" in joined and ("error" in joined or "reconstruction" in joined)
    mentions_six_params = all(token.lower() in joined for token in ["l", "w", "d"]) and "wld" in joined

    rows = [
        {
            "topic": "piao_route",
            "finding": "Piao-style route maps three-axis MFL through physics/NLS-inspired features to RBC six-parameter profile reconstruction.",
            "project_implication": "Keep exact_piao_rbc=false unless exact equations are implemented; current work remains RBC-style/Piao-inspired.",
            "evidence": "available route alignment summaries and method mapping matrix",
            "confidence": "medium",
        },
        {
            "topic": "w_parameters_role",
            "finding": "wLD/wWD/wLW control the RBC profile shape as geometry parameters.",
            "project_implication": "Treat wMAE as an auxiliary diagnostic, not the only primary success metric.",
            "evidence": "six-parameter RBC representation in project mapping",
            "confidence": "high" if mentions_six_params else "medium",
        },
        {
            "topic": "profile_reconstruction_metric",
            "finding": "Profile reconstruction error is better aligned with the Piao-style goal than isolated curvature-parameter MAE.",
            "project_implication": "Promote profile_depth_rmse_m / Er-like profile error to true-3D branch primary metric.",
            "evidence": "Piao title/route summaries emphasize 3-D defect profile reconstruction",
            "confidence": "high" if mentions_profile else "medium",
        },
        {
            "topic": "projected_mask_metric",
            "finding": "Projected mask IoU/Dice evaluates the 2-D footprint bridge, not full 3-D curvature.",
            "project_implication": "Keep projected mask as QA/comparator; do not let good Dice hide bad 3-D profile curvature.",
            "evidence": "20.78/20.81 summaries report high Dice with curvature risk",
            "confidence": "high",
        },
        {
            "topic": "current_project_metric_boundary",
            "finding": "Current per-parameter curvature MAE is a strict diagnostic and can conflict with profile-level quality.",
            "project_implication": "Stage 20.82 should audit profile-vs-parameter behavior before further model changes.",
            "evidence": "20.80 and 20.81 diagnostics",
            "confidence": "high",
        },
    ]
    write_csv(MATRIX, rows, ["topic", "finding", "project_implication", "evidence", "confidence"])

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "20.82 Piao RBC evaluation alignment summary",
        "",
        "scope: label/output-representation audit only; no COMSOL, no data generation, no NPZ modification, no training, no baseline update.",
        "method_claim: Piao-style / Piao-inspired alignment audit; not exact Piao reproduction.",
        "exact_piao_rbc: false",
        "rbc_style_approximation: true",
        "",
        "source_status:",
    ]
    lines.extend(
        f"- exists={row['exists']} read_as_text={row['read_as_text']} size_bytes={row['size_bytes']} path={row['path']}"
        for row in source_status
    )
    lines.extend(
        [
            "",
            "answer_1_RBC_six_params_generate_profile: yes; L/W/D/wLD/wWD/wLW are geometry/profile parameters used to form a 3-D defect profile.",
            "answer_2_Piao_reports_w_params_as_headline_metric: no evidence in available summaries that isolated wLD/wWD/wLW MAE is the final headline metric.",
            "answer_3_profile_reconstruction_error_alignment: profile-level reconstruction error is the closer Piao-style primary metric.",
            "answer_4_current_w_param_MAE_is_over_strict: yes as a primary metric; keep it as an auxiliary diagnostic.",
            "answer_5_recommended_primary_metrics: profile_depth_rmse_m, Er-like depth/profile reconstruction error, volume/depth descriptors; projected mask IoU/Dice remains footprint QA.",
            "answer_6_keep_wMAE_auxiliary: yes.",
            "",
            "boundary: this audit does not update CURRENT_BASELINE and does not claim exact Piao reproduction.",
        ]
    )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

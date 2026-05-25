"""Decide the fixed-scope v3_240 formal benchmark candidate status."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_benchmark_candidate_metrics.csv"
CURVATURE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_curvature_failure_audit_summary.txt"
SUMMARY_OUT = ROOT / "results/summaries/true_3d_rbc_v3_240_benchmark_candidate_decision_summary.txt"
MATRIX_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_benchmark_candidate_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric(rows: list[dict[str, str]], name: str) -> str:
    matches = [r for r in rows if r.get("metric") == name]
    if not matches:
        raise RuntimeError(f"missing metric: {name}")
    return matches[0].get("value", "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question", "answer", "evidence", "decision"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    rows = read_csv(BENCHMARK_METRICS)
    curv = CURVATURE_SUMMARY.read_text(encoding="utf-8")
    baseline_ready = metric(rows, "baseline_ready")
    exact_piao = metric(rows, "exact_piao_rbc")
    rbc_style = metric(rows, "rbc_style_approximation")
    neural_test = float(metric(rows, "train_val_test_normalized_mae").split("/")[-1])
    feature_test = float(metric(rows, "test_normalized_mae"))
    v2_delta = float(metric(rows, "v3_vs_v2_neural_test_mae_delta"))
    v2_d_delta = float(metric(rows, "v3_vs_v2_D_mae_mm_delta"))
    v2_curv_delta = float(metric(rows, "v3_vs_v2_curvature_mae_delta"))

    must_pass = neural_test < feature_test and v2_delta < 0 and v2_d_delta < 0 and baseline_ready == "False"
    curvature_risk = v2_curv_delta > 0
    decision = "benchmark_candidate_ready_for_formal_eval" if must_pass else "not_ready"
    if curvature_risk and must_pass:
        decision = "promising_but_curvature_risk"
    recommended_next = "B_model_refinement"

    matrix = [
        {"question": "Can this be called a formal benchmark candidate?", "answer": str(must_pass), "evidence": f"neural_test={neural_test:.6f}; feature_test={feature_test:.6f}; v2_delta={v2_delta:.6f}", "decision": decision},
        {"question": "Can this be called a baseline?", "answer": "False", "evidence": f"baseline_ready={baseline_ready}; CURRENT_BASELINE not updated", "decision": decision},
        {"question": "Should the next step be formal multi-seed benchmark rerun?", "answer": "Possible but not first", "evidence": "20.77 already used three seeds; curvature risk remains.", "decision": decision},
        {"question": "Should model refinement come first?", "answer": "True", "evidence": "L/W/D learned but wLD/wWD/wLW remain unstable; model/feature representation is the main risk.", "decision": decision},
        {"question": "Should curvature-targeted data come first?", "answer": "Second choice", "evidence": "boxy/sharp are weak, but N=240 expansion alone did not improve curvature vs N=112.", "decision": decision},
        {"question": "Should exact Piao NLS/LS-SVM feature pipeline be implemented?", "answer": "Strong candidate", "evidence": f"exact_piao_rbc={exact_piao}; rbc_style_approximation={rbc_style}; feature curvature is slightly better than neural.", "decision": decision},
        {"question": "Should the next step be expand to 480?", "answer": "Not first", "evidence": "Blind expansion improved L/W/D and D_m but not curvature.", "decision": decision},
        {"question": "Recommended unique next step", "answer": recommended_next, "evidence": "Start benchmark-candidate model refinement with curvature-risk controls; do not replace baseline.", "decision": decision},
    ]
    write_csv(MATRIX_OUT, matrix)

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 benchmark candidate decision summary",
                "",
                f"decision_category: {decision}",
                "formal_benchmark_candidate: True",
                "baseline_replacement: False",
                f"neural_beats_feature: {neural_test:.6f} < {feature_test:.6f}",
                f"improved_vs_N112_neural_mae: delta={v2_delta:.6f}",
                f"D_m_improved_vs_N112: delta_mm={v2_d_delta:.6f}",
                f"curvature_improved_vs_N112: False; delta={v2_curv_delta:.6f}",
                f"exact_piao_rbc: {exact_piao}",
                f"rbc_style_approximation: {rbc_style}",
                "",
                "answer_1_formal_candidate: yes, with curvature risk.",
                "answer_2_baseline: no; baseline_ready remains false and CURRENT_BASELINE must not change.",
                "answer_3_formal_multi_seed_rerun: useful for formal packaging, but not the first technical fix.",
                "answer_4_model_refinement: yes; prioritize curvature-aware model/head/loss and stronger sequence encoder.",
                "answer_5_curvature_targeted_data: useful after model/feature diagnostics, especially boxy/sharp.",
                "answer_6_exact_piao_features: yes; NLS/LS-SVM-inspired features are a strong next diagnostic because feature curvature is slightly better than neural.",
                "answer_7_expand_to_480: not first; N=240 expansion did not fix curvature.",
                "recommended_unique_next_step: B_model_refinement",
                "",
                "curvature_risk_reference:",
                curv.strip(),
                "",
                "Boundary: this is a benchmark candidate audit, not baseline replacement.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

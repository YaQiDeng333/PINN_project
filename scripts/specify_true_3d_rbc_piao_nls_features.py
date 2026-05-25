#!/usr/bin/env python
"""Write the Stage 20.80 Piao/NLS-inspired feature specification.

This is a feature diagnostic specification, not an exact Piao 2019
reproduction. The executable stages consume delta_b through the explicit
dataset_id registry/manifest gate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_piao_nls_feature_spec_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_spec_matrix.csv"
DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"


FIELDS = [
    "feature_group",
    "feature_family",
    "feature_name_pattern",
    "source",
    "requires_scipy",
    "can_fail",
    "imputation_rule",
    "expected_relation",
    "exact_piao_feature",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def feature_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(group: str, family: str, pattern: str, source: str, scipy: bool, can_fail: bool, impute: str, relation: str, notes: str) -> None:
        rows.append(
            {
                "feature_group": group,
                "feature_family": family,
                "feature_name_pattern": pattern,
                "source": source,
                "requires_scipy": scipy,
                "can_fail": can_fail,
                "imputation_rule": impute,
                "expected_relation": relation,
                "exact_piao_feature": False,
                "notes": notes,
            }
        )

    add("F0_existing_handcrafted", "control", "F0__ch{0..8}_{max|min|ptp|arg|max|min|energy|gradient}", "delta_b flattened to 9 channels", False, False, "none", "generic signal amplitude/position sanity", "Reuses the 20.77 hand-crafted feature baseline as a control.")
    add("F1_peak_shape", "peak_width_energy", "F1__{axis}_{line}_{width25|width50|width75|pos_area|neg_area|sharpness}", "per Bx/By/Bz axis and scan_line_y", False, True, "train median with failure flag", "width and flatness proxies for RBC curvature", "Adds width and lobe features missing from the 135-feature control.")
    add("F2_gradient_asymmetry", "gradient_asymmetry", "F2__{axis}_{line}_{grad_energy|zero_crossings|left_right_energy_ratio|asymmetry}", "per axis/line signal and first/second differences", False, True, "train median with failure flag", "edge sharpness and lobe asymmetry", "Detects shape imbalance that projected mask Dice can miss.")
    add("F3_cross_axis", "cross_axis_ratio", "F3__{line}_{Bx_Bz|By_Bz|energy_ratio|peak_offset|axis_corr}", "aligned Bx/By/Bz at each scan line", False, True, "train median with small-denominator flag", "tri-axis MFL coupling", "Engineering substitute for Piao three-axis MFL physics features.")
    add("F3_cross_axis", "vector_magnitude", "F3__vmag_{line}_{peak|width|energy|grad_energy}", "sqrt(Bx^2+By^2+Bz^2)", False, True, "train median with failure flag", "field magnitude shape independent of sign", "Useful when signed components have bipolar lobes.")
    add("F4_nls_curve_fit", "gaussian_abs", "F4__{axis}_{line}_gauss_{A|x0|sigma|C|rmse|success}", "bounded scipy.optimize.curve_fit on abs(signal)", True, True, "train median with fit_success flag", "NLS-like peak width/location/residual", "NLS proxy only; not the paper's exact two-stage 18-feature extraction.")
    add("F4_nls_curve_fit", "derivative_gaussian", "F4__{axis}_{line}_dog_{A|x0|sigma|C|rmse|success}", "bounded scipy.optimize.curve_fit on signed signal", True, True, "train median with fit_success flag", "bipolar MFL lobe shape", "Captures derivative-like MFL signatures if stable.")
    add("F5_curvature_focused", "ratio_derived", "F5__{axis|line}_{width_ratio|sigma_ratio|center_outer|lobe_distance}", "derived from F1-F4 only", False, True, "train median with denominator flag", "wLD/wWD/wLW-focused curvature proxies", "No labels, templates, split, or bins are used as feature inputs.")
    return rows


def run(args: argparse.Namespace) -> int:
    rows = feature_rows()
    write_csv(args.matrix, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 Piao/NLS-inspired feature spec summary",
                "",
                f"dataset_id: {DATASET_ID}",
                "scope: feature diagnostic only; no COMSOL, no data generation, no NPZ modification, no neural training, no baseline update.",
                "method_claim: Piao/NLS-inspired engineering approximation; not exact Piao 2019 reproduction.",
                "piao_source_status: local PDF text extraction was not reliable in preflight; implementation uses existing fullpaper alignment summary and project metrics.",
                "exact_piao_rbc: false",
                "rbc_style_approximation: true",
                "input_boundary: features are extracted from delta_b / BxByBz only.",
                "forbidden_inputs: rbc_params, projected_mask_2d, split, sample_id, curvature_template, depth_bin, aspect_bin, size_bin.",
                "",
                "feature_groups:",
                "- F0_existing_handcrafted: 20.77 135-feature control.",
                "- F1_peak_shape: peak/width/energy/lobe shape per axis and scan line.",
                "- F2_gradient_asymmetry: gradients, zero crossings, left-right asymmetry.",
                "- F3_cross_axis: Bx/By/Bz ratios, offsets, correlations, vector magnitude.",
                "- F4_nls_curve_fit: bounded gaussian and derivative-of-gaussian NLS proxies when scipy is available.",
                "- F5_curvature_focused: width/sigma/lobe/center-outer ratios for wLD/wWD/wLW diagnostics.",
                "",
                "feature_failure_policy: extraction records fit/width/denominator failure flags; regression uses train-only imputation/scaling.",
                f"feature_spec_rows: {len(rows)}",
                f"matrix_path: {args.matrix}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

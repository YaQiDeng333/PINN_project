"""Design the 20.87 true 3D RBC robustness expansion plan.

This is a planning-only script. It writes summary and CSV design artifacts, and
does not read/write NPZ data, run COMSOL, or train any model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "results" / "summaries"
METRICS_DIR = ROOT / "results" / "metrics"
MANIFEST = ROOT / "results" / "manifests" / "comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json"

BASELINE = {
    "profile_depth_rmse_m": 0.000387737,
    "er_like_profile_error": 0.340544,
    "L_mae_mm": 1.892,
    "W_mae_mm": 2.186,
    "D_mae_mm": 0.800,
    "projected_mask_dice": 0.847727,
    "projected_mask_iou": 0.750650,
    "wMAE_auxiliary": 0.201076,
}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifest() -> dict[str, object]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def factor_rows() -> list[dict[str, object]]:
    return [
        {
            "layer": "Layer 1 observation perturbation",
            "factor": "additive_noise",
            "levels": "0%,5%,10%,15%,20% relative to train signal scale",
            "simulation_type": "postprocess_delta_b",
            "requires_comsol": "false",
            "requires_new_labels": "false",
            "expected_failure_mode": "profile RMSE and L/W/D degrade as SNR drops; channel-specific noise may expose Bx/By/Bz reliance",
            "next_stage": "20.88",
        },
        {
            "layer": "Layer 1 observation perturbation",
            "factor": "amplitude_scaling_sensor_gain",
            "levels": "per-axis and global scale: 0.8,0.9,1.0,1.1,1.2",
            "simulation_type": "postprocess_delta_b",
            "requires_comsol": "false",
            "requires_new_labels": "false",
            "expected_failure_mode": "dimension/depth bias from gain mismatch",
            "next_stage": "20.88",
        },
        {
            "layer": "Layer 1 observation perturbation",
            "factor": "baseline_offset_zero_drift",
            "levels": "global/per-axis DC offset at 0%,2%,5%,10% signal p95",
            "simulation_type": "postprocess_delta_b",
            "requires_comsol": "false",
            "requires_new_labels": "false",
            "expected_failure_mode": "false profile broadening or depth bias from nonzero background",
            "next_stage": "20.88",
        },
        {
            "layer": "Layer 1 observation perturbation",
            "factor": "no_defect_reference_error",
            "levels": "low-frequency residual field and per-line bias at 2%,5%,10%",
            "simulation_type": "postprocess_delta_b",
            "requires_comsol": "false",
            "requires_new_labels": "false",
            "expected_failure_mode": "background subtraction error corrupts delta_b and confuses profile depth",
            "next_stage": "20.88",
        },
        {
            "layer": "Layer 1 diagnostic; Layer 2 for formal spatial claim",
            "factor": "sensor_x_resampling_jitter",
            "levels": "sub-sample x jitter: 0.25,0.5,1.0 sensor steps",
            "simulation_type": "postprocess_interpolation for 20.88 diagnostic; formal_by_COMSOL for spatial claim",
            "requires_comsol": "false for diagnostic-only 20.88; true for formal 20.89 claim",
            "requires_new_labels": "false",
            "expected_failure_mode": "edge/profile localization error and degraded L/W inference",
            "next_stage": "20.88_then_20.89_if_sensitive",
        },
        {
            "layer": "Layer 1 observation perturbation",
            "factor": "channel_dropout_partial_missing",
            "levels": "drop Bx, By, Bz, one scan line, or one axis-line pair",
            "simulation_type": "postprocess_delta_b_masking",
            "requires_comsol": "false",
            "requires_new_labels": "false",
            "expected_failure_mode": "identifies whether a single axis/line dominates prediction",
            "next_stage": "20.88",
        },
        {
            "layer": "Layer 2 approximate now; COMSOL preferred",
            "factor": "liftoff_variation",
            "levels": "sensor_z around 0.008 m: 0.006,0.007,0.008,0.009,0.010",
            "simulation_type": "approximate_by_filtering_or_amplitude_decay; formal_by_COMSOL",
            "requires_comsol": "true for formal claim",
            "requires_new_labels": "false",
            "expected_failure_mode": "depth and amplitude ambiguity under changing sensor height",
            "next_stage": "20.89",
        },
        {
            "layer": "Layer 2 approximate now; COMSOL preferred",
            "factor": "scan_line_y_offset",
            "levels": "line offsets +/-0.25, +/-0.5, +/-1.0 mm",
            "simulation_type": "approximate_by_line_interpolation; formal_by_COMSOL",
            "requires_comsol": "true for formal claim",
            "requires_new_labels": "false",
            "expected_failure_mode": "width/aspect/profile shift error from off-center scan",
            "next_stage": "20.89",
        },
        {
            "layer": "Layer 2 approximate now; COMSOL preferred",
            "factor": "Bx_By_Bz_spatial_misalignment",
            "levels": "axis-specific x/y shifts +/-0.25 to +/-1.0 sensor step",
            "simulation_type": "postprocess_interpolation; formal_by_COMSOL_or_calibrated_sensor_model",
            "requires_comsol": "true for formal claim",
            "requires_new_labels": "false",
            "expected_failure_mode": "multi-axis phase mismatch corrupts profile features",
            "next_stage": "20.89",
        },
        {
            "layer": "Layer 2 approximate now; COMSOL preferred",
            "factor": "source_strength_variation",
            "levels": "Jscale/gain proxy: 0.8,0.9,1.0,1.1,1.2",
            "simulation_type": "amplitude_proxy_for_screen; formal_by_COMSOL",
            "requires_comsol": "true for formal claim",
            "requires_new_labels": "false",
            "expected_failure_mode": "confounds material/source calibration with defect size/depth",
            "next_stage": "20.89",
        },
        {
            "layer": "Layer 2 approximate now; COMSOL preferred",
            "factor": "material_BH_proxy_variation",
            "levels": "small permeability/B-H proxy bins if COMSOL supports stable material sweep",
            "simulation_type": "proxy_only_for_design; formal_by_COMSOL",
            "requires_comsol": "true",
            "requires_new_labels": "false",
            "expected_failure_mode": "field amplitude/shape changes not explainable by current labels",
            "next_stage": "20.89_or_later",
        },
        {
            "layer": "Layer 3 must new COMSOL/new labels",
            "factor": "surface_shape_extension",
            "levels": "cuboid, ellipsoid, flat-bottom, RBC-like",
            "simulation_type": "new_COMSOL_pack",
            "requires_comsol": "true",
            "requires_new_labels": "true for shape_type and shape-specific descriptors",
            "expected_failure_mode": "six RBC params cannot represent non-RBC profiles without shape conditioning",
            "next_stage": "20.90",
        },
        {
            "layer": "Layer 3 must new COMSOL/new labels",
            "factor": "internal_buried_defect",
            "levels": "burial_depth/depth_to_surface prototype bins",
            "simulation_type": "new_COMSOL_pack_and_schema_design",
            "requires_comsol": "true",
            "requires_new_labels": "true",
            "expected_failure_mode": "surface profile labels and projected mask no longer define the defect fully",
            "next_stage": "20.91",
        },
        {
            "layer": "Layer 3 must new COMSOL/new labels",
            "factor": "multi_defect",
            "levels": "two or more true joint-solved defects",
            "simulation_type": "new_COMSOL_joint_solve",
            "requires_comsol": "true",
            "requires_new_labels": "true for instance/topology labels",
            "expected_failure_mode": "linear delta_b superposition is not sufficient for formal data; instance ambiguity",
            "next_stage": "future_after_20.90",
        },
        {
            "layer": "Layer 3 must new COMSOL/new labels",
            "factor": "arbitrary_free_form_profile",
            "levels": "profile basis/free-form surface descriptors",
            "simulation_type": "new_COMSOL_pack_and_representation_design",
            "requires_comsol": "true",
            "requires_new_labels": "true",
            "expected_failure_mode": "RBC six-param head underfits non-RBC geometry",
            "next_stage": "future_profile_native_branch",
        },
    ]


def stage_rows() -> list[dict[str, object]]:
    return [
        {
            "stage": "20.88 observation perturbation robustness audit",
            "goal": "Stress current v3_240 baseline against observation-space perturbations without new physics claims",
            "input_source": "existing v3_240 delta_b via explicit dataset_id/manifest",
            "comsol_required": "false",
            "data_generated": "false",
            "training_required": "false",
            "target_scale": "val/test 78 samples first; optionally all 240 for diagnostics",
            "gate": "baseline replay; light perturbations profile RMSE <= 0.000410 m and Dice >= 0.835; identify dominant sensitive factor",
            "next_step_on_pass": "20.89 only for sensitive liftoff/offset factors, or 20.90 if Layer 1 is stable",
            "next_step_on_fail": "augmentation/noise training investigation; fix observation definition if channel/line sensitivity is pathological",
        },
        {
            "stage": "20.89 liftoff/sensor-offset COMSOL diagnostic pack",
            "goal": "Separate true sensor geometry effects from post-processing approximations",
            "input_source": "new small COMSOL diagnostic pack design",
            "comsol_required": "true",
            "data_generated": "planned only in 20.87; actual generation later",
            "training_required": "false",
            "target_scale": "30 RBC base samples, 5 curvature templates x 6; about 7 observation variants each, about 210 rows",
            "gate": "COMSOL success >=95%; z=0.008/offset=0 replay near 20.86; mild offset/liftoff RMSE <= 0.000465 m or explainable by metadata",
            "next_step_on_pass": "20.90 shape-extension pilot or 20.92 augmentation if offset metadata is useful",
            "next_step_on_fail": "fix COMSOL/metadata/delta protocol before any training",
        },
        {
            "stage": "20.90 shape-extension pilot",
            "goal": "Design controlled surface defect type expansion without mixing buried/internal defects",
            "input_source": "new surface-defect shape pack or representation pilot",
            "comsol_required": "true for cuboid/ellipsoid/flat-bottom data; false only for representation design on v3_240",
            "data_generated": "planned only in 20.87; actual generation later",
            "training_required": "not in 20.87; later validation-only pilot",
            "target_scale": "surface shapes: cuboid, ellipsoid, flat-bottom, RBC-like; start small before broad pack",
            "gate": "shape labels defined; old surface RBC slice does not regress; profile RMSE remains primary; Dice-only gains are negative evidence",
            "next_step_on_pass": "shape_type-conditioned model or profile-native representation candidate",
            "next_step_on_fail": "return to representation or COMSOL diagnostic; do not merge into current baseline",
        },
        {
            "stage": "20.91 internal defect feasibility design",
            "goal": "Define buried/internal defect labels and observability before any training",
            "input_source": "design-only; later 12-24 prototype smoke pack if labels are sound",
            "comsol_required": "true for any actual sample",
            "data_generated": "false in design stage",
            "training_required": "false",
            "target_scale": "design first; prototype 12-24 only after schema decision",
            "gate": "burial_depth/depth_to_surface and profile metrics defined; not mixed with surface RBC baseline",
            "next_step_on_pass": "separate internal-defect COMSOL feasibility branch",
            "next_step_on_fail": "defer real/internal route; continue surface robustness",
        },
        {
            "stage": "20.92 robustness training / augmentation gate",
            "goal": "Train robustness variant only after evidence identifies useful perturbations",
            "input_source": "clean v3_240 plus 20.88/20.89 confirmed observation factors; shape augmentation only if 20.90 passes separately",
            "comsol_required": "depends on selected factors",
            "data_generated": "not in 20.87",
            "training_required": "true in 20.92 only",
            "target_scale": "3 seeds; clean test remains final-only",
            "gate": "clean RMSE <= 0.000410 m; L/W/D <= 2.08/2.40/0.88 mm; Dice >=0.835; perturbed RMSE improves >=10% vs frozen baseline",
            "next_step_on_pass": "consider robustness candidate, not automatic baseline replacement",
            "next_step_on_fail": "keep 20.86 baseline; retain variant only for deployment-specific diagnostics",
        },
    ]


def acceptance_rows() -> list[dict[str, object]]:
    return [
        {
            "metric": "profile_depth_rmse_m",
            "baseline_value": BASELINE["profile_depth_rmse_m"],
            "green_threshold": "<= baseline * 1.10 = 0.0004265 m for robustness; no-regression target <=0.000410 m",
            "warning_threshold": ">+10% to +25% degradation",
            "fail_threshold": ">+25% degradation or clean RMSE >0.000430 m",
            "applies_to": "all robustness and expansion stages",
            "decision": "primary gate; if profile RMSE fails, do not upgrade even if Dice improves",
        },
        {
            "metric": "L_m_MAE_mm",
            "baseline_value": BASELINE["L_mae_mm"],
            "green_threshold": "<= +15% = 2.176 mm; practical no-regression <=2.08 mm",
            "warning_threshold": "+15% to +30%",
            "fail_threshold": ">+30%",
            "applies_to": "all stages using six-parameter output",
            "decision": "dimension guard; profile-only branches with L fail cannot replace six-param baseline",
        },
        {
            "metric": "W_m_MAE_mm",
            "baseline_value": BASELINE["W_mae_mm"],
            "green_threshold": "<= +15% = 2.514 mm; practical no-regression <=2.40 mm",
            "warning_threshold": "+15% to +30%",
            "fail_threshold": ">+30%",
            "applies_to": "all stages using six-parameter output",
            "decision": "dimension guard",
        },
        {
            "metric": "D_m_MAE_mm",
            "baseline_value": BASELINE["D_mae_mm"],
            "green_threshold": "<= +15% = 0.920 mm; practical no-regression <=0.88 mm",
            "warning_threshold": "+15% to +30%",
            "fail_threshold": ">+30%",
            "applies_to": "all stages using six-parameter output",
            "decision": "depth guard; strong indicator for profile RMSE failures",
        },
        {
            "metric": "projected_mask_dice",
            "baseline_value": BASELINE["projected_mask_dice"],
            "green_threshold": "drop <=0.02, Dice >=0.8277; practical no-regression >=0.835",
            "warning_threshold": "drop 0.02 to 0.05",
            "fail_threshold": "drop >0.05",
            "applies_to": "2D footprint QA",
            "decision": "secondary QA; high Dice cannot override profile RMSE failure",
        },
        {
            "metric": "perturbed_RMSE_improvement_vs_frozen_baseline",
            "baseline_value": "frozen 20.86 under same perturbation",
            "green_threshold": ">=10% improvement with clean no-regression",
            "warning_threshold": "0-10% improvement or minor clean regression",
            "fail_threshold": "no improvement or clean profile RMSE regression",
            "applies_to": "20.92 robustness training",
            "decision": "augmentation is useful only if clean baseline remains intact",
        },
        {
            "metric": "COMSOL_generation_success",
            "baseline_value": "N/A",
            "green_threshold": ">=95% for 20.89 diagnostic pack",
            "warning_threshold": "90-95%",
            "fail_threshold": "<90% or delta_b check failure",
            "applies_to": "20.89 and Layer 3 generation",
            "decision": "COMSOL/metadata blocker; do not train on unstable generated data",
        },
        {
            "metric": "label_schema_defined",
            "baseline_value": "surface RBC six params",
            "green_threshold": "all new labels and metrics defined before generation",
            "warning_threshold": "partial labels, no training",
            "fail_threshold": "ambiguous internal/free-form labels",
            "applies_to": "20.90/20.91/Layer 3",
            "decision": "new schema required before internal, multi-defect, or free-form branches",
        },
    ]


def main() -> None:
    manifest = load_manifest()
    if manifest.get("dataset_id") != "comsol_true_3d_rbc_imported_watertight_pilot_v3_240":
        raise RuntimeError("Unexpected manifest dataset_id")

    factors = factor_rows()
    stages = stage_rows()
    acceptance = acceptance_rows()

    write_csv(
        METRICS_DIR / "true_3d_rbc_robustness_factor_matrix.csv",
        factors,
        [
            "layer",
            "factor",
            "levels",
            "simulation_type",
            "requires_comsol",
            "requires_new_labels",
            "expected_failure_mode",
            "next_stage",
        ],
    )
    write_csv(
        METRICS_DIR / "true_3d_rbc_robustness_stage_table.csv",
        stages,
        [
            "stage",
            "goal",
            "input_source",
            "comsol_required",
            "data_generated",
            "training_required",
            "target_scale",
            "gate",
            "next_step_on_pass",
            "next_step_on_fail",
        ],
    )
    write_csv(
        METRICS_DIR / "true_3d_rbc_robustness_acceptance_matrix.csv",
        acceptance,
        [
            "metric",
            "baseline_value",
            "green_threshold",
            "warning_threshold",
            "fail_threshold",
            "applies_to",
            "decision",
        ],
    )

    write_text(
        SUMMARY_DIR / "true_3d_rbc_robustness_expansion_plan_summary.txt",
        [
            "20.87 true 3D RBC robustness expansion plan summary",
            "",
            "Boundary:",
            "- Design-only. No COMSOL, no training, no data/NPZ generation.",
            "- CURRENT_BASELINE remains unchanged.",
            "- Baseline is 20.86 true 3D RBC profile-depth, dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240.",
            "- exact_piao_rbc=false; rbc_style_approximation=true.",
            "",
            "Recommended order:",
            "1. 20.88 observation perturbation robustness audit on existing v3_240 delta_b.",
            "2. 20.89 liftoff/sensor-offset COMSOL diagnostic pack for factors that cannot be trusted as post-processing.",
            "3. 20.90 surface shape-extension pilot with cuboid/ellipsoid/flat-bottom/RBC-like single defects.",
            "4. 20.91 internal defect feasibility design with new burial labels; do not mix into current surface baseline.",
            "5. 20.92 observation robustness training/augmentation gate only after 20.88/20.89 evidence; shape augmentation remains conditional on separate 20.90 evidence.",
            "",
            "Core rule:",
            "- Profile_depth_rmse_m is the headline metric. Dice-only improvement is not a baseline upgrade.",
        ],
    )
    write_text(
        SUMMARY_DIR / "true_3d_rbc_robustness_stage_plan.txt",
        [
            "20.87 robustness stage plan",
            "",
            "20.88: Observation perturbation audit",
            "- No COMSOL, no training. Perturb existing v3_240 delta_b only in evaluation.",
            "- Factors: noise, gain, zero drift, no-defect reference error, x jitter, channel dropout.",
            "- Gate: baseline replay; light perturbation RMSE <=0.000410 m and Dice >=0.835.",
            "",
            "20.89: Liftoff/sensor-offset COMSOL diagnostic pack",
            "- Target 30 RBC base samples, five curvature templates x six samples.",
            "- Around seven observation variants per base sample, about 210 rows.",
            "- Gate: COMSOL success >=95%, zero-offset replay near 20.86, mild offset/liftoff RMSE <=0.000465 m or metadata-explainable.",
            "",
            "20.90: Shape-extension pilot",
            "- Controlled surface shapes: cuboid, ellipsoid, flat-bottom, RBC-like.",
            "- Must carry shape_type labels and old-scope regression checks.",
            "- Do not treat non-RBC shape results as current RBC baseline replacement without separate candidate review.",
            "",
            "20.91: Internal defect feasibility design",
            "- Design labels first: burial_depth, depth_to_surface, projected mask meaning, profile metric meaning.",
            "- Internal/buried defects are a separate route, not robustness of the current surface RBC baseline.",
            "",
            "20.92: Robustness training / augmentation gate",
            "- Observation robustness training runs only after 20.88/20.89 identify stable perturbation factors.",
            "- Shape augmentation is allowed only if 20.90 separately validates shape labels and single-defect surface scope.",
            "- Clean v3_240 must remain no-regression; robust improvement must be measured against frozen 20.86 under same perturbations.",
        ],
    )
    write_text(
        SUMMARY_DIR / "true_3d_rbc_robustness_acceptance_criteria.txt",
        [
            "20.87 robustness acceptance criteria",
            "",
            "Baseline values:",
            "- profile_depth_rmse_m=0.000387737",
            "- L/W/D MAE=1.892/2.186/0.800 mm",
            "- projected mask Dice=0.847727",
            "",
            "Primary gate:",
            "- Profile RMSE green <= +10%; warning +10% to +25%; fail > +25%.",
            "- Practical clean no-regression target: RMSE <=0.000410 m.",
            "- Upgrade target: RMSE <=0.000370 m with L/W/D and Dice no-regression.",
            "",
            "Dimension guard:",
            "- L/W/D green <= +15%; warning +15% to +30%; fail > +30%.",
            "- Practical no-regression values: L<=2.08 mm, W<=2.40 mm, D<=0.88 mm.",
            "",
            "Projected mask QA:",
            "- Dice green drop <=0.02; warning drop 0.02-0.05; fail drop >0.05.",
            "- Dice is secondary. High Dice cannot override profile RMSE failure.",
            "",
            "Routing:",
            "- Layer 1 failures imply augmentation/noise training investigation.",
            "- Layer 2 failures imply new COMSOL diagnostic data before robustness claims.",
            "- Layer 3 failures imply schema/label redesign, not current-baseline retraining.",
            "- Real experimental data remains after clean simulation robustness evidence.",
        ],
    )


if __name__ == "__main__":
    main()

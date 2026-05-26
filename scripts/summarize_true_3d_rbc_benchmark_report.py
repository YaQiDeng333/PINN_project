"""Generate the 20.86 true 3D RBC benchmark report package.

This script is intentionally metadata-only. It reads committed summaries,
metrics, registry, and manifest files; it never reads or writes NPZ/data files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"

SUMMARY_OUT = ROOT / "results" / "summaries" / "true_3d_rbc_benchmark_report_summary.txt"
METRICS_OUT = ROOT / "results" / "metrics" / "true_3d_rbc_benchmark_report_metrics.csv"
COMPARISON_OUT = ROOT / "results" / "metrics" / "true_3d_rbc_benchmark_comparison_matrix.csv"

FORMAL_COMPARISON = ROOT / "results" / "metrics" / "true_3d_rbc_formal_benchmark_comparison_matrix.csv"
FORMAL_SEEDS = ROOT / "results" / "metrics" / "true_3d_rbc_formal_benchmark_20_77_seed_summary.csv"
MANIFEST = ROOT / "results" / "manifests" / f"{DATASET_ID}.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
CURRENT_BASELINE = ROOT / "CURRENT_BASELINE.md"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required file missing or empty: {path}")


def main() -> None:
    for path in [FORMAL_COMPARISON, FORMAL_SEEDS, MANIFEST, REGISTRY, CURRENT_BASELINE]:
        require_file(path)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    comparison_rows = read_csv_rows(FORMAL_COMPARISON)
    seed_rows = read_csv_rows(FORMAL_SEEDS)

    formal = next(row for row in comparison_rows if row["candidate_id"] == "20.85_formal_rerun_20.77_protocol")
    visual = next(row for row in comparison_rows if row["candidate_id"] == "20.81_feature_fusion")
    negative = next(row for row in comparison_rows if row["candidate_id"] == "20.83_profile_primary_negative_gate")
    selected_seed = next(row for row in seed_rows if row.get("selected_seed") == "True")

    report_rows: list[dict[str, object]] = [
        {"section": "identity", "metric": "dataset_id", "value": DATASET_ID, "unit": "", "role": "current_true3d_baseline"},
        {"section": "identity", "metric": "route", "value": manifest["route"], "unit": "", "role": "current_true3d_baseline"},
        {"section": "identity", "metric": "schema_version", "value": manifest["schema_version"], "unit": "", "role": "current_true3d_baseline"},
        {"section": "identity", "metric": "n_samples", "value": manifest["n_samples"], "unit": "samples", "role": "current_true3d_baseline"},
        {"section": "identity", "metric": "split_counts", "value": json.dumps(manifest["split_counts"], sort_keys=True), "unit": "", "role": "current_true3d_baseline"},
        {"section": "method", "metric": "input", "value": "delta_b Bx/By/Bz shape=(N,3,3,201), Conv1D=(N,9,201)", "unit": "", "role": "current_true3d_baseline"},
        {"section": "method", "metric": "model", "value": "20.77 small Conv1D encoder + MLP six-parameter head", "unit": "", "role": "current_true3d_baseline"},
        {"section": "method", "metric": "output", "value": "L,W,D,wLD,wWD,wLW -> RBC-style 3D profile/depth/projected mask", "unit": "", "role": "current_true3d_baseline"},
        {"section": "selection", "metric": "selected_seed", "value": formal["selected_seed"], "unit": "", "role": "current_true3d_baseline"},
        {"section": "selection", "metric": "best_epoch", "value": selected_seed["best_epoch"], "unit": "epoch", "role": "current_true3d_baseline"},
        {"section": "main", "metric": "test_total_normalized_mae", "value": formal["test_total_mae"], "unit": "", "role": "current_true3d_baseline"},
        {"section": "main", "metric": "test_profile_depth_rmse_m", "value": formal["test_profile_depth_rmse_m"], "unit": "m", "role": "current_true3d_baseline"},
        {"section": "main", "metric": "test_er_like_profile_error", "value": formal["test_er_like_profile_error"], "unit": "", "role": "current_true3d_baseline"},
        {"section": "main", "metric": "test_L_mae_mm", "value": formal["test_L_mae_mm"], "unit": "mm", "role": "current_true3d_baseline"},
        {"section": "main", "metric": "test_W_mae_mm", "value": formal["test_W_mae_mm"], "unit": "mm", "role": "current_true3d_baseline"},
        {"section": "main", "metric": "test_D_mae_mm", "value": formal["test_D_mae_mm"], "unit": "mm", "role": "current_true3d_baseline"},
        {"section": "auxiliary", "metric": "test_wMAE_auxiliary", "value": formal["test_wMAE_auxiliary"], "unit": "", "role": "diagnostic_only"},
        {"section": "auxiliary", "metric": "test_wLD_abs_error", "value": formal["test_wLD_abs_error"], "unit": "", "role": "diagnostic_only"},
        {"section": "auxiliary", "metric": "test_wWD_abs_error", "value": formal["test_wWD_abs_error"], "unit": "", "role": "diagnostic_only"},
        {"section": "auxiliary", "metric": "test_wLW_abs_error", "value": formal["test_wLW_abs_error"], "unit": "", "role": "diagnostic_only"},
        {"section": "mask", "metric": "test_projected_mask_iou", "value": formal["test_projected_mask_iou"], "unit": "", "role": "2d_footprint_qa"},
        {"section": "mask", "metric": "test_projected_mask_dice", "value": formal["test_projected_mask_dice"], "unit": "", "role": "2d_footprint_qa"},
        {"section": "comparator", "metric": "20.81_projected_mask_dice", "value": visual["test_projected_mask_dice"], "unit": "", "role": "visual_mask_comparator"},
        {"section": "comparator", "metric": "20.81_profile_depth_rmse_m", "value": visual["test_profile_depth_rmse_m"], "unit": "m", "role": "visual_mask_comparator"},
        {"section": "comparator", "metric": "20.83_projected_mask_dice", "value": negative["test_projected_mask_dice"], "unit": "", "role": "negative_gate"},
        {"section": "comparator", "metric": "20.83_profile_depth_rmse_m", "value": negative["test_profile_depth_rmse_m"], "unit": "m", "role": "negative_gate"},
        {"section": "limitation", "metric": "exact_piao_rbc", "value": str(manifest["exact_piao_rbc"]).lower(), "unit": "", "role": "scope_boundary"},
        {"section": "limitation", "metric": "rbc_style_approximation", "value": str(manifest["rbc_style_approximation"]).lower(), "unit": "", "role": "scope_boundary"},
        {"section": "limitation", "metric": "real_experimental_validation", "value": "false", "unit": "", "role": "scope_boundary"},
        {"section": "limitation", "metric": "arbitrary_freeform_multidefect", "value": "false", "unit": "", "role": "scope_boundary"},
    ]

    comparison_out_rows = [
        {
            "candidate_id": formal["candidate_id"],
            "role": "CURRENT_BASELINE_true3d_profile_depth",
            "primary_metric": "profile_depth_rmse_m",
            "profile_depth_rmse_m": formal["test_profile_depth_rmse_m"],
            "er_like_profile_error": formal["test_er_like_profile_error"],
            "total_normalized_mae": formal["test_total_mae"],
            "LWD_mae_mm": f'{formal["test_L_mae_mm"]}/{formal["test_W_mae_mm"]}/{formal["test_D_mae_mm"]}',
            "wMAE_auxiliary": formal["test_wMAE_auxiliary"],
            "projected_mask_iou": formal["test_projected_mask_iou"],
            "projected_mask_dice": formal["test_projected_mask_dice"],
            "baseline_role_after_20_86": "current_baseline",
            "notes": "Formal rerun of 20.77; profile/depth main baseline after project target transition.",
        },
        {
            "candidate_id": visual["candidate_id"],
            "role": "projected_mask_visual_comparator",
            "primary_metric": "projected_mask_dice",
            "profile_depth_rmse_m": visual["test_profile_depth_rmse_m"],
            "er_like_profile_error": visual["test_er_like_profile_error"],
            "total_normalized_mae": visual["test_total_mae"],
            "LWD_mae_mm": f'{visual["test_L_mae_mm"]}/{visual["test_W_mae_mm"]}/{visual["test_D_mae_mm"]}',
            "wMAE_auxiliary": visual["test_wMAE_auxiliary"],
            "projected_mask_iou": visual["test_projected_mask_iou"],
            "projected_mask_dice": visual["test_projected_mask_dice"],
            "baseline_role_after_20_86": "comparator",
            "notes": "Higher Dice than 20.77 but worse profile RMSE; not current baseline.",
        },
        {
            "candidate_id": negative["candidate_id"],
            "role": "profile_primary_negative_gate",
            "primary_metric": "profile_depth_rmse_m",
            "profile_depth_rmse_m": negative["test_profile_depth_rmse_m"],
            "er_like_profile_error": negative["test_er_like_profile_error"],
            "total_normalized_mae": negative["test_total_mae"],
            "LWD_mae_mm": f'{negative["test_L_mae_mm"]}/{negative["test_W_mae_mm"]}/{negative["test_D_mae_mm"]}',
            "wMAE_auxiliary": negative["test_wMAE_auxiliary"],
            "projected_mask_iou": negative["test_projected_mask_iou"],
            "projected_mask_dice": negative["test_projected_mask_dice"],
            "baseline_role_after_20_86": "negative_gate",
            "notes": "High Dice did not beat 20.77 profile RMSE; does not replace baseline.",
        },
        {
            "candidate_id": "old_v3_complex_mask_forward_consistency",
            "role": "archived_2d_mask_comparator",
            "primary_metric": "2d_mask_iou_dice_area_error",
            "profile_depth_rmse_m": "N/A",
            "er_like_profile_error": "N/A",
            "total_normalized_mae": "N/A",
            "LWD_mae_mm": "N/A",
            "wMAE_auxiliary": "N/A",
            "projected_mask_iou": "0.3563",
            "projected_mask_dice": "0.5017",
            "baseline_role_after_20_86": "archived_comparator",
            "notes": "Previous CURRENT_BASELINE for 2D v3_complex boundary task; retained for history/comparison only.",
        },
    ]

    lines = [
        "20.86 true 3D RBC benchmark report summary",
        "",
        "Task transition:",
        "- Previous CURRENT_BASELINE targeted 2D mask/boundary prediction on v3_complex.",
        "- New CURRENT_BASELINE targets true 3D RBC-style profile/depth reconstruction from Bx/By/Bz delta_b.",
        "",
        "Dataset identity:",
        f"- dataset_id: {DATASET_ID}",
        f"- route: {manifest['route']}",
        f"- schema_version: {manifest['schema_version']}",
        f"- n_samples: {manifest['n_samples']}",
        f"- split_counts: {json.dumps(manifest['split_counts'], sort_keys=True)}",
        f"- exact_piao_rbc: {str(manifest['exact_piao_rbc']).lower()}",
        f"- rbc_style_approximation: {str(manifest['rbc_style_approximation']).lower()}",
        "",
        "Method chain:",
        "- Bx/By/Bz delta_b -> Conv1D encoder -> six RBC-style parameters -> 3D profile/depth -> projected mask.",
        "- Input shape: delta_b=(N,3,3,201), Conv1D=(N,9,201).",
        "- Model: 20.77 small Conv1D encoder + MLP six-parameter head.",
        "- Selection: validation-only checkpoint/seed selection; selected seed 42.",
        "",
        "Main metrics:",
        f"- test_total_normalized_mae: {formal['test_total_mae']}",
        f"- profile_depth_rmse_m: {formal['test_profile_depth_rmse_m']}",
        f"- er_like_profile_error: {formal['test_er_like_profile_error']}",
        f"- L/W/D MAE mm: {formal['test_L_mae_mm']} / {formal['test_W_mae_mm']} / {formal['test_D_mae_mm']}",
        "",
        "Auxiliary diagnostics:",
        f"- wMAE: {formal['test_wMAE_auxiliary']}",
        f"- wLD/wWD/wLW: {formal['test_wLD_abs_error']} / {formal['test_wWD_abs_error']} / {formal['test_wLW_abs_error']}",
        "",
        "Projected mask QA:",
        f"- IoU/Dice: {formal['test_projected_mask_iou']} / {formal['test_projected_mask_dice']}",
        "",
        "Comparator roles:",
        "- 20.77/20.85: CURRENT_BASELINE true 3D profile-depth candidate.",
        "- 20.81: projected-mask / visual comparator only.",
        "- 20.83: profile-primary negative gate.",
        "- old v3_complex 2D mask baseline: archived comparator.",
        "",
        "Limitations:",
        "- exact_piao_rbc=false; this is RBC-style / Piao-inspired approximation.",
        "- wLD/wWD/wLW are auxiliary diagnostics, not the headline success metric.",
        "- Not validated on real experimental data.",
        "- Not yet arbitrary free-form, multi-defect, or real-deployment ready.",
    ]
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(METRICS_OUT, report_rows, ["section", "metric", "value", "unit", "role"])
    write_csv(
        COMPARISON_OUT,
        comparison_out_rows,
        [
            "candidate_id",
            "role",
            "primary_metric",
            "profile_depth_rmse_m",
            "er_like_profile_error",
            "total_normalized_mae",
            "LWD_mae_mm",
            "wMAE_auxiliary",
            "projected_mask_iou",
            "projected_mask_dice",
            "baseline_role_after_20_86",
            "notes",
        ],
    )


if __name__ == "__main__":
    main()

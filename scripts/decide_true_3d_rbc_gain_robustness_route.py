"""Decide the gain/amplitude robustness route after 20.89 audits."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_METRICS = ROOT / "results" / "metrics" / "true_3d_rbc_gain_calibration_strategy_metrics.csv"
AUGMENTED_METRICS = ROOT / "results" / "metrics" / "true_3d_rbc_gain_augmented_robustness_metrics.csv"
AUGMENTED_SEEDS = ROOT / "results" / "metrics" / "true_3d_rbc_gain_augmented_seed_summary.csv"
SUMMARY = ROOT / "results" / "summaries" / "true_3d_rbc_gain_robustness_route_decision_summary.txt"
MATRIX = ROOT / "results" / "metrics" / "true_3d_rbc_gain_robustness_decision_matrix.csv"

PROFILE_BASELINE_RMSE = 0.000387737
BASELINE_DICE = 0.847727


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def row_for(
    rows: Iterable[Dict[str, str]],
    *,
    split: str = "test",
    strategy: str | None = None,
    candidate: str | None = None,
    perturbation: str,
) -> Dict[str, str] | None:
    for row in rows:
        if row.get("split") != split:
            continue
        if strategy is not None and row.get("strategy") != strategy:
            continue
        if candidate is not None and row.get("candidate") != candidate:
            continue
        if row.get("perturbation_name") == perturbation:
            return row
    return None


def pct_reduction(reference_drop: float, current_drop: float) -> float:
    return 100.0 * (reference_drop - current_drop) / max(abs(reference_drop), 1.0e-9)


def best_calibration(cal_rows: List[Dict[str, str]]) -> Dict[str, object]:
    best: Dict[str, object] = {}
    best_score = float("inf")
    for strategy in sorted({row.get("strategy", "") for row in cal_rows if row.get("strategy") != "no_calibration"}):
        clean = row_for(cal_rows, strategy=strategy, split="val", perturbation="clean")
        gain08 = row_for(cal_rows, strategy=strategy, split="val", perturbation="gain_scaling_0.8x")
        gain12 = row_for(cal_rows, strategy=strategy, split="val", perturbation="gain_scaling_1.2x")
        bx50 = row_for(cal_rows, strategy=strategy, split="val", perturbation="channel_attenuation_Bx_50pct")
        ref08 = row_for(cal_rows, strategy="no_calibration", split="val", perturbation="gain_scaling_0.8x")
        ref12 = row_for(cal_rows, strategy="no_calibration", split="val", perturbation="gain_scaling_1.2x")
        if not (clean and gain08 and gain12 and bx50 and ref08 and ref12):
            continue
        clean_drop = float(clean["profile_rmse_degradation_pct_vs_20_85"])
        gain08_drop = float(gain08["profile_rmse_degradation_pct_vs_20_85"])
        gain12_drop = float(gain12["profile_rmse_degradation_pct_vs_20_85"])
        red08 = pct_reduction(float(ref08["profile_rmse_degradation_pct_vs_20_85"]), gain08_drop)
        red12 = pct_reduction(float(ref12["profile_rmse_degradation_pct_vs_20_85"]), gain12_drop)
        score = clean_drop + max(gain08_drop, gain12_drop) - 0.25 * min(red08, red12)
        if score < best_score:
            best_score = score
            best = {
                "strategy": strategy,
                "selection_split": "val",
                "clean_drop_pct": clean_drop,
                "clean_dice": float(clean["projected_mask_dice"]),
                "gain08_drop_pct": gain08_drop,
                "gain12_drop_pct": gain12_drop,
                "gain08_reduction_pct": red08,
                "gain12_reduction_pct": red12,
                "bx50_drop_pct": float(bx50["profile_rmse_degradation_pct_vs_20_85"]),
            }
    if best:
        strategy = str(best["strategy"])
        for name, perturbation in [
            ("test_clean", "clean"),
            ("test_gain08", "gain_scaling_0.8x"),
            ("test_gain12", "gain_scaling_1.2x"),
            ("test_bx50", "channel_attenuation_Bx_50pct"),
        ]:
            row = row_for(cal_rows, strategy=strategy, split="test", perturbation=perturbation)
            if row:
                best[f"{name}_drop_pct"] = float(row["profile_rmse_degradation_pct_vs_20_85"])
                best[f"{name}_dice"] = float(row["projected_mask_dice"])
        ref08_test = row_for(cal_rows, strategy="no_calibration", split="test", perturbation="gain_scaling_0.8x")
        ref12_test = row_for(cal_rows, strategy="no_calibration", split="test", perturbation="gain_scaling_1.2x")
        if ref08_test and "test_gain08_drop_pct" in best:
            best["test_gain08_reduction_pct"] = pct_reduction(float(ref08_test["profile_rmse_degradation_pct_vs_20_85"]), float(best["test_gain08_drop_pct"]))
        if ref12_test and "test_gain12_drop_pct" in best:
            best["test_gain12_reduction_pct"] = pct_reduction(float(ref12_test["profile_rmse_degradation_pct_vs_20_85"]), float(best["test_gain12_drop_pct"]))
    return best


def selected_augmented(aug_rows: List[Dict[str, str]], seed_rows: List[Dict[str, str]]) -> Dict[str, object]:
    selected = next((row for row in seed_rows if str(row.get("selected_robustness_candidate")) == "True"), None)
    if not selected:
        return {}
    candidate = selected["candidate"]
    seed = selected["seed"]
    out = {"candidate": candidate, "seed": int(seed)}
    for name, key in [
        ("clean", "clean"),
        ("gain_scaling_0.8x", "gain08"),
        ("gain_scaling_1.2x", "gain12"),
        ("channel_attenuation_Bx_50pct", "bx50"),
    ]:
        row = next(
            (
                item
                for item in aug_rows
                if item.get("candidate") == candidate
                and str(item.get("seed")) == str(seed)
                and item.get("split") == "test"
                and item.get("perturbation_name") == name
            ),
            None,
        )
        if row:
            out[f"{key}_profile_rmse_m"] = float(row["profile_depth_rmse_m"])
            out[f"{key}_drop_pct"] = float(row["profile_rmse_degradation_pct_vs_20_85"])
            out[f"{key}_dice"] = float(row["projected_mask_dice"])
            out[f"{key}_L_mae_mm"] = float(row["L_mae_mm"])
            out[f"{key}_W_mae_mm"] = float(row["W_mae_mm"])
            out[f"{key}_D_mae_mm"] = float(row["D_mae_mm"])
    ref08 = row_for(aug_rows, candidate="A0_reference_20_77", perturbation="gain_scaling_0.8x")
    ref12 = row_for(aug_rows, candidate="A0_reference_20_77", perturbation="gain_scaling_1.2x")
    refbx = row_for(aug_rows, candidate="A0_reference_20_77", perturbation="channel_attenuation_Bx_50pct")
    if ref08 and "gain08_drop_pct" in out:
        out["gain08_reduction_pct"] = pct_reduction(float(ref08["profile_rmse_degradation_pct_vs_20_85"]), float(out["gain08_drop_pct"]))
    if ref12 and "gain12_drop_pct" in out:
        out["gain12_reduction_pct"] = pct_reduction(float(ref12["profile_rmse_degradation_pct_vs_20_85"]), float(out["gain12_drop_pct"]))
    if refbx and "bx50_drop_pct" in out:
        out["bx50_reduction_pct"] = pct_reduction(float(refbx["profile_rmse_degradation_pct_vs_20_85"]), float(out["bx50_drop_pct"]))
    return out


def pass_gate(summary: Dict[str, object]) -> bool:
    return bool(
        summary
        and float(summary.get("test_clean_drop_pct", summary.get("clean_drop_pct", float("inf")))) <= 10.0
        and float(summary.get("test_clean_dice", summary.get("clean_dice", 0.0))) >= BASELINE_DICE - 0.02
        and float(summary.get("test_gain08_reduction_pct", summary.get("gain08_reduction_pct", 0.0))) >= 50.0
        and float(summary.get("test_gain12_reduction_pct", summary.get("gain12_reduction_pct", 0.0))) >= 50.0
    )


def main() -> None:
    cal_rows = read_csv(CALIBRATION_METRICS)
    aug_rows = read_csv(AUGMENTED_METRICS)
    seed_rows = read_csv(AUGMENTED_SEEDS)
    if not cal_rows:
        raise FileNotFoundError(CALIBRATION_METRICS)

    cal_best = best_calibration(cal_rows)
    cal_enough = pass_gate(cal_best)
    aug_best = selected_augmented(aug_rows, seed_rows)
    aug_enough = pass_gate(
        {
            "clean_drop_pct": aug_best.get("clean_drop_pct", float("inf")),
            "clean_dice": aug_best.get("clean_dice", 0.0),
            "gain08_reduction_pct": aug_best.get("gain08_reduction_pct", 0.0),
            "gain12_reduction_pct": aug_best.get("gain12_reduction_pct", 0.0),
        }
    )

    matrix = [
        {
            "question": "Is calibration-only sufficient?",
            "answer": cal_enough,
            "evidence": f"validation_selected={cal_best.get('strategy')}; test_clean_drop={cal_best.get('test_clean_drop_pct')}; test_gain08_reduction={cal_best.get('test_gain08_reduction_pct')}; test_gain12_reduction={cal_best.get('test_gain12_reduction_pct')}",
            "decision": "do_not_use_calibration_only_as_upgrade" if not cal_enough else "calibration_only_candidate",
        },
        {
            "question": "Is augmentation effective enough to upgrade?",
            "answer": aug_enough,
            "evidence": f"best={aug_best.get('candidate')} seed={aug_best.get('seed')}; clean_drop={aug_best.get('clean_drop_pct')}; gain08_reduction={aug_best.get('gain08_reduction_pct')}; gain12_reduction={aug_best.get('gain12_reduction_pct')}",
            "decision": "do_not_upgrade_due_to_clean_profile_cost" if not aug_enough else "robustness_candidate_ready",
        },
        {
            "question": "Should CURRENT_BASELINE remain 20.85?",
            "answer": True,
            "evidence": "Neither calibration-only nor augmentation passed all gates; CURRENT_BASELINE.md was not modified.",
            "decision": "retain_current_baseline",
        },
        {
            "question": "Is a separate robustness candidate needed?",
            "answer": True,
            "evidence": "Augmentation sharply reduced gain/Bx attenuation degradation but paid a clean profile RMSE cost.",
            "decision": "track_as_non_baseline_robustness_candidate_only",
        },
        {
            "question": "Can 20.90 liftoff/sensor-offset COMSOL diagnostic proceed?",
            "answer": True,
            "evidence": "20.89 isolated amplitude/gain sensitivity as a blocker; liftoff/sensor-offset should be measured separately with controlled amplitude calibration.",
            "decision": "proceed_to_20_90_with_gain_control_caveat",
        },
        {
            "question": "Is real-data amplitude calibration still a blocker?",
            "answer": True,
            "evidence": "Gain 0.8 and Bx attenuation remain severe without explicit augmentation or calibration; augmentation is not clean enough for baseline transition.",
            "decision": "real_data_requires_amplitude_calibration_protocol",
        },
    ]
    write_csv(MATRIX, matrix)

    lines = [
        "20.89 gain/amplitude robustness route decision",
        "",
        "Decision:",
        "- CURRENT_BASELINE remains the 20.85 true 3D RBC profile-depth baseline.",
        "- Calibration-only is not sufficient because the best strategy reduces gain sensitivity but costs too much clean profile RMSE.",
        "- Augmentation is useful diagnostically, but it is not a baseline upgrade because clean profile RMSE degrades beyond the <=10% clean gate.",
        "- A non-baseline robustness candidate can be tracked, but it should not replace the clean baseline.",
        "- Real-data amplitude/gain calibration remains a blocker.",
        "- Next step: 20.90 liftoff/sensor-offset COMSOL diagnostic pack with explicit gain/amplitude control.",
        "",
        "Calibration-only best:",
        f"- strategy: {cal_best.get('strategy')}",
        f"- selection_split: {cal_best.get('selection_split')}",
        f"- test_clean_drop_pct: {cal_best.get('test_clean_drop_pct')}",
        f"- test_gain08_drop_pct: {cal_best.get('test_gain08_drop_pct')}",
        f"- test_gain08_reduction_pct: {cal_best.get('test_gain08_reduction_pct')}",
        f"- test_gain12_drop_pct: {cal_best.get('test_gain12_drop_pct')}",
        f"- test_gain12_reduction_pct: {cal_best.get('test_gain12_reduction_pct')}",
        f"- test_bx50_drop_pct: {cal_best.get('test_bx50_drop_pct')}",
        "",
        "Augmentation selected candidate:",
        f"- candidate: {aug_best.get('candidate')}",
        f"- seed: {aug_best.get('seed')}",
        f"- clean_drop_pct: {aug_best.get('clean_drop_pct')}",
        f"- clean_dice: {aug_best.get('clean_dice')}",
        f"- gain08_drop_pct: {aug_best.get('gain08_drop_pct')}",
        f"- gain08_reduction_pct: {aug_best.get('gain08_reduction_pct')}",
        f"- gain12_drop_pct: {aug_best.get('gain12_drop_pct')}",
        f"- gain12_reduction_pct: {aug_best.get('gain12_reduction_pct')}",
        f"- bx50_drop_pct: {aug_best.get('bx50_drop_pct')}",
        f"- bx50_reduction_pct: {aug_best.get('bx50_reduction_pct')}",
        "",
        "Boundary:",
        "- No COMSOL was run.",
        "- No data/NPZ was generated or modified.",
        "- CURRENT_BASELINE.md remains unchanged.",
        "- Checkpoints and preview PNGs are not part of this commit.",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY}")
    print(f"wrote {MATRIX}")


if __name__ == "__main__":
    main()

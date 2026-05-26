"""Diagnose COMSOL V2 hard-threshold margin collapse from runner outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    return float(value)


def _avg(rows, key: str) -> float:
    if not rows:
        return 0.0
    return sum(_float(row, key) for row in rows) / len(rows)


def _all_hard_area_zero(metric_rows) -> bool:
    if not metric_rows:
        return False
    return all(_float(row, "defect_area_pred") == 0.0 for row in metric_rows)


def diagnose_run(run_dir: Path, mu_threshold: float, mask_temperature: float):
    metrics = {
        "train": _read_csv(run_dir / "metrics.csv"),
        "val": _read_csv(run_dir / "eval_metrics.csv"),
        "test": _read_csv(run_dir / "test_metrics.csv"),
    }
    history_rows = _read_csv(run_dir / "training_history.csv")

    hard_area_zero = any(_all_hard_area_zero(rows) for rows in metrics.values() if rows)
    final_history = history_rows[-1] if history_rows else {}
    final_pred_area_soft = _float(final_history, "pred_area_soft_mean")
    final_true_area = _float(final_history, "true_area_mean")
    final_min_mu = _float(final_history, "min_mu")
    final_mean_soft_defect = _float(final_history, "mean_soft_defect")
    final_area_loss = _float(final_history, "area_loss")

    soft_hard_mismatch = hard_area_zero and final_pred_area_soft > 0.0
    no_threshold_crossing = bool(history_rows) and final_min_mu > mu_threshold
    soft_collapse = bool(history_rows) and final_mean_soft_defect < 1.0e-3
    foreground_collapse = bool(history_rows) and final_true_area > 0.0 and final_pred_area_soft < 1.0e-3

    split_rows = []
    for split_name, rows in metrics.items():
        if not rows:
            continue
        split_rows.append(
            {
                "split": split_name,
                "num_rows": len(rows),
                "avg_defect_iou": _avg(rows, "defect_iou"),
                "avg_defect_area_pred": _avg(rows, "defect_area_pred"),
                "avg_defect_area_label": _avg(rows, "defect_area_label"),
                "avg_mu_mse": _avg(rows, "mu_mse"),
                "avg_mu_mae": _avg(rows, "mu_mae"),
                "hard_area_all_zero": _all_hard_area_zero(rows),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "mu_threshold": mu_threshold,
        "mask_temperature": mask_temperature,
        "history_rows": len(history_rows),
        "hard_area_zero": hard_area_zero,
        "soft_hard_mismatch": soft_hard_mismatch,
        "no_threshold_crossing": no_threshold_crossing,
        "soft_collapse": soft_collapse,
        "foreground_collapse": foreground_collapse,
        "final_mean_mu": _float(final_history, "mean_mu"),
        "final_min_mu": final_min_mu,
        "final_max_mu": _float(final_history, "max_mu"),
        "final_mean_soft_defect": final_mean_soft_defect,
        "final_pred_area_soft_mean": final_pred_area_soft,
        "final_true_area_mean": final_true_area,
        "final_area_loss": final_area_loss,
    }
    return summary, split_rows


def write_outputs(summary, split_rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "threshold_margin_summary.csv"
    fields = [
        "run_dir",
        "mu_threshold",
        "mask_temperature",
        "history_rows",
        "hard_area_zero",
        "soft_hard_mismatch",
        "no_threshold_crossing",
        "soft_collapse",
        "foreground_collapse",
        "final_mean_mu",
        "final_min_mu",
        "final_max_mu",
        "final_mean_soft_defect",
        "final_pred_area_soft_mean",
        "final_true_area_mean",
        "final_area_loss",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(summary)

    split_csv = output_dir / "threshold_margin_split_metrics.csv"
    split_fields = [
        "split",
        "num_rows",
        "avg_defect_iou",
        "avg_defect_area_pred",
        "avg_defect_area_label",
        "avg_mu_mse",
        "avg_mu_mae",
        "hard_area_all_zero",
    ]
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=split_fields)
        writer.writeheader()
        writer.writerows(split_rows)

    summary_md = output_dir / "summary.md"
    recommendation = (
        "需要 threshold-margin loss：soft foreground 已非零但 hard mask 仍为 0，且 mu_pred 没有跨过阈值。"
        if summary["soft_hard_mismatch"] or summary["no_threshold_crossing"]
        else "当前 run 未显示明确的 hard-threshold margin 风险。"
    )
    lines = [
        "# COMSOL threshold margin diagnostics",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- hard_area_zero: `{summary['hard_area_zero']}`",
        f"- soft_hard_mismatch: `{summary['soft_hard_mismatch']}`",
        f"- no_threshold_crossing: `{summary['no_threshold_crossing']}`",
        f"- soft_collapse: `{summary['soft_collapse']}`",
        f"- foreground_collapse: `{summary['foreground_collapse']}`",
        f"- final_min_mu: `{summary['final_min_mu']:.6e}`",
        f"- final_mean_mu: `{summary['final_mean_mu']:.6e}`",
        f"- final_mean_soft_defect: `{summary['final_mean_soft_defect']:.6e}`",
        f"- final_pred_area_soft_mean: `{summary['final_pred_area_soft_mean']:.6e}`",
        f"- final_true_area_mean: `{summary['final_true_area_mean']:.6e}`",
        "",
        "## 判断",
        "",
        recommendation,
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--mu-threshold", type=float, default=500.0)
    parser.add_argument("--mask-temperature", type=float, default=50.0)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run_dir or not args.output_dir:
        print("Diagnose hard mask / soft defect / mu-threshold margin from an existing run directory.")
        print(
            "Example: python comsol_threshold_margin_diagnostics.py "
            "--run-dir experiments/.../run --output-dir experiments/.../diagnostics"
        )
        return 0

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"run directory does not exist: {run_dir}")
    summary, split_rows = diagnose_run(run_dir, args.mu_threshold, args.mask_temperature)
    write_outputs(summary, split_rows, Path(args.output_dir))
    print(f"Wrote threshold margin diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

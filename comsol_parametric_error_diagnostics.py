"""Diagnose COMSOL parametric inverse metrics against oracle rasterization bounds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_FILES = {
    "train": "metrics.csv",
    "val": "eval_metrics.csv",
    "test": "test_metrics.csv",
}


def _read_single_row_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows[0]


def _as_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of the expected columns are present: {names}")


def _read_oracle_iou(oracle_dir: Path, split: str) -> float:
    aggregate = oracle_dir / split / "oracle_parametric_mask_aggregate.csv"
    if aggregate.exists():
        row = _read_single_row_csv(aggregate)
        return _as_float(row, "avg_oracle_iou", "oracle_mask_iou")
    metrics = oracle_dir / split / "oracle_parametric_mask_metrics.csv"
    if not metrics.exists():
        raise FileNotFoundError(f"Missing oracle metrics for split {split}: {aggregate} or {metrics}")
    with metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Oracle metrics has no rows: {metrics}")
    values = [float(row["oracle_mask_iou"]) for row in rows]
    return sum(values) / len(values)


def build_diagnostics(run_dir: Path, oracle_dir: Path, label: str) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    gap_rows: list[dict] = []
    for split, filename in METRIC_FILES.items():
        metric_path = run_dir / filename
        if not metric_path.exists():
            continue
        row = _read_single_row_csv(metric_path)
        mask_iou = _as_float(row, "param_mask_iou", "mask_iou")
        oracle_iou = _read_oracle_iou(oracle_dir, split)
        out = {
            "label": label,
            "split": split,
            "presence_acc": _as_float(row, "presence_accuracy", "presence_acc"),
            "type_acc": _as_float(row, "type_accuracy_present", "type_acc"),
            "continuous_mae": _as_float(row, "continuous_mae_mean", "continuous_mae"),
            "center_mae": _as_float(row, "center_mae"),
            "axis_mae": _as_float(row, "axis_mae"),
            "rotation_mae": _as_float(row, "rotation_mae"),
            "depth_mae": _as_float(row, "depth_mae"),
            "mask_iou": mask_iou,
            "oracle_mask_iou": oracle_iou,
            "oracle_gap": oracle_iou - mask_iou,
        }
        summary_rows.append(out)
        gap_rows.append(
            {
                "label": label,
                "split": split,
                "mask_iou": mask_iou,
                "oracle_mask_iou": oracle_iou,
                "oracle_gap": oracle_iou - mask_iou,
                "oracle_gap_fraction": (oracle_iou - mask_iou) / oracle_iou if oracle_iou else float("nan"),
            }
        )
    if not summary_rows:
        raise ValueError(f"No metrics files found in run_dir: {run_dir}")
    return summary_rows, gap_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dominant_error(row: dict) -> str:
    issues = []
    if row["type_acc"] < 0.8:
        issues.append("type")
    if row["rotation_mae"] > 5.0:
        issues.append("rotation")
    if row["oracle_gap"] > 0.25:
        issues.append("oracle_gap")
    if row["center_mae"] > 0.002:
        issues.append("center")
    if row["axis_mae"] > 0.001:
        issues.append("axis")
    return ", ".join(issues) if issues else "no_single_dominant_error"


def write_summary(path: Path, label: str, run_dir: Path, oracle_dir: Path, rows: list[dict]) -> None:
    lines = [
        f"# {label} parametric error diagnostics",
        "",
        f"- run_dir: `{run_dir}`",
        f"- oracle_dir: `{oracle_dir}`",
        "",
        "## Split summary",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['split']}: mask_iou={row['mask_iou']:.6e}, "
            f"oracle_iou={row['oracle_mask_iou']:.6e}, oracle_gap={row['oracle_gap']:.6e}, "
            f"type_acc={row['type_acc']:.6e}, rotation_mae={row['rotation_mae']:.6e}, "
            f"dominant={_dominant_error(row)}"
        )
    val_test = [row for row in rows if row["split"] in {"val", "test"}]
    avg_gap = sum(row["oracle_gap"] for row in val_test) / len(val_test) if val_test else float("nan")
    avg_type = sum(row["type_acc"] for row in val_test) / len(val_test) if val_test else float("nan")
    avg_rotation = sum(row["rotation_mae"] for row in val_test) / len(val_test) if val_test else float("nan")
    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            f"- val/test avg oracle gap: `{avg_gap:.6e}`。",
            f"- val/test avg type accuracy: `{avg_type:.6e}`。",
            f"- val/test avg rotation MAE: `{avg_rotation:.6e}` degree。",
            "- 当前 run 目录没有 per-sample predictions，因此本诊断只做 aggregate decomposition；未伪造 type/rotation bins。",
            "- 如果 oracle gap 大且 type/rotation 同时偏弱，下一步优先改 head / encoder / loss 分解，而不是修 target/mask schema。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    run_dir = Path(args.run_dir)
    oracle_dir = Path(args.oracle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, gap_rows = build_diagnostics(run_dir, oracle_dir, args.label)
    _write_csv(output_dir / "parametric_error_summary.csv", rows)
    _write_csv(output_dir / "oracle_gap_summary.csv", gap_rows)
    write_summary(output_dir / "summary.md", args.label, run_dir, oracle_dir, rows)
    print(f"Saved parametric error diagnostics to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--oracle-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--label", default="")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not all([args.run_dir, args.oracle_dir, args.output_dir, args.label]):
        parser.print_help()
        print(
            "\nExample: python comsol_parametric_error_diagnostics.py "
            "--run-dir run --oracle-dir oracle --output-dir out --label s115_raw"
        )
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

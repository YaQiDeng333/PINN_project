"""Run parameter-level oracle ablations for COMSOL parametric predictions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from comsol_parametric_rasterizer import mask_iou_dice, rasterize_components, write_csv


RAW_SCHEMA = [
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "depth_or_shape_param",
    "rotation_angle",
]

RASTER_SCHEMA = np.array(
    [
        "center_x",
        "center_y",
        "axis_x",
        "axis_y",
        "depth_or_shape_param",
        "rotation_sin",
        "rotation_cos",
    ],
    dtype="U64",
)

VARIANTS = [
    "pred_all",
    "gt_type",
    "gt_rotation",
    "gt_type_rotation",
    "gt_center",
    "gt_axis",
    "gt_depth",
    "gt_continuous_all",
    "gt_type_continuous",
    "gt_all",
]

REQUIRED_PREDICTION_FIELDS = {
    "sample_index",
    "component_slot",
    "matched_slot",
    "presence_true",
    "presence_pred",
    "type_true",
    "type_pred",
    "center_x_true",
    "center_x_pred",
    "center_y_true",
    "center_y_pred",
    "axis_x_true",
    "axis_x_pred",
    "axis_y_true",
    "axis_y_pred",
    "depth_true",
    "depth_pred",
    "rotation_true",
    "rotation_pred",
    "target_schema",
    "type_vocab",
}


@dataclass(frozen=True)
class TargetData:
    masks: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sample_indices: np.ndarray
    presence: np.ndarray
    type_targets: np.ndarray
    continuous_raw: np.ndarray
    type_vocab: list[str]


@dataclass(frozen=True)
class PredictionData:
    presence: np.ndarray
    type_targets: np.ndarray
    continuous_raw: np.ndarray


def _str_list(values) -> list[str]:
    return [str(v.decode("utf-8") if isinstance(v, bytes) else v) for v in values]


def _split_pipe(value: str) -> list[str]:
    return [part for part in str(value).split("|") if part]


def _as_float(row: dict, field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse numeric CSV field {field!r}: {row.get(field)!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite CSV field {field!r}: {row.get(field)!r}")
    return value


def _as_int(row: dict, field: str) -> int:
    try:
        return int(float(row[field]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse integer CSV field {field!r}: {row.get(field)!r}") from exc


def _continuous_to_degree_raw(continuous: np.ndarray, target_schema) -> np.ndarray:
    """Convert supported target schemas to RAW_SCHEMA without degree/radian heuristics."""

    schema = _str_list(target_schema)
    raw = np.zeros((*continuous.shape[:2], len(RAW_SCHEMA)), dtype=np.float32)
    for name in ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param"]:
        if name not in schema:
            raise ValueError(f"target_schema missing required field: {name}")
        raw[:, :, RAW_SCHEMA.index(name)] = continuous[:, :, schema.index(name)]

    if "rotation_angle" in schema:
        raw[:, :, RAW_SCHEMA.index("rotation_angle")] = continuous[:, :, schema.index("rotation_angle")]
    elif "rotation_sin" in schema and "rotation_cos" in schema:
        sin_v = continuous[:, :, schema.index("rotation_sin")]
        cos_v = continuous[:, :, schema.index("rotation_cos")]
        raw[:, :, RAW_SCHEMA.index("rotation_angle")] = np.rad2deg(np.arctan2(sin_v, cos_v))
    else:
        raise ValueError("target_schema missing rotation_angle or rotation_sin/rotation_cos.")
    return raw.astype(np.float32)


def _degree_raw_to_raster_continuous(continuous_raw: np.ndarray) -> np.ndarray:
    """Encode degree rotation as sin/cos so the hard rasterizer skips its angle heuristic."""

    raster = np.zeros((*continuous_raw.shape[:2], len(RASTER_SCHEMA)), dtype=np.float32)
    raster[:, :, :5] = continuous_raw[:, :, :5]
    theta = np.deg2rad(continuous_raw[:, :, RAW_SCHEMA.index("rotation_angle")])
    raster[:, :, 5] = np.sin(theta)
    raster[:, :, 6] = np.cos(theta)
    return raster


def load_target_data(npz_path: Path, targets_path: Path, max_components: int) -> TargetData:
    with np.load(npz_path, allow_pickle=True) as data:
        for key in ["masks", "x", "y"]:
            if key not in data:
                raise ValueError(f"{npz_path} missing required field: {key}")
        masks = data["masks"].astype(np.float32)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)

    with np.load(targets_path, allow_pickle=True) as data:
        for key in ["presence_targets", "type_targets", "continuous_targets", "target_schema", "type_vocab"]:
            if key not in data:
                raise ValueError(f"{targets_path} missing required field: {key}")
        continuous = (
            data["continuous_targets_raw"].astype(np.float32)
            if "continuous_targets_raw" in data
            else data["continuous_targets"].astype(np.float32)
        )
        target_schema = data["raw_target_schema"] if "raw_target_schema" in data else data["target_schema"]
        sample_indices = (
            data["sample_indices"].astype(np.int64)
            if "sample_indices" in data
            else np.arange(continuous.shape[0], dtype=np.int64)
        )
        target = TargetData(
            masks=masks,
            x=x,
            y=y,
            sample_indices=sample_indices,
            presence=data["presence_targets"].astype(np.float32),
            type_targets=data["type_targets"].astype(np.int64),
            continuous_raw=_continuous_to_degree_raw(continuous, target_schema),
            type_vocab=_str_list(data["type_vocab"]),
        )

    if target.masks.shape[0] != target.continuous_raw.shape[0]:
        raise ValueError("NPZ masks and parametric targets sample counts do not match.")
    if target.presence.shape != target.type_targets.shape or target.presence.shape != target.continuous_raw.shape[:2]:
        raise ValueError("Target presence/type/continuous component dimensions do not align.")
    if target.presence.shape[1] != max_components:
        raise ValueError(f"Expected max_components={max_components}, got targets K={target.presence.shape[1]}.")
    return target


def _assert_close(field: str, csv_value: float, target_value: float, *, atol: float = 1e-5) -> None:
    if not np.isclose(csv_value, float(target_value), rtol=1e-4, atol=atol):
        raise ValueError(f"CSV true field {field}={csv_value} does not match target value {target_value}.")


def load_prediction_data(predictions_csv: Path, target: TargetData, split: str) -> PredictionData:
    with predictions_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_PREDICTION_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{predictions_csv} missing required fields: {sorted(missing)}")
        rows = list(reader)

    n_samples, max_components = target.presence.shape
    if len(rows) != n_samples * max_components:
        raise ValueError(
            f"Expected {n_samples * max_components} prediction rows for {n_samples} samples and "
            f"{max_components} components, got {len(rows)}."
        )

    type_to_index = {name: idx for idx, name in enumerate(target.type_vocab)}
    sample_to_pos = {int(sample_index): pos for pos, sample_index in enumerate(target.sample_indices)}
    presence = np.zeros((n_samples, max_components), dtype=np.float32)
    type_targets = np.full((n_samples, max_components), -1, dtype=np.int64)
    continuous = np.zeros((n_samples, max_components, len(RAW_SCHEMA)), dtype=np.float32)
    seen: set[tuple[int, int]] = set()

    expected_schema = "|".join(RAW_SCHEMA)
    expected_vocab = "|".join(target.type_vocab)
    for row in rows:
        if row.get("split") and str(row["split"]) != split:
            raise ValueError(f"Prediction row split {row['split']!r} does not match requested split {split!r}.")
        sample_index = _as_int(row, "sample_index")
        if sample_index not in sample_to_pos:
            raise ValueError(f"Prediction sample_index={sample_index} is not present in targets.")
        sample_pos = sample_to_pos[sample_index]
        slot = _as_int(row, "component_slot")
        matched_slot = _as_int(row, "matched_slot")
        if slot != matched_slot:
            raise ValueError(
                "This oracle ablation expects fixed-order exports; "
                f"component_slot={slot} but matched_slot={matched_slot}."
            )
        if slot < 0 or slot >= max_components:
            raise ValueError(f"component_slot={slot} out of range for max_components={max_components}.")
        key = (sample_pos, slot)
        if key in seen:
            raise ValueError(f"Duplicate prediction row for sample_index={sample_index}, component_slot={slot}.")
        seen.add(key)

        if _split_pipe(row["target_schema"]) != RAW_SCHEMA:
            raise ValueError(f"Unsupported prediction target_schema: {row['target_schema']!r}; expected {expected_schema!r}.")
        if _split_pipe(row["type_vocab"]) != target.type_vocab:
            raise ValueError(f"Prediction type_vocab {row['type_vocab']!r} does not match target {expected_vocab!r}.")

        presence[sample_pos, slot] = float(_as_int(row, "presence_pred"))
        type_name = str(row["type_pred"])
        if type_name not in type_to_index:
            raise ValueError(f"Unknown type_pred {type_name!r}; expected one of {target.type_vocab}.")
        type_targets[sample_pos, slot] = type_to_index[type_name]

        true_type_name = str(row["type_true"])
        expected_type_id = int(target.type_targets[sample_pos, slot])
        expected_type_name = target.type_vocab[expected_type_id] if expected_type_id >= 0 else ""
        if true_type_name != expected_type_name:
            raise ValueError(
                f"CSV type_true {true_type_name!r} does not match target type {expected_type_name!r} "
                f"for sample_index={sample_index}, slot={slot}."
            )
        if _as_int(row, "presence_true") != int(target.presence[sample_pos, slot] > 0.5):
            raise ValueError(f"CSV presence_true mismatch for sample_index={sample_index}, slot={slot}.")

        aliases = [
            ("center_x", "center_x"),
            ("center_y", "center_y"),
            ("axis_x", "axis_x"),
            ("axis_y", "axis_y"),
            ("depth", "depth_or_shape_param"),
            ("rotation", "rotation_angle"),
        ]
        for csv_prefix, schema_name in aliases:
            schema_idx = RAW_SCHEMA.index(schema_name)
            _assert_close(
                f"{csv_prefix}_true",
                _as_float(row, f"{csv_prefix}_true"),
                target.continuous_raw[sample_pos, slot, schema_idx],
            )
            continuous[sample_pos, slot, schema_idx] = _as_float(row, f"{csv_prefix}_pred")

    if len(seen) != n_samples * max_components:
        raise ValueError("Prediction CSV does not contain a complete sample/component grid.")
    return PredictionData(presence=presence, type_targets=type_targets, continuous_raw=continuous)


def make_variant(variant: str, pred: PredictionData, target: TargetData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported ablation variant: {variant}")
    presence = pred.presence.copy()
    type_targets = pred.type_targets.copy()
    continuous = pred.continuous_raw.copy()

    if variant in {"gt_type", "gt_type_rotation", "gt_type_continuous", "gt_all"}:
        type_targets = target.type_targets.copy()
    if variant in {"gt_rotation", "gt_type_rotation"}:
        continuous[:, :, RAW_SCHEMA.index("rotation_angle")] = target.continuous_raw[:, :, RAW_SCHEMA.index("rotation_angle")]
    if variant == "gt_center":
        continuous[:, :, [RAW_SCHEMA.index("center_x"), RAW_SCHEMA.index("center_y")]] = target.continuous_raw[
            :, :, [RAW_SCHEMA.index("center_x"), RAW_SCHEMA.index("center_y")]
        ]
    if variant == "gt_axis":
        continuous[:, :, [RAW_SCHEMA.index("axis_x"), RAW_SCHEMA.index("axis_y")]] = target.continuous_raw[
            :, :, [RAW_SCHEMA.index("axis_x"), RAW_SCHEMA.index("axis_y")]
        ]
    if variant == "gt_depth":
        continuous[:, :, RAW_SCHEMA.index("depth_or_shape_param")] = target.continuous_raw[
            :, :, RAW_SCHEMA.index("depth_or_shape_param")
        ]
    if variant in {"gt_continuous_all", "gt_type_continuous", "gt_all"}:
        continuous = target.continuous_raw.copy()
    if variant == "gt_all":
        presence = target.presence.copy()
    return presence, type_targets, continuous


def score_variants(pred: PredictionData, target: TargetData, split: str) -> tuple[list[dict], list[dict]]:
    target_masks = target.masks > 0.5
    target_areas = target_masks.sum(axis=(1, 2)).astype(np.int64)
    rows: list[dict] = []
    aggregate_base: dict[str, dict] = {}

    for variant in VARIANTS:
        presence, type_targets, continuous_raw = make_variant(variant, pred, target)
        masks = rasterize_components(
            _degree_raw_to_raster_continuous(continuous_raw),
            type_targets,
            presence,
            RASTER_SCHEMA,
            target.type_vocab,
            target.x,
            target.y,
        )
        ious, dices = mask_iou_dice(masks, target_masks)
        pred_areas = masks.sum(axis=(1, 2)).astype(np.int64)
        for sample_pos, sample_index in enumerate(target.sample_indices):
            rows.append(
                {
                    "split": split,
                    "sample_index": int(sample_index),
                    "variant": variant,
                    "mask_iou": float(ious[sample_pos]),
                    "dice": float(dices[sample_pos]),
                    "pred_area": int(pred_areas[sample_pos]),
                    "target_area": int(target_areas[sample_pos]),
                    "area_diff": int(pred_areas[sample_pos] - target_areas[sample_pos]),
                }
            )
        aggregate_base[variant] = {
            "split": split,
            "variant": variant,
            "samples": int(len(ious)),
            "avg_mask_iou": float(np.mean(ious)),
            "avg_dice": float(np.mean(dices)),
            "avg_pred_area": float(np.mean(pred_areas)),
            "avg_target_area": float(np.mean(target_areas)),
            "avg_abs_area_diff": float(np.mean(np.abs(pred_areas - target_areas))),
        }

    pred_iou = aggregate_base["pred_all"]["avg_mask_iou"]
    oracle_iou = aggregate_base["gt_all"]["avg_mask_iou"]
    aggregate_rows = []
    for variant in VARIANTS:
        row = dict(aggregate_base[variant])
        row["delta_iou_vs_pred_all"] = float(row["avg_mask_iou"] - pred_iou)
        row["delta_iou_vs_oracle"] = float(row["avg_mask_iou"] - oracle_iou)
        row["oracle_gap"] = float(oracle_iou - row["avg_mask_iou"])
        aggregate_rows.append(row)
    return rows, aggregate_rows


def _format_metric(value: float) -> str:
    return f"{value:.6e}"


def write_summary(path: Path, split: str, aggregate_rows: list[dict]) -> None:
    by_variant = {row["variant"]: row for row in aggregate_rows}
    single_variants = ["gt_type", "gt_rotation", "gt_center", "gt_axis", "gt_depth"]
    best_single = max(single_variants, key=lambda name: by_variant[name]["delta_iou_vs_pred_all"])
    best_overall = max(
        [name for name in VARIANTS if name not in {"pred_all", "gt_all"}],
        key=lambda name: by_variant[name]["delta_iou_vs_pred_all"],
    )
    type_delta = by_variant["gt_type"]["delta_iou_vs_pred_all"]
    rotation_delta = by_variant["gt_rotation"]["delta_iou_vs_pred_all"]
    gt_all_iou = by_variant["gt_all"]["avg_mask_iou"]
    gate_note = "接近 S117 oracle gate" if gt_all_iou >= 0.70 else "低于 0.70，需优先检查对齐或 rasterizer 语义"
    type_note = (
        "当前 hard rasterizer 将 `rectangular_notch` 和 `rotated_rect` 都近似为 rotated rectangle；"
        "因此单独替换 type 通常不会改变 mask。"
    )
    lines = [
        f"# S157 oracle ablation split summary: {split}",
        "",
        "## Aggregate IoU",
        "",
        "| variant | avg_mask_iou | delta_vs_pred_all | oracle_gap | avg_abs_area_diff |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {variant} | {iou} | {delta} | {gap} | {area} |".format(
                variant=row["variant"],
                iou=_format_metric(row["avg_mask_iou"]),
                delta=_format_metric(row["delta_iou_vs_pred_all"]),
                gap=_format_metric(row["oracle_gap"]),
                area=_format_metric(row["avg_abs_area_diff"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `pred_all` IoU: `{_format_metric(by_variant['pred_all']['avg_mask_iou'])}`.",
            f"- `gt_all` IoU: `{_format_metric(gt_all_iou)}`; {gate_note}。",
            f"- 单项替换最大提升: `{best_single}`，delta `{_format_metric(by_variant[best_single]['delta_iou_vs_pred_all'])}`。",
            f"- 非 full-oracle 最大提升: `{best_overall}`，delta `{_format_metric(by_variant[best_overall]['delta_iou_vs_pred_all'])}`。",
            f"- `gt_type` delta: `{_format_metric(type_delta)}`。{type_note}",
            f"- `gt_rotation` delta: `{_format_metric(rotation_delta)}`。",
            "",
            "## Self-review",
            "",
            "- 本脚本只读取已有 predictions / targets / masks，不训练模型，不保存权重、checkpoint 或图片。",
            "- rotation 按 raw degree 语义处理，并在 rasterization 前转为 sin/cos schema，避免 hard rasterizer 的 degree/radian heuristic。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = load_target_data(Path(args.npz_path), Path(args.targets_path), args.max_components)
    pred = load_prediction_data(Path(args.predictions_csv), target, args.split)
    per_sample_rows, aggregate_rows = score_variants(pred, target, args.split)
    write_csv(output_dir / "per_sample_oracle_ablation.csv", per_sample_rows)
    write_csv(output_dir / "aggregate_oracle_ablation.csv", aggregate_rows)
    write_summary(output_dir / "summary.md", args.split, aggregate_rows)
    print(f"Saved COMSOL parametric oracle ablation results to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--targets-path", default="")
    parser.add_argument("--predictions-csv", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--max-components", type=int, default=3)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.npz_path or not args.targets_path or not args.predictions_csv or not args.output_dir or not args.split:
        parser.print_help()
        print(
            "\nExample: python comsol_parametric_oracle_ablation.py "
            "--npz-path val.npz --targets-path parametric_targets.npz "
            "--predictions-csv val_predictions.csv --output-dir out --split val"
        )
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

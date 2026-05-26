"""Smoke test for the minimal conditional supervised training runner."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def _write_npz(path: Path, samples: int, seed: int, multi_channel: bool = False):
    x = np.linspace(-15.0, 15.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 10.0, 10, dtype=np.float32)
    signal_shape = (samples, 3, 20) if multi_channel else (samples, 20)
    signals = np.random.default_rng(seed).normal(size=signal_shape).astype(np.float32)
    mu_maps = np.full((samples, 10, 20), 1000.0, dtype=np.float32)
    mu_maps[:, 4:6, 8:12] = 1.0
    masks = (mu_maps < 500.0).astype(np.float32)
    np.savez(path, x=x, y=y, signals=signals, mu_maps=mu_maps, masks=masks)


def _assert_no_forbidden_files(output_dir: Path):
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".npy"}
    found = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if found:
        raise AssertionError(f"forbidden files were generated: {found}")


def _run_runner_variant(repo_root: Path, npz_path: Path, eval_npz_path: Path, test_npz_path: Path, output_dir: Path, extra_args, summary_terms):
    cmd = [
        sys.executable,
        str(repo_root / "train_conditional_dual.py"),
        "--npz-path",
        str(npz_path),
        "--eval-npz-path",
        str(eval_npz_path),
        "--test-npz-path",
        str(test_npz_path),
        "--output-dir",
        str(output_dir),
        "--sample-indices",
        "0,1,2",
        "--eval-sample-indices",
        "0,1",
        "--test-sample-indices",
        "0,1",
        "--steps",
        "5",
        "--hidden-dim",
        "32",
        "--num-layers",
        "2",
        "--latent-dim",
        "16",
    ] + list(extra_args)
    result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise AssertionError(f"runner variant failed with return code {result.returncode}: {extra_args}")
    for filename in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv", "run_summary.md"]:
        if not (output_dir / filename).exists():
            raise AssertionError(f"variant {filename} was not created for {extra_args}")
    summary_text = (output_dir / "run_summary.md").read_text(encoding="utf-8")
    for term in summary_terms:
        if term not in summary_text:
            raise AssertionError(f"variant run_summary.md is missing {term}")
    _assert_no_forbidden_files(output_dir)


def main():
    repo_root = Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz_path = tmp_path / "conditional_train.npz"
        eval_npz_path = tmp_path / "conditional_eval.npz"
        test_npz_path = tmp_path / "conditional_test.npz"
        output_dir = tmp_path / "runner_output"
        _write_npz(npz_path, samples=4, seed=50)
        _write_npz(eval_npz_path, samples=2, seed=51)
        _write_npz(test_npz_path, samples=2, seed=52)

        cmd = [
            sys.executable,
            str(repo_root / "train_conditional_dual.py"),
            "--npz-path",
            str(npz_path),
            "--eval-npz-path",
            str(eval_npz_path),
            "--test-npz-path",
            str(test_npz_path),
            "--output-dir",
            str(output_dir),
            "--sample-indices",
            "0,1,2",
            "--eval-sample-indices",
            "0,1",
            "--test-sample-indices",
            "0,1",
            "--signal-ablation",
            "--signal-normalization",
            "train_zscore",
            "--signal-feature-mode",
            "raw_abs_grad",
            "--conditioning-mode",
            "concat",
            "--encoder-type",
            "cnn",
            "--point-signal-mode",
            "local_value",
            "--mask-head-mode",
            "direct",
            "--mask-source",
            "masks",
            "--train-point-subsample",
            "50",
            "--steps",
            "5",
            "--hidden-dim",
            "32",
            "--num-layers",
            "2",
            "--latent-dim",
            "16",
        ]
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise AssertionError(f"train_conditional_dual.py failed with return code {result.returncode}")

        metrics_path = output_dir / "metrics.csv"
        eval_metrics_path = output_dir / "eval_metrics.csv"
        test_metrics_path = output_dir / "test_metrics.csv"
        eval_zero_path = output_dir / "eval_metrics_zero_signal.csv"
        eval_shuffled_path = output_dir / "eval_metrics_shuffled_signal.csv"
        test_zero_path = output_dir / "test_metrics_zero_signal.csv"
        test_shuffled_path = output_dir / "test_metrics_shuffled_signal.csv"
        summary_path = output_dir / "run_summary.md"
        if not metrics_path.exists():
            raise AssertionError("metrics.csv was not created")
        if not eval_metrics_path.exists():
            raise AssertionError("eval_metrics.csv was not created")
        if not test_metrics_path.exists():
            raise AssertionError("test_metrics.csv was not created")
        for path in [eval_zero_path, eval_shuffled_path, test_zero_path, test_shuffled_path]:
            if not path.exists():
                raise AssertionError(f"{path.name} was not created")
        if not summary_path.exists():
            raise AssertionError("run_summary.md was not created")

        with metrics_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 3:
            raise AssertionError(f"metrics.csv should contain 3 rows, got {len(rows)}")
        with eval_metrics_path.open(newline="", encoding="utf-8") as f:
            eval_rows = list(csv.DictReader(f))
        if len(eval_rows) != 2:
            raise AssertionError(f"eval_metrics.csv should contain 2 rows, got {len(eval_rows)}")
        with test_metrics_path.open(newline="", encoding="utf-8") as f:
            test_rows = list(csv.DictReader(f))
        if len(test_rows) != 2:
            raise AssertionError(f"test_metrics.csv should contain 2 rows, got {len(test_rows)}")
        for column in ["defect_iou", "mu_mse", "mu_mae"]:
            if column not in rows[0]:
                raise AssertionError(f"metrics.csv is missing column {column}")
            if column not in eval_rows[0]:
                raise AssertionError(f"eval_metrics.csv is missing column {column}")
            if column not in test_rows[0]:
                raise AssertionError(f"test_metrics.csv is missing column {column}")
        if "mask_bce_mode" not in rows[0]:
            raise AssertionError("metrics.csv is missing column mask_bce_mode")
        if "area_loss_mode" not in rows[0]:
            raise AssertionError("metrics.csv is missing column area_loss_mode")
        if "lambda_area_loss" not in rows[0]:
            raise AssertionError("metrics.csv is missing column lambda_area_loss")
        if "threshold_margin_mode" not in rows[0]:
            raise AssertionError("metrics.csv is missing column threshold_margin_mode")
        if "lambda_threshold_margin" not in rows[0]:
            raise AssertionError("metrics.csv is missing column lambda_threshold_margin")
        if rows[0].get("signal_normalization") != "train_zscore":
            raise AssertionError("metrics.csv did not record signal_normalization=train_zscore")
        if rows[0].get("signal_feature_mode") != "raw_abs_grad":
            raise AssertionError("metrics.csv did not record signal_feature_mode=raw_abs_grad")
        if rows[0].get("conditioning_mode") != "concat":
            raise AssertionError("metrics.csv did not record conditioning_mode=concat")
        if rows[0].get("encoder_type") != "cnn":
            raise AssertionError("metrics.csv did not record encoder_type=cnn")
        if rows[0].get("point_signal_mode") != "local_value":
            raise AssertionError("metrics.csv did not record point_signal_mode=local_value")
        if rows[0].get("mask_head_mode") != "direct":
            raise AssertionError("metrics.csv did not record mask_head_mode=direct")
        if rows[0].get("mask_source") != "masks":
            raise AssertionError("metrics.csv did not record mask_source=masks")
        if rows[0].get("mask_bce_mode") != "bce":
            raise AssertionError("metrics.csv did not record mask_bce_mode=bce")
        if rows[0].get("area_loss_mode") != "none":
            raise AssertionError("metrics.csv did not record area_loss_mode=none")
        summary_text = summary_path.read_text(encoding="utf-8")
        if "signal_normalization" not in summary_text:
            raise AssertionError("run_summary.md is missing signal_normalization")
        if "signal_feature_mode" not in summary_text:
            raise AssertionError("run_summary.md is missing signal_feature_mode")
        if "encoder_input_length" not in summary_text:
            raise AssertionError("run_summary.md is missing encoder_input_length")
        if "conditioning_mode" not in summary_text:
            raise AssertionError("run_summary.md is missing conditioning_mode")
        if "encoder_type" not in summary_text:
            raise AssertionError("run_summary.md is missing encoder_type")
        if "point_signal_mode" not in summary_text:
            raise AssertionError("run_summary.md is missing point_signal_mode")
        if "mask_head_mode" not in summary_text:
            raise AssertionError("run_summary.md is missing mask_head_mode")
        if "mask_source" not in summary_text:
            raise AssertionError("run_summary.md is missing mask_source")
        if "train_point_subsample" not in summary_text:
            raise AssertionError("run_summary.md is missing train_point_subsample")
        if "mask_bce_mode" not in summary_text:
            raise AssertionError("run_summary.md is missing mask_bce_mode")
        if "point_sampling_mode" not in summary_text:
            raise AssertionError("run_summary.md is missing point_sampling_mode")
        if "area_loss_mode" not in summary_text:
            raise AssertionError("run_summary.md is missing area_loss_mode")
        if "val_selection_metric" not in summary_text:
            raise AssertionError("run_summary.md is missing val_selection_metric")

        _assert_no_forbidden_files(output_dir)

        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            tmp_path / "runner_pos_weighted_output",
            ["--mask-bce-mode", "pos_weighted_bce", "--pos-weight", "5"],
            ["mask_bce_mode", "pos_weight"],
        )
        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            tmp_path / "runner_focal_output",
            ["--mask-bce-mode", "focal_bce", "--focal-gamma", "2", "--focal-alpha", "0.25"],
            ["mask_bce_mode", "focal_gamma", "focal_alpha"],
        )
        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            tmp_path / "runner_positive_balanced_output",
            [
                "--train-point-subsample",
                "50",
                "--point-sampling-mode",
                "positive_balanced",
                "--positive-fraction",
                "0.5",
            ],
            ["train_point_subsample", "point_sampling_mode", "positive_fraction"],
        )

        history_output_dir = tmp_path / "runner_history_output"
        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            history_output_dir,
            ["--history-interval", "2"],
            ["history_interval"],
        )
        history_path = history_output_dir / "training_history.csv"
        if not history_path.exists():
            raise AssertionError("training_history.csv was not created")
        with history_path.open(newline="", encoding="utf-8") as f:
            history_rows = list(csv.DictReader(f))
        if len(history_rows) < 2:
            raise AssertionError("training_history.csv should contain at least 2 rows")
        for column in ["phase", "step", "total_loss", "batch_iou"]:
            if column not in history_rows[0]:
                raise AssertionError(f"training_history.csv is missing column {column}")
        _assert_no_forbidden_files(history_output_dir)

        pretrain_npz_path = tmp_path / "conditional_pretrain.npz"
        pretrain_output_dir = tmp_path / "runner_pretrain_output"
        _write_npz(pretrain_npz_path, samples=4, seed=53)
        pretrain_cmd = [
            sys.executable,
            str(repo_root / "train_conditional_dual.py"),
            "--npz-path",
            str(npz_path),
            "--eval-npz-path",
            str(eval_npz_path),
            "--test-npz-path",
            str(test_npz_path),
            "--pretrain-npz-path",
            str(pretrain_npz_path),
            "--pretrain-sample-indices",
            "0,1,2",
            "--pretrain-steps",
            "3",
            "--output-dir",
            str(pretrain_output_dir),
            "--sample-indices",
            "0,1,2",
            "--eval-sample-indices",
            "0,1",
            "--test-sample-indices",
            "0,1",
            "--steps",
            "5",
            "--history-interval",
            "1",
            "--hidden-dim",
            "32",
            "--num-layers",
            "2",
            "--latent-dim",
            "16",
        ]
        pretrain_result = subprocess.run(pretrain_cmd, cwd=repo_root, text=True, capture_output=True, check=False)
        if pretrain_result.returncode != 0:
            print(pretrain_result.stdout)
            print(pretrain_result.stderr)
            raise AssertionError(f"pretrain curriculum run failed with return code {pretrain_result.returncode}")
        for filename in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv", "run_summary.md", "training_history.csv"]:
            if not (pretrain_output_dir / filename).exists():
                raise AssertionError(f"pretrain curriculum {filename} was not created")
        with (pretrain_output_dir / "training_history.csv").open(newline="", encoding="utf-8") as f:
            pretrain_history_rows = list(csv.DictReader(f))
        phases = {row["phase"] for row in pretrain_history_rows}
        if phases != {"pretrain", "finetune"}:
            raise AssertionError(f"training_history.csv phases should be pretrain and finetune, got {phases}")
        pretrain_summary = (pretrain_output_dir / "run_summary.md").read_text(encoding="utf-8")
        for required_text in ["pretrain_npz_path", "pretrain_steps", "pretrain_sample_indices_count"]:
            if required_text not in pretrain_summary:
                raise AssertionError(f"pretrain run_summary.md is missing {required_text}")
        _assert_no_forbidden_files(pretrain_output_dir)

        area_ratio_output_dir = tmp_path / "runner_area_ratio_output"
        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            area_ratio_output_dir,
            ["--lambda-area-loss", "1.0", "--area-loss-mode", "batch_ratio_mse", "--history-interval", "1"],
            ["area_loss_mode", "lambda_area_loss"],
        )
        area_history_path = area_ratio_output_dir / "training_history.csv"
        if not area_history_path.exists():
            raise AssertionError("batch_ratio_mse training_history.csv was not created")
        with area_history_path.open(newline="", encoding="utf-8") as f:
            area_history_rows = list(csv.DictReader(f))
        for column in ["area_loss", "pred_area_soft_mean", "true_area_mean"]:
            if column not in area_history_rows[0]:
                raise AssertionError(f"training_history.csv is missing {column}")
        _assert_no_forbidden_files(area_ratio_output_dir)

        floor_output_dir = tmp_path / "runner_foreground_floor_output"
        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            floor_output_dir,
            [
                "--lambda-area-loss",
                "1.0",
                "--area-loss-mode",
                "foreground_floor",
                "--foreground-floor-ratio",
                "0.5",
                "--train-point-subsample",
                "50",
                "--point-sampling-mode",
                "positive_balanced",
                "--history-interval",
                "1",
            ],
            ["area_loss_mode", "lambda_area_loss", "foreground_floor_ratio"],
        )
        floor_history_path = floor_output_dir / "training_history.csv"
        if not floor_history_path.exists():
            raise AssertionError("foreground_floor training_history.csv was not created")
        with floor_history_path.open(newline="", encoding="utf-8") as f:
            floor_history_rows = list(csv.DictReader(f))
        for column in ["area_loss", "pred_area_soft_mean", "true_area_mean"]:
            if column not in floor_history_rows[0]:
                raise AssertionError(f"training_history.csv is missing {column} for foreground_floor")
        _assert_no_forbidden_files(floor_output_dir)

        positive_margin_output_dir = tmp_path / "runner_positive_margin_output"
        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            positive_margin_output_dir,
            [
                "--lambda-threshold-margin",
                "1.0",
                "--threshold-margin-mode",
                "positive_hinge",
                "--positive-mu-margin",
                "50",
                "--history-interval",
                "1",
            ],
            ["threshold_margin_mode", "lambda_threshold_margin", "positive_mu_margin"],
        )
        positive_margin_history_path = positive_margin_output_dir / "training_history.csv"
        if not positive_margin_history_path.exists():
            raise AssertionError("positive_hinge training_history.csv was not created")
        with positive_margin_history_path.open(newline="", encoding="utf-8") as f:
            positive_margin_history_rows = list(csv.DictReader(f))
        for column in [
            "threshold_margin_loss",
            "positive_margin_loss",
            "sampled_positive_count",
            "sampled_mu_positive_mean",
        ]:
            if column not in positive_margin_history_rows[0]:
                raise AssertionError(f"training_history.csv is missing {column} for positive_hinge")
        _assert_no_forbidden_files(positive_margin_output_dir)

        bidirectional_margin_output_dir = tmp_path / "runner_bidirectional_margin_output"
        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            bidirectional_margin_output_dir,
            [
                "--lambda-threshold-margin",
                "1.0",
                "--threshold-margin-mode",
                "bidirectional_hinge",
                "--positive-mu-margin",
                "50",
                "--negative-mu-margin",
                "50",
                "--history-interval",
                "1",
            ],
            [
                "threshold_margin_mode",
                "lambda_threshold_margin",
                "positive_mu_margin",
                "negative_mu_margin",
            ],
        )
        bidirectional_margin_history_path = bidirectional_margin_output_dir / "training_history.csv"
        if not bidirectional_margin_history_path.exists():
            raise AssertionError("bidirectional_hinge training_history.csv was not created")
        with bidirectional_margin_history_path.open(newline="", encoding="utf-8") as f:
            bidirectional_margin_history_rows = list(csv.DictReader(f))
        for column in [
            "threshold_margin_loss",
            "positive_margin_loss",
            "negative_margin_loss",
            "sampled_negative_count",
            "sampled_mu_negative_mean",
        ]:
            if column not in bidirectional_margin_history_rows[0]:
                raise AssertionError(f"training_history.csv is missing {column} for bidirectional_hinge")
        _assert_no_forbidden_files(bidirectional_margin_output_dir)

        val_selection_output_dir = tmp_path / "runner_val_selection_output"
        _run_runner_variant(
            repo_root,
            npz_path,
            eval_npz_path,
            test_npz_path,
            val_selection_output_dir,
            [
                "--val-selection-metric",
                "eval_iou",
                "--val-selection-interval",
                "2",
                "--history-interval",
                "1",
            ],
            ["val_selection_metric", "val_selection_interval", "best_step"],
        )
        val_selection_history_path = val_selection_output_dir / "training_history.csv"
        if not val_selection_history_path.exists():
            raise AssertionError("val-selection training_history.csv was not created")
        with val_selection_history_path.open(newline="", encoding="utf-8") as f:
            val_selection_history_rows = list(csv.DictReader(f))
        for column in ["eval_iou_at_step", "eval_loss_at_step", "is_best_step"]:
            if column not in val_selection_history_rows[0]:
                raise AssertionError(f"training_history.csv is missing {column} for validation selection")
        if not any(row["is_best_step"] == "True" for row in val_selection_history_rows):
            raise AssertionError("validation selection did not mark any best step")
        _assert_no_forbidden_files(val_selection_output_dir)

        multi_train_npz = tmp_path / "conditional_multi_train.npz"
        multi_eval_npz = tmp_path / "conditional_multi_eval.npz"
        multi_test_npz = tmp_path / "conditional_multi_test.npz"
        multi_output_dir = tmp_path / "runner_multi_output"
        _write_npz(multi_train_npz, samples=4, seed=60, multi_channel=True)
        _write_npz(multi_eval_npz, samples=2, seed=61, multi_channel=True)
        _write_npz(multi_test_npz, samples=2, seed=62, multi_channel=True)

        multi_cmd = [
            sys.executable,
            str(repo_root / "train_conditional_dual.py"),
            "--npz-path",
            str(multi_train_npz),
            "--eval-npz-path",
            str(multi_eval_npz),
            "--test-npz-path",
            str(multi_test_npz),
            "--output-dir",
            str(multi_output_dir),
            "--sample-indices",
            "0,1,2",
            "--eval-sample-indices",
            "0,1",
            "--test-sample-indices",
            "0,1",
            "--steps",
            "5",
            "--hidden-dim",
            "32",
            "--num-layers",
            "2",
            "--latent-dim",
            "16",
        ]
        multi_result = subprocess.run(multi_cmd, cwd=repo_root, text=True, capture_output=True, check=False)
        if multi_result.returncode != 0:
            print(multi_result.stdout)
            print(multi_result.stderr)
            raise AssertionError(
                f"multi-channel train_conditional_dual.py failed with return code {multi_result.returncode}"
            )

        for filename in ["metrics.csv", "eval_metrics.csv", "test_metrics.csv", "run_summary.md"]:
            if not (multi_output_dir / filename).exists():
                raise AssertionError(f"multi-channel {filename} was not created")
        multi_summary_text = (multi_output_dir / "run_summary.md").read_text(encoding="utf-8")
        for required_text in [
            "original_signals_shape",
            "flattened_signal_length",
            "signal_channels",
            "signal_length_per_channel",
            "encoder_input_length",
        ]:
            if required_text not in multi_summary_text:
                raise AssertionError(f"multi-channel run_summary.md is missing {required_text}")
        _assert_no_forbidden_files(multi_output_dir)

    print("Conditional supervised training runner smoke test passed.")


if __name__ == "__main__":
    main()

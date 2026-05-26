"""Smoke test for COMSOL-style long CSV to multi-height NPZ converter."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from comsol_multiheight_npz_utils import validate_comsol_multiheight_npz
from conditional_dual_data_utils import get_conditional_batch, infer_signal_len, load_conditional_npz
from conditional_dual_models import ConditionalDualNet


CHANNELS = [
    (0, "Bz_liftoff_0p5", 0.5, "Bz"),
    (1, "Bz_liftoff_1p0", 1.0, "Bz"),
    (2, "Bz_liftoff_2p0", 2.0, "Bz"),
]


def _assert_shape(array_or_tensor, expected_shape, name):
    actual = tuple(array_or_tensor.shape)
    if actual != tuple(expected_shape):
        raise AssertionError(f"{name} shape mismatch: expected {expected_shape}, got {actual}")


def _write_target_npz(path: Path):
    x = np.linspace(-15.0, 15.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 10.0, 10, dtype=np.float32)
    mu_maps = np.full((3, 10, 20), 1000.0, dtype=np.float32)
    mu_maps[:, 4:6, 8:12] = 1.0
    np.savez(path, x=x, y=y, mu_maps=mu_maps)
    return x


def _write_signals_csv(path: Path, x_values, omit_row=False, missing_column=False, nonfinite_value=False):
    fieldnames = [
        "sample_index",
        "channel_index",
        "channel_name",
        "lift_off",
        "field_component",
        "x_index",
        "x",
        "value",
    ]
    if missing_column:
        fieldnames = [name for name in fieldnames if name != "value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index in range(3):
            for channel_index, channel_name, lift_off, field_component in CHANNELS:
                for x_index, x in enumerate(x_values):
                    if omit_row and sample_index == 2 and channel_index == 2 and x_index == 19:
                        continue
                    row = {
                        "sample_index": sample_index,
                        "channel_index": channel_index,
                        "channel_name": channel_name,
                        "lift_off": lift_off,
                        "field_component": field_component,
                        "x_index": x_index,
                        "x": float(x),
                        "value": float(sample_index + channel_index * 0.1 + x_index * 0.01),
                    }
                    if nonfinite_value and sample_index == 0 and channel_index == 0 and x_index == 0:
                        row["value"] = "nan"
                    if missing_column:
                        row.pop("value")
                    writer.writerow(row)


def _run_converter(repo_root: Path, signals_csv: Path, target_npz: Path, output_npz: Path):
    cmd = [
        sys.executable,
        str(repo_root / "convert_comsol_multiheight_csv_to_npz.py"),
        "--signals-csv",
        str(signals_csv),
        "--target-npz",
        str(target_npz),
        "--output-npz",
        str(output_npz),
    ]
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def main():
    repo_root = Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target_npz = tmp_path / "target.npz"
        signals_csv = tmp_path / "signals.csv"
        output_npz = tmp_path / "converted.npz"
        x_values = _write_target_npz(target_npz)
        _write_signals_csv(signals_csv, x_values)

        result = _run_converter(repo_root, signals_csv, target_npz, output_npz)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise AssertionError(f"converter failed with return code {result.returncode}")
        if not output_npz.exists():
            raise AssertionError("converted output NPZ was not created")

        with np.load(output_npz, allow_pickle=False) as data:
            _assert_shape(data["signals"], (3, 3, 20), "signals")
            _assert_shape(data["mu_maps"], (3, 10, 20), "mu_maps")
            _assert_shape(data["x"], (20,), "x")
            _assert_shape(data["y"], (10,), "y")
            for name in ["signal_channel_names", "lift_off_values", "field_components"]:
                if name not in data.files:
                    raise AssertionError(f"converted output missing {name}")

        summary = validate_comsol_multiheight_npz(output_npz)
        if summary["num_samples"] != 3 or summary["num_channels"] != 3 or summary["signal_len"] != 20:
            raise AssertionError(f"unexpected validator summary: {summary}")

        dataset = load_conditional_npz(output_npz)
        batch = get_conditional_batch(dataset, [0, 1])
        _assert_shape(batch["signals"], (2, 60), "batch signals")
        if infer_signal_len(dataset) != 60:
            raise AssertionError("infer_signal_len should return 60")
        model = ConditionalDualNet(signal_len=60, latent_dim=16, hidden_dim=32, num_layers=2)
        out = model(batch["signals"], batch["coords"])
        _assert_shape(out["latent"], (2, 16), "latent")
        _assert_shape(out["mu"], (2, 200, 1), "mu")
        _assert_shape(out["phi"], (2, 200, 1), "phi")

        missing_column_csv = tmp_path / "missing_column.csv"
        _write_signals_csv(missing_column_csv, x_values, missing_column=True)
        missing_result = _run_converter(repo_root, missing_column_csv, target_npz, tmp_path / "missing_out.npz")
        if missing_result.returncode == 0:
            raise AssertionError("converter should fail for missing CSV column")

        incomplete_csv = tmp_path / "incomplete.csv"
        _write_signals_csv(incomplete_csv, x_values, omit_row=True)
        incomplete_result = _run_converter(repo_root, incomplete_csv, target_npz, tmp_path / "incomplete_out.npz")
        if incomplete_result.returncode == 0:
            raise AssertionError("converter should fail for incomplete channel/x_index coverage")

        nonfinite_csv = tmp_path / "nonfinite.csv"
        _write_signals_csv(nonfinite_csv, x_values, nonfinite_value=True)
        nonfinite_result = _run_converter(repo_root, nonfinite_csv, target_npz, tmp_path / "nonfinite_out.npz")
        if nonfinite_result.returncode == 0:
            raise AssertionError("converter should fail for non-finite signal values")

    print("COMSOL multi-height CSV to NPZ converter smoke test passed.")


if __name__ == "__main__":
    main()

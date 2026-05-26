"""End-to-end dry-run for the COMSOL pilot handoff path."""

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


def _write_mock_targets(path: Path):
    x = np.linspace(-10.0, 10.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 9.0, 10, dtype=np.float32)
    mu_maps = np.full((4, 10, 20), 1000.0, dtype=np.float32)
    for sample_idx in range(4):
        x0 = 6 + sample_idx
        mu_maps[sample_idx, 4:6, x0 : x0 + 4] = 1.0
    np.savez(path, x=x, y=y, mu_maps=mu_maps)
    return x


def _write_mock_signals_csv(path: Path, x_values):
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index in range(4):
            for channel_index, channel_name, lift_off, field_component in CHANNELS:
                scale = 1.0 / (1.0 + lift_off)
                for x_index, x in enumerate(x_values):
                    center = -2.0 + sample_index
                    value = scale * np.exp(-0.05 * float((x - center) ** 2))
                    writer.writerow(
                        {
                            "sample_index": sample_index,
                            "channel_index": channel_index,
                            "channel_name": channel_name,
                            "lift_off": lift_off,
                            "field_component": field_component,
                            "x_index": x_index,
                            "x": float(x),
                            "value": float(value),
                        }
                    )


def _run_converter(repo_root: Path, signals_csv: Path, targets_npz: Path, output_npz: Path):
    cmd = [
        sys.executable,
        str(repo_root / "convert_comsol_multiheight_csv_to_npz.py"),
        "--signals-csv",
        str(signals_csv),
        "--target-npz",
        str(targets_npz),
        "--output-npz",
        str(output_npz),
    ]
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def main():
    repo_root = Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        export_dir = Path(tmp) / "comsol_pilot_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        signals_csv = export_dir / "signals_multiheight.csv"
        targets_npz = export_dir / "targets.npz"
        converted_npz = export_dir / "comsol_multiheight_pilot.npz"

        x_values = _write_mock_targets(targets_npz)
        _write_mock_signals_csv(signals_csv, x_values)

        result = _run_converter(repo_root, signals_csv, targets_npz, converted_npz)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise AssertionError(f"converter failed with return code {result.returncode}")
        if not converted_npz.exists():
            raise AssertionError("converted pilot NPZ was not created")

        summary = validate_comsol_multiheight_npz(converted_npz)
        if summary["num_samples"] != 4 or summary["num_channels"] != 3 or summary["signal_len"] != 20:
            raise AssertionError(f"unexpected validator summary: {summary}")

        dataset = load_conditional_npz(converted_npz)
        batch = get_conditional_batch(dataset, [0, 1, 2])
        _assert_shape(batch["signals"], (3, 60), "batch signals")
        if infer_signal_len(dataset) != 60:
            raise AssertionError("infer_signal_len should return 60")

        model = ConditionalDualNet(signal_len=60, latent_dim=16, hidden_dim=32, num_layers=2)
        out = model(batch["signals"], batch["coords"])
        _assert_shape(out["mu"], (3, 200, 1), "mu")
        _assert_shape(out["phi"], (3, 200, 1), "phi")

    print("COMSOL pilot handoff end-to-end smoke test passed.")


if __name__ == "__main__":
    main()

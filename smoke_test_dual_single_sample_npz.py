"""Smoke test for the single-sample .npz dual-network loop.

The test creates a temporary synthetic .npz file, runs the single-sample loop
through subprocess, and verifies the expected progress output. It does not
read real data, save models, save images, create checkpoints, or call
train_pinn.py.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np


def _make_temp_npz(tmpdir):
    x = np.linspace(-15.0, 15.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 10.0, 10, dtype=np.float32)
    signals = (-0.1 * np.exp(-np.power(x / 5.0, 2))).reshape(1, 20).astype(np.float32)
    mu_maps = np.ones((1, 10, 20), dtype=np.float32) * 1000.0
    mu_maps[0, 4:6, 8:12] = 1.0

    npz_path = os.path.join(tmpdir, "dual_single_sample_smoke.npz")
    np.savez_compressed(npz_path, x=x, y=y, signals=signals, mu_maps=mu_maps)
    return npz_path


def _run_loop(npz_path):
    command = [
        sys.executable,
        "minimal_dual_single_sample_loop.py",
        "--npz-path",
        npz_path,
        "--sample-index",
        "0",
        "--outer-steps",
        "1",
        "--phi-steps",
        "1",
        "--mu-steps",
        "1",
        "--test-radius",
        "5.0",
        "--center-mode",
        "three",
        "--lambda-area-prior",
        "0.0",
        "--lambda-mask-prior",
        "0.0",
    ]
    return subprocess.run(command, text=True, capture_output=True)


def _assert_success(result):
    if result.returncode != 0:
        print("--- stdout ---")
        print(result.stdout)
        print("--- stderr ---")
        print(result.stderr)
        raise AssertionError(f"single-sample loop failed with code {result.returncode}")

    required_snippets = [
        "Minimal dual-network single-sample loop passed.",
        "loss_phi=",
        "loss_mu=",
        "mu_pred_min=",
        "mu_pred_max=",
        "mu_label_min=",
        "mu_label_max=",
        "test_grads_shape=",
        "center_mode=three",
        "test_centers=3",
        "mu_mse=",
        "mu_mae=",
        "defect_iou=",
        "area_prior_loss=",
        "pred_defect_fraction=",
        "target_defect_fraction=",
        "dice_loss=",
        "lambda_mask_prior=",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in result.stdout]
    if missing:
        print("--- stdout ---")
        print(result.stdout)
        print("--- stderr ---")
        print(result.stderr)
        raise AssertionError(f"missing expected stdout snippets: {missing}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_temp_npz(tmpdir)
        result = _run_loop(npz_path)
        _assert_success(result)

    print("Dual-network single-sample npz smoke test passed.")


if __name__ == "__main__":
    main()

# S195 COMSOL center-bin failure diagnostics script

S195 adds `comsol_center_bin_failure_diagnostics.py`, a read-only diagnostic script for existing parametric prediction exports.

Inputs per prediction directory:

- `train_predictions.csv`, `val_predictions.csv`, `test_predictions.csv`;
- `train_prediction_mask_metrics.csv`, `val_prediction_mask_metrics.csv`, `test_prediction_mask_metrics.csv`;
- `run_summary.md`.

If a split is missing, the script skips that split and reports it in the per-run summary. At least one of val/test must be diagnosable.

Outputs:

- `per_component_center_bin_errors.csv`;
- `per_sample_center_bin_errors.csv`;
- `grouped_center_bin_errors.csv`;
- `worst_samples.csv`;
- `summary.md`.

The S191 prediction CSVs do not directly export raw center-bin logits or offset heads. The script therefore reconstructs center bins and bin-normalized offsets from decoded `center_x/center_y` values and the bin/grid parameters in `run_summary.md`. This is sufficient for sample-level failure diagnostics, but it should not be interpreted as a direct logit-level calibration analysis.

`smoke_test_comsol_center_bin_failure_diagnostics.py` covers x-bin wrong, y-bin wrong, and both-correct mock cases.

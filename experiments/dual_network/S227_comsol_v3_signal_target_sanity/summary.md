# S227 normalized V3 signal-target sanity

S227 checks normalized V3 data before any tiny-overfit run.

## Split Summary

| split | samples | signal_std_min | signal_std_max | floor_rate | mask_threshold_mismatch | max_bbox_center_error_cells | center_bin_in_range | offset_min | offset_max |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| train | 30 | `4.734403e-10` | `9.312454e-09` | `1.000000` | `0` | `0.444445` | `True` | `-0.477500` | `0.471250` |
| val | 10 | `4.631334e-10` | `7.501884e-09` | `1.000000` | `0` | `0.444446` | `True` | `-0.417333` | `0.394445` |
| test | 10 | `1.152059e-09` | `7.419892e-09` | `1.000000` | `0` | `0.344448` | `True` | `-0.348889` | `0.450000` |

## Decision

All splits trigger the runner `std < 1e-8` signal floor for every sample. Tiny-overfit training is therefore skipped; the next check should target COMSOL signal export / probe height / field expression / signal scaling / runner normalization floor.

## Notes

- Mask threshold alignment, bbox/defect alignment, and center-bin target ranges are reported separately from signal scale.
- This diagnostic does not train and does not modify the normalized V3 data.

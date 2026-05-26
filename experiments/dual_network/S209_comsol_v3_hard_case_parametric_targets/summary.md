# S209 COMSOL V3 hard-case parametric target summary

S209 built fixed-order parametric targets for the S208 converted V3 hard-case pilot. No model training was run.

## Outputs

| split | samples | target path |
|---|---:|---|
| train | 30 | `experiments/dual_network/S209_comsol_v3_hard_case_parametric_targets/train/parametric_targets.npz` |
| val | 10 | `experiments/dual_network/S209_comsol_v3_hard_case_parametric_targets/val/parametric_targets.npz` |
| test | 10 | `experiments/dual_network/S209_comsol_v3_hard_case_parametric_targets/test/parametric_targets.npz` |

Each split also includes `parametric_target_preview.csv` and `parametric_target_summary.md`.

## Schema

- `presence_targets`: `[N, 3]`
- `type_targets`: `[N, 3]`
- `continuous_targets`: `[N, 3, 6]`
- `continuous_targets_raw`: `[N, 3, 6]`
- `target_schema`: `center_x`, `center_y`, `axis_x`, `axis_y`, `depth_or_shape_param`, `rotation_angle`
- `type_vocab`: `rectangular_notch`
- `angle_unit`: radians
- `component_counts`: one present component per sample in this fallback pilot.

## Boundary

The target builder is compatible with the fallback pack. However, the pack currently represents one solved rectangular notch per sample. It should be treated as a hard-case pilot for ingest and candidate evaluation, not as evidence for rotated or multi-component COMSOL generalization.

## Self-review

- Parametric target NPZ files were generated for train/val/test.
- Schema matches the existing hard rasterizer expectations.
- The next gate is oracle rasterization in S210.

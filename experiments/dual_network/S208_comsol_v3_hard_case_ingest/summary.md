# S208 COMSOL V3 hard-case ingest summary

本阶段将真实 COMSOL V3 hard-case fallback pack 接入 dual-network 支线实验目录，只做数据接入、转换和自检，不运行训练。

## Source

- source export root: `comsol_geometry_v3_hard_case_exports/`
- ingest copy: `experiments/dual_network/S208_comsol_v3_hard_case_ingest/raw/`
- converted output: `experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted/`
- split sizes: train `30`, val `10`, test `10`
- source type: real COMSOL 6.3 solves from `magnetic_prospecting.mph`
- fallback: yes, `30/10/10` was used because direct real COMSOL per-sample solve is available but slow.

The pack is not synthetic and is not copied from V2. Current generation is still a fallback pilot: it uses single unrotated rectangular Block hard cases and does not yet cover true rotated or multi-component solved geometries.

## Converted NPZ Check

| split | signals shape | CSV rows | finite values | complete x_index | mask threshold match |
|---|---:|---:|---|---|---|
| train | `[30, 3, 200]` | `18000` | yes | yes | yes |
| val | `[10, 3, 200]` | `6000` | yes | yes | yes |
| test | `[10, 3, 200]` | `6000` | yes | yes | yes |

Converted NPZ fields include `mu_maps`, `masks`, `x`, `y`, `defect_params`, `source_sample_ids`, `source_global_indices`, `signal_channel_names`, `lift_off_values`, `field_components`, `source_type`, `signal_flatten_order`, `geometry_units`, `field_units`, `metadata_json`, `signals`, and converter bookkeeping fields.

## Hard Case Distribution

| split | x_bin_wrong_like | both_bins_wrong_like | bins_correct_center_or_offset_bad | geometry_or_type_interaction | rare_y_bin_wrong |
|---|---:|---:|---:|---:|---:|
| train | 10 | 5 | 7 | 5 | 3 |
| val | 3 | 2 | 2 | 2 | 1 |
| test | 3 | 2 | 2 | 2 | 1 |

## Self-review

- V2-compatible CSV and NPZ schema is present.
- `signals` are finite and have expected `[N,3,200]` shape.
- `masks` exactly match `mu_maps < 500`.
- Every sample/channel has `x_index=0..199`.
- It is safe to proceed to S209 parametric target construction.

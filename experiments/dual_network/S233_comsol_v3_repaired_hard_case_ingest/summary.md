# S233 COMSOL V3 repaired hard-case ingest

本阶段将真实 COMSOL repaired Bz hard-case fallback pack 接入支线实验树，只做数据接入和验证，不训练模型，不修改 runner。

## Source

- Source root: `comsol_geometry_v3_repaired_bz_hard_case_exports/`
- Ingest copy: `experiments/dual_network/S233_comsol_v3_repaired_hard_case_ingest/raw/`
- Generation route: per-sample fresh COMSOL model, near-defect reduced/anomaly field `mfnc.redBz`
- Boundary: single unrotated Block `rectangular_notch` fallback pilot; not true rotated or multi-component COMSOL geometry

## Defect Params Gate

`hard_case_type` 分布从 S233 接入副本的 `defect_params.csv` 重新统计，未使用外部摘要文本。

| split | samples | x_bin_wrong_like | both_bins_wrong_like | bins_correct_center_or_offset_bad | geometry_or_type_interaction | rare_y_bin_wrong |
|---|---:|---:|---:|---:|---:|---:|
| train | 30 | 10 | 5 | 7 | 5 | 3 |
| val | 10 | 3 | 2 | 2 | 2 | 1 |
| test | 10 | 3 | 2 | 2 | 2 | 1 |

All split sample indices are unique and cover `0..N-1`.

## Converted Data

| split | converted shape | CSV rows | signal std range | peak-to-peak range | mask threshold mismatch |
|---|---:|---:|---:|---:|---:|
| train | `[30,3,200]` | 18000 | `1.678613e-06` - `3.162381e-06` | `7.409328e-06` - `1.984153e-05` | 0 |
| val | `[10,3,200]` | 6000 | `1.943240e-06` - `2.826707e-06` | `1.030256e-05` - `1.845700e-05` | 0 |
| test | `[10,3,200]` | 6000 | `2.101787e-06` - `2.876971e-06` | `8.955499e-06` - `1.965699e-05` | 0 |

All values are finite, every sample/channel has complete `x_index=0..199`, and every sample/channel passes `std > 1e-8` and `peak_to_peak > 1e-8`.

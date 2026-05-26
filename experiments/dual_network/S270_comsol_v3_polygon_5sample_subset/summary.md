# S270 COMSOL V3 Polygon 5-Sample Subset

S270 builds a 5-sample polygon subset from the S254-S258 ingested polygon V3 hard-case pack. The subset is created with `make_comsol_polygon_subset_package.py`; no COMSOL data is generated or modified.

## Selected Samples

| subset sample | source train sample | hard_case_type | defect_type | component_type_combination |
| ---: | ---: | --- | --- | --- |
| `0` | `0` | `x_bin_wrong_like` | `rotated_rect` | `rotated_rect` |
| `1` | `11` | `both_bins_wrong_like` | `multi_defect` | `rotated_rect\|rectangular_notch` |
| `2` | `15` | `bins_correct_center_or_offset_bad` | `rotated_rect` | `rotated_rect` |
| `3` | `22` | `geometry_or_type_interaction` | `multi_defect` | `rotated_rect\|rectangular_notch` |
| `4` | `27` | `rare_y_bin_wrong` | `rotated_rect` | `rotated_rect` |

## Outputs

- `train_5sample_polygon_v3.npz`
- `train_5sample_polygon_targets.npz`
- `hard_case_coverage.csv`

The subset signal shape is `[5,3,200]`; polygon target shape is `[5,3,4,2]`. `sample_indices` are renumbered to `0..4`, and `subset_source_indices_json` records the source train indices.

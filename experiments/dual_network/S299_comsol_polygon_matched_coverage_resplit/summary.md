# S299 COMSOL polygon matched-coverage resplit

This resplit uses the existing S254-S258 polygon V3 pack only. It does not generate COMSOL data and does not overwrite the original train/val/test split.

## Split Counts

- train: samples `30`, hard_case counts x_bin_wrong_like=10, both_bins_wrong_like=5, bins_correct_center_or_offset_bad=7, geometry_or_type_interaction=5, rare_y_bin_wrong=3, rotated `21`, multi-component `8`.
- val: samples `10`, hard_case counts x_bin_wrong_like=3, both_bins_wrong_like=2, bins_correct_center_or_offset_bad=2, geometry_or_type_interaction=2, rare_y_bin_wrong=1, rotated `8`, multi-component `3`.
- test: samples `10`, hard_case counts x_bin_wrong_like=3, both_bins_wrong_like=2, bins_correct_center_or_offset_bad=2, geometry_or_type_interaction=2, rare_y_bin_wrong=1, rotated `7`, multi-component `2`.

## Coverage

- Search score: `(0, 0, 18, 1, 18, 0)`.
- Held-out samples with all component bins exactly covered by train: `4` / `20`.
- Held-out samples with all component bins within train distance <= 1: `20` / `20`.
- Coverage is prioritized over exact split provenance; hard-case counts remain at the requested targets.

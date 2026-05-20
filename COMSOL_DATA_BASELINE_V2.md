# COMSOL DATA BASELINE V2

## Baseline Name

COMSOL combined single+multi-defect baseline v2.

## Scope

This is a COMSOL data-domain baseline for combined single-defect and true multi_defect controlled synthetic data.

It does not replace `CURRENT_BASELINE.md`. It is not a v3_complex baseline and should not be compared as a formal replacement for the v3_complex route.

It also does not replace `COMSOL_DATA_BASELINE.md` or `COMSOL_MULTI_DEFECT_DATA_BASELINE.md`. Those files remain the single-defect and multi_defect standalone COMSOL data-domain baselines. This document records the combined reference.

## Dataset

- NPZ path: `data/comsol_mfl/prepared/comsol_combined_single_multi_defect_baseline_v2_candidate.npz`
- N: 840
- split: train 562 / val 139 / test 139
- single_defect samples: 600
- multi_defect samples: 240
- input: `delta_bz`, shape `(840, 3, 201)`; per sample `(3, 201)`
- output: defect union mask, shape `(840, 64, 128)`; per sample `(64, 128)`
- source single-defect pack: `comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect`
- source multi_defect pack: `comsol_multi_defect_multiline_forward_pack_v3_pilot`

Defect composition:

| group | sample_count |
|---|---:|
| single_defect | 600 |
| multi_defect | 240 |

Defect type composition:

| defect_type | sample_count |
|---|---:|
| rectangular_notch | 200 |
| rotated_rect | 200 |
| polygon | 200 |
| multi_defect | 240 |

The combined pack preserves the already reviewed source splits. It does not reshuffle samples. Coordinates are matched across both source packs before merge, and `delta_bz = bz_defect - bz_no_defect` is checked after readback.

## Model

- model family: mask-only grid decoder
- input encoder: Conv1d / BzEncoder for `(3, 201)` multi-line `delta_bz`
- output: mask logits, shape `(64, 128)`
- no metadata input
- no `bz_defect` / `bz_no_defect` input
- no defect type or defect group conditional input
- no component detector
- no instance segmentation
- no forward consistency

## Training Protocol

- seeds: 42 / 123 / 2026
- epochs: 200
- batch size: 8
- normalization: per-channel `delta_bz` mean/std computed from train split only
- checkpoint selection: validation-only, using `IoU + Dice - area_error`
- threshold selection: validation-only, candidates `0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90`
- test set: final evaluation only
- loss: BCEWithLogits + soft Dice

Per-seed best epoch and selected threshold:

| seed | best_epoch | selected_threshold |
|---:|---:|---:|
| 42 | 20 | 0.50 |
| 123 | 29 | 0.80 |
| 2026 | 16 | 0.60 |

## Headline Metrics

3-seed mean +/- std:

| split | IoU | Dice | area_error | center_error |
|---|---:|---:|---:|---:|
| train | `0.6948 +/- 0.0131` | `0.8174 +/- 0.0092` | `0.1491 +/- 0.0116` | `2.6555 +/- 0.2016` |
| val | `0.6527 +/- 0.0038` | `0.7868 +/- 0.0027` | `0.1866 +/- 0.0023` | `3.1714 +/- 0.0806` |
| test | `0.6454 +/- 0.0021` | `0.7818 +/- 0.0018` | `0.2124 +/- 0.0071` | `2.8901 +/- 0.0740` |

Test split by defect group:

| defect_group | IoU | Dice | area_error |
|---|---:|---:|---:|
| single_defect | `0.6601` | `0.7926` | `0.2373` |
| multi_defect | `0.6088` | `0.7550` | `0.1507` |

Test split by defect type:

| defect_type | IoU | Dice |
|---|---:|---:|
| rectangular_notch | `0.6765` | `0.8051` |
| rotated_rect | `0.6385` | `0.7775` |
| polygon | `0.6654` | `0.7954` |
| multi_defect | `0.6088` | `0.7550` |

Multi_defect connected-component test metrics:

- `pred_cc_is_2 = 1.0000 +/- 0.0000`
- `pred_cc_correct = 1.0000 +/- 0.0000`
- missed component rate: 0
- merged component rate: 0
- split / extra-fragment rate: 0

## Comparison To Standalone COMSOL Baselines

Against `COMSOL_DATA_BASELINE.md` single-defect baseline:

- standalone single-defect test IoU / Dice: `0.6515 +/- 0.0064` / `0.7861 +/- 0.0046`
- combined model on single_defect test IoU / Dice: `0.6601` / `0.7926`
- conclusion: no substantial single-defect degradation was observed.

Against `COMSOL_MULTI_DEFECT_DATA_BASELINE.md`:

- standalone multi_defect test IoU / Dice: `0.6118 +/- 0.0014` / `0.7573 +/- 0.0011`
- combined model on multi_defect test IoU / Dice: `0.6088` / `0.7550`
- standalone multi_defect `pred_cc_is_2`: `1.0000 +/- 0.0000`
- combined multi_defect `pred_cc_is_2`: `1.0000 +/- 0.0000`
- conclusion: the small multi_defect metric dip is within the documented review tolerance, and connected-component behavior is preserved.

## Failure Audit

- Combined training did not introduce a source_dataset or source_pack issue.
- The dominant failure mode is area error, with localization as a secondary issue on some polygon-like shapes.
- The hardest group in the combined audit is `rectangular_notch+rectangular_notch`.
- No schema, coordinate, mask, or components_json issue was found.
- The result is stable enough to document as `COMSOL_DATA_BASELINE_V2`.

## Artifacts

- `results/summaries/comsol_combined_baseline_v2_pack_summary.txt`
- `results/summaries/comsol_combined_baseline_v2_training_summary.txt`
- `results/summaries/comsol_combined_baseline_v2_failure_audit_summary.txt`
- `results/summaries/claude_review_comsol_baseline_v2_candidate.txt`
- `results/metrics/comsol_combined_baseline_v2_seed_summary.csv`
- `results/metrics/comsol_combined_baseline_v2_defect_group_summary.csv`
- `results/metrics/comsol_combined_baseline_v2_defect_type_summary.csv`
- `results/metrics/comsol_combined_baseline_v2_connected_component_summary.csv`
- `results/metrics/comsol_combined_baseline_v2_component_combination_summary.csv`
- `results/metrics/comsol_combined_baseline_v2_source_summary.csv`

## Limitations

- Pilot-level controlled synthetic COMSOL data only.
- No real experimental data.
- Multi_defect samples are limited to `component_count = 2`.
- No `polygon+polygon` multi_defect samples.
- No `component_count > 2` samples.
- This is not a v3_complex baseline.
- This is not a replacement for `CURRENT_BASELINE.md`.
- This is not a replacement for standalone `COMSOL_DATA_BASELINE.md`.
- This is not a replacement for standalone `COMSOL_MULTI_DEFECT_DATA_BASELINE.md`.

## Next Step

Recommended next direction: increase `component_count > 2`.

The combined baseline already merges single-defect and true two-component multi_defect data without a clear degradation in either domain. The next meaningful stress test is whether the same pipeline and model family remain stable when true COMSOL multi_defect samples contain more than two separated components.

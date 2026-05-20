# COMSOL Three-Component Data Baseline

## Baseline Name

COMSOL 3-component multi_defect pilot_v4 mask-only multi-line baseline.

## Scope

This is a COMSOL data-domain baseline for true multi_defect samples with `component_count = 3`.

It is limited to controlled synthetic COMSOL data with `rectangular_notch` and `rotated_rect` components only. It does not include polygon components.

This baseline does not replace `CURRENT_BASELINE.md`. It is not a v3_complex baseline and should not be compared as a formal replacement for the v3_complex route.

This baseline also does not replace `COMSOL_DATA_BASELINE.md`, `COMSOL_DATA_BASELINE_V2.md`, or `COMSOL_MULTI_DEFECT_DATA_BASELINE.md`. Those documents remain the single-defect, combined single+two-component, and two-component multi_defect COMSOL references.

## Dataset

- NPZ path: `data/comsol_mfl/prepared/comsol_multi_defect_three_component_multiline_forward_pack_v4_pilot.npz`
- N: 480
- split: train 320 / val 80 / test 80
- input: `delta_bz`, shape `(480, 3, 201)`; per sample `(3, 201)`
- output: union mask, shape `(480, 64, 128)`; per sample `(64, 128)`
- `component_count = 3`
- `connected_component_count = 3` for all samples
- distance bins: near / medium / far
- sensor_z_m / liftoff: 0.008 m
- scan_line_y: `[-0.001, 0.0, 0.001]` m

Component combinations:

| component_type_combination | sample_count |
|---|---:|
| `rectangular_notch+rectangular_notch+rectangular_notch` | 120 |
| `rectangular_notch+rectangular_notch+rotated_rect` | 120 |
| `rectangular_notch+rotated_rect+rotated_rect` | 120 |
| `rotated_rect+rotated_rect+rotated_rect` | 120 |

Distance-bin distribution:

| distance_bin | sample_count |
|---|---:|
| near | 160 |
| medium | 160 |
| far | 160 |

Source-pack distribution:

| source_pack | sample_count |
|---|---:|
| pilot_v3 | 240 |
| pilot_v4_topup | 240 |

The pack uses true joint COMSOL three-component multi_defect solves. `Bz_defect` is solved with all three components present in the same COMSOL model; `delta_bz = Bz_defect - Bz_no_defect`. The target mask is a union mask rasterized from the same `components_json` used for geometry construction.

## Model

- model family: mask-only grid decoder
- input encoder: Conv1d / BzEncoder for `(3, 201)` multi-line `delta_bz`
- output: mask logits, shape `(64, 128)`
- no metadata input
- no `bz_defect` / `bz_no_defect` input
- no `components_json`, `component_types`, `component_counts`, `source_pack`, or `distance_bin` input
- no instance segmentation
- no component detector
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
| 42 | 36 | 0.60 |
| 123 | 38 | 0.70 |
| 2026 | 39 | 0.60 |

## Headline Metrics

3-seed mean +/- std:

| split | IoU | Dice |
|---|---:|---:|
| train | `0.7923 +/- 0.0130` | `0.8825 +/- 0.0087` |
| val | `0.6895 +/- 0.0046` | `0.8053 +/- 0.0035` |
| test | `0.6761 +/- 0.0034` | `0.7958 +/- 0.0037` |

Connected-component test metrics:

- `pred_cc_is_3 = 0.9875 +/- 0.0177`
- missed component rate: `0.0083 +/- 0.0059`
- merged component rate: `0.0000 +/- 0.0000`
- split component rate: `0.0125 +/- 0.0177`
- extra fragment rate: `0.0125 +/- 0.0177`

## Group Findings

- Hardest component combination: `rectangular_notch+rotated_rect+rotated_rect`.
- Main failure mode: boundary / shape smoothing.
- Near-distance samples did not show obvious degradation in the pilot_v4 failure audit.
- No schema, mask, or `components_json` issue was found.

## Artifacts

- `results/summaries/comsol_three_component_multi_defect_pilot_v4_training_gate_summary.txt`
- `results/summaries/comsol_three_component_multi_defect_pilot_v4_failure_audit_summary.txt`
- `results/summaries/comsol_three_component_multi_defect_baseline_readiness_plan.txt`
- `results/metrics/comsol_three_component_multi_defect_pilot_v4_seed_summary.csv`
- `results/metrics/comsol_three_component_multi_defect_pilot_v4_component_combination_summary.csv`
- `results/metrics/comsol_three_component_multi_defect_pilot_v4_connected_component_summary.csv`
- `results/summaries/claude_review_20_40_three_component_pilot_v4.txt`

## External Review

Claude Code review for Stage 20.40 found no must-fix or suggested-fix items and supported recording this result as a COMSOL component_count=3 data-domain baseline.

## Limitations

- Pilot-level controlled synthetic COMSOL data only.
- `component_count` fixed to 3.
- No polygon component.
- No `component_count > 3`.
- No real experimental data.
- Not a v3_complex baseline.
- Not a replacement for `CURRENT_BASELINE.md`.
- Not a replacement for `COMSOL_DATA_BASELINE.md`.
- Not a replacement for `COMSOL_DATA_BASELINE_V2.md`.
- Not a replacement for `COMSOL_MULTI_DEFECT_DATA_BASELINE.md`.

## Next Step

Recommended next direction: combine `component_count = 2` and `component_count = 3` COMSOL multi_defect data into a `COMSOL_DATA_BASELINE_V3` candidate, with an independent pack-preparation step, training gate, and review.

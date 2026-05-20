# COMSOL Multi-Defect Data Baseline

## Baseline Name

COMSOL multi_defect pilot_v3 mask-only multi-line baseline.

## Scope

This is a COMSOL data-domain baseline for true multi_defect samples. It is limited to controlled synthetic COMSOL data with `component_count = 2`.

This baseline does not replace `CURRENT_BASELINE.md`. It is not a v3_complex baseline and should not be compared as a formal replacement for the current v3_complex route.

This baseline also does not replace `COMSOL_DATA_BASELINE.md`, which records the single-defect COMSOL data-domain baseline.

## Dataset

- NPZ path: `data/comsol_mfl/prepared/comsol_multi_defect_multiline_forward_pack_v3_pilot.npz`
- N: 240
- split: train 160 / val 40 / test 40
- input: `delta_bz`, shape `(240, 3, 201)`
- output: union mask, shape `(240, 64, 128)`
- `connected_component_count = 2` for all samples

Component combinations:

| component_type_combination | sample_count |
|---|---:|
| `rectangular_notch+rectangular_notch` | 48 |
| `rectangular_notch+rotated_rect` | 48 |
| `rotated_rect+rotated_rect` | 48 |
| `rectangular_notch+polygon` | 48 |
| `rotated_rect+polygon` | 48 |

The pack uses true joint COMSOL multi_defect solves. `Bz_defect` is solved with both components present in the same COMSOL model; `delta_bz = Bz_defect - Bz_no_defect`. The target mask is a union mask rasterized from the same `components_json` used for geometry construction.

## Model

- model family: mask-only grid decoder
- input encoder: Conv1d / BzEncoder for `(3, 201)` multi-line `delta_bz`
- output: mask logits, shape `(64, 128)`
- no component detector
- no instance segmentation
- no forward consistency
- no geometry metadata input
- no `bz_defect` / `bz_no_defect` input

## Training Protocol

- seeds: 42 / 123 / 2026
- epochs: 200
- batch size: 8
- normalization: per-channel `delta_bz` mean/std computed from train split only
- checkpoint selection: validation-only, using `IoU + Dice - area_error`
- threshold selection: validation-only, candidates `0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90`
- test set: final evaluation only
- loss: BCEWithLogits + soft Dice

## Headline Metrics

3-seed mean +/- std:

| split | IoU | Dice |
|---|---:|---:|
| train | `0.6932 +/- 0.0089` | `0.8169 +/- 0.0063` |
| val | `0.6246 +/- 0.0037` | `0.7660 +/- 0.0028` |
| test | `0.6118 +/- 0.0014` | `0.7573 +/- 0.0011` |

Connected-component test metrics:

- `pred_cc_is_2 = 1.0000 +/- 0.0000`
- missed component rate: 0
- merged component rate: 0
- split / extra-fragment rate: 0

## Group Findings

- Polygon component is not clearly harder: polygon test Dice is `0.7487`, non-polygon test Dice is `0.7631`.
- Hardest component combination: `rectangular_notch+polygon`.
- Main failure mode: boundary / shape smoothing.
- No schema, mask, or `components_json` issue was found.

## Artifacts

- `results/summaries/comsol_multi_defect_pilot_v3_training_gate_summary.txt`
- `results/summaries/comsol_multi_defect_pilot_v3_failure_audit_summary.txt`
- `results/summaries/comsol_multi_defect_baseline_readiness_plan.txt`
- `results/metrics/comsol_multi_defect_pilot_v3_seed_summary.csv`
- `results/metrics/comsol_multi_defect_pilot_v3_component_combination_summary.csv`
- `results/metrics/comsol_multi_defect_pilot_v3_connected_component_summary.csv`
- `results/summaries/claude_review_20_34_multi_defect_pilot_v3.txt`

## Limitations

- Pilot-level only.
- Controlled synthetic COMSOL data only.
- `component_count` fixed to 2.
- No `polygon+polygon`.
- No `component_count > 2`.
- No real experimental data.
- Not a v3_complex baseline.
- Not a replacement for `CURRENT_BASELINE.md`.
- Not a replacement for the single-defect `COMSOL_DATA_BASELINE.md`.

## Next Step

Recommended next step: build a combined `COMSOL_DATA_BASELINE_V2` candidate using single-defect pilot_v9 plus multi_defect pilot_v3.

That next step requires an independent combined data package and training gate. It is not executed as part of this baseline record.

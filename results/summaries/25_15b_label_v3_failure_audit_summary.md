# 25.15b Label-V3 Failure Audit

- acceptance_decision: `NEEDS_PINN_LABEL_DERIVATION_V3B`
- route_decision: `A. enter 25.16 label-v3b derivation + validator, no training`
- main_conclusion: 25.15 label-v3 relieved near-empty masks but converted the failure into union-like merged masks: soft/valid support is too broad relative to component identity, while raw labels remain sufficient for a stricter v3b derivation.

## Metric Evidence

- 25.15 test: recall `0.674419`, missed `0.325581`, extra `0.340909`, merged `1.000000`, component Dice `0.034245`, union Dice `0.061694`, depth RMSE `0.001106223 m`.
- vs 25.13: component Dice `0.028710`, union Dice `0.058865`, merged `1.000000`, depth RMSE `0.000863332 m`.
- vs 25.10: component Dice `-0.075316`, union Dice `-0.068785`, merged `0.800000`, depth RMSE `0.000862908 m`.

## V3 Target Audit

- soft OR / raw union ratio mean/p95/max: `2.010499` / `2.397522` / `2.451613`
- soft duplicate fraction mean/p95/max: `0.061993` / `0.243511` / `0.327496`
- valid duplicate fraction mean/p95/max: `0.061993` / `0.243511` / `0.327496`
- soft union-like sample count: `112/112`
- separated soft duplicate fraction mean/max: `0.003718` / `0.076063`
- close soft duplicate fraction mean/max: `0.050723` / `0.253333`
- depth nonzero outside depth-valid mean/max: `0.000000` / `0.000000`
- depth-valid duplicate target mean/max: `2.651786` / `42.000000`
- SDF multi-valid overlap mean/max: `32.133929` / `232.000000`

## Failure Grouping

- test overall merged: `1.000000`
- component_count=2 merged: `1.000000`
- component_count=3 merged: `1.000000`
- separated merged: `1.000000`
- close merged: `1.000000`
- touching merged: `1.000000`
- partially_overlapping merged: `1.000000`

## Cause Ranking
- rank 1: `soft support too broad` supported=`True`; v3 soft OR expands well beyond raw union while 25.15 merged rate is global
- rank 2: `valid region leakage` supported=`True`; component valid/soft regions overlap across components, including separated/close rows
- rank 3: `SDF identity too weak in multi-valid boundary zones` supported=`True`; multiple component valid regions include near-boundary SDF pixels for more than one component
- rank 4: `depth valid region dilution` supported=`True`; component depth valid region leakage would indicate target-side depth dilution
- rank 5: `genuine topology-only hard case` supported=`False`; would require merged collapse to be concentrated in touching/overlap only
- rank 6: `evaluator / threshold artifact` supported=`False`; not supported: predicted component count remains non-empty and all hard slices merge

## V3B Recommendation

- Add `component_exclusive_hard_core` and keep it mutually exclusive.
- Split valid region into `hard_core_region`, `boundary_halo_region`, and `ignore_overlap_region`.
- Limit soft halo to a narrow non-overlapping boundary auxiliary signal.
- Treat partially-overlapping shared pixels as ignore/diagnostic unless ownership confidence is explicit.
- Supervise depth only on hard core plus narrow owned boundary; union mask/depth remain evaluation-only.

## Boundary

- No training, loss tuning, COMSOL run, data/NPZ mutation, checkpoint/preview export, baseline transition, or `CURRENT_BASELINE.md` update.

# 25.16 Label-V3B Derivation Validator

- acceptance_decision: `READY_FOR_25_17_TRAINING`
- route_decision: `A. enter 25.17 label-v3b training gate using 25.10 loss mainline + label-v3b supervision; do not use the 25.11/25.12 rebalance stack`
- main_conclusion: Label-v3b is derivable inside PINN_project from existing raw masks/depths: it keeps v2-style exclusive hard identity, adds capped narrow soft halo support to avoid v2 near-empty sparsity, and removes the broad v3 union-like support leakage.

## V3B Core Result

- hard core px mean/min: `99.851695` / `47`
- boundary halo px mean/min: `24.728814` / `10`
- soft support px mean/min: `124.580508` / `57`
- soft support / hard-core ratio mean: `1.253712`
- v3b soft OR / raw union ratio mean/max: `1.247726` / `1.250000`
- v3b / v3 soft OR shrink ratio mean: `0.627196`
- duplicate hard ownership count: `0`
- identity conflict px total: `915`
- ignore overlap px total: `378`
- depth valid region px mean/min: `124.580508` / `57`
- empty slot violation count: `0`
- ready_for_training_v3b: `True`

## Group Slices

- component_count=2 v3b soft OR/raw union mean/max: `1.248330` / `1.250000`
- component_count=3 v3b soft OR/raw union mean/max: `1.242697` / `1.250000`
- separated soft duplicate fraction max: `0.000000`
- close soft duplicate fraction max: `0.000000`
- touching identity conflict px mean/max: `13.458333` / `59`
- partially_overlapping identity conflict px mean/max: `19.875000` / `73`
- touching_boundary ignore overlap px mean/max: `5.541667` / `28`
- partially_overlapping ignore overlap px mean/max: `8.500000` / `42`

## Validator Checks
- `existing_hard_core_nonempty`: `True`
- `existing_depth_valid_nonempty`: `True`
- `empty_slots_clean`: `True`
- `duplicate_hard_ownership_zero`: `True`
- `hard_core_above_v2_sparse_lower_bound`: `True`
- `v3b_soft_or_ratio_below_1p35`: `True`
- `v3b_shrinks_v3_support`: `True`
- `separated_close_cross_component_soft_overlap_low`: `True`
- `touching_overlap_conflict_captured`: `True`
- `depth_valid_region_nonempty_and_nonduplicated`: `True`
- `raw_union_invariant_preserved`: `True`
- `strict_json_allow_nan_false`: `True`

## V3B Schema

- `component_hard_core_mask_v3b`: exclusive ownership-resolved component core.
- `component_boundary_halo_mask_v3b`: capped one-pixel cross-neighborhood halo, stripped of cross-component claims.
- `component_ignore_overlap_mask_v3b`: non-owner or ambiguous overlap/contact pixels for ignore/diagnostics.
- `component_mask_target_v3b_soft`: hard core = 1.0, halo = 0.35.
- `component_sdf_target_v3b`: clipped SDF from hard core, consumed only inside valid region.
- `component_valid_region_mask_v3b`: hard core plus exclusive capped halo.
- `component_depth_target_v3b`: hard-core depth plus nearest-core halo depth.
- `component_identity_conflict_mask_v3b`: raw overlap, touching contact, or multi-halo claim diagnostics.
- Union mask/depth remain raw OR/max evaluation targets only.

## Boundary

- No training, loss tuning, model expansion, COMSOL run, data/NPZ mutation, checkpoint/preview export, baseline transition, or `CURRENT_BASELINE.md` update.

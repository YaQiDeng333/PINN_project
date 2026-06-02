# 25.14 Label-V3 Derivation Validator

- acceptance_decision: `READY_FOR_25.15_TRAINING`
- route_decision: `A. enter 25.15 label-v3 training gate using 25.10 loss mainline + label-v3 supervision; do not use the 25.11/25.12 rebalance stack`
- main_conclusion: Label-v3 can be derived inside PINN_project from existing raw component masks/depths: hard ownership remains unique, union/depth invariants remain intact, and soft/local valid-region support substantially increases the component-mask learning signal that collapsed under target-v2.

## V3 Support Check

- active components: `236`
- v2 hard foreground px mean/min: `99.851695` / `47`
- v3 soft positive px mean/min/p05: `210.110169` / `126` / `147.750000`
- v3/v2 positive support ratio mean/min: `2.203184` / `1.697183`
- v3 valid-region ratio mean: `2.203184`
- v3 depth-valid ratio mean: `1.016763`
- existing v3 tiny/empty count: `0`
- existing depth-valid empty count: `0`
- inactive slot violations: `0`

## Group Slices

- component_count=3 v3/v2 support ratio mean/min: `2.445849` / `1.972789`
- partially_overlapping v3/v2 support ratio mean/min: `2.210708` / `1.795775`
- touching_boundary v3/v2 support ratio mean/min: `2.279145` / `1.842105`

## Invariants

- duplicate hard ownership: `297 -> 0`
- overlap-depth-conflict under hard ownership: `271 -> 0`
- raw OR to union mismatch px sum: `0`
- v2 OR to union mismatch px sum: `0`
- raw max-depth to union RMSE max: `0.000000000000 m`
- overlap region px sum: `290`
- contact boundary px sum: `959`

## V3 Schema

- `raw_component_mask_raw`: original component binary masks.
- `component_ownership_map`: hard unique owner map for deterministic evaluation and diagnostics.
- `component_mask_target_v3_soft`: owned=1.0, raw=0.8, one-pixel band=0.5, two-pixel band=0.25.
- `component_sdf_target_v3`: clipped signed distance field from raw component mask.
- `component_valid_region_mask`: local two-pixel supervision region.
- `component_depth_target_v3`: raw foreground depth with explicit depth-valid region.
- `overlap_region_mask` and `contact_boundary_mask`: topology diagnostics.

## Boundary

- No training, COMSOL run, loss tuning, data/NPZ mutation, checkpoint/preview export, baseline transition, or `CURRENT_BASELINE.md` update.

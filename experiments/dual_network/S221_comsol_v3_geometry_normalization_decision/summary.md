# S221 COMSOL V3 geometry normalization decision

S217-S220 repaired the V3 hard-case pilot coordinate convention enough to run the branch center-bin route against normalized V3 splits.

## Evidence

- S217 confirmed S208 raw V3 is internally self-consistent but not V2-compatible: raw V3 grid is `[0,4500] x [0,3000]`, while V2 uses centered meter-scale `[-0.04,0.04] x [-0.01,0.01]`.
- S218 created a normalized V3 copy with V2-compatible `x/y`, transformed centers, and transformed x/y axes. `signals`, `mu_maps`, and `masks` were unchanged.
- S219 rebuilt normalized parametric targets and oracle rasterization remained train/val/test IoU `1.000000` / `1.000000` / `1.000000`.
- S220 verified V2 train to normalized V3 val/test runability; the previous center-bin grid range error is gone.

## Decision

The geometry convention repair passes. The current S185 candidate is not replaced or degraded by this stage. The next stage may rerun V3 candidate evaluation on the normalized V3 pack, but it must be treated as a new evaluation stage.

## Boundary

Depth / z convention remains raw and unresolved. This is acceptable for mask oracle gating because depth is not used by the hard rasterizer, but it must be revisited before making claims about depth MAE or 3D geometry quality.

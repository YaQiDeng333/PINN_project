# S176 COMSOL center-bin route summary

## Purpose

S176 starts the center representation route after S171-S175 consolidated the current COMSOL parametric candidate. The stage keeps `raw MLP / shared head / fixed-order + lambda_center_grid=0.1` as the reference and tests whether a structured center representation can further reduce held-out center localization error.

## Preflight conclusion

- Continue the center-bin route.
- Use per-axis coarse bins rather than a 2D joint bin.
- Use bin-normalized offsets, not raw meter offsets.
- Keep fixed-order component slots.
- Keep `lambda_center_grid=0.1` as a comparison / decoded-center calibration term, but do not sweep it.

## Target representation

- `center_x_bin`
- `center_y_bin`
- `center_x_offset`
- `center_y_offset`

The default bin size is `8` grid cells. On S84 V2 grids this corresponds to about `3.216e-3 m` in x and `1.616e-3 m` in y, with about `25 x 13` bins.

## Boundary

This is still a `feature/dual-network-variational` branch experiment. It is not a main baseline replacement and does not change the current default CLI behavior.

## Self-review

The route directly targets the center bottleneck identified by S158/S161 and avoids returning to type / rotation / forward / raster loss sweeps.

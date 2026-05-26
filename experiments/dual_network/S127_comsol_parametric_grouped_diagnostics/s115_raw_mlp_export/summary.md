# COMSOL parametric grouped diagnostics summary

## Key grouped findings

- Best / worst type by type accuracy: `rectangular_notch` / `rotated_rect`.
- Best / worst slot by mask IoU: `0` / `2`.
- Best / worst rotation-error bin by mask IoU: `0-5` / `10-20`.
- Best / worst area bin by mask IoU: `medium` / `large`.
- Worst sample: `test sample 10 IoU=3.188098e-03, gap=7.423350e-01`.

## Interpretation

- This diagnostic uses exported per-component rows and per-sample rasterized mask metrics.
- Low mask IoU with non-trivial oracle gap indicates model prediction error rather than only rasterizer ceiling.
- Type or rotation groups with high error support follow-up component matching or set-style decoding tests.

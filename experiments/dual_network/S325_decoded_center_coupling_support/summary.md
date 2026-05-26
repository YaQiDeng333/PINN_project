# S325 decoded-center coupling support

S325 adds default-off decoded-center consistency support to `train_comsol_center_anchored_polygon_inverse.py`. The model structure is unchanged.

New CLI:

- `--center-consistency-mode none|soft_decoded_center|soft_decoded_vertex`, default `none`.
- `--lambda-center-consistency`, default `0.0`.
- `--center-consistency-smoothl1-beta`, default `0.1`.

Behavior:

- `none` preserves previous training and evaluation behavior.
- `soft_decoded_center` computes `softmax(center_bin_logits) @ bin_centers + center_offset * bin_width` and applies SmoothL1 in grid-cell units against `center_targets_norm`.
- `soft_decoded_vertex` adds effective local vertices to the soft decoded center and applies SmoothL1 in grid-cell units against `polygon_vertices_norm`.
- Official mask metrics and prediction export still use the hard argmax center decode. Soft decoded geometry is diagnostic/training support only.

Prediction export now records hard decoded center error, soft expected-center error, center-bin top1/top2 probability margins, and offset x/y errors per component.

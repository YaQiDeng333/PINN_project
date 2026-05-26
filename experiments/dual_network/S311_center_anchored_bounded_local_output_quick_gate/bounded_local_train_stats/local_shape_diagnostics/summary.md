# S309 center-anchored local-shape diagnostics

- prediction_dir: `experiments\dual_network\S311_center_anchored_bounded_local_output_quick_gate\bounded_local_train_stats`
- scope: read-only diagnostics; no training, model changes, or COMSOL generation.

## Target local-shape range

- train: components `39`, local_abs_x max/p95 `17.467779` / `16.870779`, local_abs_y max/p95 `3.798028` / `3.617103`, area mean/max `112.548468` / `241.782339`.
- val: components `13`, local_abs_x max/p95 `16.804443` / `16.406443`, local_abs_y max/p95 `3.567063` / `3.307825`, area mean/max `103.413414` / `210.727735`.
- test: components `12`, local_abs_x max/p95 `17.467779` / `17.102946`, local_abs_y max/p95 `3.567063` / `3.456479`, area mean/max `109.870784` / `235.186186`.

## Reference prediction linkage

- train: mean IoU `0.989528`, zero-IoU `0/30`, local_vertex_mae_grid `0.012475`, both-bin acc `1.000000`.
- val: mean IoU `0.029174`, zero-IoU `7/10`, local_vertex_mae_grid `3.447587`, both-bin acc `0.076923`.
- test: mean IoU `0.067532`, zero-IoU `9/10`, local_vertex_mae_grid `2.507451`, both-bin acc `0.166667`.

## Interpretation

- both-bin-correct components local_vertex_mae_grid: `0.235142`.
- bin-wrong components local_vertex_mae_grid: `2.978118`.
- The bounded-output gate should preserve the center-bin path and only constrain effective local vertices used by loss, decode, metrics, and prediction export.

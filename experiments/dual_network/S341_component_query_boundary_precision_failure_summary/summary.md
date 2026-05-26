# S341 Component-Query Boundary Precision Failure Summary

S341 starts a 1-sample-only boundary precision repair after S336-S340. The route remains blocked before 5-sample because the best prior repair did not reach hard IoU `>=0.99`.

Locked reference evidence:

- current_reference: IoU `0.974226804`, pred/target area `194 / 189`, FP/FN `5 / 0`
- decoded_center_aux_small: IoU `0.984126984`, pred/target area `186 / 189`, FP/FN `0 / 3`
- polygon_centroid_aux_small: IoU `0.963917526`, pred/target area `192 / 189`

The failure is now a few hard-raster boundary pixels, not presence/type/bin classification. This stage only tests one smaller center aux and one center-plus-tiny-area variant on sample `0`; it does not run 5-sample, train30, multi-seed, or new COMSOL generation.

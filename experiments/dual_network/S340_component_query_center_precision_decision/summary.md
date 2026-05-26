# S340 Component-Query Center Precision Decision

S336-S340 does not pass the component-query 1-sample repair gate.

The smallest decoded-center auxiliary is directionally useful but insufficient: IoU improves from `0.974226804` to `0.984126984`, while pred/target area shifts from `194 / 189` to `186 / 189`. The error changes from extra boundary pixels to missing boundary pixels, so this is not a clean center/centroid precision repair. The polygon-centroid auxiliary alone reduces IoU to `0.963917526`.

Acceptance required hard polygon IoU `>=0.99`, symmetric diff `<=1`, `abs(area_diff) <= 1`, stable presence/type/bin accuracy, no out-of-grid vertices, and no signed-area flip. The best repair still has IoU `<0.99`, area diff `-3`, and symmetric diff `3`, so 5-sample and train30 remain blocked.

Next unique recommendation: do not enter 5-sample yet. Plan a more targeted 1-sample precision repair that balances center shift with local polygon boundary/area, rather than continuing tiny center/centroid aux as-is or tuning lambda blindly.

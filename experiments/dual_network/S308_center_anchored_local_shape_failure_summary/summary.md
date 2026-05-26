# S308 center-anchored local-shape failure summary

S303-S307 showed that y-bin soft targets are not the main lever for the center-anchored polygon held-out failure. The same-run reference on the matched split kept strong train fit but weak held-out masks, while `neighbor_soft_y` only partially improved y-bin accuracy and did not stabilize final IoU.

This stage keeps the existing center-bin path, target schema, and model structure. The only repair under test is an optional bounded decode for `local_vertices_grid`, so the local head can no longer emit unbounded grid-cell offsets when explicitly enabled. Default `raw` mode remains unchanged.

Boundaries:

- no new COMSOL data;
- no multi-seed;
- no extra steps beyond the established `20000` gate;
- no model capacity increase;
- no S185/S181 candidate replacement;
- no main baseline replacement claim.

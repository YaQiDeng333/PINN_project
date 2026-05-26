# S314 local-shape conditioning support

Implemented default-off local-shape conditioning in the center-anchored polygon model and runner.

New CLI:

- `--local-shape-conditioning-mode none|center_bin|center_bin_slot|center_bin_slot_type`, default `none`;
- `--local-shape-conditioning-dim 16`.

Behavior:

- `none` preserves the previous local head path from shared latent to `local_vertices_grid`.
- `center_bin` feeds the local head with shared latent, predicted x/y bin soft embeddings, and predicted center offset.
- `center_bin_slot` adds learned component-slot embedding.
- `center_bin_slot_type` adds predicted type soft embedding.

The conditioning context from center-bin, offset, and type predictions is detached for the local-shape loss path, so local vertex loss does not train the center-bin or type heads through the conditioning channel.

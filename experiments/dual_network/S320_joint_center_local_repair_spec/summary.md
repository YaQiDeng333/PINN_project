# S320 joint center/local-shape repair spec

Because S319 proves center decode is the dominant held-out bottleneck, S320 defines one default-off quick repair: `joint_center_shape_mode=soft_center_scheduled`.

Implementation intent:

- Keep the existing center-bin, center-offset, presence, type, and local-vertex losses unchanged.
- Add a joint local vertex head that consumes shared signal latent plus a continuous center context.
- During training, linearly blend GT center context into predicted soft continuous center context from weight `1.0` to `0.0`.
- At eval/export, use predicted soft continuous center only.
- Do not detach center context in this joint mode, so local vertex loss can influence center logits and offset through the context path.

This is deliberately one narrow variant. It does not add differentiable raster loss, forward consistency, bbox/scale target, shared-query architecture, y-loss tuning, bound sweep, or additional local-conditioning combinations.

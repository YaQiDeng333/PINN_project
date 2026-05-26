# S312 bounded local output decision

S308-S312 completed the local-shape bounded-output repair gate without new COMSOL data, multi-seed, extra steps, model capacity changes, or S185/S181 replacement.

Decision: **bounded local output did not pass the held-out repair gate**.

Evidence:

- train fit remained strong for both bounded variants;
- val/test mean IoU did not improve over the same-run reference on both splits;
- val/test zero-IoU did not decrease on both splits;
- out-of-grid vertices and signed-area flips remained `0`;
- local output saturation stayed `0.0`, so fixed/train-stat bounds are not the active limiting mechanism.

Mechanism update: the failure is not simple unbounded local vertex explosion. The next repair should target **local-shape conditioning**, for example conditioning local-shape prediction on center bin, component slot/type, or explicit bbox/scale, while keeping this bounded-output support available as a safety option.

Next unique recommendation: plan a local-shape conditioning gate; do not continue bounded-bound sweeps, y-bin loss tuning, multi-seed, or new COMSOL data generation yet.

# S313 center-anchored local-shape conditioning plan summary

S308-S312 showed that bounded local output is not the active repair: saturation stayed `0.0`, out-of-grid vertices stayed `0`, signed-area flips stayed `0`, and held-out IoU did not improve over the same-run reference.

This stage tests default-off local-shape conditioning. The center-bin path, target schema, loss weights, matched split, training steps, and S185/S181 branch candidate remain unchanged.

The local vertex head can optionally condition on inference-time predictions only: predicted center-bin distributions, predicted center offsets, component slot, and predicted type distribution. Ground-truth center/type/bbox values are not used as conditioning inputs.

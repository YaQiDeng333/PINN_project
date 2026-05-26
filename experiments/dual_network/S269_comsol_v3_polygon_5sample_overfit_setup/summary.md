# S269 COMSOL V3 Polygon 5-Sample Overfit Setup

S269 starts the next staged polygon inverse gate after S264-S268 repaired the one-sample stop condition. This stage only validates 5-sample train overfit and does not run train30.

## Boundary

- Do not train the old center-bin candidate.
- Do not replace the S185/S181 branch candidate.
- Do not generate new COMSOL data or modify existing COMSOL exports.
- Do not save checkpoints, weights, images, or `.npy` files.

## Configuration

The 5-sample overfit uses the S267 successful longer-overfit configuration:

- steps: `10000`
- lr: `1e-3`
- hidden_dim: `128`
- latent_dim: `64`
- max_components: `3`
- max_vertices: `4`
- lambda_presence: `1.0`
- lambda_type: `1.0`
- lambda_vertex: `50.0`
- vertex_loss_space: `norm`
- vertex_smoothl1_beta: `0.005`
- seed: `1`
- export_predictions: `true`

The train/val/test inputs are all the same 5-sample subset so this is a pure overfit gate, not a held-out evaluation.

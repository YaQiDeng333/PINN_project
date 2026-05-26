# S274 COMSOL V3 Polygon Train30 Quick Probe Setup

S274 starts the train30 / val10 / test10 polygon inverse quick probe after the 1-sample and 5-sample overfit gates passed.

## Boundary

- Use the S254-S258 polygon V3 hard-case pack.
- Do not train the old center-bin candidate.
- Do not change model structure or runner code.
- Do not generate new COMSOL data.
- Do not save checkpoint, weights, image, or `.npy` artifacts.

## Configuration

The run reuses the S271 successful 5-sample overfit configuration:

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

Acceptance is based only on train fit. Val/test are observation metrics and are not promotion criteria in this stage.

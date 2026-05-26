# S66 COMSOL-style multi-height Bz dataset interface

## Purpose

S66 starts the transition from synthetic multi-height proxy signals to a real COMSOL / multi-height Bz dataset interface. It creates the schema plan, validator utility, and mock COMSOL-style smoke test needed before importing real COMSOL data.

## Recommended NPZ schema

Required fields:

- `signals`
- `mu_maps` or `masks`
- `x`
- `y`

Recommended `signals` shape:

```text
[num_samples, num_channels, signal_len]
```

Recommended metadata:

- `source_type = "comsol_multiheight"`
- `signal_channels`
- `signal_channel_names`
- `lift_off_values`
- `field_components`
- `probe_line_y_values`
- `signal_flatten_order = "channels_first"`
- `geometry_units`
- `field_units`

## Validator coverage

`comsol_multiheight_npz_utils.py` adds `validate_comsol_multiheight_npz(npz_path)`.

It checks:

- `signals` exists;
- `signals` is 3D `[N,C,L]`;
- `C >= 2`;
- `x/y` or `coords` exists;
- `mu_maps` or `masks` exists;
- `mu_maps.shape[0]` or `masks.shape[0]` matches `signals.shape[0]`;
- optional `x` is 1D and whether `len(x)` matches `signal_len`;
- optional metadata can be read into a summary dict.

The summary includes `num_samples`, `num_channels`, `signal_len`, `has_mu_maps`, `has_masks`, `has_x_y`, `has_coords`, `channel_names`, `lift_off_values`, `field_components`, `source_type`, and `notes`.

## Smoke test coverage

`smoke_test_comsol_multiheight_npz_utils.py` creates a mock COMSOL-style `.npz` with:

- `signals shape [5,3,20]`
- `x shape [20]`
- `y shape [10]`
- `mu_maps shape [5,10,20]`
- `signal_channel_names = ["Bz_liftoff_0p5", "Bz_liftoff_1p0", "Bz_liftoff_2p0"]`
- `lift_off_values = [0.5, 1.0, 2.0]`
- `field_components = ["Bz", "Bz", "Bz"]`
- `source_type = "mock_comsol_multiheight"`

The smoke test validates the schema, loads the same file through `conditional_dual_data_utils.py`, checks flattening to `[3,60]`, verifies `infer_signal_len == 60`, and runs a `ConditionalDualNet(signal_len=60)` forward pass. It also checks expected `ValueError` cases for missing fields, 2D signals, and `C < 2`.

## Boundary

S66 does not call COMSOL, does not generate real COMSOL data, and does not run formal training. It is an interface skeleton only.

## Next-step recommendation

S67 can prepare a COMSOL exporter / converter, or use the COMSOL MCP project to generate the first real multi-height Bz pilot dataset. The next step should keep schema validation explicit before any training run.

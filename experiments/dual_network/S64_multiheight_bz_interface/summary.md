# S64 multi-height Bz signal interface skeleton

## Purpose

S64 moves the conditional branch from single-Bz feature tweaks toward a multi-height / COMSOL-style Bz input interface. The goal is to make the data utilities and supervised runner accept both single-channel and multi-channel Bz signals before any formal multi-height experiment is run.

## Supported signal shapes

The conditional `.npz` interface now supports:

- Single-channel signals: `signals shape [num_samples, signal_len]`
- Multi-channel signals: `signals shape [num_samples, num_channels, signal_len]`

Other signal ranks raise `ValueError`.

## 2D / 3D signal handling

For 2D signals, the old behavior is preserved:

```text
[B, L] -> [B, L]
```

For 3D signals, `conditional_dual_data_utils.py` flattens channels-first:

```text
[B, C, L] -> [B, C * L]
```

The flatten order is:

```text
[channel0 all x, channel1 all x, ...]
```

`get_conditional_batch` also returns:

- `signal_original_shape`
- `signal_channels`
- `signal_length_per_channel`
- `flattened_signal_length`
- `signal_flatten_order`

`infer_signal_len(dataset)` returns the flattened encoder input length, so current `BzEncoder` can keep receiving `[B, signal_len]`.

## Smoke test coverage

S64 updates:

- `smoke_test_conditional_dual_data_utils.py`
  - verifies existing `[4,20]` signals behavior;
  - adds `[4,3,20]` multi-channel signals;
  - checks `get_conditional_batch(...).signals shape == [3,60]`;
  - checks `infer_signal_len(...) == 60`;
  - checks metadata and `ConditionalDualNet(signal_len=60)` forward.

- `smoke_test_train_conditional_dual.py`
  - keeps the existing supervised runner smoke test;
  - adds a multi-channel train / eval / test tempfile run with `signals shape [B,3,20]`;
  - checks train / eval / test metrics and run summary output.

## Current boundary

S64 does not run formal training and does not generate COMSOL data. It only verifies the interface skeleton. No model weights, checkpoints, arrays, or images are expected from this stage.

## Next-step recommendation

S65 can either run a synthetic multi-channel proxy probe or create a COMSOL multi-height Bz `.npz` conversion entrypoint. The next stage should preserve the current branch boundary: multi-channel Bz support is an input-interface step, not a result claim.

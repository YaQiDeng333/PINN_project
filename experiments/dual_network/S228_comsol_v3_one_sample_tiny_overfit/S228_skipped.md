# S228 skipped: one-sample tiny-overfit

S228 was skipped because S227 triggered the signal-scale stop condition.

All normalized V3 splits have `std_floor_trigger_rate=1.0` under the runner `std < 1e-8` rule. A one-sample tiny-overfit run would therefore test a near-zero signal input rather than a meaningful signal-to-geometry mapping.

No training was run. The next check should target COMSOL signal export, probe height, field expression, signal scaling, or the runner normalization floor.

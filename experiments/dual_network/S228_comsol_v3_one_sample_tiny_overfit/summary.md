# S228 COMSOL V3 one-sample tiny-overfit

Status: skipped.

S227 showed that every train/val/test normalized V3 sample triggers the runner `std < 1e-8` signal floor. The planned 1-sample overfit gate would not be interpretable, so S228 stops before running training.

Acceptance criterion was train mask IoU `>=0.8`; stop threshold was train mask IoU `<0.5`. Neither was evaluated because training did not start.

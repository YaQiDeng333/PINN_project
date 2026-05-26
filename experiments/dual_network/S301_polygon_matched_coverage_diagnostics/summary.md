# S301 Polygon Matched-Coverage Diagnostics

## Split Results

- train: mean/min IoU `0.995598` / `0.969697`, zero-IoU `0/30`, x/y bin acc `1.000000` / `1.000000`.
- val: mean/min IoU `0.037245` / `0.000000`, zero-IoU `8/10`, x/y bin acc `0.300000` / `0.250000`.
- test: mean/min IoU `0.072368` / `0.000000`, zero-IoU `9/10`, x/y bin acc `0.650000` / `0.100000`.

## Required Answers

1. Matched coverage did not reduce zero-IoU: held-out zero-IoU is `17/20`, versus original `16/20`.
2. Coverage constraint succeeded at distance <= 1 for `20/20` held-out samples, but exact same-bin coverage is only `4/20`.
3. Remaining zero-IoU is still center-bin dominated: y-bin wrong `17/17`, x-bin wrong `9/17`.
4. True rotated / multi-component remain hard: zero-IoU rotated samples `14`, multi-component zero-IoU samples `5`.
5. Because distance<=1 coverage did not improve val/test, the remaining problem is model-side y-bin/local-shape generalization unless a stricter exact-bin resplit is tested first.

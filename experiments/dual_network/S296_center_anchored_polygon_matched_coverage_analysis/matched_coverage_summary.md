# S296 Matched Coverage Analysis

## Coverage

- Held-out samples: `20`; zero-IoU samples: `16`.
- Held-out uncovered component bins: `19`; zero-IoU samples with uncovered bins: `15`; nonzero samples with uncovered bins: `1`.
- Nearest-train center-bin distance mean for zero/nonzero held-out samples: `1.468750` / `0.250000`.
- Nearest-train center-coordinate distance mean for zero/nonzero held-out samples: `2.650308e-03` / `7.604312e-04`.

## Decision Signal

- Zero-IoU correlates with train center-bin coverage gaps; matched-coverage resplit is the first diagnostic repair.

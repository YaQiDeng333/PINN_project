# S289 COMSOL V3 Center-Anchored Polygon Route Summary

S289 starts a center-anchored polygon representation route after S284-S288 showed that the absolute-vertex polygon inverse can fit train30 but fails held-out vertex/shape extrapolation.

The new route keeps the polygon target/oracle path and changes only the inverse parameterization: component center is predicted with center-bin classification plus offset, and the four polygon vertices are predicted as grid-cell local offsets relative to that decoded center. This separates global localization from local shape prediction.

Boundaries: this stage does not replace the S185/S181 center-bin candidate, does not replace the absolute-vertex polygon runner, does not run multi-seed validation, does not generate new COMSOL data, and is not a main baseline replacement.

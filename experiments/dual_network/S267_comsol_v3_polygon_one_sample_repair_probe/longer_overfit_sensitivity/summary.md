# COMSOL polygon vertex-to-raster sensitivity summary

- label: `longer_overfit`
- samples: `1`
- x grid spacing: `0.000402010041266`
- y grid spacing: `0.000202020197505`
- mean hard polygon IoU: `1.000000`
- min hard polygon IoU: `1.000000`
- mean oracle IoU from target vertices: `1.000000`
- min oracle IoU from target vertices: `1.000000`
- worst sample: `0` with IoU `1.000000` and area diff `0` pixels
- worst component max dx/dy cells: `0.003405` / `0.005855`

Interpretation: if oracle IoU remains 1.0 while predicted area and grid-cell vertex errors are non-trivial, the failure is a vertex-to-hard-raster sensitivity / loss-alignment issue rather than a polygon target or rasterizer alignment failure.

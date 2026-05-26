# COMSOL polygon vertex-to-raster sensitivity summary

- label: `train30_val`
- samples: `10`
- x grid spacing: `0.000402010041266`
- y grid spacing: `0.000202020197505`
- mean hard polygon IoU: `0.033122`
- min hard polygon IoU: `0.000000`
- mean oracle IoU from target vertices: `1.000000`
- min oracle IoU from target vertices: `1.000000`
- worst sample: `1` with IoU `0.000000` and area diff `718` pixels
- worst component max dx/dy cells: `108.803296` / `349.456515`

Interpretation: if oracle IoU remains 1.0 while predicted area and grid-cell vertex errors are non-trivial, the failure is a vertex-to-hard-raster sensitivity / loss-alignment issue rather than a polygon target or rasterizer alignment failure.

# COMSOL polygon vertex-to-raster sensitivity summary

- label: `five_sample_overfit`
- samples: `5`
- x grid spacing: `0.000402010041266`
- y grid spacing: `0.000202020197505`
- mean hard polygon IoU: `0.996028`
- min hard polygon IoU: `0.985401`
- mean oracle IoU from target vertices: `1.000000`
- min oracle IoU from target vertices: `1.000000`
- worst sample: `4` with IoU `0.985401` and area diff `-2` pixels
- worst component max dx/dy cells: `0.020669` / `0.067136`

Interpretation: if oracle IoU remains 1.0 while predicted area and grid-cell vertex errors are non-trivial, the failure is a vertex-to-hard-raster sensitivity / loss-alignment issue rather than a polygon target or rasterizer alignment failure.

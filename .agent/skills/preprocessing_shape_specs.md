# Objective
Enforce strict tensor shape parity during the `train_one_batch` preprocessing step. 

# Rules
Before proposing any changes or refactoring to the data loader, preprocessing functions, or model forward passes, you MUST verify that the input/output arrays match the following schemas exactly. This is critical for preventing shape mismatch errors during scatter/gather operations or latent grid projections.

## Sample BEFORE Preprocess
```text
vertices: shape=(1, 3586, 3)
vertex_normals: shape=(1, 3586, 3)
triangle_normals: shape=(1, 7168, 3)
centroids: shape=(1, 7168, 3)
triangle_areas: shape=(1, 7168)
distance: shape=(1, 32, 32, 32, 1)
closest_points: shape=(1, 32, 32, 32, 3)
normalized_triangle_areas: shape=(1, 7168)
press: shape=(1, 1, 3586)
neighbors_in: dict with keys ['neighbors_index', 'segment_ids', 'counts']
  neighbors_index: shape=(16681,)
  segment_ids: shape=(16681,)
  counts: shape=(32768,)
neighbors_out: dict with keys ['neighbors_index', 'segment_ids', 'counts']
  neighbors_index: shape=(16681,)
  segment_ids: shape=(16681,)
  counts: shape=(3586,)
query_points: shape=(1, 32, 32, 32, 3)
```

## Sample AFTER Preprocess
```text
input_geom: shape=(1, 3586, 3)
latent_queries: shape=(1, 32, 32, 32, 3)
output_queries: shape=(1, 3586, 3)
latent_features: shape=(1, 32, 32, 32, 1)
y: shape=(1, 1, 3586, 1)
x: type=NoneType
neighbors_in: dict with keys ['neighbors_index', 'segment_ids', 'counts']
  neighbors_index: shape=(16681,)
  segment_ids: shape=(16681,)
  counts: shape=(32768,)
neighbors_out: dict with keys ['neighbors_index', 'segment_ids', 'counts']
  neighbors_index: shape=(16681,)
  segment_ids: shape=(16681,)
  counts: shape=(3586,)
```
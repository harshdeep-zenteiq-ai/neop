# JAX/Flax Version of Car CFD Dataset

This guide explains the JAX/Flax versions of the Car CFD dataset modules.

## Files Created

1. **`car_cfd_dataset_jax.py`** - JAX version of CarCFDDataset
2. **`mesh_datamodule_jax.py`** - JAX version of MeshDataModule

## Key Changes from PyTorch

### Data Representation
- **PyTorch**: `torch.Tensor` objects
- **JAX**: `jax.numpy.ndarray` (jnp arrays)
- Both work with numpy arrays internally for compatibility with Open3D

### Dataset Structure
- **PyTorch**: Inherits from `torch.utils.data.Dataset`, uses DataLoader
- **JAX**: Returns data as Python lists of dictionaries, custom loader methods
  - Simple, no dependencies on PyTorch DataLoader
  - Returns batches as stacked JAX arrays

### Data Loading
```python
# PyTorch
loader = dataset.train_loader(batch_size=32)
for batch in loader:
    # process batch

# JAX
batches = dataset.train_loader(batch_size=32)
for batch in batches:
    # process batch
```

### Array Operations
- **PyTorch**: `torch.cat()`, `torch.stack()`, etc.
- **JAX**: `jnp.concatenate()`, `jnp.stack()`, etc.

### No Module Inheritance
- Removed `nn.Module` as base class (not needed in JAX)
- Classes are just regular Python classes
- All data is stored as JAX arrays or dictionaries

## Usage Example

```python
from neuralop.data.datasets import CarCFDDataset

# Initialize dataset
dataset = CarCFDDataset(
    root_dir="~/data/car-pressure-data",
    n_train=10,
    n_test=2,
    query_res=[32, 32, 32],
)

# Get training batches
train_batches = dataset.train_loader(batch_size=1, shuffle=True)
test_batches = dataset.test_loader(batch_size=1, shuffle=False)

# Process batches
for batch in train_batches:
    input_geom = batch["vertices"]      # JAX array
    queries = batch["query_points"]      # JAX array
    sdf = batch["distance"]              # JAX array
    pressure = batch["press"]            # JAX array

    # Use with JAX/Flax model
```

## Data Dictionary Keys

Each batch contains:
- `vertices`: Mesh vertex coordinates [n_vertices, 3]
- `query_points`: SDF query grid points [n_queries, 3]
- `distance`: Signed distance function values [n_queries, 1]
- `press`: Pressure values [batch, n_vertices]
- `centroids`: Triangle centroids [n_triangles, 3]
- `triangle_areas`: Triangle areas [n_triangles]
- `vertex_normals`: Vertex normals [n_vertices, 3]
- `triangle_normals`: Triangle normals [n_triangles, 3]

## Compatibility

- **Open3D**: Still used for mesh loading and distance computation (unchanged)
- **JAX**: All arrays converted to `jnp.ndarray` after loading
- **NumPy**: Used internally during computation, converted to JAX arrays at the end

## Performance Considerations

1. **Lazy Loading**: Batches are created on-the-fly from lists
2. **Memory**: All data kept in memory (simple approach, can be optimized with generators)
3. **JAX Compatibility**: Arrays are JAX arrays, ready for JIT compilation

## Future Enhancements

To further optimize for JAX:
1. Implement generator-based data loading for large datasets
2. Add batch prefetching with `jax.experimental.io_callback`
3. Implement `tf.data` style pipeline for better performance
4. Add multi-GPU data distribution utilities

## Testing

```python
# Quick test
from neuralop.data.datasets import CarCFDDataset

dataset = CarCFDDataset(
    root_dir="./data",
    n_train=1,
    n_test=1,
    query_res=[32, 32, 32]
)

print(f"Train batches: {len(dataset.train_loader())}")
print(f"Test batches: {len(dataset.test_loader())}")

batch = dataset.train_loader()[0]
print(f"Batch keys: {batch.keys()}")
print(f"Vertices shape: {batch['vertices'].shape}")
```

## Notes

- Both files follow the original PyTorch logic closely
- Minimal architectural changes to preserve correctness
- JAX arrays are immutable, so no in-place operations
- All numerical computations remain identical to PyTorch version

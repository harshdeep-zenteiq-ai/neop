# Neighbor Search Complexity Analysis

## Current Implementation: `native_neighbor_search_gpu()`

### Algorithm Overview
```
Input:
  - data: [n, d] where n = number of data points, d = dimensionality
  - queries: [m, d] where m = number of query points
  - radius: float
  
Output:
  - neighbors_index: [total_neighbors] (variable length)
  - neighbors_row_splits: [m+1]
  - weights (optional): [total_neighbors]
```

### Pseudocode
```
1. Precompute data squared norms: O(n*d) → store [n]
2. For each query i in 1..m:
   3. Compute query squared norm: O(d)
   4. Compute dot product with all data: O(n*d)
   5. Compute squared distances: O(n)
   6. Take sqrt: O(n)
   7. Find neighbors: O(n)
   8. Store indices + count: O(neighbors_per_query)
9. Concatenate all results: O(total_neighbors)
```

---

## Time Complexity Analysis

### Dominant Operations

| Step | Complexity | Explanation |
|------|-----------|-------------|
| Precompute data norms | O(n·d) | Sum over d dimensions for n points |
| Per query dot product | O(n·d) | Matrix-vector multiplication |
| Per query sqrt + distance | O(n) | Element-wise operations |
| Per query neighbor finding | O(n) | `jnp.where()` scan through all points |
| Concatenate results | O(total_neighbors) | Linear in output size |

### **Total Time Complexity: O(m·n·d + total_neighbors)**

Breaking it down:
- **Dominant term**: `O(m·n·d)` from dot products (m queries × n data points × d dimensions)
- **Secondary term**: `O(m·n)` from distance computation and neighbor finding
- **Output term**: `O(total_neighbors)` for concatenation

**Simplified**: **O(m·n·d)** where d is typically small (d=3 for 3D coordinates)

### For Your Problem
- m = 611 queries (500 train + 111 test)
- n = ~3600 data points per mesh
- d = 3 (3D coordinates)

**Operations per sample**:
```
m·n·d = 611 × 3600 × 3 = ~6.6 Million dot products per sample
```

**Total for full dataset**:
```
611 samples × 6.6M ops = ~4 Billion operations per sample
```

---

## Space Complexity Analysis

### Memory Usage Breakdown

| Component | Size | Complexity |
|-----------|------|-----------|
| Input: `data` | n·d float32 | O(n·d) |
| Input: `queries` | m·d float32 | O(m·d) |
| Precomputed `data_sq_norms` | n float32 | O(n) |
| Per-iteration: `query` (scalar) | 1 float32 | O(d) |
| Per-iteration: `dot_product` | n float32 | O(n) |
| Per-iteration: `sq_dists` | n float32 | O(n) |
| Per-iteration: `dists` | n float32 | O(n) |
| Per-iteration: `mask` (boolean) | n bool | O(n) |
| Per-iteration: `idxs` | neighbors_per_query int64 | O(k) where k ≤ n |
| Output: `nbr_indices_list` | total_neighbors int64 | O(total_neighbors) |
| Output: `weights_list` (optional) | total_neighbors float32 | O(total_neighbors) |

### **Total Space Complexity: O(n·d + m·d + total_neighbors + 5n)**

**Simplified**: **O(n·d + total_neighbors)** where the dominant terms are:
- **O(n·d)** for input data
- **O(total_neighbors)** for output

### Peak Memory During Execution

At any given time during a single iteration:
```
Peak = O(n) for temporary arrays (dot_product, sq_dists, dists, mask)
     + O(total_neighbors_so_far) for accumulated results
     + O(n·d) for input data
     ≈ O(n·d) since input dominates
```

### For Your Problem
With n=3600, d=3:
```
Input data:       3600 × 3 × 4 bytes = 43.2 KB per sample
Precomputed norm: 3600 × 4 bytes = 14.4 KB
Temporary arrays: 5 × 3600 × 4 bytes = 72 KB (dot product, sq_dists, etc)

Peak per sample ≈ 130 KB during neighbor search
Output (worst case, all neighbors): 3600 × m × 8 bytes ≈ variable

For m=611 queries:
Total neighbors ≈ ~5-10% of m×n = 300K-600K neighbors (empirical)
Output size ≈ 2.4-4.8 MB per sample
```

---

## Comparison: Previous vs Current Approach

### Previous: `native_neighbor_search_numpy` (Batched)

```python
# Batch size = 512
# Creates: [512, n] distance matrix
```

**Space**:
```
Batch distance matrix: 512 × 3600 × 4 bytes = 7.3 MB per batch
With 611 queries: ~2 batches → 7.3 MB peak
```

**Time**:
```
Same O(m·n·d) but with:
- Better cache locality (larger batches)
- NumPy on CPU (no GPU benefits)
```

### Current: `native_neighbor_search_gpu` (Per-Query)

**Space**:
```
Per-query vectors: 5 × 3600 × 4 bytes = 72 KB
Much lower peak memory (~100x reduction)
But suffers from:
- Loop overhead (m iterations = 611 iterations)
- Worse GPU utilization (small work per iteration)
```

**Time**:
```
Same O(m·n·d) operations but:
- GPU can parallelize dot products across data dimension
- More iterations → more overhead
- Smaller kernel launches → suboptimal GPU efficiency
```

---

## Performance Characteristics

### GPU vs CPU Trade-off

| Aspect | CPU Batch | GPU Per-Query |
|--------|-----------|---------------|
| Memory Peak | Higher (7.3 MB) | Lower (70 KB) |
| Parallelism | Low (vectorized ops) | Medium (per-query loops) |
| Kernel Launch Overhead | Low (2-3 launches) | High (611 launches) |
| Memory Bandwidth | CPU PCIe | GPU HBM (faster) |
| **Best for** | Small datasets | Large datasets with OOM risk |

---

## Optimization Opportunities

### Option 1: Batch on GPU (Hybrid)
```
Process M queries at a time where [M, n, d] fits in GPU memory
- Reduce launch overhead: 611/32 = ~20 launches
- Better GPU utilization
- Same asymptotic complexity
```

**Estimated**: 5-10x faster with same memory ~500 KB

### Option 2: Chunked Data
```
Process one query against chunks of data
- Smaller per-iteration memory: [chunk_size, d]
- More iterations but better control
- Useful if n is very large
```

**Estimated**: Similar speed, more fine-grained memory control

### Option 3: JIT Compilation
```
@jax.jit the inner loop
- Reduce Python overhead
- Better JAX graph optimization
- More memory efficient compilation
```

**Estimated**: 2-3x faster for Python loop overhead

### Option 4: Sparse Distance Computation
```
Use approximate nearest neighbors (ANN) first
- Reduce from O(n) to O(log n) candidates
- Only compute exact distances for candidates
- Trade-off: accuracy vs speed
```

**Estimated**: 50-100x faster if radius selects <1% neighbors

---

## Memory Estimates for Your Full Workload

### Setup
- **Samples**: 611 (500 train + 111 test)
- **Points per mesh**: ~3600
- **Dimensions**: 3
- **Average neighbors per query**: ~500-1000 (estimated 15-30% of points)

### Current Implementation (Per-Query Loop)

```
Per sample:
  Input data:        43.2 KB
  Data norms:        14.4 KB
  Temp arrays (peak): 72 KB
  Output:            2.4-4.8 MB (neighbors)
  ─────────────────────────────
  Total per sample:  ~3-5 MB
  
Full 611 samples:   ~2-3 GB (if stored)
But processed sequentially, so actual GPU memory: ~5 MB at a time
```

### Time Estimate (Single GPU)

```
Per sample: 611 queries × 3600 points × 3 dims
          = 6.6M dot product operations
          
At ~1 TFLOPS (conservative for dot products on GPU):
  6.6M ops = 6.6 microseconds per sample
  
But with Python loop + JAX overhead:
  ~1-5 ms per sample (realistic)
  
611 samples × 2-5 ms = 1.2-3 seconds total
```

---

## Recommendations

### Current Implementation
✅ **Pros**:
- No OOM errors (small per-iteration memory)
- Works on any GPU
- Simple and clear logic

❌ **Cons**:
- Suboptimal GPU utilization (611 kernel launches)
- Slow for large m (loops in Python/JAX)
- Missing out on batching benefits

### Suggested Improvement: Hybrid Batch

```python
def native_neighbor_search_gpu_batched(data, queries, radius, batch_size=32):
    """Batch M queries at a time on GPU"""
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]  # [batch_size, d]
        # Distance computation: [batch_size, n, d]
        # Can handle 32-64 queries without OOM
```

**Expected**:
- 10-20x faster than per-query loop
- Still 10x lower memory than original numpy batch (512)
- Good GPU utilization

---

## Summary Table

| Metric | Per-Query Loop | Batched (32) | Full Batch (611) |
|--------|---|---|---|
| **Time** | O(m·n·d) | O(m·n·d) | O(m·n·d) |
| **Space** | O(n·d) | O(32·n·d) | O(m·n·d) |
| **Peak Memory** | ~70 KB | ~430 KB | ~7.3 MB |
| **GPU Utilization** | Low | Medium | High |
| **Kernel Launches** | 611 | ~20 | 1-2 |
| **Estimated Speed (611 samples)** | 1-3 sec | 100-500 ms | OOM risk |


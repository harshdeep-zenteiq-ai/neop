# JAX/XLA Porting Analysis: IntegralTransform & segment_csr

## 1. What XLA compilation actually requires

When you call `jax.jit(fn)`, XLA traces `fn` once with **abstract values** (shape + dtype,
no actual data). It then compiles a binary for that exact shape.  
**Every tensor shape must be a compile-time constant.** Any operation whose output shape
depends on runtime values forces a recompile or a hard crash.

---

## 2. The exact lines that break in JAX

### 2a. `segment_csr` — the Python for-loop

```python
# segment_csr.py  lines 78-97
for i in range(n_out):
    if batched:
        from_idx = (slice(None), slice(indptr[0, i], indptr[0, i + 1]))
        ...
    else:
        from_idx = slice(indptr[i], indptr[i + 1])
        ...
    src_from = src[from_idx]          # <-- dynamic slice, length changes each iteration
    if n_nbrs > 0:                    # <-- branch on runtime value
        to_reduce = einsum(ein_str, src_from)
        ...
```

**Problem 1 — dynamic slice length:**  
`indptr[i+1] - indptr[i]` is a runtime value. `src[start:end]` produces a tensor whose
size is unknown at trace time. XLA cannot compile this — it does not know what shape
`src_from` will be.

**Problem 2 — data-dependent branch:**  
`if n_nbrs > 0` is a Python `if` over a runtime tensor value. During JAX tracing,
`n_nbrs` is an abstract scalar; Python cannot evaluate it as True/False.
This raises `ConcretizationTypeError` the moment JAX hits this line.

**Problem 3 — the loop itself:**  
`range(n_out)` is fine only if `n_out` is a Python int known at trace time. If `n_out`
comes from `indptr.shape[point_dim] - 1` and `indptr` has a static shape, this is
technically OK — but every *body* of the loop still contains the two problems above,
so the loop cannot be traced.

---

### 2b. `torch_scatter.segment_csr` path

```python
# segment_csr.py  lines 49-53
if importlib.util.find_spec("torch_scatter") is not None and use_scatter:
    import torch_scatter.segment_csr as scatter_segment_csr
    return scatter_segment_csr(src, indptr, reduce=reduction)
```

`torch_scatter` is a PyTorch C++ extension. It does not exist in JAX at all.
Even if you re-implement it as a JAX primitive, `segment_csr` is a **variable-output-shape**
operation: given `src` of shape `[E, C]` and `indptr` of length `N+1`, it produces
`[N, C]`. The output size `N` = number of query points is static, which is fine.
But internally it performs **gather-then-reduce over variable-length groups** — the
gather index tensor length `E` (total edges) is a runtime value that changes per sample.

---

### 2c. `IntegralTransform.forward` — dynamic neighbor indexing

```python
# integral_transform.py  line ~155
rep_features = y[neighbors["neighbors_index"]]
```

`neighbors["neighbors_index"]` has shape `[total_edges]`. `total_edges` varies per
sample (different cars have different mesh densities). XLA traces are shape-specific:
if sample A has 14957 total edges and sample B has 16943, you get **two different
compiled programs**, or more precisely a recompile on every new shape.

In PyTorch this is fine — PyTorch is eager and re-executes. In JAX `jit`, the first
call compiles for shape `[14957]`; the second call sees shape `[16943]` and
**recompiles** (if using `jit`) or raises an error (if using `jit` with
`static_argnums` incorrectly set).

```python
# integral_transform.py  lines ~165-170
num_reps = (
    neighbors["neighbors_row_splits"][1:]
    - neighbors["neighbors_row_splits"][:-1]
)
self_features = torch.repeat_interleave(x, num_reps, dim=0)
```

`torch.repeat_interleave` with a **tensor** of repeats is **not XLA-compilable**.
The output shape `[total_edges, D_X]` depends on the sum of `num_reps`, which is
a runtime value. XLA must know output shapes at compile time — this crashes with
`NonConcreteBooleanIndexError` or a shape inference error.

```python
# integral_transform.py  lines ~195-205
if rep_features.ndim == 2 and batched:
    rep_features = rep_features.unsqueeze(0).repeat(...)
rep_features.mul_(in_features)
```

`rep_features.ndim == 2` is a Python check on a static property (ndim is always
known at trace time), so this particular `if` is fine. But `.repeat()` with a
runtime-derived count is not.

---

## 3. Summary table

| Location | Line(s) | What breaks in JAX | Why |
|---|---|---|---|
| `segment_csr.py` | 78 | `for i in range(n_out)` | Loop body has dynamic-shape slices |
| `segment_csr.py` | 82/87 | `src[indptr[i]:indptr[i+1]]` | Slice bounds are runtime values → unknown output shape |
| `segment_csr.py` | 93 | `if n_nbrs > 0` | Branch on concrete runtime value during abstract trace |
| `integral_transform.py` | ~155 | `y[neighbors_index]` | `neighbors_index` length varies per sample → shape recompile |
| `integral_transform.py` | ~170 | `repeat_interleave(x, num_reps)` | Output shape = sum(num_reps), unknown at compile time |
| Both | — | Variable `total_edges` | Different samples → different shapes → no stable compiled kernel |

---

## 4. Two ways forward

---

### Way 1: Dense padding to `max_neighbors`

**Idea:** Replace the CSR format with a fixed-shape dense tensor.  
Precompute for each query point its neighbor indices, pad with a dummy index (`-1` or `0`)
up to `max_neighbors`, and store a boolean mask.

```
# CSR (current, variable shape)
neighbors_index:       [total_edges]        varies per sample
neighbors_row_splits:  [N_X + 1]

# Dense padded (static shape)
nbr_idx_padded:  [N_X, max_neighbors]       always same shape
nbr_mask:        [N_X, max_neighbors]       True = real neighbor
```

**Forward pass in JAX:**

```python
# all shapes static: [N_X, max_neighbors, D_coord]
nbr_coords  = y[jnp.clip(nbr_idx_padded, 0, N_Y-1)]
self_coords = jnp.broadcast_to(x[:, None, :], (N_X, max_neighbors, D_X))

agg = jnp.concatenate([nbr_coords, self_coords], axis=-1)  # [N_X, max, D_Y+D_X]
k_xy = mlp(agg)                                             # [N_X, max, D_out]

# gather f_y features
nbr_feats = f_y[:, jnp.clip(nbr_idx_padded, 0, N_Y-1), :]  # [B, N_X, max, D_F]

# zero out padded slots AFTER mlp (critical — see below)
k_xy = jnp.where(nbr_mask[None, :, :, None], k_xy[None], 0.0)

# integrate: sum over the max_neighbors axis
out = jnp.sum(k_xy * nbr_feats, axis=2)   # [B, N_X, D_out]
```

No loops, no dynamic slices, no scatter. Single `jnp.sum` over a static axis.  
XLA compiles this to a single fused kernel.

**Why the mask must go AFTER the MLP:**  
Padded slots use `y[0]` coordinates (due to `clip`). The MLP will compute a nonzero
`k(x_i, y_0)` for these fake edges. You must zero them with the mask *before* summing,
or they corrupt the integral. Apply mask to `k_xy`, not to `nbr_coords` as input.

**Memory cost:**

| | gno_out | gno_in |
|---|---|---|
| max_neighbors | **8** | **25** |
| mean_neighbors | ~4.5 | ~0.49 |
| padding overhead | **1.78x** | **51x** |
| verdict | excellent | expensive |

gno_out is a perfect fit.  
gno_in at 51x overhead means 98% of the `[N_grid, 25, ...]` tensor is zeros.  
For gno_in specifically you need to decide if you can afford the memory (see Way 2).

**XLA friendliness:** ✅ fully static shapes, one `jnp.sum`, compiles once.

---

### Way 2: Sparse aggregation via `jax.ops.segment_sum`

**Idea:** Keep the flat `neighbors_index` tensor but pad *it* to a fixed length
`max_total_edges`, and use `jax.ops.segment_sum` (JAX's native scatter-reduce).

```python
# Pad flat edge list to max_total_edges (from dataset statistics)
nbr_idx_padded:    [max_total_edges]     padded with 0
segment_ids:       [max_total_edges]     which query point each edge belongs to
edge_mask:         [max_total_edges]     True = real edge

f_gathered = f_y[:, nbr_idx_padded, :]   # [B, max_total_edges, D_F]
f_gathered = jnp.where(edge_mask[None, :, None], f_gathered, 0.0)

# kernel mlp on agg coords...
k_xy = mlp(agg)                           # [max_total_edges, D_out]
k_xy = jnp.where(edge_mask[:, None], k_xy, 0.0)

out = jax.ops.segment_sum(
    k_xy * f_gathered[b],
    segment_ids,
    num_segments=N_X
)                                         # [N_X, D_out]
```

`jax.ops.segment_sum` is XLA-native and fully JIT-compilable as long as
`num_segments` and the length of `segment_ids` are static.

**Memory cost:**

| | gno_out | gno_in |
|---|---|---|
| max_total_edges | ~8 * N_X | ~16943 (from stats) |
| mean_total_edges | ~4.5 * N_X | ~16167 |
| padding overhead | **1.78x** | **1.05x** |

For gno_in this is dramatically more memory-efficient than Way 1 (1.05x vs 51x).

**XLA friendliness:** ✅ static shapes, JAX-native scatter.  
**Extra complexity:** you need to precompute `segment_ids` alongside `neighbors_index`.

---

## 5. Head-to-head comparison

| | Way 1 (dense pad) | Way 2 (flat pad + segment_sum) |
|---|---|---|
| XLA compilable | ✅ | ✅ |
| gno_out memory overhead | 1.78x ✅ | 1.78x ✅ |
| gno_in memory overhead | 51x ❌ | 1.05x ✅ |
| Implementation complexity | low | medium |
| MLP call structure | batched over neighbors dim (nice) | flat edge list (less intuitive) |
| Best for | gno_out | gno_in |

---

## 6. Recommended hybrid approach

Given your statistics:

- **gno_out → Way 1 (dense pad to 8):** max/mean = 1.78x, trivial overhead, clean code.
- **gno_in → Way 2 (flat pad to 16943):** max/mean = 1.05x, no wasted memory.

This gives you the best of both worlds. Both are fully JAX/XLA compatible.  
Both require only changes to `precompute_neighbors` (add padded tensors) and
the GNO forward pass (replace `repeat_interleave` + `segment_csr` with the
static-shape equivalents above). The neighbor search itself (Open3D / torch fallback)
stays in PyTorch preprocessing — JAX only sees the precomputed padded arrays.

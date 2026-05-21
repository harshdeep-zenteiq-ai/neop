# JAX JIT + Neighbor Padding Plan

## Root Causes of Slowness

1. **No JIT anywhere**: Every `train_step` runs in pure eager mode. JAX traces
   and dispatches XLA kernels from scratch on every single batch. No compiled
   computation is reused.

2. **Variable neighbor-array shapes**: `neighbors_index` length differs per
   sample (e.g. 15832 – 16681 edges for a 5-sample set). JAX treats shape as
   part of the "trace key", so even if JIT were applied it would recompile for
   every new shape. This also forces `segment_csr` to call `np.asarray(indptr)`
   which blocks XLA tracing.

---

## Planned Changes

### 1. `neuralop/layers/segment_csr_jax.py`

**What**: Replace the current interface that accepts a CSR `indptr` array with
one that accepts pre-computed, pre-padded `segment_ids` and a `counts` array.

**New signature**:
```
segment_csr(src, segment_ids, counts, n_out, reduction='mean', use_scatter=False)
```

- `src`          – edge features `[max_edges, C]` or `[B, max_edges, C]`
- `segment_ids`  – int array `[max_edges]`; padded edges have value `n_out` (dummy)
- `counts`       – real-edge count per node `[n_out]`, used for mean division
- `n_out`        – int, number of real output nodes

**Why**: All inputs now have **static shapes** across batches. JAX can JIT-compile
`jax.ops.segment_sum` once and reuse it. The Python `np.asarray` / `np.repeat`
calls (which break tracing) move to a one-time preprocessing step.

---

### 2. `neuralop/layers/integral_transform_jax.py`

**What**: Change the `segment_csr` call to pass the new arguments.

Current call:
```python
splits = neighbors["neighbors_row_splits"]
out_features = segment_csr(rep_features, splits, reduction=reduction, ...)
```

New call:
```python
out_features = segment_csr(
    rep_features,
    neighbors["segment_ids"],   # precomputed, padded
    neighbors["counts"],        # per-node edge counts
    neighbors["n_out"],         # number of real output nodes
    reduction=reduction,
)
```

Also: `rep_features = y[neighbors["neighbors_index"]]` already works with a
padded `neighbors_index` (padded entries point to row 0 of `y`, which is safe
because the corresponding `segment_ids` value is `n_out`, so their contribution
is discarded by `segment_csr`).

---

### 3. `scripts/train_gino_carcfd_jax.py` — padding helper

**What**: Add a `pad_neighbor_data(data_list, max_in_edges, max_out_edges)`
helper function called **once** after `precompute_neighbors`.

Steps inside the helper:
1. Compute `max_in_edges` and `max_out_edges` across all samples in the dataset.
2. For each sample, for each of `neighbors_in` / `neighbors_out`:
   a. Read `neighbors_index` (shape `[n_real]`) and `neighbors_row_splits`
      (shape `[n_out+1]`).
   b. Compute `counts = splits[1:] - splits[:-1]`  → `[n_out]`.
   c. Compute `segment_ids = np.repeat(np.arange(n_out), counts)` → `[n_real]`.
   d. Pad `neighbors_index` to `max_edges` with `0`
      (safe — padded entries map to row 0 of `y`, but get discarded).
   e. Pad `segment_ids` to `max_edges` with `n_out`
      (dummy segment that gets dropped by `segment_csr`).
   f. Replace the neighbors dict with:
      ```python
      {
          "neighbors_index": jnp.array(padded_index),   # [max_edges]
          "segment_ids":     jnp.array(padded_seg_ids), # [max_edges]
          "counts":          jnp.array(counts),          # [n_out]
          "n_out":           n_out,                      # Python int
      }
      ```
      (Drop `neighbors_row_splits` — no longer needed at runtime.)

Call site (after both `precompute_neighbors` calls):
```python
max_in, max_out = pad_neighbor_data(
    data_module_jax.train_data.data_list + data_module_jax.test_data.data_list
)
print(f"Padded neighbors: max_in_edges={max_in}, max_out_edges={max_out}")
```

---

### 4. `scripts/train_gino_carcfd_jax.py` — JIT the train step

**What**: Restructure `FlaxModelWrapper.train_step` so the inner gradient
function is compiled **once** with `jax.jit` and reused every batch.

New design inside `FlaxModelWrapper`:

```python
self._jit_fn = None  # cached after first init

def _build_jit_fn(self, loss_fn):
    module = self.module
    tx = self.tx

    @jax.jit
    def _step(params, opt_state, model_inputs, y):
        def forward_loss(p):
            out = module.apply(p, **model_inputs)
            return loss_fn(out, y=y)
        loss, grads = jax.value_and_grad(forward_loss)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    return _step

def train_step(self, kwargs, loss_fn):
    if not self._initialized:
        self._init(kwargs)
    if self._jit_fn is None:
        self._jit_fn = self._build_jit_fn(loss_fn)

    model_keys = {'input_geom', 'latent_queries', 'output_queries',
                  'x', 'latent_features', 'ada_in', 'neighbors_in', 'neighbors_out'}
    model_inputs = {k: v for k, v in kwargs.items() if k in model_keys}
    y = kwargs['y']

    loss, self.params, self.opt_state = self._jit_fn(
        self.params, self.opt_state, model_inputs, y
    )
    return float(loss)
```

**Why this works**: `@jax.jit` is applied once at `_build_jit_fn` time. The
function takes explicit JAX-array arguments (`params`, `opt_state`,
`model_inputs`, `y`). With fixed-shape inputs (from padding), JAX compiles the
full forward+backward pass once and reuses the XLA executable every batch.

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `neuralop/layers/segment_csr_jax.py` | New signature: `(src, segment_ids, counts, n_out, reduction)` |
| `neuralop/layers/integral_transform_jax.py` | Pass `segment_ids`, `counts`, `n_out` from neighbors dict |
| `scripts/train_gino_carcfd_jax.py` | Add `pad_neighbor_data()` helper + JIT-based `FlaxModelWrapper` |

No changes to the model, data loading, or trainer files.

# Neighbor Search Precomputation — All Changes

---

## Step 1 — `neuralop/layers/gno_block.py`

**What:** Added optional `neighbors` parameter to `GNOBlock.forward()`.

```python
# Before
def forward(self, y, x, f_y=None):
    neighbors_dict = self.neighbor_search(data=y, queries=x, radius=self.radius)

# After
def forward(self, y, x, f_y=None, neighbors=None):
    if neighbors is None:
        neighbors_dict = self.neighbor_search(data=y, queries=x, radius=self.radius)
    else:
        neighbors_dict = neighbors
```

**Why:** Every other change feeds precomputed neighbors into here. This is the gate — if `neighbors` is provided, the search is skipped entirely. If not, falls back to normal behavior. Nothing else in the codebase breaks.

---

## Step 2 — `neuralop/models/gino.py` (new method)

**What:** Added `precompute_neighbors(dataset, output_n_points)` method to the `GINO` class.

```python
def precompute_neighbors(self, dataset, output_n_points=None):
    latent_queries = dataset.constant["query_points"]          # fixed grid, shape [d1,d2,d3,3]
    latent_queries_flat = latent_queries.view(-1, latent_queries.shape[-1])  # [m, 3]

    # gno_out: compute once — same for every sample
    first_vertices = dataset.data_list[0]["vertices"]
    if output_n_points is not None:
        first_out_p = first_vertices[:output_n_points]
    else:
        first_out_p = first_vertices

    gno_out_neighbors = self.gno_out.neighbor_search(
        data=latent_queries_flat,
        queries=first_out_p,
        radius=self.gno_out.radius,
    )

    # gno_in: compute per sample — depends on each car's mesh
    for sample in dataset.data_list:
        vertices = sample["vertices"]
        sample["neighbors_in"] = self.gno_in.neighbor_search(
            data=vertices,
            queries=latent_queries_flat,
            radius=self.gno_in.radius,
        )
        sample["neighbors_out"] = gno_out_neighbors  # same reference for all
```

**Why:**
- `gno_out` neighbors only depend on `latent_queries` → `output_queries`, both fixed. Computed once, shared across all samples via the same dict reference.
- `gno_in` neighbors depend on each car's mesh (`vertices`), so computed once per sample.
- `output_n_points=3586` matches the truncation `out_p = out_p[:output_vertices]` in `GINOCFDDataProcessor.preprocess()`. Must stay in sync.

---

## Step 3 — `neuralop/models/gino.py` (`GINO.forward()`)

**What:** Added `neighbors_in` and `neighbors_out` to the forward signature, passed into both GNO calls.

```python
# Before
def forward(self, input_geom, latent_queries, output_queries,
            x=None, latent_features=None, ada_in=None, **kwargs):

# After
def forward(self, input_geom, latent_queries, output_queries,
            x=None, latent_features=None, ada_in=None,
            neighbors_in=None, neighbors_out=None, **kwargs):
```

Three call sites updated:

```python
# gno_in call
in_p = self.gno_in(
    y=input_geom,
    x=latent_queries.view((-1, latent_queries.shape[-1])),
    f_y=x,
    neighbors=neighbors_in,      # NEW
)

# gno_out call (tensor branch)
out = self.gno_out(
    y=latent_queries.reshape((-1, latent_queries.shape[-1])),
    x=output_queries,
    f_y=latent_embed,
    neighbors=neighbors_out,     # NEW
)

# gno_out call (dict branch)
sub_output = self.gno_out(
    y=latent_queries.reshape((-1, latent_queries.shape[-1])),
    x=out_p,
    f_y=latent_embed,
    neighbors=neighbors_out,     # NEW
)
```

**Why:** `GINO.forward()` is called via `model(**sample)` by the trainer. Since `neighbors_in` and `neighbors_out` are now keys in the sample dict, they arrive here automatically. Passing `None` (when not precomputed) falls back cleanly to Step 1's behavior.

---

## Step 4 — `scripts/train_gino_carcfd.py`

### 4a — Precompute before training

```python
# Added right after model = get_model(config)
model.precompute_neighbors(data_module.train_data, output_n_points=3586)
model.precompute_neighbors(data_module.test_data,  output_n_points=3586)
```

**Why:** This is the single trigger for all precomputation. Must happen after both the model and datamodule are built, and before the trainer starts.

### 4b — Move neighbors to device in `GINOCFDDataProcessor.preprocess()`

```python
# Added at the end of preprocess(), after sample.update(batch_dict)
for key in ("neighbors_in", "neighbors_out"):
    if key in sample and sample[key] is not None:
        sample[key] = {
            k: v.to(self.device) for k, v in sample[key].items()
        }
```

**Why:** Neighbor dicts contain `neighbors_index` and `neighbors_row_splits` tensors. They are precomputed on CPU. The model runs on GPU. Without this, the integral transform would fail with a device mismatch error. This moves them to the correct device each batch (cheap — just a pointer move if already there).

---

## Things to watch out for

| Issue | Detail |
|---|---|
| `output_n_points=3586` | Must match `truth.shape[1]` in your dataset. If cars have fewer vertices, precomputed `neighbors_out` will be wrong. |
| `gno_out` is shared | All samples point to the same `neighbors_out` dict. Do not mutate it. |
| Test set | `precompute_neighbors` must be called on both `train_data` and `test_data` separately. |
| New cars / data | If you reload the dataset or add samples, call `precompute_neighbors` again. |
| Model on GPU before precompute | `precompute_neighbors` runs neighbor search via the model's `gno_in`/`gno_out` modules. If the model is on GPU at that point, results will be on GPU already — remove the `.to(self.device)` step or it becomes a no-op. Currently model stays on CPU during precompute which is fine. |

"""
trace_integral_transform.py

A step-by-step walkthrough of IntegralTransform.forward() with
printed shapes and values at every stage.

We simulate a realistic GINO scenario:
  - OUTPUT GNO pass:
      y = latent grid points  (flattened from e.g. 8x8x8 = 512 points, dim=3)
      x = output surface query points  (e.g. 200 mesh vertices, dim=3)
      f_y = latent FNO embeddings on the grid  (batch=2, 512 points, 64 channels)
      neighbors = for each surface point x_i, which grid points y_j are within radius r?

In the actual GINO forward pass:
  - gno_in:  y=input_geom (car mesh, ~10k pts), x=latent_grid_flat (~512 pts),
             f_y=input features (batch, n_mesh, in_channels)
  - gno_out: y=latent_grid_flat (~512 pts), x=output_queries (car mesh, ~10k pts),
             f_y=latent_embed (batch, 512, fno_hidden_channels=64)

Run with:
    cd /home/deepan/neuraloperator
    python scripts/trace_integral_transform.py
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 0: Imports")
print("="*70)

import sys
sys.path.insert(0, "/home/deepan/neuraloperator")

from neuralop.layers.channel_mlp import LinearChannelMLP
from neuralop.layers.segment_csr import segment_csr

print("  Imported LinearChannelMLP, segment_csr — OK")

# ---------------------------------------------------------------------------
# 1. Define problem sizes (realistic GINO-scale, but small for readability)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 1: Problem setup")
print("="*70)

torch.manual_seed(0)

BATCH       = 2        # number of samples in the batch
N_Y         = 16       # number of 'y' points  (latent grid pts; real GINO: ~512-4096)
N_X         = 8        # number of 'x' points  (output surface pts; real GINO: ~10k)
D_Y         = 3        # spatial dim of y  (3-D coordinates)
D_X         = 3        # spatial dim of x
D_F         = 4        # feature channels on y  (real GINO gno_out: fno_hidden_channels=64)
D_OUT       = 4        # output channels of the kernel MLP  (same as D_F here)
RADIUS      = 0.5      # neighbor search radius

print(f"  BATCH={BATCH}, N_Y={N_Y} (grid pts), N_X={N_X} (surface pts)")
print(f"  D_Y={D_Y} (coord dim), D_F={D_F} (feature channels), RADIUS={RADIUS}")
print()
print("  In real GINO gno_out:")
print("    N_Y  ~ 512  (8x8x8 latent grid, flattened)")
print("    N_X  ~ 10k  (car-surface mesh vertices)")
print("    D_F  = 64   (fno_hidden_channels)")
print("    BATCH varies per training run (often 1-4 for large meshes)")

# ---------------------------------------------------------------------------
# 2. Create inputs
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 2: Create y, x, f_y")
print("="*70)

# y: the source points  (latent grid flattened)
y = torch.rand(N_Y, D_Y)
print(f"\n  y (source/grid points):  shape={tuple(y.shape)}")
print(f"    Each row is a 3-D coordinate of a latent grid point.")
print(f"    In GINO gno_out: y = latent_queries.reshape(-1, 3)  shape=({N_Y}, 3)")
print(f"    y[:3] =\n{y[:3]}")

# x: the query points  (output mesh vertices)
x = torch.rand(N_X, D_X)
print(f"\n  x (query/output points): shape={tuple(x.shape)}")
print(f"    Each row is a 3-D coordinate of an output surface vertex.")
print(f"    In GINO gno_out: x = output_queries  shape=({N_X}, 3)")
print(f"    x[:3] =\n{x[:3]}")

# f_y: feature function defined on y  (batched)
f_y = torch.rand(BATCH, N_Y, D_F)
print(f"\n  f_y (features on y):     shape={tuple(f_y.shape)}")
print(f"    In GINO gno_out: f_y = latent_embed  shape=({BATCH}, {N_Y}, 64)")
print(f"    These are the FNO latent embeddings we want to integrate.")
print(f"    f_y[0, :3] =\n{f_y[0, :3]}")

# ---------------------------------------------------------------------------
# 3. Build neighbor structure (CRS format)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 3: Build neighbors dict (CRS format)")
print("="*70)

print("""
  'neighbors' dict has two keys:
    'neighbors_index'      : 1-D tensor — concatenated list of y-indices
                             that are neighbors of each x point.
    'neighbors_row_splits' : 1-D tensor of length N_X+1 — cumulative offsets.
                             neighbors of x[i] are at
                             neighbors_index[ row_splits[i] : row_splits[i+1] ]

  This is the standard CSR (Compressed Sparse Row) graph format.
  In real GINO: produced by gno_out.neighbor_search(data=y, queries=x, radius=r)
  which calls Open3D (or PyTorch fallback) radius search.
""")

# Build fake neighbors: for each x_i, pick a random subset of y indices
neighbors_index = []
row_splits = [0]
for i in range(N_X):
    # pick 2-5 random y neighbors
    n_nbrs = torch.randint(2, 6, ()).item()
    nbrs   = torch.randperm(N_Y)[:n_nbrs].sort().values
    neighbors_index.append(nbrs)
    row_splits.append(row_splits[-1] + n_nbrs)

neighbors_index = torch.cat(neighbors_index)  # shape: [total_edges]
row_splits_t    = torch.tensor(row_splits, dtype=torch.long)

neighbors = {
    "neighbors_index":      neighbors_index,
    "neighbors_row_splits": row_splits_t,
}

total_edges = neighbors_index.shape[0]
print(f"  neighbors_index:      shape={tuple(neighbors_index.shape)}  (total edges = {total_edges})")
print(f"    Each value is an index into y (0..{N_Y-1}).")
print(f"    neighbors_index[:10] = {neighbors_index[:10].tolist()}")

print(f"\n  neighbors_row_splits: shape={tuple(row_splits_t.shape)}  (length = N_X+1 = {N_X+1})")
print(f"    row_splits_t = {row_splits_t.tolist()}")
print(f"\n  Meaning: x[0] has neighbors y[ {row_splits_t[0]}:{row_splits_t[1]} ]  -> {row_splits_t[1]-row_splits_t[0]} neighbors")
print(f"           x[1] has neighbors y[ {row_splits_t[1]}:{row_splits_t[2]} ]  -> {row_splits_t[2]-row_splits_t[1]} neighbors")

# ---------------------------------------------------------------------------
# 4. Build the kernel MLP
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 4: Build the kernel MLP (channel_mlp)")
print("="*70)

print(f"""
  The MLP takes concatenated [y_neighbor_coords, x_query_coords] as input.
  Input dim  = D_Y + D_X = {D_Y} + {D_X} = {D_Y+D_X}
  (For transform_type='nonlinear', f_y features are also concatenated,
   making input dim = D_Y + D_X + D_F = {D_Y+D_X+D_F})

  In real GINO gno_out (transform_type='linear'):
    Input dim  = 3 + 3 = 6  (or more if positional embeddings are used)
    Hidden layers = [512, 256]
    Output dim = fno_hidden_channels = 64
""")

mlp_layers = [D_Y + D_X, 8, D_OUT]   # small MLP for demo
channel_mlp = LinearChannelMLP(layers=mlp_layers, non_linearity=F.gelu)
print(f"  MLP layers: {mlp_layers}  (input -> hidden -> output)")
print(f"  channel_mlp = {channel_mlp}")

transform_type = "linear"
print(f"\n  transform_type = '{transform_type}'")
print("""  Options:
    'linear_kernelonly' -> out = integral of k(x,y) dy                [no f_y]
    'linear'            -> out = integral of k(x,y) * f(y) dy         [kernel * features]
    'nonlinear_kernelonly' -> out = integral of k(x,y,f(y)) dy
    'nonlinear'         -> out = integral of k(x,y,f(y)) * f(y) dy
""")

# ---------------------------------------------------------------------------
# 5. Walk through forward() line by line
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 5: Walking through IntegralTransform.forward()")
print("="*70)

# ----- line: if x is None: x = y  -----
print("\n--- Line: 'if x is None: x = y' ---")
print(f"  x is provided: shape={tuple(x.shape)}  (not replaced)")
print(f"  In some GNO calls (e.g. autoencoder), x may equal y for self-integration.")

# ----- line: rep_features = y[neighbors["neighbors_index"]] -----
print("\n--- Line: rep_features = y[neighbors['neighbors_index']] ---")
rep_features = y[neighbors["neighbors_index"]]
print(f"  We index into y using neighbors_index to gather the coordinates")
print(f"  of every neighbor, for every query point, into one flat tensor.")
print(f"  y.shape            = {tuple(y.shape)}")
print(f"  neighbors_index    = {neighbors_index.tolist()}")
print(f"  rep_features.shape = {tuple(rep_features.shape)}")
print(f"    Shape: [total_edges, D_Y] = [{total_edges}, {D_Y}]")
print(f"    rep_features[:4] =\n{rep_features[:4]}")
print(f"\n  Think of it as: for every (x_i, y_j) edge in the neighbor graph,")
print(f"  grab the coordinates of y_j.")

# ----- batching detection -----
print("\n--- Lines: detect batching from f_y ---")
batched = False
if f_y is not None:
    if f_y.ndim == 3:
        batched = True
        batch_size = f_y.shape[0]
        in_features = f_y[:, neighbors["neighbors_index"], :]
    elif f_y.ndim == 2:
        batched = False
        in_features = f_y[neighbors["neighbors_index"]]

print(f"  f_y.ndim = {f_y.ndim}  -> batched = {batched}, batch_size = {batch_size}")
print(f"  in_features = f_y[:, neighbors_index, :]")
print(f"  in_features.shape = {tuple(in_features.shape)}")
print(f"    Shape: [batch, total_edges, D_F] = [{BATCH}, {total_edges}, {D_F}]")
print(f"    This gathers, for each edge (x_i, y_j), the feature vector f(y_j).")
print(f"    in_features[0, :4] =\n{in_features[0, :4]}")

# ----- num_reps -----
print("\n--- Line: num_reps = row_splits[1:] - row_splits[:-1] ---")
num_reps = (
    neighbors["neighbors_row_splits"][1:]
    - neighbors["neighbors_row_splits"][:-1]
)
print(f"  This gives the neighbor count for each query point x_i.")
print(f"  num_reps = {num_reps.tolist()}")
print(f"  num_reps.shape = {tuple(num_reps.shape)}  (one entry per x point)")

# ----- self_features -----
print("\n--- Line: self_features = torch.repeat_interleave(x, num_reps, dim=0) ---")
self_features = torch.repeat_interleave(x, num_reps, dim=0)
print(f"  x.shape            = {tuple(x.shape)}")
print(f"  self_features.shape= {tuple(self_features.shape)}")
print(f"    Shape: [total_edges, D_X] = [{total_edges}, {D_X}]")
print(f"    For each edge (x_i, y_j), we repeat x_i's coordinates.")
print(f"    So row k of self_features = coordinates of the query point")
print(f"    that 'owns' the k-th edge.")
print(f"    self_features[:4] =\n{self_features[:4]}")

# ----- agg_features -----
print("\n--- Line: agg_features = torch.cat([rep_features, self_features], dim=-1) ---")
agg_features = torch.cat([rep_features, self_features], dim=-1)
print(f"  Concatenate neighbor coords + query coords along the channel dim.")
print(f"  rep_features.shape  = {tuple(rep_features.shape)}  (neighbor y_j coords)")
print(f"  self_features.shape = {tuple(self_features.shape)}  (query    x_i coords)")
print(f"  agg_features.shape  = {tuple(agg_features.shape)}")
print(f"    Shape: [total_edges, D_Y+D_X] = [{total_edges}, {D_Y+D_X}]")
print(f"    Each row encodes one (x_i, y_j) pair: [y_j_coords | x_i_coords].")
print(f"    This is the input to the kernel MLP k(x,y).")
print(f"    agg_features[:4] =\n{agg_features[:4]}")

print(f"\n  NOTE: For transform_type='nonlinear', f_y features would also be")
print(f"  concatenated here: agg_features = cat([rep_features, self_features, in_features])")
print(f"  giving shape [{total_edges}, {D_Y+D_X+D_F}]. We skip this since transform_type='{transform_type}'.")

# ----- channel_mlp -----
print("\n--- Line: rep_features = self.channel_mlp(agg_features) ---")
rep_features = channel_mlp(agg_features)
print(f"  Run every (x_i, y_j) pair through the kernel MLP.")
print(f"  agg_features.shape  = {tuple(agg_features.shape)}  (input)")
print(f"  rep_features.shape  = {tuple(rep_features.shape)}  (output)")
print(f"    Shape: [total_edges, D_OUT] = [{total_edges}, {D_OUT}]")
print(f"    Each row is now k(x_i, y_j), the kernel value for that edge.")
print(f"    rep_features[:4] =\n{rep_features[:4]}")

# ----- multiply by f_y (linear transform) -----
print("\n--- Lines: for transform_type='linear' -> rep_features *= in_features ---")
print(f"  This is the step that turns k(x,y) into k(x,y)*f(y).")
if rep_features.ndim == 2 and batched:
    rep_features = rep_features.unsqueeze(0).repeat([batch_size] + [1] * rep_features.ndim)
    print(f"  rep_features unsqueezed and broadcast to batch dim.")
    print(f"  rep_features.shape = {tuple(rep_features.shape)}")

rep_features.mul_(in_features)
print(f"  in_features.shape  = {tuple(in_features.shape)}")
print(f"  rep_features.shape = {tuple(rep_features.shape)}")
print(f"    Shape: [batch, total_edges, D_OUT] = [{BATCH}, {total_edges}, {D_OUT}]")
print(f"    Element-wise product: k(x_i, y_j) * f(y_j)  for every edge.")
print(f"    rep_features[0, :4] =\n{rep_features[0, :4]}")

# ----- no weights in this example -----
print("\n--- Lines: check for weights / weighting_fn ---")
nbr_weights = neighbors.get("weights")
print(f"  neighbors.get('weights') = {nbr_weights}  (None -> no per-neighbor weights)")
print(f"  weighting_fn             = None  (no mollified GNO weighting)")
reduction = "sum"
print(f"  reduction = '{reduction}'")
print(f"\n  In mollified GNO (gno_weighting_function != None):")
print(f"    neighbors dict includes 'weights' key with shape [total_edges].")
print(f"    These are evaluated by weighting_fn (e.g. bump/quartic kernel)")
print(f"    and multiplied into rep_features before aggregation.")

# ----- segment_csr aggregation -----
print("\n--- Line: out_features = segment_csr(rep_features, splits, reduction='sum') ---")
splits = neighbors["neighbors_row_splits"]
if batched:
    splits = splits.unsqueeze(0).repeat([batch_size] + [1] * (splits.ndim))
print(f"  splits.shape       = {tuple(splits.shape)}")
print(f"    Shape: [batch, N_X+1] = [{BATCH}, {N_X+1}]")
print(f"    splits[b, i] to splits[b, i+1] = range of edges belonging to x_i in batch b.")
print(f"\n  segment_csr sums up all k(x_i,y_j)*f(y_j) for j in A(x_i).")
print(f"  This is the discrete approximation of the integral:")
print(f"    out(x_i) ≈ Σ_{{j∈A(x_i)}} k(x_i, y_j) * f(y_j)")

out_features = segment_csr(
    rep_features,
    splits,
    reduction=reduction,
    use_scatter=False,   # use pure-Python fallback; set True if torch-scatter installed
)
print(f"\n  out_features.shape = {tuple(out_features.shape)}")
print(f"    Shape: [batch, N_X, D_OUT] = [{BATCH}, {N_X}, {D_OUT}]")
print(f"    This is the final output: one D_OUT-dim vector per output query point.")
print(f"    out_features[0] =\n{out_features[0]}")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 6: End-to-end shape summary")
print("="*70)
print(f"""
  INPUT
    y         {tuple(y.shape)}          source (grid) point coordinates
    x         {tuple(x.shape)}           query (surface) point coordinates
    f_y       {tuple(f_y.shape)}       features on y  [batch, N_Y, D_F]

  INTERMEDIATE
    rep_features (after index)  [{total_edges}, {D_Y}]   y-coords of each edge's neighbor
    in_features                 [{BATCH}, {total_edges}, {D_F}]   f(y) for each edge  [batched]
    num_reps                    [{N_X}]              neighbor count per query pt
    self_features               [{total_edges}, {D_X}]   x-coords repeated per edge
    agg_features                [{total_edges}, {D_Y+D_X}]   [y_j | x_i] per edge -> MLP input
    rep_features (after MLP)    [{total_edges}, {D_OUT}]   k(x_i, y_j) per edge
    rep_features (after mul)    [{BATCH}, {total_edges}, {D_OUT}]   k(x_i,y_j)*f(y_j) per edge

  OUTPUT
    out_features  {tuple(out_features.shape)}       integral result  [batch, N_X, D_OUT]

  In real GINO gno_out:
    y          (512, 3)     latent grid flattened
    x          (10000, 3)   car surface vertices
    f_y        (B, 512, 64) FNO latent embeddings
    out_features (B, 10000, 64)  -> passed to projection MLP -> (B, 10000, out_channels)
""")

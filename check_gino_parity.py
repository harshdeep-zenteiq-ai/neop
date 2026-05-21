"""
Standalone GINO PyTorch vs JAX parity checker.

This script:
1. Instantiates the PyTorch GINO model and the JAX (Flax) GINO model with
   identical hyperparameters.
2. Initializes JAX params (random Flax init) to get the canonical pytree.
3. Walks the PyTorch state_dict and overwrites every leaf in the JAX
   params dict with the PyTorch values (handling shape/transposition
   differences between Conv1d, Dense, soft-gating, and SpectralConv weights).
4. Builds a small synthetic problem (small grid + a few input/output points),
   precomputes neighbors once with torch, converts them to both formats,
   and runs the forward pass on both models.
5. Reports max-abs / mean-abs error between PyTorch and JAX outputs.

Run:
    python check_gino_parity.py
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import numpy as np
import torch
import jax
import jax.numpy as jnp

from neuralop.models.gino import GINO as TorchGINO
from neuralop.models.gino_jax import GINO as JaxGINO
from neuralop.layers.neighbor_search import native_neighbor_search


# ---------------------------------------------------------------------------
# Shared model config — mirrors the actual training config used in
# train_gino_carcfd_jax.py: GINO_Small3d + GINOConfig defaults from
# config/models.py + GINO model class defaults from neuralop/models/gino.py.
#
# Key choices vs. the previous (small) parity config:
#   - fno_factorization="tucker", fno_rank=0.4 — exercises the Tucker
#     transfer path. Safe after the round-vs-ceil fix in spectral_convolution_jax.
#   - fno_norm="instance_norm" — exercises the JAX InstanceNorm against
#     torch.nn.functional.instance_norm. Both should be parameter-free.
#   - fno_n_modes=(16,16,16), fno_hidden_channels=64, fno_n_layers=4 — full
#     production model size; per-case forward will be ~100x slower than before.
#   - in_channels=0 (data_channels=0 in GINO_Small3d) -> x is None in forward.
#   - fno_channel_mlp_expansion=1.0 — GINOConfig overrides GINO default 0.5.
#   - gno_use_open3d / gno_use_torch_scatter forced False so PyTorch and JAX
#     follow the same neighbor/reduction code paths (we precompute neighbors
#     externally anyway, so open3d availability doesn't matter).
# ---------------------------------------------------------------------------
COMMON_KWARGS = dict(
    in_channels=0,                         # GINO_Small3d.data_channels=0 -> x is None
    out_channels=1,
    latent_feature_channels=1,
    projection_channel_ratio=4,
    gno_coord_dim=3,
    gno_coord_embed_dim=16,
    in_gno_radius=0.033,                   # GINO_Small3d.gno_radius=0.033
    out_gno_radius=0.033,
    in_gno_transform_type="linear",
    out_gno_transform_type="linear",
    in_gno_pos_embed_type="transformer",   # GINO default; GINOConfig only sets gno_pos_embed_type
    out_gno_pos_embed_type="transformer",
    fno_in_channels=3,
    fno_n_modes=(16, 16, 16),
    fno_hidden_channels=64,
    fno_lifting_channel_ratio=2,
    fno_n_layers=4,
    gno_embed_channels=32,
    gno_embed_max_positions=10000,
    in_gno_channel_mlp_hidden_layers=[80, 80, 80],
    out_gno_channel_mlp_hidden_layers=[512, 256],
    gno_use_open3d=False,                  # neighbors are precomputed; this is irrelevant
    gno_use_torch_scatter=False,           # match JAX path (it has no torch_scatter equivalent)
    fno_use_channel_mlp=True,
    fno_channel_mlp_expansion=1.0,         # GINOConfig overrides the GINO default 0.5
    fno_norm="instance_norm",
    fno_ada_in_features=32,                # unused when fno_norm != "ada_in"; kept to mirror config
    fno_skip="linear",
    fno_channel_mlp_skip="soft-gating",
    fno_factorization="tucker",
    fno_rank=0.4,
    fno_implementation="factorized",       # required for Tucker contraction in PyTorch
    fno_domain_padding=0.125,
)

# Synthetic problem sized to match the training config.
#   - GRID_RES=32 matches CarCFDDatasetConfig.sdf_query_resolution=32.
#   - With in/out_gno_radius=0.033 in a [0,1]^3 cube, each grid query point
#     covers a ball of volume ~1.5e-4. We need a lot of input/output points
#     for the random neighbor search to return non-empty neighborhoods;
#     bump N_INPUT_PTS / N_OUTPUT_PTS if you see many SKIP rows.
#   - N_TEST_CASES is reduced because the full-size model is much slower.
GRID_RES = 32
N_INPUT_PTS = 2000
N_OUTPUT_PTS = 1000
SEED = 0              # used for model weight init (PyTorch + JAX)
N_TEST_CASES = 5      # full-size model -> each case takes notably longer


def make_inputs(input_seed: int):
    """Build (1, n_in, 3) input geom, (1, R, R, R, 3) latent queries,
    (1, n_out, 3) output queries, (1, n_in, in_channels) `x`, and
    (1, R, R, R, 1) latent_features — all as numpy float32 arrays.

    `input_seed` controls the input draw only; model weights are separate."""
    rng = np.random.RandomState(input_seed)
    input_geom    = rng.rand(1, N_INPUT_PTS, 3).astype(np.float32)
    output_queries = rng.rand(1, N_OUTPUT_PTS, 3).astype(np.float32)

    R = GRID_RES
    grid1d = np.linspace(0, 1, R, dtype=np.float32)
    gx, gy, gz = np.meshgrid(grid1d, grid1d, grid1d, indexing='ij')
    latent_queries = np.stack([gx, gy, gz], axis=-1)[None, ...]   # (1, R, R, R, 3)

    in_ch = COMMON_KWARGS['in_channels']
    # CarCFD config uses in_channels=0 (no per-vertex input function) -> x is None.
    # GINO.forward handles x=None by treating the input GNO as kernel-only.
    if in_ch == 0:
        x = None
    else:
        x = rng.randn(1, N_INPUT_PTS, in_ch).astype(np.float32)
    latent_features = rng.randn(1, R, R, R, COMMON_KWARGS['latent_feature_channels']).astype(np.float32)

    return input_geom, latent_queries, output_queries, x, latent_features

def precompute_neighbors(input_geom, latent_queries, output_queries,
                         in_radius, out_radius):
    """Run torch native neighbor search for input GNO (data=input_geom,
    queries=grid) and output GNO (data=grid, queries=output_queries).
    Returns a dict with torch-format (index, row_splits) AND the padded
    JAX-format (index, segment_ids, counts)."""
    R = latent_queries.shape[1]
    grid_flat = torch.from_numpy(latent_queries[0].reshape(-1, 3))
    in_pts    = torch.from_numpy(input_geom[0])
    out_pts   = torch.from_numpy(output_queries[0])

    nbr_in_t  = native_neighbor_search(in_pts, grid_flat, in_radius, return_norm=False)
    nbr_out_t = native_neighbor_search(grid_flat, out_pts, out_radius, return_norm=False)

    def to_jax_padded(t_dict, n_out):
        idx    = t_dict['neighbors_index'].cpu().numpy().astype(np.int64)
        splits = t_dict['neighbors_row_splits'].cpu().numpy().astype(np.int64)
        counts = (splits[1:] - splits[:-1]).astype(np.int64)
        seg_ids = np.repeat(np.arange(n_out, dtype=np.int64), counts)
        return dict(
            neighbors_index=jnp.asarray(idx),
            segment_ids=jnp.asarray(seg_ids),
            counts=jnp.asarray(counts),
        )

    nbr_in_jax  = to_jax_padded(nbr_in_t,  n_out=R**3)
    nbr_out_jax = to_jax_padded(nbr_out_t, n_out=N_OUTPUT_PTS)

    return dict(torch_in=nbr_in_t, torch_out=nbr_out_t,
                jax_in=nbr_in_jax, jax_out=nbr_out_jax)


# ---------------------------------------------------------------------------
# Weight transfer: PyTorch -> Flax param tree
# ---------------------------------------------------------------------------
def _t2n(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def assign_channel_mlp(jax_subtree, torch_module_prefix, torch_state):
    """ChannelMLP (Conv1d-based) -> Flax Conv(kernel_size=(1,)).
    Torch weight: (out, in, 1)  ->  Flax kernel: (1, in, out)."""
    i = 0
    while f"{torch_module_prefix}.fcs.{i}.weight" in torch_state:
        w = _t2n(torch_state[f"{torch_module_prefix}.fcs.{i}.weight"])  # (out, in, 1)
        b = _t2n(torch_state[f"{torch_module_prefix}.fcs.{i}.bias"])    # (out,)
        # (out, in, 1) -> (1, in, out)
        kernel = np.transpose(w, (2, 1, 0))
        jax_subtree[f"fc_{i}"] = {"kernel": jnp.asarray(kernel),
                                  "bias":   jnp.asarray(b)}
        i += 1

def assign_linear_mlp(jax_subtree, torch_module_prefix, torch_state):
    """LinearChannelMLP (nn.Linear-based) -> Flax Dense.
    Torch Linear weight: (out, in)  ->  Flax kernel: (in, out)."""
    i = 0
    while f"{torch_module_prefix}.fcs.{i}.weight" in torch_state:
        w = _t2n(torch_state[f"{torch_module_prefix}.fcs.{i}.weight"])  # (out, in)
        b = _t2n(torch_state[f"{torch_module_prefix}.fcs.{i}.bias"])    # (out,)
        kernel = np.transpose(w, (1, 0))
        jax_subtree[f"fc_{i}"] = {"kernel": jnp.asarray(kernel),
                                  "bias":   jnp.asarray(b)}
        i += 1


def assign_gno_block(jax_subtree, torch_prefix, torch_state):
    """GNOBlock contains integral_transform.channel_mlp (LinearChannelMLP).
    The Flax side stores it under integral_transform._channel_mlp."""
    jax_subtree.setdefault("integral_transform", {})
    jax_subtree["integral_transform"].setdefault("channel_mlp", {})
    assign_linear_mlp(
        jax_subtree["integral_transform"]["channel_mlp"],
        f"{torch_prefix}.integral_transform.channel_mlp",
        torch_state,
    )


def assign_spectral_conv_dense(jax_subtree, torch_prefix, torch_state):
    """SpectralConv with dense factorization. PyTorch stores the parameter
    under `<prefix>.weight.tensor` (FactorizedTensor.DenseTensor wraps
    nn.Parameter as `.tensor`). Bias is `<prefix>.bias`. Both keep the same
    shape on the Flax side."""
    w = _t2n(torch_state[f"{torch_prefix}.weight.tensor"])  # complex64
    b = _t2n(torch_state[f"{torch_prefix}.bias"])           # float32
    jax_subtree["weight"] = jnp.asarray(w)
    jax_subtree["bias"]   = jnp.asarray(b)

def assign_spectral_conv_tucker(jax_subtree, torch_prefix, torch_state):
    """SpectralConv with Tucker factorization.

    tltorch stores the core as `<prefix>.weight.core` (complex64, shape =
    tucker_ranks) and per-dimension factor matrices as
    `<prefix>.weight.factors.{i}` (complex64, shape (d_i, rank_i)).
    Bias lives at `<prefix>.bias`.

    Flax stores these as `<prefix>.w_core` and `<prefix>.w_U{i}` plus
    `<prefix>.bias` (see spectral_convolution_jax.py:setup, Tucker branch).
    """
    core = _t2n(torch_state[f"{torch_prefix}.weight.core"])
    jax_subtree["w_core"] = jnp.asarray(core)

    # i = 0
    # while f"{torch_prefix}.weight.factors.{i}" in torch_state:
    #     f = _t2n(torch_state[f"{torch_prefix}.weight.factors.{i}"])
    #     jax_subtree[f"w_U{i}"] = jnp.asarray(f)
    #     i += 1

    i = 0
    while f"{torch_prefix}.weight.factors.factor_{i}" in torch_state:
        factor = _t2n(torch_state[f"{torch_prefix}.weight.factors.factor_{i}"])
        jax_subtree[f"w_U{i}"] = jnp.asarray(factor)
        i += 1
        
    b = _t2n(torch_state[f"{torch_prefix}.bias"])
    jax_subtree["bias"] = jnp.asarray(b)


def assign_fno_skip_linear(jax_subtree, torch_prefix, torch_state):
    """Flattened1dConv around a Conv1d.
    Torch conv.weight: (out, in, 1) -> Flax kernel: (1, in, out)."""
    w = _t2n(torch_state[f"{torch_prefix}.conv.weight"])
    kernel = np.transpose(w, (2, 1, 0))
    jax_subtree["conv"] = {"kernel": jnp.asarray(kernel)}
    bias_key = f"{torch_prefix}.conv.bias"
    if bias_key in torch_state:
        jax_subtree["conv"]["bias"] = jnp.asarray(_t2n(torch_state[bias_key]))


def assign_soft_gating(jax_subtree, torch_prefix, torch_state):
    """SoftGating has a single `weight` of shape (1, in, 1, 1, 1...).
    Same shape on both sides, just rename."""
    w = _t2n(torch_state[f"{torch_prefix}.weight"])
    jax_subtree["weight"] = jnp.asarray(w)
    bias_key = f"{torch_prefix}.bias"
    if bias_key in torch_state:
        jax_subtree["bias_param"] = jnp.asarray(_t2n(torch_state[bias_key]))


def copy_torch_to_flax(torch_model, flax_params):
    """Walk the torch model and overwrite every matching leaf in flax_params.

    flax_params here is the *inner* params dict (i.e. flax_params['params'])
    that we got from `module.init(...)['params']`. Returns a new pytree (we
    mutate in place but also return for clarity)."""
    flax_params = jax.tree_util.tree_map(lambda x: x, flax_params)  # shallow copy
    state = dict(torch_model.state_dict())
    n_layers = torch_model.fno_blocks.n_layers

    # --- gno_in ---
    flax_params["gno_in"] = {}
    assign_gno_block(flax_params["gno_in"], "gno_in", state)

    # --- lifting ---
    flax_params["lifting"] = {}
    assign_channel_mlp(flax_params["lifting"], "lifting", state)

    # --- fno_blocks ---
    fb = flax_params.setdefault("fno_blocks", {})
    for i in range(n_layers):
        # spectral conv
        # fb[f"convs_{i}"] = {}
        # assign_spectral_conv_dense(fb[f"convs_{i}"], f"fno_blocks.convs.{i}", state)
        
        fb[f"convs_{i}"] = {}
        conv_prefix = f"fno_blocks.convs.{i}"
        if f"{conv_prefix}.weight.tensor" in state:
            assign_spectral_conv_dense(fb[f"convs_{i}"], conv_prefix, state)
        elif f"{conv_prefix}.weight.core" in state:
            assign_spectral_conv_tucker(fb[f"convs_{i}"], conv_prefix, state)
        else:
            raise KeyError(
                f"Cannot locate spectral conv weights under '{conv_prefix}'. "
                f"Expected '{conv_prefix}.weight.tensor' (Dense) or "
                f"'{conv_prefix}.weight.core' (Tucker)."
            )
        
        # fno_skip (linear -> Flattened1dConv)
        fb[f"fno_skips_{i}"] = {}
        assign_fno_skip_linear(fb[f"fno_skips_{i}"], f"fno_blocks.fno_skips.{i}", state)
        # channel_mlp post-FNO (Conv1d-based ChannelMLP)
        fb[f"channel_mlp_{i}"] = {}
        assign_channel_mlp(fb[f"channel_mlp_{i}"], f"fno_blocks.channel_mlp.{i}", state)
        # channel_mlp_skip (soft-gating)
        fb[f"channel_mlp_skips_{i}"] = {}
        assign_soft_gating(fb[f"channel_mlp_skips_{i}"], f"fno_blocks.channel_mlp_skips.{i}", state)

    # --- gno_out ---
    flax_params["gno_out"] = {}
    assign_gno_block(flax_params["gno_out"], "gno_out", state)

    # --- projection ---
    flax_params["projection"] = {}
    assign_channel_mlp(flax_params["projection"], "projection", state)

    return flax_params


# ---------------------------------------------------------------------------
# Forward pass + comparison
# ---------------------------------------------------------------------------
def _run_one_case(case_idx, input_seed, torch_model, jax_model, jax_params, tol):
    """Run one parity case (fresh inputs, fixed weights). Returns metrics dict."""
    input_geom_np, latent_q_np, output_q_np, x_np, lat_feat_np = make_inputs(input_seed)

    nbrs = precompute_neighbors(
        input_geom_np, latent_q_np, output_q_np,
        in_radius=COMMON_KWARGS['in_gno_radius'],
        out_radius=COMMON_KWARGS['out_gno_radius'],
    )

    # If the neighbor search returns zero edges in either GNO, the model
    # output isn't meaningful and segment_csr can fail — skip cleanly.
    n_in_edges  = nbrs['torch_in']['neighbors_index'].shape[0]
    n_out_edges = nbrs['torch_out']['neighbors_index'].shape[0]
    if n_in_edges == 0 or n_out_edges == 0:
        return dict(case=case_idx, seed=input_seed, skipped=True,
                    n_in=n_in_edges, n_out=n_out_edges)

    x_torch = None if x_np is None else torch.from_numpy(x_np)
    x_jax   = None if x_np is None else jnp.asarray(x_np)

    # PyTorch forward
    with torch.no_grad():
        out_torch = torch_model(
            input_geom=torch.from_numpy(input_geom_np),
            latent_queries=torch.from_numpy(latent_q_np),
            output_queries=torch.from_numpy(output_q_np),
            x=x_torch,
            latent_features=torch.from_numpy(lat_feat_np),
            ada_in=None,
            neighbors_in={k: v for k, v in nbrs['torch_in'].items()},
            neighbors_out={k: v for k, v in nbrs['torch_out'].items()},
        )
    out_torch_np = out_torch.detach().cpu().numpy()

    # JAX forward
    out_jax = jax_model.apply(
        {'params': jax_params},
        input_geom=jnp.asarray(input_geom_np),
        latent_queries=jnp.asarray(latent_q_np),
        output_queries=jnp.asarray(output_q_np),
        x=x_jax,
        latent_features=jnp.asarray(lat_feat_np),
        ada_in=None,
        neighbors_in=nbrs['jax_in'],
        neighbors_out=nbrs['jax_out'],
    )
    out_jax_np = np.asarray(out_jax)

    diff = np.abs(out_torch_np - out_jax_np)
    rel  = diff / (np.abs(out_torch_np) + 1e-8)
    return dict(
        case=case_idx, seed=input_seed, skipped=False,
        n_in=n_in_edges, n_out=n_out_edges,
        max_abs=float(diff.max()), mean_abs=float(diff.mean()),
        max_rel=float(rel.max()),
        torch_min=float(out_torch_np.min()), torch_max=float(out_torch_np.max()),
        jax_min=float(out_jax_np.min()),     jax_max=float(out_jax_np.max()),
        passed=bool(diff.max() < tol),
    )


def run_parity_check():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tol = 1e-4

    print("=" * 78)
    print("GINO PyTorch <-> JAX parity check — multi-case")
    print("=" * 78)
    print(f"grid={GRID_RES}^3   n_input_pts={N_INPUT_PTS}   n_output_pts={N_OUTPUT_PTS}")
    print(f"fno_n_modes={COMMON_KWARGS['fno_n_modes']}   fno_hidden={COMMON_KWARGS['fno_hidden_channels']}   "
          f"factorization={COMMON_KWARGS['fno_factorization']}")
    print(f"fno_norm={COMMON_KWARGS['fno_norm']}   fno_skip={COMMON_KWARGS['fno_skip']}   "
          f"channel_mlp_skip={COMMON_KWARGS['fno_channel_mlp_skip']}")
    print(f"n_test_cases={N_TEST_CASES}   tol={tol:.0e}")
    print()

    # -----------------------------------------------------------------------
    # Build both models ONCE. Transfer torch -> jax weights ONCE. The N
    # cases share these weights; only the inputs vary across cases.
    # -----------------------------------------------------------------------
    torch_model = TorchGINO(**COMMON_KWARGS)
    torch_model.eval()
    jax_model = JaxGINO(**COMMON_KWARGS)

    # Init JAX with the first case's inputs to materialize the param pytree.
    init_geom, init_lat_q, init_out_q, init_x, init_lat_f = make_inputs(input_seed=0)
    init_nbrs = precompute_neighbors(
        init_geom, init_lat_q, init_out_q,
        in_radius=COMMON_KWARGS['in_gno_radius'],
        out_radius=COMMON_KWARGS['out_gno_radius'],
    )
    init_variables = jax_model.init(
        jax.random.PRNGKey(SEED),
        input_geom=jnp.asarray(init_geom),
        latent_queries=jnp.asarray(init_lat_q),
        output_queries=jnp.asarray(init_out_q),
        x=None if init_x is None else jnp.asarray(init_x),
        latent_features=jnp.asarray(init_lat_f),
        ada_in=None,
        neighbors_in=init_nbrs['jax_in'],
        neighbors_out=init_nbrs['jax_out'],
    )
    jax_params = copy_torch_to_flax(torch_model, init_variables['params'])

    def _flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            full = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flat(v, full))
            else:
                out[full] = v.shape
        return out
    before = set(_flat(init_variables['params']).keys())
    after  = set(_flat(jax_params).keys())
    missing = before - after
    if missing:
        print(f"WARNING: {len(missing)} JAX param paths NOT overwritten by transfer "
              f"(still random JAX values). First few:")
        for p in list(missing)[:10]:
            print(f"   - {p}")
        print()

    # -----------------------------------------------------------------------
    # Run N_TEST_CASES with different input seeds
    # -----------------------------------------------------------------------
    # Input seeds 1000+i to avoid collision with the model weight seed.
    header = (f"{'case':>4} {'seed':>5} {'n_in':>5} {'n_out':>5} "
              f"{'max_abs':>11} {'mean_abs':>11} {'max_rel':>11} {'status':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for i in range(N_TEST_CASES):
        seed_i = 1000 + i
        r = _run_one_case(i, seed_i, torch_model, jax_model, jax_params, tol)
        results.append(r)
        if r['skipped']:
            print(f"{r['case']:>4} {r['seed']:>5} {r['n_in']:>5} {r['n_out']:>5} "
                  f"{'-':>11} {'-':>11} {'-':>11} {'SKIP':>7}")
        else:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"{r['case']:>4} {r['seed']:>5} {r['n_in']:>5} {r['n_out']:>5} "
                  f"{r['max_abs']:>11.3e} {r['mean_abs']:>11.3e} "
                  f"{r['max_rel']:>11.3e} {status:>7}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    ran = [r for r in results if not r['skipped']]
    skipped = [r for r in results if r['skipped']]
    n_pass = sum(1 for r in ran if r['passed'])
    n_fail = sum(1 for r in ran if not r['passed'])

    print()
    print("=" * 78)
    print(f"Summary: {len(ran)} cases ran, {len(skipped)} skipped (zero-edge neighbors)")
    print(f"         {n_pass} PASS, {n_fail} FAIL  (tol={tol:.0e})")
    if ran:
        max_abs_arr  = np.array([r['max_abs']  for r in ran])
        mean_abs_arr = np.array([r['mean_abs'] for r in ran])
        max_rel_arr  = np.array([r['max_rel']  for r in ran])
        print(f"         max_abs  across cases: min={max_abs_arr.min():.3e}  "
              f"mean={max_abs_arr.mean():.3e}  max={max_abs_arr.max():.3e}")
        print(f"         mean_abs across cases: min={mean_abs_arr.min():.3e}  "
              f"mean={mean_abs_arr.mean():.3e}  max={mean_abs_arr.max():.3e}")
        print(f"         max_rel  across cases: min={max_rel_arr.min():.3e}  "
              f"mean={max_rel_arr.mean():.3e}  max={max_rel_arr.max():.3e}")
    print("=" * 78)


if __name__ == "__main__":
    run_parity_check()

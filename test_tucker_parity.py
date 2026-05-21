"""
Parity test: SpectralConv Tucker factorization — PyTorch vs JAX.

Sections:
  A. Inspect Tucker ranks from each implementation.
  B. Forward-pass parity using shared dense weights (copied from PyTorch).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
IN_CH   = 8
OUT_CH  = 8
N_MODES = (4, 4, 4)
RANK    = 0.5
BATCH   = 20
SPATIAL = (8, 8, 8)
SEED    = 42

print(f'No. of batch: {BATCH}')
# ─────────────────────────────────────────────────────────────────────────────
# A. PYTORCH — Tucker ranks and reconstruction
# ─────────────────────────────────────────────────────────────────────────────
import tensorly as tl
tl.set_backend("pytorch")

import torch
from neuralop.layers.spectral_convolution import SpectralConv as TorchSpectralConv

torch_model = TorchSpectralConv(
    in_channels=IN_CH, out_channels=OUT_CH, n_modes=N_MODES,
    factorization="tucker", rank=RANK, bias=True, implementation="factorized",
)
torch_model.eval()

tw = torch_model.weight  # tltorch TuckerTensor
torch_core_shape = tuple(tw.core.shape)
torch_factor_shapes = [tuple(f.shape) for f in tw.factors]
torch_ranks = tuple(f.shape[1] for f in tw.factors)

# Reconstruct dense — set backend to pytorch first to avoid cross-contamination
tl.set_backend("pytorch")
torch_dense_np = tw.to_tensor().detach().cpu().numpy().astype(np.complex64)
torch_bias_np  = torch_model.bias.detach().cpu().numpy().astype(np.float32)

print("=== [PyTorch] Tucker weight ===")
print(f"  core shape     : {torch_core_shape}")
for i, fs in enumerate(torch_factor_shapes):
    print(f"  factor[{i}] shape: {fs}")
print(f"  per-dim ranks  : {torch_ranks}")
print(f"  reconstructed  : {torch_dense_np.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# B. JAX — Tucker ranks and reconstruction
# ─────────────────────────────────────────────────────────────────────────────
tl.set_backend("jax")

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax

from neuralop.layers.spectral_convolution_jax import (
    SpectralConv as JaxSpectralConv,
    JAXTuckerTensor,
)

jax_model = JaxSpectralConv(
    in_channels=IN_CH, out_channels=OUT_CH, n_modes=N_MODES,
    factorization="tucker", rank=RANK, bias=True, implementation="factorized",
)

rng = jax.random.PRNGKey(SEED)
dummy_x = jnp.ones((BATCH, IN_CH, *SPATIAL), dtype=jnp.float32)
params = jax_model.init(rng, dummy_x)
bound  = jax_model.bind(params)

jw: JAXTuckerTensor = bound.weight
jax_core_shape    = tuple(jw.core.shape)
jax_factor_shapes = [tuple(f.shape) for f in jw.factors]
jax_ranks         = tuple(f.shape[1] for f in jw.factors)

# Reconstruct via multi-mode product (pure JAX math)
from tensorly.tucker_tensor import tucker_to_tensor
jax_dense_np = np.array(tucker_to_tensor((jw.core, jw.factors))).astype(np.complex64)

print("\n=== [JAX] Tucker weight ===")
print(f"  core shape     : {jax_core_shape}")
for i, fs in enumerate(jax_factor_shapes):
    print(f"  factor[{i}] shape: {fs}")
print(f"  per-dim ranks  : {jax_ranks}")
print(f"  reconstructed  : {jax_dense_np.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# C. Rank summary
# ─────────────────────────────────────────────────────────────────────────────
# weight_shape used when building Tucker (last mode is n_modes[-1]//2+1 for real FFT)
weight_shape = (IN_CH, OUT_CH) + tuple(
    m // 2 + 1 if i == len(N_MODES) - 1 else m for i, m in enumerate(N_MODES)
)

import math
from tensorly.tucker_tensor import validate_tucker_rank

jax_ranks_formula  = tuple(max(1, math.ceil(RANK * d)) for d in weight_shape)
canonical_ranks    = tuple(int(r) for r in validate_tucker_rank(weight_shape, rank=RANK))

print("\n=== Tucker rank comparison ===")
print(f"  weight_shape used           : {weight_shape}")
print(f"  PyTorch ranks (tltorch)     : {torch_ranks}")
print(f"  JAX ranks (ceil(rank*d))    : {jax_ranks_formula}")
print(f"  canonical validate_tucker   : {canonical_ranks}")
print(f"  Shapes match (recon)        : {torch_dense_np.shape == jax_dense_np.shape}")
print(f"  Ranks: JAX==PyTorch?        : {jax_ranks == torch_ranks}")
print(f"  Ranks: JAX==canonical?      : {jax_ranks == canonical_ranks}")
print(f"  Ranks: PyTorch==canonical?  : {torch_ranks == canonical_ranks}")

# ─────────────────────────────────────────────────────────────────────────────
# D. Forward-pass parity — inject PyTorch dense weights into JAX dense model
#    This tests whether the forward-pass *logic* (FFT, slicing, contraction)
#    is identical, isolating it from any weight-init or rank differences.
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Forward-pass parity (shared dense weights) ===")

# JAX dense model with same hyperparams
jax_dense_model = JaxSpectralConv(
    in_channels=IN_CH, out_channels=OUT_CH, n_modes=N_MODES,
    factorization="dense", rank=RANK, bias=True, implementation="factorized",
)
params_dense = jax_dense_model.init(rng, dummy_x)
jax_w_shape  = params_dense['params']['weight'].shape
print(f"  JAX dense param shape : {jax_w_shape}")
print(f"  PyTorch recon shape   : {torch_dense_np.shape}")

if jax_w_shape == torch_dense_np.shape:
    # Inject PyTorch weights
    params_dense = flax.core.unfreeze(params_dense)
    params_dense['params']['weight'] = jnp.array(torch_dense_np)
    params_dense['params']['bias']   = jnp.array(torch_bias_np)
    params_dense = flax.core.freeze(params_dense)

    np.random.seed(SEED)
    x_np    = np.random.randn(BATCH, IN_CH, *SPATIAL).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_jax   = jnp.array(x_np)

    # PyTorch forward (dense, same weight)
    # DenseTensor (from tltorch) stores parameters under .tensor, not .data.
    tl.set_backend("pytorch")
    torch_dense_model = TorchSpectralConv(
        in_channels=IN_CH, out_channels=OUT_CH, n_modes=N_MODES,
        factorization="dense", rank=RANK, bias=True, implementation="reconstructed",
    )
    torch_dense_model.eval()
    with torch.no_grad():
        # For tltorch DenseTensor the underlying nn.Parameter is .tensor
        torch_dense_model.weight.tensor.data = torch.tensor(torch_dense_np)
        torch_dense_model.bias.data          = torch.tensor(torch_bias_np)
        y_torch = torch_dense_model(x_torch).cpu().numpy()

    # JAX forward — must reset backend after PyTorch forward above
    tl.set_backend("jax")
    y_jax    = jax_dense_model.apply(params_dense, x_jax)
    y_jax_np = np.array(y_jax)

    abs_diff = np.abs(y_torch - y_jax_np)
    print(f"\n  Output shape (torch)     : {y_torch.shape}")
    print(f"  Output shape (jax)       : {y_jax_np.shape}")
    print(f"  Max absolute difference  : {abs_diff.max():.6e}")
    print(f"  Mean absolute difference : {abs_diff.mean():.6e}")
    print(f"  Rel max diff             : {(abs_diff / (np.abs(y_torch) + 1e-8)).max():.6e}")

    tol = 1e-3
    verdict = "PASS" if abs_diff.max() < tol else "FAIL"
    print(f"\n  [{verdict}] Forward-pass logic {'agrees' if verdict=='PASS' else 'differs'} within tol={tol}")

else:
    print(f"\n  SKIP: weight shapes differ ({jax_w_shape} vs {torch_dense_np.shape})")
    print("        Rank mismatch confirmed — fix JAX rank computation to proceed.")

# ─────────────────────────────────────────────────────────────────────────────
# E. Tucker forward-pass parity — inject same core+factors into JAX Tucker model
#    This tests whether _contract_tucker + JAXTuckerTensor.__getitem__ is correct.
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Tucker-path forward-pass parity (shared Tucker weights) ===")

if torch_ranks == jax_ranks:
    print("  Ranks match — can directly copy core and factors.")
else:
    print(f"  Ranks differ (torch={torch_ranks}, jax={jax_ranks}).")
    print("  Building a common Tucker with JAX ranks and identical data for both paths.")

# Use JAX ranks as the ground truth Tucker structure.
# Build a PyTorch Tucker model with the same ranks as JAX, copy to JAX.
tl.set_backend("pytorch")

# We'll use a full-rank (rank=1.0) Tucker for parity — same structure guaranteed.
torch_fr_model = TorchSpectralConv(
    in_channels=IN_CH, out_channels=OUT_CH, n_modes=N_MODES,
    factorization="tucker", rank=1.0, bias=True, implementation="factorized",
)
torch_fr_model.eval()
tw_fr = torch_fr_model.weight
fr_core_np    = tw_fr.core.detach().cpu().numpy().astype(np.complex64)
fr_factors_np = [f.detach().cpu().numpy().astype(np.complex64) for f in tw_fr.factors]
fr_bias_np    = torch_fr_model.bias.detach().cpu().numpy().astype(np.float32)

print(f"  Full-rank Tucker core     : {fr_core_np.shape}")
print(f"  Full-rank Tucker factors  : {[f.shape for f in fr_factors_np]}")

# PyTorch forward with full-rank Tucker
tl.set_backend("pytorch")
x_torch = torch.from_numpy(x_np)
with torch.no_grad():
    y_torch_fr = torch_fr_model(x_torch).cpu().numpy()

# Build JAX Tucker model with rank=1.0 and inject same weights
tl.set_backend("jax")
jax_tucker_model = JaxSpectralConv(
    in_channels=IN_CH, out_channels=OUT_CH, n_modes=N_MODES,
    factorization="tucker", rank=1.0, bias=True, implementation="factorized",
)
params_tucker = jax_tucker_model.init(rng, dummy_x)
params_tucker = flax.core.unfreeze(params_tucker)

params_tucker['params']['w_core'] = jnp.array(fr_core_np)
for i, f_np in enumerate(fr_factors_np):
    params_tucker['params'][f'w_U{i}'] = jnp.array(f_np)
params_tucker['params']['bias'] = jnp.array(fr_bias_np)

params_tucker = flax.core.freeze(params_tucker)

x_jax = jnp.array(x_np)
y_jax_tucker    = jax_tucker_model.apply(params_tucker, x_jax)
y_jax_tucker_np = np.array(y_jax_tucker)

abs_diff_t = np.abs(y_torch_fr - y_jax_tucker_np)
print(f"\n  Output shape (torch Tucker) : {y_torch_fr.shape}")
print(f"  Output shape (jax Tucker)   : {y_jax_tucker_np.shape}")
print(f"  Max absolute difference     : {abs_diff_t.max():.6e}")
print(f"  Mean absolute difference    : {abs_diff_t.mean():.6e}")

for tol, label in [(1e-3, "tight"), (5e-3, "loose (float32 noise)")]:
    verdict_t = "PASS" if abs_diff_t.max() < tol else "FAIL"
    print(f"  [{verdict_t}] Tucker-path logic tol={tol} ({label})")

# ─────────────────────────────────────────────────────────────────────────────
# F. Bug 2 test — n_modes < max_n_modes forces real (non-trivial) factor slicing
#
#    When starts != 0, JAXTuckerTensor.__getitem__ slices factor rows to select
#    the active frequency modes.  We test two variants:
#      F1. Current code  — factors sliced in-place (factorized contraction)
#      F2. Reconstruction fix — __getitem__ reconstructs to dense, then slices
#          (avoids multi-factor einsum accumulation error)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Bug 2: n_modes < max_n_modes (non-trivial factor slicing) ===")

MAX_N_MODES = (8, 8, 8)   # larger capacity
N_MODES_SMALL = (4, 4, 4) # active modes — forces starts != 0

# ── PyTorch reference ────────────────────────────────────────────────────────
tl.set_backend("pytorch")
torch_max_model = TorchSpectralConv(
    in_channels=IN_CH, out_channels=OUT_CH,
    n_modes=N_MODES_SMALL, max_n_modes=MAX_N_MODES,
    factorization="tucker", rank=1.0, bias=True, implementation="factorized",
)
torch_max_model.eval()
tw_max = torch_max_model.weight
max_core_np    = tw_max.core.detach().cpu().numpy().astype(np.complex64)
max_factors_np = [f.detach().cpu().numpy().astype(np.complex64) for f in tw_max.factors]
max_bias_np    = torch_max_model.bias.detach().cpu().numpy().astype(np.float32)
print(f"  PyTorch core shape    : {max_core_np.shape}")
print(f"  PyTorch factor shapes : {[f.shape for f in max_factors_np]}")

np.random.seed(SEED + 1)
x_np2    = np.random.randn(BATCH, IN_CH, *SPATIAL).astype(np.float32)
x_torch2 = torch.from_numpy(x_np2)
with torch.no_grad():
    y_torch_max = torch_max_model(x_torch2).cpu().numpy()

# ── F1: JAX current code (factorized slicing) ────────────────────────────────
tl.set_backend("jax")
jax_max_model = JaxSpectralConv(
    in_channels=IN_CH, out_channels=OUT_CH,
    n_modes=N_MODES_SMALL, max_n_modes=MAX_N_MODES,
    factorization="tucker", rank=1.0, bias=True, implementation="factorized",
)
dummy_x2   = jnp.ones((BATCH, IN_CH, *SPATIAL), dtype=jnp.float32)
params_max = jax_max_model.init(rng, dummy_x2)
params_max = flax.core.unfreeze(params_max)
params_max['params']['w_core'] = jnp.array(max_core_np)
for i, f_np in enumerate(max_factors_np):
    params_max['params'][f'w_U{i}'] = jnp.array(f_np)
params_max['params']['bias'] = jnp.array(max_bias_np)
params_max = flax.core.freeze(params_max)

x_jax2 = jnp.array(x_np2)
y_f1_np = np.array(jax_max_model.apply(params_max, x_jax2))

diff_f1 = np.abs(y_torch_max - y_f1_np)
print(f"\n  [F1 factorized slicing]  max diff: {diff_f1.max():.6e}  mean: {diff_f1.mean():.6e}")

# ── F2: Reconstruction fix — patch __getitem__ to reconstruct→dense→slice ───
from tensorly.tucker_tensor import tucker_to_tensor as _ttt
import neuralop.layers.spectral_convolution_jax as _sc_jax

_orig_getitem = JAXTuckerTensor.__getitem__

def _reconstructed_getitem(self, slices):
    """Reconstruct Tucker to dense, then slice — avoids factor-slicing path."""
    dense = jnp.array(_ttt((self.core, self.factors)))
    sliced = dense[slices]
    return sliced   # plain jnp.ndarray; _contract_dense will handle it

JAXTuckerTensor.__getitem__ = _reconstructed_getitem

# Also need _contract_tucker to fall back to _contract_dense when weight is ndarray.
# We patch the forward call locally by wrapping the model apply.
from neuralop.layers.spectral_convolution_jax import _contract_dense as _cd

_orig_contract_tucker = _sc_jax._contract_tucker

def _patched_contract_tucker(x, weight, separable=False):
    if isinstance(weight, jnp.ndarray):
        return _cd(x, weight, separable=separable)
    return _orig_contract_tucker(x, weight, separable=separable)

_sc_jax._contract_tucker = _patched_contract_tucker

# Re-bind model so it picks up the patched functions (contract is stored as closure)
# The easiest way: re-init a fresh model instance after patching
jax_max_model_f2 = JaxSpectralConv(
    in_channels=IN_CH, out_channels=OUT_CH,
    n_modes=N_MODES_SMALL, max_n_modes=MAX_N_MODES,
    factorization="tucker", rank=1.0, bias=True, implementation="factorized",
)
params_max_f2 = jax_max_model_f2.init(rng, dummy_x2)
params_max_f2 = flax.core.unfreeze(params_max_f2)
params_max_f2['params']['w_core'] = jnp.array(max_core_np)
for i, f_np in enumerate(max_factors_np):
    params_max_f2['params'][f'w_U{i}'] = jnp.array(f_np)
params_max_f2['params']['bias'] = jnp.array(max_bias_np)
params_max_f2 = flax.core.freeze(params_max_f2)

y_f2_np = np.array(jax_max_model_f2.apply(params_max_f2, x_jax2))

diff_f2 = np.abs(y_torch_max - y_f2_np)
print(f"  [F2 reconstruct+slice]   max diff: {diff_f2.max():.6e}  mean: {diff_f2.mean():.6e}")

# Restore originals
JAXTuckerTensor.__getitem__ = _orig_getitem
_sc_jax._contract_tucker    = _orig_contract_tucker

print(f"\n  Improvement (F2 vs F1)   : {diff_f1.max()/diff_f2.max():.2f}x")
print(f"  Output shape             : {y_f1_np.shape}")
for tol in [1e-4, 1e-3, 5e-3]:
    v1 = "PASS" if diff_f1.max() < tol else "FAIL"
    v2 = "PASS" if diff_f2.max() < tol else "FAIL"
    print(f"  tol={tol:.0e}  F1={v1}  F2={v2}")

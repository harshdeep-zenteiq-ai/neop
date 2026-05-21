"""
Standalone gradient parity check for SpectralConv (JAX vs PyTorch).

Copies JAX weights into the PyTorch layer, runs identical inputs through
both, computes d(sum(output))/d(input), and compares norms + values.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_ENABLE_X64"] = "0"  # stay in f32

import numpy as np
import jax
import jax.numpy as jnp
import torch

# ---------------------------------------------------------------------------
# Parameters — match what GINO uses
# ---------------------------------------------------------------------------
IN_C   = 64
OUT_C  = 64
MODES  = (6, 6, 6)        # n_modes before the +1 adjustment
BATCH  = 1
SPATIAL = (16, 16, 16)    # d1, d2, d3
SEED   = 42
# ---------------------------------------------------------------------------

# ── JAX SpectralConv ──────────────────────────────────────────────────────
import sys
sys.path.insert(0, "/home/deepan/neuraloperator")
from neuralop.layers.spectral_convolution_jax import SpectralConv as JaxSpectralConv

jax_model = JaxSpectralConv(
    in_channels=IN_C,
    out_channels=OUT_C,
    n_modes=MODES,
    factorization=None,        # dense weight
    implementation="reconstructed",
    enforce_hermitian_symmetry=True,
    fft_norm="forward",
    bias=True,
)

# Init with a dummy forward pass so params are created
rng = jax.random.PRNGKey(SEED)
dummy_x = jnp.zeros((BATCH, IN_C, *SPATIAL), dtype=jnp.float32)
jax_params = jax_model.init(rng, dummy_x)

# ── PyTorch SpectralConv ───────────────────────────────────────────────────
from neuralop.layers.spectral_convolution import SpectralConv as TorchSpectralConv

torch_model = TorchSpectralConv(
    in_channels=IN_C,
    out_channels=OUT_C,
    n_modes=MODES,
    factorization=None,
    implementation="reconstructed",
    enforce_hermitian_symmetry=True,
    fft_norm="forward",
    bias=True,
)

# ── Copy JAX weights → PyTorch ────────────────────────────────────────────
jax_w = np.array(jax_params["params"]["weight"])          # complex64
torch_w = torch.from_numpy(jax_w)
# DenseTensor wraps a single Parameter; copy into it via its underlying tensor
with torch.no_grad():
    torch_model.weight.tensor.data.copy_(torch_w)

jax_b  = np.array(jax_params["params"]["bias"])            # float32
torch_b = torch.from_numpy(jax_b)
with torch.no_grad():
    torch_model.bias.data.copy_(torch_b)

print(f"Weight shape  JAX={jax_w.shape}  Torch={tuple(torch_model.weight.shape)}")
print(f"Bias shape    JAX={jax_b.shape}   Torch={tuple(torch_model.bias.shape)}")
print(f"Max weight diff after copy: {(torch_w - torch_model.weight.tensor.data).abs().max().item():.2e}")

# ── Matching input ────────────────────────────────────────────────────────
np.random.seed(SEED)
x_np = np.random.randn(BATCH, IN_C, *SPATIAL).astype(np.float32)
x_jax   = jnp.array(x_np)
x_torch = torch.from_numpy(x_np).requires_grad_(True)

# ── Forward + backward (PyTorch) ──────────────────────────────────────────
y_torch = torch_model(x_torch)
loss_torch = y_torch.sum()
loss_torch.backward()
g_torch = x_torch.grad.detach().numpy()

# ── Forward + backward (JAX) ──────────────────────────────────────────────
def fwd(x):
    y = jax_model.apply(jax_params, x)
    return y.sum()

g_jax_fn = jax.grad(fwd)
g_jax = np.array(g_jax_fn(x_jax))

# ── Compare ───────────────────────────────────────────────────────────────
gnorm_t = np.linalg.norm(g_torch)
gnorm_j = np.linalg.norm(g_jax)
diff    = np.linalg.norm(g_jax - g_torch)
ratio   = gnorm_j / gnorm_t

print(f"\n=== Gradient parity: SpectralConv input gradient ===")
print(f"  ||g_jax||   = {gnorm_j:.6e}")
print(f"  ||g_torch|| = {gnorm_t:.6e}")
print(f"  ratio j/t   = {ratio:.4f}   (1.000 = perfect)")
print(f"  ||g_j - g_t|| = {diff:.6e}")
print(f"  max(|g_j - g_t|) = {np.abs(g_jax - g_torch).max():.6e}")

# Also compare forward outputs
y_torch_np = y_torch.detach().numpy()
y_jax_np   = np.array(jax_model.apply(jax_params, x_jax))
fwd_diff   = np.abs(y_jax_np - y_torch_np).max()
print(f"\n=== Forward parity ===")
print(f"  max |y_j - y_t| = {fwd_diff:.6e}")
print(f"  sum y_j = {float(y_jax_np.sum()):.6e}  sum y_t = {float(y_torch_np.sum()):.6e}")

# ── Isolate: test rfftn VJP alone ─────────────────────────────────────────
print("\n=== Isolate: rfftn + irfft round-trip gradient ===")
fft_dims = list(range(-3, 0))

# PyTorch
x_t2 = torch.from_numpy(x_np).requires_grad_(True)
y_t2 = torch.fft.rfftn(x_t2, norm="forward", dim=fft_dims)
y_t2 = torch.fft.irfftn(y_t2, s=SPATIAL, dim=fft_dims, norm="forward")
y_t2.sum().backward()
gt2 = x_t2.grad.detach().numpy()

# JAX
def fwd2(x):
    y = jnp.fft.rfftn(x, norm="forward", axes=fft_dims)
    y = jnp.fft.irfftn(y, s=SPATIAL, axes=fft_dims, norm="forward")
    return y.sum()

gj2 = np.array(jax.grad(fwd2)(x_jax))

ratio2 = np.linalg.norm(gj2) / np.linalg.norm(gt2)
print(f"  rfftn→irfftn ratio j/t = {ratio2:.4f}  (should be 1.000)")

# ── Isolate: enforce_hermitian_symmetry path ───────────────────────────────
print("\n=== Isolate: enforce_hermitian_symmetry path gradient ===")
mode_sizes = list(SPATIAL)

# PyTorch
x_t3 = torch.from_numpy(x_np).requires_grad_(True)
X_t3 = torch.fft.rfftn(x_t3, norm="forward", dim=fft_dims)
if len(fft_dims) > 1:
    X_t3 = torch.fft.fftshift(X_t3, dim=fft_dims[:-1])
# Skip weight multiply, just pass through
out_t3 = X_t3.clone()
if len(fft_dims) > 1:
    out_t3 = torch.fft.ifftshift(out_t3, dim=fft_dims[:-1])
# enforce hermitian
out_t3 = torch.fft.ifftn(out_t3, s=mode_sizes[:-1], dim=fft_dims[:-1], norm="forward")
out_t3[..., 0] = out_t3[..., 0].real + 0j
if mode_sizes[-1] % 2 == 0:
    out_t3[..., -1] = out_t3[..., -1].real + 0j
out_t3 = torch.fft.irfft(out_t3, n=mode_sizes[-1], dim=fft_dims[-1], norm="forward")
out_t3.sum().backward()
gt3 = x_t3.grad.detach().numpy()

# JAX
def fwd3(x):
    X = jnp.fft.rfftn(x, norm="forward", axes=fft_dims)
    if len(fft_dims) > 1:
        X = jnp.fft.fftshift(X, axes=fft_dims[:-1])
    out = X
    if len(fft_dims) > 1:
        out = jnp.fft.ifftshift(out, axes=fft_dims[:-1])
    out = jnp.fft.ifftn(out, s=mode_sizes[:-1], axes=fft_dims[:-1], norm="forward")
    out = out.at[..., 0].set((out[..., 0].real + 0j).astype(out.dtype))
    if mode_sizes[-1] % 2 == 0:
        out = out.at[..., -1].set((out[..., -1].real + 0j).astype(out.dtype))
    out = jnp.fft.irfft(out, n=mode_sizes[-1], axis=fft_dims[-1], norm="forward")
    return out.sum()

gj3 = np.array(jax.grad(fwd3)(x_jax))

ratio3 = np.linalg.norm(gj3) / np.linalg.norm(gt3)
print(f"  full hermitian-sym path ratio j/t = {ratio3:.4f}  (should be 1.000)")
print(f"  ||g_j3 - g_t3|| = {np.linalg.norm(gj3 - gt3):.6e}")

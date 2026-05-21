"""
Standalone gradient parity check for FNOBlocks components (JAX vs PyTorch).

Tests each sub-component of block[0] individually to isolate the 2x gradient bug.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_ENABLE_X64"] = "0"

import numpy as np
import jax
import jax.numpy as jnp
import torch
import sys
sys.path.insert(0, "/home/deepan/neuraloperator")

# ---------------------------------------------------------------------------
# Parameters — match GINO defaults
# ---------------------------------------------------------------------------
IN_C   = 64
OUT_C  = 64
MODES  = (6, 6, 6)
SPATIAL = (16, 16, 16)
BATCH  = 1
SEED   = 42
# ---------------------------------------------------------------------------

np.random.seed(SEED)
x_np = np.random.randn(BATCH, IN_C, *SPATIAL).astype(np.float32)

# ── Build JAX FNOBlocks (n_layers=1 so only block[0]) ─────────────────────
from neuralop.layers.fno_block_jax import FNOBlocks as JaxFNOBlocks

jax_fno = JaxFNOBlocks(
    in_channels=IN_C,
    out_channels=OUT_C,
    n_modes=MODES,
    n_layers=1,
    use_channel_mlp=True,
    channel_mlp_expansion=0.5,
    fno_skip="linear",
    channel_mlp_skip="soft-gating",
    norm=None,
    non_linearity=jax.nn.gelu,
)
rng = jax.random.PRNGKey(SEED)
x_jax = jnp.array(x_np)
jax_params = jax_fno.init(rng, x_jax, index=0)

# ── Build PyTorch FNOBlocks ────────────────────────────────────────────────
from neuralop.layers.fno_block import FNOBlocks as TorchFNOBlocks

torch_fno = TorchFNOBlocks(
    in_channels=IN_C,
    out_channels=OUT_C,
    n_modes=MODES,
    n_layers=1,
    use_channel_mlp=True,
    channel_mlp_expansion=0.5,
    fno_skip="linear",
    channel_mlp_skip="soft-gating",
    norm=None,
    non_linearity=torch.nn.functional.gelu,
)

# ── Copy weights JAX → PyTorch ─────────────────────────────────────────────
def copy_param(torch_param, jax_arr):
    with torch.no_grad():
        torch_param.data.copy_(torch.from_numpy(np.array(jax_arr)))

p = jax_params["params"]

# SpectralConv weight & bias
copy_param(torch_fno.convs[0].weight.tensor, p["convs_0"]["weight"])
copy_param(torch_fno.convs[0].bias,          p["convs_0"]["bias"])

# fno_skip (Flattened1dConv → nn.Conv)
# JAX conv uses (spatial, features) layout; kernel shape is (kernel, in, out)
jax_conv_kernel = np.array(p["fno_skips_0"]["conv"]["kernel"])  # (1, IN_C, OUT_C)
# PyTorch Conv1d weight shape: (out_channels, in_channels, kernel_size) = (OUT_C, IN_C, 1)
torch_conv_w = torch.from_numpy(jax_conv_kernel.transpose(2, 1, 0))  # (OUT_C, IN_C, 1)
with torch.no_grad():
    torch_fno.fno_skips[0].conv.weight.data.copy_(torch_conv_w)

# channel_mlp_skip (SoftGating)
copy_param(torch_fno.channel_mlp_skips[0].weight,
           p["channel_mlp_skips_0"]["weight"])

# ChannelMLP weights: JAX nn.Conv kernel shape is (1, in_c, out_c)
# PyTorch Conv1d weight shape is (out_c, in_c, 1) → transpose axes (2,1,0)
print("ChannelMLP param keys:", list(p["channel_mlp_0"].keys()))
for i, fc_torch in enumerate(torch_fno.channel_mlp[0].fcs):
    jax_k = np.array(p["channel_mlp_0"][f"fc_{i}"]["kernel"])  # (1, in_c, out_c)
    torch_k = torch.from_numpy(jax_k.transpose(2, 1, 0))       # (out_c, in_c, 1)
    jax_b = np.array(p["channel_mlp_0"][f"fc_{i}"]["bias"])     # (out_c,)
    with torch.no_grad():
        fc_torch.weight.data.copy_(torch_k)
        fc_torch.bias.data.copy_(torch.from_numpy(jax_b))
print(f"ChannelMLP weights copied: {len(torch_fno.channel_mlp[0].fcs)} layers")

# ── Gradient comparison: full FNOBlock ─────────────────────────────────────
x_torch = torch.from_numpy(x_np).requires_grad_(True)

y_torch = torch_fno(x_torch, index=0)
y_torch.sum().backward()
g_torch = x_torch.grad.detach().numpy()

def fwd_jax(x):
    return jax_fno.apply(jax_params, x, index=0).sum()

g_jax = np.array(jax.grad(fwd_jax)(x_jax))

gnorm_t = np.linalg.norm(g_torch)
gnorm_j = np.linalg.norm(g_jax)
ratio_full = gnorm_j / gnorm_t
print(f"\n=== Full FNOBlock gradient ===")
print(f"  ||g_jax||   = {gnorm_j:.6e}")
print(f"  ||g_torch|| = {gnorm_t:.6e}")
print(f"  ratio j/t   = {ratio_full:.4f}   (1.000 = perfect)")
print(f"  ||g_j - g_t|| = {np.linalg.norm(g_jax - g_torch):.6e}")

# ── Access submodules via bind so setup() fields are visible ──────────────
bound = jax_fno.bind(jax_params)

# ── Isolate: fno_skip (Flattened1dConv) only ──────────────────────────────
x_torch_s = torch.from_numpy(x_np).requires_grad_(True)
y_ts = torch_fno.fno_skips[0](x_torch_s)
y_ts.sum().backward()
gs_torch = x_torch_s.grad.detach().numpy()

def fwd_skip(x):
    return bound.fno_skips[0](x).sum()

gs_jax = np.array(jax.grad(fwd_skip)(x_jax))

ratio_skip = np.linalg.norm(gs_jax) / np.linalg.norm(gs_torch)
print(f"\n=== fno_skip (Flattened1dConv) gradient ===")
print(f"  ratio j/t = {ratio_skip:.4f}")
print(f"  ||g_j - g_t|| = {np.linalg.norm(gs_jax - gs_torch):.6e}")

# ── Isolate: channel_mlp_skip (SoftGating) only ───────────────────────────
x_torch_sg = torch.from_numpy(x_np).requires_grad_(True)
y_tsg = torch_fno.channel_mlp_skips[0](x_torch_sg)
y_tsg.sum().backward()
gsg_torch = x_torch_sg.grad.detach().numpy()

def fwd_sg(x):
    return bound.channel_mlp_skips[0](x).sum()

gsg_jax = np.array(jax.grad(fwd_sg)(x_jax))

ratio_sg = np.linalg.norm(gsg_jax) / np.linalg.norm(gsg_torch)
print(f"\n=== channel_mlp_skip (SoftGating) gradient ===")
print(f"  ratio j/t = {ratio_sg:.4f}")
print(f"  ||g_j - g_t|| = {np.linalg.norm(gsg_jax - gsg_torch):.6e}")

# ── Isolate: ChannelMLP only ───────────────────────────────────────────────
x_torch_mlp = torch.from_numpy(x_np).requires_grad_(True)
y_tmlp = torch_fno.channel_mlp[0](x_torch_mlp)
y_tmlp.sum().backward()
gmlp_torch = x_torch_mlp.grad.detach().numpy()

def fwd_mlp(x):
    return bound.channel_mlp[0](x).sum()

gmlp_jax = np.array(jax.grad(fwd_mlp)(x_jax))

ratio_mlp = np.linalg.norm(gmlp_jax) / np.linalg.norm(gmlp_torch)
print(f"\n=== ChannelMLP gradient ===")
print(f"  ratio j/t = {ratio_mlp:.4f}")
print(f"  ||g_j - g_t|| = {np.linalg.norm(gmlp_jax - gmlp_torch):.6e}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n=== Summary ===")
print(f"  SpectralConv:         1.0000 (confirmed in test_spectral_conv_grad.py)")
print(f"  fno_skip:             {ratio_skip:.4f}")
print(f"  channel_mlp_skip:     {ratio_sg:.4f}")
print(f"  ChannelMLP:           {ratio_mlp:.4f}")
print(f"  Full FNOBlock:        {ratio_full:.4f}")

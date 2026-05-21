"""
Targeted gradient parity test for FNOBlocks with:
  - n_layers=4 (matches actual GINO)
  - Tucker factorization (matches actual GINO)
  - norm="instance_norm" (matches actual GINO)
  - channel_mlp_expansion=1.0 (matches actual GINO)

Also compares intermediate FORWARD ACTIVATIONS to find where divergence starts.
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
IN_C   = 64
OUT_C  = 64
MODES  = (6, 6, 6)
SPATIAL = (32, 32, 32)
BATCH  = 1
N_LAYERS = 4
SEED   = 42
RANK   = 0.4
# ---------------------------------------------------------------------------

np.random.seed(SEED)
x_np = np.random.randn(BATCH, IN_C, *SPATIAL).astype(np.float32)
x_jax = jnp.array(x_np)

# ── Build JAX FNOBlocks ────────────────────────────────────────────────────
from neuralop.layers.fno_block_jax import FNOBlocks as JaxFNOBlocks

jax_fno = JaxFNOBlocks(
    in_channels=IN_C,
    out_channels=OUT_C,
    n_modes=MODES,
    n_layers=N_LAYERS,
    use_channel_mlp=True,
    channel_mlp_expansion=1.0,
    fno_skip="linear",
    channel_mlp_skip="soft-gating",
    norm="instance_norm",
    non_linearity=jax.nn.gelu,
    factorization="tucker",
    rank=RANK,
    implementation="factorized",
)

# Initialize by running ALL indices so all layer params get created.
# Each index creates its own set of sub-module params; merge them.
rng = jax.random.PRNGKey(SEED)
jax_params = jax_fno.init(rng, x_jax, index=0)
for _idx in range(1, N_LAYERS):
    _p_extra = jax_fno.init(jax.random.PRNGKey(SEED + _idx), x_jax, index=_idx)
    jax_params["params"].update(_p_extra["params"])

print("JAX param keys after full init:", sorted(jax_params["params"].keys()))
assert f"convs_{N_LAYERS-1}" in jax_params["params"], "Missing last layer params!"

# ── Build PyTorch FNOBlocks ────────────────────────────────────────────────
from neuralop.layers.fno_block import FNOBlocks as TorchFNOBlocks

torch_fno = TorchFNOBlocks(
    in_channels=IN_C,
    out_channels=OUT_C,
    n_modes=MODES,
    n_layers=N_LAYERS,
    use_channel_mlp=True,
    channel_mlp_expansion=1.0,
    fno_skip="linear",
    channel_mlp_skip="soft-gating",
    norm="instance_norm",
    non_linearity=torch.nn.functional.gelu,
    factorization="tucker",
    rank=RANK,
    implementation="factorized",
)
torch_fno.eval()

# ── Copy weights JAX → PyTorch ─────────────────────────────────────────────
def copy_param(torch_param, arr):
    with torch.no_grad():
        torch_param.data.copy_(torch.from_numpy(np.array(arr)))

p = jax_params["params"]

for i in range(N_LAYERS):
    pi = p[f"convs_{i}"]
    copy_param(torch_fno.convs[i].weight.core, pi["w_core"])
    j = 0
    while f"w_U{j}" in pi:
        copy_param(torch_fno.convs[i].weight.factors[j], pi[f"w_U{j}"])
        j += 1
    copy_param(torch_fno.convs[i].bias, pi["bias"])

    jax_k = np.array(p[f"fno_skips_{i}"]["conv"]["kernel"])   # (1, IN_C, OUT_C)
    torch_k = torch.from_numpy(jax_k.transpose(2, 1, 0))
    with torch.no_grad():
        torch_fno.fno_skips[i].conv.weight.data.copy_(torch_k)

    copy_param(torch_fno.channel_mlp_skips[i].weight, p[f"channel_mlp_skips_{i}"]["weight"])

    for fc_i, fc_torch in enumerate(torch_fno.channel_mlp[i].fcs):
        jax_k2 = np.array(p[f"channel_mlp_{i}"][f"fc_{fc_i}"]["kernel"])  # (1, in, out)
        torch_k2 = torch.from_numpy(jax_k2.transpose(2, 1, 0))
        jax_b2 = np.array(p[f"channel_mlp_{i}"][f"fc_{fc_i}"]["bias"])
        with torch.no_grad():
            fc_torch.weight.data.copy_(torch_k2)
            fc_torch.bias.data.copy_(torch.from_numpy(jax_b2))

print("Weights copied.\n")

# ── Compare FORWARD ACTIVATIONS at each block ──────────────────────────────
print("=== Forward activation parity (block-by-block) ===")
jax_x = x_jax
torch_x = torch.from_numpy(x_np)

for idx in range(N_LAYERS):
    jax_out = jax_fno.apply(jax_params, jax_x, index=idx)
    torch_out = torch_fno(torch_x, index=idx)
    j_np = np.array(jax_out)
    t_np = torch_out.detach().numpy()
    max_diff = np.abs(j_np - t_np).max()
    norm_j = np.linalg.norm(j_np)
    norm_t = np.linalg.norm(t_np)
    print(f"  block[{idx}]: ||jax||={norm_j:.4e}  ||torch||={norm_t:.4e}  "
          f"ratio={norm_j/(norm_t+1e-12):.4f}  max_diff={max_diff:.4e}")
    jax_x = jax_out
    torch_x = torch_out.detach()

# ── Gradient parity: full 4-block forward ─────────────────────────────────
print("\n=== Parameter gradient parity (4-block loop) ===")

def fwd_jax_full(params):
    out = x_jax
    for idx in range(N_LAYERS):
        out = jax_fno.apply(params, out, index=idx)
    return out.sum()

g_jax = jax.grad(fwd_jax_full)(jax_params)["params"]

for _p in torch_fno.parameters():
    _p.grad = None
torch_x = torch.from_numpy(x_np)
out_torch = torch_x
for idx in range(N_LAYERS):
    out_torch = torch_fno(out_torch, index=idx)
out_torch.sum().backward()

print(f"\n{'Param':<50} {'t_norm':>10} {'j_norm':>10} {'ratio':>8}")
print("-" * 80)

def compare(name, t_g, j_g):
    t = np.array(t_g.detach().cpu())
    j = np.array(j_g)
    if t.shape != j.shape:
        print(f"  {name:<48}  SHAPE MISMATCH: torch={t.shape} jax={j.shape}")
        return
    tn = np.linalg.norm(t)
    jn = np.linalg.norm(j)
    ratio = jn / (tn + 1e-12)
    max_diff = np.abs(t - j).max()
    flag = "  <-- FAIL" if abs(ratio - 1.0) > 0.1 else ""
    print(f"  {name:<48} {tn:>10.3e} {jn:>10.3e} {ratio:>8.4f}{flag}")

for i in range(N_LAYERS):
    compare(f"convs_{i}/w_core",
            torch_fno.convs[i].weight.core.grad, g_jax[f"convs_{i}"]["w_core"])
    compare(f"convs_{i}/w_U0",
            torch_fno.convs[i].weight.factors[0].grad, g_jax[f"convs_{i}"]["w_U0"])
    compare(f"convs_{i}/bias",
            torch_fno.convs[i].bias.grad, g_jax[f"convs_{i}"]["bias"])
    compare(f"channel_mlp_{i}/fc_0/kernel",
            torch_fno.channel_mlp[i].fcs[0].weight.grad.permute(2,1,0),
            g_jax[f"channel_mlp_{i}"]["fc_0"]["kernel"])
    compare(f"fno_skips_{i}/conv/kernel",
            torch_fno.fno_skips[i].conv.weight.grad.permute(2,1,0),
            g_jax[f"fno_skips_{i}"]["conv"]["kernel"])
    compare(f"channel_mlp_skips_{i}/weight",
            torch_fno.channel_mlp_skips[i].weight.grad,
            g_jax[f"channel_mlp_skips_{i}"]["weight"])

print("-" * 80)

# Also compare input gradient
x_torch_g = torch.from_numpy(x_np).requires_grad_(True)
out_torch2 = x_torch_g
for idx in range(N_LAYERS):
    out_torch2 = torch_fno(out_torch2, index=idx)
out_torch2.sum().backward()
g_torch_x = x_torch_g.grad.detach().numpy()

def fwd_jax_full_x(x):
    out = x
    for idx in range(N_LAYERS):
        out = jax_fno.apply(jax_params, out, index=idx)
    return out.sum()

g_jax_x = np.array(jax.grad(fwd_jax_full_x)(x_jax))
ratio_x = np.linalg.norm(g_jax_x) / np.linalg.norm(g_torch_x)
print(f"\n  Input grad ratio j/t = {ratio_x:.4f}  (1.000 = perfect)")
print(f"  ||g_jax_x - g_torch_x|| = {np.linalg.norm(g_jax_x - g_torch_x):.4e}")

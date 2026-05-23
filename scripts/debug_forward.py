"""
Step 3 diagnostic: compare PyTorch and JAX FNO forward passes
on the same deterministic input tensor.

Checks:
  - output shape and dtype
  - output statistics (min, max, mean, std)
  - whether JAX SpectralConv weights have real == imag (the suspected bug)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config (matches darcy_config.py Default) ────────────────────────────────
DATA_CHANNELS   = 1
OUT_CHANNELS    = 1
N_MODES         = [16, 16]
HIDDEN_CHANNELS = 24
PROJ_RATIO      = 2
BATCH           = 2
RESOLUTION      = 16  # train_resolution from config

# ── Build a shared config object ────────────────────────────────────────────
from zencfg import ConfigBase
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.darcy_config import Default
from zencfg import make_config_from_cli

config = Default()
config = config.to_dict()

# ── PyTorch forward ──────────────────────────────────────────────────────────
import torch
from neuralop import get_model

pt_model = get_model(config)
pt_model.eval()

torch.manual_seed(42)
pt_x = torch.ones(BATCH, DATA_CHANNELS, RESOLUTION, RESOLUTION, dtype=torch.float32)
with torch.no_grad():
    pt_out = pt_model(pt_x).numpy()

# ── JAX/Flax forward ─────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from neuralop import get_model_jax

jax_module = get_model_jax(config)

key = jax.random.PRNGKey(42)
jax_x = jnp.ones((BATCH, DATA_CHANNELS, RESOLUTION, RESOLUTION), dtype=jnp.float32)
params = jax_module.init(key, x=jax_x)
jax_out = np.array(jax_module.apply(params, x=jax_x))

# ── Report ───────────────────────────────────────────────────────────────────
def stats(name, arr):
    return (f"  shape={arr.shape}  dtype={arr.dtype}"
            f"  min={arr.min():.6f}  max={arr.max():.6f}"
            f"  mean={arr.mean():.6f}  std={arr.std():.6f}")

print("=== Forward pass output ===")
print(f"[PT ] {stats('pt_out',  pt_out)}")
print(f"[JAX] {stats('jax_out', jax_out)}")

# ── Inspect JAX SpectralConv weights for the real==imag bug ──────────────────
print("\n=== SpectralConv weight audit (JAX) ===")
n_checked = 0
n_identical = 0
for path, w in jax.tree_util.tree_leaves_with_path(params):
    path_str = "/".join(
        (k.key if hasattr(k, 'key') else str(k)) for k in path
    )
    if "weight" in path_str.lower() and jnp.iscomplexobj(w):
        r, i = w.real, w.imag
        max_diff = float(jnp.abs(r - i).max())
        are_identical = bool(jnp.allclose(r, i, atol=1e-6))
        flag = "  <-- real==imag BUG" if are_identical else ""
        print(f"  {path_str}  shape={w.shape}  max|real-imag|={max_diff:.6e}{flag}")
        n_checked += 1
        if are_identical:
            n_identical += 1

if n_checked == 0:
    print("  (no complex weight parameters found)")
elif n_identical == n_checked:
    print(f"\n[FAIL] All {n_checked} complex weight(s) have real == imag "
          "(same RNG key used for both parts in _cx_init).")
elif n_identical == 0:
    print(f"\n[PASS] All {n_checked} complex weight(s) have real != imag.")
else:
    print(f"\n[PARTIAL] {n_identical}/{n_checked} complex weight(s) have real == imag.")

# ── Check for f64 contamination in JAX params ────────────────────────────────
print("\n=== dtype audit (JAX params) ===")
f64_paths = []
for path, w in jax.tree_util.tree_leaves_with_path(params):
    path_str = "/".join(
        (k.key if hasattr(k, 'key') else str(k)) for k in path
    )
    arr = np.array(w)
    if arr.dtype in (np.float64, np.complex128):
        f64_paths.append((path_str, arr.dtype, arr.shape))

if f64_paths:
    print("[WARN] f64/c128 tensors found (performance killer on RTX 3050):")
    for p, d, s in f64_paths:
        print(f"  {p}  dtype={d}  shape={s}")
else:
    print("[PASS] All JAX params are f32/c64.")

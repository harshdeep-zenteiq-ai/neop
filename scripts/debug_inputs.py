"""
Step 2 diagnostic: verify PyTorch and JAX dataloaders yield identical batches.
Checks shapes, dtypes, min, max, and mean for one batch of x and y.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path("~/data/darcy/").expanduser()
N_TRAIN = 1000
BATCH_SIZE = 8
TEST_RESOLUTIONS = [16, 32]
N_TESTS = [100, 50]
TEST_BATCH_SIZES = [16, 16]

# ── PyTorch loader ──────────────────────────────────────────────────────────
import torch
from neuralop.data.datasets import load_darcy_flow_small as load_pt

pt_train, pt_test, _ = load_pt(
    data_root=DATA_ROOT,
    n_train=N_TRAIN,
    batch_size=BATCH_SIZE,
    test_resolutions=TEST_RESOLUTIONS,
    n_tests=N_TESTS,
    test_batch_sizes=TEST_BATCH_SIZES,
    encode_input=False,
    encode_output=False,
)

pt_batch = next(iter(pt_train))
pt_x = pt_batch["x"].numpy()
pt_y = pt_batch["y"].numpy()

# ── JAX loader ──────────────────────────────────────────────────────────────
from neuralop.data.datasets.darcy_jax import load_darcy_flow_small as load_jax

jax_train, jax_test, _ = load_jax(
    data_root=DATA_ROOT,
    n_train=N_TRAIN,
    batch_size=BATCH_SIZE,
    test_resolutions=TEST_RESOLUTIONS,
    n_tests=N_TESTS,
    test_batch_sizes=TEST_BATCH_SIZES,
    encode_input=False,
    encode_output=False,
)

jax_batch = next(iter(jax_train))
jax_x = np.array(jax_batch["x"])
jax_y = np.array(jax_batch["y"])

# ── Report ───────────────────────────────────────────────────────────────────
def stats(name, arr):
    return (f"  shape={arr.shape}  dtype={arr.dtype}"
            f"  min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}")

print("=== x ===")
print(f"[PT ] {stats('pt_x',  pt_x)}")
print(f"[JAX] {stats('jax_x', jax_x)}")

print("\n=== y ===")
print(f"[PT ] {stats('pt_y',  pt_y)}")
print(f"[JAX] {stats('jax_y', jax_y)}")

# ── Assertions ───────────────────────────────────────────────────────────────
ATOL = 1e-5

def check(field, pt_arr, jax_arr):
    ok = True
    if pt_arr.shape != jax_arr.shape:
        print(f"\n[FAIL] {field} shape mismatch: PT={pt_arr.shape}, JAX={jax_arr.shape}")
        ok = False
    if str(pt_arr.dtype) != str(jax_arr.dtype):
        print(f"\n[WARN] {field} dtype mismatch: PT={pt_arr.dtype}, JAX={jax_arr.dtype}")
    if ok:
        if not np.allclose(pt_arr, jax_arr, atol=ATOL):
            diff = np.abs(pt_arr - jax_arr)
            print(f"\n[FAIL] {field} values differ: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
            ok = False
        else:
            print(f"\n[PASS] {field}: shapes and values match (atol={ATOL})")
    return ok

x_ok = check("x", pt_x, jax_x)
y_ok = check("y", pt_y, jax_y)

if not (x_ok and y_ok):
    print("\n[INFO] Value mismatch likely means dataloaders use different shuffle seeds "
          "or orderings — check whether each uses the same random seed / same sample order.")
    sys.exit(1)
else:
    print("\nAll checks passed — dataloaders are in parity.")

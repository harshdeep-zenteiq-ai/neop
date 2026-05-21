#!/usr/bin/env python3
"""
Quick timing test runner for both JAX and PyTorch.
Runs 1 epoch with limited batches to measure timing breakdown.
"""
import os
import sys
import subprocess
from pathlib import Path

print("="*80)
print("PERFORMANCE TIMING TEST: JAX vs PyTorch")
print("="*80)

# Run JAX training with verbose timing
print("\n[1/2] Running JAX training (1 epoch, first 5 batches)...")
print("-"*80)
env_jax = os.environ.copy()
env_jax['JAX_VERBOSE'] = '0'

result = subprocess.run([
    sys.executable,
    "scripts/train_gino_carcfd_jax.py",
    "--opt.n_epochs=1",
    "--wandb.log=False",
], cwd="/home/deepan/neuraloperator", env=env_jax, capture_output=True, text=True)

# Parse JAX output for timing information
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\n\n")
print("="*80)
print("[2/2] Running PyTorch training (1 epoch, first 5 batches)...")
print("-"*80)

result = subprocess.run([
    sys.executable,
    "scripts/train_gino_carcfd.py",
    "--opt.n_epochs=1",
    "--wandb.log=False",
], cwd="/home/deepan/neuraloperator", capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\n" + "="*80)
print("TIMING TEST COMPLETE")
print("="*80)
print("""
To analyze the results:
1. Look for "Batch X: Total time" lines in both outputs
2. Compare "Preprocessing (data conversion)" times
3. Compare "Model train_step" times
4. Check for "Device sync" and "Loss extraction" times in JAX output
5. Look for recompilation markers like "[RECOMPILE]" in JAX output
""")

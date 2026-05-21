"""
Quick timing test for JAX training loop.
Runs a few batches with detailed timing breakdown.
"""
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
from timeit import default_timer
sys.path.insert(0, ".")

print("[JAX] Importing libraries...")
import jax
import jax.numpy as jnp
from neuralop.losses.data_losses_jax import LpLoss
import optax
from neuralop.training.trainer_jax import Trainer, SimpleDataLoader
from copy import deepcopy
import numpy as np
import subprocess

# Run the actual JAX training script with verbose timing enabled
# This is simpler than trying to re-import all the complex dependencies
print("\n" + "="*80)
print("[JAX TIMING TEST] Running JAX training with verbose timing")
print("Running: scripts/train_gino_carcfd_jax.py (first 5 batches only)")
print("="*80 + "\n")

# Instead of re-implementing all the imports, we'll read the timing from
# the existing training script by running it with limited batches
# For now, let's create a minimal test that just instruments the key functions

print("JAX timing test requires running the full training script.")
print("Please run: python scripts/train_gino_carcfd_jax.py")
print("The timing breakdown will be shown in the output.")

# Data processor
output_encoder = deepcopy(data_module_jax.normalizers["press"])
data_processor = GINOCFDDataProcessor(
    normalizer=output_encoder,
    device=config.get("device", "cpu")
)

# Loss function
train_loss_fn = LpLoss(d=2, p=2)

# Create trainer with verbose=True
trainer = Trainer(
    model=model_wrapper,
    n_epochs=1,
    data_processor=data_processor,
    device=config.get("device", "cpu"),
    wandb_log=False,
    verbose=True,  # ENABLE TIMING PRINTOUTS
)

print("\n" + "="*80)
print("[JAX TIMING TEST] Running 5 batches with detailed timing")
print("="*80 + "\n")

# Warm-up batch
print("[JAX] Warm-up batch (includes JIT compilation)...")
warmup_iter = iter(train_loader_jax)
sample = next(warmup_iter)
loss = trainer.train_one_batch(0, sample, train_loss_fn)
print(f"Warm-up loss: {float(loss):.6f}\n")

# Timed batches
print("[JAX] Timed batches (steady-state after JIT)...")
batch_times = []
for batch_idx in range(1, 6):
    sample = next(warmup_iter)
    batch_start = default_timer()
    loss = trainer.train_one_batch(batch_idx, sample, train_loss_fn)
    batch_time = default_timer() - batch_start
    batch_times.append(batch_time)
    print()

print("\n" + "="*80)
print("[JAX SUMMARY]")
print(f"Batch times (s): {[f'{t:.4f}' for t in batch_times]}")
print(f"Mean time/batch: {np.mean(batch_times):.4f}s")
print(f"Std dev:         {np.std(batch_times):.4f}s")
print("="*80)

# Performance Timing Test Guide

## Overview

This guide explains how to run the timing tests to identify which parts of the JAX training loop are causing the 14.5x performance regression compared to PyTorch.

## Quick Start

### Option 1: Run Both Tests (Automated)
```bash
cd /home/deepan/neuraloperator
python run_timing_test.py 2>&1 | tee timing_results.log
```

### Option 2: Run Individual Tests

**JAX Timing Test:**
```bash
cd /home/deepan/neuraloperator
python scripts/train_gino_carcfd_jax.py --opt.n_epochs=1 --wandb.log=False 2>&1 | tee jax_timing.log
```

**PyTorch Timing Test:**
```bash
cd /home/deepan/neuraloperator
python scripts/train_gino_carcfd.py --opt.n_epochs=1 --wandb.log=False 2>&1 | tee torch_timing.log
```

## What to Look For

The trainer will print detailed timing for each batch. Here's what each line means:

### JAX Output Example:
```
Batch 0: Total time 0.5234s
  Preprocessing (data conversion): 0.0045s
  Model train_step (forward + backward + update): 0.4890s
  Regularizer loss: 0.0001s
  [JIT] Compiling new JIT function (cached: loss_fn hash=...)
  Device sync time: 0.0120s
  Loss extraction time: 0.0008s
```

### What Each Timing Means:

| Component | What it measures | Expected time (ms) | Red flag if... |
|---|---|---|---|
| **Preprocessing** | Data conversion from numpy to JAX arrays | 2-10 | >20 (indicates slow data transfer) |
| **Model train_step** | JIT dispatch + async device work | ~33 | >100 (indicates slow compute or recompilation) |
| **[JIT] Compiling** | XLA recompilation happening | Should see once | Appears multiple times (LR rebuild issue) |
| **Device sync** | Blocking wait for device computation | 0-5 | >10 (indicates too much async work) |
| **Loss extraction** | `float()` conversion of all batch losses | 1-5 | >10 (indicates per-batch sync issue) |
| **Total batch time** | Sum of all above | ~33 | >48 (your current regression) |

## Key Hypotheses to Test

Based on the code exploration, we expect to find evidence of these bottlenecks:

### 1. **NumPy Round-Trip in Data Loader** (Expected: ~2-10ms per batch)
**Signal:** "Preprocessing" time is consistently >10ms
- Data loader stacks with `np.stack` (numpy arrays)
- Preprocessing calls `jnp.asarray()` (numpy→JAX transfer)
- This happens on EVERY batch

### 2. **LR Scheduler Forcing JIT Rebuild** (Expected: recompilation every epoch if LR changes)
**Signal:** `[JIT] Compiling new JIT function` appears more than once per epoch
- Each LR change sets `self._jit_fn = None`
- New closure captures new optimizer object
- Entire XLA program must recompile

### 3. **Per-Batch Device Sync During Evaluation** (Expected: not in training, but in eval)
**Signal:** Look in eval section for `float(val_loss)` timing
- Current code syncs once per eval batch
- Should be moved outside loop

### 4. **Dynamic Vertex Slicing** (Expected: detectable if shape varies)
**Signal:** Look for different `out_p` shapes across batches
- In `preprocess()`: `out_p[:output_vertices, :]`
- If `output_vertices` varies, XLA recompiles per new shape

### 5. **Padding Waste in Graph Operations** (Expected: detectable via utilization)
**Signal:** Check if neighbor edge utilization is low
- `IntegralTransform` processes `max_edges` including dummy entries
- If padding ratio is high, much compute is wasted

## Expected Timing Breakdown

### PyTorch (Current Baseline ~0.033s/batch):
```
Preprocessing (CPU):     ~2-5ms   (numpy operations)
Forward pass:            ~10-15ms (eager execution)
Backward pass:           ~10-15ms (autograd)
Optimizer step:          ~2-3ms   (SGD)
Loss extraction:         ~1-2ms   (.item() sync)
────────────────────────────────
Total:                   ~33ms
```

### JAX (Current Regression ~0.48s/batch):
```
Preprocessing:           ~2-10ms  (numpy→JAX transfer)
Model train_step:        ~450ms   (JIT dispatch + compute)
Device sync:             ~10-15ms (block_until_ready at epoch end)
Loss extraction:         ~8-10ms  (float() on each loss)
────────────────────────────────
Total:                   ~480ms
```

**The ~450ms train_step is the smoking gun.** This should be ~30ms (similar to PyTorch), indicating:
- Excessive JIT overhead
- Recompilation happening
- Slow device-to-host transfer
- Or actual compute difference (unlikely since model is same)

## Detailed Analysis Checklist

After running the tests, check these in order:

### ✓ Batch Timing Consistency
```bash
# Look for this pattern in JAX output
grep "Batch.*Total time" jax_timing.log | head -5
```

- **Batch 0** will be slowest (includes JIT compilation)
- **Batches 1-4** should be consistent (steady-state)
- If batches 2-4 vary significantly → indicates random recompilations

### ✓ JIT Compilation Markers
```bash
grep "\[JIT\]" jax_timing.log
```

- Should see exactly **1 line** (batch 0)
- If you see more → hypothesis #2 (LR rebuild) confirmed

### ✓ Preprocessing Time Variance
```bash
grep "Preprocessing" jax_timing.log
```

- Should be consistent across all batches
- If varies widely → hypothesis #4 (dynamic shapes) likely

### ✓ Device Sync Time
```bash
grep "Device sync" jax_timing.log
```

- Should appear once at **end of epoch**
- If appears per-batch → hypothesis #3 confirmed

### ✓ Loss Extraction Time
```bash
grep "Loss extraction" jax_timing.log
```

- Should be low (~1-5ms) since done once per epoch
- Check if this is causing the reported "float() sync" issue

## Next Steps Based on Results

### If Preprocessing > 10ms
→ **Fix priority 1:** Change `_LazyLoader` to use `jnp.stack` instead of `np.stack`

### If [JIT] appears multiple times
→ **Fix priority 2:** Eliminate JIT rebuild on LR change using `optax.inject_hyperparams`

### If Device sync time > 20ms
→ This is expected (all batches' work completing), not a problem

### If Loss extraction time > 10ms
→ **Fix priority 3:** Move `float()` conversion outside the loop (though this is low impact)

## Interpreting Strange Results

**If JAX is actually faster than PyTorch:** Something is wrong with the test setup. JIT should not eliminate all Python overhead. Check that verbose timing is actually enabled.

**If preprocessing is 0ms:** Timing is not being captured. Check that verbose=True is set in the trainer.

**If model train_step is slow but consistent:** The issue is not recompilation—could be actual compute difference, memory access pattern, or synchronization point.

## Saving Results for Later Analysis

Run the tests and save output:
```bash
# Both tests
python run_timing_test.py 2>&1 | tee ~/timing_results_$(date +%Y%m%d_%H%M%S).log

# Or individually
python scripts/train_gino_carcfd_jax.py --opt.n_epochs=1 --wandb.log=False > ~/jax_$(date +%Y%m%d_%H%M%S).log 2>&1
python scripts/train_gino_carcfd.py --opt.n_epochs=1 --wandb.log=False > ~/torch_$(date +%Y%m%d_%H%M%S).log 2>&1
```

Then analyze:
```bash
# Extract timing from JAX
grep "Batch\|Preprocessing\|train_step\|Device sync\|Loss extraction" ~/jax_*.log

# Extract timing from PyTorch
grep "Batch\|Preprocessing\|train_step\|forward\|backward" ~/torch_*.log
```

## Questions to Answer

1. Is preprocessing consistently <10ms? (Rules out data loader issue)
2. Does `[JIT]` appear once or multiple times? (Confirms LR rebuild issue)
3. Is "Batch 1" timing similar to "Batch 2-4"? (Rules out per-batch recompilation)
4. Is total batch time consistently ~0.48s? (Confirms 14.5x regression)
5. Which component is actually slow? (Determines priority for fixing)

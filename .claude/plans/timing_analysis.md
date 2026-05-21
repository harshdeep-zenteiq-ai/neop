# Timing Analysis Results

## Executive Summary

**The bottleneck is the core JAX compute itself (~0.477s/batch), not data loading or scheduling.**

| Component | JAX | PyTorch | Ratio | Status |
|---|---|---|---|---|
| **Preprocessing** | 0.001s | 0.0002s | 5-10x slower | ✗ NOT the issue (negligible absolute time) |
| **Model train_step** | 0.477s | 0.0335s | **14.3x slower** | ✓ **THIS IS THE BOTTLENECK** |
| **Device sync** | 0.0002s | - | - | ✗ NOT an issue |
| **Loss extraction** | 0.0013s | - | - | ✗ NOT an issue |
| **JIT recompilation** | Once (warm-up) | N/A | - | ✗ NOT happening |
| **Total per batch** | 0.477s | 0.0335s | **14.3x** | **Confirmed regression** |

---

## Hypothesis Testing Results

### ✓ RULED OUT: NumPy Round-Trip in Data Loader (Hypothesis #1)
**Finding:** Preprocessing is only 0.001-0.002s (JAX) vs 0.0002s (PyTorch)
- Even if we eliminated it entirely, we'd save only ~1ms per batch
- This is only 0.2% of the 477ms bottleneck
- **Conclusion:** Not worth fixing

### ✓ RULED OUT: LR Scheduler Forcing JIT Rebuild (Hypothesis #2)
**Finding:** `[JIT] Compiling` appears exactly once (batch 0 only)
- All subsequent batches use the cached JIT function
- No recompilation markers appear
- LR changes are not happening per-batch or per-sample
- **Conclusion:** Not happening; not the issue

### ✓ RULED OUT: Per-Batch Device Sync (Hypothesis #3)
**Finding:** Device sync and loss extraction at epoch-end are negligible
- Device sync: 0.0002s (once per epoch)
- Loss extraction: 0.0013s (once per epoch)
- These don't appear in per-batch timings
- **Conclusion:** Not a bottleneck; current pattern is correct

### ✓ RULED OUT: Dynamic Vertex Slicing (Hypothesis #4)
**Finding:** Batches 1-9 show consistent timing (~0.477s ± 0.008s)
- If shapes were varying per batch, you'd see spikes on new shapes
- Timing is rock-stable → shapes are constant
- **Conclusion:** Not recompiling due to shape changes

### ✓ RULED OUT: Padding Waste in Graph Ops (Hypothesis #5)
**Finding:** If padding waste was the issue (50% utilization), we'd see a 2x slowdown
- Current slowdown is 14.3x, not 2x
- Padding is the wrong scale of problem
- **Conclusion:** Contributes negligibly to the regression

---

## The Real Bottleneck

**The JAX implementation of the model itself is ~14.3x slower than PyTorch.**

The "Model train_step" timings are:
```
JAX:      0.477s (forward + backward + update via jax.value_and_grad + optax)
PyTorch:  0.0335s (forward + backward + update via autograd + optimizer)
```

This includes:
1. Forward pass through GINO + FNO + IntegralTransform
2. Loss computation
3. Gradient computation via `jax.value_and_grad`
4. Optimizer update via optax

### Possible Root Causes (in order of likelihood):

#### 1. **XLA Compilation Overhead** (High probability)
- JAX compiles to XLA; even cached JIT has dispatch overhead
- The warm-up batch (4.24s) suggests significant compilation cost
- Steady-state (0.477s) suggests the cached JIT is not as optimized as PyTorch's eager execution
- **Check:** Is XLA doing full optimization on the compiled graph, or is it leaving slow ops?

#### 2. **Inefficient JAX/Flax Operations** (High probability)
- `jax.vmap` over scatter operations (segment_sum)
- `jax.lax.scan` (currently commented out, but if used)
- Inefficient gather/scatter for graph operations
- FFT operations in SpectralConv
- **Check:** Are there unnecessary vmaps or redundant operations?

#### 3. **Hidden Device-to-Host Sync in Compute Path** (Medium probability)
- Something in the model code is calling `jax.device_get()` implicitly
- Or a control-flow operation that forces materialization
- **Check:** Look for Python `if` statements that depend on JAX array values inside the JIT

#### 4. **Memory Allocation/Deallocation Overhead** (Medium probability)
- JAX may be allocating/freeing more aggressively than PyTorch
- Or the graph structure requires more intermediate buffers
- **Check:** GPU memory usage during the train_step

#### 5. **Different Numerical Path Length** (Low probability)
- Loss values differ (JAX: 0.5265, PyTorch: 0.2961)
- Suggests different initialization or model structure
- Could cause different computation graphs
- **Check:** Are the models truly identical? Same weight init?

---

## Next Steps: Targeted Investigation

### Phase 1: Pinpoint the Slow Operation (30 minutes)

Add sub-second timing within the train_step to isolate which component is slow:

```python
# In train_step (scripts/train_gino_carcfd_jax.py)
@jax.jit
def _step(params, opt_state, model_inputs, y):
    t0 = time.perf_counter()
    
    # Forward pass
    out = module.apply(params, **model_inputs)
    t_forward = time.perf_counter() - t0
    
    # Loss + gradients
    loss, grads = jax.value_and_grad(forward_loss)(params)
    t_grad = time.perf_counter() - t0 - t_forward
    
    # Optimizer update
    updates, new_opt = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    t_update = time.perf_counter() - t0 - t_forward - t_grad
    
    # Print timing (async, won't block)
    jax.debug.print("Forward: {}, Grad: {}, Update: {}", 
                    t_forward, t_grad, t_update)
    
    return loss, new_params, new_opt
```

**Expected output:** Will show whether it's forward, gradient, or optimizer that's slow

### Phase 2: Profile XLA Compilation (1 hour)

Use JAX's built-in tracing to see what XLA is generating:

```python
import jax.profiler

# Before training loop
jax.profiler.start_trace("/tmp/jax_trace")

# Run 1-2 batches
for i, sample in enumerate(train_loader):
    if i == 2:
        break
    trainer.train_one_batch(i, sample, loss_fn)

jax.block_until_ready(...)
jax.profiler.stop_trace()

# View: tensorboard --logdir=/tmp/jax_trace
```

Look for:
- Total op count (if very high, suggests inefficient graph)
- Slow ops (which primitives are taking time?)
- Memory throughput (is it bandwidth-limited?)

### Phase 3: Compare Model Code Line-by-Line (2 hours)

Since we know it's the model compute, directly compare:
- Forward pass: `neuralop/models/gino_jax.py` vs `neuralop/models/gino.py`
- Layers: `integral_transform_jax.py` vs `integral_transform.py`
- Convolutions: `spectral_convolution_jax.py` vs `spectral_convolution.py`

Look for:
- Unnecessary `jax.vmap` wrapping
- Redundant shape manipulations
- Type conversions that force computation
- Loop structures that should be scanned

---

## Summary Table: What We Know

| Finding | Status | Implication |
|---|---|---|
| Preprocessing is slow | ✗ Ruled out | Keep as-is |
| LR rebuild causes recompilation | ✗ Ruled out | No optimization needed |
| Per-batch sync is bottleneck | ✗ Ruled out | Current async pattern is correct |
| Shape variability causes recompilation | ✗ Ruled out | Padding is working |
| **Core JAX compute is 14.3x slower** | ✓ **CONFIRMED** | **Needs investigation** |
| Warm-up includes heavy XLA compilation | ✓ Observed | Expected; 4.24s → 0.477s |
| Steady-state is stable (no more recompilation) | ✓ Observed | JIT cache is working |

---

## Recommendation

**Do NOT fix the ruled-out hypotheses.** They won't move the needle.

**Focus on:** Why is the JAX model compute so slow?

1. **Immediate:** Add sub-second timing within _step to identify forward vs grad vs update
2. **Follow-up:** Profile with JAX tracer to see XLA optimization
3. **Deep-dive:** Compare model implementations line-by-line

The issue is not in the training infrastructure—it's in how the model is expressed in JAX.

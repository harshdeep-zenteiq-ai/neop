# FlaxModelWrapper Instantiation (Lines 337-341)

## Overview
When `model = FlaxModelWrapper(model, ...)` executes, a wrapper object is created that combines the JAX/Flax neural network module with optax optimization. This happens in **two phases**: immediate wrapper setup and lazy parameter initialization.

---

## Phase 1: Immediate Wrapper Creation (Lines 337-341)

### What Gets Called
```python
model = FlaxModelWrapper(
    model,                                      # Flax GINO module instance
    learning_rate=config.opt.learning_rate,    # 1e-3 (from config)
    weight_decay=config.opt.weight_decay,      # 1e-4 (from config)
)
```

### FlaxModelWrapper.__init__() Execution (lines 213-221)
At instantiation, the following attributes are **immediately assigned**:

| Attribute | Value | Type | Purpose |
|-----------|-------|------|---------|
| `self.module` | Original GINO model | Flax Linen module | Holds the neural network |
| `self.key` | `jax.random.PRNGKey(0)` | JAX PRNG key | Random seed for model init |
| `self.tx` | `optax.adamw(learning_rate=1e-3, weight_decay=1e-4)` | optax Optimizer | Creates AdamW optimizer with specified hyperparameters |
| `self.params` | `None` | NoneType | **Not initialized yet** |
| `self.opt_state` | `None` | NoneType | **Not initialized yet** |
| `self._initialized` | `False` | bool | Lazy initialization flag |
| `self._jit_fn` | `None` | NoneType | JIT-compiled training step (built on first use) |

### Key Points
- **No model parameters are created yet** — `self.params` is None
- **No optimizer state is created yet** — `self.opt_state` is None
- **No forward pass has run** — The model has not been traced/compiled
- **Instantiation is lightweight** — Only creates the wrapper and optimizer object

### Size After Instantiation
```
FlaxModelWrapper memory footprint:
  - Wrapper object overhead: ~1-2 KB
  - Optimizer object (optax.adamw): ~10-50 bytes (no param-dependent state yet)
  - Self references: ~1-2 KB
Total: ~15-60 KB (negligible compared to model params)
```

---

## Phase 2: Lazy Initialization (Deferred to First Training Step)

### When Initialization Happens
The actual parameter initialization is **deferred until the first training step** via the `_init()` method (lines 223-228):

```python
def _init(self, sample_kwargs):
    """Initialize params and optimizer state from the first real sample."""
    model_inputs = {k: v for k, v in sample_kwargs.items() if k in self._MODEL_KEYS}
    self.params    = self.module.init(self.key, **model_inputs)      # JAX init
    self.opt_state = self.tx.init(self.params)                       # Optimizer init
    self._initialized = True
```

### What _init() Does (Triggered Later)
1. **Extracts model inputs** from sample data using allowed keys: 
   - `input_geom`, `latent_queries`, `output_queries`, `x`, `latent_features`, `ada_in`, `neighbors_in`, `neighbors_out`
2. **Initializes GINO parameters** via `self.module.init(key, **model_inputs)`:
   - Shape inference: Traces model with real sample shapes
   - Parameter tree created: ~1.5M float32 parameters (based on GINO_Small3d)
   - Returns FrozenDict of parameter tensors
3. **Initializes optimizer state** via `self.tx.init(self.params)`:
   - Creates AdamW state: momentum and variance accumulators for each parameter
   - Size: ~2x model params (momentum + variance per param)
   - Memory: ~12 MB for GINO_Small3d

### Timeline
```
Line 337-341: FlaxModelWrapper created, params=None, opt_state=None
    ↓
    [Training starts...]
    ↓
Line 344-347: First sample accessed from dataloader
    ↓
trainer.fit() or train_step() called
    ↓
FlaxModelWrapper.train_step() → calls _init() on first sample
    ↓
self.params and self.opt_state initialized from real data
```

---

## State After Instantiation (Before Training)

```python
model.module        # → <GINO instance>
model.params        # → None
model.opt_state     # → None
model.key           # → PRNGKey([0, 0])
model.tx            # → <optax.adamw>
model._initialized  # → False
model._jit_fn       # → None
```

---

## Methods Available After Instantiation

### Forward Pass (Eval)
```python
output = model(input_geom=x, latent_queries=q_lat, output_queries=q_out, x=...)
# Calls __call__() → lazy inits if needed → module.apply(self.params, ...)
```

### Training Step
```python
loss = model.train_step(kwargs, loss_fn)
# Calls train_step() → builds _jit_fn if needed → JIT-compiled gradient update
```

### Neighbor Precomputation
```python
model.precompute_neighbors(train_data, output_n_points=...)
# Delegates to self.module.precompute_neighbors() if available
```

---

## JIT Compilation Status After Instantiation

| Component | Status | Cache |
|-----------|--------|-------|
| `_jit_fn` | Not built | None |
| `_scan_fn` | Commented out | N/A |
| Model forward | Not traced | Only runs on first eval/train |

First call to `train_step()` will:
1. Call `_build_jit_fn(loss_fn)` → Creates `@jax.jit` decorated function
2. Cache in `self._jit_fn`
3. Reuse for all subsequent training steps

---

## Memory Breakdown After Line 341

```
Immediate Memory:
  Wrapper object:           ~60 KB
  optax.adamw object:       ~100 B
  ─────────────────────────────────
  Total:                    ~61 KB

Deferred (at first train_step):
  Model params (GINO):      ~6 MB (1.5M × 4 bytes)
  Optimizer state (2×):     ~12 MB (momentum + variance)
  Cached JIT functions:     ~1-5 MB
  ─────────────────────────────────
  Total at first step:      ~20 MB

Full Memory After Training Starts:
  Wrapper + params + opt_state:  ~20-61 MB
```

---

## Comparison: Before vs After Wrapper

### Before (GINO only)
```python
model = get_model_jax(config)  # Returns GINO Flax module
# Memory: Just the module definition (~1 MB)
# Has: model.init(), model.apply()
# Does: Inference only (no optimization)
```

### After (FlaxModelWrapper)
```python
model = FlaxModelWrapper(model, ...)  # Wraps GINO
# Memory: ~61 KB wrapper + module definition
# Has: train_step(), __call__(), precompute_neighbors()
# Does: Training with gradient updates via AdamW
```

---

## Summary

**What Happens (Lines 337-341):**
1. FlaxModelWrapper object created with empty params/opt_state
2. optax.adamw optimizer instantiated with lr=1e-3, weight_decay=1e-4
3. PRNG key set to PRNGKey(0) for reproducibility
4. Ready state: `_initialized=False` (will initialize on first data sample)

**What Does NOT Happen Yet:**
- ❌ Model parameters NOT created
- ❌ Optimizer state NOT created
- ❌ Model NOT traced/compiled
- ❌ JIT functions NOT built

**Next Steps:** 
When `trainer.fit()` or `model.train_step(sample, loss_fn)` is called with the first training sample, lazy initialization triggers and the full training machinery is activated.

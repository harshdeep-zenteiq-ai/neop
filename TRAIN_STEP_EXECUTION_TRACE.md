# train_step() Execution Trace - Complete Flow

**Location:** `neuralop/training/trainer_jax.py` line 535

**Call:** `loss = self.model.train_step(sample, training_loss)`

**Where:**
- `self.model` → `FlaxModelWrapper` instance (created at line 337-341 in train_gino_carcfd_jax.py)
- `sample` → Preprocessed data dict with keys: `input_geom`, `latent_queries`, `output_queries`, `latent_features`, `y`, `x`, `neighbors_in`, `neighbors_out`
- `training_loss` → Loss function (e.g., `LpLoss(d=2)`)

---

## Execution Sequence

### PHASE 1: Entry to train_step()

**Location:** `scripts/train_gino_carcfd_jax.py` lines 319-329

```python
def train_step(self, kwargs, loss_fn):
    """Per-sample JIT step (fallback)."""
```

**Called with:**
```python
kwargs = sample  # Dict with 8 keys
loss_fn = training_loss  # LpLoss instance
```

---

### PHASE 2: Check Initialization Status (Line 321)

```python
if not self._initialized:
    self._init(kwargs)
```

**First-time execution (epoch 0, batch 0):**
- `self._initialized` = `False` (set in `__init__` line 219)
- **Condition: TRUE** → Execute `_init(kwargs)`

**Subsequent executions:**
- `self._initialized` = `True` (set after first call)
- **Condition: FALSE** → Skip to line 323

---

### PHASE 2.1: Lazy Initialization via _init() (First Batch Only)

**Location:** `scripts/train_gino_carcfd_jax.py` lines 223-228

```python
def _init(self, sample_kwargs):
    """Initialize params and optimizer state from the first real sample."""
    model_inputs = {k: v for k, v in sample_kwargs.items() if k in self._MODEL_KEYS}
    self.params    = self.module.init(self.key, **model_inputs)
    self.opt_state = self.tx.init(self.params)
    self._initialized = True
```

#### Line 225: Extract Model Inputs

```python
model_inputs = {k: v for k, v in sample_kwargs.items() if k in self._MODEL_KEYS}
```

**`self._MODEL_KEYS`** (defined line 208-211):
```python
frozenset({
    'input_geom', 'latent_queries', 'output_queries',
    'x', 'latent_features', 'ada_in', 'neighbors_in', 'neighbors_out',
})
```

**Input sample dict keys:**
```
'input_geom', 'latent_queries', 'output_queries', 'latent_features', 'y', 'x', 
'neighbors_in', 'neighbors_out'
```

**Result after filtering:**
```python
model_inputs = {
    'input_geom': ndarray (batch, 3586, 3),
    'latent_queries': ndarray (batch, 32, 32, 32, 3),
    'output_queries': ndarray (batch, 3586, 3),
    'latent_features': ndarray (batch, 32, 32, 32, 1),
    'x': None,
    'neighbors_in': dict {
        'neighbors_index': ndarray (16681,),
        'segment_ids': ndarray (16681,),
        'counts': ndarray (32768,)
    },
    'neighbors_out': dict {
        'neighbors_index': ndarray (16681,),
        'segment_ids': ndarray (16681,),
        'counts': ndarray (3586,)
    }
}
# Note: 'y' and 'ada_in' NOT included (not in _MODEL_KEYS)
```

**Excluded keys:** `'y'` (ground truth), `'ada_in'` (not in sample anyway)

---

#### Line 226: Initialize Model Parameters

```python
self.params = self.module.init(self.key, **model_inputs)
```

**What happens:**
1. **Flax module tracing:** `self.module.init()` calls the Flax model's initialization
2. **Shape inference:** JAX traces through the GINO model with sample shapes
3. **Parameter creation:** All learnable parameters are created based on inferred shapes
4. **Return:** FrozenDict of all parameters (weights, biases, etc.)

**Example output structure:**
```python
self.params = FrozenDict({
    'params': {
        'GINOModel_0': {
            'embed_pos_0': {
                'kernel': ndarray (3, 128),      # Position embedding
                'bias': ndarray (128,)
            },
            'gnn_0': {
                'kernel': ndarray (128, 128),    # Graph layer
                'bias': ndarray (128,)
            },
            # ... many more layers ...
            'output_head': {
                'kernel': ndarray (128, 1),      # Output projection
                'bias': ndarray (1,)
            }
        }
    }
})
```

**Timing:** ~100-500ms (one-time cost)

**Size:** ~6 MB (1.5M float32 parameters)

---

#### Line 227: Initialize Optimizer State

```python
self.opt_state = self.tx.init(self.params)
```

**What happens:**
1. **Optimizer trace:** `self.tx.init()` creates optimizer state (AdamW)
2. **State dict creation:** Momentum and variance accumulators for each parameter
3. **Return:** optax optimizer state matching parameter tree structure

**Example optimizer state:**
```python
self.opt_state = {
    'count': ndarray([], dtype=int32),     # Step counter (value: 0)
    'mu': {                                # First moment (momentum)
        'params': {
            'GINOModel_0': {
                'embed_pos_0': {
                    'kernel': ndarray (3, 128),  # Momentum for position embedding kernel
                    'bias': ndarray (128,)       # Momentum for bias
                },
                # ... mirror of params structure ...
            }
        }
    },
    'nu': {                                # Second moment (variance)
        'params': {
            'GINOModel_0': {
                'embed_pos_0': {
                    'kernel': ndarray (3, 128),  # Variance for position embedding kernel
                    'bias': ndarray (128,)
                },
                # ... mirror of params structure ...
            }
        }
    }
}
```

**Timing:** ~10-50ms

**Size:** ~12 MB (2× model params)

---

#### Line 228: Set Initialized Flag

```python
self._initialized = True
```

**Effect:** Future calls to `train_step()` will skip the `_init()` block

---

### PHASE 3: Build JIT Function (First Batch Only)

**Location:** `scripts/train_gino_carcfd_jax.py` lines 323-324

```python
if self._jit_fn is None:
    self._jit_fn = self._build_jit_fn(loss_fn)
```

**First-time execution:**
- `self._jit_fn` = `None` (set in `__init__` line 220)
- **Condition: TRUE** → Execute `_build_jit_fn(loss_fn)`

**Subsequent executions:**
- `self._jit_fn` = cached JIT function
- **Condition: FALSE** → Skip, reuse cached function

---

### PHASE 3.1: Build JIT Function via _build_jit_fn()

**Location:** `scripts/train_gino_carcfd_jax.py` lines 297-310

```python
def _build_jit_fn(self, loss_fn):
    module = self.module
    tx     = self.tx

    @jax.jit
    def _step(params, opt_state, model_inputs, y):
        def forward_loss(p):
            out = module.apply(p, **model_inputs)
            return loss_fn(out, y=y)
        loss, grads = jax.value_and_grad(forward_loss)(params)
        updates, new_opt = tx.update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), new_opt

    return _step
```

**What gets returned:**
- A **JIT-compiled function** `_step` decorated with `@jax.jit`
- This function is NOT executed yet—just defined

**Return value stored in:**
```python
self._jit_fn = _step  # JIT-compiled closure with loss_fn captured
```

**Note:** The actual JIT compilation happens on first call to `self._jit_fn()`, not here.

---

### PHASE 4: Extract Model Inputs (Line 325)

```python
model_inputs = {k: v for k, v in kwargs.items() if k in self._MODEL_KEYS}
```

**Same filtering as line 225:**
```python
model_inputs = {
    'input_geom': ndarray (batch, 3586, 3),
    'latent_queries': ndarray (batch, 32, 32, 32, 3),
    'output_queries': ndarray (batch, 3586, 3),
    'latent_features': ndarray (batch, 32, 32, 32, 1),
    'x': None,
    'neighbors_in': dict {...},
    'neighbors_out': dict {...}
}
```

---

### PHASE 5: Execute JIT Training Step (Lines 326-328)

```python
loss, self.params, self.opt_state = self._jit_fn(
    self.params, self.opt_state, model_inputs, kwargs['y']
)
```

**Called with:**
- `self.params` → FrozenDict of model parameters from phase 2.1
- `self.opt_state` → Optimizer state from phase 2.1
- `model_inputs` → Dict from phase 4
- `kwargs['y']` → Ground truth pressure, shape (batch, 1, 3586, 1)

---

### PHASE 5.1: Inside _step() - Forward Pass (Line 304)

```python
out = module.apply(self.params, **model_inputs)
```

**What happens:**
1. **JAX functional inference:** `module.apply()` runs the GINO model in pure functional mode
2. **Input shapes:**
   - `input_geom`: (batch, 3586, 3)
   - `latent_queries`: (batch, 32, 32, 32, 3)
   - `output_queries`: (batch, 3586, 3)
   - `latent_features`: (batch, 32, 32, 32, 1)
   - `neighbors_in`: edge index + segment IDs + counts
   - `neighbors_out`: edge index + segment IDs + counts
   - `x`: None

3. **Model computation:**
   - Position encoding of geometry
   - Graph neural operator on input mesh
   - Latent space projection (FFT-based)
   - Output space projection
   - Vertex-wise pressure prediction

4. **Output shape:** `(batch, 3586, 1)` or `(batch, n_output_vertices, 1)`

**Return value:**
```python
out = ndarray, shape (batch, 3586, 1)  # Predicted pressure
```

**Timing:** ~50-200ms (forward pass)

---

### PHASE 5.2: Compute Loss (Lines 303-305)

```python
def forward_loss(p):
    out = module.apply(p, **model_inputs)
    return loss_fn(out, y=y)
loss, grads = jax.value_and_grad(forward_loss)(params)
```

**What happens:**
1. **Define closure:** `forward_loss(p)` captures `module`, `model_inputs`, `loss_fn`, `y`
2. **Compute loss and gradients:**
   - Forward pass: `out = module.apply(params, **model_inputs)`
   - Loss computation: `loss = loss_fn(out, y=y)`
   - Backward pass: Gradient computation w.r.t. all parameters
3. **Return:**
   - `loss` → scalar float32 (e.g., 0.0234)
   - `grads` → FrozenDict matching parameter structure with gradients

**Loss computation (assuming LpLoss with d=2):**
```python
loss = loss_fn(out, y=y)  # Computes L2 norm: ||out - y||_2
# Result: scalar tensor, e.g., shape=()
```

**Gradient structure:**
```python
grads = {
    'params': {
        'GINOModel_0': {
            'embed_pos_0': {
                'kernel': ndarray (3, 128),    # ∂loss/∂kernel
                'bias': ndarray (128,)         # ∂loss/∂bias
            },
            # ... all parameters have corresponding gradients ...
        }
    }
}
```

**Timing:** ~100-300ms (backward pass)

---

### PHASE 5.3: Optimizer Update (Line 307)

```python
updates, new_opt = tx.update(grads, opt_state, params)
```

**What happens:**
1. **Gradient transformation:** AdamW processes gradients
   - Compute exponential moving averages of gradient (momentum)
   - Compute exponential moving averages of squared gradient (variance)
   - Compute adaptive learning rate: `lr_t = lr / (sqrt(variance) + eps)`
   - Apply weight decay: `update = -lr_t * (gradient + weight_decay * param)`

2. **Return:**
   - `updates` → Dict matching parameter structure with parameter updates
   - `new_opt` → Updated optimizer state (incremented count, updated momentum/variance)

**Example AdamW computation:**
```python
# For each parameter:
m_t = beta1 * m_{t-1} + (1 - beta1) * grad
v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
update = -lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
```

**Timing:** ~1-10ms (element-wise operations)

---

### PHASE 5.4: Apply Updates to Parameters (Line 308)

```python
return loss, optax.apply_updates(params, updates), new_opt
```

**What happens:**
1. **Add updates to parameters:**
   ```python
   new_params = params + updates
   ```

2. **Return tuple:**
   ```python
   (loss, new_params, new_opt)
   ```

**Result:**
- `loss` → scalar, e.g., `0.0234`
- `new_params` → Updated FrozenDict (all weights moved in direction of negative gradient)
- `new_opt` → Updated optimizer state for next step

**Timing:** <1ms

---

### PHASE 6: Update Model State (Lines 326-328)

```python
loss, self.params, self.opt_state = self._jit_fn(
    self.params, self.opt_state, model_inputs, kwargs['y']
)
```

**After JIT execution completes:**
```python
self.params = new_params      # Updated parameters
self.opt_state = new_opt      # Updated optimizer state
loss = 0.0234                 # Scalar loss value
```

---

### PHASE 7: Return to train_one_batch() (Line 329)

```python
return loss
```

**Returns to caller:**
```python
loss = 0.0234  # Scalar float32
```

**Back in trainer_jax.py line 535:**
```python
loss = self.model.train_step(sample, training_loss)
# loss is now 0.0234
```

---

## Summary: Complete Execution Flow

### First Batch (Epoch 0, Batch 0)

```
train_step(sample, training_loss)
├─ Phase 1: Enter train_step
├─ Phase 2: Check init → TRUE
│  └─ _init() [~600ms]
│     ├─ Line 225: Extract model_inputs
│     ├─ Line 226: Initialize params via module.init() [~100-500ms]
│     ├─ Line 227: Initialize opt_state via tx.init() [~10-50ms]
│     └─ Line 228: Set _initialized = True
├─ Phase 3: Check JIT cache → NULL
│  └─ _build_jit_fn(loss_fn) [~1ms, builds closure]
│     ├─ Creates @jax.jit decorated function
│     └─ Stores in self._jit_fn
├─ Phase 4: Extract model_inputs [~1ms]
├─ Phase 5: Execute JIT step [~300-700ms, FIRST TIME TRIGGERS XLA COMPILATION]
│  └─ Inside @jax.jit _step():
│     ├─ Line 304: Forward pass [~50-200ms]
│     ├─ Line 306: Compute loss & gradients [~100-300ms]
│     ├─ Line 307: AdamW update computation [~1-10ms]
│     └─ Line 308: Apply updates & return
├─ Phase 6: Update self.params and self.opt_state [<1ms]
└─ Phase 7: Return loss

TOTAL TIME: ~900ms-1.2s (includes XLA compilation)
```

### Subsequent Batches (Epoch 0, Batch 1+)

```
train_step(sample, training_loss)
├─ Phase 1: Enter train_step
├─ Phase 2: Check init → FALSE [SKIP _init]
├─ Phase 3: Check JIT cache → EXISTS [REUSE]
├─ Phase 4: Extract model_inputs [~1ms]
├─ Phase 5: Execute JIT step [~150-400ms, REUSES COMPILED XLA CODE]
│  └─ XLA kernel runs pre-compiled
├─ Phase 6: Update self.params and self.opt_state [<1ms]
└─ Phase 7: Return loss

TOTAL TIME: ~150-410ms (no compilation overhead)
```

---

## Key Outputs at Each Phase

| Phase | Output | Type | Shape/Value |
|-------|--------|------|-------------|
| 2.1 | `self.params` | FrozenDict | ~1.5M parameters |
| 2.1 | `self.opt_state` | dict | 2× params (momentum + variance) |
| 3.1 | `self._jit_fn` | JAX JIT function | closure capturing loss_fn |
| 4 | `model_inputs` | dict | 8 keys, arrays + dicts |
| 5.1 | `out` | ndarray | (batch, 3586, 1) |
| 5.2 | `loss` | ndarray | scalar, e.g., 0.0234 |
| 5.2 | `grads` | FrozenDict | Same structure as params |
| 5.3 | `updates` | dict | Parameter-shaped updates |
| 5.3 | `new_opt` | dict | Updated optimizer state |
| 5.4 | `new_params` | FrozenDict | Updated parameters |
| 7 | `loss` | float32 | Scalar return value |

---

## Print Output Expected

If the line is called with debug prints enabled:

```
sample before preprocess in train_one_batch:
  vertices: shape=(3586, 3)
  query_points: shape=(32768, 3)
  distance: shape=(3586,)
  press: shape=(3586,)
  neighbors_in: dict with keys ['neighbors_index', 'neighbors_row_splits']
    neighbors_index: shape=(16681,)
    neighbors_row_splits: shape=(32769,)
  neighbors_out: dict with keys ['neighbors_index', 'neighbors_row_splits']
    neighbors_index: shape=(16681,)
    neighbors_row_splits: shape=(3587,)

sample after preprocess in train_one_batch:
  input_geom: shape=(3586, 3)
  latent_queries: shape=(32768, 3)
  output_queries: shape=(3586, 3)
  latent_features: shape=(3586,)
  y: shape=(1, 3586, 1)
  x: type=NoneType
  neighbors_in: dict with keys ['neighbors_index', 'segment_ids', 'counts']
    neighbors_index: shape=(16681,)
    segment_ids: shape=(16681,)
    counts: shape=(32768,)
  neighbors_out: dict with keys ['neighbors_index', 'segment_ids', 'counts']
    neighbors_index: shape=(16681,)
    segment_ids: shape=(16681,)
    counts: shape=(3586,)

# Then train_step executes...
# On first batch: initialization + XLA compilation (~900ms-1.2s)
# On subsequent batches: just inference + backward + optimizer step (~150-410ms)

# Returns: loss (scalar float32, e.g., 0.0234)
```

---

## Critical Points

1. **Lazy Initialization:** `_init()` runs only on the first sample. Parameters and optimizer state are created from real data shapes.

2. **XLA Compilation:** The first call to the JIT function triggers XLA compilation, which takes 300-700ms. Subsequent calls reuse the compiled kernel.

3. **No Stochasticity:** The training step is deterministic (PRNG keys are not involved in training, only in initialization).

4. **Shape Consistency:** All samples must have the same shape after preprocessing for the JIT-compiled code to be reusable.

5. **Memory Updates:** Both `self.params` and `self.opt_state` are updated in-place in the wrapper object. Subsequent calls use the updated parameters.

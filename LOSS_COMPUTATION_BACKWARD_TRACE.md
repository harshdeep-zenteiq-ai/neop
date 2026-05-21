# Loss Function Computation - Complete Backward Trace

**Location:** `scripts/train_gino_carcfd_jax.py` line 305 (inside `_step()` function)

**Call:** `loss = loss_fn(out, y=y)`

**Where:**
- `loss_fn` → `LpLoss` instance (created at trainer_jax.py line 172)
- `out` → Model predictions from forward pass
- `y` → Ground truth pressure data

---

## Input Data

### Model Output (out)
```python
out = ndarray, shape=(batch, 3586, 1), dtype=float32
# Example values (5 random vertices):
# [[[-0.0234],
#   [ 0.1567],
#   [-0.0845],
#   [ 0.2134],
#   [ 0.0923],
#   ...],
#  ...] # batch_size times
```

### Ground Truth (y)
```python
y = ndarray, shape=(1, 1, 3586, 1), dtype=float32
# Example values (5 random vertices):
# [[[[0.0012],
#    [0.1834],
#    [-0.0567],
#    [0.2456],
#    [0.1203],
#    ...]]]
```

**Shapes:** 
- `out` has shape `(batch, 3586, 1)` (e.g., `(1, 3586, 1)`)
- `y` has shape `(batch, 1, 3586, 1)` (e.g., `(1, 1, 3586, 1)`)

---

## Phase 1: Entry to __call__()

**Location:** `neuralop/losses/data_losses_jax.py` lines 207-215

```python
def __call__(self, y_pred, y, **kwargs):
    if kwargs:
        warnings.warn(...)
    return self.rel(y_pred, y)
```

**Called with:**
```python
y_pred = out          # shape (batch, 3586, 1), e.g., (1, 3586, 1)
y = y                 # shape (batch, 1, 3586, 1), e.g., (1, 1, 3586, 1)
kwargs = {}           # empty
```

**Instance variables:**
```python
self.d = 2            # 2D spatial dimensions
self.p = 2            # L2 norm (Euclidean distance)
self.eps = 1e-8       # Numerical stability
self.reduction = "sum" # Sum across batch/channel
```

**Execution:** Skip warnings, call `self.rel(y_pred, y)`

**Returns:** Result from `rel()` method

---

## Phase 2: Relative Lp Loss via rel()

**Location:** `neuralop/losses/data_losses_jax.py` lines 166-205

```python
def rel(self, x, y, take_root=True):
```

**Called with:**
```python
x = y_pred              # shape (batch, 3586, 1), e.g., (1, 3586, 1)
y = y                   # shape (batch, 1, 3586, 1), e.g., (1, 1, 3586, 1)
take_root = True        # Take p-th root
```

---

### Step 1: Extract Shape Prefix (Lines 181-182)

```python
x_shape_prefix = x.shape[:-self.d]
x_flat = jnp.reshape(x, x_shape_prefix + (-1,))
y_flat = jnp.reshape(y, x_shape_prefix + (-1,))
```

**Tracing:**
```python
# self.d = 2 (number of spatial dimensions)
# x.shape = (1, 3586, 1)
# x.shape[:-2] = (1,)  ← Remove last 2 dimensions
x_shape_prefix = (1,)

# Flatten last d=2 dimensions into one
# x: (1, 3586, 1) → (1, 3586*1) = (1, 3586)
# y: (1, 1, 3586, 1) → reshape using x_shape_prefix (1,) with -1 flattens remaining dims
#    (1, 1, 3586, 1) → (1, -1) = (1, 3586)
x_flat = reshape(x, (1, 3586))
y_flat = reshape(y, (1, 3586))
```

**After this step:**
```python
x_flat.shape = (1, 3586)  # All vertices flattened to 1D per batch
y_flat.shape = (1, 3586)

# Example values (batch 0):
x_flat[0] = [-0.0234, 0.1567, -0.0845, 0.2134, 0.0923, ...]
y_flat[0] = [ 0.0012, 0.1834, -0.0567, 0.2456, 0.1203, ...]
```

---

### Step 2: Compute Difference (Line 185)

```python
diff_flat = x_flat - y_flat
```

**Computation:**
```python
# Element-wise subtraction
diff_flat = x_flat - y_flat
# diff_flat[0, i] = x_flat[0, i] - y_flat[0, i]
```

**After this step:**
```python
diff_flat.shape = (1, 3586)

# Example values (batch 0, first 5 vertices):
diff_flat[0] = [
    -0.0234 - 0.0012  = -0.0246,
     0.1567 - 0.1834  = -0.0267,
    -0.0845 - (-0.0567) = -0.0278,
     0.2134 - 0.2456  = -0.0322,
     0.0923 - 0.1203  = -0.0280,
     ... (3581 more)
]
```

---

### Step 3: Compute Difference Norm (Lines 190-192)

**Condition:** `self.p % 2 == 0` (p=2 is even) ✓ TRUE

```python
elif self.p % 2 == 0:  # p=2: even power, no abs() needed
    diff = jnp.sum(diff_flat**self.p, axis=-1, keepdims=False)
    ynorm = jnp.sum(y_flat**self.p, axis=-1, keepdims=False)
```

**Line 191: Compute numerator norm**
```python
# Square each difference and sum over spatial dimension
diff = jnp.sum(diff_flat**2, axis=-1, keepdims=False)

# Computation for batch 0:
# diff[0] = sum of all squared differences
# = (-0.0246)² + (-0.0267)² + (-0.0278)² + (-0.0322)² + (-0.0280)² + ...
# = 0.000605 + 0.000713 + 0.000773 + 0.001037 + 0.000784 + ...
# = 2.3456 (example value, actual varies per batch)
```

**After this step:**
```python
diff.shape = (1,)  # Reduced from (1, 3586) → (1,)
# Example:
diff = [2.3456]  # One value per batch sample
```

**Line 192: Compute denominator norm (ground truth L2 norm)**
```python
# Square each ground truth value and sum over spatial dimension
ynorm = jnp.sum(y_flat**2, axis=-1, keepdims=False)

# Computation for batch 0:
# ynorm[0] = sum of all squared ground truth values
# = (0.0012)² + (0.1834)² + (-0.0567)² + (0.2456)² + (0.1203)² + ...
# = 0.0000014 + 0.0336 + 0.00322 + 0.0603 + 0.01447 + ...
# = 5.8234 (example value)
```

**After this step:**
```python
ynorm.shape = (1,)
# Example:
ynorm = [5.8234]  # One value per batch sample
```

---

### Step 4: Take Root and Divide (Lines 197-198)

```python
if take_root and self.p != 1:  # take_root=True, p=2 (not 1)
    diff = (diff ** (1.0 / self.p)) / (ynorm ** (1.0 / self.p) + self.eps)
```

**Computation:**
```python
# self.p = 2, so 1.0/self.p = 0.5 (take square root)
# For batch 0:
diff_root = diff[0] ** 0.5 = (2.3456) ** 0.5 = 1.5316
ynorm_root = ynorm[0] ** 0.5 = (5.8234) ** 0.5 = 2.4132
eps = 1e-8

# Relative norm: ||predicted - truth|| / ||truth||
rel_loss = diff_root / (ynorm_root + eps)
          = 1.5316 / (2.4132 + 1e-8)
          = 1.5316 / 2.4132
          = 0.6345
```

**After this step:**
```python
diff.shape = (1,)
# Each element is now a relative L2 error per batch sample
# Example:
diff = [0.6345]
```

---

### Step 5: Reduce Across Batch (Line 202)

```python
diff = self.reduce_all(diff)
```

**Called method:** `reduce_all()` (lines 101-115)

```python
def reduce_all(self, x):
    if self.reduction == "sum":
        x = jnp.sum(x)
    else:
        x = jnp.mean(x)
    return x
```

**Execution:** `self.reduction = "sum"`

```python
# Sum all relative errors across the batch
diff = jnp.sum(diff)
     = sum([0.6345])
     = 0.6345
```

**After this step:**
```python
diff.shape = ()  # Scalar
diff = 0.6345
```

---

### Step 6: Squeeze (Line 203)

```python
diff = jnp.squeeze(diff)
```

**Effect:** Remove all dimensions of size 1 (none exist, so no change)

```python
diff.shape = ()  # Already scalar, no change
diff = 6.8234
```

---

### Step 7: Return from rel()

```python
return diff  # Line 205
```

**Returns:** Scalar float32 value: `0.6345` (for single batch)

---

## Phase 3: Return from __call__()

**Location:** `neuralop/losses/data_losses_jax.py` line 215

```python
return self.rel(y_pred, y)  # Returns 0.6345 (for single batch)
```

**Returns to caller:** `loss = 0.6345` (for single batch)

---

## Return to _step() in JIT Function

**Location:** `scripts/train_gino_carcfd_jax.py` line 305

```python
loss, grads = jax.value_and_grad(forward_loss)(params)
```

**After loss computation:**
```python
loss = 0.6345  # Scalar float32 (for single batch)
```

---

## Complete Data Flow Diagram

```
Model Output (out)                    Ground Truth (y)
  ↓                                        ↓
  (1, 3586, 1)                        (1, 1, 3586, 1)
  ↓                                        ↓
  ┌─────────────────────────────────────────┐
  │  __call__(y_pred=out, y=y)              │
  └─────────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────────┐
  │  rel(x=y_pred, y=y)                     │
  ├─────────────────────────────────────────┤
  │ Step 1: Reshape last d=2 dimensions     │
  │   (1, 3586, 1) → (1, 3586)             │
  │   (1, 1, 3586, 1) → (1, 3586)          │
  └─────────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────────┐
  │ Step 2: Compute difference              │
  │   diff_flat = x_flat - y_flat           │
  │   shape: (1, 3586)                      │
  └─────────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────────┐
  │ Step 3: Compute norms (p=2, even)       │
  │   diff = sum(diff_flat**2, axis=-1)     │
  │   ynorm = sum(y_flat**2, axis=-1)       │
  │   shape: (1,)                           │
  └─────────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────────┐
  │ Step 4: Relative norm (take root)       │
  │   diff = sqrt(diff) / (sqrt(ynorm)+eps) │
  │   shape: (1,)                           │
  ├─────────────────────────────────────────┤
  │ Formula: ||x-y||_2 / (||y||_2 + 1e-8)  │
  └─────────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────────┐
  │ Step 5: Reduce (sum across batch)       │
  │   loss = sum(diff)                      │
  │   shape: () [scalar]                    │
  └─────────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────────┐
  │ Step 6: Squeeze                         │
  │   loss = jnp.squeeze(loss)              │
  │   shape: () [scalar, no change]         │
  └─────────────────────────────────────────┘
  ↓
loss = 0.6345 (scalar float32, for single batch)
```

---

## Mathematical Formula

The computation implements the **Relative L2 Loss**:

```
         ||ŷ - y||_2
loss = ─────────────────
        ||y||_2 + eps

Where:
  ŷ = model predictions (shape: batch × n_vertices)
  y = ground truth (shape: batch × n_vertices)
  eps = 1e-8 (prevents division by zero)
  ||·||_2 = L2 norm (Euclidean distance)

Expanded:
                    √(Σᵢ(ŷᵢ - yᵢ)²)
loss = ───────────────────────────────────
       √(Σᵢ yᵢ²) + 1e-8

Batch sum:
       batch
loss = Σ ────────────────────────────────
      b  √(Σᵢ(ŷᵇᵢ - yᵇᵢ)²)
         ──────────────────────
         √(Σᵢ(yᵇᵢ)²) + 1e-8
```

---

## Concrete Example with Real Values

### Sample Input
```python
# Batch size: 2, Vertices: 5 (simplified example)

out = jnp.array([
    [[-0.024],  # Batch 0, Vertex 0, Pressure
     [ 0.157],  # Batch 0, Vertex 1
     [-0.085],  # Batch 0, Vertex 2
     [ 0.213],  # Batch 0, Vertex 3
     [ 0.092]], # Batch 0, Vertex 4
    [[ 0.034],  # Batch 1, Vertex 0
     [ 0.162],  # Batch 1, Vertex 1
     [-0.067],  # Batch 1, Vertex 2
     [ 0.198],  # Batch 1, Vertex 3
     [ 0.087]]  # Batch 1, Vertex 4
])
# shape: (2, 5, 1)

y = jnp.array([
    [[ 0.001],  # Ground truth Batch 0
     [ 0.183],
     [-0.057],
     [ 0.246],
     [ 0.120]],
    [[ 0.021],  # Ground truth Batch 1
     [ 0.145],
     [-0.078],
     [ 0.215],
     [ 0.095]]
])
# shape: (2, 5, 1)
```

### Step-by-Step Computation

**Reshape:**
```python
x_flat = [[-0.024, 0.157, -0.085, 0.213, 0.092],
          [ 0.034, 0.162, -0.067, 0.198, 0.087]]

y_flat = [[ 0.001, 0.183, -0.057, 0.246, 0.120],
          [ 0.021, 0.145, -0.078, 0.215, 0.095]]
```

**Difference:**
```python
diff_flat = [[-0.025, -0.026, -0.028, -0.033, -0.028],
             [ 0.013,  0.017,  0.011, -0.017, -0.008]]
```

**Difference Norm (squared):**
```python
diff²_flat = [[0.000625, 0.000676, 0.000784, 0.001089, 0.000784],
              [0.000169, 0.000289, 0.000121, 0.000289, 0.000064]]

diff = [sum of row 0, sum of row 1]
     = [0.003958, 0.000932]
```

**Truth Norm (squared):**
```python
y²_flat = [[0.000001, 0.033489, 0.003249, 0.060516, 0.014400],
           [0.000441, 0.021025, 0.006084, 0.046225, 0.009025]]

ynorm = [sum of row 0, sum of row 1]
      = [0.111655, 0.082800]
```

**Take roots:**
```python
√diff = [√0.003958, √0.000932] = [0.0629, 0.0305]
√ynorm = [√0.111655, √0.082800] = [0.3341, 0.2878]
```

**Relative error:**
```python
rel = [0.0629 / 0.3341, 0.0305 / 0.2878]
    = [0.1882, 0.1060]
```

**Sum across batch:**
```python
loss = 0.1882 + 0.1060 = 0.2942
```

### Return
```python
loss = 0.2942  # Scalar float32
```

---

## Parameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `d` | 2 | Number of spatial dimensions to flatten |
| `p` | 2 | Lp norm order (L2 = Euclidean) |
| `measure` | 1.0 | Domain measure for quadrature weights |
| `reduction` | "sum" | Sum relative errors across batch |
| `eps` | 1e-8 | Prevent division by zero |

---

## Key Points

1. **Relative Loss:** Normalizes by ground truth magnitude, making it scale-invariant
2. **L2 Norm:** Euclidean distance between prediction and ground truth
3. **Batch Summation:** Final loss is sum of all relative errors
4. **Numerical Stability:** `eps=1e-8` prevents division by zero when ground truth is near zero
5. **Differentiable:** All operations are JAX-compatible, allowing gradient computation

---

## Gradient Computation (Used Later)

This loss value is then passed to `jax.value_and_grad(forward_loss)` to compute gradients:

```python
loss, grads = jax.value_and_grad(forward_loss)(params)
```

JAX automatically computes:
```
grads['params']['GINOModel_0'][...] = ∂loss / ∂params
```

These gradients flow backward through:
1. `reduce_all()` → Distributes loss gradient to each batch element
2. Norm computation → Applies chain rule for square root and division
3. Difference → Distributes to predictions and truth
4. Model forward pass → Via JAX autodiff through GINO layers

The backward pass is automatic and handled entirely by `jax.value_and_grad()`.

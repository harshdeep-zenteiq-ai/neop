# GINOCFDDataProcessor - Detailed Analysis

## Class Overview

**Location:** `scripts/train_gino_carcfd_jax.py` (Lines 468-585)

The `GINOCFDDataProcessor` is a **data pipeline adapter** that transforms raw CFD mesh data into the exact format required by the GINO model. It inherits from `DataProcessor` and acts as middleware between the dataset and the model during training and inference.

### Purpose
Convert raw CFD car-pressure dataset samples into GINO-compatible inputs with proper shape handling, normalization, and neighbor data preprocessing.

---

## Method Breakdown

### 1. `__init__(self, normalizer, device="cpu")`

**What it does:** Initialize the data processor with normalization and device settings.

**Parameters:**
| Parameter | Type | Purpose |
|-----------|------|---------|
| `normalizer` | callable | Handles normalization/denormalization of pressure values |
| `device` | str | Device for computation ("cpu" or GPU identifier) |

**Attributes created:**
```python
self.normalizer = normalizer          # Normalization function
self.device = device                  # Target device
self.model = None                     # Placeholder for model reference (set later via wrap())
self.training = True                  # Mode flag (set during __call__)
```

**Key insight:** The processor doesn't initialize anything model-specific yet—parameters and shapes are unknown until data arrives.

---

### 2. `preprocess(self, sample)`

**What it does:** Transform raw CFD sample into GINO input format.

**Input Structure (from dataset):**
```python
sample = {
    "vertices": ndarray of shape (n_vertices, 3),        # 3D mesh coordinates
    "query_points": ndarray of shape (n_query_points, 3), # Grid points for latent space
    "distance": ndarray of shape (n_vertices,),           # SDF distance field
    "press": ndarray of shape (n_vertices,),              # Pressure values (ground truth)
    "neighbors_in": dict with sparse connectivity,        # Precomputed graph edges
    "neighbors_out": dict with sparse connectivity
}
```

**Processing Steps:**

**Step 1: Extract and Convert to NumPy**
```python
in_p = np.asarray(sample["vertices"])                 # Input geometry
latent_queries = np.asarray(sample["query_points"])   # Latent query points
out_p = np.asarray(sample["vertices"])                # Output geometry (same as input)
f = np.asarray(sample["distance"])                    # Features (SDF distance)
truth = np.asarray(sample["press"])                   # Ground truth pressure
```

**Step 2: Reshape Ground Truth to (batch, n_vertices, 1)**

Three cases handled:
- **Case 1: ndim == 2** (single sample, shape (n_vertices,))
  ```python
  truth = np.expand_dims(np.expand_dims(truth, axis=0), axis=-1)
  # Result: (1, n_vertices, 1)
  ```

- **Case 2: ndim == 3** (batch provided, shape (batch, n_vertices))
  ```python
  truth = np.expand_dims(truth, axis=-1)
  # Result: (batch, n_vertices, 1)
  ```

- **Case 3: Already correct shape (batch, n_vertices, 1)**
  - No change needed

**Step 3: Align Output Geometry to Ground Truth**

This handles mesh size mismatch:
```python
output_vertices = truth.shape[0] if truth.ndim > 1 else 1
if out_p.shape[0] > output_vertices:
    out_p = out_p[:output_vertices, :]
```

**Why?** The output mesh might have more vertices than pressure data points. Truncate to match.

**Step 4: Create Output Dictionary**
```python
batch_dict = {
    "input_geom": in_p,              # (n_vertices, 3) → mesh coordinates as input
    "latent_queries": latent_queries, # (n_query_points, 3) → grid for latent encoding
    "output_queries": out_p,          # (n_output_vertices, 3) → where to predict
    "latent_features": f,             # (n_vertices,) → SDF distance as features
    "y": truth,                       # (batch, n_vertices, 1) → target pressure
    "x": None,                        # Not used in this implementation
}
```

**Step 5: Convert Neighbor Data to NumPy**

Neighbor indices are precomputed and stored as sparse representations:
```python
for key in ("neighbors_in", "neighbors_out"):
    if key in sample and sample[key] is not None:
        batch_dict[key] = {
            k: np.asarray(v)
            for k, v in sample[key].items()
        }
```

After `pad_neighbor_data()` preprocessing (earlier in pipeline), structure is:
```python
neighbors_in = {
    "neighbors_index": ndarray,   # Edge indices (padded)
    "segment_ids": ndarray,       # Which node each edge belongs to
    "counts": ndarray             # Edge count per node
}
neighbors_out = {
    "neighbors_index": ndarray,
    "segment_ids": ndarray,
    "counts": ndarray
}
```

**Output Structure:**
```python
return batch_dict  # 9 keys total: 6 tensor inputs + 3 dicts (neighbor info + y + x)
```

---

### 3. `postprocess(self, out, sample)`

**What it does:** Denormalize predictions when in evaluation mode.

**Execution:**
```python
if not self.training:  # Only apply when evaluating (not training)
    out = self.normalizer.inverse_transform(out)
    y = jnp.asarray(sample["y"])
    if y.ndim > 1:
        y = jnp.squeeze(y, axis=0)  # Remove batch dim if present
    y = self.normalizer.inverse_transform(y)
    sample["y"] = y
```

**Why?** During preprocessing, the normalizer standardizes pressure values (zero mean, unit variance). After inference, undo this transformation to get real pressure values in original scale.

**Returns:** `(out, sample)` with denormalized pressure values

---

### 4. `to(self, device)`

**What it does:** Move processor and its components to a specific device.

```python
self.device = device
if hasattr(self.normalizer, 'to'):
    self.normalizer = self.normalizer.to(device)  # Move normalizer if possible
return self
```

**Use case:** When switching from CPU to GPU computation (or vice versa)

---

### 5. `wrap(self, model)`

**What it does:** Register the model with this processor for the forward pass.

```python
self.model = model  # Store reference to the GINO model
return self         # Enable method chaining
```

**Timing:** Called after model is instantiated but before training starts

---

### 6. `__call__(self, sample, training=True)`

**What it does:** Complete forward pass: preprocessing → model inference → postprocessing

**Execution Flow:**

**Step 1: Set Mode**
```python
self.training = training  # Flag for postprocessing behavior
```

**Step 2: Preprocess**
```python
sample = self.preprocess(sample)
# Transforms sample into GINO input format with 9 keys
```

**Step 3: Filter to Model Keys**
```python
model_keys = {
    'input_geom', 'latent_queries', 'output_queries',
    'x', 'latent_features', 'ada_in', 'neighbors_in', 'neighbors_out', 'y'
}
model_input = {k: v for k, v in sample.items() if k in model_keys}
```

This subset ensures only expected keys are passed to the model. Note: some keys like `'ada_in'` may not be in sample; they're just included in the filter set.

**Step 4: Model Inference**
```python
out = self.model(**model_input)
# GINO model processes the input and returns predictions
# out shape: (batch, n_output_vertices, 1) → predicted pressure
```

**Step 5: Postprocess**
```python
out, sample = self.postprocess(out, sample)
# Denormalize if not training
```

**Step 6: Return**
```python
return out, sample
```

**Returns:** `(predictions, updated_sample)`

---

## Data Flow Summary

### Training Mode (`training=True`)
```
Raw Sample (from dataset)
    ↓
preprocess()
  - Extract vertices, queries, distance, pressure
  - Reshape pressure to (batch, n_vertices, 1)
  - Align output geometry to pressure shape
  - Pass neighbors as-is
    ↓
Filter to model keys
    ↓
self.model(**model_input)
  - GINO performs inference
    ↓
postprocess() [SKIPPED - not training]
    ↓
Return (predictions, sample)
```

### Evaluation Mode (`training=False`)
```
Raw Sample (from dataset)
    ↓
preprocess() [same as training]
    ↓
Filter to model keys
    ↓
self.model(**model_input)
  - GINO performs inference
    ↓
postprocess() [APPLIED]
  - Denormalize model output
  - Denormalize ground truth y
    ↓
Return (denormalized predictions, updated sample)
```

---

## Key Design Decisions

### 1. **Why Keep Neighbors as Dicts?**
After `pad_neighbor_data()` conversion, neighbors are no longer simple 1D arrays. They're dicts with `neighbors_index`, `segment_ids`, and `counts`. The processor preserves this structure for the model to use during aggregation operations.

### 2. **Why Use NumPy Not JAX?**
Preprocessing happens on CPU before data reaches the model. JAX arrays would require unnecessary compilation. NumPy is faster for data transformation.

### 3. **Why Denormalize Only in Eval?**
- **Training:** Model learns on normalized data (better numerical stability)
- **Evaluation:** Need real pressure values for metrics and visualization

### 4. **Why Filter to Model Keys?**
The processor handles many intermediate computations. Filtering ensures only relevant inputs reach the model, preventing shape mismatches or unexpected arguments.

### 5. **Why Align Output Geometry to Ground Truth?**
CFD meshes can have many vertices. Only the vertices with corresponding pressure measurements matter for supervision. Truncating aligns the two.

---

## Integration with Training Pipeline

### In Context of `train_gino_carcfd_jax.py`:

```python
# 1. Create processor (line ~350)
processor = GINOCFDDataProcessor(normalizer=normalizer, device="cpu")

# 2. Wrap with model (line ~360)
processor = processor.wrap(model)

# 3. During training loop:
for sample in train_dataloader:
    # Inside trainer.fit() or train_step():
    output, sample = processor(sample, training=True)
    loss = loss_fn(output, sample["y"])
    # Backward pass, optimizer step...

# 4. During evaluation:
for sample in val_dataloader:
    output, sample = processor(sample, training=False)
    # output is now denormalized pressure predictions
    # sample["y"] is denormalized ground truth
    metrics = compute_metrics(output, sample["y"])
```

---

## Summary Table

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `__init__` | normalizer, device | self | Initialize processor state |
| `preprocess` | raw sample dict | GINO input dict | Transform mesh data to model format |
| `postprocess` | model output, sample | denormalized values | Undo normalization when evaluating |
| `to` | device str | self | Move to device |
| `wrap` | model instance | self | Register model |
| `__call__` | sample, training flag | (output, sample) | Complete pipeline: preprocess → model → postprocess |

---

## Memory and Compute

**Size Profile:**
- Normalizer: ~1-10 KB (depends on implementation)
- Device reference: ~100 bytes
- Model reference: pointer only (~8 bytes)

**Computation Time (per sample):**
- Preprocessing: ~1-5 ms (NumPy operations)
- Model forward: ~10-100 ms (depends on GINO size)
- Postprocessing: ~1-5 ms (denormalization)
- **Total: ~15-110 ms per sample**

**Note:** No parameters or gradients stored in processor—it's stateless except for the model and normalizer references.

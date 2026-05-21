# GINO CAR CFD Training Script - Complete Data Flow

## Overview
This document traces the complete data transformation pipeline from raw CFD data to model parameter updates in `train_gino_carcfd_jax.py`.

---

## 1. Configuration & Setup

### Configuration Values (from `config/gino_carcfd_config.py`)
```
sdf_query_resolution: 32        # Creates 32³ = 32,768 query points
n_train: 250                     # Training samples
n_test: 50                       # Test samples
batch_size: 1                    # Per-sample processing
learning_rate: 1e-3
weight_decay: 1e-4
n_epochs: 301
```

### JAX Environment Setup
```python
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'       # Async memory allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'      # Don't reserve 90% VRAM upfront
os.environ["JAX_ENABLE_X64"] = "False"                     # Use float32 (not float64)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"     # Platform-specific allocator
```

---

## 2. Data Loading Phase

### Step 2.1: Load Dataset from Disk
**Input**: Raw CFD mesh files from disk
```
Location: ~/data/car-pressure-data/processed-car-pressure-data
```

**Output**: `CarCFDDatasetjax` object with:
- `train_data.data_list`: List of 250 training samples (on **CPU RAM**)
- `test_data.data_list`: List of 50 test samples (on **CPU RAM**)

### Step 2.2: Sample Dictionary Structure (Before Preprocessing)
Each sample in `data_list` is a dictionary:

```python
sample = {
    'vertices': ndarray,                    # Shape: (n_verts, 3)
                                           # Typical: (10000, 3)
                                           # Size: 10,000 * 3 * 4 bytes = 120 KB
    
    'query_points': ndarray,               # Shape: (32, 32, 32, 3)
                                           # Size: 32,768 * 3 * 4 bytes = ~390 KB
    
    'distance': ndarray,                   # Shape: (32, 32, 32, 1)
                                           # Size: 32,768 * 1 * 4 bytes = ~130 KB
    
    'press': ndarray,                      # Shape: (n_pressure_points,)
                                           # Typical: (3586,)
                                           # Size: 3,586 * 4 bytes = ~14 KB
    
    'neighbors_in': dict with:
        'neighbors_index': ndarray,        # Shape: (n_edges_in,)
                                           # Variable length, depends on graph
        'neighbors_row_splits': ndarray,   # Shape: (n_query_points + 1,)
                                           # Row pointers for CSR format
    
    'neighbors_out': dict with:
        'neighbors_index': ndarray,        # Shape: (n_edges_out,)
        'neighbors_row_splits': ndarray,   # Shape: (n_output_points + 1,)
}
```

**Total per sample**: ~650 KB (before preprocessing)

---

## 3. Neighbor Precomputation Phase

### Step 3.1: Precompute Neighbors
**Function**: `model.precompute_neighbors()` (called at lines 366-369)

**Process**:
1. For each sample in training data:
   - Build k-nearest neighbor graph for input query points
   - Build k-nearest neighbor graph for output vertices
2. Store neighbor indices and row splits in each sample's dict

**Output**: Updates `sample['neighbors_in']` and `sample['neighbors_out']` with graph connectivity

---

## 4. Neighbor Data Padding Phase

### Step 4.1: Pad Neighbor Arrays
**Function**: `pad_neighbor_data()` (lines 118-177)

**Problem**: Variable-length neighbor arrays → incompatible with `jax.jit`

**Solution**: Pad all samples to maximum edge count

**Detailed Process**:

For each sample and each of `neighbors_in` / `neighbors_out`:

```python
# Input
neighbors_index: shape (n_real_edges,)           # Variable length
neighbors_row_splits: shape (n_output_nodes + 1,) # Row pointers

# Processing
1. Compute segment_ids from row_splits:
   segment_ids: shape (n_real_edges,)
   # Maps each edge to its output node

2. Find max edges across all samples:
   max_edges = max(len(neighbors_index) for all samples)
   # Typical: max_in ~500, max_out ~1000 edges

3. Pad arrays with zeros/dummy values:
   neighbors_index_padded: shape (max_edges,)    # Pad with 0s
   segment_ids_padded: shape (max_edges,)        # Pad with dummy segment ID

4. Store counts (real edges per node):
   counts: shape (n_output_nodes,)               # Number of real neighbors
```

**Output Structure** (stored as **numpy arrays on CPU**):
```python
sample['neighbors_in'] = {
    'neighbors_index': ndarray shape (max_in_edges,)     # ~500
    'segment_ids': ndarray shape (max_in_edges,)         # ~500
    'counts': ndarray shape (n_query_points,)            # 32,768
}

sample['neighbors_out'] = {
    'neighbors_index': ndarray shape (max_out_edges,)    # ~1000
    'segment_ids': ndarray shape (max_out_edges,)        # ~1000
    'counts': ndarray shape (n_output_points,)           # 3,586
}
```

**Memory Impact**: ~4 KB per sample (small compared to vertex data)

---

## 5. Data Loading & Batching Phase

### Step 5.1: Create Data Loaders
**Code** (lines 382-383):
```python
train_loader_jax = SimpleDataLoader(
    data_module_jax.train_loader(batch_size=1, shuffle=True)
)
test_loader_jax = SimpleDataLoader(
    data_module_jax.test_loader(batch_size=1, shuffle=False)
)
```

**Batch Structure**:
```python
batch = {
    'vertices': ndarray shape (1, 10000, 3),           # [batch=1, verts, 3D]
    'query_points': ndarray shape (1, 32, 32, 32, 3),  # [batch=1, 32³, 3D]
    'distance': ndarray shape (1, 32, 32, 32),         # [batch=1, 32³]
    'press': ndarray shape (1, 3586),                   # [batch=1, pressure points]
    'neighbors_in': dict (as above),                    # Graph connectivity
    'neighbors_out': dict (as above),
}
```

**Location**: Data remains on **CPU RAM**

---

## 6. Data Preprocessing Phase

### Step 6.1: GINOCFDDataProcessor.preprocess()
**Called by**: Trainer's `train_one_batch()` at each iteration
**Location**: Lines 503-563

**Input Shapes** (batch_size=1):
```
vertices:       (1, 10000, 3)
query_points:   (1, 32, 32, 32, 3)
distance:       (1, 32, 32, 32)
press:          (1, 3586)
neighbors_*:    As defined above
```

**Processing Steps**:

#### Step 6.1a: Convert to Numpy Arrays (Keep on CPU)
```python
in_p = np.asarray(sample["vertices"])              # (1, 10000, 3) → float32 numpy
latent_queries = np.asarray(sample["query_points"]) # (1, 32, 32, 32, 3) → float32 numpy
out_p = np.asarray(sample["vertices"])              # (1, 10000, 3) → float32 numpy
f = np.asarray(sample["distance"])                  # (1, 32, 32, 32) → float32 numpy
truth = np.asarray(sample["press"])                 # (1, 3586) → float32 numpy
```

#### Step 6.1b: Reshape Truth Data
```python
if truth.ndim == 2:
    # Shape: (1, 3586) → (1, 3586, 1)
    truth = np.expand_dims(np.expand_dims(truth, axis=0), axis=-1)
    # Final shape: (1, 1, 3586, 1)
```

#### Step 6.1c: Truncate Output Vertices
```python
output_vertices = truth.shape[0]  # Get from truth shape
if out_p.shape[0] > output_vertices:
    out_p = out_p[:output_vertices, :]
    # Typical: (10000, 3) → (3586, 3)
```

#### Step 6.1d: Convert Neighbor Data
```python
batch_dict = {
    'input_geom': in_p,                    # (1, 10000, 3) numpy
    'latent_queries': latent_queries,      # (1, 32, 32, 32, 3) numpy
    'output_queries': out_p,               # (1, 3586, 3) numpy
    'latent_features': f,                  # (1, 32, 32, 32) numpy
    'y': truth,                            # (1, 1, 3586, 1) numpy
    'x': None,
    'neighbors_in': {
        'neighbors_index': np.asarray(...), # (max_in_edges,) numpy
        'segment_ids': np.asarray(...),     # (max_in_edges,) numpy
        'counts': np.asarray(...),          # (n_query_points,) numpy
    },
    'neighbors_out': {
        'neighbors_index': np.asarray(...), # (max_out_edges,) numpy
        'segment_ids': np.asarray(...),     # (max_out_edges,) numpy
        'counts': np.asarray(...),          # (n_output_points,) numpy
    }
}
```

**Output Location**: Data still on **CPU RAM** as numpy arrays

---

## 7. Model Training Step Phase

### Step 7.1: Extract Model Inputs
**Location**: Trainer's `train_one_batch()` + FlaxModelWrapper.train_step()

```python
model_inputs = {
    k: v for k, v in batch_dict.items() 
    if k in ['input_geom', 'latent_queries', 'output_queries',
             'x', 'latent_features', 'ada_in', 'neighbors_in', 'neighbors_out']
}

# Result:
model_inputs = {
    'input_geom': (1, 10000, 3) numpy,
    'latent_queries': (1, 32, 32, 32, 3) numpy,
    'output_queries': (1, 3586, 3) numpy,
    'latent_features': (1, 32, 32, 32) numpy,
    'neighbors_in': {...} numpy,
    'neighbors_out': {...} numpy,
}
```

### Step 7.2: JIT Compilation & GPU Transfer
**Location**: FlaxModelWrapper._jit_fn (lines 320-329)

**JAX JIT Boundary** ← **DATA TRANSFERS FROM CPU TO GPU HERE**

```python
@jax.jit
def _step(params, opt_state, model_inputs, y):
    # numpy arrays auto-converted to JAX device arrays (on GPU)
    
    def forward_loss(p):
        out = module.apply(p, **model_inputs)  # Forward pass
        return loss_fn(out, y=y)               # Loss computation
    
    loss, grads = jax.value_and_grad(forward_loss)(params)  # Backward pass
    updates, new_opt = tx.update(grads, opt_state, params)  # Optimizer update
    return loss, optax.apply_updates(params, updates), new_opt
```

---

## 8. Model Forward Pass

### Step 8.1: GINO Model Forward Pass
**Input**:
```
input_geom: (1, 10000, 3)           # Input vertex coordinates
latent_queries: (1, 32, 32, 32, 3)  # Query points in latent space (32³=32,768 points)
output_queries: (1, 3586, 3)        # Output vertex coordinates
latent_features: (1, 32, 32, 32)    # Distance/SDF features
neighbors_in: {...}                 # Input graph connectivity
neighbors_out: {...}                # Output graph connectivity
```

### Step 8.2: Model Architecture Flow
The GINO model applies transformers on graphs:

```
1. Input Encoding:
   input_geom (1, 10000, 3) → Embed to latent features
   
2. Query Point Processing:
   latent_queries (1, 32, 32, 32, 3) → Flatten to (32768, 3)
   
3. Graph Neural Operations (using precomputed neighbors):
   Apply attention/aggregation on neighbors_in graph
   → (32768, hidden_dim)
   
4. Output Generation:
   latent space (32768, hidden_dim) → Decode to output
   
5. Output Vertices Processing:
   Aggregate predictions at output_queries (1, 3586, 3)
   
6. Final Prediction:
   Output shape: (1, 3586, 1) or (1, 3586)
   Predicts pressure at 3,586 output vertices
```

**Output Shape**: 
```
out: (1, 3586, 1)  # Predicted pressure values
```

---

## 9. Loss Computation

### Step 9.1: L2 Loss Calculation
**Function**: `LpLoss(d=2, p=2)` (line 472)

```python
predicted = out                    # Shape: (1, 3586, 1)
target = y                         # Shape: (1, 3586, 1)

loss = LpLoss(predicted, target)   # L2 norm: ||pred - target||²

# Result
loss: scalar (float32)             # Single loss value
```

---

## 10. Backward Pass & Optimization

### Step 10.1: Gradient Computation
**Inside JIT-compiled _step function**:

```python
loss, grads = jax.value_and_grad(forward_loss)(params)
```

**Process**:
1. Compute loss (forward pass)
2. Backpropagate to compute gradients for all parameters
3. Gradients have same shape as parameters

**Gradient Shapes** (examples):
```
grads = {
    'layer_1/dense': {
        'kernel': gradient array,    # Same shape as weight matrix
        'bias': gradient array,      # Same shape as bias vector
    },
    'layer_2/dense': {...},
    ...
}
```

### Step 10.2: Optimizer State Update
**Optimizer**: AdamW (momentum buffers for each parameter)

```python
# Adam optimizer maintains:
opt_state = {
    'counts': int,                   # Step count
    'mu': {...},                     # First moment (momentum)
    'nu': {...},                     # Second moment (variance)
}

# Update step
updates, new_opt = tx.update(grads, opt_state, params)
```

**Size**: opt_state ≈ 2× params size (momentum + variance buffers)

### Step 10.3: Parameter Update
```python
new_params = optax.apply_updates(params, updates)
# new_params = params + updates (with learning rate scaling)
```

---

## 11. Complete Training Loop

### Step 11.1: Per-Epoch Training Loop

```
For each epoch (0 to 300):
    For each batch in train_loader (250 batches for 250 samples):
        1. Load batch from CPU RAM
        2. Preprocess: numpy arrays
        3. JIT call: transfer to GPU
        4. Forward pass: compute prediction
        5. Loss: compute L2 loss
        6. Backward: compute gradients
        7. Update: apply optimizer step
        8. Return loss
    
    Evaluate on test set (50 samples)
    Update learning rate scheduler
    Log metrics
```

---

## 12. Memory Usage Breakdown (Per Sample)

### GPU Memory (During JIT Forward+Backward):
```
Model Parameters:        ~50-200 MB (depends on model size)
Optimizer State (Adam):  ~100-400 MB (2× params)
Input Tensors:           ~1-2 MB
  - input_geom: 120 KB
  - latent_queries: 390 KB
  - distance: 130 KB
  - neighbors: 10 KB
Intermediate Activations: ~500 MB - 1 GB (model-dependent)
Output & Gradients:      ~50-100 MB
```

**Total**: 750 MB - 1.7 GB per sample

### CPU Memory (Data Storage):
```
All 250 training samples: ~150 MB
  - vertices: 120 KB × 250 = 30 MB
  - query_points: 390 KB × 250 = 97 MB
  - distance: 130 KB × 250 = 32 MB
  - press: 14 KB × 250 = 3.5 MB
```

---

## 13. Current Memory Issue Analysis

### Problem
```
jaxlib._jax.XlaRuntimeError: RESOURCE_EXHAUSTED
Failed to allocate 1.29GiB on device ordinal 0
```

### Root Cause
The JIT compilation of the forward+backward pass creates **large intermediate tensors** during:
1. Forward pass through graph transformers
2. Gradient computation (backpropagation)
3. Storing activation intermediates for autograd

With 32³ = 32,768 query points, each going through attention/aggregation layers, the intermediate feature maps can become very large.

### Why Chunked Scanning Helped (Before)
The `preprocess_and_stack` + `train_epoch_scan` approach:
- Kept data on CPU in chunks
- Transferred one chunk (10 samples) to GPU at a time
- Reduced instantaneous memory pressure
- But introduced OOM when constructing the full stacked tensor in memory

### Why Per-Sample Training Still Fails
- Even a single sample with 32³ query points creates large intermediate tensors
- The model architecture might not be optimized for JAX's memory profile
- JAX doesn't do as much memory optimization as PyTorch

---

## 14. Recommended Solutions

### Option A: Reduce Query Resolution
```python
# In config/gino_carcfd_config.py
sdf_query_resolution: int = 16  # Reduce from 32
# This reduces query points from 32,768 to 4,096 (8× less memory)
```

### Option B: Gradient Checkpointing (if available)
```python
# In model definition, enable gradient checkpointing
# This trades compute for memory
```

### Option C: Mixed Precision Training
```python
jax.config.update("jax_default_matmul_precision", "medium")
# Use lower precision for matrix multiplies (saves memory)
```

### Option D: Return to Chunked Scanning (with fixes)
- Keep the `preprocess_and_stack` approach
- But reduce chunk size to 5 instead of 10
- Avoid stacking all samples at once

---

## Summary Table

| Phase | Input Shape | Output Shape | Location | Memory Type |
|-------|-------------|--------------|----------|------------|
| Load | Disk files | batch dict | RAM | CPU |
| Precompute Neighbors | Vertices | Graph indices | RAM | CPU |
| Pad Neighbors | Variable edges | Max edges | RAM | CPU |
| Batch Load | Sample dict | Batch dict | RAM | CPU |
| Preprocess | Numpy arrays | Numpy arrays | RAM | CPU |
| **JIT Transfer** | Numpy → JAX | JAX arrays | **GPU** | **GPU** |
| Forward Pass | (1,32,32,32,3) + graph | (1,3586,1) | GPU | GPU |
| Loss | Pred + Target | Scalar | GPU | GPU |
| Backward | Loss | Grads | GPU | GPU |
| Update | Grads + Params | New Params | GPU | GPU |


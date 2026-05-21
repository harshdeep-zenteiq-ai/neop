# GINO Model Training Flow - Complete Mental Model

> A comprehensive guide to understanding `train_gino_carcfd.py` and how the GINO architecture works in the context of this training script.

---

## 📋 Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Script Execution Flow (Step-by-Step)](#script-execution-flow)
3. [Key Files and Their Purposes](#key-files)
4. [GINO Model Architecture Explained](#gino-architecture)
5. [Data Flow Through the System](#data-flow)
6. [Training Loop Mechanics](#training-loop)
7. [Quick Reference Guide](#quick-reference)

---

## High-Level Overview

### What is the Script Doing?

The `train_gino_carcfd.py` script trains a **GINO (Geometry-Informed Neural Operator)** model to predict **pressure fields** on car surfaces from **geometric mesh data** using CFD (Computational Fluid Dynamics) data.

**Input to Model:** Car mesh geometry (vertices) + regular query grid + distance field
**Output from Model:** Predicted pressure at each vertex
**Purpose:** Learn a function that maps geometry → pressure

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING SCRIPT WORKFLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Load Configuration        (gino_carcfd_config.py)           │
│  2. Setup Device              (GPU/CPU)                         │
│  3. Load Dataset              (CarCFDDataset)                   │
│  4. Create Data Loaders       (train/test batches)              │
│  5. Initialize GINO Model     (get_model)                       │
│  6. Setup Training Components (optimizer, loss, scheduler)      │
│  7. Create Data Processor     (GINOCFDDataProcessor)            │
│  8. Create Trainer            (Trainer)                         │
│  9. Train Model               (trainer.train loop)              │
│                                                                   │
│  Output: Trained model weights saved to disk                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Script Execution Flow

### Phase 0: Setup & Configuration (Lines 1-38)

#### What: Load configuration from `gino_carcfd_config.py`
```python
# Line 27-28: Add config path to Python path
sys.path.insert(0, "../")
from config.gino_carcfd_config import Default

# Line 30-31: Load and parse configuration
config = make_config_from_cli(Default)
config = config.to_dict()
```

**Why:** GINO has many hyperparameters:
- Model architecture (number of layers, channels, FNO modes)
- Training parameters (learning rate, batch size, epochs)
- Data parameters (grid resolution, train/test split)
- Device settings (GPU/CPU selection)

#### What: Setup distributed training and device
```python
# Line 34: Setup GPU/CPU and distributed training
device, is_logger = setup(config)
```

**Why:** Enable multi-GPU training if configured

#### What: Adjust model based on data resolution
```python
# Line 37-38: Ensure FNO modes don't exceed data resolution
if config.data.sdf_query_resolution < config.model.fno_n_modes[0]:
    config.model.fno_n_modes = [config.data.sdf_query_resolution] * 3
```

**Why:** FNO operates in Fourier space; modes must be ≤ Nyquist frequency

---

### Phase 1: Logging Setup (Lines 40-63)

#### What: Initialize Weights & Biases logging
```python
# Line 43-63: Setup WandB for experiment tracking
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    wandb.init(**wandb_init_args)
```

**Why:** Track metrics, visualize training, log hyperparameters

---

### Phase 2: Data Loading (Lines 66-80)

#### What: Create dataset and data loaders
```python
# Line 67-73: Load CarCFDDataset
data_module = CarCFDDataset(
    root_dir=config.data.root,
    query_res=[config.data.sdf_query_resolution] * 3,
    n_train=config.data.n_train,
    n_test=config.data.n_test,
    download=config.data.download,
)

# Line 79-80: Create train/test data loaders
train_loader = data_module.train_loader(batch_size=1, shuffle=True)
test_loader = data_module.test_loader(batch_size=1, shuffle=False)
```

**What is CarCFDDataset?**

A wrapper around `MeshDataModule` that:
1. Loads car mesh geometry files (.ply)
2. Computes signed distance fields (SDF) between meshes and regular grid
3. Loads pressure data (.npy files)
4. Normalizes data using Gaussian normalization
5. Splits into train/test sets

**Data Structure (per sample):**
```
{
    "vertices": shape (1, n_vertices, 3)           # Car mesh coordinates
    "query_points": shape (1, 32, 32, 32, 3)       # Regular grid
    "distance": shape (1, n_vertices, 1)           # SDF (vertex→grid)
    "press": shape (1, n_vertices, 1)              # Target pressure
}
```

---

### Phase 3: Model Initialization (Lines 82-83)

#### What: Create GINO model from configuration
```python
# Line 83: Build GINO model using config parameters
model = get_model(config)
```

**What is get_model()?**
- Reads `config.model` parameters
- Instantiates GINO class with:
  - Input/output channels
  - FNO block configuration
  - GNO radius for neighbor search
  - Weighting functions
  - Activations and normalizations

**Model Object:** A PyTorch `nn.Module` with forward method:
```python
def forward(self, input_geom, latent_queries, output_queries,
            latent_features=None, x=None, **kwargs):
    # Returns: predicted pressure at output_queries
    # shape: (batch, n_output_points, out_channels)
```

---

### Phase 4: Optimizer & Scheduler Setup (Lines 85-108)

#### What: Create optimizer to update model weights
```python
# Line 86-90: AdamW optimizer with weight decay
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)
```

**Why AdamW?** Adaptive learning rates + weight decay (better generalization)

#### What: Create learning rate scheduler
```python
# Line 92-108: Choose scheduler based on config
if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(...)
```

**Why Scheduler?** Reduce learning rate during training to fine-tune weights

---

### Phase 5: Loss Function Setup (Lines 111-121)

#### What: Create loss functions for training and evaluation
```python
# Line 111: Create Lp loss (L2 by default)
l2loss = LpLoss(d=2, p=2)

# Line 113-116: Assign training loss
if config.opt.training_loss == "l2":
    train_loss_fn = l2loss

# Line 118-121: Assign testing loss
if config.opt.testing_loss == "l2":
    test_loss_fn = l2loss
```

**What is LpLoss?**
- Computes ||predicted - target||_p norm
- `d=2` means spatial dimension is 2 (for pressure on 2D surface)
- `p=2` means L2 norm (Euclidean distance)

**Loss Computation:**
```python
loss = mean(||predicted_pressure - target_pressure||_2)
```

---

### Phase 6: Data Processor Class Definition (Lines 126-208)

#### What: Custom class to transform data between dataset format and model format

```python
class GINOCFDDataProcessor(DataProcessor):
```

**Why?** Raw dataset format ≠ GINO input format

**Key Methods:**

##### `__init__(normalizer, device)`
```python
def __init__(self, normalizer, device="cuda"):
    self.normalizer = normalizer        # Denormalization for eval
    self.device = device                # GPU/CPU
    self.model = None                   # Filled by wrap() (unused)
```

##### `preprocess(sample)` - **Critical Step**
Transforms raw batch → GINO-compatible format

```python
def preprocess(self, sample):
    # INPUT: Raw batch from DataLoader
    # {
    #   "vertices": (1, 3500, 3),
    #   "query_points": (1, 32, 32, 32, 3),
    #   "distance": (1, 3500, 1),
    #   "press": (1, 3500, 1),
    #   ...
    # }

    # Step 1: Extract and squeeze batch dimension
    in_p = sample["vertices"].squeeze(0).to(self.device)        # (3500, 3)
    latent_queries = sample["query_points"].squeeze(0)          # (32, 32, 32, 3)
    out_p = sample["vertices"].squeeze(0).to(self.device)       # (3500, 3)
    f = sample["distance"].to(self.device)                      # (1, 3500, 1)

    # Step 2: Prepare target
    truth = sample["press"].squeeze(0).unsqueeze(-1)            # (3500, 1)

    # Step 3: Truncate output if needed
    output_vertices = truth.shape[1]
    if out_p.shape[0] > output_vertices:
        out_p = out_p[:output_vertices, :]

    # Step 4: Reshape latent_queries to flat grid
    latent_queries = latent_queries.view((-1, 3))               # (32768, 3)

    # Step 5: Build model input dict
    batch_dict = dict(
        input_geom=in_p,                # Vertices (data points)
        latent_queries=latent_queries,  # Regular grid
        output_queries=out_p,           # Output vertices
        latent_features=f,              # Distance feature
        y=truth,                        # Target pressure
        x=None,
    )

    sample.update(batch_dict)
    return sample

    # OUTPUT: sample with GINO keys
```

##### `postprocess(out, sample)` - Data Denormalization
```python
def postprocess(self, out, sample):
    if not self.training:  # During evaluation
        out = self.normalizer.inverse_transform(out)
        y = self.normalizer.inverse_transform(sample["y"].squeeze(0))
        sample["y"] = y
    return out, sample
```

**Why?** Training on normalized data is numerically stable, but we evaluate on real-world scale

##### `to(device)` - Move to GPU
```python
def to(self, device):
    self.device = device
    self.normalizer = self.normalizer.to(device)
    return self
```

---

### Phase 7: Data Processor Instantiation (Lines 212-213)

#### What: Create an instance of GINOCFDDataProcessor
```python
# Line 212: Deep copy the normalizer to avoid modifying original
output_encoder = deepcopy(data_module.normalizers["press"]).to(device)

# Line 213: Create data processor with independent normalizer
data_processor = GINOCFDDataProcessor(
    normalizer=output_encoder,
    device=device
)
```

**Why deepcopy?** Ensure processor modifications don't affect original normalizer

---

### Phase 8: Trainer Initialization (Lines 216-223)

#### What: Create trainer object that manages the training loop
```python
trainer = Trainer(
    model=model,                        # GINO model
    n_epochs=config.opt.n_epochs,       # Number of epochs
    data_processor=data_processor,      # Data transformer
    device=device,                      # GPU/CPU
    wandb_log=config.wandb.log,        # Enable logging
    verbose=is_logger,                 # Print progress
)
```

**What is Trainer?**
An orchestrator class that:
- Manages training loop (epochs, batches)
- Calls preprocess → model → postprocess for each batch
- Computes loss and backprop
- Handles validation
- Saves checkpoints
- Logs metrics

---

### Phase 9: Start Training (Lines 230-238)

#### What: Execute the training loop
```python
trainer.train(
    train_loader=train_loader,           # Training data
    test_loaders={"test": test_loader},  # Validation data
    optimizer=optimizer,                 # Weight updater
    scheduler=scheduler,                 # Learning rate scheduler
    training_loss=train_loss_fn,        # Loss function
    eval_losses={                        # Evaluation metrics
        config.opt.testing_loss: test_loss_fn
    },
    regularizer=None,                   # Optional regularization
)
```

---

## Key Files

### 1. `train_gino_carcfd.py` (Main Script)
**Location:** `scripts/train_gino_carcfd.py` (239 lines)

**Purpose:** Orchestrate the entire training pipeline

**Key Components:**
- Configuration loading
- Dataset instantiation
- Model creation
- Data processor creation
- Trainer setup
- Training loop execution

---

### 2. `gino_carcfd_config.py` (Configuration)
**Location:** `config/gino_carcfd_config.py`

**Purpose:** Centralized hyperparameter management

**Example Structure:**
```python
class Default:
    # Data parameters
    class data:
        root = "~/data/car-pressure-data/processed-car-pressure-data"
        sdf_query_resolution = 32    # 32³ regular grid
        n_train = 800
        n_test = 200

    # Model parameters
    class model:
        in_channels = 1              # Input: distance field
        out_channels = 1             # Output: pressure
        fno_n_modes = (16, 16, 16)   # Fourier modes
        in_gno_radius = 0.033        # Neighbor search radius
        out_gno_radius = 0.033

    # Training parameters
    class opt:
        learning_rate = 0.001
        weight_decay = 0.0005
        scheduler = "CosineAnnealingLR"
        n_epochs = 200

    class wandb:
        log = True
        project = "neural-operator"
```

---

### 3. `CarCFDDataset` (Data Loading)
**Location:** `neuralop/data/datasets/__init__.py`

**What it does:**
- Wraps `MeshDataModule` for car CFD data
- Loads mesh vertices, pressure targets, SDF
- Normalizes pressure with Gaussian normalization
- Creates train/test splits

**Key Method:**
```python
data_module = CarCFDDataset(
    root_dir="...",
    query_res=[32, 32, 32],
)
# Returns: dataset with .train_loader() and .test_loader() methods
```

---

### 4. `GINO` Model Class
**Location:** `neuralop/models/gino.py` (600+ lines)

**What it does:**
- Converts irregular mesh data to regular grid (Input GNO)
- Processes in latent Fourier space (FNO)
- Projects back to output mesh (Output GNO)

**Architecture Overview:**
```
Input Mesh Vertices
       ↓
[INPUT GNO] ← Neighbor search to grid points
       ↓
Features on regular grid
       ↓
[FNO BLOCKS] ← Spectral convolution in Fourier space
       ↓
Processed features on grid
       ↓
[OUTPUT GNO] ← Neighbor search to output vertices
       ↓
Output Mesh Pressures
```

---

### 5. `Trainer` Class
**Location:** `neuralop/training/trainer.py` (700+ lines)

**Key Methods:**
- `train()` - Main training loop
- `train_one_epoch()` - Single epoch
- `train_one_batch()` - Single batch
- `eval_one_batch()` - Validation step
- `evaluate_all()` - Full evaluation

---

## GINO Architecture Explained

### What is GINO?

GINO = **Geometry-Informed Neural Operator**

A neural network that learns operators (functions) that map input functions on irregular meshes to output functions on irregular meshes.

### Architecture Components

```
┌──────────────────────────────────────────────────────────────┐
│                        GINO ARCHITECTURE                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT LAYER                                                  │
│  ├─ Irregular vertices (3500 points in 3D)                   │
│  ├─ Regular grid (32³ = 32768 points)                        │
│  └─ Distance field (as features)                             │
│                                                                │
│  ↓ [GNO INPUT LAYER] - Kernel Integral Transform            │
│  │  • Find neighbors within radius 0.033                     │
│  │  • Integrate vertex features → grid points                │
│  │  • Output: 64 channels on regular grid                    │
│                                                                │
│  ↓ [FNO BLOCKS] × 4 - Spectral Processing (Latent Space)   │
│  │  • Transform to Fourier space                             │
│  │  • Keep 16³ low-frequency modes                           │
│  │  • Linear transformation in Fourier domain                │
│  │  • Transform back to spatial domain                       │
│  │  • Each block: Conv → MLP → Residual                      │
│  │  • Output: 64 channels on regular grid                    │
│                                                                │
│  ↓ [GNO OUTPUT LAYER] - Kernel Integral Transform           │
│  │  • Find neighbors within radius 0.033                     │
│  │  • Integrate grid features → output vertices              │
│  │  • Output: 1 channel on mesh (pressure)                   │
│                                                                │
│  OUTPUT LAYER                                                 │
│  └─ Pressure at each vertex (3500 values)                    │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### GNO (Graph Neural Operator) - The Neighbor Search Component

**Purpose:** Kernel integral transform between different point clouds

**Process:**
```
For each query point x:
    1. Find all data points y within radius r of x
    2. For each neighbor: compute kernel(x, y) × features(y)
    3. Sum all contributions (integral)
    4. Output integrated feature at x
```

**Code:**
```python
# In GNOBlock.forward()
neighbors_dict = self.neighbor_search(
    data=y,              # Source points (vertices or grid)
    queries=x,           # Query points (grid or output vertices)
    radius=0.033         # Neighborhood radius
)
# Returns: indices of neighbors + row splits for CSR format

out = self.integral_transform(
    y=y,                 # Source point embeddings
    x=x,                 # Query point embeddings
    neighbors=neighbors_dict,
    f_y=features         # Feature values at source points
)
```

### FNO (Fourier Neural Operator) - The Global Processing Component

**Purpose:** Efficient long-range feature mixing using spectral convolution

**Why Fourier Space?**
- Regular grid allows FFT (fast)
- Low-frequency modes capture global patterns
- Linear transformation in Fourier = global conv in space

**Process:**
```
Input grid (64 channels, 32³ points)
       ↓
FFT to Fourier space
       ↓
Keep 16³ modes (discard high-frequency noise)
       ↓
Linear transformation (learnable weights)
       ↓
Inverse FFT back to spatial space
       ↓
Output grid (64 channels, 32³ points)
```

### Data Flow Through GINO (Single Forward Pass)

```
STEP 1: INPUT GNO
────────────────
Input:
  • Vertices (irregular): 3500 points in 3D
  • Latent grid (regular): 32,768 points (32³) in 3D
  • Distance features: 3500 values

Process:
  • Neighbor search: for each grid point, find nearby vertices
  • Kernel integral: weighted sum of vertex features
  • Neighbors per grid point: ~100-500 (depending on radius 0.033)

Output:
  • Features on regular grid: [32, 32, 32, 64 channels]

STEP 2: FNO BLOCKS (4 layers)
──────────────────────────
Input:
  • Grid features: [32, 32, 32, 64]

Each FNO Block:
  • FFT: to Fourier space
  • Conv: linear transformation in Fourier (16³ modes)
  • iFFT: back to spatial
  • MLP: pointwise nonlinearity
  • Residual: skip connection

Output:
  • Grid features: [32, 32, 32, 64] (same shape)

STEP 3: OUTPUT GNO
──────────────────
Input:
  • Grid features: [32, 32, 32, 64]
  • Output vertices (irregular): 3500 points
  • Latent grid: 32,768 points (to enable neighbor search)

Process:
  • Neighbor search: for each output vertex, find nearby grid points
  • Kernel integral: weighted sum of grid features

Output:
  • Pressure at each vertex: [3500, 1]

STEP 4: PROJECTION (MLP)
────────────────────────
Input:
  • Features at vertices: [3500, 64]

Process:
  • 3-layer MLP with 512 hidden units
  • Reduce from 64 channels → 1 channel (pressure)

Output:
  • Final pressure prediction: [3500, 1]
```

---

## Data Flow Through the System

### 1. From Disk to DataLoader

```
disk: /path/to/car_data/
    ├─ sample_0/
    │   ├─ tri_mesh.ply          (mesh geometry)
    │   └─ press.npy             (target pressure)
    ├─ sample_1/
    │   ├─ tri_mesh.ply
    │   └─ press.npy
    └─ ...

        ↓ CarCFDDataset loads and preprocesses

in-memory: DictDataset
    ├─ sample_0 = {
    │   "vertices": tensor(3500, 3),
    │   "query_points": tensor(32, 32, 32, 3),
    │   "distance": tensor(3500, 1),
    │   "press": tensor(3500, 1),
    │   "vertex_normals": tensor(3500, 3),
    │   "centroids": tensor(3682, 3),
    │   ...
    │ }
    └─ ...

        ↓ DataLoader batches samples

        ↓ trainer.train_one_batch()
```

### 2. From DataLoader to Model Output

```
DataLoader batch:
  {
    "vertices": tensor(1, 3500, 3),
    "query_points": tensor(1, 32, 32, 32, 3),
    "distance": tensor(1, 3500, 1),
    "press": tensor(1, 3500, 1),
    ...
  }

    ↓ data_processor.preprocess()

GINOCFDDataProcessor format:
  {
    "input_geom": tensor(3500, 3),           ← vertices
    "latent_queries": tensor(32768, 3),      ← flattened grid
    "output_queries": tensor(3500, 3),       ← output vertices
    "latent_features": tensor(1, 3500, 1),   ← distance
    "y": tensor(3500, 1),                    ← target pressure
    "x": None,
  }

    ↓ model(**sample)

GINO.forward():
  1. INPUT GNO:   vertices → grid features
  2. FNO BLOCKS:  grid features → processed features
  3. OUTPUT GNO:  grid features → vertex features
  4. PROJECTION:  vertex features → pressure

  returns: tensor(1, 3500, 1)  ← predicted pressure

    ↓ data_processor.postprocess()

If training:
  Skip denormalization (loss on normalized scale)

If evaluating:
  Denormalize output and target

    ↓ loss = training_loss(out, **sample)

Loss value: scalar
```

---

## Training Loop Mechanics

### Single Training Step

```
1. Sample batch from train_loader
   sample = next(iter(train_loader))

2. Preprocess data
   sample = data_processor.preprocess(sample)

3. Zero gradients
   optimizer.zero_grad()

4. Forward pass through model
   output = model(**sample)

5. Postprocess (no denormalization during training)
   output, sample = data_processor.postprocess(output, sample)

6. Compute loss
   loss = training_loss(output, sample["y"])

7. Backward pass (compute gradients)
   loss.backward()

8. Update weights
   optimizer.step()

9. Log metrics
   wandb.log({"loss": loss})
```

### Full Training Epoch

```
for epoch in range(n_epochs):

    # Training phase
    data_processor.train()  # Set training=True
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        loss = train_one_batch(batch)
        avg_loss += loss

        # Update learning rate (some schedulers)
        if scheduler_type == "OneCycleLR":
            scheduler.step()

    # Validation phase (every eval_interval epochs)
    if epoch % eval_interval == 0:
        data_processor.eval()   # Set training=False
        model.eval()

        for batch_idx, batch in enumerate(test_loader):
            with torch.no_grad():
                loss = eval_one_batch(batch)
                metrics[f"test_{metric}"] += loss

        # Update learning rate scheduler
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(avg_test_loss)

    # Log epoch metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_loss,
        "test_loss": avg_test_loss,
        "lr": optimizer.param_groups[0]['lr'],
    })

    # Save checkpoint if best
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        save_checkpoint(model, optimizer, epoch)
```

### Training Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Epochs** | 200 | Total training iterations |
| **Batch Size** | 1 | Samples per gradient step |
| **Learning Rate** | 0.001 | Weight update magnitude |
| **Weight Decay** | 0.0005 | L2 regularization |
| **Optimizer** | AdamW | Momentum-based SGD |
| **Scheduler** | CosineAnnealingLR | Learning rate annealing |
| **Loss** | L2 (MSE) | Mean squared error |

---

## GINO Model Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **in_gno_radius** | 0.033 | Input neighbor search radius |
| **out_gno_radius** | 0.033 | Output neighbor search radius |
| **fno_n_modes** | (16, 16, 16) | Number of Fourier modes |
| **fno_hidden_channels** | 64 | Intermediate channel size |
| **fno_n_layers** | 4 | Number of FNO blocks |
| **in_gno_channel_mlp_hidden_layers** | [80, 80, 80] | Input GNO MLP layers |
| **out_gno_channel_mlp_hidden_layers** | [512, 256] | Output GNO MLP layers |
| **in_channels** | 1 | Input channels (distance) |
| **out_channels** | 1 | Output channels (pressure) |

---

## Quick Reference Guide

### Class Hierarchy & Instantiation

```python
# 1. CONFIGURATION
config = make_config_from_cli(Default)
# config.model, config.data, config.opt, config.wandb

# 2. DATASET & LOADERS
data_module = CarCFDDataset(root_dir, query_res, n_train, n_test)
# has: data_module.normalizers, train_loader(), test_loader()

train_loader = data_module.train_loader(batch_size=1, shuffle=True)
test_loader = data_module.test_loader(batch_size=1, shuffle=False)

# 3. MODEL
model = get_model(config)  # Returns: GINO instance
# has: model.forward(), model.train(), model.eval()

# 4. OPTIMIZER
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
# has: optimizer.zero_grad(), optimizer.step()

# 5. SCHEDULER
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# has: scheduler.step()

# 6. LOSS FUNCTION
loss_fn = LpLoss(d=2, p=2)
# usage: loss = loss_fn(output, target)

# 7. DATA PROCESSOR
normalizer = deepcopy(data_module.normalizers["press"]).to(device)
data_processor = GINOCFDDataProcessor(normalizer=normalizer, device=device)
# has: preprocess(), postprocess(), to(device), train(), eval()

# 8. TRAINER
trainer = Trainer(model, n_epochs=200, data_processor=data_processor, ...)
# has: train(), train_one_epoch(), train_one_batch(), eval_one_batch()

# 9. TRAINING EXECUTION
trainer.train(
    train_loader=train_loader,
    test_loaders={"test": test_loader},
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=loss_fn,
    eval_losses={"l2": loss_fn},
)
```

### Key Method Calls in Sequence

```
trainer.train()
  ├─ trainer.__init__()
  │   └─ data_processor.to(device)
  │
  ├─ for epoch in range(n_epochs):
  │   │
  │   ├─ trainer.train_one_epoch()
  │   │   ├─ data_processor.train()
  │   │   ├─ model.train()
  │   │   └─ for batch in train_loader:
  │   │       ├─ trainer.train_one_batch(batch)
  │   │       │   ├─ sample = data_processor.preprocess(sample)
  │   │       │   ├─ output = model(**sample)
  │   │       │   ├─ output, sample = data_processor.postprocess(out, sample)
  │   │       │   ├─ loss = training_loss(output, sample["y"])
  │   │       │   ├─ loss.backward()
  │   │       │   └─ optimizer.step()
  │   │
  │   └─ if epoch % eval_interval == 0:
  │       ├─ trainer.evaluate_all()
  │       │   ├─ data_processor.eval()
  │       │   ├─ model.eval()
  │       │   └─ for batch in test_loader:
  │       │       ├─ trainer.eval_one_batch(batch)
  │       │       │   ├─ sample = data_processor.preprocess(sample)
  │       │       │   ├─ output = model(**sample)
  │       │       │   ├─ output, sample = data_processor.postprocess(out, sample)
  │       │       │   └─ loss = eval_loss(output, sample["y"])
  │       │
  │       ├─ scheduler.step(test_loss)
  │       └─ wandb.log(metrics)
  │
  └─ return final_metrics
```

---

## Mental Model Summary

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHAT HAPPENS AT EACH STAGE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ BEFORE TRAINING:                                                 │
│  • Load CAR MESHES from disk (3500 vertices each)               │
│  • Compute DISTANCE FIELD between mesh and regular grid         │
│  • Load PRESSURE data (ground truth)                            │
│  • NORMALIZE pressure for numerical stability                   │
│  • Create TRAIN/TEST split                                      │
│                                                                   │
│ DURING EACH TRAINING STEP:                                      │
│  • Extract one sample: mesh + distance + pressure              │
│  • PREPROCESS: reformat to model input format                  │
│  • FORWARD: run through GINO                                    │
│    - INPUT GNO: mesh → grid (neighbor search)                  │
│    - FNO: grid processing (Fourier spectral conv)              │
│    - OUTPUT GNO: grid → mesh (neighbor search)                 │
│  • COMPUTE LOSS: compare prediction vs target                  │
│  • BACKWARD: compute gradients                                  │
│  • UPDATE: gradient descent                                     │
│                                                                   │
│ DURING VALIDATION:                                              │
│  • Same preprocess/forward steps                                │
│  • But NO gradient computation                                  │
│  • DENORMALIZE output for real-world metrics                   │
│  • Evaluate on actual pressure scale                            │
│                                                                   │
│ AFTER TRAINING:                                                 │
│  • Save trained weights                                         │
│  • Can now predict pressure on new car meshes                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **GINO** = Graph Neural Operator
   - Maps functions on irregular meshes to functions on irregular meshes
   - Uses neighbor search (GNO) + spectral processing (FNO)

2. **Neighbor Search** = Find nearby points within radius
   - INPUT GNO: mesh → grid
   - OUTPUT GNO: grid → mesh
   - Enables local information flow

3. **Fourier Neural Operator** = Global information flow
   - Uses FFT for efficient computation
   - Spectral convolution in Fourier space
   - Can learn long-range dependencies

4. **Data Processor** = Format transformation
   - Raw dataset format → model input format
   - Handles normalization/denormalization
   - Handles device placement (CPU/GPU)

5. **Trainer** = Orchestrator
   - Manages training loop
   - Calls model, computes loss, updates weights
   - Handles validation and checkpoints

---

## Conclusion

The training script is a well-structured pipeline that:

1. **Loads configuration** → defines all hyperparameters
2. **Loads data** → CAR meshes, distances, pressures
3. **Creates model** → GINO with specified architecture
4. **Sets up training** → optimizer, scheduler, loss
5. **Processes data** → transforms formats as needed
6. **Trains model** → iterative weight updates
7. **Validates** → measures performance on test set
8. **Saves results** → trained weights for inference

The **GINO architecture** is specifically designed for this task:
- **GNO layers** handle irregular-to-regular and regular-to-irregular transformations
- **FNO layers** enable efficient global feature processing
- **Neighbor search** creates local connectivity graphs
- This combination allows learning mappings between arbitrary geometries

The beauty of this approach is that the same architecture can work on **any car geometry** once trained, making it a true neural operator!


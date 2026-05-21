# GINO Training - Quick Reference & Cheat Sheet

## 📊 Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                  train_gino_carcfd.py EXECUTION FLOW                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 1: INITIALIZATION                                          │ │
│ ├─────────────────────────────────────────────────────────────────┤ │
│ │ Line 27-28:  Load configuration (gino_carcfd_config.py)        │ │
│ │ Line 34:     Setup device (GPU/CPU)                            │ │
│ │ Line 37-38:  Adjust model based on data resolution             │ │
│ │ Line 43-63:  Setup WandB logging                               │ │
│ │                                                                  │ │
│ │ Objects Created: config, device, is_logger                     │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 2: DATA                                                    │ │
│ ├─────────────────────────────────────────────────────────────────┤ │
│ │ Line 67-73:  Create CarCFDDataset                              │ │
│ │              └─ Loads meshes, computes SDF, normalizes data    │ │
│ │ Line 79-80:  Create train/test loaders (batch_size=1)          │ │
│ │                                                                  │ │
│ │ Objects Created: data_module, train_loader, test_loader       │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 3: MODEL & TRAINING SETUP                                 │ │
│ ├─────────────────────────────────────────────────────────────────┤ │
│ │ Line 83:     Create GINO model (get_model)                     │ │
│ │ Line 86-90:  Create AdamW optimizer                            │ │
│ │ Line 92-108: Create learning rate scheduler                    │ │
│ │ Line 111:    Create L2 loss function                           │ │
│ │                                                                  │ │
│ │ Objects Created: model, optimizer, scheduler, loss_fn          │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 4: DATA PROCESSOR & TRAINER                               │ │
│ ├─────────────────────────────────────────────────────────────────┤ │
│ │ Line 126-208: Define GINOCFDDataProcessor class                │ │
│ │ Line 212-213: Create data processor instance                   │ │
│ │ Line 216-223: Create Trainer                                   │ │
│ │                                                                  │ │
│ │ Objects Created: data_processor, trainer                       │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 5: TRAINING EXECUTION                                     │ │
│ ├─────────────────────────────────────────────────────────────────┤ │
│ │ Line 230-238: Call trainer.train()                             │ │
│ │                                                                  │ │
│ │ FOR EACH EPOCH (200 epochs):                                   │ │
│ │   FOR EACH BATCH IN TRAIN LOADER:                              │ │
│ │     1. data_processor.preprocess() - reshape data              │ │
│ │     2. model.forward() - GINO forward pass                     │ │
│ │     3. data_processor.postprocess() - denormalize (if eval)   │ │
│ │     4. loss_fn() - compute loss                                │ │
│ │     5. loss.backward() - compute gradients                     │ │
│ │     6. optimizer.step() - update weights                       │ │
│ │                                                                  │ │
│ │   IF EVAL EPOCH:                                               │ │
│ │     FOR EACH BATCH IN TEST LOADER:                             │ │
│ │       Same as above but no gradient computation                │ │
│ │                                                                  │ │
│ │ Result: Trained model weights                                  │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 GINO Architecture Pipeline

```
                        INPUT BATCH
                            │
                    ┌───────┴────────┐
                    │                │
              VERTICES           QUERY GRID
              (3500 pts)          (32768 pts)
                    │                │
                    └───────┬────────┘
                            │
            ┌───────────────────────────────┐
            │                               │
            │      INPUT GNO LAYER          │
            │                               │
            │  • Neighbor search (r=0.033)  │
            │  • Kernel integral transform  │
            │  • Vertices → Grid features   │
            │                               │
            └───────────────┬───────────────┘
                            │
                    GRID FEATURES
                    (32×32×32, 64 ch)
                            │
         ┌──────────────────┴──────────────────┐
         │                                     │
    ┌────┴─────────────────────────────────┐   │
    │    FNO BLOCK 1 (Spectral Conv)       │   │
    │    • FFT → 16 modes → iFFT           │   │
    │    • MLP → Residual                  │   │
    └────┬─────────────────────────────────┘   │
         │                                     │
    ┌────┴─────────────────────────────────┐   │
    │    FNO BLOCK 2 (Spectral Conv)       │   │
    │    • FFT → 16 modes → iFFT           │   │
    │    • MLP → Residual                  │   │
    └────┬─────────────────────────────────┘   │
         │                                     │
    ┌────┴─────────────────────────────────┐   │
    │    FNO BLOCK 3 (Spectral Conv)       │   │
    │    • FFT → 16 modes → iFFT           │   │
    │    • MLP → Residual                  │   │
    └────┬─────────────────────────────────┘   │
         │                                     │
    ┌────┴─────────────────────────────────┐   │
    │    FNO BLOCK 4 (Spectral Conv)       │   │
    │    • FFT → 16 modes → iFFT           │   │
    │    • MLP → Residual                  │   │
    └────┬─────────────────────────────────┘   │
         │                                     │
         └──────────────┬──────────────────────┘
                        │
                PROCESSED GRID FEATURES
                (32×32×32, 64 ch)
                        │
            ┌───────────┴────────────┐
            │                        │
            │   OUTPUT GNO LAYER     │
            │                        │
            │ • Neighbor search      │
            │ • Kernel integral      │
            │ • Grid → Vertices      │
            │                        │
            └───────────┬────────────┘
                        │
                VERTEX FEATURES
                (3500 pts, 64 ch)
                        │
            ┌───────────┴────────────┐
            │                        │
            │ PROJECTION (MLP)       │
            │                        │
            │ 64 ch → 1 ch           │
            │ (pressure output)      │
            │                        │
            └───────────┬────────────┘
                        │
                PRESSURE OUTPUT
                (3500 pts, 1 ch)
```

---

## 🔄 Single Training Step

```
STEP 1: GET BATCH
────────────────
sample = {
  "vertices": (1, 3500, 3),
  "query_points": (1, 32, 32, 32, 3),
  "distance": (1, 3500, 1),
  "press": (1, 3500, 1),
  ...
}

            ↓

STEP 2: PREPROCESS
──────────────────
sample = data_processor.preprocess(sample)
sample = {
  "input_geom": (3500, 3),           ← vertices
  "latent_queries": (32768, 3),      ← flattened grid
  "output_queries": (3500, 3),       ← output vertices
  "latent_features": (1, 3500, 1),   ← distance
  "y": (3500, 1),                    ← target pressure
  "x": None,
}

            ↓

STEP 3: FORWARD
───────────────
output = model(**sample)
output shape: (1, 3500, 1)

            ↓

STEP 4: POSTPROCESS
───────────────────
output, sample = data_processor.postprocess(output, sample)

If training: Skip inverse normalization
If evaluating: Apply inverse normalization

            ↓

STEP 5: COMPUTE LOSS
────────────────────
loss = loss_fn(output, sample["y"])
loss = scalar value

            ↓

STEP 6: BACKWARD
────────────────
loss.backward()
Compute ∂loss/∂w for all weights

            ↓

STEP 7: UPDATE
──────────────
optimizer.step()
w = w - lr × ∂loss/∂w

            ↓

STEP 8: LOG
───────────
wandb.log({"loss": loss.item()})

            ↓

STEP 9: NEXT BATCH
──────────────────
Repeat for next sample
```

---

## 📋 Key Classes & Methods

### 1. CarCFDDataset
```python
data_module = CarCFDDataset(
    root_dir="...",
    query_res=[32, 32, 32],
    n_train=800,
    n_test=200,
)

# Methods
train_loader = data_module.train_loader(batch_size=1, shuffle=True)
test_loader = data_module.test_loader(batch_size=1, shuffle=False)

# Attributes
data_module.normalizers["press"]  # UnitGaussianNormalizer
data_module.time_to_distance     # Average SDF computation time
```

### 2. GINO Model
```python
model = GINO(
    in_channels=1,
    out_channels=1,
    fno_n_modes=(16, 16, 16),
    fno_hidden_channels=64,
    fno_n_layers=4,
    in_gno_radius=0.033,
    out_gno_radius=0.033,
)

# Methods
output = model(
    input_geom=(b, n_vertices, 3),
    latent_queries=(b, n_grid, 3),
    output_queries=(b, n_output, 3),
    latent_features=(b, n_features, 1),
)
# Returns: (b, n_output, 1)
```

### 3. GINOCFDDataProcessor
```python
processor = GINOCFDDataProcessor(
    normalizer=normalizer,
    device="cuda"
)

# Methods
sample = processor.preprocess(sample)          # Dict transform
output, sample = processor.postprocess(out, sample)  # Denormalize
processor = processor.to(device)               # Move to device
processor.train()  # or processor.eval()       # Set mode
```

### 4. Trainer
```python
trainer = Trainer(
    model=model,
    n_epochs=200,
    data_processor=data_processor,
    device=device,
)

# Methods
trainer.train(
    train_loader=train_loader,
    test_loaders={"test": test_loader},
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=loss_fn,
    eval_losses={"l2": loss_fn},
)
```

---

## ⚙️ Important Hyperparameters

### Data Hyperparameters
```python
config.data = {
    'root': '~/data/car-pressure-data/processed-car-pressure-data',
    'sdf_query_resolution': 32,      # 32³ regular grid
    'n_train': 800,                  # Training samples
    'n_test': 200,                   # Test samples
}
```

### Model Hyperparameters
```python
config.model = {
    'in_channels': 1,                # Input: distance field
    'out_channels': 1,               # Output: pressure
    'fno_n_modes': (16, 16, 16),    # Fourier modes (low-frequency)
    'fno_hidden_channels': 64,       # Channel size in FNO
    'fno_n_layers': 4,               # Number of FNO blocks
    'in_gno_radius': 0.033,          # Input neighbor search radius
    'out_gno_radius': 0.033,         # Output neighbor search radius
}
```

### Training Hyperparameters
```python
config.opt = {
    'learning_rate': 0.001,          # Weight update magnitude
    'weight_decay': 0.0005,          # L2 regularization
    'n_epochs': 200,                 # Training iterations
    'batch_size': 1,                 # Samples per step
    'optimizer': 'AdamW',            # Optimizer type
    'scheduler': 'CosineAnnealingLR',# Learning rate schedule
    'training_loss': 'l2',           # Loss function
    'testing_loss': 'l2',            # Evaluation loss
}
```

---

## 🎯 Quick Answers

### Q: What does the script do?
**A:** Trains a GINO model to predict pressure on car surfaces from geometry.

### Q: What is the input to the model?
**A:**
- Car mesh vertices (3500 points in 3D)
- Regular grid (32³ = 32,768 points)
- Distance field (as features)

### Q: What is the output?
**A:** Predicted pressure at each vertex (3500 values)

### Q: How is data transformed?
**A:**
1. Load meshes from disk (.ply files)
2. Compute signed distance field
3. Normalize pressure values
4. Create DataLoader for batching
5. Preprocess before model: reshape, move to GPU
6. Postprocess after model: denormalize for evaluation

### Q: What does GINO do?
**A:**
1. INPUT GNO: Project vertex data onto regular grid (neighbor search)
2. FNO: Process globally in Fourier space (spectral convolution)
3. OUTPUT GNO: Project grid features back to vertices (neighbor search)
4. PROJECTION: Convert to final output (pressure)

### Q: Why neighbor search?
**A:** To handle irregular point clouds by finding nearby points within a fixed radius.

### Q: Why Fourier space?
**A:** Efficient global processing using FFT; low-frequency modes capture important patterns.

### Q: When is data normalized?
**A:** Always during training (numerical stability). Denormalized during evaluation (real-world scale metrics).

### Q: What's the training loop?
**A:**
```
for epoch in epochs:
    for batch in train_loader:
        preprocess → forward → loss → backward → update weights
    validate on test set
    save if best model
```

---

## 🔍 Debugging Tips

### If model doesn't learn:
- Check learning rate (try 0.01 or 0.0001)
- Verify data normalization is correct
- Check batch size (try 2 or 4)
- Verify loss function is correct

### If CUDA out of memory:
- Reduce batch size (already 1, can't go lower)
- Reduce grid resolution (sdf_query_resolution)
- Reduce model channels (fno_hidden_channels)

### If training is slow:
- Open3D neighbor search is faster (default)
- Reduce number of FNO blocks
- Reduce training epochs for testing

### If evaluation metrics are wrong:
- Check data_processor.eval() is called before validation
- Verify postprocess() is inverting normalization correctly
- Ensure test batch uses same normalizer as training

---

## 📚 File Locations

```
neuraloperator/
├── scripts/
│   └── train_gino_carcfd.py              ← MAIN SCRIPT
├── config/
│   └── gino_carcfd_config.py             ← HYPERPARAMETERS
├── neuralop/
│   ├── models/
│   │   └── gino.py                       ← GINO ARCHITECTURE
│   ├── layers/
│   │   ├── gno_block.py                  ← GNO BLOCKS
│   │   ├── neighbor_search.py            ← NEIGHBOR SEARCH
│   │   └── integral_transform.py         ← KERNEL INTEGRAL
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── car_cfd_dataset.py        ← DATASET WRAPPER
│   │   │   └── mesh_datamodule.py        ← MESH LOADING
│   │   └── transforms/
│   │       └── data_processors.py        ← BASE PROCESSOR CLASS
│   └── training/
│       └── trainer.py                    ← TRAINER CLASS
└── GINO_TRAINING_FLOW.md                 ← FULL DOCUMENTATION
```

---

## 🚀 How to Modify the Script

### To change model architecture:
Edit `config/gino_carcfd_config.py`:
```python
class Default:
    class model:
        fno_n_modes = (32, 32, 32)  # Increase for larger models
        fno_hidden_channels = 128   # More channels
        fno_n_layers = 8            # More processing blocks
```

### To change training parameters:
```python
class Default:
    class opt:
        learning_rate = 0.01        # Try different values
        n_epochs = 500              # Train longer
```

### To change data:
```python
class Default:
    class data:
        sdf_query_resolution = 64   # Finer grid
        n_train = 1000              # More training samples
```

### To add custom data processor:
Subclass `GINOCFDDataProcessor` and override:
- `preprocess()` - custom input transformation
- `postprocess()` - custom output transformation

---

## ✅ Checklist Before Running

- [ ] Configuration file exists and is correct
- [ ] Data directory path is valid
- [ ] GPU has enough memory (check with nvidia-smi)
- [ ] All dependencies installed (torch, open3d, etc.)
- [ ] Output directory exists for saving checkpoints
- [ ] WandB token is set (if using logging)
- [ ] Verify first batch loads without errors
- [ ] Check loss decreases in first few steps


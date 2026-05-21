# Neighbor Search Logging Implementation

This document explains the neighbor search logging system that has been integrated into the GINO training pipeline.

## Overview

The system logs neighbor search results from both the input GNO (`gno_in`) and output GNO (`gno_out`) layers during training. For each epoch:
- **gno_in**: Logs neighbors found for each query point in the latent space
- **gno_out**: Logs neighbors found for each output mesh vertex

## Files Modified

### 1. `neuralop/models/gino.py`
- Added storage fields for neighbor search results:
  - `last_gno_in_neighbors`, `last_gno_in_query`, `last_gno_in_data`
  - `last_gno_out_neighbors`, `last_gno_out_query`, `last_gno_out_data`
- Modified `forward()` method to capture neighbor search results from both GNO blocks
- Results are stored after neighbor search but before the GNO forward pass (minimal overhead)

### 2. `neuralop/utils.py`
- Added `NeighborSearchLogger` class
- Writes logs in human-readable text format
- Logs include:
  - Summary statistics (total queries, data points, neighbors)
  - Per-query details with neighbor indices and distances
  - Squared distances for weighted neighbors

### 3. `neuralop/training/trainer.py`
- Added import of `NeighborSearchLogger`
- Added storage fields in `__init__()` for first batch neighbor data
- Modified `train_one_epoch()` to capture first batch during training
- Modified `evaluate()` to capture first batch during testing
- Modified epoch loop in `train()` to call logger after each eval interval

## Log File Format

Logs are written to `./neighbor_logs/` directory with naming:
```
gno_in_train_epoch_0.txt
gno_in_test_epoch_0.txt
gno_out_train_epoch_0.txt
gno_out_test_epoch_0.txt
... (one per epoch)
```

### Log Content Format

```
================================================================================
Neighbor Search Log: GNO_IN (TRAIN)
Epoch: 0
================================================================================

Summary Statistics:
  Total query points: 32768
  Total data points: 3500
  Total neighbors found: 105432
  Average neighbors per query: 3.22

================================================================================

Query Point 0:
  Coordinates: [0.25 0.50 0.75]
  Number of neighbors: 3
  Neighbor indices: [105, 234, 1023]
    [0] data_idx=105, coords=[0.24 0.51 0.76], sq_distance=0.000321
    [1] data_idx=234, coords=[0.26 0.49 0.74], sq_distance=0.000512
    [2] data_idx=1023, coords=[0.23 0.52 0.77], sq_distance=0.000841

Query Point 1:
  ...
```

## How It Works

### During Training

1. **First batch only**: When `train_one_batch()` processes the first batch (idx==0):
   - After the model forward pass, neighbor info is captured from GINO model
   - Stored in trainer fields: `gno_in_neighbors_train`, etc.

2. **After epoch evaluation**: When `evaluate_all()` runs at eval interval:
   - Test data first batch is processed in `evaluate()`
   - Neighbor info is captured similarly and stored in test fields

3. **Logging**: After `evaluate_all()` completes:
   - `neighbor_logger.log_epoch()` is called with all captured data
   - 4 files are written (gno_in train/test, gno_out train/test) for that epoch

### Data Captured

For **gno_in**:
- Query points: latent space points (SDF grid)
- Data points: input geometry (car mesh vertices)
- Neighbors: vertices near each query point

For **gno_out**:
- Query points: output mesh vertices
- Data points: latent space points (from FNO)
- Neighbors: latent points near each output vertex

## Performance

- **Minimal overhead**: Neighbor search is called once per batch in normal flow; we just store the results
- **First batch only**: Only logs the first batch of train and test to keep files manageable
- **Happens at eval_interval**: Logging only occurs when model is evaluated (not every epoch if eval_interval > 1)

## Usage

No additional code needed! The logging happens automatically:

```python
trainer = Trainer(model=model, n_epochs=10, ...)
trainer.train(
    train_loader=train_loader,
    test_loaders={"test": test_loader},
    ...
)
```

Logs will automatically be written to `./neighbor_logs/` after each evaluation.

## Configuration

To change log directory:

```python
from neuralop.utils import NeighborSearchLogger

logger = NeighborSearchLogger(log_dir="./my_logs")
trainer.neighbor_logger = logger
```

## Example Log Inspection

```bash
# View train logs for epoch 0
cat ./neighbor_logs/gno_in_train_epoch_0.txt

# Count total neighbors in an epoch
grep "data_idx=" ./neighbor_logs/gno_in_train_epoch_0.txt | wc -l

# Find maximum neighbors for a single query
grep -A 100 "Query Point" ./neighbor_logs/gno_in_train_epoch_0.txt | grep "Number of neighbors" | sort -t: -k2 -n | tail -1
```

## Summary

This implementation provides visibility into how the GNO layers search for neighbors during training, without affecting training speed or model behavior. The logs help understand:
- How many neighbors are found at different stages
- The neighborhood structure of the data
- Whether radius parameters are appropriate
- How neighbor counts change with training

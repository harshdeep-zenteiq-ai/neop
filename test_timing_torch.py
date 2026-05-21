"""
Quick timing test for PyTorch training loop.
Runs a few batches with detailed timing breakdown.
"""
import sys
from timeit import default_timer
sys.path.insert(0, "./")

import torch
import numpy as np
from neuralop.data.datasets.car_cfd_dataset import CarCFDDataset
from neuralop.data.transforms.data_processors import DataProcessor
from neuralop import get_model
from neuralop.losses.data_losses import LpLoss
from neuralop.training.trainer import Trainer
from copy import deepcopy
from zencfg import make_config_from_cli
from config.gino_carcfd_config import Default

# Load config
config = make_config_from_cli(Default)
config = config.to_dict()

# Model setup
if config.data.sdf_query_resolution < config.model.fno_n_modes[0]:
    config.model.fno_n_modes = [config.data.sdf_query_resolution] * 3

# Data loading
print("[PyTorch] Loading data...")
data_module = CarCFDDataset(
    data_dir=config.data.data_dir,
    n_vertices=config.data.n_vertices,
    query_resolution=config.data.sdf_query_resolution,
    normalize_sdf=config.data.get("normalize_sdf", False),
)

# Model creation
model = get_model(config.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create data loader (same batch size as JAX)
train_loader = torch.utils.data.DataLoader(
    data_module.train_data,
    batch_size=1,
    shuffle=False,
)

# Data processor
output_encoder = deepcopy(data_module.normalizers["press"])
data_processor = DataProcessor()

# Loss function
train_loss_fn = LpLoss(d=2, p=2)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay
)

# Create trainer with verbose=True
trainer = Trainer(
    model=model,
    n_epochs=1,
    data_processor=data_processor,
    device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    wandb_log=False,
    verbose=True,  # ENABLE TIMING PRINTOUTS
)
trainer.optimizer = optimizer

print("\n" + "="*80)
print("[PyTorch TIMING TEST] Running 5 batches with detailed timing")
print("="*80 + "\n")

# Warm-up batch
print("[PyTorch] Warm-up batch...")
train_iter = iter(train_loader)
sample = next(train_iter)
loss = trainer.train_one_batch(0, sample, train_loss_fn)
print(f"Warm-up loss: {float(loss.detach()):.6f}\n")

# Timed batches
print("[PyTorch] Timed batches...")
batch_times = []
for batch_idx in range(1, 6):
    sample = next(train_iter)
    batch_start = default_timer()
    loss = trainer.train_one_batch(batch_idx, sample, train_loss_fn)
    batch_time = default_timer() - batch_start
    batch_times.append(batch_time)
    print()

print("\n" + "="*80)
print("[PyTorch SUMMARY]")
print(f"Batch times (s): {[f'{t:.4f}' for t in batch_times]}")
print(f"Mean time/batch: {np.mean(batch_times):.4f}s")
print(f"Std dev:         {np.std(batch_times):.4f}s")
print("="*80)

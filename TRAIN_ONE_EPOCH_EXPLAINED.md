# `train_one_epoch` — A Line-by-Line Walkthrough

This is the heart of every training loop in `neuraloperator`. It runs the model over the entire training dataset once (one **epoch**) and returns loss statistics.

---

## Function Signature

```python
def train_one_epoch(self, epoch, train_loader, training_loss):
```

```
>>> Starting epoch 3
>>> train_loader has 200 batches (e.g. 1600 samples / batch_size=8)
>>> training_loss = RelativeL2Loss()
```

---

## Phase 1 — Setup

```python
self.on_epoch_start(epoch)
```
```
>>> [Hook] on_epoch_start called for epoch=3
>>> (e.g. resets internal state, updates epoch counter: self.epoch = 3)
```

---

```python
avg_loss = 0
avg_lasso_loss = 0
```
```
>>> avg_loss       = 0.0   # will accumulate loss * n_samples per batch
>>> avg_lasso_loss = 0.0   # same but for regularizer (e.g. L1 weight penalty)
```

---

```python
self.model.train()
```
```
>>> Model switched to TRAIN mode
>>> Dropout layers are now active
>>> BatchNorm uses batch statistics (not running stats)
```

---

```python
if self.data_processor:
    self.data_processor.train()
```
```
>>> data_processor.train() called
>>> (e.g. UnitGaussianNormalizer will now update its running stats if applicable)
```

---

```python
t1 = default_timer()
```
```
>>> t1 = 1712345678.123   # wall-clock time at epoch start, used to measure epoch duration
```

---

```python
train_err = 0.0
self.n_samples = 0
```
```
>>> train_err  = 0.0   # accumulates loss.item() per batch (raw sum, before dividing by n_batches)
>>> n_samples  = 0     # counts total training examples seen this epoch (used as denominator for avg_loss)
```

---

## Phase 2 — The Batch Loop

```python
for idx, sample in enumerate(train_loader):
```
```
>>> idx=0,  sample = {'x': Tensor[8,3,64,64], 'y': Tensor[8,1,64,64]}
>>> idx=1,  sample = {'x': Tensor[8,3,64,64], 'y': Tensor[8,1,64,64]}
>>> ...
>>> idx=199 (last batch)
```

Each `sample` is a dict — `x` is input, `y` is the ground truth the model must predict.

---

```python
loss = self.train_one_batch(idx, sample, training_loss)
```

This is where the forward pass happens. Internally it does:
1. `optimizer.zero_grad()` — clears old gradients
2. `data_processor.preprocess(sample)` — normalizes inputs, moves to device
3. `out = self.model(**sample)` — forward pass through the neural operator
4. `loss = training_loss(out, sample['y'])` — computes the loss

```
>>> idx=0: loss = 0.8412   (large early on — model is random)
>>> idx=1: loss = 0.7903
>>> ...
>>> idx=199: loss = 0.1204  (smaller by end of epoch — gradients have accumulated)
```

---

```python
loss.backward()
```
```
>>> Backprop through the entire compute graph
>>> Every parameter p now has p.grad populated
>>> e.g. layer_0.weight.grad = Tensor[256,64] of partial derivatives
```

This is the core of learning — computing how much each weight contributed to the loss.

---

```python
self.optimizer.step()
```
```
>>> Adam update: p = p - lr * (m / (sqrt(v) + eps))
>>> All parameters nudged slightly in the direction that reduces loss
>>> lr=1e-3 (example), so weights shift by a small amount
```

---

```python
train_err += loss.item()
```
```
>>> After batch 0:   train_err = 0.8412
>>> After batch 1:   train_err = 1.6315
>>> ...
>>> After batch 199: train_err = 42.71   (raw sum across all 200 batches)
```

`.item()` extracts the Python float from the tensor — detaches from the compute graph.

---

```python
with torch.no_grad():
    avg_loss += loss.item()
    if self.regularizer:
        avg_lasso_loss += self.regularizer.loss
```
```
>>> torch.no_grad(): no gradients tracked here — just bookkeeping
>>> avg_loss       accumulates loss per batch (same as train_err — used for per-sample avg later)
>>> avg_lasso_loss accumulates the regularizer penalty (e.g. sum of |weights|)
```

Why `no_grad`? These are just logging accumulations — no need to build a compute graph for them.

---

## Phase 3 — Scheduler Step

```python
if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    self.scheduler.step(train_err)
else:
    self.scheduler.step()
```

```
>>> ReduceLROnPlateau: "if loss hasn't improved for N epochs, reduce lr by factor 0.5"
>>>   self.scheduler.step(42.71)  → monitors train_err to decide whether to reduce lr
>>>
>>> CosineAnnealingLR / StepLR / etc:
>>>   self.scheduler.step()  → just advance the schedule (no loss needed)
```

The learning rate schedule determines *how fast* the model learns over time.

---

## Phase 4 — Compute Epoch Statistics

```python
epoch_train_time = default_timer() - t1
```
```
>>> epoch_train_time = 47.3s   (wall clock time for all 200 batches)
```

---

```python
train_err /= len(train_loader)
```
```
>>> train_err = 42.71 / 200 = 0.2136   (mean loss per batch this epoch)
```

---

```python
avg_loss /= self.n_samples
```
```
>>> n_samples = 1600  (200 batches × 8 samples each)
>>> avg_loss  = 42.71 / 1600 = 0.02669   (mean loss per sample — more comparable across batch sizes)
```

`avg_loss` is more robust than `train_err` when batch sizes vary.

---

```python
if self.regularizer:
    avg_lasso_loss /= self.n_samples
else:
    avg_lasso_loss = None
```
```
>>> avg_lasso_loss = 0.00341   (mean L1 weight penalty per sample, if regularizer active)
>>> avg_lasso_loss = None      (if no regularizer — clean sentinel value)
```

---

## Phase 5 — Logging

```python
lr = None
for pg in self.optimizer.param_groups:
    lr = pg["lr"]
```
```
>>> optimizer.param_groups = [{'lr': 0.00087, 'params': [...]}, ...]
>>> lr = 0.00087   (current learning rate — may have been reduced by scheduler)
```

Iterates param groups (useful when different layers have different lrs — `lr` ends up as the last group's value).

---

```python
if self.verbose and epoch % self.eval_interval == 0:
    self.log_training(
        epoch=epoch,
        time=epoch_train_time,
        avg_loss=avg_loss,
        train_err=train_err,
        avg_lasso_loss=avg_lasso_loss,
        lr=lr,
    )
```
```
>>> [Epoch 3 | 47.3s] train_err=0.2136  avg_loss=0.02669  lasso=0.00341  lr=8.7e-4
```

Only logs every `eval_interval` epochs to avoid console spam.

---

## Return Values

```python
return train_err, avg_loss, avg_lasso_loss, epoch_train_time
```

| Value | Meaning | Example |
|---|---|---|
| `train_err` | Mean loss per batch | `0.2136` |
| `avg_loss` | Mean loss per sample | `0.02669` |
| `avg_lasso_loss` | Mean regularizer penalty per sample | `0.00341` or `None` |
| `epoch_train_time` | Wall-clock seconds for this epoch | `47.3` |

These are passed back to the main `train()` loop, logged, and used to decide whether to save a checkpoint.

---

## Big Picture: What One Epoch Actually Does

```
For each batch in the dataset:
  1. Forward pass   →  prediction
  2. Loss           →  how wrong were we?
  3. Backward pass  →  how do we fix each weight?
  4. Optimizer step →  nudge weights in the right direction

After all batches:
  5. Scheduler step →  adjust learning rate
  6. Compute stats  →  average loss over epoch
  7. Log            →  print/wandb progress
```

One epoch = the model has seen every training sample exactly once. You run this for `n_epochs` total, and the model gradually learns to predict better each time.

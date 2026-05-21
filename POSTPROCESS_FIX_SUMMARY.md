# Postprocess Fix for Training Flow - Summary

## Problem Identified
The `train_one_batch` method was missing the postprocess step that exists in `eval_one_batch`, creating an inconsistency in the data flow:

### eval_one_batch (Correct Flow)
```
1. Preprocess(sample)
2. Forward pass: out = model(sample)
3. Postprocess: out, sample = data_processor.postprocess(out, sample)
4. Loss: loss = training_loss(out, sample)
```

### train_one_batch (Before Fix - Missing Postprocess)
```
1. Preprocess(sample)
2. Forward pass + Loss inside train_step() - NO POSTPROCESS
```

## Solution Implemented

### Changes to `scripts/train_gino_carcfd_jax.py`

**Modified `_build_jit_fn` method:**
- Now accepts `postprocess_fn` as a parameter to the internal `_step` function
- Applies `postprocess_fn(out)` inside the JIT-compiled function before computing loss
- Maintains JIT compatibility by passing postprocess_fn as an argument, not capturing in closure

**Modified `train_step` method:**
- Now accepts optional `postprocess_fn` parameter
- Passes it to the JIT function during execution

```python
def _build_jit_fn(self, loss_fn):
    @jax.jit
    def _step(params, opt_state, model_inputs, y, postprocess_fn=None):
        def forward_loss(p):
            out = module.apply(p, **model_inputs)
            if postprocess_fn is not None:
                out = postprocess_fn(out)  # ← POSTPROCESS BEFORE LOSS
            return loss_fn(out, y=y)
        ...
```

### Changes to `neuralop/training/trainer_jax.py`

**Modified `train_one_batch` method:**
- Creates a JAX-compatible postprocess wrapper when `data_processor` exists
- Wrapper function only applies denormalization if NOT in training mode
- Passes wrapper to `train_step`

```python
if hasattr(self.model, 'train_step'):
    postprocess_fn = None
    if self.data_processor is not None:
        def postprocess_fn(out):
            if not self.data_processor.training:
                out = self.data_processor.normalizer.inverse_transform(out)
            return out
    loss = self.model.train_step(sample, training_loss, postprocess_fn=postprocess_fn)
```

## Key Design Decisions

1. **JIT Compatibility:** 
   - Postprocess function passed as argument to JIT function, not captured in closure
   - Pure JAX operations only inside JIT boundary
   - Conditional logic (denormalization check) happens outside JIT

2. **Training vs Evaluation:**
   - During training: `data_processor.training=True` → postprocess does nothing (returns output unchanged)
   - During evaluation: `data_processor.training=False` → postprocess applies denormalization
   - This preserves existing behavior while ensuring code consistency

3. **Loss Computation:**
   - Training: Loss computed on normalized data (postprocess is no-op)
   - Evaluation: Loss computed on denormalized data (postprocess applies normalization)
   - Both follow the same code path now

## Updated Data Flow

### train_one_batch (After Fix - With Postprocess)
```
1. Preprocess(sample)
2. Inside train_step:
   a. Forward pass: out = model(sample)
   b. Postprocess: out = postprocess_fn(out)  [no-op during training]
   c. Loss: loss = training_loss(out, y)
   d. Gradients + Optimizer update
```

### Result
✅ `train_one_batch` now follows same postprocess pattern as `eval_one_batch`
✅ Maintains JIT compilation for performance
✅ Code is future-proof for potential denormalization during training

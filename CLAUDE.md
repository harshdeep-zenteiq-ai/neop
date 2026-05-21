# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NeuralOperator** is a library for learning neural operators — resolution-invariant mappings between function spaces. The repo contains a PyTorch reference implementation and an in-progress JAX port. The current focus is **GINO (Geometry-informed Neural Operator)** trained on the Car CFD dataset for pressure field prediction.

## Installation & Setup

```bash
pip install -e .
pip install -r requirements.txt
```

JAX-specific environment variables (set in training scripts, not globally):
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async
```

## Commands

```bash
# Run all tests
pytest neuralop/

# Run a single test file
pytest neuralop/layers/tests/test_spectral_convolution.py

# Run a specific test
pytest neuralop/layers/tests/test_fno_block.py::test_fno_block_2d -v

# Train GINO (PyTorch)
cd scripts && python train_gino_carcfd.py

# Train GINO (JAX)
cd scripts && python train_gino_carcfd_jax.py

# Override config values from CLI
python train_gino_carcfd_jax.py --data.n_train 10 --data.n_test 5

# TensorBoard profiling
tensorboard --logdir=./scripts/jax_profiles_epoch
```

## Architecture

### Model Pipeline (GINO)

```
Input geometry (irregular point cloud) + SDF query grid
    ↓
Input GNOBlock  — radius-based neighbor search → IntegralTransform kernel
    ↓
ChannelMLP lifting → FNOBlocks (SpectralConv in frequency domain)
    ↓
Output GNOBlock — projects latent grid → output point cloud
    ↓
ChannelMLP projection → pressure predictions
```

### Key Modules

**`neuralop/models/`**
- `gino.py` / `gino_jax.py` — GINO model; the JAX version uses Flax `nn.Module` with `setup()` pattern
- `fno.py` — Standard FNO for regular grids
- `base_model.py` / `base_model_jax.py` — Registry-based base; models register by `name=` kwarg

**`neuralop/layers/`**
- `spectral_convolution.py` / `_jax.py` — Core FFT-based convolution; supports dense/CP/Tucker weight factorizations via `tensorly-torch`. Tucker path invokes `einsum_complexhalf` only when input is `complex64` (not `complex128`)
- `fno_block.py` / `_jax.py` — Sequence of Fourier layers with optional ChannelMLP, skip connections, normalization
- `gno_block.py` / `_jax.py` — Graph Neural Operator layer; wraps `NeighborSearch` + `IntegralTransform`
- `integral_transform.py` — Computes kernel integrals over neighbor graphs in CSR format
- `channel_mlp.py` / `_jax.py` — Pointwise MLP applied channel-wise; uses `Conv(kernel_size=1)` to preserve spatial structure (critical: do NOT flatten spatial dims)
- `skip_connections_jax.py` — `SoftGating` initializes weights with `nn.initializers.ones`; **must pass `jnp.float32` as dtype** or params default to f64
- `embeddings_jax.py` — `SinusoidalEmbedding`: frequency arithmetic inside `setup()` produces f64; **cast `freqs` to `x.dtype` before einsum**

**`neuralop/data/`**
- `datasets/mesh_datamodule_jax.py` — `CarCFDDatasetjax`; loads geometry + SDF pressure data, pads neighbor arrays to fixed max length before JIT (shapes must be static)
- `transforms/data_processors_jax.py` — `DataProcessor` (Flax module): `preprocess` / `postprocess` called around model forward

**`neuralop/training/`**
- `trainer.py` — PyTorch trainer (reference)
- `trainer_jax.py` — JAX trainer; uses `jax.jit`-compiled `_step` function with `jax.value_and_grad`; includes TensorBoard profiling per epoch/batch
- `training_state_jax.py` — Checkpoint save/load for JAX/Flax models

**`config/`** — `zencfg`-based dataclass configs. `Default` in `gino_carcfd_config.py` is the root. Note: `opt` field is typed as `ConfigBase` (abstract), so `--opt.n_epochs` CLI override fails; use `--data.*` overrides instead.

**`scripts/`** — Training entry points. `train_gino_carcfd_jax.py` contains `FlaxModelWrapper` that manages Flax parameter state and the JIT-compiled step function.

**`parity_checker/`** — Agent-based framework for comparing JAX vs PyTorch layer outputs numerically.

## JAX Port: Critical Constraints

**Tensor layout**: PyTorch uses `(batch, channels, spatial...)`, Flax Conv uses `(batch, spatial..., channels)`. Transpositions are frequent and expected in `_jax.py` files.

**Static shapes for JIT**: Neighbor arrays must be padded to a fixed max size before building data loaders. Variable-length neighbor lists cause recompilation on every batch.

**f64 contamination kills performance**: RTX 3050 FP64 throughput is ~1/64 of FP32, causing ~14× slowdown. Two known sources:
1. `SoftGating.setup()` — `nn.initializers.ones` defaults to f64; always pass `jnp.float32` as the dtype arg to `self.param()`
2. `SinusoidalEmbedding.__call__` — `jnp.arange(N) / N` inside a `setup()`-pattern module produces f64; cast `freqs = freqs.astype(x.dtype)` before einsum

When `rfft` receives f64 input it returns `complex128`; Tucker contractions then fall through to `tl.einsum` (slow) instead of `einsum_complexhalf` (fast f16 path).

**`value_and_grad` does not recompute the forward pass** — JAX traces the full forward+backward in a single pass. Adding `jax.remat()` rematerializes (drops and recomputes) intermediates; use it only deliberately for memory, not to "fix" recomputation.

## Configuration

Override config from CLI using `--section.key value` syntax:
```bash
python train_gino_carcfd_jax.py --data.n_train 10 --model.fno_n_modes 8 8 8
```
`fno_n_modes` is automatically clamped to `sdf_query_resolution` if larger.

# What `get_model(config)` Returns for `gino_carcfd_config.py`

## Trace through `get_model`

`get_model` lives in `neuralop/models/base_model.py`. It does the following:

### Step 1 — Read `model_arch`

```python
arch = config.model["model_arch"].lower()  # → "gino"
```

`config.model` is `GINO_Small3d()`, which inherits from `GINOConfig`, which sets:
```python
model_arch: str = "gino"
```

### Step 2 — Compute `in_channels`

```python
data_channels = model_config.pop("data_channels")  # → 0  (from GINO_Small3d)
patching_levels = 0                                 # PatchingConfig default
model_config["in_channels"] = data_channels         # → 0
```

### Step 3 — Dispatch

```python
BaseModel._models["gino"](**model_config)
# → GINO(**model_config)
```

`GINO` is registered in `_models` automatically via `__init_subclass__` when the class is defined.

---

## What is instantiated

```
GINO(
    in_channels                    = 0,      # data_channels from GINO_Small3d
    out_channels                   = 1,
    latent_feature_channels        = 1,

    # GNO geometry
    gno_coord_dim                  = 3,
    gno_coord_embed_dim            = 16,
    gno_radius                     = 0.033,  # fallback; in/out_gno_radius default to this
    in_gno_radius                  = 0.033,  # GINO.__init__ default
    out_gno_radius                 = 0.033,  # GINO.__init__ default
    in_gno_transform_type          = "linear",
    out_gno_transform_type         = "linear",
    gno_pos_embed_type             = "nerf",

    # GNO MLP
    gno_embed_channels             = 32,
    gno_embed_max_positions        = 10000,
    in_gno_channel_mlp_hidden_layers  = [80, 80, 80],
    out_gno_channel_mlp_hidden_layers = [512, 256],
    gno_channel_mlp_non_linearity  = F.gelu,
    gno_use_open3d                 = True,
    gno_use_torch_scatter          = True,
    gno_weighting_function         = None,   # GINOConfig default
    gno_weight_function_scale      = None,   # GINOConfig default

    # FNO
    fno_n_modes                    = [16, 16, 16],
    fno_hidden_channels            = 64,
    fno_lifting_channel_ratio      = 2,      # GINO.__init__ default
    fno_n_layers                   = 4,      # GINO.__init__ default
    fno_use_channel_mlp            = True,
    fno_channel_mlp_expansion      = 1.0,   # GINOConfig
    fno_norm                       = "instance_norm",
    fno_ada_in_features            = 32,
    fno_factorization              = "tucker",
    fno_rank                       = 0.4,
    fno_domain_padding             = 0.125,
    fno_resolution_scaling_factor  = 1,
)
```

---

## Architecture summary

```
Input coords (3D)
      │
      ▼
[SinusoidalEmbedding / NerfEmbedding]   ← gno_pos_embed_type="nerf"
      │
      ▼
[GNOBlock — in_gno]                     ← radius=0.033, transform_type="linear"
  ChannelMLP hidden layers: [80, 80, 80]
  Lifts irregular mesh → latent 3D grid (32³)
      │
      ▼
[FNO — 4 layers]                        ← n_modes=(16,16,16), hidden=64, factorization="tucker", rank=0.4
  (instance_norm, channel_mlp after each layer)
      │
      ▼
[GNOBlock — out_gno]                    ← radius=0.033, transform_type="linear"
  ChannelMLP hidden layers: [512, 256]
  Projects latent grid → output mesh points
      │
      ▼
[ChannelMLP projection]                 ← ratio=4 → 256 channels → out_channels=1
      │
      ▼
Output: scalar pressure per mesh point (out_channels=1)
```

---

## Key observations

| Property | Value | Why |
|---|---|---|
| `in_channels=0` | No raw input features | `data_channels=0` in `GINO_Small3d`; the model is purely geometry-driven |
| `latent_feature_channels=1` | SDF channel concatenated onto latent grid | `GINO_Small3d` sets this to 1 |
| `fno_factorization="tucker"` | Tucker-factorized spectral conv | Reduces parameters; `fno_rank=0.4` means ~40% of full-rank |
| `gno_pos_embed_type="nerf"` | NeRF-style sinusoidal positional encoding | Set in `GINOConfig` default |
| `out_channels=1` | Predicts surface pressure (scalar) | CarCFD problem setup |

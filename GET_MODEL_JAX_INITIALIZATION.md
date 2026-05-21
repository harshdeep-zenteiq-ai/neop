# get_model_jax(config) - Detailed Initialization Documentation

## Overview
When `model = get_model_jax(config)` is called, it instantiates a GINO (Geometry-Informed Neural Operator) model based on the provided configuration. This document details the complete initialization pipeline, component structure, and expected shapes.

---

## Phase 1: get_model_jax() Function Call

### Function Signature
```python
def get_model_jax(config):
    """Returns an instantiated model for the given config
    
    Parameters
    ----------
    config : Bunch or dict-like
        configuration with config.model containing model_arch and model parameters
        
    Returns
    -------
    model : nn.Module (Flax Linen Module)
        the instantiated GINO model
    """
```

### Execution Flow

#### Step 1: Extract Architecture Name
```python
arch = config.model["model_arch"].lower()

# For GINO_Small3d config:
arch = "gino"  # lowercase
```

#### Step 2: Copy and Prepare Model Config
```python
model_config = config.model.copy()

# Remove architecture identifier
model_config.pop("model_arch", None)

# Extract data channels
data_channels = model_config.pop("data_channels")  # Default: 0 for GINO_Small3d

# Check for patching configuration
try:
    patching_levels = config["patching"]["levels"]
except KeyError:
    patching_levels = 0

# Adjust data_channels based on patching
if patching_levels:
    data_channels *= patching_levels + 1

# Set in_channels in model config
model_config["in_channels"] = data_channels
```

**For GINO_Small3d with default config:**
```
Original data_channels: 0
Patching levels: 0 (default)
Calculated in_channels: 0
model_config["in_channels"] = 0
```

#### Step 3: Instantiate Model Class
```python
try:
    return BaseModel._models[arch](**model_config)
    # Looks up "gino" in registry and calls GINO(**model_config)
except KeyError:
    raise ValueError(f"Got config.arch={arch}, expected one of {available_models()}.")
```

---

## Phase 2: GINO Model Instantiation

### GINO Class Definition
```python
class GINO(BaseModel, name="gino"):
    """Geometry-informed Neural Operator
    
    Inherits from:
    - BaseModel: Provides model registration, checkpoint loading/saving
    - nn.Module (Flax): Provides JAX/Flax functionality
    """
```

### Constructor Parameters (with GINO_Small3d defaults)

```python
GINO(
    # Basic I/O
    in_channels: int = 0,                    # From data_channels (0 for GINO_Small3d)
    out_channels: int = 1,
    
    # Input GNO parameters
    gno_coord_dim: int = 3,                  # 3D geometry
    gno_coord_embed_dim: int = 16,
    in_gno_radius: float = 0.033,
    in_gno_transform_type: str = "linear",
    in_gno_pos_embed_type: str = "transformer",
    in_gno_channel_mlp_hidden_layers: tuple = (80, 80, 80),
    
    # Output GNO parameters
    out_gno_radius: float = 0.033,
    out_gno_transform_type: str = "linear",
    out_gno_pos_embed_type: str = "transformer",
    out_gno_channel_mlp_hidden_layers: tuple = (512, 256),
    gno_weighting_function: str = None,
    gno_weight_function_scale: float = 1,
    
    # FNO parameters
    fno_n_modes: tuple = (16, 16, 16),       # Query resolution
    fno_hidden_channels: int = 64,
    fno_n_layers: int = 4,
    fno_lifting_channel_ratio: int = 2,      # lifting_channels = 2 * 64 = 128
    fno_use_channel_mlp: bool = True,
    fno_channel_mlp_expansion: float = 0.5,
    fno_norm: str = None,
    fno_domain_padding: float = 0.125,
    fno_factorization: str = "tucker",
    fno_rank: float = 0.4,
    
    # Latent features
    latent_feature_channels: int = 1,
    
    # Other parameters
    projection_channel_ratio: int = 4,       # projection_channels = 4 * 64 = 256
    gno_use_open3d: bool = True,
)
```

### BaseModel.__new__() Interception

Before GINO.__init__() runs, BaseModel.__new__() intercepts and:

```python
def __new__(cls, *args, **kwargs):
    # 1. Inspect method signature
    sig = inspect.signature(cls)
    
    # 2. Validate and fill in default parameters
    for key, value in sig.parameters.items():
        if (value.default is not inspect._empty) and (key not in kwargs):
            kwargs[key] = value.default
    
    # 3. Store initialization kwargs
    kwargs["_version"] = cls._version
    kwargs["_name"] = cls._name
    kwargs["args"] = args
    
    instance = super().__new__(cls)
    instance._init_kwargs = kwargs  # Stored for checkpoint saving/loading
    
    return instance
```

**Result**: Instance has `_init_kwargs` attribute containing all initialization parameters.

---

## Phase 3: GINO.setup() - Component Initialization

The `setup()` method is called by Flax and initializes all sub-modules. This is where the actual architecture gets built.

### 3.1 Calculate Intermediate Dimensions

```python
fno_hidden_channels = 64

# Lifting layer dimensions
lifting_channels = fno_lifting_channel_ratio * fno_hidden_channels
                 = 2 * 64 = 128

# Input GNO output channels
if in_gno_transform_type == "linear":  # Yes, it is
    in_gno_out_channels = fno_in_channels = 3  # fno_in_channels
else:
    in_gno_out_channels = in_channels

# FNO input channels calculation
fno_in_channels = 3  # from in_gno_out_channels
if latent_feature_channels is not None:  # Yes, it's 1
    fno_in_channels += latent_feature_channels
    fno_in_channels = 3 + 1 = 4

self._fno_in_channels = 4

# Projection layer dimensions
projection_channels = projection_channel_ratio * fno_hidden_channels
                    = 4 * 64 = 256
```

### 3.2 Initialize Input GNO Block

```python
self.gno_in = GNOBlock(
    in_channels=0,                              # From config (in_channels)
    out_channels=3,                             # fno_in_channels (without latent)
    coord_dim=3,
    pos_embedding_type="transformer",           # in_gno_pos_embed_type
    pos_embedding_channels=32,                  # gno_embed_channels
    pos_embedding_max_positions=10000,
    radius=0.033,                               # in_gno_radius
    reduction="mean",
    weighting_fn=None,
    channel_mlp_layers=[80, 80, 80],           # in_gno_channel_mlp_hidden_layers
    channel_mlp_non_linearity=nn.gelu,
    transform_type="linear",                    # in_gno_transform_type
    use_torch_scatter_reduce=False,
    use_open3d_neighbor_search=True,
)

# GNOBlock components:
# - Transformer-based position embeddings
# - Channel MLP layers
# - Neighbor search (via Open3D)
# - Graph convolution operations
```

**GNOBlock Purpose**: Encode input geometry (mesh vertices + function values) into a latent representation.

### 3.3 Initialize Lifting MLP

```python
self.lifting = ChannelMLP(
    in_channels=4,                              # fno_in_channels (3 + 1 latent)
    hidden_channels=128,                        # lifting_channels
    out_channels=64,                            # fno_hidden_channels
    n_layers=2,
)

# ChannelMLP structure:
# Layer 1: Linear(4, 128) → GELU
# Layer 2: Linear(128, 64)
```

**Purpose**: Project concatenated input and latent features to FNO hidden space.

### 3.4 Initialize FNO Blocks

```python
self.fno_blocks = FNOBlocks(
    n_modes=(16, 16, 16),                       # fno_n_modes (3D)
    in_channels=64,                             # fno_hidden_channels
    out_channels=64,                            # fno_hidden_channels
    n_layers=4,                                 # fno_n_layers
    resolution_scaling_factor=None,
    fno_block_precision="full",
    use_channel_mlp=True,
    channel_mlp_expansion=0.5,                  # 32 hidden channels
    channel_mlp_dropout=0,
    non_linearity=nn.gelu,
    stabilizer=None,
    norm=None,                                  # fno_norm
    ada_in_features=None,                       # ada_in_dim (no ada_in)
    preactivation=False,
    fno_skip="linear",
    channel_mlp_skip="soft-gating",
    separable=False,
    factorization="tucker",                     # fno_factorization
    rank=0.4,                                   # fno_rank
    fixed_rank_modes=False,
    implementation="factorized",
    decomposition_kwargs={},
    conv_module=SpectralConv,
    enforce_hermitian_symmetry=True,
)

# FNOBlocks structure (4 layers):
# Each layer contains:
#   - Spectral Convolution: 64 → 64 channels (Tucker-factorized)
#   - Channel MLP: 64 → 32 (expansion 0.5) → 64 (with soft-gating skip)
#   - GELU activations
```

**Purpose**: Spectral processing in the latent space using Fourier modes.

### 3.5 Initialize Output GNO Block

```python
# Determine weighting function
if gno_weighting_function is None:  # Yes, default is None
    weight_fn = None
else:
    weight_fn = dispatch_weighting_fn(...)

self.gno_out = GNOBlock(
    in_channels=64,                             # fno_hidden_channels
    out_channels=64,                            # fno_hidden_channels
    coord_dim=3,
    radius=0.033,                               # out_gno_radius
    reduction="sum",                            # Unlike input GNO ("mean")
    weighting_fn=weight_fn,                     # None by default
    pos_embedding_type="transformer",           # out_gno_pos_embed_type
    pos_embedding_channels=32,                  # gno_embed_channels
    pos_embedding_max_positions=10000,
    channel_mlp_layers=[512, 256],             # out_gno_channel_mlp_hidden_layers
    channel_mlp_non_linearity=nn.gelu,
    transform_type="linear",                    # out_gno_transform_type
    use_torch_scatter_reduce=False,
    use_open3d_neighbor_search=True,
)

# GNOBlock structure (output):
# - Larger channel MLP (512, 256) vs input (80, 80, 80)
# - Sum reduction (vs mean for input)
# - Decodes latent features back to output mesh
```

**Purpose**: Decode latent features onto output query points (mesh vertices).

### 3.6 Initialize Projection MLP

```python
self.projection = ChannelMLP(
    in_channels=64,                             # fno_hidden_channels
    out_channels=1,                             # out_channels
    hidden_channels=256,                        # projection_channels
    n_layers=2,
    n_dim=1,                                    # 1D processing (per-point)
    non_linearity=nn.gelu,
)

# ChannelMLP structure (1D):
# Layer 1: Linear(64, 256) → GELU
# Layer 2: Linear(256, 1)
```

**Purpose**: Project FNO hidden features to output channels (pressure).

---

## Phase 4: Model Instance Structure

### What gets stored in the model instance:

```python
model = GINO(
    in_channels=0,
    out_channels=1,
    ...
)

model attributes:
├─ _init_kwargs: dict                  # All initialization parameters
│  └─ Contains all parameters for checkpoint loading
│
├─ _version: str = "0.1.0"
├─ _name: str = "gino"
│
├─ gno_in: GNOBlock                   # Input geometry encoder
│  ├─ pos_embed: SinusoidalEmbedding
│  ├─ channel_mlp: ChannelMLP
│  └─ neighbor_search: function
│
├─ lifting: ChannelMLP                # 4 → 128 → 64
│
├─ fno_blocks: FNOBlocks              # 4 spectral convolution layers
│  └─ [4 layers] each with:
│     ├─ spectral_conv: SpectralConv (Tucker-factorized)
│     └─ channel_mlp: ChannelMLP
│
├─ gno_out: GNOBlock                  # Output decoder
│  ├─ pos_embed: SinusoidalEmbedding
│  ├─ channel_mlp: ChannelMLP
│  └─ neighbor_search: function
│
├─ projection: ChannelMLP             # 64 → 256 → 1
│
└─ adain_pos_embed: None              # Not used (fno_norm != "ada_in")
```

### Flax Module Registration

Flax automatically wraps all sub-modules in the model:

```python
# When .setup() completes, Flax has:
model = GINO(...)
# Flax creates sub-module collections:
model.module_dict = {
    'gno_in': GNOBlock instance,
    'lifting': ChannelMLP instance,
    'fno_blocks': FNOBlocks instance,
    'gno_out': GNOBlock instance,
    'projection': ChannelMLP instance,
}
```

---

## Phase 5: Parameter Initialization (Not Yet Done)

**Important**: At this point, the model has been instantiated but **parameters are NOT initialized yet**.

Parameters are initialized when you call:
```python
params = model.init(
    rng,
    input_geom=...,
    latent_queries=...,
    output_queries=...,
)
```

### Expected Parameter Shapes (once initialized)

```
model parameters structure:
params = {
    'gno_in': {
        'pos_embed': {...},
        'channel_mlp': {
            'Dense_0': {'kernel': (input_dim, 80), 'bias': (80,)},
            'Dense_1': {'kernel': (80, 80), 'bias': (80,)},
            'Dense_2': {'kernel': (80, 3), 'bias': (3,)},
        }
    },
    'lifting': {
        'Dense_0': {'kernel': (4, 128), 'bias': (128,)},
        'Dense_1': {'kernel': (128, 64), 'bias': (64,)},
    },
    'fno_blocks': {
        'FNOBlock_0': {
            'spectral_conv': {'weight': (16, 16, 16, 64, 64)},
            'channel_mlp': {...},
        },
        'FNOBlock_1': {...},
        'FNOBlock_2': {...},
        'FNOBlock_3': {...},
    },
    'gno_out': {
        'pos_embed': {...},
        'channel_mlp': {
            'Dense_0': {'kernel': (64, 512), 'bias': (512,)},
            'Dense_1': {'kernel': (512, 256), 'bias': (256,)},
            'Dense_2': {'kernel': (256, 64), 'bias': (64,)},
        }
    },
    'projection': {
        'Dense_0': {'kernel': (64, 256), 'bias': (256,)},
        'Dense_1': {'kernel': (256, 1), 'bias': (1,)},
    },
}
```

---

## Phase 6: Forward Pass Signature

### __call__() Method

```python
def __call__(
    self,
    input_geom: jnp.ndarray,           # (1, n_vertices, 3) - mesh vertices
    latent_queries: jnp.ndarray,       # (1, 32, 32, 32, 3) - query grid
    output_queries: jnp.ndarray,       # (1, n_output_vertices, 3) - output points
    x: jnp.ndarray = None,             # (batch, n_vertices, in_channels) - input function
    latent_features: jnp.ndarray = None,  # (batch, 32, 32, 32, 1) - latent features
    ada_in: jnp.ndarray = None,        # Adaptive instance norm (not used by default)
    neighbors_in: dict = None,         # Precomputed input neighbors
    neighbors_out: dict = None,        # Precomputed output neighbors
    **kwargs
) -> jnp.ndarray:
    """
    Returns:
        out: jnp.ndarray of shape (batch, n_output_vertices, out_channels)
             For pressure prediction: (batch, n_vertices, 1)
    """
```

### Typical Data Shapes for Car CFD

```
Given CarCFD sample from train_data[i]:

Input:
  - input_geom (vertices): (N, 3), e.g., (5000, 3)
  - latent_queries: (32, 32, 32, 3) - from config.data.sdf_query_resolution
  - output_queries: (N, 3), e.g., (5000, 3) - same as vertices
  - x (distance SDF): (32, 32, 32, 1) - can be fed as input_geom aux
  - latent_features (distance): (1, 32, 32, 32, 1) - optional
  - neighbors_in: dict with precomputed neighbor indices
  - neighbors_out: dict with precomputed neighbor indices

Forward pass:
  input_geom (N, 3) + x (32, 32, 32, 1)
  ↓ gno_in ↓ (processes N vertices + 32³ query points)
  → (batch, 32, 32, 32, 3) + (batch, 32, 32, 32, 1) latent
  ↓ concatenate latent features ↓
  → (batch, 32, 32, 32, 4)
  ↓ lifting ↓
  → (batch, 64, 32, 32, 32)
  ↓ 4× FNO layers ↓
  → (batch, 64, 32, 32, 32)
  ↓ gno_out ↓ (decode to output vertices)
  → (batch, N, 64)
  ↓ projection ↓
  → (batch, N, 1)

Output:
  - pressure predictions: (batch, N, 1)
```

---

## Complete Data Flow Visualization

```
CONFIG
  ├─ model.model_arch: "gino"
  ├─ model.in_channels: 0 (adjusted from data_channels)
  ├─ model.out_channels: 1
  ├─ model.fno_n_modes: [16, 16, 16]
  ├─ model.fno_hidden_channels: 64
  ├─ model.latent_feature_channels: 1
  └─ ... [other parameters]
    ↓
get_model_jax(config)
    ├─ arch = "gino"
    ├─ Extract & validate model_config
    └─ Adjust in_channels (0 * (patching_levels + 1) = 0)
    ↓
GINO.__new__()
    ├─ Store all kwargs in _init_kwargs
    └─ Create instance
    ↓
GINO.setup() (called by Flax)
    ├─ Calculate dimensions:
    │  ├─ lifting: 4 → 128 → 64
    │  ├─ fno_in: 4 (3 from gno_in + 1 latent)
    │  └─ projection: 64 → 256 → 1
    │
    ├─ Initialize gno_in (GNOBlock)
    │  └─ Processes: vertices + query points → 3D latent
    │
    ├─ Initialize lifting (ChannelMLP: 4 → 128 → 64)
    │  └─ Projects: (input + latent_features) → FNO space
    │
    ├─ Initialize fno_blocks (4 spectral convolution layers)
    │  └─ Each: SpectralConv (Tucker) + ChannelMLP
    │
    ├─ Initialize gno_out (GNOBlock)
    │  └─ Decodes: latent space → output vertices
    │
    └─ Initialize projection (ChannelMLP: 64 → 256 → 1)
       └─ Projects: hidden features → output channels
    ↓
MODEL INSTANCE CREATED
    ├─ All sub-modules initialized
    ├─ Parameters NOT yet initialized (wait for model.init())
    └─ Ready for: init(), apply(), and training
```

---

## Parameter Count Summary (Approximate)

```
Component              In        Out       Params (approx)
─────────────────────────────────────────────────────────
gno_in pos_embed       -         32        ~10K
gno_in channel_mlp     0→80→80→3 -         ~26K
lifting                4→128→64  -         ~28K
fno_blocks[0-3]        64→64     -         ~1.2M (Tucker factorized)
gno_out pos_embed      -         32        ~10K
gno_out channel_mlp    64→512→256→64 -     ~220K
projection             64→256→1  -         ~16K
─────────────────────────────────────────────────────────
TOTAL ESTIMATED:                           ~1.5M parameters
```

---

## Key Characteristics

### Architecture Type
- **Hybrid Graph-Spectral**: Combines geometric graph processing (GNO) with spectral methods (FNO)
- **3D-Capable**: Designed for 3D point clouds and meshes

### Strengths
1. **Geometry-Aware**: Directly processes mesh vertices and topology
2. **Global Integration**: FNO layers provide global receptive field
3. **Flexible**: Can handle variable mesh sizes and irregular point clouds
4. **Factorized**: Tucker decomposition reduces parameters

### Typical Use
- **PDE Operators**: Learn mapping from geometry/boundary conditions to outputs
- **Car CFD**: Predict pressure from mesh geometry and inlet velocity
- **Mesh-based Problems**: Any problem defined on unstructured meshes

---

## Usage After Initialization

```python
# After model is created
model = get_model_jax(config)

# Initialize parameters
import jax
key = jax.random.PRNGKey(0)
params = model.init(
    key,
    input_geom=jnp.zeros((1, 100, 3)),
    latent_queries=jnp.zeros((1, 32, 32, 32, 3)),
    output_queries=jnp.zeros((1, 100, 3)),
)

# Use in forward pass
from functools import partial
apply_fn = partial(model.apply, params)

output = apply_fn(
    input_geom=vertices,      # (batch, n_vertices, 3)
    latent_queries=query_grid, # (batch, 32, 32, 32, 3)
    output_queries=output_points,
    x=input_function,         # (batch, n_vertices, 0) - dummy for in_channels=0
    latent_features=sdf_distance,
)
# output shape: (batch, n_vertices, 1) for pressure prediction
```

---

## Configuration Changes Visible to Model

```python
# If you modified these config values:

# 1. Data channels (affects in_channels)
config.model.data_channels = 3  # Would affect input dimension
↑ Would require different handling in training loop

# 2. Query resolution (affects FNO modes)
config.data.sdf_query_resolution = 64  # Would expect 64x64x64 queries
↑ Model would still work but expects different latent_queries shape

# 3. FNO hidden channels (scales entire model)
config.model.fno_hidden_channels = 128  # Would double most layer sizes
↑ Parameters would increase significantly

# 4. Latent feature channels (affects lifting input)
config.model.latent_feature_channels = 8  # Would change fno_in_channels to 11
↑ Lifting layer dimensions would change

# For current GINO_Small3d defaults: all above are default values
```

---

## Notes

1. **Flax Linen Module**: This is a Flax module, not PyTorch. Uses functional JAX paradigm.
2. **No Parameters Yet**: Model is instantiated but params are uninitialized until `model.init()` is called.
3. **Dynamic Shapes**: GNO blocks can handle variable input sizes (adaptive).
4. **Functional**: No state mutation; JAX arrays are immutable.
5. **Neighbor Search**: Can be precomputed for efficiency (see `model.precompute_neighbors()`).


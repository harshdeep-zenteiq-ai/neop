"""
Training script for GINO on Car CFD dataset.

This script trains a Graph Neural Operator (GINO) on computational fluid
dynamics data for car pressure prediction. The model learns to predict
pressure fields from geometric inputs using graph-based representations.
"""

from timeit import default_timer

# import torch
import wandb
import sys

from neuralop import get_model_jax 

import os
# Prevents memory fragmentation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Stops JAX from blindly reserving 90% of your VRAM on startup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ["JAX_ENABLE_X64"] = "True" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from neuralop.data.datasets.car_cfd_dataset_jax import CarCFDDatasetjax
from neuralop.losses.data_losses_jax import LpLoss
from neuralop.training.adamw_jax import AdamW
from neuralop.training.trainer_jax import Trainer
from neuralop.data.transforms.data_processors_jax import DataProcessor
from copy import deepcopy
import numpy as np 
# query points is [sdf_query_resolution] * 3 (taken from config ahmed)
# Read the configuration
config_name = "cfd"
from zencfg import make_config_from_cli
import sys

sys.path.insert(0, "../")
from config.gino_carcfd_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()

# Distributed training setup, if enabled
# device, is_logger = setup(config)

# Model architecture adjustment for query resolution
if config.data.sdf_query_resolution < config.model.fno_n_modes[0]:
    config.model.fno_n_modes = [config.data.sdf_query_resolution] * 3

# WandB logging configuration
# wandb_init_args = {}
# config_name = "car-pressure"
# if config.wandb.log and is_logger:
#     wandb.login(key=get_wandb_api_key())
#     if config.wandb.name:
#         wandb_name = config.wandb.name
#     else:
#         wandb_name = "_".join(
#             f"{var}" for var in [config_name, config.data.sdf_query_resolution]
#         )

#     wandb_init_args = dict(
#         config=config,
#         name=wandb_name,
#         group=config.wandb.group,
#         project=config.wandb.project,
#         entity=config.wandb.entity,
#     )

#     if config.wandb.sweep:
#         for key in wandb.config.keys():
#             config.params[key] = wandb.config[key]
#     wandb.init(**wandb_init_args)

    root: str = "~/data/car-pressure-data/processed-car-pressure-data"

data_module_jax = CarCFDDatasetjax(
    root_dir=config.data.root,
    query_res=[config.data.sdf_query_resolution] * 3,
    n_train=config.data.n_train,
    n_test=config.data.n_test,
    download=config.data.download,
)

# Model initialization
model = get_model_jax(config)

# from functools import partial
# import jax.nn
# _gelu_exact = partial(jax.nn.gelu, approximate=False)

# Construct the JAX GINO with exact (erf-based) gelu to match torch's F.gelu default.
# get_model_jax dispatches via config.model.model_arch; we replicate its logic but
# inject the non_linearity overrides directly.
# from neuralop.models.gino_jax import GINO as _JaxGINO
# _model_kwargs = config.model.copy()
# _model_kwargs.pop("model_arch", None)
# _model_kwargs["in_channels"] = _model_kwargs.pop("data_channels")
# _model_kwargs["fno_non_linearity"] = _gelu_exact
# _model_kwargs["gno_channel_mlp_non_linearity"] = _gelu_exact
# model = _JaxGINO(**_model_kwargs)

# JAX/Flax requires explicit parameter initialization + gradient-based updates via optax
import optax
import numpy as np_cpu

def pad_neighbor_data(data_lists):
    """Pad neighbor arrays across all samples to a fixed maximum length.

    Converts variable-length ``neighbors_index`` / ``neighbors_row_splits``
    into fixed-shape arrays that are compatible with ``jax.jit``.

    For each sample and each of ``neighbors_in`` / ``neighbors_out``:
    - Computes ``segment_ids`` from ``row_splits`` (one ID per edge).
    - Pads ``neighbors_index`` to ``max_edges`` with 0 (safe dummy index).
    - Pads ``segment_ids``     to ``max_edges`` with ``n_out`` (dummy segment
      that is discarded by ``segment_csr``).
    - Stores ``counts`` (real edges per output node) for mean reduction.
    - Drops ``neighbors_row_splits`` (no longer needed at runtime).

    Parameters
    ----------
    data_lists : list of lists
        One or more ``dataset.data_list`` collections to pad jointly so that
        train and test sets share the same ``max_edges`` value.

    Returns
    -------
    max_in_edges, max_out_edges : int
    """
    all_samples = [s for dl in data_lists for s in dl]

    max_in  = max(s['neighbors_in']['neighbors_index'].shape[0]  for s in all_samples)
    max_out = max(s['neighbors_out']['neighbors_index'].shape[0] for s in all_samples)
    # print(f"[pad_neighbor_data] max_in_edges={max_in}, max_out_edges={max_out}")

    for s in all_samples:
        for key, max_edges in [('neighbors_in', max_in), ('neighbors_out', max_out)]:
            nbrs   = s[key]
            idx    = np_cpu.asarray(nbrs['neighbors_index'])        # [n_real]
            splits = np_cpu.asarray(nbrs['neighbors_row_splits'])   # [n_out+1]
            n_real = idx.shape[0]
            n_out  = splits.shape[0] - 1

            counts     = splits[1:] - splits[:-1]                          # [n_out]
            seg_ids    = np_cpu.repeat(np_cpu.arange(n_out), counts)       # [n_real]
            pad_len    = max_edges - n_real

            idx_padded    = np_cpu.concatenate([idx,     np_cpu.zeros(pad_len, dtype=idx.dtype)])
            seg_ids_padded = np_cpu.concatenate([seg_ids, np_cpu.full(pad_len, n_out, dtype=seg_ids.dtype)])

            s[key] = {
                'neighbors_index': idx_padded,     # [max_edges] - keep as numpy
                'segment_ids':     seg_ids_padded, # [max_edges] - keep as numpy
                'counts':          counts,          # [n_out] - keep as numpy
            }


    return max_in, max_out

class FlaxModelWrapper:
    """Wraps a Flax module with optax optimization.

    Two training modes are supported:
    - ``train_step``: per-sample JIT call (fallback).
    - ``train_epoch_scan``: entire epoch compiled as one XLA program via
      ``jax.lax.scan``, eliminating all Python-loop overhead (preferred).
    """

    _MODEL_KEYS = frozenset({
        'input_geom', 'latent_queries', 'output_queries',
        'x', 'latent_features', 'ada_in', 'neighbors_in', 'neighbors_out',
    })

    def __init__(self, flax_module, learning_rate=1e-3, weight_decay=1e-4, key=None):
        self.module       = flax_module
        self.learning_rate = learning_rate
        self.params       = None
        self.opt_state    = None
        self.key          = key if key is not None else jax.random.PRNGKey(0)
        # self.tx           = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        self.tx = optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate, weight_decay=weight_decay)
        self._initialized = False
        self._jit_fn      = None   # per-step JIT cache
        # self._scan_fn     = None   # epoch-level scan JIT cache

    def _init(self, sample_kwargs):
        """Initialize params and optimizer state from the first real sample."""
        model_inputs = {k: v for k, v in sample_kwargs.items() if k in self._MODEL_KEYS}
        self.params    = self.module.init(self.key, **model_inputs)
        self.opt_state = self.tx.init(self.params)
        self._initialized = True

    def set_lr(self, new_lr):
        self.learning_rate = new_lr
        self.opt_state.hyperparams['learning_rate'] = jnp.array(new_lr) 

    def _build_jit_fn(self, loss_fn):
        module = self.module
        tx     = self.tx

        @jax.jit
        def _step(params, opt_state, model_inputs, y):
            def forward_loss(p):
                out = module.apply(p, **model_inputs)
                return loss_fn(out, y=y)
            loss, grads = jax.value_and_grad(forward_loss)(params)
            updates, new_opt = tx.update(grads, opt_state, params)
            return loss, optax.apply_updates(params, updates), new_opt

        return _step


    def __call__(self, **kwargs):
        """Forward-only pass (eval)."""
        if not self._initialized:
            self._init(kwargs)
        model_inputs = {k: v for k, v in kwargs.items() if k in self._MODEL_KEYS}
        return self.module.apply(self.params, **model_inputs)

    def train_step(self, kwargs, loss_fn):
        """Per-sample JIT step (fallback)."""
        if not self._initialized:
            self._init(kwargs)
        if self._jit_fn is None:
            self._jit_fn = self._build_jit_fn(loss_fn)
        model_inputs = {k: v for k, v in kwargs.items() if k in self._MODEL_KEYS}
        loss, self.params, self.opt_state = self._jit_fn(
            self.params, self.opt_state, model_inputs, kwargs['y']
        )
        return loss

    def precompute_neighbors(self, *args, **kwargs):
        if hasattr(self.module, 'precompute_neighbors'):
            return self.module.precompute_neighbors(*args, **kwargs)

# Wrap the model
model = FlaxModelWrapper(
    model,
    learning_rate=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

# Precompute neighbor search results for all samples before training.
_sample0 = data_module_jax.train_data.data_list[0]
output_n_points = _sample0["press"].shape[1]
tt = default_timer()
model.precompute_neighbors(data_module_jax.train_data, output_n_points=output_n_points,
                           log_dir="./neighbor_logs", split="train")
model.precompute_neighbors(data_module_jax.test_data,  output_n_points=output_n_points,
                           log_dir="./neighbor_logs", split="test")
elapsed = default_timer() - tt
# print('neighbor precomputation done in jax version with time', elapsed)

# Pad all neighbor arrays to a fixed maximum length so shapes are static
# across batches — required for jax.jit to compile once and reuse.
pad_neighbor_data([
    data_module_jax.train_data.data_list,
    data_module_jax.test_data.data_list,
])

# Create data loaders AFTER padding so batches include padded neighbor dicts
from neuralop.training.trainer_jax import SimpleDataLoader
train_loader_jax = SimpleDataLoader(data_module_jax.train_loader(batch_size=1, shuffle=False))
test_loader_jax  = SimpleDataLoader(data_module_jax.test_loader(batch_size=1, shuffle=False))

# Create the optimizer
optimizer = AdamW(
    params=[],  # JAX models handle parameters differently
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

# Simple scheduler implementations for JAX
class SchedulerBase:
    """Base scheduler class for JAX"""
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.step_count = 0

    def step(self, metric=None):
        self.step_count += 1

class ReduceLROnPlateau(SchedulerBase):
    """Simple implementation of ReduceLROnPlateau for JAX"""
    def __init__(self, optimizer, factor=0.1, patience=10, mode="min"):
        super().__init__(optimizer)
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.best_metric = None
        self.patience_counter = 0

    def step(self, metric=None):
        super().step()
        if metric is not None:
            if self.best_metric is None:
                self.best_metric = metric
            elif (self.mode == "min" and metric < self.best_metric) or (self.mode == "max" and metric > self.best_metric):
                self.best_metric = metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.optimizer.learning_rate *= self.factor
                    self.patience_counter = 0

class CosineAnnealingLR(SchedulerBase):
    """Simple implementation of CosineAnnealingLR for JAX"""
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.learning_rate

    def step(self, metric=None):
        super().step()
        self.optimizer.learning_rate = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + jnp.cos(jnp.pi * self.step_count / self.T_max)
        ) / 2

class StepLR(SchedulerBase):
    """Simple implementation of StepLR for JAX"""
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.learning_rate

    def step(self, metric=None):
        super().step()
        if self.step_count % self.step_size == 0:
            # self.optimizer.learning_rate = self.base_lr * (self.gamma ** (self.step_count // self.step_size))
            self.optimizer.set_lr(self.base_lr * (self.gamma ** (self.step_count // self.step_size)))

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    # scheduler = StepLR(
    #     optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    # )
    scheduler = StepLR(
        model, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got {config.opt.scheduler=}")


l2loss = LpLoss(d=2, p=2)

if config.opt.training_loss == "l2":
    train_loss_fn = l2loss
else:
    raise ValueError(f"Got {config.opt.training_loss=}")

if config.opt.testing_loss == "l2":
    test_loss_fn = l2loss
else:
    raise ValueError(f"Got {config.opt.testing_loss=}")

# Custom data processor for GINO CFD training
class GINOCFDDataProcessor(DataProcessor):
    """
    Data processor for GINO training on CFD car-pressure dataset.

    This processor handles the conversion of CFD mesh data into the format
    expected by the GINO model, including graph construction and
    feature extraction from geometric inputs.
    """

    def __init__(self, normalizer, device="cpu"):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None
        self.training = True

    def preprocess(self, sample):
        """
        Convert CFD mesh data into GINO input format.

        Transforms the data dictionary from MeshDataModule's DictDataset
        into the form expected by the GINO model.
        """

        in_p = np.asarray(sample["vertices"])

        latent_queries = np.asarray(sample["query_points"])

        out_p = np.asarray(sample["vertices"])

        f = np.asarray(sample["distance"])

        # Output pressure data — mirror PyTorch: press.squeeze(0).unsqueeze(-1)
        # Loader produces press of shape (1, 1, n_press); after squeeze+unsqueeze
        # we get (1, n_press, 1), matching the PyTorch reference exactly.
        press = np.asarray(sample["press"])
        truth = np.expand_dims(np.squeeze(press, axis=0), axis=-1)  # (1, n_press, 1)

        # Truncate out_p to the first n_press vertices to match the pressure data.
        # out_p still has the batch dim at axis 0, so n_vertices is at axis 1.
        output_vertices = truth.shape[1]
        if out_p.shape[1] > output_vertices:
            out_p = out_p[:, :output_vertices, :]
        
        # Create new sample dict with only model inputs
        batch_dict = dict(
            input_geom=in_p,
            latent_queries=latent_queries,
            output_queries=out_p,
            latent_features=f,
            y=truth,
            x=None,
        )

        # Convert precomputed neighbor indices to JAX arrays (no batch dim to squeeze)
        for key in ("neighbors_in", "neighbors_out"):
            if key in sample and sample[key] is not None:
                batch_dict[key] = {
                    k: np.asarray(v)
                    for k, v in sample[key].items()
                }

        return batch_dict

    def postprocess(self, out, sample):
        """
        Postprocess model output and ground truth data.

        Applies inverse normalization to both predictions and ground truth
        when not in training mode.
        """
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = jnp.asarray(sample["y"])
            if y.ndim > 1:
                y = jnp.squeeze(y, axis=0)
            y = self.normalizer.inverse_transform(y)
            sample["y"] = y

        return out, sample

    def to(self, device):
        self.device = device
        if hasattr(self.normalizer, 'to'):
            self.normalizer = self.normalizer.to(device)
        return self

    def wrap(self, model):
        self.model = model
        return self

    def __call__(self, sample, training=True):
        """
        Complete forward pass through the data processor and model.
        """
        self.training = training
        sample = self.preprocess(sample)
        # Filter sample to only include keys the model expects
        model_keys = {'input_geom', 'latent_queries', 'output_queries',
                      'x', 'latent_features', 'ada_in', 'neighbors_in', 'neighbors_out', 'y'}
        model_input = {k: v for k, v in sample.items() if k in model_keys}
        out = self.model(**model_input)
        out, sample = self.postprocess(out, sample)
        return out, sample


# Initialize data processor
output_encoder = deepcopy(data_module_jax.normalizers["press"])
data_processor = GINOCFDDataProcessor(normalizer=output_encoder, device=config.get("device", "cpu"))

# =========================================================================
# Initialize JAX/Flax weights from a freshly-constructed PyTorch GINO so
# both implementations start training from byte-identical params.
#
# Order matters:
#   1. build PyTorch model with the SAME config (same arch kwargs => same
#      parameter shapes).
#   2. run one real preprocessed sample through Flax.init to materialize
#      the canonical Flax param pytree (gives us the shapes + tree structure).
#   3. walk the torch state_dict and overwrite every Flax leaf using
#      check_gino_parity.copy_torch_to_flax.
#   4. assign onto the FlaxModelWrapper and pre-build the optax opt_state
#      so the lazy `_init` inside `train_step` is bypassed.
# =========================================================================

# import torch
# from neuralop import get_model as get_model_torch   # PyTorch GINO factory
# # check_gino_parity.py lives in the project root; '../' is already on sys.path
# # (added at line 41 for `from config.gino_carcfd_config import Default`)
# from check_gino_parity import copy_torch_to_flax
# # Step 1: build a torch GINO with the same config
# torch.manual_seed(0)
# torch_model = get_model_torch(config)
# torch_model.gno_in.integral_transform.use_torch_scatter  = False
# torch_model.gno_out.integral_transform.use_torch_scatter = False
# torch_model.eval()

# # Step 2: grab one real sample, run the processor preprocess so model_inputs
# # match exactly what the trainer feeds in (including padded neighbors).
# _sample0_jax = next(iter(train_loader_jax))
# _sample0_jax = data_processor.preprocess(_sample0_jax)
# _model_inputs = {
#     k: v for k, v in _sample0_jax.items()
#     if k in FlaxModelWrapper._MODEL_KEYS
# }

# # Step 3: materialize Flax param pytree (random init), then overwrite with torch values.
# _init_variables = model.module.init(model.key, **_model_inputs)   # {'params': pytree}
# _inner = copy_torch_to_flax(torch_model, _init_variables['params'])
# transferred_variables = {'params': _inner}

# # Sanity check: every leaf should still be present.
# import jax as _jax
# n_before = len(_jax.tree_util.tree_leaves(_init_variables['params']))
# n_after  = len(_jax.tree_util.tree_leaves(_inner))
# assert n_before == n_after, (
#     f"Param leaf count changed during transfer: {n_before} -> {n_after}. "
#     f"copy_torch_to_flax missed some Flax paths."
# )

# # Step 4: pre-seed the wrapper so its lazy _init() is skipped on the first batch.
# model.params       = transferred_variables
# model.opt_state    = model.tx.init(model.params)
# model._initialized = True

# print(f"[weight transfer] PyTorch -> Flax: {n_after} param leaves transferred.")

# # =========================================================================
# # Gate A: forward-parity check on the actual training sample.
# #
# # Pushes the same preprocessed sample through the freshly-built torch_model
# # AND the weight-transferred Flax model. Compares raw outputs. If they
# # diverge beyond float32 noise, refuses to start the 301-epoch run — that
# # usually means InstanceNorm reduction axes differ, padded-neighbor handling
# # is off, or the Tucker contraction order differs. Cheap to run (~1 second).
# # =========================================================================

# # pad_neighbor_data() rewrote each sample's neighbors dict to the JAX-side
# # padded format {neighbors_index (padded), segment_ids (padded), counts (real)}.
# # Torch's GINO expects {neighbors_index, neighbors_row_splits}. We reconstruct
# # the torch format here purely for this one parity call.
# def _padded_jax_to_torch_nbrs(padded):
#     counts = np.asarray(padded['counts']).astype(np.int64)
#     n_real = int(counts.sum())
#     idx_real = np.asarray(padded['neighbors_index'])[:n_real].astype(np.int64)
#     row_splits = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
#     return {
#         'neighbors_index':       torch.from_numpy(idx_real),
#         'neighbors_row_splits':  torch.from_numpy(row_splits),
#     }

# # ---- Torch forward on the same preprocessed sample ----------------------
# _in_p_t  = torch.from_numpy(np.asarray(_sample0_jax['input_geom']).astype(np.float32))
# _lat_q_t = torch.from_numpy(np.asarray(_sample0_jax['latent_queries']).astype(np.float32))
# _out_q_t = torch.from_numpy(np.asarray(_sample0_jax['output_queries']).astype(np.float32))
# _lat_f_t = torch.from_numpy(np.asarray(_sample0_jax['latent_features']).astype(np.float32))
# _x_v     = _sample0_jax.get('x', None)
# _x_t     = None if _x_v is None else torch.from_numpy(np.asarray(_x_v).astype(np.float32))

# _nbrs_in_t  = _padded_jax_to_torch_nbrs(_sample0_jax['neighbors_in'])
# _nbrs_out_t = _padded_jax_to_torch_nbrs(_sample0_jax['neighbors_out'])

# with torch.no_grad():
#     _out_torch = torch_model(
#         input_geom=_in_p_t,
#         latent_queries=_lat_q_t,
#         output_queries=_out_q_t,
#         x=_x_t,
#         latent_features=_lat_f_t,
#         ada_in=None,
#         neighbors_in=_nbrs_in_t,
#         neighbors_out=_nbrs_out_t,
#     )
# _out_torch_np = _out_torch.detach().cpu().numpy()

# # ---- JAX forward using the transferred params via the wrapper ----------
# # FlaxModelWrapper.__call__ applies the module with self.params, which is
# # the dict we just overwrote with torch values. No data-processor postprocess
# # is invoked here — we want raw model output for the comparison.
# _out_jax_arr = model(**_model_inputs)
# _out_jax_np  = np.asarray(_out_jax_arr)

# # ---- Compare ------------------------------------------------------------
# _diff = np.abs(_out_torch_np - _out_jax_np)
# print()
# print("=" * 70)
# print("[Gate A] Forward parity check on first preprocessed training sample")
# print("=" * 70)
# print(f"  torch out shape={_out_torch_np.shape}  dtype={_out_torch_np.dtype}")
# print(f"  jax   out shape={_out_jax_np.shape}    dtype={_out_jax_np.dtype}")
# print(f"  torch range=[{_out_torch_np.min():.4f}, {_out_torch_np.max():.4f}]   "
#       f"mean={_out_torch_np.mean():.4e}   std={_out_torch_np.std():.4e}")
# print(f"  jax   range=[{_out_jax_np.min():.4f}, {_out_jax_np.max():.4f}]   "
#       f"mean={_out_jax_np.mean():.4e}   std={_out_jax_np.std():.4e}")
# print(f"  max_abs  = {_diff.max():.3e}")
# print(f"  mean_abs = {_diff.mean():.3e}")

# _GATE_A_TOL_MAX  = 5e-4
# _GATE_A_TOL_MEAN = 5e-5
# if _diff.max() < _GATE_A_TOL_MAX or _diff.mean() < _GATE_A_TOL_MEAN:
#     print(f"[Gate A] PASS  (tol_max={_GATE_A_TOL_MAX:.0e}, tol_mean={_GATE_A_TOL_MEAN:.0e})")
#     print("=" * 70)
# else:
#     print("=" * 70)
#     raise RuntimeError(
#         f"[Gate A] FAIL — forward outputs diverge beyond float32 noise.\n"
#         f"        max_abs={_diff.max():.3e}  (tol={_GATE_A_TOL_MAX:.0e})\n"
#         f"        mean_abs={_diff.mean():.3e}  (tol={_GATE_A_TOL_MEAN:.0e})\n"
#         f"        Refusing to start the 301-epoch run. Likely culprits:\n"
#         f"          - InstanceNorm reduction axes mismatch\n"
#         f"          - padded-neighbor segment_csr behavior\n"
#         f"          - Tucker einsum contraction order"
#     )

# # =========================================================================
# # Gate C: gradient parity on the same training sample.
# #
# # Runs forward + LpLoss + .backward() on the torch model, and jax.value_and_grad
# # of the same loss on the Flax model with the transferred params. Converts torch
# # grads into the Flax-shaped pytree using the same name mapping as the weight
# # transfer (copy_torch_to_flax), then diffs leaf-by-leaf.
# #
# # If forward parity holds (Gate A) but gradient parity fails, the backward
# # implementations diverge somewhere (most often: padded segment_csr scatter
# # or einsum contraction order in Tucker). The 301-epoch run will then drift.
# # =========================================================================
# from neuralop.losses.data_losses import LpLoss as _TorchLpLoss

# # ---- Torch: fresh forward with grads enabled, then backward -------------
# # Clear any stale grads — defensive; torch_model is fresh so this is a no-op.
# for _p in torch_model.parameters():
#     _p.grad = None

# # Re-run forward; Gate A used torch.no_grad() so no autograd graph survived.
# _out_torch_g = torch_model(
#     input_geom=_in_p_t,
#     latent_queries=_lat_q_t,
#     output_queries=_out_q_t,
#     x=_x_t,
#     latent_features=_lat_f_t,
#     ada_in=None,
#     neighbors_in=_nbrs_in_t,
#     neighbors_out=_nbrs_out_t,
# )

# # Target from the preprocessed sample (already shaped (1, n_out, 1)).
# _y_np = np.asarray(_sample0_jax['y']).astype(np.float32)
# _y_t  = torch.from_numpy(_y_np)

# _torch_loss_fn_local = _TorchLpLoss(d=2, p=2)
# _torch_loss = _torch_loss_fn_local(_out_torch_g, _y_t)
# _torch_loss.backward()

# # ---- JAX: value_and_grad on the same loss ------------------------------
# # train_loss_fn was created earlier in this script (LpLoss(d=2, p=2)).
# def _jax_loss_fn(_inner_params, _inputs, _y):
#     _out = model.module.apply({'params': _inner_params}, **_inputs)
#     return train_loss_fn(_out, y=_y)

# _y_jax = jnp.asarray(_y_np)
# _jax_loss, _jax_grads = jax.value_and_grad(_jax_loss_fn)(
#     model.params['params'], _model_inputs, _y_jax
# )
# # --- Diagnostic: compare ∂L/∂(input to gno_out) i.e. ∂L/∂(latent_embed) ----
# # This tells us whether the divergence is INSIDE gno_out's backward
# # (then ∂L/∂latent_embed will differ by ~3.5x) or downstream in FNO/norm
# # (then ∂L/∂latent_embed will match but FNO grads still differ).

# # Torch: register a backward hook on latent_embed during a FRESH forward.
# # Need a new forward because the previous .backward() already ran.
# _torch_latent_grad = {}
# def _hook(g):
#     _torch_latent_grad['g'] = g.detach().clone()
#     return None

# for _p in torch_model.parameters():
#     _p.grad = None

# # We need the hook on the tensor produced INSIDE GINO.forward — easiest is
# # to monkey-patch gno_out for one call so we can intercept its f_y input.
# _orig_gno_out_fwd = torch_model.gno_out.forward
# def _patched_gno_out_fwd(*a, **kw):
#     _f = kw.get('f_y', None) if 'f_y' in kw else (a[2] if len(a) > 2 else None)
#     if _f is not None and _f.requires_grad:
#         _f.register_hook(_hook)
#     return _orig_gno_out_fwd(*a, **kw)
# torch_model.gno_out.forward = _patched_gno_out_fwd

# # _torch_lifting_grad = {}
# # def _hook_lifting(g):
# #     _torch_lifting_grad['g'] = g.detach().clone()
# # torch_model.lifting.register_forward_hook(
# #     lambda m, inp, out: out.register_hook(_hook_lifting)
# # )
# _torch_lifting_grad = {}
# def _hook_lifting(g):
#     _torch_lifting_grad['g'] = g.detach().clone()
# def _lifting_fwd_hook(m, inp, out):
#     if isinstance(out, torch.Tensor):
#         out.register_hook(_hook_lifting)
# torch_model.lifting.register_forward_hook(_lifting_fwd_hook)

# _torch_fno_grads = {}
# _orig_fno_fwd = torch_model.fno_blocks.forward
# def _patched_fno_fwd(x, idx, *a, **kw):
#     out = _orig_fno_fwd(x, idx, *a, **kw)
#     def _h(g, _idx=idx): _torch_fno_grads[_idx] = g.detach().clone()
#     if isinstance(out, torch.Tensor):
#         out.register_hook(_h)
#     return out
# torch_model.fno_blocks.forward = _patched_fno_fwd

# # Hook: gradient at block[0] spectral conv output (equivalent of _s1)
# _torch_spectral0_grad = {}
# def _spectral0_fwd_hook(m, inp, out):
#     out.register_hook(lambda g: _torch_spectral0_grad.update({'g': g.detach().clone()}))
# torch_model.fno_blocks.convs[0].register_forward_hook(_spectral0_fwd_hook)

# _out2 = torch_model(
#     input_geom=_in_p_t, latent_queries=_lat_q_t, output_queries=_out_q_t,
#     x=_x_t, latent_features=_lat_f_t, ada_in=None,
#     neighbors_in=_nbrs_in_t, neighbors_out=_nbrs_out_t,
# )
# _torch_loss_fn_local(_out2, _y_t).backward()
# torch_model.gno_out.forward = _orig_gno_out_fwd  # restore
# torch_model.fno_blocks.forward = _orig_fno_fwd  # restore

# # JAX: compute ∂L/∂(latent_embed) by splitting the forward into two halves.
# # Easier: use jax.grad with a wrapper that takes latent_embed as a free
# # variable. But latent_embed is computed inside the model. Cleanest is to
# # re-run the JAX forward with `has_aux` and a wrapper. For a quick check,
# # instead compare the NORM of f_y.grad on torch with the equivalent quantity
# # we extract from _jax_grads — but _jax_grads only has param grads.
# #
# # Practical alternative: compute ∂L/∂(f_y_jax) by manually computing
# # gno_out's backward in JAX. Skip for now; the torch hook value is enough
# # to compare against a JAX run when we instrument the JAX side analogously.

# print(f"\n[diag] torch  ∂L/∂(gno_out f_y)  norm = {_torch_latent_grad['g'].norm():.6e}  "
#       f"max_abs = {_torch_latent_grad['g'].abs().max():.6e}")

# # JAX: compute ∂L/∂(latent_embed) using flax.linen.bind to split the forward.
# # bind() injects params so we can call sub-modules directly without apply().
# _bound = model.module.bind({'params': model.params['params']})

# # --- First half: gno_in → reshape → concat latent_features → latent_embedding ---
# _input_geom_sq = jnp.squeeze(jnp.asarray(_model_inputs['input_geom']), axis=0)
# _lat_q_arr     = jnp.asarray(_model_inputs['latent_queries'])
# _lat_q_sq      = jnp.squeeze(_lat_q_arr, axis=0)            # [n1, n2, n3, 3]
# _out_q_sq      = jnp.squeeze(jnp.asarray(_model_inputs['output_queries']), axis=0)
# _lat_f_arr     = (jnp.asarray(_model_inputs['latent_features'])
#                   if _model_inputs.get('latent_features') is not None else None)

# _gno_in_out = _bound.gno_in(
#     y=_input_geom_sq,
#     x=_lat_q_sq.reshape((-1, _lat_q_sq.shape[-1])),
#     f_y=None,
#     neighbors=_model_inputs['neighbors_in'],
# )
# _grid_shape = _lat_q_sq.shape[:-1]          # (n1, n2, n3)
# _in_p = _gno_in_out.reshape((1, *_grid_shape, -1))
# if _lat_f_arr is not None:
#     _in_p = jnp.concatenate((_in_p, _lat_f_arr), axis=-1)

# _le_val = _bound.latent_embedding(_in_p)    # [1, C, n1, n2, n3]

# # --- Second half: latent_embed → gno_out → projection → loss ---
# _fno_hidden = model.module.fno_hidden_channels   # 64

# def _loss_from_le(le_var):
#     # Mirror GINO.__call__ from latent_embed onward
#     _perm = (0, 2, 3, 4, 1)                      # (b,c,n1,n2,n3) → (b,n1,n2,n3,c)
#     le = jnp.transpose(le_var, _perm).reshape(1, -1, _fno_hidden)
#     out = _bound.gno_out(
#         y=_lat_q_sq.reshape((-1, _lat_q_sq.shape[-1])),
#         x=_out_q_sq,
#         f_y=le,
#         neighbors=_model_inputs['neighbors_out'],
#     )
#     out = jnp.transpose(out, (0, 2, 1))
#     out = jnp.transpose(_bound.projection(out), (0, 2, 1))
#     return train_loss_fn(out, y=_y_jax)

# _jax_le_grad = jax.grad(_loss_from_le)(_le_val)
# print(f"[diag] jax    ∂L/∂(gno_out f_y)  norm = {float(jnp.linalg.norm(_jax_le_grad)):.6e}  "
#       f"max_abs = {float(jnp.abs(_jax_le_grad).max()):.6e}")
# print(f"[diag] ratio  j/t norm = "
#       f"{float(jnp.linalg.norm(_jax_le_grad)) / float(_torch_latent_grad['g'].norm()):.3f}")

# # --- Intermediate activation grad diagnostics ---
# # 1a. ∂L/∂(lifting output) — i.e. gradient at the FNO block input
# _in_p_perm = jnp.transpose(_in_p, (0, len(_in_p.shape)-1, *range(1, len(_in_p.shape)-1)))
# _in_p_perm = jnp.transpose(_in_p, (0, _in_p.ndim-1, *range(1, _in_p.ndim-1)))
# _lifting_out = _bound.lifting(_in_p_perm)

# def _loss_from_lifting(lv):
#     # lv shape: (b, c, n1, n2, n3)  — same as _lifting_out
#     fno_out = lv
#     for _idx in range(_bound.fno_blocks.n_layers):
#         fno_out = _bound.fno_blocks(fno_out, _idx, ada_in_embeddings=None)
#     # fno_out is now latent_embed — same shape as _le_val
#     _perm = (0, 2, 3, 4, 1)
#     le = jnp.transpose(fno_out, _perm).reshape(1, -1, _fno_hidden)
#     out = _bound.gno_out(
#         y=_lat_q_sq.reshape((-1, _lat_q_sq.shape[-1])),
#         x=_out_q_sq,
#         f_y=le,
#         neighbors=_model_inputs['neighbors_out'],
#     )
#     out = jnp.transpose(out, (0, 2, 1))
#     out = jnp.transpose(_bound.projection(out), (0, 2, 1))
#     return train_loss_fn(out, y=_y_jax)


# _lift_grad = jax.grad(_loss_from_lifting)(_lifting_out)
# print(f"[diag] jax  ∂L/∂(lifting_out) norm = {float(jnp.linalg.norm(_lift_grad)):.6e}")

# print(f"[diag] torch ∂L/∂(lifting_out) norm = {_torch_lifting_grad['g'].norm():.6e}")
# print(f"[diag] ratio lifting j/t = {float(jnp.linalg.norm(_lift_grad)) / float(_torch_lifting_grad['g'].norm()):.3f}")

# # Per-FNO-block grad diagnostics
# _fno_intermediates = []
# _cur = _lifting_out
# for _idx in range(_bound.fno_blocks.n_layers):
#     _cur = _bound.fno_blocks(_cur, _idx, ada_in_embeddings=None)
#     _fno_intermediates.append(_cur)

# for _idx, _fno_mid in enumerate(_fno_intermediates):
#     def _loss_from_fno(fv, _idx=_idx):
#         fno_out = fv
#         for _jdx in range(_idx + 1, _bound.fno_blocks.n_layers):
#             fno_out = _bound.fno_blocks(fno_out, _jdx, ada_in_embeddings=None)
#         _perm = (0, 2, 3, 4, 1)
#         le = jnp.transpose(fno_out, _perm).reshape(1, -1, _fno_hidden)
#         out = _bound.gno_out(y=_lat_q_sq.reshape((-1, _lat_q_sq.shape[-1])), x=_out_q_sq, f_y=le, neighbors=_model_inputs['neighbors_out'])
#         out = jnp.transpose(out, (0, 2, 1))
#         out = jnp.transpose(_bound.projection(out), (0, 2, 1))
#         return train_loss_fn(out, y=_y_jax)
#     _fno_grad = jax.grad(_loss_from_fno)(_fno_mid)
#     print(f"[diag] jax  ∂L/∂(fno_block[{_idx}]_out) norm = {float(jnp.linalg.norm(_fno_grad)):.6e}")
    
# for _idx in range(torch_model.fno_blocks.n_layers):
#     if _idx in _torch_fno_grads:
#         _t_norm = float(_torch_fno_grads[_idx].norm())
#         print(f"[diag] torch ∂L/∂(fno_block[{_idx}]_out) norm = {_t_norm:.6e}")

# # ─── Surgical block[0] internal gradient diagnostics ────────────────────────
# # Run block[0] forward step by step, capture each intermediate,
# # then compute jax.grad(loss_from_that_point)(intermediate) for each.

# _b = _bound.fno_blocks          # shorthand

# # -- step-by-step forward through block[0] --
# _s0 = _lifting_out              # input to block[0]

# # 1. skip connections (computed from same input x)
# _x_skip_fno    = _b.fno_skips[0](_s0)
# _x_skip_fno    = _b.convs[0].transform(_x_skip_fno)
# _x_skip_mlp    = _b.channel_mlp_skips[0](_s0)

# # 2. spectral conv
# _s1 = _b.convs[0](_s0)         # after spectral conv

# # 3. norm[0]
# _s2 = _b._apply_norm(_s1, 0, None, 0)   # after first norm

# # 4. add fno skip
# _s3 = _s2 + _x_skip_fno        # after skip add

# # 5. first GELU
# _s4 = _b._non_linearity(_s3)   # after first non-linearity

# # 6. channel MLP + mlp skip
# _s5 = _b.channel_mlp[0](_s4) + _x_skip_mlp   # after channel MLP

# # 7. norm[1]
# _s6 = _b._apply_norm(_s5, 1, None, 0)   # after second norm

# # 8. second GELU → should equal _fno_intermediates[0]
# _s7 = _b._non_linearity(_s6)   # after second non-linearity

# def _tail_to_loss(fno_out):
#     """Run fno_out (block[0] output) through blocks 1-3 and to loss."""
#     for _jdx in range(1, _bound.fno_blocks.n_layers):
#         fno_out = _bound.fno_blocks(fno_out, _jdx, ada_in_embeddings=None)
#     _perm = (0, 2, 3, 4, 1)
#     le = jnp.transpose(fno_out, _perm).reshape(1, -1, _fno_hidden)
#     out = _bound.gno_out(y=_lat_q_sq.reshape((-1, _lat_q_sq.shape[-1])), x=_out_q_sq, f_y=le, neighbors=_model_inputs['neighbors_out'])
#     out = jnp.transpose(out, (0, 2, 1))
#     out = jnp.transpose(_bound.projection(out), (0, 2, 1))
#     return train_loss_fn(out, y=_y_jax)

# # Each continuation properly completes block[0] from the given intermediate.
# # _x_skip_fno and _x_skip_mlp are treated as constants (they depend on _s0, not sv).
# def _from_s1(sv):  # after spectral conv
#     s2 = _b._apply_norm(sv, 0, None, 0)
#     s3 = s2 + _x_skip_fno
#     s4 = _b._non_linearity(s3)
#     s5 = _b.channel_mlp[0](s4) + _x_skip_mlp
#     s6 = _b._apply_norm(s5, 1, None, 0)
#     s7 = _b._non_linearity(s6)
#     return _tail_to_loss(s7)

# def _from_s2(sv):  # after norm[0]
#     s3 = sv + _x_skip_fno
#     s4 = _b._non_linearity(s3)
#     s5 = _b.channel_mlp[0](s4) + _x_skip_mlp
#     s6 = _b._apply_norm(s5, 1, None, 0)
#     s7 = _b._non_linearity(s6)
#     return _tail_to_loss(s7)

# def _from_s3(sv):  # after skip add
#     s4 = _b._non_linearity(sv)
#     s5 = _b.channel_mlp[0](s4) + _x_skip_mlp
#     s6 = _b._apply_norm(s5, 1, None, 0)
#     s7 = _b._non_linearity(s6)
#     return _tail_to_loss(s7)

# def _from_s4(sv):  # after first GELU
#     s5 = _b.channel_mlp[0](sv) + _x_skip_mlp
#     s6 = _b._apply_norm(s5, 1, None, 0)
#     s7 = _b._non_linearity(s6)
#     return _tail_to_loss(s7)

# def _from_s5(sv):  # after channel MLP + skip
#     s6 = _b._apply_norm(sv, 1, None, 0)
#     s7 = _b._non_linearity(s6)
#     return _tail_to_loss(s7)

# def _from_s6(sv):  # after norm[1]
#     s7 = _b._non_linearity(sv)
#     return _tail_to_loss(s7)

# _jax_spectral_norm = None
# for _label, _sv, _fn in [
#     ("after_spectral",  _s1, _from_s1),
#     ("after_norm0",     _s2, _from_s2),
#     ("after_skip_add",  _s3, _from_s3),
#     ("after_gelu0",     _s4, _from_s4),
#     ("after_mlp",       _s5, _from_s5),
#     ("after_norm1",     _s6, _from_s6),
# ]:
#     _g = jax.grad(_fn)(_sv)
#     _gnorm = float(jnp.linalg.norm(_g))
#     if _label == "after_spectral":
#         _jax_spectral_norm = _gnorm
#     print(f"[diag] jax  ∂L/∂({_label}) norm = {_gnorm:.6e}")

# if 'g' in _torch_spectral0_grad and _jax_spectral_norm is not None:
#     _t_spec_norm = float(_torch_spectral0_grad['g'].norm())
#     print(f"[diag] torch ∂L/∂(after_spectral) norm = {_t_spec_norm:.6e}")
#     print(f"[diag] ratio after_spectral j/t        = {_jax_spectral_norm / _t_spec_norm:.3f}")

# print()
# print("=" * 86)
# print("[Gate C] Gradient parity check")
# print("=" * 86)
# print(f"  torch loss = {float(_torch_loss):.6e}")
# print(f"  jax   loss = {float(_jax_loss):.6e}")
# print(f"  |Δloss|    = {abs(float(_torch_loss) - float(_jax_loss)):.3e}")
# print()

# # ---- Convert torch grads -> Flax-shaped pytree --------------------------
# # Reuse copy_torch_to_flax by faking a torch_model whose state_dict() returns
# # parameter gradients instead of values. Same key names, same shapes.
# class _GradsAsState:
#     def __init__(self, _tm):
#         self._state = {n: p.grad for n, p in _tm.named_parameters() if p.grad is not None}
#         self._n_layers = _tm.fno_blocks.n_layers
#     def state_dict(self):
#         return self._state
#     @property
#     def fno_blocks(self):
#         class _FB: pass
#         _fb = _FB()
#         _fb.n_layers = self._n_layers
#         return _fb

# _torch_grads_flax_shape = copy_torch_to_flax(
#     _GradsAsState(torch_model),
#     jax.tree_util.tree_map(lambda x: x, transferred_variables['params']),  # template
# )

# # ---- Walk both pytrees and compare leaf-by-leaf ------------------------
# def _flatten(pytree, prefix=""):
#     out = {}
#     for _k, _v in pytree.items():
#         _full = f"{prefix}/{_k}" if prefix else _k
#         if isinstance(_v, dict):
#             out.update(_flatten(_v, _full))
#         else:
#             out[_full] = _v
#     return out

# _torch_flat = _flatten(_torch_grads_flax_shape)
# _jax_flat   = _flatten(_jax_grads)

# # Gradients accumulate forward noise through the chain rule; widen tolerance.
# _C_TOL_MAX  = 1e-3
# _C_TOL_MEAN = 1e-4
# _C_REL_TOL  = 1e-3   # relative-to-torch-norm safety net for tiny-magnitude grads

# print(f"{'param path':<55} {'t_norm':>11} {'j_norm':>11} {'max_abs':>11} {'rel':>9}")
# print("-" * 86)

# _all_ok = True
# _overall_max = 0.0
# _n_pass = 0
# _n_fail = 0
# for _path in sorted(_torch_flat.keys()):
#     _tg = np.asarray(_torch_flat[_path])
#     _jg = np.asarray(_jax_flat[_path])
#     _d  = np.abs(_tg - _jg)
#     _t_norm = float(np.linalg.norm(_tg))
#     _j_norm = float(np.linalg.norm(_jg))
#     _rel    = float(_d.max() / (_t_norm + 1e-12))
#     _ok = bool(_d.max() < _C_TOL_MAX or _d.mean() < _C_TOL_MEAN or _rel < _C_REL_TOL)
#     _overall_max = max(_overall_max, float(_d.max()))
#     _n_pass += int(_ok)
#     _n_fail += int(not _ok)
#     _all_ok &= _ok
#     _flag = "" if _ok else "  <-- FAIL"
#     print(f"{_path:<55} {_t_norm:>11.3e} {_j_norm:>11.3e} {_d.max():>11.3e} {_rel:>9.2e}{_flag}")

# print("-" * 86)
# print(f"[Gate C] {_n_pass} leaves PASS, {_n_fail} leaves FAIL  "
#       f"(tol_max={_C_TOL_MAX:.0e}, tol_mean={_C_TOL_MEAN:.0e}, tol_rel={_C_REL_TOL:.0e})")
# print(f"[Gate C] worst leaf max_abs = {_overall_max:.3e}")
# print("=" * 86)
# if not _all_ok:
#     raise RuntimeError(
#         "[Gate C] FAIL — gradient parity broken on at least one leaf.\n"
#         "        See per-row 'FAIL' tags above. Refusing 301-epoch run."
#     )
# print("[Gate C] PASS")
# 
# # Free torch model now that Gates A + C are done.
# del torch_model

# Trainer setup
trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=config.get("device", "cpu"),
    wandb_log=config.wandb.get("log", False) if isinstance(config.wandb, dict) else False,
    verbose=True,
)

# Start training process
trainer.train(
    train_loader=train_loader_jax,
    test_loaders={"test": test_loader_jax},
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss_fn,
    eval_losses={config.opt.testing_loss: test_loss_fn},
    regularizer=None,
)

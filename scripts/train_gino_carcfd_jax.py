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
from neuralop.training.adamw_jax import AdamW, adamw_cx
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

# ── Run overrides ────────────────────────────────────────────────────────────
config.data.n_train = 500
config.data.n_test  = 111
config.opt.n_epochs = 10
# ────────────────────────────────────────────────────────────────────────────

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
        self.tx = optax.inject_hyperparams(adamw_cx)(learning_rate=learning_rate, weight_decay=weight_decay)
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

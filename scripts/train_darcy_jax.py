"""
Training script for Darcy flow equation using neural operators.

This script trains a neural operator on the 2D Darcy flow equation,
which models fluid flow through porous media. The script supports
distributed training and multi-grid patching for high-resolution data.
"""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import optax

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    pass

from neuralop.losses.data_losses_jax import LpLoss, H1Loss
from neuralop.data.datasets.darcy_jax import load_darcy_flow_small, DataLoader
from neuralop.data.transforms.data_processors_jax import MGPatchingDataProcessor
from neuralop.training.torch_setup_jax import setup
from neuralop.training.adamw_jax import AdamW, adamw_cx
from neuralop.training.trainer_jax import Trainer
from neuralop.mpu.comm_jax import get_local_rank
from neuralop.utils_jax import get_wandb_api_key, count_model_params
from neuralop import get_model_jax


class FlaxModelWrapper:
    """Wraps a Flax FNO module with optax optimization for use with the Trainer.

    Manages Flax parameter state and the JIT-compiled training step so the
    stateless Flax module can be used with the stateful Trainer interface.
    """

    _MODEL_KEYS = frozenset({'x'})

    def __init__(self, flax_module, learning_rate=1e-3, weight_decay=1e-4, key=None):
        self.module = flax_module
        self.learning_rate = learning_rate
        self.params = None
        self.opt_state = None
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self.tx = optax.inject_hyperparams(adamw_cx)(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        self._initialized = False
        self._jit_fn = None

    def _init(self, sample_kwargs):
        """Initialize params and optimizer state from the first real sample."""
        model_inputs = {k: v for k, v in sample_kwargs.items() if k in self._MODEL_KEYS}
        self.params = self.module.init(self.key, **model_inputs)
        self.opt_state = self.tx.init(self.params)
        self._initialized = True

    def set_lr(self, new_lr):
        self.learning_rate = new_lr
        self.opt_state.hyperparams['learning_rate'] = jnp.array(new_lr)

    def _build_jit_fn(self, loss_fn):
        module = self.module
        tx = self.tx

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
        """Per-sample JIT step."""
        if not self._initialized:
            self._init(kwargs)
        if self._jit_fn is None:
            self._jit_fn = self._build_jit_fn(loss_fn)
        model_inputs = {k: v for k, v in kwargs.items() if k in self._MODEL_KEYS}
        loss, self.params, self.opt_state = self._jit_fn(
            self.params, self.opt_state, model_inputs, kwargs['y']
        )
        return loss


# Read the configuration
from zencfg import make_config_from_cli

sys.path.insert(0, "../")
from config.darcy_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()

# Distributed training setup, if enabled
device, is_logger = setup(config)

# Set up WandB logging
wandb_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config.model.model_arch,
                config.model.n_layers,
                config.model.n_modes,
                config.model.hidden_channels,
            ]
        )
    wandb_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_args)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print configuration details
if config.verbose and is_logger:
    print(f"##### CONFIG #####\n")
    print(config)
    sys.stdout.flush()

# Load the Darcy flow dataset
data_root = Path(config.data.folder).expanduser()
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    data_root=data_root,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=False,
    encode_output=False,
)

# Model initialization
_fno_module = get_model_jax(config)
model = FlaxModelWrapper(
    _fno_module,
    learning_rate=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

# convert dataprocessor to an MGPatchingDataProcessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(
        model=model,
        in_normalizer=data_processor.in_normalizer,
        out_normalizer=data_processor.out_normalizer,
        padding_fraction=config.patching.padding,
        stitching=config.patching.stitching,
        levels=config.patching.levels,
        use_distributed=config.distributed.use_distributed,
        device=device,
    )

# Distributed data parallel setup
# Reconfigure DataLoaders for distributed mode.
# Note: JAX DataLoader does not support DistributedSampler; data sharding
# is handled via JAX pmap rather than a sampler.
if config.distributed.use_distributed:
    train_db = train_loader.dataset
    train_loader = DataLoader(
        dataset=train_db, batch_size=config.data.batch_size
    )
    for (res, loader), batch_size in zip(
        test_loaders.items(), config.data.test_batch_sizes
    ):
        test_db = loader.dataset
        test_loaders[res] = DataLoader(
            dataset=test_db, batch_size=batch_size, shuffle=False
        )

# Create the optimizer
# Note: JAX/Flax models carry no .parameters(); the optimizer state is managed
# inside FlaxModelWrapper via optax. AdamW here provides the lr-tracking
# interface that the Trainer and scheduler use to read/update the learning rate.
optimizer = AdamW(
    params=[],
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)


# Scheduler implementations for JAX (replaces torch.optim.lr_scheduler)
class SchedulerBase:
    """Base scheduler class for JAX."""
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.step_count = 0

    def step(self, metric=None):
        self.step_count += 1


class ReduceLROnPlateau(SchedulerBase):
    """Simple implementation of ReduceLROnPlateau for JAX."""
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
            elif (self.mode == "min" and metric < self.best_metric) or \
                 (self.mode == "max" and metric > self.best_metric):
                self.best_metric = metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    new_lr = model.learning_rate * self.factor
                    model.set_lr(new_lr)
                    self.optimizer.learning_rate = new_lr
                    self.patience_counter = 0


class CosineAnnealingLR(SchedulerBase):
    """Simple implementation of CosineAnnealingLR for JAX."""
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.learning_rate

    def step(self, metric=None):
        super().step()
        new_lr = float(self.eta_min + (self.base_lr - self.eta_min) * (
            1 + jnp.cos(jnp.pi * self.step_count / self.T_max)
        ) / 2)
        model.set_lr(new_lr)
        self.optimizer.learning_rate = new_lr


class StepLR(SchedulerBase):
    """Simple implementation of StepLR for JAX."""
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.learning_rate

    def step(self, metric=None):
        super().step()
        if self.step_count % self.step_size == 0:
            new_lr = self.base_lr * (self.gamma ** (self.step_count // self.step_size))
            model.set_lr(new_lr)
            self.optimizer.learning_rate = new_lr


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
    scheduler = StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Create the losses functions
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f"Got training_loss={config.opt.training_loss} "
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    data_processor=data_processor,
    mixed_precision=config.opt.mixed_precision,
    wandb_log=config.wandb.log,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
)

# Initialize model parameters eagerly so param count is available before training.
# JAX/Flax models are stateless; params are separate from the module and only
# exist after the first .init() call.
_resolution = config.data.train_resolution
_dummy_x = jnp.zeros((1, config.model.data_channels, _resolution, _resolution))
if not model._initialized:
    model._init({'x': _dummy_x})

# Log model parameter count
if is_logger:
    # JAX: count_model_params takes the param pytree, not the model object
    n_params = count_model_params(model.params)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        # Note: wandb.watch is PyTorch-only and is not supported for Flax models

# Start training process
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

# Finalize WandB logging
if config.wandb.log and is_logger:
    wandb.finish()

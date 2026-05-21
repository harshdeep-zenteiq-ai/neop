from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings

import jax
import jax.numpy as jnp
from flax import linen as nn

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

from neuralop.losses import LpLoss
from .training_state_jax import load_training_state, save_training_state
import sys


class SimpleDataLoader:
    """Wrapper for a list of batches to mimic a PyTorch DataLoader."""
    def __init__(self, batches):
        self.batches = batches
        # Create a simple dataset object with len
        class Dataset:
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
        self.dataset = Dataset(len(batches))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class Trainer:
    """
    A general Trainer class to train neural-operators on given datasets.

    .. note ::
        Our Trainer expects datasets to provide batches as key-value dictionaries, ex.:
        ``{'x': x, 'y': y}``, that are keyed to the arguments expected by models and losses.
        For specifics and an example, check ``neuralop.data.datasets.DarcyDataset``.

    Parameters
    ----------
    model : flax.linen.Module
        model to train
    n_epochs : int
        number of training epochs
    wandb_log : bool, default is False
        whether to log results to wandb
    device : str, default 'cpu'
        device for computation (JAX handles placement automatically)
    mixed_precision : bool, default is False
        whether to use mixed precision (JAX-specific implementation)
    data_processor : DataProcessor class to transform data, default is None
        if not None, data from the loaders is transform first with data_processor.preprocess,
        then after getting an output from the model, that is transformed with data_processor.postprocess.
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    use_distributed : bool, default is False
        whether to use distributed training (JAX pmap)
    verbose : bool, default is False
        whether to print training progress
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = "cpu",
        mixed_precision: bool = False,
        data_processor=None,
        eval_interval: int = 1,
        log_output: bool = False,
        use_distributed: bool = False,
        verbose: bool = False,
        stacked_train_data=None,
    ):
        """Initialize Trainer."""
        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = False
        if wandb_available:
            self.wandb_log = wandb_log and wandb.run is not None
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor
        # Pre-stacked training data for jax.lax.scan-based epoch loop
        # self.stacked_train_data = stacked_train_data

        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        eval_modes=None,
        save_every: int = None,
        save_best: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
        max_autoregressive_steps: int = None,
    ):
        """Trains the given model on the given dataset.

        Parameters
        ----------
        train_loader : DataLoader
            training dataloader
        test_loaders : dict
            testing dataloaders keyed by name
        optimizer : optax optimizer
            optimizer to use during training
        scheduler : scheduler object
            learning rate scheduler to use during training
        training_loss : loss function, optional
            cost function to minimize
        eval_losses : dict, optional
            dict of losses to use in evaluation
        eval_modes : dict, optional
            optional mapping from loader name to evaluation mode
            ('single_step' or 'autoregressive')
        save_every : int, optional
            if provided, interval at which to save checkpoints
        save_best : str, optional
            if provided, key of metric to monitor for best model
        save_dir : str | Path, default "./ckpt"
            directory at which to save training states
        resume_from_dir : str | Path, default None
            if provided, resumes training state from this directory
        max_autoregressive_steps : int, default None
            max number of autoregressive steps for evaluation

        Returns
        -------
        epoch_metrics : dict
            dictionary of metrics for the last validation epoch
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, "reduction"):
            if training_loss.reduction == "mean":
                warnings.warn(
                    f"{training_loss.reduction=}. This means that the loss is "
                    "initialized to average across the batch dim. The Trainer "
                    "expects losses to sum across the batch dim."
                )

        if eval_losses is None:
            eval_losses = dict(l2=training_loss)

        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # create default eval modes
        if eval_modes is None:
            eval_modes = {}

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert (
                self.save_best in metrics
            ), f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float("inf")
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f"Training on {len(train_loader.dataset)} samples")
            print(f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
                f"         on resolutions {[name for name in test_loaders]}.")
            sys.stdout.flush()

        for epoch in range(self.start_epoch, self.n_epochs):
            (
                train_err,
                avg_loss,
                avg_lasso_loss,
                epoch_train_time,
            ) = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time,
            )

            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(
                    epoch=epoch,
                    eval_losses=eval_losses,
                    test_loaders=test_loaders,
                    eval_modes=eval_modes,
                    max_autoregressive_steps=max_autoregressive_steps,
                )
                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """Train for one epoch and return training metrics.

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : DataLoader
            training data loader
        training_loss : loss function
            training loss function

        Returns
        -------
        train_err : float
            average training error
        avg_loss : float
            average loss per sample
        avg_lasso_loss : float or None
            average regularization loss
        epoch_train_time : float
            time taken for epoch
        """
        self.on_epoch_start(epoch)

        if self.data_processor:
            if hasattr(self.data_processor, 'train'):
                self.data_processor.train()
            elif hasattr(self.data_processor, 'training'):
                self.data_processor.training = True

        t1 = default_timer()

        # --- Fast path: jax.lax.scan over pre-stacked data ---
        # Compiles the entire epoch as one XLA program — no Python loop overhead.
        # if (self.stacked_train_data is not None
        #         and hasattr(self.model, 'train_epoch_scan')):
        #     train_err, avg_loss = self.model.train_epoch_scan(
        #         self.stacked_train_data, training_loss
        #     )
        #     avg_lasso_loss = None
        #     self.n_samples = 1   # placeholder; scan processed all samples

        # # --- Per-batch JIT loop (async dispatch, single GPU-CPU sync) ---
        # else:
        
        self.n_samples = 0
        batch_losses = []   # accumulate as JAX arrays — no per-batch sync

        for idx, sample in enumerate(train_loader):
            loss = self.train_one_batch(idx, sample, training_loss)
            batch_losses.append(loss)  # stays on GPU (no float() call)

            # Single GPU-CPU sync at end of epoch
        jax.block_until_ready(batch_losses[-1])
        total = sum(float(l) for l in batch_losses)
        n     = len(batch_losses)
        train_err      = total / n
        avg_loss       = total / max(self.n_samples, 1)
        avg_lasso_loss = None
        if self.regularizer:
            avg_lasso_loss /= max(self.n_samples, 1)
        else:
            avg_lasso_loss = None

        # Update scheduler
        if hasattr(self.scheduler, 'step'):
            self.scheduler.step(train_err)

        epoch_train_time = default_timer() - t1

        lr = None
        if hasattr(self.optimizer, 'learning_rate'):
            lr = self.optimizer.learning_rate

        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(
        self,
        epoch,
        eval_losses,
        test_loaders,
        eval_modes,
        max_autoregressive_steps=None,
    ):
        """Evaluate on all test loaders.

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_losses : dict
            keyed ``loss_name: loss_obj`` for each loss
        test_loaders : dict
            keyed ``loader_name: loader`` for each test loader
        eval_modes : dict, optional
            keyed ``loader_name: eval_mode`` for each test loader
        max_autoregressive_steps : int, optional
            max number of autoregressive steps

        Returns
        -------
        all_metrics : dict
            collected eval metrics for each loader
        """
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_eval_mode = eval_modes.get(loader_name, "single_step")
            loader_metrics = self.evaluate(
                eval_losses,
                loader,
                log_prefix=loader_name,
                mode=loader_eval_mode,
                max_steps=max_autoregressive_steps,
            )
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics

    def evaluate(
        self,
        loss_dict,
        data_loader,
        log_prefix="",
        epoch=None,
        mode="single_step",
        max_steps=None,
    ):
        """Evaluate the model on a dictionary of losses.

        Parameters
        ----------
        loss_dict : dict
            each function takes as input (prediction, ground_truth)
            and returns the corresponding loss
        data_loader : DataLoader
            data loader to evaluate on
        log_prefix : str, default ''
            used as prefix in output dictionary
        epoch : int | None
            current epoch for logging
        mode : str
            'single_step' or 'autoregression'
        max_steps : int, optional
            max number of steps for autoregressive rollout

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}'] = loss value
        """
        self.model.eval() if hasattr(self.model, 'eval') else None
        if self.data_processor:
            if hasattr(self.data_processor, 'eval'):
                self.data_processor.eval()
            elif hasattr(self.data_processor, 'training'):
                self.data_processor.training = False

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        # Warn the user if any of the eval losses is reducing across the batch
        for _, eval_loss in loss_dict.items():
            if hasattr(eval_loss, "reduction"):
                if eval_loss.reduction == "mean":
                    warnings.warn(
                        f"{eval_loss.reduction=}. This means that the loss is "
                        "initialized to average across the batch dim. The Trainer "
                        "expects losses to sum across the batch dim."
                    )

        self.n_samples = 0
        for idx, sample in enumerate(data_loader):
            return_output = False
            if idx == len(data_loader) - 1:
                return_output = True
            if mode == "single_step":
                eval_step_losses, outs = self.eval_one_batch(
                    sample, loss_dict, return_output=return_output
                )
            elif mode == "autoregression":
                eval_step_losses, outs = self.eval_one_batch_autoreg(
                    sample,
                    loss_dict,
                    return_output=return_output,
                    max_steps=max_steps,
                )

            for loss_name, val_loss in eval_step_losses.items():
                errors[f"{log_prefix}_{loss_name}"] += float(val_loss) if hasattr(val_loss, '__float__') else val_loss

        for key in errors.keys():
            errors[key] /= max(self.n_samples, 1)

        # on last batch, log model outputs
        if self.log_output and self.wandb_log:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)

        return errors

    def on_epoch_start(self, epoch):
        """Run at the beginning of each training epoch.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample, training_loss):
        """Run one batch through model and return training loss.

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch
        training_loss : loss function
            loss function to compute

        Returns
        -------
        loss : float or array
            training loss value
        """
        if self.regularizer:
            self.regularizer.reset() if hasattr(self.regularizer, 'reset') else None

        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data if no preprocessor exists
            sample = {k: jnp.asarray(v) for k, v in sample.items() if isinstance(v, jnp.ndarray)}

        if "y" in sample and hasattr(sample["y"], 'shape'):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        # Model forward pass with gradient update if model supports train_step
        if hasattr(self.model, 'train_step'):
            # JAX model: compute loss + gradients + update params in one step
            loss = self.model.train_step(sample, training_loss)
        else:
            # Fallback: plain forward pass (no gradient update — eval only)
            out = self.model(**sample)
            if self.epoch == 0 and idx == 0 and self.verbose and hasattr(out, 'shape'):
                print(f"Raw outputs of shape {out.shape}")
            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample)
            loss = training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss if hasattr(self.regularizer, 'loss') else 0.0

        return loss

    def eval_one_batch(
        self, sample: dict, eval_losses: dict, return_output: bool = False
    ):
        """Run inference on one batch and return eval losses.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_output : bool
            whether to return model outputs for plotting

        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss
        outputs : array or None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data if no preprocessor exists
            sample = {k: jnp.asarray(v) for k, v in sample.items() if isinstance(v, jnp.ndarray)}

        if "y" in sample and hasattr(sample["y"], 'shape'):
            self.n_samples += sample["y"].shape[0]

        out = self.model(**sample)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, **sample)
            eval_step_losses[loss_name] = val_loss

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def eval_one_batch_autoreg(
        self,
        sample: dict,
        eval_losses: dict,
        return_output: bool = False,
        max_steps: int = None,
    ):
        """Run autoregressive inference on one batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_output : bool
            whether to return model outputs for plotting
        max_steps : int
            max number of timesteps to roll out

        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss
        outputs : array or None
            optionally returns batch outputs
        """
        eval_step_losses = {loss_name: 0.0 for loss_name in eval_losses.keys()}

        t = 0
        if max_steps is None:
            max_steps = float("inf")

        # only increment the sample count once
        sample_count_incr = False

        while sample is not None and t < max_steps:
            if self.data_processor is not None:
                sample = self.data_processor.preprocess(sample, step=t) if hasattr(self.data_processor.preprocess, '__code__') and 'step' in self.data_processor.preprocess.__code__.co_varnames else self.data_processor.preprocess(sample)
            else:
                # load data if no preprocessor exists
                sample = {
                    k: jnp.asarray(v)
                    for k, v in sample.items()
                    if isinstance(v, jnp.ndarray)
                }

            if sample is None:
                break

            # only increment the sample count once
            if not sample_count_incr:
                if "y" in sample and hasattr(sample["y"], 'shape'):
                    self.n_samples += sample["y"].shape[0]
                sample_count_incr = True

            out = self.model(**sample)

            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample, step=t) if hasattr(self.data_processor.postprocess, '__code__') and 'step' in self.data_processor.postprocess.__code__.co_varnames else self.data_processor.postprocess(out, sample)

            for loss_name, loss in eval_losses.items():
                step_loss = loss(out, **sample)
                eval_step_losses[loss_name] += float(step_loss) if hasattr(step_loss, '__float__') else step_loss

            t += 1

        # average over all steps of the final rollout
        for loss_name in eval_step_losses.keys():
            eval_step_losses[loss_name] /= max(t, 1)

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def log_training(
        self,
        epoch: int,
        time: float,
        avg_loss: float,
        train_err: float,
        avg_lasso_loss: float = None,
        lr: float = None,
    ):
        """Log results from a single training epoch.

        Parameters
        ----------
        epoch : int
            epoch number
        time : float
            training time of epoch
        avg_loss : float
            average loss per sample
        train_err : float
            training error for entire epoch
        avg_lasso_loss : float, optional
            average lasso loss from regularizer
        lr : float, optional
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"

        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=False)

    def log_eval(self, epoch: int, eval_metrics: dict):
        """Log evaluation metrics to stdout and wandb.

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics keyed f"{test_loader_name}_{metric}"
        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, jnp.ndarray):
                msg += f"{metric}={float(value):.4f}, "
            if self.wandb_log:
                values_to_log[metric] = value

        msg = f"Eval: " + msg[:-2] if msg else "Eval: "
        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=True)

    def resume_state_from_dir(self, save_dir):
        """Resume training from a saved state directory.

        Parameters
        ----------
        save_dir : str | Path
            directory in which training state is saved
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # check for save model exists
        if (save_dir / "best_model_state_dict.pkl").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pkl").exists():
            save_name = "model"
        else:
            raise FileNotFoundError(
                "Error: resume_from_dir expects a model state dict named "
                "model.pkl or best_model.pkl."
            )

        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.regularizer,
            resume_epoch,
        ) = load_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=self.model,
            optimizer=self.optimizer,
            regularizer=self.regularizer,
            scheduler=self.scheduler,
        )

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose:
                    print(f"Trainer resuming from epoch {resume_epoch}")

    def checkpoint(self, save_dir):
        """Save current training state to a directory.

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        if self.save_best is not None:
            save_name = "best_model"
        else:
            save_name = "model"

        save_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            regularizer=self.regularizer,
            epoch=self.epoch,
        )
        if self.verbose:
            print(f"Saved training state to {save_dir}")

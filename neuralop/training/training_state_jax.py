"""
Snippet to load all artifacts of training state as Modules
without constraining to use inside a default Trainer
"""
from typing import Union
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from neuralop.mpu.comm import get_local_rank


def load_training_state(
    save_dir: Union[str, Path],
    save_name: str,
    model=None,
    optimizer=None,
    scheduler=None,
    regularizer=None,
    map_location: dict = None,
) -> dict:
    """load_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    model : object, optional
        model object to load (Flax module or similar), by default None
    optimizer : object, optional
        optimizer object to load, by default None
    scheduler : object, optional
        scheduler object to load, by default None
    regularizer : object, optional
        regularizer object to load, by default None
    map_location : dict, optional
        mapping dictionary (for API compatibility), by default None

    Returns
    -------
    tuple of training state
        ``model, optimizer, scheduler, regularizer, epoch``

    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    # optionally load epoch
    epoch = None
    manifest_pth = save_dir / "manifest.pkl"
    if manifest_pth.exists():
        with open(manifest_pth, 'rb') as f:
            manifest = pickle.load(f)
        epoch = manifest.get("epoch")

    # Load model if state exists
    if model is not None:
        save_pth = save_dir / f"{save_name}_state_dict.pkl"
        if save_pth.exists():
            with open(save_pth, 'rb') as f:
                state_dict = pickle.load(f)
            # For Flax models, assign loaded state to model
            # This assumes the model has a method to load parameters, or
            # the caller handles parameter assignment
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(state_dict)
            else:
                # If model is a dict of parameters, return the loaded state
                model = state_dict
        else:
            print(f"Warning: requested to load model state, but no saved model state exists in {save_dir}.")

    # load optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.pkl"
        if optimizer_pth.exists():
            with open(optimizer_pth, 'rb') as f:
                optimizer = pickle.load(f)
        else:
            print(f"Warning: requested to load optimizer state, but no saved optimizer state exists in {save_dir}.")

    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.pkl"
        if scheduler_pth.exists():
            with open(scheduler_pth, 'rb') as f:
                scheduler = pickle.load(f)
        else:
            print(f"Warning: requested to load scheduler state, but no saved scheduler state exists in {save_dir}.")

    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.pkl"
        if regularizer_pth.exists():
            with open(regularizer_pth, 'rb') as f:
                regularizer = pickle.load(f)
        else:
            print(f"Warning: requested to load regularizer state, but no saved regularizer state exists in {save_dir}.")

    return model, optimizer, scheduler, regularizer, epoch


def save_training_state(
    save_dir: Union[str, Path],
    save_name: str,
    model=None,
    optimizer=None,
    scheduler=None,
    regularizer=None,
    epoch: int = None,
) -> None:
    """save_training_state saves model and optional other training modules
    for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory to save training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to save
    model : object, optional
        model object to save (Flax module or state dict), by default None
    optimizer : object, optional
        optimizer object to save, by default None
    scheduler : object, optional
        scheduler object to save, by default None
    regularizer : object, optional
        regularizer object to save, by default None
    epoch : int, optional
        epoch number to save, by default None
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True, parents=True)
    manifest = {}

    # Save model
    if model is not None:
        model_pth = save_dir / f"{save_name}_state_dict.pkl"
        # Extract state dict if model has the method, otherwise save directly
        if hasattr(model, 'state_dict'):
            state_to_save = model.state_dict()
        else:
            state_to_save = model

        with open(model_pth, 'wb') as f:
            pickle.dump(state_to_save, f)
        manifest["model"] = f"{save_name}_state_dict.pkl"

    # save optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.pkl"
        # Extract state dict if optimizer has the method, otherwise save directly
        if hasattr(optimizer, 'state_dict'):
            opt_state = optimizer.state_dict()
        else:
            opt_state = optimizer

        with open(optimizer_pth, 'wb') as f:
            pickle.dump(opt_state, f)
        manifest["optimizer"] = "optimizer.pkl"

    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.pkl"
        # Extract state dict if scheduler has the method, otherwise save directly
        if hasattr(scheduler, 'state_dict'):
            sched_state = scheduler.state_dict()
        else:
            sched_state = scheduler

        with open(scheduler_pth, 'wb') as f:
            pickle.dump(sched_state, f)
        manifest["scheduler"] = "scheduler.pkl"

    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.pkl"
        # Extract state dict if regularizer has the method, otherwise save directly
        if hasattr(regularizer, 'state_dict'):
            reg_state = regularizer.state_dict()
        else:
            reg_state = regularizer

        with open(regularizer_pth, 'wb') as f:
            pickle.dump(reg_state, f)
        manifest["regularizer"] = "regularizer.pkl"

    if epoch is not None:
        manifest["epoch"] = epoch

    manifest_pth = save_dir / "manifest.pkl"
    with open(manifest_pth, 'wb') as f:
        pickle.dump(manifest, f)

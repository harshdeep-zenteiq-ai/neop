import math
import warnings
from typing import Callable, Iterable, Tuple, Union, Dict, Any

import jax
import jax.numpy as jnp
from .tensor_galore_projector_jax import TensorGaLoreProjector


class AdamW:
    """
    Implements AdamW (Adam with weight decay fix [1]_), and offers
    optional Tensor-GaLore projection (see [2]_ and [3]_) for memory-efficient training.

    Note: This JAX implementation maintains optimizer state but requires
    external parameter update logic (typically handled by training loop).

    Parameters
    ----------
    params : Iterable, optional
        Iterable of parameters to optimize or dictionaries defining parameter groups.
    lr : float, optional, defaults to 0.001
        The learning rate to use.
    betas : Tuple[float,float], optional, defaults to (0.9, 0.999)
        Adam's betas parameters (b1, b2).
    eps : float, optional, defaults to 1e-06
        Adam's epsilon for numerical stability.
    weight_decay : float, optional, defaults to 0.0
        Decoupled weight decay to apply.
    correct_bias : bool, optional, defaults to True
        Whether or not to correct bias in Adam.
    galore_params : Iterable, optional
        Iterable of parameters to optimize in low-rank subspace
    galore_rank : float, int or int tuple
        Rank of low-rank subspace in which to optimize galore_params.
        Either a float corresponding to a percentage of params
        to preserve, or an list of int ranks corresponding to
        each mode of the tensor. If a single int is given, it is used
        for all modes (see `neuralop/training/tensor_galore_projector_jax.py`)
    galore_update_proj_gap : int, defaults to 50
        Number of optimizer steps before projection tensors are recomputed.
    galore_scale : float, defaults to 1.0
        Additional lr-like scalar by which galore parameters are multiplied before update
    activation_checkpoint : bool, default False
        Whether to use activation checkpointing during projection
    warm_restart : bool, default True
        Whether to use warm restart for GaLore projection

    References
    ----------
    .. _[1] : Loschchilov, I. and Hutter, F. (2019). Decoupled Decay Regularization.
         ICLR 2019, https://arxiv.org/pdf/1711.05101.

    .. _[2] : Zhao, J, Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., Tian Y. (2024).
        GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. ICML 2024,
        https://arxiv.org/abs/2403.03507.

    .. _[3] : George, R., Pitt, D., Zhao, J., Kossaifi, J., Luo, C., Tian, Y., Anandkumar, A (2024).
        Tensor-GaLore: Memory-Efficient Training via Gradient Tensor Decomposition. arXiv preprint,
        https://arxiv.org/pdf/2501.02379.
    """

    def __init__(
        self,
        params: Iterable = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        galore_params: Iterable = None,
        galore_rank: Union[float, int, Tuple[int]] = 1.0,
        galore_update_proj_gap: int = 50,
        galore_scale: float = 1.0,
        activation_checkpoint: bool = False,
        warm_restart: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        self.param_groups = []

        # Add default param group
        self.param_groups.append({
            "params": params if params is not None else [],
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "galore": False,
        })

        # Keep optional GaLore parameters separate for projection
        if galore_params is not None:
            self.add_param_group({
                "params": galore_params,
                "rank": galore_rank,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "correct_bias": correct_bias,
                "galore": True,
            })

        self.galore_rank = galore_rank
        self.activation_checkpoint = activation_checkpoint
        self.warm_restart = warm_restart
        self.galore_update_proj_gap = galore_update_proj_gap
        self.galore_scale = galore_scale

        # Optimizer state
        self.state = {}
        self.learning_rate = lr

    def add_param_group(self, param_group):
        """Add a param group to the optimizer."""
        self.param_groups.append(param_group)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss : float or None
            Loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        updates = {}

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group["params"]):
                if not hasattr(p, 'grad') or p.grad is None:
                    continue

                grad = p.grad

                # Use (group_idx, param_idx) as unique key for parameter state
                param_key = (group_idx, param_idx)

                if param_key not in self.state:
                    self.state[param_key] = {}

                state = self.state[param_key]

                if "step" not in state:
                    state["step"] = 0

                # GaLore Projection
                if group.get("galore", False):
                    if "projector" not in state:
                        state["projector"] = TensorGaLoreProjector(
                            rank=self.galore_rank,
                            update_proj_gap=self.galore_update_proj_gap,
                            scale=self.galore_scale,
                            activation_checkpoint=self.activation_checkpoint,
                            warm_restart=self.warm_restart,
                        )

                    # track tensor shape for projection back
                    proj_input = grad
                    grad = state["projector"].project(proj_input, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = jnp.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = jnp.zeros_like(grad)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
                if jnp.iscomplexobj(grad):
                    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * jnp.real(grad * jnp.conj(grad))
                else:
                    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (grad * grad)

                denom = jnp.sqrt(exp_avg_sq) + group["eps"]

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                # GaLore Projection Back
                if group.get("galore", False):
                    norm_grad = state["projector"].project_back(norm_grad)

                # Compute parameter update
                param_update = -step_size * norm_grad

                # Add weight decay (decoupled weight decay, not L2 regularization)
                # This is the correct way to use weight decay with Adam
                if group["weight_decay"] > 0.0:
                    param_update = param_update - group["lr"] * group["weight_decay"] * p

                # Store update for this parameter
                updates[param_key] = param_update

                # Update state
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss, updates

    def get_state_dict(self):
        """Returns optimizer state as a dictionary."""
        return {
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict):
        """Loads optimizer state from a dictionary."""
        self.state = state_dict.get("state", {})
        self.param_groups = state_dict.get("param_groups", self.param_groups)

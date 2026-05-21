import math
import warnings
from typing import List

import jax
import jax.numpy as jnp

# Set warning filter to show each warning only once
warnings.filterwarnings("once", category=UserWarning)

#loss function with rel/abs Lp loss
class LpLoss(object):
    """LpLoss provides the Lp norm between two discretized d-dimensional functions.

    Note that LpLoss always averages over the spatial dimensions.

    .. note::
        In function space, the Lp norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to LpLoss, we assume a regular
        discretization and take ``1 / measure`` as the quadrature weights.

    Parameters
    ----------
    d : int, optional
        dimension of data on which to compute, by default 1
    p : int, optional
        order of L-norm, by default 2
        L-p norm: [\\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::
            To perform quadrature, ``LpLoss`` scales ``measure`` by the size
            of each spatial dimension of ``x``, and multiplies them with
            ||x-y||, such that the final norm is a scaled average over the spatial
            dimensions of ``x``.
    reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')

        .. warning::
            ``LpLoss`` always reduces over the spatial dimensions according to ``self.measure``.
            `reduction` only applies to the batch and channel dimensions.
    eps : float, optional
        small number added to the denominator for numerical stability when using the relative loss

    Examples
    --------
    See the module docstring or the user guide for usage examples.
    """

    def __init__(self, d=1, p=2, measure=1.0, reduction="sum", eps=1e-8):
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps

        allowed_reductions = ["sum", "mean"]
        assert (
            reduction in allowed_reductions
        ), f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure] * self.d
        else:
            self.measure = measure

    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"

    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that
        ``LpLoss`` computes the average over spatial dims.

        Parameters
        ----------
        x : jnp.ndarray
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0] * self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j] / x.shape[-j]

        return quadrature

    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Parameters
        ----------
        x: jnp.ndarray
            inputs
        """
        if self.reduction == "sum":
            x = jnp.sum(x)
        else:
            x = jnp.mean(x)

        return x

    def abs(self, x, y, quadrature=None, take_root=True):
        """absolute Lp-norm

        Parameters
        ----------
        x : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        quadrature : float or list, optional
            quadrature weights for integral
            either single scalar or one per dimension
        take_root : bool, optional
            whether to take the p-th root of the norm, by default True
        """
        # Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d

        # Flatten last d dimensions
        x_shape_prefix = x.shape[:-self.d]
        x_flat = jnp.reshape(x, x_shape_prefix + (-1,))
        y_flat = jnp.reshape(y, x_shape_prefix + (-1,))

        diff_flat = x_flat - y_flat

        if self.p == 1:
            const = math.prod(quadrature)
            diff = const * jnp.sum(jnp.abs(diff_flat), axis=-1, keepdims=False)
        elif self.p % 2 == 0:  # Even power p: no need for abs() since x^p > 0
            const = math.prod(quadrature)
            diff = const * jnp.sum(diff_flat**self.p, axis=-1, keepdims=False)
        else:
            const = math.prod(quadrature)
            diff = const * jnp.sum(
                jnp.abs(diff_flat) ** self.p, axis=-1, keepdims=False
            )

        if take_root and self.p != 1:
            diff = diff ** (1.0 / self.p)

        diff = self.reduce_all(diff)
        diff = jnp.squeeze(diff)

        return diff

    def rel(self, x, y, take_root=True):
        """
        rel: relative LpLoss
        computes ||x-y||/(||y|| + eps)

        Parameters
        ----------
        x : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        take_root : bool, optional
            whether to take the p-th root of the norm, by default True
        """
        # Flatten last d dimensions
        x_shape_prefix = x.shape[:-self.d]
        x_flat = jnp.reshape(x, x_shape_prefix + (-1,))
        y_flat = jnp.reshape(y, x_shape_prefix + (-1,))

        diff_flat = x_flat - y_flat

        if self.p == 1:
            diff = jnp.sum(jnp.abs(diff_flat), axis=-1, keepdims=False)
            ynorm = jnp.sum(jnp.abs(y_flat), axis=-1, keepdims=False)
        elif self.p % 2 == 0:  # Even power p: no need for abs() since x^p > 0
            diff = jnp.sum(diff_flat**self.p, axis=-1, keepdims=False)
            ynorm = jnp.sum(y_flat**self.p, axis=-1, keepdims=False)
        else:
            diff = jnp.sum(jnp.abs(diff_flat) ** self.p, axis=-1, keepdims=False)
            ynorm = jnp.sum(jnp.abs(y_flat) ** self.p, axis=-1, keepdims=False)

        if take_root and self.p != 1:
            diff = (diff ** (1.0 / self.p)) / (ynorm ** (1.0 / self.p) + self.eps)
        else:
            diff = diff / (ynorm + self.eps)

        diff = self.reduce_all(diff)
        diff = jnp.squeeze(diff)

        return diff

    def __call__(self, y_pred, y, **kwargs):
        if kwargs:
            warnings.warn(
                f"LpLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2
            )
        return self.rel(y_pred, y)

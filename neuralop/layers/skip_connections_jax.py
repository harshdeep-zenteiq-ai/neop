import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


def skip_connection(
    in_features, out_features, n_dim=2, bias=False, skip_type="soft-gating"
):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    n_dim : int, optional
        Dimensionality of the input (excluding batch-size and channels).
        n_dim=2 corresponds to having Module2D, by default 2
    bias : bool, optional
        Whether to use a bias, by default False
    skip_type : {'identity', 'linear', 'soft-gating'}, optional
        Kind of skip connection to use. Options: 'identity', 'linear', 'soft-gating', by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if skip_type.lower() == "soft-gating":
        return SoftGating(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    elif skip_type.lower() == "linear":
        return Flattened1dConv(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            bias=bias,
        )
    elif skip_type.lower() == "identity":
        return Identity()
    else:
        raise ValueError(
            f"Got skip-connection type={skip_type}, expected one of"
            f" {'soft-gating', 'linear', 'id'}."
        )


class Identity(nn.Module):
    """Simple identity module for use as a skip connection."""

    @nn.compact
    def __call__(self, x):
        return x


class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input.

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : None
        This is provided for API compatibility with nn.Linear only
    n_dim : int, optional
        Dimensionality of the input (excluding batch-size and channels).
        n_dim=2 corresponds to having Module2D, by default 2
    bias : bool, optional
        Whether to use a bias, by default False
    """

    in_features: int
    out_features: Optional[int] = None
    n_dim: int = 2
    bias: bool = False

    def setup(self):
        if self.out_features is not None and self.in_features != self.out_features:
            raise ValueError(
                f"Got in_features={self.in_features} and out_features={self.out_features}, "
                "but these two must be the same for soft-gating"
            )
        weight_shape = (1, self.in_features) + (1,) * self.n_dim
        self.weight = self.param("weight", nn.initializers.ones, weight_shape)
        if self.bias:
            self.bias_param = self.param("bias", nn.initializers.ones, weight_shape)
        else:
            self.bias_param = None

    def __call__(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias_param is not None:
            return self.weight * x + self.bias_param
        else:
            return self.weight * x


class Flattened1dConv(nn.Module):
    """Flattened1dConv is a Conv-based skip layer for
    input tensors of ndim > 3 (batch, channels, d1, ...) that flattens all dimensions
    past the batch and channel dims into one dimension, applies the Conv,
    and un-flattens.

    Parameters
    ----------
    in_channels : int
        in_channels of Conv1d
    out_channels : int
        out_channels of Conv1d
    kernel_size : int
        kernel_size of Conv1d
    bias : bool, optional
        bias of Conv1d, by default False
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    bias: bool = False

    def setup(self):
        self.conv = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            use_bias=self.bias,
        )

    def __call__(self, x):
        # x.shape: b, c, x1, ..., xn  (x_ndim > 1)
        size = list(x.shape)
        # flatten everything past 1st data dim
        x = x.reshape(*size[:2], -1)
        # JAX Conv expects (batch, spatial, features) — transpose in and out
        x = jnp.transpose(x, (0, 2, 1))
        x = self.conv(x)
        x = jnp.transpose(x, (0, 2, 1))
        # reshape x into an Nd tensor b, c, x1, x2, ...
        x = x.reshape(size[0], self.out_channels, *size[2:])
        return x

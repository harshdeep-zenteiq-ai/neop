from typing import List, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

from neuralop.utils_jax import validate_scaling_factor


class DomainPadding(nn.Module):
    """Applies domain padding scaled automatically to the input's resolution.

    Parameters
    ----------
    domain_padding : float or list
        Typically, between zero and one, percentage of padding to use
        if a list, make sure if matches the dim of (d1, ..., dN)
    resolution_scaling_factor : int, optional
        Resolution scaling factor, by default 1

    Notes
    -----
    This class works for any input resolution. Expects inputs of shape
    (batch-size, channels, d1, ...., dN)
    """

    domain_padding: Union[float, list]
    resolution_scaling_factor: Union[int, List[int]] = 1

    def setup(self):
        if self.resolution_scaling_factor is None:
            # mirror the None-guard from the PyTorch __init__
            object.__setattr__(self, 'resolution_scaling_factor', 1)

        # dict(f'{resolution}'=pad_width) such that padded = jnp.pad(x, pad_width)
        object.__setattr__(self, '_padding', dict())

        # dict(f'{resolution}'=indices_to_unpad) such that unpadded = x[indices]
        object.__setattr__(self, '_unpad_indices', dict())

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        forward pass: pad the input
        """
        return self.pad(x)

    def pad(self, x: jax.Array, verbose: bool = False) -> jax.Array:
        """Take an input and pad it by the desired fraction

        The amount of padding will be automatically scaled with the resolution
        """
        resolution = x.shape[2:]

        # if domain_padding is list, then to pass on
        domain_padding = self.domain_padding
        if isinstance(domain_padding, (float, int)):
            domain_padding = [float(domain_padding)] * len(resolution)

        assert len(domain_padding) == len(resolution), (
            "domain_padding length must match the number of spatial/time dimensions "
            "(excluding batch, ch)"
        )

        resolution_scaling_factor = self.resolution_scaling_factor
        if not isinstance(self.resolution_scaling_factor, list):
            # if unset by the user, scaling_factor will be 1 by default,
            # so `resolution_scaling_factor` should never be None.
            resolution_scaling_factor: List[float] = validate_scaling_factor(
                self.resolution_scaling_factor, len(resolution), n_layers=None
            )

        try:
            pad_width = self._padding[f"{resolution}"]
            return jnp.pad(x, pad_width, mode="constant")

        except KeyError:
            padding = [round(p * r) for (p, r) in zip(domain_padding, resolution)]

            if verbose:
                print(
                    f"Padding inputs of resolution={resolution} with "
                    f"padding={padding}, symmetric"
                )

            output_pad = padding

            output_pad = [
                round(i * j) for (i, j) in zip(resolution_scaling_factor, output_pad)
            ]

            # jnp.pad pad_width: no padding on (batch, channel), symmetric on spatial.
            # PyTorch F.pad takes a reversed flat list; jnp.pad takes a list of
            # (before, after) pairs in forward spatial order — mathematically identical.
            pad_width = [(0, 0), (0, 0)] + [(p, p) for p in padding]

            unpad_list = list()
            for p in output_pad:
                if p == 0:
                    padding_end = None
                    padding_start = None
                else:
                    padding_end = p
                    padding_start = -p
                unpad_list.append(slice(padding_end, padding_start, None))
            unpad_indices = (Ellipsis,) + tuple(unpad_list)

            self._padding[f"{resolution}"] = pad_width

            padded = jnp.pad(x, pad_width, mode="constant")

            output_shape = padded.shape[2:]

            output_shape = [
                round(i * j) for (i, j) in zip(resolution_scaling_factor, output_shape)
            ]

            self._unpad_indices[f"{[i for i in output_shape]}"] = unpad_indices

            return padded

    def unpad(self, x: jax.Array) -> jax.Array:
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices[f"{list(x.shape[2:])}"]
        return x[unpad_indices]

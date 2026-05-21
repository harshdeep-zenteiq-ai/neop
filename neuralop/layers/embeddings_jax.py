from abc import ABC, abstractmethod
from typing import List, Optional

import jax.numpy as jnp
import flax.linen as nn


class Embedding(nn.Module, ABC):
    @property
    @abstractmethod
    def out_channels(self):
        pass


class GridEmbedding2D(Embedding):
    """GridEmbedding2D applies a simple positional embedding as a regular 2D grid.

    It expects inputs of shape (batch, channels, d_1, d_2)

    Parameters
    ----------
    in_channels : int
        number of channels in input. Fixed for output channel interface
    grid_boundaries : list, optional
        coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
    """

    in_channels: int
    grid_boundaries: list = ((0, 1), (0, 1))

    def setup(self):
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + 2

    def grid(self, spatial_dims, dtype):
        """grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : tuple
            sizes of spatial resolution
        dtype :
            dtype to encode data.

        Returns
        -------
        jnp.ndarray
            output grids to concatenate
        """
        if self._grid is None or self._res != spatial_dims:
            grid_x, grid_y = regular_grid_2d(
                spatial_dims, grid_boundaries=self.grid_boundaries
            )
            grid_x = grid_x.astype(dtype)[jnp.newaxis, jnp.newaxis, ...]
            grid_y = grid_y.astype(dtype)[jnp.newaxis, jnp.newaxis, ...]
            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def __call__(self, data, batched=True):
        if not batched:
            if data.ndim == 3:
                data = data[jnp.newaxis, ...]
        batch_size = data.shape[0]
        x, y = self.grid(data.shape[-2:], data.dtype)
        out = jnp.concatenate(
            (data,
             jnp.broadcast_to(x, (batch_size, 1, *data.shape[-2:])),
             jnp.broadcast_to(y, (batch_size, 1, *data.shape[-2:]))),
            axis=1,
        )
        if not batched and batch_size == 1:
            return out.squeeze(0)
        else:
            return out


class GridEmbeddingND(nn.Module):
    """GridEmbeddingND applies a simple positional embedding as a regular ND grid.

    It expects inputs of shape (batch, channels, d_1, ..., d_n)

    Parameters
    ----------
    in_channels : int
        number of channels in input
    dim : int
        dimensions of positional encoding to apply
    grid_boundaries : list, optional
        coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
    """

    in_channels: int
    dim: int = 2
    grid_boundaries: tuple = ((0, 1), (0, 1))

    def setup(self):
        assert self.dim == len(self.grid_boundaries), (
            f"Error: expected grid_boundaries to be an iterable of length {self.dim}, "
            f"received {self.grid_boundaries}"
        )
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + self.dim

    def grid(self, spatial_dims, dtype):
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : tuple
            Sizes of spatial resolution.
        dtype :
            dtype to encode data.

        Returns
        -------
        jnp.ndarray
            output grids to concatenate
        """
        if self._grid is None or self._res != spatial_dims:
            grids_by_dim = regular_grid_nd(spatial_dims, grid_boundaries=self.grid_boundaries)
            grids_by_dim = [x.astype(dtype)[jnp.newaxis, jnp.newaxis, ...] for x in grids_by_dim]
            self._grid = grids_by_dim
            self._res = spatial_dims

        return self._grid

    def __call__(self, data, batched=True):
        """
        Parameters
        ----------
        data: jnp.ndarray
            assumes shape (batch (optional), channels, x_1, x_2, ...x_n)
        batched: bool
            whether data has a batch dim
        """
        if not batched:
            if data.ndim == self.dim + 1:
                data = data[jnp.newaxis, ...]
        batch_size = data.shape[0]
        grids = self.grid(spatial_dims=data.shape[2:], dtype=data.dtype)
        grids = [jnp.broadcast_to(x, (batch_size, 1, *data.shape[2:])) for x in grids]
        out = jnp.concatenate((data, *grids), axis=1)
        return out


class SinusoidalEmbedding(Embedding):
    """
    Sinusoidal positional embedding for enriching coordinate inputs with spectral information.

    Parameters
    ----------
    in_channels : int
        Number of input channels to embed (dimensionality of input coordinates)
    num_frequencies : int, optional
        Number of frequency levels L in the embedding.
    embedding_type : {'transformer', 'nerf'}, optional
        Type of embedding to apply, by default 'transformer'
    max_positions : int, optional
        Maximum number of positions for transformer-style encoding, by default 10000.
        Only used when embedding_type='transformer'.
    """

    in_channels: int
    num_frequencies: Optional[int] = None
    embedding_type: str = "transformer"
    max_positions: int = 10000

    def setup(self):
        allowed_embeddings = ["nerf", "transformer"]
        assert self.embedding_type in allowed_embeddings, (
            f"Error: embedding_type expected one of {allowed_embeddings}, "
            f"received {self.embedding_type}"
        )
        if self.embedding_type == "transformer":
            assert self.max_positions is not None, (
                "Error: max_positions must have an int value for transformer embedding."
            )

    @property
    def out_channels(self):
        return 2 * self.num_frequencies * self.in_channels

    def __call__(self, x):
        """
        Parameters
        ----------
        x: jnp.ndarray
            shape (n_in, self.in_channels) or (batch, n_in, self.in_channels)
        """
        assert x.ndim in [2, 3], (
            f"Error: expected inputs of shape (batch, n_in, {self.in_channels}) "
            f"or (n_in, channels), got inputs with ndim={x.ndim}, shape={x.shape}"
        )
        if x.ndim == 2:
            batched = False
            x = x[jnp.newaxis, ...]
        else:
            batched = True
        batch_size, n_in, _ = x.shape

        if self.embedding_type == "nerf":
            freqs = 2 ** jnp.arange(0, self.num_frequencies) * jnp.pi

        elif self.embedding_type == "transformer":
            freqs = jnp.arange(0, self.num_frequencies) / self.num_frequencies * 2
            freqs = (1 / self.max_positions) ** freqs

        # outer product of wavenumbers and position coordinates
        # shape b, n_in, channels, len(freqs)
        freqs = jnp.einsum("bij, k -> bijk", x, freqs)

        # shape b, n_in, channels, len(freqs), 2
        freqs = jnp.stack((jnp.sin(freqs), jnp.cos(freqs)), axis=-1)

        # transpose the inner per-entry matrix and ravel to interleave sin and cos
        freqs = freqs.reshape(batch_size, n_in, -1)

        if not batched:
            freqs = freqs.squeeze(0)
        return freqs


class RotaryEmbedding2D(nn.Module):
    dim: int
    min_freq: float = 1 / 64
    scale: float = 1.0

    def setup(self):
        """
        Applying rotary positional embedding (https://arxiv.org/abs/2104.09864) to the input feature tensor.
        """
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        self.inv_freq = inv_freq
        self.out_channels = 2

    def __call__(self, coordinates):
        """coordinates is array of [batch_size, num_points]"""
        coordinates = coordinates * (self.scale / self.min_freq)
        freqs = jnp.einsum("... i , j -> ... i j", coordinates, self.inv_freq)  # [b, n, d//2]
        return jnp.concatenate((freqs, freqs), axis=-1)  # [b, n, d]

    @staticmethod
    def apply_1d_rotary_pos_emb(t, freqs):
        return apply_rotary_pos_emb(t, freqs)

    @staticmethod
    def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
        """Split the last dimension of features into two equal halves
        and apply 1d rotary positional embedding to each half."""
        d = t.shape[-1]
        t_x, t_y = t[..., : d // 2], t[..., d // 2 :]
        return jnp.concatenate(
            (apply_rotary_pos_emb(t_x, freqs_x), apply_rotary_pos_emb(t_y, freqs_y)),
            axis=-1,
        )


# Utility functions for GridEmbedding
def regular_grid_2d(spatial_dims, grid_boundaries=((0, 1), (0, 1))):
    """
    Creates a 2 x height x width stack of positional encodings A, where
    A[:,i,j] = [[x,y]] at coordinate (i,j) on a (height, width) grid.
    """
    height, width = spatial_dims

    xt = jnp.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = jnp.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = jnp.meshgrid(xt, yt, indexing="ij")

    return grid_x, grid_y


def regular_grid_nd(
    resolutions: List[int], grid_boundaries: List[List[int]] = ((0, 1), (0, 1))
):
    """regular_grid_nd generates a tuple of coordinate arrays for a bounded regular grid.

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the regular grid.

    Returns
    -------
    grid: tuple of jnp.ndarray
    """
    assert len(resolutions) == len(grid_boundaries), \
        "Error: inputs must have same number of dimensions"

    meshgrid_inputs = []
    for res, (start, stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(jnp.linspace(start, stop, res + 1)[:-1])
    grid = jnp.meshgrid(*meshgrid_inputs, indexing="ij")
    return tuple(grid)


# Utility functions for Rotary embedding
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
def rotate_half(x):
    """Split x's channels into two equal halves."""
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x[..., 0, :], x[..., 1, :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply rotation matrix computed based on freqs to rotate t.
    t: array of shape [batch_size, num_points, dim]
    freqs: array of shape [batch_size, num_points, 1]

    Formula: see equation (34) in https://arxiv.org/pdf/2104.09864.pdf
    """
    return (t * jnp.cos(freqs)) + (rotate_half(t) * jnp.sin(freqs))

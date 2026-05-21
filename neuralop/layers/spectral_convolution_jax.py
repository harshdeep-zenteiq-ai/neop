from typing import List, Optional, Tuple, Union

from ..utils_jax import validate_scaling_factor

import jax
import jax.numpy as jnp
import flax.linen as nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from .einsum_utils_jax import einsum_complexhalf
from .base_spectral_conv_jax import BaseSpectralConv
from .resample_jax import resample

tl.set_backend("jax")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

class JAXTuckerTensor:
    """Lightweight Tucker tensor for JAX — replaces tltorch.TuckerTensor.

    Stores a core tensor and a list of factor matrices as plain JAX arrays.
    The interface mirrors what ``_contract_tucker`` expects (.core, .factors).
    """

    def __init__(self, core: jnp.ndarray, factors):
        self.core    = core
        self.factors = list(factors)

    @property
    def shape(self):
        """Shape of the full (reconstructed) tensor, derived from factor rows."""
        return tuple(f.shape[0] for f in self.factors)

    def __getitem__(self, slices):
        """Slice the Tucker tensor by slicing each factor matrix's row dimension."""
        if not isinstance(slices, tuple):
            slices = (slices,)
        new_factors = []
        for i, factor in enumerate(self.factors):
            s = slices[i] if i < len(slices) else slice(None)
            new_factors.append(factor[s, :])   # slice rows; keep rank dim intact
        return JAXTuckerTensor(self.core, new_factors)

def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not isinstance(weight, jnp.ndarray):
        weight = weight.to_tensor()

    # if x.dtype == jnp.complex64:
        # return einsum_complexhalf(eq, x, weight)
    # else:
        # return tl.einsum(eq, x, weight)
    return tl.einsum(eq, x, weight)


def _contract_dense_separable(x, weight, separable):
    if not isinstance(weight, jnp.ndarray):
        weight = weight.to_tensor()
    return x * weight


def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    # if x.dtype == jnp.complex64:
        # return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    # else:
        # return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)
    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)

def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]
    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    # if x.dtype == jnp.complex64:
        # return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    # else:
        # return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)
    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    # if x.dtype == jnp.complex64:
        # return einsum_complexhalf(eq, x, *tt_weight.factors)
    # else:
        # return tl.einsum(eq, x, *tt_weight.factors)
    return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors.
        if False,
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    # JAX-native Tucker is always contracted factorized (no reconstruction)
    if isinstance(weight, JAXTuckerTensor):
        return _contract_tucker

    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if isinstance(weight, jnp.ndarray):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


Number = Union[int, float]


class SpectralConv(BaseSpectralConv):
    """SpectralConv implements the Spectral Convolution component of a Fourier layer.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    n_modes : int or int tuple
    complex_data : bool, optional
    max_n_modes : int tuple or None, optional
    bias : bool, optional
    separable : bool, optional
    resolution_scaling_factor : float, list of float, or None, optional
    fno_block_precision : str, optional
    rank : float, optional
    factorization : str or None, optional
    implementation : {'factorized', 'reconstructed'}, optional
    enforce_hermitian_symmetry : bool, optional
    fixed_rank_modes : bool, optional
    decomposition_kwargs : dict or None, optional
    init_std : float or 'auto', optional
    fft_norm : str, optional
    """

    in_channels: int
    out_channels: int
    n_modes: Union[int, Tuple]
    complex_data: bool = False
    max_n_modes: Optional[Tuple] = None
    bias: bool = True
    separable: bool = False
    resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None
    fno_block_precision: str = "full"
    rank: float = 1.0
    factorization: Optional[str] = None
    implementation: str = "reconstructed"
    enforce_hermitian_symmetry: bool = True
    fixed_rank_modes: Union[bool, List[int]] = False
    decomposition_kwargs: Optional[dict] = None
    init_std: Union[str, float] = "auto"
    fft_norm: str = "forward"

    def setup(self):
        n_modes = self.n_modes
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes
        self.order = len(self._n_modes)

        max_n_modes = self.max_n_modes
        if max_n_modes is None:
            max_n_modes = self._n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self._max_n_modes = list(max_n_modes)

        self._resolution_scaling_factor = validate_scaling_factor(
            self.resolution_scaling_factor, self.order
        )

        init_std = self.init_std
        if init_std == "auto":
            init_std = (2 / (self.in_channels + self.out_channels)) ** 0.5
        self._init_std = init_std

        fixed_rank_modes = self.fixed_rank_modes
        if isinstance(fixed_rank_modes, bool):
            fixed_rank_modes = [0] if fixed_rank_modes else None
        self._fixed_rank_modes = fixed_rank_modes

        # Use the config factorization (tucker or dense).
        # CP/TT fall back to dense since only tucker is implemented in JAX.
        factorization = self.factorization if self.factorization is not None else "Dense"
        if factorization.lower() not in ("dense", "tucker"):
            factorization = "Dense"

        if self.separable:
            if self.in_channels != self.out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={self.in_channels} and "
                    f"out_channels={self.out_channels}",
                )
            weight_shape = (self.in_channels, *self._max_n_modes)
        else:
            weight_shape = (self.in_channels, self.out_channels, *self._max_n_modes)

        # Create spectral weight tensor as native Flax parameters.
        # Dense: one complex array of shape weight_shape.
        # Tucker: core tensor + per-dim factor matrices — 70x smaller than dense.
        def _cx_init(rng, shape):
            std = self._init_std
            return (std * jax.random.normal(rng, shape, dtype=jnp.float32)
                    + 1j * std * jax.random.normal(rng, shape, dtype=jnp.float32)
                    ).astype(jnp.complex64)

        if factorization.lower() == "dense":
            self.weight = self.param("weight", _cx_init, weight_shape)
            
        elif factorization.lower() == "tucker":
            # Use tensorly's validate_tucker_rank for parity with PyTorch's
            # tltorch.TuckerTensor.new(...), which calls the same function.
            # For a float rank in (0, 1], it is treated as a compression ratio
            # (target Tucker params = rank * full tensor params), NOT a
            # per-dim multiplier. A simple `round(rank * d)` would under-shoot
            # by a large factor and produce a different shape than PyTorch.

            from tensorly.tucker_tensor import validate_tucker_rank
            tucker_ranks = tuple(
                int(r) for r in validate_tucker_rank(
                    weight_shape,
                    rank=self.rank,
                    fixed_modes=self._fixed_rank_modes,
                )
            )
               
            # Core tensor
            self._w_core = self.param("w_core", _cx_init, tucker_ranks)

            # Factor matrices — one per dimension of weight_shape
            for i, (d, r) in enumerate(zip(weight_shape, tucker_ranks)):
                setattr(self, f'_w_U{i}', self.param(f'w_U{i}', _cx_init, (d, r)))

            n_dims = len(weight_shape)
            self.weight = JAXTuckerTensor(
                self._w_core,
                [getattr(self, f'_w_U{i}') for i in range(n_dims)],
            )
        else:
            raise ValueError(
                f"JAX port only supports 'dense' or 'tucker' factorization, got '{factorization}'"
            )

        self._contract = get_contract_fun(
            self.weight, implementation=self.implementation, separable=self.separable
        )

        if self.bias:
            # Flax parameter: shape (out_channels, 1, 1, ...) with `order` trailing 1s
            self._bias = self.param(
                "bias",
                lambda rng, shape: self._init_std * jnp.zeros(shape, dtype=jnp.float32),
                tuple([self.out_channels]) + (1,) * self.order,
            )
        else:
            self._bias = None

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if self._resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self._resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == list(out_shape):
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    @nn.nowrap
    def _get_n_modes(self):
        return self._n_modes

    def __call__(self, x: jnp.ndarray, output_shape: Optional[Tuple[int]] = None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : jnp.ndarray
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        import time
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.astype(jnp.float16)

        t_fft_start = time.perf_counter()
        if self.complex_data:
            x = jnp.fft.fftn(x, norm=self.fft_norm, axes=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = jnp.fft.rfftn(x, norm=self.fft_norm, axes=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]

        if self.order > 1:
            x = jnp.fft.fftshift(x, axes=dims_to_fft_shift)
        t_fft_end = time.perf_counter()
        # print(f"    [SpectralConv FFT] Time: {(t_fft_end - t_fft_start)*1000:.3f}ms")

        if self.fno_block_precision == "mixed":
            x = x.astype(jnp.complex64)  # JAX has no chalf; use complex64

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = jnp.complex64   # JAX has no complex32/chalf
        else:
            out_dtype = jnp.complex64

        out_fft = jnp.zeros(
            [batchsize, self.out_channels, *fft_size], dtype=out_dtype
        )

        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self._n_modes, self._max_n_modes)
        ]

        if self.separable:
            slices_w = [slice(None)]  # channels
        else:
            slices_w = [slice(None), slice(None)]  # in_channels, out_channels

        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]

        if self.separable:
            weight_start_idx = 1
        else:
            weight_start_idx = 2

        slices_x = [slice(None), slice(None)]  # batch_size, channels

        for all_modes, kept_modes in zip(fft_size, list(weight.shape[weight_start_idx:])):
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        slices_x = tuple(slices_x)
        t_contract_start = time.perf_counter()
        out_fft = out_fft.at[slices_x].set(
            self._contract(x[slices_x], weight, separable=self.separable).astype(out_fft.dtype)
        )
        t_contract_end = time.perf_counter()
        # print(f"    [SpectralConv Contraction] Time: {(t_contract_end - t_contract_start)*1000:.3f}ms")

        if self._resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self._resolution_scaling_factor)])

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = jnp.fft.ifftshift(out_fft, axes=fft_dims[:-1])

        t_ifft_start = time.perf_counter()
        # Inverse FFT
        if self.complex_data:
            x = jnp.fft.ifftn(out_fft, s=mode_sizes, axes=fft_dims, norm=self.fft_norm)
        else:
            if self.enforce_hermitian_symmetry:
                out_fft = jnp.fft.ifftn(out_fft, s=mode_sizes[:-1], axes=fft_dims[:-1], norm=self.fft_norm)

                # Enforce Hermitian symmetry conditions for irfft
                # 0th frequency must be real
                # out_fft = out_fft.at[..., 0].set(out_fft[..., 0].real + 0j)
                out_fft = out_fft.at[..., 0].set((out_fft[..., 0].real + 0j).astype(out_fft.dtype))

                # Nyquist frequency must be real if the spatial size is even
                if mode_sizes[-1] % 2 == 0:
                    # out_fft = out_fft.at[..., -1].set(out_fft[..., -1].real + 0j)
                    out_fft = out_fft.at[..., -1].set((out_fft[..., -1].real + 0j).astype(out_fft.dtype))

                x = jnp.fft.irfft(out_fft, n=mode_sizes[-1], axis=fft_dims[-1], norm=self.fft_norm)
            else:
                x = jnp.fft.irfftn(out_fft, s=mode_sizes, axes=fft_dims, norm=self.fft_norm)
        t_ifft_end = time.perf_counter()
        # print(f"    [SpectralConv IFFT] Time: {(t_ifft_end - t_ifft_start)*1000:.3f}ms")

        if self._bias is not None:
            x = x + self._bias

        return x


        # elif factorization.lower() == "tucker":
        #     import math
        #     rank = self.rank if isinstance(self.rank, float) else 1.0
        #     # Compute per-dimension Tucker ranks
        #     # tucker_ranks = tuple(max(1, math.ceil(rank * d)) for d in weight_shape)
        #     tucker_ranks = tuple(max(1, int(round(rank * d))) for d in weight_shape)
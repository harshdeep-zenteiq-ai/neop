"""
Functionality for handling complex-valued spatial data
"""

import jax.numpy as jnp
import flax.linen as nn


def CGELU(x: jnp.ndarray):
    """Complex GELU activation function
    Follows the formulation of CReLU from [1]_.
    Applies GELU to real and imaginary parts of the input
    separately, then combine as complex number


    Parameters
    -----------
    x : jnp.ndarray (dtype=complex)
        pre-activation inputs

    References
    ----------
    .. [1] :

    Trabelsi, C., et al. (2018). "Deep Complex Networks".
        ICLR 2018, https://openreview.net/pdf?id=H1T2hmZAb.
    """

    return nn.gelu(x.real).astype(jnp.complex64) + 1j * nn.gelu(x.imag).astype(jnp.complex64)


def cselu(x: jnp.ndarray):
    """Complex-valued SELU activation
    Apply SELU to real and imag part of the input separately, then combine as complex number
    Args:
        x: complex array
    """
    return nn.selu(x.real).astype(jnp.complex64) + 1j * nn.selu(x.imag).astype(jnp.complex64)


def ctanh(x: jnp.ndarray):
    """Complex-valued tanh stabilizer
    Apply ctanh is real and imag part of the input separately, then combine as complex number
    Args:
        x: complex array
    """
    return jnp.tanh(x.real).astype(jnp.complex64) + 1j * jnp.tanh(x.imag).astype(jnp.complex64)


def apply_complex(real_func, imag_func, x, dtype=jnp.complex64):
    """
    fr: a function (e.g., conv) to be applied on real part of x
    fi: a function (e.g., conv) to be applied on imag part of x
    x: complex input.
    """
    return (real_func(x.real) - imag_func(x.imag)).astype(dtype) + 1j * (real_func(x.imag) + imag_func(x.real)).astype(dtype)


class ComplexValued(nn.Module):
    """
    Wrapper class that converts a standard nn.Module that operates on real data
    into a module that operates on complex-valued spatial data.
    """

    module: nn.Module

    def setup(self):
        self.fr = self.module.clone()
        self.fi = self.module.clone()

    def __call__(self, x):
        return apply_complex(self.fr, self.fi, x)

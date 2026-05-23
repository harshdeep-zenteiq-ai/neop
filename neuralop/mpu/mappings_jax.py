# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
from typing import Any

import jax

from .comm_jax import get_model_parallel_group

# helper functions
from .helpers_jax import split_tensor_along_dim
from .helpers_jax import _reduce
from .helpers_jax import _split
from .helpers_jax import _gather


# model parallel
class _CopyToModelParallelRegion:
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=get_model_parallel_group())

    @staticmethod
    def apply(input_):
        # JAX custom_vjp: forward = identity, backward = all-reduce gradient
        @jax.custom_vjp
        def fn(x):
            return x

        def fn_fwd(x):
            return x, None

        def fn_bwd(_, g):
            return (_reduce(g, group=get_model_parallel_group()),)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn(input_)


class _ReduceFromModelParallelRegion:
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_, group=get_model_parallel_group())

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    @staticmethod
    def apply(input_):
        # JAX custom_vjp: forward = all-reduce, backward = identity
        @jax.custom_vjp
        def fn(x):
            return _reduce(x, group=get_model_parallel_group())

        def fn_fwd(x):
            return fn(x), None

        def fn_bwd(_, g):
            return (g,)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn(input_)


class _ScatterToModelParallelRegion:
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _split(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _split(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, group=get_model_parallel_group()), None

    @staticmethod
    def apply(input_, dim_):
        # dim_ is closed over so fn takes only the differentiable array input
        @jax.custom_vjp
        def fn(x):
            return _split(x, dim_, group=get_model_parallel_group())

        def fn_fwd(x):
            return fn(x), None

        def fn_bwd(_, g):
            return (_gather(g, dim_, group=get_model_parallel_group()),)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn(input_)


class _GatherFromModelParallelRegion:
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _gather(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _gather(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, group=get_model_parallel_group()), None

    @staticmethod
    def apply(input_, dim_):
        # dim_ is closed over so fn takes only the differentiable array input
        @jax.custom_vjp
        def fn(x):
            return _gather(x, dim_, group=get_model_parallel_group())

        def fn_fwd(x):
            return fn(x), None

        def fn_bwd(_, g):
            return (_split(g, dim_, group=get_model_parallel_group()),)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn(input_)


# -----------------
# Helper functions.
# -----------------
# matmul parallel
def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_, dim):
    return _ScatterToModelParallelRegion.apply(input_, dim)


def gather_from_model_parallel_region(input_, dim):
    return _GatherFromModelParallelRegion.apply(input_, dim)

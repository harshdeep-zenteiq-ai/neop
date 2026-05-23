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


import jax
import jax.numpy as jnp


def get_memory_format(tensor):
    # JAX arrays have no memory_format attribute; always row-major (C-contiguous)
    return None


def pad_helper(tensor, dim, new_size, mode="zero"):
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    # Convert PyTorch F.pad flat reversed list to jnp.pad (before, after) pairs in forward order
    n_pairs = len(output_shape) // 2
    pairs = [(output_shape[2 * i], output_shape[2 * i + 1]) for i in range(n_pairs)]
    jnp_pad_widths = [(0, 0)] * (ndim - n_pairs) + list(reversed(pairs))
    tensor_pad = jnp.pad(tensor, jnp_pad_widths, mode="constant")

    if mode == "conj":
        lhs_slice = [
            slice(0, x) if idx != dim else slice(orig_size, new_size)
            for idx, x in enumerate(tensor.shape)
        ]
        rhs_slice = [
            slice(0, x) if idx != dim else slice(1, output_shape[1] + 1)
            for idx, x in enumerate(tensor.shape)
        ]
        tensor_pad = tensor_pad.at[tuple(lhs_slice)].set(
            jnp.flip(jnp.conj(tensor_pad[tuple(rhs_slice)]), axis=dim)
        )

    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        slice(0, x) if idx != dim else slice(0, new_size)
        for idx, x in enumerate(tensor.shape)
    ]
    tensor_trunc = tensor[tuple(output_slice)]

    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    assert (
        dim < tensor.ndim
    ), f"Error, tensor dimension is {tensor.ndim} which cannot be split along {dim}"
    assert (
        tensor.shape[dim] % num_chunks == 0
    ), f"Error, cannot split dim {dim} evenly. Dim size is \
                                                  {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    tensor_list = jnp.split(tensor, num_chunks, axis=dim)

    return tensor_list


# distributed primitives
def _transpose(tensor, dim0, dim1, group=None, async_op=False):
    # get input format
    input_format = get_memory_format(tensor)

    # get comm params
    comm_size = len(group) if group is not None else jax.process_count()

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [y for y in jnp.split(tensor, comm_size, axis=dim0)]
    x_recv = [jnp.empty_like(x_send[0]) for _ in range(comm_size)]

    # global transposition
    transposed = jax.lax.all_to_all(
        tensor, axis_name='batch', split_axis=dim0, concat_axis=dim1
    )
    x_recv = list(jnp.split(transposed, comm_size, axis=dim1))
    req = None

    return x_recv, req


def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    comm_size = len(group) if group is not None else jax.process_count()
    if comm_size == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.astype(jnp.float32)
        inputf_ = jax.lax.psum(inputf_, axis_name='batch')
        input_ = inputf_.astype(dtype)
    else:
        input_ = jax.lax.psum(input_, axis_name='batch')

    return input_


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = len(group) if group is not None else jax.process_count()
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: jnp.split does not create contiguous tensors by default.
    rank = jax.process_index()
    output = input_list[rank]

    return output


def _gather(input_, dim_, group=None):
    """Gather tensors and concatinate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_)

    comm_size = len(group) if group is not None else jax.process_count()
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert (
        dim_ < input_.ndim
    ), f"Error, cannot gather along {dim_} for tensor with {input_.ndim} dimensions."

    # Size and dimension.
    comm_rank = jax.process_index()

    tensor_list = [jnp.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    gathered = jax.lax.all_gather(input_, axis_name='batch')

    # Note: jnp.concatenate already creates a contiguous tensor.
    output = jnp.concatenate([gathered[i] for i in range(comm_size)], axis=dim_)

    return output

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


import os
import logging
import jax
import datetime as dt


class disable_logging(object):
    def __init__(self, level=logging.ERROR):
        logging.disable(level=level)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        logging.disable(level=logging.NOTSET)


# dummy placeholders
# In JAX, process groups are represented as lists of process-index integers
# rather than torch.distributed group objects.
_initialized = False
_DATA_PARALLEL_GROUP = None
_MODEL_PARALLEL_GROUP = None


# world comm
def get_world_size():
    if not _initialized:
        return 1
    else:
        return jax.process_count()


def get_local_rank():
    if not _initialized:
        return 0
    else:
        return jax.process_index()


def get_global_rank():
    if not _initialized:
        return 0
    else:
        return jax.process_index()


# data parallel
def get_data_parallel_size():
    if not _initialized:
        return 1
    else:
        return len(_DATA_PARALLEL_GROUP) if _DATA_PARALLEL_GROUP is not None else jax.process_count()


def get_data_parallel_rank():
    if not _initialized:
        return 0
    else:
        return jax.process_index()


def get_data_parallel_group():
    assert _initialized, "Error, initialize JAX distributed first"
    return _DATA_PARALLEL_GROUP


# model parallel
def get_model_parallel_size():
    if not _initialized or (_MODEL_PARALLEL_GROUP is None):
        return 1
    else:
        return len(_MODEL_PARALLEL_GROUP)


def get_model_parallel_rank():
    if not _initialized or (_MODEL_PARALLEL_GROUP is None):
        return 0
    else:
        return jax.process_index()


def get_model_parallel_group():
    assert _initialized, "Error, initialize JAX distributed first"
    return _MODEL_PARALLEL_GROUP


def init(model_parallel_size: int = 1, verbose: bool = False):
    """
    Set up global and local communicator.
    JAX distributed must be initialized externally via jax.distributed.initialize()
    before calling this function. Process groups are represented as lists of
    process-index integers (JAX has no group-handle objects).
    """

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    global_rank = int(os.getenv("RANK", 0))
    world_size = jax.local_device_count()

    global _initialized
    global _DATA_PARALLEL_GROUP
    global _MODEL_PARALLEL_GROUP

    if world_size > 1:
        with disable_logging():
            _initialized = True

            # once initialized, get true values for rank and size using jax
            world_size = get_world_size()
            local_rank = get_local_rank()

            # set a barrier until all processes reach this point
            jax.effects_barrier()

    # process 0 is logger
    is_logger = get_local_rank() == 0

    # get model groups
    model_group_size = model_parallel_size

    # compute data parallel size
    data_group_size = world_size // model_group_size

    if is_logger:
        print(f"Using {world_size} in {model_group_size} x {data_group_size} decomposition (#model-ranks x #data-ranks)")

    assert ( (model_group_size <= world_size) and (world_size % model_group_size == 0) ), \
        "Error, please make sure matmul_parallel_size * spatial_parallel_size <= world size and that world size is evenly divisible by matmul_parallel_size * spatial_parallel_size"

    # number of model groups
    num_model_groups = world_size // model_group_size

    if is_logger:
        print("Starting Wireup")

    if world_size > 1:
        if model_group_size > 1:
            model_groups = []
            for i in range(num_model_groups):
                start = i * model_group_size
                end = start + model_group_size
                model_groups.append(list(range(start, end)))

            data_groups = [sorted(list(i)) for i in zip(*model_groups)]

            if verbose and is_logger:
                print("Model Parallel Groups w/ respect to world rank:")
                for grp in model_groups:
                    print(grp)

            if verbose and is_logger:
                print("Data Parallel Groups w/ respect to world rank:")
                for grp in data_groups:
                    print(grp)

            # initialize groups — stored as lists of process-index integers
            with disable_logging():
                # data groups
                for grp in data_groups:
                    if global_rank in grp:
                        _DATA_PARALLEL_GROUP = grp
                # model groups
                for grp in model_groups:
                    if global_rank in grp:
                        _MODEL_PARALLEL_GROUP = grp

        else:
            # technically unnecessary but we do it to be clean
            with disable_logging():
                _MODEL_PARALLEL_GROUP = [global_rank]
                _DATA_PARALLEL_GROUP = list(range(world_size))

    # barrier
    if _initialized:
        jax.effects_barrier()

    if is_logger:
        print("Finished Wireup")

    return

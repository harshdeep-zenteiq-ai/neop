import jax
import neuralop.mpu.comm_jax as comm


def setup(config):
    """A convenience function to intialize the device, setup JAX settings and
    check multi-grid and other values. It sets up distributed communication, if used.

    Parameters
    ----------
    config : dict
        this function checks:
        * config.distributed (use_distributed, seed)
        * config.data (n_train, batch_size, test_batch_sizes, n_tests, test_resolutions)

    Returns
    -------
    device, is_logger
        device : jax.Device
        is_logger : bool
    """
    seed = config.distributed.seed

    if config.distributed.use_distributed:
        comm.init(
            model_parallel_size=config.distributed.model_parallel_size,
            verbose=config.verbose,
        )

        # Set process 0 to log screen and wandb
        is_logger = comm.get_local_rank() == 0

        # Set device for local rank
        try:
            gpus = jax.devices('gpu')
        except RuntimeError:
            gpus = []
        if gpus:
            device = gpus[comm.get_local_rank() % len(gpus)]
        else:
            device = jax.devices('cpu')[0]

        if seed is not None:
            seed = seed + comm.get_data_parallel_rank()

        # Ensure batch can be evenly split among the model-parallel group
        if config.patching.levels > 0:
            assert(config.data.batch_size*(2**(2*config.patching.levels)) % comm.get_model_parallel_size() == 0), (
                f'With MG patching, total batch-size of {config.data.batch_size*(2**(2*config.patching.levels))}'
                f' ({config.data.batch_size} times {(2**(2*config.patching.levels))}).'
                f' However, this total batch-size cannot be evenly split among the {comm.get_model_parallel_size()} model-parallel groups.'
            )
            for b_size in config.data.test_batch_sizes:
                assert (b_size*(2**(2*config.patching.levels)) % comm.get_model_parallel_size() == 0), (
                f'With MG patching, for test resolution of {config.data.test_resolutions[j]}'
                f' the total batch-size is {config.data.batch_size*(2**(2*config.patching.levels))}'
                f' ({config.data.batch_size} times {(2**(2*config.patching.levels))}).'
                f' However, this total batch-size cannot be evenly split among the {comm.get_model_parallel_size()} model-parallel groups.'
                )

    else:
        is_logger = True
        try:
            gpus = jax.devices('gpu')
        except RuntimeError:
            gpus = []
        if gpus:
            device = gpus[0]
        else:
            device = jax.devices('cpu')[0]

    # Set device, optimization settings
    # JAX manages random state via explicit PRNG keys rather than global seeds;
    # seed is retained here for downstream construction of jax.random.PRNGKey(seed).
    has_gpu = any(d.platform == 'gpu' for d in jax.devices())
    if has_gpu:
        try:
            jax.config.update("jax_default_device", device)
        except Exception:
            pass

        increase_l2_fetch_granularity()
        try:
            jax.config.update("jax_default_matmul_precision", "high")
        except AttributeError:
            pass

    return device, is_logger


def increase_l2_fetch_granularity():
    try:
        import ctypes

        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128
    except:
        return

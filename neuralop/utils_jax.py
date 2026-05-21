from typing import List, Optional, Union
from math import prod
from pathlib import Path
import jax 
import jax.numpy as jnp 
import numpy as np  

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

def count_model_params(params):
    """Returns the total number of parameters of a JAX/Flax model.

    Notes
    -----
    One complex number is counted as two parameters (real and imaginary parts).
    """
    leaves = jax.tree_util.tree_leaves(params)
    return sum(
        p.size * 2 if np.iscomplexobj(p) else p.size
        for p in leaves
    )

def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : jax.numpy array
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if np.iscomplexobj(tensor):
        return 2 * n_params
    return n_params


def wandb_login(api_key_file="../config/wandb_api_key.txt", key=None):
    if key is None:
        key = get_wandb_api_key(api_key_file)

    wandb.login(key=key)


def set_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        os.environ["WANDB_API_KEY"] = key.strip()


def get_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()


# Define the function to compute the spectrum
def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Parameters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.
    normalize: bool
        whether to apply normalization to the output of the 2D FFT.
        If True, normalizes the outputs by ``1/n_observations``
        (actually ``1/sqrt(n_observations * n_observations)``).
    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
        The spectrum is computed using a square approximation to radial
        binning, meaning that the wavenumber 'bin' into which a particular
        coefficient is the coefficient's location along the diagonal, indexed
        from the top-left corner of the 2d FFT output.
    """
    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = jnp.fft.fft2(signal, norm="ortho")
    else:
        signal = jnp.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2d wavenumbers following fft convention
    k_max = n_observations // 2
    wavenumbers = jnp.concatenate(
        (
            jnp.arange(start=0, stop=k_max, step=1),
            jnp.arange(start=-k_max, stop=0, step=1),
        )
    )
    wavenumbers = jnp.tile(wavenumbers, (n_observations, 1))
    k_x = wavenumbers.T
    k_y = wavenumbers

    # Sum wavenumbers
    sum_k = jnp.sqrt(k_x**2 + k_y**2)

    # Remove symmetric components from wavenumbers
    index = -1.0 * jnp.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index = index.at[0:k_max1, 0:k_max1].set(sum_k[0:k_max1, 0:k_max1])

    spectrum = jnp.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = jnp.where(index == j)
        spectrum = spectrum.at[:, j - 1].set(
            (jnp.abs(signal[:, ind[0], ind[1]]) ** 2).sum(axis=1)
        )

    spectrum = spectrum.mean(axis=0)
    return spectrum

Number = Union[float, int]


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        if n_layers is None and len(scaling_factor) == n_dim:
            # this is a dim-wise scaling
            return [float(s) for s in scaling_factor]
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None


def compute_rank(tensor):
    # Compute the matrix rank of a tensor
    rank = jnp.linalg.matrix_rank(tensor)
    return rank


def compute_stable_rank(tensor):
    # Compute the stable rank of a tensor
    fro_norm = jnp.linalg.norm(tensor, ord="fro") ** 2
    l2_norm = jnp.linalg.norm(tensor, ord=2) ** 2
    rank = fro_norm / l2_norm
    return rank


def compute_explained_variance(frequency_max, s):
    # Compute the explained variance based on frequency_max and singular values (s)
    s_current = s.at[frequency_max:].set(0)
    return 1 - jnp.var(s - s_current) / jnp.var(s)

def get_project_root():
    root = Path(__file__).parent.parent
    return root


class NeighborSearchLogger:
    """Logger for GNO neighbor search results."""

    def __init__(self, log_dir: str = "./neighbor_logs"):
        """
        Initialize the logger.

        Parameters
        ----------
        log_dir : str, optional
            Directory to store log files, by default "./neighbor_logs"
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_gno_neighbors(
        self,
        neighbors_dict,
        query_points,
        data_points,
        epoch: int,
        stage: str,  # "train" or "test"
        gno_type: str,  # "gno_in" or "gno_out"
    ):
        """
        Log neighbor search results to file.

        Parameters
        ----------
        neighbors_dict : Dict[str, torch.Tensor]
            Dictionary from neighbor_search with keys:
            - 'neighbors_index': [n_neighbors] flat list of neighbor indices
            - 'neighbors_row_splits': [n_queries + 1] cumulative indices
            - 'weights' (optional): [n_neighbors] squared distances
        query_points : torch.Tensor
            Query points of shape [n_queries, d] or [batch, n_queries, d]
        data_points : torch.Tensor
            Data points of shape [n_data, d] or [batch, n_data, d]
        epoch : int
            Epoch number
        stage : str
            "train" or "test"
        gno_type : str
            "gno_in" or "gno_out"
        """
        # Handle batched inputs - use first batch if batched
        if query_points.ndim > 2:
            query_points = query_points[0]
        if data_points.ndim > 2:
            data_points = data_points[0]

        # Move to CPU and convert to numpy for easier handling
        neighbors_index = neighbors_dict["neighbors_index"].cpu().numpy()
        neighbors_row_splits = neighbors_dict["neighbors_row_splits"].cpu().numpy()
        weights = neighbors_dict.get("weights")
        if weights is not None:
            weights = weights.cpu().numpy()

        query_points = query_points.cpu().numpy()
        data_points = data_points.cpu().numpy()

        n_queries = len(neighbors_row_splits) - 1

        # Create filename
        filename = self.log_dir / f"{gno_type}_{stage}_epoch_{epoch}.txt"

        with open(filename, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"Neighbor Search Log: {gno_type.upper()} ({stage.upper()})\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Summary Statistics:\n")
            f.write(f"  Total query points: {n_queries}\n")
            f.write(f"  Total data points: {len(data_points)}\n")
            f.write(f"  Total neighbors found: {len(neighbors_index)}\n")
            f.write(f"  Average neighbors per query: {len(neighbors_index) / n_queries:.2f}\n")
            f.write(f"\n{'='*80}\n\n")

            # Log details for each query point
            for q_idx in range(n_queries):
                start_idx = int(neighbors_row_splits[q_idx])
                end_idx = int(neighbors_row_splits[q_idx + 1])
                n_neighbors = end_idx - start_idx

                f.write(f"Query Point {q_idx}:\n")
                f.write(f"  Coordinates: {query_points[q_idx]}\n")
                f.write(f"  Number of neighbors: {n_neighbors}\n")

                if n_neighbors > 0:
                    neighbor_indices = neighbors_index[start_idx:end_idx]
                    f.write(f"  Neighbor indices: {neighbor_indices.tolist()}\n")

                    # Write neighbor details
                    for i, nbr_idx in enumerate(neighbor_indices):
                        nbr_idx = int(nbr_idx)
                        distance = None
                        if weights is not None:
                            distance = weights[start_idx + i]

                        f.write(f"    [{i}] data_idx={nbr_idx}, ")
                        f.write(f"coords={data_points[nbr_idx]}, ")
                        if distance is not None:
                            f.write(f"sq_distance={distance:.6f}\n")
                        else:
                            f.write(f"\n")
                else:
                    f.write(f"  No neighbors found within radius\n")

                f.write(f"\n")

        print(f"✓ Logged {gno_type} {stage} neighbors to: {filename}")

    def log_epoch(
        self,
        gno_in_neighbors_train,
        gno_in_query_points_train,
        gno_in_data_points_train,
        gno_out_neighbors_train,
        gno_out_query_points_train,
        gno_out_data_points_train,
        gno_in_neighbors_test=None,
        gno_in_query_points_test=None,
        gno_in_data_points_test=None,
        gno_out_neighbors_test=None,
        gno_out_query_points_test=None,
        gno_out_data_points_test=None,
        epoch: int = 0,
    ):
        """
        Log all neighbor search results for an epoch.

        Parameters
        ----------
        gno_in_neighbors_train : Dict
            Neighbors dict from gno_in during training
        gno_in_query_points_train : torch.Tensor
            Query points for gno_in during training
        gno_in_data_points_train : torch.Tensor
            Data points for gno_in during training
        gno_out_neighbors_train : Dict
            Neighbors dict from gno_out during training
        gno_out_query_points_train : torch.Tensor
            Query points for gno_out during training
        gno_out_data_points_train : torch.Tensor
            Data points for gno_out during training
        gno_in_neighbors_test : Dict, optional
            Neighbors dict from gno_in during testing
        gno_in_query_points_test : torch.Tensor, optional
            Query points for gno_in during testing
        gno_in_data_points_test : torch.Tensor, optional
            Data points for gno_in during testing
        gno_out_neighbors_test : Dict, optional
            Neighbors dict from gno_out during testing
        gno_out_query_points_test : torch.Tensor, optional
            Query points for gno_out during testing
        gno_out_data_points_test : torch.Tensor, optional
            Data points for gno_out during testing
        epoch : int, optional
            Epoch number, by default 0
        """
        # Log training
        if gno_in_neighbors_train is not None:
            self.log_gno_neighbors(
                gno_in_neighbors_train,
                gno_in_query_points_train,
                gno_in_data_points_train,
                epoch=epoch,
                stage="train",
                gno_type="gno_in",
            )

        if gno_out_neighbors_train is not None:
            self.log_gno_neighbors(
                gno_out_neighbors_train,
                gno_out_query_points_train,
                gno_out_data_points_train,
                epoch=epoch,
                stage="train",
                gno_type="gno_out",
            )

        # Log testing
        if gno_in_neighbors_test is not None:
            self.log_gno_neighbors(
                gno_in_neighbors_test,
                gno_in_query_points_test,
                gno_in_data_points_test,
                epoch=epoch,
                stage="test",
                gno_type="gno_in",
            )

        if gno_out_neighbors_test is not None:
            self.log_gno_neighbors(
                gno_out_neighbors_test,
                gno_out_query_points_test,
                gno_out_data_points_test,
                epoch=epoch,
                stage="test",
                gno_type="gno_out",
            )

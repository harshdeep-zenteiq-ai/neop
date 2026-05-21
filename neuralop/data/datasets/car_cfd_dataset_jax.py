"""JAX/Flax version of Car CFD Dataset"""

from typing import List, Union
from pathlib import Path
import jax.numpy as jnp
import sys

from .mesh_datamodule_jax import MeshDataModule
from .web_utils import download_from_zenodo_record
from neuralop.utils import get_project_root

class CarCFDDatasetjax(MeshDataModule):
    """Processed version of the Car-CFD dataset for JAX/Flax.

    CarCFDDataset is a processed version of the dataset introduced in
    [1]_, which encodes a triangular mesh over the surface of a 3D model car
    and provides the air pressure at each centroid and vertex of the mesh when
    the car is placed in a simulated wind tunnel with a recorded inlet velocity.
    In our case, inputs are a signed distance function evaluated over a regular
    3D grid of query points, as well as the inlet velocity. Outputs are pressure
    values at each centroid of the triangle mesh.

    Data is also stored on Zenodo: https://zenodo.org/records/13936501

    Parameters
    ----------
    root_dir : Union[str, Path]
        root directory at which data is stored.
    n_train : int, optional
        Number of training instances to load, by default 1
    n_test : int, optional
        Number of testing instances to load, by default 1
    query_res : List[int], optional
        Dimension-wise resolution of signed distance function
        (SDF) query cube, by default [32,32,32]
    download : bool, optional
        Whether to download data from Zenodo, by default True

    Attributes
    ----------
    train_data: dict
        dictionary of training examples
    test_data: dict
        dictionary of testing examples
    normalizers: dict
        normalizers for data

    References
    ----------
    .. [1] : Umetani, N. and Bickel, B. (2018). "Learning three-dimensional flow for interactive
        aerodynamic design". ACM Transactions on Graphics, 2018.
        https://dl.acm.org/doi/10.1145/3197517.3201325.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        n_train: int = 1,
        n_test: int = 1,
        query_res: List[int] = [32, 32, 32],
        download: bool = True,
    ):
        """Initialize the CarCFDDataset."""
        self.zenodo_record_id = "13936501"

        if isinstance(root_dir, str):
            root_dir = Path(root_dir).expanduser().resolve()
        print('root_dir', root_dir)

        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        if download:
            download_from_zenodo_record(record_id=self.zenodo_record_id, root=root_dir)

        # Initialize mesh datamodule
        super().__init__(
            root_dir=root_dir,
            item_dir_name="processed-car-pressure-data",
            n_train=n_train,
            n_test=n_test,
            query_res=query_res,
            attributes=["press"],
        )

        # process data list to remove specific vertices from pressure to match number of vertices
        for i, data in enumerate(self.train_data.data_list):
            press = data["press"]
            self.train_data.data_list[i]["press"] = jnp.concatenate(
                (press[:, 0:16], press[:, 112:]), axis=1
            )
        for i, data in enumerate(self.test_data.data_list):
            press = data["press"]
            self.test_data.data_list[i]["press"] = jnp.concatenate(
                (press[:, 0:16], press[:, 112:]), axis=1
            )


    # def train_loader(self, batch_size=1, shuffle=True):
    #     """Return training data as list of batches."""
    #     return self._create_loader(self.train_data, batch_size, shuffle)

    # def test_loader(self, batch_size=1, shuffle=False):
    #     """Return test data as list of batches."""
    #     return self._create_loader(self.test_data, batch_size, shuffle)

    # def _create_loader(self, data, batch_size=1, shuffle=False):
    #     """Create a simple data loader that yields batches."""
    #     n_samples = len(data)
    #     indices = list(range(n_samples))

    #     if shuffle:
    #         import random
    #         random.shuffle(indices)

    #     batches = []
    #     for i in range(0, n_samples, batch_size):
    #         batch_indices = indices[i:i + batch_size]
    #         batch = {
    #             key: jnp.stack([data[idx][key] for idx in batch_indices])
    #             for key in data[0].keys()
    #         }
    #         batches.append(batch)

    #     return batches


def load_mini_car():
    """Load the 3-example mini Car-CFD dataset we package along with our module.

    See `neuralop.data.datasets.CarCFDDataset` for more detailed references
    """
    import pickle
    return pickle.load(open(get_project_root() / "neuralop/data/datasets/data/mini_car.pt", 'rb'))

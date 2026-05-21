import warnings

warnings.filterwarnings("once", category=UserWarning)

import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Callable

from .base_model_jax import BaseModel
from ..layers.channel_mlp_jax import ChannelMLP
from ..layers.embeddings_jax import SinusoidalEmbedding
from ..layers.fno_block_jax import FNOBlocks
from ..layers.spectral_convolution_jax import SpectralConv
from ..layers.gno_block_jax import GNOBlock
from ..layers.gno_weighting_functions_jax import dispatch_weighting_fn


class GINO(BaseModel, name="gino"):
    """
    GINO: Geometry-informed Neural Operator - learns a mapping between
    functions presented over arbitrary coordinate meshes. The model carries
    global integration through spectral convolution layers in an intermediate
    latent space, as described in [1]_. Optionally enables a weighted output
    GNO for use in a Mollified Graph Neural Operator scheme, as introduced in [2]_.

    Parameters
    ----------
    in_channels : int
        Feature dimension of input points. Determined by the problem.
    out_channels : int
        Feature dimension of output points. Determined by the problem.
    fno_n_modes : tuple, optional
        Number of modes along each dimension to use in FNO. Default: (16, 16, 16)
    fno_hidden_channels : int, optional
        Hidden channels for use in FNO. Default: 64
    fno_n_layers : int, optional
        Number of layers in FNO. Default: 4
    latent_feature_channels : int, optional
        Number of channels in optional latent feature map. Default: None
    projection_channel_ratio : int, optional
        Ratio of pointwise projection channels to fno_hidden_channels. Default: 4
    gno_coord_dim : int, optional
        Geometric dimension of input/output queries. Default: 3
    in_gno_radius : float, optional
        Radius in input space for GNO neighbor search. Default: 0.033
    out_gno_radius : float, optional
        Radius in output space for GNO neighbor search. Default: 0.033

    References
    ----------
    .. [1] : Li, Z., Kovachki, N., Choy, C., Li, B., Kossaifi, J., Otta, S.,
        Nabian, M., Stadler, M., Hundt, C., Azizzadenesheli, K., Anandkumar, A. (2023)
        Geometry-Informed Neural Operator for Large-Scale 3D PDEs. NeurIPS 2023.
    .. [2] : Lin, R. et al. Placeholder reference for Mollified Graph Neural Operators.
    """

    in_channels: int
    out_channels: int
    latent_feature_channels: Optional[int] = None
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 3
    gno_coord_embed_dim: int = 16
    in_gno_radius: float = 0.033
    out_gno_radius: float = 0.033
    gno_radius: float = 0.033
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    gno_pos_embed_type: str = "nerf"
    fno_domain_padding: float = 0.125
    gno_weighting_function: Any = None
    gno_weight_function_scale: float = 1
    in_gno_pos_embed_type: str = "transformer"
    out_gno_pos_embed_type: str = "transformer"
    fno_in_channels: int = 3
    fno_n_modes: Any = (16, 16, 16)
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 2
    fno_n_layers: int = 4
    gno_embed_channels: int = 32
    gno_embed_max_positions: int = 10000
    in_gno_channel_mlp_hidden_layers: Any = (80, 80, 80)
    out_gno_channel_mlp_hidden_layers: Any = (512, 256)
    gno_channel_mlp_non_linearity: Callable = nn.gelu
    gno_use_open3d: bool = True
    gno_use_torch_scatter: bool = False
    out_gno_tanh: Any = None
    fno_resolution_scaling_factor: Any = None
    fno_block_precision: str = "full"
    fno_use_channel_mlp: bool = True
    fno_channel_mlp_dropout: float = 0
    fno_channel_mlp_expansion: float = 0.5
    fno_non_linearity: Callable = nn.gelu
    fno_stabilizer: Any = None
    fno_norm: Any = None
    fno_ada_in_features: int = 4
    fno_ada_in_dim: int = 1
    fno_preactivation: bool = False
    fno_skip: str = "linear"
    fno_channel_mlp_skip: str = "soft-gating"
    fno_separable: bool = False
    fno_factorization: Any = None
    fno_rank: float = 1.0
    fno_fixed_rank_modes: bool = False
    fno_implementation: str = "factorized"
    fno_decomposition_kwargs: Any = None
    fno_conv_module: Any = SpectralConv
    fno_enforce_hermitian_symmetry: bool = True

    def setup(self):
        
        # Initialize call counter for debugging
        # self._latent_embedding_call_count = 0        
        # print('[GINO.setup] Initializing GINO model')

        fno_hidden_channels = self.fno_hidden_channels
        fno_n_modes = self.fno_n_modes
        decomposition_kwargs = self.fno_decomposition_kwargs or {}

        lifting_channels = self.fno_lifting_channel_ratio * fno_hidden_channels

        # If the input GNO performs a nonlinear kernel, the GNO's output
        # features must be the same dimension as its input.
        if self.in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            in_gno_out_channels = self.in_channels
        else:
            in_gno_out_channels = self.fno_in_channels

        # The actual input channels to the FNO are computed here.
        fno_in_channels = in_gno_out_channels
        if self.latent_feature_channels is not None:
            fno_in_channels += self.latent_feature_channels
        self._fno_in_channels = fno_in_channels

        gno_use_open3d = self.gno_use_open3d
        if self.gno_coord_dim != 3 and gno_use_open3d:
            warnings.warn(
                f'GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data',
                UserWarning,
                stacklevel=2,
            )
            gno_use_open3d = False

        in_coord_dim = len(fno_n_modes)
        if in_coord_dim != self.gno_coord_dim:
            warnings.warn(
                f'FNO expects {in_coord_dim}-d data while input GNO expects {self.gno_coord_dim}-d data',
                UserWarning,
                stacklevel=2,
            )
        self._in_coord_dim = in_coord_dim
        self._in_coord_dim_forward_order = list(range(in_coord_dim))
        # tensor indices starting at 2 to permute everything after channel and batch dims
        self._in_coord_dim_reverse_order = [j + 2 for j in self._in_coord_dim_forward_order]

        if self.fno_norm == "ada_in":
            if self.fno_ada_in_features is not None and self.out_gno_pos_embed_type is not None:
                self.adain_pos_embed = SinusoidalEmbedding(
                    in_channels=self.fno_ada_in_dim,
                    num_frequencies=self.fno_ada_in_features,
                    max_positions=10000,
                    embedding_type=self.out_gno_pos_embed_type,
                )
                ada_in_dim = self.adain_pos_embed.out_channels
            else:
                ada_in_dim = self.fno_ada_in_dim
                self.adain_pos_embed = None
        else:
            self.adain_pos_embed = None
            ada_in_dim = None
        self._ada_in_dim = ada_in_dim

        ### input GNO
        self.gno_in = GNOBlock(
            in_channels=self.in_channels,
            out_channels=in_gno_out_channels,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=self.in_gno_pos_embed_type,
            pos_embedding_channels=self.gno_embed_channels,
            pos_embedding_max_positions=self.gno_embed_max_positions,
            radius=self.in_gno_radius,
            reduction="mean",
            weighting_fn=None,
            channel_mlp_layers=list(self.in_gno_channel_mlp_hidden_layers),
            channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
            transform_type=self.in_gno_transform_type,
            use_torch_scatter_reduce=self.gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        ### Lifting layer before FNOBlocks
        self.lifting = ChannelMLP(
            in_channels=fno_in_channels,
            hidden_channels=lifting_channels,
            out_channels=fno_hidden_channels,
            n_layers=2,
        )

        ### FNOBlocks in latent space
        self.fno_blocks = FNOBlocks(
            n_modes=fno_n_modes,
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            n_layers=self.fno_n_layers,
            resolution_scaling_factor=self.fno_resolution_scaling_factor,
            fno_block_precision=self.fno_block_precision,
            use_channel_mlp=self.fno_use_channel_mlp,
            channel_mlp_expansion=self.fno_channel_mlp_expansion,
            channel_mlp_dropout=self.fno_channel_mlp_dropout,
            non_linearity=self.fno_non_linearity,
            stabilizer=self.fno_stabilizer,
            norm=self.fno_norm,
            ada_in_features=ada_in_dim,
            preactivation=self.fno_preactivation,
            fno_skip=self.fno_skip,
            channel_mlp_skip=self.fno_channel_mlp_skip,
            separable=self.fno_separable,
            factorization=self.fno_factorization,
            rank=self.fno_rank,
            fixed_rank_modes=self.fno_fixed_rank_modes,
            implementation=self.fno_implementation,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=self.fno_conv_module,
            enforce_hermitian_symmetry=self.fno_enforce_hermitian_symmetry,
        )

        ### output GNO
        if self.gno_weighting_function is not None:
            weight_fn = dispatch_weighting_fn(
                self.gno_weighting_function,
                sq_radius=self.out_gno_radius ** 2,
                scale=self.gno_weight_function_scale,
            )
        else:
            weight_fn = None

        self.gno_out = GNOBlock(
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.out_gno_radius,
            reduction="sum",
            weighting_fn=weight_fn,
            pos_embedding_type=self.out_gno_pos_embed_type,
            pos_embedding_channels=self.gno_embed_channels,
            pos_embedding_max_positions=self.gno_embed_max_positions,
            channel_mlp_layers=list(self.out_gno_channel_mlp_hidden_layers),
            channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
            transform_type=self.out_gno_transform_type,
            use_torch_scatter_reduce=self.gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        projection_channels = self.projection_channel_ratio * fno_hidden_channels
        self.projection = ChannelMLP(
            in_channels=fno_hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=self.fno_non_linearity,
        )

    # returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, ada_in=None):
        # in_p : (batch, n_1 , ... , n_k, in_channels + k)
        # ada_in : (fno_ada_in_dim, )
                
        # permute (b, n_1, ..., n_k, c) -> (b, c, n_1, ..., n_k)
        perm = (0, len(in_p.shape) - 1, *list(range(1, len(in_p.shape) - 1)))
        in_p = jnp.transpose(in_p, perm)

        # Compute Ada IN embedding
        ada_in_embeddings = None
        if ada_in is not None:
            if ada_in.ndim == 2:
                ada_in = jnp.squeeze(ada_in, axis=0)
            if self.adain_pos_embed is not None:
                ada_in_embed = jnp.squeeze(
                    self.adain_pos_embed(jnp.expand_dims(ada_in, axis=0)), axis=0
                )
            else:
                ada_in_embed = ada_in
            if self.fno_norm == "ada_in":
                ada_in_embeddings = [ada_in_embed]

        # Apply FNO blocks
        in_p = self.lifting(in_p)

        for idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(in_p, idx, ada_in_embeddings=ada_in_embeddings)

        return in_p

    def __call__(self, input_geom, latent_queries, output_queries,
                 x=None, latent_features=None, ada_in=None,
                 neighbors_in=None, neighbors_out=None, **kwargs):
        if kwargs:
            warnings.warn(
                f"GINO.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        """The GINO's forward call:
        Input GNO --> FNOBlocks --> output GNO + projection Refactor training script for JAX compatibilityto output queries.

        Parameters
        ----------
        input_geom : jnp.ndarray
            input domain coordinate mesh, shape (1, n_in, gno_coord_dim)
        latent_queries : jnp.ndarray
            latent geometry on which to compute FNO latent embeddings,
            shape (1, n_gridpts_1, ..., n_gridpts_n, gno_coord_dim)
        output_queries : jnp.ndarray | dict[jnp.ndarray]
            points at which to query the final GNO layer to get output,
            shape (1, n_out, gno_coord_dim) per tensor.
        x : jnp.ndarray, optional
            input function defined on the input domain, shape (batch, n_in, in_channels)
        latent_features : jnp.ndarray, optional
            optional feature map to concatenate onto latent embedding
        ada_in : jnp.ndarray, optional
            adaptive scalar instance parameter
        neighbors_in : dict, optional
            precomputed neighbor search result for the input GNO
        neighbors_out : dict, optional
            precomputed neighbor search result for the output GNO

        Returns
        -------
        out : jnp.ndarray | dict[jnp.ndarray]
            Function over the output query coordinates
        """

        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        if latent_features is not None:
            assert (
                self.latent_feature_channels is not None
            ), "if passing latent features, latent_feature_channels must be set."
            assert latent_features.shape[-1] == self.latent_feature_channels
            assert (
                latent_features.ndim == self.gno_coord_dim + 2
            ), f"Latent features must be of shape (batch, n_gridpts_1, ...n_gridpts_n, gno_coord_dim), got {latent_features.shape}"
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = jnp.tile(
                        latent_features,
                        (batch_size, *[1] * (latent_features.ndim - 1)),
                    )

        input_geom = jnp.squeeze(input_geom, axis=0)
        latent_queries = jnp.squeeze(latent_queries, axis=0)

        # Pass through input GNOBlock
        in_p = self.gno_in(
            y=input_geom,
            x=latent_queries.reshape((-1, latent_queries.shape[-1])),
            f_y=x,
            neighbors=neighbors_in,
        )

        grid_shape = latent_queries.shape[:-1]  # disregard positional encoding dim

        # shape (batch_size, grid1, ...gridn, -1)
        in_p = in_p.reshape((batch_size, *grid_shape, -1))

        if latent_features is not None:
            in_p = jnp.concatenate((in_p, latent_features), axis=-1)

        # apply fno in latent space
        latent_embed = self.latent_embedding(in_p=in_p, ada_in=ada_in)

        # Integrate latent space to output queries
        # latent_embed shape (b, c, n_1, n_2, ..., n_k)
        batch_size = latent_embed.shape[0]
        # permute to (b, n_1, n_2, ...n_k, c) then reshape to (b, n_1 * n_2 * ...n_k, out_channels)
        perm = (0, *self._in_coord_dim_reverse_order, 1)
        latent_embed = jnp.transpose(latent_embed, perm).reshape(
            batch_size, -1, self.fno_hidden_channels
        )

        if self.out_gno_tanh in ["latent_embed", "both"]:
            latent_embed = jnp.tanh(latent_embed)

        # integrate over the latent space
        if isinstance(output_queries, dict):
            out = {}
            for key, out_p in output_queries.items():
                out_p = jnp.squeeze(out_p, axis=0)

                sub_output = self.gno_out(
                    y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                    x=out_p,
                    f_y=latent_embed,
                    neighbors=neighbors_out,
                )
                sub_output = jnp.transpose(sub_output, (0, 2, 1))

                # Project pointwise to out channels
                sub_output = jnp.transpose(self.projection(sub_output), (0, 2, 1))

                out[key] = sub_output
        else:
            output_queries = jnp.squeeze(output_queries, axis=0)

            out = self.gno_out(
                y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                x=output_queries,
                f_y=latent_embed,
                neighbors=neighbors_out,
            )
            out = jnp.transpose(out, (0, 2, 1))

            # Project pointwise to out channels
            out = jnp.transpose(self.projection(out), (0, 2, 1))

        return out

    def precompute_neighbors(self, dataset, output_n_points=None, log_dir=None, split="train"):
        """Precompute and cache neighbor search results for all samples.

        Parameters
        ----------
        dataset : DictDataset
            The train or test dataset whose data_list will be updated in-place.
        output_n_points : int, optional
            Number of output query points. If None, all vertices are used.
        log_dir : str or Path, optional
            Directory to write per-sample neighbor count log files.
        split : str, optional
            Label for the dataset split. Default: "train".
        """
        from pathlib import Path

        latent_queries = dataset.constant["query_points"]   # shape [d1, d2, d3, 3]
        if hasattr(latent_queries, "numpy"):
            latent_queries = jnp.array(latent_queries.detach().cpu().numpy())
        else:
            latent_queries = jnp.array(latent_queries)
        latent_queries_flat = latent_queries.reshape(-1, latent_queries.shape[-1])  # [m, 3]

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        log_file = None
        if log_dir is not None:
            log_path = log_dir / f"{split}.txt"
            log_file = open(log_path, "w")
            log_file.write(f"Neighbor search summary — {split}\n")
            log_file.write("=" * 60 + "\n\n")

        # Directly call the standalone neighbor search function —
        # we cannot instantiate nn.Module subclasses (like NeighborSearch) outside
        # of Flax's init/apply lifecycle, and Flax wraps every method on this class.
        from ..layers.neighbor_search_jax import kdtree_neighbor_search as _ns_fn
        return_norm = self.gno_weighting_function is not None

        def _to_jnp(t):
            """Convert torch tensor or numpy array to jnp array."""
            if hasattr(t, "numpy"):      # torch.Tensor
                return jnp.array(t.detach().cpu().numpy())
            return jnp.array(t)

        # print(f"\n[PRECOMPUTE NEIGHBORS] Starting precomputation for {split} split ({len(dataset.data_list)} samples)...", flush=True)

        try:
            for i, sample in enumerate(dataset.data_list):
                # sample_start_time = time.time()
                vertices = _to_jnp(sample["vertices"])   # [n, 3]

                sample["neighbors_in"] = _ns_fn(
                    data=vertices,
                    queries=latent_queries_flat,
                    radius=self.in_gno_radius,
                    return_norm=return_norm,
                )

                out_p = vertices if output_n_points is None else vertices[:output_n_points]
                sample["neighbors_out"] = _ns_fn(
                    data=latent_queries_flat,
                    queries=out_p,
                    radius=self.out_gno_radius,
                    return_norm=return_norm,
                )

                # sample_elapsed_time = time.time() - sample_start_time
                # print(f"[PRECOMPUTE NEIGHBORS] Sample {i+1:3d}/{len(dataset.data_list)} done in {sample_elapsed_time:.3f}s", flush=True)

                if log_file is not None:
                    splits_in  = sample["neighbors_in"]["neighbors_row_splits"]
                    counts_in  = (splits_in[1:] - splits_in[:-1]).tolist()

                    splits_out = sample["neighbors_out"]["neighbors_row_splits"]
                    counts_out = (splits_out[1:] - splits_out[:-1]).tolist()

                    log_file.write(f"Sample {i+1:03d}\n")
                    log_file.write(f"  gno_in  — total: {sum(counts_in)}  "
                                   f"min: {min(counts_in)}  max: {max(counts_in)}  "
                                   f"mean: {sum(counts_in)/len(counts_in):.2f}\n")
                    log_file.write(f"  gno_out — total: {sum(counts_out)}  "
                                   f"min: {min(counts_out)}  max: {max(counts_out)}  "
                                   f"mean: {sum(counts_out)/len(counts_out):.2f}\n\n")
        finally:
            if log_file is not None:
                log_file.close()

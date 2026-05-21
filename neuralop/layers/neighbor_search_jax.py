import jax 
import jax.numpy as jnp
import flax.linen as nn

# only import open3d if built
open3d_built = False
try:
    from open3d.ml.torch.layers import FixedRadiusSearch

    open3d_built = True
except:
    pass


# Uses open3d by default which, as of October 2024, requires torch 2.0 and cuda11.*
class NeighborSearch(nn.Module):
    """Neighborhood search between two arbitrary coordinate meshes.

    For each point `x` in `queries`, returns a set of the indices of all points `y` in `data`
    within the ball of radius r `B_r(x)`

    Parameters
    ----------
    use_open3d : bool, optional
        Whether to use open3d or native JAX implementation, by default True
        NOTE: open3d implementation requires 3d data
    return_norm : bool, optional
        Whether to return normalized distances, by default False
    """

    use_open3d: bool = True
    return_norm: bool = False

    def setup(self):
        if self.use_open3d and open3d_built:  # slightly faster, works on GPU in 3d only
            self.search_fn = FixedRadiusSearch()
            self._use_open3d = True
        else:  # slower fallback, works on GPU and CPU
            self.search_fn = None
            self._use_open3d = False

    def __call__(self, data, queries, radius):
        """
        Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : jnp.ndarray of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : jnp.ndarray of shape [m, d]
            Points for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)

        Returns
        -------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: jnp.ndarray with dtype=int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the
                    neighbors for every point.
                neighbors_row_splits: jnp.ndarray of shape [m+1] with dtype=int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}

        if self._use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict["neighbors_index"] = jnp.array(search_return.neighbors_index, dtype=jnp.int64)
            return_dict["neighbors_row_splits"] = jnp.array(search_return.neighbors_row_splits, dtype=jnp.int64)
        else:
            return_dict = native_neighbor_search_chunked(data, queries, radius, self.return_norm)

        return return_dict


def native_neighbor_search(
    data: jnp.ndarray, queries: jnp.ndarray, radius: float, return_norm: bool = False
):
    """
    Native JAX implementation of a neighborhood search
    between two arbitrary coordinate meshes.

    Parameters
    ----------
    data : jnp.ndarray
        vector of data points from which to find neighbors
    queries : jnp.ndarray
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """
    import sys

    print(f"\n[native_neighbor_search] Starting...", file=sys.stderr)
    print(f"[native_neighbor_search] queries shape: {queries.shape}", file=sys.stderr)
    print(f"[native_neighbor_search] data shape: {data.shape}", file=sys.stderr)

    nbr_dict = {}
    
    dists = jnp.linalg.norm(queries[:, None, :] - data[None, :, :], axis=-1)
    
    print('After dists print')

    # compute pairwise distances
    print(f"[native_neighbor_search] Computing: queries[:, None, :] shape will be {queries.shape[0], 1, queries.shape[1]}", file=sys.stderr)
    print(f"[native_neighbor_search] Computing: data[None, :, :] shape will be {1, data.shape[0], data.shape[1]}", file=sys.stderr)
    print(f"[native_neighbor_search] Intermediate subtraction will allocate [{queries.shape[0]}, {data.shape[0]}, {queries.shape[1]}] = {queries.shape[0] * data.shape[0] * queries.shape[1] * 4 / 1e9:.2f} GB", file=sys.stderr)

    print(f"[native_neighbor_search] Starting subtraction...", file=sys.stderr)
    subtraction = queries[:, None, :] - data[None, :, :]
    print(f"[native_neighbor_search] Subtraction done, shape: {subtraction.shape}", file=sys.stderr)

    print(f"[native_neighbor_search] Computing norm...", file=sys.stderr)
    all_dists = jnp.linalg.norm(subtraction, axis=-1)
    print(f"[native_neighbor_search] Norm done, shape: {all_dists.shape}", file=sys.stderr)

    # keep zero-distance points
    eps = 1e-7
    print(f"[native_neighbor_search] Replacing zero distances...", file=sys.stderr)
    all_dists = jnp.where(all_dists == 0.0, eps, all_dists)
    print(f"[native_neighbor_search] Zero replacement done", file=sys.stderr)

    print(f"[native_neighbor_search] Masking by radius...", file=sys.stderr)
    dists = jnp.where(all_dists <= radius, all_dists, 0.0)  # i,j is nonzero if j is i's neighbor
    print(f"[native_neighbor_search] Radius masking done, dists shape: {dists.shape}", file=sys.stderr)

    print(f"[native_neighbor_search] Finding nonzero indices...", file=sys.stderr)
    nbr_indices = jnp.nonzero(dists, size=dists.size)[1]  # only keep the column indices
    print(f"[native_neighbor_search] Nonzero indices found", file=sys.stderr)

    # filter to only actual nonzero entries
    print(f"[native_neighbor_search] Filtering nonzero entries...", file=sys.stderr)
    mask = dists.reshape(-1) > 0
    nbr_indices = nbr_indices[mask]
    print(f"[native_neighbor_search] Filtering done, nbr_indices shape: {nbr_indices.shape}", file=sys.stderr)

    if return_norm:
        weights = dists[dists > 0]
        nbr_dict["weights"] = weights ** 2  # weighting function computed on squared norms

    print(f"[native_neighbor_search] Computing neighborhood sizes...", file=sys.stderr)
    in_nbr = jnp.where(dists > 0, 1.0, 0.0)
    nbrhd_sizes = jnp.cumsum(jnp.sum(in_nbr, axis=1), axis=0)  # cumulative neighborhood sizes
    splits = jnp.concatenate((jnp.array([0.0]), nbrhd_sizes))
    print(f"[native_neighbor_search] Done! splits shape: {splits.shape}", file=sys.stderr)

    nbr_dict["neighbors_index"] = nbr_indices.astype(jnp.int64)
    nbr_dict["neighbors_row_splits"] = splits.astype(jnp.int64)

    return nbr_dict


# class NeighborSearchJAX:
#     """
#     JIT-safe fixed-size CSR neighbor search (no boolean indexing).
#     """

#     def __init__(self, max_neighbors: int, return_norm: bool = False):
#         self.max_neighbors = max_neighbors
#         self.return_norm = return_norm

#     def __call__(self, data: jnp.ndarray, queries: jnp.ndarray, radius: float):
#         # 1. Compute all pairwise distances: (m, n)
#         dists = jnp.linalg.norm(queries[:, None, :] - data[None, :, :], axis=-1)
        
#         # 2. Find the indices of the K-nearest points
#         # Use jax.lax.top_k to get the smallest distances (negate them)
#         val, neighbors_index = jax.lax.top_k(-dists, self.max_neighbors)
#         actual_dists = -val # (m, max_neighbors)
        
#         # 3. Create a validity mask: True if distance <= radius
#         mask = actual_dists <= radius
        
#         # 4. Crucial: We must pass this mask to the IntegralTransform!
#         out = {
#             "neighbors_index": neighbors_index.reshape(-1).astype(jnp.int32),
#             "neighbors_mask": mask.reshape(-1), # ADD THIS
#             "neighbors_row_splits": jnp.arange(0, (queries.shape[0] + 1) * self.max_neighbors, self.max_neighbors).astype(jnp.int32)
#         }
#         return out

# def native_neighbor_search_gpu(data: jnp.ndarray, queries: jnp.ndarray, radius: float, return_norm: bool = False):
#     """JAX GPU-friendly version of native_neighbor_search.

#     Uses memory-efficient distance computation to avoid OOM on GPU
#     during large-scale precomputation (500+ cases).

#     Parameters
#     ----------
#     data : jnp.ndarray of shape [n, d]
#     queries : jnp.ndarray of shape [m, d]
#     radius : float
#     return_norm : bool
#     """
#     import numpy as np

#     nbr_dict = {}

#     # Precompute squared norms of data points for efficient distance calculation
#     data_sq_norms = jnp.sum(data ** 2, axis=1)  # [n]

#     nbr_indices_list = []
#     counts = []
#     weights_list = []

#     # Process queries one at a time to minimize memory usage
#     # Formula: ||q - d||^2 = ||q||^2 - 2(q·d) + ||d||^2
#     for i in range(len(queries)):
#         query = queries[i]  # [d]
#         query_sq_norm = jnp.sum(query ** 2)

#         # Compute squared distances: [n]
#         # Using: ||q - d||^2 = ||q||^2 - 2(q·d) + ||d||^2
#         dot_product = jnp.dot(query, data.T)  # [n]
#         sq_dists = query_sq_norm - 2 * dot_product + data_sq_norms

#         # Handle zero-distance points
#         eps = 1e-7
#         dists = jnp.sqrt(jnp.maximum(sq_dists, eps**2))

#         # Find neighbors within radius
#         mask = dists <= radius

#         # Convert to numpy for eager evaluation (we're in a loop anyway)
#         # This avoids JAX's dynamic shape inference that causes OOM
#         mask_np = np.asarray(mask)
#         idxs = np.where(mask_np)[0].astype(np.int64)

#         nbr_indices_list.append(jnp.array(idxs, dtype=jnp.int64))
#         counts.append(len(idxs))

#         if return_norm:
#             weights = sq_dists[mask]  # Use squared distances as weights
#             weights_list.append(jnp.asarray(weights))

#     # Concatenate all results
#     if nbr_indices_list:
#         nbr_indices = jnp.concatenate(nbr_indices_list)
#     else:
#         nbr_indices = jnp.array([], dtype=jnp.int64)

#     splits = jnp.concatenate([jnp.array([0]), jnp.array(counts, dtype=jnp.int64)])
#     splits = jnp.cumsum(splits).astype(jnp.int64)

#     nbr_dict["neighbors_index"] = nbr_indices.astype(jnp.int64)
#     nbr_dict["neighbors_row_splits"] = splits

#     if return_norm:
#         if weights_list:
#             weights = jnp.concatenate(weights_list)
#         else:
#             weights = jnp.array([], dtype=jnp.float32)
#         nbr_dict["weights"] = weights

#     return nbr_dict


def native_neighbor_search_chunked(data: jnp.ndarray, queries: jnp.ndarray, radius: float, return_norm: bool = False, batch_size: int = 1024):
    """
    Memory-efficient chunked neighbor search.

    Processes queries in small batches to avoid allocating the full [m, n, d]
    intermediate tensor. Uses vectorized JAX operations on each batch.

    Parameters
    ----------
    data : jnp.ndarray of shape [n, d]
        Search space of possible neighbors
    queries : jnp.ndarray of shape [m, d]
        Points for which to find neighbors
    radius : float
        Radius of each ball: B(queries[j], radius)
    return_norm : bool, optional
        Whether to return normalized distances, by default False
    batch_size : int, optional
        Number of queries to process at once, by default 1024

    Returns
    -------
    dict
        Dictionary with keys: neighbors_index, neighbors_row_splits, and optional weights
    """
    import sys

    print(f"\n[native_neighbor_search_chunked] Starting with batch_size={batch_size}...", file=sys.stderr)
    print(f"[native_neighbor_search_chunked] queries shape: {queries.shape}", file=sys.stderr)
    print(f"[native_neighbor_search_chunked] data shape: {data.shape}", file=sys.stderr)

    nbr_dict = {}
    nbr_indices_list = []
    counts = []
    weights_list = []

    # Process queries in chunks
    n_chunks = (len(queries) + batch_size - 1) // batch_size
    print(f"[native_neighbor_search_chunked] Processing {len(queries)} queries in {n_chunks} chunks", file=sys.stderr)

    for chunk_idx in range(0, len(queries), batch_size):
        end_idx = min(chunk_idx + batch_size, len(queries))
        q_batch = queries[chunk_idx:end_idx]  # [batch_size, d]

        # print(f"[native_neighbor_search_chunked] Processing chunk {chunk_idx//batch_size + 1}/{n_chunks}, queries [{chunk_idx}:{end_idx}]", file=sys.stderr)

        # Compute distances for this batch: [batch_size, n]
        # Intermediate: [batch_size, n, d] which is much smaller than [m, n, d]
        batch_dists = jnp.linalg.norm(
            q_batch[:, None, :] - data[None, :, :], axis=-1
        )

        # print(f"[native_neighbor_search_chunked]   Computed distances shape: {batch_dists.shape}", file=sys.stderr)

        # keep zero-distance points
        eps = 1e-7
        batch_dists = jnp.where(batch_dists == 0.0, eps, batch_dists)

        # Find neighbors within radius
        mask = batch_dists <= radius  # [batch_size, n]

        # Process each query in the batch
        for i in range(len(q_batch)):
            query_mask = mask[i]  # [n]
            query_dists = batch_dists[i]  # [n]

            # Find neighbor indices
            nbr_idx = jnp.nonzero(query_mask, size=len(data))[0]

            # Count actual neighbors
            actual_count = jnp.sum(query_mask)
            nbr_idx = nbr_idx[:actual_count]  # Trim to actual count

            nbr_indices_list.append(nbr_idx.astype(jnp.int64))
            counts.append(int(actual_count))

            if return_norm:
                weights = query_dists[query_mask]
                weights_list.append(weights ** 2)

    # print(f"[native_neighbor_search_chunked] Concatenating results...", file=sys.stderr)

    # Concatenate all results
    if nbr_indices_list:
        nbr_indices = jnp.concatenate(nbr_indices_list)
    else:
        nbr_indices = jnp.array([], dtype=jnp.int64)

    splits = jnp.concatenate([jnp.array([0]), jnp.array(counts, dtype=jnp.int64)])
    splits = jnp.cumsum(splits).astype(jnp.int64)

    nbr_dict["neighbors_index"] = nbr_indices.astype(jnp.int64)
    nbr_dict["neighbors_row_splits"] = splits

    if return_norm:
        if weights_list:
            weights = jnp.concatenate(weights_list)
        else:
            weights = jnp.array([], dtype=jnp.float32)
        nbr_dict["weights"] = weights

    print(f"[native_neighbor_search_chunked] Done! Total neighbors: {len(nbr_indices)}", file=sys.stderr)
    return nbr_dict

def native_neighbor_search_chunked_optimized(data: jnp.ndarray, queries: jnp.ndarray, radius: float, return_norm: bool = False, batch_size: int = 2048):
    
    nbr_indices_list = []
    weights_list = []
    
    # We will build the row splits dynamically 
    row_splits = [0]
    current_split = 0
    eps = 1e-7

    for chunk_idx in range(0, len(queries), batch_size):
        end_idx = min(chunk_idx + batch_size, len(queries))
        q_batch = queries[chunk_idx:end_idx]

        # 1. Vectorized distance calculation
        batch_dists = jnp.linalg.norm(q_batch[:, None, :] - data[None, :, :], axis=-1)
        batch_dists = jnp.where(batch_dists == 0.0, eps, batch_dists)

        # 2. Vectorized mask
        mask = batch_dists <= radius

        # --- NO INNER LOOP BEYOND THIS POINT ---

        # 3. Get all valid indices for the ENTIRE batch in one GPU sweep
        # row_idx is the query index within the batch, data_idx is the neighbor index
        row_idx, data_idx = jnp.nonzero(mask)
        
        nbr_indices_list.append(data_idx.astype(jnp.int64))

        if return_norm:
            # Extract weights for the whole batch in one sweep
            weights = batch_dists[row_idx, data_idx]
            weights_list.append(weights ** 2)

        # 4. Get counts for the whole batch in one sweep
        batch_counts = jnp.sum(mask, axis=1)
        
        # We only sync `batch_counts` to CPU here, which is 1024x less syncing than your loop
        import numpy as np
        for count in np.array(batch_counts):
            current_split += int(count)
            row_splits.append(current_split)

    # Concatenate chunked results
    nbr_dict = {
        "neighbors_index": jnp.concatenate(nbr_indices_list) if nbr_indices_list else jnp.array([], dtype=jnp.int64),
        "neighbors_row_splits": jnp.array(row_splits, dtype=jnp.int64)
    }

    if return_norm:
        nbr_dict["weights"] = jnp.concatenate(weights_list) if weights_list else jnp.array([], dtype=jnp.float32)

    return nbr_dict 

def native_neighbor_search_numpy(data, queries, radius: float, return_norm: bool = False):
    """Delegates to native_neighbor_search_chunked for memory efficiency.

    This function is kept for backwards compatibility with the original API.
    """
    # Convert inputs to JAX arrays if needed
    if not isinstance(data, jnp.ndarray):
        data = jnp.array(data)
    if not isinstance(queries, jnp.ndarray):
        queries = jnp.array(queries)

    return native_neighbor_search_chunked_optimized(data, queries, radius, return_norm, batch_size=2048)

import numpy as np
from scipy.spatial import cKDTree

def kdtree_neighbor_search(data: np.ndarray, queries: np.ndarray, radius: float, return_norm: bool = False):
    """
    Lightning-fast O(N log M) neighbor search using Scipy's C++ KD-Tree.
    Run this eagerly on the CPU inside your data loader.
    """
    # 1. Build the KD-Tree on the search space (data)
    tree = cKDTree(data)
    
    # 2. Query the tree for points within the radius
    # Returns a list of lists containing neighbor indices for each query
    # neighbor_lists = tree.query_ball_point(queries, r=radius)
    safe_radius = radius + 1e-6 
    neighbor_lists = tree.query_ball_point(queries, r=safe_radius)
    
    # 3. Format the outputs to match your exact dictionary structure
    all_indices = []
    row_splits = [0]
    current_split = 0
    all_weights = []
    
    for i, nbrs in enumerate(neighbor_lists):
        n_count = len(nbrs)
        current_split += n_count
        row_splits.append(current_split)
        
        if n_count > 0:
            all_indices.extend(nbrs)
            
            if return_norm:
                # Calculate distances only for actual neighbors, saving massive computation
                q_point = queries[i]
                d_points = data[nbrs]
                sq_dists = np.sum((d_points - q_point)**2, axis=-1)
                all_weights.extend(sq_dists)

    nbr_dict = {
        "neighbors_index": np.array(all_indices, dtype=np.int32),
        "neighbors_row_splits": np.array(row_splits, dtype=np.int32)
    }
    
    if return_norm:
        nbr_dict["weights"] = np.array(all_weights, dtype=np.float32)
        
    return nbr_dict
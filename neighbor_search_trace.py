"""
Detailed trace of native_neighbor_search to understand each step
"""
import torch

# ===== EXAMPLE SETUP =====
# 3 query points, 5 data points, 3D coordinates, radius=0.5
queries = torch.tensor([
    [0.0, 0.0, 0.0],  # query point 0
    [1.0, 0.0, 0.0],  # query point 1
    [2.0, 0.0, 0.0],  # query point 2
], dtype=torch.float32)

data = torch.tensor([
    [0.1, 0.0, 0.0],  # data point 0 - close to query 0
    [0.0, 0.15, 0.0], # data point 1 - close to query 0
    [1.0, 0.2, 0.0],  # data point 2 - close to query 1
    [1.05, 0.0, 0.0], # data point 3 - close to query 1
    [5.0, 5.0, 5.0],  # data point 4 - far from all
], dtype=torch.float32)

radius = 0.5
return_norm = False 

print("="*80)
print("INPUT SHAPES AND VALUES")
print("="*80)
print(f"queries.shape: {queries.shape}")
print(f"queries:\n{queries}\n")
print(f"data.shape: {data.shape}")
print(f"data:\n{data}\n")
print(f"radius: {radius}")
print(f"return_norm: {return_norm}\n")

# ===== LINE 1: nbr_dict = {} =====
nbr_dict = {}
print("="*80)
print("LINE 1: nbr_dict = {}")
print("="*80)
print(f"OUTPUT: nbr_dict = {nbr_dict}")
print(f"DESCRIPTION: Initialize empty dictionary to store results\n")

# ===== LINE 2: all_dists = torch.cdist(queries, data).to(queries.device) =====
all_dists = torch.cdist(queries, data).to(queries.device)
print("="*80)
print("LINE 2: all_dists = torch.cdist(queries, data).to(queries.device)")
print("="*80)
print(f"OUTPUT shape: {all_dists.shape}  # [num_queries, num_data]")
print(f"OUTPUT values:\n{all_dists}")
print(f"DESCRIPTION: Compute pairwise L2 distances between all query and data points")
print(f"  - Row i, Col j = distance from query_i to data_j")
print(f"  - query 0 to data 0: {all_dists[0,0]:.4f}")
print(f"  - query 1 to data 2: {all_dists[1,2]:.4f}")
print(f"  - query 2 to data 4: {all_dists[2,4]:.4f}\n")

# ===== LINE 3: eps = 1e-7 =====
eps = 1e-7
print("="*80)
print("LINE 3: eps = 1e-7")
print("="*80)
print(f"OUTPUT: eps = {eps}")
print(f"DESCRIPTION: Define small epsilon to handle zero-distance points (self-proximity)\n")

# ===== LINE 4: all_dists = torch.where(all_dists == 0., eps, all_dists) =====
all_dists = torch.where(all_dists == 0., eps, all_dists)
print("="*80)
print("LINE 4: all_dists = torch.where(all_dists == 0., eps, all_dists)")
print("="*80)
print(f"OUTPUT shape: {all_dists.shape}")
print(f"OUTPUT values:\n{all_dists}")
print(f"DESCRIPTION: Replace any exactly-zero distances with eps")
print(f"  - Prevents exact zero distances (only matters if queries contain exact data points)\n")

# ===== LINE 5: dists = torch.where(all_dists <= radius, all_dists, 0.) =====
dists = torch.where(all_dists <= radius, all_dists, 0.)
print("="*80)
print("LINE 5: dists = torch.where(all_dists <= radius, all_dists, 0.)")
print("="*80)
print(f"OUTPUT shape: {dists.shape}  # [num_queries, num_data]")
print(f"OUTPUT values:\n{dists}")
print(f"DESCRIPTION: Mask distances - keep only neighbors within radius, zero out others")
print(f"  - 0.0 means 'NOT a neighbor' (distance > radius)")
print(f"  - Non-zero means 'IS a neighbor' (distance <= radius)")
print(f"  - For query 0: neighbors are data points {(dists[0] > 0).nonzero(as_tuple=True)[0].tolist()}")
print(f"  - For query 1: neighbors are data points {(dists[1] > 0).nonzero(as_tuple=True)[0].tolist()}")
print(f"  - For query 2: neighbors are data points {(dists[2] > 0).nonzero(as_tuple=True)[0].tolist()}\n")

# ===== LINE 6: nbr_indices = dists.nonzero()[:,1:].reshape(-1,) =====
nonzero_result = dists.nonzero()
print("="*80)
print("LINE 6a: INTERMEDIATE - dists.nonzero()")
print("="*80)
print(f"OUTPUT shape: {nonzero_result.shape}  # [num_neighbors, 2]")
print(f"OUTPUT:\n{nonzero_result}")
print(f"DESCRIPTION: Find all (row, col) coordinates where distance > 0")
print(f"  - Row index = which query point")
print(f"  - Col index = which data point is its neighbor\n")

nbr_indices = dists.nonzero()[:,1:].reshape(-1,)
print("="*80)
print("LINE 6b: nbr_indices = dists.nonzero()[:,1:].reshape(-1,)")
print("="*80)
print(f"OUTPUT shape: {nbr_indices.shape}  # [num_neighbors] - flat list of neighbor indices")
print(f"OUTPUT:\n{nbr_indices}")
print(f"DESCRIPTION: Extract only the data point indices (column indices) and flatten")
print(f"  - Ordered as: [neighbors of q0, neighbors of q1, neighbors of q2, ...]")
print(f"  - In this example: query 0 has 2 neighbors (0,1), query 1 has 2 neighbors (2,3), query 2 has 0\n")

# ===== LINE 7: if return_norm =====
if return_norm:
    print("="*80)
    print("LINE 7-8: if return_norm: weights = dists[dists.nonzero(as_tuple=True)]")
    print("="*80)
    weights_nonzero = dists.nonzero(as_tuple=True)
    print(f"dists.nonzero(as_tuple=True) returns 2 tensors:")
    print(f"  - Row indices: {weights_nonzero[0]}")
    print(f"  - Col indices: {weights_nonzero[1]}")

    weights = dists[dists.nonzero(as_tuple=True)]
    print(f"\nweights = dists[nonzero_coords]")
    print(f"OUTPUT shape: {weights.shape}")
    print(f"OUTPUT:\n{weights}")
    print(f"DESCRIPTION: Extract the actual distance values for valid neighbors\n")

    print("="*80)
    print("LINE 9: nbr_dict['weights'] = weights ** 2")
    print("="*80)
    weights_squared = weights ** 2
    nbr_dict["weights"] = weights_squared
    print(f"OUTPUT:\n{weights_squared}")
    print(f"DESCRIPTION: Store SQUARED distances as weights (weighting function)\n")

# ===== LINE 10: in_nbr = torch.where(dists > 0, 1., 0.,) =====
in_nbr = torch.where(dists > 0, 1., 0.,)
print("="*80)
print("LINE 10: in_nbr = torch.where(dists > 0, 1., 0.,)")
print("="*80)
print(f"OUTPUT shape: {in_nbr.shape}  # [num_queries, num_data]")
print(f"OUTPUT:\n{in_nbr}")
print(f"DESCRIPTION: Convert to binary mask - 1.0 if neighbor, 0.0 if not")
print(f"  - Same as dists but binarized\n")

# ===== LINE 11: nbrhd_sizes = torch.cumsum(torch.sum(in_nbr, dim=1), dim=0) =====
sum_per_query = torch.sum(in_nbr, dim=1)
print("="*80)
print("LINE 11a: INTERMEDIATE - torch.sum(in_nbr, dim=1)")
print("="*80)
print(f"OUTPUT shape: {sum_per_query.shape}  # [num_queries]")
print(f"OUTPUT:\n{sum_per_query}")
print(f"DESCRIPTION: Count neighbors per query point")
print(f"  - query 0 has {int(sum_per_query[0])} neighbors")
print(f"  - query 1 has {int(sum_per_query[1])} neighbors")
print(f"  - query 2 has {int(sum_per_query[2])} neighbors\n")

nbrhd_sizes = torch.cumsum(sum_per_query, dim=0)
print("="*80)
print("LINE 11b: nbrhd_sizes = torch.cumsum(torch.sum(in_nbr, dim=1), dim=0)")
print("="*80)
print(f"OUTPUT shape: {nbrhd_sizes.shape}  # [num_queries]")
print(f"OUTPUT:\n{nbrhd_sizes}")
print(f"DESCRIPTION: Cumulative sum of neighbor counts")
print(f"  - Position 0: {int(nbrhd_sizes[0])} = sum of neighbors up to query 0")
print(f"  - Position 1: {int(nbrhd_sizes[1])} = sum of neighbors up to query 1")
print(f"  - Position 2: {int(nbrhd_sizes[2])} = sum of neighbors up to query 2\n")

# ===== LINE 12: splits = torch.cat((torch.tensor([0.]).to(queries.device), nbrhd_sizes)) =====
initial_zero = torch.tensor([0.]).to(queries.device)
print("="*80)
print("LINE 12: splits = torch.cat((torch.tensor([0.]).to(queries.device), nbrhd_sizes))")
print("="*80)
print(f"torch.tensor([0.]) = {initial_zero}")
print(f"nbrhd_sizes = {nbrhd_sizes}")

splits = torch.cat((initial_zero, nbrhd_sizes))
print(f"\nOUTPUT shape: {splits.shape}  # [num_queries + 1]")
print(f"OUTPUT:\n{splits}")
print(f"DESCRIPTION: Row split indices for accessing neighbors (Compressed Row Storage format)")
print(f"  - splits[i] = index where neighbors of query i START in nbr_indices")
print(f"  - splits[i+1] = index where neighbors of query i END in nbr_indices")
print(f"  - To get neighbors of query 0: nbr_indices[splits[0]:splits[1]] = nbr_indices[0:2]")
print(f"  - To get neighbors of query 1: nbr_indices[splits[1]:splits[2]] = nbr_indices[2:4]")
print(f"  - To get neighbors of query 2: nbr_indices[splits[2]:splits[3]] = nbr_indices[4:4]\n")

# ===== LINE 13-14: Store in dict =====
nbr_dict["neighbors_index"] = nbr_indices.long().to(queries.device)
nbr_dict["neighbors_row_splits"] = splits.long()

print("="*80)
print("LINES 13-14: Store results in nbr_dict")
print("="*80)
print(f"nbr_dict['neighbors_index'] shape: {nbr_dict['neighbors_index'].shape}")
print(f"nbr_dict['neighbors_index']:\n{nbr_dict['neighbors_index']}\n")
print(f"nbr_dict['neighbors_row_splits'] shape: {nbr_dict['neighbors_row_splits'].shape}")
print(f"nbr_dict['neighbors_row_splits']:\n{nbr_dict['neighbors_row_splits']}")
if return_norm:
    print(f"\nnbr_dict['weights'] shape: {nbr_dict['weights'].shape}")
    print(f"nbr_dict['weights']:\n{nbr_dict['weights']}")

# ===== LINE 15: return =====
print("\n" + "="*80)
print("FINAL RETURN")
print("="*80)
print(f"Returns: nbr_dict with keys: {list(nbr_dict.keys())}")
print(f"\nUSAGE EXAMPLE - Access neighbors of query 0:")
q_idx = 0
start = int(nbr_dict['neighbors_row_splits'][q_idx])
end = int(nbr_dict['neighbors_row_splits'][q_idx + 1])
neighbors = nbr_dict['neighbors_index'][start:end]
print(f"  nbr_dict['neighbors_row_splits'][{q_idx}:{q_idx+1}] = [{start}:{end}]")
print(f"  nbr_dict['neighbors_index'][{start}:{end}] = {neighbors.tolist()}")
if return_norm:
    weights = nbr_dict['weights'][start:end]
    print(f"  nbr_dict['weights'][{start}:{end}] = {weights.tolist()}")

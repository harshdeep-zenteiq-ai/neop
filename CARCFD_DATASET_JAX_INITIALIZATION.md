# CarCFDDatasetjax Initialization - Detailed Flow Documentation

## Overview
When you instantiate `CarCFDDatasetjax`, a complete data pipeline is executed that loads, processes, and normalizes Car CFD dataset for JAX/Flax training. This document details the entire initialization sequence with all operations, intermediate states, and final outputs.

---

## Initialization Call Example
```python
data_module_jax = CarCFDDatasetjax(
    root_dir=config.data.root,
    query_res=[config.data.sdf_query_resolution] * 3,
    n_train=config.data.n_train,
    n_test=config.data.n_test,
    download=config.data.download,
)
```

---

## Phase 1: CarCFDDatasetjax.__init__() Setup

### 1.1 Parameter Initialization
```
Input Parameters:
- root_dir: Union[str, Path]          # Root directory for dataset
- n_train: int = 1                    # Number of training samples to load
- n_test: int = 1                     # Number of test samples to load
- query_res: List[int] = [32, 32, 32] # Resolution of SDF query grid
- download: bool = True                # Whether to download data from Zenodo
```

### 1.2 Set Zenodo Record ID
```python
self.zenodo_record_id = "13936501"
```
- Stores the Zenodo record ID for downloading the Car CFD dataset
- Dataset URL: https://zenodo.org/records/13936501

### 1.3 Root Directory Processing
```
Input: root_dir (can be string or Path)
↓
If string:
  - Convert to Path object
  - Expand user home directory (~)
  - Resolve to absolute path
↓
Check directory existence:
  - If NOT exists:
    - Create directory with parents=True
    - Prints: "root_dir [path]"
```

### 1.4 Optional Data Download
```
If download=True:
  ├─ Call: download_from_zenodo_record(
  │   record_id="13936501",
  │   root=root_dir
  │ )
  └─ Downloads entire Car CFD dataset from Zenodo to root_dir
  
If download=False:
  └─ Skip download (assumes data already exists)
```

**Zenodo Dataset Contents:**
- Pre-processed car CFD data including:
  - Triangle mesh files (*.ply format)
  - Pressure data (*.npy files)
  - train.txt / test.txt index files
  - Stored in "processed-car-pressure-data" subdirectory

---

## Phase 2: Parent Class (MeshDataModule) Initialization

### 2.1 Directory Validation
```
Input:
  - root_dir: resolved absolute path
  - item_dir_name: "processed-car-pressure-data"
  - dataset_root = root_dir / "processed-car-pressure-data"

Validation Steps:
  ✓ Check dataset_root.exists() → Assert True
  ✓ Check dataset_root.is_dir() → Assert True
  ✓ Fail if directory missing (user must have downloaded data)
```

### 2.2 Load Train/Test Indices
```
Read Files:
  1. dataset_root / "train.txt"
     └─ Single line with comma-separated sample IDs
     └─ Example: "sample_001,sample_002,sample_003,..."
  
  2. dataset_root / "test.txt"
     └─ Single line with comma-separated sample IDs
     └─ Example: "sample_101,sample_102,..."

Process Indices:
  train_ind = file_content.split(",")
  test_ind = file_content.split(",")
  
  If n_train provided:
    train_ind = train_ind[0:n_train]
  If n_test provided:
    test_ind = test_ind[0:n_test]
  
  n_train = len(train_ind)  # Actual count after trimming
  n_test = len(test_ind)
  
  Print: "[INIT] n_train={n_train}, n_test={n_test}"

Result:
  - train_ind: list of training sample IDs
  - test_ind: list of testing sample IDs
  - mesh_ind: train_ind + test_ind (combined list)
```

### 2.3 Locate Data Directory
```
Check for correct data directory structure:
  
  Path 1 (Try): root_dir / "processed-car-pressure-data" / "data" / {sample_id} / "tri_mesh.ply"
  
  If NOT found:
    Path 2 (Retry): root_dir / "processed-car-pressure-data" / "data" / "data" / {sample_id} / "tri_mesh.ply"
    └─ Use this nested structure
  
  Else:
    Use original path
```

### 2.4 Load Mesh Files
```
For each sample_id in mesh_ind:
  ├─ mesh_path = data_dir / f"{sample_id}/tri_mesh.ply"
  ├─ mesh = o3d.io.read_triangle_mesh(mesh_path)
  │  └─ Load 3D triangle mesh using Open3D
  │  └─ Returns Open3D mesh object with:
  │     - vertices: Nx3 array of 3D vertex coordinates
  │     - triangles: Mx3 array of vertex indices forming triangles
  │     - (optionally) vertex_colors, normals, etc.
  │
  └─ meshes.append(mesh)

Output:
  - meshes: list of Open3D triangle mesh objects
  - Total meshes loaded = len(train_ind) + len(test_ind)
```

### 2.5 Compute Global Bounding Box
```
Process each mesh:
  for j, mesh in enumerate(meshes):
    min_b[:, j] = mesh.get_min_bound()  # 3D min corner
    max_b[:, j] = mesh.get_max_bound()  # 3D max corner

Compute Global Bounds:
  min_b = min_b.min(axis=1)  # Minimum across all meshes (3,)
  max_b = max_b.max(axis=1)  # Maximum across all meshes (3,)

Output:
  - min_b: (3,) array [x_min, y_min, z_min]
  - max_b: (3,) array [x_max, y_max, z_max]
  - Defines a global bounding cube encompassing all meshes
```

### 2.6 Create Query Points Grid
```
Input:
  - query_res = [query_res[0], query_res[1], query_res[2]]
  - For example: [32, 32, 32]
  - min_b, max_b from previous step

Process:
  1. Create 1D linspace for each dimension:
     tx = np.linspace(min_b[0], max_b[0], query_res[0])
     ty = np.linspace(min_b[1], max_b[1], query_res[1])
     tz = np.linspace(min_b[2], max_b[2], query_res[2])
  
  2. Create 3D meshgrid:
     query_points = np.stack(
       np.meshgrid(tx, ty, tz, indexing="ij"),
       axis=-1
     )
  
  3. Convert to float32:
     query_points = query_points.astype(np.float32)

Output:
  - query_points: (query_res[0], query_res[1], query_res[2], 3) array
  - For [32, 32, 32]: shape (32, 32, 32, 3)
  - Contains 32,768 3D coordinates uniformly distributed in bounding box
  - Each point has [x, y, z] in original coordinate space
```

### 2.7 Process Each Mesh - Core Data Processing

#### For each mesh i:
```
Step A: Compute Mesh Properties
  ├─ mesh.compute_triangle_normals()
  │  └─ Add normal vectors for each triangle
  │
  ├─ mesh.compute_vertex_normals()
  │  └─ Add normal vectors for each vertex
  │
  ├─ item_dict["vertices"] = mesh.vertices (Nx3)
  │  └─ Array of vertex coordinates
  │
  ├─ item_dict["vertex_normals"] = mesh.vertex_normals (Nx3)
  │  └─ Normal vectors at each vertex
  │
  └─ item_dict["triangle_normals"] = mesh.triangle_normals (Mx3)
     └─ Normal vectors for each triangle

Step B: Compute Triangle Centroids and Areas
  ├─ For each triangle defined by vertices A, B, C:
  │  ├─ centroid = (A + B + C) / 3
  │  └─ area = sqrt(sum(cross(B-A, C-A)^2)) / 2
  │
  ├─ item_dict["centroids"] = centroids (Mx3)
  │  └─ Center point of each triangle
  │
  └─ item_dict["triangle_areas"] = areas (M,)
     └─ Surface area of each triangle

Step C: Normalize Vertices and Centroids to [0, 1]
  ├─ Convert from global [min_b, max_b] to [0, 1]
  │
  ├─ item_dict["vertices"] = range_normalize(vertices, min_b, max_b, 0, 1)
  │  └─ Formula: (vertices - min_b) / (max_b - min_b)
  │
  └─ item_dict["centroids"] = range_normalize(centroids, min_b, max_b, 0, 1)
     └─ Same normalization applied to centroids

Step D: Compute Signed Distance Function (SDF)
  ├─ Convert Open3D mesh to tensor format for raycasting:
  │  mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
  │
  ├─ Create raycasting scene with BVH acceleration:
  │  scene = o3d.t.geometry.RaycastingScene()
  │  scene.add_triangles(mesh)
  │
  ├─ Compute signed distances from all query points to mesh:
  │  distance = scene.compute_signed_distance(query_points)
  │  └─ Positive inside mesh, negative outside
  │  └─ Shape: (query_res[0], query_res[1], query_res[2])
  │
  ├─ Compute closest points on mesh to each query point:
  │  closest = scene.compute_closest_points(query_points)["points"]
  │  └─ Shape: (query_res[0], query_res[1], query_res[2], 3)
  │
  └─ item_dict["distance"] = expand_dims(distance, -1)
     item_dict["closest_points"] = closest
     item_dict["closest_points"] = range_normalize(closest, min_b, max_b, 0, 1)
     └─ Add dimension and normalize closest points to [0, 1]

Output for each mesh:
  item_dict = {
    "vertices": (N, 3),                          # Vertex coordinates
    "vertex_normals": (N, 3),                    # Normals at vertices
    "triangle_normals": (M, 3),                  # Normals at triangles
    "centroids": (M, 3),                         # Triangle centers
    "triangle_areas": (M,),                      # Triangle areas
    "distance": (32, 32, 32, 1),                 # SDF (expanded)
    "closest_points": (32, 32, 32, 3),           # Closest points on mesh
  }
```

### 2.8 Handle Processing Errors
```
If distance computation fails for any mesh:
  ├─ Add sample_id to deleted_meshes list
  ├─ Skip this mesh with continue
  └─ Adjust n_train: n_train -= len(deleted_meshes)
```

### 2.9 Compute Data Normalization Bounds
```
Using training data only (data[0:n_train]):

1. Find distance bounds:
   min_dist = min(all distances in training data)
   max_dist = max(all distances in training data)

2. Find triangle area bounds:
   min_area = min(all areas in training data)
   max_area = max(all areas in training data)

These bounds are used for normalizing test data as well
(Important: test set normalized using training statistics)
```

### 2.10 Normalize All Data
```
For each data sample:
  
  1. Normalize distance to [1e-6, 1]:
     distance = range_normalize(distance, min_dist, max_dist, 1e-6, 1)
     └─ Shift away from 0 to avoid numerical issues
  
  2. Normalize triangle areas to [1e-6, 1]:
     normalized_triangle_areas = range_normalize(
       triangle_areas, min_area, max_area, 1e-6, 1
     )
     └─ Stored separately from original areas
```

### 2.11 Convert to JAX Arrays
```
For each data_dict in data:
  For each key in data_dict:
    data_dict[key] = jnp.array(data_dict[key], dtype=jnp.float32)
    └─ Convert all numpy arrays to JAX float32 arrays
    └─ Enables JAX/Flax compatibility
```

### 2.12 Load Attributes (Pressure Data)
```
Input: attributes = ["press"]

For each sample_id in mesh_ind (training + test):
  For each attribute in ["press"]:
    path = data_dir / f"{sample_id}/press.npy"
    attr_data = np.load(path)
    └─ Shape typically: (T, P) where T=timesteps, P=pressures
    
    data[j]["press"] = jnp.array(attr_data, dtype=jnp.float32)

Output:
  - Each data sample now has "press" key with pressure values
```

### 2.13 Create Gaussian Normalizers
```
1. Identify which attributes need normalization:
   normalizer_keys = ["press"]  # From attributes with JAX array values
   
2. Compute statistics from training data only:
   self.normalizers = UnitGaussianNormalizer.from_dataset(
     data[0:n_train],
     axis=[1],
     keys=["press"]
   )
   
   UnitGaussianNormalizer computes:
   - mean and std from training set
   - Stores per-attribute normalizers
   
   Output: dict mapping "press" → UnitGaussianNormalizer instance

3. Apply normalization to all data (train + test):
   For each sample and attribute "press":
     ├─ Unsqueeze if needed: (P,) → (1, P)
     ├─ Transform using normalizer:
     │  data[j]["press"] = normalizer.transform(data[j]["press"])
     │  └─ Subtracts mean and divides by std
     │
     └─ Result: zero-mean, unit-variance data
```

### 2.14 Normalize Query Points
```
Convert query_points from [0, 1] (from bounding box normalization):
  query_points = range_normalize(
    query_points, min_b, max_b, 0, 1
  )

Result:
  - query_points: (32, 32, 32, 3) in [0, 1] range
```

### 2.15 Convert Query Points to JAX Array
```
query_points = jnp.array(query_points, dtype=jnp.float32)

Output:
  - query_points: JAX float32 array ready for model input
```

### 2.16 Create Train/Test Datasets
```
1. Create constant dictionary:
   constant = {"query_points": query_points}
   └─ Shared across all samples (same query grid for all)

2. Create training dataset:
   self.train_data = DictDataset(
     data_list=data[0:n_train],
     constant=constant
   )
   └─ Type: DictDataset
   └─ Length: n_train samples
   └─ Each sample: dict with all keys + query_points

3. Create test dataset:
   self.test_data = DictDataset(
     data_list=data[n_train:],
     constant=constant
   )
   └─ Type: DictDataset
   └─ Length: n_test samples
   └─ Each sample: dict with all keys + query_points
```

---

## Phase 3: Post-Processing in CarCFDDatasetjax

### 3.1 Adjust Pressure Data
```
Problem:
  - Original pressure array has 128 elements (with padding)
  - Actual mesh vertices are 48
  - Elements 16-112 (96 values) are padding that need removal

Solution applied to training data:
  for i, data in enumerate(self.train_data.data_list):
    press = data["press"]  # Shape: (..., 128)
    
    # Keep columns 0-16 and 112-end (skip 16-112)
    self.train_data.data_list[i]["press"] = jnp.concatenate(
      (press[:, 0:16], press[:, 112:]),
      axis=1
    )
    
    Result:
    - Column 0-16 (16 values) + columns 112-128 (16 values) = 32 values
    - Wait, original has 128 elements...
    
    Actual result: 16 + 16 = 32 values (if 128 total)
    OR: depends on actual dimension, but removes middle padding

Solution applied to test data:
  for i, data in enumerate(self.test_data.data_list):
    press = data["press"]  # Shape: (..., 128)
    
    self.test_data.data_list[i]["press"] = jnp.concatenate(
      (press[:, 0:16], press[:, 112:]),
      axis=1
    )
    
    Same transformation as training data

Output:
  - Pressure data shaped to match actual mesh vertices
  - Padding removed from both train and test sets
```

---

## Final State: What Gets Returned

### CarCFDDatasetjax Instance Contains:

```python
data_module = CarCFDDatasetjax(...)

data_module attributes:
├─ self.zenodo_record_id: str = "13936501"
│
├─ self.train_data: DictDataset
│  ├─ Length: n_train
│  └─ Each item (via __getitem__):
│     {
│       "vertices": (N, 3) JAX array - normalized vertex coords
│       "vertex_normals": (N, 3) JAX array - vertex normals
│       "triangle_normals": (M, 3) JAX array - triangle normals
│       "centroids": (M, 3) JAX array - triangle centers
│       "triangle_areas": (M,) JAX array - triangle areas
│       "distance": (32, 32, 32, 1) JAX array - normalized SDF
│       "closest_points": (32, 32, 32, 3) JAX array - closest points on mesh
│       "press": (32, 32, 32) or similar JAX array - normalized pressure
│       "query_points": (32, 32, 32, 3) JAX array - query grid points
│     }
│
├─ self.test_data: DictDataset
│  ├─ Length: n_test
│  └─ Each item: same structure as train_data
│
└─ self.normalizers: dict
   ├─ Keys: ["press"]
   └─ Values: UnitGaussianNormalizer instances
      ├─ Stores mean/std computed from training data
      ├─ Can be used to denormalize predictions
      └─ Applied during __init__ to all data
```

### DictDataset.__getitem__() Behavior:
```python
item = data_module.train_data[index]

Returns:
  dict containing:
  ├─ Copy of data_list[index]
  └─ Merged with constant dictionary (query_points)
```

---

## Data Flow Summary

```
Input Parameters
    ↓
[Phase 1] CarCFDDatasetjax Setup
    ↓
Root directory validation/creation
    ↓
Optional Zenodo download
    ↓
[Phase 2] MeshDataModule parent initialization
    ├─ Load train/test indices from text files
    ├─ Load triangle meshes from .ply files
    ├─ Compute bounding box
    ├─ Create query point grid (32×32×32)
    ├─ For each mesh:
    │  ├─ Compute vertex/triangle normals
    │  ├─ Compute triangle centroids and areas
    │  ├─ Compute signed distance function to all query points
    │  └─ Normalize all geometry to [0, 1]
    ├─ Load pressure attribute data from .npy files
    ├─ Compute Gaussian normalizers from training data
    ├─ Normalize all pressure values to zero-mean, unit-variance
    ├─ Create training and test DictDataset objects
    └─ Initialize self.train_data and self.test_data
    ↓
[Phase 3] Post-Processing
    └─ Remove padding from pressure data (columns 16-112)
    ↓
CarCFDDatasetjax instance ready for training
```

---

## Key Normalizations Applied

| Data | Source Range | Target Range | When Applied |
|------|--------------|--------------|--------------|
| Vertices | Global bounds | [0, 1] | During mesh processing |
| Centroids | Global bounds | [0, 1] | During mesh processing |
| Distance | [min_dist, max_dist] | [1e-6, 1] | After SDF computation |
| Triangle Areas | [min_area, max_area] | [1e-6, 1] | After area computation |
| Closest Points | Global bounds | [0, 1] | During SDF processing |
| Query Points | Global bounds | [0, 1] | After grid creation |
| Pressure | Mean/std (train data) | 0 mean, σ=1 | Via UnitGaussianNormalizer |

---

## Memory Considerations

```
For n_train + n_test samples with query_res=[32, 32, 32]:

Per Sample:
  - vertices: ~(1KB - 10KB depending on mesh density)
  - distance: 32×32×32×4 bytes = ~512 KB per sample
  - closest_points: 32×32×32×3×4 bytes = ~1.5 MB per sample
  - pressure: variable (depends on timesteps)
  
Total:
  For 5 training + 2 test samples:
  ~10 MB - 15 MB in memory (approximate)

Lazy Loading:
  - DictDataset doesn't pre-allocate batches
  - _LazyLoader creates batches on-demand during iteration
  - Avoids GPU OOM issues with large datasets
```

---

## Error Handling

```
Potential failures during initialization:

1. Directory doesn't exist:
   └─ Creates with mkdir(parents=True)

2. Dataset directory missing:
   └─ Assert fails, user must download data

3. Mesh file missing:
   └─ o3d.io.read_triangle_mesh fails with file not found

4. Distance computation fails:
   └─ Sample skipped, n_train decremented

5. Pressure file missing:
   └─ np.load fails with file not found

6. Normalizer computation fails:
   └─ UnitGaussianNormalizer.from_dataset may fail if no valid data
```

---

## Usage Example

```python
# Initialize dataset
data_module = CarCFDDatasetjax(
    root_dir="/path/to/data",
    n_train=100,
    n_test=20,
    query_res=[32, 32, 32],
    download=True  # Downloads from Zenodo if not present
)

# Access training data
train_loader = data_module.train_loader(batch_size=4, shuffle=True)

for batch in train_loader:
    # batch is a dict:
    # {
    #   "vertices": (4, N, 3),
    #   "distance": (4, 32, 32, 32, 1),
    #   "press": (4, 32, 32, 32),
    #   "query_points": (4, 32, 32, 32, 3),
    #   ... other keys ...
    # }
    
    # Use with JAX/Flax model
    predictions = model(batch)

# Denormalize pressure predictions
denorm_predictions = data_module.normalizers["press"].inverse_transform(predictions)
```

---

## References

- **Dataset Source**: [Zenodo Record 13936501](https://zenodo.org/records/13936501)
- **Original Paper**: Umetani, N. and Bickel, B. (2018). "Learning three-dimensional flow for interactive aerodynamic design". ACM Transactions on Graphics.
- **Libraries Used**:
  - `open3d`: Mesh processing and raycasting
  - `jax`/`jax.numpy`: Array operations
  - `numpy`: Numerical computations
  - `vtk`: Visualization support

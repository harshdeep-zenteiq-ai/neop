"""JAX/Flax version of Mesh DataModule for handling mesh-based datasets"""

from pathlib import Path
from timeit import default_timer
from typing import List, Union
import jax
import numpy as np
import jax.numpy as jnp
import sys

from neuralop.data.transforms.normalizers_jax import UnitGaussianNormalizer
from .dict_dataset_jax import DictDataset 

try:
    import open3d as o3d
    o3d_warn = False
except ModuleNotFoundError:
    o3d_warn = True

# For visualization of meshes and pressure fields
import vtk
from vtk.util.numpy_support import numpy_to_vtk


class _LazyLoader:
    """Lazy data loader that builds each batch on demand.

    Batches are constructed only during iteration, so no GPU memory is
    consumed at construction time.  ``batch_size=1`` is assumed for dict-
    valued keys such as ``neighbors_in`` / ``neighbors_out``.
    """

    def __init__(self, data, index_groups):
        self._data         = data
        self._index_groups = index_groups
        # Expose a minimal `.dataset` object so the trainer can call len()
        _n = len(index_groups)
        self.dataset = type('Dataset', (), {'__len__': lambda _: _n})()

    def __len__(self):
        return len(self._index_groups)

    def __iter__(self):
        for batch_indices in self._index_groups:
            batch = {}
            for key in self._data[0].keys():
                values = [self._data[idx][key] for idx in batch_indices]
                if isinstance(values[0], dict):
                    batch[key] = values[0] if len(values) == 1 else values
                else:
                    batch[key] = jnp.stack(values)
            yield batch


class MeshDataModule:
    """JAX/Flax version of MeshDataModule for irregular coordinate meshes.

    Provides a general dataset for irregular coordinate meshes for use in
    GNO-based architectures with JAX/Flax.

    Parameters
    ----------
    root_dir : Union[str, Path]
        str or Path to root directory of CFD dataset
    item_dir_name : Union[str, Path]
        directory in which individual item subdirs are stored
    n_train : int, optional
        hard limit on number of training examples
    n_test : int, optional
        hard limit on number of test examples
    query_res : List[int], optional
        resolution of latent query points along each dimension
    attributes : List[str], optional
        list of string keys for attributes in the dataset to return
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        item_dir_name: Union[str, Path],
        n_train: int = None,
        n_test: int = None,
        query_res: List[int] = None,
        attributes: List[str] = None,
    ):
        """MeshDataModule initialization."""

        if o3d_warn:
            print("Warning: MeshDataModule requires open3d dependency")
            raise ModuleNotFoundError()

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
            # print(f"[INIT] Converted root_dir to Path: {root_dir}")

        root_dir = root_dir.expanduser().resolve()
        # print(f"[INIT] root_dir expanded/resolved: {root_dir}")
        assert root_dir.exists(), "Path does not exist"
        assert root_dir.is_dir(), "Path is not a directory"

        if isinstance(item_dir_name, (list, tuple)):
            item_dir_name = item_dir_name[0]

        dataset_root = (root_dir / item_dir_name).expanduser().resolve()
        # print(f"[INIT] dataset_root: {dataset_root}")
        assert dataset_root.exists(), "Dataset folder does not exist"

        # Read train and test indices
        with open(dataset_root / "train.txt") as file:
            train_ind = file.readline().split(",")
        with open(dataset_root / "test.txt") as file:
            test_ind = file.readline().split(",")

        if n_train is not None and n_train < len(train_ind):
            train_ind = train_ind[0:n_train]
        if n_test is not None and n_test < len(test_ind):
            test_ind = test_ind[0:n_test]

        train_ind = train_ind[0:n_train]
        test_ind = test_ind[0:n_test]
        n_train = len(train_ind)
        n_test = len(test_ind)
        print(f"[INIT] n_train={n_train}, n_test={n_test}")

        mesh_ind = train_ind + test_ind
        mesh_ind = [x.rstrip() for x in mesh_ind]

        data_dir = dataset_root / "data"
        mesh_path_check = data_dir / train_ind[0] / "tri_mesh.ply"
        if not mesh_path_check.exists():
            data_dir = data_dir / "data"

        # Load meshes
        # print(f"\n[MESH LOADING] Starting to load {len(mesh_ind)} meshes...")
        meshes = []

        for ind in mesh_ind:
            mesh_path = str(data_dir / (ind + "/tri_mesh.ply"))
            # print(f"[MESH LOADING] Loading mesh from: {mesh_path}")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            # print(f"[MESH LOADING] ✓ Loaded mesh [{ind}]: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

            # Save mesh information to external file
            # mesh_output_dir = Path("./mesh_data")
            # mesh_output_dir.mkdir(parents=True, exist_ok=True)
            # mesh_file_path = mesh_output_dir / f"mesh_{ind}.txt"

            # vertices = np.asarray(mesh.vertices)
            # triangles = np.asarray(mesh.triangles)

            # with open(mesh_file_path, 'w') as f:
            #     print(f'ind: {ind} and number of vertices: {len(vertices)} and number of triangles: {len(triangles)}')
            #     f.write(f"Number of vertices: {len(vertices)}\n")
            #     f.write(f"Number of triangles: {len(triangles)}\n")
            #     f.write(f"\n{'='*80}\n")
            #     f.write(f"VERTEX COORDINATES\n")
            #     f.write(f"{'='*80}\n")
            #     for v_idx, vertex in enumerate(vertices):
            #         f.write(f"Vertex {v_idx}: {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            #     f.write(f"\n{'='*80}\n")
            #     f.write(f"TRIANGLE CONNECTIVITY\n")
            #     f.write(f"{'='*80}\n")
            #     for t_idx, triangle in enumerate(triangles):
            #         f.write(f"Triangle {t_idx}: {triangle[0]} {triangle[1]} {triangle[2]}\n")

            # print(f"[MESH LOADING] ✓ Saved mesh data to: {mesh_file_path}")

            meshes.append(mesh)

            # # 1. Load pressure data
            # pressure = np.load(str(data_dir / (ind + "/press.npy"))).squeeze()

            # # --- Diagnostic: verify shapes align ---
            # vertices_check = np.asarray(mesh.vertices)
            # triangles_check = np.asarray(mesh.triangles)
            # print(f"[{ind}] pressure shape  : {pressure.shape}  dtype={pressure.dtype}")
            # print(f"[{ind}] pressure min/max : {pressure.min():.4f} / {pressure.max():.4f}")
            # print(f"[{ind}] n_vertices       : {len(vertices_check)}")
            # print(f"[{ind}] n_triangles      : {len(triangles_check)}")
            # if pressure.shape[0] == len(vertices_check):
            #     print(f"[{ind}] ✓ pressure is VERTEX-centered")
            # elif pressure.shape[0] == len(triangles_check):
            #     print(f"[{ind}] ✗ pressure is CELL-centered — needs interpolation to vertices")
            # else:
            #     print(f"[{ind}] ✗ pressure shape {pressure.shape[0]} matches NEITHER vertices ({len(vertices_check)}) nor triangles ({len(triangles_check)})")
            # # --- End diagnostic ---

            # # 2. Build a vtkPolyData from the Open3D mesh
            # vertices = np.asarray(mesh.vertices)
            # triangles = np.asarray(mesh.triangles)

            # vtk_points = vtk.vtkPoints()
            # vtk_points.SetData(numpy_to_vtk(vertices, deep=True))

            # vtk_cells = vtk.vtkCellArray()
            # # vtkCellArray expects [n_pts, i0, i1, i2, ...] per cell
            # cell_array = np.hstack(
            #     [np.full((len(triangles), 1), 3, dtype=np.int64), triangles]
            # ).ravel()
            # vtk_id_array = vtk.vtkIdTypeArray()
            # vtk_id_array.SetNumberOfValues(len(cell_array))
            # for i, v in enumerate(cell_array):
            #     vtk_id_array.SetValue(i, int(v))
            # vtk_cells.SetCells(len(triangles), vtk_id_array)

            # poly_data = vtk.vtkPolyData()
            # poly_data.SetPoints(vtk_points)
            # poly_data.SetPolys(vtk_cells)

            # # 3. Drop indices 16:112 (96 padding values) to align with vertex count
            # # car_cfd_dataset.py applies the same slice: press[:, 0:16] + press[:, 112:]
            # pressure_vis = np.concatenate([pressure[0:16], pressure[112:]])
            # assert len(pressure_vis) == len(vertices), (
            #     f"After slice: {len(pressure_vis)} values vs {len(vertices)} vertices"
            # )
            # print(f"[{ind}] pressure_vis shape after slice: {pressure_vis.shape} ✓")

            # pressure_vtk = numpy_to_vtk(pressure_vis.astype(np.float64), deep=True)
            # pressure_vtk.SetName("Pressure")
            # poly_data.GetPointData().SetScalars(pressure_vtk)

            # # 4a. Smooth geometry — scalars are now correctly vertex-aligned
            # smoother = vtk.vtkSmoothPolyDataFilter()
            # smoother.SetInputData(poly_data)
            # smoother.SetNumberOfIterations(50)
            # smoother.SetRelaxationFactor(0.1)
            # smoother.FeatureEdgeSmoothingOff()
            # smoother.BoundarySmoothingOn()
            # smoother.Update()
            # smooth_poly = smoother.GetOutput()

            # # 5. Build a proper jet LUT (256 table values, blue → red)
            # lut = vtk.vtkLookupTable()
            # lut.SetNumberOfTableValues(256)
            # lut.SetHueRange(0.667, 0.0)   # HSV: blue (240°) → red (0°)
            # lut.SetSaturationRange(1.0, 1.0)
            # lut.SetValueRange(1.0, 1.0)
            # lut.SetRange(float(pressure_vis.min()), float(pressure_vis.max()))
            # lut.Build()

            # surface_mapper = vtk.vtkPolyDataMapper()
            # surface_mapper.SetInputData(smooth_poly)
            # surface_mapper.SetScalarRange(float(pressure_vis.min()), float(pressure_vis.max()))
            # surface_mapper.SetLookupTable(lut)
            # surface_mapper.InterpolateScalarsBeforeMappingOn()

            # surface_actor = vtk.vtkActor()
            # surface_actor.SetMapper(surface_mapper)
            # surface_actor.GetProperty().SetOpacity(0.85)

            # # 6. Scalar bar (colorbar)
            # scalar_bar = vtk.vtkScalarBarActor()
            # scalar_bar.SetLookupTable(lut)
            # scalar_bar.SetTitle("Pressure")
            # scalar_bar.SetNumberOfLabels(5)

            # # 8. Render
            # renderer = vtk.vtkRenderer()
            # renderer.AddActor(surface_actor)
            # renderer.AddActor2D(scalar_bar)
            # renderer.SetBackground(0.1, 0.1, 0.1)

            # render_window = vtk.vtkRenderWindow()
            # render_window.SetWindowName(f"Car CFD Pressure – {ind}")
            # render_window.SetSize(1024, 768)
            # render_window.AddRenderer(renderer)

            # interactor = vtk.vtkRenderWindowInteractor()
            # interactor.SetRenderWindow(render_window)
            # interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

            # render_window.Render()
            # interactor.Start()
        
        
        # print(f'\n[BOUNDING BOX] Computing global bounding box...')
        # Dataset wide bounding box
        # print(f'meshes', meshes)

        # print(f'\n[BOUNDING BOX] Computing global bounding box...')
        min_b, max_b = self.get_global_bounding_box(meshes)
        # print(f'[BOUNDING BOX] min_b: {min_b}, max_b: {max_b}')

        are_watertight = True

        # Create query points
        # print(f"\n[QUERY POINTS] Creating query grid...")
        if isinstance(query_res, (list, tuple)):
            tx = np.linspace(min_b[0], max_b[0], query_res[0])
            ty = np.linspace(min_b[1], max_b[1], query_res[1])
            tz = np.linspace(min_b[2], max_b[2], query_res[2])

            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
            # print(f"[QUERY POINTS] ✓ query_points shape: {query_points.shape}")
        else:
            raise TypeError(f"query_res must be list/tuple, got {type(query_res)}")

        # Process data from meshes
        data = []
        deleted_meshes = []
        self.time_to_distance = 0.0 
        # print(f"\n[DATA PROCESSING] Starting data processing for {len(meshes)} meshes...")

        for i, mesh in enumerate(meshes):
            # print(f"\n[DATA {i}] Processing mesh {i}...")
            item_dict = {}

            mesh = mesh.compute_triangle_normals()
            mesh = mesh.compute_vertex_normals()

            item_dict["vertices"] = np.asarray(mesh.vertices)
            item_dict["vertex_normals"] = np.asarray(mesh.vertex_normals)
            item_dict["triangle_normals"] = np.asarray(mesh.triangle_normals)

            centroids, area = self.compute_triangle_centroids(
                item_dict["vertices"], np.asarray(mesh.triangles)
            )
            item_dict["centroids"] = centroids
            item_dict["triangle_areas"] = area

            # Normalize vertices
            item_dict["vertices"] = self.range_normalize(
                item_dict["vertices"], min_b, max_b, 0, 1
            )
            item_dict["centroids"] = self.range_normalize(
                item_dict["centroids"], min_b, max_b, 0, 1
            )

            # Compute distances
            if query_points is not None:
                # print(f"[DATA {i}] Computing signed distances...")
                tt = default_timer()
                try:
                    distance, closest = self.compute_distances(
                        mesh, query_points, are_watertight
                    )
                    elapsed = default_timer() - tt
                    # print(f"[DATA {i}] ✓ Distances computed in {elapsed:.2f}s")
                    # self.time_to_distance += elapsed
                except Exception as e:
                    # print(f"[DATA {i}] ✗ ERROR computing distances: {e}")
                    deleted_meshes.append(mesh_ind[i])
                    continue
                
                self.time_to_distance += elapsed
                item_dict["distance"] = np.expand_dims(distance, -1)
                item_dict["closest_points"] = closest
                item_dict["closest_points"] = self.range_normalize(
                    closest, min_b, max_b, 0, 1
                )

            data.append(item_dict)

        self.time_to_distance /= len(meshes) if len(meshes) > 0 else 1
        
        del meshes  
        # Normalize data
        n_train -= len(deleted_meshes)
        # print(f"[DATA PROCESSING] deleted_meshes: {deleted_meshes}, n_train adjusted to {n_train}")

        # Bounds based on training data
        # print(f"\n[NORMALIZATION] Computing training set statistics...")
        min_dist, max_dist = self.get_bounds_from_data(data[0:n_train], "distance")
        # print(f"[NORMALIZATION] distance bounds (training): [{min_dist:.6f}, {max_dist:.6f}]")

        min_area, max_area = self.get_bounds_from_data(
            data[0:n_train], "triangle_areas"
        )
        # print(f"[NORMALIZATION] triangle_areas bounds (training): [{min_area:.6e}, {max_area:.6e}]")

        # print(f"[NORMALIZATION] Normalizing all data...")
        for i, data_dict in enumerate(data):
            data_dict["distance"] = self.range_normalize(
                data_dict["distance"], min_dist, max_dist, 1e-6, 1
            )
            # print(f"[NORMALIZATION] data[{i}] distance normalized: range [{data_dict['distance'].min():.6f}, {data_dict['distance'].max():.6f}]")

            data_dict["normalized_triangle_areas"] = self.range_normalize(
                data_dict["triangle_areas"], min_area, max_area, 1e-6, 1
            )
            # print(f"[NORMALIZATION] data[{i}] triangle_areas normalized: range [{data_dict['normalized_triangle_areas'].min():.6f}, {data_dict['normalized_triangle_areas'].max():.6f}]")

        # Convert to torch
        # print(f"\n[TORCH CONVERSION] Converting numpy arrays to torch tensors...")
        for i, data_dict in enumerate(data):
            for key in data_dict:
                # data_dict[key] = torch.from_numpy(data_dict[key]).to(torch.float32)
                data_dict[key] = jnp.array(data_dict[key], dtype=jnp.float32)
            # print(f"[TORCH CONVERSION] data[{i}]: {len(data_dict)} keys converted to torch.float32")

        # Load non-mesh data
        # print(f"\n[ATTRIBUTES] Loading attributes: {attributes}")
        if attributes is not None:
            for j, ind in enumerate(mesh_ind):
                if ind in deleted_meshes:
                    continue
                for attr in attributes:
                    path = str(data_dir / (ind + "/" + attr + ".npy"))
                    attr_data = np.load(path)
        
                    data[j][attr] = jnp.array(attr_data, dtype=jnp.float32)

            # Compute Gaussian normalizers based on training data
            # print(f"\n[GAUSSIAN NORM] Computing Gaussian normalizers from training data...")
            normalizer_keys = []
            for attr in attributes:
                # if isinstance(data[0][attr], torch.Tensor):
                #     normalizer_keys.append(attr)
                if isinstance(data[0][attr], jax.Array) or isinstance(data[0][attr], jnp.ndarray):
                    normalizer_keys.append(attr)
                    # print(f"[GAUSSIAN NORM] Added {attr} to normalizer_keys")
            # print(f"[GAUSSIAN NORM] normalizer_keys: {normalizer_keys}")

            # returns keyed dict of UnitGaussianNormalizer instances
            self.normalizers = UnitGaussianNormalizer.from_dataset(
                data, axis=[1], keys=normalizer_keys
            )
            # print(f"[GAUSSIAN NORM] Normalizers created: {list(self.normalizers.keys()) if self.normalizers else 'None'}")

            # Encode all data
            # print(f"[GAUSSIAN NORM] Applying normalizers to all data...")
            for attr in normalizer_keys:
                # print(f"[GAUSSIAN NORM] Normalizing {attr} across all samples...")
                for j in range(len(data)):
                    data_elem = data[j][attr]
                    shape_before = data_elem.shape
                    if data_elem.shape[0] != 1:
                        data_elem = jnp.expand_dims(data_elem, axis=0)
                        # print(f"[GAUSSIAN NORM] data[{j}][{attr}] unsqueezed: {shape_before} → {data_elem.shape}")
                    data[j][attr] = self.normalizers[attr].transform(data_elem)
                    # print(f"[GAUSSIAN NORM] data[{j}][{attr}] normalized: mean={data[j][attr].mean():.4f}, std={data[j][attr].std():.4f}")

            if not bool(self.normalizers):
                self.normalizers = None
                # print(f"[GAUSSIAN NORM] Normalizers is empty, set to None")
        else:
            self.normalizers = None
            # print(f"[GAUSSIAN NORM] No attributes provided, normalizers set to None")

        # Set-up constant dict
        # print(f"\n[FINAL SETUP] Normalizing query points...")
        query_points = self.range_normalize(query_points, min_b, max_b, 0, 1)
        # print(f"[FINAL SETUP] query_points normalized: shape {query_points.shape}, range [{query_points.min():.4f}, {query_points.max():.4f}]")

        # query_points = torch.from_numpy(query_points).to(torch.float32)
        query_points = jnp.array(query_points, dtype=jnp.float32)
        # print(f"[FINAL SETUP] query_points to torch: shape {query_points.shape}, dtype {query_points.dtype}")

        constant = {"query_points": query_points}
        # print(f"[FINAL SETUP] constant dict created with keys: {list(constant.keys())}")

        # Datasets
        # print(f"\n[FINAL SETUP] Creating train/test datasets...")
        self.train_data = DictDataset(data[0:n_train], constant)
        # print(f"[FINAL SETUP] ✓ train_data created: {len(self.train_data)} samples")

        self.test_data = DictDataset(data[n_train:], constant)
        # print(f"[FINAL SETUP] ✓ test_data created: {len(self.test_data)} samples")
        # print(f"\n[INIT] ✓✓✓ MeshDataModule initialization complete! ✓✓✓")
        
        # sys.exit(0)  # TEMP EXIT TO CHECK OUTPUT DURING INIT 
################################################################################################

    def get_global_bounding_box(self, meshes):
        # print(f"[GET_BBOX] Computing bounding boxes for {len(meshes)} meshes...")
        min_b = np.zeros((3, len(meshes)))
        max_b = np.zeros((3, len(meshes)))
        for j, mesh in enumerate(meshes):
            try:
                min_b[:, j] = mesh.get_min_bound()
                max_b[:, j] = mesh.get_max_bound()
                # print(f"[GET_BBOX] Mesh {j}: min={min_b[:, j]}, max={max_b[:, j]}")
            except IndexError as e:
                # print(f"[GET_BBOX] ✗ Mesh {j} could not be bounded: {e}")
                pass

        min_b = min_b.min(axis=1)
        max_b = max_b.max(axis=1)
        # print(f"[GET_BBOX] Global bounds: min={min_b}, max={max_b}")

        return min_b, max_b

    def are_watertight(self, meshes):
        for mesh in meshes:
            if not mesh.is_watertight():
                return False
        return True


    def compute_triangle_centroids(self, vertices, triangles):
        # print(f"[COMPUTE_CENT] Computing centroids for {len(triangles)} triangles...")
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )
        # print(f"[COMPUTE_CENT] Triangle vertices extracted")

        centroids = (A + B + C) / 3
        # print(f"[COMPUTE_CENT] Centroids computed: shape {centroids.shape}, range [{centroids.min():.4f}, {centroids.max():.4f}]")

        areas = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2
        # print(f"[COMPUTE_CENT] Areas computed: shape {areas.shape}, range [{areas.min():.6e}, {areas.max():.6e}]")

        return centroids, areas

    def compute_distances(self, mesh, query_points, signed_distance):
        # print(f"[COMPUTE_DIST] Converting mesh to tensor format...")
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        # print(f"[COMPUTE_DIST] ✓ Mesh converted, vertices: {len(mesh.vertex.positions)}")

        # print(f"[COMPUTE_DIST] Creating raycasting scene...")
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        # print(f"[COMPUTE_DIST] ✓ Scene created with BVH acceleration")

        if signed_distance:
            # print(f"[COMPUTE_DIST] Computing SIGNED distances...")
            dist = scene.compute_signed_distance(query_points).numpy()
            # print(f"[COMPUTE_DIST] ✓ Signed distance computed: shape {dist.shape}, range [{dist.min():.4f}, {dist.max():.4f}]")
        else:
            # print(f"[COMPUTE_DIST] Computing UNSIGNED distances...")
            dist = scene.compute_distance(query_points).numpy()
            # print(f"[COMPUTE_DIST] ✓ Unsigned distance computed: shape {dist.shape}, range [{dist.min():.4f}, {dist.max():.4f}]")

        # print(f"[COMPUTE_DIST] Computing closest points...")
        closest = scene.compute_closest_points(query_points)["points"].numpy()
        # print(f"[COMPUTE_DIST] ✓ Closest points computed: shape {closest.shape}")

        return dist, closest

    def range_normalize(self, data, min_b, max_b, new_min, new_max):
        # Normalize from [min_b, max_b] → [new_min, new_max]
        data = (data - min_b) / (max_b - min_b)
        data = (new_max - new_min) * data + new_min
        # Note: print statements omitted here to avoid excessive output during normalization
        return data

    def get_bounds_from_data(self, data, key):
        # print(f"[GET_BOUNDS] Computing bounds for '{key}' from {len(data)} samples...")
        global_min = data[0][key].min()
        global_max = data[0][key].max()
        # print(f"[GET_BOUNDS] data[0][{key}]: min={global_min:.6e}, max={global_max:.6e}")

        for j in range(1, len(data)):
            current_min = data[j][key].min()
            current_max = data[j][key].max()
            # print(f"[GET_BOUNDS] data[{j}][{key}]: min={current_min:.6e}, max={current_max:.6e}")

            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max

        # print(f"[GET_BOUNDS] Global bounds for '{key}': [{global_min:.6e}, {global_max:.6e}]")
        return global_min, global_max

    # ------------------------------------------------------------------
    # Lazy loader helpers
    # ------------------------------------------------------------------
    def train_loader(self, batch_size=1, shuffle=True):
        """Return training dataloader."""
        return self._create_loader(self.train_data, batch_size, shuffle)

    def test_loader(self, batch_size=1, shuffle=False):
        """Return test dataloader."""
        return self._create_loader(self.test_data, batch_size, shuffle)

    def _create_loader(self, data, batch_size=1, shuffle=False):
        """Return a lazy data loader that builds each batch on demand.

        Batches are constructed only when iterated, avoiding the GPU OOM
        that would result from pre-building all batches as JAX arrays at once.
        """
        n_samples = len(data)
        indices = list(range(n_samples))

        if shuffle:
            import random
            random.shuffle(indices)

        # Store only index groups; actual tensors are built in __iter__
        index_groups = [
            indices[i: i + batch_size]
            for i in range(0, n_samples, batch_size)
        ]

        return _LazyLoader(data, index_groups)

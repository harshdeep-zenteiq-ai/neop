from pathlib import Path
from timeit import default_timer
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt 

# import open3d for io if built. Otherwise,
# the class will build, but no files will be loaded.
try:
    import open3d as o3d

    o3d_warn = False
except ModuleNotFoundError:
    o3d_warn = True

import torch
from torch.utils.data import DataLoader

from .dict_dataset import DictDataset
from ..transforms.normalizers import UnitGaussianNormalizer
import sys 

# For visualization of meshes and pressure fields
import vtk
from vtk.util.numpy_support import numpy_to_vtk

class MeshDataModule:
    def __init__(
        self,
        root_dir: Union[str, Path],
        item_dir_name: Union[str, Path],
        n_train: int = None,
        n_test: int = None,
        query_res: List[int] = None,
        attributes: List[str] = None,
    ):
        """MeshDataModule provides a general dataset for irregular coordinate meshes
            for use in a GNO-based architecture

        Parameters
        ----------
        root_dir : Union[str, Path]
            str or Path to root directory of CFD dataset
        item_dir_name : Union[str, Path]
            directory in which individual item subdirs are stored
        n_train : int, optional
            hard limit on number of training examples
            if n_train is greater than the actual number
            of training examples available, nothing is changed
        n_test : int, optional
            hard limit on number of test examples
            if n_test is greater than the actual number
            of testing examples available, nothing is changed
        query_res : List[int], optional
            resolution of latent query points along each dimension
        attributes : List[str], optional
            list of string keys for attributes in the dataset to return
            as keys for each batch dict
        """

        if o3d_warn:
            print("Warning: you are attempting to run MeshDataModule without the required dependency open3d.")
            raise ModuleNotFoundError()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
            # print(f"[INIT] Converted root_dir to Path: {root_dir}")

        # Ensure path is valid
        root_dir = root_dir.expanduser().resolve()
        # print(f"[INIT] root_dir expanded/resolved: {root_dir}")
        assert root_dir.exists(), "Path does not exist"
        # print(f"[INIT] ✓ root_dir exists")
        assert root_dir.is_dir(), "Path is not a directory"
        # print(f"[INIT] ✓ root_dir is a directory")

        if isinstance(item_dir_name, (list, tuple)):
            item_dir_name = item_dir_name[0]
            # print(f"[INIT] item_dir_name was list/tuple, selected first: {item_dir_name}")

        dataset_root = (root_dir / item_dir_name).expanduser().resolve()
        # print(f"[INIT] dataset_root: {dataset_root}")
        assert dataset_root.exists(), "Dataset folder does not exist"
        # print(f"[INIT] ✓ dataset_root exists")
        # sys.exit(0)                    
        # Read train and test indicies
        with open(dataset_root / "train.txt") as file:
            train_ind = file.readline().split(",")
        # print(f"[INIT] train_ind loaded: {len(train_ind)} samples, first={train_ind[0] if train_ind else 'EMPTY'}")

        with open(dataset_root / "test.txt") as file:
            test_ind = file.readline().split(",")
        # print(f"[INIT] test_ind loaded: {len(test_ind)} samples, first={test_ind[0] if test_ind else 'EMPTY'}")
        
        # print(str(dataset_root / "train.txt"))
                       
        # print(f"[INIT] Requested: n_train={n_train}, n_test={n_test}")
        if n_train is not None:
            if n_train < len(train_ind):
                train_ind = train_ind[0:n_train]
                # print(f"[INIT] Sliced train_ind to {n_train} samples")

        if n_test is not None:
            if n_test < len(test_ind):
                test_ind = test_ind[0:n_test]
                # print(f"[INIT] Sliced test_ind to {n_test} samples")

        # set train and test sizes
        train_ind = train_ind[0:n_train]
        test_ind = test_ind[0:n_test]
        n_train = len(train_ind)
        n_test = len(test_ind)
        # print(f"[INIT] Final n_train={n_train}, n_test={n_test}")

        mesh_ind = train_ind + test_ind
        # print(f'[INIT] mesh_ind combined: {len(mesh_ind)} samples')
        # remove trailing newlines from train and test indices
        mesh_ind = [x.rstrip() for x in mesh_ind]
        # print(f'[INIT] mesh_ind after rstrip: {mesh_ind}')
        # print(f'dataset_root: {dataset_root}')
    
        # data_dir = root_dir / "data"
        data_dir = dataset_root / "data"
        # print(f'[INIT] data_dir initialized: {data_dir}')

        mesh_path_check = data_dir / train_ind[0] / "tri_mesh.ply"
        # print(f'[INIT] Checking mesh path: {mesh_path_check}')
        if not mesh_path_check.exists():
            # print('[INIT] Mesh path NOT found, adjusting data_dir...')
            data_dir = data_dir / "data"
            # print(f'[INIT] data_dir adjusted to: {data_dir}')
        else:
            print(f'[INIT] ✓ Mesh path exists')
        # Load all meshes
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
        
        min_b, max_b = self.get_global_bounding_box(meshes)
        # print(f'[BOUNDING BOX] min_b: {min_b}')
        # print(f'[BOUNDING BOX] max_b: {max_b}')
        # sys.exit(0)
        # are_watertight = self.are_watertight(meshes)
        are_watertight = True

        # Uniform query points if not provided
        # print(f"\n[QUERY POINTS] Creating query grid...")
        if isinstance(query_res, list) or isinstance(query_res, tuple):
            # print(f"[QUERY POINTS] query_res: {query_res}")
            tx = np.linspace(min_b[0], max_b[0], query_res[0])
            ty = np.linspace(min_b[1], max_b[1], query_res[1])
            tz = np.linspace(min_b[2], max_b[2], query_res[2])
            # print(f"[QUERY POINTS] Grid created: tx({query_res[0]}), ty({query_res[1]}), tz({query_res[2]})")

            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
            # print(f"[QUERY POINTS] ✓ query_points shape: {query_points.shape}, dtype: {query_points.dtype}")
        else:
            raise TypeError(f"query_res must be list/tuple, got {type(query_res)}")

        # Compute data from meshes
        data = []
        deleted_meshes = []
        self.time_to_distance = 0.0
        # print(f"\n[DATA PROCESSING] Starting data processing for {len(meshes)} meshes...")
        for i, mesh in enumerate(meshes):
            # print(f"\n[DATA {i}] Processing mesh {i}...")
            item_dict = {}

            mesh = mesh.compute_triangle_normals()
            # print(f"[DATA {i}] Triangle normals computed: shape {np.asarray(mesh.triangle_normals).shape}")

            mesh = mesh.compute_vertex_normals()
            # print(f"[DATA {i}] Vertex normals computed: shape {np.asarray(mesh.vertex_normals).shape}")

            item_dict["vertices"] = np.asarray(mesh.vertices)
            # print(f"[DATA {i}] vertices: shape {item_dict['vertices'].shape}")

            item_dict["vertex_normals"] = np.asarray(mesh.vertex_normals)
            # print(f"[DATA {i}] vertex_normals: shape {item_dict['vertex_normals'].shape}")

            item_dict["triangle_normals"] = np.asarray(mesh.triangle_normals)
            # print(f"[DATA {i}] triangle_normals: shape {item_dict['triangle_normals'].shape}")

            centroids, area = self.compute_triangle_centroids(
                item_dict["vertices"], np.asarray(mesh.triangles)
            )
            # print(f"[DATA {i}] centroids: shape {centroids.shape}, area: shape {area.shape}")

            item_dict["centroids"] = centroids
            item_dict["triangle_areas"] = area

            # Normalize vertex data based on global bound
            item_dict["vertices"] = self.range_normalize(
                item_dict["vertices"], min_b, max_b, 0, 1
            )
            # print(f"[DATA {i}] vertices normalized to [0,1]: range [{item_dict['vertices'].min():.4f}, {item_dict['vertices'].max():.4f}]")

            item_dict["centroids"] = self.range_normalize(
                item_dict["centroids"], min_b, max_b, 0, 1
            )
            # print(f"[DATA {i}] centroids normalized to [0,1]: range [{item_dict['centroids'].min():.4f}, {item_dict['centroids'].max():.4f}]")

            if query_points is not None:
                # print(f"[DATA {i}] Computing signed distances (this may take ~45-60s)...")
                tt = default_timer()
                try:
                    distance, closest = self.compute_distances(
                        mesh, query_points, are_watertight
                    )
                    elapsed = default_timer() - tt
                    # print(f"[DATA {i}] ✓ Distances computed in {elapsed:.2f}s")
                    # print(f"[DATA {i}] distance shape: {distance.shape}, range [{distance.min():.4f}, {distance.max():.4f}]")
                    # print(f"[DATA {i}] closest shape: {closest.shape}")
                except Exception as e:
                    # print(f"[DATA {i}] ✗ ERROR computing distances: {e}")
                    deleted_meshes.append(mesh_ind[i])
                    # print(f"[DATA {i}] Mesh marked for deletion")
                    continue
                self.time_to_distance += elapsed
                item_dict["distance"] = np.expand_dims(distance, -1)
                # print(f"[DATA {i}] distance expanded: shape {item_dict['distance'].shape}")

                item_dict["closest_points"] = closest

                # Normalize vertex data based on global bound
                item_dict["closest_points"] = self.range_normalize(
                    item_dict["closest_points"], min_b, max_b, 0, 1
                )
                # print(f"[DATA {i}] closest_points normalized to [0,1]")
            data.append(item_dict)
            # print(f"[DATA {i}] ✓ Mesh {i} added to data, total items: {len(data)}")

        self.time_to_distance /= len(meshes) if len(meshes) > 0 else 1
        # print(f"\n[DATA PROCESSING] Average time per mesh: {self.time_to_distance:.2f}s")

        del meshes
        # print(f"[DATA PROCESSING] Meshes deleted from memory")

        # remove all broken meshes from training set
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
                data_dict[key] = torch.from_numpy(data_dict[key]).to(torch.float32)
            # print(f"[TORCH CONVERSION] data[{i}]: {len(data_dict)} keys converted to torch.float32")

        # Load non-mesh data
        # print(f"\n[ATTRIBUTES] Loading attributes: {attributes}")
        if attributes is not None:
            for j, ind in enumerate(mesh_ind):
                # skip corrupted meshes we caught while adding to dataset
                if ind in deleted_meshes:
                    # print(f"[ATTRIBUTES] Skipping j={j} (mesh {ind} was deleted)")
                    continue
                # print(f"[ATTRIBUTES] Loading attributes for mesh j={j} (ind={ind})...")
                for attr in attributes:
                    path = str(data_dir / (ind + "/" + attr + ".npy"))
                    # print(f"[ATTRIBUTES] Loading {attr} from: {path}")
                    attr_data = np.load(path)
                    # print(f"[ATTRIBUTES] Loaded {attr}: shape {attr_data.shape}, dtype {attr_data.dtype}")

                    data[j][attr] = torch.from_numpy(attr_data)
                    # print(f"[ATTRIBUTES] Converted to torch: shape {data[j][attr].shape}, dtype {data[j][attr].dtype}")

                    if isinstance(data[j][attr], torch.Tensor):
                        data[j][attr] = data[j][attr].to(torch.float32)
                        # print(f"[ATTRIBUTES] Cast to float32: shape {data[j][attr].shape}")

            # Compute Gaussian normalizers based on training data
            # print(f"\n[GAUSSIAN NORM] Computing Gaussian normalizers from training data...")
            normalizer_keys = []
            for attr in attributes:
                if isinstance(data[0][attr], torch.Tensor):
                    normalizer_keys.append(attr)
                    # print(f"[GAUSSIAN NORM] Added {attr} to normalizer_keys")
            # print(f"[GAUSSIAN NORM] normalizer_keys: {normalizer_keys}")

            # returns keyed dict of UnitGaussianNormalizer instances
            self.normalizers = UnitGaussianNormalizer.from_dataset(
                data, dim=[1], keys=normalizer_keys
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
                        data_elem = data_elem.unsqueeze(0)
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

        query_points = torch.from_numpy(query_points).to(torch.float32)
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

    def train_loader(self, **kwargs):
        # print(f"[LOADER] Creating train DataLoader with {len(self.train_data)} samples")
        loader = DataLoader(self.train_data, **kwargs)
        # print(f"[LOADER] ✓ Train DataLoader created")
        return loader

    def test_loader(self, **kwargs):
        # print(f"[LOADER] Creating test DataLoader with {len(self.test_data)} samples")
        loader = DataLoader(self.test_data, **kwargs)
        # print(f"[LOADER] ✓ Test DataLoader created")
        return loader

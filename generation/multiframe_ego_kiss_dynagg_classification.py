import os
import sys
import pdb
import time
import yaml
import torch
import torch.nn.functional as F
# import chamfer
import mmcv
import numpy as np
from pytorch3d.ops.iou_box3d import box3d_overlap
from truckscenes.truckscenes import TruckScenes  ### Truckscenes
from truckscenes.utils import splits  ### Truckscenes
from tqdm import tqdm
from truckscenes.utils.data_classes import Box, LidarPointCloud
from truckscenes.utils.geometry_utils import view_points  ### Truckscenes
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import defaultdict
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
# from mmdet3d.core.bbox import box_np_ops
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation
from truckscenes.utils.geometry_utils import transform_matrix, points_in_box
import os.path as osp
from functools import reduce

import open3d as o3d
from copy import deepcopy
import matplotlib.pyplot as plt
import kiss_icp
from kiss_icp.pipeline import OdometryPipeline
from pathlib import Path
import json

from custom_datasets import InMemoryDataset, InMemoryDatasetMapMOS
import math
from shapely.geometry import Polygon, Point, MultiPolygon

from numba import njit, prange, cuda
from mapmos.pipeline import MapMOSPipeline

weights_path_mapmos = Path("/home/max/Desktop/Masterarbeit/Python/MapMOS_full/mapmos.ckpt")

CLASS_COLOR_MAP = {
    0: [0.6, 0.6, 0.6],  # noise - gray
    1: [0.9, 0.1, 0.1],  # barrier - red
    2: [1.0, 0.6, 0.0],  # bicycle - orange
    3: [0.5, 0.0, 0.5],  # bus - purple
    4: [0.0, 0.0, 1.0],  # car - blue
    5: [0.3, 0.3, 0.0],  # construction_vehicle - olive
    6: [1.0, 0.0, 1.0],  # motorcycle - magenta
    7: [1.0, 1.0, 0.0],  # pedestrian - yellow
    8: [1.0, 0.5, 0.5],  # traffic_cone - light red
    9: [0.5, 0.5, 0.0],  # trailer - mustard
    10: [0.0, 1.0, 0.0],  # truck - green
    11: [0.2, 0.8, 0.8],  # ego_vehicle - cyan
    12: [1.0, 0.8, 0.0],  # traffic_sign - gold
    13: [0.4, 0.4, 0.8],  # other_vehicle - steel blue
    14: [0.0, 0.5, 0.5],  # train - teal
    15: [0.8, 0.8, 0.8],  # Unknown - light gray
    16: [0.0, 1.0, 1.0],  # Background - bright cyan
    # (Assuming this is your FREE_LEARNING_INDEX or a general background for free/unobserved)
}
DEFAULT_COLOR = [0.3, 0.3, 0.3]  # Default color for labels not in map (darker gray)

# Constants for voxel states matching your usage
STATE_UNOBSERVED = 0
STATE_FREE = 1
STATE_OCCUPIED = 2


def transform_points(points_n_features, transform_4x4):
    """
    Transforms points (N, features) using a 4x4 matrix.
    Assumes input points_n_features has shape (N, features), where features >= 3 (x, y, z, ...).
    Outputs transformed points in the same (N, features) format.
    """
    # Debug prints (optional, remove after verification)
    # print(f"-- Inside transform_points --")
    # print(f"Input points shape: {points_n_features.shape}")
    # print(f"Transform matrix shape: {transform_4x4.shape}")

    # Check if there are any points
    if points_n_features.shape[0] == 0:
        # print("Input points array is empty, returning empty array.")
        return points_n_features  # Return empty array if no points

    # --- Core Logic ---
    # Extract XYZ coordinates (N, 3) - Select first 3 COLUMNS
    points_xyz_n3 = points_n_features[:, :3]
    # print(f"Extracted XYZ shape: {points_xyz_n3.shape}")

    # Convert to homogeneous coordinates (N, 4)
    points_homo_n4 = np.hstack((points_xyz_n3, np.ones((points_xyz_n3.shape[0], 1))))
    # print(f"Homogeneous points shape: {points_homo_n4.shape}")

    # Apply transformation: (4, 4) @ (4, N) -> (4, N)
    # Note the transpose of points_homo_n4 before multiplication
    transformed_homo_4n = transform_4x4 @ points_homo_n4.T
    # print(f"Transformed homogeneous shape (before T): {transformed_homo_4n.shape}")

    # Transpose back and extract XYZ: (N, 4) -> (N, 3)
    transformed_xyz_n3 = transformed_homo_4n.T[:, :3]
    # print(f"Transformed XYZ shape: {transformed_xyz_n3.shape}")

    # Combine transformed XYZ with original extra features (if any)
    if points_n_features.shape[1] > 3:  # Check if there were features beyond XYZ (columns > 3)
        # Get the remaining features from the original input
        extra_features = points_n_features[:, 3:]
        # print(f"Extra features shape: {extra_features.shape}")
        # Stack horizontally to keep the (N, features) shape
        transformed_n_features = np.hstack((transformed_xyz_n3, extra_features))
    else:
        # If only XYZ, the transformed XYZ is the result
        transformed_n_features = transformed_xyz_n3

    # print(f"Output transformed features shape: {transformed_n_features.shape}")
    # print(f"-- Exiting transform_points --")
    return transformed_n_features


def transform_matrix_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Arguments:
        x: (n,)
        xp: (m,)
        fp: (4, 4, m)

    Returns:
        y: (4, 4, n)
    """
    # Initialize interpolated transformation matrices
    y = np.repeat(np.eye(4, dtype=fp.dtype)[..., None], x.size, axis=-1)

    # Split homogeneous transformation matrices in rotational and translational part
    rot = fp[:3, :3, :]
    trans = fp[:3, 3, :]

    # Get interpolated rotation matrices
    slerp = Slerp(xp, Rotation.from_matrix(np.moveaxis(rot, -1, 0)))
    y[:3, :3, :] = np.moveaxis(slerp(x).as_matrix(), 0, -1)

    # Get interpolated translation vectors
    y[:3, 3, :] = np.vstack((
        interp1d(xp, trans[0, :])(x),
        interp1d(xp, trans[1, :])(x),
        interp1d(xp, trans[2, :])(x),
    ))

    return y


def transform_pointwise(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Retruns a transformed point cloud

    Point cloud transformation with a transformation matrix for each point.

    Arguments:
        points: Point cloud with dimensions (3, n).
        transforms: Homogeneous transformation matrices with dimesnion (4, 4, n).

    Retruns:
        points: Transformed point cloud with dimension (3, n).
    """
    # Add extra dimesnion to points (3, n) -> (4, n)
    points = np.vstack((points[:3, :], np.ones(points.shape[1], dtype=points.dtype)))

    # Point cloud transformation as 3D dot product
    # T@P^T with dimensions (n, 4, 4) x (n, 1, 4) -> (n, 1, 4)
    points = np.einsum('nij,nkj->nki', np.moveaxis(transforms, -1, 0), points.T[:, None, :])

    # Remove extra dimensions (n, 1, 4) -> (n, 3)
    points = np.squeeze(points)[:, :3]

    return points.T


def get_pointwise_fused_pointcloud(trucksc: TruckScenes, sample: Dict[str, Any], allowed_sensors: List[str]) -> Tuple[
    LidarPointCloud, np.ndarray]:
    """ Returns a fused lidar point cloud for the given sample.

    Fuses the point clouds of the given sample and returns them in the ego
    vehicle frame at the timestamp of the given sample. Uses the timestamps
    of the individual point clouds to transform them to a uniformed frame.

    Does not consider the timestamps of the individual points during the
    fusion.

    Arguments:
        trucksc: TruckScenes dataset instance.
        sample: Reference sample to fuse the point clouds of.

    Returns:
        fused_point_cloud: Fused lidar point cloud in the ego vehicle frame at the
            timestamp of the sample.
        sensor_ids:      numpy array of shape (N_points,) indicating which sensor each point came from
    """
    # Initialize
    points = np.zeros((LidarPointCloud.nbr_dims(), 0), dtype=np.float64)
    timestamps = np.zeros((1, 0), dtype=np.uint64)
    fused_point_cloud = LidarPointCloud(points, timestamps)
    sensor_ids = np.zeros((0,), dtype=int)

    # Get reference ego pose (timestamp of the sample/annotations)
    ref_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])

    # Homogeneous transformation matrix from global to ref ego car frame.
    car_from_global = transform_matrix(ref_ego_pose['translation'],
                                       Quaternion(ref_ego_pose['rotation']),
                                       inverse=True)

    # Iterate over all lidar sensors and fuse their point clouds
    for sensor_idx, sensor in enumerate(allowed_sensors):
        if sensor not in sample['data']:
            print(f"Skipping sensor {sensor} as it is not in sample data.")
            continue
        if 'lidar' not in sensor.lower():
            continue

        # Aggregate current and previous sweeps.
        sd = trucksc.get('sample_data', sample['data'][sensor])

        # Load pointcloud
        pc = LidarPointCloud.from_file(osp.join(trucksc.dataroot, sd['filename']))

        # Get ego pose for the first and last point of the point cloud
        t_min = np.min(pc.timestamps)
        t_max = np.max(pc.timestamps)
        ego_pose_t_min = trucksc.getclosest('ego_pose', t_min)
        ego_pose_t_max = trucksc.getclosest('ego_pose', t_max)

        # Homogeneous transformation matrix from ego car frame to global frame.
        global_from_car_t_min = transform_matrix(ego_pose_t_min['translation'],
                                                 Quaternion(ego_pose_t_min['rotation']),
                                                 inverse=False)

        global_from_car_t_max = transform_matrix(ego_pose_t_max['translation'],
                                                 Quaternion(ego_pose_t_max['rotation']),
                                                 inverse=False)

        globals_from_car = transform_matrix_interp(x=np.squeeze(pc.timestamps),
                                                   xp=np.stack((t_min, t_max)),
                                                   fp=np.dstack((global_from_car_t_min, global_from_car_t_max)))

        # Get sensor calibration information
        cs = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        car_from_current = transform_matrix(cs['translation'],
                                            Quaternion(cs['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        pc.transform(car_from_current)

        pc.points[:3, :] = transform_pointwise(pc.points[:3, :], globals_from_car)

        pc.transform(car_from_global)

        M = pc.points.shape[1]
        sensor_ids = np.hstack((sensor_ids, np.full(M, sensor_idx, dtype=int)))

        # Merge with key pc.
        fused_point_cloud.points = np.hstack((fused_point_cloud.points, pc.points))
        if pc.timestamps is not None:
            fused_point_cloud.timestamps = np.hstack((fused_point_cloud.timestamps, pc.timestamps))

    return fused_point_cloud, sensor_ids


def get_rigid_fused_pointcloud(trucksc: TruckScenes, sample: Dict[str, Any], allowed_sensors: List[str]) -> Tuple[
    LidarPointCloud, np.ndarray]:
    """ Returns a fused lidar point cloud for the given sample.

    Fuses the point clouds of the given sample and returns them in the ego
    vehicle frame at the timestamp of the given sample. Uses the timestamps
    of the individual point clouds to transform them to a uniformed frame.

    Does not consider the timestamps of the individual points during the
    fusion.

    Arguments:
        trucksc: TruckScenes dataset instance.
        sample: Reference sample to fuse the point clouds of.

    Returns:
        fused_point_cloud: Fused lidar point cloud in the ego vehicle frame at the
            timestamp of the sample.
        sensor_ids:      numpy array of shape (N_points,) indicating which sensor each point came from
    """
    # Initialize
    points = np.zeros((LidarPointCloud.nbr_dims(), 0), dtype=np.float64)
    timestamps = np.zeros((1, 0), dtype=np.uint64)
    fused_point_cloud = LidarPointCloud(points, timestamps)
    sensor_ids = np.zeros((0,), dtype=int)

    # Get reference ego pose (timestamp of the sample/annotations)
    ref_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])

    # Homogeneous transformation matrix from global to ref ego car frame.
    car_from_global = transform_matrix(ref_ego_pose['translation'],
                                       Quaternion(ref_ego_pose['rotation']),
                                       inverse=True)

    # Iterate over all lidar sensors and fuse their point clouds
    for sensor_idx, sensor in enumerate(allowed_sensors):
        if sensor not in sample['data']:
            print(f"Skipping sensor {sensor} as it is not in sample data.")
            continue
        if 'lidar' not in sensor.lower():
            continue

        # Aggregate current and previous sweeps.
        sd = trucksc.get('sample_data', sample['data'][sensor])

        # Load pointcloud
        pc = LidarPointCloud.from_file(osp.join(trucksc.dataroot, sd['filename']))

        # Get ego pose (timestamp of the sample data/point cloud)
        sensor_ego_pose = trucksc.getclosest('ego_pose', sd['timestamp'])

        # Homogeneous transformation matrix from ego car frame to global frame.
        global_from_car = transform_matrix(sensor_ego_pose['translation'],
                                           Quaternion(sensor_ego_pose['rotation']),
                                           inverse=False)

        # Get sensor calibration information
        cs = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        car_from_current = transform_matrix(cs['translation'],
                                            Quaternion(cs['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        pc.transform(trans_matrix)

        M = pc.points.shape[1]
        sensor_ids = np.hstack((sensor_ids, np.full(M, sensor_idx, dtype=int)))

        # Merge with key pc.
        fused_point_cloud.points = np.hstack((fused_point_cloud.points, pc.points))
        if pc.timestamps is not None:
            fused_point_cloud.timestamps = np.hstack((fused_point_cloud.timestamps, pc.timestamps))

    return fused_point_cloud, sensor_ids


def get_boxes(trucksc: TruckScenes, sample: Dict[str, Any]) -> List[Box]:
    """ Retruns the bounding boxes of the given sample.

    Arguments:
        trucksc: TruckScenes dataset instance.
        sample: Reference sample to get the boxes from.

    Returns:
        boxes: List of box instances in the ego vehicle frame at the
            timestamp of the sample.
    """
    # Retrieve all sample annotations
    boxes = list(map(trucksc.get_box, sample['anns']))

    # Get reference ego pose (timestamp of the sample/annotations)
    ref_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])

    # Transform boxes to ego frame
    for box in boxes:
        box.translate(-np.array(ref_ego_pose['translation']))
        box.rotate(Quaternion(ref_ego_pose['rotation']).inverse)

    return boxes


# Function to perform poisson surface reconstruction on a given point cloud and returns a mesh representation of the point cloud, along with vertex info
# Inputs pcd: input point cloud,
# depth: parameter to control resolution of mesh. Higher depth results in a more detailed mesh but requires more computation
# n_threads: Number of threads for parallel processing
# min_density: threshold for removing low-density vertices from generated mesh. Used to clean up noisy or sparse areas
def run_poisson(pcd, depth, n_threads, min_density=None):
    # creates triangular mesh form pcd using poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=n_threads
    )
    # returns mesh and densities: list of density values corresponding to each vertex in the mesh. Density indicates how well a vertex is supported by underlying points

    # Post-process the mesh
    # Purpose: to clean up the mesh by removing low-density vertices (e.g. noise or poorly supported areas)
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)  # min_density should be between 0 and 1
        mesh.remove_vertices_by_mask(vertices_to_remove)  # removes vertices where density is below threshold
    mesh.compute_vertex_normals()  # computes the normals of the vertices

    return mesh, densities


# Function that creates a 3D mesh from a given point cloud or a buffer of points
# Inputs buffer: a list of point clouds that are combined if no original is given
# depth: resolution for poisson surface reconstruction
# n_threads: Number of threads for parallel processing
# min_density: Optional threshold for removing low-density vertices
# point_cloud_original: provides a preprocessed point cloud, if given buffer is ignored
def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original=None):
    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(
            buffer)  # Calls buffer_to_pointcloud(buffer) to create a combined point cloud from the list of smaller point clouds
    else:
        pcd = point_cloud_original  # Uses the given point cloud directly

    return run_poisson(pcd, depth, n_threads, min_density)  # calls run_poisson function to generate mesh


# Function to combine multiple point clouds from a list (buffer) into a single point cloud and optionally estimating normals in the resulting point cloud
# Input: buffer: a list of individual point clouds (each being an instance of open3d.geometry.PointCloud)
# compute_normals: boolean flag that if set to True estimates normals of the final point cloud
def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()  # Initialize empty point cloud object using Open3d
    for cloud in buffer:
        pcd += cloud  # combine each point cloud with current point cloud object
    if compute_normals:
        pcd.estimate_normals()  # estimate normals for each point in the combined point cloud

    return pcd


# Function to preprocess a given point cloud by estimating and orienting the normals of each point
# Input: pcd: input point cloud
# max_nn: maximum number of nearest neighbors to use when estimating normals with default 20
# normals: boolean flag whether to estimate normals
def preprocess_cloud(
        pcd,
        max_nn=20,
        normals=None,
):
    cloud = deepcopy(
        pcd)  # create a deep copy of the input point cloud to ensure that original point cloud is not modified
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)  # estimate normals based on nearest max_nn points
        cloud.orient_normals_towards_camera_location()  # orients all computed normals to point towards the camera location to get consistent normal direction for visualization and mesh generation

    return cloud


# Wrapper function that calls preprocess_cloud function with parameters derived from a config file
# Input: pcd: input point cloud
# config: Dictionary containing all configuration parameters loaded from a YAML file
def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )


def icp_align(source_np, target_np, init_trans=np.eye(4), voxel_size=0.2):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_np)
    target.points = o3d.utility.Vector3dVector(target_np)

    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)

    source.estimate_normals()
    target.estimate_normals()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)

    reg = o3d.pipelines.registration.registration_icp(
        source, target, 0.5, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria
    )
    return reg.transformation


def denoise_pointcloud(pcd: o3d.geometry.PointCloud, filter_mode: str, config: dict,
                       location_msg: str = "point cloud") -> o3d.geometry.PointCloud:
    """
    Applies noise filtering to the given point cloud using the specified method.

    Args:
        pcd: Open3D point cloud.
        filter_mode: One of 'none', 'sor', 'ror', 'both'.
        config: Dictionary from config.yaml with noise filtering parameters.

    Returns:
        Filtered point cloud.
    """

    initial_pcd = pcd  # Keep reference to original
    initial_count = np.asarray(initial_pcd.points).shape[0]
    kept_indices = np.arange(initial_count)  # Start with all indices
    filtered_pcd = initial_pcd

    if initial_count == 0:  # Handle empty input cloud
        print(f"Skipping filtering at '{location_msg}' on empty input.")
        return filtered_pcd, kept_indices

    if filter_mode == 'none':  # Explicitly handle 'none' case
        return filtered_pcd, kept_indices

    try:  # Add error handling for filtering operations
        if filter_mode == 'sor':
            filtered_pcd, ind = initial_pcd.remove_statistical_outlier(
                nb_neighbors=config['sor_nb_neighbors'],
                std_ratio=config['sor_std_ratio']
            )
            kept_indices = np.array(ind)
        elif filter_mode == 'ror':
            filtered_pcd, ind = initial_pcd.remove_radius_outlier(
                nb_points=config['ror_nb_points'],
                radius=config['ror_radius']
            )
            kept_indices = np.array(ind)
        elif filter_mode == 'both':
            sor_filtered_pcd, sor_ind = initial_pcd.remove_statistical_outlier(
                nb_neighbors=config['sor_nb_neighbors'],
                std_ratio=config['sor_std_ratio']
            )
            if np.asarray(sor_filtered_pcd.points).shape[0] > 0:
                filtered_pcd, ror_ind = sor_filtered_pcd.remove_radius_outlier(
                    nb_points=config['ror_nb_points'],
                    radius=config['ror_radius']
                )
                kept_indices = np.array(sor_ind)[ror_ind]
            else:
                filtered_pcd = sor_filtered_pcd
                kept_indices = np.array([], dtype=int)

        final_count = np.asarray(filtered_pcd.points).shape[0]
        # --- 2. REMOVED internal location_msg = "point cloud" ---

        # --- 3. Use the passed location_msg argument here ---
        print(f"Filtering {location_msg} with filter mode {filter_mode}. Reduced from {initial_count} to {final_count} points.")

    except Exception as e:
        print(f"Error during filtering ({filter_mode}) at {location_msg}: {e}. Returning original.")
        filtered_pcd = initial_pcd
        kept_indices = np.arange(initial_count)

        # Ensure return type matches definition
    return filtered_pcd, kept_indices


def in_range_mask(points, pc_range):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    return (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )


def visualize_pointcloud(points, colors=None, title="Point Cloud"):
    """
    Visualize a point cloud using Open3D.
    Args:
        points: Nx3 or Nx4 numpy array of XYZ[+label/feature].
        colors: Optional Nx3 RGB array or a string-based colormap (e.g., "label").
        title: Optional window title.
    """
    if points.shape[1] < 3:
        print("Invalid point cloud shape:", points.shape)
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    if colors is not None:
        if isinstance(colors, str) and colors == "label" and points.shape[1] > 3:
            labels = points[:, 3].astype(int)
            max_label = labels.max() + 1
            cmap = plt.get_cmap("tab20", max_label)
            rgb = cmap(labels)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        elif isinstance(colors, np.ndarray) and colors.shape == points[:, :3].shape:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    elif points.shape[1] > 3:
        # Use label to colorize by default
        labels = points[:, 3].astype(int)
        max_label = labels.max() + 1
        cmap = plt.get_cmap("tab20", max_label)
        rgb = cmap(labels)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    print(f"Visualizing point cloud with {np.asarray(pcd.points).shape[0]} points")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_pointcloud_bbox(points: np.ndarray,
                              boxes: Optional[List] = None,  # Use List[Box] if Box class is imported
                              colors: Optional[Union[np.ndarray, str]] = None,
                              title: str = "Point Cloud with BBoxes",
                              self_vehicle_range: Optional[List[float]] = None,  # New parameter
                              vis_self_vehicle: bool = False):
    """
    Visualize a point cloud and optional bounding boxes using Open3D.

    Args:
        points: Nx3 or Nx(>3) numpy array of XYZ[+label/feature].
        boxes: List of Box objects (e.g., from truckscenes) in the same coordinate frame as points.
               Assumes Box objects have .center, .wlh, and .orientation (pyquaternion.Quaternion) attributes.
        colors: Optional Nx3 RGB array or a string-based colormap (e.g., "label").
                If "label", assumes the 4th column of `points` contains integer labels.
        title: Optional window title.
        self_vehicle_range: Optional list [x_min, y_min, z_min, x_max, y_max, z_max] for the ego vehicle box.
        vis_self_vehicle: If True and self_vehicle_range is provided, draws the ego vehicle box.
    """

    geometries = []

    # --- Point cloud ---
    if points.ndim != 2 or points.shape[1] < 3:
        print(f"Error: Invalid point cloud shape: {points.shape}. Needs Nx(>=3).")
        return
    if points.shape[0] == 0:
        print("Warning: Point cloud is empty.")
        # Continue to potentially draw boxes

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # --- Point Cloud Coloring ---
    use_label_coloring = False
    if colors is not None:
        if isinstance(colors, str) and colors.lower() == "label":
            if points.shape[1] > 3:
                use_label_coloring = True
            else:
                print("Warning: 'colors' set to 'label' but points array has < 4 columns.")
        elif isinstance(colors, np.ndarray) and colors.shape == points[:, :3].shape:
            # Ensure colors are float64 and in range [0, 1] for Open3D
            colors_float = colors.astype(np.float64)
            if np.max(colors_float) > 1.0:  # Basic check if maybe 0-255 range
                colors_float /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_float, 0.0, 1.0))
        else:
            print(
                f"Warning: Invalid 'colors' argument. Type: {type(colors)}, Value/Shape: {colors if isinstance(colors, str) else colors.shape}. Using default colors.")
    elif points.shape[1] > 3:  # Default to label coloring if 4th column exists and colors=None
        print("Info: No 'colors' provided, attempting to color by 4th column (label).")
        use_label_coloring = True

    if use_label_coloring:
        try:
            labels = points[:, 3].astype(int)
            unique_labels = np.unique(labels)
            if unique_labels.size > 0:
                # Map labels to colors
                min_label = unique_labels.min()
                max_label = unique_labels.max()
                label_range = max_label - min_label + 1
                # Use a colormap suitable for categorical data
                cmap = plt.get_cmap("tab20", label_range)
                # Normalize labels to 0..label_range-1 for colormap indexing
                normalized_labels = labels - min_label
                rgb = cmap(normalized_labels)[:, :3]  # Get RGB, ignore alpha
                pcd.colors = o3d.utility.Vector3dVector(rgb)
            else:
                print("Warning: Found label column, but no unique labels detected.")
        except Exception as e:
            print(f"Error applying label coloring: {e}. Using default colors.")
            use_label_coloring = False  # Revert if error occurred

    geometries.append(pcd)

    # --- Ego Vehicle Bounding Box ---
    if vis_self_vehicle and self_vehicle_range is not None:
        if len(self_vehicle_range) == 6:
            x_min_s, y_min_s, z_min_s, x_max_s, y_max_s, z_max_s = self_vehicle_range
            center_s = np.array([(x_min_s + x_max_s) / 2.0,
                                 (y_min_s + y_max_s) / 2.0,
                                 (z_min_s + z_max_s) / 2.0])
            # Open3D extent is [length(x), width(y), height(z)]
            extent_s = np.array([x_max_s - x_min_s,
                                 y_max_s - y_min_s,
                                 z_max_s - z_min_s])
            R_s = np.eye(3)  # Ego vehicle box is axis-aligned in its own coordinate frame
            ego_obb = o3d.geometry.OrientedBoundingBox(center_s, R_s, extent_s)
            ego_obb.color = (0.0, 0.8, 0.2)  # Green color for ego vehicle
            geometries.append(ego_obb)
        else:
            print(
                f"Warning: self_vehicle_range provided for ego vehicle but not of length 6. Got: {self_vehicle_range}")

    # --- Other Bounding boxes (Annotations) ---
    num_boxes_drawn = 0
    if boxes is not None:
        for i, box in enumerate(boxes):
            try:
                # Create Open3D OrientedBoundingBox from truckscenes Box properties
                center = box.center  # Should be numpy array (3,)
                # truckscenes Box.wlh = [width(y), length(x), height(z)]
                # o3d OrientedBoundingBox extent = [length(x), width(y), height(z)]
                extent = np.array([box.wlh[1], box.wlh[0], box.wlh[2]])
                # Get rotation matrix from pyquaternion.Quaternion
                R = box.orientation.rotation_matrix  # Should be 3x3 numpy array

                obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
                obb.color = (1.0, 0.0, 0.0)  # Set color to red
                geometries.append(obb)
                num_boxes_drawn += 1

            except AttributeError as e:
                print(
                    f"Error processing box {i} (Token: {getattr(box, 'token', 'N/A')}): Missing attribute {e}. Skipping box.")
            except Exception as e:
                print(f"Error processing box {i} (Token: {getattr(box, 'token', 'N/A')}): {e}. Skipping box.")

    # --- Visualize ---
    if not geometries:
        print("No geometries (point cloud or boxes) to visualize.")
        return

    point_count = np.asarray(pcd.points).shape[0]
    num_ego_box = 1 if (vis_self_vehicle and self_vehicle_range is not None and len(self_vehicle_range) == 6) else 0
    print(
        f"Visualizing point cloud with {point_count} points, {num_boxes_drawn} annotation boxes, and {num_ego_box} ego vehicle box.")
    o3d.visualization.draw_geometries(geometries, window_name=title)


def ray_casting(ray_start, ray_end, pc_range, voxel_size, spatial_shape, EPS=1e-9, DISTANCE=0.5):
    """
    3-D DDA / Amanatides–Woo ray casting.
    Returns a list of integer 3-tuples (i,j,k) of all voxels traversed by the ray.
    """
    # shift into voxel grid coords
    new_start = ray_start[:3] - pc_range[:3]
    new_end = ray_end[:3] - pc_range[:3]

    ray = new_end - new_start
    step = np.sign(ray).astype(int)
    tDelta = np.empty(3, float)
    cur_voxel = np.empty(3, int)
    last_voxel = np.empty(3, int)
    tMax = np.empty(3, float)

    # init
    for k in range(3):
        if ray[k] != 0:
            tDelta[k] = (step[k] * voxel_size[k]) / ray[k]
        else:
            tDelta[k] = np.finfo(float).max

        # nudge start/end inside to avoid boundary cases
        new_start[k] += step[k] * voxel_size[k] * EPS
        new_end[k] -= step[k] * voxel_size[k] * EPS

        cur_voxel[k] = int(np.floor(new_start[k] / voxel_size[k]))
        last_voxel[k] = int(np.floor(new_end[k] / voxel_size[k]))

    # compute initial tMax
    for k in range(3):
        if ray[k] != 0:
            # boundary coordinate
            coord = cur_voxel[k] * voxel_size[k]
            if step[k] < 0 and coord < new_start[k]:
                boundary = coord
            else:
                boundary = coord + step[k] * voxel_size[k]
            tMax[k] = (boundary - new_start[k]) / ray[k]
        else:
            tMax[k] = np.finfo(float).max

    visited = []
    # traverse until we've gone past last_voxel in any dimension
    while np.all(step * (cur_voxel - last_voxel) < DISTANCE):
        # record
        visited.append(tuple(cur_voxel.copy()))
        # step to next voxel
        # pick axis with smallest tMax
        m = np.argmin(tMax)
        cur_voxel[m] += step[m]
        if not (0 <= cur_voxel[m] < spatial_shape[m]):
            break
        tMax[m] += tDelta[m]
    return visited


def calculate_lidar_visibility(points, points_origin, points_label,
                               pc_range, voxel_size, spatial_shape, occupancy_grid, FREE_LEARNING_INDEX,
                               points_sensor_indices,
                               sensor_max_ranges):
    """
    points:        (N,3) array of LiDAR hits
    points_origin:(N,3) corresponding sensor origins
    points_label:  (N,) integer semantic labels per point
    Returns:
      voxel_state: (H,W,Z) 0=NOT_OBS,1=FREE,2=OCC
      voxel_label: (H,W,Z) semantic label (FREE_LABEL if no hit)
    """
    NOT_OBS, FREE, OCC = STATE_UNOBSERVED, STATE_FREE, STATE_OCCUPIED

    voxel_occ_count = np.zeros(spatial_shape, int)
    voxel_free_count = np.zeros(spatial_shape, int)
    voxel_label = np.full(spatial_shape, FREE_LEARNING_INDEX, int)

    # for each LiDAR point
    for i in tqdm(range(points.shape[0]), desc='Processing lidar points...'):
        start = points_origin[i]
        end = points[i]
        # direct hit voxel
        actual_hit_voxel_indices = ((end - pc_range[:3]) / voxel_size).astype(int)
        if np.all((0 <= actual_hit_voxel_indices) & (actual_hit_voxel_indices < spatial_shape)):
            voxel_occ_count[tuple(actual_hit_voxel_indices)] += 1
            voxel_label[tuple(actual_hit_voxel_indices)] = int(points_label[i])
        # walk the ray up to the point
        sensor_idx_of_point = points_sensor_indices[i]
        max_range_for_this_sensor = sensor_max_ranges[sensor_idx_of_point]
        current_distance_to_hit = np.linalg.norm(end - start)

        if current_distance_to_hit <= max_range_for_this_sensor:
            for vox_tuple in ray_casting(start, end, pc_range, voxel_size, spatial_shape):
                if np.array_equal(np.array(vox_tuple), actual_hit_voxel_indices):
                    continue

                occupancy_grid_value = occupancy_grid[vox_tuple]
                if occupancy_grid_value != FREE_LEARNING_INDEX:
                    break
                else:
                    voxel_free_count[vox_tuple] += 1

    # build state mask
    voxel_state = np.full(spatial_shape, NOT_OBS, int)
    voxel_state[voxel_free_count > 0] = FREE
    voxel_state[voxel_occ_count > 0] = OCC
    return voxel_state, voxel_label


# --- Numba CUDA Device Function for Ray Casting Steps ---
@cuda.jit(device=True)
def _ray_casting_gpu_step_logic(
        # Inputs for one ray
        ray_start_x, ray_start_y, ray_start_z,  # Origin of this ray (sensor)
        ray_end_x, ray_end_y, ray_end_z,  # Target of this ray (LiDAR hit)
        # Grid parameters
        pc_range_min_x, pc_range_min_y, pc_range_min_z,
        voxel_sx, voxel_sy, voxel_sz,  # Voxel sizes
        grid_dx, grid_dy, grid_dz,  # Grid dimensions in voxels
        # Pre-computed occupancy for early exit
        occupancy_grid_gpu,  # Read-only, shows where aggregated matter is
        FREE_LEARNING_INDEX_CONST,  # Make sure this is the correct constant name
        # Output array to update
        voxel_free_count_gpu,  # This will be updated atomically
        EPS, DISTANCE  # Constants from your ray_casting
):
    # --- Inline DDA logic from your ray_casting function ---

    new_start_x = ray_start_x - pc_range_min_x
    new_start_y = ray_start_y - pc_range_min_y
    new_start_z = ray_start_z - pc_range_min_z

    new_end_x = ray_end_x - pc_range_min_x
    new_end_y = ray_end_y - pc_range_min_y
    new_end_z = ray_end_z - pc_range_min_z

    ray_vx = new_end_x - new_start_x
    ray_vy = new_end_y - new_start_y
    ray_vz = new_end_z - new_start_z

    step_ix, step_iy, step_iz = 0, 0, 0
    if ray_vx > 0:
        step_ix = 1
    elif ray_vx < 0:
        step_ix = -1
    if ray_vy > 0:
        step_iy = 1
    elif ray_vy < 0:
        step_iy = -1
    if ray_vz > 0:
        step_iz = 1
    elif ray_vz < 0:
        step_iz = -1

    t_delta_x = float('inf')
    if ray_vx != 0: t_delta_x = (step_ix * voxel_sx) / ray_vx
    t_delta_y = float('inf')
    if ray_vy != 0: t_delta_y = (step_iy * voxel_sy) / ray_vy
    t_delta_z = float('inf')
    if ray_vz != 0: t_delta_z = (step_iz * voxel_sz) / ray_vz

    # Nudge
    adj_start_x = new_start_x + step_ix * voxel_sx * EPS
    adj_start_y = new_start_y + step_iy * voxel_sy * EPS
    adj_start_z = new_start_z + step_iz * voxel_sz * EPS

    adj_end_x = new_end_x - step_ix * voxel_sx * EPS
    adj_end_y = new_end_y - step_iy * voxel_sy * EPS
    adj_end_z = new_end_z - step_iz * voxel_sz * EPS

    cur_vox_ix = int(math.floor(adj_start_x / voxel_sx))
    cur_vox_iy = int(math.floor(adj_start_y / voxel_sy))
    cur_vox_iz = int(math.floor(adj_start_z / voxel_sz))

    last_vox_ix = int(math.floor(adj_end_x / voxel_sx))
    last_vox_iy = int(math.floor(adj_end_y / voxel_sy))
    last_vox_iz = int(math.floor(adj_end_z / voxel_sz))

    t_max_x = float('inf')
    if ray_vx != 0:
        coord_x = float(cur_vox_ix * voxel_sx)
        boundary_x = coord_x + step_ix * voxel_sx if not (step_ix < 0 and coord_x < adj_start_x) else coord_x
        t_max_x = (boundary_x - adj_start_x) / ray_vx

    t_max_y = float('inf')
    if ray_vy != 0:
        coord_y = float(cur_vox_iy * voxel_sy)
        boundary_y = coord_y + step_iy * voxel_sy if not (step_iy < 0 and coord_y < adj_start_y) else coord_y
        t_max_y = (boundary_y - adj_start_y) / ray_vy

    t_max_z = float('inf')
    if ray_vz != 0:
        coord_z = float(cur_vox_iz * voxel_sz)
        boundary_z = coord_z + step_iz * voxel_sz if not (step_iz < 0 and coord_z < adj_start_z) else coord_z
        t_max_z = (boundary_z - adj_start_z) / ray_vz

    max_iterations = grid_dx + grid_dy + grid_dz + 3  # Max iterations for safety

    for _ in range(max_iterations):
        term_x = True if step_ix == 0 else (step_ix * (cur_vox_ix - last_vox_ix) >= DISTANCE)
        term_y = True if step_iy == 0 else (step_iy * (cur_vox_iy - last_vox_iy) >= DISTANCE)
        term_z = True if step_iz == 0 else (step_iz * (cur_vox_iz - last_vox_iz) >= DISTANCE)
        if term_x and term_y and term_z:
            break

        actual_hit_vx = int(math.floor((ray_end_x - pc_range_min_x) / voxel_sx))
        actual_hit_vy = int(math.floor((ray_end_y - pc_range_min_y) / voxel_sy))
        actual_hit_vz = int(math.floor((ray_end_z - pc_range_min_z) / voxel_sz))

        is_current_voxel_the_actual_hit = (cur_vox_ix == actual_hit_vx and \
                                           cur_vox_iy == actual_hit_vy and \
                                           cur_vox_iz == actual_hit_vz)

        if not is_current_voxel_the_actual_hit:
            if (0 <= cur_vox_ix < grid_dx and
                    0 <= cur_vox_iy < grid_dy and
                    0 <= cur_vox_iz < grid_dz):

                if occupancy_grid_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz] != FREE_LEARNING_INDEX_CONST:
                    return  # Ray hit an obstruction
                else:
                    cuda.atomic.add(voxel_free_count_gpu, (cur_vox_ix, cur_vox_iy, cur_vox_iz), 1)
            else:  # Current voxel out of bounds
                return

        # Step to next voxel
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                cur_vox_ix += step_ix
                if not (0 <= cur_vox_ix < grid_dx): return
                t_max_x += t_delta_x
            else:
                cur_vox_iz += step_iz
                if not (0 <= cur_vox_iz < grid_dz): return
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                cur_vox_iy += step_iy
                if not (0 <= cur_vox_iy < grid_dy): return
                t_max_y += t_delta_y
            else:
                cur_vox_iz += step_iz
                if not (0 <= cur_vox_iz < grid_dz): return
                t_max_z += t_delta_z


# --- Main Numba CUDA Kernel ---
@cuda.jit
def visibility_kernel(
        points_gpu,  # (N,3) LiDAR hits
        points_origin_gpu,  # (N,3) Sensor origins
        points_label_gpu,  # (N,) Semantic labels for hits
        pc_range_min_gpu,  # (3,) [xmin, ymin, zmin] - expecting np.array
        voxel_size_gpu,  # (3,) [vx, vy, vz] - expecting np.array
        spatial_shape_gpu,  # (3,) [Dx, Dy, Dz] - expecting np.array (int32)
        occupancy_grid_gpu,  # (Dx,Dy,Dz) Pre-computed, read-only (uint8)
        FREE_LEARNING_INDEX_CONST_UINT8,  # Scalar (uint8)
        voxel_occ_count_out_gpu,  # (Dx,Dy,Dz) for writing (int32)
        voxel_free_count_out_gpu,  # (Dx,Dy,Dz) for writing (int32)
        voxel_label_out_gpu,  # (Dx,Dy,Dz) for writing (int32)
        FREE_LABEL_CONST_FOR_INIT_INT32,  # Scalar (int32, e.g. -1)
        EPS_CONST, DISTANCE_CONST,  # Scalars (float64)
        points_sensor_indices_gpu,
        sensor_max_ranges_gpu
):
    i = cuda.grid(1)
    if i >= points_gpu.shape[0]:
        return

    ray_start_x = points_origin_gpu[i, 0]
    ray_start_y = points_origin_gpu[i, 1]
    ray_start_z = points_origin_gpu[i, 2]

    ray_end_x = points_gpu[i, 0]
    ray_end_y = points_gpu[i, 1]
    ray_end_z = points_gpu[i, 2]

    point_label = points_label_gpu[i]  # This is int32

    # Grid parameters from input arrays
    pc_min_x, pc_min_y, pc_min_z = pc_range_min_gpu[0], pc_range_min_gpu[1], pc_range_min_gpu[2]
    voxel_sx, voxel_sy, voxel_sz = voxel_size_gpu[0], voxel_size_gpu[1], voxel_size_gpu[2]
    grid_dx, grid_dy, grid_dz = spatial_shape_gpu[0], spatial_shape_gpu[1], spatial_shape_gpu[2]

    # --- 1. Mark Occupied Voxel (for the actual LiDAR hit 'ray_end') ---
    actual_hit_vx = int(math.floor((ray_end_x - pc_min_x) / voxel_sx))
    actual_hit_vy = int(math.floor((ray_end_y - pc_min_y) / voxel_sy))
    actual_hit_vz = int(math.floor((ray_end_z - pc_min_z) / voxel_sz))

    if (0 <= actual_hit_vx < grid_dx and
            0 <= actual_hit_vy < grid_dy and
            0 <= actual_hit_vz < grid_dz):
        cuda.atomic.add(voxel_occ_count_out_gpu, (actual_hit_vx, actual_hit_vy, actual_hit_vz), 1)
        voxel_label_out_gpu[actual_hit_vx, actual_hit_vy, actual_hit_vz] = point_label  # Direct write (last wins)

    sensor_idx_of_point = points_sensor_indices_gpu[i]
    max_range_for_this_sensor = sensor_max_ranges_gpu[sensor_idx_of_point]

    # Calculate squared distance to avoid sqrt in kernel if possible, or just use norm if math.sqrt is acceptable
    dx = ray_end_x - ray_start_x
    dy = ray_end_y - ray_start_y
    dz = ray_end_z - ray_start_z
    distance_sq_to_hit = dx * dx + dy * dy + dz * dz

    # --- 2. Perform Ray Casting ---
    if distance_sq_to_hit <= max_range_for_this_sensor * max_range_for_this_sensor:
        _ray_casting_gpu_step_logic(
            ray_start_x, ray_start_y, ray_start_z,
            ray_end_x, ray_end_y, ray_end_z,
            pc_min_x, pc_min_y, pc_min_z,
            voxel_sx, voxel_sy, voxel_sz,
            grid_dx, grid_dy, grid_dz,
            occupancy_grid_gpu,
            FREE_LEARNING_INDEX_CONST_UINT8,  # Pass the constant for comparison
            voxel_free_count_out_gpu,
            EPS_CONST, DISTANCE_CONST
        )


# --- Host Function to Manage GPU Execution ---
def calculate_lidar_visibility_gpu_host(
        points_cpu, points_origin_cpu, points_label_cpu,
        pc_range_cpu_list,  # Original list [xmin,ymin,zmin,xmax,ymax,zmax]
        voxel_size_cpu_scalar,  # Original scalar voxel size
        spatial_shape_cpu_list,  # Original list [Dx,Dy,Dz]
        occupancy_grid_cpu,  # (Dx,Dy,Dz) np.uint8
        FREE_LEARNING_INDEX_cpu,  # scalar int/uint8 for free space semantic label
        FREE_LABEL_placeholder_cpu,  # scalar int (e.g., -1 for internal init)
        points_sensor_indices_cpu: np.ndarray,
        sensor_max_ranges_cpu: np.ndarray
):
    num_points = points_cpu.shape[0]
    if num_points == 0:
        voxel_state = np.full(tuple(spatial_shape_cpu_list), STATE_UNOBSERVED, dtype=np.uint8)
        voxel_label = np.full(tuple(spatial_shape_cpu_list), FREE_LEARNING_INDEX_cpu, dtype=np.uint8)
        return voxel_state, voxel_label

    # Prepare data for GPU (ensure contiguous and correct types)
    points_gpu_data = cuda.to_device(np.ascontiguousarray(points_cpu, dtype=np.float64))
    points_origin_gpu_data = cuda.to_device(np.ascontiguousarray(points_origin_cpu, dtype=np.float64))
    points_label_gpu_data = cuda.to_device(np.ascontiguousarray(points_label_cpu, dtype=np.int32))

    pc_range_min_gpu_data = cuda.to_device(np.ascontiguousarray(pc_range_cpu_list[:3], dtype=np.float64))
    voxel_size_gpu_data = cuda.to_device(np.array([voxel_size_cpu_scalar] * 3, dtype=np.float64))
    spatial_shape_gpu_data = cuda.to_device(np.array(spatial_shape_cpu_list, dtype=np.int32))

    occupancy_grid_gpu_data = cuda.to_device(np.ascontiguousarray(occupancy_grid_cpu, dtype=np.uint8))

    points_sensor_indices_gpu_data = cuda.to_device(
        np.ascontiguousarray(points_sensor_indices_cpu, dtype=np.int32))
    sensor_max_ranges_gpu_data = cuda.to_device(np.ascontiguousarray(sensor_max_ranges_cpu, dtype=np.float32))

    # Output arrays on GPU
    voxel_occ_count_gpu = cuda.to_device(np.zeros(tuple(spatial_shape_cpu_list), dtype=np.int32))
    voxel_free_count_gpu = cuda.to_device(np.zeros(tuple(spatial_shape_cpu_list), dtype=np.int32))
    voxel_label_out_gpu = cuda.to_device(
        np.full(tuple(spatial_shape_cpu_list), np.int32(FREE_LABEL_placeholder_cpu), dtype=np.int32))

    # Kernel launch configuration
    threads_per_block = 256
    blocks_per_grid = (num_points + (threads_per_block - 1)) // threads_per_block

    EPS_CONST_val = 1e-9  # Standard DDA constant
    DISTANCE_CONST_val = 0.5  # Standard DDA constant

    print(f"Launching GPU kernel: {blocks_per_grid} blocks, {threads_per_block} threads/block for {num_points} points.")
    visibility_kernel[blocks_per_grid, threads_per_block](
        points_gpu_data, points_origin_gpu_data, points_label_gpu_data,
        pc_range_min_gpu_data, voxel_size_gpu_data, spatial_shape_gpu_data,
        occupancy_grid_gpu_data,
        np.uint8(FREE_LEARNING_INDEX_cpu),  # Pass as uint8 for comparison with occupancy_grid_gpu
        voxel_occ_count_gpu, voxel_free_count_gpu, voxel_label_out_gpu,
        np.int32(FREE_LABEL_placeholder_cpu),  # For initializing voxel_label_out_gpu
        EPS_CONST_val, DISTANCE_CONST_val,
        points_sensor_indices_gpu_data,
        sensor_max_ranges_gpu_data
    )
    cuda.synchronize()

    # Copy results back to CPU
    voxel_occ_count_cpu = voxel_occ_count_gpu.copy_to_host()
    voxel_free_count_cpu = voxel_free_count_gpu.copy_to_host()
    voxel_label_from_gpu_cpu = voxel_label_out_gpu.copy_to_host()  # This is int32

    # Final state assignment (on CPU)
    final_voxel_states = np.full(tuple(spatial_shape_cpu_list), STATE_UNOBSERVED, dtype=np.uint8)
    final_voxel_states[voxel_free_count_cpu > 0] = STATE_FREE
    final_voxel_states[voxel_occ_count_cpu > 0] = STATE_OCCUPIED

    # Create final semantic labels grid
    final_semantic_labels = np.full(tuple(spatial_shape_cpu_list), FREE_LEARNING_INDEX_cpu, dtype=np.uint8)
    # Populate labels for occupied voxels
    occupied_mask = (final_voxel_states == STATE_OCCUPIED)
    # voxel_label_from_gpu_cpu contains actual semantic labels for occupied cells,
    # and FREE_LABEL_placeholder_cpu for others.
    final_semantic_labels[occupied_mask] = voxel_label_from_gpu_cpu[occupied_mask].astype(np.uint8)

    print("GPU visibility calculation finished.")
    return final_voxel_states, final_semantic_labels


# --- Helper function to get camera parameters ---
def get_camera_parameters(trucksc: TruckScenes, sample_data_token: str, ego_pose_timestamp: int):
    """
    Retrieves and transforms camera parameters to the current ego vehicle frame.
    """
    sd = trucksc.get('sample_data', sample_data_token)
    cs = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])

    # Camera intrinsics (K)
    cam_intrinsics = np.array(cs['camera_intrinsic'])

    # Transformation: camera frame -> ego vehicle frame AT THE TIMESTAMP OF THE CAMERA IMAGE
    # This is P_cam2ego in the paper's notation, specific to this camera's capture time.
    cam_extrinsic_translation = np.array(cs['translation'])
    cam_extrinsic_rotation = Quaternion(cs['rotation'])
    T_ego_at_cam_timestamp_from_cam = transform_matrix(
        cam_extrinsic_translation, cam_extrinsic_rotation, inverse=False
    )

    # If the camera's timestamp differs from the reference ego_pose_timestamp,
    # we need to bring the camera pose into the *current* ego frame.
    # Current ego pose (at ego_pose_timestamp, e.g., LiDAR keyframe time)
    current_ego_pose_rec = trucksc.getclosest('ego_pose', ego_pose_timestamp)
    T_global_from_current_ego = transform_matrix(
        current_ego_pose_rec['translation'], Quaternion(current_ego_pose_rec['rotation']), inverse=False
    )
    T_current_ego_from_global = np.linalg.inv(T_global_from_current_ego)

    # Ego pose at the camera's capture time
    cam_timestamp_ego_pose_rec = trucksc.getclosest('ego_pose', sd['timestamp'])
    T_global_from_ego_at_cam_timestamp = transform_matrix(
        cam_timestamp_ego_pose_rec['translation'], Quaternion(cam_timestamp_ego_pose_rec['rotation']), inverse=False
    )

    # Final transformation: camera frame -> current_ego_frame
    # P_current_ego_from_cam = P_current_ego_from_global @ P_global_from_ego_at_cam_timestamp @ P_ego_at_cam_timestamp_from_cam
    T_current_ego_from_cam = T_current_ego_from_global @ T_global_from_ego_at_cam_timestamp @ T_ego_at_cam_timestamp_from_cam

    # Camera origin in the current ego frame
    cam_origin_in_current_ego = T_current_ego_from_cam[:3, 3]

    # Rotation part for transforming ray directions
    R_current_ego_from_cam = T_current_ego_from_cam[:3, :3]

    return {
        'intrinsics': cam_intrinsics,  # 3x3 K matrix
        'T_current_ego_from_cam': T_current_ego_from_cam,  # 4x4 matrix
        'origin_in_current_ego': cam_origin_in_current_ego,  # (3,)
        'R_current_ego_from_cam': R_current_ego_from_cam,  # (3,3)
        'width': sd['width'],
        'height': sd['height']
    }


# --- CPU Function for Camera Visibility (Algorithm 3 from Occ3D) ---
def calculate_camera_visibility_cpu(
        # Inputs based on Algorithm 3 and practical needs
        trucksc: TruckScenes,
        current_sample_token: str,  # To get camera data for the current keyframe
        lidar_voxel_state: np.ndarray,  # (Dx,Dy,Dz) - output from LiDAR visibility (0=UNOBS, 1=FREE, 2=OCC)
        pc_range_params: list,  # [xmin,ymin,zmin,xmax,ymax,zmax]
        voxel_size_params: np.ndarray,  # [vx,vy,vz]
        spatial_shape_params: np.ndarray,  # [Dx,Dy,Dz]
        camera_names: List[str],  # List of camera sensor names to use
        DEPTH_MAX: float = 100.0
):
    print("Calculating Camera Visibility (CPU)...")

    # Output camera visibility mask: 1 if observed by any camera (and LiDAR), 0 otherwise
    # Initialize to 0 (unobserved by camera)
    camera_visibility_mask = np.zeros(spatial_shape_params, dtype=np.uint8)

    # Get timestamp of the current sample (e.g., the keyframe for which we do this)
    current_sample_rec = trucksc.get('sample', current_sample_token)
    current_ego_pose_ts = current_sample_rec['timestamp']

    # Iterate over each camera specified
    for cam_name in tqdm(camera_names, desc="Processing Cameras"):
        if cam_name not in current_sample_rec['data']:
            print(f"Warning: Camera {cam_name} not found in sample data for token {current_sample_token}. Skipping.")
            continue

        cam_sample_data_token = current_sample_rec['data'][cam_name]
        cam_params = get_camera_parameters(trucksc, cam_sample_data_token, current_ego_pose_ts)

        K = cam_params['intrinsics']  # 3x3
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        cam_origin_ego = cam_params['origin_in_current_ego']  # (3,)
        R_ego_from_cam = cam_params['R_current_ego_from_cam']  # (3,3)

        # Create a meshgrid of pixel coordinates
        u_coords = np.arange(cam_params['width'])
        v_coords = np.arange(cam_params['height'])
        uu, vv = np.meshgrid(u_coords, v_coords)  # vv: (H,W), uu: (H,W)

        # Unproject pixels to get ray directions in camera frame (at Z_cam=1)
        x_cam_norm = (uu - cx) / fx
        y_cam_norm = (vv - cy) / fy
        # Ray directions in camera frame (Z_cam=1 plane)
        # Shape: (H, W, 3)
        ray_dirs_cam = np.stack([x_cam_norm, y_cam_norm, np.ones_like(x_cam_norm)], axis=-1)

        # Transform ray directions to ego frame
        # Reshape for batch matrix multiplication: (H*W, 3)
        ray_dirs_cam_flat = ray_dirs_cam.reshape(-1, 3)
        ray_dirs_ego_flat = (R_ego_from_cam @ ray_dirs_cam_flat.T).T  # (H*W, 3)

        # Normalize ray directions in ego frame
        ray_dirs_ego_flat_norm = ray_dirs_ego_flat / (np.linalg.norm(ray_dirs_ego_flat, axis=1, keepdims=True) + 1e-9)

        # Define ray endpoints (far points)
        far_points_ego = cam_origin_ego + ray_dirs_ego_flat_norm * DEPTH_MAX

        # Iterate through each pixel ray for this camera
        num_pixel_rays = far_points_ego.shape[0]
        for ray_idx in tqdm(range(num_pixel_rays), desc=f" Rays for {cam_name}", leave=False):

            ray_start_ego = cam_origin_ego
            ray_end_ego = far_points_ego[ray_idx]

            # Use your existing ray_casting function
            # It expects (ray_hit_point, ray_sensor_origin, ...)
            # Here, ray_start_ego is the sensor origin, ray_end_ego is the far point
            for vox_tuple in ray_casting(
                    ray_start=ray_start_ego,  # Physical start of ray
                    ray_end=ray_end_ego,  # Physical end of ray (far point)
                    pc_range=pc_range_params,
                    voxel_size=voxel_size_params,
                    spatial_shape=spatial_shape_params
            ):
                # vox_tuple is (vx, vy, vz)
                # Check bounds (ray_casting should handle this, but an extra check is safe)
                if not (0 <= vox_tuple[0] < spatial_shape_params[0] and
                        0 <= vox_tuple[1] < spatial_shape_params[1] and
                        0 <= vox_tuple[2] < spatial_shape_params[2]):
                    continue

                lidar_state_at_vox = lidar_voxel_state[vox_tuple]

                if lidar_state_at_vox == STATE_OCCUPIED:
                    camera_visibility_mask[vox_tuple] = STATE_OCCUPIED  # Observed by camera, was occupied by LiDAR
                    break  # Ray is blocked by a LiDAR-occupied voxel
                elif lidar_state_at_vox == STATE_FREE:
                    camera_visibility_mask[vox_tuple] = STATE_FREE  # Observed by camera, was free by LiDAR
                    # Ray continues through free space
                else:
                    camera_visibility_mask[vox_tuple] = STATE_UNOBSERVED

    print("Finished Camera Visibility (CPU).")
    return camera_visibility_mask


# --- Numba CUDA Device Function for Camera Ray Traversal & Mask Update ---
@cuda.jit(device=True)
def _camera_ray_trace_and_update_mask_device(
        # Ray properties
        ray_start_x, ray_start_y, ray_start_z,  # Camera origin in ego frame
        ray_end_x, ray_end_y, ray_end_z,  # Far point for this pixel ray in ego frame
        # Grid parameters
        pc_range_min_x, pc_range_min_y, pc_range_min_z,
        voxel_sx, voxel_sy, voxel_sz,
        grid_dx, grid_dy, grid_dz,
        # Input LiDAR visibility state
        lidar_voxel_state_gpu,  # (Dx,Dy,Dz) uint8, read-only
        # Output camera visibility mask to update
        camera_visibility_mask_gpu,  # (Dx,Dy,Dz) uint8, for writing
        # Constants
        STATE_OCCUPIED_CONST, STATE_FREE_CONST, STATE_UNOBSERVED_CONST,
        EPS, DISTANCE
):
    # --- Inline DDA logic (adapted from your ray_casting) ---
    new_start_x = ray_start_x - pc_range_min_x
    new_start_y = ray_start_y - pc_range_min_y
    new_start_z = ray_start_z - pc_range_min_z

    new_end_x = ray_end_x - pc_range_min_x
    new_end_y = ray_end_y - pc_range_min_y
    new_end_z = ray_end_z - pc_range_min_z

    ray_vx = new_end_x - new_start_x
    ray_vy = new_end_y - new_start_y
    ray_vz = new_end_z - new_start_z

    step_ix, step_iy, step_iz = 0, 0, 0
    if ray_vx > 0:
        step_ix = 1
    elif ray_vx < 0:
        step_ix = -1
    if ray_vy > 0:
        step_iy = 1
    elif ray_vy < 0:
        step_iy = -1
    if ray_vz > 0:
        step_iz = 1
    elif ray_vz < 0:
        step_iz = -1

    t_delta_x = float('inf')
    if ray_vx != 0: t_delta_x = (step_ix * voxel_sx) / ray_vx
    t_delta_y = float('inf')
    if ray_vy != 0: t_delta_y = (step_iy * voxel_sy) / ray_vy
    t_delta_z = float('inf')
    if ray_vz != 0: t_delta_z = (step_iz * voxel_sz) / ray_vz

    adj_start_x = new_start_x + step_ix * voxel_sx * EPS
    adj_start_y = new_start_y + step_iy * voxel_sy * EPS
    adj_start_z = new_start_z + step_iz * voxel_sz * EPS

    # For camera rays, the 'last_voxel' is effectively the one at DEPTH_MAX
    # The loop should continue as long as we are within bounds and haven't hit an occluder
    cur_vox_ix = int(math.floor(adj_start_x / voxel_sx))
    cur_vox_iy = int(math.floor(adj_start_y / voxel_sy))
    cur_vox_iz = int(math.floor(adj_start_z / voxel_sz))

    # No explicit last_voxel needed for termination if we check bounds and DEPTH_MAX (implicitly by ray_end)
    # The DDA termination based on DISTANCE to last_voxel is less relevant here;
    # we trace until occlusion or max depth (implicitly handled by ray_end_x/y/z) or out of bounds.

    t_max_x = float('inf')
    if ray_vx != 0:
        coord_x = float(cur_vox_ix * voxel_sx)
        boundary_x = coord_x + step_ix * voxel_sx if not (step_ix < 0 and coord_x < adj_start_x) else coord_x
        t_max_x = (boundary_x - adj_start_x) / ray_vx

    t_max_y = float('inf')
    if ray_vy != 0:
        coord_y = float(cur_vox_iy * voxel_sy)
        boundary_y = coord_y + step_iy * voxel_sy if not (step_iy < 0 and coord_y < adj_start_y) else coord_y
        t_max_y = (boundary_y - adj_start_y) / ray_vy

    t_max_z = float('inf')
    if ray_vz != 0:
        coord_z = float(cur_vox_iz * voxel_sz)
        boundary_z = coord_z + step_iz * voxel_sz if not (step_iz < 0 and coord_z < adj_start_z) else coord_z
        t_max_z = (boundary_z - adj_start_z) / ray_vz

    max_iterations = grid_dx + grid_dy + grid_dz + 3  # Safety break

    for _ in range(max_iterations):
        # Check if current voxel is within grid bounds
        if not (0 <= cur_vox_ix < grid_dx and \
                0 <= cur_vox_iy < grid_dy and \
                0 <= cur_vox_iz < grid_dz):
            return  # Ray went out of bounds

        # Check LiDAR state at this voxel
        lidar_state_at_vox = lidar_voxel_state_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz]

        if lidar_state_at_vox == STATE_OCCUPIED_CONST:
            # Write STATE_OCCUPIED to the camera mask
            camera_visibility_mask_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz] = STATE_OCCUPIED_CONST
            return  # Ray is blocked
        elif lidar_state_at_vox == STATE_FREE_CONST:
            camera_visibility_mask_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz] = STATE_FREE_CONST
            # Ray continues through this free voxel
        elif lidar_state_at_vox == STATE_UNOBSERVED_CONST:
            # Voxel is unobserved by LiDAR. For Occ3D compatibility, this means
            # it's not considered "observed" in the joint camera-LiDAR sense.
            # The camera ray itself might physically continue, but we stop marking
            # voxels as camera-visible along this path if it enters LiDAR-unobserved space.
            camera_visibility_mask_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz] = STATE_UNOBSERVED_CONST
            return  # Stop considering this ray for camera visibility updates

        # Termination condition: check if we've effectively reached the ray_end
        # This is a bit tricky with DDA. The loop usually stops when out of bounds
        # or after a certain number of steps if ray_end is far.
        # The DISTANCE check from original ray_casting might be adapted.
        # For simplicity, we rely on max_iterations or going out of bounds if DEPTH_MAX is large.
        # Or, if cur_voxel is the voxel containing ray_end_x,y,z, we can stop.
        end_vox_ix = int(math.floor((ray_end_x - pc_range_min_x) / voxel_sx))
        end_vox_iy = int(math.floor((ray_end_y - pc_range_min_y) / voxel_sy))
        end_vox_iz = int(math.floor((ray_end_z - pc_range_min_z) / voxel_sz))
        if cur_vox_ix == end_vox_ix and cur_vox_iy == end_vox_iy and cur_vox_iz == end_vox_iz:
            return  # Reached the far point of the ray

        # Step to next voxel
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                cur_vox_ix += step_ix
                t_max_x += t_delta_x
            else:
                cur_vox_iz += step_iz
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                cur_vox_iy += step_iy
                t_max_y += t_delta_y
            else:
                cur_vox_iz += step_iz
                t_max_z += t_delta_z


# --- CUDA Kernel for a single camera's visibility ---
@cuda.jit
def camera_visibility_kernel_per_cam(
        # Ray origins and directions for this camera
        cam_origin_ego_gpu,  # (3,) XYZ of camera in ego frame
        pixel_ray_dirs_ego_gpu,  # (Num_pixels, 3) Normalized ray directions in ego frame
        # LiDAR visibility and grid parameters (read-only)
        lidar_voxel_state_gpu,  # (Dx,Dy,Dz) uint8
        pc_range_min_gpu,  # (3,)
        voxel_size_gpu,  # (3,)
        spatial_shape_gpu,  # (3,) int32
        # Output
        camera_visibility_mask_gpu,  # (Dx,Dy,Dz) uint8, for writing
        # Constants
        DEPTH_MAX_CONST,
        STATE_OCCUPIED_CONST, STATE_FREE_CONST, STATE_UNOBSERVED_CONST,
        EPS_CONST, DISTANCE_CONST
):
    pixel_idx = cuda.grid(1)  # Global index for the current pixel ray
    if pixel_idx >= pixel_ray_dirs_ego_gpu.shape[0]:
        return

    # Ray start is the camera origin (same for all threads in this launch)
    ray_start_x = cam_origin_ego_gpu[0]
    ray_start_y = cam_origin_ego_gpu[1]
    ray_start_z = cam_origin_ego_gpu[2]

    # Ray direction for this specific pixel
    dir_x = pixel_ray_dirs_ego_gpu[pixel_idx, 0]
    dir_y = pixel_ray_dirs_ego_gpu[pixel_idx, 1]
    dir_z = pixel_ray_dirs_ego_gpu[pixel_idx, 2]

    # Calculate far end-point of the ray
    ray_end_x = ray_start_x + dir_x * DEPTH_MAX_CONST
    ray_end_y = ray_start_y + dir_y * DEPTH_MAX_CONST
    ray_end_z = ray_start_z + dir_z * DEPTH_MAX_CONST

    # Grid parameters for device function
    pc_min_x, pc_min_y, pc_min_z = pc_range_min_gpu[0], pc_range_min_gpu[1], pc_range_min_gpu[2]
    voxel_sx, voxel_sy, voxel_sz = voxel_size_gpu[0], voxel_size_gpu[1], voxel_size_gpu[2]
    grid_dx, grid_dy, grid_dz = spatial_shape_gpu[0], spatial_shape_gpu[1], spatial_shape_gpu[2]

    _camera_ray_trace_and_update_mask_device(
        ray_start_x, ray_start_y, ray_start_z,
        ray_end_x, ray_end_y, ray_end_z,
        pc_min_x, pc_min_y, pc_min_z,
        voxel_sx, voxel_sy, voxel_sz,
        grid_dx, grid_dy, grid_dz,
        lidar_voxel_state_gpu,
        camera_visibility_mask_gpu,  # This is where updates happen
        STATE_OCCUPIED_CONST, STATE_FREE_CONST, STATE_UNOBSERVED_CONST,  # Pass constants
        EPS_CONST, DISTANCE_CONST
    )


# --- Host Function to Manage Camera Visibility GPU Execution ---
def calculate_camera_visibility_gpu_host(
        trucksc: TruckScenes,
        current_sample_token: str,
        lidar_voxel_state_cpu: np.ndarray,  # (Dx,Dy,Dz) uint8
        pc_range_cpu_list: list,
        voxel_size_cpu_scalar: float,
        spatial_shape_cpu_list: list,  # [Dx,Dy,Dz]
        camera_names: List[str],
        DEPTH_MAX_val: float = 100.0
):
    print("Calculating Camera Visibility (GPU)...")

    spatial_shape_tuple = tuple(spatial_shape_cpu_list)

    # Transfer common data to GPU once
    lidar_voxel_state_gpu = cuda.to_device(np.ascontiguousarray(lidar_voxel_state_cpu, dtype=np.uint8))
    pc_range_min_gpu = cuda.to_device(np.ascontiguousarray(pc_range_cpu_list[:3], dtype=np.float64))
    voxel_size_gpu = cuda.to_device(np.array([voxel_size_cpu_scalar] * 3, dtype=np.float64))
    spatial_shape_gpu_dims = cuda.to_device(np.array(spatial_shape_cpu_list, dtype=np.int32))

    # Output mask on GPU, initialized to 0
    camera_visibility_mask_gpu = cuda.to_device(
        np.full(spatial_shape_tuple, STATE_UNOBSERVED, dtype=np.uint8)  # Explicit initialization
    )

    current_sample_rec = trucksc.get('sample', current_sample_token)
    current_ego_pose_ts = current_sample_rec['timestamp']

    EPS_CONST_val = 1e-9
    DISTANCE_CONST_val = 0.5  # From your CPU ray_casting, might not be strictly needed for camera version's termination

    for cam_name in tqdm(camera_names, desc="GPU Processing Cameras"):
        if cam_name not in current_sample_rec['data']:
            print(f"Warning: Camera {cam_name} not found in sample data for token {current_sample_token}. Skipping.")
            continue

        cam_sample_data_token = current_sample_rec['data'][cam_name]
        cam_params = get_camera_parameters(trucksc, cam_sample_data_token, current_ego_pose_ts)

        K = cam_params['intrinsics']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        cam_origin_ego_cpu = np.ascontiguousarray(cam_params['origin_in_current_ego'], dtype=np.float64)
        R_ego_from_cam_cpu = np.ascontiguousarray(cam_params['R_current_ego_from_cam'], dtype=np.float64)

        u_coords = np.arange(cam_params['width'])
        v_coords = np.arange(cam_params['height'])
        uu, vv = np.meshgrid(u_coords, v_coords)

        x_cam_norm = (uu.astype(np.float64) - cx) / fx
        y_cam_norm = (vv.astype(np.float64) - cy) / fy
        ray_dirs_cam = np.stack([x_cam_norm, y_cam_norm, np.ones_like(x_cam_norm)], axis=-1)

        ray_dirs_cam_flat = np.ascontiguousarray(ray_dirs_cam.reshape(-1, 3))
        ray_dirs_ego_flat = (R_ego_from_cam_cpu @ ray_dirs_cam_flat.T).T

        ray_dirs_ego_flat_norm = ray_dirs_ego_flat / (np.linalg.norm(ray_dirs_ego_flat, axis=1, keepdims=True) + 1e-9)
        ray_dirs_ego_flat_norm = np.ascontiguousarray(ray_dirs_ego_flat_norm, dtype=np.float64)

        # Transfer per-camera data
        cam_origin_ego_gpu_current = cuda.to_device(cam_origin_ego_cpu)
        pixel_ray_dirs_ego_gpu_current = cuda.to_device(ray_dirs_ego_flat_norm)

        num_pixel_rays = pixel_ray_dirs_ego_gpu_current.shape[0]
        if num_pixel_rays == 0:
            continue

        threads_per_block_cam = 256
        blocks_per_grid_cam = (num_pixel_rays + (threads_per_block_cam - 1)) // threads_per_block_cam

        # print(f"  Launching Camera Kernel for {cam_name}: {blocks_per_grid_cam} blocks, {threads_per_block_cam} threads")
        camera_visibility_kernel_per_cam[blocks_per_grid_cam, threads_per_block_cam](
            cam_origin_ego_gpu_current,
            pixel_ray_dirs_ego_gpu_current,
            lidar_voxel_state_gpu,  # Already on GPU
            pc_range_min_gpu,  # Already on GPU
            voxel_size_gpu,  # Already on GPU
            spatial_shape_gpu_dims,  # Already on GPU
            camera_visibility_mask_gpu,  # Output, already on GPU
            DEPTH_MAX_val,
            STATE_OCCUPIED, STATE_FREE, STATE_UNOBSERVED,  # Pass constants
            EPS_CONST_val, DISTANCE_CONST_val
        )
        cuda.synchronize()  # Wait for this camera's kernel to finish

    # Copy final camera visibility mask back to CPU
    camera_visibility_mask_cpu = camera_visibility_mask_gpu.copy_to_host()

    print("Finished Camera Visibility (GPU).")
    return camera_visibility_mask_cpu


def visualize_occupancy_o3d(voxel_state, voxel_label, pc_range, voxel_size,
                            class_color_map, default_color,
                            show_semantics=False, show_free=False, show_unobserved=False):
    """
    Visualizes occupancy grid using Open3D.

    Args:
        voxel_state (np.ndarray): 3D array, STATE_OCCUPIED (2), STATE_FREE (1), STATE_UNOBSERVED (0).
        voxel_label (np.ndarray): 3D array of same shape, per-voxel semantic label.
        pc_range (list or np.ndarray): [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size (list or np.ndarray): [vx, vy, vz].
        class_color_map (dict): Mapping from semantic label index to RGB color.
        default_color (list): Default RGB color for labels not in class_color_map.
        show_semantics (bool): If True, color occupied voxels by their semantic label.
                               Otherwise, occupied voxels are red.
        show_free (bool): If True, visualize free voxels (colored light blue).
        show_unobserved (bool): If True, visualize unobserved voxels (colored gray).
    """
    geometries = []

    # --- Process Occupied Voxels ---
    occ_indices = np.where(voxel_state == STATE_OCCUPIED)
    if len(occ_indices[0]) > 0:
        # Convert voxel indices to world-coords of their centers
        xs_occ = (occ_indices[0].astype(float) + 0.5) * voxel_size[0] + pc_range[0]
        ys_occ = (occ_indices[1].astype(float) + 0.5) * voxel_size[1] + pc_range[1]
        zs_occ = (occ_indices[2].astype(float) + 0.5) * voxel_size[2] + pc_range[2]

        occupied_points_world = np.vstack((xs_occ, ys_occ, zs_occ)).T

        pcd_occupied = o3d.geometry.PointCloud()
        pcd_occupied.points = o3d.utility.Vector3dVector(occupied_points_world)

        if show_semantics:
            labels_occ = voxel_label[occ_indices]
            colors_occ = np.array([class_color_map.get(int(label), default_color) for label in labels_occ])
            pcd_occupied.colors = o3d.utility.Vector3dVector(colors_occ)
        else:
            pcd_occupied.paint_uniform_color([1.0, 0.0, 0.0])  # Red for occupied
        geometries.append(pcd_occupied)
    else:
        print("No occupied voxels to show.")

    # --- Process Free Voxels (Optional) ---
    if show_free:
        free_indices = np.where(voxel_state == STATE_FREE)
        if len(free_indices[0]) > 0:
            xs_free = (free_indices[0].astype(float) + 0.5) * voxel_size[0] + pc_range[0]
            ys_free = (free_indices[1].astype(float) + 0.5) * voxel_size[1] + pc_range[1]
            zs_free = (free_indices[2].astype(float) + 0.5) * voxel_size[2] + pc_range[2]
            free_points_world = np.vstack((xs_free, ys_free, zs_free)).T

            pcd_free = o3d.geometry.PointCloud()
            pcd_free.points = o3d.utility.Vector3dVector(free_points_world)
            pcd_free.paint_uniform_color([0.5, 0.7, 1.0])  # Light blue for free
            geometries.append(pcd_free)
        else:
            print("No free voxels to show (or show_free=False).")

    # --- Process Unobserved Voxels (Optional) ---
    if show_unobserved:
        unobserved_indices = np.where(voxel_state == STATE_UNOBSERVED)
        if len(unobserved_indices[0]) > 0:
            xs_unobs = (unobserved_indices[0].astype(float) + 0.5) * voxel_size[0] + pc_range[0]
            ys_unobs = (unobserved_indices[1].astype(float) + 0.5) * voxel_size[1] + pc_range[1]
            zs_unobs = (unobserved_indices[2].astype(float) + 0.5) * voxel_size[2] + pc_range[2]
            unobserved_points_world = np.vstack((xs_unobs, ys_unobs, zs_unobs)).T

            pcd_unobserved = o3d.geometry.PointCloud()
            pcd_unobserved.points = o3d.utility.Vector3dVector(unobserved_points_world)
            pcd_unobserved.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray for unobserved
            geometries.append(pcd_unobserved)
        else:
            print("No unobserved voxels to show (or show_unobserved=False).")

    if not geometries:
        print("Nothing to visualize.")
        return

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Occupancy Grid (Open3D)')
    for geom in geometries:
        vis.add_geometry(geom)

    vis.run()
    vis.destroy_window()

def load_lidar_entries(trucksc, sample, lidar_sensors):
    entries = []
    for sensor in lidar_sensors:
        token = sample['data'][sensor]
        while token:
            sd = trucksc.get('sample_data', token)
            entries.append({
                'sensor': sensor,
                'timestamp': sd['timestamp'],
                'token': token,
                'keyframe': sd['is_key_frame']
            })
            token = sd['next']
    entries.sort(key=lambda x: x['timestamp'])
    return entries


def group_entries(entries, lidar_sensors, max_time_diff):
    used_tokens = set()
    groups = []

    for i, ref_entry in enumerate(entries):
        if ref_entry['token'] in used_tokens:
            continue

        ref_keyframe_flag = ref_entry['keyframe']
        group = {ref_entry['sensor']: ref_entry}
        group_tokens = {ref_entry['token']}

        for j in range(i + 1, len(entries)):
            cand = entries[j]
            if cand['keyframe'] != ref_keyframe_flag:
                continue
            if cand['token'] in used_tokens or cand['sensor'] in group:
                continue
            # Check that the new candidate is close to ALL current group timestamps
            if any(abs(cand['timestamp'] - e['timestamp']) > max_time_diff for e in group.values()):
                continue
            group[cand['sensor']] = cand
            group_tokens.add(cand['token'])

        if len(group) == len(lidar_sensors):
            groups.append(group)
            used_tokens.update(group_tokens)

    return groups

def transform_boxes_to_ego(boxes, ego_pose_record):
    """
    Transforms a list of boxes from global coordinates into ego vehicle coordinates.

    Args:
        boxes: List of Box instances in global/world frame.
        ego_pose_record: A dictionary with keys 'translation' and 'rotation'
                         describing the ego vehicle pose in global frame.

    Returns:
        A new list of Box instances in the ego vehicle coordinate frame.
    """
    transformed_boxes = []
    ego_translation = np.array(ego_pose_record['translation'])
    ego_rotation_inv = Quaternion(ego_pose_record['rotation']).inverse

    for box in boxes:
        box_copy = deepcopy(box)
        # Translate: global -> ego
        box_copy.translate(-ego_translation)
        # Rotate: global -> ego
        box_copy.rotate(ego_rotation_inv)
        transformed_boxes.append(box_copy)

    return transformed_boxes

def visualize_mapmos_predictions(points_xyz: np.ndarray,
                                 predicted_labels: np.ndarray,
                                 scan_index: int,
                                 window_title_prefix: str = "MapMOS Prediction"):
    """
    Visualizes a point cloud, coloring points based on MapMOS predicted labels.

    Args:
        points_xyz (np.ndarray): Point cloud array of shape (N, 3) for XYZ coordinates.
        predicted_labels (np.ndarray): NumPy array of shape (N,) containing predicted labels.
                                      Assumes: 0=static, 1=dynamic, -1=not classified.
        scan_index (int): The index of the scan for the window title.
        window_title_prefix (str): Prefix for the visualization window title.
    """
    if points_xyz.shape[0] == 0:
        print(f"Scan {scan_index}: No points to visualize.")
        return
    if points_xyz.shape[0] != predicted_labels.shape[0]:
        print(
            f"Scan {scan_index}: Mismatch between number of points ({points_xyz.shape[0]}) and labels ({predicted_labels.shape[0]}). Skipping visualization.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz[:, :3])

    # Define colors:
    # Static (label 0) = Blue
    # Dynamic (label 1) = Red
    # Not Classified (label -1) = Gray

    colors = np.zeros_like(points_xyz)  # Initialize with black

    static_mask = (predicted_labels == 0)
    dynamic_mask = (predicted_labels == 1)
    unclassified_mask = (predicted_labels == -1)

    colors[static_mask] = [0.0, 0.0, 1.0]  # Blue for static
    colors[dynamic_mask] = [1.0, 0.0, 0.0]  # Red for dynamic
    colors[unclassified_mask] = [0.5, 0.5, 0.5]  # Gray for not classified/out of range

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Visualizing Scan Index: {scan_index}")
    print(f"  Static points (blue): {np.sum(static_mask)}")
    print(f"  Dynamic points (red): {np.sum(dynamic_mask)}")
    print(f"  Unclassified points (gray): {np.sum(unclassified_mask)}")

    o3d.visualization.draw_geometries([pcd], window_name=f"{window_title_prefix} - Scan {scan_index}")


def save_pointcloud_for_annotation(points_n_features, output_path):
    """
    Saves a point cloud to a .pcd file for annotation tools.

    Args:
        points_n_features (np.ndarray): The point cloud array, shape (N, features).
                                       Assumes XYZ are the first 3 columns.
        output_path (str): The full path to save the .pcd file.
    """
    print(f"Saving point cloud with shape {points_n_features.shape[0]} for annotation to {output_path}")

    if points_n_features.shape[0] == 0:
        print(f"Warning: Attempting to save an empty point cloud to {output_path}. Skipping.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_n_features[:, :3])  # Use only XYZ

    # If you have intensity, you can add it as a color or custom attribute if needed,
    # but for bounding box annotation, XYZ is usually sufficient.

    o3d.io.write_point_cloud(output_path, pcd)
    # print(f"Saved point cloud to {output_path}")


def parse_single_annotation_file(json_filepath):
    """
    Loads annotations for a single frame from a JSON file.
    The JSON file is expected to contain a list of objects for one point cloud.

    Args:
        json_filepath (str): Path to the single annotation JSON file.

    Returns:
        tuple: A tuple containing:
               - frame_idx (int): The index of the frame, parsed from the filename.
               - boxes (list): A list of truckscenes.utils.data_classes.Box objects.
               Returns (None, []) if the file cannot be parsed.
    """
    if not os.path.exists(json_filepath):
        # This is not an error, just means no manual annotation for this frame.
        return None, []

    with open(json_filepath, 'r') as f:
        data = json.load(f)

    boxes_for_this_frame = []
    if 'objects' not in data or not isinstance(data['objects'], list):
        return [] # File exists but has no objects

    for label_obj in data['objects']:
        try:
            # --- 3. Access centroid, dimensions, and rotations as dictionaries ---
            center = [label_obj['centroid']['x'], label_obj['centroid']['y'], label_obj['centroid']['z']]

            # Reorder dimensions: exported is (l, w, h), your Box class expects (w, l, h)
            dims = [label_obj['dimensions']['width'], label_obj['dimensions']['length'], label_obj['dimensions']['height']]

            # Get yaw from the 'z' rotation (in radians)
            yaw = label_obj['rotations']['z']

            # Create the Box object
            box = Box(
                center=center,
                size=dims,
                orientation=Quaternion(axis=[0, 0, 1], angle=yaw)
            )
            box.name = label_obj['name']
            boxes_for_this_frame.append(box)

        except KeyError as e:
            print(f"Warning: Skipping an object in {json_filepath} due to missing key: {e}")
            continue

    return boxes_for_this_frame


def calculate_3d_iou_monte_carlo(box1_params, box2_params, num_samples=3000):
    """
    Calculates the 3D Intersection over Union (IoU) of two oriented bounding boxes
    using Monte Carlo approximation.

    (Docstring remains the same)
    """
    # --- 1. Create Open3D OrientedBoundingBox objects ---
    # (This part is correct and remains unchanged)
    center1 = box1_params[:3]
    extent1 = box1_params[3:6]
    R1 = Rotation.from_euler('z', box1_params[6], degrees=False).as_matrix()
    obb1 = o3d.geometry.OrientedBoundingBox(center1, R1, extent1)

    center2 = box2_params[:3]
    extent2 = box2_params[3:6]
    R2 = Rotation.from_euler('z', box2_params[6], degrees=False).as_matrix()
    obb2 = o3d.geometry.OrientedBoundingBox(center2, R2, extent2)

    # --- 2. Monte Carlo Approximation of Intersection Volume ---

    # Generate random points uniformly within the axis-aligned version of box 1
    points_local1 = np.random.rand(num_samples, 3) - 0.5
    points_local1 *= extent1

    ### MODIFICATION START: Correctly transform points to world frame ###

    # Manually create the 4x4 transformation matrix for obb1
    transform_matrix1 = np.eye(4)
    transform_matrix1[:3, :3] = obb1.R
    transform_matrix1[:3, 3] = obb1.center

    # Convert local points to homogeneous coordinates for transformation
    points_local1_homo = np.hstack((points_local1, np.ones((num_samples, 1))))

    # Apply the transformation: (4x4 matrix) @ (4xN points)
    points_world_homo = transform_matrix1 @ points_local1_homo.T

    # Convert back to 3D coordinates (N, 3)
    points_world = points_world_homo[:3, :].T

    ### MODIFICATION END ###

    # --- 3. Count how many of these points are inside box 2 ---
    # (This part is correct and remains unchanged)
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(points_world)
    indices_in_box2 = obb2.get_point_indices_within_bounding_box(pcd_temp.points)

    num_intersecting_points = len(indices_in_box2)

    # --- 4. Calculate Volumes and IoU ---
    # (This part is correct and remains unchanged)
    volume1 = obb1.volume()
    volume2 = obb2.volume()

    intersection_volume = volume1 * (num_intersecting_points / num_samples)
    union_volume = volume1 + volume2 - intersection_volume

    if union_volume < 1e-6:
        return 0.0

    iou = intersection_volume / union_volume

    return iou


# MODIFIED Function to match the [width, length, height] convention
def convert_boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
  """
  Converts parameterized 3D boxes to the 8 corner coordinates.
  MODIFIED: Assumes input dimensions are [width, length, height] and maps them
  directly to the box's local X, Y, and Z axes respectively.

  Args:
      boxes (torch.Tensor): A tensor of shape (N, 7) with parameters
                            [cx, cy, cz, w, l, h, yaw_rad].

  Returns:
      torch.Tensor: A tensor of shape (N, 8, 3) representing the box corners.
  """
  if boxes.ndim != 2 or boxes.shape[1] != 7:
    raise ValueError("Input tensor must be of shape (N, 7).")

  device = boxes.device
  corners_norm = torch.tensor([
    [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
    [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5], [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
  ], dtype=torch.float32, device=device)

  centers = boxes[:, 0:3]

  # --- FIX: Use dimensions directly without reordering ---
  # The input tensor's dimensions are [width, length, height].
  # We will use these to scale the x, y, and z axes of the canonical box.
  dims_wlh = boxes[:, 3:6]

  yaws = boxes[:, 6]

  # Create rotation matrices from yaw
  cos_yaws, sin_yaws = torch.cos(yaws), torch.sin(yaws)
  zeros, ones = torch.zeros_like(cos_yaws), torch.ones_like(cos_yaws)
  rot_matrices = torch.stack([
    cos_yaws, -sin_yaws, zeros,
    sin_yaws, cos_yaws, zeros,
    zeros, zeros, ones
  ], dim=1).reshape(-1, 3, 3)

  # --- FIX: Scale the corners using the original [w, l, h] dimensions ---
  # This now scales the box's local X-axis by width, Y-axis by length, and Z-axis by height.
  scaled_corners = corners_norm.unsqueeze(0) * dims_wlh.unsqueeze(1)

  # Rotate and translate corners
  rotated_corners = torch.bmm(scaled_corners, rot_matrices.transpose(1, 2))
  final_corners = rotated_corners + centers.unsqueeze(1)

  return final_corners

def calculate_3d_iou_pytorch3d(boxes1_params: np.ndarray, boxes2_params: np.ndarray) -> np.ndarray:
    """
    Calculates the exact 3D IoU using PyTorch3D.
    Takes parameterized boxes, converts them to corners, and computes IoU.

    Args:
        boxes1_params (np.ndarray): Box parameters of shape (N, 7) [cx,cy,cz,w,l,h,yaw].
        boxes2_params (np.ndarray): Box parameters of shape (M, 7) [cx,cy,cz,w,l,h,yaw].

    Returns:
        np.ndarray: A matrix of IoU values of shape (N, M).
    """
    # Convert numpy arrays to GPU tensors
    b1_t = torch.from_numpy(boxes1_params).float().cuda()
    b2_t = torch.from_numpy(boxes2_params).float().cuda()

    # Convert parameterized boxes to 8 corner coordinates
    corners1 = convert_boxes_to_corners(b1_t)
    corners2 = convert_boxes_to_corners(b2_t)

    # Calculate IoU using the PyTorch3D CUDA kernel
    # The function returns (intersection_volume, iou_matrix)
    _, iou_matrix_gpu = box3d_overlap(corners1, corners2)

    return iou_matrix_gpu.cpu().numpy()

def get_object_overlap_signature(frame_data, target_obj_idx_in_frame, iou_min_threshold=0.01):
    """
    Calculates a robust overlap signature for the target object in the given frame.

    The signature is a sorted list of tuples, where each tuple represents a significant
    overlap with another object.
    Format: [(iou_with_obj1, obj1_instance_token), (iou_with_obj2, obj2_instance_token), ...]

    Args:
        frame_data (dict): The dictionary for a single frame from your `dict_list`.
        target_obj_idx_in_frame (int): The index of the target object.
        iou_min_threshold (float): The minimum IoU to be considered a significant overlap.

    Returns:
        list: A sorted list of (float, str) tuples representing the signature.
    """
    # target_obj_bbox_params = frame_data['gt_bbox_3d'][target_obj_idx_in_frame]
    # gt_bbox_3d_target = frame_data['gt_bbox_3d_unmodified'][target_obj_idx_in_frame]
    gt_bbox_3d_target = frame_data['gt_bbox_3d'][target_obj_idx_in_frame]

    """dims = gt_bbox_3d_target[:, 3:6]

    gt_bbox_3d_target[:, 6] += np.pi / 2.
    gt_bbox_3d_target[:, 2] -= dims[:, 2] / 2.
    gt_bbox_3d_target[:, 2] -= 0.1
    gt_bbox_3d_target[:, 3:6] *= 1.1"""

    target_obj_bbox_params = gt_bbox_3d_target.copy()

    # This list will store all significant overlaps
    overlaps = []

    for other_obj_idx, other_obj_token in enumerate(frame_data['object_tokens']):
        # Skip comparing the object to itself
        if other_obj_idx == target_obj_idx_in_frame:
            continue

        # gt_bbox_3d_other_obj = frame_data['gt_bbox_3d_unmodified'][other_obj_idx]
        gt_bbox_3d_other_obj = frame_data['gt_bbox_3d'][other_obj_idx]

        """dims = gt_bbox_3d_other_obj[:, 3:6]

        gt_bbox_3d_other_obj[:, 6] += np.pi / 2.
        gt_bbox_3d_other_obj[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d_other_obj[:, 2] -= 0.1
        gt_bbox_3d_other_obj[:, 3:6] *= 1.1"""

        other_obj_bbox_params = gt_bbox_3d_other_obj.copy()

        # Calculate the 3D IoU (using the Monte Carlo function from before)
        iou = calculate_3d_iou_monte_carlo(target_obj_bbox_params, other_obj_bbox_params)
        #iou = calculate_3d_iou_pytorch3d(target_obj_bbox_params, other_obj_bbox_params)[0]

        # If the overlap is significant, add it to our list
        if iou > iou_min_threshold:
            relative_yaw = calculate_relative_yaw(target_obj_bbox_params, other_obj_bbox_params)
            other_obj_category = frame_data['converted_object_category'][other_obj_idx]
            overlaps.append((iou, other_obj_token, other_obj_category, relative_yaw))

    # VERY IMPORTANT: Sort the list to create a canonical signature.
    # We sort by the instance token (the second element of the tuple).
    # This ensures that the order doesn't affect the comparison later.
    overlaps.sort(key=lambda x: x[1])

    return overlaps


def get_object_overlap_signature_COMPARISON(frame_data, target_obj_idx_in_frame, iou_min_threshold=0.01):
    """
    TEMPORARY a_iou_comparison_function.
    Calculates IoU using both Monte Carlo and PyTorch3D methods for side-by-side comparison
    and prints the results. This is intentionally not batched for easy comparison.
    """
    # --- Using the EXPANDED boxes as requested ---
    # This is frame_data['gt_bbox_3d'] from your main loop
    all_boxes_in_frame = frame_data['gt_bbox_3d']
    target_obj_bbox_params = all_boxes_in_frame[target_obj_idx_in_frame]

    # Get the token for logging
    target_token_short = frame_data['object_tokens'][target_obj_idx_in_frame].split('-')[0]

    overlaps = []
    print(f"\n--- IoU Comparison for Target Box: {target_token_short}... ---")

    for other_obj_idx, other_obj_token in enumerate(frame_data['object_tokens']):
        if other_obj_idx == target_obj_idx_in_frame:
            continue

        other_obj_bbox_params = all_boxes_in_frame[other_obj_idx]

        # --- 1. Calculate IoU with Monte Carlo method ---
        iou_mc = calculate_3d_iou_monte_carlo(target_obj_bbox_params, other_obj_bbox_params)

        # --- 2. Calculate IoU with exact PyTorch3D method ---
        # Reshape inputs from (7,) to (1, 7) for the batch-based function
        target_for_pt3d = target_obj_bbox_params.reshape(1, 7)
        other_for_pt3d = other_obj_bbox_params.reshape(1, 7)
        # The result is a (1, 1) matrix, so we get the single value
        iou_pt3d = calculate_3d_iou_pytorch3d(target_for_pt3d, other_for_pt3d)[0, 0]

        # --- 3. Compare and build signature ---
        # We will use the more accurate pt3d result for the actual signature
        iou = iou_pt3d

        # Only print and process if there is a significant overlap
        if iou > iou_min_threshold or iou_mc > iou_min_threshold:
            other_token_short = other_obj_token.split('-')[0]
            print(
                f"  vs {other_token_short:<8} | Monte Carlo IoU: {iou_mc:.6f} | PyTorch3D (Exact) IoU: {iou_pt3d:.6f}")

            if iou > iou_min_threshold:
                relative_yaw = calculate_relative_yaw(target_obj_bbox_params, other_obj_bbox_params)
                other_obj_category = frame_data['converted_object_category'][other_obj_idx]
                overlaps.append((iou, other_obj_token, other_obj_category, relative_yaw))

    overlaps.sort(key=lambda x: x[1])
    return overlaps


def get_object_overlap_signature_BATCH(frame_data, target_obj_idx_in_frame, iou_min_threshold=0.01):
    """
    FINAL VERSION.
    Calculates a robust overlap signature for the target object in the given frame
    using a single, efficient batch call to the PyTorch3D IoU function.
    """
    # --- Using the EXPANDED boxes as requested ---
    # This is frame_data['gt_bbox_3d'] from your main loop
    all_boxes_in_frame = frame_data['gt_bbox_3d']

    # Isolate the single target box for the batch call
    target_box = all_boxes_in_frame[target_obj_idx_in_frame:target_obj_idx_in_frame + 1]  # Shape: (1, 7)

    # --- Perform ONE batch calculation for maximum efficiency ---
    # This computes IoU between the target box and all boxes, returning a (1, M) matrix.
    iou_row = calculate_3d_iou_pytorch3d(target_box, all_boxes_in_frame)[0]

    overlaps = []
    for other_obj_idx, other_obj_token in enumerate(frame_data['object_tokens']):
        if other_obj_idx == target_obj_idx_in_frame:
            continue

        # Look up the pre-calculated IoU
        iou = iou_row[other_obj_idx]

        if iou > iou_min_threshold:
            target_obj_bbox_params = all_boxes_in_frame[target_obj_idx_in_frame]
            other_obj_bbox_params = all_boxes_in_frame[other_obj_idx]

            relative_yaw = calculate_relative_yaw(target_obj_bbox_params, other_obj_bbox_params)
            other_obj_category = frame_data['converted_object_category'][other_obj_idx]
            overlaps.append((iou, other_obj_token, other_obj_category, relative_yaw))

    overlaps.sort(key=lambda x: x[1])
    return overlaps

def compare_signatures(sig1, sig2,
                       iou_strong_threshold=0.05,
                       iou_strong_tolerance=0.01,
                       yaw_tolerance_rad=0.1):
    """
    Compares two overlap signatures with tiered logic for weak and strong overlaps.
    """
    # 1. Check for same number of interacting objects
    if len(sig1) != len(sig2):
        return False

    # 2. Handle the isolated case
    if not sig1:
        return True

    # 3. Iterate and compare each corresponding interaction
    for i in range(len(sig1)):
        iou1, token1, cat1, yaw1 = sig1[i]
        iou2, token2, cat2, yaw2 = sig2[i]

        # 3a. Tokens must match
        if token1 != token2:
            return False

        # 3b. Relative yaws must be similar
        if abs(yaw1 - yaw2) > yaw_tolerance_rad:
            return False

        # ### MODIFICATION: Tiered IoU Comparison Logic ###
        is_strong1 = iou1 > iou_strong_threshold
        is_strong2 = iou2 > iou_strong_threshold

        if is_strong1 and is_strong2:
            # BOTH are strong overlaps. Compare their values with a tolerance.
            if abs(iou1 - iou2) > iou_strong_tolerance:
                return False
        elif not is_strong1 and not is_strong2:
            # BOTH are weak overlaps. We consider them a match without checking the
            # exact IoU value, as they are both just "slight touches".
            pass
        else:
            # One is strong and one is weak. This is a clear mismatch in state.
            return False

    # If all checks passed, the signatures are a match
    return True


# --- 2. NEW CLASS-BASED SIGNATURE COMPARISON FUNCTION ---
def compare_signatures_class_based(sig1, target_cat1, sig2, target_cat2, thresholds_config):
    """
    Compares two overlap signatures using class-based, dynamic thresholds.

    Args:
        sig1 (list): Signature of the first object.
        target_cat1 (int): Class ID of the first target object.
        sig2 (list): Signature of the second object.
        target_cat2 (int): Class ID of the second target object.
        thresholds_config (dict): The configuration dictionary with class-based thresholds.

    Returns:
        bool: True if the signatures are a match based on class-specific rules.
    """
    # 1. Check if the target objects are even the same class
    if target_cat1 != target_cat2:
        return False

    # 2. Check for same number of interacting objects
    if len(sig1) != len(sig2):
        return False

    # 3. Handle the isolated case
    if not sig1:
        return True

    # 4. Iterate and compare each corresponding interaction
    for i in range(len(sig1)):
        iou1, token1, neighbor_cat1, yaw1 = sig1[i]
        iou2, token2, neighbor_cat2, yaw2 = sig2[i]

        # 4a. Tokens and neighbor classes must match
        if token1 != token2 or neighbor_cat1 != neighbor_cat2:
            return False

        # --- DYNAMIC THRESHOLD LOOKUP ---
        # Create the key for the dictionary lookup
        class_pair_key = frozenset({target_cat1, neighbor_cat1})

        # Get the specific thresholds for this pair, falling back to 'default' if not found
        params = thresholds_config.get(class_pair_key, thresholds_config['default'])

        iou_strong_threshold = params['iou_strong_threshold']
        iou_strong_tolerance = params['iou_strong_tolerance']
        yaw_tolerance_rad = params['yaw_tolerance_rad']
        # --- END DYNAMIC LOOKUP ---

        # 4b. Relative yaws must be similar (using the dynamic tolerance)
        if abs(yaw1 - yaw2) > yaw_tolerance_rad:
            return False

        # 4c. Tiered IoU Comparison Logic (using the dynamic thresholds)
        is_strong1 = iou1 > iou_strong_threshold
        is_strong2 = iou2 > iou_strong_threshold

        if is_strong1 and is_strong2:
            # BOTH are strong overlaps. Compare their values with the dynamic tolerance.
            if abs(iou1 - iou2) > iou_strong_tolerance:
                return False
        elif not is_strong1 and not is_strong2:
            # BOTH are weak overlaps. This is a match.
            pass
        else:
            # One is strong and one is weak. This is a clear mismatch in state.
            return False

    # If all checks passed for all interactions, the signatures are a match
    return True

def are_box_sizes_similar(box1_params, box2_params, volume_ratio_tolerance=1.05):
    """
    Checks if the dimensions of two bounding boxes are similar based on their volume ratio.

    Args:
        box1_params (np.ndarray): Parameters for box 1 [cx, cy, cz, l, w, h, yaw].
        box2_params (np.ndarray): Parameters for box 2 in the same format.
        volume_ratio_tolerance (float): How much larger one volume can be than the other.
                                        E.g., 1.2 means a 20% difference is allowed.

    Returns:
        bool: True if the box sizes are considered similar.
    """
    # Extract dimensions (l, w, h)
    dims1 = box1_params[3:6]
    dims2 = box2_params[3:6]

    # Calculate volumes
    volume1 = dims1[0] * dims1[1] * dims1[2]
    volume2 = dims2[0] * dims2[1] * dims2[2]

    # Avoid division by zero if a box has no volume
    if volume1 < 1e-6 or volume2 < 1e-6:
        # Consider them not similar if one has volume and the other doesn't,
        # but similar if both have no volume.
        return (volume1 < 1e-6) and (volume2 < 1e-6)

    # Calculate the ratio of the larger volume to the smaller volume
    if volume1 > volume2:
        ratio = volume1 / volume2
    else:
        ratio = volume2 / volume1

    # Check if the ratio is within the allowed tolerance
    return ratio <= volume_ratio_tolerance


def is_point_centroid_z_similar(points1, points2, category_id,
                                target_category_id, z_tolerance=0.2):
    """
    Checks if the Z-centroid of two point clouds are similar, but only for a specific category.
    This is highly effective for vehicles like forklifts where the point cloud's center of mass
    changes vertically with the load.

    Args:
        points1 (np.ndarray): Point cloud (N,3) for the object in the keyframe.
        points2 (np.ndarray): Point cloud (M,3) for the object in the candidate frame.
        category_id (int): The learning category ID of the object.
        target_category_id (int): The specific category ID to apply this check to.
        z_tolerance (float): The maximum allowed vertical distance (in meters) between point centroids.

    Returns:
        bool: True if the point cloud Z-centroids are considered similar.
    """
    # If the object is not the target category, we don't apply this check.
    if category_id != target_category_id:
        return True

    # Handle cases where one or both point clouds might be empty.
    if points1.shape[0] == 0 and points2.shape[0] == 0:
        return True  # Both empty is a match.
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        return False  # One empty and one not is a mismatch.

    # Calculate the mean of the Z-coordinate (the 3rd column, index 2) for each cloud.
    centroid_z1 = np.mean(points1[:, 2])
    centroid_z2 = np.mean(points2[:, 2])

    # Return True only if the vertical distance is within our tolerance.
    return abs(centroid_z1 - centroid_z2) <= z_tolerance


def calculate_relative_yaw(box1_params, box2_params):
    """
    Calculates the shortest angle difference between the yaws of two boxes.

    Args:
        box1_params (np.ndarray): Parameters for box 1 [..., yaw].
        box2_params (np.ndarray): Parameters for box 2 [..., yaw].

    Returns:
        float: The relative yaw in radians, in the range [-pi, pi].
    """
    yaw1 = box1_params[6]
    yaw2 = box2_params[6]

    # Calculate the difference
    delta_yaw = yaw1 - yaw2

    # Normalize the angle to the range [-pi, pi] to get the shortest path
    relative_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))

    return relative_yaw

def polygon_to_3d_lineset(poly2d, y_min, y_max, col=(1, 0, 0)):
    """
    Extrude a 2D Shapely polygon (or MultiPolygon) in X–Z up to a Y-thickness,
    and return a list of Open3D LineSet wireframes.
    (This function is kept as is, but will be used by a new helper)
    """
    if isinstance(poly2d, MultiPolygon):
        linesets = []
        for part in poly2d.geoms:
            linesets.extend(polygon_to_3d_lineset(part, y_min, y_max, col))
        return linesets

    ring = list(poly2d.exterior.coords)
    n = len(ring)

    verts = []
    for (x, z) in ring:
        verts.append([x, y_min, z])
    for (x, z) in ring:
        verts.append([x, y_max, z])

    lines = []
    for i in range(n):
        lines.append([i, (i + 1) % n])
    for i in range(n, 2 * n):
        lines.append([i, n + ((i + 1) % n)])
    for i in range(n):
        lines.append([i, i + n])

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(verts)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    ).paint_uniform_color(col)

    return [ls]


def create_side_L_shapes(box_tr, box_tl, h_hitch=1.5):
    """
    Given two LOCAL-FRAME box tuples (cx,cy,cz, L,W,H, yaw=0), returns:
      - tractor_L2D : the L-shape for the tractor in X–Z
      - trailer_L2D : the complementary shape for the trailer
      - z_ground_tr, z_roof_tr
    """
    # NOTE: The box format is assumed to be (center_x, center_y, center_z, Length, Width, Height, yaw)
    # Length is along the vehicle's forward axis (X), Width is across (Y).
    cxt, _, szt, Lt, _, Ht, _ = box_tr
    cxl, _, szl, Ll, _, Hl, _ = box_tl

    # half‐sizes & key Z planes for tractor vs. trailer
    z_ground_tr = szt - Ht / 2.0
    z_roof_tr = szt + Ht / 2.0
    z_ground_tl = szl - Hl / 2.0
    z_roof_tl = szl + Hl / 2.0
    z_hitch = z_ground_tr + h_hitch

    # X‐planes (in local frame)
    x_front_tr = cxt + Lt / 2.0
    x_back_tr = cxt - Lt / 2.0
    x_front_tl = cxl + Ll / 2.0
    x_back_tl = cxl - Ll / 2.0

    # 1) tractor L‐shape in X–Z (using tractor’s Z bounds)
    # This shape now has 6 points for a more accurate "step" profile.
    tractor_pts = [
        (x_front_tr, z_ground_tr),  # Bottom-front of cab
        (x_front_tr, z_roof_tr),  # Top-front of cab
        (x_front_tl, z_roof_tr),  # Top-rear of cab (at trailer coupling)
        (x_front_tl, z_hitch),  # Drop down to hitch height
        (x_back_tr, z_hitch),  # Back of tractor chassis at hitch height
        (x_back_tr, z_ground_tr),  # Bottom-rear of tractor chassis
    ]
    tractor_L2D = Polygon(tractor_pts).buffer(0)

    # 2) full trailer rectangle side‐view (using trailer’s Z bounds)
    trailer_rect = Polygon([
        (x_front_tl, z_ground_tl),
        (x_front_tl, z_roof_tl),
        (x_back_tl, z_roof_tl),
        (x_back_tl, z_ground_tl),
    ]).buffer(0)

    # 3) subtract tractor‐L from trailer rect
    trailer_L2D = trailer_rect.difference(tractor_L2D)
    if not trailer_L2D.is_valid:
        trailer_L2D = trailer_L2D.buffer(0)

    return tractor_L2D, trailer_L2D, z_ground_tr, z_roof_tr


# ------------------------------------------------------------------
def assign_label_by_L_shape(
        overlap_idxs, pt_to_box_map,
        points, boxes, box_cls_labels, pt_labels
):
    """
    Reassign only overlap points between each truck(10)/trailer(9) pair.
    All others keep pt_labels.
    (This function was logically correct and is kept as is).
    """
    new_labels = pt_labels.flatten().copy()
    h_hitch = 1.3

    pair_to_pts = defaultdict(list)
    for pi in overlap_idxs:
        b = sorted(pt_to_box_map[pi])
        if len(b) == 2:
            pair_to_pts[tuple(b)].append(pi)

    for (i, j), pts in pair_to_pts.items():
        li, lj = box_cls_labels[i], box_cls_labels[j]
        if set((li, lj)) != {9, 10}:
            continue

        idx_tr = i if li == 10 else j
        idx_tl = j if idx_tr == i else i

        # Unpack world‐coords. Note: Your npy file seems to have w,l,h not l,w,h
        # boxes[:,3:6] -> width, length, height
        cx_tr, cy_tr, cz_tr, w_tr, l_tr, h_tr, yaw_tr, _ = boxes[idx_tr]
        cx_tl, cy_tl, cz_tl, w_tl, l_tl, h_tl, _, _ = boxes[idx_tl]

        c, s = math.cos(-yaw_tr), math.sin(-yaw_tr)

        # Build LOCAL‐frame boxes (Length is X, Width is Y)
        box_tr_local = (0.0, 0.0, cz_tr, l_tr, w_tr, h_tr, 0.0)

        dx0 = cx_tl - cx_tr
        dy0 = cy_tl - cy_tr
        x_loc_tl = dx0 * c - dy0 * s
        y_loc_tl = dx0 * s + dy0 * c
        box_tl_local = (x_loc_tl, y_loc_tl, cz_tl, l_tl, w_tl, h_tl, 0.0)

        L_tr, L_tl, z_ground, z_roof = create_side_L_shapes(
            box_tr_local, box_tl_local, h_hitch
        )

        half_W_tr = w_tr / 2.0
        for pi in pts:
            xg, yg, zg = points[pi]
            # Rotate point into tractor's local frame
            dx_pt = (xg - cx_tr) * c - (yg - cy_tr) * s
            dy_pt = (xg - cx_tr) * s + (yg - cy_tr) * c

            # Check if point is laterally within the tractor's width and vertically within its profile
            if abs(dy_pt) > half_W_tr or not (z_ground <= zg <= z_roof):
                continue

            # Check against the 2D side-view polygons
            point_2d = Point(dx_pt, zg)
            if L_tr.contains(point_2d):
                new_labels[pi] = 10  # Truck
            elif L_tl.contains(point_2d):
                new_labels[pi] = 9  # Trailer

    return new_labels.reshape(-1, 1)


def create_rotated_wireframes(box_tr_world, box_tl_world):
    """
    Creates the L-shape wireframes by generating each in its own local frame
    and then applying its own direct world transformation. This is the most
    robust method for visualization.
    """
    # --- Truck Data ---
    cx_tr, cy_tr, cz_tr, w_tr, l_tr, h_tr, yaw_tr = box_tr_world
    # Define truck box in its own pure local frame (origin at 0,0,0)
    box_tr_local = (0.0, 0.0, 0.0, l_tr, w_tr, h_tr, 0.0)

    # --- Trailer Data ---
    cx_tl, cy_tl, cz_tl, w_tl, l_tl, h_tl, yaw_tl = box_tl_world
    # Define trailer box in ITS OWN pure local frame
    box_tl_local = (0.0, 0.0, 0.0, l_tl, w_tl, h_tl, 0.0)

    # --- Create Polygons ---
    # To get the shape of the trailer, we still need to know where the truck is relative to it.
    # We will simulate the truck's position from the trailer's perspective.
    c_trailer_inv, s_trailer_inv = math.cos(-yaw_tl), math.sin(-yaw_tl)
    dx0_rel = cx_tr - cx_tl
    dy0_rel = cy_tr - cy_tl
    x_loc_tr_in_tl_frame = dx0_rel * c_trailer_inv - dy0_rel * s_trailer_inv
    y_loc_tr_in_tl_frame = dx0_rel * s_trailer_inv + dy0_rel * c_trailer_inv
    z_loc_tr_in_tl_frame = cz_tr - cz_tl

    # We call create_side_L_shapes from the trailer's perspective to get the trailer_poly
    # The first argument is the truck, the second is the trailer.
    _, trailer_poly, _, _ = create_side_L_shapes(
        box_tr=(x_loc_tr_in_tl_frame, y_loc_tr_in_tl_frame, z_loc_tr_in_tl_frame, l_tr, w_tr, h_tr, 0.0),
        box_tl=box_tl_local,  # Trailer is at the origin of this frame
        h_hitch=1.3
    )

    # For the truck polygon, we do the reverse.
    c_truck_inv, s_truck_inv = math.cos(-yaw_tr), math.sin(-yaw_tr)
    dx0_rel_tr = cx_tl - cx_tr
    dy0_rel_tr = cy_tl - cy_tr
    x_loc_tl_in_tr_frame = dx0_rel_tr * c_truck_inv - dy0_rel_tr * s_truck_inv
    y_loc_tl_in_tr_frame = dx0_rel_tr * s_truck_inv + dy0_rel_tr * c_truck_inv
    z_loc_tl_in_tr_frame = cz_tl - cz_tr

    truck_poly, _, _, _ = create_side_L_shapes(
        box_tr=box_tr_local,  # Truck is at the origin of this frame
        box_tl=(x_loc_tl_in_tr_frame, y_loc_tl_in_tr_frame, z_loc_tl_in_tr_frame, l_tl, w_tl, h_tl, 0.0),
        h_hitch=1.3
    )

    # --- Create Local Wireframes ---
    ls_tr_local = polygon_to_3d_lineset(truck_poly, -w_tr / 2.0, w_tr / 2.0, col=(0, 1, 0))  # Green
    ls_tl_local = polygon_to_3d_lineset(trailer_poly, -w_tl / 2.0, w_tl / 2.0, col=(1, 0, 1))  # Magenta

    # --- Apply Direct World Transformations ---

    # 1. Transform Truck Wireframe using its own world pose
    transform_tr = np.identity(4)
    transform_tr[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw_tr])
    transform_tr[:3, 3] = [cx_tr, cy_tr, cz_tr]
    for ls in ls_tr_local:
        ls.transform(transform_tr)

    # 2. Transform Trailer Wireframe using ITS OWN world pose
    transform_tl = np.identity(4)
    transform_tl[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw_tl])
    transform_tl[:3, 3] = [cx_tl, cy_tl, cz_tl]
    for ls in ls_tl_local:
        ls.transform(transform_tl)

    return ls_tr_local + ls_tl_local


def visualize_l_shape_results_3d(
        points_xyz,
        final_labels,
        overlap_idxs,
        pt_to_box_map,
        boxes_with_labels,
        box_cls_labels,
        class_color_map
):
    """
    Visualizes the final point cloud colored by refined labels, with 3D L-shape wireframes overlaid.

    Args:
        points_xyz (np.ndarray): The (N,3) XYZ coordinates of the full point cloud.
        final_labels (np.ndarray): The (N,1) final semantic labels after L-shape refinement.
        overlap_idxs (np.ndarray): Indices of points in overlapping boxes.
        pt_to_box_map (defaultdict): Mapping from point index to list of box indices.
        boxes_with_labels (np.ndarray): The (M, 8) box parameters used for classification.
        box_cls_labels (np.ndarray): The (M,) class labels for each box.
        class_color_map (dict): The dictionary for coloring points.
    """
    print("\n--- Generating Final 3D L-Shape Visualization ---")

    # 1. Create and color the main point cloud based on the FINAL labels
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(
        np.array([class_color_map.get(l, [0.5, 0.5, 0.5]) for l in final_labels.flatten()])
    )

    geometries_to_draw = [pcd]

    # 2. Find truck/trailer pairs and generate their 3D L-shape wireframes
    seen_pairs = set()
    for pi in overlap_idxs:
        # Use .get() for safety in case a point is no longer in the map
        box_ids = tuple(sorted(pt_to_box_map.get(pi, [])))
        if len(box_ids) != 2 or box_ids in seen_pairs:
            continue

        i, j = box_ids
        if {box_cls_labels[i], box_cls_labels[j]} == {9, 10}:  # Truck and Trailer
            seen_pairs.add(box_ids)
            idx_tr = i if box_cls_labels[i] == 10 else j
            idx_tl = j if box_cls_labels[i] == i else i

            # Get the world-frame box parameters (without the label)
            box_tr_world = boxes_with_labels[idx_tr][:7]
            box_tl_world = boxes_with_labels[idx_tl][:7]

            # Create the 3D wireframes using the helper function
            l_shape_wireframes = create_rotated_wireframes(box_tr_world, box_tl_world)
            geometries_to_draw.extend(l_shape_wireframes)

    # 3. Display the final scene
    print("Displaying point cloud with final labels and 3D L-Shape wireframes.")
    print("  - Truck L-Shape: Green")
    print("  - Trailer L-Shape: Magenta")
    o3d.visualization.draw_geometries(
        geometries_to_draw,
        window_name="Final 3D L-Shape Classification Result",
        width=1920, height=1080
    )

def main(trucksc, val_list, indice, truckscenesyaml, args, config):
    # Extract necessary parameters from the arguments and configs
    save_path = args.save_path  # Directory where processed data will be saved
    data_root = args.dataroot  # Root directory of dataset
    learning_map = truckscenesyaml['learning_map']  # dictionary that maps raw semantic labels to learning labels
    voxel_size = config['voxel_size']  # Size of each voxel in the occupancy grid
    pc_range = config['pc_range']  # Range of point cloud coordinates to consider (bounding box)
    occ_size = config['occ_size']  # Dimensions of the output occupancy grid
    load_mode = args.load_mode
    self_range = config[
        'self_range']  # Parameter in config file that specifies a range threshold for the vehicle's own points
    x_min_self, y_min_self, z_min_self, x_max_self, y_max_self, z_max_self = self_range

    intensity_threshold = config['intensity_threshold']
    distance_intensity_threshold = config['distance_intensity_threshold']

    # sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT']
    sensors = config['sensors']
    print(f"Lidar sensors: {sensors}")
    sensors_max_range_list = []
    for sensor in sensors:
        if sensor in ['LIDAR_LEFT', 'LIDAR_RIGHT']:
            sensors_max_range_list.append(200)
        if sensor in ['LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR']:
            sensors_max_range_list.append(35)
    print(f"Lidar sensors max range: {sensors_max_range_list}")
    sensor_max_ranges_arr = np.array(sensors_max_range_list, dtype=np.float64)

    cameras = config['cameras']
    print(f"Cameras: {cameras}")

    max_time_diff = config['max_time_diff']

    # Retrieves a specific scene from the truckScenes dataset
    my_scene = trucksc.scene[indice]  # scene is selected by indice parameter
    scene_name = my_scene['name']  ### Extract scene name for saving
    print(f"Processing scene: {scene_name}")
    scene_description = my_scene['description']
    print(f"Scene description: {scene_description}")
    # load the first sample from a scene to start
    first_sample_token = my_scene[
        'first_sample_token']  # access the first sample token: contains token of first frame of the scene
    my_sample = trucksc.get('sample',
                            first_sample_token)  # retrieve the first sample as dictionary. Dictionary includes data from multiple sensors

    # Data split handling
    if args.split == 'train':
        if my_scene['token'] in val_list:  # Ensures that no validation data is mistakenly used in training
            return
    elif args.split == 'val':
        if my_scene['token'] not in val_list:  # Ensures that no training data is used during validation
            return
    elif args.split == 'all':  # Proceeds without filtering, useful for generating predictions or evaluations on the entire dataset
        pass
    else:  # Error if split type is not recognized
        raise NotImplementedError

    # Define the numeric index for the 'Unknown' class
    # Assuming key 36 maps to 'Unknown' in labels and learning_map[36] is 15
    unknown_label_key = 36
    UNKNOWN_LEARNING_INDEX = learning_map.get(unknown_label_key, 15)

    # Define the numeric index for the 'Background' class
    # Assuming key 37 maps to 'Background' in labels and learning_map[37] is 16
    background_label_key = 37
    BACKGROUND_LEARNING_INDEX = learning_map.get(background_label_key, 16)

    free_label_key = 38
    FREE_LEARNING_INDEX = learning_map.get(free_label_key, 17)

    # Create a mapping from category name (e.g., "car") to learning ID (e.g., 4)
    category_name_to_learning_id = {}
    for label_key, label_name_in_yaml in truckscenesyaml['labels'].items():
        mapped_label_index = learning_map.get(label_key)
        if mapped_label_index is not None:
            category_name_to_learning_id[label_name_in_yaml] = mapped_label_index

    # Create the reverse mapping from learning ID (e.g., 10) to name (e.g., "truck")
    # We ensure the keys are integers for correct lookups.
    learning_id_to_name = {int(k): v for k, v in truckscenesyaml['labels_16'].items()}

    # Define the learning ID for the category we want to apply the Z-centroid check to
    OTHER_VEHICLE_LEARNING_ID = category_name_to_learning_id.get('vehicle.other', 13)

    TRUCK_ID = category_name_to_learning_id.get('truck', 10)
    TRAILER_ID = category_name_to_learning_id.get('trailer', 9)
    CAR_ID = category_name_to_learning_id.get('car', 10)
    OTHER_VEHICLE_ID = category_name_to_learning_id.get('other', 13)
    BARRIER_ID = category_name_to_learning_id.get('barrier', 10)
    PEDESTRIAN_ID = category_name_to_learning_id.get('pedestrian', 10)

    CLASS_BASED_THRESHOLDS = {
        # --- Interaction: truck <-> trailer ---
        # This is a tight, expected interaction. We use a high threshold.
        frozenset({TRUCK_ID, TRAILER_ID}): {
            'iou_strong_threshold': 0.10,
            'iou_strong_tolerance': 0.10,  # Allow a larger difference since they are large objects
            'yaw_tolerance_rad': 0.1
        },
        # --- Interaction: car <-> car ---
        # Two cars shouldn't have a high IoU unless they are crashing.
        frozenset({CAR_ID, CAR_ID}): {
            'iou_strong_threshold': 0.05,
            'iou_strong_tolerance': 0.05,
            'yaw_tolerance_rad': 0.2
        },
        # --- Interaction: truck <-> other_vehicle ---
        # Based on your logs, this seems to be an incidental, "nearby" interaction.
        # We can use a lower threshold to still capture it in the signature,
        # but the logic will correctly classify it as 'weak'.
        frozenset({TRUCK_ID, OTHER_VEHICLE_ID}): {
            'iou_strong_threshold': 0.01,
            'iou_strong_tolerance': 0.001,
            'yaw_tolerance_rad': 0.2
        },

        frozenset({TRAILER_ID, OTHER_VEHICLE_ID}): {
            'iou_strong_threshold': 0.01,
            'iou_strong_tolerance': 0.001,
            'yaw_tolerance_rad': 0.2
        },
        # --- DEFAULT FALLBACK ---
        # This is crucial for any pair you haven't explicitly defined.
        'default': {
            'iou_strong_threshold': 0.05,
            'iou_strong_tolerance': 0.01,
            'yaw_tolerance_rad': 0.15
        }
    }

    lidar_entries = load_lidar_entries(trucksc=trucksc, sample=my_sample, lidar_sensors=sensors)
    print(f"Number of lidar entries: {len(lidar_entries)}")

    groups = group_entries(entries=lidar_entries, lidar_sensors=sensors, max_time_diff=max_time_diff)
    print(f"\n✅ Total groups found: {len(groups)}")

    dict_list = []

    reference_ego_pose = None
    ref_ego_from_global = None

    for i, group in enumerate(groups):
        print(f"Processing group {i}, timestamps:")
        for sensor in sensors:
            print(f"  {sensor}: {group[sensor]['timestamp']} | keyframe: {group[sensor]['keyframe']}")

        ref_sensor = sensors[0]

        sample_data_dict = {sensor: group[sensor]['token'] for sensor in sensors}
        sample = {
            'timestamp': np.mean([group[s]['timestamp'] for s in sensors]),
            'data': sample_data_dict,
            'sample_data_token': sample_data_dict[ref_sensor],
            'is_key_frame': group[ref_sensor]['keyframe'],
        }

        ########### Load point cloud #############
        if load_mode == 'pointwise':
            sensor_fused_pc, sensor_ids_points = get_pointwise_fused_pointcloud(trucksc, sample,
                                                                                allowed_sensors=sensors)
        elif load_mode == 'rigid':
            sensor_fused_pc, sensor_ids_points = get_rigid_fused_pointcloud(trucksc, sample, allowed_sensors=sensors)
        else:
            raise ValueError(f'Fusion mode {load_mode} is not supported')

        if sensor_fused_pc.timestamps is not None:
            print(
                f"The fused sensor pc at frame {i} has the shape: {sensor_fused_pc.points.shape} with timestamps: {sensor_fused_pc.timestamps.shape}")
        else:
            print(f"The fused sensor pc at frame {i} has the shape: {sensor_fused_pc.points.shape} with no timestamps.")

        ##########################################
        if args.version == 'v1.0-trainval':
            scene_terminal_list = [4, 68, 69, 70, 203, 205, 206, 241, 272, 273, 423, 492, 597]
        elif args.version == 'v1.0-test':
            scene_terminal_list = [3, 114, 115, 116]

        ################################# Uncomment if you need to save pointclouds for annotation ###########################

        """if indice in scene_terminal_list:
            annotation_base = os.path.join(data_root, 'annotation')
            # Format the filename with leading zeros to maintain order (e.g., 000000.pcd, 000001.pcd)

            if sample['is_key_frame']:
                output_pcd_filename = f"{i:06d}_keyframe.pcd"
                annotation_data_save_path = os.path.join(annotation_base, scene_name, 'pointclouds', output_pcd_filename)
            else:
                output_pcd_filename = f"{i:06d}_nonkeyframe.pcd"
                annotation_data_save_path = os.path.join(annotation_base, scene_name, 'pointclouds', 'nonkeyframes', output_pcd_filename)

            # Save the raw fused point cloud before any filtering
            # We use .points.T to get the (N, features) shape
            save_pointcloud_for_annotation(sensor_fused_pc.points.T, annotation_data_save_path)"""


        ########### get boxes #####################
        # --- Step 1: Get original boxes from the dataset ---
        boxes_global = trucksc.get_boxes(sample['sample_data_token'])

        # Convert to ego or sensor frame (optional depending on fusion mode)
        pose_record = trucksc.getclosest('ego_pose', trucksc.get('sample_data', sample['sample_data_token'])['timestamp'])

        boxes_ego = transform_boxes_to_ego(
            boxes=boxes_global,
            ego_pose_record=pose_record
        )

        original_boxes_token = [box.token for box in boxes_ego]  # retrieves a list of tokens from the bounding box
        # Extract object categories
        original_object_category_names = [truckscenes.get('sample_annotation', box_token)['category_name'] for box_token in
                           original_boxes_token]  # retrieves category name for each bounding box

        # Convert original category names to numeric learning IDs
        converted_object_category = []
        for category_name in original_object_category_names:
            numeric_label = category_name_to_learning_id.get(category_name, UNKNOWN_LEARNING_INDEX)
            converted_object_category.append(numeric_label)

        ##################################################### get manual boxes for terminal scenes ###################################
        if indice in scene_terminal_list:
            annotation_base = os.path.join(data_root, 'annotation')
            if sample['is_key_frame']:
                load_pcd_annotation_filename = f"{i:06d}_keyframe.json"
                load_annotation_ending = os.path.join('keyframes', load_pcd_annotation_filename)
            else:
                load_pcd_annotation_filename = f"{i:06d}_nonkeyframe.json"
                load_annotation_ending = os.path.join('nonkeyframes', load_pcd_annotation_filename)

            annotation_data_load_path = os.path.join(annotation_base, scene_name, 'annotations',
                                                     load_annotation_ending)
            manual_boxes = parse_single_annotation_file(annotation_data_load_path)

            if manual_boxes:  # This checks if the list is not empty
                print(f"Frame {i}: Loaded {len(manual_boxes)} manual annotations. Augmenting GT.")
                print(manual_boxes)
                for box in manual_boxes:
                    boxes_ego.append(box)
                    manual_category_name = box.name
                    numeric_label = category_name_to_learning_id.get(manual_category_name, UNKNOWN_LEARNING_INDEX)
                    converted_object_category.append(numeric_label)

        # Extract object tokens. Each instance token represents a unique object
        #original_object_tokens = [truckscenes.get('sample_annotation', box_token)['instance_token'] for
         #                         box_token in
          #                        original_boxes_token]  # Uses sample_annotation data to get instance_token fore each bb

        object_tokens = []
        for box in boxes_ego:
            if box.token is not None:
                # This is an original box from the dataset, get its instance token
                instance_token = truckscenes.get('sample_annotation', box.token)['instance_token']
                object_tokens.append(instance_token)
            else:
                # This is a manually added box, it has no token in the dataset
                manual_token = f"manual_{box.name}"
                object_tokens.append(manual_token)

        # print(object_tokens)

        ############################# get object categories ##########################
        """converted_object_category = []  # Initialize empty list

        # Iterate over each object category extracted earlier
        for category in object_category:
            found_match = False
            # Iterate over label mappings defined in truckscenes.yaml file
            for label_key, label_name_in_yaml in truckscenesyaml['labels'].items():
                # Check category and map to learning label
                if category == label_name_in_yaml:
                    mapped_label_index = learning_map.get(label_key)
                    if mapped_label_index is not None:
                        converted_object_category.append(mapped_label_index)
                        found_match = True
                        break  # Found the mapping, move to the next category
                    else:
                        # This case means label_key exists in 'labels' but not 'learning_map'
                        print(
                            f"Warning: Category '{category}' mapped to label_key '{label_key}', but '{label_key}' not found in learning_map. Using 'Unknown' label.")
                        # --- CHANGE: Use UNKNOWN_LEARNING_INDEX ---
                        converted_object_category.append(UNKNOWN_LEARNING_INDEX)
                        found_match = True
                        break

            # If the category was not found in the truckscenesyaml['labels'] mapping at all
            if not found_match:
                print(
                    f"Warning: Category '{category}' not found in truckscenes.yaml mapping. Using 'Unknown' label.")
                # --- CHANGE: Use UNKNOWN_LEARNING_INDEX ---
                converted_object_category.append(UNKNOWN_LEARNING_INDEX)"""

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes_ego]).reshape(-1,
                                                               3)  # gets center coordinates (x,y,z) of each bb
        dims = np.array([b.wlh for b in boxes_ego]).reshape(-1,
                                                            3)  # extract dimension width, length, height of each bb
        rots = np.array([b.orientation.yaw_pitch_roll[0]  # extract rotations (yaw angles)
                         for b in boxes_ego]).reshape(-1, 1)
        gt_bbox_3d_unmodified = np.concatenate([locs, dims, rots], axis=1).astype(
            np.float32)  # combines location, dimensions and rotation into a 2D array

        gt_bbox_3d = gt_bbox_3d_unmodified.copy()

        gt_bbox_3d[:, 6] += np.pi / 2.  # adjust yaw angles by 90 degrees
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        # gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.05  # Experiment
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        # gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.05 # Experiment
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.2  # Slightly expand the bbox to wrap all object points

        ############################### Visualize if specified in arguments ###########################################
        # visualize_pointcloud(sensor_fused_pc.points.T, title=f"Fused sensor PC in ego coordinates - Frame {i}")
        """visualize_pointcloud_bbox(sensor_fused_pc.points.T,
                                          boxes=boxes_ego,
                                          title=f"Fused filtered static sensor PC + BBoxes + Ego BBox - Frame {i}",
                                          self_vehicle_range=self_range,
                                          vis_self_vehicle=True)"""

        if args.vis_raw_pc:
            visualize_pointcloud_bbox(sensor_fused_pc.points.T,
                                      boxes=boxes_ego,
                                      title=f"Fused raw sensor PC + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)


        ############################## Filter raw pc #################################################################
        if args.filter_raw_pc and args.filter_mode != 'none':
            # 1) prepare
            raw_pts = sensor_fused_pc.points.T # (N, 3+…)
            raw_sids = sensor_ids_points.copy()  # (N,)
            pcd_raw = o3d.geometry.PointCloud()
            pcd_raw.points = o3d.utility.Vector3dVector(raw_pts[:, :3])

            # 2) filter
            filtered_raw_pcd, kept_raw_idx = denoise_pointcloud(pcd_raw, args.filter_mode, config,
                                                                location_msg=f"raw pc at frame {i}")
            # 3) re‐assemble arrays
            raw_pts = np.asarray(filtered_raw_pcd.points)
            raw_sids = raw_sids[kept_raw_idx]

            # 4) if you had extra features beyond XYZ, re‐append them:
            if sensor_fused_pc.points.shape[0] > 3:
                raw_pts = np.hstack([raw_pts, sensor_fused_pc.points.T[kept_raw_idx, 3:]])

            # Now overwrite your fused_pc & sensor_ids:
            sensor_fused_pc.points = raw_pts.T
            sensor_fused_pc.timestamps = sensor_fused_pc.timestamps[:, kept_raw_idx]
            sensor_ids_points = raw_sids.copy()

        assert sensor_fused_pc.points.shape[1] == sensor_ids_points.shape[0], \
            f"point count {sensor_fused_pc.points.shape[1]} vs sensor_ids {sensor_ids_points.shape[0]}"
        ##############################################################################################################

        ############################### Visualize if specified in arguments ###########################################
        # visualize_pointcloud(sensor_fused_pc.points.T, title=f"Fused sensor PC in ego coordinates - Frame {i}")
        """visualize_pointcloud_bbox(sensor_fused_pc.points.T,
                                          boxes=boxes_ego,
                                          title=f"Fused filtered static sensor PC + BBoxes + Ego BBox - Frame {i}",
                                          self_vehicle_range=self_range,
                                          vis_self_vehicle=True)"""

        if args.vis_raw_pc and args.filter_raw_pc:
            visualize_pointcloud_bbox(sensor_fused_pc.points.T,
                                      boxes=boxes_ego,
                                      title=f"Fused filtered raw sensor PC (filter mode {args.filter_mode}) + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)

        ############################# cut out movable object points and masks ##########################
        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(sensor_fused_pc.points.T[:, :3][np.newaxis, :, :]),
                                              torch.from_numpy(gt_bbox_3d[np.newaxis,
                                                               :]))  # use function to identify which points belong to which bounding box

        num_points = sensor_fused_pc.points.shape[1]
        points_label = np.full((num_points, 1), BACKGROUND_LEARNING_INDEX, dtype=np.uint8)

        # Assign object labels to points inside bounding boxes
        # Ensure converted_object_category has the correct mapped labels
        for box_idx in range(gt_bbox_3d.shape[0]):
            # Get the mask for points in the current box
            object_points_mask = points_in_boxes[0][:, box_idx].bool()
            # Get the semantic label for this object type
            object_label = converted_object_category[box_idx]
            # Assign the object label to the corresponding points in the points_label
            points_label[object_points_mask] = object_label

        pc_with_semantic = np.concatenate([sensor_fused_pc.points.T[:, :3], points_label], axis=1)
        visualize_pointcloud_bbox(pc_with_semantic,
                                  boxes=boxes_ego,
                                  title=f"Fused filtered raw sensor PC before L-shape refinement + BBoxes + Ego BBox - Frame {i}",
                                  self_vehicle_range=self_range,
                                  vis_self_vehicle=True)

        ###################################### L-SHAPE Refinement Logic ############################################
        print(f"Frame {i}: Refining truck/trailer points with L-shape logic...")

        # 2. Prepare inputs for the L-shape function
        # Find points that fall into more than one box
        point_counts_in_boxes = points_in_boxes[0].sum(axis=1).numpy()
        overlap_idxs = np.where(point_counts_in_boxes > 1)[0]
        print(f"Found {len(overlap_idxs)} points in overlapping bounding boxes.")

        overlap_points = sensor_fused_pc.points.T[:, :3][overlap_idxs]
        visualize_pointcloud_bbox(overlap_points,
                                  boxes=boxes_ego,
                                  title=f"Fused filtered raw sensor PC before L-shape refinement + BBoxes + Ego BBox - Frame {i}",
                                  self_vehicle_range=self_range,
                                  vis_self_vehicle=True)

        if len(overlap_idxs) > 0:
            # Create the mapping from each point to the boxes it's in
            pt_to_box_map = defaultdict(list)
            # This is more efficient than iterating through all points
            for pi in overlap_idxs:
                box_indices_for_point = np.where(points_in_boxes[0][pi, :].bool())[0]
                pt_to_box_map[pi] = box_indices_for_point.tolist()

            # Get the raw XYZ points
            points_xyz_for_lshape = sensor_fused_pc.points.T[:, :3]

            # Prepare the box data in the format your function expects.
            # It needs [cx, cy, cz, w, l, h, yaw, label]
            # gt_bbox_3d is [cx, cy, cz, w, l, h, yaw]
            # converted_object_category is [label]
            box_cls_labels = np.array(converted_object_category)
            # The `boxes_for_polygons` from your new script used unmodified l, h, but scaled w.
            # Your main pipeline's `gt_bbox_3d` is ALREADY scaled, so we create the required
            # variant from `gt_bbox_3d_unmodified`.
            boxes_for_polygons = gt_bbox_3d_unmodified.copy()
            boxes_for_polygons[:, [3]] *= 1.2  # Scale Width (Y-dim)
            #boxes_for_polygons[:, 6] += np.pi / 2.  # Adjust yaw
            #boxes_for_polygons[:, 2] -= gt_bbox_3d_unmodified[:, 5] / 2.  # Adjust Z center
            #boxes_for_polygons[:, 2] -= 0.1  # Move down

            # Also need to add the label column for the function call
            boxes_for_polygons_with_labels = np.concatenate(
                [boxes_for_polygons, box_cls_labels.reshape(-1, 1)], axis=1
            )

            # 3. Call the refinement function
            final_labels = assign_label_by_L_shape(
                overlap_idxs=overlap_idxs,
                pt_to_box_map=pt_to_box_map,
                points=points_xyz_for_lshape,
                boxes=boxes_for_polygons_with_labels,  # Use the specially prepared boxes
                box_cls_labels=box_cls_labels,
                pt_labels=points_label
            )

            # 4. CRITICAL: Update the main `points_label` variable with the refined labels
            points_label = final_labels
            print("L-shape refinement complete. Point labels have been updated.")
        else:
            print("No overlapping points found, skipping L-shape refinement.")

        pc_with_semantic = np.concatenate([sensor_fused_pc.points.T[:, :3], points_label], axis=1)

        visualize_pointcloud_bbox(pc_with_semantic,
                                  boxes=boxes_ego,
                                  title=f"Fused filtered raw sensor PC after L-shape refinement + BBoxes + Ego BBox - Frame {i}",
                                  self_vehicle_range=self_range,
                                  vis_self_vehicle=True)

        print("Rebuilding object_points_list from refined labels using instance-specific masks...")

        # This pc_with_semantic will now be correct
        pc_with_semantic = np.concatenate([sensor_fused_pc.points.T[:, :3], points_label], axis=1)

        object_points_list = []
        objects_points_list_sensor_ids = []

        # Ensure points_label is a flat numpy array for masking
        refined_labels_flat = points_label.flatten()

        # Iterate through each BOX/OBJECT INSTANCE by its index
        for box_idx in range(gt_bbox_3d.shape[0]):
            object_label_for_this_box = converted_object_category[box_idx]

            # --- Create the final mask by combining two conditions ---

            # Condition 1: Points that are GEOMETRICALLY inside this specific box's original volume.
            # We get this from the original points_in_boxes result.
            mask_geometric = points_in_boxes[0][:, box_idx].bool().numpy()

            # Condition 2: Points whose final SEMANTIC label matches this box's class.
            mask_semantic = (refined_labels_flat == object_label_for_this_box)

            # The final mask is the logical AND of the two. A point must satisfy both.
            final_instance_mask = mask_geometric & mask_semantic

            # Extract points and their sensor IDs using this precise instance mask
            object_points = sensor_fused_pc.points.T[final_instance_mask]
            object_points_sensor_ids = sensor_ids_points.T[final_instance_mask]

            # Append the clean list of points to the final list
            object_points_list.append(object_points)
            objects_points_list_sensor_ids.append(object_points_sensor_ids)

            # --- Sanity Check (Now it should give meaningful results) ---
            num_original_pts = np.sum(mask_geometric)
            num_refined_pts = np.sum(final_instance_mask)
            if num_original_pts != num_refined_pts:
                print(
                    f"  Box {box_idx} (Label: {object_label_for_this_box}): Points changed from {num_original_pts} to {num_refined_pts}")
            else:
                # If the count didn't change, we can just log it for completeness
                print(f"  Box {box_idx} (Label: {object_label_for_this_box}): Points unchanged at {num_refined_pts}")

        print("Finished creating refined, instance-specific object point lists.")

        visualize_l_shape_results_3d(
            points_xyz=sensor_fused_pc.points.T[:, :3],
            final_labels=points_label,
            overlap_idxs=overlap_idxs,
            pt_to_box_map=pt_to_box_map,
            boxes_with_labels=boxes_for_polygons_with_labels,
            box_cls_labels=box_cls_labels,
            class_color_map=CLASS_COLOR_MAP
        )

        """object_points_list = []  # creates an empty list to store points associated with each object
        objects_points_list_sensor_ids = []
        j = 0
        # Iterate through each bounding box along the last dimension
        while j < points_in_boxes.shape[-1]:
            # Create a boolean mask indicating whether each point belongs to the current bounding box.
            object_points_mask = points_in_boxes[0][:, j].bool()
            # Extract points using mask to filter points
            object_points = sensor_fused_pc.points.T[object_points_mask]
            object_points_sensor_ids = sensor_ids_points.copy().T[object_points_mask]
            # Store the filtered points, Result is a list of arrays, where each element contains the points belonging to a particular object
            object_points_list.append(object_points)
            objects_points_list_sensor_ids.append(object_points_sensor_ids)
            j = j + 1"""

        # shape: (1, Npoints, Nboxes)
        point_box_mask = points_in_boxes[0]  # Remove batch dim: shape (Npoints, Nboxes)

        # Point is dynamic if it falls inside *any* box
        dynamic_mask = point_box_mask.any(dim=-1)  # shape: (Npoints,)

        # Count
        num_dynamic_points = dynamic_mask.sum().item()
        print(f"Number of dynamic points: {num_dynamic_points}")

        # Get static mask (inverse)
        static_mask = ~dynamic_mask

        # Get points from the fused point cloud (transposed for shape [Npoints, 4])
        points_xyz = sensor_fused_pc.points.T[:, :3]

        # Create a mask for points outside the ego vehicle bounding box
        # Mask calculation: filters out points that are too close to the vehicle in x, y or z directions

        inside_x = torch.from_numpy(points_xyz[:, 0] >= x_min_self) & torch.from_numpy(points_xyz[:, 0] <= x_max_self)
        inside_y = torch.from_numpy(points_xyz[:, 1] >= y_min_self) & torch.from_numpy(points_xyz[:, 1] <= y_max_self)
        inside_z = torch.from_numpy(points_xyz[:, 2] >= z_min_self) & torch.from_numpy(points_xyz[:, 2] <= z_max_self)

        inside_ego_mask = inside_x & inside_y & inside_z

        num_ego_points = inside_ego_mask.sum().item()
        print(f"Number of points on ego vehicle: {num_ego_points}")

        dynamic_mask_mapmos = dynamic_mask | inside_ego_mask

        current_frame_mapmos_labels = dynamic_mask_mapmos.cpu().numpy().astype(np.int32).reshape(-1, 1)
        assert current_frame_mapmos_labels.shape[0] == sensor_fused_pc.points.T.shape[0], \
            "Mismatch between number of points in scan and generated labels"

        total_mapmos_dynamic_labels = dynamic_mask_mapmos.sum().item()
        print(f"Total points labeled as dynamic for MapMOS input (annotated dynamic + ego): {total_mapmos_dynamic_labels}")

        ego_filter_mask = ~inside_ego_mask

        points_mask = static_mask & ego_filter_mask

        pc_ego_unfiltered = sensor_fused_pc.points.T[points_mask]
        pc_ego_unfiltered_sensors = sensor_ids_points.copy().T[points_mask]
        print(
            f"Number of static points extracted: {pc_ego_unfiltered.shape} with sensor_ids {pc_ego_unfiltered_sensors.shape}")

        """for box_idx in range(gt_bbox_3d.shape[0]):
            num_in_box = points_in_boxes[0][:, box_idx].sum().item()
            print(f"Box {box_idx} contains {num_in_box} points")"""

        pc_with_semantic_ego_unfiltered = pc_with_semantic[points_mask]
        pc_with_semantic_ego_unfiltered_sensors = sensor_ids_points.copy().T[points_mask]
        print(
            f"Number of semantic static points extracted: {pc_with_semantic_ego_unfiltered.shape} with sensor_ids {pc_with_semantic_ego_unfiltered_sensors.shape}")

        ############################### Visualize if specified in arguments ###########################################
        """visualize_pointcloud_bbox(pc_with_semantic_ego_unfiltered,
                                          boxes=boxes_ego,
                                          title=f"Fused filtered static sensor PC + BBoxes + Ego BBox - Frame {i}",
                                          self_vehicle_range=self_range,
                                          vis_self_vehicle=True)"""

        if args.vis_static_pc:
            visualize_pointcloud_bbox(pc_with_semantic_ego_unfiltered,
                                      boxes=boxes_ego,
                                      title=f"Fused static sensor PC + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)
        ############################################################################################################

        pc_ego = pc_ego_unfiltered.copy()
        # pc_with_semantic_ego = pc_with_semantic_ego_unfiltered.copy()

        ########################################## Lidar intensity filtering #######################################
        if args.filter_lidar_intensity:
            # find the part that starts with "weather."
            weather_tag = next(tag for tag in scene_description.split(';') if tag.startswith('weather.'))
            # split on the dot and take the second piece
            weather = weather_tag.split('.', 1)[1]

            if weather == 'snow' or weather == 'rain' or weather == 'fog':
                print(
                    f"Lidar intensity filtering for bad weather: {weather} with intensity {intensity_threshold} and distance threshold {distance_intensity_threshold} metres")
                print(f'Shape of pc_ego before weather filtering: {pc_ego.shape}')

                distances_to_ego = np.linalg.norm(pc_ego[:, :3], axis=1)

                pc_lidar_intensities = pc_ego[:, 3]

                filter_keep_mask = (distances_to_ego > distance_intensity_threshold) | \
                                   ((distances_to_ego <= distance_intensity_threshold) & (
                                           pc_lidar_intensities > intensity_threshold))

                # intensity_mask = pc_lidar_intensities > intensity_threshold
                # pc_ego = pc_ego[intensity_mask]
                # pc_ego_unfiltered_sensors = pc_ego_unfiltered_sensors[intensity_mask]
                # pc_with_semantic_ego = pc_with_semantic_ego[intensity_mask]
                # pc_with_semantic_ego_unfiltered_sensors = pc_with_semantic_ego_unfiltered_sensors[intensity_mask]

                pc_ego = pc_ego[filter_keep_mask]
                pc_ego_unfiltered_sensors = pc_ego_unfiltered_sensors[filter_keep_mask]
                # pc_with_semantic_ego = pc_with_semantic_ego[filter_keep_mask]
                # pc_with_semantic_ego_unfiltered_sensors = pc_with_semantic_ego_unfiltered_sensors[filter_keep_mask]

                print(f'Shape of pc_ego after weather filtering: {pc_ego.shape}')

                ################################ Visualize if specified in arguments ##################################
                if args.vis_lidar_intensity_filtered:
                    visualize_pointcloud_bbox(pc_ego,
                                                      boxes=boxes_ego,
                                                      title=f"Fused filtered static sensor PC + BBoxes + Ego BBox - Frame {i}",
                                                      self_vehicle_range=self_range,
                                                      vis_self_vehicle=True)
                #######################################################################################################
            else:
                print(f"No lidar intensity filtering for good weather: {weather}")
        else:
            print(f"No lidar intensity filtering according to arguments")

        ############################# Apply filtering to static points in ego frame #################################
        if args.filter_static_pc and args.filter_mode != 'none':
            pcd_static = o3d.geometry.PointCloud()
            pcd_static.points = o3d.utility.Vector3dVector(pc_ego[:, :3])
            filtered_pcd_static, kept_indices = denoise_pointcloud(
                pcd_static, args.filter_mode, config, location_msg=f"static ego points at frame {i}"
            )
            pc_ego = np.asarray(filtered_pcd_static.points)
            pc_ego_unfiltered_sensors = pc_ego_unfiltered_sensors[kept_indices]
            # pc_with_semantic_ego = pc_with_semantic_ego[kept_indices]  # ✅ Only filter here
            # pc_with_semantic_ego_unfiltered_sensors = pc_with_semantic_ego_unfiltered_sensors[kept_indices]

        assert pc_ego.shape[0] == pc_ego_unfiltered_sensors.shape[0], (
            f"static points ({pc_ego.shape[0]}) != sensor_ids ({pc_ego_unfiltered_sensors.shape[0]})"
        )
        """assert pc_with_semantic_ego.shape[0] == pc_with_semantic_ego_unfiltered_sensors.shape[0], (
            f"semantic points ({pc_with_semantic_ego.shape[0]}) != semantic_sensor_ids "
            f"({pc_with_semantic_ego_unfiltered_sensors.shape[0]})"
        )"""

        ############################ Visualization #############################################################
        if args.vis_static_pc and args.filter_static_pc:
            visualize_pointcloud_bbox(pc_with_semantic_ego_unfiltered,
                                      boxes=boxes_ego,
                                      title=f"Fused filtered static sensor PC (filter mode {args.filter_mode}) + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)
        #######################################################################################################

        # Get ego pose for this sample
        ego_pose_i = trucksc.getclosest('ego_pose', sample['timestamp'])

        # Transformation from ego to global
        global_from_ego_i = transform_matrix(ego_pose_i['translation'], Quaternion(ego_pose_i['rotation']),
                                             inverse=False)

        if reference_ego_pose is None:
            reference_ego_pose = ego_pose_i  # Store reference pose
            # Calculate transform from global TO reference ego
            ref_ego_from_global = transform_matrix(reference_ego_pose['translation'],
                                                   Quaternion(reference_ego_pose['rotation']), inverse=True)
            # Transformation from current ego (i) to reference ego is identity
            ego_ref_from_ego_i = np.eye(4)
            print(f"Frame {i}: Set as reference frame.")
        else:
            # Calculate transformation from current ego (i) TO reference ego
            # ego_ref <- global <- ego_i
            ego_ref_from_ego_i = ref_ego_from_global @ global_from_ego_i
            print(f"Frame {i}: Calculated transform to reference frame.")

        # --- Transform FILTERED static points TO REFERENCE EGO FRAME ---
        # Inputs to transform_points should be the filtered points: pc_ego, pc_with_semantic_ego
        points_in_ref_frame = transform_points(pc_ego, ego_ref_from_ego_i)
        #semantic_points_in_ref_frame = transform_points(pc_with_semantic_ego, ego_ref_from_ego_i)
        print(f"Frame {i}: Transformed static points to ref ego. Shape: {points_in_ref_frame.shape}")
        #print(f"Frame {i}: Transformed semantic static points to ref ego. Shape: {semantic_points_in_ref_frame.shape}")

        # --- Transform FILTERED static points TO GLOBAL FRAME ---
        points_in_global_frame = transform_points(pc_ego, global_from_ego_i)
        #semantic_points_in_global_frame = transform_points(pc_with_semantic_ego, global_from_ego_i)
        print(f"Frame {i}: Transformed static points to global. Shape: {points_in_global_frame.shape}")
        #print(
         #   f"Frame {i}: Transformed semantic static points to global. Shape: {semantic_points_in_global_frame.shape}")

        if args.vis_static_pc_global:
            visualize_pointcloud(points_in_ref_frame, title=f"Fused sensor PC in world coordinates - Frame {i}")

        # Assign the calculated variables to the desired keys
        pc_ego_i_save = pc_ego.copy()  # Filtered points in current ego frame (Features, N)
        print(f"Frame {i}: Static points in ego frame shape: {pc_ego_i_save.shape}")
        # pc_with_semantic_ego_i_save = pc_with_semantic_ego.copy()  # Filtered semantic points in current ego frame (Features+1, N)
        # print(f"Frame {i}: Static semantic points in ego frame shape: {pc_with_semantic_ego_i_save.shape}")
        pc_ego_ref_save = points_in_ref_frame.copy()  # Filtered points transformed to reference ego frame (Features, N)
        # pc_with_semantic_ego_ref_save = semantic_points_in_ref_frame.copy()  # Filtered semantic points transformed to reference ego frame (Features+1, N)
        pc_global_save = points_in_global_frame.copy()  # Filtered points transformed to global frame (Features, N)
        # pc_with_semantic_global_save = semantic_points_in_global_frame.copy()  # Filtered semantic points transformed to global frame (Features+1, N)

        ################## record information into a dict  ########################
        ref_sd = trucksc.get('sample_data', sample['sample_data_token'])

        print(sensor_ids_points.T.shape)

        frame_dict = {
            "sample_timestamp": sample['timestamp'],
            "scene_name": scene_name,
            "sample_token": trucksc.get('sample', ref_sd['sample_token'])['token'],
            "is_key_frame": sample['is_key_frame'],
            "converted_object_category": converted_object_category,
            "boxes_ego": boxes_ego,
            "gt_bbox_3d": gt_bbox_3d,  # BBox in current frame's ego coords
            "gt_bbox_3d_unmodified": gt_bbox_3d_unmodified,
            "object_tokens": object_tokens,
            "object_points_list": object_points_list,  # Raw object points in current ego frame
            "object_points_list_sensor_ids": objects_points_list_sensor_ids,
            "raw_lidar_ego": sensor_fused_pc.points.T,
            "raw_lidar_ego_sensor_ids": sensor_ids_points.T,
            "mapmos_per_point_labels": current_frame_mapmos_labels,
            "lidar_pc_ego_i": pc_ego_i_save,  # Filtered static points in CURRENT ego frame (i)
            "lidar_pc_ego_sensor_ids": pc_ego_unfiltered_sensors,
            # "lidar_pc_with_semantic_ego_i": pc_with_semantic_ego_i_save,
            # "lidar_pc_with_semantic_ego_sensor_ids": pc_with_semantic_ego_unfiltered_sensors,
            # Filtered semantic static points in CURRENT ego frame (i)
            "lidar_pc_ego_ref": pc_ego_ref_save,  # Filtered static points transformed to REFERENCE ego frame
            # "lidar_pc_with_semantic_ego_ref": pc_with_semantic_ego_ref_save,
            # Filtered semantic static points transformed to REFERENCE ego frame
            "lidar_pc_global": pc_global_save,  # Filtered static points transformed to GLOBAL frame
            # "lidar_pc_with_semantic_global": pc_with_semantic_global_save,
            # Filtered semantic static points transformed to GLOBAL frame
            "ego_pose": ego_pose_i,  # Current frame's ego pose dictionary
            # "lidar_calibrated_sensor": ref_calibrated_sensor # Uncomment if needed
            # Add other necessary fields like ego_ref_from_ego_i if needed later for ICP refinement logic
            "ego_ref_from_ego_i": ego_ref_from_ego_i,
            # Store originals if your ICP logic needs them (Optional based on full implementation)
            # "original_pc_ego": pc_ego_unfiltered.copy(),
            # "original_pc_with_semantic_ego": pc_with_semantic_ego_unfiltered.copy(),
        }

        # append the dictionary to list
        dict_list.append(frame_dict)  # appends dictionary containing frame data to the list dict_list
        # After iterating through the entire scene, this list will contain information for all frames in the scene

    ################# Prepare Lists for Static Scene Points (in Reference Ego Frame) ########################

    # These lists will hold points already transformed into the reference ego frame
    # using the ORIGINAL (unrefined) poses calculated in the previous loop.
    # The subsequent ICP step (if enabled) should refine the poses and potentially
    # regenerate these lists with higher accuracy.

    print("Extracting static points previously transformed to reference ego frame...")

    # Extract static points (already in ref ego frame, Features x N format)
    # Use the correct key from the dictionary populated earlier
    unrefined_pc_ego_list = [frame_dict['lidar_pc_ego_i'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_ego_list)} static point clouds (in ego i frame).")

    unrefined_pc_ego_list_sensor_ids = [frame_dict['lidar_pc_ego_sensor_ids'] for frame_dict in dict_list]

    # Extract semantic static points (already in ref ego frame, Features+1 x N format)
    # Use the correct key from the dictionary populated earlier
    # unrefined_sem_pc_ego_list = [frame_dict['lidar_pc_with_semantic_ego_i'] for frame_dict in dict_list]
    # print(f"Extracted {len(unrefined_sem_pc_ego_list)} semantic point clouds (in ego i frame).")

    # unrefined_sem_pc_ego_list_sensor_ids = [frame_dict['lidar_pc_with_semantic_ego_sensor_ids'] for frame_dict in
      #                                      dict_list]

    pc_ego_combined_draw = np.concatenate(unrefined_pc_ego_list, axis=0)
    print(f"Pc ego i combined shape: {pc_ego_combined_draw.shape}")

    ######################## Visualization #################################################
    if args.vis_aggregated_static_ego_i_pc:
        pc_ego_to_draw = o3d.geometry.PointCloud()
        pc_coordinates = pc_ego_combined_draw[:, :3]
        pc_ego_to_draw.points = o3d.utility.Vector3dVector(pc_coordinates)
        o3d.visualization.draw_geometries([pc_ego_to_draw], window_name="Combined static point clouds (in ego i frame)")
    ########################################################################################

    unrefined_pc_ego_ref_list = [frame_dict['lidar_pc_ego_ref'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_ego_ref_list)} static point clouds (in ego ref frame).")
    unrefined_pc_ego_ref_list_sensor_ids = [frame_dict['lidar_pc_ego_sensor_ids'] for frame_dict in dict_list]
    # unrefined_sem_pc_ego_ref_list = [frame_dict['lidar_pc_with_semantic_ego_ref'] for frame_dict in dict_list]
    # print(f"Extracted {len(unrefined_sem_pc_ego_ref_list)} semantic static point clouds (in ego ref frame).")
    # unrefined_sem_pc_ego_ref_list_sensor_ids = [frame_dict['lidar_pc_with_semantic_ego_sensor_ids'] for frame_dict in
      #                                          dict_list]

    pc_ego_ref_combined_draw = np.concatenate(unrefined_pc_ego_ref_list, axis=0)
    print(f"Pc ego ref combined shape: {pc_ego_ref_combined_draw.shape}")

    ###################### Visualization ##################################################
    if args.vis_aggregated_static_ego_ref_pc:
        pc_ego_ref_to_draw = o3d.geometry.PointCloud()
        pc_ego_ref_coordinates = pc_ego_ref_combined_draw[:, :3]
        pc_ego_ref_to_draw.points = o3d.utility.Vector3dVector(pc_ego_ref_coordinates)
        o3d.visualization.draw_geometries([pc_ego_ref_to_draw],
                                          window_name="Combined static point clouds (in ego ref frame)")
    #######################################################################################

    unrefined_pc_global_list = [frame_dict['lidar_pc_global'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_global_list)} static point clouds (in world frame).")
    # unrefined_sem_pc_global_list = [frame_dict['lidar_pc_with_semantic_global'] for frame_dict in dict_list]
    # print(f"Extracted {len(unrefined_sem_pc_global_list)} semantic static point clouds (in world frame).")

    pc_global_combined_draw = np.concatenate(unrefined_pc_global_list, axis=0)
    print(f"Pc global shape: {pc_global_combined_draw.shape}")

    ####################### Visualization ################################################
    if args.vis_aggregated_static_global_pc:
        pc_global_to_draw = o3d.geometry.PointCloud()
        pc_global_coordinates = pc_global_combined_draw[:, :3]
        pc_global_to_draw.points = o3d.utility.Vector3dVector(pc_global_coordinates)
        o3d.visualization.draw_geometries([pc_global_to_draw],
                                          window_name="Combined static point clouds (in global frame)")
    ######################################################################################

    raw_pc_list = [frame_dict['raw_lidar_ego'] for frame_dict in dict_list]
    print(f"Extracted {len(raw_pc_list)} static and dynamic point clouds (in ego i frame).")
    raw_pc_list_sensor_ids = [frame_dict['raw_lidar_ego_sensor_ids'] for frame_dict in dict_list]
    print(f"Extracted {len(raw_pc_list_sensor_ids)} sensor ids for static and dynamic point clouds (in ego i frame).")
    raw_pc_draw = np.concatenate(raw_pc_list, axis=0)
    print(f"Raw Pc with static and dynamic points shape: {raw_pc_draw.shape}")

    ########################## Visualization #############################################
    if args.vis_aggregated_raw_pc_ego_i:
        raw_pc_to_draw = o3d.geometry.PointCloud()
        raw_pc_coordinates = raw_pc_draw[:, :3]
        raw_pc_to_draw.points = o3d.utility.Vector3dVector(raw_pc_coordinates)
        o3d.visualization.draw_geometries([raw_pc_to_draw],
                                          window_name="Combined static and dynamic point clouds (in ego i frame)")

    ######################################################################################
    ##################### Prepare lidar timestamps for Kiss-ICP ##########################
    # Extract timestamps associated with each frame's original ego pose
    try:
        # Adjust key if needed based on your ego_pose dictionary structure
        lidar_timestamps = [frame_dict['sample_timestamp'] for frame_dict in dict_list]
        print(f"Extracted {len(lidar_timestamps)} timestamps.")
    except KeyError:
        print("Timestamp key not found in ego_pose, setting lidar_timestamps to None.")
        lidar_timestamps = None  # Fallback

    print(f"Lidar timestamps: {lidar_timestamps}")

    ######################### Process ego_ref_from_ego_i for kissicp #############################
    if not dict_list:
        print("dict_list is empty. Cannot proceed with pose comparison.")
    else:
        gt_relative_poses_list = [fd['ego_ref_from_ego_i'] for fd in dict_list]
        gt_relative_poses_arr = np.array(gt_relative_poses_list)  # Shape: (num_frames, 4, 4)
        print(f"Collected {gt_relative_poses_arr.shape[0]} GT relative poses for comparison.")

    ##############################################################################################
    ##################################### MapMOS #################################################

    # mos_config = MOSConfig(voxel_size_belief=0.25, delay_mos=10, ...)
    # odometry_config = OdometryConfig(voxel_size=0.5, ...)
    # data_config = DataConfig(max_range=100.0, ...)
    # pipeline_config = { "mos": mos_config, "odom": odometry_config, "data": data_config }

    in_memory_dataset_mapmos = None
    mapmos_pipeline = None
    estimated_poses_kiss = None
    log_dir_mapmos = osp.join(save_path, scene_name, "mapmos_logs")

    """mapmos_labels_per_scan = []
    for point_cloud_in_frame in raw_pc_list:
        num_points_in_frame = point_cloud_in_frame.shape[0]
        random_labels_for_frame = np.random.randint(0, 2, size=(num_points_in_frame, 1), dtype=np.int32)
        mapmos_labels_per_scan.append(random_labels_for_frame)"""

    mapmos_labels_per_scan = [frame_dict['mapmos_per_point_labels'] for frame_dict in dict_list]

    try:
        print("Initializing InMemoryDatasetMapMOS...")

        in_memory_dataset_mapmos = InMemoryDatasetMapMOS(
            lidar_scans=raw_pc_list,
            scan_timestamps=lidar_timestamps,
            labels_per_scan=mapmos_labels_per_scan,
            gt_global_poses=gt_relative_poses_arr,  # Optional
            sequence_id=f"{scene_name}_mapmos_run"
        )
        print(f"InMemoryDatasetMapMOS initialized with {len(in_memory_dataset_mapmos)} scans.")
    except Exception as e:
        print(f"Error creating InMemoryDatasetMapMos: {e}. Skipping MapMOS.")

    config_path_mapmos = None
    if in_memory_dataset_mapmos:
        try:
            print(f"Initializing MapMOSPipeline with weights: {weights_path_mapmos}")

            if not weights_path_mapmos.is_file():
                raise FileNotFoundError(f"MapMOS weights not found at: {weights_path_mapmos}")

            mapmos_pipeline = MapMOSPipeline(
                dataset=in_memory_dataset_mapmos,
                weights=weights_path_mapmos,
                config=config_path_mapmos,
                visualize=False,
                save_ply=True,
                save_kitti=False,
                n_scans=-1,
                jump=0
            )

            print("MapMOS pipeline initialized.")

            mapmos_start_time = time.time()
            print("Running MapMOS pipeline...")
            run_output, all_frame_predictions = mapmos_pipeline.run()
            mapmos_end_time = time.time()
            print(f"MapMOS pipeline finished in {mapmos_end_time - mapmos_start_time:.2f} seconds.")

            # Process or print results
            if hasattr(run_output, 'print') and callable(getattr(run_output, 'print')):
                print("Printing MapMOS results:")
                run_output.print()
            else:
                print(
                    "MapMOS run completed. Inspect 'mapmos_pipeline_instance' or 'run_output' for results if available.")
                # For example, the results might be saved to files if save_ply=True,
                # or accessible via methods on mapmos_pipeline_instance.

            # Now you can iterate through all_frame_predictions
            """for frame_data in all_frame_predictions:
                scan_idx = frame_data["scan_index"]
                points = frame_data["points"]
                predicted_labels = frame_data["predicted_labels"]  # These are your belief_labels_query
                gt_labels = frame_data["gt_labels"]  # These are your query_labels

                print(
                    f"Scan Index: {scan_idx}, Points shape: {points.shape}, Predicted Labels shape: {predicted_labels.shape}, GT Labels shape: {gt_labels.shape}")

                visualize_mapmos_predictions(
                    points_xyz=points,
                    predicted_labels=predicted_labels,
                    scan_index=scan_idx,
                    window_title_prefix=f"MapMOS Output ({scene_name})"  # Using your scene_name
                )

                # Example: Filter dynamic points based on MapMOS prediction
                # Assuming 1 means dynamic, 0 means static in predicted_labels
                dynamic_points = points[predicted_labels == 1]
                static_points = points[predicted_labels == 0]
                print(f"  - Found {dynamic_points.shape[0]} dynamic points and {static_points.shape[0]} static points.")"""

        except Exception as e:
            print(f"Error during MapMOS pipeline execution: {e}")

    static_points_mapmos = []
    static_points_mapmos_sensor_ids = []

    if all_frame_predictions:
        print(f"\n--- Filtering static points for all {len(all_frame_predictions)} frames ---")
        for frame_idx, frame_data in enumerate(all_frame_predictions):
            original_points_in_frame = raw_pc_list[frame_idx]
            original_points_in_frame_sensor_ids = raw_pc_list_sensor_ids[frame_idx]

            # Condition 1: Point was considered STATIC in your input GT to MapMOS
            input_gt_is_static_mask = (mapmos_labels_per_scan[frame_idx].flatten() == 0)

            # Condition 2: Point was NOT predicted as DYNAMIC by MapMOS
            mapmos_predicted_labels = frame_data["predicted_labels"]
            mapmos_not_predicted_dynamic_mask = (
                    mapmos_predicted_labels != 1)  # True if MapMOS predicted 0 (static) or -1 (unclassified)

            # Combine both conditions with logical AND
            final_static_mask_for_frame = input_gt_is_static_mask & mapmos_not_predicted_dynamic_mask

            # Apply the mask to get the static points for the current frame
            static_points_this_frame = original_points_in_frame[final_static_mask_for_frame]
            static_points_mapmos.append(static_points_this_frame)
            static_points_this_frame_sensor_ids = original_points_in_frame_sensor_ids[final_static_mask_for_frame]
            static_points_mapmos_sensor_ids.append(static_points_this_frame_sensor_ids)

    print(f"Static points list mapmos: {len(static_points_mapmos)} frames with sensor ids: {len(static_points_mapmos_sensor_ids)}")

    """raw_pc_0 = raw_pc_list[0]

    # Condition 1: Point is NOT dynamic in your GT input to MapMOS
    input_gt_is_static_mask = (mapmos_labels_per_scan[0].flatten() == 0)

    # Condition 2: Point is NOT classified as dynamic by MapMOS prediction
    mapmos_predicted_labels = all_frame_predictions[0]["predicted_labels"]
    mapmos_not_predicted_dynamic_mask = (
                mapmos_predicted_labels != 1)  # True if MapMOS predicted 0 (static) or -1 (unclassified)

    # Combine both conditions: A point is truly static if both are true
    final_static_mask = input_gt_is_static_mask & mapmos_not_predicted_dynamic_mask

    # Filter raw_pc_0 using this final static mask
    static_points_from_raw_pc_0 = raw_pc_0[final_static_mask]

    print(f"--- Filtering for Frame 0 ---")
    print(f"Original number of points in raw_pc_0: {raw_pc_0.shape[0]}")
    print(f"Number of points static according to your input GT: {np.sum(input_gt_is_static_mask)}")
    print(f"Number of points NOT predicted as dynamic by MapMOS: {np.sum(mapmos_not_predicted_dynamic_mask)}")
    print(f"Number of points satisfying BOTH static conditions: {static_points_from_raw_pc_0.shape[0]}")


    pcd_static_vis = o3d.geometry.PointCloud()
    pcd_static_vis.points = o3d.utility.Vector3dVector(static_points_from_raw_pc_0[:, :3])
    pcd_static_vis.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for these static points
    o3d.visualization.draw_geometries([pcd_static_vis],
                                      window_name=f"Final Static Points (Frame 0)")"""


    poses_kiss_icp = None
    ###############################################################################################
    ################# Refinement using KISS-ICP ###################################################
    if args.icp_refinement and len(dict_list) > 1:
        print(f"--- Performing KISS-ICP refinement on static global point clouds for scene {scene_name} ---")

        # --- 1. Prepare Dataset and Pipeline ---
        in_memory_dataset = None
        pipeline = None
        estimated_poses_kiss = None  # Will hold the results from pipeline.poses
        log_dir_kiss = osp.join(save_path, scene_name, "kiss_icp_logs")

        try:
            in_memory_dataset = InMemoryDataset(
                lidar_scans=raw_pc_list,
                gt_relative_poses=gt_relative_poses_arr,
                timestamps=lidar_timestamps,
                # Use a descriptive sequence ID, incorporating scene name if possible
                sequence_id=f"{scene_name}_icp_run",
                log_dir=log_dir_kiss
            )
            print(f"Created InMemoryDataset with {len(in_memory_dataset)} scans.")

        except Exception as e:
            print(f"Error creating InMemoryDataset: {e}. Skipping refinement.")
            args.icp_refinement = False  # Disable refinement

        if args.icp_refinement and in_memory_dataset:
            try:
                kiss_config_path = Path('kiss_config.yaml')  # Ensure this file exists
                pipeline = OdometryPipeline(dataset=in_memory_dataset, config=kiss_config_path)
                print("KISS-ICP pipeline initialized.")
            except Exception as e:
                print(f"Error initializing KISS-ICP: {e}. Skipping refinement.")
                args.icp_refinement = False  # Disable refinement if init fails

        # --- 2. Run KISS-ICP Pipeline ---
        if args.icp_refinement and pipeline is not None:  # Check again in case init failed
            kiss_start_time = time.time()
            print("Running KISS-ICP pipeline...")
            try:
                # --- THIS IS THE CORRECT WAY TO RUN THE PIPELINE ---
                results = pipeline.run()
                # ----------------------------------------------------
                kiss_end_time = time.time()
                print(f"KISS-ICP pipeline finished. Time: {kiss_end_time - kiss_start_time:.2f} sec.")
                print("Pipeline Results:", results)

                # --- Get the calculated poses AFTER run() completes ---
                estimated_poses_kiss = pipeline.poses  # This is the NumPy array (NumScans, 4, 4)
                poses_kiss_icp = pipeline.poses

                # Basic check on returned poses
                if not isinstance(estimated_poses_kiss, np.ndarray) or \
                        estimated_poses_kiss.shape != (len(raw_pc_list), 4, 4):
                    print(f"Error: Unexpected pose results shape {estimated_poses_kiss.shape}. "
                          f"Expected ({len(raw_pc_list)}, 4, 4). Skipping refinement application.")
                    args.icp_refinement = False  # Disable further steps

            except Exception as e:
                print(f"Error during KISS-ICP pipeline run: {e}. Skipping refinement application.")
                args.icp_refinement = False  # Disable further steps if run fails
                import traceback
                traceback.print_exc()

        # --- 3. Apply Refined Poses to Original Point Clouds (Only if ICP succeeded) ---
        if args.icp_refinement and estimated_poses_kiss is not None:
            print("Applying refined poses from KISS-ICP...")
            refined_lidar_pc_list = []
            # refined_lidar_pc_with_semantic_list = []

            for idx, points_ego in enumerate(unrefined_pc_ego_list): #(static_points_mapmos): # (unrefined_pc_ego_list):
                pose = estimated_poses_kiss[idx]
                print(f"Applying refined pose {idx}: {pose}")

                print(f"Points ego shape: {points_ego.shape}")

                # points_xyz = points_ego.T[:, :3]
                points_xyz = points_ego[:, :3]
                points_homo = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
                points_transformed = (pose @ points_homo.T)[:3, :].T

                if points_ego.shape[1] > 3:
                    # other_features = points_ego.T[:, 3:]
                    other_features = points_ego[:, 3:]
                    points_transformed = np.hstack((points_transformed, other_features))

                refined_lidar_pc_list.append(points_transformed)

            """for idx, points_semantic_ego in enumerate(unrefined_sem_pc_ego_list):
                pose = estimated_poses_kiss[idx]
                print(f"Applying refined pose {idx}: {pose}")

                print(f"Points semantic ego shape: {points_semantic_ego.shape}")
                points_xyz = points_semantic_ego[:, :3]
                points_homo = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
                points_transformed = (pose @ points_homo.T)[:3, :].T

                if points_semantic_ego.shape[0] > 3:
                    other_features = points_semantic_ego[:, 3:]
                    points_transformed = np.hstack((points_transformed, other_features))

                refined_lidar_pc_with_semantic_list.append(points_transformed)"""

        # --- 4. Compare KISS-ICP Poses with Ground Truth ---
        if 'gt_relative_poses_arr' in locals() and gt_relative_poses_arr.shape[0] > 0:  # Check if GT poses were loaded
            if poses_kiss_icp.shape[0] == gt_relative_poses_arr.shape[0]:
                print("\n--- Comparing KISS-ICP Poses with Ground Truth Poses ---")
                trans_errors = []
                rot_errors_rad = []  # Store rotational errors in radians
                trans_errors_x = []
                trans_errors_y = []
                trans_errors_z = []

                kiss_relative_poses_arr = poses_kiss_icp

                for k_idx in range(kiss_relative_poses_arr.shape[0]):
                    pose_kiss_k = kiss_relative_poses_arr[k_idx]  # Pose of frame k in KISS-ICP's frame 0 system
                    pose_gt_k = gt_relative_poses_arr[k_idx]  # Pose of frame k in dataset's frame 0 system

                    # Translational error
                    t_kiss = pose_kiss_k[:3, 3]
                    t_gt = pose_gt_k[:3, 3]

                    # Translational error vector (GT - Estimated)
                    t_error_vec = t_gt - t_kiss
                    trans_errors_x.append(t_error_vec[0])
                    trans_errors_y.append(t_error_vec[1])
                    trans_errors_z.append(t_error_vec[2])

                    # Overall translational error
                    trans_error = np.linalg.norm(t_error_vec)  # Same as np.linalg.norm(t_gt - t_kiss)
                    trans_errors.append(trans_error)

                    # Rotational error
                    R_kiss = pose_kiss_k[:3, :3]
                    R_gt = pose_gt_k[:3, :3]

                    # Relative rotation: R_error = inv(R_kiss) @ R_gt
                    # or R_error = R_gt @ R_kiss.T (if R_kiss is orthogonal, its transpose is its inverse)
                    R_error = R_kiss.T @ R_gt

                    # Angle from rotation matrix trace
                    # angle = arccos((trace(R_error) - 1) / 2)
                    trace_val = np.trace(R_error)
                    # Clip to avoid domain errors with arccos due to numerical inaccuracies for values slightly outside [-1, 1]
                    clipped_arg = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
                    rot_error_rad = np.arccos(clipped_arg)
                    rot_errors_rad.append(rot_error_rad)

                avg_trans_error = np.mean(trans_errors)
                median_trans_error = np.median(trans_errors)
                avg_rot_error_deg = np.mean(np.degrees(rot_errors_rad))
                median_rot_error_deg = np.median(np.degrees(rot_errors_rad))

                # Calculate statistics for component-wise errors
                mae_trans_error_x = np.mean(np.abs(trans_errors_x))
                mae_trans_error_y = np.mean(np.abs(trans_errors_y))
                mae_trans_error_z = np.mean(np.abs(trans_errors_z))

                mean_trans_error_x = np.mean(trans_errors_x)  # To see bias
                mean_trans_error_y = np.mean(trans_errors_y)  # To see bias
                mean_trans_error_z = np.mean(trans_errors_z)  # To see bias

                print(f"Sequence: {scene_name}")
                print(f"  Average Translational Error : {avg_trans_error:.4f} m")
                print(f"  Median Translational Error  : {median_trans_error:.4f} m")
                print(f"  Average Rotational Error    : {avg_rot_error_deg:.4f} degrees")
                print(f"  Median Rotational Error     : {median_rot_error_deg:.4f} degrees")
                print(f"  MAE X: {mae_trans_error_x:.4f} m (Mean X Bias: {mean_trans_error_x:+.4f} m)")
                print(f"  MAE Y: {mae_trans_error_y:.4f} m (Mean Y Bias: {mean_trans_error_y:+.4f} m)")
                print(f"  MAE Z: {mae_trans_error_z:.4f} m (Mean Z Bias: {mean_trans_error_z:+.4f} m)")

                # Plotting (updated to 2x2 layout)
                fig, axs = plt.subplots(2, 2, figsize=(17, 10))  # Adjusted figsize for 2x2
                fig.suptitle(f'Scene {scene_name}: KISS-ICP vs GT Relative Pose Errors', fontsize=16)

                # Top-left: Overall Translational Error
                axs[0, 0].plot(trans_errors, label="Overall Trans. Error")
                axs[0, 0].set_title('Overall Translational Error')
                axs[0, 0].set_ylabel('Error (m)')
                axs[0, 0].grid(True)
                axs[0, 0].legend()
                axs[0, 0].set_xlabel('Frame Index')

                # Top-right: Overall Rotational Error
                axs[0, 1].plot(np.degrees(rot_errors_rad), label="Overall Rot. Error")
                axs[0, 1].set_title('Overall Rotational Error')
                axs[0, 1].set_ylabel('Error (degrees)')
                axs[0, 1].grid(True)
                axs[0, 1].legend()
                axs[0, 1].set_xlabel('Frame Index')

                # Bottom-left: X and Y Translational Errors
                axs[1, 0].plot(trans_errors_x, label="X Error (GT - Est)", alpha=0.9)
                axs[1, 0].plot(trans_errors_y, label="Y Error (GT - Est)", alpha=0.9)
                axs[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.7, label="Zero Error")
                axs[1, 0].set_title('X & Y Translational Component Errors')
                axs[1, 0].set_xlabel('Frame Index')
                axs[1, 0].set_ylabel('Error (m)')
                axs[1, 0].grid(True)
                axs[1, 0].legend()

                # Bottom-right: Z Translational Error
                axs[1, 1].plot(trans_errors_z, label="Z Error (GT - Est)")
                axs[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.7, label="Zero Error")
                axs[1, 1].set_title('Z Translational Component Error')
                axs[1, 1].set_xlabel('Frame Index')
                axs[1, 1].set_ylabel('Error (m)')
                axs[1, 1].grid(True)
                axs[1, 1].legend()

                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for subtitle

                plot_save_dir = Path(args.save_path) / scene_name / "kiss_icp_logs"
                plot_save_dir.mkdir(parents=True, exist_ok=True)
                plot_filename = plot_save_dir / f"errors_scene_{scene_name}.png"
                plt.savefig(plot_filename)
                print(f"Saved pose error plot to {plot_filename}")

                if args.pose_error_plot:
                    plt.show()
                plt.close(fig)

            else:
                print(f"Warning: Number of KISS-ICP poses ({poses_kiss_icp.shape[0]}) "
                      f"does not match GT relative poses ({gt_relative_poses_arr.shape[0]}). Cannot compare.")

        elif not ('gt_relative_poses_arr' in locals() and gt_relative_poses_arr.shape[0] > 0):
            print("GT relative poses not available for comparison.")



                    # Determine the source lists for aggregation based on ICP refinement
    if not args.icp_refinement:
        print("ICP refinement is OFF. Using unrefined points (in reference ego frame) for aggregation.")
        # These are lists of (N, D) arrays, already in reference ego frame
        source_pc_list_all_frames = unrefined_pc_ego_ref_list
        # source_sem_pc_list_all_frames = unrefined_sem_pc_ego_ref_list
        source_pc_sids_list_all_frames = unrefined_pc_ego_ref_list_sensor_ids
        # source_sem_sids_list_all_frames = unrefined_sem_pc_ego_ref_list_sensor_ids
    else:
        print("ICP refinement is ON. Using KISS-ICP refined points for aggregation.")
        # These are lists of (N, D) arrays, in the KISS-ICP refined global/map frame
        source_pc_list_all_frames = refined_lidar_pc_list
        # source_sem_pc_list_all_frames = refined_lidar_pc_with_semantic_list
        # SIDs lists for refined PCs are typically the same as their unrefined counterparts,
        # as ICP only affects poses, not point identities or origins relative to sensor.
        # IMPORTANT: Ensure these unrefined SIDs lists correspond to the frames in refined_lidar_pc_list
        source_pc_sids_list_all_frames = unrefined_pc_ego_list_sensor_ids  # SIDs from ego_i list
        # source_pc_sids_list_all_frames = static_points_mapmos_sensor_ids
        # source_sem_sids_list_all_frames = unrefined_sem_pc_ego_list_sensor_ids  # SIDs from ego_i list

    #################################### Filtering based on if only keyframes should be used ###########################
    print(f"Static map aggregation: --static_map_keyframes_only is {args.static_map_keyframes_only}")

    lidar_pc_list_for_concat = []
    lidar_pc_sids_list_for_concat = []
    # lidar_pc_with_semantic_list_for_concat = []
    # lidar_pc_with_semantic_sids_list_for_concat = []

    for idx, frame_info in enumerate(dict_list):
        is_key = frame_info['is_key_frame']
        # Decide whether to include this frame's points in the static map
        include_in_static_map = True
        if args.static_map_keyframes_only and not is_key:
            include_in_static_map = False

        if include_in_static_map:
            print(f"  Including frame {idx} (Keyframe: {is_key}) in static map aggregation.")
            # Add static points
            if idx < len(source_pc_list_all_frames) and source_pc_list_all_frames[idx].shape[0] > 0:
                lidar_pc_list_for_concat.append(source_pc_list_all_frames[idx])
                if idx < len(source_pc_sids_list_all_frames) and source_pc_sids_list_all_frames[idx].shape[0] > 0:
                    lidar_pc_sids_list_for_concat.append(source_pc_sids_list_all_frames[idx])
                elif source_pc_list_all_frames[idx].shape[
                    0] > 0:  # Points exist but SIDs might be empty if something went wrong
                    print(
                        f"Warning: Frame {idx} has {source_pc_list_all_frames[idx].shape[0]} static points but missing/empty SIDs.")

            """# Add semantic static points
            if idx < len(source_sem_pc_list_all_frames) and source_sem_pc_list_all_frames[idx].shape[0] > 0:
                lidar_pc_with_semantic_list_for_concat.append(source_sem_pc_list_all_frames[idx])
                if idx < len(source_sem_sids_list_all_frames) and source_sem_sids_list_all_frames[idx].shape[
                    0] > 0:
                    lidar_pc_with_semantic_sids_list_for_concat.append(source_sem_sids_list_all_frames[idx])
                elif source_sem_pc_list_all_frames[idx].shape[0] > 0:
                    print(
                        f"Warning: Frame {idx} has {source_sem_pc_list_all_frames[idx].shape[0]} semantic points but missing/empty SIDs.")"""
        else:
            print(
                f"  Skipping frame {idx} (Keyframe: {is_key}) for static map aggregation due to --static_map_keyframes_only.")

    ###################################################################################################################

    if lidar_pc_list_for_concat:
        print(f"Concatenating pc from {len(lidar_pc_list_for_concat)} frames")
        # Concatenate along points axis (axis=0)
        lidar_pc_final_global = np.concatenate(lidar_pc_list_for_concat, axis=0)
        print(f"Concatenated refined static global points. Shape: {lidar_pc_final_global.shape}")
    else:
        sys.exit()

    if lidar_pc_sids_list_for_concat:
        print(f"Concatenating pc sensor ids from {len(lidar_pc_sids_list_for_concat)} frames")
        lidar_pc_final_global_sensor_ids = np.concatenate(lidar_pc_sids_list_for_concat, axis=0)
        print(f"Concatenated refined static global point sensor ids. Shape: {lidar_pc_final_global_sensor_ids.shape}")
    else:
        sys.exit()

    assert lidar_pc_final_global.shape[0] == lidar_pc_final_global_sensor_ids.shape[0]

    ################################## Visualization #############################################################
    if args.vis_aggregated_static_kiss_refined:
        visualize_pointcloud(lidar_pc_final_global, title=f"Aggregated Refined Static PC (Global) - Scene {scene_name}")
    #############################################################################################################

    ################## concatenate all semantic scene segments ########################
    """if lidar_pc_with_semantic_list_for_concat:
        print(f"Concatenating semantic pc from {len(lidar_pc_with_semantic_list_for_concat)} frames")
        lidar_pc_with_semantic_final_global = np.concatenate(lidar_pc_with_semantic_list_for_concat,
                                                             axis=0)  # Shape (N_total, Features)
        print(f"Concatenated refined semantic global points. Shape: {lidar_pc_with_semantic_final_global.shape}")
    else:
        sys.exit()

    if lidar_pc_with_semantic_sids_list_for_concat:
        print(f"Concatenating pc from {len(lidar_pc_with_semantic_sids_list_for_concat)} frames")
        lidar_pc_with_semantic_final_global_sensor_ids = np.concatenate(lidar_pc_with_semantic_sids_list_for_concat, axis=0)
        print(f"Concatenated refined semantic static global point sensor ids. Shape: {lidar_pc_with_semantic_final_global_sensor_ids.shape}")
    else:
        sys.exit()

    assert lidar_pc_with_semantic_final_global.shape[0] == lidar_pc_with_semantic_final_global_sensor_ids.shape[0]"""

    ####################################################################################################################

    lidar_pc = lidar_pc_final_global.T
    # lidar_pc_with_semantic = lidar_pc_with_semantic_final_global.T

    lidar_pc_sensor_ids = lidar_pc_final_global_sensor_ids
    # lidar_pc_with_semantic_sensor_ids = lidar_pc_with_semantic_final_global_sensor_ids

    ########################################### Visualization #########################################################
    if args.vis_static_frame_comparison_kiss_refined:
        # Define frames to visualize
        frame_indices_to_viz = [1, 25]  # Frame 1 (index 0), Frame 25 (index 24)
        colors = [
            [1.0, 0.0, 0.0],  # Red for frame 1
            [0.0, 0.0, 1.0]  # Blue for frame 25
        ]

        # Check if the list is long enough
        if len(lidar_pc_list_for_concat) <= max(frame_indices_to_viz):
            print(
                f"Error: refined_lidar_pc_list only has {len(lidar_pc_list_for_concat)} elements. Cannot access frame {max(frame_indices_to_viz) + 1}.")
            # Handle error appropriately, maybe exit or skip visualization
            sys.exit(1)  # Or 'pass' if you want to continue without this visualization

        point_clouds_to_visualize = []

        for i, frame_idx in enumerate(frame_indices_to_viz):
            # Get the point cloud for the specific frame
            pc_np = lidar_pc_list_for_concat[frame_idx]

            # Ensure it's not empty and has the right dimensions
            if pc_np is None or pc_np.shape[0] == 0:
                print(f"Warning: Point cloud for frame {frame_idx + 1} (index {frame_idx}) is empty. Skipping.")
                continue
            if pc_np.ndim != 2 or pc_np.shape[1] < 3:
                print(
                    f"Warning: Point cloud for frame {frame_idx + 1} (index {frame_idx}) has unexpected shape {pc_np.shape}. Needs (N, >=3). Skipping.")
                continue

            # Extract XYZ coordinates (assuming they are the first 3 columns)
            xyz = pc_np[:, :3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # Assign color
            pcd.paint_uniform_color(colors[i])

            point_clouds_to_visualize.append(pcd)

        # Visualize if we have point clouds to show
        if point_clouds_to_visualize:
            print(
                f"Visualizing Frame {frame_indices_to_viz[0] + 1} (Red) and Frame {frame_indices_to_viz[1] + 1} (Blue)...")
            o3d.visualization.draw_geometries(
                point_clouds_to_visualize,
                window_name=f"Scene {scene_name} - Frame {frame_indices_to_viz[0] + 1} (Red) & {frame_indices_to_viz[1] + 1} (Blue)",
                width=800,
                height=600
            )
        else:
            print("No valid point clouds found for the selected frames to visualize.")
    ###############################################################################################################

    ################################## Filtering of static aggregated point cloud #####################################
    if args.filter_aggregated_static_pc and args.filter_mode != 'none':
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(lidar_pc[:3, :].T)
        filtered_pcd_o3d, kept_indices = denoise_pointcloud(pcd_o3d, args.filter_mode, config,
                                                            location_msg="aggregated static points")
        lidar_pc = lidar_pc[:, kept_indices]
        lidar_pc_sensor_ids = lidar_pc_sensor_ids[kept_indices]
        #lidar_pc_with_semantic = lidar_pc_with_semantic[:,
         #                        kept_indices]  # Only here if lidar_pc and lidar_pc_semantics are the same
        #lidar_pc_with_semantic_sensor_ids = lidar_pc_with_semantic_sensor_ids[kept_indices]


    """if args.filter_aggregated_static_pc and args.filter_mode != 'none':
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(lidar_pc_with_semantic[:3, :].T)
        denoise_pointcloud.location_msg = "aggregated static semantic points"
        filtered_pcd_o3d, kept_indices = denoise_pointcloud(pcd_o3d, args.filter_mode, config)
        lidar_pc_with_semantic = lidar_pc_with_semantic[:, kept_indices]"""  # Only needed if lidar_pc and lidar_pc are different pcs
    ###################################################################################################################
    # ensure each point has a sensor id
    assert lidar_pc.shape[1] == lidar_pc_sensor_ids.shape[0], (
        f"point count ({lidar_pc.shape[1]}) != sensor_ids count ({lidar_pc_sensor_ids.shape[0]})"
    )

    """# ensure each semantic point has a sensor id
    assert lidar_pc_with_semantic.shape[1] == lidar_pc_with_semantic_sensor_ids.shape[0], (
        f"semantic point count ({lidar_pc_with_semantic.shape[1]}) != "
        f"semantic_sensor_ids count ({lidar_pc_with_semantic_sensor_ids.shape[0]})"
    )"""

    ############################## Visualization ######################################################

    if args.vis_filtered_aggregated_static:
        visualize_pointcloud(lidar_pc.T, title=f"Aggregated Refined Static PC (Global) - Scene {scene_name}")
    ###################################################################################################

    source_dict_list_for_objects = dict_list

    if args.dynamic_map_keyframes_only:
        print("Dynamic object aggregation will use ONLY KEYFRAMES.")
        source_dict_list_for_objects = [fd for fd in dict_list if fd['is_key_frame']]
        if not source_dict_list_for_objects:
            print(
                "Warning: --dynamic_map_keyframes_only is set, but no keyframes found in dict_list. Object data will be empty.")
    else:
        print("Dynamic object aggregation will use ALL FRAMES.")

    print(f"Dynamic object aggregation will use {len(source_dict_list_for_objects)} frames.")

    ############################# Old aggregation logic #################################
    """
    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = []  # stores unique object tokens from all frames
    object_semantic = []  # stores semantic category corresponding to each unique object
    for frame_dict in source_dict_list_for_objects:  # Iterate through frames and collect unique objects
        for i, object_token in enumerate(frame_dict['object_tokens']):
            if object_token not in object_token_zoo:  # Filter and append object tokens
                if (frame_dict['object_points_list'][i].shape[
                    0] > 0):  # only appends objects that have at least one point
                    object_token_zoo.append(object_token)
                    object_semantic.append(frame_dict['converted_object_category'][i])
                else:
                    continue

    object_points_dict = {}  # initialize an empty dictionary to hold aggregated object points
    object_sids_dict = {}

    print("\nAggregating object points...")
    for query_object_token in tqdm(object_token_zoo,
                                   desc="Aggregating Objects"):  # Loop through each unique object token
        canonical_segments_list = []
        canonical_segments_sids_list = []
        for frame_dict in source_dict_list_for_objects:  # iterates through all frames
            if query_object_token in frame_dict['object_tokens']:  # Check if the object exists in this frame
                obj_idx = frame_dict['object_tokens'].index(query_object_token)  # Find its index
                object_points = frame_dict['object_points_list'][obj_idx]  # retrieve raw object points
                object_points_sids = frame_dict['object_points_list_sensor_ids'][obj_idx]

                if object_points is not None and object_points.shape[0] > 0:
                    # Canonicalization: Translate to center and rotate based on box yaw
                    center = frame_dict['gt_bbox_3d'][obj_idx][:3]
                    yaw = frame_dict['gt_bbox_3d'][obj_idx][6]
                    translated_points = object_points[:, :3] - center  # Use only XYZ
                    if translated_points.shape[0] > 0:
                        Rot = Rotation.from_euler('z', -yaw, degrees=False)
                        rotated_object_points = Rot.apply(translated_points)  # Canonical segment (N_seg, 3)
                        if rotated_object_points.shape[0] > 0:
                            canonical_segments_list.append(rotated_object_points)  # Add segment to the list
                            canonical_segments_sids_list.append(object_points_sids)

        if canonical_segments_list:
            object_points_dict[query_object_token] = np.concatenate(canonical_segments_list, axis=0)
            object_sids_dict[query_object_token] = np.concatenate(canonical_segments_sids_list, axis=0)
        else:
            object_points_dict[query_object_token] = np.zeros((0, 3), dtype=float)
            object_sids_dict[query_object_token] = np.zeros((0,), dtype=int)

    object_data = []  # Stores all unique objects across frames

    for idx, token in enumerate(object_token_zoo):
        obj = {
            "token": token,
            "points": object_points_dict[token][:, :3],  # Canonicalized x, y, z
            "points_sids": object_sids_dict[token],
            "semantic": object_semantic[idx]  # Integer category
        }
        object_data.append(obj)
    """

    ###################################### Loop to save data per frame ########################################
    # Loop over frames to process scene
    tranforms_ref_to_k = []
    i = 0
    while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames to avoid infinite loops
        print(f"Processing frame {i} for saving")
        if i >= len(dict_list):  # If the frame index exceeds the number of frames in the scene exit the function
            print('finish scene!')
            return
        frame_dict = dict_list[i]  # retrieves dictionary corresponding to the current frame
        # Only processes key frames since non-key frames do not have ground truth annotations
        is_key_frame = frame_dict['is_key_frame']
        if not is_key_frame:  # only use key frame as GT
            i = i + 1
            continue  # skips to the next frame if not a key frame

        # extract points from transformed point clouds
        point_cloud = lidar_pc[:3,
                      :]  # extract transformed static points, T to switch dims from (3,N) to (N, 3)
        # point_cloud_with_semantic = lidar_pc_with_semantic  # retrieves transformed semantic points with labels (N, 4): [x, y, z, label]
        point_cloud_sensor_ids = lidar_pc_sensor_ids
        # point_cloud_with_semantic_sensor_ids = lidar_pc_with_semantic_sensor_ids
        print(f"Original point_cloud: {point_cloud.shape} with sensor ids: {point_cloud_sensor_ids.shape}")
        #print(
         #   f"Original semantic point_cloud: {point_cloud_with_semantic.shape} with sensor ids: {point_cloud_with_semantic_sensor_ids.shape}")

        # === Transform back to ego frame ===
        sample = truckscenes.get('sample', frame_dict['sample_token'])
        ego_pose = truckscenes.getclosest('ego_pose', sample['timestamp'])
        ego_from_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=True)

        sensor_origins = []
        for idx, sensor in enumerate(sensors):
            sd = trucksc.get('sample_data', sample['data'][sensor])
            cs = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])

            T_s_to_ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
            sensor_origins.append(T_s_to_ego[:3, 3])
        sensor_origins = np.stack(sensor_origins, axis=0)  # shape (S,3)

        if args.icp_refinement:
            T_ref_from_k = poses_kiss_icp[i]
            T_k_from_ref = np.linalg.inv(T_ref_from_k)
            tranforms_ref_to_k.append(T_k_from_ref)
        else:
            T_ref_from_k = frame_dict['ego_ref_from_ego_i']
            T_k_from_ref = np.linalg.inv(T_ref_from_k)
            tranforms_ref_to_k.append(T_ref_from_k)

        # Get global point clouds for this iteration (use copies to avoid modifying originals accidentally)
        point_cloud_global_xyz = lidar_pc[:3, :].copy()  # Global static XYZ (3, N)
        # point_cloud_global_xyzl = lidar_pc_with_semantic.copy()  # Global static XYZ+L (4+, N)
        print(f"Global point_cloud shape: {point_cloud_global_xyz.shape}")
        # print(f"Global semantic point_cloud shape: {point_cloud_global_xyzl.shape}")

        # --- Transform scene static points (point_cloud) ---
        if point_cloud_global_xyz.shape[1] > 0:
            points_homo = np.hstack((point_cloud_global_xyz.T, np.ones((point_cloud_global_xyz.shape[1], 1))))  # (N, 4)
            points_k_ego = (T_k_from_ref @ points_homo.T).T[:, :3]  # Shape (N, 4)
            # Assign the result to the loop-local variable 'point_cloud'
            point_cloud = points_k_ego.T  # Ego coords (3, N)
        else:
            point_cloud = np.zeros((3, 0), dtype=point_cloud_global_xyz.dtype)

        """# --- Transform semantic points as well (point_cloud_with_semantic) ---
        if point_cloud_global_xyzl.shape[1] > 0:
            # Create homogeneous coords from the global semantic points' XYZ
            sem_homo = np.hstack(
                (point_cloud_global_xyzl[:3, :].T, np.ones((point_cloud_global_xyzl.shape[1], 1))))  # (N, 4)
            # Perform transformation to get ego XYZ
            sem_ego_xyz = (T_k_from_ref @ sem_homo.T).T[:, :3]  # (N, 3) - contains ego XYZ

            # --- Create the NEW ego semantic array ---
            # Combine the new ego XYZ with the original labels/features (from row 3 onwards)
            point_cloud_with_semantic_ego = np.vstack((
                sem_ego_xyz.T,  # New Ego XYZ (3, N)
                point_cloud_global_xyzl[3:, :]  # Original Labels/Features (1+, N)
            ))
            # Assign the newly created array to the loop-local variable
            point_cloud_with_semantic = point_cloud_with_semantic_ego  # Ego coords + Labels (4+, N)

        else:
            # Handle empty case - ensure correct number of rows if needed
            num_sem_rows = lidar_pc_with_semantic.shape[0]  # Get original number of rows
            point_cloud_with_semantic = np.zeros((num_sem_rows, 0), dtype=lidar_pc_with_semantic.dtype)"""

        # Prints after transformation:
        print(f"Ego point_cloud shape: {point_cloud.shape} with sensor ids: {point_cloud_sensor_ids.shape}")
        # Optional: print first few points' ego coords: print(point_cloud[:,:5])
        """print(
            f"Ego semantic point_cloud shape: {point_cloud_with_semantic.shape} with sensor ids: {point_cloud_with_semantic_sensor_ids.shape}")
        # Optional: print first few points' ego coords + labels: print(point_cloud_with_semantic[:,:5])

        assert point_cloud.shape[1] == point_cloud_with_semantic.shape[1], \
            f"Ego point counts mismatch after transform: {point_cloud.shape[1]} vs {point_cloud_with_semantic.shape[1]}"
        """
        assert point_cloud.shape[1] == point_cloud_sensor_ids.shape[0]

        """################## load bbox of target frame ##############
        sample = truckscenes.get('sample', frame_dict['sample_token'])
        boxes = get_boxes(truckscenes, sample)

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)

        boxes_token = [box.token for box in boxes]  # retrieves a list of tokens from the bounding box
        # Extract object tokens. Each instance token represents a unique object

        # Extract object tokens. Each instance token represents a unique object
        object_tokens = [truckscenes.get('sample_annotation', box_token)['instance_token'] for box_token in
                         boxes_token]  # Uses sample_annotation data to get instance_token fore each bb
        # Extract object categories
        object_category = [truckscenes.get('sample_annotation', box_token)['category_name'] for box_token in
                           boxes_token]  # retrieves category name for each bounding box

        ############################# get object categories ##########################
        converted_object_category = []  # Initialize empty list

        # Iterate over each object category extracted earlier
        for category in object_category:
            found_match = False
            # Iterate over label mappings defined in truckscenes.yaml file
            for label_key, label_name_in_yaml in truckscenesyaml['labels'].items():
                # Check category and map to learning label
                if category == label_name_in_yaml:
                    mapped_label_index = learning_map.get(label_key)
                    if mapped_label_index is not None:
                        converted_object_category.append(mapped_label_index)
                        found_match = True
                        break  # Found the mapping, move to the next category
                    else:
                        # This case means label_key exists in 'labels' but not 'learning_map'
                        print(
                            f"Warning: Category '{category}' mapped to label_key '{label_key}', but '{label_key}' not found in learning_map. Using 'Unknown' label.")
                        # --- CHANGE: Use UNKNOWN_LEARNING_INDEX ---
                        converted_object_category.append(UNKNOWN_LEARNING_INDEX)
                        found_match = True
                        break

            # If the category was not found in the truckscenesyaml['labels'] mapping at all
            if not found_match:
                print(
                    f"Warning: Category '{category}' not found in truckscenes.yaml mapping. Using 'Unknown' label.")
                # --- CHANGE: Use UNKNOWN_LEARNING_INDEX ---
                converted_object_category.append(UNKNOWN_LEARNING_INDEX)"""

        """labels_array = np.array(converted_object_category).reshape(-1, 1)

        # Check if the number of labels matches the number of boxes
        if locs.shape[0] == labels_array.shape[0]:
            # Concatenate geometric data AND the labels array
            # Resulting shape will be (N, 3+3+1+1) = (N, 8)
            gt_bbox_3d_with_labels = np.concatenate([locs, dims, rots, labels_array], axis=1).astype(np.float32)

            # ---- EXPORT gt_bbox_3d (derived from boxes) ----
            # Create a unique filename using the frame index 'i'
            gt_bbox_filename = f'frame_{i}_gt_bbox_3d_labeled.npy'
            dirs = os.path.join(save_path, scene_name, frame_dict['sample_token'])  #### Save in folder with scene name
            if not os.path.exists(dirs):  # create directory if does not exist
                os.makedirs(dirs)
            output_filepath_bbox = os.path.join(dirs, gt_bbox_filename)
            np.save(output_filepath_bbox, gt_bbox_3d_with_labels)
            print(f"Saved bounding box data for frame {i} to {output_filepath_bbox}")
            print(f"Saved array shape: {gt_bbox_3d_with_labels.shape}")
        else:
            print(
                f"ERROR: Mismatch between number of boxes ({locs.shape[0]}) and number of labels ({labels_array.shape[0]}) for frame {i}. Skipping save.")

            # ------------------------------------------------"""

        ################ load boxes of target frame ############################
        gt_bbox_3d_unmodified = frame_dict['gt_bbox_3d_unmodified']
        boxes = frame_dict['boxes_ego']

        object_tokens = frame_dict['object_tokens']
        converted_object_category = frame_dict['converted_object_category']

        ################################################
        percentage_factor = 1.20  # Target increase of 20%
        max_absolute_increase_m = 0.4  # The maximum total increase for any dimension (e.g., 20cm per side)
        # Get the original dimensions [w, l, h]
        dims = gt_bbox_3d_unmodified[:, 3:6].copy()

        # 1. Calculate the dimensions if extended by the percentage factor
        dims_extended_by_percentage = dims * percentage_factor

        # 2. Calculate the maximum allowed dimensions based on the absolute cap
        dims_with_max_absolute_increase = dims + max_absolute_increase_m

        # 3. For each dimension, take the SMALLER of the two options
        new_dims = np.minimum(dims_extended_by_percentage, dims_with_max_absolute_increase)

        gt_bbox_3d = gt_bbox_3d_unmodified.copy()
        gt_bbox_3d[:, 3:6] = new_dims

        gt_bbox_3d[:, 6] += np.pi / 2.  # adjust yaw angles by 90 degrees
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        # gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.05  # Experiment
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction

        ################################################

        """# Create a copy to modify
        gt_bbox_3d = gt_bbox_3d_unmodified.copy()

        dims = gt_bbox_3d[:, 3:6]

        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] -= 0.1
        gt_bbox_3d[:, 3:6] *= 1.1"""

        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################## bbox placement ##############
        current_keyframe_object_tokens = frame_dict['object_tokens']
        current_keyframe_gt_bboxes_unmodified = frame_dict['gt_bbox_3d_unmodified']
        current_keyframe_object_categories = frame_dict['converted_object_category']

        # These lists will store the final point clouds for dynamic objects for THIS scene
        dynamic_object_points_list = []  # Final transformed points for scene
        dynamic_object_points_sids_list = []
        dynamic_object_points_semantic_list = []  # Final semantic-labeled points for scene
        dynamic_object_points_semantic_sids_list = []

        # Iterate through each object present in THIS keyframe `i`
        for keyframe_obj_idx, current_object_token in enumerate(current_keyframe_object_tokens):
            # Get the object's class ID and name for logging
            class_id = current_keyframe_object_categories[keyframe_obj_idx]
            class_name = learning_id_to_name.get(class_id, 'Unknown')  # Safely get the name
            short_token = current_object_token.split('-')[0]

            print(f"  Processing dynamic object: {class_name} ('{short_token}...') for keyframe {i}...")

            # --- 1. Get the interaction signature for the object in THIS keyframe ---
            # We pass `frame_dict` which is `dict_list[i]`
            signature_at_keyframe_i = get_object_overlap_signature_BATCH(frame_dict, keyframe_obj_idx)

            if signature_at_keyframe_i:
                # Create a temporary map to resolve tokens to names for the signature string
                token_to_name_map = {
                    tok: learning_id_to_name.get(cat_id, 'Unknown')
                    for tok, cat_id in zip(current_keyframe_object_tokens, current_keyframe_object_categories)
                }
                # Now build the descriptive signature string
                sig_str = ", ".join([f"({iou:.4f}, {learning_id_to_name.get(cat_id, '?')}, {yaw:.4f} rad)"
                                     for iou, tok, cat_id, yaw in signature_at_keyframe_i])
                print(f"    Signature for '{class_name}': [{sig_str}]")

            # These will collect points from all frames with a similar context
            contextual_canonical_segments = []
            contextual_canonical_sids_segments = []

            # --- 2. Find all frames with a similar context and collect their points ---
            for candidate_frame_data in source_dict_list_for_objects:
                if current_object_token in candidate_frame_data['object_tokens']:
                    candidate_obj_idx = candidate_frame_data['object_tokens'].index(current_object_token)

                    # Get the category IDs for the target object in both the keyframe and candidate frame
                    category_id_keyframe = current_keyframe_object_categories[keyframe_obj_idx]
                    category_id_candidate = candidate_frame_data['converted_object_category'][candidate_obj_idx]

                    # Get the signatures for both
                    signature_at_candidate_frame = get_object_overlap_signature_BATCH(candidate_frame_data,
                                                                                     candidate_obj_idx)

                    # Get the bounding box parameters for both the keyframe and candidate frame object
                    bbox_params_keyframe = current_keyframe_gt_bboxes_unmodified[keyframe_obj_idx]
                    bbox_params_candidate = candidate_frame_data['gt_bbox_3d_unmodified'][candidate_obj_idx]
                    points_keyframe = frame_dict['object_points_list'][keyframe_obj_idx]
                    points_candidate = candidate_frame_data['object_points_list'][candidate_obj_idx]

                    # Compare the keyframe's signature with the candidate's signature
                    if (compare_signatures_class_based(
                            signature_at_keyframe_i, category_id_keyframe,
                            signature_at_candidate_frame, category_id_candidate,
                            CLASS_BASED_THRESHOLDS
                    ) and
                            are_box_sizes_similar(bbox_params_keyframe, bbox_params_candidate) and
                            is_point_centroid_z_similar(points_keyframe, points_candidate,
                                                        category_id_keyframe, OTHER_VEHICLE_ID,
                                                        z_tolerance=0.5)):
                        # Contexts match! Get the points from this candidate frame.
                        obj_points_in_candidate = candidate_frame_data['object_points_list'][candidate_obj_idx]
                        obj_sids_in_candidate = candidate_frame_data['object_points_list_sensor_ids'][candidate_obj_idx]

                        if obj_points_in_candidate is not None and obj_points_in_candidate.shape[0] > 0:
                            # Canonicalize the points using the candidate frame's bbox parameters
                            bbox_params_cand = candidate_frame_data['gt_bbox_3d_unmodified'][candidate_obj_idx]
                            center_cand, yaw_cand = bbox_params_cand[:3], bbox_params_cand[6]

                            translated_pts = obj_points_in_candidate[:, :3] - center_cand
                            Rot_can = Rotation.from_euler('z', -yaw_cand, degrees=False)
                            canonical_pts = Rot_can.apply(translated_pts)

                            contextual_canonical_segments.append(canonical_pts)
                            contextual_canonical_sids_segments.append(obj_sids_in_candidate)

            num_aggregated_frames = len(contextual_canonical_segments)
            # Use the class_name in the log
            print(f"      -> For object '{class_name} ({short_token})', found {num_aggregated_frames} similar context frame(s).")

            # --- 3. Aggregate the collected points and place them in the scene ---
            aggregated_canonical_points = np.zeros((0, 3), dtype=float)
            aggregated_canonical_sids = np.zeros((0,), dtype=int)

            if contextual_canonical_segments:
                aggregated_canonical_points = np.concatenate(contextual_canonical_segments, axis=0)
                aggregated_canonical_sids = np.concatenate(contextual_canonical_sids_segments, axis=0)
                print(f"      -> Aggregated {aggregated_canonical_points.shape[0]} points for this context.")
            else:
                # FALLBACK: If no similar contexts are found, use only the points from the current keyframe
                # This prevents an object from disappearing if its context is unique.
                print(f"      -> No similar contexts found. Using points from keyframe {i} only.")
                obj_points_in_keyframe = frame_dict['object_points_list'][keyframe_obj_idx]
                if obj_points_in_keyframe is not None and obj_points_in_keyframe.shape[0] > 0:
                    # Canonicalize points from the current keyframe itself
                    bbox_params_keyframe = current_keyframe_gt_bboxes_unmodified[keyframe_obj_idx]
                    center_key, yaw_key = bbox_params_keyframe[:3], bbox_params_keyframe[6]
                    translated = obj_points_in_keyframe[:, :3] - center_key
                    Rot_key = Rotation.from_euler('z', -yaw_key, degrees=False)
                    aggregated_canonical_points = Rot_key.apply(translated)
                    aggregated_canonical_sids = frame_dict['object_points_list_sensor_ids'][keyframe_obj_idx]

            # --- 4. De-canonicalize and store for final scene assembly ---
            if aggregated_canonical_points.shape[0] > 0:
                # De-canonicalize using the pose from the CURRENT keyframe `i`
                bbox_params_decanon = current_keyframe_gt_bboxes_unmodified[keyframe_obj_idx]
                decanon_center, decanon_yaw = bbox_params_decanon[:3], bbox_params_decanon[6]

                Rot_decan = Rotation.from_euler('z', decanon_yaw, degrees=False)
                points_in_scene_xyz = Rot_decan.apply(aggregated_canonical_points) + decanon_center

                # Keep track of the sensor IDs for these points
                sids_for_points = aggregated_canonical_sids

                bbox_for_filtering = gt_bbox_3d[keyframe_obj_idx: keyframe_obj_idx + 1]

                original_count = points_in_scene_xyz.shape[0]

                # Only filter if there are points to filter
                if original_count > 0:
                    points_in_box_mask = points_in_boxes_cpu(
                        torch.from_numpy(points_in_scene_xyz[np.newaxis, :, :]),
                        torch.from_numpy(bbox_for_filtering[np.newaxis, :, :])
                    )[0, :, 0].bool()

                    # Apply the mask to the points and their corresponding sensor IDs
                    points_in_scene_xyz = points_in_scene_xyz[points_in_box_mask]
                    sids_for_points = sids_for_points[points_in_box_mask]

                    filtered_count = points_in_scene_xyz.shape[0]
                    if original_count > filtered_count:
                        print(f"      [Final Clip] Object {class_name}: Clipped aggregated points from {original_count} to {filtered_count}.")

                # Add to lists for final concatenation
                dynamic_object_points_list.append(points_in_scene_xyz)
                dynamic_object_points_sids_list.append(sids_for_points)

                # Create the semantic version
                semantic_label = current_keyframe_object_categories[keyframe_obj_idx]
                semantic_column = np.full((points_in_scene_xyz.shape[0], 1), semantic_label,
                                          dtype=points_in_scene_xyz.dtype)
                points_with_semantics = np.concatenate([points_in_scene_xyz, semantic_column], axis=1)
                dynamic_object_points_semantic_list.append(points_with_semantics)
                dynamic_object_points_semantic_sids_list.append(sids_for_points)

        # --- Final Scene Assembly for frame i ---
        print("\n  Assembling final scene for saving...")

        """# Find matching object entry
        for obj in object_data:
            if current_object_token == obj["token"]:
                points = obj["points"]  # Canonical (x, y, z)
                points_sids = obj["points_sids"]

                # Rotate and translate object points to place back into current scene
                Rot = Rotation.from_euler('z', rots[keyframe_obj_idx], degrees=False)
                rotated_object_points = Rot.apply(points)
                points = rotated_object_points + locs[keyframe_obj_idx]  # Translate to bbox center

                # Filter points inside bounding box
                if points.shape[0] >= 5:
                    original_count = points.shape[0]
                    points_in_boxes = points_in_boxes_cpu(
                        torch.from_numpy(points[np.newaxis, :, :]),
                        torch.from_numpy(gt_bbox_3d[keyframe_obj_idx:keyframe_obj_idx + 1][np.newaxis, :])
                    )
                    points = points[points_in_boxes[0, :, 0].bool()]
                    points_sids = points_sids[points_in_boxes[0, :, 0].bool()]

                    assert points.shape[0] == points_sids.shape[0]

                    filtered_count = points.shape[0]
                    if original_count > filtered_count:
                        print(
                            f"[Box Filter] Object {current_object_token}: reduced points from {original_count} to {filtered_count}")

                # Append only if points remain
                if points.shape[0] > 0:
                    dynamic_object_points_list.append(points)
                    dynamic_object_points_sids_list.append(points_sids)

                    semantics = np.ones((points.shape[0], 1)) * obj["semantic"]
                    points_with_semantics = np.concatenate([points, semantics], axis=1)
                    dynamic_object_semantic_list.append(points_with_semantics)
                    dynamic_object_semantic_sids_list.append(points_sids)

                break  # No need to check further once matched"""

        if args.vis_static_before_combined_dynamic:
            visualize_pointcloud_bbox(point_cloud.T,
                                      boxes=boxes,
                                      title=f"Fused static PC before combining with dynamic points + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)

        """temp_points_filename = f'frame_{i}_temp_points.npy'
        sem_temp_points_filename = f'frame_{i}_sem_temp_points.npy'
        dirs = os.path.join(save_path, scene_name, frame_dict['sample_token'])  #### Save in folder with scene name
        if not os.path.exists(dirs):  # create directory if does not exist
            os.makedirs(dirs)
        output_filepath_temp = os.path.join(dirs, temp_points_filename)
        output_filepath_sem_temp = os.path.join(dirs, sem_temp_points_filename)"""
        # Combine Scene Points with Object Points
        try:  # avoid concatenate an empty array
            temp = np.concatenate(dynamic_object_points_list)
            temp_sids = np.concatenate(dynamic_object_points_sids_list)
            """# ---- EXPORT temp points ----
            np.save(output_filepath_temp, temp)
            print(f"Saved temp object points for frame {i} to {output_filepath_temp}")
            # ----------------------------"""
            # scene_points = point_cloud.T
            scene_points = np.concatenate(
                [point_cloud.T, temp])  # Merge static scene points and object points

            scene_points_sids = np.concatenate([point_cloud_sensor_ids, temp_sids])
        except:
            print("Error concatenating static and object points.")
            scene_points = point_cloud  # If no object points, only use static scene points
            scene_points_sids = point_cloud_sensor_ids
        try:
            temp = np.concatenate(dynamic_object_points_semantic_list)
            temp_sids = np.concatenate(dynamic_object_points_semantic_sids_list)

            pts_xyz = point_cloud.T

            num_points_pc = point_cloud.shape[1]
            points_label = np.full((num_points_pc, 1), BACKGROUND_LEARNING_INDEX, dtype=np.uint8)
            point_cloud_with_semantic = np.concatenate([pts_xyz, points_label], axis=1)

            point_cloud_with_semantic_sensor_ids = point_cloud_sensor_ids.copy()
            """# ---- EXPORT temp points ----
            np.save(output_filepath_sem_temp, temp)
            print(f"Saved temp object points for frame {i} to {output_filepath_sem_temp}")
            # ----------------------------"""
            # scene_semantic_points = point_cloud_with_semantic.T
            scene_semantic_points = np.concatenate(
                [point_cloud_with_semantic, temp])  # Merge semantic points from objects and static scenes
            scene_semantic_points_sids = np.concatenate([point_cloud_with_semantic_sensor_ids, temp_sids])
        except:
            print("Error concatenating semantic static and object points.")
            pts_xyz = point_cloud.T
            background_label = np.full((pts_xyz.shape[0], 1), BACKGROUND_LEARNING_INDEX, dtype=np.uint8)
            scene_semantic_points = np.concatenate([pts_xyz, background_label], axis=1)
            scene_semantic_points_sids = point_cloud_sensor_ids.copy()
            #scene_semantic_points = point_cloud_with_semantic  # If no object points, only use static scene points
            #scene_semantic_points_sids = point_cloud_with_semantic_sensor_ids

        assert scene_points_sids.shape == scene_semantic_points_sids.shape

        print(f"Scene points before applying range filtering: {scene_points.shape}")
        mask = in_range_mask(scene_points, pc_range)

        scene_points = scene_points[mask]

        print(f"Scene points sids applying range filtering: {scene_points_sids.shape}")
        scene_points_sids = scene_points_sids[mask]

        print(f"Scene points after applying range filtering: {scene_points.shape}")

        ################################## Visualize #####################################################
        """visualize_pointcloud_bbox(scene_points,
                                  boxes=boxes,
                                  title=f"Fused dynamic and static PC + BBoxes + Ego BBox - Frame {i}",
                                  self_vehicle_range=self_range,
                                  vis_self_vehicle=True)"""
        if args.vis_combined_static_dynamic_pc:
            visualize_pointcloud_bbox(scene_points,
                                      boxes=boxes,
                                      title=f"Fused dynamic and static PC + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)
        ################################################################################################

        ################## get semantics of sparse points  ##############
        print(f"Scene semantic points before applying range filtering: {scene_semantic_points.shape}")
        mask = in_range_mask(scene_semantic_points, pc_range)

        scene_semantic_points = scene_semantic_points[mask]  # Filter points within a spatial range
        scene_semantic_points_sids = scene_semantic_points_sids[mask]
        print(f"Scene semantic points after applying range filtering: {scene_semantic_points.shape}")

        ################################## Filtering #################################################
        if args.filter_combined_static_dynamic_pc and args.filter_mode != 'none':
            print(f"Filtering {scene_points.shape}")
            pcd_to_filter = o3d.geometry.PointCloud()
            pcd_to_filter.points = o3d.utility.Vector3dVector(scene_points[:, :3])

            filtered_pcd, kept_indices = denoise_pointcloud(pcd_to_filter, args.filter_mode, config,
                                                 location_msg="final scene points")
            scene_points = np.asarray(filtered_pcd.points)
            scene_points_sids = scene_points_sids[kept_indices]
            scene_semantic_points = scene_semantic_points[kept_indices]
            scene_semantic_points_sids = scene_semantic_points_sids[kept_indices]
        ############################################################################################

        # ensure each combined scene point has a sensor id
        assert scene_points.shape[0] == scene_points_sids.shape[0], (
            f"scene_points count ({scene_points.shape[0]}) != scene_points_sids count ({scene_points_sids.shape[0]})"
        )

        # ensure each semantic scene point has a sensor id
        assert scene_semantic_points.shape[0] == scene_semantic_points_sids.shape[0], (
            f"scene_semantic_points count ({scene_semantic_points.shape[0]}) != "
            f"scene_semantic_points_sids count ({scene_semantic_points_sids.shape[0]})"
        )

        ############################ Meshing #######################################################
        if args.meshing:
            print("--- Starting Meshing and Voxelization---")
            print(f"Shape of scene points before meshing: {scene_points.shape}")

            ################## get mesh via Possion Surface Reconstruction ##############
            point_cloud_original = o3d.geometry.PointCloud()  # Initialize point cloud object
            with_normal2 = o3d.geometry.PointCloud()  # Initialize point cloud object
            point_cloud_original.points = o3d.utility.Vector3dVector(
                scene_points[:, :3])  # converts scene_points array into open3D point cloud format
            with_normal = preprocess(point_cloud_original, config)  # Uses the preprocess function to compute normals
            with_normal2.points = with_normal.points  # copies the processed points and normals to another point clouds
            with_normal2.normals = with_normal.normals  # copies the processed points and normals to another point clouds

            # Generate mesh from point cloud using Poisson Surface Reconstruction
            mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                           config['min_density'], with_normal2)
            scene_points = np.asarray(mesh.vertices, dtype=float)

            print(f"Shape of scene points after meshing: {scene_points.shape}")

            ################## remain points with a spatial range ##############
            mask_meshing = in_range_mask(scene_points, pc_range)

            scene_points = scene_points[mask_meshing]  # Filter points within a spatial range

            pcd_meshed_vox_indices = scene_points.copy()

            pcd_meshed_vox_indices[:, 0] = (pcd_meshed_vox_indices[:, 0] - pc_range[0]) / voxel_size
            pcd_meshed_vox_indices[:, 1] = (pcd_meshed_vox_indices[:, 1] - pc_range[1]) / voxel_size
            pcd_meshed_vox_indices[:, 2] = (pcd_meshed_vox_indices[:, 2] - pc_range[2]) / voxel_size
            pcd_meshed_vox_indices = np.floor(pcd_meshed_vox_indices).astype(int)

            pcd_meshed_vox_indices[:, 0] = np.clip(pcd_meshed_vox_indices[:, 0], 0, occ_size[0] - 1)
            pcd_meshed_vox_indices[:, 1] = np.clip(pcd_meshed_vox_indices[:, 1], 0, occ_size[1] - 1)
            pcd_meshed_vox_indices[:, 2] = np.clip(pcd_meshed_vox_indices[:, 2], 0, occ_size[2] - 1)

            binary_voxel_grid = np.zeros(occ_size, dtype=bool)
            unique_meshed_voxel_indices = np.unique(pcd_meshed_vox_indices[:, :3], axis=0)
            if unique_meshed_voxel_indices.shape[0] > 0:
                binary_voxel_grid[
                    unique_meshed_voxel_indices[:, 0], unique_meshed_voxel_indices[:, 1], unique_meshed_voxel_indices[:,
                                                                                          2]] = True
            # Get voxel indices (vx,vy,vz) of occupied cells
            x_occ_idx, y_occ_idx, z_occ_idx = np.where(binary_voxel_grid)
            fov_voxel_indices = np.stack([x_occ_idx, y_occ_idx, z_occ_idx], axis=-1)

            if fov_voxel_indices.shape[0] == 0:
                print(
                    "No fov_voxels (occupied centers) found after voxelizing mesh. Occupancy grid will be empty/free.")
            else:
                # Convert these occupied voxel indices to world coordinates (centers of voxels)
                fov_voxels_world_xyz = fov_voxel_indices.astype(float)
                fov_voxels_world_xyz[:, :3] = (fov_voxels_world_xyz[:, :3] + 0.5) * voxel_size
                fov_voxels_world_xyz[:, 0] += pc_range[0]
                fov_voxels_world_xyz[:, 1] += pc_range[1]
                fov_voxels_world_xyz[:, 2] += pc_range[2]

                # Assign semantics using Chamfer distance
                dense_voxels_needing_labels_world = fov_voxels_world_xyz

                if scene_semantic_points.shape[0] == 0:
                    print(
                        "WARNING: scene_semantic_points is empty for Chamfer. Labeled occupancy will be based on FREE_LEARNING_INDEX.")
                    # occupancy_grid is already initialized to FREE
                else:
                    print(
                        f"Chamfer - Dense (target needing labels) world shape: {dense_voxels_needing_labels_world.shape}")
                    print(f"Chamfer - Sparse (source with labels) world shape: {scene_semantic_points.shape}")

                    x_chamfer = torch.from_numpy(dense_voxels_needing_labels_world).cuda().unsqueeze(0).float()
                    y_chamfer = torch.from_numpy(scene_semantic_points[:, :3]).cuda().unsqueeze(
                        0).float()  # Use XYZ for distance

                    _, _, idx1_chamfer, _ = chamfer.forward(x_chamfer, y_chamfer)
                    indices_from_chamfer = idx1_chamfer[0].cpu().numpy()

                    assigned_labels_for_dense = scene_semantic_points[
                        indices_from_chamfer, 3]  # Assuming label is 4th col (index 3)

                    # Combine world coords of dense voxels with their new labels
                    dense_voxels_world_with_semantic = np.concatenate(
                        [dense_voxels_needing_labels_world, assigned_labels_for_dense[:, np.newaxis]], axis=1
                    )

                    # Convert these world points (now with labels) to final voxel coordinates [vx,vy,vz,label]
                    temp_voxel_coords = dense_voxels_world_with_semantic.copy()
                    temp_voxel_coords[:, 0] = (temp_voxel_coords[:, 0] - pc_range[0]) / voxel_size
                    temp_voxel_coords[:, 1] = (temp_voxel_coords[:, 1] - pc_range[1]) / voxel_size
                    temp_voxel_coords[:, 2] = (temp_voxel_coords[:, 2] - pc_range[2]) / voxel_size

                    # Store integer voxel coordinates and labels
                    dense_voxels_with_semantic_voxelcoords = np.zeros_like(temp_voxel_coords, dtype=int)
                    dense_voxels_with_semantic_voxelcoords[:, :3] = np.floor(temp_voxel_coords[:, :3]).astype(int)
                    dense_voxels_with_semantic_voxelcoords[:, 3] = temp_voxel_coords[:, 3].astype(int)  # Labels column

                    # Clip to ensure within bounds
                    dense_voxels_with_semantic_voxelcoords[:, 0] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 0],
                                                                           0, occ_size[0] - 1)
                    dense_voxels_with_semantic_voxelcoords[:, 1] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 1],
                                                                           0, occ_size[1] - 1)
                    dense_voxels_with_semantic_voxelcoords[:, 2] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 2],
                                                                           0, occ_size[2] - 1)

                    # Populate the final occupancy_grid
                    # Handle cases where multiple dense points might map to the same voxel cell
                    # by taking the label of the first one encountered (due to np.unique)
                    unique_final_vox_indices, unique_idx_map = np.unique(dense_voxels_with_semantic_voxelcoords[:, :3],
                                                                         axis=0, return_index=True)
                    labels_for_unique_final_voxels = dense_voxels_with_semantic_voxelcoords[unique_idx_map, 3]

                    # Initialize 3D occupancy grid with the "Unknown" or "Background" label (e.g. 16)
                    occupancy_grid = np.full(occ_size, FREE_LEARNING_INDEX, dtype=np.uint8)

                    if unique_final_vox_indices.shape[0] > 0:
                        occupancy_grid[
                            unique_final_vox_indices[:, 0],
                            unique_final_vox_indices[:, 1],
                            unique_final_vox_indices[:, 2]
                        ] = labels_for_unique_final_voxels

                    occupied_mask = occupancy_grid != FREE_LEARNING_INDEX
                    total_occupied_voxels = np.sum(occupied_mask)

                    if np.any(occupied_mask):  # Check if there are any occupied voxels
                        vx, vy, vz = np.where(occupied_mask)  # Get the indices (vx, vy, vz) of all occupied voxels
                        labels_at_occupied = occupancy_grid[vx, vy, vz]  # Get the labels at these occupied locations

                        # Stack them into the [vx, vy, vz, label] format
                        dense_voxels_with_semantic_voxelcoords_save = np.stack([vx, vy, vz, labels_at_occupied],
                                                                               axis=-1).astype(int)
                    else:
                        # If no voxels are occupied (e.g., the entire grid is FREE_LEARNING_INDEX)
                        dense_voxels_with_semantic_voxelcoords_save = np.zeros((0, 4), dtype=int)

        else:
            print("--- Starting Voxelization without Meshing ---")
            if scene_semantic_points.shape[0] == 0:
                print("No semantic points available. Occupancy grid will be empty/free.")
                # occupancy_grid already initialized to FREE, dense_voxels_with_semantic_voxelcoords is empty
            else:

                print("Creating Lidar visibility masks")

                # Directly use scene_semantic_points (which are XYZL)
                # Convert their world XYZ to voxel coordinates, keep their labels
                points_to_voxelize = scene_semantic_points.copy()

                labels = points_to_voxelize[:, 3].astype(int)  # Assuming label is 4th col

                voxel_indices_float = np.zeros_like(points_to_voxelize[:, :3])
                voxel_indices_float[:, 0] = (points_to_voxelize[:, 0] - pc_range[0]) / voxel_size
                voxel_indices_float[:, 1] = (points_to_voxelize[:, 1] - pc_range[1]) / voxel_size
                voxel_indices_float[:, 2] = (points_to_voxelize[:, 2] - pc_range[2]) / voxel_size

                voxel_indices_int = np.floor(voxel_indices_float).astype(int)

                dense_voxels_with_semantic_voxelcoords = np.concatenate(
                    [voxel_indices_int, labels[:, np.newaxis]], axis=1
                )

                # Clip to ensure within bounds
                dense_voxels_with_semantic_voxelcoords[:, 0] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 0], 0,
                                                                       occ_size[0] - 1)
                dense_voxels_with_semantic_voxelcoords[:, 1] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 1], 0,
                                                                       occ_size[1] - 1)
                dense_voxels_with_semantic_voxelcoords[:, 2] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 2], 0,
                                                                       occ_size[2] - 1)

                # Initialize 3D occupancy grid with the "Unknown" or "Background" label (e.g. 16)
                occupancy_grid = np.full(occ_size, FREE_LEARNING_INDEX, dtype=np.uint8)
                # Populate the final occupancy_grid
                # If multiple original semantic points fall into the same voxel, the last one's label will apply.
                if dense_voxels_with_semantic_voxelcoords.shape[0] > 0:
                    occupancy_grid[
                        dense_voxels_with_semantic_voxelcoords[:, 0],
                        dense_voxels_with_semantic_voxelcoords[:, 1],
                        dense_voxels_with_semantic_voxelcoords[:, 2]
                    ] = dense_voxels_with_semantic_voxelcoords[:, 3]

                # pick the proper origin for each LiDAR hit by indexing with your per‐point sensor‐ID
                #    scene_points_sids is (N,) telling which sensor produced each point
                points_origin = sensor_origins[scene_semantic_points_sids]  # (N,3)
                print(f"Points origin shape: {points_origin.shape}")
                points_label = scene_semantic_points[:, 3].astype(int)
                print(f"Points label shape: {points_label.shape}")
                points = scene_semantic_points[:, :3]
                print(f"Points shape: {points.shape}")

                print("Creating Lidar visibility masks using GPU...")
                # --- Time the GPU execution ---
                print("\nTiming GPU Lidar visibility calculation...")
                start_time_gpu = time.perf_counter()

                # Call the GPU host function
                voxel_state_gpu, voxel_label_gpu = calculate_lidar_visibility_gpu_host(
                    points_cpu=points,  # Your (N,3) hits
                    points_origin_cpu=points_origin,  # Your (N,3) original sensor origins
                    points_label_cpu=points_label,  # Your (N,) semantic labels (ensure int32)
                    pc_range_cpu_list=pc_range,  # Your [xmin,ymin,zmin,xmax,ymax,zmax] list
                    voxel_size_cpu_scalar=voxel_size,  # Your scalar voxel_size from config
                    spatial_shape_cpu_list=occ_size,  # Your [Dx,Dy,Dz] list from config
                    occupancy_grid_cpu=occupancy_grid,  # Your pre-computed (Dx,Dy,Dz) aggregated occupancy (uint8)
                    FREE_LEARNING_INDEX_cpu=FREE_LEARNING_INDEX,  # Your semantic index for free space
                    FREE_LABEL_placeholder_cpu=-1,  # The internal placeholder for initializing labels on GPU
                    points_sensor_indices_cpu=scene_semantic_points_sids.astype(np.int32),
                    sensor_max_ranges_cpu=sensor_max_ranges_arr
                )

                end_time_gpu = time.perf_counter()
                print(f"GPU Lidar visibility calculation took: {end_time_gpu - start_time_gpu:.4f} seconds")

                print(f"GPU Voxel state shape: {voxel_state_gpu.shape}")
                print(f"GPU Voxel label shape: {voxel_label_gpu.shape}")
                print("Finished Lidar visibility masks (GPU).")

                if args.vis_lidar_visibility:
                    voxel_size_for_viz = np.array([voxel_size] * 3)
                    visualize_occupancy_o3d(
                        voxel_state=voxel_state_gpu,
                        voxel_label=voxel_label_gpu,
                        pc_range=pc_range,
                        voxel_size=voxel_size_for_viz,
                        class_color_map=CLASS_COLOR_MAP,  # Make sure this is globally defined
                        default_color=DEFAULT_COLOR,  # Make sure this is globally defined
                        show_semantics=True,
                        show_free=True,
                        show_unobserved=False
                    )

                run_cpu_comparison_lidar = False
                if run_cpu_comparison_lidar:
                    if isinstance(voxel_size, (int, float)):
                        voxel_size_masks = [voxel_size, voxel_size, voxel_size]

                    # --- Time the CPU execution ---
                    print("\nTiming CPU Lidar visibility calculation...")
                    start_time_cpu = time.perf_counter()

                    voxel_state, voxel_label = calculate_lidar_visibility(
                        points=scene_semantic_points[:, :3],  # (N,3) hits in ego–i
                        points_origin=points_origin,  # (N,3) ray‐starts in ego–i
                        points_label=points_label,  # (N,) semantic of each hit
                        pc_range=pc_range,  # [xmin,ymin,zmin,xmax,ymax,zmax]
                        voxel_size=voxel_size_masks,  # [vx,vy,vz]
                        spatial_shape=occ_size,  # (H,W,Z)
                        occupancy_grid=occupancy_grid,
                        FREE_LEARNING_INDEX=FREE_LEARNING_INDEX,
                        points_sensor_indices=scene_semantic_points_sids,
                        sensor_max_ranges=sensor_max_ranges_arr
                    )

                    end_time_cpu = time.perf_counter()
                    print(f"CPU Lidar visibility calculation took: {end_time_cpu - start_time_cpu:.4f} seconds")

                    print(voxel_state.shape)
                    print(voxel_label.shape)

                    print("Finished Lidar visibility masks")

                    if args.vis_lidar_visibility:
                        print("Visualizing with Semantics and Free")
                        visualize_occupancy_o3d(
                            voxel_state=voxel_state,
                            voxel_label=voxel_label,
                            pc_range=pc_range,
                            voxel_size=voxel_size_masks,  # Pass as array
                            class_color_map=CLASS_COLOR_MAP,
                            default_color=DEFAULT_COLOR,
                            show_semantics=True,
                            show_free=True,
                            show_unobserved=False
                        )

                occupied_mask = occupancy_grid != FREE_LEARNING_INDEX
                total_occupied_voxels = np.sum(occupied_mask)

                if np.any(occupied_mask):  # Check if there are any occupied voxels
                    vx, vy, vz = np.where(occupied_mask)  # Get the indices (vx, vy, vz) of all occupied voxels
                    labels_at_occupied = occupancy_grid[vx, vy, vz]  # Get the labels at these occupied locations

                    # Stack them into the [vx, vy, vz, label] format
                    dense_voxels_with_semantic_voxelcoords_save = np.stack([vx, vy, vz, labels_at_occupied],
                                                                           axis=-1).astype(int)
                else:
                    # If no voxels are occupied (e.g., the entire grid is FREE_LEARNING_INDEX)
                    dense_voxels_with_semantic_voxelcoords_save = np.zeros((0, 4), dtype=int)

                # --- Saving Logic (Using GPU results by default) ---
                # final_voxel_state_to_save will be the grid with 0 (UNOBS), 1 (FREE), 2 (OCC)
                final_voxel_state_to_save = voxel_state_gpu
                # final_voxel_label_to_save will be the grid with semantic labels for OCC,
                # and FREE_LEARNING_INDEX for FREE and UNOBS. This is the 'semantics' array for Occ3D.
                final_voxel_label_to_save = voxel_label_gpu

                # --- Calculate Camera Visibility Mask ---

                print(f"Calculating camera visibility for cameras (GPU): {cameras}")
                start_time_cam_vis_gpu = time.perf_counter()

                mask_camera = calculate_camera_visibility_gpu_host(  # Call the GPU host function
                    trucksc=trucksc,
                    current_sample_token=frame_dict['sample_token'],  # Pass current sample token
                    lidar_voxel_state_cpu=voxel_state_gpu,  # Output from LiDAR visibility
                    pc_range_cpu_list=pc_range,
                    voxel_size_cpu_scalar=voxel_size,
                    spatial_shape_cpu_list=occ_size,
                    camera_names=cameras,
                    DEPTH_MAX_val=config.get('camera_ray_depth_max', 100.0)  # e.g., 70 meters
                )

                print(f"Camera visibility mask cpu has the shape: {mask_camera.shape}")

                end_time_cam_vis_gpu = time.perf_counter()
                print(
                    f"GPU Camera visibility calculation took: {end_time_cam_vis_gpu - start_time_cam_vis_gpu:.4f} seconds")

                mask_camera_binary = np.zeros_like(mask_camera, dtype=np.uint8)
                mask_camera_binary[mask_camera == STATE_OCCUPIED] = 1
                mask_camera_binary[mask_camera == STATE_FREE] = 1

                if args.vis_camera_visibility:
                    print("Visualizing GPU Camera Visibility Mask Results...")

                    temp_voxel_state_for_cam_viz = mask_camera.copy()

                    voxel_size_arr_viz = np.array([voxel_size] * 3) if isinstance(voxel_size, (int, float)) else np.array(
                        voxel_size)

                    visualize_occupancy_o3d(
                        voxel_state=temp_voxel_state_for_cam_viz,
                        voxel_label=voxel_label_gpu,
                        pc_range=pc_range,
                        voxel_size=voxel_size_arr_viz,
                        class_color_map=CLASS_COLOR_MAP,
                        default_color=DEFAULT_COLOR,
                        show_semantics=True,  # Show semantics of camera-visible regions
                        show_free=True,  # Not showing LiDAR-free for this specific mask viz
                        show_unobserved=False  # Shows what's NOT camera visible as unobserved
                    )

                run_cpu_comparison_camera = False
                if run_cpu_comparison_camera:
                    print(f"Calculating camera visibility for cameras (CPU): {cameras}")

                    # Ensure voxel_size and occ_size are passed as numpy arrays if the function expects them
                    voxel_size_arr = np.array([voxel_size] * 3) if isinstance(voxel_size, (int, float)) else np.array(
                        voxel_size)
                    occ_size_arr = np.array(occ_size)

                    start_time_cam_vis = time.perf_counter()
                    mask_camera = calculate_camera_visibility_cpu(
                        trucksc=trucksc,
                        current_sample_token=frame_dict['sample_token'],  # Pass current sample token
                        lidar_voxel_state=final_voxel_state_to_save,  # Output from LiDAR visibility
                        pc_range_params=pc_range,
                        voxel_size_params=voxel_size_arr,
                        spatial_shape_params=occ_size_arr,
                        camera_names=cameras,
                        DEPTH_MAX=config.get('camera_ray_depth_max', 100.0)  # Make it configurable
                    )
                    print(f"Camera visibility mask cpu has the shape: {mask_camera.shape}")

                    end_time_cam_vis = time.perf_counter()
                    print(f"CPU Camera visibility calculation took: {end_time_cam_vis - start_time_cam_vis:.4f} seconds")

                    mask_camera_binary = np.zeros_like(mask_camera, dtype=np.uint8)
                    mask_camera_binary[mask_camera == STATE_OCCUPIED] = 1
                    mask_camera_binary[mask_camera == STATE_FREE] = 1

                    if args.vis_camera_visibility:

                        print("Visualizing Camera Visibility Mask Results...")
                        temp_voxel_state_for_cam_viz = mask_camera.copy()

                        voxel_size_arr_viz = np.array([voxel_size] * 3) if isinstance(voxel_size,
                                                                                      (int, float)) else np.array(
                            voxel_size)

                        visualize_occupancy_o3d(
                            voxel_state=temp_voxel_state_for_cam_viz,
                            voxel_label=voxel_label_gpu,
                            pc_range=pc_range,
                            voxel_size=voxel_size_arr_viz,
                            class_color_map=CLASS_COLOR_MAP,
                            default_color=DEFAULT_COLOR,
                            show_semantics=True,  # Show semantics of camera-visible regions
                            show_free=True,  # Not showing LiDAR-free for this specific mask viz
                            show_unobserved=False  # Shows what's NOT camera visible as unobserved
                        )

        print(
            f"Shape of dense_voxels_with_semantic_voxelcoords for saving: {dense_voxels_with_semantic_voxelcoords_save.shape}")
        print(
            f"Occupancy shape: Occsize: {occupancy_grid.shape}, Total number voxels: {occupancy_grid.shape[0] * occupancy_grid.shape[1] * occupancy_grid.shape[2]}, Occupied: {total_occupied_voxels}")

        print(f"\nPreparing data for saving (using GPU results by default)...")

        # Create the binary mask_lidar (0 for unobserved, 1 for observed)
        # Observed means either FREE or OCCUPIED.
        mask_lidar_to_save = (final_voxel_state_to_save != STATE_UNOBSERVED).astype(np.uint8)
        mask_camera_to_save = mask_camera_binary

        ##########################################save like Occ3D #######################################
        # Save as .npz
        dirs = os.path.join(save_path, scene_name, frame_dict['sample_token'])
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        # Save in .npz format, matching Occ3D keys
        output_filepath_npz = os.path.join(dirs, 'labels.npz')
        print(f"Saving semantic occupancy and LiDAR visibility mask to {output_filepath_npz}...")
        np.savez_compressed(
            output_filepath_npz,
            semantics=final_voxel_label_to_save,  # This is your (Dx,Dy,Dz) semantic grid
            mask_lidar=mask_lidar_to_save,  # This is your (Dx,Dy,Dz) 0-1 LiDAR visibility mask
            mask_camera=mask_camera_to_save,
            # If you also compute camera visibility, you would add:
            # mask_camera=your_camera_visibility_mask
        )
        print(f"  Saved 'semantics' shape: {final_voxel_label_to_save.shape}")
        print(f"  Saved 'mask_lidar' shape: {mask_lidar_to_save.shape} (0=unobserved, 1=observed)")
        print(
            f"  Saved 'mask_camera' shape: {mask_camera_to_save.shape} (0=unobserved, 1=observed)")

        """##########################################Save as .npy ##########################################
        # Save the resulting dense voxels with semantics
        dirs = os.path.join(save_path, scene_name, frame_dict['sample_token'])  #### Save in folder with scene name
        if not os.path.exists(dirs):  # create directory if does not exist
            os.makedirs(dirs)

        save_path_base = 'labels'  ### copy file_name to save_path_base
        suffix_to_remove = '.pcd'  ### define suffix to remove
        if save_path_base.endswith(suffix_to_remove):
            save_path_base = save_path_base[:-len(suffix_to_remove)]  ### Slice off suffix

        output_filepath = os.path.join(dirs, save_path_base + '.npy')  ### Generate output filepath
        # Save the dense semantic voxels as a numpy file with a filename corresponding to the frame
        print(f"Saving GT to {output_filepath}...")  ####
        np.save(output_filepath, dense_voxels_with_semantic_voxelcoords_save)  ### saving point cloud
        print(f"Dense voxels with semantic shape {dense_voxels_with_semantic_voxelcoords_save.shape} saved.")"""

        i = i + 1
        continue  # moves to the next frame for processing


# Main entry point of the script
if __name__ == '__main__':
    # argument parsing allows users to customize parameters when running the script
    from argparse import ArgumentParser

    parse = ArgumentParser()

    ############################## Define Command-Line Arguments #########################################################################
    parse.add_argument('--dataset', type=str, default='truckscenes')  # Dataset selection with default: "truckScenes"
    parse.add_argument('--config_path', type=str,
                       default='config_truckscenes.yaml')  # Configuration file path with default: "config.yaml"
    parse.add_argument('--split', type=str,
                       default='train')  # data split, default: "train", options: "train", "val", "all"
    parse.add_argument('--save_path', type=str,
                       default='./data/GT_occupancy/')  # save path, default: "./data/GT/GT_occupancy"
    parse.add_argument('--start', type=int,
                       default=0)  # start indice, default: 0, determines range of sequences to process
    parse.add_argument('--end', type=int,
                       default=850)  # end indice, default: 850, determines range of sequences to process
    parse.add_argument('--dataroot', type=str,
                       default='./data/truckscenes/')  # data root path, default: "./data/truckScenes
    parse.add_argument('--version', type=str, default='v1.0-trainval')
    parse.add_argument('--trucksc_val_list', type=str,
                       default='./truckscenes_val_list.txt')  # text file containing validation scene tokens, default: "./truckscenes_val_list.txt"
    parse.add_argument('--label_mapping', type=str,
                       default='truckscenes.yaml')  # YAML file containing label mappings, default: "truckscenes.yaml"

    parse.add_argument('--load_mode', type=str, default='pointwise')  # pointwise or rigid

    parse.add_argument('--static_map_keyframes_only', action='store_true', help='Build the final static map using only keyframes (after ICP, if enabled, ran on all frames).')
    parse.add_argument('--dynamic_map_keyframes_only', action='store_true', help='Aggregate dynamic object points using only segments from keyframes..')

    ####################### Kiss-ICP refinement ##########################################
    parse.add_argument('--icp_refinement', action='store_true', help='Enable ICP refinement')
    parse.add_argument('--pose_error_plot', action='store_true', help='Plot pose error')

    ####################### Meshing #####################################################
    parse.add_argument('--meshing', action='store_true', help='Enable meshing')

    ######################## Filtering ####################################################
    parse.add_argument('--filter_mode', type=str, default='none', choices=['none', 'sor', 'ror', 'both'],
                       help='Noise filtering method to apply before meshing')

    parse.add_argument('--filter_lidar_intensity', action='store_true', help='Enable lidar intensity filtering')

    parse.add_argument('--filter_raw_pc', action='store_true', help='Enable raw pc filtering')
    parse.add_argument('--filter_static_pc', action='store_true', help='Enable static pc filtering')
    parse.add_argument('--filter_aggregated_static_pc', action='store_true', help='Enable aggregated static pc filtering')
    parse.add_argument('--filter_combined_static_dynamic_pc', action='store_true', help='Enable combined static and dynamic pc filtering')

    ########################## Visualization ################################################
    parse.add_argument('--vis_raw_pc', action='store_true', help='Enable raw pc visualization')
    parse.add_argument('--vis_static_pc', action='store_true', help='Enable static pc visualization')
    parse.add_argument('--vis_static_pc_global', action='store_true', help='Enable static pc global visualization')
    parse.add_argument('--vis_lidar_intensity_filtered', action='store_true', help='Enable lidar intensity filtered visualization')

    parse.add_argument('--vis_aggregated_static_ego_i_pc', action='store_true', help='Enable aggregated static ego i pc visualization')
    parse.add_argument('--vis_aggregated_static_ego_ref_pc', action='store_true', help='Enable aggregated static ego ref pc visualization')
    parse.add_argument('--vis_aggregated_static_global_pc', action='store_true', help='Enable aggregated static global pc visualization')
    parse.add_argument('--vis_aggregated_raw_pc_ego_i', action='store_true', help='Enable aggregated raw pc ego i visualization')
    parse.add_argument('--vis_static_frame_comparison_kiss_refined', action='store_true', help='Enable static frame comparison kiss refinement')
    parse.add_argument('--vis_aggregated_static_kiss_refined', action='store_true', help='Enable aggregated static kiss refinement')
    parse.add_argument('--vis_filtered_aggregated_static', action='store_true', help='Enable filtered aggregated static kiss refinement')
    parse.add_argument('--vis_static_before_combined_dynamic', action='store_true', help='Enable static pc visualization before combined with dynmanic points')
    parse.add_argument('--vis_combined_static_dynamic_pc', action='store_true', help='Enable combined static and dynamic pc visualization')

    parse.add_argument('--vis_lidar_visibility', action='store_true', help='Enable lidar visibility visualization')
    parse.add_argument('--vis_camera_visibility', action='store_true', help='Enable camera visibility visualization')

    args = parse.parse_args()

    if args.dataset == 'truckscenes':  # check dataset type
        val_list = []
        with open(args.trucksc_val_list, 'r') as file:  # Load validation scene tokens to val_list
            for item in file:
                val_list.append(item[:-1])
        file.close()

        # Load the truckScenes dataset
        truckscenes = TruckScenes(version=args.version,
                                  dataroot=args.dataroot,
                                  verbose=True)  # verbose True to print informational messages
        train_scenes = splits.train  # train_scenes and val_scenes contain the scene tokens for the training and validation splits
        val_scenes = splits.val
    else:
        raise NotImplementedError

    # load config with hyperparameters and settings
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map to map raw semantic labels to learning labels
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        truckscenesyaml = yaml.safe_load(stream)

    # Process sequences in a loop
    for i in range(args.start, args.end):
        print('processing sequence:', i)
        # call the main function
        # Inputs: nusc: initialized truckscenes dataset object
        # val_list: list of validation scene tokens
        # indice: current scene index
        # truckscenesyaml: loaded label mapping
        # args: parsed command-line arguments
        # config: configuration settings
        main(truckscenes, val_list, indice=i,
             truckscenesyaml=truckscenesyaml, args=args, config=config)
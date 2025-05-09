import os
import sys
import pdb
import time
import yaml
import torch
import chamfer
import mmcv
import numpy as np
from truckscenes.truckscenes import TruckScenes ### Truckscenes
from truckscenes.utils import splits ### Truckscenes
from tqdm import tqdm
from truckscenes.utils.data_classes import Box, LidarPointCloud
from truckscenes.utils.geometry_utils import view_points ### Truckscenes
from typing import Any, Dict, List, Optional, Union
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

from custom_datasets import InMemoryDataset


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
        return points_n_features # Return empty array if no points

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
    if points_n_features.shape[1] > 3: # Check if there were features beyond XYZ (columns > 3)
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

def get_pointwise_fused_pointcloud(trucksc: TruckScenes, sample: Dict[str, Any], allowed_sensors: List[str]) -> LidarPointCloud:
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
    """
    # Initialize
    points = np.zeros((LidarPointCloud.nbr_dims(), 0), dtype=np.float64)
    timestamps = np.zeros((1, 0), dtype=np.uint64)
    fused_point_cloud = LidarPointCloud(points, timestamps)

    # Get reference ego pose (timestamp of the sample/annotations)
    ref_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])

    # Homogeneous transformation matrix from global to ref ego car frame.
    car_from_global = transform_matrix(ref_ego_pose['translation'],
                                       Quaternion(ref_ego_pose['rotation']),
                                       inverse=True)

    # Iterate over all lidar sensors and fuse their point clouds
    for sensor in sample['data'].keys():
        if sensor not in allowed_sensors:
            print(f"Skipping sensor {sensor} as it is not allowed.")
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

        # Merge with key pc.
        fused_point_cloud.points = np.hstack((fused_point_cloud.points, pc.points))
        if pc.timestamps is not None:
            fused_point_cloud.timestamps = np.hstack((fused_point_cloud.timestamps, pc.timestamps))

    return fused_point_cloud

def get_rigid_fused_pointcloud(trucksc: TruckScenes, sample: Dict[str, Any], allowed_sensors: List[str]) -> LidarPointCloud:
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
    """
    # Initialize
    points = np.zeros((LidarPointCloud.nbr_dims(), 0), dtype=np.float64)
    timestamps = np.zeros((1, 0), dtype=np.uint64)
    fused_point_cloud = LidarPointCloud(points, timestamps)

    # Get reference ego pose (timestamp of the sample/annotations)
    ref_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])

    # Homogeneous transformation matrix from global to ref ego car frame.
    car_from_global = transform_matrix(ref_ego_pose['translation'],
                                       Quaternion(ref_ego_pose['rotation']),
                                       inverse=True)

    # Iterate over all lidar sensors and fuse their point clouds
    for sensor in sample['data'].keys():
        if sensor not in allowed_sensors:
            print(f"Skipping sensor {sensor} as it is not allowed.")
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

        # Merge with key pc.
        fused_point_cloud.points = np.hstack((fused_point_cloud.points, pc.points))
        if pc.timestamps is not None:
            fused_point_cloud.timestamps = np.hstack((fused_point_cloud.timestamps, pc.timestamps))

    return fused_point_cloud


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
        vertices_to_remove = densities < np.quantile(densities, min_density) # min_density should be between 0 and 1
        mesh.remove_vertices_by_mask(vertices_to_remove) # removes vertices where density is below threshold
    mesh.compute_vertex_normals() # computes the normals of the vertices

    return mesh, densities

# Function that creates a 3D mesh from a given point cloud or a buffer of points
# Inputs buffer: a list of point clouds that are combined if no original is given
# depth: resolution for poisson surface reconstruction
# n_threads: Number of threads for parallel processing
# min_density: Optional threshold for removing low-density vertices
# point_cloud_original: provides a preprocessed point cloud, if given buffer is ignored
def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer) # Calls buffer_to_pointcloud(buffer) to create a combined point cloud from the list of smaller point clouds
    else:
        pcd = point_cloud_original # Uses the given point cloud directly

    return run_poisson(pcd, depth, n_threads, min_density) # calls run_poisson function to generate mesh

# Function to combine multiple point clouds from a list (buffer) into a single point cloud and optionally estimating normals in the resulting point cloud
# Input: buffer: a list of individual point clouds (each being an instance of open3d.geometry.PointCloud)
# compute_normals: boolean flag that if set to True estimates normals of the final point cloud
def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud() # Initialize empty point cloud object using Open3d
    for cloud in buffer:
        pcd += cloud # combine each point cloud with current point cloud object
    if compute_normals:
        pcd.estimate_normals() # estimate normals for each point in the combined point cloud

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

    cloud = deepcopy(pcd) # create a deep copy of the input point cloud to ensure that original point cloud is not modified
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params) # estimate normals based on nearest max_nn points
        cloud.orient_normals_towards_camera_location() # orients all computed normals to point towards the camera location to get consistent normal direction for visualization and mesh generation

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

def denoise_pointcloud(pcd: o3d.geometry.PointCloud, filter_mode: str, config: dict, location_msg: str = "point cloud") -> o3d.geometry.PointCloud:
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
        print(f"[Filtering ({filter_mode}) at {location_msg}] Reduced from {initial_count} to {final_count} points.")

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
                              boxes: Optional[List] = None, # Use List[Box] if Box class is imported
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
             if np.max(colors_float) > 1.0: # Basic check if maybe 0-255 range
                 colors_float /= 255.0
             pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_float, 0.0, 1.0))
        else:
            print(f"Warning: Invalid 'colors' argument. Type: {type(colors)}, Value/Shape: {colors if isinstance(colors, str) else colors.shape}. Using default colors.")
    elif points.shape[1] > 3: # Default to label coloring if 4th column exists and colors=None
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
                rgb = cmap(normalized_labels)[:, :3] # Get RGB, ignore alpha
                pcd.colors = o3d.utility.Vector3dVector(rgb)
            else:
                print("Warning: Found label column, but no unique labels detected.")
        except Exception as e:
            print(f"Error applying label coloring: {e}. Using default colors.")
            use_label_coloring = False # Revert if error occurred

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
                center = box.center # Should be numpy array (3,)
                # truckscenes Box.wlh = [width(y), length(x), height(z)]
                # o3d OrientedBoundingBox extent = [length(x), width(y), height(z)]
                extent = np.array([box.wlh[1], box.wlh[0], box.wlh[2]])
                # Get rotation matrix from pyquaternion.Quaternion
                R = box.orientation.rotation_matrix # Should be 3x3 numpy array

                obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
                obb.color = (1.0, 0.0, 0.0)  # Set color to red
                geometries.append(obb)
                num_boxes_drawn += 1

            except AttributeError as e:
                print(f"Error processing box {i} (Token: {getattr(box, 'token', 'N/A')}): Missing attribute {e}. Skipping box.")
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
    object_icp_voxel_size = config.get("object_icp_voxel_size", 0.1)

    # sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT']
    sensors = config['sensors']
    print(sensors)

    # Retrieves a specific scene from the truckScenes dataset
    my_scene = trucksc.scene[indice]  # scene is selected by indice parameter
    scene_name = my_scene['name']  ### Extract scene name for saving
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

    dict_list = []
    i = 0
    reference_ego_pose = None
    ref_ego_from_global = None

    while True:
        print (f"Processing frame {i}")
        ########### Load point cloud #############
        if load_mode == 'pointwise':
            sensor_fused_pc = get_pointwise_fused_pointcloud(trucksc, my_sample, allowed_sensors=sensors)
        elif load_mode == 'rigid':
            sensor_fused_pc = get_rigid_fused_pointcloud(trucksc, my_sample, allowed_sensors=sensors)
        else:
            raise ValueError(f'Fusion mode {load_mode} is not supported')

        print(f"The fused sensor pc at frame {i} has the shape: {sensor_fused_pc.points.shape}")

        # visualize_pointcloud(sensor_fused_pc.points.T, title=f"Fused sensor PC in ego coordinates - Frame {i}")

        if args.vis and args.vis_position == 'fused_sensors_ego':
            visualize_pointcloud(sensor_fused_pc.points.T, title=f"Fused sensor PC in ego coordinates - Frame {i}")

        """geometries_to_draw = []
        pcd = o3d.geometry.PointCloud()
        #points_xyz_1 = sensor_fused_pc.points.T[:, :3]
        points_xyz_1 = sensor_fused_pc.points[:3, :].T
        pcd.points = o3d.utility.Vector3dVector(points_xyz_1)
        pcd.paint_uniform_color([0.0, 0.0, 1.0])

        geometries_to_draw.append(pcd)

        center = np.array([(x_min_self + x_max_self) / 2.0,
                           (y_min_self + y_max_self) / 2.0,
                           (z_min_self + z_max_self) / 2.0])
        extent = np.array([x_max_self - x_min_self,
                           y_max_self - y_min_self,
                           z_max_self - z_min_self])
        # Rotation matrix is identity since it's axis-aligned in ego frame
        R = np.eye(3)

        # Create the Open3D oriented bounding box object
        ego_filter_o3d_bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
        ego_filter_o3d_bbox.color = (1.0, 0.0, 0.0)  # Set color to red
        geometries_to_draw.append(ego_filter_o3d_bbox)

        o3d.visualization.draw_geometries(
            geometries_to_draw,
            window_name=f"Frame {i}: Fused PC & Ego Filter Box (Ego Coords)"
        )"""

        ##########################################

        ########### get boxes #####################
        boxes = get_boxes(trucksc, my_sample)

        boxes_token = [box.token for box in boxes]  # retrieves a list of tokens from the bounding box
        # Extract object tokens. Each instance token represents a unique object
        object_tokens = [truckscenes.get('sample_annotation', box_token)['instance_token'] for box_token in
                         boxes_token]  # Uses sample_annotation data to get instance_token fore each bb
        # Extract object categories
        object_category = [truckscenes.get('sample_annotation', box_token)['category_name'] for box_token in
                           boxes_token]  # retrieves category name for each bounding box

        """visualize_pointcloud_bbox(sensor_fused_pc.points.T,
                                  boxes=boxes,
                                  title=f"Fused filtered static sensor PC + BBoxes + Ego BBox - Frame {i}",
                                  self_vehicle_range=self_range,
                                  vis_self_vehicle=True)"""

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
                converted_object_category.append(UNKNOWN_LEARNING_INDEX)

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1,
                                                           3)  # gets center coordinates (x,y,z) of each bb
        dims = np.array([b.wlh for b in boxes]).reshape(-1,
                                                        3)  # extract dimension width, length, height of each bb
        rots = np.array([b.orientation.yaw_pitch_roll[0]  # extract rotations (yaw angles)
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(
            np.float32)  # combines location, dimensions and rotation into a 2D array

        gt_bbox_3d[:, 6] += np.pi / 2.  # adjust yaw angles by 90 degrees
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        # gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.05  # Experiment
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        # gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.05 # Experiment
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1  # Slightly expand the bbox to wrap all object points

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

        object_points_list = []  # creates an empty list to store points associated with each object
        j = 0
        # Iterate through each bounding box along the last dimension
        while j < points_in_boxes.shape[-1]:
            # Create a boolean mask indicating whether each point belongs to the current bounding box.
            object_points_mask = points_in_boxes[0][:, j].bool()
            # Extract points using mask to filter points
            object_points = sensor_fused_pc.points.T[object_points_mask]
            # Store the filtered points, Result is a list of arrays, where each element contains the points belonging to a particular object
            object_points_list.append(object_points)
            j = j + 1

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
        ego_filter_mask = ~inside_ego_mask

        points_mask = static_mask & ego_filter_mask

        pc_ego_unfiltered = sensor_fused_pc.points.T[points_mask]
        print(f"Number of static points extracted: {pc_ego_unfiltered.shape}")

        """for box_idx in range(gt_bbox_3d.shape[0]):
            num_in_box = points_in_boxes[0][:, box_idx].sum().item()
            print(f"Box {box_idx} contains {num_in_box} points")"""

        pc_with_semantic_ego_unfiltered = pc_with_semantic[points_mask]
        print(f"Number of semantic static points extracted: {pc_with_semantic_ego_unfiltered.shape}")

        pc_ego = pc_ego_unfiltered.copy()
        pc_with_semantic_ego = pc_with_semantic_ego_unfiltered.copy()

        # Optional: Apply filtering to static points in ego frame
        if args.filter_location in ['ego_static', 'all'] and args.filter_mode != 'none':
            print(f"Filtering static ego points at frame {i}...")

            if pc_ego.shape[0] > 0:
                pcd_static = o3d.geometry.PointCloud()
                pcd_static.points = o3d.utility.Vector3dVector(pc_ego[:, :3])
                filtered_pcd_static, kept_indices = denoise_pointcloud(
                    pcd_static, args.filter_mode, config, location_msg="static ego points"
                )
                pc_ego = np.asarray(filtered_pcd_static.points)
                pc_with_semantic_ego = pc_with_semantic_ego[kept_indices]  # ✅ Only filter here


        if args.vis and args.vis_position == 'fused_sensors_ego_boxes_static':
            visualize_pointcloud_bbox(pc_ego, boxes=boxes,
                                      title=f"Fused filtered static sensor PC + BBoxes - Frame {i}")

        # Get ego pose for this sample
        ego_pose_i = trucksc.getclosest('ego_pose', my_sample['timestamp'])

        # Transformation from ego to global
        global_from_ego_i = transform_matrix(ego_pose_i['translation'], Quaternion(ego_pose_i['rotation']), inverse=False)

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
        semantic_points_in_ref_frame = transform_points(pc_with_semantic_ego, ego_ref_from_ego_i)
        print(f"Frame {i}: Transformed static points to ref ego. Shape: {points_in_ref_frame.shape}")
        print(f"Frame {i}: Transformed semantic static points to ref ego. Shape: {semantic_points_in_ref_frame.shape}")

        # --- Transform FILTERED static points TO GLOBAL FRAME ---
        points_in_global_frame = transform_points(pc_ego, global_from_ego_i)
        semantic_points_in_global_frame = transform_points(pc_with_semantic_ego, global_from_ego_i)
        print(f"Frame {i}: Transformed static points to global. Shape: {points_in_global_frame.shape}")
        print(
            f"Frame {i}: Transformed semantic static points to global. Shape: {semantic_points_in_global_frame.shape}")


        if args.vis and args.vis_position == 'fused_sensors_global_static':
            visualize_pointcloud(points_in_ref_frame, title=f"Fused sensor PC in world coordinates - Frame {i}")

        # Assign the calculated variables to the desired keys
        pc_ego_i_save = pc_ego.copy()  # Filtered points in current ego frame (Features, N)
        print(f"Frame {i}: Static points in ego frame shape: {pc_ego_i_save.shape}")
        pc_with_semantic_ego_i_save = pc_with_semantic_ego.copy()  # Filtered semantic points in current ego frame (Features+1, N)
        print(f"Frame {i}: Static semantic points in ego frame shape: {pc_with_semantic_ego_i_save.shape}")
        pc_ego_ref_save = points_in_ref_frame.copy()  # Filtered points transformed to reference ego frame (Features, N)
        pc_with_semantic_ego_ref_save = semantic_points_in_ref_frame.copy()  # Filtered semantic points transformed to reference ego frame (Features+1, N)
        pc_global_save = points_in_global_frame.copy()  # Filtered points transformed to global frame (Features, N)
        pc_with_semantic_global_save = semantic_points_in_global_frame.copy()  # Filtered semantic points transformed to global frame (Features+1, N)

        ################## record information into a dict  ########################
        ref_sd = trucksc.get('sample_data', my_sample['data']['LIDAR_LEFT'])

        frame_dict = {
            "sample_timestamp": my_sample['timestamp'],
            "scene_name": scene_name,
            "sample_token": my_sample['token'],
            "is_key_frame": ref_sd['is_key_frame'],
            "converted_object_category": converted_object_category,
            "gt_bbox_3d": gt_bbox_3d,  # BBox in current frame's ego coords
            "object_tokens": object_tokens,
            "object_points_list": object_points_list,  # Raw object points in current ego frame
            "raw_lidar_ego": sensor_fused_pc.points.T,
            "lidar_pc_ego_i": pc_ego_i_save,  # Filtered static points in CURRENT ego frame (i)
            "lidar_pc_with_semantic_ego_i": pc_with_semantic_ego_i_save,
            # Filtered semantic static points in CURRENT ego frame (i)
            "lidar_pc_ego_ref": pc_ego_ref_save,  # Filtered static points transformed to REFERENCE ego frame
            "lidar_pc_with_semantic_ego_ref": pc_with_semantic_ego_ref_save,
            # Filtered semantic static points transformed to REFERENCE ego frame
            "lidar_pc_global": pc_global_save,  # Filtered static points transformed to GLOBAL frame
            "lidar_pc_with_semantic_global": pc_with_semantic_global_save,
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

        next_sample_token = my_sample['next']
        i += 1
        if next_sample_token != '':
            my_sample = trucksc.get('sample', next_sample_token)
        else:
            break

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
    # Optional: Check shape of first element if list is not empty
    # if unrefined_pc_ref_ego_list:
    #     print(f"  Shape of first cloud: {unrefined_pc_ref_ego_list[0].shape}")

    # Extract semantic static points (already in ref ego frame, Features+1 x N format)
    # Use the correct key from the dictionary populated earlier
    unrefined_sem_pc_ego_list = [frame_dict['lidar_pc_with_semantic_ego_i'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_sem_pc_ego_list)} semantic point clouds (in ego i frame).")
    # Optional: Check shape of first element if list is not empty
    # if unrefined_sem_pc_ref_ego_list:
    #     print(f"  Shape of first semantic cloud: {unrefined_sem_pc_ref_ego_list[0].shape}")

    pc_ego_combined_draw = np.concatenate(unrefined_pc_ego_list, axis=0)
    print(f"Pc ego i combined shape: {pc_ego_combined_draw.shape}")
    pc_ego_to_draw = o3d.geometry.PointCloud()
    pc_coordinates = pc_ego_combined_draw[:, :3]
    pc_ego_to_draw.points = o3d.utility.Vector3dVector(pc_coordinates)
    o3d.visualization.draw_geometries([pc_ego_to_draw], window_name="Combined static point clouds (in ego i frame)")

    unrefined_pc_ego_ref_list = [frame_dict['lidar_pc_ego_ref'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_ego_ref_list)} static point clouds (in ego ref frame).")
    unrefined_sem_pc_ego_ref_list = [frame_dict['lidar_pc_with_semantic_ego_ref'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_sem_pc_ego_ref_list)} semantic static point clouds (in ego ref frame).")

    pc_ego_ref_combined_draw = np.concatenate(unrefined_pc_ego_ref_list, axis=0)
    print(f"Pc ego ref combined shape: {pc_ego_ref_combined_draw.shape}")
    pc_ego_ref_to_draw = o3d.geometry.PointCloud()
    pc_ego_ref_coordinates = pc_ego_ref_combined_draw[:, :3]
    pc_ego_ref_to_draw.points = o3d.utility.Vector3dVector(pc_ego_ref_coordinates)
    o3d.visualization.draw_geometries([pc_ego_ref_to_draw], window_name="Combined static point clouds (in ego ref frame)")

    unrefined_pc_global_list = [frame_dict['lidar_pc_global'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_global_list)} static point clouds (in world frame).")
    unrefined_sem_pc_global_list = [frame_dict['lidar_pc_with_semantic_global'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_sem_pc_global_list)} semantic static point clouds (in world frame).")

    pc_global_combined_draw = np.concatenate(unrefined_pc_global_list, axis=0)
    print(f"Pc global shape: {pc_global_combined_draw.shape}")
    pc_global_to_draw = o3d.geometry.PointCloud()
    pc_global_coordinates = pc_global_combined_draw[:, :3]
    pc_global_to_draw.points = o3d.utility.Vector3dVector(pc_global_coordinates)
    o3d.visualization.draw_geometries([pc_global_to_draw], window_name="Combined static point clouds (in ego ref frame)")

    raw_pc_list = [frame_dict['raw_lidar_ego'] for frame_dict in dict_list]
    print(f"Extracted {len(raw_pc_list)} static and dynamic point clouds (in ego i frame).")
    raw_pc_draw = np.concatenate(raw_pc_list, axis=0)
    print(f"Raw Pc with static and dynamic points shape: {raw_pc_draw.shape}")
    raw_pc_to_draw = o3d.geometry.PointCloud()
    raw_pc_coordinates = raw_pc_draw[:, :3]
    raw_pc_to_draw.points = o3d.utility.Vector3dVector(raw_pc_coordinates)
    o3d.visualization.draw_geometries([raw_pc_to_draw], window_name="Combined static and dynamic point clouds (in ego frame)")

    # Extract timestamps associated with each frame's original ego pose
    try:
        # Adjust key if needed based on your ego_pose dictionary structure
        lidar_timestamps = [frame_dict['sample_timestamp'] for frame_dict in dict_list]
        print(f"Extracted {len(lidar_timestamps)} timestamps.")
    except KeyError:
        print("Timestamp key not found in ego_pose, setting lidar_timestamps to None.")
        lidar_timestamps = None  # Fallback

    print(f"Lidar timestamps: {lidar_timestamps}")

    print("Finished preparing initial lists of points in reference ego frame.")

    poses_kiss_icp = None
    ################# Refinement using KISS-ICP ###################################################
    if args.icp_refinement and len(dict_list) > 1:
        print(f"--- Performing KISS-ICP refinement on static global point clouds for scene {scene_name} ---")

        # --- 1. Prepare Dataset and Pipeline ---
        in_memory_dataset = None
        pipeline = None
        estimated_poses_kiss = None  # Will hold the results from pipeline.poses

        try:
            in_memory_dataset = InMemoryDataset(
                lidar_scans=raw_pc_list,
                timestamps=lidar_timestamps,
                # Use a descriptive sequence ID, incorporating scene name if possible
                sequence_id=f"{scene_name}_icp_run"
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
            refined_lidar_pc_with_semantic_list = []

            for idx, points_ego in enumerate(unrefined_pc_ego_list):
                pose = estimated_poses_kiss[idx]
                print(f"Applying refined pose {idx}: {pose}")

                print(f"Points ego shape: {points_ego.shape}")

                #points_xyz = points_ego.T[:, :3]
                points_xyz = points_ego[:, :3]
                points_homo = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
                points_transformed = (pose @ points_homo.T)[:3, :].T

                if points_ego.shape[1] > 3:
                    #other_features = points_ego.T[:, 3:]
                    other_features = points_ego[:, 3:]
                    points_transformed = np.hstack((points_transformed, other_features))

                refined_lidar_pc_list.append(points_transformed)

            for idx, points_semantic_ego in enumerate(unrefined_sem_pc_ego_list):
                pose = estimated_poses_kiss[idx]
                print(f"Applying refined pose {idx}: {pose}")

                print(f"Points semantic ego shape: {points_semantic_ego.shape}")
                points_xyz = points_semantic_ego[:, :3]
                points_homo = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
                points_transformed = (pose @ points_homo.T)[:3, :].T

                if points_semantic_ego.shape[0] > 3:
                    other_features = points_semantic_ego[:, 3:]
                    points_transformed = np.hstack((points_transformed, other_features))

                refined_lidar_pc_with_semantic_list.append(points_transformed)


    if not args.icp_refinement:
        refined_lidar_pc_list = unrefined_pc_ego_ref_list
        refined_lidar_pc_with_semantic_list = unrefined_sem_pc_ego_ref_list
        poses_kiss_icp = None

    # Assuming refined lists contain (Features, N) based on appending .T earlier
    lidar_pc_list_for_concat = [pc for pc in refined_lidar_pc_list if
                                pc.shape[1] > 0]  # Transpose to (N, Features) and filter empty
    lidar_pc_with_semantic_list_for_concat = [pc for pc in refined_lidar_pc_with_semantic_list if
                                              pc.shape[1] > 0]  # Transpose to (N, Features) and filter empty

    if lidar_pc_list_for_concat:
        # Concatenate along points axis (axis=0)
        lidar_pc_final_global = np.concatenate(lidar_pc_list_for_concat, axis=0)
        print(f"Concatenated refined static global points. Shape: {lidar_pc_final_global.shape}")
    else:
        lidar_pc_final_global = np.zeros((0, refined_lidar_pc_list[0].shape[0] if refined_lidar_pc_list else 3),
                                         dtype=np.float64)  # Empty array with correct feature count
        print("Concatenated refined static global points resulted in an empty array.")

    # --- Visualization Part ---

    # Define frames to visualize
    frame_indices_to_viz = [1, 25]  # Frame 1 (index 0), Frame 25 (index 24)
    colors = [
        [1.0, 0.0, 0.0],  # Red for frame 1
        [0.0, 0.0, 1.0]  # Blue for frame 25
    ]

    # Check if the list is long enough
    if len(refined_lidar_pc_list) <= max(frame_indices_to_viz):
        print(
            f"Error: refined_lidar_pc_list only has {len(refined_lidar_pc_list)} elements. Cannot access frame {max(frame_indices_to_viz) + 1}.")
        # Handle error appropriately, maybe exit or skip visualization
        sys.exit(1)  # Or 'pass' if you want to continue without this visualization

    point_clouds_to_visualize = []

    for i, frame_idx in enumerate(frame_indices_to_viz):
        # Get the point cloud for the specific frame
        pc_np = refined_lidar_pc_list[frame_idx]

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

    if args.vis and args.vis_position == 'aggregated_frames_global_static':
        visualize_pointcloud(lidar_pc_final_global, title=f"Aggregated Refined Static PC (Global) - Scene {scene_name}")

        ################## concatenate all semantic scene segments ########################
    if lidar_pc_with_semantic_list_for_concat:
        if lidar_pc_with_semantic_list_for_concat[0].shape[1] > 5:
            lidar_pc_with_semantic_final_global = np.concatenate(lidar_pc_with_semantic_list_for_concat,
                                                                 axis=1)  # Shape (N_total, Features)
        else:
            lidar_pc_with_semantic_final_global = np.concatenate(lidar_pc_with_semantic_list_for_concat,
                                                             axis=0)  # Shape (N_total, Features)
        print(f"Concatenated refined semantic global points. Shape: {lidar_pc_with_semantic_final_global.shape}")
    else:
        lidar_pc_with_semantic_final_global = np.zeros(
            (0, refined_lidar_pc_with_semantic_list[0].shape[0] if refined_lidar_pc_with_semantic_list else 4),
            dtype=np.float64)  # Empty array with correct feature count
        print("Concatenated refined semantic global points resulted in an empty array.")

        # --- Transpose final arrays IF the rest of your code expects (Features, N) ---
        # Example:
        # lidar_pc = lidar_pc_final_global.T
        # lidar_pc_with_semantic = lidar_pc_with_semantic_final_global.T
        # --- OR ---
        # --- Modify the rest of your code to handle (N, Features) ---
        # This depends on how 'lidar_pc' and 'lidar_pc_with_semantic' are used later.
        # Assuming the rest of the code needs (Features, N) based on original structure:
    lidar_pc = lidar_pc_final_global.T
    lidar_pc_with_semantic = lidar_pc_with_semantic_final_global.T


    if args.vis and args.vis_position == 'aggregated_frames_global_static':
        visualize_pointcloud(lidar_pc.T, title=f"Aggregated static PC in world coordinates")


    if args.filter_location in ['static_agg', 'all'] and args.filter_mode != 'none':
        if lidar_pc.shape[1] > 0:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(lidar_pc[:3, :].T)
            filtered_pcd_o3d, kept_indices = denoise_pointcloud(pcd_o3d, args.filter_mode, config, location_msg="aggregated static points")
            lidar_pc = lidar_pc[:, kept_indices]
            lidar_pc_with_semantic = lidar_pc_with_semantic[:, kept_indices] # Only here if lidar_pc and lidar_pc_semantics are the same

    """if args.filter_location in ['static_agg', 'both'] and args.filter_mode != 'none':
        if lidar_pc_with_semantic.shape[0] > 0:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(lidar_pc_with_semantic[:3, :].T)
            denoise_pointcloud.location_msg = "aggregated static semantic points"
            filtered_pcd_o3d, kept_indices = denoise_pointcloud(pcd_o3d, args.filter_mode, config)
            lidar_pc_with_semantic = lidar_pc_with_semantic[:, kept_indices]""" # Only needed if lidar_pc and lidar_pc are different pcs

    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = []  # stores unique object tokens from all frames
    object_semantic = []  # stores semantic category corresponding to each unique object
    for frame_dict in dict_list:  # Iterate through frames and collect unique objects
        for i, object_token in enumerate(frame_dict['object_tokens']):
            if object_token not in object_token_zoo:  # Filter and append object tokens
                if (frame_dict['object_points_list'][i].shape[
                    0] > 0):  # only appends objects that have at least one point
                    object_token_zoo.append(object_token)
                    object_semantic.append(frame_dict['converted_object_category'][i])
                else:
                    continue


    object_points_dict = {}  # initialize an empty dictionary to hold aggregated object points
    object_icp_voxel_size = config.get('object_icp_voxel_size', 0.1)  # Get object ICP voxel size from config

    print("\nAggregating object points...")
    for query_object_token in tqdm(object_token_zoo,
                                   desc="Aggregating Objects"):  # Loop through each unique object token
        canonical_segments_list = []
        for frame_dict in dict_list:  # iterates through all frames
            if query_object_token in frame_dict['object_tokens']:  # Check if the object exists in this frame
                obj_idx = frame_dict['object_tokens'].index(query_object_token)  # Find its index
                object_points = frame_dict['object_points_list'][obj_idx]  # retrieve raw object points

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

        # Check if any segments were collected
        if not canonical_segments_list:
            object_points_dict[query_object_token] = np.zeros((0, 3))  # Assign empty array
            continue  # Skip to the next object token

        # Conditionally Refine the collected segments using ICP >>>
        segments_to_concatenate = canonical_segments_list
        if args.icp_refinement_objects and len(canonical_segments_list) > 1:
            # print(f"  Refining object {query_object_token} ({len(canonical_segments_list)} segments) with ICP...") # Optional verbose

            MIN_ICP_POINTS = 10  # Min points required for a segment to participate in ICP

            # 2a. Choose reference segment (e.g., the largest)
            segment_lengths = [len(seg) for seg in canonical_segments_list]
            reference_idx = np.argmax(segment_lengths)
            reference_segment = canonical_segments_list[reference_idx]

            # Only proceed if reference is usable
            if reference_segment.shape[0] >= MIN_ICP_POINTS:
                aligned_segments_list = [reference_segment]  # Start new list with the reference

                # 2b. Align other segments TO the reference
                for i, segment in enumerate(canonical_segments_list):
                    if i == reference_idx:
                        continue  # Skip the reference itself

                    # Align only if source segment is also usable
                    if segment.shape[0] >= MIN_ICP_POINTS:
                        # print(f"   Aligning segment {i} ({segment.shape[0]} pts) to reference ({reference_segment.shape[0]} pts)...") # Optional
                        try:
                            # Perform ICP
                            icp_transform_obj = icp_align(
                                segment,  # Source (N_seg, 3)
                                reference_segment,  # Target (N_ref, 3)
                                init_trans=np.eye(4),
                                voxel_size=object_icp_voxel_size
                            )
                            # Apply transformation
                            xyz_seg = segment
                            xyz_seg_homo = np.hstack((xyz_seg, np.ones((xyz_seg.shape[0], 1))))
                            transformed_homo_seg = (icp_transform_obj @ xyz_seg_homo.T).T
                            segment_aligned = transformed_homo_seg[:, :3]

                            aligned_segments_list.append(segment_aligned)  # Add ALIGNED segment
                        except Exception as e:
                            print(
                                f"   ICP failed for segment {i} of object {query_object_token}: {e}. Adding UNALIGNED segment.")
                            aligned_segments_list.append(segment)  # Add original on failure
                    else:
                        # Keep small segments unaligned
                        aligned_segments_list.append(segment)

                # CRUCIAL: Point to the list containing aligned segments for concatenation
                segments_to_concatenate = aligned_segments_list
            # else: # Optional: Handle case where reference is too small
            # print(f"  Skipping ICP refinement for {query_object_token}: Reference segment too small.")
            # segments_to_concatenate remains canonical_segments_list

        if segments_to_concatenate:  # Final check if the list is not empty
            object_points_dict[query_object_token] = np.concatenate(segments_to_concatenate, axis=0)
        else:
            object_points_dict[query_object_token] = np.zeros((0, 3))


    object_data = []  # Stores all unique objects across frames

    for idx, token in enumerate(object_token_zoo):
        obj = {
            "token": token,
            "points": object_points_dict[token][:, :3],  # Canonicalized x, y, z
            "semantic": object_semantic[idx]  # Integer category
        }
        object_data.append(obj)

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
        point_cloud_with_semantic = lidar_pc_with_semantic  # retrieves transformed semantic points with labels (N, 4): [x, y, z, label]
        print(f"Original point_cloud: {point_cloud.shape}")
        print(f"Original semantic point_cloud: {point_cloud_with_semantic.shape}")

        # === Transform back to ego frame ===
        sample = truckscenes.get('sample', frame_dict['sample_token'])
        ego_pose = truckscenes.getclosest('ego_pose', sample['timestamp'])
        ego_from_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=True)

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
        point_cloud_global_xyzl = lidar_pc_with_semantic.copy()  # Global static XYZ+L (4+, N)
        print(f"Global point_cloud shape: {point_cloud_global_xyz.shape}")
        print(f"Global semantic point_cloud shape: {point_cloud_global_xyzl.shape}")

        # --- Transform scene static points (point_cloud) ---
        if point_cloud_global_xyz.shape[1] > 0:
            points_homo = np.hstack((point_cloud_global_xyz.T, np.ones((point_cloud_global_xyz.shape[1], 1))))  # (N, 4)
            points_k_ego = (T_k_from_ref @ points_homo.T).T [:, :3] # Shape (N, 4)
            # Assign the result to the loop-local variable 'point_cloud'
            point_cloud = points_k_ego.T  # Ego coords (3, N)
        else:
            point_cloud = np.zeros((3, 0), dtype=point_cloud_global_xyz.dtype)

        # --- Transform semantic points as well (point_cloud_with_semantic) ---
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
            point_cloud_with_semantic = np.zeros((num_sem_rows, 0), dtype=lidar_pc_with_semantic.dtype)

        # Prints after transformation:
        print(f"Ego point_cloud shape: {point_cloud.shape}")
        # Optional: print first few points' ego coords: print(point_cloud[:,:5])
        print(f"Ego semantic point_cloud shape: {point_cloud_with_semantic.shape}")
        # Optional: print first few points' ego coords + labels: print(point_cloud_with_semantic[:,:5])

        # --- Check: Do shapes still match? ---
        assert point_cloud.shape[1] == point_cloud_with_semantic.shape[1], \
            f"Ego point counts mismatch after transform: {point_cloud.shape[1]} vs {point_cloud_with_semantic.shape[1]}"

        ################## load bbox of target frame ##############
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
                converted_object_category.append(UNKNOWN_LEARNING_INDEX)

        labels_array = np.array(converted_object_category).reshape(-1, 1)

        """# Check if the number of labels matches the number of boxes
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

        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] -= 0.1
        gt_bbox_3d[:, 3:6] *= 1.1

        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################## bbox placement ##############
        object_points_list = []  # Final transformed points for scene
        object_semantic_list = []  # Final semantic-labeled points for scene

        for j, object_token in enumerate(frame_dict['object_tokens']):
            # Find matching object entry
            for obj in object_data:
                if object_token == obj["token"]:
                    points = obj["points"]  # Canonical (x, y, z)

                    # Rotate and translate object points to place back into current scene
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]  # Translate to bbox center

                    # Filter points inside bounding box
                    if points.shape[0] >= 5:
                        original_count = points.shape[0]
                        points_in_boxes = points_in_boxes_cpu(
                            torch.from_numpy(points[np.newaxis, :, :]),
                            torch.from_numpy(gt_bbox_3d[j:j + 1][np.newaxis, :])
                        )
                        points = points[points_in_boxes[0, :, 0].bool()]
                        filtered_count = points.shape[0]
                        if original_count > filtered_count:
                            print(
                                f"[Box Filter] Object {object_token}: reduced points from {original_count} to {filtered_count}")


                    # Append only if points remain
                    if points.shape[0] > 0:
                        object_points_list.append(points)

                        semantics = np.ones((points.shape[0], 1)) * obj["semantic"]
                        points_with_semantics = np.concatenate([points, semantics], axis=1)
                        object_semantic_list.append(points_with_semantics)

                    break  # No need to check further once matched

        """temp_points_filename = f'frame_{i}_temp_points.npy'
        sem_temp_points_filename = f'frame_{i}_sem_temp_points.npy'
        dirs = os.path.join(save_path, scene_name, frame_dict['sample_token'])  #### Save in folder with scene name
        if not os.path.exists(dirs):  # create directory if does not exist
            os.makedirs(dirs)
        output_filepath_temp = os.path.join(dirs, temp_points_filename)
        output_filepath_sem_temp = os.path.join(dirs, sem_temp_points_filename)"""
        # Combine Scene Points with Object Points
        try:  # avoid concatenate an empty array
            temp = np.concatenate(object_points_list)

            """# ---- EXPORT temp points ----
            np.save(output_filepath_temp, temp)
            print(f"Saved temp object points for frame {i} to {output_filepath_temp}")
            # ----------------------------"""
            # scene_points = point_cloud.T
            scene_points = np.concatenate(
                [point_cloud.T, temp])  # Merge static scene points and object points
        except:
            scene_points = point_cloud  # If no object points, only use static scene points
        try:
            temp = np.concatenate(object_semantic_list)
            """# ---- EXPORT temp points ----
            np.save(output_filepath_sem_temp, temp)
            print(f"Saved temp object points for frame {i} to {output_filepath_sem_temp}")
            # ----------------------------"""
            # scene_semantic_points = point_cloud_with_semantic.T
            scene_semantic_points = np.concatenate(
                [point_cloud_with_semantic.T, temp])  # Merge semantic points from objects and static scenes
        except:
            scene_semantic_points = point_cloud_with_semantic  # If no object points, only use static scene points

        print(f"Scene points before applying range filtering: {scene_points.shape}")
        mask = in_range_mask(scene_points, pc_range)

        scene_points = scene_points[mask]
        print(f"Scene points after applying range filtering: {scene_points.shape}")

        visualize_pointcloud_bbox(scene_points,
                                  boxes=boxes,
                                  title=f"Fused dynamic and static PC + BBoxes + Ego BBox - Frame {i}",
                                  self_vehicle_range=self_range,
                                  vis_self_vehicle=True)

        ################## get semantics of sparse points  ##############
        print(f"Scene semantic points before applying range filtering: {scene_semantic_points.shape}")
        mask = in_range_mask(scene_semantic_points, pc_range)

        scene_semantic_points = scene_semantic_points[mask]  # Filter points within a spatial range
        print(f"Scene semantic points after applying range filtering: {scene_semantic_points.shape}")

        if args.filter_location in ['final', 'all'] and args.filter_mode != 'none':
            print(f"Filtering {scene_points.shape}")
            if scene_points.shape[0] > 0:
                pcd_to_filter = o3d.geometry.PointCloud()
                pcd_to_filter.points = o3d.utility.Vector3dVector(scene_points[:, :3])
                print('test')

                filtered_pcd, _ = denoise_pointcloud(pcd_to_filter, args.filter_mode, config, location_msg="final scene points")
                scene_points = np.asarray(filtered_pcd.points)


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

        # --- Common finalization and saving logic ---
        print(
            f"Shape of dense_voxels_with_semantic_voxelcoords for saving: {dense_voxels_with_semantic_voxelcoords_save.shape}")
        print(
            f"Occupancy shape: Occsize: {occupancy_grid.shape}, Total number voxels: {occupancy_grid.shape[0] * occupancy_grid.shape[1] * occupancy_grid.shape[2]}, Occupied: {total_occupied_voxels}")

        ##########################################save like Occ3D #######################################
        # Save as .npz
        dirs = os.path.join(save_path, scene_name, frame_dict['sample_token'])
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        save_path_base = 'labels'
        suffix_to_remove = '.pcd'
        if save_path_base.endswith(suffix_to_remove):
            save_path_base = save_path_base[:-len(suffix_to_remove)]

        output_filepath = os.path.join(dirs, save_path_base + '.npz')
        print(f"Saving semantic occupancy grid to {output_filepath}...")
        np.savez_compressed(output_filepath, semantics=occupancy_grid)

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

    # Define Command-Line Arguments
    parse.add_argument('--dataset', type=str, default='truckscenes') # Dataset selection with default: "truckScenes
    parse.add_argument('--config_path', type=str, default='config_truckscenes.yaml') # Configuration file path with default: "config.yaml"
    parse.add_argument('--split', type=str, default='train') # data split, default: "train", options: "train", "val", "all"
    parse.add_argument('--save_path', type=str, default='./data/GT_occupancy/') # save path, default: "./data/GT/GT_occupancy"
    parse.add_argument('--start', type=int, default=0) # start indice, default: 0, determines range of sequences to process
    parse.add_argument('--end', type=int, default=850) # end indice, default: 850, determines range of sequences to process
    parse.add_argument('--dataroot', type=str, default='./data/truckscenes/') # data root path, default: "./data/truckScenes
    parse.add_argument('--trucksc_val_list', type=str, default='./truckscenes_val_list.txt') # text file containing validation scene tokens, default: "./truckscenes_val_list.txt"
    parse.add_argument('--label_mapping', type=str, default='truckscenes.yaml') # YAML file containing label mappings, default: "truckscenes.yaml"
    parse.add_argument('--meshing', action='store_true', help='Enable meshing')
    parse.add_argument('--icp_refinement', action='store_true', help='Enable ICP refinement')
    parse.add_argument('--icp_refinement_objects', action='store_true',
                       help='Enable ICP refinement of canonical object segments')
    parse.add_argument('--load_mode', type=str, default='pointwise') # pointwise or rigid
    parse.add_argument('--filter_mode', type=str, default='none', choices=['none', 'sor', 'ror', 'both'],
                       help='Noise filtering method to apply before meshing')
    parse.add_argument('--filter_location', type=str, default='none', choices=['none', 'static_agg', 'final', 'ego_static', 'all'],
                       help='Where to apply noise filtering: none, static_agg (aggregated static), final (final scene points), or all')
    parse.add_argument('--vis', action='store_true', help='Enable visualization of intermediate point clouds')
    parse.add_argument('--vis_position', type=str, default='fused_sensors_ego',
                       choices=['fused_sensors_ego', 'fused_sensors_ego_boxes_static', 'fused_sensors_global_static',
                                'aggregated_frames_global_static', 'final'], )
    args=parse.parse_args()


    if args.dataset=='truckscenes': # check dataset type
        val_list = []
        with open(args.trucksc_val_list, 'r') as file: # Load validation scene tokens to val_list
            for item in file:
                val_list.append(item[:-1])
        file.close()

        # Load the truckScenes dataset
        truckscenes = TruckScenes(version='v1.0-trainval',
                               dataroot=args.dataroot,
                               verbose=True) # verbose True to print informational messages
        train_scenes = splits.train # train_scenes and val_scenes contain the scene tokens for the training and validation splits
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
    for i in range(args.start,args.end):
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
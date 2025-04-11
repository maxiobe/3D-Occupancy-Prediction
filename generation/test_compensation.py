import argparse
import os.path as osp

from functools import reduce
from typing import Any, Dict, List

import matplotlib
import numpy as np
import open3d as o3d

from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from pyquaternion import Quaternion

from truckscenes import TruckScenes
from truckscenes.utils import colormap
from truckscenes.utils.data_classes import Box, LidarPointCloud
from truckscenes.utils.geometry_utils import transform_matrix, points_in_box


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


def get_rigit_fused_pointcloud(trucksc: TruckScenes, sample: Dict[str, Any]) -> LidarPointCloud:
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


def get_pointwise_fused_pointcloud(trucksc: TruckScenes, sample: Dict[str, Any]) -> LidarPointCloud:
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


def visualize_pointcloud(point_cloud: LidarPointCloud, trucksc: TruckScenes,
                         sample: Dict[str, Any], with_anns: bool = True) -> None:
    """Visualizes the given point cloud and annotations.
    """
    # Extract points and intensities
    points = point_cloud.points[:3, :].T
    intensities = point_cloud.points[3, :].T

    # Convert intensities to colors
    rgb = matplotlib.colormaps['viridis'](intensities)[..., :3]

    # Initialize vizualization objets
    vis_obj = []

    # Define point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    vis_obj.append(pcd)

    if with_anns:
        # Get boxes in ego frame
        boxes = get_boxes(trucksc, sample)

        # Define bounding boxes
        for box in boxes:
            # Initialize bounding box
            bbox = o3d.geometry.OrientedBoundingBox()

            # Set bounding box center and orientation
            bbox.center = box.center
            bbox.extent = box.wlh[[1, 0, 2]]
            bbox.R = Quaternion(box.orientation).rotation_matrix

            # Set bounding box color
            bbox.color = np.asarray(colormap.get_colormap()[box.name]) / 255

            # Add bounding box to visualization objects
            vis_obj.append(bbox)

    # Set visualization options
    rend = o3d.visualization.RenderOption()
    rend.line_width = 8.0
    vis = o3d.visualization.Visualizer()
    vis.update_renderer()
    vis.create_window()

    # Visualize all objects (point cloud and boxes)
    for obj in vis_obj:
        vis.add_geometry(obj)
        vis.poll_events()
        vis.update_geometry(obj)

    # Visualize
    vis.run()


def main(src: str, version: str = 'v1.0-mini', mode: str = 'pointwise') -> None:
    """
    src: Dataset root path.
    version: Dataset version.
    mode: One of either: rigit, sectorwise, or pointwise.
    """
    # Load TruckScenes dataset
    trucksc = TruckScenes(version=version, dataroot=src)

    for scene in trucksc.scene:
        print(f"Scene {scene['token']}:")

        # Get first sample of the scene
        sample = trucksc.get('sample', scene['first_sample_token'])

        # Iterate over all samples in the scene
        for _ in range(scene['nbr_samples'] - 1):
            # Get fused point cloud
            if mode == 'rigit':
                point_cloud = get_rigit_fused_pointcloud(trucksc, sample)
            elif mode == 'pointwise':
                point_cloud = get_pointwise_fused_pointcloud(trucksc, sample)
            else:
                raise ValueError(f'Fusion mode {mode} is not supported')

            # Initialize counter for empty bounding boxes
            empty_boxes = 0

            # Count annotations without points
            boxes = get_boxes(trucksc, sample)
            boxes_with_points = [np.any(points_in_box(box, point_cloud.points[:3, :])) for box in boxes]

            empty_boxes += len(boxes) - sum(boxes_with_points)

            print(f"Sample {sample['token']} has {empty_boxes} annotations without points!")

            # Visualize fused point cloud
            visualize_pointcloud(point_cloud, trucksc, sample)

            # Get next sample in the scene
            sample = trucksc.get('sample', sample['next'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default=r'D:\data\truckscenes',
                        help="Path to the dataset folder.")
    parser.add_argument('--version', type=str, default='v1.0-test',
                        help="Dataset version.")
    args = parser.parse_args()
    main(args.src, args.version)
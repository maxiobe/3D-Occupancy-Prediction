import argparse
import os
import open3d as o3d
import yaml
import numpy as np
from truckscenes.truckscenes import TruckScenes
from typing import Any, Dict, List
from pyquaternion import Quaternion
from truckscenes.utils.data_classes import Box, LidarPointCloud

from pathlib import Path
import yaml
import sys

# Define a default color for labels not in the map
DEFAULT_COLOR = [0.0, 0.0, 0.0]  # Black

def load_yaml(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: YAML file not found at {path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {path}: {e}", file=sys.stderr)
        sys.exit(1)

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


def visualize_bboxes(trucksc, truckscenes_dataroot, sample_token, config_path, npz_path, ref_sensor):
    config = load_yaml(config_path)

    voxel_size = config.get("voxel_size")
    pc_range = config.get("pc_range")
    if voxel_size is None or pc_range is None:
        print("Error: 'voxel_size' or 'pc_range' not found in config file.", file=sys.stderr)
        sys.exit(1)

    npz_load_path = Path(npz_path)
    if not npz_load_path.is_file():
        print(f"Error: File not found at {npz_load_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {npz_load_path}...")
    try:
        data = np.load(npz_load_path)
        available_keys_message = f"Available keys/arrays in the archive: {data.files}"
        print(available_keys_message)
        semantics = data['semantics']
    except Exception as e:
        print(f"Error loading .npz file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Shape of semantics: {semantics.shape}")

    valid_mask = (semantics >= 0) & (semantics <= 16)
    voxel_indices = np.stack(np.nonzero(valid_mask), axis=-1)

    if voxel_indices.shape[0] == 0:
        print("No valid voxels found to visualize.", file=sys.stderr)
        sys.exit(1)

    # Convert voxel indices to world coordinates (center of voxel)
    world_coords = (voxel_indices + 0.5) * voxel_size + np.array(pc_range[:3])

    print("voxel_indices shape:", voxel_indices.shape)
    print("world_coords shape:", world_coords.shape)

    # Get label values for each voxel
    labels = semantics[tuple(voxel_indices.T)]

    print("labels shape:", labels.shape)

    # Define class colormap for 0–16
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
    }

    # Assign colors using color map
    point_colors = np.array([CLASS_COLOR_MAP.get(int(label), DEFAULT_COLOR) for label in labels])

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coords)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    my_sample = trucksc.get('sample', sample_token)
    # print(my_sample)
    boxes = get_boxes(trucksc, my_sample)

    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

    if locs.size > 0:
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.  # fix yaw mismatch
        #gt_bbox_3d[:, 2] -= dims[:, 2] / 2.  # lower center Z to base
        #gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # slightly lower
        #gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1  # make box a bit bigger
    else:
        gt_bbox_3d = np.empty((0, 7), dtype=np.float32)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for i in range(gt_bbox_3d.shape[0]):
        center = gt_bbox_3d[i, 0:3]
        w, l, h = gt_bbox_3d[i, 3:6]
        yaw = gt_bbox_3d[i, 6]

        # Convert yaw to rotation matrix (yaw around Z axis)
        rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

        bbox = o3d.geometry.OrientedBoundingBox(center, rot_mat, [w, l, h])
        bbox.color = (1, 0, 0)
        vis.add_geometry(bbox)

    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize boxes from truck scenes')
    parser.add_argument("--npz_path", type=str, help="Path to npy file")
    parser.add_argument("--config_path", type=str, default="config_truckscenes.yaml")
    parser.add_argument("--truckscenes_dataroot", type=str, default=None)
    parser.add_argument("--truckscenes_yaml", type=str, default="truckscenes.yaml")
    parser.add_argument("--trsc_version", type=str, default="v1.0-trainval")
    parser.add_argument("--sample_token", type=str, default=None)
    parser.add_argument("--ref_sensor", type=str, default='LIDAR_LEFT')

    args = parser.parse_args()

    trucksc = TruckScenes(version=args.trsc_version, dataroot=args.truckscenes_dataroot, verbose=True)


    visualize_bboxes(trucksc, args.truckscenes_dataroot, args.sample_token, args.config_path, args.npz_path, args.ref_sensor)
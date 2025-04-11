import argparse
import open3d as o3d
import yaml
import numpy as np
from pathlib import Path
import sys


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


def visualize_pointcloud(config_path, npy_path):
    config = load_yaml(config_path)
    voxel_size = config.get("voxel_size")
    pc_range = config.get("pc_range")

    npy_load_path = Path(npy_path)
    if not npy_load_path.exists():
        print(f"Error: .npy file not found at {npy_path}")
        return

    voxel_data = np.load(str(npy_load_path))
    voxel_xyz = voxel_data[:, :3].astype(np.float32)
    labels = voxel_data[:, 3:].astype(np.int32)

    world_coords = np.zeros_like(voxel_xyz)
    world_coords[:, 0] = pc_range[0] + (voxel_xyz[:, 0] + 0.5) * voxel_size
    world_coords[:, 1] = pc_range[1] + (voxel_xyz[:, 1] + 0.5) * voxel_size
    world_coords[:, 2] = pc_range[2] + (voxel_xyz[:, 2] + 0.5) * voxel_size

    CLASS_COLOR_MAP = {
        0: [0.6, 0.6, 0.6],  # noise
        1: [0.9, 0.1, 0.1],  # barrier
        2: [1.0, 0.6, 0.0],  # bicycle
        3: [0.5, 0.0, 0.5],  # bus
        4: [0.0, 0.0, 1.0],  # car
        5: [0.3, 0.3, 0.0],  # construction_vehicle
        6: [1.0, 0.0, 1.0],  # motorcycle
        7: [1.0, 1.0, 0.0],  # pedestrian
        8: [1.0, 0.5, 0.5],  # traffic_cone
        9: [0.5, 0.5, 0.0],  # trailer
        10: [0.0, 1.0, 0.0],  # truck
        11: [0.2, 0.8, 0.8],  # ego_vehicle
        12: [1.0, 0.8, 0.0],  # traffic_sign
        13: [0.4, 0.4, 0.8],  # other_vehicle
        14: [0.0, 0.5, 0.5],  # train
        15: [0.8, 0.8, 0.8],  # Unknown
        16: [0.0, 1.0, 1.0],  # Background
    }

    colors = np.array([CLASS_COLOR_MAP.get(label[0], [1.0, 1.0, 1.0]) for label in labels], dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Semantic Point Cloud")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize semantic point cloud from npy')
    parser.add_argument("--npy_path", type=str, required=True, help="Path to .npy file")
    parser.add_argument("--config_path", type=str, default="config_truckscenes.yaml", help="Path to config YAML")

    args = parser.parse_args()
    visualize_pointcloud(args.config_path, args.npy_path)

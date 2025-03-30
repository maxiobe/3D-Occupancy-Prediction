import open3d as o3d
import numpy as np
import yaml
import os
import sys
import argparse
from pathlib import Path

DEFAULT_COLOR_MAP_RAW = {
    0: [0, 0, 0],        # Noise / Ignore? Default: Black
    1: [70, 130, 180],   # Barrier (SteelBlue)
    2: [220, 20, 60],    # Bicycle (Crimson)
    3: [255, 127, 80],   # Bus (Coral)
    4: [255, 0, 0],      # Car (Red) - Often prominent
    5: [255, 158, 0],    # Construction Vehicle (Orange variant)
    6: [233, 150, 70],   # Motorcycle (DaskOrange)
    7: [0, 0, 230],      # Pedestrian (Blue) - Often prominent
    8: [255, 61, 99],    # Traffic Cone (Red variant)
    9: [0, 207, 191],    # Trailer (Turquoise variant)
    10: [75, 0, 75],     # Truck (Indigo variant)
    11: [160, 160, 160], # Driveable Surface (Gray)
    12: [107, 142, 35],  # Other Flat (OliveDrab)
    13: [245, 245, 245], # Sidewalk (WhiteSmoke)
    14: [0, 175, 0],     # Terrain (Green variant)
    15: [102, 102, 102], # Manmade (Gray)
    16: [60, 179, 113],  # Vegetation (MediumSeaGreen)
    17: [255, 255, 255], # Free Space? (White) - Use if label 17 exists
    # Add more indices if your learning_map created more than 17 classes
}
# Normalize to 0-1 range for Open3D
DEFAULT_COLOR_MAP = {k: [c/255.0 for c in v] for k, v in DEFAULT_COLOR_MAP_RAW.items()}
DEFAULT_COLOR = [0.5, 0.5, 0.5] # Gray for unknown labels

def load_yaml(path):
    """Loads a YAML file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: YAML file not found at {path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {path}: {e}", file=sys.stderr)
        sys.exit(1)

def get_semantic_color_map(nuscenes_yaml_path):
    """Tries to load color map from nuscenes.yaml, otherwise uses default."""
    if not nuscenes_yaml_path or not Path(nuscenes_yaml_path).is_file():
        print("Warning: nuscenes.yaml not found or specified. Using default color map.", file=sys.stderr)
        return DEFAULT_COLOR_MAP

    nuscenes_config = load_yaml(nuscenes_yaml_path)
    if 'color_map' not in nuscenes_config:
        print("Warning: 'color_map' not found in nuscenes.yaml. Using default color map.", file=sys.stderr)
        return DEFAULT_COLOR_MAP

    try:
        # Assuming color_map in yaml is {label_index: [R, G, B]} with 0-255 values
        color_map_raw = nuscenes_config['color_map']
        color_map_normalized = {k: [c/255.0 for c in v] for k, v in color_map_raw.items()}
        print(f"Loaded color map from {nuscenes_yaml_path}")
        return color_map_normalized
    except Exception as e:
        print(f"Error processing color map from nuscenes.yaml: {e}. Using default.", file=sys.stderr)
        return DEFAULT_COLOR_MAP


def visualize_npy(npy_path, config_path, nuscenes_yaml_path):
    """Loads, processes, and visualizes the semantic voxel .npy file."""
    npy_file = Path(npy_path)
    if not npy_file.is_file():
        print(f"Error: NPY file not found at {npy_path}", file=sys.stderr)
        sys.exit(1)

    # Load config to get voxel size and pc range
    config = load_yaml(config_path)
    voxel_size = config.get('voxel_size')
    pc_range = config.get('pc_range')
    if voxel_size is None or pc_range is None:
        print("Error: 'voxel_size' or 'pc_range' not found in config file.", file=sys.stderr)
        sys.exit(1)
    if len(pc_range) < 3:
        print("Error: 'pc_range' in config must have at least 3 values (xmin, ymin, zmin).", file=sys.stderr)
        sys.exit(1)

    # Load the semantic color map
    color_map = get_semantic_color_map(nuscenes_yaml_path)

    # Load the NPY file
    print(f"Loading data from {npy_path}...")
    try:
        # Data is expected as [vx, vy, vz, label] (Integers)
        voxel_data = np.load(npy_path)
    except Exception as e:
        print(f"Error loading NPY file {npy_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if voxel_data.ndim != 2 or voxel_data.shape[1] != 4:
        print(f"Error: Expected NPY data to have shape (N, 4), but got {voxel_data.shape}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {voxel_data.shape[0]} voxels.")

    # Separate coordinates and labels
    voxel_coords = voxel_data[:, :3].astype(float) # Use float for calculations
    labels = voxel_data[:, 3].astype(int)

    # Convert voxel coordinates to world coordinates (center of voxel)
    world_coords = np.zeros_like(voxel_coords)
    world_coords[:, 0] = (voxel_coords[:, 0] + 0.5) * voxel_size + pc_range[0]
    world_coords[:, 1] = (voxel_coords[:, 1] + 0.5) * voxel_size + pc_range[1]
    world_coords[:, 2] = (voxel_coords[:, 2] + 0.5) * voxel_size + pc_range[2]

    # Assign colors based on labels
    colors = np.zeros((world_coords.shape[0], 3))
    unique_labels = np.unique(labels)
    print(f"Found unique labels: {unique_labels}")
    for lab in unique_labels:
        mask = (labels == lab)
        colors[mask] = color_map.get(lab, DEFAULT_COLOR) # Use default color if label not in map

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    print("Displaying point cloud... Close the window to exit.")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize nuScenes GT occupancy .npy files.")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to the .npy file to visualize.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config.yaml file used for generation.")
    parser.add_argument("--nuscenes_yaml", type=str, default=None, help="(Optional) Path to the nuscenes.yaml file containing label/color maps.")

    args = parser.parse_args()

    visualize_npy(args.npy_path, args.config_path, args.nuscenes_yaml)
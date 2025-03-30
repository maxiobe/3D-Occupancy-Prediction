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


def visualize_npz(npz_path, config_path, nuscenes_yaml_path=None):
    """Visualize voxel centers from a dense .npz voxel grid using color map."""
    npz_file = Path(npz_path)
    if not npz_file.is_file():
        print(f"Error: File not found at {npz_path}", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = load_yaml(config_path)
    voxel_size = config.get('voxel_size')
    pc_range = config.get('pc_range')
    if voxel_size is None or pc_range is None:
        print("Error: 'voxel_size' or 'pc_range' not found in config file.", file=sys.stderr)
        sys.exit(1)

    # Load color map
    color_map = get_semantic_color_map(nuscenes_yaml_path)

    print(f"Loading data from {npz_path}...")
    try:
        data = np.load(npz_path)
        semantics = data['semantics']
        mask_lidar = data['mask_lidar']
    except Exception as e:
        print(f"Error loading .npz file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Shape of semantics: {semantics.shape}")
    print(f"Shape of mask_lidar: {mask_lidar.shape}")

    # Only visualize valid voxels with labels 1-16 (ignore 0 and 17)
    valid_mask = mask_lidar & (semantics > 0) & (semantics < 17)

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

    # Assign colors using color map
    colors = np.array([color_map.get(int(label), DEFAULT_COLOR) for label in labels])

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Displaying {len(world_coords)} voxel centers with label-based colors...")
    o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize nuScenes GT occupancy .npz files.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the .npz file to visualize.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config.yaml file used for generation.")
    parser.add_argument("--nuscenes_yaml", type=str, default=None, help="(Optional) Path to the nuscenes.yaml file containing label/color maps.")

    args = parser.parse_args()

    visualize_npz(args.npz_path, args.config_path, args.nuscenes_yaml)
import open3d as o3d
import numpy as np
import yaml
import os
import sys
import argparse
from pathlib import Path
# Use try-except for optional import
try:
    from truckscenes.truckscenes import TruckScenes # Import TruckScenes devkit
    from truckscenes.utils.data_classes import Box # To handle TruckScenes boxes
    from pyquaternion import Quaternion # For rotation handling
    TRUCKSCENES_AVAILABLE = True
except ImportError:
    TRUCKSCENES_AVAILABLE = False
    print("Warning: 'truckscenes-devkit' not found. Bounding box visualization will be unavailable.", file=sys.stderr)
    print("Install with: pip install git+https://github.com/ika-rwth-aachen/truckscenes-devkit.git", file=sys.stderr)


# --- (Keep your existing DEFAULT_COLOR_MAP_RAW and DEFAULT_COLOR_MAP) ---
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
}
DEFAULT_COLOR_MAP = {k: [c/255.0 for c in v] for k, v in DEFAULT_COLOR_MAP_RAW.items()}
DEFAULT_COLOR = [0.5, 0.5, 0.5] # Gray for unknown labels
BBOX_COLOR = [0, 1, 0] # Green for bounding boxes

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
        print("Info: nuscenes.yaml not found or specified. Using default color map.", file=sys.stderr)
        return DEFAULT_COLOR_MAP

    nuscenes_config = load_yaml(nuscenes_yaml_path)
    if 'color_map' not in nuscenes_config:
        print("Warning: 'color_map' not found in nuscenes.yaml. Using default color map.", file=sys.stderr)
        return DEFAULT_COLOR_MAP

    try:
        # Assuming color_map in yaml is {label_index: [R, G, B]} with 0-255 values
        color_map_raw = nuscenes_config['color_map']
        color_map_normalized = {k: [c/255.0 for c in v] for k, v in color_map_raw.items()}
        print(f"Loaded semantic color map from {nuscenes_yaml_path}")
        return color_map_normalized
    except Exception as e:
        print(f"Error processing color map from nuscenes.yaml: {e}. Using default.", file=sys.stderr)
        return DEFAULT_COLOR_MAP

def get_truckscenes_bboxes(trsc, sample_token):
    """Loads GT bounding boxes for a given sample token from TruckScenes."""
    # This function requires TRUCKSCENES_AVAILABLE to be True
    if not TRUCKSCENES_AVAILABLE:
        print("Error: Cannot load bounding boxes, truckscenes-devkit is not available.", file=sys.stderr)
        return []

    boxes = []
    if not sample_token:
         print("Error: No sample_token provided for bounding box lookup.", file=sys.stderr)
         return boxes # Return empty list if no token

    try:
        sample = trsc.get('sample', sample_token)
    except KeyError:
        print(f"Error: Sample token '{sample_token}' not found in TruckScenes dataset.", file=sys.stderr)
        return boxes
    except Exception as e:
        print(f"Error retrieving sample '{sample_token}' from TruckScenes: {e}", file=sys.stderr)
        return boxes

    print(f"Found {len(sample['anns'])} annotations for sample {sample_token}.")
    for ann_token in sample['anns']:
        try:
            ann_record = trsc.get('sample_annotation', ann_token)
            # Create a TruckScenes Box object
            box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']),
                      name=ann_record['category_name'], token=ann_record['token'])

            # Create an Open3D OrientedBoundingBox
            center = box.center
            rotation_matrix = box.rotation_matrix
            # TruckScenes size is [width, length, height].
            # Open3D extent uses [length, width, height] convention more often.
            extent = [box.wlh[1], box.wlh[0], box.wlh[2]] # l, w, h

            obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
            obb.color = BBOX_COLOR # Assign a specific color to boxes
            boxes.append(obb)
        except KeyError as e:
            print(f"Warning: Skipping annotation {ann_token} due to missing key: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Skipping annotation {ann_token} due to error: {e}", file=sys.stderr)


    print(f"Created {len(boxes)} Open3D bounding box geometries.")
    return boxes

def extract_token_from_filename(filename):
    """
    Extracts the sample token from the NPY filename as a fallback.
    Attempts to find a 32-character hex string.
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    for part in parts:
        if len(part) == 32 and all(c in '0123456789abcdefABCDEF' for c in part):
            print(f"Extracted sample token from filename (fallback): {part}")
            return part
    print(f"Warning: Could not extract 32-char token from filename '{filename}'. BBox loading might fail if --sample_token not provided.", file=sys.stderr)
    return None


def visualize_npy(npy_path, config_path, nuscenes_yaml_path, truckscenes_dataroot, sample_token_arg):
    """Loads, processes, and visualizes the semantic voxel .npy file and optionally TruckScenes bboxes."""
    npy_file = Path(npy_path)
    if not npy_file.is_file():
        print(f"Error: NPY file not found at {npy_path}", file=sys.stderr)
        sys.exit(1)

    # Load config to get voxel size and pc range
    config = load_yaml(config_path)
    voxel_size_cfg = config.get('voxel_size') # Might be list [vx, vy, vz]
    pc_range = config.get('pc_range')
    if voxel_size_cfg is None or pc_range is None:
        print("Error: 'voxel_size' or 'pc_range' not found in config file.", file=sys.stderr)
        sys.exit(1)
    if len(pc_range) < 3:
        print("Error: 'pc_range' in config must have at least 3 values (xmin, ymin, zmin).", file=sys.stderr)
        sys.exit(1)

    # Handle voxel size
    if isinstance(voxel_size_cfg, list):
        if len(voxel_size_cfg) < 3:
             print("Error: 'voxel_size' list must have at least 3 values (vx, vy, vz).", file=sys.stderr)
             sys.exit(1)
        voxel_size_x, voxel_size_y, voxel_size_z = voxel_size_cfg[0], voxel_size_cfg[1], voxel_size_cfg[2]
        print(f"Using voxel sizes: vx={voxel_size_x}, vy={voxel_size_y}, vz={voxel_size_z}")
    else: # Assume float/int -> uniform voxel size
        voxel_size_x = voxel_size_y = voxel_size_z = float(voxel_size_cfg)
        print(f"Using uniform voxel size: {voxel_size_x}")


    # Load the semantic color map
    color_map = get_semantic_color_map(nuscenes_yaml_path)

    # Load the NPY file
    print(f"Loading semantic occupancy data from {npy_path}...")
    try:
        voxel_data = np.load(npy_path)
    except Exception as e:
        print(f"Error loading NPY file {npy_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if voxel_data.ndim != 2 or voxel_data.shape[1] != 4:
        print(f"Error: Expected NPY data to have shape (N, 4), but got {voxel_data.shape}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {voxel_data.shape[0]} voxels.")

    # Separate coordinates and labels
    voxel_indices = voxel_data[:, :3].astype(float)
    labels = voxel_data[:, 3].astype(int)

    # Convert voxel indices to world coordinates
    world_coords = np.zeros_like(voxel_indices)
    world_coords[:, 0] = pc_range[0] + (voxel_indices[:, 0] + 0.5) * voxel_size_x
    world_coords[:, 1] = pc_range[1] + (voxel_indices[:, 1] + 0.5) * voxel_size_y
    world_coords[:, 2] = pc_range[2] + (voxel_indices[:, 2] + 0.5) * voxel_size_z

    # Assign colors based on labels
    colors = np.zeros((world_coords.shape[0], 3))
    unique_labels = np.unique(labels)
    print(f"Found unique semantic labels in NPY: {unique_labels}")
    for lab in unique_labels:
        mask = (labels == lab)
        colors[mask] = color_map.get(lab, DEFAULT_COLOR)

    # Create Open3D point cloud for semantic voxels
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- Load TruckScenes Bounding Boxes ---
    geometries_to_visualize = [pcd] # Start with the point cloud
    effective_sample_token = sample_token_arg # Use the argument first

    if truckscenes_dataroot:
        # Check if devkit is actually available
        if not TRUCKSCENES_AVAILABLE:
             print("Error: Cannot load bounding boxes because truckscenes-devkit is not installed.", file=sys.stderr)
        # If dataroot is given, a sample token is required
        elif not effective_sample_token:
            print("Info: --sample_token not provided, attempting to extract from filename...")
            effective_sample_token = extract_token_from_filename(npy_file.name)
            if not effective_sample_token:
                 print("Error: --truckscenes_dataroot provided, but could not determine sample_token (provide --sample_token argument).", file=sys.stderr)
                 # Decide whether to exit or just continue without boxes
                 # sys.exit(1) # Option: exit if token is absolutely required
                 print("Continuing visualization without bounding boxes.", file=sys.stderr)

        # Proceed if we have a dataroot, the devkit, and a token
        if TRUCKSCENES_AVAILABLE and effective_sample_token:
            dataroot_path = Path(truckscenes_dataroot)
            if not dataroot_path.is_dir():
                print(f"Error: TruckScenes dataroot directory not found at '{truckscenes_dataroot}'", file=sys.stderr)
            else:
                try:
                    print(f"Initializing TruckScenes (version: v1.0-trainval, dataroot: {truckscenes_dataroot})")
                    trsc = TruckScenes(version='v1.0-trainval', dataroot=str(dataroot_path), verbose=True)

                    print(f"Loading bounding boxes for sample token: {effective_sample_token}")
                    bboxes = get_truckscenes_bboxes(trsc, effective_sample_token)
                    if bboxes:
                       geometries_to_visualize.extend(bboxes) # Add boxes to the list

                except Exception as e:
                     print(f"Error loading or processing TruckScenes data: {e}", file=sys.stderr)
                     print("Please ensure the dataroot and version are correct and the dataset is intact.", file=sys.stderr)

    # Visualize
    print(f"Displaying {len(geometries_to_visualize)} geometries... Close the window to exit.")
    if not geometries_to_visualize:
        print("Warning: No geometries to display.", file=sys.stderr)
        return

    try:
        o3d.visualization.draw_geometries(geometries_to_visualize)
    except Exception as e:
        print(f"Error during Open3D visualization: {e}", file=sys.stderr)
        print("Ensure you have a display environment available (e.g., not running headless).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize semantic occupancy .npy files and optionally TruckScenes GT bounding boxes.")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to the semantic occupancy .npy file to visualize.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config.yaml file used for occupancy generation (contains voxel_size, pc_range).")
    parser.add_argument("--truckscenes_yaml", type=str, default=None, help="(Optional) Path to the nuscenes.yaml/truckscenes.yaml file containing label/color maps for semantic voxels.")
    parser.add_argument("--truckscenes_dataroot", type=str, default=None, help="(Optional) Path to the root directory of the TruckScenes dataset (e.g., /path/to/truckscenes). If provided, enables bounding box loading.")
    # --- Added argument for sample_token ---
    parser.add_argument("--sample_token", type=str, default=None, help="(Optional) The specific sample_token for the frame to load bounding boxes for. Preferred over filename extraction if --truckscenes_dataroot is used.")
    # --- Optional: Argument for TruckScenes version ---
    #parser.add_argument("--trsc_version", type=str, default="v1.0-trainval", help="Version string for TruckScenes dataset (e.g., 'v1.0-trainval', 'v1.0-test')")

    args = parser.parse_args()

    # Pass the sample_token argument to the main function
    visualize_npy(
        args.npy_path,
        args.config_path,
        args.truckscenes_yaml, # Note: Renamed arg for clarity, keep nuscenes_yaml internally for now
        args.truckscenes_dataroot,
        args.sample_token

    )
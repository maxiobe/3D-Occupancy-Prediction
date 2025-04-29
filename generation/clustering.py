import numpy as np
import open3d as o3d
import os
from collections import defaultdict # Added for counting

# Define class colormap (provided by user)
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
# Add a default color for labels not in the map
DEFAULT_COLOR = [0.5, 0.5, 0.5] # Medium gray
OVERLAP_COLOR = [1.0, 0.1, 0.7] # Bright Pink for overlapping points

def visualize_object_data(data_directory, highlight_overlap=True): # Added flag
    """
    Loads bounding boxes and object points, applies transformations to boxes,
    identifies points in multiple boxes, and visualizes them.

    Args:
        data_directory (str): The path to the directory containing the .npy files.
        highlight_overlap (bool): If True, colors points in multiple boxes white.
    """
    print(f"--- Loading data from: {data_directory} ---")

    # --- Define File Paths ---
    bbox_file = os.path.join(data_directory, 'frame_0_gt_bbox_3d.npy')
    # obj_pts_file = os.path.join(data_directory, 'frame_0_temp_points.npy') # Original points without labels
    obj_sem_pts_file = os.path.join(data_directory, 'frame_0_sem_temp_points.npy') # Points with labels

    # --- Load Data ---
    object_points = np.empty((0, 3))
    object_labels = np.empty((0, 1), dtype=int)
    gt_bboxes_data_original = np.empty((0, 7)) # Store original loaded boxes here

    # Load Object Points with Semantic Labels
    try:
        loaded_obj_sem = np.load(obj_sem_pts_file)
        if loaded_obj_sem.ndim == 2 and loaded_obj_sem.shape[1] >= 4:
            object_points = loaded_obj_sem[:, :3]
            # Ensure labels are integers and have shape (N, 1)
            object_labels = loaded_obj_sem[:, 3].astype(int).reshape(-1, 1)
            print(f"Loaded object points: {object_points.shape}, labels: {object_labels.shape}")
        else:
            print(f"Warning: Object semantic points file {obj_sem_pts_file} has unexpected shape {loaded_obj_sem.shape}. Expected Nx4.")
    except FileNotFoundError:
        print(f"Warning: Object semantic points file not found at {obj_sem_pts_file}.")
    except Exception as e:
        print(f"Error loading object semantic points {obj_sem_pts_file}: {e}")

    # Load ORIGINAL Bounding Boxes
    try:
        loaded_bbox = np.load(bbox_file)
        # Expecting original format: [x, y, center_z, w, l, h, yaw]
        if loaded_bbox.ndim == 2 and loaded_bbox.shape[1] == 7:
            gt_bboxes_data_original = loaded_bbox
            print(f"Loaded original bounding boxes: {gt_bboxes_data_original.shape}")
        elif loaded_bbox.shape != (0, 7) and loaded_bbox.size > 0: # Check size > 0 to avoid warning on empty files
             print(f"Warning: Bounding box file {bbox_file} has unexpected shape {loaded_bbox.shape}. Expected Nx7. Skipping boxes.")
        elif loaded_bbox.size == 0:
             print(f"Info: Bounding box file {bbox_file} is empty.")
        else: # Catch other unexpected cases
             print(f"Warning: Bounding box file {bbox_file} could not be loaded correctly. Shape: {loaded_bbox.shape}. Skipping boxes.")

    except FileNotFoundError:
        print(f"Warning: Bounding box file not found at {bbox_file}.")
    except Exception as e:
        print(f"Error loading bounding boxes {bbox_file}: {e}")

    # --- Exit if nothing significant was loaded ---
    # We need points to check for overlaps, even if boxes aren't loaded (though pointless then)
    if object_points.shape[0] == 0:
        print("Error: Failed to load points. Cannot perform overlap check or visualization.")
        return

    # --- Prepare Point Cloud (if loaded) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points)
    # Assign initial colors based on semantic labels
    colors = np.array([CLASS_COLOR_MAP.get(label[0], DEFAULT_COLOR) for label in object_labels], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Created point cloud geometry with {object_points.shape[0]} points.")

    # --- Prepare Bounding Boxes (Apply Transformations) ---
    bboxes_list = []
    # This list will store the OrientedBoundingBox objects from Open3D
    o3d_bboxes = []
    if gt_bboxes_data_original.shape[0] > 0:
        # Make a copy to modify
        gt_bboxes_data_modified = gt_bboxes_data_original.copy()

        # --- Apply the transformations ---
        h_original = gt_bboxes_data_original[:, 5].copy() # Original height
        gt_bboxes_data_modified[:, 6] += np.pi / 2.0
        print("Adjusted Yaw (+pi/2).")
        # gt_bboxes_data_modified[:, 2] -= h_original / 2.0 # Z to base (Optional)
        # print("Adjusted Z from center to base.")
        gt_bboxes_data_modified[:, 2] -= 0.1
        print("Applied Z offset (-0.1).")
        gt_bboxes_data_modified[:, 3:6] *= 1.1
        print("Applied scaling (1.1x) to box dimensions.")

        # --- Create Open3D Boxes using MODIFIED values ---
        for i in range(gt_bboxes_data_modified.shape[0]):
            center = gt_bboxes_data_modified[i, 0:3]
            extent = gt_bboxes_data_modified[i, 3:6] # w, l, h -> extent for O3D
            yaw = gt_bboxes_data_modified[i, 6]
            # Convert yaw to rotation matrix (yaw around Z axis)
            # Note: Open3D OrientedBoundingBox uses rotation matrix directly
            # R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw)) # Use this if center is relative to world Z
            # For yaw adjustment like this, it's simpler to think axis-angle
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

            # Create the Open3D bounding box object
            # Constructor: center, rotation matrix R, extent [width, length, height]
            bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
            bbox.color = (1, 0, 0) # Red color for boxes
            bboxes_list.append(bbox) # Add to list for visualization
            o3d_bboxes.append(bbox) # Keep separate list of O3D objects for point checking

        print(f"Processed {len(bboxes_list)} bounding boxes using specified transformations.")
    else:
         print("No bounding boxes loaded or processed.")

    # --- Find Points in Multiple Bounding Boxes ---
    overlapping_point_indices = []
    if object_points.shape[0] > 0 and len(o3d_bboxes) > 0:
        print("\n--- Checking for points in multiple boxes ---")
        # Create a count array, initialized to zeros
        point_counts = np.zeros(object_points.shape[0], dtype=int)

        # Iterate through each bounding box
        for i, bbox in enumerate(o3d_bboxes):
            # Find indices of points inside this bounding box
            # Note: Requires points as o3d.utility.Vector3dVector
            indices_in_box = bbox.get_point_indices_within_bounding_box(pcd.points)

            if indices_in_box:
                # Increment the count for points found in this box
                point_counts[indices_in_box] += 1
            # print(f"Box {i}: Found {len(indices_in_box)} points inside.") # Debug print

        # Find indices where the count is greater than 1
        overlapping_point_indices = np.where(point_counts > 1)[0]

        if len(overlapping_point_indices) > 0:
            print(f"Found {len(overlapping_point_indices)} points belonging to 2 or more bounding boxes.")
            # print("Indices of overlapping points:", overlapping_point_indices) # Optional: print indices

            if highlight_overlap:
                # Modify the colors of the overlapping points in the main point cloud
                overlap_colors = np.asarray(pcd.colors)
                overlap_colors[overlapping_point_indices] = OVERLAP_COLOR
                pcd.colors = o3d.utility.Vector3dVector(overlap_colors)
                print(f"Highlighted overlapping points in white.")

        else:
            print("No points found in multiple bounding boxes.")

    elif object_points.shape[0] == 0:
        print("Skipping overlap check: No points loaded.")
    else: # No boxes loaded
        print("Skipping overlap check: No bounding boxes loaded.")

    # --- Visualize ---
    geometries_to_draw = []
    if object_points.shape[0] > 0:
        geometries_to_draw.append(pcd) # Add point cloud (potentially with highlighted overlaps)
    if bboxes_list:
        geometries_to_draw.extend(bboxes_list)

    if not geometries_to_draw:
        print("No geometries available to draw.")
        return

    print("\n--- Opening Visualization Window ---")
    print("Points in original semantic colors (overlapping points may be white).")
    print("Boxes are shown in red.")
    print("Press 'Q' or close the window to exit.")
    o3d.visualization.draw_geometries(geometries_to_draw,
                                      window_name=f"Object Points & Transformed GT Boxes - {os.path.basename(data_directory)}",
                                      width=1280, height=720)
    print("--- Visualization window closed ---")


# ==================================================
# --- User Configuration ---
# ==================================================
# !!!! IMPORTANT !!!!
# SET THIS PATH: Point it to the specific directory containing the .npy files
#                (e.g., the '1bb41855cb724ae6980bababdbd1865e2' folder from your image)
# Example Linux/macOS: '/home/user/data/nuscenes_subset/sce...11/1bb41855cb724ae6980bababdbd1865e2'
# Example Windows: r'C:\data\nuscenes_subset\sce...11\1bb41855cb

DATA_DIR = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/gt/all_sensors_icp_boxes/scene-0044384af3d8494e913fb8b14915239e-11/1bb41855cb724ae6980bababbd1865e2' # <-- CHANGE THIS
# ==================================================


if __name__ == "__main__":
    # Basic check if the path might be the placeholder
    if 'path/to/your/data' in DATA_DIR or not os.path.isdir(DATA_DIR):
        print("="*60)
        print("ERROR: Please update the 'DATA_DIR' variable in the script!")
        print(f"       Set it to the directory containing the frame_0_...npy files.")
        print(f"       Current value: '{DATA_DIR}'")
        print("="*60)
    else:
        visualize_object_data(DATA_DIR, highlight_overlap=True)
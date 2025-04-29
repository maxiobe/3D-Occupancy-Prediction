import numpy as np
import open3d as o3d
import os

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

def visualize_object_data(data_directory):
    """
    Loads bounding boxes and object points, applies transformations to boxes,
    and visualizes them.

    Args:
        data_directory (str): The path to the directory containing the .npy files
                               (e.g., '.../1bb41855cb724ae6980bababdbd1865e2').
    """
    print(f"--- Loading data from: {data_directory} ---")

    # --- Define File Paths ---
    bbox_file = os.path.join(data_directory, 'frame_0_gt_bbox_3d.npy')
    obj_pts_file = os.path.join(data_directory, 'frame_0_temp_points.npy')

    # --- Load Data ---
    object_points = np.empty((0, 3))
    gt_bboxes_data_original = np.empty((0, 7)) # Store original loaded boxes here

    # Load Object Points
    try:
        loaded_obj = np.load(obj_pts_file)
        if loaded_obj.ndim == 2 and loaded_obj.shape[1] >= 3:
            object_points = loaded_obj[:, :3]
            print(f"Loaded object points: {object_points.shape}")
        else:
            print(f"Warning: Object points file {obj_pts_file} has unexpected shape {loaded_obj.shape}. Cannot visualize points.")
            object_points = np.empty((0, 3))
    except FileNotFoundError:
        print(f"Warning: Object points file not found at {obj_pts_file}.")
    except Exception as e:
        print(f"Error loading object points {obj_pts_file}: {e}")

    # Load ORIGINAL Bounding Boxes
    try:
        loaded_bbox = np.load(bbox_file)
        # Expecting original format: [x, y, center_z, w, l, h, yaw]
        if loaded_bbox.ndim == 2 and loaded_bbox.shape[1] == 7:
            gt_bboxes_data_original = loaded_bbox
            print(f"Loaded original bounding boxes: {gt_bboxes_data_original.shape}")
        elif loaded_bbox.shape != (0, 7):
            print(f"Warning: Bounding box file {bbox_file} has unexpected shape {loaded_bbox.shape}. Expected Nx7. Skipping boxes.")
    except FileNotFoundError:
        print(f"Warning: Bounding box file not found at {bbox_file}.")
    except Exception as e:
        print(f"Error loading bounding boxes {bbox_file}: {e}")

    # --- Exit if nothing significant was loaded ---
    if object_points.shape[0] == 0 and gt_bboxes_data_original.shape[0] == 0:
        print("Error: Failed to load points and bounding boxes. Nothing to visualize.")
        return

    # --- Prepare Point Cloud (if loaded) ---
    pcd = o3d.geometry.PointCloud()
    if object_points.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(object_points)
        colors = np.tile([1.0, 0.0, 0.0], (object_points.shape[0], 1))  # Red
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"Created point cloud geometry with {object_points.shape[0]} points.")
    else:
        print("No object points loaded.")


    # --- Prepare Bounding Boxes (Apply Transformations) ---
    bboxes_list = []
    if gt_bboxes_data_original.shape[0] > 0:
        # Make a copy to modify
        gt_bboxes_data_modified = gt_bboxes_data_original.copy()

        # --- Apply the transformations ---
        # Store original height needed for z-adjustment BEFORE scaling height
        h_original = gt_bboxes_data_original[:, 5].copy() # Original height from column index 5

        gt_bboxes_data_modified[:, 6] += np.pi / 2.0
        print("Adjusted Yaw (+pi/2).")

        #gt_bboxes_data_modified[:, 2] -= h_original / 2.0
        #print("Adjusted Z from center to base.")

        gt_bboxes_data_modified[:, 2] -= 0.1
        print("Applied Z offset (-0.1).")

        gt_bboxes_data_modified[:, 3:6] *= 1.1
        print("Applied scaling (1.1x) to box dimensions.")


        # --- Create Open3D Boxes using MODIFIED values ---
        for i in range(gt_bboxes_data_modified.shape[0]):
            # Extract modified values
            center = gt_bboxes_data_modified[i, 0:3]
            w, l, h = gt_bboxes_data_modified[i, 3:6]
            yaw = gt_bboxes_data_modified[i, 6]

            # Convert yaw to rotation matrix (yaw around Z axis)
            rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

            bbox = o3d.geometry.OrientedBoundingBox(center, rot_mat, [w, l, h])
            bbox.color = (1, 0, 0)
            bboxes_list.append(bbox)
        print(f"Processed {len(bboxes_list)} bounding boxes using specified transformations.")
    else:
         print("No bounding boxes loaded.")


    # --- Visualize ---
    geometries_to_draw = []
    if object_points.shape[0] > 0:
        geometries_to_draw.append(pcd)
    if bboxes_list:
        geometries_to_draw.extend(bboxes_list)

    if not geometries_to_draw:
        print("No geometries available to draw.")
        return

    print("\n--- Opening Visualization Window ---")
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
        visualize_object_data(DATA_DIR)
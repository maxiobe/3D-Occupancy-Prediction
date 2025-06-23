import numpy as np
import torch
from pyquaternion import Quaternion
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Essential imports from the TruckScenes library ---
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box

# --- Configuration ---
# ---> SET THESE VALUES
TRUCKSCENES_DATA_ROOT = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'  # Path to your TruckScenes data
TRUCKSCENES_VERSION = 'v1.0-trainval'  # Dataset version

# Check if a CUDA-enabled GPU is available
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. This script will run on the CPU, which will be very slow.")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda:0")
    print(f"Using CUDA device: {torch.cuda.get_device_name(DEVICE)}")


# --- Helper Functions ---

def visualize_all_boxes_in_frame(list_of_corners: List[np.ndarray], title=""):
    """
    Creates an interactive 3D plot of all bounding boxes in a frame.
    `list_of_corners` is a list where each item is a numpy array of shape (8, 3).
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Added black

    all_points = []

    # Loop through each set of corners and plot it with a cycling color
    for i, corners in enumerate(list_of_corners):
        color = colors[i % len(colors)]
        for j, k in edges:
            ax.plot(corners[[j, k], 0], corners[[j, k], 1], corners[[j, k], 2], color=color)
        all_points.append(corners)

    # Set plot limits for a correct aspect ratio
    if not all_points:
        return  # Don't plot if there are no boxes

    all_points = np.vstack(all_points)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    # Ensure the plot has a cubic aspect ratio
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    mid_x, mid_y, mid_z = (x_max + x_min) * 0.5, (y_max + y_min) * 0.5, (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X axis');
    ax.set_ylabel('Y axis');
    ax.set_zlabel('Z axis')
    ax.set_title(title)

    print(f"Displaying 3D plot for: {title}. Close the plot window to continue to the next frame...")
    plt.show()


def get_boxes_for_sample(trucksc: TruckScenes, sample: Dict[str, Any]) -> List[Box]:
    """
    Returns the bounding boxes for a given sample, transformed into the
    ego vehicle's coordinate frame at the sample's timestamp.
    """
    annotation_tokens = sample['anns']
    if not annotation_tokens: return []
    boxes = [trucksc.get_box(token) for token in annotation_tokens]
    ego_pose_record = trucksc.getclosest('ego_pose', sample['timestamp'])
    ego_translation = np.array(ego_pose_record['translation'])
    ego_rotation_inv = Quaternion(ego_pose_record['rotation']).inverse
    for box in boxes:
        box.translate(-ego_translation)
        box.rotate(ego_rotation_inv)
    return boxes


def convert_boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts parameterized 3D boxes to their 8 corner coordinates.
    Assumes input `boxes` has dimensions as (length, width, height).
    """
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("Input tensor must be of shape (N, 7).")
    corners_norm = torch.tensor([
        [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5], [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
    ], dtype=torch.float32, device=boxes.device)
    # The first dim of corners_norm (X) is scaled by the first dim of boxes (length)
    # The second dim of corners_norm (Y) is scaled by the second dim of boxes (width)

    centers = boxes[:, 0:3]
    dims_wlh = boxes[:, 3:6]
    dims_lwh = dims_wlh[:, [1, 0, 2]]
    yaws = boxes[:, 6]

    cos_yaws, sin_yaws = torch.cos(yaws), torch.sin(yaws)
    zeros, ones = torch.zeros_like(cos_yaws), torch.ones_like(cos_yaws)
    rot_matrices = torch.stack([
        cos_yaws, -sin_yaws, zeros, sin_yaws, cos_yaws, zeros, zeros, zeros, ones
    ], dim=1).reshape(-1, 3, 3)
    scaled_corners = corners_norm.unsqueeze(0) * dims_lwh.unsqueeze(1)
    rotated_corners = torch.bmm(scaled_corners, rot_matrices.transpose(1, 2))
    return rotated_corners + centers.unsqueeze(1)


def main():
    """
    Main function to visualize all bounding boxes for each frame in a scene.
    """
    print(f"Initializing TruckScenes dataset from: {TRUCKSCENES_DATA_ROOT}")
    trucksc = TruckScenes(version=TRUCKSCENES_VERSION, dataroot=TRUCKSCENES_DATA_ROOT, verbose=True)

    print("\nStarting analysis. A 3D plot will appear for each frame.")

    total_scenes = len(trucksc.scene)
    for scene_idx, scene_record in enumerate(trucksc.scene):
        if scene_idx > 10:  # Limit number of scenes for testing
            break
        scene_name = scene_record['name']
        print(f"\nProcessing Scene {scene_idx + 1}/{total_scenes}: {scene_name}")

        frame_counter = 0
        sample_token = scene_record['first_sample_token']

        while sample_token:
            frame_counter += 1
            sample_record = trucksc.get('sample', sample_token)
            boxes = get_boxes_for_sample(trucksc, sample_record)

            locs = np.array([b.center for b in boxes]).reshape(-1,
                                                               3)  # gets center coordinates (x,y,z) of each bb
            dims = np.array([b.wlh for b in boxes]).reshape(-1,
                                                            3)  # extract dimension width, length, height of each bb
            rots = np.array([b.orientation.yaw_pitch_roll[0]  # extract rotations (yaw angles)
                             for b in boxes]).reshape(-1, 1)

            gt_bbox_3d_unmodified = np.concatenate([locs, dims, rots], axis=1).astype(
                np.float32)

            gt_bbox_3d = gt_bbox_3d_unmodified.copy()

            #gt_bbox_3d[:, 6] += np.pi / 2.  # adjust yaw angles by 90 degrees
            gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
            gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
            gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.0

            if not boxes:
                print(f"-- Frame {frame_counter}: No boxes found.")
                sample_token = sample_record['next']
                continue

            print(f"-- Frame {frame_counter}: Found {len(boxes)} boxes. Visualizing...")
            all_box_params = np.array([
                [b.center[0], b.center[1], b.center[2], b.wlh[0], b.wlh[1], b.wlh[2], b.orientation.yaw_pitch_roll[0]]
                for b in boxes
            ])

            all_box_params = gt_bbox_3d

            # 2. Convert all box parameters to corner coordinates
            all_box_params_t = torch.from_numpy(all_box_params).float().to(DEVICE)
            all_corners_t = convert_boxes_to_corners(all_box_params_t)  # Tensor of shape (num_boxes, 8, 3)

            # 3. Convert tensor to a list of numpy arrays for plotting
            list_of_corners_np = [c.cpu().numpy() for c in all_corners_t]

            # 4. Call the visualization function with all the corners
            visualize_all_boxes_in_frame(
                list_of_corners_np,
                title=f"Scene {scene_idx + 1} ('{scene_name}'), Frame {frame_counter}"
            )

            sample_token = sample_record['next']

    print("\n--- Visualization Finished ---")


if __name__ == '__main__':
    main()
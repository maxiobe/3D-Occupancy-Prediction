import numpy as np
import torch
from pyquaternion import Quaternion
from typing import Dict, Any, List
from collections import defaultdict

# --- Essential imports from the TruckScenes and PyTorch3D libraries ---
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import transform_matrix
from pytorch3d.ops.iou_box3d import box3d_overlap

# --- Configuration ---
# ---> SET THESE VALUES
TRUCKSCENES_DATA_ROOT = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval' # Path to your TruckScenes data
TRUCKSCENES_VERSION = 'v1.0-trainval'         # Dataset version
IOU_THRESHOLD = 0.01                           # Minimum IoU to be considered a valid overlap for statistics

# Check if a CUDA-enabled GPU is available
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. This script will run on the CPU, which will be very slow.")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda:0")
    print(f"Using CUDA device: {torch.cuda.get_device_name(DEVICE)}")


# --- Helper Functions (Unchanged) ---

def get_boxes_for_sample(trucksc: TruckScenes, sample: Dict[str, Any]) -> List[Box]:
    """
    Returns the bounding boxes for a given sample, transformed into the
    ego vehicle's coordinate frame at the sample's timestamp.
    """
    annotation_tokens = sample['anns']
    if not annotation_tokens:
        return []

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
    Converts parameterized 3D boxes to their 8 corner coordinates for PyTorch3D.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("Input tensor must be of shape (N, 7).")

    corners_norm = torch.tensor([
        [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5], [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
    ], dtype=torch.float32, device=boxes.device)

    centers = boxes[:, 0:3]
    dims_wlh = boxes[:, 3:6]
    yaws = boxes[:, 6]

    cos_yaws, sin_yaws = torch.cos(yaws), torch.sin(yaws)
    zeros, ones = torch.zeros_like(cos_yaws), torch.ones_like(cos_yaws)
    rot_matrices = torch.stack([
        cos_yaws, -sin_yaws, zeros,
        sin_yaws, cos_yaws, zeros,
        zeros, zeros, ones
    ], dim=1).reshape(-1, 3, 3)

    scaled_corners = corners_norm.unsqueeze(0) * dims_wlh.unsqueeze(1)
    rotated_corners = torch.bmm(scaled_corners, rot_matrices.transpose(1, 2))
    final_corners = rotated_corners + centers.unsqueeze(1)

    return final_corners

def calculate_3d_iou_pytorch3d(boxes1_params: np.ndarray, boxes2_params: np.ndarray) -> np.ndarray:
    """
    Calculates the exact 3D IoU for two sets of boxes using PyTorch3D.
    """
    if boxes1_params.shape[0] == 0 or boxes2_params.shape[0] == 0:
        return np.empty((boxes1_params.shape[0], boxes2_params.shape[0]))

    b1_t = torch.from_numpy(boxes1_params).float().to(DEVICE)
    b2_t = torch.from_numpy(boxes2_params).float().to(DEVICE)

    corners1 = convert_boxes_to_corners(b1_t)
    corners2 = convert_boxes_to_corners(b2_t)

    _, iou_matrix_gpu = box3d_overlap(corners1, corners2)

    return iou_matrix_gpu.cpu().numpy()

def main():
    """
    Main function to analyze object overlaps across all scenes and calculate statistics.
    """
    print(f"Initializing TruckScenes dataset from: {TRUCKSCENES_DATA_ROOT}")
    trucksc = TruckScenes(version=TRUCKSCENES_VERSION, dataroot=TRUCKSCENES_DATA_ROOT, verbose=True)

    # Use a defaultdict to easily collect all IoU values for each class pair
    overlaps_data = defaultdict(list)

    print("\nStarting analysis for all scenes... This may take a while.")

    # Loop over every scene in the dataset
    total_scenes = len(trucksc.scene)
    for scene_idx, scene_record in enumerate(trucksc.scene):
        scene_name = scene_record['name']
        print(f"Processing Scene {scene_idx + 1}/{total_scenes}: {scene_name}")

        sample_token = scene_record['first_sample_token']
        while sample_token:
            sample_record = trucksc.get('sample', sample_token)
            boxes = get_boxes_for_sample(trucksc, sample_record)

            if len(boxes) < 2:
                sample_token = sample_record['next']
                continue

            all_box_params = np.array([
                [b.center[0], b.center[1], b.center[2], b.wlh[0], b.wlh[1], b.wlh[2], b.orientation.yaw_pitch_roll[0]]
                for b in boxes
            ])

            iou_matrix = calculate_3d_iou_pytorch3d(all_box_params, all_box_params)

            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = iou_matrix[i, j]
                    if iou > IOU_THRESHOLD:
                        category_name1 = trucksc.get('sample_annotation', boxes[i].token)['category_name']
                        category_name2 = trucksc.get('sample_annotation', boxes[j].token)['category_name']

                        # Create a canonical (sorted) key for the class pair
                        class_pair_key = tuple(sorted((category_name1, category_name2)))

                        # Store the IoU value
                        overlaps_data[class_pair_key].append(iou)

            sample_token = sample_record['next']

    print("\n--- All scenes processed. Calculating statistics... ---")

    # --- Calculate and Display Final Statistics ---
    if not overlaps_data:
        print("No significant overlaps were found in the entire dataset.")
        return

    results = []
    for class_pair, iou_list in overlaps_data.items():
        iou_array = np.array(iou_list)
        results.append({
            'pair': class_pair,
            'count': len(iou_array),
            'avg_iou': np.mean(iou_array),
            'median_iou': np.median(iou_array),
            'min_iou': np.min(iou_array),
            'max_iou': np.max(iou_array)
        })

    # Sort results by the number of occurrences (most frequent first)
    results.sort(key=lambda x: x['count'], reverse=True)

    # Print the header for our summary table
    print("\n" + "="*85)
    print(" " * 25 + "Overlap Statistics Across Dataset")
    print("="*85)
    header = f"{'Object Pair':<40} | {'Count':>8} | {'Avg IoU':>10} | {'Median IoU':>12} | {'Max IoU':>8}"
    print(header)
    print("-"*len(header))

    # Print each result row
    for res in results:
        pair_str = f"{res['pair'][0]} & {res['pair'][1]}"
        print(f"{pair_str:<40} | {res['count']:>8} | {res['avg_iou']:>10.4f} | {res['median_iou']:>12.4f} | {res['max_iou']:>8.4f}")
    print("="*85)

if __name__ == '__main__':
    main()
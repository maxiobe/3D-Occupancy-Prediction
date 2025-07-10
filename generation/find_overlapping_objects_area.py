import numpy as np
import torch
from pyquaternion import Quaternion
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import textwrap

# --- Essential imports from the TruckScenes and PyTorch3D libraries ---
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import transform_matrix
from pytorch3d.ops.iou_box3d import box3d_overlap
from copy import deepcopy

# --- Configuration ---
# ---> SET THESE VALUES
TRUCKSCENES_DATA_ROOT = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'  # Path to your TruckScenes data
TRUCKSCENES_VERSION = 'v1.0-trainval'  # Dataset version
IOU_THRESHOLD = 0.01  # Minimum IoU to be considered a valid overlap for statistics

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
    Assumes input `boxes` has dimensions as (length, width, height).
    """
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("Input tensor must be of shape (N, 7).")

    corners_norm = torch.tensor([
        [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5], [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
    ], dtype=torch.float32, device=boxes.device)

    """corners_norm = torch.tensor([
        [-0.5, +0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, +0.5, +0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5],
    ], dtype=torch.float32, device=boxes.device)"""

    centers = boxes[:, 0:3]
    dims_wlh = boxes[:, 3:6]
    dims_lwh = dims_wlh[:, [1, 0, 2]]
    yaws = boxes[:, 6]

    cos_yaws, sin_yaws = torch.cos(yaws), torch.sin(yaws)
    zeros, ones = torch.zeros_like(cos_yaws), torch.ones_like(cos_yaws)
    rot_matrices = torch.stack([
        cos_yaws, -sin_yaws, zeros,
        sin_yaws, cos_yaws, zeros,
        zeros, zeros, ones
    ], dim=1).reshape(-1, 3, 3)

    scaled_corners = corners_norm.unsqueeze(0) * dims_lwh.unsqueeze(1)
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

    intersection_volume_matrix_gpu, iou_matrix_gpu = box3d_overlap(corners1, corners2)

    return iou_matrix_gpu.cpu().numpy(), intersection_volume_matrix_gpu.cpu().numpy()

def main():
    """
    Main function to analyze object overlaps across all scenes and calculate statistics.
    """
    print(f"Initializing TruckScenes dataset from: {TRUCKSCENES_DATA_ROOT}")
    trucksc = TruckScenes(version=TRUCKSCENES_VERSION, dataroot=TRUCKSCENES_DATA_ROOT, verbose=True)

    overlaps_data = defaultdict(lambda: defaultdict(list))

    print("\nStarting analysis for all scenes... This may take a while.")

    total_scenes = len(trucksc.scene)
    for scene_idx, scene_record in enumerate(trucksc.scene):
        scene_name = scene_record['name']
        print(f"Processing Scene {scene_idx + 1}/{total_scenes}: {scene_name}")

        frame_counter = 0
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = trucksc.get('sample', sample_token)
            boxes = get_boxes_for_sample(trucksc, sample_record)

            if len(boxes) < 2:
                sample_token = sample_record['next']
                continue

            locs = np.array([b.center for b in boxes]).reshape(-1,
                                                                   3)  # gets center coordinates (x,y,z) of each bb
            dims = np.array([b.wlh for b in boxes]).reshape(-1,
                                                                3)  # extract dimension width, length, height of each bb
            rots = np.array([b.orientation.yaw_pitch_roll[0]  # extract rotations (yaw angles)
                             for b in boxes]).reshape(-1, 1)

            gt_bbox_3d_unmodified = np.concatenate([locs, dims, rots], axis=1).astype(
                np.float32)

            gt_bbox_3d = gt_bbox_3d_unmodified.copy()

            # gt_bbox_3d[:, 6] += np.pi / 2.  # not needed as we changed wlh to lwh
            # gt_bbox_3d[:, 2] -= dims[:, 2] / 2. #--> not needed as we want the center
            # gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # not needed as we do not want to shift and it has no effect on everything
            gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.0

            """all_box_params = np.array([
                [b.center[0], b.center[1], b.center[2], b.wlh[0], b.wlh[1], b.wlh[2], b.orientation.yaw_pitch_roll[0]]
                for b in boxes
            ])"""
            all_box_params = gt_bbox_3d

            iou_matrix, intersection_volume_matrix = calculate_3d_iou_pytorch3d(all_box_params, all_box_params)

            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = iou_matrix[i, j]
                    if iou > IOU_THRESHOLD:
                        intersection_volume = intersection_volume_matrix[i, j]
                        # Calculate volumes of the two boxes
                        volume1 = dims[i, 0] * dims[i, 1] * dims[i, 2]
                        volume2 = dims[j, 0] * dims[j, 1] * dims[j, 2]
                        smallest_volume = min(volume1, volume2)

                        # Calculate the new metric: overlap_volume / smallest_box_volume
                        # Avoid division by zero, though volume should always be positive
                        overlap_ratio = intersection_volume / smallest_volume if smallest_volume > 1e-6 else 0

                        category_name1 = trucksc.get('sample_annotation', boxes[i].token)['category_name']
                        category_name2 = trucksc.get('sample_annotation', boxes[j].token)['category_name']

                        if overlap_ratio > 1:
                            print(f"Overlap ratio: {overlap_ratio}, intersection volume: {intersection_volume}, volume1: {volume1}, volume2: {volume2}, category_name1: {category_name1}, category_name2: {category_name2}")
                        class_pair_key = tuple(sorted((category_name1, category_name2)))
                        # Store all three metrics
                        overlaps_data[class_pair_key][scene_idx].append((iou, intersection_volume, overlap_ratio))

            sample_token = sample_record['next']
            frame_counter += 1

    print("\n--- All scenes processed. Calculating and presenting results... ---")

    if not overlaps_data:
        print("No significant overlaps were found in the entire dataset.")
        return

    # --- Process and store results ---
    results = []
    for class_pair, scene_data_lists in overlaps_data.items():
        all_metrics_for_pair = [item for scene_list in scene_data_lists.values() for item in scene_list]

        # Unzip the metrics into separate lists
        all_ious, all_volumes, all_ratios = zip(*all_metrics_for_pair)

        overall_iou_array = np.array(all_ious)
        overall_volume_array = np.array(all_volumes)
        overall_ratio_array = np.array(all_ratios)

        overall_stats = {
            'scene_count': len(scene_data_lists),
            'total_overlaps': len(overall_iou_array),
            'avg_iou': np.mean(overall_iou_array),
            'median_iou': np.median(overall_iou_array),
            'min_iou': np.min(overall_iou_array),
            'max_iou': np.max(overall_iou_array),
            'avg_volume': np.mean(overall_volume_array),
            'median_volume': np.median(overall_volume_array),
            'min_volume': np.min(overall_volume_array),
            'max_volume': np.max(overall_volume_array),
            'avg_ratio': np.mean(overall_ratio_array),
            'median_ratio': np.median(overall_ratio_array),
            'min_ratio': np.min(overall_ratio_array),
            'max_ratio': np.max(overall_ratio_array),
        }

        results.append({
            'pair': class_pair,
            'overall_stats': overall_stats,
        })

    results.sort(key=lambda x: x['overall_stats']['total_overlaps'], reverse=True)

    # --- Presentation ---

    # --- IoU Statistics Table ---
    print("\n" + "=" * 120)
    print(" " * 45 + "Overall IoU Statistics")
    print("=" * 120)
    header_iou = (f"{'Object Pair':<60} | {'Total Overlaps':>14} | {'Unique Scenes':>13} | "
                  f"{'Overall Avg IoU':>15} | {'Overall Median':>14}")
    print(header_iou)
    print("-" * len(header_iou))
    for res in results:
        pair_str = f"{res['pair'][0]} & {res['pair'][1]}"
        stats = res['overall_stats']
        print(f"{pair_str:<60} | {stats['total_overlaps']:>14} | {stats['scene_count']:>13} | "
              f"{stats['avg_iou']:>15.4f} | {stats['median_iou']:>14.4f}")
    print("=" * 120)

    # --- NEW: Overlapping Volume Statistics Table ---
    print("\n" + "=" * 130)
    print(" " * 45 + "Overlapping Volume Statistics (in cubic meters)")
    print("=" * 130)
    header_vol = (f"{'Object Pair':<60} | {'Avg Volume':>15} | {'Median Volume':>15} | "
                  f"{'Min Volume':>15} | {'Max Volume':>15}")
    print(header_vol)
    print("-" * len(header_vol))
    for res in results:
        pair_str = f"{res['pair'][0]} & {res['pair'][1]}"
        stats = res['overall_stats']
        print(f"{pair_str:<60} | {stats['avg_volume']:>15.4f} | {stats['median_volume']:>15.4f} | "
              f"{stats['min_volume']:>15.4f} | {stats['max_volume']:>15.4f}")
    print("=" * 130)

    # --- NEW: Volume Ratio Statistics Table ---
    print("\n" + "=" * 145)
    print(" " * 35 + "Overlap Volume / Smallest BBox Volume Ratio Statistics")
    print("=" * 145)
    header_ratio = (f"{'Object Pair':<60} | {'Avg Ratio':>15} | {'Median Ratio':>15} | "
                    f"{'Min Ratio':>15} | {'Max Ratio':>15}")
    print(header_ratio)
    print("-" * len(header_ratio))
    for res in results:
        pair_str = f"{res['pair'][0]} & {res['pair'][1]}"
        stats = res['overall_stats']
        # Note: A ratio > 1.0 is possible if the intersection volume is larger
        # than the smaller of the two boxes' volumes (which can happen with rotations)
        # but is capped by the union for IoU.
        print(f"{pair_str:<60} | {stats['avg_ratio']:>15.4f} | {stats['median_ratio']:>15.4f} | "
              f"{stats['min_ratio']:>15.4f} | {stats['max_ratio']:>15.4f}")
    print("=" * 145)


if __name__ == '__main__':
    main()
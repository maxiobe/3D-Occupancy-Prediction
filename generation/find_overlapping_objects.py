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

    _, iou_matrix_gpu = box3d_overlap(corners1, corners2)

    return iou_matrix_gpu.cpu().numpy()

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

            if len(boxes) < 2:
                sample_token = sample_record['next']
                continue

            """all_box_params = np.array([
                [b.center[0], b.center[1], b.center[2], b.wlh[0], b.wlh[1], b.wlh[2], b.orientation.yaw_pitch_roll[0]]
                for b in boxes
            ])"""
            all_box_params = gt_bbox_3d

            iou_matrix = calculate_3d_iou_pytorch3d(all_box_params, all_box_params)

            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = iou_matrix[i, j]
                    if iou > IOU_THRESHOLD:
                        category_name1 = trucksc.get('sample_annotation', boxes[i].token)['category_name']
                        category_name2 = trucksc.get('sample_annotation', boxes[j].token)['category_name']
                        class_pair_key = tuple(sorted((category_name1, category_name2)))
                        overlaps_data[class_pair_key][scene_idx].append((iou, frame_counter))

            sample_token = sample_record['next']
            frame_counter += 1

    print("\n--- All scenes processed. Calculating and presenting results... ---")

    if not overlaps_data:
        print("No significant overlaps were found in the entire dataset.")
        return

    results = []
    for class_pair, scene_iou_lists in overlaps_data.items():
        all_ious_for_pair = []
        per_scene_stats_list = []
        min_ious_per_scene = []
        ### --- NEW: List to store the maximum IoU from each scene --- ###
        max_ious_per_scene = []


        for scene_idx, iou_frame_pairs in sorted(scene_iou_lists.items()):
            iou_values = [pair[0] for pair in iou_frame_pairs]
            iou_array = np.array(iou_values)
            all_ious_for_pair.extend(iou_values)

            min_iou_in_scene = np.min(iou_array)
            max_iou_in_scene = np.max(iou_array) # Get max for this scene
            min_ious_per_scene.append(min_iou_in_scene)
            max_ious_per_scene.append(max_iou_in_scene) # Add this scene's max IoU to our new list


            per_scene_stats_list.append({
                'scene_idx': scene_idx,
                'count': len(iou_array),
                'avg_iou': np.mean(iou_array),
                'median_iou': np.median(iou_array),
                'min_iou': min_iou_in_scene,
                'max_iou': max_iou_in_scene,
            })

        overall_iou_array = np.array(all_ious_for_pair)
        overall_stats = {
            'scene_count': len(scene_iou_lists),
            'total_overlaps': len(overall_iou_array),
            'avg_iou': np.mean(overall_iou_array),
            'median_iou': np.median(overall_iou_array),
            'min_iou': np.min(overall_iou_array),
            'max_iou': np.max(overall_iou_array),
            'avg_min_iou': np.mean(np.array(min_ious_per_scene)) if min_ious_per_scene else 0,
            ### --- NEW: Calculate the average of the maximums --- ###
            'avg_max_iou': np.mean(np.array(max_ious_per_scene)) if max_ious_per_scene else 0,
        }

        results.append({
            'pair': class_pair,
            'overall_stats': overall_stats,
            'per_scene_stats': per_scene_stats_list
        })

    results.sort(key=lambda x: x['overall_stats']['total_overlaps'], reverse=True)

    ### --- MODIFIED: Rebuilt this entire table to match your request --- ###
    print("\n" + "=" * 200)
    print(" " * 100 + "Overall Object Pair Overlap Statistics")
    print("=" * 200)
    header = (f"{'Object Pair':<60} | {'Total Overlaps':>14} | {'Unique Scenes':>13} | "
              f"{'Overall Avg IoU':>15} | {'Overall Median':>14} | {'Overall Min':>11} | {'Overall Max':>11} | "
              f"{'Avg Min IoU':>11} | {'Avg Max IoU':>11}")
    print(header)
    print("-" * len(header))
    for res in results:
        pair_str = f"{res['pair'][0]} & {res['pair'][1]}"
        stats = res['overall_stats']
        print(f"{pair_str:<60} | {stats['total_overlaps']:>14} | {stats['scene_count']:>13} | "
              f"{stats['avg_iou']:>15.4f} | {stats['median_iou']:>14.4f} | {stats['min_iou']:>11.4f} | {stats['max_iou']:>11.4f} | "
              f"{stats['avg_min_iou']:>11.4f} | {stats['avg_max_iou']:>11.4f}")
    print("=" * 200)

    # --- Detailed Scene-by-Scene Information Table (unchanged) ---
    print("\n" + "=" * 90)
    print(" " * 27 + "Detailed Per-Scene Overlap Statistics")
    print("=" * 90)

    if not results:
        print("No overlaps to report.")
    else:
        for res in results:
            pair_str = f"-> {res['pair'][0]} & {res['pair'][1]}"
            print(f"\n{pair_str} (Overall Min: {res['overall_stats']['min_iou']:.4f}, "
                  f"Median: {res['overall_stats']['median_iou']:.4f}, "
                  f"Avg: {res['overall_stats']['avg_iou']:.4f})")
            print("-" * 90)

            scene_header = (f"{'Scene Idx':>9} | {'Count':>7} | {'Avg IoU':>9} | {'Median IoU':>11} | "
                            f"{'Min IoU':>9} | {'Max IoU':>9}")
            print(scene_header)
            print("-" * len(scene_header))

            for scene_stats in res['per_scene_stats']:
                print(f"{scene_stats['scene_idx']:>9} | {scene_stats['count']:>7} | {scene_stats['avg_iou']:>9.4f} | "
                      f"{scene_stats['median_iou']:>11.4f} | {scene_stats['min_iou']:>9.4f} | {scene_stats['max_iou']:>9.4f}")

    print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
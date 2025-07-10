import numpy as np
import torch
from pyquaternion import Quaternion
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import textwrap

from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import transform_matrix
from pytorch3d.ops.iou_box3d import box3d_overlap

from copy import deepcopy

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


# --- Helper Functions ---

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


def convert_boxes_to_corners(centers: torch.Tensor, dims: torch.Tensor, rot_matrices: torch.Tensor) -> torch.Tensor:
    """
    Corrected function for TruckScenes → PyTorch3D box convention.
    dims: (width, length, height)
    """
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3).")

    w, l, h = dims[:, 0], dims[:, 1], dims[:, 2]

    # Re-map axes explicitly to match PyTorch3D:
    x_corners = (l / 2).unsqueeze(1) * torch.tensor(
        [-1, 1, 1, -1, -1, 1, 1, -1], device=DEVICE, dtype=centers.dtype
    )

    y_corners = (h / 2).unsqueeze(1) * torch.tensor(
        [-1, -1, 1, 1, -1, -1, 1, 1], device=DEVICE, dtype=centers.dtype
    )

    z_corners = (w / 2).unsqueeze(1) * torch.tensor(
        [-1, -1, -1, -1, 1, 1, 1, 1], device=DEVICE, dtype=centers.dtype
    )

    corners_local = torch.stack([x_corners, y_corners, z_corners], dim=1)

    # Rotate & translate correctly
    rotated_corners = torch.bmm(rot_matrices, corners_local)
    final_corners = rotated_corners.transpose(1, 2) + centers.unsqueeze(1)

    return final_corners



def main():
    """
    Main function to analyze object overlaps across all scenes and calculate statistics.
    """
    print(f"Initializing TruckScenes dataset from: {TRUCKSCENES_DATA_ROOT}")
    trucksc = TruckScenes(version=TRUCKSCENES_VERSION, dataroot=TRUCKSCENES_DATA_ROOT, verbose=True)

    # Data structure to hold: {class_pair: {scene_idx: [(iou, overlap_volume, overlap_ratio), ...]}}
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

            # Extract full 3D orientation and other parameters.
            centers_np = np.array([b.center for b in boxes])
            dims_np = np.array([b.wlh for b in boxes])  # This is w, l, h
            rot_matrices_np = np.array([b.orientation.rotation_matrix for b in boxes])

            # Convert to tensors for GPU processing
            centers_t = torch.from_numpy(centers_np).float().to(DEVICE)
            dims_t = torch.from_numpy(dims_np).float().to(DEVICE)
            rot_matrices_t = torch.from_numpy(rot_matrices_np).float().to(DEVICE)

            # Get corners using the corrected function for accurate 3D representation
            corners = convert_boxes_to_corners(centers_t, dims_t, rot_matrices_t)

            # Calculate intersection and IoU using the accurate corners
            intersection_volume_matrix_gpu, iou_matrix_gpu = box3d_overlap(corners, corners)
            iou_matrix = iou_matrix_gpu.cpu().numpy()
            intersection_volume_matrix = intersection_volume_matrix_gpu.cpu().numpy()

            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = iou_matrix[i, j]
                    if iou > IOU_THRESHOLD:
                        intersection_volume = intersection_volume_matrix[i, j]

                        # Calculate volumes of the two boxes using the numpy array
                        volume1 = dims_np[i, 0] * dims_np[i, 1] * dims_np[i, 2]  # w * l * h
                        volume2 = dims_np[j, 0] * dims_np[j, 1] * dims_np[j, 2]  # w * l * h
                        smallest_volume = min(volume1, volume2)

                        # The ratio should now be physically correct.
                        overlap_ratio = intersection_volume / smallest_volume if smallest_volume > 1e-6 else 0

                        category_name1 = trucksc.get('sample_annotation', boxes[i].token)['category_name']
                        category_name2 = trucksc.get('sample_annotation', boxes[j].token)['category_name']
                        class_pair_key = tuple(sorted((category_name1, category_name2)))

                        if overlap_ratio > 0.99:
                            print(
                                f"High overlap in frame {frame_counter}: Overlap ratio {overlap_ratio}, category 1 {category_name1} vs 2 {category_name2}, intersection volume {intersection_volume}, volume 1 {volume1}, volume 2 {volume2}")

                        # Store all three metrics
                        overlaps_data[class_pair_key][scene_idx].append((iou, intersection_volume, overlap_ratio))

            sample_token = sample_record['next']
            frame_counter += 1

    print("\n--- All scenes processed. Calculating and presenting results... ---")

    if not overlaps_data:
        print("No significant overlaps were found in the entire dataset.")
        return

    # --- Process and store results (logic unchanged) ---
    results = []
    for class_pair, scene_data_lists in overlaps_data.items():
        all_metrics_for_pair = [item for scene_list in scene_data_lists.values() for item in scene_list]
        if not all_metrics_for_pair: continue

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

    # --- Presentation (unchanged) ---
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
        print(f"{pair_str:<60} | {stats['avg_ratio']:>15.4f} | {stats['median_ratio']:>15.4f} | "
              f"{stats['min_ratio']:>15.4f} | {stats['max_ratio']:>15.4f}")
    print("=" * 145)


if __name__ == '__main__':
    main()


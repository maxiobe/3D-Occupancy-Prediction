# File: preprocess_and_interpolate.py
import os
import sys
import yaml
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation, Slerp

# --- Assuming these are in the same directory or your python path ---
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box


# Helper functions from your original script
def load_lidar_entries(trucksc, sample, lidar_sensors):
    """Gathers all lidar data entries for a scene starting from the first sample."""
    entries = []
    for sensor in lidar_sensors:
        token = sample['data'][sensor]
        while token:
            sd = trucksc.get('sample_data', token)
            entries.append({
                'sensor': sensor,
                'timestamp': sd['timestamp'],
                'token': token,
                'keyframe': sd['is_key_frame']
            })
            token = sd['next']
    entries.sort(key=lambda x: x['timestamp'])
    return entries


def group_entries(entries, lidar_sensors, max_time_diff):
    """Groups lidar entries that were captured at approximately the same time."""
    used_tokens = set()
    groups = []
    for i, ref_entry in enumerate(entries):
        if ref_entry['token'] in used_tokens:
            continue
        ref_keyframe_flag = ref_entry['keyframe']
        group = {ref_entry['sensor']: ref_entry}
        group_tokens = {ref_entry['token']}
        for j in range(i + 1, len(entries)):
            cand = entries[j]
            if cand['keyframe'] != ref_keyframe_flag:
                continue
            if cand['token'] in used_tokens or cand['sensor'] in group:
                continue
            if any(abs(cand['timestamp'] - e['timestamp']) > max_time_diff for e in group.values()):
                continue
            group[cand['sensor']] = cand
            group_tokens.add(cand['token'])
        if len(group) == len(lidar_sensors):
            groups.append(group)
            used_tokens.update(group_tokens)
    return groups


# Function to parse your manually created JSON files
def parse_single_annotation_file(json_filepath):
    """
    Loads manual annotations for a single frame from a JSON file.
    """
    if not os.path.exists(json_filepath):
        return []

    with open(json_filepath, 'r') as f:
        data = json.load(f)

    boxes_for_this_frame = []
    if 'objects' not in data or not isinstance(data['objects'], list):
        return []

    for label_obj in data['objects']:
        try:
            center = [label_obj['centroid']['x'], label_obj['centroid']['y'], label_obj['centroid']['z']]
            # Annotation tool format: [length, width, height]
            # Your Box class expects: [width, length, height], so we reorder
            dims = [label_obj['dimensions']['width'], label_obj['dimensions']['length'],
                    label_obj['dimensions']['height']]
            yaw = label_obj['rotations']['z']

            box = Box(
                center=center,
                size=dims,
                orientation=Quaternion(axis=[0, 0, 1], angle=yaw)
            )
            box.name = label_obj['name']
            boxes_for_this_frame.append(box)
        except KeyError as e:
            print(f"Warning: Skipping an object in {json_filepath} due to missing key: {e}")
            continue

    return boxes_for_this_frame


# Function to save the final data into your desired JSON format
def save_boxes_to_json(output_filepath, pcd_base_dir, frame_index, boxes_list):
    """
    Saves a list of Box objects to a JSON file in the specified format.
    """
    objects_data = []
    for box in boxes_list:
        yaw, pitch, roll = box.orientation.yaw_pitch_roll
        w, l, h = box.wlh

        obj_dict = {
            "name": box.name,
            "centroid": {"x": box.center[0], "y": box.center[1], "z": box.center[2]},
            "dimensions": {"length": l, "width": w, "height": h},
            "rotations": {"x": roll, "y": pitch, "z": yaw}
        }
        objects_data.append(obj_dict)

    final_json_data = {
        "folder": "pointclouds",
        "filename": f"{frame_index:06d}_nonkeyframe.pcd",
        "path": f"{pcd_base_dir}/{frame_index:06d}_nonkeyframe.pcd",
        "objects": objects_data
    }

    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_filepath, 'w') as f:
        json.dump(final_json_data, f, indent=4)


def main():
    # --- Configuration ---
    dataroot = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    output_dir_base = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/annotation'
    version = 'v1.0-trainval'

    #dataroot = '/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test'
    #output_dir_base = '/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test/annotation'
    #version = 'v1.0-test'

    trucksc = TruckScenes(version=version, dataroot=dataroot, verbose=True)

    # You can loop through scenes here, for now we do scene 0
    scene_idx = 423
    my_scene = trucksc.scene[scene_idx]
    scene_name = my_scene['name']
    first_sample_token = my_scene['first_sample_token']
    my_sample = trucksc.get('sample', first_sample_token)

    sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR']
    max_time_diff = 90000

    lidar_entries = load_lidar_entries(trucksc=trucksc, sample=my_sample, lidar_sensors=sensors)
    groups = group_entries(entries=lidar_entries, lidar_sensors=sensors, max_time_diff=max_time_diff)
    print(f"\n✅ Scene: {scene_name}, Total frames found: {len(groups)}")

    # =================================================================
    # PASS 1: GATHER DATA & ANNOTATIONS FOR KEYFRAMES
    # =================================================================
    sample_list = []
    print("\n--- Pass 1: Gathering frame data and keyframe annotations ---")
    for i, group in enumerate(tqdm(groups, desc="Gathering frames")):
        ref_sensor = sensors[0]
        sample_data_dict = {sensor: group[sensor]['token'] for sensor in group}

        sample = {
            'timestamp': np.mean([g['timestamp'] for g in group.values()]),
            'data': sample_data_dict,
            'is_key_frame': group[ref_sensor]['keyframe'],
            'manual_boxes': []  # Initialize with empty list for ALL frames
        }

        if sample['is_key_frame']:
            annotation_base = os.path.join(dataroot, 'annotation')
            annotation_filename = f"{i:06d}_keyframe.json"
            annotation_data_load_path = os.path.join(annotation_base, scene_name, 'annotations', 'keyframes',
                                                     annotation_filename)

            manual_boxes = parse_single_annotation_file(annotation_data_load_path)
            if manual_boxes:
                sample['manual_boxes'] = manual_boxes

        sample_list.append(sample)

    # =================================================================
    # PASS 2: INTERPOLATE BOXES FOR NON-KEYFRAMES
    # =================================================================
    print("\n--- Pass 2: Interpolating data for non-keyframes ---")

    keyframes = sorted([(i, f) for i, f in enumerate(sample_list) if f['is_key_frame']],
                       key=lambda x: x[1]['timestamp'])

    if len(keyframes) < 2:
        print("Warning: Need at least 2 keyframes to interpolate. Can't process non-keyframes.")
    else:
        for i, frame_dict in enumerate(tqdm(sample_list, desc="Interpolating frames")):
            if frame_dict['is_key_frame']:
                continue

            current_ts = frame_dict['timestamp']
            kf_before_tuple = next((kf for kf in reversed(keyframes) if kf[0] < i), None)
            kf_after_tuple = next((kf for kf in keyframes if kf[0] > i), None)

            if not kf_before_tuple or not kf_after_tuple:
                nearest_kf_tuple = kf_before_tuple or kf_after_tuple
                if nearest_kf_tuple:
                    frame_dict['manual_boxes'] = deepcopy(nearest_kf_tuple[1]['manual_boxes'])
                continue

            kf_before_data, kf_after_data = kf_before_tuple[1], kf_after_tuple[1]
            ts_before, ts_after = kf_before_data['timestamp'], kf_after_data['timestamp']

            boxes_before_map = {box.name: box for box in kf_before_data['manual_boxes']}
            boxes_after_map = {box.name: box for box in kf_after_data['manual_boxes']}

            common_names = boxes_before_map.keys() & boxes_after_map.keys()

            if ts_after == ts_before: continue  # Avoid division by zero
            alpha = (current_ts - ts_before) / (ts_after - ts_before)

            interpolated_boxes = []
            for name in common_names:
                box_before = boxes_before_map[name]
                box_after = boxes_after_map[name]

                center_interp = (1 - alpha) * np.array(box_before.center) + alpha * np.array(box_after.center)

                # Scipy SLERP expects quaternions in [x, y, z, w] format
                q_before = box_before.orientation.elements[[1, 2, 3, 0]]
                q_after = box_after.orientation.elements[[1, 2, 3, 0]]
                key_rots = Rotation.from_quat([q_before, q_after])
                slerp = Slerp([ts_before, ts_after], key_rots)
                rot_interp = slerp(current_ts)

                quat_interp_xyzw = rot_interp.as_quat()
                quat_interp = Quaternion(w=quat_interp_xyzw[3], x=quat_interp_xyzw[0], y=quat_interp_xyzw[1],
                                         z=quat_interp_xyzw[2])

                interpolated_boxes.append(Box(
                    center=list(center_interp),
                    size=box_before.wlh,
                    orientation=quat_interp,
                    name=name
                ))
            frame_dict['manual_boxes'] = interpolated_boxes

    # =================================================================
    # PASS 3: SAVE THE FINAL ANNOTATIONS TO JSON FILES
    # =================================================================
    print("\n--- Pass 3: Saving final annotations to JSON ---")
    pcd_base_dir = os.path.join(output_dir_base, scene_name, 'pointclouds')
    output_scene_dir = os.path.join(output_dir_base, scene_name, 'annotations', 'nonkeyframes')

    for i, frame_dict in enumerate(tqdm(sample_list, desc="Saving JSON files")):
        output_filepath = os.path.join(output_scene_dir, f"{i:06d}_nonkeyframe.json")

        if not frame_dict['is_key_frame'] and frame_dict['manual_boxes']:
            save_boxes_to_json(output_filepath, pcd_base_dir, i, frame_dict['manual_boxes'])

    print(f"\n✅ Done! Interpolated annotations saved to: {output_scene_dir}")


if __name__ == '__main__':
    main()
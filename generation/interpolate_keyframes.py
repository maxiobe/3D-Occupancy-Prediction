# File: interpolate_KEYFRAMES_only_v2.py
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation, Slerp
import json

# --- Assuming these are in the same directory or your python path ---
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box

# =====================================================================================
# HELPER FUNCTIONS (UNCHANGED)
# =====================================================================================
def load_lidar_entries(trucksc, sample, lidar_sensors):
    entries = []
    for sensor in lidar_sensors:
        token = sample['data'][sensor]
        while token:
            sd = trucksc.get('sample_data', token)
            entries.append(
                {'sensor': sensor, 'timestamp': sd['timestamp'], 'token': token, 'keyframe': sd['is_key_frame']})
            token = sd['next']
    entries.sort(key=lambda x: x['timestamp'])
    return entries

def group_entries(entries, lidar_sensors, max_time_diff):
    """
    This function is now restored to its ORIGINAL state.
    It returns ALL frame groups (keyframes and non-keyframes) to preserve the scene's timeline.
    """
    used_tokens = set()
    groups = []
    for i, ref_entry in enumerate(entries):
        if ref_entry['token'] in used_tokens: continue
        ref_keyframe_flag = ref_entry['keyframe']
        group = {ref_entry['sensor']: ref_entry};
        group_tokens = {ref_entry['token']}
        for j in range(i + 1, len(entries)):
            cand = entries[j]
            # Match keyframe flag to avoid mixing
            if cand['keyframe'] != ref_keyframe_flag: continue
            if cand['token'] in used_tokens or cand['sensor'] in group: continue
            if any(abs(cand['timestamp'] - e['timestamp']) > max_time_diff for e in group.values()): continue
            group[cand['sensor']] = cand;
            group_tokens.add(cand['token'])
        if len(group) == len(lidar_sensors):
            groups.append(group);
            used_tokens.update(group_tokens)
    return groups

def parse_single_annotation_file(json_filepath):
    if not os.path.exists(json_filepath): return []
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    boxes_for_this_frame = []
    if 'objects' not in data or not isinstance(data['objects'], list): return []
    for label_obj in data['objects']:
        try:
            box = Box(center=[label_obj['centroid']['x'], label_obj['centroid']['y'], label_obj['centroid']['z']],
                      size=[label_obj['dimensions']['width'], label_obj['dimensions']['length'],
                            label_obj['dimensions']['height']],
                      orientation=Quaternion(axis=[0, 0, 1], angle=label_obj['rotations']['z']),
                      name=label_obj['name'])
            boxes_for_this_frame.append(box)
        except KeyError as e:
            print(f"Warning: Skipping an object in {json_filepath} due to missing key: {e}"); continue
    return boxes_for_this_frame

def save_boxes_to_json(output_filepath, pcd_base_dir, original_frame_index, boxes_list):
    objects_data = []
    for box in boxes_list:
        yaw, pitch, roll = box.orientation.yaw_pitch_roll;
        w, l, h = box.wlh
        obj_dict = {"name": box.name, "centroid": {"x": box.center[0], "y": box.center[1], "z": box.center[2]},
                    "dimensions": {"length": l, "width": w, "height": h},
                    "rotations": {"x": roll, "y": pitch, "z": yaw}}
        objects_data.append(obj_dict)
    # The output filename uses the ORIGINAL frame index
    final_json_data = {"folder": "pointclouds", "filename": f"{original_frame_index:06d}_keyframe.pcd",
                       "path": f"{pcd_base_dir}/{original_frame_index:06d}_keyframe.pcd", "objects": objects_data}
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w') as f: json.dump(final_json_data, f, indent=4)

# =====================================================================================
# MAIN SCRIPT LOGIC
# =====================================================================================
def main():
    # --- 1. Configuration ---
    dataroot = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    version = 'v1.0-trainval'

    #dataroot = '/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test'
    #version = 'v1.0-test'
    scene_idx_to_process = 423

    # --- Initialize ---
    trucksc = TruckScenes(version=version, dataroot=dataroot, verbose=True)
    my_scene = trucksc.scene[scene_idx_to_process]
    scene_name = my_scene['name']
    first_sample_token = my_scene['first_sample_token']
    my_sample = trucksc.get('sample', first_sample_token)
    sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR']
    max_time_diff = 90000

    # --- 2. Gather ALL frames to preserve original indexing ---
    print(f"\nGathering and grouping ALL frames for scene: {scene_name}...")
    lidar_entries = load_lidar_entries(trucksc=trucksc, sample=my_sample, lidar_sensors=sensors)
    all_scene_frames = group_entries(entries=lidar_entries, lidar_sensors=sensors, max_time_diff=max_time_diff)
    print(f"✅ Found {len(all_scene_frames)} total frames in the scene.")

    # --- 3. Build a new timeline containing ONLY keyframes ---
    # Each keyframe object will store its original index from the full timeline.
    print("\nFiltering for keyframes and identifying manual 'anchor' annotations...")
    keyframe_timeline = []
    manual_anchor_indices = [] # This will store the index *within the keyframe_timeline*

    # Iterate through ALL frames to find the keyframes
    for i, group in enumerate(tqdm(all_scene_frames, desc="Finding keyframes")):
        # Check if the current frame group is a keyframe
        if not group[sensors[0]]['keyframe']:
            continue

        # It's a keyframe, so add it to our new timeline
        frame_info = {
            'original_index': i,  # <-- CRITICAL: Store the absolute index (e.g., 195)
            'timestamp': np.mean([g['timestamp'] for g in group.values()]),
            'is_manual_anchor': False,
            'boxes': []
        }

        # Check if a manual annotation exists for this keyframe using its ORIGINAL index
        manual_annotation_path = Path(dataroot) / 'annotation' / scene_name / 'annotations' / 'keyframes_manual' / f"{i:06d}_keyframe.json"

        if manual_annotation_path.exists():
            frame_info['is_manual_anchor'] = True
            frame_info['boxes'] = parse_single_annotation_file(manual_annotation_path)
            # Store the index of this anchor *within the keyframe_timeline*
            manual_anchor_indices.append(len(keyframe_timeline))

        keyframe_timeline.append(frame_info)

    print(f"✅ Found {len(keyframe_timeline)} keyframes in total.")
    print(f"✅ Identified {len(manual_anchor_indices)} manually annotated anchor keyframes.")


    # --- 4. Pass 2: Interpolate the EMPTY Keyframes ---
    print("\n--- Pass 2: Generating annotations for empty keyframes... ---")
    if len(manual_anchor_indices) < 2:
        print("Warning: Need at least 2 manually annotated 'anchor' keyframes to interpolate. Skipping Pass 2.")
    else:
        # Now we loop through our clean `keyframe_timeline`
        for i, frame in enumerate(tqdm(keyframe_timeline, desc="Interpolating keyframes")):
            if frame['is_manual_anchor']:
                continue

            anchor_before_idx = max([k_idx for k_idx in manual_anchor_indices if k_idx < i], default=None)
            anchor_after_idx = min([k_idx for k_idx in manual_anchor_indices if k_idx > i], default=None)

            if anchor_before_idx is None or anchor_after_idx is None:
                continue

            anchor_before_data = keyframe_timeline[anchor_before_idx]
            anchor_after_data = keyframe_timeline[anchor_after_idx]
            boxes_before_map = {box.name: box for box in anchor_before_data['boxes']}
            boxes_after_map = {box.name: box for box in anchor_after_data['boxes']}
            common_box_names = boxes_before_map.keys() & boxes_after_map.keys()

            ts_before, ts_after, ts_current = anchor_before_data['timestamp'], anchor_after_data['timestamp'], frame['timestamp']
            if ts_after == ts_before: continue
            alpha = (ts_current - ts_before) / (ts_after - ts_before)

            interpolated_boxes = []
            for name in common_box_names:
                box_before, box_after = boxes_before_map[name], boxes_after_map[name]
                center_interp = (1 - alpha) * np.array(box_before.center) + alpha * np.array(box_after.center)
                key_rots = Rotation.from_quat(
                    [box_before.orientation.elements[[1, 2, 3, 0]], box_after.orientation.elements[[1, 2, 3, 0]]])
                slerp = Slerp([ts_before, ts_after], key_rots)
                rot_interp = slerp(ts_current)
                quat_interp_xyzw = rot_interp.as_quat()
                quat_interp = Quaternion(w=quat_interp_xyzw[3], x=quat_interp_xyzw[0], y=quat_interp_xyzw[1], z=quat_interp_xyzw[2])
                interpolated_boxes.append(
                    Box(center=list(center_interp), size=box_before.wlh, orientation=quat_interp, name=name))

            frame['boxes'] = interpolated_boxes
    print("✅ Pass 2 complete.")

    # --- 5. Pass 3: Save the newly generated keyframe annotations ---
    print("\n--- Pass 3: Saving interpolated keyframe annotations for your review... ---")
    output_dir_for_review = Path(dataroot) / 'annotation' / scene_name / 'annotations' / 'keyframes_interpolated'
    pcd_base_dir = Path(dataroot) / 'annotation' / scene_name / 'pointclouds'

    for frame in tqdm(keyframe_timeline, desc="Saving files"):
        if not frame['is_manual_anchor'] and frame['boxes']:
            # Use the stored ORIGINAL index for the filename
            original_index = frame['original_index']
            output_filepath = output_dir_for_review / f"{original_index:06d}_keyframe.json"
            save_boxes_to_json(output_filepath, str(pcd_base_dir), original_index, frame['boxes'])

    print(f"\n🎉 Success! New keyframes for review have been saved to: {output_dir_for_review}")

if __name__ == '__main__':
    main()
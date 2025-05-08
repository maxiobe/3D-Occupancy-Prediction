import os
from truckscenes.truckscenes import TruckScenes


# Config
DATAROOT = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'  # Adjust path as needed
SCENE_INDEX = 0  # Change to the index of the scene you want to inspect

MAX_TIME_DIFF = 80000  # microseconds = 80 ms
LIDAR_SENSOR_NAMES = [
    'LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT',
    'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR'
]

def load_lidar_entries(trucksc, sample):
    entries = []
    for sensor in LIDAR_SENSOR_NAMES:
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

def group_entries(entries):
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
            if abs(cand['timestamp'] - ref_entry['timestamp']) > MAX_TIME_DIFF:
                break
            group[cand['sensor']] = cand
            group_tokens.add(cand['token'])

        if len(group) == len(LIDAR_SENSOR_NAMES):
            groups.append(group)
            used_tokens.update(group_tokens)

    return groups

def main():

    trucksc = TruckScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)
    scene = trucksc.scene[SCENE_INDEX]
    print(f"Inspecting scene: {scene['name']}")
    sample = trucksc.get('sample', scene['first_sample_token'])

    """sample_time_previous = None
    while True:
        print(f"Sample timestamp: {sample['timestamp']}")
        if sample_time_previous is not None:
            print(f"Sample timestamp difference: {abs(sample['timestamp'] - sample_time_previous)}")

        timestamp_sensor = []
        for sensor in lidar_sensor_names:
            sd = trucksc.get('sample_data', sample['data'][sensor])
            sd_timestamp = sd['timestamp']
            timestamp_sensor.append(sd_timestamp)
        print(timestamp_sensor)
        print(max(timestamp_sensor) - min(timestamp_sensor))

        next_sample_token = sample['next']
        if next_sample_token != '':
            sample_time_previous = sample['timestamp']
            sample = trucksc.get('sample', next_sample_token)
        else:
            break"""

    # Load entries
    lidar_entries = load_lidar_entries(trucksc, sample)

    print(f"Number of lidar entries: {len(lidar_entries)}")

    groups = group_entries(lidar_entries)
    # Summary
    print(f"\n✅ Total groups found: {len(groups)}")

    # Example output
    for idx, group in enumerate(groups[:10]):
        print(f"\n🔍 Group {idx + 1} (keyframe={list(group.values())[0]['keyframe']}):")
        for sensor in LIDAR_SENSOR_NAMES:
            e = group[sensor]
            print(f"  {sensor}: ts={e['timestamp']} token={e['token']}")


if __name__ == "__main__":
    main()
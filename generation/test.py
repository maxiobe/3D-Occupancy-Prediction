import os
from truckscenes.truckscenes import TruckScenes

from generation.visu import my_sample

# Config
DATAROOT = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'  # Adjust path as needed
SCENE_INDEX = 0  # Change to the index of the scene you want to inspect
REF_SENSOR = 'LIDAR_RIGHT'  # or LIDAR_RIGHT depending on your dataset

def main():
    # Load dataset
    trucksc = TruckScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)

    # Get a scene
    scene = trucksc.scene[SCENE_INDEX]
    print(f"Inspecting scene: {scene['name']}")

    # Get first sample (keyframe)
    sample = trucksc.get('sample', scene['first_sample_token'])
    lidar_token = sample['data'][REF_SENSOR]
    next_sample_token = sample['next']

    """print("\nWalking through LIDAR samples chain...\n")

    while next_sample_token:
        lidar_data = trucksc.get('sample_data', sample['data'][REF_SENSOR])
        print(lidar_data)
        print(lidar_data['filename'])

        next_sample_token = sample['next']
        if next_sample_token != '':
            sample = trucksc.get('sample', next_sample_token)"""



    print("\nWalking through LIDAR sample_data chain...\n")

    while lidar_token:
        lidar_data = trucksc.get('sample_data', lidar_token)

        print(f"SampleData Token: {lidar_data['token']}")
        print(f"  is_key_frame: {lidar_data['is_key_frame']}")
        print(f"  filename: {lidar_data['filename']}")

        # Move to next LIDAR frame (may be key or non-key)
        lidar_token = lidar_data['next']

if __name__ == "__main__":
    main()

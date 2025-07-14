import json
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.splits import train, val, test, mini_train, mini_val
from tqdm import tqdm
import os

# Trainval
#  = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
# TRUCKSCENES_VERSION = 'v1.0-trainval'

# Test
TRUCKSCENES_DATA_ROOT = '/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test'
TRUCKSCENES_VERSION = 'v1.0-test'

# Mini
#TRUCKSCENES_DATA_ROOT = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini'
#TRUCKSCENES_VERSION = 'v1.0-mini'

OUTPUT_FILENAME = 'annotations.json'

def main():
    if TRUCKSCENES_VERSION == 'v1.0-mini':
        train_split = mini_train
        val_split = mini_val
    elif TRUCKSCENES_VERSION == 'v1.0-trainval':
        train_split = train
        val_split = val
    elif TRUCKSCENES_VERSION == 'v1.0-test':
        test_split = test
    else:
        raise NotImplementedError(f"The version '{TRUCKSCENES_VERSION}' is not a valid or recognized version.")

    scene_infos = {}
    print(f"Initializing TruckScenes dataset from: {TRUCKSCENES_DATA_ROOT}")
    trucksc = TruckScenes(version=TRUCKSCENES_VERSION, dataroot=TRUCKSCENES_DATA_ROOT, verbose=True)

    camera_names = ['CAMERA_LEFT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_BACK']

    # Iterate through every scene
    for scene_record in tqdm(trucksc.scene, desc="Processing Scenes"):
        scene_name = scene_record['name']
        scene_infos[scene_name] = {}
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = trucksc.get('sample', sample_token)

            sample_token_next = sample_record['next']
            sample_token_previous = sample_record['prev']
            gt_path_sample = f"gts/{scene_name}/{sample_token}/labels.npz"
            timestamp = sample_record['timestamp']
            ego_pose_record = trucksc.getclosest('ego_pose', timestamp)
            ego_pose = {
                'translation': ego_pose_record['translation'],
                'rotation': ego_pose_record['rotation'],
            }

            sample_data = sample_record['data']

            camera_sensor_info = {}
            for cam_name in camera_names:
                if cam_name in sample_data:
                    cam_token = sample_data[cam_name]
                    cam_record = trucksc.get('sample_data', cam_token)
                    calibrated_sensor = trucksc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
                    cam_ego_pose_record = trucksc.get('ego_pose', cam_record['ego_pose_token'])
                    cam_ego_pose = {
                        'token': cam_ego_pose_record['token'],
                        'timestamp': cam_ego_pose_record['timestamp'],
                        'rotation': cam_ego_pose_record['rotation'],
                        'translation': cam_ego_pose_record['translation'],
                    }
                    camera_sensor_info[cam_name] = {
                        'intrinsics': calibrated_sensor['camera_intrinsic'],
                        'extrinsic': {
                            'translation': calibrated_sensor['translation'],
                            'rotation': calibrated_sensor['rotation'],
                        },
                        'ego_pose': cam_ego_pose,
                        'img_path': cam_record['filename']
                    }

            scene_infos[scene_name][sample_token] = {
                'timestamp': timestamp,
                'ego_pose': ego_pose,
                'camera_sensor': camera_sensor_info,
                'gt_path': gt_path_sample,
                'prev': sample_token_previous,
                'next': sample_token_next,
            }

            sample_token = sample_record['next']

    if TRUCKSCENES_VERSION == 'v1.0-test':
        output_data = {
            'test_split': test_split,
            'scene_infos': scene_infos,
        }
    else:
        output_data = {
            'train_split': train_split,
            'val_split': val_split,
            'scene_infos': scene_infos,
        }

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(output_data, f)

    print(f"\nSuccessfully created '{OUTPUT_FILENAME}'")
    print(f"Total scenes: {len(trucksc.scene)}")
    if TRUCKSCENES_VERSION == 'v1.0-trainval' or TRUCKSCENES_VERSION == 'v1.0-mini':
        print(f"Training scenes: {len(train_split)}")
        print(f"Validation scenes: {len(val_split)}")
    else:
        print(f"Test scenes: {len(test_split)}")


if __name__ == '__main__':
    main()
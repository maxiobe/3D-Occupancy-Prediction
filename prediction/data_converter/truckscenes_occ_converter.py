import mmcv
import numpy as np
import os
from collections import OrderedDict
from os import path as osp
from pyquaternion import Quaternion
from typing import List, Tuple, Union

from shapely.geometry import MultiPoint, box
from mmdet3d.core.bbox.box_np_ops import points_cam2img
import simplejson as json

from truckscenes.truckscenes import TruckScenes
from truckscenes.utils import splits as truckscenes_splits
from truckscenes.utils.geometry_utils import view_points


def _get_imu_info(trsc, sample):
    """
    Fetch IMU data from the chassis sensor around the sample's timestamp.
    """
    sample_timestamp = sample['timestamp']

    ego_pose = trsc.getclosest('ego_pose', sample_timestamp)
    position = ego_pose['translation']
    orientation = ego_pose['rotation']

    imu = trsc.getclosest('ego_motion_chassis', sample_timestamp)

    velocity = [imu['vx'], imu['vy'], imu['vz']]
    acceleration = [imu['ax'], imu['ay'], imu['az']]
    rotation_rate = [imu['roll_rate'], imu['pitch_rate'], imu['yaw_rate']]

    # imu_pitch = imu['pitch']
    # imu_roll = imu['roll']
    # imu_yaw = imu['yaw']

    features = []
    features.extend(position)
    features.extend(orientation)
    features.extend(velocity)
    features.extend(acceleration)
    features.extend(rotation_rate)
    features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def get_available_scenes(trsc):
    """Get available scenes from the input truckscenes class."""
    available_scenes = []
    print(f'total scene num: {len(trsc.scene)}')
    for scene in trsc.scene:
        lidar_sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR'] # very slow
        # lidar_sensors = ['LIDAR_LEFT']
        first_sample_token = scene['first_sample_token']
        sample_rec = trsc.get('sample', first_sample_token)

        scene_available = True

        for lidar_sensor in lidar_sensors:
            if lidar_sensor not in sample_rec['data']:
                scene_available = False
                break
            sd_token = sample_rec['data'][lidar_sensor]
            lidar_path, _, _ = trsc.get_sample_data(sd_token)

            if not mmcv.is_filepath(str(lidar_path)):
                scene_available = False
                break

        if scene_available:
            available_scenes.append(scene)

    print(f'Existing scene number: {len(available_scenes)}')
    return available_scenes


def _get_sensor2primarylidar_transform(trsc, sensor_token, pl2e_t, pl2e_r_mat, e2g_t, e2g_r_mat, sensor_type):
    """
    Get the transformation matrix from a sensor to the ego vehicle's frame.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        info (dict): Sweep information after transformation.
    """
    sd_rec = trsc.get('sample_data', sensor_token)
    cs_record = trsc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

    # Transformation from sensor to ego frame
    sensor2ego_trans = cs_record['translation']
    sensor2ego_rot = cs_record['rotation']

    pose_record = trsc.get('ego_pose', sd_rec['ego_pose_token'])
    ego2global_trans = pose_record['translation']
    ego2global_rot = pose_record['rotation']

    info = {
        'data_path': str(trsc.get_sample_data_path(sd_rec['token'])),
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': sensor2ego_trans,
        'sensor2ego_rotation': sensor2ego_rot,
        'ego2global_translation': ego2global_trans,
        'ego2global_rotation': ego2global_rot,
        'timestamp': sd_rec['timestamp'],
    }

    l2e_r_sweep_mat = Quaternion(sensor2ego_rot).rotation_matrix
    e2g_r_sweep_mat = Quaternion(ego2global_rot).rotation_matrix

    R = (l2e_r_sweep_mat.T @ e2g_r_sweep_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(pl2e_r_mat).T)

    T = (sensor2ego_trans @ e2g_r_sweep_mat.T + ego2global_trans) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(pl2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(pl2e_r_mat).T
                  ) + pl2e_t @ np.linalg.inv(pl2e_r_mat).T

    info['sensor2primarylidar_rotation'] = R
    info['sensor2primarylidar_translation'] = T

    return info


def _fill_occ_trainval_infos(trsc,
                             occ_anno,
                             token2name,
                             train_scenes,
                             val_scenes,
                             test=False,
                             max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        trsc (:obj:`TruckScenes`): Dataset class in the truckScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """

    train_trsc_infos = []
    val_trsc_infos = []
    frame_idx = 0
    scene_infos = occ_anno.get('scene_infos', {})

    for sample in mmcv.track_iter_progress(trsc.sample):
        scene_token = sample['scene_token']
        scene_name = token2name[scene_token]

        # Check if the sample belongs to a scene we want to process
        if scene_token not in train_scenes and scene_token not in val_scenes:
            print(f"{scene_token} not in train and val scenes. Skipping.")
            continue

        primary_lidar = 'LIDAR_LEFT'
        primary_lidar_token = sample['data'].get(primary_lidar)
        if not primary_lidar_token:
            print(f"Primary lidar token {primary_lidar_token} not found. Skipping.")
            continue

        sd_rec = trsc.get('sample_data', primary_lidar_token)
        sample_token = sd_rec['sample_token']

        if sample_token not in scene_infos.get(scene_name, {}):
            print(f"{sample_token} not in scene_infos. Skipping.")
            continue

        occ_sample = scene_infos[scene_name][sample_token]
        pose_record = trsc.get('ego_pose', sd_rec['ego_pose_token'])
        ego2global_trans = pose_record['translation']
        ego2global_rot = pose_record['rotation']

        cs_record = trsc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        primary_lidar_path, boxes, _ = trsc.get_sample_data(primary_lidar_token)

        primary_lidar2ego_trans = cs_record['translation']
        primary_lidar2ego_rot = cs_record['rotation']

        primary_lidar2ego_rot_mat = Quaternion(primary_lidar2ego_rot).rotation_matrix
        ego2global_rot_mat = Quaternion(ego2global_rot).rotation_matrix

        # Get IMU data instead of CAN bus
        imu_data = _get_imu_info(trsc, sample)

        info = {
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'imu_data': imu_data,  # Replaced 'can_bus'
            'frame_idx': frame_idx, # temporal related info
            'sweeps': [],  # Sweeps will be for the primary LiDAR
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'ego2global_translation': ego2global_trans,
            'ego2global_rotation': ego2global_rot,
            'timestamp': sample['timestamp'],
            'primary_lidar_path': primary_lidar_path,
            'primarylidar2ego_translation': primary_lidar2ego_trans,
            'primarylidar2ego_rotation': primary_lidar2ego_rot,
            'lidars': dict(),  # Store all LiDARs
        }

        # Add occupancy ground truth path
        if 'gt_path' in occ_sample:
            info['occ_gt_path'] = occ_sample['gt_path']

        camera_types = [
            'CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_BACK'
        ]

        for cam in camera_types:
            if cam in sample['data']:
                cam_token = sample['data'][cam]
                cam_path, _, cam_intrinsic = trsc.get_sample_data(cam_token)
                cam_info = _get_sensor2primarylidar_transform(trsc, cam_token, primary_lidar2ego_trans, primary_lidar2ego_rot_mat, ego2global_trans, ego2global_rot_mat, cam)
                cam_info.update(cam_intrinsic=cam_intrinsic)
                cam_info.update(cam_path=cam_path)
                info['cams'].update({cam: cam_info})

        lidar_types = [
            'LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT',
            'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR'
        ]
        # Get data for all 6 LiDARs
        for lidar in lidar_types:
            if lidar in sample['data']:
                lidar_token = sample['data'][lidar]
                lidar_path, boxes, _ = trsc.get_sample_data(lidar_token)
                lidar_info = _get_sensor2primarylidar_transform(trsc, lidar_token, primary_lidar2ego_trans, primary_lidar2ego_rot_mat, ego2global_trans, ego2global_rot_mat, lidar)
                info['lidars'][lidar] = lidar_info

        # Obtain sweeps for the single primary LiDAR
        sweeps = []
        sweep_sd_rec = trsc.get('sample_data', primary_lidar_token)
        while len(sweeps) < max_sweeps:
            if not sweep_sd_rec['prev'] == '':
                # Get transformation from this sweep's sensor frame to the *current* ego frame
                sweep_token = sweep_sd_rec['prev']
                sweep_info = _get_sensor2primarylidar_transform(trsc, sweep_token, primary_lidar2ego_trans, primary_lidar2ego_rot_mat, ego2global_trans, ego2global_rot_mat, primary_lidar)

                sweeps.append(sweep_info)
                sweep_sd_rec = trsc.get('sample_data', sweep_sd_rec['prev'])
            else:
                break
            info['sweeps'] = sweeps


        if not test and 'anns' in sample:
            pass

        if sample['scene_token'] in train_scenes:
            train_trsc_infos.append(info)
        else:
            val_trsc_infos.append(info)

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

    return train_trsc_infos, val_trsc_infos


def create_truckscenes_occ_infos(root_path,
                          occ_path,
                          out_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of truckscenes dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """

    print(f"Creating TruckScenes infos for version: {version} from {root_path}")
    trsc = TruckScenes(version=version, dataroot=root_path, verbose=True)

    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers, f'Version {version} is not supported.'

    if version == 'v1.0-trainval':
        train_scenes = truckscenes_splits.train
        val_scenes = truckscenes_splits.val
    elif version == 'v1.0-test':
        train_scenes = truckscenes_splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = truckscenes_splits.mini_train
        val_scenes = truckscenes_splits.mini_val
    else:
        raise ValueError(f'Version {version} is not supported.')

    print("Splits created....")

    with open(os.path.join(occ_path, 'annotations.json'), 'r') as f:
        occ_anno = json.load(f)

    print('File annotations.json loaded....')

    available_scenes = get_available_scenes(trsc)

    train_scenes = {s['token'] for s in available_scenes if s['name'] in train_scenes}
    val_scenes = {s['token'] for s in available_scenes if s['name'] in val_scenes}

    token2name = {scene['token']: scene['name'] for scene in trsc.scene}

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_trsc_infos, val_trsc_infos = _fill_occ_trainval_infos(
        trsc, occ_anno, token2name, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print(f'test sample: {len(train_trsc_infos)}')
        data = dict(infos=train_trsc_infos, metadata=metadata)
        info_path = osp.join(out_path, f'{info_prefix}_infos_temporal_test.pkl')
        mmcv.dump(data, info_path)
    else:
        print(f'train sample: {len(train_trsc_infos)}, val sample: {len(val_trsc_infos)}')
        # Save training data
        data = dict(infos=train_trsc_infos, metadata=metadata)
        info_path = osp.join(out_path, f'{info_prefix}_infos_temporal_train.pkl')
        mmcv.dump(data, info_path)
        # Save validation data
        data['infos'] = val_trsc_infos
        info_val_path = osp.join(out_path, f'{info_prefix}_infos_temporal_val.pkl')
        mmcv.dump(data, info_val_path)

    print(f"Successfully generated .pkl files in {out_path}")




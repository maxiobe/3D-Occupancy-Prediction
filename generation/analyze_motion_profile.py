import numpy as np
import torch
from pyquaternion import Quaternion
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import textwrap
import matplotlib.pyplot as plt
import os

# --- Essential imports from the TruckScenes and PyTorch3D libraries ---
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box

TRUCKSCENES_DATA_ROOT = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'  # Path to your TruckScenes data
TRUCKSCENES_VERSION = 'v1.0-trainval'  # Dataset version

OUTPUT_FOLDER = '/home/max/Desktop/Masterarbeit/IMU_check_scenes'


def plot_imu_data(scene_idx: int, scene_name: str, data: Dict[str, List[float]], output_dir: str):
    """
    Generates and saves plots for the collected IMU data for a single scene.

    Args:
        scene_idx (int): The index of the scene.
        scene_name (str): The name of the scene for titling the plot.
        data (Dict[str, List[float]]): A dictionary containing the collected time-series data.
        output_dir (str): The directory where the plot image will be saved.
    """
    print(f"Generating plot for scene {scene_idx}: {scene_name}")

    # --- NEW: Update the title format ---
    title = f'Scene {scene_idx}: {scene_name}\nIMU Data Comparison: Cabin vs. Chassis'

    # Create a figure with 2x2 subplots for better visualization
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)

    time_s = data['time_s']

    # --- Plot 1 (Top-Left): Longitudinal Acceleration (ax) ---
    axs[0, 0].plot(time_s, data['chassis_ax'], label='Chassis', color='blue', alpha=0.8)
    axs[0, 0].plot(time_s, data['cabin_ax'], label='Cabin', color='red', linestyle='--', alpha=0.8)
    axs[0, 0].set_title('Longitudinal Acceleration ($a_x$)')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Acceleration ($m/s^2$)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- Plot 2 (Top-Right): Lateral Acceleration (ay) ---
    axs[0, 1].plot(time_s, data['chassis_ay'], label='Chassis', color='blue', alpha=0.8)
    axs[0, 1].plot(time_s, data['cabin_ay'], label='Cabin', color='red', linestyle='--', alpha=0.8)
    axs[0, 1].set_title('Lateral Acceleration ($a_y$)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Acceleration ($m/s^2$)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- Plot 3 (Bottom-Left): Vertical Acceleration (az) ---
    axs[1, 0].plot(time_s, data['chassis_az'], label='Chassis', color='blue', alpha=0.8)
    axs[1, 0].plot(time_s, data['cabin_az'], label='Cabin', color='red', linestyle='--', alpha=0.8)
    axs[1, 0].set_title('Vertical Acceleration ($a_z$)')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Acceleration ($m/s^2$)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- Plot 4 (Bottom-Right): Pitch Rate ---
    axs[1, 1].plot(time_s, data['chassis_pitch_rate'], label='Chassis', color='blue', alpha=0.8)
    axs[1, 1].plot(time_s, data['cabin_pitch_rate'], label='Cabin', color='red', linestyle='--', alpha=0.8)
    axs[1, 1].set_title('Pitch Rate')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- NEW: Update the filename format and save path ---
    output_filename = f"scene_{scene_idx}_{scene_name}.png"
    full_path = os.path.join(output_dir, output_filename)
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")

    # --- NEW: Close the plot to free up memory ---
    # This is important when creating many plots in a loop.
    plt.close(fig)



def main():
    """
    Main function to analyze object overlaps across all scenes and calculate statistics.
    """
    print(f"Initializing TruckScenes dataset from: {TRUCKSCENES_DATA_ROOT}")
    trucksc = TruckScenes(version=TRUCKSCENES_VERSION, dataroot=TRUCKSCENES_DATA_ROOT, verbose=True)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")

    print("\nStarting analysis for all scenes...")

    total_scenes = len(trucksc.scene)
    for scene_idx, scene_record in enumerate(trucksc.scene):
        scene_name = scene_record['name']
        print(f"Processing Scene {scene_idx + 1}/{total_scenes}: {scene_name}")

        scene_data = defaultdict(list)
        first_sample_timestamp = None

        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = trucksc.get('sample', sample_token)

            sample_timestamp = sample_record['timestamp']

            if first_sample_timestamp is None:
                first_sample_timestamp = sample_timestamp

            imu_chassis = trucksc.getclosest('ego_motion_chassis', sample_timestamp)
            imu_cabin = trucksc.getclosest('ego_motion_cabin', sample_timestamp)

            # Timestamps are in microseconds, convert to seconds relative to the start
            relative_time_s = (sample_timestamp - first_sample_timestamp) / 1e6
            scene_data['time_s'].append(relative_time_s)

            # Collect chassis data
            scene_data['chassis_ax'].append(imu_chassis['ax'])
            scene_data['chassis_ay'].append(imu_chassis['ay'])
            scene_data['chassis_az'].append(imu_chassis['az'])
            scene_data['chassis_pitch_rate'].append(imu_chassis['pitch_rate'])

            # Collect cabin data
            scene_data['cabin_ax'].append(imu_cabin['ax'])
            scene_data['cabin_ay'].append(imu_cabin['ay'])
            scene_data['cabin_az'].append(imu_cabin['az'])
            scene_data['cabin_pitch_rate'].append(imu_cabin['pitch_rate'])

            sample_token = sample_record['next']

        if scene_data:
            plot_imu_data(scene_idx, scene_name, scene_data, OUTPUT_FOLDER)

if __name__ == '__main__':
    main()
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box, LidarPointCloud
import numpy as np
from pyquaternion import Quaternion
from typing import Any, Dict, List, Optional, Union


def get_boxes(trucksc: TruckScenes, sample: Dict[str, Any]) -> List[Box]:
    """ Retruns the bounding boxes of the given sample.

    Arguments:
        trucksc: TruckScenes dataset instance.
        sample: Reference sample to get the boxes from.

    Returns:
        boxes: List of box instances in the ego vehicle frame at the
            timestamp of the sample.
    """
    # Retrieve all sample annotations
    boxes = list(map(trucksc.get_box, sample['anns']))

    # Get reference ego pose (timestamp of the sample/annotations)
    ref_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])

    # Transform boxes to ego frame
    for box in boxes:
        box.translate(-np.array(ref_ego_pose['translation']))
        box.rotate(Quaternion(ref_ego_pose['rotation']).inverse)

    return boxes

def main(trucksc: TruckScenes, testset: bool = False):
    """
    Finds all instances of 'vehicle.other' across all scenes and calculates
    the min, max, and average dimensions.
    """
    # Initialize a single list to hold all found vehicle dictionaries
    all_vehicles_data = []
    vehicles_above_8m = []

    seen_instance_tokens = set()

    # Using a direct loop over trucksc.scene is more Pythonic
    for scene_idx, my_scene in enumerate(trucksc.scene):
        scene_name = my_scene['name']
        print(f"\n--- Processing Scene {scene_idx}: '{scene_name}' ---")

        # Start with the first sample in the scene
        sample_token = my_scene['first_sample_token']

        while sample_token:
            my_sample = trucksc.get('sample', sample_token)

            boxes = get_boxes(trucksc, my_sample)

            if not boxes:
                # Move to the next sample if no boxes are found
                sample_token = my_sample.get('next', '')
                continue

            # Extract necessary data for all boxes in the current sample
            box_tokens = [box.token for box in boxes]
            instance_tokens = [trucksc.get('sample_annotation', bt)['instance_token'] for bt in box_tokens]
            category_names = [trucksc.get('sample_annotation', bt)['category_name'] for bt in box_tokens]
            dims = np.array([b.wlh for b in boxes])  # Shape: (N, 3) for N boxes

            # Loop through all boxes found in this sample
            for i, category_name in enumerate(category_names):
                instance_token = instance_tokens[i]
                if category_name == 'vehicle.truck' and instance_token not in seen_instance_tokens:
                    # If it's the category we want, create a dictionary and append it
                    vehicle_dict = {
                        'scene_name': scene_name,
                        'sample_token': my_sample['token'],
                        'box_token': box_tokens[i],
                        'instance_token': instance_tokens[i],
                        'category_name': category_names[i],
                        'wlh': dims[i]  # Storing the [width, length, height] array
                    }
                    print(vehicle_dict)
                    all_vehicles_data.append(vehicle_dict)

                    if dims[i][1] > 8:
                        print(f"Above 8m: {vehicle_dict}")
                        vehicles_above_8m.append(vehicle_dict)

                    seen_instance_tokens.add(instance_token)

            # Move to the next sample token
            sample_token = my_sample.get('next', '')

    print(f"\n--- Processing Complete ---")
    print(f"Found a total of {len(all_vehicles_data)} instances of 'vehicle.other'.")

    # Now, let's analyze the dimensions from the collected data
    if not all_vehicles_data:
        print("No 'vehicle.other' instances were found to analyze.")
        return



if __name__ == '__main__':

    truckscenes = TruckScenes(version='v1.0-trainval',
                              dataroot='/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval',
                              verbose=True)

    #truckscenes = TruckScenes(version='v1.0-test',
     #                              dataroot='/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test',
      #                             verbose=True)
    main(truckscenes, testset=False)

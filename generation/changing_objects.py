#from tkinter import Image
from PIL import Image
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box, LidarPointCloud
import numpy as np
from pyquaternion import Quaternion
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import open3d as o3d
import os.path as osp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from truckscenes.utils.geometry_utils import transform_matrix, points_in_box
from collections import defaultdict
import os
import cv2

def visualize_pointcloud_bbox(points: np.ndarray,
                              gt_boxes, # Use List[Box] if Box class is imported
                              colors: Optional[Union[np.ndarray, str]] = None,
                              title: str = "Point Cloud with BBoxes"):
    """
    Visualize a point cloud and optional bounding boxes using Open3D.

    Args:
        points: Nx3 or Nx(>3) numpy array of XYZ[+label/feature].
        boxes: List of Box objects (e.g., from truckscenes) in the same coordinate frame as points.
               Assumes Box objects have .center, .wlh, and .orientation (pyquaternion.Quaternion) attributes.
        colors: Optional Nx3 RGB array or a string-based colormap (e.g., "label").
                If "label", assumes the 4th column of `points` contains integer labels.
        title: Optional window title.
    """
    geometries = []

    # --- Point cloud ---
    if points.ndim != 2 or points.shape[1] < 3:
        print(f"Error: Invalid point cloud shape: {points.shape}. Needs Nx(>=3).")
        return
    if points.shape[0] == 0:
        print("Warning: Point cloud is empty.")
        # Continue to potentially draw boxes

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # --- Point Cloud Coloring ---
    use_label_coloring = False
    if colors is not None:
        if isinstance(colors, str) and colors.lower() == "label":
            if points.shape[1] > 3:
                use_label_coloring = True
            else:
                print("Warning: 'colors' set to 'label' but points array has < 4 columns.")
        elif isinstance(colors, np.ndarray) and colors.shape == points[:, :3].shape:
             # Ensure colors are float64 and in range [0, 1] for Open3D
             colors_float = colors.astype(np.float64)
             if np.max(colors_float) > 1.0: # Basic check if maybe 0-255 range
                 colors_float /= 255.0
             pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_float, 0.0, 1.0))
        else:
            print(f"Warning: Invalid 'colors' argument. Type: {type(colors)}, Value/Shape: {colors if isinstance(colors, str) else colors.shape}. Using default colors.")
    elif points.shape[1] > 3: # Default to label coloring if 4th column exists and colors=None
         print("Info: No 'colors' provided, attempting to color by 4th column (label).")
         use_label_coloring = True

    if use_label_coloring:
        try:
            labels = points[:, 3].astype(int)
            unique_labels = np.unique(labels)
            if unique_labels.size > 0:
                # Map labels to colors
                min_label = unique_labels.min()
                max_label = unique_labels.max()
                label_range = max_label - min_label + 1
                # Use a colormap suitable for categorical data
                cmap = plt.get_cmap("tab20", label_range)
                # Normalize labels to 0..label_range-1 for colormap indexing
                normalized_labels = labels - min_label
                rgb = cmap(normalized_labels)[:, :3] # Get RGB, ignore alpha
                pcd.colors = o3d.utility.Vector3dVector(rgb)
            else:
                print("Warning: Found label column, but no unique labels detected.")
        except Exception as e:
            print(f"Error applying label coloring: {e}. Using default colors.")
            use_label_coloring = False # Revert if error occurred


    geometries.append(pcd)

    # --- Bounding boxes ---
    num_boxes_drawn = 0
    if gt_boxes is not None:
        for i in range(gt_boxes.shape[0]):
            # Create Open3D OrientedBoundingBox from truckscenes Box properties
            center = gt_boxes[i, 0:3] # Should be numpy array (3,)
            # truckscenes Box.wlh = [width(y), length(x), height(z)]
            # o3d OrientedBoundingBox extent = [length(x), width(y), height(z)]
            w, l, h = gt_boxes[i, 3:6]
            yaw = gt_boxes[i, 6]
            # Convert yaw to rotation matrix (yaw around Z axis)
            rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

            bbox = o3d.geometry.OrientedBoundingBox(center, rot_mat, [w, l, h])
            bbox.color = (1.0, 0.0, 0.0)  # Set color to red
            geometries.append(bbox)
            num_boxes_drawn += 1

    # --- Visualize ---
    if not geometries:
        print("No geometries (point cloud or boxes) to visualize.")
        return

    point_count = np.asarray(pcd.points).shape[0]
    print(f"Visualizing point cloud with {point_count} points and {num_boxes_drawn} boxes.")
    o3d.visualization.draw_geometries(geometries, window_name=title)

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

def transform_matrix_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Arguments:
        x: (n,)
        xp: (m,)
        fp: (4, 4, m)

    Returns:
        y: (4, 4, n)
    """
    # Initialize interpolated transformation matrices
    y = np.repeat(np.eye(4, dtype=fp.dtype)[..., None], x.size, axis=-1)

    # Split homogeneous transformation matrices in rotational and translational part
    rot = fp[:3, :3, :]
    trans = fp[:3, 3, :]

    # Get interpolated rotation matrices
    slerp = Slerp(xp, Rotation.from_matrix(np.moveaxis(rot, -1, 0)))
    y[:3, :3, :] = np.moveaxis(slerp(x).as_matrix(), 0, -1)

    # Get interpolated translation vectors
    y[:3, 3, :] = np.vstack((
        interp1d(xp, trans[0, :])(x),
        interp1d(xp, trans[1, :])(x),
        interp1d(xp, trans[2, :])(x),
    ))

    return y


def transform_pointwise(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Retruns a transformed point cloud

    Point cloud transformation with a transformation matrix for each point.

    Arguments:
        points: Point cloud with dimensions (3, n).
        transforms: Homogeneous transformation matrices with dimesnion (4, 4, n).

    Retruns:
        points: Transformed point cloud with dimension (3, n).
    """
    # Add extra dimesnion to points (3, n) -> (4, n)
    points = np.vstack((points[:3, :], np.ones(points.shape[1], dtype=points.dtype)))

    # Point cloud transformation as 3D dot product
    # T@P^T with dimensions (n, 4, 4) x (n, 1, 4) -> (n, 1, 4)
    points = np.einsum('nij,nkj->nki', np.moveaxis(transforms, -1, 0), points.T[:, None, :])

    # Remove extra dimensions (n, 1, 4) -> (n, 3)
    points = np.squeeze(points)[:, :3]

    return points.T

def get_pointwise_fused_pointcloud(trucksc: TruckScenes, sample: Dict[str, Any], allowed_sensors: List[str]) -> LidarPointCloud:
    """ Returns a fused lidar point cloud for the given sample.

    Fuses the point clouds of the given sample and returns them in the ego
    vehicle frame at the timestamp of the given sample. Uses the timestamps
    of the individual point clouds to transform them to a uniformed frame.

    Does not consider the timestamps of the individual points during the
    fusion.

    Arguments:
        trucksc: TruckScenes dataset instance.
        sample: Reference sample to fuse the point clouds of.

    Returns:
        fused_point_cloud: Fused lidar point cloud in the ego vehicle frame at the
            timestamp of the sample.
    """
    # Initialize
    points = np.zeros((LidarPointCloud.nbr_dims(), 0), dtype=np.float64)
    timestamps = np.zeros((1, 0), dtype=np.uint64)
    fused_point_cloud = LidarPointCloud(points, timestamps)

    # Get reference ego pose (timestamp of the sample/annotations)
    ref_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])

    # Homogeneous transformation matrix from global to ref ego car frame.
    car_from_global = transform_matrix(ref_ego_pose['translation'],
                                       Quaternion(ref_ego_pose['rotation']),
                                       inverse=True)

    # Iterate over all lidar sensors and fuse their point clouds
    for sensor in sample['data'].keys():
        if sensor not in allowed_sensors:
            print(f"Skipping sensor {sensor} as it is not allowed.")
            continue
        if 'lidar' not in sensor.lower():
            continue

        # Aggregate current and previous sweeps.
        sd = trucksc.get('sample_data', sample['data'][sensor])

        # Load pointcloud
        pc = LidarPointCloud.from_file(osp.join(trucksc.dataroot, sd['filename']))

        # Get ego pose for the first and last point of the point cloud
        t_min = np.min(pc.timestamps)
        t_max = np.max(pc.timestamps)
        ego_pose_t_min = trucksc.getclosest('ego_pose', t_min)
        ego_pose_t_max = trucksc.getclosest('ego_pose', t_max)

        # Homogeneous transformation matrix from ego car frame to global frame.
        global_from_car_t_min = transform_matrix(ego_pose_t_min['translation'],
                                                 Quaternion(ego_pose_t_min['rotation']),
                                                 inverse=False)

        global_from_car_t_max = transform_matrix(ego_pose_t_max['translation'],
                                                 Quaternion(ego_pose_t_max['rotation']),
                                                 inverse=False)

        globals_from_car = transform_matrix_interp(x=np.squeeze(pc.timestamps),
                                                   xp=np.stack((t_min, t_max)),
                                                   fp=np.dstack((global_from_car_t_min, global_from_car_t_max)))

        # Get sensor calibration information
        cs = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        car_from_current = transform_matrix(cs['translation'],
                                            Quaternion(cs['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        pc.transform(car_from_current)

        pc.points[:3, :] = transform_pointwise(pc.points[:3, :], globals_from_car)

        pc.transform(car_from_global)

        # Merge with key pc.
        fused_point_cloud.points = np.hstack((fused_point_cloud.points, pc.points))
        if pc.timestamps is not None:
            fused_point_cloud.timestamps = np.hstack((fused_point_cloud.timestamps, pc.timestamps))

    return fused_point_cloud

def track_box_dimensions(trucksc: 'TruckScenes', scene_index: int = 0, precision: int = 4):
    """
    Tracks the dimensions of object bounding boxes across all samples in a scene
    and reports if they change for any object instance.

    Args:
        trucksc: Initialized TruckScenes dataset instance.
        scene_index: Index of the scene to analyze.
        precision: Number of decimal places to round dimensions to for comparison.
                   This helps ignore tiny floating-point variations.
    """
    if not trucksc.scene or scene_index >= len(trucksc.scene):
        print(f"Error: Scene index {scene_index} is out of bounds (max: {len(trucksc.scene)-1}).")
        return

    my_scene = trucksc.scene[scene_index]
    scene_name = my_scene['name']
    print(f"\n--- Analyzing Scene: '{scene_name}' (Index: {scene_index}) for Dimension Changes ---")

    # Use defaultdict for easier handling of new instance tokens
    # Stores {instance_token: list of (timestamp, (w, l, h))}
    tracked_dimensions = defaultdict(list)

    current_sample_token = my_scene.get('first_sample_token')
    sample_count = 0

    if not current_sample_token:
        print(f"Scene '{scene_name}' has no samples.")
        return

    while current_sample_token:
        my_sample = trucksc.get('sample', current_sample_token)
        timestamp = my_sample['timestamp']
        sample_count += 1
        # print(f"Processing Sample {sample_count}, Token: {current_sample_token}, Timestamp: {timestamp}") # Optional: progress indicator

        # Get boxes (already transformed to ego frame by get_boxes, but wlh is intrinsic)
        boxes = get_boxes(trucksc, my_sample) # Use the provided get_boxes function

        if not boxes:
            # Move to the next sample if no boxes found
            print("No boxes found. Continuing...")
            current_sample_token = my_sample.get('next', '')
            continue

        for box in boxes:
            try:
                # box.token is the sample_annotation_token
                sample_annotation = trucksc.get('sample_annotation', box.token)
                instance_token = sample_annotation['instance_token']

                # Get dimensions and round them to handle potential float inaccuracies
                # box.wlh = [width(y), length(x), height(z)]
                dimensions = tuple(np.round(box.wlh, precision))

                # Store timestamp and dimensions for this instance
                tracked_dimensions[instance_token].append((timestamp, dimensions))

            except KeyError:
                 # This might happen if an annotation token is somehow invalid
                 print(f"Warning: Could not find sample_annotation or instance_token for box token {getattr(box, 'token', 'N/A')} in sample {current_sample_token}")
            except Exception as e:
                 print(f"Warning: Error processing box token {getattr(box, 'token', 'N/A')} in sample {current_sample_token}: {e}")


        # Move to the next sample
        current_sample_token = my_sample.get('next', '')
        if not current_sample_token:
             print(f"Processed {sample_count} samples. Reached end of scene '{scene_name}'.") # Optional: end of scene message
             break # Explicit break

    # --- Analysis ---
    print(f"\n--- Dimension Analysis Results for Scene '{scene_name}' ---")
    changed_count = 0
    constant_count = 0
    instances_analyzed = 0

    if not tracked_dimensions:
        print("No object instances with annotations found in this scene.")
        return

    for instance_token, history in tracked_dimensions.items():
        instances_analyzed += 1
        # Get unique dimension tuples observed for this instance
        # We stored rounded tuples, so set comparison works well
        unique_dims = set(dims for ts, dims in history)

        if len(unique_dims) > 1:
            changed_count += 1
            print(f"\n[!] Dimensions CHANGED for Instance Token: {instance_token}")
            # Sort history by timestamp for clarity before printing
            history.sort(key=lambda item: item[0])
            last_dims = None
            for ts, dims in history:
                 # Print only when dimensions change compared to the last printed entry for this instance
                 if dims != last_dims:
                    print(f"  - Timestamp: {ts}, Dimensions (W, L, H): {dims}")
                    last_dims = dims
        else:
            constant_count += 1
            # Optional: Print info about constant boxes if needed
            # if history: # Check if history is not empty
            #    first_ts, first_dims = history[0]
            #    print(f"\n[OK] Dimensions CONSTANT for Instance Token: {instance_token}")
            #    print(f"   Dimensions (W, L, H): {first_dims} (Observed across {len(history)} frames)")


    print("\n--- Summary ---")
    print(f"Total unique object instances analyzed: {instances_analyzed}")
    print(f"Instances with changing dimensions:     {changed_count}")
    print(f"Instances with constant dimensions:   {constant_count}")
    print("---------------\n")

    return changed_count


def get_images(trucksc, my_sample, sensor):
    my_sample_data = my_sample['data']
    cam_left_front_data = trucksc.get('sample_data', my_sample_data[sensor])
    print(my_sample_data)
    print(cam_left_front_data)
    img_path_left_front = cam_left_front_data['filename']

    img = Image.open(os.path.join(trucksc.dataroot, img_path_left_front))

    return img


def main(trucksc, testset=False):
    sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR']
    my_scene = trucksc.scene[0]
    scene_name = my_scene['name']
    print(f"\n--- Scene '{scene_name}' ---")

    first_sample_token = my_scene['first_sample_token']
    my_sample = trucksc.get('sample', first_sample_token)
    print(my_sample)

    sample_idx = 0
    while True:
        boxes = get_boxes(trucksc, my_sample)

        print(f"Found {len(boxes)} boxes in ego frame via get_boxes.")

        print("--- Box Dimensions ---")
        if boxes:  # Check if boxes list is not empty before iterating
            for i, box in enumerate(boxes):
                token = getattr(box, 'token', 'N/A')  # Use getattr for safety
                wlh = getattr(box, 'wlh', [np.nan, np.nan, np.nan])  # Use getattr for safety
                # Format the output clearly. Note: box.wlh = [width(y), length(x), height(z)]
                category = getattr(box, 'name', 'N/A')
                print(f"  Box {i + 1}: Token={token} | Category={category} | Dimensions (W, L, H)=[{wlh[0]:.2f}, {wlh[1]:.2f}, {wlh[2]:.2f}]")
        else:
            print("  No valid boxes to display dimensions for in this sample.")
        print("----------------------")

        if not boxes and not testset:
            print(
                "No boxes found or processed successfully. Skipping attribute extraction and visualization for this sample.")
            # Proceed to the next sample
            next_sample_token = my_sample.get('next', '')  # Use .get for safety
            if next_sample_token:
                my_sample = trucksc.get('sample', next_sample_token)
                sample_idx += 1
            else:
                print("End of scene reached.")
                break  # Exit the while loop
            continue  # Skip the rest of the loop for this sample

        # Extract object tokens. Each instance token represents a unique object
        boxes_token = [box.token for box in boxes]  # retrieves a list of tokens from the bounding box
        # Extract object tokens. Each instance token represents a unique object
        object_tokens = [truckscenes.get('sample_annotation', box_token)['instance_token'] for box_token in
                         boxes_token]  # Uses sample_annotation data to get instance_token fore each bb

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1,
                                                           3)  # gets center coordinates (x,y,z) of each bb
        dims = np.array([b.wlh for b in boxes]).reshape(-1,
                                                        3)  # extract dimension width, length, height of each bb
        rots = np.array([b.orientation.yaw_pitch_roll[0]  # extract rotations (yaw angles)
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(
            np.float32)  # combines location, dimensions and rotation into a 2D array

        gt_bbox_3d[:, 6] +=  np.pi / 2.  # adjust yaw angles by 90 degrees
        # gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        # gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.05  # Experiment
        #gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        # gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.05 # Experiment
        #gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1  # Slightly expand the bbox to wrap all object points

        sensor_fused_pc = get_pointwise_fused_pointcloud(trucksc, my_sample, allowed_sensors=sensors)

        print(sensor_fused_pc.points.shape)

        pc_fused_ego = sensor_fused_pc.points.T

        camera = 'CAMERA_RIGHT_FRONT'

        images = get_images(trucksc, my_sample, camera)

        plt.figure(figsize=(12, 8))  # Size in inches (12x8 inches, adjust as needed)
        plt.imshow(images)
        plt.title(f"Left Front Camera Image {sample_idx}", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        camera = 'CAMERA_RIGHT_BACK'
        #camera = 'CAMERA_RIGHT_FRONT'
        images_2 = get_images(trucksc, my_sample, camera)

        plt.figure(figsize=(12, 8))  # Size in inches (12x8 inches, adjust as needed)
        plt.imshow(images_2)
        # plt.title("Left Back Camera Image", fontsize=16)
        # plt.title("Right Back Camera Image", fontsize=16)
        plt.axis('off')
        plt.title(f"Right Front Camera Image {sample_idx}", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        visualize_pointcloud_bbox(pc_fused_ego, gt_boxes=gt_bbox_3d,
                                  title=f"Fused filtered static sensor PC + BBoxes - Frame {sample_idx}")


        next_sample_token = my_sample['next']
        if next_sample_token != '':
            my_sample = trucksc.get('sample', next_sample_token)
            sample_idx += 1
        else:
            break


if __name__ == '__main__':

    truckscenes = TruckScenes(version='v1.0-trainval',
                              dataroot='/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval',
                              verbose=True)

    #truckscenes = TruckScenes(version='v1.0-test',
     #                              dataroot='/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test',
      #                             verbose=True)
    main(truckscenes, testset=True)

    changed_count = 0
    changed_scenes_list = []
    for i in range(0, 597):
        changed_count = track_box_dimensions(trucksc=truckscenes, scene_index=i)
        if changed_count:
            changed_scenes_list.append(i)

    print(f"Changed {len(changed_scenes_list)} scenes.")
    print("Changed object size in scene:", changed_scenes_list)

    #main(truckscenes)
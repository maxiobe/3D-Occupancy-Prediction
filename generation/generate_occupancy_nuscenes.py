import os
import sys
import pdb
import time
import yaml
import torch
import chamfer
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
# from mmdet3d.core.bbox import box_np_ops
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation

import open3d
import open3d as o3d
from copy import deepcopy

# Function to perform poisson surface reconstruction on a given point cloud and returns a mesh representation of the point cloud, along with vertex info
# Inputs pcd: input point cloud,
    # depth: parameter to control resolution of mesh. Higher depth results in a more detailed mesh but requires more computation
    # n_threads: Number of threads for parallel processing
    # min_density: threshold for removing low-density vertices from generated mesh. Used to clean up noisy or sparse areas
def run_poisson(pcd, depth, n_threads, min_density=None):
    # creates triangular mesh form pcd using poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )
    # returns mesh and densities: list of density values corresponding to each vertex in the mesh. Density indicates how well a vertex is supported by underlying points

    # Post-process the mesh
    # Purpose: to clean up the mesh by removing low-density vertices (e.g. noise or poorly supported areas)
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density) # min_density should be between 0 and 1
        mesh.remove_vertices_by_mask(vertices_to_remove) # removes vertices where density is below threshold
    mesh.compute_vertex_normals() # computes the normals of the vertices

    return mesh, densities

# Function that creates a 3D mesh from a given point cloud or a buffer of points
# Inputs buffer: a list of point clouds that are combined if no original is given
# depth: resolution for poisson surface reconstruction
# n_threads: Number of threads for parallel processing
# min_density: Optional threshold for removing low-density vertices
# point_cloud_original: provides a preprocessed point cloud, if given buffer is ignored
def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer) # Calls buffer_to_pointcloud(buffer) to create a combined point cloud from the list of smaller point clouds
    else:
        pcd = point_cloud_original # Uses the given point cloud directly

    return run_poisson(pcd, depth, n_threads, min_density) # calls run_poisson function to generate mesh

# Function to combine multiple point clouds from a list (buffer) into a single point cloud and optionally estimating normals in the resulting point cloud
# Input: buffer: a list of individual point clouds (each being an instance of open3d.geometry.PointCloud)
# compute_normals: boolean flag that if set to True estimates normals of the final point cloud
def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud() # Initialize empty point cloud object using Open3d
    for cloud in buffer:
        pcd += cloud # combine each point cloud with current point cloud object
    if compute_normals:
        pcd.estimate_normals() # estimate normals for each point in the combined point cloud

    return pcd

# Function to preprocess a given point cloud by estimating and orienting the normals of each point
# Input: pcd: input point cloud
# max_nn: maximum number of nearest neighbors to use when estimating normals with default 20
# normals: boolean flag whether to estimate normals
def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):

    cloud = deepcopy(pcd) # create a deep copy of the input point cloud to ensure that original point cloud is not modified
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params) # estimate normals based on nearest max_nn points
        cloud.orient_normals_towards_camera_location() # orients all computed normals to point towards the camera location to get consistent normal direction for visualization and mesh generation

    return cloud

# Wrapper function that calls preprocess_cloud function with parameters derived from a config file
# Input: pcd: input point cloud
# config: Dictionary containing all configuration parameters loaded from a YAML file
def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )

# Function to find the nearest neighbor (NN) in one set of 3D points (verts1) for each point in another set of 3D points (verts2)
# Input: verts1: numpy array of shape (n, 3) representing n 3D points
# verts2: numpy array of shape (m, 3) representing m 3D points
def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
    import open3d as o3d

    indices = []
    distances = []

    # checks if either of the input point sets is empty and returns empty lists if so
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud() # creates open3d point cloud object
    pcd.points = o3d.utility.Vector3dVector(verts1) # Converts input numpy array verts1 to an open3d Vector3dVector which allows to efficiently handle point data
    kdtree = o3d.geometry.KDTreeFlann(pcd) # Builds a KD-Tree using the point cloud. KD-Tree is a data-structure optimized for nearest neighbor search in high-dimensional space

    # Nearest neighbor search by looping through each point in verts2 and searches the 1 nearest neighbor in verts1
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1) # vert: current point in verts2, 1: number of nearest neighbors to find, dist: list of squared distances to nn
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0])) # appends euclidean distance by taking the square root of the returned squared distance

    # Returns two lists:
    # 1. A list of nearest neighbor indices from verts1 corresponding to each point in verts2.
    # 2. A list of distances between each point in verts2 and its nearest point in verts1.
    return indices, distances

# Function to transform a LiDAR point cloud from one coordinate to another. Essential when dealing with multiple sensors or coordinate frames.
# Input: pc: point cloud represented as a numpy arry of shape (n,3) where N is the number of points
# lidar_calibrated_sensor: dictionary containing the rotation and translation of the LiDAR sensor in the vehicle coordinate system
# lidar_ego_pose: Dictionary containing the rotation and translation of the ego vehicle pose
# cam_calibrated_sensor: Dictionary containing the rotation and translation of the camera sensor in the vehicle coordinate system
# cam_ego_pose: Dictionary containing the rotation and translation of the ego vehicle pose for the camera
def lidar_to_world_to_lidar(pc,lidar_calibrated_sensor,lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose):

    # LiDAR coordinate system -> vehicle coordinate system -> world coordinate system -> vehicle coordinate system -> reference coordinate system (e.g. first lidar frame)

    pc = LidarPointCloud(pc.T) # Converts input point cloud pc to a LidarPointCloud object, uses transpose because input might be in form (N, 3) while function expects (3, N)
    # Rotate the LiDAR point cloud to align with the LiDAR sensor's rotation
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix) # use a quarternion rotation to align the point cloud with the sensor's orientation, extracts rotation matrix from quaternion
    # Translate to the LiDAR sensor position
    pc.translate(np.array(lidar_calibrated_sensor['translation'])) # applies a translation vector to move the point cloud to the LiDAR sensor position

    # Transform Points from LiDAR to Ego Vehicle Coordinate System
    # Rotate to align with the vehicle pose
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix) # transforms the point cloud to the vehicle's coordinate system using ego pose rotation
    # Translate to the vehicle position
    pc.translate(np.array(lidar_ego_pose['translation'])) # applies a translation to the ego vehicle position

    # Transform Points from ego vehicle to camera coordinate system
    # Translate to move the point cloud to the camera pose
    pc.translate(-np.array(cam_ego_pose['translation'])) # uses negative translation to move the point cloud back from the camera's position
    # Rotate to align with the camera's coordinate system
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T) # uses transpose of the rotation matrix to undo the rotation applied earlier
    # Moves point cloud from ego vehicle coordinate system to camera coordinate system

    # Translate to remove the camera calibration offset
    pc.translate(-np.array(cam_calibrated_sensor['translation'])) # uses negative translation to reverse the camera calibration translation
    # Rotate to reverse camera calibration rotation:
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T) # uses the inverse rotation matrix to undo the camera's calibration rotation

    return pc

# main function as entry point for processing a specific scene from the nuScenes dataset
# Input: nusc: a nuScenes object representing the dataset instance
# val_list: a list of scene tokens that are part of the validation split
# indice: index of the scene to be processed
# nuscenesyaml: A dictionary loaded from a YAML configuration file specific for nuScenes
# args: Parsed command-line arguments
# config: a dictionary containing configuration settings
def main(nusc, val_list, indice, nuscenesyaml, args, config):

    # Extract necessary parameters from the arguments and configs
    save_path = args.save_path # Directory where processed data will be saved
    data_root = args.dataroot # Root directory of dataset
    learning_map = nuscenesyaml['learning_map'] # dictionary that maps raw semantic labels to learning labels
    voxel_size = config['voxel_size'] # Size of each voxel in the occupancy grid
    pc_range = config['pc_range'] # Range of point cloud coordinates to consider (bounding box)
    occ_size = config['occ_size'] # Dimensions of the output occupancy grid

    # Retrieves a specific scene from the nuScenes dataset
    my_scene = nusc.scene[indice] # scene is selected by indice parameter
    scene_name = my_scene['name'] ### Extract scene name for saving
    sensor = 'LIDAR_TOP' # Specifies the LiDAR sensor name

    # Data split handling
    if args.split == 'train':
        if my_scene['token'] in val_list: # Ensures that no validation data is mistakenly used in training
            return
    elif args.split == 'val':
        if my_scene['token'] not in val_list: # Ensures that no training data is used during validation
            return
    elif args.split == 'all': # Proceeds without filtering, useful for generating predictions or evaluations on the entire dataset
        pass
    else: # Error if split type is not recognized
        raise NotImplementedError

    # load the first sample from a scene to start
    first_sample_token = my_scene['first_sample_token'] # access the first sample token: contains token of first frame of the scene
    my_sample = nusc.get('sample', first_sample_token) # retrieve the first sample as dictionary. Dictionary includes data from multiple sensors
    lidar_data = nusc.get('sample_data', my_sample['data'][sensor]) # retrieves LiDAR sample data using the specific sensor 'LIDAR_TOP', my_sample['data'][sensor] retrieves token associated with sensor
    # Lidar_data contains sample_token, channel, ego_pose_token, calibrated_sensor_token, filename ...

    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token']) # use the ego pose token from lidar_data to get pose information of ego vehicle when LiDAR data was captured
    # lidar_ego_pose0 contains translation, rotation, timestamp

    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token']) # use calibrated sensor token to load sensor calibration data
    # lidar_calibrated_sensor0 contains sensor_token, translation, rotation, camera_intrinsics

    # collect LiDAR sequence
    dict_list = []

    # loop that processes each sample (frame) in a scene until break
    # Performs several operations:
    # 1. extract and process bounding boxes (objects)
    # 2. convert object categories into a suitable format
    # 3. extract and preprocess LiDAR point cloud data
    # 4. load semantic annotations if available
    while True:
        ############################# get boxes ##########################
        # Loading sample data
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token']) # retrieve lidar_path and boxes
        # Extract bounding box tokens
        boxes_token = [box.token for box in boxes] # retrieves a list of tokens from the bounding box
        # Extract object tokens. Each instance token represents a unique object
        object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token] # Uses sample_annotation data to get instance_token fore each bb
        # Extract object categories
        object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token] # retrieves category name for each bounding box

        ############################# get object categories ##########################
        converted_object_category = [] # Initialize empty list to store converted category labels
        # Iterate over each object category extracted earlier
        for category in object_category:
            # Iterate over label mappings defined in nuscenes.yaml file
            for (j, label) in enumerate(nuscenesyaml['labels']):
                # Check category and map to learning label
                if category == nuscenesyaml['labels'][label]:
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1, 3) # gets center coordinates (x,y,z) of each bb
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3) # extract dimension width, length, height of each bb
        rots = np.array([b.orientation.yaw_pitch_roll[0] # extract rotations (yaw angles)
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32) # combines location, dimensions and rotation into a 2D array
        gt_bbox_3d[:, 6] += np.pi / 2. # adjust yaw angles by 90 degrees
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1  # Slightly expand the bbox to wrap all object points

        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data['filename']  # load LiDAR names
        # Load LiDAR Point Cloud Data
        pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
                          dtype=np.float32,
                          count=-1).reshape(-1, 5)[..., :4] # Keeps only first four columns (x, y, z, intensity)

        # Load semantic annotations if key frame
        if lidar_data['is_key_frame']:  # only key frame has semantic annotations
            sample_token = nusc.get('sample_data', lidar_data['token'])['sample_token']
            # print(sample_token)
            lidar_sd_token = lidar_data['token'] # get token of LiDAR sample data, token is used to locate corresponding semantic label
            lidarseg_labels_filename = os.path.join(nusc.dataroot, # Construct path to LiDAR Semantic Annotations file
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])

            # Load semantic labels from binary file
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1]) # Reads entire binary file as 1-dimensional array of 8-bit unsigned integers
            # each element in the array corresponds to a semantic label for a point in the LiDAR point cloud,
            # convert flat array into a 2D array with one column where each row corresponds to a point's semantic label
            # If N points in the point cloud, the shape will be (N, 1)

            # Map semantic labels to learning class (to match the label format required for training)
            # learning_map is a dictionary that maps raw labels to learning labels
            points_label = np.vectorize(learning_map.__getitem__)(points_label) # np.vectorize() to apply a function element-wise to an array

            # Combine points with semantic labels
            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1) # Combines point coordinates with semantic labels to form a 4D array: (x, y, z, semantic_label)

        ############################# cut out movable object points and masks ##########################
        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                              torch.from_numpy(gt_bbox_3d[np.newaxis, :])) # use function to identify which points belong to which bounding box
        # pc0[:, :3]: Extracts the x, y, z coordinates from the LiDAR point cloud
        # gt_box_3d: Array containing 3d bounding box attributes (location, size, rotation)
        # both converted to Torch tensors and reshaped with additional dimension using np.newaxis to make the shapes compatible
        # points_in_boxes: Output is a tensor of shape (1, num_boxes, numb_points), where: each element is 1 if the point is inside a bounding box and 0 otherwise

        object_points_list = [] # creates an empty list to store points associated with each object
        j = 0
        # Iterate through each bounding box along the last dimension
        while j < points_in_boxes.shape[-1]:
            # Create a boolean mask indicating whether each point belongs to the current bounding box.
            object_points_mask = points_in_boxes[0][:, j].bool()
            # Extract points using mask to filter points
            object_points = pc0[object_points_mask]
            #Store the filtered points, Result is a list of arrays, where each element contains the points belonging to a particular object
            object_points_list.append(object_points)
            j = j + 1


        # Create moving mask of ones with the same shape as points_in_boxes
        moving_mask = torch.ones_like(points_in_boxes) # mask is used to aggregate all points that belong to any bounding box
        # Combine Masks of all bounding boxes
        points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool() # Sums masks across all bounding boxes to get a composite mask
        # result is a boolean tensor of shape (1, num_points), each element is true if the point belongs to any object and false otherwise

        # Invert the mask to get static points
        points_mask = ~(points_in_boxes[0]) # negates the boolean tensor to identify points that do not belong to any bounding box -> static background points

        ############################# get point mask of the vehicle itself ##########################
        # goal here is to filter out points that belong to the vehicle itself
        range = config['self_range'] # Parameter in config file that specifies a range threshold for the vehicle's own points

        # Mask calculation: filters out points that are too close to the vehicle in x, y or z directions
        oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > range[0]) |
                                        (np.abs(pc0[:, 1]) > range[1]) |
                                        (np.abs(pc0[:, 2]) > range[2]))

        ############################# get static scene segment ##########################
        # Combine background mask and the self-filter mask using a logical AND
        # Ensures that the final mask excludes both moving objects and the vehicle itself
        points_mask = points_mask & oneself_mask
        # Extract static points
        pc = pc0[points_mask] # uses final mask to filter the static points from the original point cloud

        ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
        # Get current frame's pose and calibration data
        lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        # Coordinate Transformation
        # transforms the point cloud to the same (first) LiDAR coordinate system
        # transformation is necessary because frames in a sequence have different poses
        # function aligns the current point cloud to the reference frame of the first LiDAR position
        lidar_pc = lidar_to_world_to_lidar(pc.copy(), lidar_calibrated_sensor.copy(), lidar_ego_pose.copy(),
                                           lidar_calibrated_sensor0,
                                           lidar_ego_pose0)

        ################## record Non-key frame information into a dict  ########################
        dict = {"object_tokens": object_tokens, # list of tokens that identify objects detected in current frame
                "object_points_list": object_points_list, # list of point clouds, where each entry contains the points associated with a specific object
                "lidar_pc": lidar_pc.points, # contains transformed point cloud data (x, y, z coordinates) after processing in common coordinate system (first LiDAR frame)
                "lidar_ego_pose": lidar_ego_pose, # stores pose information of current LiDAR sensor
                "lidar_calibrated_sensor": lidar_calibrated_sensor, # stores calibration information of current LiDAR sensor
                "lidar_token": lidar_data['token'], # unique identifier of the current LiDAR sample
                "is_key_frame": lidar_data['is_key_frame'], # boolean flag indicating if key frame
                "gt_bbox_3d": gt_bbox_3d, # list of bounding box coordinates for all detected objects in the frame in format [x, y, z, width, legth, height, yaw]
                "converted_object_category": converted_object_category, # stores learning friendly object categories
                "pc_file_name": pc_file_name.split('/')[-1],
                } # extracts file name from the full path

        ################## record semantic information into the dict if it's a key frame  ########################
        if lidar_data['is_key_frame']: # check if keyframe
            dict["sample_token"] = sample_token
            # Apply point mask
            pc_with_semantic = pc_with_semantic[points_mask] # filters out points based on computed mask, retaining only static points (excluding dynamic objects and the vehicle itself)
            # Transform points to the first LiDAR Coordinate System
            lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
                                                             lidar_calibrated_sensor.copy(),
                                                             lidar_ego_pose.copy(),
                                                             lidar_calibrated_sensor0,
                                                             lidar_ego_pose0)
            # Store Transformed Semantic Points
            dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points # saves the transformed semantic point cloud into the dictionary

        # append the dictionary to list
        dict_list.append(dict) # appends dictionary containing frame data to the list dict_list
        # After iterating through the entire scene, this list will contain information for all frames in the scene

        ################## go to next frame of the sequence  ########################
        next_token = lidar_data['next'] # get token for the next frame from the current LiDAR data, if no next frame (end of scene) next_token is empty string
        if next_token != '':  # Load next frame data if next frame exists
            lidar_data = nusc.get('sample_data', next_token)
        else:
            break # break if end of scene is reached

    ################## concatenate all static scene segments (including non-key frames)  ########################
    lidar_pc_list = [dict['lidar_pc'] for dict in dict_list] # iterate through the list of dictionaries, extracts the static scene point cloud (lidar_pc) from each frame's dictionary
    lidar_pc = np.concatenate(lidar_pc_list, axis=1).T # Merge list of point clouds along the point dimension (axis=1). Transpose to switch from (3, N) to (N, 3)

    ################## concatenate all semantic scene segments (only key frames)  ########################
    lidar_pc_with_semantic_list = [] # initialize a list to hold semantic point clouds from key frames
    for dict in dict_list:  # loops through list of dictionaries
        if dict['is_key_frame']:
            lidar_pc_with_semantic_list.append(dict['lidar_pc_with_semantic']) # only appends semantic point clouds from key frames to the list
    lidar_pc_with_semantic = np.concatenate(lidar_pc_with_semantic_list, axis=1).T # concatenate semantic points

    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = [] # stores unique object tokens from all frames
    object_semantic = [] # stores semantic category corresponding to each unique object
    for dict in dict_list: # Iterate through frames and collect unique objects
        for i, object_token in enumerate(dict['object_tokens']):
            if object_token not in object_token_zoo: # Filter and append object tokens
                if (dict['object_points_list'][i].shape[0] > 0): # only appends objects that have at least one point
                    object_token_zoo.append(object_token)
                    object_semantic.append(dict['converted_object_category'][i])
                else:
                    continue

    # Aggregate object points
    object_points_dict = {} # initialize an empty dictionary to hold aggregated object points for each unique object token

    for query_object_token in object_token_zoo: # Loop through each unique object token
        object_points_dict[query_object_token] = [] # initializes an empty list for each token to store points from different frames
        for dict in dict_list: # iterates through all frames
            for i, object_token in enumerate(dict['object_tokens']):
                if query_object_token == object_token: # find matching object tokens
                    object_points = dict['object_points_list'][i] # retrieve object points
                    if object_points.shape[0] > 0:
                        object_points = object_points[:, :3] - dict['gt_bbox_3d'][i][:3] # translates the object points to the center of the bounding box
                        # Rotate points to align with BBox orientation
                        rots = dict['gt_bbox_3d'][i][6]
                        Rot = Rotation.from_euler('z', -rots, degrees=False)
                        rotated_object_points = Rot.apply(object_points) # uses yaw angle from bounding box to align the points
                        object_points_dict[query_object_token].append(rotated_object_points) # aggregate transformed points
                else:
                    continue
        object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                axis=0) # Concatenate points across multiple frames to form a single object point cloud

    # Aggregate object point vertices
    object_points_vertice = [] # initialize empty list to store object points from multiple frames
    for key in object_points_dict.keys(): # iterates through each object token in object_points_dict
        point_cloud = object_points_dict[key] # retrieves the aggregated points for a given object token
        object_points_vertice.append(point_cloud[:, :3]) # append only first three coordinates (x, y, z)
    # print('object finish')

    # Loop over frames to process scene
    i = 0
    while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames to avoid infinite loops
        if i >= len(dict_list): # If the frame index exceeds the number of frames in the scene exit the function
            print('finish scene!')
            return
        dict = dict_list[i] # retrieves dictionary corresponding to the current frame
        # Only processes key frames since non-key frames do not have ground truth annotations
        is_key_frame = dict['is_key_frame']
        if not is_key_frame:  # only use key frame as GT
            i = i + 1
            continue # skips to the next frame if not a key frame

        ################## convert the static scene to the target coordinate system ##############
        # retrieve calibration and pose data for current frame
        lidar_calibrated_sensor = dict['lidar_calibrated_sensor']
        lidar_ego_pose = dict['lidar_ego_pose']

        # Transform the static point cloud to the current LiDAR coordinate system, use initial calibration and pose as reference frame
        # Use current frame's calibration and pose to align the points
        lidar_pc_i = lidar_to_world_to_lidar(lidar_pc.copy(),
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)
        # coordinate transformation for the semantic scene, ensures that static and semantic scenes are in the same coordinate frame
        lidar_pc_i_semantic = lidar_to_world_to_lidar(lidar_pc_with_semantic.copy(),
                                                      lidar_calibrated_sensor0.copy(),
                                                      lidar_ego_pose0.copy(),
                                                      lidar_calibrated_sensor,
                                                      lidar_ego_pose)

        # extract points from transformed point clouds
        point_cloud = lidar_pc_i.points.T[:, :3] # extract transformed static points, T to switch dims from (3,N) to (N, 3)
        point_cloud_with_semantic = lidar_pc_i_semantic.points.T # retrieves transformed semantic points with labels (N, 4): [x, y, z, label]

        ################## load bbox of target frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(dict['lidar_token']) # retrieve LiDAR path and bounding boxes
        locs = np.array([b.center for b in boxes]).reshape(-1, 3) # extracts the center coordinates (x, y, z) of each bounding box
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3) # extract dimensions (w, l, h) of bounding boxes
        rots = np.array([b.orientation.yaw_pitch_roll[0] # Extract rotations yaw angle of bounding boxes
                         for b in boxes]).reshape(-1, 1)
        # Combine location, dimensions and rotations to form a single array of shape (N, 7)
        # Columns [0:3] center coordinates, [3:6] dimensions, [6] yaw rotation
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2. # Adjust bb orientation (yaw angle by 90 degrees to match coordinate system convention
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2. # Adjust bounding box height to better fit ground level
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1 # Increase size of bounding box by 10% to ensure that all points are contained
        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################## bbox placement ##############
        object_points_list = [] # initialize list for point clouds
        object_semantic_list = [] # initialize list for semantic data
        for j, object_token in enumerate(dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token == object_token_in_zoo: # check if current object's token is present in object zoo
                    points = object_points_vertice[k]
                    # Rotate and translate object points
                    Rot = Rotation.from_euler('z', rots[j], degrees=False) # uses yaw angle (rots[j] to rotate object's point cloud
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j] # translate the rotated points to bounding box center (locs[j])

                    # Filter Points inside bounding box
                    if points.shape[0] >= 5: # ensure that the object has at least 5 points
                        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                              torch.from_numpy(gt_bbox_3d[j:j + 1][np.newaxis, :])) # check which points are inside bounding box
                        points = points[points_in_boxes[0, :, 0].bool()] # only keep points that fall inside the bounding box

                    # Append Points and Semantics to list
                    object_points_list.append(points) # adds final object points to list
                    semantics = np.ones_like(points[:, 0:1]) * object_semantic[k] # creates semantic label array of same length as num of points
                    object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1)) # Combines 3D coordinates and semantic label

        # Combine Scene Points with Object Points
        try:  # avoid concatenate an empty array
            temp = np.concatenate(object_points_list)
            scene_points = np.concatenate([point_cloud, temp]) # Merge static scene points and object points
        except:
            scene_points = point_cloud # If no object points, only use static scene points
        try:
            temp = np.concatenate(object_semantic_list)
            scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp]) # Merge semantic points from objects and static scenes
        except:
            scene_semantic_points = point_cloud_with_semantic # If no object points, only use static scene points

        ################## remain points with a spatial range ##############
        mask = (np.abs(scene_points[:, 0]) < 50.0) & (np.abs(scene_points[:, 1]) < 50.0) \
               & (scene_points[:, 2] > -5.0) & (scene_points[:, 2] < 3.0) # Generate mask for filtering points in x, y [-50, 50], z [-5, 3]
        scene_points = scene_points[mask] # Filter points within a spatial range

        ################## get mesh via Possion Surface Reconstruction ##############
        point_cloud_original = o3d.geometry.PointCloud() # Initialize point cloud object
        with_normal2 = o3d.geometry.PointCloud() # Initialize point cloud object
        point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3]) # converts scene_points array into open3D point cloud format
        with_normal = preprocess(point_cloud_original, config) # Uses the preprocess function to compute normals
        with_normal2.points = with_normal.points # copies the processed points and normals to another point clouds
        with_normal2.normals = with_normal.normals # copies the processed points and normals to another point clouds

        # Generate a mesh from the point cloud, uses poisson surface reconstruction algorithm to create a 3D mesh
        mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'], # min_density: filters low-density vertices
                                       config['min_density'], with_normal2) # depth: level of details, threads: number of parallel processing
        scene_points = np.asarray(mesh.vertices, dtype=float) # Converts the vertices of the mesh into a numpy array

        ################## remain points with a spatial range ##############
        mask = (np.abs(scene_points[:, 0]) < 50.0) & (np.abs(scene_points[:, 1]) < 50.0) \
               & (scene_points[:, 2] > -5.0) & (scene_points[:, 2] < 3.0) # Generate mask for filtering points in x, y [-50, 50], z [-5, 3]
        scene_points = scene_points[mask] # Filter points within a spatial range

        ################## convert points to voxels ##############
        pcd_np = scene_points
        # Normalize points to voxel grid, transforms 3D points into voxel coordinates
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size # voxel size: controls the granularity of voxel grid
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size # point cloud range (pc_range): defines bounding box
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
        pcd_np = np.floor(pcd_np).astype(np.int) # Round down to nearest integer
        voxel = np.zeros(occ_size) # initialize voxel grid with zeros to represent empty voxels
        voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1 # marks occupied voxels as 1

        ################## convert voxel coordinates to LiDAR system  ##############
        gt_ = voxel
        # Generate grid coordinates (3D grid with voxel coordinates)
        x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
        y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
        z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        #Stack coordinates together to create a 3D arry of coordinate triples
        vv = np.stack([X, Y, Z], axis=-1)
        fov_voxels = vv[gt_ > 0] # Extracts coordinates of occupied voxels

        # convert voxel coordinates back to the LiDAR coordinate system
        fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size # scaling back (multiply voxel coordinates by voxel size), centering voxels (add 0.5 to center of voxel) and adjusting range
        fov_voxels[:, 0] += pc_range[0]
        fov_voxels[:, 1] += pc_range[1]
        fov_voxels[:, 2] += pc_range[2]

        ################## get semantics of sparse points  ##############
        mask = (np.abs(scene_semantic_points[:, 0]) < 50.0) & (np.abs(scene_semantic_points[:, 1]) < 50.0) \
               & (scene_semantic_points[:, 2] > -5.0) & (scene_semantic_points[:, 2] < 3.0) # Generate mask for filtering points in x, y [-50, 50], z [-5, 3]
        scene_semantic_points = scene_semantic_points[mask] # Filter points within a spatial range

        ################## Nearest Neighbor to assign semantics ##############
        dense_voxels = fov_voxels # voxel points that need semantic labels
        sparse_voxels_semantic = scene_semantic_points # Voxel points that already have semantic labels

        # convert dense voxel points to torch tensor
        x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float() # uses cuda to move to a GPU and adds batch dimension by unsqueeze(0)
        # Convert sparse semantic points to torch tensor
        y = torch.from_numpy(sparse_voxels_semantic[:, :3]).cuda().unsqueeze(0).float() # uses cuda to move to GPU and add batch dim
        d1, d2, idx1, idx2 = chamfer.forward(x, y) # Chamfer Distance to calculate nearest neighbors between dense and sparse points
        # d1, d2: distances between nearest neighbors
        # idx11, idx2: Indices of the nearest neighbors

        indices = idx1[0].cpu().numpy() # convert torch tensor indices to a numpy array

        dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)] # uses the nearest neighbor indices to find the corresponding semantic label

        # Concatenates dense voxel coordinates and their assigned semantic labels
        dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

        # to voxel coordinate
        pcd_np = dense_voxels_with_semantic
        # Transforms world coordinates to voxel coordinates by normalizing
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size

        dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int) # Use flooring to convert to discrete voxel indices

        # Save the resulting dense voxels with semantics
        #dirs = os.path.join(save_path, scene_name, dict['lidar_token']) #### Save in folder with scene name
        dirs = os.path.join(save_path, scene_name, dict['sample_token'])
        if not os.path.exists(dirs): # create directory if does not exist
            os.makedirs(dirs)

        save_path_base = dict['pc_file_name'] ### copy file_name to save_path_base
        suffix_to_remove = '.pcd.bin' ### define suffix to remove
        if save_path_base.endswith(suffix_to_remove):
            save_path_base = save_path_base[:-len(suffix_to_remove)] ### Slice off suffix

        output_filepath = os.path.join(dirs, save_path_base + '.npy') ### Generate output filepath
        # Save the dense semantic voxels as a numpy file with a filename corresponding to the frame
        print(f"Saving GT to {output_filepath}...") ####
        np.save(output_filepath, dense_voxels_with_semantic) ### saving point cloud

        i = i + 1
        continue # moves to the next frame for processing

# Function to save point cloud as ply file
def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud() # initialize empty point cloud
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:, :3]) # converts points array into vectorized format for open3d
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original) # Exports point cloud to a PLY file which can be viewed using 3D visualization tools

# Main entry point of the script
if __name__ == '__main__':
    # argument parsing allows users to customize parameters when running the script
    from argparse import ArgumentParser
    parse = ArgumentParser()

    # Define Command-Line Arguments
    parse.add_argument('--dataset', type=str, default='nuscenes') # Dataset selection with default: "nuscenes"
    parse.add_argument('--config_path', type=str, default='config.yaml') # Configuration file path with default: "config.yaml"
    parse.add_argument('--split', type=str, default='train') # data split, default: "train", options: "train", "val", "all"
    parse.add_argument('--save_path', type=str, default='./data/GT_occupancy/') # save path, default: "./data/GT/GT_occupancy"
    parse.add_argument('--start', type=int, default=0) # start indice, default: 0, determines range of sequences to process
    parse.add_argument('--end', type=int, default=850) # end indice, default: 850, determines range of sequences to process
    parse.add_argument('--dataroot', type=str, default='./data/nuScenes/') # data root path, default: "./data/nuScenes/"
    parse.add_argument('--nusc_val_list', type=str, default='./nuscenes_val_list.txt') # text file containing validation scene tokens, default: "./nuscenes_val_list.txt"
    parse.add_argument('--label_mapping', type=str, default='nuscenes.yaml') # YAML file containing label mappings, default: "nuscenes.yaml"
    args=parse.parse_args()


    if args.dataset=='nuscenes': # check dataset type
        val_list = []
        with open(args.nusc_val_list, 'r') as file: # Load validation scene tokens to val_list
            for item in file:
                val_list.append(item[:-1])
        file.close()

        # Load the nuScenes dataset
        nusc = NuScenes(version='v1.0-trainval',
                        dataroot=args.dataroot,
                        verbose=True) # verbose True to print informational messages
        train_scenes = splits.train # train_scenes and val_scenes contain the scene tokens for the training and validation splits
        val_scenes = splits.val
    else:
        raise NotImplementedError

    # load config with hyperparameters and settings
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map to map raw semantic labels to learning labels
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)

    # Process sequences in a loop
    for i in range(args.start,args.end):
        print('processing sequence:', i)
        # call the main function
        # Inputs: nusc: initialized nuScenes dataset object
        # val_list: list of validation scene tokens
        # indice: current scene index
        # nuscenesyaml: loaded label mapping
        # args: parsed command-line arguments
        # config: configuration settings
        main(nusc, val_list, indice=i,
             nuscenesyaml=nuscenesyaml, args=args, config=config)
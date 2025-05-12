import os.path
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud
from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def main(trucksc, args):

    ############## Getting scene and sample data #################
    my_scene = trucksc.scene[args.scene_id]

    first_sample_token = my_scene['first_sample_token']
    my_sample = trucksc.get('sample', first_sample_token)
    sample_data_token = my_sample['data']['LIDAR_LEFT']
    sample_data = trucksc.get('sample_data', sample_data_token)

    #trucksc.render_scene(my_scene['token'])

    filename = sample_data['filename']
    load_path = os.path.join(trucksc.dataroot, filename)

    ############ Load lidar data ########################

    pc = LidarPointCloud.from_file(load_path)
    print("Original point cloud shape:", pc.points.shape)

    xyz = pc.points[:3, :].T  # shape (N,3)
    intensities = pc.points[3, :]

    print(f"Lidar intensity {intensities}")

    ################ Visualize original point cloud ######################

    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.points[:3, :].T)

    # Visualize before and after filtering
    print("Displaying Original Point Cloud...")
    o3d.visualization.draw_geometries([pcd], window_name=f'Original point cloud')

    ############### Filtering based on lidar intensity ####################

    mask = intensities > 0
    xyz_filtered = xyz[mask]

    pcd_intens = o3d.geometry.PointCloud()
    pcd_intens.points = o3d.utility.Vector3dVector(xyz_filtered)

    print(f'Removed {xyz.shape[0] - len(pcd_intens.points)} points')

    o3d.visualization.draw_geometries([pcd_intens], window_name=f'Filtered lidar intensity')

    # prepare an (N,3) color array
    colors = np.zeros((xyz.shape[0], 3))
    colors[mask] = [0.0, 1.0, 0.0]  # green
    colors[~mask] = [1.0, 0.0, 0.0]  # red

    # create a colored PointCloud
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(xyz)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    # display
    o3d.visualization.draw_geometries(
        [pcd_colored],
        window_name="Intensity>0 = Green, ≤0 = Red"
    )

    ############### Apply SOR ##################################

    # Perform Statistical Outlier Removal
    print("Applying Statistical Outlier Removal...")
    pcd_sor, ind_sor = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"SOR removed {len(pcd.points) - len(ind_sor)} points.")

    print("Displaying Original Point Cloud...")
    o3d.visualization.draw_geometries([pcd_sor], window_name=f'Filtered SOR point cloud')

    ############# Apply SOR + ROR #############################

    # Perform Radius Outlier Removal
    print("Applying Radius Outlier Removal...")
    pcd_ror, ind_ror = pcd_sor.remove_radius_outlier(nb_points=16, radius=0.5)
    print(f"ROR removed {len(pcd.points) - len(ind_ror)} points.")

    print("Displaying Filtered Point Cloud (SOR + ROR)...")
    o3d.visualization.draw_geometries([pcd_ror], window_name=f'Filtered SOR + ROR point cloud')

    ####################### DBSCAN noise removal ##############################
    # eps: cluster radius in meters, min_points: minimum points to form a cluster
    eps = 0.3
    min_points = 10
    print(f"Running DBSCAN (eps={eps}, min_points={min_points})...")
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )
    # noise points are labeled -1
    inlier_idx = np.where(labels >= 0)[0]
    pcd_db = pcd.select_by_index(inlier_idx)
    num_removed = len(labels) - len(inlier_idx)
    print(f"DBSCAN removed {num_removed} noise points, remaining {len(inlier_idx)}")

    o3d.visualization.draw_geometries([pcd_db], window_name="After DBSCAN")

if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument('--scene_id', type=int, default=0)
    parse.add_argument('--version', type=str, default='v1.0-trainval')
    parse.add_argument('--data_root', type=str, default='/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval')

    args = parse.parse_args()

    trucksc = TruckScenes(version=args.version, dataroot=args.data_root, verbose=True)

    main(trucksc, args)

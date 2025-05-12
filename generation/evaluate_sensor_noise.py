import os.path
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud
from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def main(trucksc, args):
    my_scene = trucksc.scene[args.scene_id]

    first_sample_token = my_scene['first_sample_token']
    my_sample = trucksc.get('sample', first_sample_token)
    sample_data_token = my_sample['data']['LIDAR_LEFT']
    sample_data = trucksc.get('sample_data', sample_data_token)

    trucksc.render_scene(my_scene['token'])

    filename = sample_data['filename']
    load_path = os.path.join(trucksc.dataroot, filename)

    pc = LidarPointCloud.from_file(load_path)
    print("Original point cloud shape:", pc.points.shape)

    xyz = pc.points[:3, :].T  # shape (N,3)
    intensities = pc.points[3, :]

    print(intensities)

    mask = intensities > 0
    xyz_filtered = xyz[mask]

    pcd_intens = o3d.geometry.PointCloud()
    pcd_intens.points = o3d.utility.Vector3dVector(xyz_filtered)

    print(f'Removed {xyz.shape[0] - len(pcd_intens.points)} points')


    o3d.visualization.draw_geometries([pcd_intens], window_name=f'Filtered lidar intensity')

    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.points[:3, :].T)

    # Perform Statistical Outlier Removal
    print("Applying Statistical Outlier Removal...")
    pcd_sor, ind_sor = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"SOR removed {len(pcd.points) - len(ind_sor)} points.")

    # Perform Radius Outlier Removal (optional but recommended)
    print("Applying Radius Outlier Removal...")
    pcd_ror, ind_ror = pcd_sor.remove_radius_outlier(nb_points=16, radius=0.5)
    print(f"ROR removed {len(pcd.points) - len(ind_ror)} points.")

    # Visualize before and after filtering
    print("Displaying Original Point Cloud...")
    o3d.visualization.draw_geometries([pcd], window_name=f'Original point cloud')

    print("Displaying Original Point Cloud...")
    o3d.visualization.draw_geometries([pcd_sor], window_name=f'Filtered SOR point cloud')

    print("Displaying Filtered Point Cloud (SOR + ROR)...")
    o3d.visualization.draw_geometries([pcd_ror], window_name=f'Filtered SOR + ROR point cloud')

    # Optionally, save the filtered point cloud
    # o3d.io.write_point_cloud("filtered_point_cloud.pcd", pcd_ror)


if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument('--scene_id', type=int, default=5)
    parse.add_argument('--version', type=str, default='v1.0-trainval')
    parse.add_argument('--data_root', type=str, default='/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval')

    args = parse.parse_args()

    trucksc = TruckScenes(version=args.version, dataroot=args.data_root, verbose=True)

    main(trucksc, args)

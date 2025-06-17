import open3d as o3d

import numpy as np
from pathlib import Path

saved_ply_path = Path('./results/2025-06-02_16-44-43/ply/000130.ply')
saved_ply_path = Path('./results/2025-06-02_17-26-19/ply/000080.ply')
saved_ply_path = Path('./results/2025-06-02_17-49-34/ply/000030.ply')
saved_ply_path = Path('./results/2025-06-02_18-12-43/ply/000040.ply')

if saved_ply_path.is_file():
    print(f"Loading PLY file: {saved_ply_path}")
    pcd = o3d.io.read_point_cloud(str(saved_ply_path))

    if not pcd.has_points():
        print("Loaded PLY is empty.")
    else:
        print(f"Loaded point cloud with {len(pcd.points)} points.")
        if pcd.has_colors():
            print("Point cloud has colors. These might represent static/moving.")
        else:
            print("Point cloud does not have colors by default.")

    o3d.visualization.draw_geometries([pcd], window_name="MapMOS Output PLY")
else:
    print(f"PLY file not found at: {saved_ply_path}")
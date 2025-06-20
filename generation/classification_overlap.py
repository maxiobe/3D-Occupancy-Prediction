import numpy as np
import open3d as o3d
import os
from collections import defaultdict
import math
from shapely.geometry import Polygon, Point, MultiPolygon

# --- USER CONFIG: CLASS COLORS ---
CLASS_COLOR_MAP = {
    0: [0.6, 0.6, 0.6], 1: [0.9, 0.1, 0.1], 2: [1.0, 0.6, 0.0],
    3: [0.5, 0.0, 0.5], 4: [0.0, 0.0, 1.0], 5: [0.3, 0.3, 0.0],
    6: [1.0, 0.0, 1.0], 7: [1.0, 1.0, 0.0], 8: [1.0, 0.5, 0.5],
    9: [0.5, 0.5, 0.0],  # Trailer - Changed to Olive for better contrast
    10: [0.0, 1.0, 0.0],  # Truck - Green
    11: [0.2, 0.8, 0.8],
    12: [1.0, 0.8, 0.0], 13: [0.4, 0.4, 0.8], 14: [0.0, 0.5, 0.5],
    15: [0.8, 0.8, 0.8], 16: [0.0, 1.0, 1.0],
}
DEFAULT_COLOR = [0.5, 0.5, 0.5]


# ------------------------------------------------------------------

def polygon_to_3d_lineset(poly2d, y_min, y_max, col=(1, 0, 0)):
    """
    Extrude a 2D Shapely polygon (or MultiPolygon) in X–Z up to a Y-thickness,
    and return a list of Open3D LineSet wireframes.
    (This function is kept as is, but will be used by a new helper)
    """
    if isinstance(poly2d, MultiPolygon):
        linesets = []
        for part in poly2d.geoms:
            linesets.extend(polygon_to_3d_lineset(part, y_min, y_max, col))
        return linesets

    ring = list(poly2d.exterior.coords)
    n = len(ring)

    verts = []
    for (x, z) in ring:
        verts.append([x, y_min, z])
    for (x, z) in ring:
        verts.append([x, y_max, z])

    lines = []
    for i in range(n):
        lines.append([i, (i + 1) % n])
    for i in range(n, 2 * n):
        lines.append([i, n + ((i + 1) % n)])
    for i in range(n):
        lines.append([i, i + n])

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(verts)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    ).paint_uniform_color(col)

    return [ls]


# ★★★ CHANGE 1: Corrected the L-Shape polygon definition ★★★
def create_side_L_shapes(box_tr, box_tl, h_hitch=1.5):
    """
    Given two LOCAL-FRAME box tuples (cx,cy,cz, L,W,H, yaw=0), returns:
      - tractor_L2D : the L-shape for the tractor in X–Z
      - trailer_L2D : the complementary shape for the trailer
      - z_ground_tr, z_roof_tr
    """
    # NOTE: The box format is assumed to be (center_x, center_y, center_z, Length, Width, Height, yaw)
    # Length is along the vehicle's forward axis (X), Width is across (Y).
    cxt, _, szt, Lt, _, Ht, _ = box_tr
    cxl, _, szl, Ll, _, Hl, _ = box_tl

    # half‐sizes & key Z planes for tractor vs. trailer
    z_ground_tr = szt - Ht / 2.0
    z_roof_tr = szt + Ht / 2.0
    z_ground_tl = szl - Hl / 2.0
    z_roof_tl = szl + Hl / 2.0
    z_hitch = z_ground_tr + h_hitch

    # X‐planes (in local frame)
    x_front_tr = cxt + Lt / 2.0
    x_back_tr = cxt - Lt / 2.0
    x_front_tl = cxl + Ll / 2.0
    x_back_tl = cxl - Ll / 2.0

    # 1) tractor L‐shape in X–Z (using tractor’s Z bounds)
    # This shape now has 6 points for a more accurate "step" profile.
    tractor_pts = [
        (x_front_tr, z_ground_tr),  # Bottom-front of cab
        (x_front_tr, z_roof_tr),  # Top-front of cab
        (x_front_tl, z_roof_tr),  # Top-rear of cab (at trailer coupling)
        (x_front_tl, z_hitch),  # Drop down to hitch height
        (x_back_tr, z_hitch),  # Back of tractor chassis at hitch height
        (x_back_tr, z_ground_tr),  # Bottom-rear of tractor chassis
    ]
    tractor_L2D = Polygon(tractor_pts).buffer(0)

    # 2) full trailer rectangle side‐view (using trailer’s Z bounds)
    trailer_rect = Polygon([
        (x_front_tl, z_ground_tl),
        (x_front_tl, z_roof_tl),
        (x_back_tl, z_roof_tl),
        (x_back_tl, z_ground_tl),
    ]).buffer(0)

    # 3) subtract tractor‐L from trailer rect
    trailer_L2D = trailer_rect.difference(tractor_L2D)
    if not trailer_L2D.is_valid:
        trailer_L2D = trailer_L2D.buffer(0)

    return tractor_L2D, trailer_L2D, z_ground_tr, z_roof_tr


# ------------------------------------------------------------------
def assign_label_by_L_shape(
        overlap_idxs, pt_to_box_map,
        points, boxes, box_cls_labels, pt_labels
):
    """
    Reassign only overlap points between each truck(10)/trailer(9) pair.
    All others keep pt_labels.
    (This function was logically correct and is kept as is).
    """
    new_labels = pt_labels.flatten().copy()
    h_hitch = 1.3

    pair_to_pts = defaultdict(list)
    for pi in overlap_idxs:
        b = sorted(pt_to_box_map[pi])
        if len(b) == 2:
            pair_to_pts[tuple(b)].append(pi)

    for (i, j), pts in pair_to_pts.items():
        li, lj = box_cls_labels[i], box_cls_labels[j]
        if set((li, lj)) != {9, 10}:
            continue

        idx_tr = i if li == 10 else j
        idx_tl = j if idx_tr == i else i

        # Unpack world‐coords. Note: Your npy file seems to have w,l,h not l,w,h
        # boxes[:,3:6] -> width, length, height
        cx_tr, cy_tr, cz_tr, w_tr, l_tr, h_tr, yaw_tr, _ = boxes[idx_tr]
        cx_tl, cy_tl, cz_tl, w_tl, l_tl, h_tl, _, _ = boxes[idx_tl]

        c, s = math.cos(-yaw_tr), math.sin(-yaw_tr)

        # Build LOCAL‐frame boxes (Length is X, Width is Y)
        box_tr_local = (0.0, 0.0, cz_tr, l_tr, w_tr, h_tr, 0.0)

        dx0 = cx_tl - cx_tr
        dy0 = cy_tl - cy_tr
        x_loc_tl = dx0 * c - dy0 * s
        y_loc_tl = dx0 * s + dy0 * c
        box_tl_local = (x_loc_tl, y_loc_tl, cz_tl, l_tl, w_tl, h_tl, 0.0)

        L_tr, L_tl, z_ground, z_roof = create_side_L_shapes(
            box_tr_local, box_tl_local, h_hitch
        )

        half_W_tr = w_tr / 2.0
        for pi in pts:
            xg, yg, zg = points[pi]
            # Rotate point into tractor's local frame
            dx_pt = (xg - cx_tr) * c - (yg - cy_tr) * s
            dy_pt = (xg - cx_tr) * s + (yg - cy_tr) * c

            # Check if point is laterally within the tractor's width and vertically within its profile
            if abs(dy_pt) > half_W_tr or not (z_ground <= zg <= z_roof):
                continue

            # Check against the 2D side-view polygons
            point_2d = Point(dx_pt, zg)
            if L_tr.contains(point_2d):
                new_labels[pi] = 10  # Truck
            elif L_tl.contains(point_2d):
                new_labels[pi] = 9  # Trailer

    return new_labels.reshape(-1, 1)


# ★★★ CHANGE 2: New helper function to create correctly rotated wireframes ★★★
def create_rotated_wireframes(box_tr_world, box_tl_world):
    """
    Creates the L-shape wireframes in the correct world-frame orientation.
    """
    # Unpack world-frame boxes to get poses and dimensions
    cx_tr, cy_tr, cz_tr, w_tr, l_tr, h_tr, yaw_tr = box_tr_world
    cx_tl, cy_tl, cz_tl, w_tl, l_tl, h_tl, yaw_tl = box_tl_world

    # --- 1. Create polygons in LOCAL frame (same logic as classification) ---
    c, s = math.cos(-yaw_tr), math.sin(-yaw_tr)
    box_tr_local = (0.0, 0.0, cz_tr, l_tr, w_tr, h_tr, 0.0)
    dx0, dy0 = cx_tl - cx_tr, cy_tl - cy_tr
    x_loc_tl, y_loc_tl = dx0 * c - dy0 * s, dx0 * s + dy0 * c
    box_tl_local = (x_loc_tl, y_loc_tl, cz_tl, l_tl, w_tl, h_tl, 0.0)

    L_tr_local, L_tl_local, _, _ = create_side_L_shapes(box_tr_local, box_tl_local)

    # --- 2. Extrude local polygons along their local Y-axis ---
    # The extrusion happens from -width/2 to +width/2 in the local Y frame.
    y0 = -w_tr / 2.0
    y1 = +w_tr / 2.0

    # Create non-rotated LineSets from the local-frame polygons
    ls_tr_local = polygon_to_3d_lineset(L_tr_local, y0, y1, col=(0, 1, 0))  # Green for Truck
    ls_tl_local = polygon_to_3d_lineset(L_tl_local, y0, y1, col=(1, 0, 1))  # Magenta for Trailer

    all_linesets = ls_tr_local + ls_tl_local

    # --- 3. Transform (rotate and translate) the LineSets to WORLD frame ---
    transform = np.identity(4)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw_tr])
    transform[:3, 3] = [cx_tr, cy_tr, 0]  # Z translation is part of the polygon's definition

    for ls in all_linesets:
        ls.transform(transform)

    return all_linesets


# ★★★ CHANGE 3: Refactored main visualization logic ★★★
def visualize_scene(data_directory, show_l_shapes=True, show_bboxes=False):
    """
    Main function that uses TWO sets of modified boxes as per the user's specific request:
    1. Boxes with W,L,H scaled for DETECTING overlap points.
    2. Boxes with only W scaled for CLASSIFYING points and VISUALIZING polygons.
    """
    print(f"--- Loading data from: {data_directory} ---")
    sem = np.load(os.path.join(data_directory, 'frame_0_sem_temp_points.npy'))
    pts = sem[:, :3]
    pt_labels = sem[:, 3].astype(int).reshape(-1, 1)

    # Load the original, unmodified boxes as our base
    boxes_unmodified = np.load(os.path.join(data_directory, 'frame_0_gt_bbox_3d_labeled.npy'))

    # <<< CHANGE 1: Create TWO distinct sets of modified boxes >>>

    # SET 1: Boxes for DETECTING overlap points (W, L, H scaled)
    boxes_for_detection = boxes_unmodified.copy()
    boxes_for_detection[:, 3:6] *= 1.2  # Scale W, L, H

    # SET 2: Boxes for generating POLYGONS (only W scaled)
    boxes_for_polygons = boxes_unmodified.copy()
    boxes_for_polygons[:, [3]] *= 1.2

    # The class labels are the same for all sets
    box_cls = boxes_unmodified[:, 7].astype(int)

    # <<< CHANGE 2: Use `boxes_for_detection` to find the overlap points >>>
    print("Finding overlap points using boxes scaled in all dimensions...")
    o3d_bboxes_detect = []
    for b in boxes_for_detection:
        # Using l,w,h from the fully scaled boxes
        c, e, y = b[0:3], [b[4], b[3], b[5]], b[6]
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, y])
        obb = o3d.geometry.OrientedBoundingBox(c, R, e)
        o3d_bboxes_detect.append(obb)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cnt = np.zeros(len(pts), int)
    pt2box = defaultdict(list)
    for i, obb in enumerate(o3d_bboxes_detect):
        idxs = obb.get_point_indices_within_bounding_box(pcd.points)
        cnt[idxs] += 1
        for pi in idxs:
            pt2box[pi].append(i)
    overlap = np.where(cnt > 1)[0]

    # <<< CHANGE 3: Use `boxes_for_polygons` for the classification logic >>>
    print("Classifying points using polygons with only width extended...")
    final_labels = assign_label_by_L_shape(
        overlap, pt2box, pts, boxes_for_polygons, box_cls, pt_labels
    )

    # Recolor the point cloud based on the final labels
    pcd.colors = o3d.utility.Vector3dVector(
        np.array([CLASS_COLOR_MAP.get(l, DEFAULT_COLOR) for l in final_labels.flatten()])
    )

    geometries_to_draw = [pcd]

    if show_l_shapes:
        seen_pairs = set()
        for pi in overlap:
            box_ids = tuple(sorted(pt2box[pi]))
            if len(box_ids) != 2 or box_ids in seen_pairs:
                continue

            i, j = box_ids
            if {box_cls[i], box_cls[j]} == {9, 10}:
                seen_pairs.add(box_ids)
                idx_tr = i if box_cls[i] == 10 else j
                idx_tl = j if idx_tr == i else i

                # <<< CHANGE 4: Use `boxes_for_polygons` to create the wireframes >>>
                # This ensures the visualization matches the classification logic.
                box_tr_world = boxes_for_polygons[idx_tr][:7]
                box_tl_world = boxes_for_polygons[idx_tl][:7]

                l_shape_wireframes = create_rotated_wireframes(box_tr_world, box_tl_world)
                geometries_to_draw.extend(l_shape_wireframes)

    if show_bboxes:
        # Decide which set of boxes to show. Let's show the polygon boxes
        # as they are most relevant to the final classification.
        for i, b in enumerate(boxes_for_polygons):
            c, e, y = b[0:3], [b[4], b[3], b[5]], b[6]
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, y])
            obb = o3d.geometry.OrientedBoundingBox(c, R, e)
            obb.color = CLASS_COLOR_MAP.get(box_cls[i], DEFAULT_COLOR)
            geometries_to_draw.append(obb)

    o3d.visualization.draw_geometries(
        geometries_to_draw,
        window_name="Dual-Geometry L-Shape Disambiguation",
        width=1920, height=1080
    )

# ====================================================
if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual path to your data directory
    DATA_DIR = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/gt/all_sensors_icp_boxes_labels/scene-0044384af3d8494e913fb8b14915239e-11/1bb41855cb724ae6980bababbd1865e2'

    # Check if the directory exists before running
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please update the DATA_DIR variable in the script.")
    else:
        # Run the visualization
        # Set show_l_shapes=True to see the new wireframes
        # Set show_bboxes=True to see the original bounding boxes
        visualize_scene(DATA_DIR, show_l_shapes=True, show_bboxes=False)
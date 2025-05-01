import numpy as np
import open3d as o3d
import os
from collections import defaultdict, Counter # Make sure Counter is imported
import scipy.stats # Make sure scipy is imported
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN

# Define class colormap (provided by user)
CLASS_COLOR_MAP = {
    0: [0.6, 0.6, 0.6],  # noise - gray
    1: [0.9, 0.1, 0.1],  # barrier - red
    2: [1.0, 0.6, 0.0],  # bicycle - orange
    3: [0.5, 0.0, 0.5],  # bus - purple
    4: [0.0, 0.0, 1.0],  # car - blue
    5: [0.3, 0.3, 0.0],  # construction_vehicle - olive
    6: [1.0, 0.0, 1.0],  # motorcycle - magenta
    7: [1.0, 1.0, 0.0],  # pedestrian - yellow
    8: [1.0, 0.5, 0.5],  # traffic_cone - light red
    9: [0.5, 0.5, 0.0],  # trailer - mustard
    10: [0.0, 1.0, 0.0],  # truck - green
    11: [0.2, 0.8, 0.8],  # ego_vehicle - cyan
    12: [1.0, 0.8, 0.0],  # traffic_sign - gold
    13: [0.4, 0.4, 0.8],  # other_vehicle - steel blue
    14: [0.0, 0.5, 0.5],  # train - teal
    15: [0.8, 0.8, 0.8],  # Unknown - light gray
    16: [0.0, 1.0, 1.0],  # Background - bright cyan
}
# Add a default color for labels not in the map
DEFAULT_COLOR = [0.5, 0.5, 0.5] # Medium gray
OVERLAP_COLOR = [1.0, 0.1, 0.7] # Bright Pink for overlapping points

def assign_label_by_local_dbscan(
    overlap_indices,
    point_to_boxes,
    points,
    original_labels, # Need original labels for mapping heuristic
    current_labels,
    # --- DBSCAN Parameters (CRITICAL TO TUNE!) ---
    eps=0.2,         # Max distance between samples for neighborhood
    min_samples=10   # Min points to form a core point
):
    """
    Assigns points in PAIRWISE overlaps using local DBSCAN.
    Maps DBSCAN clusters back to semantic labels based on majority original label.
    Points in overlaps involving 3+ boxes or marked as noise by DBSCAN keep original labels.
    """
    print(f"Applying disambiguation: Local DBSCAN (Pairwise Overlaps, eps={eps}, min_samples={min_samples})")
    new_labels = current_labels.copy() # Start with current labels (which should be original)

    # --- 1. Group points by the PAIR of boxes they overlap ---
    print("Grouping points by overlapping box pairs...")
    overlap_pairs_to_points = defaultdict(list)
    points_in_higher_overlaps = set()

    for point_idx in overlap_indices:
        overlapping_box_indices = sorted(point_to_boxes.get(point_idx, []))
        num_overlaps = len(overlapping_box_indices)

        if num_overlaps == 2:
            box_pair_key = tuple(overlapping_box_indices)
            overlap_pairs_to_points[box_pair_key].append(point_idx)
        elif num_overlaps > 2:
            points_in_higher_overlaps.add(point_idx)

    print(f"Found {len(overlap_pairs_to_points)} distinct pairwise overlap zones.")
    if points_in_higher_overlaps:
        print(f"Note: {len(points_in_higher_overlaps)} points are in overlaps of 3+ boxes and will keep original labels.")

    # --- 2. Process each pairwise overlap zone ---
    print("Running DBSCAN locally on pairwise overlaps...")
    processed_point_count = 0
    noise_point_count = 0

    for box_pair, point_indices_in_overlap in overlap_pairs_to_points.items():
        # Need enough points for DBSCAN to potentially form clusters
        if len(point_indices_in_overlap) < min_samples:
             continue

        overlap_points_coords = points[point_indices_in_overlap]
        np_point_indices_in_overlap = np.array(point_indices_in_overlap) # NumPy array for easier indexing

        # --- Run DBSCAN ---
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1) # n_jobs=-1 uses all CPU cores
            # Labels are -1 (noise), 0, 1, 2, ...
            point_cluster_labels = dbscan.fit_predict(overlap_points_coords)
        except Exception as e:
            print(f"Warning: DBSCAN failed for overlap between boxes {box_pair[0]} & {box_pair[1]}: {e}")
            continue # Keep original labels for points in this overlap

        # --- Map DBSCAN clusters (>=0) back using majority original label ---
        unique_dbscan_labels = np.unique(point_cluster_labels)

        for dbscan_label in unique_dbscan_labels:
            if dbscan_label == -1:
                # Handle noise points - keep original labels (already in new_labels)
                noise_mask = (point_cluster_labels == -1)
                noise_point_count += np.sum(noise_mask)
                continue # Go to next dbscan label

            # Find original indices for points in this DBSCAN cluster
            current_dbscan_cluster_mask = (point_cluster_labels == dbscan_label)
            original_indices_in_cluster = np_point_indices_in_overlap[current_dbscan_cluster_mask]

            if len(original_indices_in_cluster) == 0: continue # Skip empty clusters if they somehow occur

            # Get the ORIGINAL semantic labels of points in this DBSCAN cluster
            original_semantic_labels = original_labels[original_indices_in_cluster].flatten()

            # Find the most frequent original label within this DBSCAN cluster
            try:
                mode_result = scipy.stats.mode(original_semantic_labels)
                if mode_result.count == 0 : # Handle case where mode is undefined (e.g., empty input)
                     print(f"Warning: Could not find mode label for DBSCAN cluster {dbscan_label} in overlap {box_pair}. Keeping original labels.")
                     continue

                # scipy.stats.mode returns mode array and count array even for single mode
                majority_label = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode


            except Exception as e:
                 print(f"Warning: Error finding mode label for DBSCAN cluster {dbscan_label} in overlap {box_pair}: {e}. Keeping original labels.")
                 continue # Skip assignment for this cluster

            # Assign this majority label to all points originally indexed in this cluster
            new_labels[original_indices_in_cluster] = majority_label
            processed_point_count += len(original_indices_in_cluster)

    print(f"Finished Local DBSCAN: Processed {processed_point_count} points in clusters, {noise_point_count} noise points kept original label.")
    return new_labels

def assign_label_by_local_kmeans(
    overlap_indices,
    point_to_boxes,
    points,
    boxes_data,     # Need original box data for centers
    box_labels,     # Map box index to class label
    current_labels
):
    """
    Assigns points in PAIRWISE overlaps using local K-Means (K=2).
    Points in overlaps involving 3+ boxes are ignored by this strategy.
    """
    print("Applying disambiguation: Local K-Means (Pairwise Overlaps, K=2)")
    new_labels = current_labels.copy()

    # --- 1. Group points by the PAIR of boxes they overlap ---
    print("Grouping points by overlapping box pairs...")
    overlap_pairs_to_points = defaultdict(list)
    points_in_higher_overlaps = set() # Keep track of points in 3+ overlaps

    for point_idx in overlap_indices:
        overlapping_box_indices = sorted(point_to_boxes.get(point_idx, [])) # Sort for consistent key
        num_overlaps = len(overlapping_box_indices)

        if num_overlaps == 2:
            box_pair_key = tuple(overlapping_box_indices) # Use tuple as dict key
            overlap_pairs_to_points[box_pair_key].append(point_idx)
        elif num_overlaps > 2:
            points_in_higher_overlaps.add(point_idx)

    print(f"Found {len(overlap_pairs_to_points)} distinct pairwise overlap zones.")
    if points_in_higher_overlaps:
        print(f"Note: {len(points_in_higher_overlaps)} points are in overlaps of 3+ boxes and will keep original labels.")

    # --- 2. Process each pairwise overlap zone ---
    print("Running K-Means locally on pairwise overlaps...")
    processed_point_count = 0
    for box_pair, point_indices_in_overlap in overlap_pairs_to_points.items():
        if len(point_indices_in_overlap) < 2: # Need at least 2 points for K=2
             continue

        box1_idx, box2_idx = box_pair
        overlap_points_coords = points[point_indices_in_overlap]

        # --- Run K-Means ---
        try:
            # n_init='auto' or 10 recommended for stability
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10, verbose=0)
            point_cluster_labels = kmeans.fit_predict(overlap_points_coords) # Labels are 0 or 1
            cluster_centers = kmeans.cluster_centers_ # Centroids of the two K-Means clusters
        except Exception as e:
            print(f"Warning: K-Means failed for overlap between boxes {box1_idx} & {box2_idx}: {e}")
            continue # Keep original labels for points in this overlap

        # --- Map K-Means clusters (0, 1) back to original box labels ---
        # Use geometric centers of the original boxes
        if box1_idx >= len(boxes_data) or box2_idx >= len(boxes_data):
            print(f"Warning: Box index out of bounds for boxes {box1_idx} or {box2_idx}. Skipping assignment.")
            continue
        box1_center = boxes_data[box1_idx, 0:3]
        box2_center = boxes_data[box2_idx, 0:3]

        # Distances: K-Means cluster center 0 to Box 1/2 centers
        dist_c0_b1 = np.linalg.norm(cluster_centers[0] - box1_center)
        dist_c0_b2 = np.linalg.norm(cluster_centers[0] - box2_center)
        # Distances: K-Means cluster center 1 to Box 1/2 centers
        dist_c1_b1 = np.linalg.norm(cluster_centers[1] - box1_center)
        dist_c1_b2 = np.linalg.norm(cluster_centers[1] - box2_center)

        # Assign K-Means cluster 0 to the closer box center's label
        if dist_c0_b1 < dist_c0_b2:
            map_cluster0_to_label = box_labels[box1_idx]
            map_cluster1_to_label = box_labels[box2_idx] # Cluster 1 gets the other label
            # Sanity check: ensure cluster 1 is closer to box 2 (usually holds)
            # if dist_c1_b2 > dist_c1_b1:
            #     print(f" Info: K-Means cluster mapping sanity check passed for boxes {box_pair}")
        elif dist_c0_b2 < dist_c0_b1:
            map_cluster0_to_label = box_labels[box2_idx]
            map_cluster1_to_label = box_labels[box1_idx] # Cluster 1 gets the other label
             # Sanity check: ensure cluster 1 is closer to box 1 (usually holds)
            # if dist_c1_b1 > dist_c1_b2:
            #      print(f" Info: K-Means cluster mapping sanity check passed for boxes {box_pair}")
        else:
            # Tie in distance for cluster 0 - ambiguous mapping. Keep original labels for this group.
            print(f"Warning: K-Means cluster 0 equidistant to box centers {box1_idx} & {box2_idx}. Keeping original labels.")
            continue # Skip assignment for this overlap pair

        # --- Assign final labels to points based on K-Means results ---
        for i, point_idx in enumerate(point_indices_in_overlap):
            kmeans_label = point_cluster_labels[i]
            if kmeans_label == 0:
                new_labels[point_idx] = map_cluster0_to_label
            else: # kmeans_label == 1
                new_labels[point_idx] = map_cluster1_to_label
            processed_point_count += 1

    print(f"Finished Local K-Means: Processed {processed_point_count} points in pairwise overlaps.")
    return new_labels

def assign_label_by_gmm(
    overlap_indices,
    point_to_boxes,
    points,
    original_labels, # Use original labels to find training data
    point_counts,    # Use counts to identify non-overlapping points
    box_labels,      # Map box index to class label
    n_components=1,  # Number of Gaussian components per class GMM
    min_points_for_train=50 # Minimum non-overlapping points needed to train a GMM
):
    """
    Assigns overlapping points based on the highest likelihood from class-specific GMMs
    trained on non-overlapping points.
    """
    print(f"Applying disambiguation: GMM (n_components={n_components})")
    new_labels = original_labels.copy() # Start with original labels

    # --- 1. Train GMMs for relevant classes ---
    print("Training GMMs...")
    class_gmms = {}
    # Find all unique labels associated with the actual boxes present
    unique_box_cls_labels = np.unique(box_labels)

    for cls_label in unique_box_cls_labels:
        if cls_label == 0: continue # Skip noise/background if necessary

        # Find indices of points ORIGINALLY labeled as this class AND are NON-OVERLAPPING
        # Non-overlapping means they fell into exactly one box (point_counts == 1)
        train_indices = np.where(
            (original_labels.flatten() == cls_label) & (point_counts == 1)
        )[0]

        if len(train_indices) >= min_points_for_train:
            print(f"  Training GMM for class {cls_label} with {len(train_indices)} non-overlapping points...")
            points_for_gmm = points[train_indices]

            try:
                # covariance_type='full' allows ellipsoids, 'diag' forces axis-aligned
                gmm = GaussianMixture(n_components=n_components, random_state=0,
                                      covariance_type='full', reg_covar=1e-6) # reg_covar avoids singular matrices
                gmm.fit(points_for_gmm)
                class_gmms[cls_label] = gmm
            except Exception as e:
                print(f"  Warning: Failed to train GMM for class {cls_label}: {e}")
        else:
            print(f"  Skipping GMM for class {cls_label} - insufficient non-overlapping points ({len(train_indices)} < {min_points_for_train}).")

    if not class_gmms:
        print("Warning: No GMMs were successfully trained. Returning original labels.")
        return new_labels

    # --- 2. Assign overlapping points based on highest likelihood ---
    print("Assigning overlapping points using trained GMMs...")
    assigned_count = 0
    failed_count = 0
    for point_idx in overlap_indices:
        point_coord = points[point_idx].reshape(1, -1) # Needs shape (1, 3) for score_samples

        # Identify candidate classes based on the boxes this point overlaps with
        overlapping_box_indices = point_to_boxes.get(point_idx, [])
        if not overlapping_box_indices:
             failed_count += 1
             continue # Should not happen if point_to_boxes is built correctly

        # Get unique class labels for these specific overlapping boxes
        candidate_labels = set(box_labels[box_idx] for box_idx in overlapping_box_indices if box_idx < len(box_labels))

        max_log_likelihood = -np.inf
        best_label = -1 # Default invalid label
        found_candidate = False

        for label in candidate_labels:
            if label in class_gmms: # Check if we successfully trained a GMM for this candidate
                try:
                    log_likelihood = class_gmms[label].score_samples(point_coord)[0]
                    if log_likelihood > max_log_likelihood:
                        max_log_likelihood = log_likelihood
                        best_label = label
                        found_candidate = True
                except Exception as e:
                    print(f"Warning: Error scoring point {point_idx} with GMM for label {label}: {e}")

        # Assign the best label found, otherwise keep the original
        if found_candidate and best_label != -1:
            new_labels[point_idx] = best_label
            assigned_count += 1
        else:
            # Keep original label if no suitable GMM candidate found or scoring failed
             failed_count += 1
            # Optionally: Assign an 'ambiguous' label here instead
            # new_labels[point_idx] = AMBIGUOUS_LABEL

    print(f"Finished GMM assignment: {assigned_count} points reassigned, {failed_count} kept original label due to issues.")
    return new_labels

def assign_label_by_nearest_center(
    overlap_indices, point_to_boxes, points, boxes_data, box_labels, current_labels
):
    # ... (function code from previous answer) ...
    print("Applying disambiguation: Nearest Box Center")
    new_labels = current_labels.copy()
    for point_idx in overlap_indices:
        point_coord = points[point_idx]
        # Use .get for safer dictionary access, provide empty list if key missing
        overlapping_box_indices = point_to_boxes.get(point_idx, [])

        if not overlapping_box_indices: # Skip if somehow no boxes listed for overlap index
             print(f"Warning: No overlapping boxes found in mapping for point index {point_idx}. Keeping original label.")
             continue

        min_dist_sq = float('inf')
        best_label = -1 # Default/error label

        for box_idx in overlapping_box_indices:
            if box_idx >= len(boxes_data):
                 print(f"Warning: Box index {box_idx} out of bounds for boxes_data ({len(boxes_data)}).")
                 continue
            box_center = boxes_data[box_idx, 0:3]
            dist_sq = np.sum((point_coord - box_center)**2)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                if box_idx >= len(box_labels):
                     print(f"Warning: Box index {box_idx} out of bounds for box_labels ({len(box_labels)}).")
                     best_label = -1 # Assign default/error label
                else:
                    best_label = box_labels[box_idx]


        if best_label != -1:
            new_labels[point_idx] = best_label
        else:
            print(f"Warning: Could not determine valid nearest box label for point index {point_idx}. Keeping original label {new_labels[point_idx]}.")
            # Keep original label if no valid nearest box found

    return new_labels

# --- Nearest Surface Helper ---
def distance_point_to_obb_surface(point, obb):
    # ... (function code from previous answer) ...
    center = obb.center
    R = obb.R  # Rotation matrix
    extent = obb.extent # Full extent [width, length, height]
    p_local = R.T @ (point - center)
    half_extent = extent / 2.0
    dist_x = half_extent[0] - abs(p_local[0])
    dist_y = half_extent[1] - abs(p_local[1])
    dist_z = half_extent[2] - abs(p_local[2])
    min_dist = min(max(0, dist_x), max(0, dist_y), max(0, dist_z))
    return min_dist


def assign_label_by_nearest_surface(
    overlap_indices, point_to_boxes, points, o3d_boxes_list, box_labels, current_labels
):
    # ... (function code from previous answer) ...
    print("Applying disambiguation: Nearest Box Surface")
    new_labels = current_labels.copy()
    for point_idx in overlap_indices:
        point_coord = points[point_idx]
        overlapping_box_indices = point_to_boxes.get(point_idx, [])

        if not overlapping_box_indices:
             print(f"Warning: No overlapping boxes found in mapping for point index {point_idx}. Keeping original label.")
             continue

        min_dist = float('inf')
        best_label = -1

        for box_idx in overlapping_box_indices:
             if box_idx >= len(o3d_boxes_list):
                 print(f"Warning: Box index {box_idx} out of bounds for o3d_boxes_list ({len(o3d_boxes_list)}).")
                 continue

             obb = o3d_boxes_list[box_idx]
             try:
                 dist = distance_point_to_obb_surface(point_coord, obb)
             except Exception as e:
                 print(f"Error calculating surface distance for point {point_idx}, box {box_idx}: {e}")
                 dist = float('inf')

             if dist < min_dist:
                min_dist = dist
                if box_idx >= len(box_labels):
                     print(f"Warning: Box index {box_idx} out of bounds for box_labels ({len(box_labels)}).")
                     best_label = -1
                else:
                    best_label = box_labels[box_idx]

        if best_label != -1:
            new_labels[point_idx] = best_label
        else:
             print(f"Warning: Could not determine valid nearest surface label for point index {point_idx}. Keeping original label {new_labels[point_idx]}.")

    return new_labels


def assign_label_by_knn_majority(
    overlap_indices, point_counts, points, current_labels, k=50 # Number of neighbours
):
    # ... (function code from previous answer) ...
    print(f"Applying disambiguation: KNN Majority Vote (k={k})")
    new_labels = current_labels.copy()
    non_overlap_indices = np.where(point_counts == 1)[0]

    if len(non_overlap_indices) == 0:
        print("Warning: No non-overlapping points found. Cannot use KNN majority. Returning original labels for overlaps.")
        return new_labels

    non_overlap_points = points[non_overlap_indices]
    pcd_non_overlap = o3d.geometry.PointCloud()
    pcd_non_overlap.points = o3d.utility.Vector3dVector(non_overlap_points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_non_overlap)
    print(f"Built KDTree on {len(non_overlap_indices)} non-overlapping points.")

    print("Querying neighbors for overlapping points...")
    for point_idx in overlap_indices:
        point_coord = points[point_idx]
        [found_k, indices, _] = kdtree.search_knn_vector_3d(point_coord, k)

        if found_k > 0:
            neighbor_original_indices = non_overlap_indices[indices]
            neighbor_labels = current_labels[neighbor_original_indices].flatten()
            try:
                mode_result = scipy.stats.mode(neighbor_labels)
                # Handle different scipy versions and empty results
                if hasattr(mode_result, 'mode') and np.isscalar(mode_result.mode):
                    majority_label = mode_result.mode
                elif hasattr(mode_result, 'mode') and mode_result.mode.size > 0 : # Check if array and not empty
                    majority_label = mode_result.mode[0]
                else:
                     print(f"Warning: KNN mode calculation failed for point {point_idx}. Keeping original label.")
                     continue # Keep original label held in new_labels

                new_labels[point_idx] = majority_label
            except Exception as e:
                 print(f"Error finding mode for point {point_idx}: {e}. Keeping original label.")
        else:
            print(f"Warning: No neighbors found for point {point_idx}. Keeping original label.")

    return new_labels


def visualize_object_data(data_directory, highlight_overlap=True): # Added flag
    """
    Loads bounding boxes and object points, applies transformations to boxes,
    identifies points in multiple boxes, and visualizes them.

    Args:
        data_directory (str): The path to the directory containing the .npy files.
        highlight_overlap (bool): If True, colors points in multiple boxes white.
    """
    print(f"--- Loading data from: {data_directory} ---")

    # --- Define File Paths ---
    bbox_file = os.path.join(data_directory, 'frame_0_gt_bbox_3d_labeled.npy')
    # obj_pts_file = os.path.join(data_directory, 'frame_0_temp_points.npy') # Original points without labels
    obj_sem_pts_file = os.path.join(data_directory, 'frame_0_sem_temp_points.npy') # Points with labels

    # --- Load Data ---
    object_points = np.empty((0, 3))
    object_labels = np.empty((0, 1), dtype=int)
    gt_bboxes_data_original = np.empty((0, 7)) # Store original loaded boxes here

    # Load Object Points with Semantic Labels
    try:
        loaded_obj_sem = np.load(obj_sem_pts_file)
        if loaded_obj_sem.ndim == 2 and loaded_obj_sem.shape[1] >= 4:
            object_points = loaded_obj_sem[:, :3]
            # Ensure labels are integers and have shape (N, 1)
            object_labels = loaded_obj_sem[:, 3].astype(int).reshape(-1, 1)
            print(f"Loaded object points: {object_points.shape}, labels: {object_labels.shape}")
        else:
            print(f"Warning: Object semantic points file {obj_sem_pts_file} has unexpected shape {loaded_obj_sem.shape}. Expected Nx4.")
    except FileNotFoundError:
        print(f"Warning: Object semantic points file not found at {obj_sem_pts_file}.")
    except Exception as e:
        print(f"Error loading object semantic points {obj_sem_pts_file}: {e}")

    # Load ORIGINAL Bounding Boxes
    try:
        loaded_bbox = np.load(bbox_file)
        # Expecting original format: [x, y, center_z, w, l, h, yaw]
        if loaded_bbox.ndim == 2 and loaded_bbox.shape[1] == 8:
            gt_bboxes_data_original = loaded_bbox
            print(f"Loaded original bounding boxes: {gt_bboxes_data_original.shape}")
        elif loaded_bbox.shape != (0, 8) and loaded_bbox.size > 0: # Check size > 0 to avoid warning on empty files
             print(f"Warning: Bounding box file {bbox_file} has unexpected shape {loaded_bbox.shape}. Expected Nx7. Skipping boxes.")
        elif loaded_bbox.size == 0:
             print(f"Info: Bounding box file {bbox_file} is empty.")
        else: # Catch other unexpected cases
             print(f"Warning: Bounding box file {bbox_file} could not be loaded correctly. Shape: {loaded_bbox.shape}. Skipping boxes.")

    except FileNotFoundError:
        print(f"Warning: Bounding box file not found at {bbox_file}.")
    except Exception as e:
        print(f"Error loading bounding boxes {bbox_file}: {e}")

    # --- Exit if nothing significant was loaded ---
    # We need points to check for overlaps, even if boxes aren't loaded (though pointless then)
    if object_points.shape[0] == 0:
        print("Error: Failed to load points. Cannot perform overlap check or visualization.")
        return

    # --- Prepare Point Cloud (if loaded) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points)
    # Assign initial colors based on semantic labels
    colors = np.array([CLASS_COLOR_MAP.get(label[0], DEFAULT_COLOR) for label in object_labels], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Created point cloud geometry with {object_points.shape[0]} points.")

    # --- Prepare Bounding Boxes (Apply Transformations) ---
    bboxes_list = []
    # This list will store the OrientedBoundingBox objects from Open3D
    o3d_bboxes = []
    if gt_bboxes_data_original.shape[0] > 0:
        # Make a copy to modify
        gt_bboxes_data_modified = gt_bboxes_data_original.copy()

        # --- Apply the transformations ---
        h_original = gt_bboxes_data_original[:, 5].copy() # Original height
        gt_bboxes_data_modified[:, 6] += np.pi / 2.0
        print("Adjusted Yaw (+pi/2).")
        # gt_bboxes_data_modified[:, 2] -= h_original / 2.0 # Z to base (Optional)
        # print("Adjusted Z from center to base.")
        gt_bboxes_data_modified[:, 2] -= 0.1
        print("Applied Z offset (-0.1).")
        gt_bboxes_data_modified[:, 3:6] *= 1.1
        print("Applied scaling (1.1x) to box dimensions.")

        # --- Create Open3D Boxes using MODIFIED values ---
        for i in range(gt_bboxes_data_modified.shape[0]):
            center = gt_bboxes_data_modified[i, 0:3]
            extent = gt_bboxes_data_modified[i, 3:6] # w, l, h -> extent for O3D
            yaw = gt_bboxes_data_modified[i, 6]

            # --- Get Label ID (from ORIGINAL array, column index 7) ---
            # Ensure label is integer for dictionary lookup
            try:
                label_id = int(gt_bboxes_data_original[i, 7])
            except IndexError:
                print(f"Error: Could not access label ID at index 7 for box {i}. Using default color.")
                label_id = 15  # Default to Unknown label index
            except ValueError:
                print(
                    f"Error: Could not convert label ID '{gt_bboxes_data_original[i, 7]}' to int for box {i}. Using default color.")
                label_id = 15  # Default to Unknown label index

            # Convert yaw to rotation matrix (yaw around Z axis)
            # Note: Open3D OrientedBoundingBox uses rotation matrix directly
            # R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw)) # Use this if center is relative to world Z
            # For yaw adjustment like this, it's simpler to think axis-angle
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

            # Create the Open3D bounding box object
            # Constructor: center, rotation matrix R, extent [width, length, height]
            bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
            # bbox.color = (1, 0, 0) # Red color for boxes
            box_color = CLASS_COLOR_MAP.get(label_id, DEFAULT_COLOR)
            bbox.color = tuple(box_color)  # Assign color (convert list to tuple)
            bboxes_list.append(bbox) # Add to list for visualization
            o3d_bboxes.append(bbox) # Keep separate list of O3D objects for point checking

        print(f"Processed {len(bboxes_list)} bounding boxes using specified transformations.")
    else:
         print("No bounding boxes loaded or processed.")

    # --- Find Points in Multiple Bounding Boxes ---
    overlapping_point_indices = []
    if object_points.shape[0] > 0 and len(o3d_bboxes) > 0:
        print("\n--- Checking for points in multiple boxes ---")
        # Create a count array, initialized to zeros
        point_counts = np.zeros(object_points.shape[0], dtype=int)

        # Iterate through each bounding box
        for i, bbox in enumerate(o3d_bboxes):
            # Find indices of points inside this bounding box
            # Note: Requires points as o3d.utility.Vector3dVector
            indices_in_box = bbox.get_point_indices_within_bounding_box(pcd.points)

            if indices_in_box:
                # Increment the count for points found in this box
                point_counts[indices_in_box] += 1
            # print(f"Box {i}: Found {len(indices_in_box)} points inside.") # Debug print

        # Find indices where the count is greater than 1
        overlapping_point_indices = np.where(point_counts > 1)[0]

        if len(overlapping_point_indices) > 0:
            print(f"Found {len(overlapping_point_indices)} points belonging to 2 or more bounding boxes.")
            # print("Indices of overlapping points:", overlapping_point_indices) # Optional: print indices

            if highlight_overlap:
                # Modify the colors of the overlapping points in the main point cloud
                overlap_colors = np.asarray(pcd.colors)
                overlap_colors[overlapping_point_indices] = OVERLAP_COLOR
                pcd.colors = o3d.utility.Vector3dVector(overlap_colors)
                print(f"Highlighted overlapping points in white.")

        else:
            print("No points found in multiple bounding boxes.")

    elif object_points.shape[0] == 0:
        print("Skipping overlap check: No points loaded.")
    else: # No boxes loaded
        print("Skipping overlap check: No bounding boxes loaded.")

    # --- Visualize ---
    geometries_to_draw = []
    if object_points.shape[0] > 0:
        geometries_to_draw.append(pcd) # Add point cloud (potentially with highlighted overlaps)
    if bboxes_list:
        geometries_to_draw.extend(bboxes_list)

    if not geometries_to_draw:
        print("No geometries available to draw.")
        return

    print("\n--- Opening Visualization Window ---")
    print("Points in original semantic colors (overlapping points may be white).")
    print("Boxes are shown in red.")
    print("Press 'Q' or close the window to exit.")
    o3d.visualization.draw_geometries(geometries_to_draw,
                                      window_name=f"Object Points & Transformed GT Boxes - {os.path.basename(data_directory)}",
                                      width=1280, height=720)
    print("--- Visualization window closed ---")


def visualize_object_data_refined(data_directory, strategy=None): # Removed highlight_overlap
    """
    Loads points and labeled boxes, identifies overlaps, applies a
    disambiguation strategy (if specified), and visualizes the result.

    Args:
        data_directory (str): Path to the directory containing the .npy files.
        strategy (str, optional): Disambiguation strategy to apply.
            Options: 'nearest_center', 'nearest_surface', 'knn_majority',
                     None (or 'highlight'). Defaults to None.
    """
    print(f"--- Loading data from: {data_directory} ---")
    print(f"--- Disambiguation Strategy: {strategy} ---")

    # --- Define File Paths ---
    # Use the _labeled file
    bbox_file = os.path.join(data_directory, 'frame_0_gt_bbox_3d_labeled.npy')
    obj_sem_pts_file = os.path.join(data_directory, 'frame_0_sem_temp_points.npy')

    # --- Load Data ---
    object_points = np.empty((0, 3))
    object_labels = np.empty((0, 1), dtype=int)
    # Expect N x 8 array: [x, y, z, w, l, h, yaw, label_id]
    gt_bboxes_data_original = np.empty((0, 8))
    gt_box_labels = np.array([], dtype=int) # Initialize empty box labels array

    # Load Object Points with Semantic Labels
    try:
        loaded_obj_sem = np.load(obj_sem_pts_file)
        if loaded_obj_sem.ndim == 2 and loaded_obj_sem.shape[1] >= 4:
            object_points = loaded_obj_sem[:, :3]
            # Ensure labels are integers and have shape (N, 1)
            object_labels = loaded_obj_sem[:, 3].astype(int).reshape(-1, 1)
            print(f"Loaded object points: {object_points.shape}, labels: {object_labels.shape}")
        else:
            print(
                f"Warning: Object semantic points file {obj_sem_pts_file} has unexpected shape {loaded_obj_sem.shape}. Expected Nx4.")
    except FileNotFoundError:
        print(f"Warning: Object semantic points file not found at {obj_sem_pts_file}.")
    except Exception as e:
        print(f"Error loading object semantic points {obj_sem_pts_file}: {e}")


    # Load ORIGINAL Bounding Boxes WITH LABELS
    try:
        loaded_bbox = np.load(bbox_file)
        # Expecting format: [x, y, center_z, w, l, h, yaw, label_id] -> Nx8
        if loaded_bbox.ndim == 2 and loaded_bbox.shape[1] == 8:
            gt_bboxes_data_original = loaded_bbox # Store N x 8 array
            # Extract labels -> ensure they are integers
            gt_box_labels = gt_bboxes_data_original[:, 7].astype(int)
            print(f"Loaded original bounding boxes with labels: {gt_bboxes_data_original.shape}")
            print(f"Extracted {len(gt_box_labels)} box labels.")
        # Handle cases with incorrect shape or empty file
        elif loaded_bbox.size == 0:
             print(f"Info: Bounding box file {bbox_file} is empty.")
        else:
             print(f"Warning: Bounding box file {bbox_file} has unexpected shape {loaded_bbox.shape}. Expected Nx8. Skipping boxes.")
             gt_bboxes_data_original = np.empty((0, 8)) # Ensure it's empty for later checks

    except FileNotFoundError:
        print(f"Warning: Bounding box file not found at {bbox_file}.")
    except Exception as e:
        print(f"Error loading bounding boxes {bbox_file}: {e}")


    # --- Exit if points not loaded ---
    if object_points.shape[0] == 0:
        print("Error: Failed to load points. Cannot perform overlap check or visualization.")
        return

    # --- Prepare Point Cloud (assign initial colors) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points)
    # Keep track of original labels before potential modification
    original_point_labels = object_labels.copy()
    # We will calculate final colors later, after potential disambiguation
    # colors = np.array([CLASS_COLOR_MAP.get(label[0], DEFAULT_COLOR) for label in object_labels], dtype=np.float32)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Created point cloud geometry with {object_points.shape[0]} points.")

    # --- Prepare Bounding Boxes (Apply Transformations & Coloring) ---
    bboxes_list = []
    o3d_bboxes = []
    # Make a copy to modify geometry, keep original for labels
    gt_bboxes_data_modified = gt_bboxes_data_original.copy()

    if gt_bboxes_data_modified.shape[0] > 0:
        # --- Apply the transformations ---
        # Using indices 0-6, leaves label in column 7 untouched in the copy
        gt_bboxes_data_modified[:, 6] += np.pi / 2.0
        # ... (other transformations like Z offset, scaling) ...
        gt_bboxes_data_modified[:, 2] -= 0.1
        gt_bboxes_data_modified[:, 3:6] *= 1.1
        print("Applied transformations to bounding box geometry.")

        # --- Create Open3D Boxes using MODIFIED geometry & ORIGINAL labels ---
        for i in range(gt_bboxes_data_modified.shape[0]):
            center = gt_bboxes_data_modified[i, 0:3]
            extent = gt_bboxes_data_modified[i, 3:6]
            yaw = gt_bboxes_data_modified[i, 6]
            # Get label ID directly from extracted gt_box_labels array
            try:
                label_id = gt_box_labels[i]
            except IndexError:
                 print(f"Warning: Box index {i} out of bounds for gt_box_labels. Using Unknown.")
                 label_id = 15 # Unknown label

            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
            bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
            # Set Box Color based on Label ID
            box_color = CLASS_COLOR_MAP.get(label_id, DEFAULT_COLOR)
            bbox.color = tuple(box_color)
            bboxes_list.append(bbox)
            o3d_bboxes.append(bbox)
        print(f"Processed {len(bboxes_list)} bounding boxes and assigned colors based on labels.")
    else:
         print("No bounding boxes loaded or processed.")


    # --- Find Points in Multiple Bounding Boxes ---
    overlapping_point_indices = np.array([], dtype=int) # Initialize empty
    point_counts = np.zeros(object_points.shape[0], dtype=int)
    point_to_overlapping_boxes = defaultdict(list) # Maps point_idx -> [box_idx1, box_idx2, ...]

    if object_points.shape[0] > 0 and len(o3d_bboxes) > 0:
        print("\n--- Checking for points in multiple boxes ---")
        # Iterate through each bounding box
        for i, bbox in enumerate(o3d_bboxes):
            indices_in_box = bbox.get_point_indices_within_bounding_box(pcd.points)
            if indices_in_box:
                point_counts[indices_in_box] += 1
                # Store which box this was for each point inside it
                for point_idx in indices_in_box:
                    point_to_overlapping_boxes[point_idx].append(i)

        # Find indices where the count is greater than 1
        overlapping_point_indices = np.where(point_counts > 1)[0]

        """# Filter the dictionary
        point_to_overlapping_boxes = {
            pt_idx: box_indices
            for pt_idx, box_indices in point_to_overlapping_boxes.items()
            if pt_idx in overlapping_point_indices
        }"""
        print(f"Filtering dictionary for {len(overlapping_point_indices)} overlapping points...")
        # Create the filtered dictionary by iterating ONLY over the overlapping indices
        # and looking them up in the original dictionary (which is fast).
        point_to_overlapping_boxes_filtered = {
            pt_idx: point_to_overlapping_boxes[pt_idx]  # Fast lookup
            for pt_idx in overlapping_point_indices  # Iterate only L times (626k)
        }
        # Assign the filtered dictionary back
        point_to_overlapping_boxes = point_to_overlapping_boxes_filtered
        print("Finished filtering dictionary.")
        # --- END OPTIMIZED FILTERING ---

        if len(overlapping_point_indices) > 0:
            print(f"Found {len(overlapping_point_indices)} points belonging to 2 or more bounding boxes.")
        else:
            print("No points found in multiple bounding boxes.")
    else:
        print("Skipping overlap check: No points or no boxes loaded.")

    # --- Apply Disambiguation Strategy ---
    # Start with the original labels loaded from the semantic points file
    final_point_labels = original_point_labels.copy()

    # Only apply strategy if there are overlaps AND a valid strategy is chosen
    if len(overlapping_point_indices) > 0 and strategy and strategy != 'highlight':
        if gt_box_labels.size == 0:
             print("Warning: Cannot apply strategy - box labels were not loaded.")
        else:
            if strategy == 'nearest_center':
                final_point_labels = assign_label_by_nearest_center(
                    overlapping_point_indices, point_to_overlapping_boxes, object_points,
                    gt_bboxes_data_modified, gt_box_labels, final_point_labels # Pass MODIFIED boxes here as centers might change if Z is adjusted
                )
            elif strategy == 'nearest_surface':
                 final_point_labels = assign_label_by_nearest_surface(
                    overlapping_point_indices, point_to_overlapping_boxes, object_points,
                    o3d_bboxes, gt_box_labels, final_point_labels # Pass o3d_bboxes list
                )
            elif strategy == 'knn_majority':
                 final_point_labels = assign_label_by_knn_majority(
                    overlapping_point_indices, point_counts, object_points,
                    final_point_labels # Pass current labels
                )

            elif strategy == 'gmm':
                final_point_labels = assign_label_by_gmm(
                    overlap_indices=overlapping_point_indices,
                    point_to_boxes=point_to_overlapping_boxes,
                    points=object_points,
                    original_labels=original_point_labels,  # Pass original labels for training GMM
                    point_counts=point_counts,  # Pass counts to find non-overlapping
                    box_labels=gt_box_labels,  # Pass box labels map
                    n_components=1  # Start with 1 component per class
                )

            elif strategy == 'local_kmeans':
                final_point_labels = assign_label_by_local_kmeans(
                    overlap_indices=overlapping_point_indices,
                    point_to_boxes=point_to_overlapping_boxes,
                    points=object_points,
                    boxes_data=gt_bboxes_data_original,  # Use ORIGINAL centers for mapping
                    box_labels=gt_box_labels,
                    current_labels=final_point_labels
                )
            elif strategy == 'local_dbscan':
                final_point_labels = assign_label_by_local_dbscan(
                    overlap_indices=overlapping_point_indices,
                    point_to_boxes=point_to_overlapping_boxes,
                    points=object_points,
                    original_labels=original_point_labels,  # Needed for mapping heuristic
                    current_labels=final_point_labels,
                    eps=0.2,  # Example: Tune this based on point density/scale (e.g., 0.1, 0.3, 0.5)
                    min_samples=15  # Example: Tune this (e.g., 5, 10, 20)
                )
            else:
                print(f"Warning: Unknown strategy '{strategy}'. No disambiguation applied.")

            print(f"Applied disambiguation strategy: {strategy}")
    elif strategy == 'highlight':
         print("Highlighting overlaps (no disambiguation applied).")
         # Apply highlight color IF overlaps exist
         if len(overlapping_point_indices) > 0:
              temp_colors_for_highlight = np.array([CLASS_COLOR_MAP.get(lbl[0], DEFAULT_COLOR) for lbl in final_point_labels])
              temp_colors_for_highlight[overlapping_point_indices] = OVERLAP_COLOR
              pcd.colors = o3d.utility.Vector3dVector(temp_colors_for_highlight)
              print(f"Highlighted {len(overlapping_point_indices)} overlapping points in pink.")


    # --- Set Final Point Cloud Colors ---
    # This happens AFTER disambiguation OR uses highlighting if strategy='highlight'
    if strategy != 'highlight': # If we applied a strategy or did nothing, use final labels
         print("Setting final point colors based on original or resolved labels.")
         final_colors = np.array([CLASS_COLOR_MAP.get(label[0], DEFAULT_COLOR) for label in final_point_labels], dtype=np.float32)
         pcd.colors = o3d.utility.Vector3dVector(final_colors)
    elif len(overlapping_point_indices) == 0 and strategy == 'highlight':
         # If highlight was requested but there were no overlaps, still need to set colors
         print("Setting final point colors (no overlaps found for highlighting).")
         final_colors = np.array([CLASS_COLOR_MAP.get(label[0], DEFAULT_COLOR) for label in final_point_labels], dtype=np.float32)
         pcd.colors = o3d.utility.Vector3dVector(final_colors)
    # Else: colors were already set by the highlight logic above


    # --- Visualize ---
    geometries_to_draw = []
    if object_points.shape[0] > 0:
        geometries_to_draw.append(pcd) # Add point cloud with final colors
    if bboxes_list:
        geometries_to_draw.extend(bboxes_list) # Add colored boxes

    if not geometries_to_draw:
        print("No geometries available to draw.")
        return

    print("\n--- Opening Visualization Window ---")
    if strategy and strategy != 'highlight':
         print(f"Displaying points with overlaps resolved by '{strategy}'.")
    elif strategy == 'highlight':
         print("Displaying points with overlaps highlighted in pink.")
    else:
         print("Displaying original point labels (no disambiguation strategy applied).")
    print("Boxes are colored by their semantic label.")
    print("Press 'Q' or close the window to exit.")
    o3d.visualization.draw_geometries(geometries_to_draw,
                                      window_name=f"Strategy: {strategy} - {os.path.basename(data_directory)}",
                                      width=1280, height=720)
    print("--- Visualization window closed ---")

# ==================================================
# --- User Configuration ---
# ==================================================
# !!!! IMPORTANT !!!!
# SET THIS PATH: Point it to the specific directory containing the .npy files
#                (e.g., the '1bb41855cb724ae6980bababdbd1865e2' folder from your image)
# Example Linux/macOS: '/home/user/data/nuscenes_subset/sce...11/1bb41855cb724ae6980bababdbd1865e2'
# Example Windows: r'C:\data\nuscenes_subset\sce...11\1bb41855cb

DATA_DIR = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/gt/all_sensors_icp_boxes_labels/scene-0044384af3d8494e913fb8b14915239e-11/1bb41855cb724ae6980bababbd1865e2'
# ==================================================


if __name__ == "__main__":
    # Basic check if the path might be the placeholder
    if 'path/to/your/data' in DATA_DIR or not os.path.isdir(DATA_DIR):
        print("="*60)
        print("ERROR: Please update the 'DATA_DIR' variable in the script!")
        print(f"       Set it to the directory containing the frame_0_...npy files.")
        print(f"       Current value: '{DATA_DIR}'")
        print("="*60)
    else:
        visualize_object_data(DATA_DIR, highlight_overlap=True)
        # --- CHOOSE STRATEGY ---
        # Options: None, 'highlight', 'nearest_center', 'nearest_surface', 'knn_majority'
        # Set to None to just show original point colors and labeled boxes
        # Set to 'highlight' to show overlaps in pink
        # Set to a strategy name to apply it and show the results
        # SELECTED_STRATEGY = 'highlight'
        # SELECTED_STRATEGY = 'nearest_center'
        # SELECTED_STRATEGY = 'knn_majority'
        # SELECTED_STRATEGY = 'nearest_surface'
        # SELECTED_STRATEGY = 'gmm'
        # SELECTED_STRATEGY = 'local_kmeans'
        SELECTED_STRATEGY = 'local_dbscan'

        visualize_object_data_refined(DATA_DIR, strategy=SELECTED_STRATEGY)
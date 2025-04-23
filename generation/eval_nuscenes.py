import argparse
import numpy as np
# from collections import defaultdict # Not strictly needed anymore
from sklearn.metrics import confusion_matrix
import warnings # Added for handling division by zero
# import matplotlib.pyplot as plt # Not used
import plotly.graph_objects as go # Import Plotly
from plotly.subplots import make_subplots # For side-by-side 3D plots


def count_labels(arr, num_classes):
    """Counts occurrences of each label from 0 to num_classes-1 in a numpy array."""
    labels, counts = np.unique(arr, return_counts=True)
    label_counts = dict(zip(labels, counts))
    full_counts = {i: label_counts.get(i, 0) for i in range(num_classes)}
    return full_counts

def get_points_labels(arr, background_label=0):
    """Extracts coordinates and labels of non-background voxels."""
    coords = np.argwhere(arr != background_label)
    if coords.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    labels = arr[coords[:, 0], coords[:, 1], coords[:, 2]]
    # Assuming array index order is (X, Y, Z) for plotting
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return x, y, z, labels


def visualize_3d_occupancy(arr1, arr2, title1="Baseline", title2="Own", num_classes=18, background_label=17,
                           main_title="3D Semantic Occupancy", overlay_debug=False, debug_color1='blue', debug_color2='red'):
    """
    Visualize two 3D occupancy grids using Plotly.
    Can show side-by-side semantic plots or a single overlay plot with fixed colors for debugging.
    """

    if arr1.ndim != 3 or arr2.ndim != 3:
        print("Warning: Visualization expects 3D arrays. Skipping visualization.")
        return

    print(f"Extracting non-background (label != {background_label}) points for 3D visualization...")
    # We always need the points, get labels too for the default mode
    x1, y1, z1, labels1 = get_points_labels(arr1, background_label)
    x2, y2, z2, labels2 = get_points_labels(arr2, background_label)

    if x1.size == 0 and x2.size == 0:
        print(f"Warning: Both arrays contain only background labels (label={background_label}). Nothing to visualize.")
        return
    print(f"Found {x1.size} points for Baseline, {x2.size} points for Own.")

    # --- Choose Visualization Mode ---
    if overlay_debug:
        # --- Overlay Mode (Fixed Colors) ---
        print(f"Creating overlay plot: {title1} ({debug_color1}), {title2} ({debug_color2})")
        fig = go.Figure()

        marker_base = dict(size=2, opacity=0.7) # Base marker style

        # Add Baseline trace with fixed color
        if x1.size > 0:
            fig.add_trace(go.Scatter3d(
                x=x1, y=y1, z=z1,
                mode='markers',
                marker=dict(**marker_base, color=debug_color1), # Fixed color
                name=title1 # For legend
            ))

        # Add Own trace with fixed color
        if x2.size > 0:
             fig.add_trace(go.Scatter3d(
                x=x2, y=y2, z=z2,
                mode='markers',
                marker=dict(**marker_base, color=debug_color2), # Fixed color
                name=title2 # For legend
            ))

        # Update layout for single plot with legend
        fig.update_layout(
            title_text=main_title + " (Overlay Debug)",
            height=800,
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=40),
            legend_title_text='Dataset'
        )

    else:
        # --- Side-by-Side Mode (Semantic Colors) ---
        print("Creating side-by-side semantic plots.")
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]], subplot_titles=(title1, title2))

        cmin_val = 0 # Show class 0 if present and not background
        cmax_val = num_classes - 2 if background_label == num_classes - 1 else num_classes - 1 # Adjust max if background is highest label
        if background_label == 0: cmin_val = 1 # Don't include 0 in color range if it's background


        marker_props = dict(
            size=2, opacity=0.8, color=None, colorscale='viridis',
            cmin=cmin_val, cmax=cmax_val, # Set color range for non-background labels
            colorbar=dict(title='Class Label', thickness=15, x=0.45 if x1.size > 0 else 1.0)
        )

        if x1.size > 0:
            marker_props['color'] = labels1
            trace1 = go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=marker_props, name=title1)
            fig.add_trace(trace1, row=1, col=1)
        else:
            print(f"No non-background points to plot for {title1}.")
            fig.add_annotation(text=f"No data (label != {background_label})", xref="paper", yref="paper", x=0.2, y=0.5, showarrow=False, row=1, col=1)

        if x2.size > 0:
            marker_props2 = marker_props.copy(); marker_props2['color'] = labels2
            if x1.size > 0: marker_props2['showscale'] = False
            else: marker_props2['colorbar']['x'] = 1.0
            trace2 = go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=marker_props2, name=title2)
            fig.add_trace(trace2, row=1, col=2)
        else:
             print(f"No non-background points to plot for {title2}.")
             fig.add_annotation(text=f"No data (label != {background_label})", xref="paper", yref="paper", x=0.8, y=0.5, showarrow=False, row=1, col=2)

        # Update layout for side-by-side plots
        fig.update_layout(
            title_text=main_title + " Comparison",
            height=700,
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    # --- End Visualization Mode Choice ---

    print("Displaying interactive 3D visualization. This might take a moment to render.")
    print("Close the browser tab/window opened by Plotly to continue metric calculation.")
    fig.show()


def compute_metrics(gt_baseline, gt_own, num_classes=18):
    """Compute accuracy, per-class accuracy and IoU"""
    gt_baseline_flat = gt_baseline.flatten()
    gt_own_flat = gt_own.flatten()

    # --- IMPORTANT: Check unique values to understand the data ---
    # print("Unique values in baseline (before mask):", np.unique(gt_baseline_flat))
    # print("Unique values in own (before mask):", np.unique(gt_own_flat))
    # ---

    mask = (gt_baseline_flat >= 0) & (gt_baseline_flat < num_classes) # Mask: ignore -1 AND values >= num_classes
    if not np.any(mask): # Check if there are any valid labels
        print("Warning: No valid labels found after masking (Range 0 to {}). Cannot compute metrics.".format(num_classes-1))
        return None, None, None, None, None # Return None for all metrics

    baseline = gt_baseline_flat[mask]
    own = gt_own_flat[mask]

    # --- IMPORTANT: Check unique values after masking ---
    # print("Unique values in baseline (after mask):", np.unique(baseline))
    # print("Unique values in own (after mask):", np.unique(own))
    # ---

    if baseline.size == 0:
         print("Warning: Arrays are empty after masking. Cannot compute metrics.")
         return None, None, None, None, None

    # 1. Overall Accuracy
    accuracy = np.mean(baseline == own)

    # 2. Confusion Matrix (Needed for Per-class Accuracy and IoU)
    # Ensure labels go from 0 to num_classes-1 for confusion_matrix
    labels = list(range(num_classes))
    conf_mat = confusion_matrix(baseline, own, labels=labels)
    # print("\nConfusion Matrix:\n", conf_mat) # Optional: print confusion matrix

    # 3. Per-Class Accuracy
    # Sum of correct predictions for each class / total instances of that class (diagonal / row sum)
    with warnings.catch_warnings(): # Suppress division by zero warnings for classes not present in baseline
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_class_acc = np.diag(conf_mat) / conf_mat.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc) # Replace NaN (from 0/0) with 0

    # Mean Per-Class Accuracy (often more informative than overall accuracy on imbalanced datasets)
    mean_per_class_acc = np.mean(per_class_acc)

    # 4. Intersection over Union (IoU) or Jaccard Index
    # Intersection = Diagonal element (True Positives)
    # Union = Row sum + Column sum - Diagonal element (TP + FP + FN)
    intersection = np.diag(conf_mat)
    union = conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - intersection
    with warnings.catch_warnings(): # Suppress division by zero warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        iou_per_class = intersection / union # Changed variable name for clarity
    iou_per_class = np.nan_to_num(iou_per_class) # Replace NaN (from 0/0) with 0

    # Mean IoU (mIoU) - a standard metric for segmentation
    mean_iou = np.mean(iou_per_class) # Calculate mean over the per-class IoUs

    return accuracy, mean_per_class_acc, mean_iou, per_class_acc, iou_per_class




def main(args):
    gt_path_base = args.dataroot_baseline
    gt_path_own = args.dataroot_own
    keyname_base = 'semantics'
    keyname_own = 'occupancy'
    num_classes = 18

    print(f"Loading baseline data from: {gt_path_base}")
    print(f"Loading own data from: {gt_path_own}")

    try:
        gt_npz_base = np.load(gt_path_base)
        gt_npz_own = np.load(gt_path_own)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return
    except Exception as e:
        print(f"An error occurred loading NPZ files: {e}")
        return

    print(f"Using key '{keyname_base}' to access data within base NPZ file.")

    # --- Access the array using the specified key ---
    if keyname_base not in gt_npz_base:
        print(f"Error: Key '{keyname_base}' not found in baseline file: {gt_path_base}")
        print(f"Available keys: {list(gt_npz_base.keys())}")
        gt_npz_base.close()
        gt_npz_own.close()
        return

    if keyname_own not in gt_npz_own:
        print(f"Error: Key '{keyname_own}' not found in own data file: {gt_path_own}")
        print(f"Available keys: {list(gt_npz_own.keys())}")
        gt_npz_base.close()
        gt_npz_own.close()
        return

    gt_base_array = gt_npz_base[keyname_base]
    gt_own_array = gt_npz_own[keyname_own]

    gt_own_array = np.transpose(gt_own_array, (1, 0, 2))
    gt_own_array = np.flip(gt_own_array, axis=1)

    # --- Data loaded ---
    print("\nBaseline Data Info:")
    print(f"  Shape: {gt_base_array.shape}")
    print(f"  Data type: {gt_base_array.dtype}")

    print("\nOwn Data Info:")
    print(f"  Shape: {gt_own_array.shape}")
    print(f"  Data type: {gt_own_array.dtype}")

    if gt_base_array.shape != gt_own_array.shape:
        print("\nError: Baseline and own data arrays have different shapes. Cannot compare.")
        gt_npz_base.close()
        gt_npz_own.close()
        return

    # --- Count Labels Step ---
    print("\n--- Voxel Label Counts ---")
    baseline_counts = count_labels(gt_base_array, num_classes)
    own_counts = count_labels(gt_own_array, num_classes)

    # Determine max width for alignment
    max_label_width = len(str(num_classes - 1))
    # Find max count length for formatting - check across both dicts
    max_count_base = max(len(str(c)) for c in baseline_counts.values()) if baseline_counts else 1
    max_count_own = max(len(str(c)) for c in own_counts.values()) if own_counts else 1
    max_count_width = max(max_count_base, max_count_own, len("Baseline"), len("Own Data"))

    header = f"  {'Label':<{max_label_width}} | {'Baseline':>{max_count_width}} | {'Own Data':>{max_count_width}}"
    separator = f"  {'-' * max_label_width}-+-{'-' * max_count_width}-+-{'-' * max_count_width}"
    print(header)
    print(separator)

    total_baseline = 0
    total_own = 0
    for i in range(num_classes):
        base_count = baseline_counts.get(i, 0)
        own_count = own_counts.get(i, 0)
        print(
            f"  {i:<{max_label_width}} | {base_count:>{max_count_width},} | {own_count:>{max_count_width},}")  # Added comma formatting
        total_baseline += base_count
        total_own += own_count

    print(separator.replace("-", "="))  # Footer separator
    print(
        f"  {'Total':<{max_label_width}} | {total_baseline:>{max_count_width},} | {total_own:>{max_count_width},}")
    print("--------------------------")
    # --- End Count Labels Step ---

    # --- Visualization Step ---
    if args.visualize or args.overlay_debug:  # Trigger visualization if either flag is set
        print("\nAttempting 3D visualization...")
        visualize_3d_occupancy(
            gt_base_array, gt_own_array,
            title1="Baseline", title2="Own",
            num_classes=num_classes,  # Pass total range needed for visualization checks
            background_label=args.background_label,
            main_title=f"3D Occupancy",
            overlay_debug=args.overlay_debug  # Pass the overlay flag
        )
    else:
        print("\n3D Visualization skipped (use --visualize or --overlay_debug flag to enable).")
    # --- End Visualization Step ---

    # --- Call compute_metrics ---
    print("\nComputing Metrics...")
    num_classes = 17  # Get num_classes from args
    results = compute_metrics(gt_base_array, gt_own_array, num_classes=num_classes)

    if results:
        accuracy, mean_per_class_acc, mean_iou, per_class_acc, iou_per_class = results
        print(f"\n--- Results (Comparing Own vs Baseline) ---")
        print(f"Number of Classes: {num_classes}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Mean Per-Class Accuracy: {mean_per_class_acc:.4f}")
        print(f"Mean IoU (mIoU): {mean_iou:.4f}")  # This is your mIoU

        # Optionally print per-class details
        print("\nPer-Class Metrics:")
        # Determine max width for class number alignment
        max_class_width = len(str(num_classes - 1))
        print(f"  {'Class':<{max_class_width + 1}} | {'Accuracy':<10} | {'IoU':<10}")  # Header
        print(f"  {'-' * (max_class_width + 1)} | {'-' * 10} | {'-' * 10}")  # Separator
        for i in range(num_classes):
            print(
                f"  {i:<{max_class_width + 1}} | {per_class_acc[i]:<10.4f} | {iou_per_class[i]:<10.4f}")  # This is IoU per class

        print("--------------------------------------------")
    else:
        print("Metrics computation failed (check warnings above).")

        # Close the npz files
    gt_npz_base.close()
    gt_npz_own.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot_baseline', type=str)
    parser.add_argument('--dataroot_own', type=str)
    parser.add_argument('--scene', type=str, default='scene-0001')
    parser.add_argument('--visualize', action='store_true', help="Enable side-by-side visualization of a Z-slice.")
    parser.add_argument('--background_label', type=int, default=17,
                        help="Specify the integer label treated as background (not visualized).")
    parser.add_argument('--overlay_debug', action='store_true',
                        help="Enable overlay 3D visualization with fixed colors for debugging.")

    args = parser.parse_args()

    main(args)



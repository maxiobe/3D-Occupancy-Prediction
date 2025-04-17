import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix




def compute_metrics(gt_baseline, gt_own, num_classes=17):
    """Compute accuracy, per-class accuracy and IoU"""
    gt_baseline_flat = gt_baseline.flatten()
    gt_own_flat = gt_own.flatten()

    mask = gt_baseline_flat >= 0  # ignore -1 or undefined
    baseline = gt_baseline_flat[mask]
    own = gt_own_flat[mask]

    accuracy = np.mean(baseline == own)




def main(args):
    gt_path_base = args.dataroot_baseline
    gt_path_own = args.dataroot_own

    gt_npz_base = np.load(gt_path_base)
    gt_npz_own = np.load(gt_path_own)

    print("Baseline:")
    print(gt_npz_base.shape)
    print(gt_npz_base)

    print("Own data:")
    print(gt_npz_own.shape)
    print(gt_npz_own)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot_baseline', type=str)
    parser.add_argument('--dataroot_own', type=str)
    parser.add_argument('--scene', type=str, default='scene-0001')

    args = parser.parse_args()

    main(args)



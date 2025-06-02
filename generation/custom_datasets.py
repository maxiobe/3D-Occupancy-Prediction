import numpy as np
from typing import List

class InMemoryDataset:
    """
    A custom dataset class for KISS-ICP that uses lidar scans already loaded into memory.
    """
    def __init__(self, lidar_scans, gt_relative_poses, timestamps=None, initial_guess=True, sequence_id="in_memory_seq", log_dir="."):
        """
        Initializes the dataset.

        Args:
            lidar_scans (list): A list where each element is a NumPy array representing a lidar scan (e.g., Nx3 or NxD).
            timestamps (list, optional): A list of timestamps corresponding to each scan.
                                         If None, dummy timestamps (0.0, 0.1, 0.2...) are generated,
                                         which might affect deskewing if enabled.
            sequence_id (str, optional): An identifier for the sequence, used for naming output files.
        """
        if not isinstance(lidar_scans, list) or not lidar_scans:
            raise ValueError("lidar_scans must be a non-empty list.")
        self.scans = lidar_scans

        if timestamps is None:
            # Generate sequential timestamps if none provided (e.g., assuming 10Hz)
            # Warning: Deskewing might not work optimally without real timestamps.
            self.timestamps = [float(i) * 0.1 for i in range(len(self.scans))]
            print("INFO: No timestamps provided to InMemoryDataset. Generating dummy timestamps (0.0, 0.1, ...).")
        elif len(timestamps) != len(self.scans):
            raise ValueError("Number of timestamps must match the number of scans.")
        else:
            self.timestamps = [float(t) for t in timestamps] # Ensure timestamps are floats

        # --- Attributes potentially used by OdometryPipeline ---
        # Provide a sequence ID for output file naming
        self.sequence_id = sequence_id
        # Add a dummy data_dir attribute, as OdometryPipeline might check for it
        self.log_dir = log_dir
        # Indicate no ground truth poses are available
        # (OdometryPipeline checks using hasattr, so not defining it works too)
        self.gt_poses = gt_relative_poses
        self.initial_guess = initial_guess

    def __len__(self):
        """Returns the number of scans in the dataset."""
        return len(self.scans)

    def __getitem__(self, index):
        """
        Returns the scan and its timestamp array, ensuring correct format for KISS-ICP.

        Args:
            index (int): The index of the scan to retrieve.

        Returns:
            tuple: (processed_frame, timestamp_array)
                   processed_frame: NumPy array (Nx3), float64, C-contiguous.
                   timestamp_array: NumPy array containing the timestamp(s) (float64).
        """
        if not 0 <= index < len(self.scans):
            raise IndexError(f"Index {index} is out of range for {len(self.scans)} scans.")

        # --- Get original data ---
        # Assuming self.scans stores arrays of shape (Features, N)
        original_frame_feat_n = self.scans[index]
        # Assuming self.timestamps is a list of floats from __init__
        timestamp = self.timestamps[index]

        # --- Validate and Prepare Point Cloud Frame ---

        # Input Validation
        if not isinstance(original_frame_feat_n, np.ndarray) or original_frame_feat_n.ndim != 2:
            print(
                f"Warning: Scan at index {index} is not a 2D numpy array (shape: {getattr(original_frame_feat_n, 'shape', 'N/A')}). Returning empty frame.")
            empty_frame = np.empty((0, 3), dtype=np.float64)
            # Return empty frame and correctly formatted timestamp array
            timestamp_array = np.array([timestamp], dtype=np.float64)
            return empty_frame, timestamp_array

        # Transpose to (N, Features) format
        frame_n_feat = original_frame_feat_n

        # Extract XYZ and handle empty/insufficient feature cases
        if frame_n_feat.shape[0] == 0:  # Handle empty cloud after transpose
            raw_frame_xyz = np.empty((0, 3), dtype=np.float64)
        elif frame_n_feat.shape[1] >= 3:
            # Extract XYZ coordinates (first 3 features)
            raw_frame_xyz = frame_n_feat[:, :3]
        else:  # Handle case with points but less than 3 features
            print(
                f"Warning: Scan at index {index} has less than 3 features (shape: {frame_n_feat.shape}). Cannot extract XYZ. Returning empty frame.")
            raw_frame_xyz = np.empty((0, 3), dtype=np.float64)
            # Return empty frame and correctly formatted timestamp array
            timestamp_array = np.array([timestamp], dtype=np.float64)
            return raw_frame_xyz, timestamp_array

        # Ensure Correct Data Type (float64 for Vector3dVector)
        if raw_frame_xyz.dtype != np.float64:
            processed_frame = raw_frame_xyz.astype(np.float64)
        else:
            processed_frame = raw_frame_xyz

        # Ensure C-Contiguity for pybind11 binding
        if not processed_frame.flags['C_CONTIGUOUS']:
            processed_frame = np.ascontiguousarray(processed_frame)

        # --- Prepare Timestamp Array ---
        # Convert the single float timestamp into a 1-element NumPy array.
        # KISS-ICP's preprocess expects an object with .ravel()
        timestamp_array = np.array([timestamp], dtype=np.float64)

        # --- Return the processed frame and timestamp array ---
        return processed_frame, timestamp_array

    def get_frames_timestamps(self):
        """
        Returns all timestamps; used for saving poses (e.g., TUM format).
        Required if saving in TUM format.
        """
        # OdometryPipeline handles slicing itself based on jump/n_scans,
        # so we return the full list of timestamps corresponding to self.scans
        return np.array(self.timestamps, dtype=np.float64)

class InMemoryDatasetMapMOS:
    """
    A custom dataset class for MapMOS that uses lidar scans, timestamps,
    and labels already loaded into memory.
    """

    def __init__(self,
                 lidar_scans: list,  # List of NxFeatures NumPy arrays
                 # Per-scan timestamps are crucial for MapMOS's 4D convolutions
                 scan_timestamps: list,  # List of floats (one per scan)
                 # Labels for segmentation (0 for static, 1 for moving)
                 labels_per_scan: list,  # List of Nx1 NumPy arrays (int32)
                 sequence_id: str = "in_memory_mapmos_seq",
                 # gt_poses are not directly used by __getitem__ for MapMOS in the
                 # same way as TruckScenesDataset, but good to have for context/evaluation later.
                 gt_global_poses: list = None  # Optional: List of 4x4 global poses
                 ):

        if not (len(lidar_scans) == len(scan_timestamps) == len(labels_per_scan)):
            raise ValueError(
                "Input lists (lidar_scans, scan_timestamps, labels_per_scan) must have the same length.")
        if gt_global_poses is not None and len(lidar_scans) != len(gt_global_poses):
            raise ValueError("If gt_global_poses is provided, it must match the length of lidar_scans.")
        if not lidar_scans:
            raise ValueError("Input lists cannot be empty.")

        self.scans = lidar_scans
        self.scan_timestamps_data = [float(t) for t in scan_timestamps]  # Ensure float
        self.labels_data = labels_per_scan  # This is your list of label arrays
        self.gt_global_poses_data = gt_global_poses

        self.sequence_id = sequence_id
        # Add other attributes MapMOSPipeline might expect from a dataset object
        # self.log_dir = log_dir # If needed by the pipeline for saving outputs

    def __len__(self):
        """Returns the number of scans in the dataset."""
        return len(self.scans)

    def __getitem__(self, index):
        """
        Returns the points, per-point timestamps (typically zeros for MapMOS like in TruckScenesDataset),
        and labels for the given index.

        Args:
            index (int): The index of the scan to retrieve.

        Returns:
            tuple: (points, timestamps_for_points, labels)
                   points: NumPy array (Nx3), float64.
                   timestamps_for_points: NumPy array (N,), float64. MapMOS itself will
                                         likely use the per-scan timestamp (self.scan_timestamps_data[index])
                                         to create the 4th dimension for its convolutions.
                                         The original TruckScenesDataset returned np.zeros here.
                   labels: NumPy array (Nx1), int32.
        """
        if not 0 <= index < len(self.scans):
            raise IndexError(f"Index {index} is out of range for {len(self.scans)} scans.")

        # --- Point Cloud ---
        points_raw = self.scans[index]
        if points_raw.ndim != 2 or points_raw.shape[1] < 3:
            # If your raw_pc_list stores (Features, N), then transpose
            if points_raw.ndim == 2 and points_raw.shape[0] >= 3 and points_raw.shape[0] < points_raw.shape[
                1]:  # Heuristic for (F,N)
                points_raw = points_raw.T  # Now (N, F)
            else:
                raise ValueError(
                    f"Scan at index {index} (shape {points_raw.shape}) is not a valid 2D numpy array with at least 3 features (expected NxD).")

        points = points_raw[:, :3].astype(np.float64)  # Extract XYZ, ensure float64

        # --- Timestamps for points ---
        # Mimicking TruckScenesDataset which returns zeros here.
        # MapMOS likely uses the per-scan timestamp (self.scan_timestamps_data[index])
        # when constructing its 4D input tensor.
        timestamps_for_points = np.zeros(len(points), dtype=np.float64)

        # --- Labels ---
        labels = self.labels_data[index]  # Should be (N, 1) int32
        if not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.shape[1] != 1:
            raise ValueError(
                f"Labels at index {index} (shape {labels.shape}) must be a 2D NumPy array with shape (N, 1).")
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                f"Mismatch in number of points ({points.shape[0]}) and labels ({labels.shape[0]}) at index {index}.")
        labels = labels.astype(np.int32)

        return points, timestamps_for_points, labels

    # --- Other methods potentially useful or expected by the pipeline ---
    @property
    def timestamps(self) -> List[float]:  # Matches kiss_icp InMemoryDataset
        """Returns the list of per-scan timestamps."""
        return self.scan_timestamps_data

    # If MapMOS needs access to ground truth poses (e.g., for evaluation or if it doesn't run odometry)
    # The MapMOS paper mentions using KISS-ICP for odometry, so it might not need GT poses
    # from the dataset object for its core processing.
    # def get_gt_poses(self) -> list: # Or np.ndarray
    #    return self.gt_global_poses_data
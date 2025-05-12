import numpy as np
from pathlib import Path # Good practice for paths, though less critical here
# Import OdometryPipeline if not already imported
from kiss_icp.pipeline import OdometryPipeline
# Import load_config if you haven't loaded it some other way
from kiss_icp.config import load_config

class InMemoryDataset:
    """
    A custom dataset class for KISS-ICP that uses lidar scans already loaded into memory.
    """
    def __init__(self, lidar_scans, timestamps=None, sequence_id="in_memory_seq"):
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
        self.data_dir = "."
        # Indicate no ground truth poses are available
        # (OdometryPipeline checks using hasattr, so not defining it works too)
        # self.has_gt = False # Or just don't define gt_poses

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

    # --- Methods/Attributes NOT needed for basic operation ---
    # def apply_calibration(self, poses): return poses # Only if calibration needed
    # gt_poses = None # Only if evaluation against ground truth needed
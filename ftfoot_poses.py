import numpy as np
import cv2
import pickle
import os
import sys
import tf.transformations
from geometry_msgs.msg import Quaternion
import argparse
import re
from tqdm import tqdm

# --- Add project root to sys.path if needed (optional, run from root preferred) ---
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# --- Camera Parameters (CRITICAL: Verify these match your setup) ---
CAMERA_HEIGHT = 0.409 + 0.1
CAMERA_X_OFFSET = 0.451
FX, FY, CX, CY = 935.2355857804463, 935.7905325732659, 656.1572332633887, 513.7144019593092
DIST_COEFFS = np.array([-0.08194476107782814, -0.06592640858415261, -0.0007043163003212235, 0.002577256982584405])
CAMERA_MATRIX = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

# --- Processing Parameters ---
MAX_FUTURE_POSES = 30  # How many poses into the future to project
DEFAULT_LINE_THICKNESS = 2 # Default trajectory line thickness

# --- 6DoF Helper Functions (Essential) ---

def quaternion_to_angle(q):
    """Convert a quaternion dict/object into Euler angles (roll, pitch, yaw) in radians."""
    try:
        if isinstance(q, dict):
            q_tuple = (q['qx'], q['qy'], q['qz'], q['qw'])
        elif isinstance(q, Quaternion):
             q_tuple = (q.x, q.y, q.z, q.w)
        else:
            raise TypeError("Input must be a dict or Quaternion object")

        # Ensure qw is positive for unique representation before conversion
        if q_tuple[3] < 0:
            q_tuple = (-q_tuple[0], -q_tuple[1], -q_tuple[2], -q_tuple[3])

        return list(tf.transformations.euler_from_quaternion(q_tuple, axes='sxyz')) # returns [roll, pitch, yaw]
    except Exception as e:
        print(f"Error converting quaternion {q} to angle: {e}")
        return [0.0, 0.0, 0.0] # Return neutral angles on error


def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles (N, 3) [roll, pitch, yaw] to rotation matrices (N, 3, 3). """
    if not isinstance(euler_angles, np.ndarray): euler_angles = np.array(euler_angles, dtype=np.float32)
    if len(euler_angles.shape) == 1: euler_angles = euler_angles.reshape(1, -1)

    cos = np.cos(euler_angles); sin = np.sin(euler_angles)
    batch_size = euler_angles.shape[0]
    zero = np.zeros(batch_size); one = np.ones(batch_size)

    R_x = np.stack([ np.stack([one, zero, zero], axis=-1),
                     np.stack([zero, cos[:, 0], -sin[:, 0]], axis=-1),
                     np.stack([zero, sin[:, 0], cos[:, 0]], axis=-1) ], axis=1)
    R_y = np.stack([ np.stack([cos[:, 1], zero, sin[:, 1]], axis=-1),
                     np.stack([zero, one, zero], axis=-1),
                     np.stack([-sin[:, 1], zero, cos[:, 1]], axis=-1) ], axis=1)
    R_z = np.stack([ np.stack([cos[:, 2], -sin[:, 2], zero], axis=-1),
                     np.stack([sin[:, 2], cos[:, 2], zero], axis=-1),
                     np.stack([zero, zero, one], axis=-1) ], axis=1)
    return np.matmul(np.matmul(R_z, R_y), R_x) # ZYX convention


def extract_euler_angles_from_se3_batch(tf3_matx):
    """Extract Euler angles [roll, pitch, yaw] from SE(3) matrices (N, 4, 4)."""
    rotation_matrices = tf3_matx[:, :3, :3]
    euler_angles = np.zeros((tf3_matx.shape[0], 3), dtype=tf3_matx.dtype)
    sy = np.sqrt(rotation_matrices[:, 0, 0]**2 + rotation_matrices[:, 1, 0]**2)
    singular = sy < 1e-6

    euler_angles[~singular, 0] = np.arctan2(rotation_matrices[~singular, 2, 1], rotation_matrices[~singular, 2, 2]) # Roll
    euler_angles[~singular, 1] = np.arctan2(-rotation_matrices[~singular, 2, 0], sy[~singular])                      # Pitch
    euler_angles[~singular, 2] = np.arctan2(rotation_matrices[~singular, 1, 0], rotation_matrices[~singular, 0, 0]) # Yaw
    euler_angles[singular, 0] = np.arctan2(-rotation_matrices[singular, 1, 2], rotation_matrices[singular, 1, 1]) # Roll
    euler_angles[singular, 1] = np.arctan2(-rotation_matrices[singular, 2, 0], sy[singular])                       # Pitch
    euler_angles[singular, 2] = 0                                                                               # Yaw
    return euler_angles


def to_robot_numpy(Robot_frame_6dof, P_target_6dof):
    """Transforms target poses (N, 6) to be relative to the Robot_frame pose (1, 6)."""
    if not isinstance(Robot_frame_6dof, np.ndarray): Robot_frame_6dof = np.array(Robot_frame_6dof, dtype=np.float32)
    if not isinstance(P_target_6dof, np.ndarray): P_target_6dof = np.array(P_target_6dof, dtype=np.float32)
    if Robot_frame_6dof.shape != (1, 6): raise ValueError("Robot_frame must have shape (1, 6)")
    if len(P_target_6dof.shape) != 2 or P_target_6dof.shape[1] != 6: raise ValueError("P_target must have shape (N, 6)")

    batch_size = P_target_6dof.shape[0]

    # Create SE(3) matrices: T_world_robot (T1) and T_world_target (T2)
    T1 = np.identity(4, dtype=Robot_frame_6dof.dtype)
    T1[:3, :3] = euler_to_rotation_matrix(Robot_frame_6dof[:, 3:])
    T1[:3, 3] = Robot_frame_6dof[:, :3].flatten()

    T2 = np.identity(4, dtype=P_target_6dof.dtype)[np.newaxis, :, :].repeat(batch_size, axis=0)
    T2[:, :3, :3] = euler_to_rotation_matrix(P_target_6dof[:, 3:])
    T2[:, :3, 3] = P_target_6dof[:, :3]

    # Calculate T_robot_world (T1_inv)
    R1_T = T1[:3, :3].T
    t1 = T1[:3, 3]
    T1_inv = np.identity(4, dtype=T1.dtype)
    T1_inv[:3, :3] = R1_T
    T1_inv[:3, 3] = -R1_T @ t1

    # Calculate T_robot_target = T_robot_world @ T_world_target = T1_inv @ T2
    tf3_mat_robot_target = np.matmul(T1_inv, T2)

    # Extract relative pose
    transform_relative = np.zeros_like(P_target_6dof)
    transform_relative[:, :3] = tf3_mat_robot_target[:, :3, 3] # Translation part
    transform_relative[:, 3:] = extract_euler_angles_from_se3_batch(tf3_mat_robot_target) # Rotation part
    return transform_relative

# --- Projection Helper Functions ---

def project_points(xyz, camera_matrix, dist_coeffs):
    """Projects 3D points (X forward, Y left, Z up) to image coordinates."""
    if xyz.shape[0] == 0: return np.empty((0, 2))
    rvec = tvec = np.zeros(3)
    # Convert ROS coords (X fwd, Y left, Z up) to OpenCV coords (Z fwd, X right, Y down)
    xyz_cv = np.stack([-xyz[:, 1], -xyz[:, 2], xyz[:, 0]], axis=-1)

    valid_indices = xyz_cv[:, 2] > 1e-3 # Points in front of camera (OpenCV Z > 0)
    if not np.any(valid_indices): return np.full((xyz.shape[0], 2), np.nan)

    xyz_cv_valid = xyz_cv[valid_indices]
    try:
        uv, _ = cv2.projectPoints(xyz_cv_valid.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs)
        uv = uv.reshape(-1, 2)
        full_uv = np.full((xyz.shape[0], 2), np.nan)
        full_uv[valid_indices] = uv
        return full_uv
    except cv2.error as e:
        # print(f"Warning: cv2.projectPoints error: {e}") # Optional debug
        return np.full((xyz.shape[0], 2), np.nan)

def get_pos_pixels(points_3d, camera_matrix, dist_coeffs, image_size):
    """Gets valid, integer pixel coordinates within image bounds."""
    if points_3d.shape[0] == 0: return np.empty((0, 2), dtype=int)

    pixels_float = project_points(points_3d, camera_matrix, dist_coeffs)
    valid_projections_mask = ~np.isnan(pixels_float).any(axis=1)
    valid_pixels_float = pixels_float[valid_projections_mask]

    if valid_pixels_float.shape[0] == 0: return np.empty((0, 2), dtype=int)

    # Clip to image bounds BEFORE converting to int
    valid_pixels_float[:, 0] = np.clip(valid_pixels_float[:, 0], 0, image_size[0] - 1)
    valid_pixels_float[:, 1] = np.clip(valid_pixels_float[:, 1], 0, image_size[1] - 1)

    return valid_pixels_float.astype(int)

# --- Mask Creation Function ---

def create_trajectory_line_mask(traj_pixels, image_size, line_thickness):
    """Creates a binary mask by drawing lines between trajectory pixels."""
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8) # H, W format for cv2
    if traj_pixels.shape[0] >= 2:
        for i in range(len(traj_pixels) - 1):
            pt1 = tuple(traj_pixels[i])     # (x, y)
            pt2 = tuple(traj_pixels[i+1])   # (x, y)
            cv2.line(mask, pt1, pt2, color=255, thickness=line_thickness)
    return mask

# --- Data Loading Function ---

def load_poses(file_path):
    """Loads poses from a text file (timestamp x y z qx qy qz qw), skipping header."""
    poses = []
    line_num = 0
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) <= 1: return [] # Handle empty or header-only files
            for line in lines[1:]: # Skip header
                line_num += 1
                try:
                    data = line.split()
                    if len(data) < 8: continue # Skip malformed lines
                    poses.append({
                        'timestamp': float(data[0]), 'x': float(data[1]), 'y': float(data[2]), 'z': float(data[3]),
                        'qx': float(data[4]), 'qy': float(data[5]), 'qz': float(data[6]), 'qw': float(data[7])
                    })
                except ValueError: continue # Skip lines with conversion errors
    except FileNotFoundError:
        print(f"Error: Poses file not found: {file_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error reading poses file {file_path}: {e}", file=sys.stderr)
        return []
    return poses

def find_closest_pose_idx(timestamp, poses_timestamps):
    """Finds the index of the closest pose timestamp."""
    if len(poses_timestamps) == 0: return -1
    time_diffs = np.abs(poses_timestamps - timestamp)
    return np.argmin(time_diffs)

# --- Main Processing Function for a Single Bag ---

def process_single_bag(bag_prefix, pickle_file, poses_directory, output_mask_base_dir, line_thickness):
    """Processes one bag's data to generate trajectory masks."""
    print(f"--- Processing Bag: {bag_prefix} ---")

    poses_file = os.path.join(poses_directory, f"{bag_prefix}_processed", "path_poses.txt")
    output_masks_dir = os.path.join(output_mask_base_dir, f"{bag_prefix}_footprint_mask") # Changed folder name
    os.makedirs(output_masks_dir, exist_ok=True)

    # --- Load Data ---
    try:
        with open(pickle_file, "rb") as f: data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {pickle_file}: {e}", file=sys.stderr)
        return

    all_poses = load_poses(poses_file)
    if not all_poses:
        print(f"Error: No valid poses loaded from {poses_file}. Skipping bag.", file=sys.stderr)
        return

    # Validate essential keys in loaded pickle data
    required_keys = ['thermal_paths', 'thermal_timestamps']
    if not all(key in data for key in required_keys):
         print(f"Error: Pickle file {pickle_file} is missing required keys ({required_keys}). Skipping bag.", file=sys.stderr)
         return

    thermal_images_paths = data['thermal_paths']
    thermal_ts_list = data['thermal_timestamps']
    num_frames = len(thermal_ts_list)

    if not thermal_images_paths or num_frames == 0:
        print(f"Warning: No thermal image paths or timestamps found in {pickle_file}. Skipping bag.")
        return

    # Pre-calculate pose timestamps
    all_poses_timestamps = np.array([p['timestamp'] for p in all_poses])

    # Get image dimensions from the first valid thermal image path
    image_size = None
    for img_p in thermal_images_paths:
        if img_p and os.path.exists(img_p):
            try:
                img_h, img_w = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE).shape[:2]
                image_size = (img_w, img_h) # W, H
                print(f"Using image size: {image_size} from {img_p}")
                break
            except Exception as e:
                print(f"Warning: Could not read image dimensions from {img_p}: {e}")
    if image_size is None:
        print(f"Error: Could not determine image size for bag {bag_prefix}. Skipping.", file=sys.stderr)
        return


    # --- Process Frames ---
    frame_indices = range(num_frames)
    for idx in tqdm(frame_indices, desc=f"Bag {bag_prefix}", unit="frame"):
        thermal_ts = thermal_ts_list[idx]
        if thermal_ts is None or thermal_ts <= 0: continue # Skip invalid timestamps

        # Find current pose
        current_pose_idx = find_closest_pose_idx(thermal_ts, all_poses_timestamps)
        if current_pose_idx == -1: continue # Skip if no pose found

        # Find future poses
        start_future_pose_idx = current_pose_idx
        while start_future_pose_idx < len(all_poses) and all_poses[start_future_pose_idx]['timestamp'] < thermal_ts:
            start_future_pose_idx += 1
        if start_future_pose_idx >= len(all_poses): continue # Skip if no poses at or after current time

        future_poses_indices = range(start_future_pose_idx, min(start_future_pose_idx + MAX_FUTURE_POSES, len(all_poses)))
        future_poses = [all_poses[p_idx] for p_idx in future_poses_indices]
        if len(future_poses) < 2: continue # Need at least 2 for relative transform

        # --- Transform and Project ---
        try:
            # Reference frame: first future pose
            first_future_pose_dict = future_poses[0]
            first_future_euler = quaternion_to_angle(first_future_pose_dict)
            first_future_6dof_ref = np.array([[first_future_pose_dict['x'], first_future_pose_dict['y'], first_future_pose_dict['z'],
                                             first_future_euler[0], first_future_euler[1], first_future_euler[2]]])

            # Target frames: all future poses
            future_6dof_targets = []
            for p in future_poses:
                p_euler = quaternion_to_angle(p)
                future_6dof_targets.append([p['x'], p['y'], p['z'], p_euler[0], p_euler[1], p_euler[2]])
            future_6dof_targets = np.array(future_6dof_targets)

            # Transform
            positions_local_6dof = to_robot_numpy(first_future_6dof_ref, future_6dof_targets)
            positions_local_3d = positions_local_6dof[:, :3] # Extract xyz

            # Remove potential duplicates
            if positions_local_3d.shape[0] > 1:
                duplicates = np.all(np.abs(np.diff(positions_local_3d, axis=0)) < 1e-6, axis=1)
                keep_indices = np.concatenate(([True], ~duplicates))
                positions_local_3d = positions_local_3d[keep_indices]

            if positions_local_3d.shape[0] < 2: continue # Need at least 2 unique points

            # Apply camera offset
            positions_local_3d[:, 0] += CAMERA_X_OFFSET
            positions_local_3d[:, 2] -= CAMERA_HEIGHT

            # Project to pixels
            traj_pixels = get_pos_pixels(positions_local_3d, CAMERA_MATRIX, DIST_COEFFS, image_size)
            if traj_pixels.shape[0] < 2: continue # Need >= 2 valid pixels

        except Exception as e:
            print(f"\nWarning: Error during transform/project for frame {idx} in {bag_prefix}: {e}", file=sys.stderr)
            continue # Skip frame on error

        # --- Create and Save Mask ---
        try:
            mask = create_trajectory_line_mask(traj_pixels, image_size, line_thickness)
            # Use consistent frame index naming (e.g., 000000.png)
            mask_output_filename = f'{idx:06d}.png'
            mask_output_path = os.path.join(output_masks_dir, mask_output_filename)
            cv2.imwrite(mask_output_path, mask)
        except Exception as e:
            print(f"\nWarning: Error creating/saving mask for frame {idx} in {bag_prefix}: {e}", file=sys.stderr)
            continue

    print(f"--- Finished Bag: {bag_prefix} ---")


# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectory line masks from pickle data and poses.")
    parser.add_argument("--pickle_dir", type=str, required=True, help="Directory containing the input *_processed.pkl files.")
    parser.add_argument("--poses_dir", type=str, required=True, help="Base directory containing pose folders (<bag_prefix>_processed/path_poses.txt).")
    parser.add_argument("--mask_output_dir", type=str, required=True, help="Base directory to save the output mask folders (<bag_prefix>_footprint_mask/).")
    parser.add_argument("--line_thickness", type=int, default=DEFAULT_LINE_THICKNESS, help="Thickness of the trajectory line in pixels for the mask.")

    args = parser.parse_args()

    # Validate input directories
    if not os.path.isdir(args.pickle_dir):
        print(f"Error: Pickle directory not found: {args.pickle_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.poses_dir):
        print(f"Error: Poses directory not found: {args.poses_dir}", file=sys.stderr)
        sys.exit(1)

    # Create base output directory if it doesn't exist
    os.makedirs(args.mask_output_dir, exist_ok=True)

    # Find pickle files
    try:
        pickle_files = sorted([os.path.join(args.pickle_dir, f) for f in os.listdir(args.pickle_dir) if f.endswith("_processed.pkl")])
        if not pickle_files:
             print(f"Error: No '*_processed.pkl' files found in {args.pickle_dir}", file=sys.stderr)
             sys.exit(1)
        print(f"Found {len(pickle_files)} pickle files to process.")
    except Exception as e:
        print(f"Error accessing pickle directory {args.pickle_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # Process each pickle file
    for pickle_file in pickle_files:
        pickle_filename = os.path.basename(pickle_file)
        bag_prefix = pickle_filename.replace("_processed.pkl", "")
        try:
            process_single_bag(
                bag_prefix,
                pickle_file,
                args.poses_dir,
                args.mask_output_dir,
                args.line_thickness
            )
        except Exception as e:
             print(f"\n!!! Critical Error processing {pickle_filename}: {e}", file=sys.stderr)
             import traceback
             traceback.print_exc() # Print full traceback for critical errors
             print("!!! Proceeding to next file...")
        print("-" * 50)

    print("\nFinished processing all pickle files.")
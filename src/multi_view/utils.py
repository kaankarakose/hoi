import os
import json
import numpy as np
import cv2
import glob
from scipy.spatial.transform import Rotation as R


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def read_score(score_file):
    """Read a score file and return the score value."""
    with open(score_file, 'r') as f:
        return float(f.read().strip())


def rle2mask(rle):
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def mask2rle(mask):
    """
    Encodes a single mask to an uncompressed RLE, in the format expected by
    pycoco tools.
    
    Args:
        mask: A 2D numpy array of shape (h, w) with binary values
        
    Returns:
        Dictionary with 'size' and 'counts' fields
    """
    h, w = mask.shape
    # Put in fortran order and flatten (transpose and flatten)
    mask_flat = mask.T.flatten()
    
    # Compute change indices
    diff = mask_flat[1:] != mask_flat[:-1]  # XOR operation for boolean arrays
    change_indices = np.where(diff)[0]
    
    # Encode run length
    cur_idxs = np.concatenate([
        np.array([0], dtype=change_indices.dtype),
        change_indices + 1,
        np.array([h * w])
    ])
    
    btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
    counts = [] if mask_flat[0] == 0 else [0]
    counts.extend(btw_idxs.tolist())
    
    return {
        "size": [h, w],
        "counts": counts
    }


def read_rle(rle_file_path):
    """
    Read an RLE mask from a file.
    
    Args:
        rle_file_path: Path to the RLE file
    
    Returns:
        numpy array: Binary mask
    """
    with open(rle_file_path, 'r') as f:
        rle_data = json.load(f)
    return rle2mask(rle_data)


def rle_to_mask(rle):
    """Convert run-length encoding to binary mask."""
    if rle is None:
        return None
    return rle2mask(rle)


def mask_to_rle(mask):
    """Convert binary mask to run-length encoding."""
    return mask2rle(mask)


def save_rle(rle, output_path):
    """Save run-length encoding to file."""
    with open(output_path, 'w') as f:
        json.dump(rle, f, cls = NumpyEncoder )


def read_frame_data(base_dir, frame_id, object_name):
    """
    Read all masks and scores for a specific frame.
    
    Args:
        base_dir: Base directory containing the prediction data.
        frame_id: Frame ID (e.g., 'frame_0001').
        object_name: Name of the object folder (e.g., 'AMF1').
        
    Returns:
        Dictionary mapping mask indices to (score, mask) tuples.
    """
    result = {}
    
    # Get scores
    scores_dir = os.path.join(base_dir, object_name, 'scores', frame_id)
    if not os.path.exists(scores_dir):
        return result
    
    score_files = glob.glob(os.path.join(scores_dir, 'score_*.txt'))
    for score_file in score_files:
        idx = int(os.path.basename(score_file).split('_')[1].split('.')[0])
        score = read_score(score_file)
        
        # Get corresponding mask
        mask_file = os.path.join(base_dir, object_name, 'masks', frame_id, f'mask_{idx}.rle')
        if os.path.exists(mask_file):
            rle = read_rle(mask_file)
            result[idx] = (score, rle)
    
    return result


def triangulate_point(camera_matrices, image_points):
    """
    Triangulate a 3D point from multiple 2D points in different camera views.
    
    Args:
        camera_matrices: List of camera projection matrices (3x4)
        image_points: List of 2D points in each camera view (Nx2)
        
    Returns:
        3D point in world coordinates (3x1)
    """
    num_cameras = len(camera_matrices)
    A = np.zeros((num_cameras * 2, 4))
    
    for i in range(num_cameras):
        x, y = image_points[i]
        P = camera_matrices[i]
        A[i*2] = x * P[2] - P[0]
        A[i*2+1] = y * P[2] - P[1]
    
    # Solve the system of equations
    _, _, vh = np.linalg.svd(A)
    point_3d_homogeneous = vh[-1]
    
    # Convert from homogeneous to 3D coordinates
    point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
    return point_3d


def get_camera_matrix(intrinsic, extrinsic):
    """
    Compute the camera projection matrix from intrinsic and extrinsic parameters.
    
    Args:
        intrinsic: 3x3 intrinsic camera matrix
        extrinsic: 4x4 extrinsic camera matrix (rotation and translation)
        
    Returns:
        3x4 camera projection matrix
    """
    # Extract rotation and translation from extrinsic matrix
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3].reshape(3, 1)
    
    # Create the 3x4 [R|t] matrix
    Rt = np.hstack((rotation, translation))
    
    # Compute the camera projection matrix
    P = np.dot(intrinsic, Rt)
    return P


def get_mask_contour(mask):
    """
    Get the contour points of a binary mask.
    
    Args:
        mask: Binary mask as a numpy array
        
    Returns:
        Array of contour points
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.array([])
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour.reshape(-1, 2)


def get_mask_centroid(mask):
    """
    Get the centroid of a binary mask.
    
    Args:
        mask: Binary mask as a numpy array
        
    Returns:
        (x, y) coordinates of the centroid
    """
    # Calculate moments of the mask
    M = cv2.moments(mask.astype(np.uint8))
    
    if M["m00"] == 0:
        # If mask is empty, return the center of the image
        raise ValueError('Maks is empty!')
        h, w = mask.shape
        return (w // 2, h // 2)
    
    # Calculate centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def triangulate_mask(masks, camera_matrices=None, camera_system=None, views=None):
    """
    Triangulate a mask from multiple camera views to get a 3D point cloud.
    
    Args:
        masks: List of binary masks from different camera views
        camera_matrices: List of camera projection matrices, or None if using camera_system
        camera_system: Optional CameraSystem object containing camera poses
        views: Optional list of view names corresponding to the masks (required if camera_system is provided)
        
    Returns:
        Numpy array of 3D points representing the object
    """
    # If using old-style camera matrices
    if camera_system is None:
        if len(masks) < 2 or len(masks) != len(camera_matrices):
            return None
        
        # Get contour points from each mask
        all_contour_points = []
        for mask in masks:
            if mask is None:
                all_contour_points.append([])
                continue
            contour_points = get_mask_contour(mask)
            all_contour_points.append(contour_points)
        
        # If any mask doesn't have contour points, return None
        if any(len(points) == 0 for points in all_contour_points):
            return None
        
        # For simplicity, just use centroids for now
        centroids = [get_mask_centroid(mask) for mask in masks]
        
        # Check if any centroid is None
        if any(centroid is None for centroid in centroids):
            return None
        
        # Triangulate the centroids
        centroid_3d = triangulate_point(camera_matrices, centroids)
        
        # In a more complete implementation, we would triangulate multiple points
        # from the contours and create a point cloud
        # This is just a placeholder for now
        
        return np.array([centroid_3d])
    
    # If using new CameraSystem
    else:
        if len(masks) < 2 or len(masks) != len(views):
            return None
            
        # Get contour points from each mask
        all_contour_points = []
        valid_masks = []
        valid_views = []
        
        for i, mask in enumerate(masks):
            if mask is None:
                continue
                
            contour_points = get_mask_contour(mask)
            if len(contour_points) > 0:
                all_contour_points.append(contour_points)
                valid_masks.append(mask)
                valid_views.append(views[i])
        
        # Need at least 2 valid masks with contour points
        if len(valid_masks) < 2:
            return None
        
        # Extract centroids for each valid mask
        centroids = []
        for mask in valid_masks:
            centroid = get_mask_centroid(mask)
            if centroid is not None:
                centroids.append(centroid)
                
        if len(centroids) < 2:
            return None
            
        # Use camera_system to triangulate points
        # For now, we'll just use one reference camera (first in the list) and triangulate from there
        reference_cam = 'cam1'  # Assuming the first camera is the reference
        
        # Get the view poses relative to the reference camera
        view_poses = {}
        for view in valid_views:
            # Look for direct transformation from reference to view
            if f"{reference_cam}_to_{view}" in camera_system.camera_poses:
                view_poses[view] = camera_system.camera_poses[f"{reference_cam}_to_{view}"]
            # Look for direct transformation from view to reference
            elif f"{view}_to_{reference_cam}" in camera_system.camera_poses:
                # Need to invert the transformation
                pose = camera_system.camera_poses[f"{view}_to_{reference_cam}"]
                R_inv = pose.R.T  # Transpose of rotation matrix is its inverse for orthogonal matrices
                T_inv = -np.dot(R_inv, pose.T)  # Negation of R^-1 * T
                view_poses[view] = {'R': R_inv, 'T': T_inv}
        
        # If we don't have enough view poses, return None
        if len(view_poses) < 2:
            return None
            
        # TODO: Implement proper triangulation using camera_system
        # This would involve projecting points from image space to 3D world space
        # using the camera extrinsics
        
        # For now, return a placeholder
        # In a real implementation, you would triangulate the centroids using the camera poses
        return np.array([[0.0, 0.0, 0.0]])  # Placeholder


def merge_masks(masks, scores, camera_params, strategy='triangulate'):
    """
    Merge masks from different views using 3D triangulation.
    
    Args:
        masks: List of binary masks from different camera views.
        scores: List of corresponding scores.
        camera_params: Either:
                       - List of dictionaries containing 'intrinsic' and 'extrinsic' matrices for each camera, or
                       - Dictionary containing 'camera_system' (CameraSystem object) and 'views' (list of view names).
        strategy: Merging strategy ('triangulate').
        
    Returns:
        Dictionary containing:
        - 'points3d': Numpy array of 3D points (if strategy is 'triangulate')
        - 'mask': Merged mask as RLE encoding (if strategy is not 'triangulate')
        - 'score': Average score of the masks
    """
    if len(masks) == 0 or len(scores) == 0:
        return {'score': 0}
    
    if strategy == 'triangulate':
        # Check if we're using the new CameraSystem or the old camera parameters
        if isinstance(camera_params, dict) and 'camera_system' in camera_params and 'views' in camera_params:
            # Using new CameraSystem
            camera_system = camera_params['camera_system']
            views = camera_params['views']
            
            # Triangulate masks to get 3D points
            points_3d = triangulate_mask(masks, None, camera_system, views)
        else:
            # Using old-style camera parameters
            # Compute camera matrices from intrinsic and extrinsic parameters
            camera_matrices = []
            for params in camera_params:
                if 'intrinsic' in params and 'extrinsic' in params:
                    P = get_camera_matrix(params['intrinsic'], params['extrinsic'])
                    camera_matrices.append(P)
                else:
                    print("Missing intrinsic or extrinsic parameters")
            
            # Triangulate masks to get 3D points
            points_3d = triangulate_mask(masks, camera_matrices)
        
        # Compute average score
        avg_score = sum(scores) / len(scores)
        
        return {
            'points3d': points_3d,
            'score': avg_score
        }
    
    # TODO: Implement other strategies (highest, average, union)

    
    else:
        raise ValueError(f"Unknown merging strategy: {strategy}")

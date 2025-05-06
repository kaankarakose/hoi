"""
HAMER feature loader class for loading Hand Mesh Recovery features.
"""

import os
import glob
import re
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the parent directory to the path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from object_interaction_detection.dataloaders.base_loader import BaseDataLoader

logging.basicConfig(level=logging.INFO)

class HAMERLoader(BaseDataLoader):
    """
    Data loader for HAMER (Hand Mesh Recovery) features.
    
    This loader handles loading hand pose/mesh data for analyzing
    hand-object interactions.
    """
    
    def __init__(self, 
                 session_name: str, 
                 data_root_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the HAMER loader.
        
        Args:
            session_name: Name of the session to process
            data_root_dir: Root directory for all data
            config: Configuration parameters (optional)
        """
        # Set HAMER-specific defaults for paths
        config = config or {}
        
        # Add default HAMER paths
        config.setdefault('pose_dir', os.path.join(data_root_dir, 'hand_detections'))
        
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # HAMER specific attributes
        self.hand_types = ['left', 'right']
        self.camera_views = ['cam_top', 'cam_side_l', 'cam_side_r']
        
        # Define hand joint indices for specific landmarks
        # These indices are based on MANO model joints
        self.landmark_indices = {
            'wrist': 0,
            'index_tip': 4,
            'middle_tip': 8,
            'ring_tip': 12,
            'pinky_tip': 16,
            'thumb_tip': 20
        }
    
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load HAMER features for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing hand pose features
        """
        # Check for cached features
        cached_features = self._get_cached_features(camera_view, frame_idx)
        if cached_features is not None:
            return cached_features
        
        # Initialize features dictionary
        features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'left_hand': {
                'pose_data': None,
                'success': False,
                'vertices': None,
                'cam_t': None,
                'crop_bbox': None,
                'bbox': None
            },
            'right_hand': {
                'pose_data': None,
                'success': False,
                'vertices': None,
                'cam_t': None,
                'crop_bbox': None,
                'bbox': None
            }
        }
        
        # Process each hand type separately
        for hand_type in self.hand_types:
            # Load pose data for this specific hand
            pose_data = self._load_pose_data(camera_view, frame_idx, hand_type)
            
            if pose_data is None:
                continue
            
            # Determine which hand data we're working with based on is_right flag
            if pose_data.get('is_right', False):  # Right hand
                hand_key = 'right_hand'
            else:  # Left hand
                hand_key = 'left_hand'
            
            # Update features for this hand
            features[hand_key]['success'] = True
            features[hand_key]['pose_data'] = pose_data
            features[hand_key]['vertices'] = pose_data.get('vertices')
            features[hand_key]['cam_t'] = pose_data.get('cam_t')
            
            # Handle crop bbox - each hand may have a different naming convention
            crop_bbox_key = f'{hand_type}_crop_bbox' if f'{hand_type}_crop_bbox' in pose_data else 'crop_bbox'
            features[hand_key]['crop_bbox'] = pose_data.get(crop_bbox_key)
            features[hand_key]['bbox'] = pose_data.get('bbox')
        
        # Cache features (store a copy without the raw vertices data to save memory)
        cache_features = self._prepare_cache_features(features)
        self._cache_features(camera_view, frame_idx, cache_features)
        
        return features
    
    def _prepare_cache_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a memory-efficient version of features for caching.
        
        Args:
            features: Full features dictionary
            
        Returns:
            Memory-efficient features dictionary for caching
        """
        cache_features = features.copy()
        
        for hand_type in self.hand_types:
            hand_key = f'{hand_type}_hand'
            if features[hand_key]['success']:
                # Create a copy without the large data fields
                hand_data = features[hand_key].copy()
                hand_data['pose_data'] = None  # Don't cache the full pose data
                hand_data['vertices'] = None   # Don't cache the vertices
                cache_features[hand_key] = hand_data
        
        return cache_features
    
    def _load_pose_data(self, camera_view: str, frame_idx: int, hand_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Load raw pose data for a specific hand at a specific frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            hand_type: Hand type ('left' or 'right'). If None, will try both hands.
            
        Returns:
            Raw pose data or None if not available
        """
        # Build path to pose directory
        poses_dir = os.path.join(
            self.config['pose_dir'],
            self.session_name,
            camera_view,
            "3D_poses"
        )
        
        if not os.path.exists(poses_dir):
            logging.warning(f"Poses directory not found: {poses_dir}")
            return None
        
        # Try to load specific hand if specified
        if hand_type is not None:
            file_pattern = f"{hand_type}_hand_{frame_idx:06d}.npz"
            file_path = os.path.join(poses_dir, file_pattern)
            
            if os.path.exists(file_path):
                try:
                    # Load the npz file
                    data = np.load(file_path, allow_pickle=True)
                    
                    # Convert to dictionary for easier access
                    pose_data = {}
                    for key in data.files:
                        pose_data[key] = data[key]
                    
                    # Set is_right flag based on hand_type
                    pose_data['is_right'] = (hand_type == 'right')
                    return pose_data
                except Exception as e:
                    logging.error(f"Error loading pose data from {file_path}: {e}")
        
        # If hand_type is None or we didn't find the specified hand, try both hands
        for hand in ['left', 'right']:
            file_pattern = f"{hand}_hand_{frame_idx:06d}.npz"
            file_path = os.path.join(poses_dir, file_pattern)
            
            if os.path.exists(file_path):
                try:
                    # Load the npz file
                    data = np.load(file_path, allow_pickle=True)
                    
                    # Convert to dictionary for easier access
                    pose_data = {}
                    for key in data.files:
                        pose_data[key] = data[key]
                    
                    # Set is_right flag based on hand
                    pose_data['is_right'] = (hand == 'right')
                    return pose_data
                except Exception as e:
                    logging.error(f"Error loading pose data from {file_path}: {e}")
        
        # If we get here, no hand data was found
        return None
    
    def get_hand_features(self, camera_view: str, frame_idx: int, hand_type: str) -> Dict[str, Any]:
        """
        Get features specifically for one hand.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            hand_type: Hand type ('left' or 'right')
            
        Returns:
            Dictionary with hand features or empty dict if not available
        """
        features = self.load_features(camera_view, frame_idx)
        hand_key = f'{hand_type}_hand'
        
        if not features[hand_key]['success']:
            return {}
            
        return features[hand_key]
    
    def get_hand_vertices(self, camera_view: str, frame_idx: int, hand_type: str) -> Optional[np.ndarray]:
        """
        Get hand vertices for a specific frame and hand.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            hand_type: Hand type ('left' or 'right')
            
        Returns:
            Numpy array of vertices or None if not available
        """
        hand_features = self.get_hand_features(camera_view, frame_idx, hand_type)
        return hand_features.get('vertices')
    
    def get_hand_crop_bbox(self, camera_view: str, frame_idx: int, hand_type: str) -> Optional[List[int]]:
        """
        Get hand crop bounding box for a specific frame and hand.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            hand_type: Hand type ('left' or 'right')
            
        Returns:
            List containing bounding box coordinates [x1, y1, x2, y2] or None if not available
        """
        hand_features = self.get_hand_features(camera_view, frame_idx, hand_type)
        return hand_features.get('crop_bbox')
    
    def get_hand_detection_status(self, camera_view: str, frame_idx: int) -> Dict[str, bool]:
        """
        Get detection status for both hands in a specific frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary with keys 'left' and 'right' containing boolean values
        """
        features = self.load_features(camera_view, frame_idx)
        return {
            'left': features['left_hand']['success'],
            'right': features['right_hand']['success']
        }

    def get_valid_frame_idx(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Get valid frame indices for each hand type and camera view.
        
        Returns:
            Dictionary with camera views as keys, each containing a dictionary with
            'left' and 'right' as keys and lists of valid frame indices as values.
        """
        valid_frames = {}
        
        # Iterate through all camera views
        for camera_view in self.camera_views:
            valid_frames[camera_view] = {
                'left': [],
                'right': []
            }
            
            # Build path to pose directory
            poses_dir = os.path.join(
                self.config['pose_dir'],
                self.session_name,
                camera_view,
                "3D_poses"
            )
            
            if not os.path.exists(poses_dir):
                logging.warning(f"Poses directory not found: {poses_dir}")
                continue
            
            # Look for left hand files
            left_hand_files = sorted(glob.glob(os.path.join(poses_dir, "left_hand_*.npz")))
            for file_path in left_hand_files:
                try:
                    # Extract frame index from filename
                    match = re.search(r'left_hand_(\d+).npz', os.path.basename(file_path))
                    if match:
                        frame_idx = int(match.group(1))
                        valid_frames[camera_view]['left'].append(frame_idx)
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")
            
            # Look for right hand files
            right_hand_files = sorted(glob.glob(os.path.join(poses_dir, "right_hand_*.npz")))
            for file_path in right_hand_files:
                try:
                    # Extract frame index from filename
                    match = re.search(r'right_hand_(\d+).npz', os.path.basename(file_path))
                    if match:
                        frame_idx = int(match.group(1))
                        valid_frames[camera_view]['right'].append(frame_idx)
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")
        
        return valid_frames
        
if __name__ == "__main__":
    # Test the loader
    hamer_loader = HAMERLoader(
        session_name="imi_session1_2", 
        data_root_dir="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data"
    )
    
    # Test loading data for a specific frame
    data = hamer_loader.load_features(camera_view="cam_top", frame_idx=100)
    
    print("Left hand detected:", data['left_hand']['success'])
    print("Right hand detected:", data['right_hand']['success'])
    
    if data['left_hand']['success']:
        print("Left hand crop bbox:", data['left_hand']['crop_bbox'])
    
    if data['right_hand']['success']:
        print("Right hand crop bbox:", data['right_hand']['crop_bbox'])

    valid_frames = hamer_loader.get_valid_frame_idx()
    print("Valid frames:", list(valid_frames['cam_top']['left']))

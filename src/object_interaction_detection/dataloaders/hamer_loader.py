"""
HAMER feature loader class for loading Hand Mesh Recovery features.
"""

import os
import glob
import re
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
# Add the parent directory to the path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from object_interaction_detection.dataloaders.base_loader import BaseDataLoader


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
                'landmarks': {}
            },
            'right_hand': {
                'pose_data': None,
                'success': False,
                'landmarks': {}
            }
        }
        
        # Process each hand type
        for hand_type in self.hand_types:
            # Load pose data for this hand
            pose_data = self._load_pose_data(camera_view, frame_idx, hand_type)
            
            if pose_data is None:
                continue
            
            # Store raw pose data (reference only, not stored in cache)
            features[f'{hand_type}_hand']['pose_data'] = pose_data
            features[f'{hand_type}_hand']['success'] = True
            
            # Extract key landmarks if available
            self._extract_landmarks(features[f'{hand_type}_hand'], pose_data)
        
        # Cache features (store a copy without the raw pose data to save memory)
        cache_features = features.copy()
        for hand_type in self.hand_types:
            if cache_features[f'{hand_type}_hand']['success']:
                # Create a copy without the large pose_data field
                hand_data = cache_features[f'{hand_type}_hand'].copy()
                hand_data['pose_data'] = None  # Don't cache the full pose data
                cache_features[f'{hand_type}_hand'] = hand_data
        
        # Cache the memory-efficient version
        self._cache_features(camera_view, frame_idx, cache_features)
        
        return features
    
    def _load_pose_data(self, camera_view: str, frame_idx: int, 
                       hand_type: str) -> Optional[Dict[str, Any]]:
        """
        Load raw pose data for a specific hand.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            hand_type: Hand type ('left' or 'right')
            
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
            return None
        
        # Try different filename patterns
        pattern = f"{hand_type}_hand_{frame_idx:06d}.npz",
 
        path = os.path.join(poses_dir, pattern)
        if os.path.exists(path):
            try:
                return dict(np.load(path, allow_pickle=True))
            except Exception as e:
                print(f"Error loading pose data: {e}")
        
        return None
    
    def _extract_landmarks(self, hand_features: Dict[str, Any], 
                          pose_data: Dict[str, Any]):
        """
        Extract key landmarks from pose data.
        :TODO
        Args:
            hand_features: Features dictionary to update
            pose_data: Raw pose data
        """
        # Check for joints in pose data
        if 'joints3d' not in pose_data and 'verts' not in pose_data:
            return
        
        # Use joints if available, otherwise use vertices
        if 'joints3d' in pose_data:
            joints = pose_data['joints3d']
            # Extract key landmarks
            for landmark_name, joint_idx in self.landmark_indices.items():
                if joint_idx < len(joints):
                    hand_features['landmarks'][landmark_name] = {
                        'position': joints[joint_idx].tolist() if isinstance(joints[joint_idx], np.ndarray) else joints[joint_idx],
                        'confidence': pose_data.get('confidence', 1.0)
                    }
        elif 'verts' in pose_data:
            # For mesh vertices, we might need to use a different approach
            # This is just a placeholder - would need to be adjusted for actual data
            verts = pose_data['verts']
            # Extract approximate tip positions (if there's a mapping available)
            if hasattr(self, 'vertex_to_landmark_mapping'):
                for landmark_name, vertex_idx in self.vertex_to_landmark_mapping.items():
                    if vertex_idx < len(verts):
                        hand_features['landmarks'][landmark_name] = {
                            'position': verts[vertex_idx].tolist() if isinstance(verts[vertex_idx], np.ndarray) else verts[vertex_idx],
                            'confidence': pose_data.get('confidence', 1.0)
                        }
        
        # Store global hand properties
        if 'global_orient' in pose_data:
            hand_features['global_orient'] = pose_data['global_orient'].tolist() if isinstance(pose_data['global_orient'], np.ndarray) else pose_data['global_orient']
        
        if 'hand_pose' in pose_data:
            hand_features['hand_pose'] = True  # Just store if it exists, not the full pose
        
        if 'betas' in pose_data:
            hand_features['betas'] = True  # Just store if it exists, not the full betas
        
        if 'confidence' in pose_data:
            hand_features['confidence'] = float(pose_data['confidence'])
        
        # Store crop bounding box if available
        hand_type = 'left' if 'left' in hand_features.get('type', '') else 'right'
        if f'{hand_type}_crop_bbox' in pose_data:
            hand_features['crop_bbox'] = pose_data[f'{hand_type}_crop_bbox'].tolist() if isinstance(pose_data[f'{hand_type}_crop_bbox'], np.ndarray) else pose_data[f'{hand_type}_crop_bbox']
        elif 'crop_bbox' in pose_data:
            hand_features['crop_bbox'] = pose_data['crop_bbox'].tolist() if isinstance(pose_data['crop_bbox'], np.ndarray) else pose_data['crop_bbox']
    
    def get_hand_velocities(self, camera_view: str, start_frame: int, 
                           end_frame: int) -> Dict[int, Dict[str, Any]]:
        """
        Calculate hand velocities for a range of frames.
        
        Args:
            camera_view: Camera view name
            start_frame: Starting frame index
            end_frame: Ending frame index
            
        Returns:
            Dictionary mapping frame indices to velocity data
        """
        # Load features for all frames in range
        features_by_frame = self.load_features_range(camera_view, start_frame, end_frame)
        
        # Initialize velocities dictionary
        velocities = {}
        
        # Process each frame (except the first one)
        for frame_idx in range(start_frame + 1, end_frame + 1):
            # Skip if this frame or previous frame doesn't have features
            if frame_idx not in features_by_frame or frame_idx - 1 not in features_by_frame:
                continue
            
            # Get current and previous frame features
            curr_features = features_by_frame[frame_idx]
            prev_features = features_by_frame[frame_idx - 1]
            
            # Initialize velocities for this frame
            velocities[frame_idx] = {
                'left_hand': {},
                'right_hand': {}
            }
            
            # Process each hand type
            for hand_type in self.hand_types:
                hand_key = f'{hand_type}_hand'
                
                # Skip if either frame doesn't have this hand
                if (not curr_features[hand_key]['success'] or 
                    not prev_features[hand_key]['success']):
                    continue
                
                # Process each landmark
                for landmark_name in self.landmark_indices.keys():
                    # Skip if either frame doesn't have this landmark
                    if (landmark_name not in curr_features[hand_key]['landmarks'] or 
                        landmark_name not in prev_features[hand_key]['landmarks']):
                        continue
                    
                    # Get current and previous positions
                    curr_pos = curr_features[hand_key]['landmarks'][landmark_name]['position']
                    prev_pos = prev_features[hand_key]['landmarks'][landmark_name]['position']
                    
                    # Calculate velocity (difference in position)
                    if isinstance(curr_pos, list) and isinstance(prev_pos, list) and len(curr_pos) == 3 and len(prev_pos) == 3:
                        velocity = [
                            curr_pos[0] - prev_pos[0],
                            curr_pos[1] - prev_pos[1],
                            curr_pos[2] - prev_pos[2]
                        ]
                        
                        # Calculate magnitude
                        magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
                        
                        # Store velocity
                        velocities[frame_idx][hand_key][landmark_name] = {
                            'velocity': velocity,
                            'magnitude': float(magnitude)
                        }
        
        return velocities
    
    def detect_hand_object_proximity(self, camera_view: str, frame_idx: int, 
                                    object_positions: Dict[str, List[float]], 
                                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect proximity between hands and objects.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            object_positions: Dictionary mapping object names to 3D positions
            threshold: Proximity threshold in meters
            
        Returns:
            Dictionary with proximity detection results
        """
        # Load hand features
        hand_features = self.load_features(camera_view, frame_idx)
        
        # Initialize results
        results = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'proximity_threshold': threshold,
            'hand_object_proximities': {
                'left_hand': {},
                'right_hand': {}
            }
        }
        
        # Process each hand type
        for hand_type in self.hand_types:
            hand_key = f'{hand_type}_hand'
            
            # Skip if no hand data
            if not hand_features[hand_key]['success']:
                continue
            
            # Process each landmark
            for landmark_name, landmark_data in hand_features[hand_key]['landmarks'].items():
                # Skip if no position data
                if 'position' not in landmark_data:
                    continue
                
                hand_pos = landmark_data['position']
                
                # Skip if invalid position
                if not isinstance(hand_pos, list) or len(hand_pos) != 3:
                    continue
                
                # Check proximity to each object
                for object_name, object_pos in object_positions.items():
                    # Skip if invalid object position
                    if not isinstance(object_pos, list) or len(object_pos) != 3:
                        continue
                    
                    # Calculate distance
                    distance = np.sqrt(
                        (hand_pos[0] - object_pos[0])**2 +
                        (hand_pos[1] - object_pos[1])**2 +
                        (hand_pos[2] - object_pos[2])**2
                    )
                    
                    # Check if within threshold
                    if distance <= threshold:
                        # Store proximity result
                        if object_name not in results['hand_object_proximities'][hand_key]:
                            results['hand_object_proximities'][hand_key][object_name] = []
                        
                        results['hand_object_proximities'][hand_key][object_name].append({
                            'landmark': landmark_name,
                            'distance': float(distance),
                            'hand_position': hand_pos,
                            'object_position': object_pos
                        })
        
        # Add summary
        results['interaction_detected'] = (
            len(results['hand_object_proximities']['left_hand']) > 0 or
            len(results['hand_object_proximities']['right_hand']) > 0
        )
        
        return results
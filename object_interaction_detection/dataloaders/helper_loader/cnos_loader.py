"""
CNOS feature loader class for loading Contact Neural Object Segmentation features.
"""

import os
import glob
import re
import json
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the parent directory to the path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from object_interaction_detection.dataloaders.base_loader import BaseDataLoader

from object_interaction_detection.utils.utils import load_rle_mask, rle2mask
import logging
logging.basicConfig(level=logging.INFO)


class CNOSLoader(BaseDataLoader):
    """
    Data loader for CNOS (Contact Neural Object Segmentation) features.
    
    This loader handles loading segmentation masks and scores for 
    hand-object interaction detection.
    """
    
    def __init__(self, 
                 session_name: str, 
                 data_root_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CNOS loader.
        
        Args:
            session_name: Name of the session to process
            data_root_dir: Root directory for all data
            config: Configuration parameters (optional)
        """
        # Set CNOS-specific defaults for paths
        config = config or {}
        
        # Add default CNOS paths
        config.setdefault('results_dir', os.path.join(data_root_dir, 'multi_cnos_result'))
        config.setdefault('cropped_frames_dir', os.path.join(data_root_dir, 'hand_detections'))
        
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # CNOS specific attributes
        self.frame_types = ['L_frames', 'R_frames']  # Left and right hand frames
        
        self.camera_views = ['cam_top', 'cam_side_l', 'cam_side_r']
        # Map of camera_view -> frame_type -> list of objects
        self.available_objects = self._discover_available_objects()
    
    
    def _discover_available_objects(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Discover available objects for each camera view and frame type.
        
        Returns:
            Dictionary mapping camera views to frame types to object lists
        """
        available_objects = {}
        
        # Process each camera view
        for cam_view in self.camera_views:
            available_objects[cam_view] = {}
            for frame_type in self.frame_types:
                # Build path to results directory
                frame_type_results_dir = os.path.join(
                    self.config['results_dir'], 
                    self.session_name, 
                    cam_view, 
                    frame_type
                )
                if not os.path.exists(frame_type_results_dir):
                    available_objects[cam_view][frame_type] = []
                    continue
                # Get object directories
                object_dirs = [d for d in glob.glob(os.path.join(frame_type_results_dir, "*")) 
                              if os.path.isdir(d) and os.path.basename(d) != "frames"]
                
                # Extract object names
                object_names = [os.path.basename(d) for d in object_dirs]
                object_names.sort()
                
                available_objects[cam_view][frame_type] = object_names
                
                logging.info(f"Found {len(object_names)} objects for {cam_view}/{frame_type}: {object_names}")
        
        return available_objects
    def get_frame_count(self, camera_view: str, frame_type: str) -> int:
        """
        Get total number of frames for a specific camera view and frame type.
        
        Args:
            camera_view: Camera view name
            frame_type: Frame type ('L_frames' or 'R_frames')
            
        Returns:
            Number of frames
        """ 
        frames_path = os.path.join(self.config['results_dir'], 
                                  self.session_name, 
                                  camera_view, 
                                  frame_type, 
                                  'frames')
        if not os.path.exists(frames_path):
            return 0
        
        # Count frames based on image files in the directory
        frame_files = [f for f in os.listdir(frames_path) 
                     if f.endswith('.jpg') or f.endswith('.png')]
        
        return len(frame_files)
    def get_available_objects(self, camera_view: str, frame_type: str) -> List[str]:
        """
        Get list of available objects for a specific camera view and frame type.
        
        Args:
            camera_view: Camera view name
            frame_type: Frame type ('L_frames' or 'R_frames')
            
        Returns:
            List of object names
        """
        if (camera_view not in self.available_objects or 
            frame_type not in self.available_objects[camera_view]):
            return []
            
        return self.available_objects[camera_view][frame_type]
    
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load CNOS features for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing features for both left and right hands
        """
        # Check for cached features
        cached_features = self._get_cached_features(camera_view, frame_idx)
        if cached_features is not None:
            return cached_features
        
        # Initialize features dictionary
        features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'L_frames': {
                'objects': {},
                'success': False
            },
            'R_frames': {
                'objects': {},
                'success': False
            }
        }
        
        # Process each frame type (left and right hand)
        for frame_type in self.frame_types:
            # Get available objects for this frame type
            objects = self.get_available_objects(camera_view, frame_type)
            if not objects:
                continue
            
            # Process each object
            for object_name in objects:
                # Get masks and scores for this object
                mask_files, scores = self._get_masks_and_scores(
                    camera_view, frame_type, object_name, frame_idx
                )
                
                if not mask_files:
                    continue
                
                # Store in features dictionary
                features[frame_type]['objects'][object_name] = {
                    'mask_files': mask_files,
                    'scores': scores,
                    'max_score': max(scores) if scores else 0.0,
                    'max_score_idx': np.argmax(scores) if scores else -1
                }
            
            # Mark as successful if we found any objects
            if features[frame_type]['objects']:
                features[frame_type]['success'] = True
        
        # Cache features
        self._cache_features(camera_view, frame_idx, features)
        
        return features
    
    def load_masks(self, camera_view: str, frame_idx: int, 
                  load_masks: bool = False) -> Dict[str, Any]:
        """
        Load CNOS features including actual mask data (not just filenames).
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            load_masks: Whether to load actual mask data (memory intensive)
            
        Returns:
            Dictionary containing features with actual mask data
        """
        # Load base features first (paths and scores)
        features = self.load_features(camera_view, frame_idx)
        

        
        # Process each frame type
        for frame_type in self.frame_types:
            if not features[frame_type]['success']:

                continue
                
            # Process each object
            for object_name, object_data in features[frame_type]['objects'].items():
                # Initialize masks list
                object_data['masks'] = []
                
                # Load each mask
                for mask_file in object_data['mask_files']:
                    
                    mask = load_rle_mask(mask_file)
                    mask = rle2mask(mask)
                    object_data['masks'].append(mask)

                
                # Add the highest-scoring mask for convenience
                max_idx = object_data['max_score_idx']
                if max_idx >= 0 and max_idx < len(object_data['masks']):
                    object_data['max_score_mask'] = object_data['masks'][max_idx]
        
        return features
    
    def _get_masks_and_scores(self, camera_view: str, frame_type: str, 
                             object_name: str, frame_idx: int) -> Tuple[List[str], List[float]]:
        """
        Get mask files and scores for a specific object, frame type, and frame index.
        
        Args:
            camera_view: Camera view name
            frame_type: Frame type ('L_frames' or 'R_frames')
            object_name: Object name
            frame_idx: Frame index
            
        Returns:
            Tuple of (mask_files, scores)
        """
        # Build path to object directory in results
        object_dir = os.path.join(
            self.config['results_dir'], 
            self.session_name, 
            camera_view, 
            frame_type, 
            object_name
        )
        
        if not os.path.exists(object_dir):
            return [], []
        
        # Format frame name with leading zeros (try different formats)
        frame_names = [
            f"frame_{frame_idx:06d}"
        ]
        
        masks_dir = None
        scores_dir = None
        
        # Find masks directory
        for frame_name in frame_names:
            test_masks_dir = os.path.join(object_dir, "masks", frame_name)
            if os.path.exists(test_masks_dir):
                masks_dir = test_masks_dir
                break
        
        # Find scores directory
        for frame_name in frame_names:
            test_scores_dir = os.path.join(object_dir, "scores", frame_name)
            if os.path.exists(test_scores_dir):
                scores_dir = test_scores_dir
                break
        
        # Get masks
        mask_files = []
        if masks_dir:
            mask_files = sorted(glob.glob(os.path.join(masks_dir, "mask_*.rle")))
        
        # Get scores
        scores = []
        if scores_dir:
            score_files = sorted(glob.glob(os.path.join(scores_dir, "score_*.txt")))
            
            for score_file in score_files:
                try:
                    with open(score_file, 'r') as f:
                        score = float(f.read().strip())
                        scores.append(score)
                except:
                    scores.append(0.0)
        
        # If we have masks but no scores, use default score of 1.0
        if mask_files and not scores:
            raise ValueError("This cannot be happened! - Data corrupted")
        
        # Make sure we have the same number of masks and scores
        mask_files = mask_files[:len(scores)] if scores else []
        scores = scores[:len(mask_files)] if mask_files else []
        
        return mask_files, scores
    
    def get_max_score_by_frame(self, camera_view: str, start_frame: int, 
                              end_frame: int) -> Dict[int, Dict[str, float]]:
        """
        Get maximum detection score for each frame in a range.
        
        Args:
            camera_view: Camera view name
            start_frame: Starting frame index
            end_frame: Ending frame index
            
        Returns:
            Dictionary mapping frame indices to max scores by object
        """
        max_scores = {}
        
        # Process each frame
        for frame_idx in range(start_frame, end_frame + 1):
            # Load features for this frame
            features = self.load_features(camera_view, frame_idx)
            
            # Initialize frame entry
            max_scores[frame_idx] = {
                'L_frames': {},
                'R_frames': {}
            }
            
            # Process each frame type
            for frame_type in self.frame_types:
                
                if not features[frame_type]['success']:
                    continue
                    
                # Get max score for each object
                for object_name, object_data in features[frame_type]['objects'].items():
                    max_scores[frame_idx][frame_type][object_name] = object_data['max_score']
        
        return max_scores



if __name__ == "__main__":
  

    cnos_loader = CNOSLoader(session_name="imi_session1_2", data_root_dir="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data")
    print("Testing cnos loader")
    print("get_max_score_by_frame -> it should return me max score for each frame")
    print(cnos_loader.get_max_score_by_frame(camera_view="cam_side_l", start_frame=100, end_frame=101))

    print("get_avaiable_objects",cnos_loader.get_available_objects(camera_view="cam_top", frame_type="L_frames"))
    
    print("get_frame_count",cnos_loader.get_frame_count(camera_view="cam_top", frame_type="L_frames"))


    data = cnos_loader.load_masks(camera_view="cam_top", frame_idx=100, load_masks=True)

   
    
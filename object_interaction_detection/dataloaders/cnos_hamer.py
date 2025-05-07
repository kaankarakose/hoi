"""
Combined CNOS and HAMER loader for hand-object interaction detection.
"""

import os
import glob
import re
import json
import sys
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the parent directory to the path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.object_interaction_detection.dataloaders.helper_loader.base_loader import BaseDataLoader
from src.object_interaction_detection.dataloaders.helper_loader.cnos_loader import CNOSLoader
from src.object_interaction_detection.dataloaders.helper_loader.hamer_loader import HAMERLoader

logging.basicConfig(level=logging.INFO)

class CNOSHAMERLoader(BaseDataLoader):
    """
    Combined loader that uses both CNOS and HAMER loaders to provide
    object segmentation masks in original image coordinates.
    """
    
    def __init__(self, 
                 session_name: str, 
                 data_root_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the combined CNOS and HAMER loader.
        
        Args:
            session_name: Name of the session to process
            data_root_dir: Root directory for all data
            config: Configuration parameters (optional)
        """
        # Set defaults for paths
        config = config or {}
        
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # Initialize both loaders
        self.cnos_loader = CNOSLoader(session_name, data_root_dir, config)
        self.hamer_loader = HAMERLoader(session_name, data_root_dir, config)
        
        # Set reference to camera views
        self.camera_views = self.cnos_loader.camera_views
        
        # Define mapping between frame types and hand types
        self.frame_type_to_hand_type = {
            'L_frames': 'left_hand',
            'R_frames': 'right_hand'
        }

        # Default original image dimensions - can be overridden via config
        self.orig_width = config.get('orig_width', 1920)
        self.orig_height = config.get('orig_height', 1200)
        
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load combined features from both CNOS and HAMER loaders.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing combined features
        """
        # Check for cached features
        cached_features = self._get_cached_features(camera_view, frame_idx)
        if cached_features is not None:
            return cached_features
        
        # Load features from both loaders
        cnos_features = self.cnos_loader.load_features(camera_view, frame_idx)
        hamer_features = self.hamer_loader.load_features(camera_view, frame_idx)
        
        # Initialize combined features dictionary
        features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'L_frames': {
                'objects': {},
                'success': cnos_features['L_frames']['success'],
                'crop_bbox': None
            },
            'R_frames': {
                'objects': {},
                'success': cnos_features['R_frames']['success'],
                'crop_bbox': None
            }
        }
        
        # Add crop bounding boxes from HAMER to each frame type
        if hamer_features['left_hand']['success']:
            features['L_frames']['crop_bbox'] = hamer_features['left_hand']['crop_bbox']
            
        if hamer_features['right_hand']['success']:
            features['R_frames']['crop_bbox'] = hamer_features['right_hand']['crop_bbox']
        
        # Copy CNOS object data
        for frame_type in self.cnos_loader.frame_types:
            if cnos_features[frame_type]['success']:
                features[frame_type]['objects'] = cnos_features[frame_type]['objects']
        
        # Cache features
        self._cache_features(camera_view, frame_idx, features)
        
        return features
    
    def load_original_masks(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load CNOS masks and transform them to original image coordinates.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing masks in original image coordinates
        """
        # Load combined features first
        features = self.load_features(camera_view, frame_idx)
        
        # Load raw masks from CNOS
        cnos_features = self.cnos_loader.load_masks(camera_view, frame_idx)
        
        # Initialize transformed masks dictionary
        transformed_features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'L_frames': {
                'objects': {},
                'success': features['L_frames']['success'],
                'crop_bbox': features['L_frames']['crop_bbox']
            },
            'R_frames': {
                'objects': {},
                'success': features['R_frames']['success'],
                'crop_bbox': features['R_frames']['crop_bbox']
            }
        }
        
        # Process each frame type
        for frame_type in self.cnos_loader.frame_types:
            # Skip if no masks for this frame type
            if not cnos_features[frame_type]['success']:
                continue
            
            # Get crop bbox for this frame type
            crop_bbox = features[frame_type]['crop_bbox']
            
            # Process each object
            for object_name, object_data in cnos_features[frame_type]['objects'].items():
                # Initialize object in transformed features
                transformed_features[frame_type]['objects'][object_name] = {
                    'mask_files': object_data['mask_files'],
                    'scores': object_data['scores'],
                    'max_score': object_data['max_score'],
                    'max_score_idx': object_data['max_score_idx'],
                    'masks': [],
                    'orig_masks': []
                }
                
                # Copy original masks
                transformed_features[frame_type]['objects'][object_name]['masks'] = object_data['masks']
                
                # Transform masks if crop_bbox is available
                if crop_bbox:
                    # Transform all masks
                    orig_masks = self._transform_masks_to_original(
                        object_data['masks'], crop_bbox
                    )
                    transformed_features[frame_type]['objects'][object_name]['orig_masks'] = orig_masks
                    
                    # Add transformed max_score_mask if available # burasi daha deepe gidebilir mesela
                    max_idx = object_data['max_score_idx']
                    if max_idx >= 0 and max_idx < len(orig_masks):
                        transformed_features[frame_type]['objects'][object_name]['orig_max_score_mask'] = orig_masks[max_idx]
            
        return transformed_features
    
    def _transform_mask_to_original(self, mask: np.ndarray, crop_bbox: List[int]) -> np.ndarray:
        """
        Transform mask from cropped frame coordinates to original image coordinates.
        
        Args:
            mask: merged mask in cropped coordinates
            crop_bbox: Crop bounding box [x1, y1, x2, y2]
            
        Returns:
            transformed masks in original image coordinates
        """
        if mask is None or crop_bbox is None:
            return None
        
        # Parse crop bbox
        x1, y1, x2, y2 = crop_bbox
     
        # Use configured original image dimensions
        orig_width = self.orig_width
        orig_height = self.orig_height

        # Create empty mask for original image
        transformed_mask = np.zeros((orig_height, orig_width), dtype=mask.dtype)
        
        # Place cropped mask into original image coordinates
        try:
            transformed_mask[y1:y2, x1:x2] = mask
            # Log the number of True pixels in the mask
            # only for debug!
            true_pixels = np.sum(mask)
            logging.info(f"Placed mask with {true_pixels} True pixels into original coordinates")
        except ValueError as e:
            logging.error(f"Error transforming mask: {e}")
            logging.error(f"Mask shape: {mask.shape}, Crop: ({x1},{y1},{x2},{y2})")
            raise e
        
        return transformed_mask


    def _transform_masks_to_original(self, masks: List[np.ndarray], crop_bbox: List[int]) -> List[np.ndarray]:
        """
        Transform masks from cropped frame coordinates to original image coordinates.
        
        Args:
            masks: List of binary masks in cropped coordinates
            crop_bbox: Crop bounding box [x1, y1, x2, y2]
            
        Returns:
            List of transformed masks in original image coordinates
        """
        if not masks or crop_bbox is None:
            return []
        
        # Parse crop bbox
        x1, y1, x2, y2 = crop_bbox
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Use configured original image dimensions
        orig_width = self.orig_width
        orig_height = self.orig_height
        
        transformed_masks = []
        
        for mask in masks:
            # Resize mask to match crop dimensions if needed
            if mask.shape[0] != crop_height or mask.shape[1] != crop_width:
                logging.info(f"Resizing mask from {mask.shape} to ({crop_height}, {crop_width})")
                mask = cv2.resize(mask.astype(np.uint8), (crop_width, crop_height))
                mask = mask.astype(bool)
            
            # Create empty mask for original image
            transformed_mask = np.zeros((orig_height, orig_width), dtype=bool)
            
            # Place cropped mask into original image coordinates
            try:
                transformed_mask[y1:y2, x1:x2] = mask
                # Log the number of True pixels in the mask
                true_pixels = np.sum(mask)
                logging.info(f"Placed mask with {true_pixels} True pixels into original coordinates")
            except ValueError as e:
                logging.error(f"Error transforming mask: {e}")
                logging.error(f"Mask shape: {mask.shape}, Crop: ({x1},{y1},{x2},{y2})")
                # Create an empty mask as fallback
                transformed_mask = np.zeros((orig_height, orig_width), dtype=bool)
            
            transformed_masks.append(transformed_mask)
        
        return transformed_masks
    
    def get_max_score_by_frame(self, camera_view: str, start_frame: int, 
                              end_frame: int) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Get maximum detection score for each frame in a range.
        
        Args:
            camera_view: Camera view name
            start_frame: Starting frame index
            end_frame: Ending frame index
            
        Returns:
            Dictionary mapping frame indices to max scores by object
        """
        # Delegate to the CNOS loader which already has this functionality
        return self.cnos_loader.get_max_score_by_frame(camera_view, start_frame, end_frame)
    
    def get_available_objects(self, camera_view: str, frame_type: str) -> List[str]:
        """
        Get list of available objects for a specific camera view and frame type.
        
        Args:
            camera_view: Camera view name
            frame_type: Frame type ('L_frames' or 'R_frames')
            
        Returns:
            List of object names
        """
        return self.cnos_loader.get_available_objects(camera_view, frame_type)
    
    def get_frame_count(self, camera_view: str, frame_type: str) -> int:
        """
        Get total number of frames for a specific camera view and frame type.
        
        Args:
            camera_view: Camera view name
            frame_type: Frame type ('L_frames' or 'R_frames')
            
        Returns:
            Number of frames
        """
        return self.cnos_loader.get_frame_count(camera_view, frame_type)
    



if __name__ == "__main__":
    # Test the CNOSHAMERLoader
    cnos_hamer_loader = CNOSHAMERLoader(session_name="imi_session1_2", data_root_dir="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data")
    print("Testing CNOSHAMERLoader")
    
    # Test get_max_score_by_frame
    print("get_max_score_by_frame -> it should return me max score for each frame")
    print(cnos_hamer_loader.get_max_score_by_frame(camera_view="cam_side_l", start_frame=100, end_frame=101))
    
    # Test get_available_objects
    print("get_available_objects", cnos_hamer_loader.get_available_objects(camera_view="cam_top", frame_type="L_frames"))
    
    # Test get_frame_count
    print("get_frame_count", cnos_hamer_loader.get_frame_count(camera_view="cam_top", frame_type="L_frames"))
    
    # Test the combined loader functionality
    print("\nLoading features")
    features = cnos_hamer_loader.load_features(camera_view="cam_top", frame_idx=100)
    print(f"L_frames success: {features['L_frames']['success']}")
    print(f"R_frames success: {features['R_frames']['success']}")
    
    # Test loading masks and transforming to original coordinates
    print("\nLoading masks and transforming to original coordinates")
    orig_masks = cnos_hamer_loader.load_original_masks(camera_view="cam_top", frame_idx=100)
    
    # Print information about the masks
    for frame_type in ['L_frames', 'R_frames']:
        if orig_masks[frame_type]['success']:
            print(f"\n{frame_type} crop_bbox:", orig_masks[frame_type]['crop_bbox'])
            print(f"{frame_type} objects:")
            for obj_name, obj_data in orig_masks[frame_type]['objects'].items():
                print(f"  - {obj_name}: {len(obj_data['masks'])} masks, max score: {obj_data['max_score']:.4f}")
                if 'orig_masks' in obj_data:
                    print(f"    - {len(obj_data['orig_masks'])} transformed masks")
                    # Print shape of first transformed mask if available
                    if obj_data['orig_masks']:
                        shape = obj_data['orig_masks'][0].shape
                        print(f"    - First transformed mask shape: {shape}")

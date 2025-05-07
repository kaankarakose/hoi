"""
Combined feature loader class for integrating CNOS and HAMER features.
This loader handles the transformation of object segmentation masks from 
hand-cropped frames back to original image coordinates.
"""

import os
import numpy as np
import logging
import random
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from PIL import Image

# Add the parent directory to the path dynamically
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from object_interaction_detection.dataloaders.base_loader import BaseDataLoader
from object_interaction_detection.dataloaders.cnos_loader import CNOSLoader
from object_interaction_detection.dataloaders.hamer_loader import HAMERLoader
from object_interaction_detection.utils.utils import load_rle_mask, rle2mask

# Predefined colors for each object (BGR format for OpenCV)
OBJECT_COLORS = {
    'AMF1': [161, 113, 146],      # Color for AMF1
    'AMF2': [131, 136, 51],       # Color for AMF2
    'AMF3': [130, 171, 146],      # Color for AMF3
    'BOX': [112, 62, 54],         # Color for BOX
    'CUP': [146, 79, 156],        # Color for CUP
    'DINOSAUR': [99, 93, 79],     # Color for DINOSAUR
    'FIRETRUCK': [94, 121, 171],  # Color for FIRETRUCK
    'HAIRBRUSH': [59, 175, 143],  # Color for HAIRBRUSH
    'PINCER': [210, 196, 209],    # Color for PINCER 
    'WRENCH': [67, 126, 87],      # Color for WRENCH
}

logging.basicConfig(level=logging.INFO)

class CombinedLoader(BaseDataLoader):
    """
    Combined data loader that integrates CNOS and HAMER features.
    
    This loader transforms object segmentation masks from cropped hand
    frames back to original image dimensions using hand crop bounding boxes.
    It also handles merging of overlapping high-scoring objects.
    """
    
    def __init__(self, 
                 session_name: str, 
                 data_root_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the combined loader.
        
        Args:
            session_name: Name of the session to process
            data_root_dir: Root directory for all data
            config: Configuration parameters (optional)
        """
        # Set defaults for paths
        config = config or {}
        
        # Add default score threshold configuration
        config.setdefault('score_threshold', 0.45)  # Default threshold of 0.5
        config.setdefault('frames_dir', os.path.join(data_root_dir, 'orginal_frames'))  # For visualization
        
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # Initialize CNOS and HAMER loaders
        self.cnos_loader = CNOSLoader(session_name, data_root_dir, config)
        self.hamer_loader = HAMERLoader(session_name, data_root_dir, config)
        
        # Get camera views and frame types from CNOS loader
        self.camera_views = self.cnos_loader.camera_views
        self.frame_types = self.cnos_loader.frame_types
        
        # Mapping between frame types and hand types
        self.frame_to_hand_map = {
            'L_frames': 'left',
            'R_frames': 'right'
        }
        
        # Store the score threshold for filtering objects
        self.score_threshold = config['score_threshold']
        
        # Set up object colors for visualization
        self.object_colors = {}
        
        logging.info(f"Initialized combined loader for session {session_name} with score threshold {self.score_threshold}")
    def _merge_objects_mask(self, combined_features):

        # Process each frame type (L_frames, R_frames)
        for frame_type in self.frame_types:
                hand_type = self.frame_to_hand_map[frame_type]
                hand_key = f'{hand_type}_hand'
                
                # Check if both hand data and object segmentation are available
                if (hamer_features[hand_key]['success'] and 
                    cnos_features[frame_type]['success']):
                    
                    # Get crop bounding box for this hand
                    bbox = hamer_features[hand_key]['bbox'] # crop bbox in original image coordinates, bbox is actual hand coordinates.
                    # Get crop bounding box for this hand
                    crop_bbox = hamer_features[hand_key]['crop_bbox'] # crop bbox in original image coordinates, bbox is actual hand coordinates.
                    
                    if crop_bbox is None:
                        logging.warning(f"No crop_bbox found for {hand_type} hand at frame {frame_idx}")
                        raise ValueError("No crop_bbox found for {hand_type} hand at frame {frame_idx}")
                    
                    # Process each object in this frame
                    for object_name, object_data in cnos_features[frame_type]['objects'].items():
                        # Check if max score passes the threshold
                        if object_data['max_score'] < self.score_threshold:
                            logging.debug(f"Skipping object {object_name} with score {object_data['max_score']} < threshold {self.score_threshold}")
                            continue
                        
                        # Load mask data if needed
                        masks = []
                        scores = []
                        
                        # Only include masks with scores above threshold
                        for i, score in enumerate(object_data['scores']):
                            #if score >= self.score_threshold: # this is important! # TODO: apply object specific thresholds.? 
                            mask_file = object_data['mask_files'][i] # mask file path
                            mask = load_rle_mask(mask_file)
                            mask = rle2mask(mask) # give back np array of the mask.
                            masks.append(mask)
                            scores.append(score)
                        
                        # Skip if no masks passed the threshold
                        if not masks:
                            continue

                        # Assign values to masks
                        valued_masks = []
                        for i, mask in enumerate(masks):
                            # Assign a value from 2 to 10 (cycling if needed)
                            value = i + 2  # This gives values 2,3,4,.5,6,7, .... except from the 0 and 1 because it may could make error.
                            valued_mask = mask * value  # Replace 1s with the value
                            valued_masks.append(valued_mask)
                        result = valued_masks[0].copy()

                        # For each additional mask
                        for mask in valued_masks[1:]:
                            # Keep value only if both masks have the same non-zero value
                            # If either mask is 0 or they have different values, set to 0
                            result = np.where((result == mask) & (result != 0), result, 0)
                        # all mask are merged into one mask.
                        # Transform masks to original image coordinates
                        transformed_masks = self._transform_mask_to_original(
                            result, crop_bbox
                        )
                        
                        # Recompute max score index after filtering
                        max_score_idx = np.argmax(scores) if scores else -1
                        max_score = scores[max_score_idx] if max_score_idx >= 0 else 0.0
                        
                        # Store in combined features
                        combined_features['preprocess'][hand_key]['objects'][object_name] = {
                            'mask': transformed_mask,
                            'scores': scores,
                            'max_score': max_score,
                            'max_score_idx': max_score_idx
                        }
                    # Mark as successful if we processed any objects
                    if combined_features['preprocess'][hand_key]['objects']:
                        combined_features['preprocess'][hand_key]['success'] = True
        return combined_features
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load combined features for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing combined features -- Mask are merged!
        """
        # Check for cached features
        cached_features = self._get_cached_features(camera_view, frame_idx)
        if cached_features is not None:
            return cached_features
        
        # Load HAMER features (hand information)
        hamer_features = self.hamer_loader.load_features(camera_view, frame_idx)
        
        # Load CNOS features (object segmentation)
        cnos_features = self.cnos_loader.load_features(camera_view, frame_idx)
        
        # Initialize combined features dictionary
        combined_features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'hamer': hamer_features,  # Original HAMER features
            'cnos': cnos_features,    # Original CNOS features
            'preprocess': {  # preprocess features
                'left_hand': {
                    'objects': {},
                    'success': False
                },
                'right_hand': {
                    'objects': {},
                    'success': False
                },
            'combined': {         # Combined objects from both hands
                'objects': {},
                'success': False
            }
            }
        }
        
        ## merged object masks into one
        combined_features =  self._merge_objects_mask(combined_features) # objects mask are merged into one


        # Merge overlapping objects from both hands
        self._merge_objects_by_hand(combined_features)
        
        # Cache features (without large arrays to save memory)
        cache_features = self._prepare_cache_features(combined_features)
        self._cache_features(camera_view, frame_idx, cache_features)
        
        ### Perform the combined mask!


        


        return combined_features
    
    def _prepare_cache_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a memory-efficient version of features for caching.
        
        Args:
            features: Full features dictionary
            
        Returns:
            Memory-efficient features dictionary for caching
        """
        # Create a deep copy without the large data arrays
        cache_features = {
            'frame_idx': features['frame_idx'],
            'camera_view': features['camera_view'],
            'merged': {
                'left_hand': {
                    'success': features['merged']['left_hand']['success'],
                    'objects': {}
                },
                'right_hand': {
                    'success': features['merged']['right_hand']['success'],
                    'objects': {}
                },
                'combined': {
                    'success': features['merged']['combined']['success'],
                    'objects': {}
                }
            }
        }
        
        # Add basic hand information from HAMER that we need for visualization
        cache_features['hamer'] = {
            'left_hand': {'crop_bbox': features['hamer']['left_hand'].get('crop_bbox')},
            'right_hand': {'crop_bbox': features['hamer']['right_hand'].get('crop_bbox')}
        }
        
        # Don't store the rest of original HAMER and CNOS features to save memory
        
        # Copy object metadata but not masks
        for hand_key in ['left_hand', 'right_hand']:
            for obj_name, obj_data in features['merged'][hand_key]['objects'].items():
                cache_features['merged'][hand_key]['objects'][obj_name] = {
                    'scores': obj_data['scores'],
                    'max_score': obj_data['max_score'],
                    'max_score_idx': obj_data['max_score_idx']
                }
        
        # Same for combined objects
        for obj_name, obj_data in features['merged']['combined']['objects'].items():
            cache_features['merged']['combined']['objects'][obj_name] = {
                'score': obj_data['score'],
                'source_hand': obj_data['source_hand']
            }
        
        return cache_features
    
    def _transform_mask_to_original(self, mask: np.ndarray, crop_bbox: List[int]) -> np.ndarray:
        """
        Transform mask from cropped frame coordinates to original image coordinates.
        
        Args:
            mask: merged mask in cropped coordinates
            crop_bbox: Crop bounding box [x1, y1, x2, y2]
            
        Returns:
            transformed masks in original image coordinates
        """
        if not masks or crop_bbox is None:
            return []
        
        # Parse crop bbox
        x1, y1, x2, y2 = crop_bbox
     
        # Original image dimensions can be determined from context or configuration
        # For now, we'll create masks that are large enough to contain the crop
        orig_width = 1920 # TODO: No need to be hard coded!
        orig_height = 1200  # TODO: No need to be hard coded!

    

        # Create empty mask for original image+-
        transformed_mask = np.zeros((orig_height, orig_width), dtype = mask.dtype)
        
        # Place cropped mask into original image coordinates
        try:
            transformed_mask[y1:y2, x1:x2] = mask
            # Log the number of True pixels in the mask
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
        
        # Original image dimensions can be determined from context or configuration
        # For now, we'll create masks that are large enough to contain the crop
        orig_width = 1920 # TODO: No need to be hard coded!
        orig_height = 1200  # TODO: No need to be hard coded!

        
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
    
    def _merge_objects_by_hand(self, features: Dict[str, Any]) -> None:
        """
        Merge overlapping objects from both hands based on scores.
        
        Args:
            features: Combined features dictionary to be updated in-place
        """
        combined_objects = {}
        
        # Process each hand
        if features['merged']['left_hand']['success']:
            raise ValueError("Left hand is not successful")
            return
        if features['merged']['right_hand']['success']:
            raise ValueError("Right hand is not successful")
            return


        left_hand = features['merged']['left_hand']
        # Process each object in this frame / I need to get the objects from cnos features. scores and masks.
        for object_name, object_data in features['cnos']['left_hand']['objects'].items():
            # Check if max score passes the threshold
            if object_data['max_score'] < self.score_threshold:
                logging.debug(f"Skipping object {object_name} with score {object_data['max_score']} < threshold {self.score_threshold}")
                continue
            
            # Load mask data if needed
            masks = []
            scores = []
            
            # Only include masks with scores above threshold
            for i, score in enumerate(object_data['scores']):
                if score >= self.score_threshold: # this is important! # TODO: apply object specific thresholds.? 
                    mask_file = object_data['mask_files'][i] # mask file path
                    mask = load_rle_mask(mask_file)
                    mask = rle2mask(mask) # give back np array of the mask.
                    masks.append(mask)
                    scores.append(score)
                         
        # Process each object in this hand
        for object_name, object_data in features['merged'][hand_key]['objects'].items():
            # Skip objects with scores below threshold
            if object_data['max_score'] < self.score_threshold:
                continue
            
            # Use object's max score mask
            if 'max_score_mask' not in object_data:
                continue
                
            mask = object_data['max_score_mask']
            score = object_data['max_score']
            
            # Check if this object already exists in combined objects
            if object_name in combined_objects:
                # If existing object has higher score, keep it
                if combined_objects[object_name]['score'] >= score:
                    continue
            

            # Add or update object in combined objects
            combined_objects[object_name] = {
                'mask': mask,
                'score': score,
                'source_hand': hand_key
            }
    
        # Store combined objects in features
        for object_name, object_data in combined_objects.items():
            features['merged']['combined']['objects'][object_name] = object_data
        
        # Mark as successful if we have any combined objects
        if combined_objects:
            features['merged']['combined']['success'] = True
    
    def get_object_mask(self, camera_view: str, frame_idx: int, 
                       object_name: str) -> Optional[np.ndarray]:
        """
        Get mask for a specific object in original image coordinates.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            object_name: Object name
            
        Returns:
            Binary mask in original image coordinates or None if not available
        """
        features = self.load_features(camera_view, frame_idx)
        
        if (not features['merged']['combined']['success'] or 
            object_name not in features['merged']['combined']['objects']):
            return None
            
        return features['merged']['combined']['objects'][object_name].get('mask')
    
    def get_available_objects(self, camera_view: str, frame_idx: int) -> List[str]:
        """
        Get list of available objects for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            List of object names
        """
        features = self.load_features(camera_view, frame_idx)
        
        if not features['merged']['combined']['success']:
            return []
            
        return list(features['merged']['combined']['objects'].keys())
    
    def get_object_scores(self, camera_view: str, frame_idx: int) -> Dict[str, float]:
        """
        Get scores for all objects in a specific frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary mapping object names to scores
        """
        features = self.load_features(camera_view, frame_idx)
        
        if not features['merged']['combined']['success']:
            return {}
            
        return {
            obj_name: obj_data['score']
            for obj_name, obj_data in features['merged']['combined']['objects'].items()
        }
    

    def set_score_threshold(self, threshold: float) -> None:
        """
        Set the confidence score threshold for object filtering.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if threshold < 0.0 or threshold > 1.0:
            logging.warning(f"Invalid threshold value {threshold}. Must be between 0.0 and 1.0. Using current value {self.score_threshold}.")
            return
            
        self.score_threshold = threshold
        logging.info(f"Updated score threshold to {self.score_threshold}")
        
        # Clear cache as threshold change affects results
        self.clear_cache()
    
    def _get_object_color(self, object_name: str) -> Tuple[int, int, int]:
        """
        Get a consistent color for an object using predefined color mapping.
        
        Args:
            object_name: Object name
            
        Returns:
            BGR color tuple
        """
        # Use predefined colors if available
        if object_name in OBJECT_COLORS:
            return OBJECT_COLORS[object_name]
            
        # Fall back to existing color if generated before
        if object_name in self.object_colors:
            return self.object_colors[object_name]
            
        # Generate a random color as a last resort
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        self.object_colors[object_name] = (b, g, r)  # OpenCV uses BGR
        
        return self.object_colors[object_name]
    
    def _load_original_frame(self, camera_view: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load the original frame image.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Frame image as numpy array or None if not available
        """
        # Frame filename format may vary depending on your dataset
        frame_path = os.path.join(self.config['frames_dir'], self.session_name, camera_view, f"frame_{frame_idx + 1:04d}.jpg")
     
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        
        logging.warning(f"Frame image not found for {camera_view}, frame {frame_idx}")
        return None
    
    
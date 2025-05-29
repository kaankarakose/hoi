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

# Use absolute or relative imports depending on how the script is run
if __name__ == "__main__":
    # When run directly as a script, use absolute imports
    sys.path.append(os.path.dirname(current_dir))  # Add parent of dataloaders
    from object_interaction_detection.dataloaders.helper_loader.base_loader import BaseDataLoader
    from object_interaction_detection.dataloaders.cnos_hamer import CNOSHAMERLoader 
    from object_interaction_detection.dataloaders.helper_loader.hamer_loader import HAMERLoader
else:
    # When imported as a module, use relative imports
    from .helper_loader.base_loader import BaseDataLoader
    from .helper_loader.hamer_loader import HAMERLoader
    from .cnos_hamer import CNOSHAMERLoader

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

    This merged Left and Right hand features into one.

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
        config.setdefault('frames_dir', os.path.join(data_root_dir, 'orginal_frames'))  # For visualization
        print(config, "combined")
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # Initialize CNOS and HAMER loaders
        self.cnos_loader = CNOSHAMERLoader(session_name, data_root_dir, config)
        self.hamer_loader = HAMERLoader(session_name, data_root_dir, config) # just need for valid index somehow
        # Get camera views and frame types from CNOS loader
        self.camera_views = self.cnos_loader.camera_views
        self.frame_types = self.cnos_loader.frame_type_to_hand_type
        
        # Mapping between frame types and hand types
        self.frame_to_hand_map = {
            'L_frames': 'left',
            'R_frames': 'right'
        }
        # Store the score threshold for filtering objects
        self.score_threshold = config['score_threshold']
        # Set up object colors for visualization
        self.object_colors = OBJECT_COLORS

        logging.info(f"Initialized combined loader for session {session_name} with score threshold {self.score_threshold}")
    
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load combined features for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing combined features -- Mask are merged!
        """
        # Load CNOS features (object segmentation)
        cnos_features = self.cnos_loader.load_original_masks(camera_view, frame_idx) # not load features because i need original mask
        # Initialize combined features dictionary
        combined_features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'cnos': cnos_features,    # Original CNOS features
            'merged' : {
                    'mask': None,
                    'object_id_map': None,
                    'success': False,
                }
        }
        # Merge overlapping objects from both hands
        self._merge_objects_by_hand(combined_features)
        return combined_features

    def _merge_objects_by_hand(self, features: Dict[str, Any]) -> None:
        """
        Merge overlapping objects from both hands based on scores using NumPy operations.
        
        Args:
            features: Combined features dictionary to be updated in-place
        """
        # Process left hand objects
        left_mask_candidates = []
        if features['cnos']['L_frames']['success']:
            left_hand = features['cnos']['L_frames']
            for object_name, object_data in left_hand['objects'].items():
                object_data = left_hand['objects'][object_name]
                if object_data['max_score'] >= self.score_threshold and 'orig_max_score_mask' in object_data:
                    left_mask_candidates.append((object_name, object_data['orig_max_score_mask'], object_data['max_score']))
        
        # Process right hand objects
        right_mask_candidates = []
        if features['cnos']['R_frames']['success']:
            right_hand = features['cnos']['R_frames']
            for object_name, object_data in right_hand['objects'].items():
                object_data = right_hand['objects'][object_name]
                if object_data['max_score'] >= self.score_threshold and 'orig_max_score_mask' in object_data:
                    right_mask_candidates.append((object_name, object_data['orig_max_score_mask'], object_data['max_score']))
        
        # Combine all candidates
        all_candidates = left_mask_candidates + right_mask_candidates
        
        if not all_candidates:
            features['cnos']['combined_objects'] = {}
            return
        
        # Get the mask shape from the first candidate
        mask_shape = all_candidates[0][1].shape
        
        # Create mapping from object names to unique IDs (starting from 2)
        unique_objects = sorted(set(candidate[0] for candidate in all_candidates))
        object_id_map = {obj_name: idx + 2 for idx, obj_name in enumerate(unique_objects)} # Escape from 0 and 1 True False

        
        # Create a score map and ID map with same dimensions as masks
        score_map = np.zeros(mask_shape, dtype=np.float32)
        id_map = np.zeros(mask_shape, dtype=np.int32)
        all_candidates.sort(key=lambda x: x[2], reverse=True) # sort by score
        # Process each mask using NumPy operations
        for object_name, mask, score in all_candidates:
            object_id = object_id_map[object_name]
            
            # Create a mask for pixels where current object has higher score than existing
            # or where no object exists yet
            better_score_mask = np.logical_or(
                    (mask > 0) & (score_map == 0),  # No object yet
                    (mask > 0) & (score > score_map)  # Current object has higher score
                )
            # Update score_map and id_map where current object is better
            score_map = np.where(better_score_mask, score, score_map)
            id_map = np.where(better_score_mask, object_id, id_map)
        
        # Create combined objects dictionary
        # combined_objects = {}
        # for object_name, object_id in object_id_map.items():
        #     # Create binary mask for this object
        #     object_mask = (id_map == object_id).astype(np.uint8)
            
        #     # Only include objects that appear in the final map
        #     if np.any(object_mask):
        #         # Find the maximum score for this object
        #         max_score = max(score for name, _, score in all_candidates if name == object_name)
                
        #         combined_objects[object_name] = {
        #             'max_score': max_score,
        #             'max_score_mask': object_mask
        #         }
        
        # Add a single merged mask to the output
        features['merged']['mask'] = id_map.copy()
        features['merged']['object_id_map'] = object_id_map
        features['merged']['success'] = True
    
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
    
    
def visualize_object_masks(id_map, rgb_frame, object_id_map, object_colors, alpha=0.5):
    """
    Create a visualization of object masks overlaid on an RGB frame.
    
    Args:
        id_map: NumPy array where each pixel value is an object ID
        rgb_frame: Original RGB image as a NumPy array
        object_id_map: Dictionary mapping object names to IDs
        object_colors: Dictionary mapping object names to RGB colors
        alpha: Transparency factor for the overlay (0-1)
        
    Returns:
        Visualization as a NumPy array
    """
    # Create a copy of the RGB frame as float for blending
    visualization = rgb_frame.astype(np.float32).copy()
    
    # Create a reverse mapping from ID to object name
    id_to_name = {id_val: name for name, id_val in object_id_map.items()}

    # Create a colored mask
    colored_mask = np.zeros_like(visualization)
    
    # Extract unique object IDs from the id_map (excluding 0/background)
    unique_ids = np.unique(id_map)
    unique_ids = unique_ids[unique_ids > 0]
    
    # For each object ID in the id_map
    for obj_id in unique_ids:
        # Get the object name
        if obj_id in id_to_name:
            obj_name = id_to_name[obj_id]
            
            # Get the color for this object
            if obj_name in object_colors:
                color = np.array(object_colors[obj_name], dtype=np.float32)
                
                # Create a binary mask for this object
                obj_mask = (id_map == obj_id)
                
                # Apply the color to this object in the colored mask
                for c in range(3):  # RGB channels
                    # This is the key fix - properly apply the color to the mask
                    colored_mask[:,:,c][obj_mask] = color[c]
    
    # Create the mask for all objects
    mask = (id_map > 0)
    
    # Blend the colored mask with the original frame
    for c in range(3):  # RGB channels
        visualization[:,:,c] = np.where(
            mask,
            (1 - alpha) * visualization[:,:,c] + alpha * colored_mask[:,:,c],
            visualization[:,:,c]
        )
    
    # Convert back to uint8
    visualization = np.clip(visualization, 0, 255).astype(np.uint8)
    
    return visualization

if __name__ == "__main__":
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from object_interaction_detection.dataloaders.combined_loader import CombinedLoader
    
    # Initialize the Combined Loader
    session_name = "imi_session1_6"
    data_root_dir = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data"
    
    # Create a configuration with a specific score threshold
    config = {
        'score_threshold': 0.35,
        'frames_dir': f"{data_root_dir}/orginal_frames"
    }
    
    # Initialize the loader
    combined_loader = CombinedLoader(session_name=session_name, data_root_dir=data_root_dir, config=config)
    print("Initialized CombinedLoader")
    
    # Get valid frame indices from the CNOS loader
    valid_indices = combined_loader.cnos_loader.hamer_loader.get_valid_frame_idx()
    print(f"Available camera views: {combined_loader.camera_views}")

    index = random.choice(valid_indices['cam_side_r']['left'])

    print("Left testing:")
    features = combined_loader.load_features(camera_view='cam_side_r', frame_idx=index)

    final_mask = features['merged']['mask']
    object_id_map = features['merged']['object_id_map']
    frame = combined_loader._load_original_frame(camera_view='cam_side_r', frame_idx=index)

    visualization = visualize_object_masks(final_mask, frame, object_id_map, combined_loader.object_colors)
    
    print(features)
    cv2.imwrite('Visualization.png', visualization)
    
    

    
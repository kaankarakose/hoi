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
else:
    # When imported as a module, use relative imports
    from .helper_loader.base_loader import BaseDataLoader
    from cnos_hamer import CNOSHAMERLoader

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
        config.setdefault('score_threshold', 0.45)  # Default threshold of 0.5
        config.setdefault('frames_dir', os.path.join(data_root_dir, 'orginal_frames'))  # For visualization
        
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # Initialize CNOS and HAMER loaders
        self.cnos_loader = CNOSHAMERLoader(session_name, data_root_dir, config)
        
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
        
      
        # Load CNOS features (object segmentation)
        cnos_features = self.cnos_loader.load_features(camera_view, frame_idx)
        
        # Initialize combined features dictionary
        combined_features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'cnos': cnos_features,    # Original CNOS features
            'merged' : {
                    'mask': None,
                    'success': False,
                }
        }
        

        # Merge overlapping objects from both hands
        self._merge_objects_by_hand(combined_features)
        
        # Cache features (without large arrays to save memory)
        cache_features = self._prepare_cache_features(combined_features)
        self._cache_features(camera_view, frame_idx, cache_features)


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
            'merged': 
                 {
                    'success': features['merged']['combined']['success'],
                    'objects': {}
                }
            }

        # Initialize combined features dictionary
        combined_features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'cnos': cnos_features,    # Original CNOS features
            'merged' : {
                    'success': False,
                    'objects': {}
                }
        }
            
        return cache_features
    
    
    
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
                if object_data['max_score'] >= self.score_threshold:
                    left_mask_candidates.append((object_name, object_data['max_score_mask'], object_data['max_score']))
        
        # Process right hand objects
        right_mask_candidates = []
        if features['cnos']['R_frames']['success']:
            right_hand = features['cnos']['R_frames']
            for object_name, object_data in right_hand['objects'].items():
                if object_data['max_score'] >= self.score_threshold:
                    right_mask_candidates.append((object_name, object_data['max_score_mask'], object_data['max_score']))
        
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
        
        # Process each mask using NumPy operations
        for object_name, mask, score in all_candidates:
            object_id = object_id_map[object_name]
            
            # Create a mask for pixels where current object has higher score than existing
            # or where no object exists yet
            better_score_mask = np.logical_or(
                mask > 0 & score_map == 0,  # No object yet
                mask > 0 & score > score_map  # Current object has higher score
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
        


              
                    
               
    
    def get_object_mask(self, camera_view: str, frame_idx: int, object_name: str) -> Optional[np.ndarray]:
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
        'score_threshold': 0.45,
        'frames_dir': f"{data_root_dir}/original_frames"
    }
    
    # Initialize the loader
    combined_loader = CombinedLoader(session_name=session_name, data_root_dir=data_root_dir, config=config)
    print("Initialized CombinedLoader")
    
    # Get valid frame indices from the CNOS loader
    valid_indices = combined_loader.cnos_loader.hamer_loader.get_valid_frame_idx()
    print(f"Available camera views: {combined_loader.camera_views}")
    
    # Choose a random valid frame from one camera view
    camera_view = "cam_top"  # Using cam_top as in the example
    if camera_view in valid_indices and 'left' in valid_indices[camera_view]:
        valid_frames = valid_indices[camera_view]['left']
        if valid_frames:
            test_frame_idx = random.choice(valid_frames)
            print(f"\nSelected test frame: {camera_view}, frame {test_frame_idx}")
        else:
            print(f"No valid frames found for {camera_view}, 'left'")
            # Try to find any valid frame in any camera
            for cam in combined_loader.camera_views:
                if cam in valid_indices and 'left' in valid_indices[cam] and valid_indices[cam]['left']:
                    camera_view = cam
                    test_frame_idx = random.choice(valid_indices[cam]['left'])
                    print(f"Using alternative: {camera_view}, frame {test_frame_idx}")
                    break
            else:
                print("No valid frames found in any camera")
                exit(1)
    else:
        print(f"Camera view {camera_view} or 'left' hand not found in valid indices")
        exit(1)
    
    # Test 1: Load features and check merged objects
    print("\nTest 1: Loading features")
    features = combined_loader.load_features(camera_view=camera_view, frame_idx=test_frame_idx)
    
    # Check if merged features exist
    if 'cnos' in features and 'combined_objects' in features['cnos']:
        print("Merged objects found:")
        for obj_name, obj_data in features['cnos']['combined_objects'].items():
            print(f"  - {obj_name}: max score = {obj_data['max_score']:.4f}, mask shape = {obj_data['max_score_mask'].shape}")
    else:
        print("No merged objects found in features")
    
    # Test 2: Get available objects
    print("\nTest 2: Getting available objects")
    objects = combined_loader.get_available_objects(camera_view=camera_view, frame_idx=test_frame_idx)
    print(f"Available objects: {objects}")
    
    # Test 3: Get object scores
    print("\nTest 3: Getting object scores")
    scores = combined_loader.get_object_scores(camera_view=camera_view, frame_idx=test_frame_idx)
    print(f"Object scores: {scores}")
    
    # Test 4: Change score threshold and reload features
    print("\nTest 4: Changing score threshold")
    new_threshold = 0.7
    combined_loader.set_score_threshold(threshold=new_threshold)
    print(f"Set score threshold to {new_threshold}")
    
    # Reload features with new threshold
    features_high_threshold = combined_loader.load_features(camera_view=camera_view, frame_idx=test_frame_idx)
    
    # Check merged objects with new threshold
    if 'cnos' in features_high_threshold and 'combined_objects' in features_high_threshold['cnos']:
        print("Merged objects with higher threshold:")
        for obj_name, obj_data in features_high_threshold['cnos']['combined_objects'].items():
            print(f"  - {obj_name}: max score = {obj_data['max_score']:.4f}")
    else:
        print("No merged objects found with higher threshold")
    
    # Test 5: Visualize the merged masks
    print("\nTest 5: Visualizing merged masks")
    # Try to load the original frame
    original_frame = combined_loader._load_original_frame(camera_view=camera_view, frame_idx=test_frame_idx)
    
    if original_frame is not None:
        # Create a copy for visualization
        vis_frame = original_frame.copy()
        
        # Get all merged objects
        if 'cnos' in features and 'combined_objects' in features['cnos']:
            for obj_name, obj_data in features['cnos']['combined_objects'].items():
                # Get the object mask
                obj_mask = obj_data['max_score_mask']
                
                # Get color for this object
                color = combined_loader._get_object_color(obj_name)
                
                # Apply mask to frame - create an overlay
                mask_overlay = np.zeros_like(original_frame)
                mask_overlay[obj_mask > 0] = color
                
                # Blend with original frame
                alpha = 0.5  # Transparency
                cv2.addWeighted(mask_overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
                
                # Add object name
                # Find centroid of mask
                if np.any(obj_mask):
                    y_indices, x_indices = np.where(obj_mask > 0)
                    centroid_y = int(np.mean(y_indices))
                    centroid_x = int(np.mean(x_indices))
                    
                    # Put text at centroid
                    cv2.putText(vis_frame, obj_name, (centroid_x, centroid_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Merged Objects - {camera_view}, Frame {test_frame_idx}")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the visualization
            output_dir = "visualizations"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/merged_objects_{camera_view}_{test_frame_idx}.png")
            print(f"Visualization saved to {output_dir}/merged_objects_{camera_view}_{test_frame_idx}.png")
            
            # Show the plot
            plt.show()
        else:
            print("No merged objects to visualize")
    else:
        print("Original frame not found for visualization")
    
    print("\nTesting completed")
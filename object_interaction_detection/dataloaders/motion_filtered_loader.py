"""
Motion-Filtered Object Loader class that integrates object segmentation masks
with optical flow to filter out non-moving objects.
"""

import os
import numpy as np
import logging
import cv2
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the parent directory to the path dynamically
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Use absolute or relative imports depending on how the script is run
if __name__ == "__main__":
    # When run directly as a script, use absolute imports
    from object_interaction_detection.dataloaders.helper_loader.base_loader import BaseDataLoader
    from object_interaction_detection.dataloaders.combined_loader import CombinedLoader, visualize_object_masks
    from object_interaction_detection.dataloaders.helper_loader.flow_loader import FlowLoader
else:
    # When imported as a module, use relative imports
    from .helper_loader.base_loader import BaseDataLoader
    from .combined_loader import CombinedLoader, visualize_object_masks
    from .helper_loader.flow_loader import FlowLoader

logging.basicConfig(level=logging.INFO)

class MotionFilteredLoader(BaseDataLoader):
    """
    Data loader that combines object segmentation masks with optical flow
    to filter out non-moving objects.
    
    This loader builds on the CombinedLoader (which merges left and right hand features)
    and uses optical flow to determine which objects are moving, removing non-moving
    objects from the final mask.
    """
    
    def __init__(self, 
                 session_name: str, 
                 data_root_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the motion-filtered loader.
        
        Args:
            session_name: Name of the session to process
            data_root_dir: Root directory for all data
            config: Configuration parameters (optional)
        """
        # Set defaults for paths
        config = config or {}
        
        
        #flow specific paths
        flow_root_dir = "/nas/project_data/B1_Behavior/rush/kaan/old_method/processed_data"

        # Add default motion threshold configuration
        config.setdefault('low_threshold', 0.28)
        config.setdefault('score_threshold', 0.45)
        config.setdefault('motion_threshold', 0.05)  # Default threshold for motion
        config.setdefault('temporal_window', 1)      # Default window for flow aggregation
        config.setdefault('frames_dir', os.path.join(data_root_dir, 'orginal_frames'))  # For visualization
        
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # Initialize component loaders
        self.combined_loader = CombinedLoader(session_name, data_root_dir, config)
        self.flow_loader = FlowLoader(session_name, flow_root_dir, config)
        
        # Copy camera views from combined loader
        self.camera_views = self.combined_loader.camera_views
        
        # Store configuration parameters
        self.motion_threshold = config['motion_threshold']
        self.temporal_window = config['temporal_window']
        
        # Initialize features cache
        self.features_cache = {}
        
        logging.info(f"Initialized motion-filtered loader for session {session_name} "
                     f"with motion threshold {self.motion_threshold}")
    
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load combined features with motion filtering for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing motion-filtered features
        """
        # Check for cached features
        cached_features = self._get_cached_features(camera_view, frame_idx)
        if cached_features is not None:
            return cached_features
        
        # Load combined features (object masks)
        combined_features = self.combined_loader.load_features(camera_view, frame_idx)
        
        
        # Initialize motion filtered features
        motion_filtered_features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'combined': combined_features,  # Store original combined features
            'motion_filtered': {
                'mask': None,
                'object_id_map': None,
                'success': False,
                'moving_objects': []
            }
        }
        
        # Check if combined features were loaded successfully
        if not combined_features.get('merged', {}).get('success', False):
            self._cache_features(camera_view, frame_idx, motion_filtered_features)
            return motion_filtered_features
        
        # Get the merged mask and object ID map
        merged_mask = combined_features['merged']['mask']
        object_id_map = combined_features['merged']['object_id_map']

        if merged_mask is None or object_id_map is None:
            self._cache_features(camera_view, frame_idx, motion_filtered_features)
            return motion_filtered_features
        
        # Apply motion filtering to the merged mask
        motion_filtered_mask, moving_objects = self._filter_by_motion(
            camera_view, frame_idx, merged_mask, object_id_map
        )
     
        
        # Update motion filtered features
        motion_filtered_features['motion_filtered']['mask'] = motion_filtered_mask
        motion_filtered_features['motion_filtered']['object_id_map'] = object_id_map
        motion_filtered_features['motion_filtered']['success'] = True
        motion_filtered_features['motion_filtered']['moving_objects'] = moving_objects
        
        # Cache features (without large arrays to save memory)
        cache_features = self._prepare_cache_features(motion_filtered_features)
        self._cache_features(camera_view, frame_idx, cache_features)
        
        return motion_filtered_features
    
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
            'combined': {
                'frame_idx': features['combined']['frame_idx'],
                'camera_view': features['combined']['camera_view'],
                'merged': {
                    'success': features['combined']['merged']['success'],
                    'object_id_map': features['combined']['merged']['object_id_map']
                    # Exclude the large 'mask' array
                }
            },
            'motion_filtered': {
                'success': features['motion_filtered']['success'],
                'object_id_map': features['motion_filtered']['object_id_map'],
                'moving_objects': features['motion_filtered']['moving_objects'],
                'mask': features['motion_filtered']['mask']
                # Exclude the large 'mask' array
            }
        }            
        return cache_features
    
    def _filter_by_motion(self, 
                         camera_view: str, 
                         frame_idx: int, 
                         merged_mask: np.ndarray, 
                         object_id_map: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
        """
        Filter object masks based on motion detected by optical flow.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            merged_mask: Merged object mask with object IDs
            object_id_map: Mapping from object names to IDs
            
        Returns:
            Tuple containing filtered mask and list of moving object names
        """
        # Create a copy of the merged mask for filtering
        filtered_mask = merged_mask.copy()
        
        # Create a reverse mapping from ID to object name
        id_to_name = {id_val: name for name, id_val in object_id_map.items()}
        
        # List to store moving object names
        moving_objects = []
        
        # Extract unique object IDs from the mask (excluding 0/background)
        unique_ids = np.unique(merged_mask)
        unique_ids = unique_ids[unique_ids > 0]
        
        # Process each object ID
        for obj_id in unique_ids:
            # Get the object name
            obj_name = id_to_name.get(obj_id)
            if obj_name is None:
                continue
            
            # Create a binary mask for this object
            obj_mask = (merged_mask == obj_id)
            
            # Check if the object is moving using the flow loader
            flow_info = self.flow_loader.process_flow_in_mask(
                camera_view=camera_view,
                frame_idx=frame_idx,
                mask=obj_mask,
                temporal_window=self.temporal_window,
                motion_threshold=self.motion_threshold
            )
      
            print(flow_info['avg_u'])
            print(flow_info['avg_v'])
            print(flow_info['avg_dir'])
            print(flow_info['avg_dir_scaled'])
            print(flow_info['avg_len'])
            print(flow_info['is_moving'])
            print(flow_info['raw_u_vectors'])
            print(flow_info['raw_v_vectors'])
            # If the object is not moving, remove it from the filtered mask
            if not flow_info['is_moving']:
                filtered_mask[obj_mask] = 0
            else:
                moving_objects.append(obj_name)
        
        return filtered_mask, moving_objects
    
    def get_motion_vectors(self, 
                          camera_view: str, 
                          frame_idx: int, 
                          object_name: str) -> Dict[str, Any]:
        """
        Get motion vectors for a specific object.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            object_name: Name of the object
            
        Returns:
            Dictionary containing motion vectors and related information
        """
        # Load filtered features
        features = self.load_features(camera_view, frame_idx)
        
        # Check if features were loaded successfully
        if not features['motion_filtered']['success']:
            return {
                'success': False,
                'object_name': object_name,
                'is_moving': False,
                'motion_vector': (0.0, 0.0)
            }
        
        # Get object ID map and filtered mask
        object_id_map = features['motion_filtered']['object_id_map']
        filtered_mask = features['motion_filtered']['mask']
        
        # Check if the object exists in the ID map
        if object_name not in object_id_map:
            return {
                'success': False,
                'object_name': object_name,
                'is_moving': False,
                'motion_vector': (0.0, 0.0)
            }
        
        # Get the object ID
        obj_id = object_id_map[object_name]
        
        # Create a binary mask for this object
        obj_mask = (filtered_mask == obj_id)
        
        # Check if any pixels belong to this object after filtering
        if not np.any(obj_mask):
            return {
                'success': True,
                'object_name': object_name,
                'is_moving': False,
                'motion_vector': (0.0, 0.0)
            }
        
        # Get flow information for this object
        flow_info = self.flow_loader.process_flow_in_mask(
            camera_view=camera_view,
            frame_idx=frame_idx,
            mask=obj_mask,
            temporal_window=self.temporal_window
        )
        
        return {
            'success': True,
            'object_name': object_name,
            'is_moving': flow_info['is_moving'],
            'motion_vector': flow_info['avg_dir'],
            'motion_vector_scaled': flow_info['avg_dir_scaled'],
            'motion_magnitude': flow_info['avg_len'],
            'raw_flow_info': flow_info
        }
    
    def get_all_moving_objects(self, 
                              camera_view: str, 
                              frame_idx: int) -> List[str]:
        """
        Get a list of all moving objects in a frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            List of moving object names
        """
        # Load filtered features
        features = self.load_features(camera_view, frame_idx)
        
        # Return moving objects if features were loaded successfully
        if features['motion_filtered']['success']:
            return features['motion_filtered']['moving_objects']
        
        # Return empty list if features could not be loaded
        return []
    
    def visualize_motion_filtered_masks(self, 
                                       camera_view: str, 
                                       frame_idx: int,
                                       with_flow_vectors: bool = True) -> np.ndarray:
        """
        Create a visualization of motion-filtered object masks with optional flow vectors.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            with_flow_vectors: Whether to include flow vectors in visualization
            
        Returns:
            Visualization as a NumPy array
        """
        # Load filtered features
        features = self.load_features(camera_view, frame_idx)
        
        # Load original frame
        frame = self._load_original_frame(camera_view, frame_idx)
        # print(features.keys())
        # print(features['motion_filtered'].keys())
        # print(features['motion_filtered']['success'])
        # print(features['motion_filtered']['mask'])
        # raise ValueError
        # Check if features were loaded successfully
        if not features['motion_filtered']['success'] or features['motion_filtered']['mask'] is None:
            raise ValueError
        
        # Get filtered mask and object ID map
        filtered_mask = features['motion_filtered']['mask']
        object_id_map = features['motion_filtered']['object_id_map']
        
        # Create visualization using the visualize_object_masks function
        visualization = visualize_object_masks(
            id_map=filtered_mask,
            rgb_frame=frame,
            object_id_map=object_id_map,
            object_colors=self.combined_loader.object_colors
        )
        
        # Add flow vectors if requested
        if with_flow_vectors:
            visualization = self._add_flow_vectors_to_visualization(
                visualization, camera_view, frame_idx, filtered_mask, object_id_map
            )
        
        return visualization
    
    def _add_flow_vectors_to_visualization(self,
                                          visualization: np.ndarray,
                                          camera_view: str,
                                          frame_idx: int,
                                          filtered_mask: np.ndarray,
                                          object_id_map: Dict[str, int]) -> np.ndarray:
        """
        Add flow vectors to object mask visualization.
        
        Args:
            visualization: Mask visualization image
            camera_view: Camera view name
            frame_idx: Frame index
            filtered_mask: Filtered object mask
            object_id_map: Mapping from object names to IDs
            
        Returns:
            Visualization with flow vectors
        """
        # Create a copy of the visualization
        result = visualization.copy()
        
        # Create a reverse mapping from ID to object name
        id_to_name = {id_val: name for name, id_val in object_id_map.items()}
        
        # Extract unique object IDs from the mask (excluding 0/background)
        unique_ids = np.unique(filtered_mask)
        unique_ids = unique_ids[unique_ids > 0]
        
        # Process each object ID
        for obj_id in unique_ids:
            # Get the object name
            obj_name = id_to_name.get(obj_id)
            if obj_name is None:
                continue
            
            # Create a binary mask for this object
            obj_mask = (filtered_mask == obj_id)
            
            # Get flow information for this object
            flow_info = self.flow_loader.process_flow_in_mask(
                camera_view=camera_view,
                frame_idx=frame_idx,
                mask=obj_mask,
                temporal_window=self.temporal_window
            )
            
            # Skip if the object is not moving
            if not flow_info['is_moving']:
                continue
            
            # Get centroid of the object mask
            y_indices, x_indices = np.where(obj_mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            centroid_y = int(np.min(y_indices))
            centroid_x = int(np.min(x_indices))
            
            # Draw an arrow from the centroid in the direction of motion
            motion_vector = flow_info['avg_dir_scaled']
            arrow_end_x = centroid_x + motion_vector[0]
            arrow_end_y = centroid_y + motion_vector[1]
            
            # Draw the arrow
            cv2.arrowedLine(
                result,
                (centroid_x, centroid_y),
                (arrow_end_x, arrow_end_y),
                (0, 255, 0),  # Green color
                3,  # Line thickness
                tipLength=6  # Length of arrow tip
            )
        
        return result
    
    def _load_original_frame(self, camera_view: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load the original frame image (delegate to combined loader).
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Frame image as numpy array or None if not available
        """
        return self.combined_loader._load_original_frame(camera_view, frame_idx)
    
    def _cache_features(self, camera_view: str, frame_idx: int, features: Dict[str, Any]) -> None:
        """
        Cache features for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            features: Features dictionary to cache
        """
        # Create cache key
        cache_key = f"{camera_view}_{frame_idx}"
        
        # Store in cache
        self.features_cache[cache_key] = features
    
    def _get_cached_features(self, camera_view: str, frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get cached features for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Cached features dictionary or None if not in cache
        """
        # Create cache key
        cache_key = f"{camera_view}_{frame_idx}"
        
        # Return cached features if available
        return self.features_cache.get(cache_key)


if __name__ == "__main__":
    import random
    
    # Initialize the Motion-Filtered Loader
    session_name = "imi_session1_6"
    data_root_dir = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data"
    
    # Create a configuration with specific thresholds
    config = {
        'score_threshold': 0.35,    # CNOS confidence threshold
        'lower_threshold': 0.28,    # Lower bound for CNOS confidence
        'motion_threshold': 0.05,   # Threshold for determining motion
        'temporal_window': 2,       # Window for optical flow aggregation
        'frames_dir': f"{data_root_dir}/orginal_frames"
    }
    
    # Initialize the loader
    motion_loader = MotionFilteredLoader(
        session_name=session_name, 
        data_root_dir=data_root_dir, 
        config=config
    )
    #print("Initialized MotionFilteredLoader")
    
    # Get valid frame indices from the CNOS loader
    valid_indices = motion_loader.combined_loader.cnos_loader.hamer_loader.get_valid_frame_idx()
    #print(f"Available camera views: {motion_loader.camera_views}")

    # Choose a random valid frame
    index = random.choice(valid_indices['cam_side_r']['left'])

    #print(f"Processing frame {index} from cam_side_r")
    features = motion_loader.load_features(camera_view='cam_side_r', frame_idx=index)

    # Display results
    moving_objects = motion_loader.get_all_moving_objects('cam_side_r', index)
    #print(f"Moving objects: {moving_objects}")
    
    # Create visualization
    visualization = motion_loader.visualize_motion_filtered_masks(
        camera_view='cam_side_r', 
        frame_idx=index,
        with_flow_vectors=True
    )
    
    # Save visualization
    cv2.imwrite('Motion_Filtered_Visualization.png', visualization)
    #print("Visualization saved to Motion_Filtered_Visualization.png")
    
    # Get motion vectors for each moving object
    for obj_name in moving_objects:
        motion_info = motion_loader.get_motion_vectors('cam_side_r', index, obj_name)
        #print(f"Object: {obj_name}, Motion vector: {motion_info['motion_vector']}, "
        #      f"Magnitude: {motion_info['motion_magnitude']}")

    #Combined
    # Initialize the loader
    combined_loader = CombinedLoader(session_name=session_name, data_root_dir=data_root_dir, config=config)
    flow_loader = FlowLoader(session_name=session_name, data_root_dir="/nas/project_data/B1_Behavior/rush/kaan/old_method/processed_data", config=config)
    #print("Initialized CombinedLoader")
    features = combined_loader.load_features(camera_view='cam_side_r', frame_idx=index)
    final_mask = features['merged']['mask']
    object_id_map = features['merged']['object_id_map']
    frame = combined_loader._load_original_frame(camera_view='cam_side_r', frame_idx=index)
    visualization = visualize_object_masks(final_mask, frame, object_id_map, combined_loader.object_colors)
    
    #print(features)
    cv2.imwrite('original.png', visualization)


    ## on the flow frame
    flow_frame = flow_loader._load_flow_frame(camera_view='cam_side_r', frame_idx=index)

    visualization = visualize_object_masks(final_mask, flow_frame, object_id_map, combined_loader.object_colors)
     #print(features)
    cv2.imwrite('flow.png', visualization)

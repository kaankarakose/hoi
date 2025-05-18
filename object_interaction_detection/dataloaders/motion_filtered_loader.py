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
    def _filter_by_motion(self, 
                     camera_view: str, 
                     frame_idx: int, 
                     merged_mask: np.ndarray, 
                     object_id_map: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
        """
        Filter object masks based on activeness detected by optical flow.
        
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
            
            # Check the activeness of the object using the flow loader
            activeness = self.flow_loader.process_flow_in_mask_active_area(
                camera_view=camera_view,
                frame_idx=frame_idx,
                mask=obj_mask
            )
            
            # If the object is not active enough, remove it from the filtered mask
            # if activeness < self.motion_threshold:
            #     filtered_mask[obj_mask] = 0
            # else:
            moving_objects.append(obj_name)
        
        return filtered_mask, moving_objects
    
    def get_activeness(self, 
                          camera_view: str, 
                          frame_idx: int, 
                          object_name: str) -> Dict[str, Any]:
        """
        Get activeness score for a specific object.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            object_name: Name of the object
            
        Returns:
            Dictionary containing activeness score and related information
        """
        # Load filtered features
        features = self.load_features(camera_view, frame_idx)
        
        # Check if features were loaded successfully
        if not features['motion_filtered']['success']:
            return {
                'success': False,
                'object_name': object_name,
                'activeness': 0
            }
        
        # Get object ID map and filtered mask
        object_id_map = features['motion_filtered']['object_id_map']
        filtered_mask = features['motion_filtered']['mask']
        
        # Check if the object exists in the ID map
        if object_name not in object_id_map:
            return {
                'success': False,
                'object_name': object_name,
                'activeness': 0,
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
                'activeness': 0,
            }

        # Get flow information for this object
        activeness = self.flow_loader.process_flow_in_mask_active_area(
            camera_view=camera_view,
            frame_idx=frame_idx,
            mask=obj_mask,
        )
        
        return {
            'success': True,
            'object_name': object_name,
            'activeness': activeness
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
    
    
    
    

    def _calculate_object_activeness(self, 
                                camera_view: str, 
                                frame_idx: int, 
                                mask: np.ndarray, 
                                object_id_map: Dict[str, int]) -> Dict[int, float]:
        """
        Calculate activeness scores for all objects in the mask.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            mask: Object mask with object IDs
            object_id_map: Mapping from object names to IDs
            
        Returns:
            Dictionary mapping object IDs to activeness scores
        """
        # Create a reverse mapping from ID to object name
        id_to_name = {id_val: name for name, id_val in object_id_map.items()}
        
        # Dictionary to store activeness for each object ID
        activeness_map = {}
        
        # Extract unique object IDs from the mask (excluding 0/background)
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]
        
        # Process each object ID
        for obj_id in unique_ids:
            # Get the object name
            obj_name = id_to_name.get(obj_id)
            if obj_name is None:
                activeness_map[obj_id] = 0.0
                continue
            
            # Create a binary mask for this object
            obj_mask = (mask == obj_id)
            
            # Get activeness for this object
            activeness = self.flow_loader.process_flow_in_mask_active_area(
                camera_view=camera_view,
                frame_idx=frame_idx,
                mask=obj_mask
            )
            
            activeness_map[obj_id] = activeness
        
        return activeness_map
        
    def _visualize_with_activeness(self,
                                id_map: np.ndarray,
                                rgb_frame: np.ndarray,
                                object_id_map: Dict[str, int],
                                object_colors: Dict[str, Tuple[int, int, int]],
                                activeness_map: Dict[int, float]) -> np.ndarray:
        """
        Visualize object masks with brightness based on activeness scores.
        
        Args:
            id_map: Object mask with object IDs
            rgb_frame: Original RGB frame
            object_id_map: Mapping from object names to IDs
            object_colors: Mapping from object names to colors
            activeness_map: Mapping from object IDs to activeness scores
            
        Returns:
            Visualization as a NumPy array
        """
        # Create a copy of the RGB frame
        visualization = rgb_frame.copy()
        
        # Create a reverse mapping from ID to object name
        id_to_name = {id_val: name for name, id_val in object_id_map.items()}
        
        # Extract unique object IDs from the mask (excluding 0/background)
        unique_ids = np.unique(id_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        # Create a separate overlay for each object
        overlay = np.zeros_like(rgb_frame, dtype=np.float32)
        
        # Define brightness levels based on activeness
        # This creates a step function with 4 levels
        def get_brightness_factor(activeness):
            if activeness < 0.125:
                return 0.25  # Very low brightness (25%)
            elif activeness < 0.25:
                return 0.5   # Low brightness (50%)
            elif activeness < 0.5:
                return 0.75  # Medium brightness (75%)
            else:
                return 1.0   # Full brightness (100%)
        
        # Process each object ID
        for obj_id in unique_ids:
            # Get the object name
            obj_name = id_to_name.get(obj_id)
            if obj_name is None:
                continue
            
            # Get the object's activeness
            activeness = activeness_map.get(obj_id, 0.0)
            
            # Get the object's color
            base_color = object_colors.get(obj_name, (255, 0, 0))  # Default to red
            
            # Calculate brightness-adjusted color
            brightness = get_brightness_factor(activeness)
            adjusted_color = tuple(int(c * brightness) for c in base_color)
            
            # Create object mask
            obj_mask = (id_map == obj_id)
            
            # Apply color to the object mask with transparency
            for c in range(3):  # RGB channels
                overlay[obj_mask, c] = adjusted_color[c]
        
        # Combine the original frame with the overlay
        alpha = 0.5  # Transparency factor
        visualization = cv2.addWeighted(visualization, 1-alpha, overlay.astype(np.uint8), alpha, 0)
        
        # Add activeness labels to each object
        for obj_id in unique_ids:
            obj_name = id_to_name.get(obj_id, "")
            activeness = activeness_map.get(obj_id, 0.0)
            
            # Create object mask and find its center point
            obj_mask = (id_map == obj_id)
            if not np.any(obj_mask):
                continue
                
            y_indices, x_indices = np.where(obj_mask)
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
            
            # Format label with activeness percentage
            label = f"{obj_name}: {activeness:.2f}"
            
            # Add activeness level indicator
            if activeness < 0.125:
                level = "Very Low"
            elif activeness < 0.25:
                level = "Low"
            elif activeness < 0.5:
                level = "Medium"
            else:
                level = "High"
                
            full_label = f"{obj_name}: {level} ({activeness:.2f})"
            
            # Add label to visualization
            cv2.putText(
                visualization,
                full_label,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )
        
        return visualization    




def visualize_object_masks(id_map, rgb_frame, object_id_map, object_colors, activeness_map=None, alpha=0.5, min_activeness=0.40):
    """
    Create a visualization of object masks overlaid on an RGB frame, with brightness controlled by activeness.
    
    Args:
        id_map: NumPy array where each pixel value is an object ID
        rgb_frame: Original RGB image as a NumPy array
        object_id_map: Dictionary mapping object names to IDs
        object_colors: Dictionary mapping object names to RGB colors
        activeness_map: Dictionary mapping object names to activeness scores (0 to 1)
        alpha: Transparency factor for the overlay (0-1)
        min_activeness: Minimum activeness threshold for displaying an object (default 0.25)
        
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
    
    # Track which pixels should be overlaid
    overlay_mask = np.zeros(id_map.shape, dtype=bool)
    
    # For each object ID in the id_map
    for obj_id in unique_ids:
        # Get the object name
        if obj_id in id_to_name:
            obj_name = id_to_name[obj_id]
            
            # Skip objects below the minimum activeness threshold
            if activeness_map is not None:
                activeness = activeness_map.get(obj_name, 0)
                if activeness < min_activeness:
                    continue
            
            # Get the color for this object
            if obj_name in object_colors:
                base_color = np.array(object_colors[obj_name], dtype=np.float32)
                
                # Adjust color brightness based on activeness (if provided)
                if activeness_map is not None:
                    activeness = activeness_map.get(obj_name, 0)
                    # Scale brightness: 
                    # - activeness of 0.25 maps to 25% brightness
                    # - activeness of 0.5 maps to 50% brightness
                    # - activeness of 1.0 maps to 100% brightness
                    brightness_factor = max(activeness, min_activeness) * 2  # Scale up for better visibility
                    brightness_factor = min(brightness_factor, 1.0)  # Cap at 1.0
                    color = base_color * brightness_factor
                else:
                    color = base_color
                
                # Create a binary mask for this object
                obj_mask = (id_map == obj_id)
                
                # Add to overlay mask
                overlay_mask = overlay_mask | obj_mask
                
                # Apply the color to this object in the colored mask
                for c in range(3):  # RGB channels
                    colored_mask[:,:,c][obj_mask] = color[c]
    
    # Blend the colored mask with the original frame
    for c in range(3):  # RGB channels
        visualization[:,:,c] = np.where(
            overlay_mask,
            (1 - alpha) * visualization[:,:,c] + alpha * colored_mask[:,:,c],
            visualization[:,:,c]
        )
    
    # Add activeness labels to each object if activeness_map is provided
    if activeness_map is not None:
        visualization = visualization.astype(np.uint8)
        for obj_id in unique_ids:
            if obj_id in id_to_name:
                obj_name = id_to_name[obj_id]
                if obj_name in activeness_map:
                    activeness = activeness_map[obj_name]
                    
                    # Skip objects below threshold
                    if activeness < min_activeness:
                        continue
                    
                    # Create a binary mask for this object
                    obj_mask = (id_map == obj_id)
                    if not np.any(obj_mask):
                        continue
                    
                    # Find center of mass for label placement
                    y_indices, x_indices = np.where(obj_mask)
                    center_y = int(np.mean(y_indices))
                    center_x = int(np.mean(x_indices))
                    
                    # Add activeness text
                    text = f"{obj_name}: {activeness:.2f}"
                    cv2.putText(
                        visualization,
                        text,
                        (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
    
    # Convert back to uint8
    visualization = np.clip(visualization, 0, 255).astype(np.uint8)
    
    return visualization
import cv2
if __name__ == "__main__":
    import random
    
    # Initialize the Motion-Filtered Loader
    session_name = "imi_session1_6"
    data_root_dir = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data"
    camera_view = 'cam_top'
    # Create a configuration with specific thresholds
    config = {
        'score_threshold': 0.40,    # CNOS confidence threshold
        'motion_threshold': 0.05,   # Threshold for determining motion - old version
        'temporal_window': 2,       # Window for optical flow aggregation
        'frames_dir': f"{data_root_dir}/orginal_frames"
    }
    
    # Initialize the loader
    motion_loader = MotionFilteredLoader(
        session_name=session_name, 
        data_root_dir=data_root_dir, 
        config=config
    )
    print("Initialized MotionFilteredLoader")
    
    # Get valid frame indices from the CNOS loader
    valid_indices = motion_loader.combined_loader.cnos_loader.hamer_loader.get_valid_frame_idx()
    print(f"Available camera views: {motion_loader.camera_views}")

    # Choose a random valid frame
    index = random.choice(valid_indices[camera_view]['left'])
    print(f"Processing frame {index} from {camera_view}")
    
    # Load features
    features = motion_loader.load_features(camera_view = camera_view, frame_idx=index)

    # Get list of moving objects
    moving_objects = motion_loader.get_all_moving_objects(camera_view, index)
    print(f"Moving objects: {moving_objects}")
    
    # Combined Loader and Flow Loader
    combined_loader = CombinedLoader(session_name=session_name, data_root_dir=data_root_dir, config=config)
    flow_loader = FlowLoader(session_name=session_name, data_root_dir="/nas/project_data/B1_Behavior/rush/kaan/old_method/processed_data", config=config)
    
    # Get original masks
    combined_features = combined_loader.load_features(camera_view=camera_view, frame_idx=index)
    original_mask = combined_features['merged']['mask']
    object_id_map = combined_features['merged']['object_id_map']
    
    # Get RGB frame and flow frame
    frame = combined_loader._load_original_frame(camera_view=camera_view, frame_idx=index)
    flow_frame = flow_loader._load_flow_frame(camera_view=camera_view, frame_idx=index)
    
    # Get all objects
    all_objects = list(object_id_map.keys())
    
    # Calculate activeness for each object
    activeness_map = {}
    for obj_name in all_objects:
        obj_id = object_id_map[obj_name]
        obj_mask = (original_mask == obj_id)
        activeness = flow_loader.process_flow_in_mask_active_area(
            camera_view=camera_view,
            frame_idx=index,
            mask=obj_mask
        )
        activeness_map[obj_name] = activeness
        print(f"Object: {obj_name}, Activeness: {activeness:.4f}")
    
    # Create visualizations
    
    # 1. Original visualization (without activeness)
    original_vis = visualize_object_masks(
        original_mask, frame, object_id_map, combined_loader.object_colors
    )
    cv2.imwrite('original_masks.png', original_vis)
    print("Saved original visualization")
    
    # 2. Enhanced visualization with activeness (all objects)
    activeness_vis_all = visualize_object_masks(
        original_mask, frame, object_id_map, combined_loader.object_colors, 
        activeness_map=activeness_map, min_activeness=0  # Show all objects
    )
    cv2.imwrite('activeness_all_objects.png', activeness_vis_all)
    print("Saved activeness visualization (all objects)")
    
    # 3. Enhanced visualization with activeness (only objects with activeness > 0.25)
    activeness_vis_filtered = visualize_object_masks(
        original_mask, frame, object_id_map, combined_loader.object_colors, 
        activeness_map=activeness_map, min_activeness=0.40  # Default threshold
    )
    cv2.imwrite('activeness_filtered_objects.png', activeness_vis_filtered)
    print("Saved activeness visualization (filtered objects)")
    
    # 4. Flow frame visualization with activeness
    flow_vis = visualize_object_masks(
        original_mask, flow_frame, object_id_map, combined_loader.object_colors,
        activeness_map=activeness_map, min_activeness=0.25
    )
    cv2.imwrite('flow_activeness.png', flow_vis)
    print("Saved flow frame visualization with activeness")
    
    # Create a side-by-side comparison
    h, w, _ = frame.shape
    composite = np.zeros((h, w*3, 3), dtype=np.uint8)
    
    # Add original visualization to first column
    composite[:, 0:w, :] = original_vis
    
    # Add activeness-based visualization (all objects) to second column
    composite[:, w:2*w, :] = activeness_vis_all
    
    # Add activeness-based visualization (filtered) to third column
    composite[:, 2*w:3*w, :] = activeness_vis_filtered
    
    # Add column titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "Original", (w//4, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(composite, "Activeness (All)", (w + w//4, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(composite, "Activeness (>0.25)", (2*w + w//4, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add activeness table
    y_offset = h - 20*len(all_objects)
    cv2.putText(composite, "Object Activeness:", (10, y_offset - 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Sort objects by activeness
    sorted_objects = sorted(all_objects, key=lambda x: activeness_map[x], reverse=True)
    
    for i, obj_name in enumerate(sorted_objects):
        activeness = activeness_map[obj_name]
        status = "VISIBLE" if activeness >= 0.25 else "HIDDEN"
        text = f"{obj_name}: {activeness:.4f} - {status}"
        y_pos = y_offset + i * 20
        color = (255, 255, 255) if activeness >= 0.25 else (128, 128, 128)
        cv2.putText(composite, text, (10, y_pos), font, 0.5, color, 1, cv2.LINE_AA)
    
    # Save the composite visualization
    cv2.imwrite('comparison_visualization.png', composite)
    print("Saved comparison visualization")
    
    print("Process completed successfully!")
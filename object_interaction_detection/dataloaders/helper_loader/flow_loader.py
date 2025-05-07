"""
Optical Flow loader class for loading and processing optical flow data.
"""

import os
import glob
import re
import sys
import math
import logging
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Add the parent directory to the path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
# Fix import path for BaseDataLoader
if __name__ == "__main__":
    # When run directly as a script, use absolute imports
    from object_interaction_detection.dataloaders.helper_loader.base_loader import BaseDataLoader
else:
    # When imported as a module, can use relative imports
    from .base_loader import BaseDataLoader

logging.basicConfig(level=logging.INFO)

class FlowLoader(BaseDataLoader):
    """
    Data loader for optical flow frames.
    
    This loader handles loading and processing optical flow data for analyzing
    motion in video frames.
    """
    
    def __init__(self, 
             session_name: str, 
             data_root_dir: str,
             config: Optional[Dict[str, Any]] = None,
             feature_types: Optional[List[str]] = None):
        """
        Initialize the Flow loader.
        
        Args:
            session_name: Name of the session to process
            data_root_dir: Root directory for all data
            config: Configuration parameters (optional)
            feature_types: List of feature types to extract ('direction', 'brightness', or both)
        """
        # Set Flow-specific defaults for paths
        config = config or {}
        
        # Add default optical flow paths
        config.setdefault('flow_dir', os.path.join(data_root_dir, 'memflow'))
        
        # Temporal window for flow aggregation
        config.setdefault('temporal_window', 1)  # Default: use only current frame
        
        # Motion threshold
        config.setdefault('motion_threshold', 0.05)  # Default motion threshold
        
        # Scale factor for visualization
        config.setdefault('scale_factor', 5.0)  # Default scale factor
        
        # Flag to include brightness information
        config.setdefault('include_brightness', False)  # Default: don't include brightness
        
        # Feature types to extract
        self.feature_types = feature_types or ['direction']
        if not isinstance(self.feature_types, list):
            self.feature_types = [self.feature_types]
        
        # Validate feature types
        valid_types = ['direction', 'brightness']
        for ft in self.feature_types:
            if ft not in valid_types:
                logging.warning(f"Unknown feature type: {ft}, ignoring")
                self.feature_types.remove(ft)
        
        if not self.feature_types:
            logging.warning("No valid feature types provided, defaulting to 'direction'")
            self.feature_types = ['direction']
        
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # Flow-specific attributes
        self.camera_views = self._discover_camera_views()
        self.aggregation_methods = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min
        }
        
    def _discover_camera_views(self) -> List[str]:
        """
        Discover available camera views for the session.
        
        Returns:
            List of camera view names
        """
        flow_dir = os.path.join(self.config['flow_dir'], self.session_name)
        if not os.path.exists(flow_dir):
            logging.warning(f"Flow directory not found: {flow_dir}")
            return ['cam_top', 'cam_side_l', 'cam_side_r']  # Default camera views
        
        # Get all subdirectories as camera views
        camera_views = [d for d in os.listdir(flow_dir) 
                       if os.path.isdir(os.path.join(flow_dir, d))]
        
        if not camera_views:
            logging.info("No camera views found, using defaults")
            return ['cam_top', 'cam_side_l', 'cam_side_r']  # Default camera views
        
        return camera_views
    
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load optical flow features for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing flow features
        """
        # Check for cached features
        cached_features = self._get_cached_features(camera_view, frame_idx)
        if cached_features is not None:
            return cached_features
        
        # Initialize features dictionary
        features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'flow_data': None,
            'hsv_data': None,
            'success': False
        }
        
        # Load optical flow frame
        flow_frame = self._load_flow_frame(camera_view, frame_idx)
        if flow_frame is None:
            self._cache_features(camera_view, frame_idx, features)
            return features
        
        # Convert to HSV for easier processing
        hsv_data = self._convert_flow_to_hsv(flow_frame)
        if hsv_data is None:
            self._cache_features(camera_view, frame_idx, features)
            return features
        
        # Update features
        features['flow_data'] = flow_frame
        features['hsv_data'] = hsv_data
        features['success'] = True
        
        # Cache features
        self._cache_features(camera_view, frame_idx, features)
        
        return features
    
    def _load_flow_frame(self, camera_view: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load optical flow frame for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Numpy array containing flow frame or None if not available
        """
        # Build path to flow directory
        flow_dir = os.path.join(
            self.config['flow_dir'],
            self.session_name,
            camera_view
        )
        
        if not os.path.exists(flow_dir):
            logging.warning(f"Flow directory not found: {flow_dir}")
            return None
        
        # Try different filename patterns
        pattern = f"flow_{frame_idx:04d}_to_{frame_idx+1:04d}.png"

        
        file_path = os.path.join(flow_dir, pattern)
        if os.path.exists(file_path):
        
            # Load the flow frame as an image
            flow_image = Image.open(file_path)
            flow_array = np.array(flow_image)
            return flow_array
           
    
        logging.warning(f"No flow frame found for {camera_view}, frame {frame_idx}")
        return None
    
    def _convert_flow_to_hsv(self, flow_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert optical flow frame to HSV for easier processing.
        
        Args:
            flow_frame: Numpy array containing flow frame
            
        Returns:
            Numpy array containing HSV data or None if conversion failed
        """
        try:
            # Convert to PIL Image and then to HSV
            flow_image = Image.fromarray(flow_frame)
            hsv_image = flow_image.convert("HSV")
            hsv_array = np.array(hsv_image)
            return hsv_array
        except Exception as e:
            logging.error(f"Error converting flow frame to HSV: {e}")
            return None
    def _extract_flow_brightness(self, hsv_data: np.ndarray, mask: np.ndarray) -> List[float]:
        """
        Extract flow intensity values from HSV data within a masked region.
        
        In optical flow visualization, the Saturation (S) channel represents the magnitude
        of flow, which is a better measure of flow intensity than the Value (V) channel.
        
        Args:
            hsv_data: HSV data array
            mask: Boolean mask array indicating region of interest
            
        Returns:
            List of flow intensity values from the masked region
        """
        # Ensure mask has the right shape
        if mask.shape[:2] != hsv_data.shape[:2]:
            logging.error(f"Mask shape {mask.shape} does not match HSV data shape {hsv_data.shape}")
            return []
        
        # Apply mask to get HSV values in the region of interest
        mask_3d = np.expand_dims(mask.astype(bool), axis=2)
        mask_3d = np.repeat(mask_3d, 3, axis=2)  # Expand to 3 channels
        
        # Get HSV values in the masked region
        masked_hsv = hsv_data[mask_3d].reshape(-1, 3)
        
        # Skip white and black pixels (no information)
        valid_pixels = ~(np.all(masked_hsv == [255, 255, 255], axis=1) | 
                        np.all(masked_hsv == [0, 0, 0], axis=1))
        
        # Filter only valid pixels
        valid_hsv = masked_hsv[valid_pixels]
        
        if len(valid_hsv) == 0:
            return []
        
        # Return flow intensity (Saturation channel) rather than brightness (Value channel)
        # For optical flow, saturation represents magnitude which is more meaningful
        # Convert to float to avoid potential overflow errors
        return valid_hsv[:, 1].astype(float).tolist()

    def get_flow_brightness(self, 
                        camera_view: str, 
                        frame_idx: int, 
                        mask: np.ndarray, 
                        method: str = 'mean') -> Dict[str, Any]:
        """
        Get flow brightness information for a masked region.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            mask: Boolean mask indicating region of interest
            method: Method for aggregating brightness values ('mean', 'median', 'max', 'min')
            
        Returns:
            Dictionary containing brightness information
        """
        # Load flow features
        features = self.load_features(camera_view, frame_idx)
        if not features['success']:
            return {
                'success': False
            }
        
        # Extract brightness values
        brightness_values = self._extract_flow_brightness(features['hsv_data'], mask)
        if not brightness_values:
            return {
                'success': False
            }
        
        # Calculate statistics
        avg_brightness = self.aggregation_methods.get(method, np.mean)(brightness_values)
        
        return {
            'success': True,
            'avg_brightness': avg_brightness,
            'raw_brightness_values': brightness_values
        }
    def process_flow_in_mask(self, 
                             camera_view: str, 
                             frame_idx: int, 
                             mask: np.ndarray,
                             temporal_window: Optional[int] = None,
                             aggregation_method: str = 'mean',
                             scale_factor: Optional[float] = None,
                             motion_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Process optical flow within a specific mask.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            mask: Boolean mask array indicating region of interest
            temporal_window: Number of frames to consider in each direction (default: from config)
            aggregation_method: Method to aggregate flow vectors ('mean', 'median', 'max', 'min')
            scale_factor: Scale factor for visualization (default: from config)
            motion_threshold: Threshold for determining if motion is present (default: from config)
            
        Returns:
            Dictionary containing processed flow information
        """
        # Use default values from config if not specified
        if temporal_window is None:
            temporal_window = self.config.get('temporal_window', 1)
        
        if scale_factor is None:
            scale_factor = self.config.get('scale_factor', 5.0)
            
        if motion_threshold is None:
            motion_threshold = self.config.get('motion_threshold', 0.05)
            
        # Get aggregation function
        if aggregation_method not in self.aggregation_methods:
            logging.warning(f"Unknown aggregation method: {aggregation_method}, using 'mean'")
            aggregation_method = 'mean'
        
        agg_func = self.aggregation_methods[aggregation_method]
        
        # Get frame indices to process based on temporal window
        frame_indices = list(range(
            max(0, frame_idx - temporal_window),
            frame_idx + temporal_window + 1
        ))
        
        # Process each frame and collect results
        all_u_vectors = []
        all_v_vectors = []
        
        for idx in frame_indices:
            # Skip if same as current frame (no motion)
            if idx == frame_idx and len(frame_indices) > 1:
                continue
                
            # Process the frame
            features = self.load_features(camera_view, idx)
            
            if not features['success'] or features['hsv_data'] is None:
                continue
                
            # Extract flow vectors from masked region
            u_vectors, v_vectors = self._extract_flow_vectors(
                features['hsv_data'], 
                mask
            )
            
            if u_vectors and v_vectors:
                all_u_vectors.extend(u_vectors)
                all_v_vectors.extend(v_vectors)
        
        # Handle case where no valid vectors were found
        if not all_u_vectors or not all_v_vectors:
            return {
                'success': False,
                'avg_u': 0.0,
                'avg_v': 0.0,
                'avg_dir': (0.0, 0.0),
                'avg_dir_scaled': (0, 0),
                'avg_len': 0.0,
                'is_moving': False,
                'raw_u_vectors': [],
                'raw_v_vectors': []
            }
        
        # Aggregate vectors using the specified method
        avg_u = float(agg_func(all_u_vectors))
        avg_v = float(agg_func(all_v_vectors))
        
        # Calculate average direction and length
        avg_dir = (avg_u, avg_v)
        avg_dir_scaled = (
            int(round(avg_u * scale_factor)),
            int(round(avg_v * scale_factor))
        )
        
        avg_len = math.sqrt(avg_u * avg_u + avg_v * avg_v)
        is_moving = avg_len > motion_threshold
        
        return {
            'success': True,
            'avg_u': avg_u,
            'avg_v': avg_v,
            'avg_dir': avg_dir,
            'avg_dir_scaled': avg_dir_scaled,
            'avg_len': avg_len,
            'is_moving': is_moving,
            'raw_u_vectors': all_u_vectors,
            'raw_v_vectors': all_v_vectors
        }
    
    def _extract_flow_vectors(self, 
                              hsv_data: np.ndarray, 
                              mask: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Extract flow vectors from HSV data within a masked region.
        
        Args:
            hsv_data: HSV data array
            mask: Boolean mask array indicating region of interest
            
        Returns:
            Tuple containing lists of u and v components of flow vectors
        """
        # Check for None values
        if hsv_data is None:
            logging.error("HSV data is None in _extract_flow_vectors")
            return [], []
            
        if mask is None:
            logging.error("Mask is None in _extract_flow_vectors")
            return [], []
            
        # Ensure mask has the right shape
        try:
            if mask.shape[:2] != hsv_data.shape[:2]:
                logging.error(f"Mask shape {mask.shape} does not match HSV data shape {hsv_data.shape}")
                return [], []
        except AttributeError as e:
            logging.error(f"Error comparing shapes: {e}")
            return [], []
        
        # Apply mask to get HSV values in the region of interest
        mask_3d = np.expand_dims(mask.astype(bool), axis=2)
        mask_3d = np.repeat(mask_3d, 3, axis=2)  # Expand to 3 channels
        
        # Get HSV values in the masked region
        masked_hsv = hsv_data[mask_3d].reshape(-1, 3)
        
        # Skip white and black pixels (no information)
        valid_pixels = ~(np.all(masked_hsv == [255, 255, 255], axis=1) | 
                        np.all(masked_hsv == [0, 0, 0], axis=1))
        
        # Filter only valid pixels
        valid_hsv = masked_hsv[valid_pixels]
        
        if len(valid_hsv) == 0:
            return [], []
        
        # Convert HSV to flow vectors using vectorized operations
        # Convert to float to avoid overflow errors
        valid_hsv_float = valid_hsv.astype(float)
        
        # H (0-255) maps to angle (0-360 degrees)
        angle_degrees = valid_hsv_float[:, 0] * 360 / 255
        
        # S (0-255) maps to length (0-1)
        length = valid_hsv_float[:, 1] / 255
        
        # Convert polar to cartesian coordinates
        angle_radians = np.radians(angle_degrees)
        direction_vector_us = length * np.cos(angle_radians)
        direction_vector_vs = length * np.sin(angle_radians)
        
        return direction_vector_us.tolist(), direction_vector_vs.tolist()
    
    def get_flow_direction(self, 
                           camera_view: str, 
                           frame_idx: int, 
                           mask: np.ndarray,
                           **kwargs) -> Tuple[float, float]:
        """
        Get flow direction vector for a masked region.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            mask: Boolean mask array indicating region of interest
            **kwargs: Additional arguments passed to process_flow_in_mask
            
        Returns:
            Tuple containing (u, v) direction vector
        """
        result = self.process_flow_in_mask(camera_view, frame_idx, mask, **kwargs)
        return result['avg_dir']
    
    def is_region_moving(self, 
                         camera_view: str, 
                         frame_idx: int, 
                         mask: np.ndarray,
                         **kwargs) -> bool:
        """
        Determine if a masked region is moving.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            mask: Boolean mask array indicating region of interest
            **kwargs: Additional arguments passed to process_flow_in_mask
            
        Returns:
            Boolean indicating if region is moving
        """
        result = self.process_flow_in_mask(camera_view, frame_idx, mask, **kwargs)
        return result['is_moving']
    
    def load_frame_batch(self, 
                        camera_view: str, 
                        frame_indices: List[int]) -> Dict[int, np.ndarray]:
        """
        Load multiple flow frames for a specific camera view.
        
        Args:
            camera_view: Camera view name
            frame_indices: List of frame indices to load
            
        Returns:
            Dictionary mapping frame indices to flow frames
        """
        result = {}
        for idx in frame_indices:
            features = self.load_features(camera_view, idx)
            if features['success']:
                result[idx] = features['flow_data']
        
        return result
    
    def process_flow_features(self,
                            camera_view: str,
                            frame_idx: int,
                            mask: np.ndarray,
                            feature_type: Optional[str] = None,
                            temporal_window: Optional[int] = None,
                            aggregation_method: str = 'mean',
                            scale_factor: Optional[float] = None,
                            motion_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Process optical flow features within a specific mask.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            mask: Boolean mask array indicating region of interest
            feature_type: Type of feature to extract ('direction' or 'brightness')
            temporal_window: Number of frames to consider in each direction (default: from config)
            aggregation_method: Method to aggregate values ('mean', 'median', 'max', 'min')
            scale_factor: Scale factor for visualization (default: from config)
            motion_threshold: Threshold for determining if motion is present (default: from config)
            
        Returns:
            Dictionary containing processed flow information
        """
        # Use default feature type if not specified
        if feature_type is None:
            feature_type = self.feature_types[0]
        
        # Validate feature type
        if feature_type not in ['direction', 'brightness']:
            logging.warning(f"Unknown feature type: {feature_type}, using 'direction'")
            feature_type = 'direction'
        
        # Process according to feature type
        if feature_type == 'direction':
            return self.process_flow_in_mask(
                camera_view, frame_idx, mask, 
                temporal_window, aggregation_method, 
                scale_factor, motion_threshold
            )
        else:  # brightness
            return self.get_flow_brightness(
                camera_view, frame_idx, mask, aggregation_method
            )


if __name__ == "__main__":
    # Test the loader with different feature types
    flow_loader = FlowLoader(
        session_name="imi_session1_6", 
        data_root_dir="/nas/project_data/B1_Behavior/rush/kaan/old_method/processed_data",
        feature_types=['direction', 'brightness']
    )
    
    # Test loading data for a specific frame
    data = flow_loader.load_features(camera_view="cam_top", frame_idx=100)
    print("Flow data loaded:", data['success'])
    
    if data['success']:
        # Create a sample mask
        mask = np.zeros_like(data['flow_data'][:, :, 0], dtype=bool)
        mask[600:800, 600:800] = True
        
        # Process flow direction in mask
        direction_result = flow_loader.process_flow_features(
            camera_view="cam_top",
            frame_idx=100,
            mask=mask,
            feature_type='direction',
            temporal_window=2,
            aggregation_method='mean'
        )
        
        print("Region moving:", direction_result['is_moving'])
        print("Average direction:", direction_result['avg_dir'])
        print("Average speed:", direction_result['avg_len'])
        
        # Process flow brightness in mask
        brightness_result = flow_loader.process_flow_features(
            camera_view="cam_top",
            frame_idx=100,
            mask=mask,
            feature_type='brightness',
            temporal_window=2,
            aggregation_method='min'
        )
        
        print("Brightness processing success:", brightness_result['success'])
        if brightness_result['success']:
            print("Average brightness:", brightness_result['avg_brightness'])
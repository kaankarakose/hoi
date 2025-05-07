"""
Frame loader class for loading camera frames.
"""

import os
import glob
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Import the base loader
from object_interaction_detection.dataloaders.base_loader import BaseDataLoader
import logging
logging.basicConfig(level=logging.INFO)


class FrameLoader(BaseDataLoader):
    """
    Data loader for camera frames.
    
    This loader handles loading frames from different camera views.
    """
    
    def __init__(self, 
                 session_name: str, 
                 data_root_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the frame loader.
        
        Args:
            session_name: Name of the session to process
            data_root_dir: Root directory for all data
            config: Configuration parameters (optional)
        """
        # Set frame-specific defaults
        config = config or {}
        
        # Add default frame paths
        config.setdefault('frames_dir', os.path.join(data_root_dir, 'orginal_frames'))
        
        # Call parent constructor
        super().__init__(session_name, data_root_dir, config)
        
        # Set standard camera views if not already discovered
        if not self.camera_views:
            self.camera_views = ['cam_top', 'cam_side_l', 'cam_side_r']
    
    def _discover_camera_views(self) -> List[str]:
        """
        Discover available camera views for the session.
        
        Returns:
            List of camera view names
        """
    
        frames_dir = os.path.join(self.config['frames_dir'], self.session_name)
        if not os.path.exists(frames_dir):
            logging.warning(f"Frames directory not found: {frames_dir}")
            return ['cam_top', 'cam_side_l', 'cam_side_r']  # Default camera views
        
        # Get all subdirectories as camera views
        camera_views = [d for d in os.listdir(frames_dir) 
                       if os.path.isdir(os.path.join(frames_dir, d))]
        
        if not camera_views:
            logging.warning(f"No camera views found in: {frames_dir}")
            return ['cam_top', 'cam_side_l', 'cam_side_r']  # Default camera views
        
        logging.info(f"Found {len(camera_views)} camera views: {camera_views}")
        return camera_views
    
    def get_frame_count(self, camera_view: str) -> int:
        """
        Get total number of frames for a specific camera view.
        
        Args:
            camera_view: Camera view name
            
        Returns:
            Number of frames
        """
        frames_path = os.path.join(self.config['frames_dir'], 
                                  self.session_name, camera_view)
        if not os.path.exists(frames_path):
            logging.warning(f"Frame path not found: {frames_path}")
            return 0
        
        # Count frames based on image files in the directory
        frame_files = [f for f in os.listdir(frames_path) 
                     if f.endswith('.jpg') or f.endswith('.png')]
        
        return len(frame_files)
    
    def get_frame_path(self, camera_view: str, frame_idx: int) -> Optional[str]:
        """
        Get the path to a specific frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Path to the frame file, or None if not found
        """
        frames_path = os.path.join(self.config['frames_dir'], 
                                  self.session_name, camera_view)
        if not os.path.exists(frames_path):
            logging.warning(f"Frame path not found: {frames_path}")
            return None
        
        # Try different filename patterns
        pattern = f"frame_{frame_idx:04d}.jpg"
    
        path = os.path.join(frames_path, pattern)

        if os.path.exists(path):
            return path
        
        logging.warning(f"No frame found for {camera_view}, frame {frame_idx}")
        return None
    
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load frame features for a specific camera view and frame.
        This method is required by the BaseDataLoader abstract class.
        
        For FrameLoader, we just return the frame path and metadata.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing frame path and metadata
        """
        # Check for cached features
        cached_features = self._get_cached_features(camera_view, frame_idx)
        if cached_features is not None:
            return cached_features
        
        # Get frame path
        frame_path = self.get_frame_path(camera_view, frame_idx)
        
        # Initialize features dictionary
        features = {
            'frame_idx': frame_idx,
            'camera_view': camera_view,
            'frame_path': frame_path,
            'success': frame_path is not None
        }
        
        # Cache features
        self._cache_features(camera_view, frame_idx, features)
        
        return features
    
    def load_frame(self, camera_view: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load the actual frame image for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Numpy array containing the frame image, or None if not found
        """
        # Get frame path
        frame_path = self.get_frame_path(camera_view, frame_idx)
        
        if not frame_path:
            return None
        
        try:
            # Read image using OpenCV
            frame = cv2.imread(frame_path)
            
            # Convert from BGR to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame
        except Exception as e:
            logging.error(f"Error loading frame {frame_path}: {e}")
            return None
    
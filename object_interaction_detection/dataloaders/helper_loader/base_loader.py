
"""
Base loader for all dataloaders
"""
import os
import glob
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    This class handles common functionality for loading and caching features
    from disk. Specific data formats are handled by subclasses.
    """
    
    def __init__(self, 
                 session_name: str, 
                 data_root_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base data loader.
        
        Args:
            session_name: Name of the session to process
            data_root_dir: Root directory for all data
            config: Configuration parameters (optional)
        """
        self.session_name = session_name
        self.data_root_dir = data_root_dir
        self.config = config or {}
        
        # Discover camera views
        self.camera_views = self._discover_camera_views()
        
  
    
    def _discover_camera_views(self) -> List[str]:
        """
        Discover available camera views for the session.
        
        Returns:
            List of camera view names
        """
        # Default implementation - subclasses may override
        frames_dir = os.path.join(self.data_root_dir, 'orginal_frames', self.session_name)
        if not os.path.exists(frames_dir):
            print(f"Warning: Frames directory not found: {frames_dir}")
            return []
        
        # Get all subdirectories as camera views
        camera_views = [d for d in os.listdir(frames_dir) 
                       if os.path.isdir(os.path.join(frames_dir, d))]
        
        assert 'cam_top' in camera_views
        assert 'cam_side_l' in camera_views
        assert 'cam_side_r' in camera_views
        
        
        return camera_views
    

    
    def get_frame_count(self, camera_view: str) -> int:
        """
        Get total number of frames for a specific camera view.
        
        Args:
            camera_view: Camera view name
            
        Returns:
            Number of frames
        """
        # Default implementation - subclasses may override
        frames_path = os.path.join(self.data_root_dir, 'original_frames', 
                                  self.session_name, camera_view)
        if not os.path.exists(frames_path):
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
        # Default implementation - subclasses may override
        frames_path = os.path.join(self.data_root_dir, 'original_frames', 
                                  self.session_name, camera_view)
        if not os.path.exists(frames_path):
            return None
        
        # Try different filename patterns
        patterns = [
            f"frame_{frame_idx:06d}.jpg",
            f"frame_{frame_idx:06d}.png",
            f"frame_{frame_idx:04d}.jpg",
            f"frame_{frame_idx:04d}.png",
            f"{frame_idx:06d}.jpg",
            f"{frame_idx:06d}.png",
            f"{frame_idx:04d}.jpg",
            f"{frame_idx:04d}.png",
        ]
        
        for pattern in patterns:
            path = os.path.join(frames_path, pattern)
            if os.path.exists(path):
                return path
        
        return None
    

    @abstractmethod
    def load_features(self, camera_view: str, frame_idx: int) -> Dict[str, Any]:
        """
        Load features for a specific camera view and frame.
        
        Args:
            camera_view: Camera view name
            frame_idx: Frame index
            
        Returns:
            Dictionary containing features
        """
        pass
    
    
    def load_features_range(self, camera_view: str, 
                           start_frame: int, end_frame: int) -> Dict[int, Dict[str, Any]]:
        """
        Load features for a range of frames.
        
        Args:
            camera_view: Camera view name
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (inclusive)
            
        Returns:
            Dictionary mapping frame indices to feature dictionaries
        """
        frame_indices = list(range(start_frame, end_frame + 1))
        return self.load_features_batch(camera_view, frame_indices)
    
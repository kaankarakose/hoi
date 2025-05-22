import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass


# Import the annotation parser class from the provided code
import sys
sys.path.append('/nas/project_data/B1_Behavior/rush/kaan/hoi')
from object_interaction_detection.evaluation.utils.annotation_parser import AnnotationData

# We'll import plots later to avoid circular imports
# from object_interaction_detection.evaluation.utils.plots import plot_detection_energy

logger = logging.getLogger(__name__)

@dataclass
class ObjectEnergy:
    """Energy of the object from optical flow information. Emprical!"""
    current_energy: float = 0.0
    decay_rate: float = 0.1  # How quickly energy decays per frame
    accumulation_rate: float = 1.0  # How quickly activeness adds to energy
    
    def update(self, activeness: float) -> float:
        """Update energy based on new activeness value and return the new energy."""
        # Add new energy from activeness
        self.current_energy += activeness * self.accumulation_rate
        
        # Apply decay
        if activeness < 0.1:  # Only decay when there's minimal activity
            self.current_energy *= (1.0 - self.decay_rate)
            
        # Ensure energy stays within reasonable bounds
        self.current_energy = max(0.0, min(1.0, self.current_energy))
        
        return self.current_energy

@dataclass
class Detection:
    """A detected object instance with frame-based activeness."""
    object_name: str
    frame_idx: int
    activeness: float
    energy: float
    
    def to_time(self, fps: float = 30.0) -> float:
        """Convert frame index to time in seconds."""
        return self.frame_idx / fps


class DetectionData:
    """Container for detection results from object tracker."""
    
    def __init__(self, session_name: str = None, camera_view: str = None, 
                 detection_root: str = None):
        """Initialize detection data container.
        
        Args:
            session_name: Name of the session (e.g., 'imi_session1_2')
            camera_view: Name of the camera view (e.g., 'cam_side_l')
            detection_root: Root directory for detection results
        """
        self.session_name = session_name
        self.camera_view = camera_view
        self.detection_root = detection_root or "/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/tracking"
        self.object_data = {}  # Dict mapping object name to its data
        self.frame_indices = []
        self.fps = 30.0
        
        # Load data if session and camera are provided
        if session_name and camera_view:
            self.load_data()
    
    def load_data(self):
        """Load detection data from saved JSON file."""
        file_path = os.path.join(
            self.detection_root, 
            self.session_name, 
            self.camera_view,
            f"{self.session_name}_{self.camera_view}_activeness.json"
        )
        
        if not os.path.exists(file_path):
            logger.warning(f"Detection file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Extract metadata
            self.camera_view = data.get("camera_view", self.camera_view)
            self.object_data = data.get("objects", {})
            # print(self.object_data.keys())
            # print(self.object_data['BOX'].keys()) #dict_keys(['activeness', 'frame_idx'])
            # Extract frame indices if available
            metadata = data.get("metadata", {})
            if "frame_range" in metadata.keys():
                start, end = metadata["frame_range"]
                self.frame_indices = list(range(start, end + 1))
            logger.info(f"Loaded detection data from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading detection data from {file_path}: {e}")
            raise ValueError(f"Error loading detection data from {file_path}: {e}")
    
    def get_active_detections(self, threshold: float = 0.5) -> List[Detection]:
        """Get all detections where activeness exceeds threshold.
        
        Args:
            threshold: Activeness threshold (0-1)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        for obj_name, obj_data in self.object_data.items():
            activeness = obj_data.get("activeness", [])
            frames = obj_data.get("frames", [])
            
            if len(activeness) != len(frames):
                logger.warning(f"Activeness and frames length mismatch for {obj_name}")
                continue
            
            for frame_idx, active_val in zip(frames, activeness):
                if active_val >= threshold:
                    detections.append(Detection(
                        object_name=obj_name,
                        frame_idx=frame_idx,
                        activeness=active_val
                    ))
        
        return detections
    
    def get_first_active_frame(self, object_name: str, threshold: float = 0.5) -> Optional[int]:
        """Get the first frame where the object activeness exceeds threshold.
        
        Args:
            object_name: Name of the object
            threshold: Activeness threshold (0-1)
            
        Returns:
            Frame index or None if not found
        """
        if object_name not in self.object_data:
            return None
        
        obj_data = self.object_data[object_name]
        activeness = obj_data.get("activeness", [])
        frames = obj_data.get("frames", [])
        
        if len(activeness) != len(frames):
            logger.warning(f"Activeness and frames length mismatch for {object_name}")
            return None
        
        for frame_idx, active_val in zip(frames, activeness):
            if active_val >= threshold:
                return frame_idx
        
        return None
    
    def get_all_detected_objects(self) -> List[str]:
        """Get a list of all detected object names."""
        return list(self.object_data.keys())

    def process_with_energy(self, energy_threshold: float = 0.3, activeness_threshold: float = 0.5) -> List[Detection]:
        """Process detections with energy accumulation.
        
        Args:
            energy_threshold: Minimum energy to consider object active
            activeness_threshold: Threshold for raw activeness values
            
        Returns:
            List of Detection objects with energy values
        """
        detections = []
        energy_trackers = {}  # Maps object_name to ObjectEnergy instance
        
        # Create energy trackers for each object
        for obj_name in self.object_data.keys():

            energy_trackers[obj_name] = ObjectEnergy()
        # Process all frames chronologically
        frame_range = range(min(self.frame_indices), max(self.frame_indices) + 1)
        for frame_idx in frame_range:
            for obj_name, tracker in energy_trackers.items():
                obj_data = self.object_data.get(obj_name, {})
                frames = obj_data.get("frame_idx", [])
                activeness = obj_data.get("activeness", [])
                # Get activeness for current frame if available
                try:
                    frame_pos = frames.index(frame_idx)
                    current_activeness = activeness[frame_pos]
                except ValueError:
                    current_activeness = 0.0
                
                # Update energy
                current_energy = tracker.update(current_activeness)
                
                # Create detection if energy or activeness is high enough
                if current_energy >= energy_threshold or current_activeness >= activeness_threshold:
                    detections.append(Detection(
                        object_name=obj_name,
                        frame_idx=frame_idx,
                        activeness=current_activeness,
                        energy=current_energy
                    ))
        
        return detections


if __name__ == "__main__":
    
    detection_data = DetectionData(session_name="imi_session1_2", camera_view="cam_side_l")
    
    detections = detection_data.process_with_energy(energy_threshold=0.3, activeness_threshold=0.5)

    annotation_data = AnnotationData(session_name="imi_session1_2")
    first_annotated_time = annotation_data.get_first_segment("HandUsed").start_time

    # Convert annotation time to frame number
    first_annotated_frame = first_annotated_time * detection_data.fps
    
    # Import only when running as main to avoid circular imports
    from object_interaction_detection.evaluation.utils.plots import plot_detection_energy
    
    plot_detection_energy(
    detection_data,
    object_names=['BOX'],
    activeness_threshold=0.5,
    energy_threshold=0.3,
    first_annotated_time=first_annotated_time,
    figsize=(12, 8),
    time_based=False,  # Changed to False to use frame numbers
    highlight_detections=True,
    save_path='./detection_energy.png')

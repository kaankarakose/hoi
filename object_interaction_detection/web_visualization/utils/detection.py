"""
Detection data handler for hand-object interaction visualization
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

@dataclass
class ObjectHit:
    """Represents a detection hit for a specific object"""
    object_name: str
    annotation_time: float
    detection_time: float
    time_diff: float
    hit_within_window: bool
    activeness: float
    energy: float
    isFirstPlayDetection: bool

@dataclass
class DetectionData:
    """Container for detection data and metrics"""
    session_name: str
    camera_view: str
    metrics: Dict[str, Any]
    object_metrics: Dict[str, Any]
    object_hits: Dict[str, List[ObjectHit]]
    first_detection_times: Dict[str, float]
    first_annotation_times: Dict[str, float]
    first_overall_annotation_time: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionData':
        """Create DetectionData from a dictionary"""
        # Process object hits to convert from dict to ObjectHit objects
        object_hits = {}
        for obj_name, hits in data.get('object_hits', {}).items():
            object_hits[obj_name] = [ObjectHit(**hit) for hit in hits]
        
        return cls(
            session_name=data.get('session_name', ''),
            camera_view=data.get('camera_view', ''),
            metrics=data.get('metrics', {}),
            object_metrics=data.get('object_metrics', {}),
            object_hits=object_hits,
            first_detection_times=data.get('first_detection_times', {}),
            first_annotation_times=data.get('first_annotation_times', {}),
            first_overall_annotation_time=data.get('first_overall_annotation_time', 0.0)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DetectionData':
        """Create DetectionData from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'DetectionData':
        """Create DetectionData from a JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DetectionData to a dictionary"""
        # Convert ObjectHit objects back to dictionaries
        object_hits_dict = {}
        for obj_name, hits in self.object_hits.items():
            object_hits_dict[obj_name] = [
                {
                    'object_name': hit.object_name,
                    'annotation_time': hit.annotation_time,
                    'detection_time': hit.detection_time,
                    'time_diff': hit.time_diff,
                    'hit_within_window': hit.hit_within_window,
                    'activeness': hit.activeness,
                    'energy': hit.energy,
                    'isFirstPlayDetection': hit.isFirstPlayDetection
                } for hit in hits
            ]
        
        return {
            'session_name': self.session_name,
            'camera_view': self.camera_view,
            'metrics': self.metrics,
            'object_metrics': self.object_metrics,
            'object_hits': object_hits_dict,
            'first_detection_times': self.first_detection_times,
            'first_annotation_times': self.first_annotation_times,
            'first_overall_annotation_time': self.first_overall_annotation_time
        }
    
    def to_json(self) -> str:
        """Convert DetectionData to a JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, file_path: str) -> None:
        """Save DetectionData to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class DetectionManager:
    """Manager for handling detection data across sessions and camera views"""
    
    def __init__(self, data_dir: str, session_name: str = '', camera_view: str = ''):
        """
        Initialize the detection manager
        
        Args:
            data_dir: Directory where detection data is stored
        """
        self.data_dir = data_dir
        self.session_name = session_name
        self.camera_view = camera_view
        self.detection_data = None # Dict[session_name, Dict[camera_view, DetectionData]]
        
    def load_data(self, session_name: str, camera_view: str) -> Optional[DetectionData]:
        """
        Load detection data for a specific session and camera view
        
        Args:
            session_name: Name of the session
            camera_view: Camera view name
            
        Returns:
            DetectionData object or None if not found
        """

        # Build file path
        file_path = os.path.join(self.data_dir,session_name, camera_view, f"{session_name}_{camera_view}_evaluation.json")
        print(f"Loading detection data from: {file_path}")
        # Check if file exists
        if not os.path.exists(file_path):
            return None
        
        # Load data
        try:
            data = DetectionData.from_file(file_path)

            return data
        except Exception as e:
            print(f"Error loading detection data: {e}")
            return None
    
    
    
    def get_detected_objects(self) -> List[str]:
        """
        Get a list of all detected objects
        
        Returns:
            List of object names
        """
        detected_objects = set()

        
        return list(detected_objects)
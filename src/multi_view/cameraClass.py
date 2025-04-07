import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


import json

@dataclass
class CameraPose:
    """Data class representing a camera pose with rotation matrix and translation vector."""
    R: np.ndarray
    T: np.ndarray
    
    def __post_init__(self):
        # Convert lists to numpy arrays if needed
        if isinstance(self.R, list):
            self.R = np.array(self.R)
        if isinstance(self.T, list):
            self.T = np.array(self.T)


@dataclass
class CameraPair:
    """Data class representing a transformation between two cameras."""
    R: np.ndarray
    T: np.ndarray
    
    def __post_init__(self):
        # Convert lists to numpy arrays if needed
        if isinstance(self.R, list):
            self.R = np.array(self.R)
        if isinstance(self.T, list):
            self.T = np.array(self.T)


@dataclass
class CameraSystem:
    """Data class representing a system of cameras with their poses and transformations."""
    camera_poses: Dict[str, CameraPose] = field(default_factory=dict)
    camera_pairs: Dict[str, CameraPair] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, json_data):
        """Create a CameraSystem instance from JSON data."""
        camera_system = cls()
        
        # Process camera poses
        for cam_name, pose_data in json_data.get("camera_poses", {}).items():
            camera_system.camera_poses[cam_name] = CameraPose(
                R=pose_data["R"],
                T=pose_data["T"]
            )
        
        # Process camera pairs
        for pair_name, pair_data in json_data.get("camera_pairs", {}).items():
            camera_system.camera_pairs[pair_name] = CameraPair(
                R=pair_data["R"],
                T=pair_data["T"]
            )
        
        return camera_system


if __name__ == "__main__":

    # Load JSON data
    with open("/nas/project_data/B1_Behavior/rush/kaan/hoi/camera_params/camera.json", "r") as f:
        data = json.load(f)
    
    # Create CameraSystem from JSON
    camera_system = CameraSystem.from_json(data)
    
    # Access data
    print(f"Number of camera poses: {len(camera_system.camera_poses)}")
    print(f"Number of camera pairs: {len(camera_system.camera_pairs)}")
    
    # Example: access first camera pose
    first_cam = next(iter(camera_system.camera_poses.keys()))
    print(f"First camera: {first_cam}")
    print(f"R matrix:\n{camera_system.camera_poses[first_cam].R}")
    print(f"T vector: {camera_system.camera_poses[first_cam].T}")

    print(camera_system.camera_poses['cam2_to_cam1'])
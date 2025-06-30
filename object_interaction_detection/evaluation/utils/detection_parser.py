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
#from object_interaction_detection.evaluation.utils.plots import *

logger = logging.getLogger(__name__)

@dataclass
class ObjectEnergy:
    """Energy of the object from optical flow information. Emprical!"""
    current_energy: float = 0.0
    decay_rate: float = 0.05  # How quickly energy decays per frame
    accumulation_rate: float = 1.5  # How quickly activeness adds to energy
    
    def update(self, activeness: float) -> float:
        """Update energy based on new activeness value and return the new energy."""
        # Add new energy from activeness
        self.current_energy += activeness * self.accumulation_rate
        
        # Apply decay
        if activeness < 0.09:  # Only decay when there's minimal activity
            self.current_energy *= (1.0 - self.decay_rate)
            
        # Ensure energy stays within reasonable bounds
        self.current_energy = max(0.0, min(1.0, self.current_energy))
        
        return self.current_energy
    def reset(self):
        """Reset the energy to zero."""
        self.current_energy = 0.0
        logger.info("Energy reset to zero.")
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
    
    def __init__(self, session_name: str = None, camera_view: str = "multi", 
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
        if self.camera_view != "multi":
            file_path = os.path.join(
                self.detection_root, 
                self.session_name, 
                self.camera_view ,
                f"{self.session_name}_{self.camera_view}_activeness.json"
            )
        else:
            
            file_path = os.path.join(
                self.detection_root, 
                self.session_name, 
                f"{self.session_name}_activeness.json"
            )

        print(file_path)
        
        if not os.path.exists(file_path):
            logger.warning(f"Detection file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Extract metadata
            self.camera_view = data.get("camera_view", self.camera_view)
            self.object_data = data.get("objects", {})
            print(self.object_data.keys())
            #print(self.object_data['BOX'].keys()) #dict_keys(['activeness', 'frame_idx'])
            # Extract frame indices if available
            metadata = data.get("metadata", {})
            if "frame_range" in metadata.keys():
                start, end = metadata["frame_range"]
                print(f"Frame range: {start} to {end}")
                self.frame_indices = list(range(start, end + 1))
            else:
                raise ValueError("Metadata does not contain 'frame_range' information.")
            logger.info(f"Loaded detection data from {file_path}")
            return True
        except Exception as e:
            raise ValueError(f"Error loading detection data from {file_path}: {e}")
    
    def get_active_detections(self, threshold: float = 0.01, isolation_window: int = 5) -> List[Detection]:
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
                raise ValueError(
                    f"Activeness and frames length mismatch for {obj_name}"
                )
            print(len(frames))
            
            active_frames = set()
            frame_to_activeness = {}
            
            for frame_idx, active_val in zip(frames, activeness):
                #if active_val >= threshold:
                active_frames.add(frame_idx)
                frame_to_activeness[frame_idx] = active_val

            # Filter out isolated detections #TODO : THIS MAY NOT BE THE BEST WAY
            for frame_idx in active_frames:
                # Check if this frame has neighbors within the isolation window
                has_neighbors = False
                for offset in range(-isolation_window, isolation_window + 1):
                    if offset != 0 and (frame_idx + offset) in active_frames:
                        has_neighbors = True
                        break

                if has_neighbors and active_val > threshold:
                    active_val = frame_to_activeness[frame_idx]
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


        safety_gap = 25
        counter = 0
        detections = []

        # Track last seen frame for each object
        last_seen_frames = {}

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


##PLOT


"""
Plots for visualizing detection results.

"""


import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING, Any

def plot_detection_energy(
    detection_data: Any,  # Type will be DetectionData
    object_names: Optional[List[str]] = None,
    activeness_threshold: float = 0.5,
    energy_threshold: float = 0.3,
    first_annotated_time: float = 0.0,
    first_annotated_objects: Optional[str] = None,
    figsize: Tuple[int, int] = (60, 40),
    time_based: bool = True,
    highlight_detections: bool = True,
    session_camera: Tuple[str, str] = ("", ""),  # (session, camera)
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot activeness scores and energy values for selected objects.
    
    Args:
        detection_data: The DetectionData object containing detection information
        object_names: List of object names to plot (None = plot all objects)
        activeness_threshold: Threshold line for activeness
        energy_threshold: Threshold line for energy
        figsize: Figure size (width, height) in inches
        time_based: If True, x-axis is time in seconds; otherwise, it's frame index
        highlight_detections: Whether to highlight frames where detections are active
        save_path: Path to save the figure (None = don't save)
        
    Returns:
        Matplotlib figure object
    """
    fps = detection_data.fps
    # Get objects to plot
    if object_names is None:
        object_names = detection_data.get_all_detected_objects()
    
    if not object_names:
        print("No objects to plot!")
        return None
    
    # Set up figure and axes
    #n_objects = len(object_names)
    n_objects = len(first_annotated_objects)
    print(n_objects)
    fig, axes = plt.subplots(n_objects, 2, figsize=figsize, sharex=True)
    
    # Handle case with only one object (axes would be 1D)
    if n_objects == 1:
        axes = np.array([axes])
    
    # Process all detections with energy
    all_detections = detection_data.process_with_energy(
        energy_threshold=energy_threshold,
        activeness_threshold=activeness_threshold
    )
    
    # Group detections by object name
    detections_by_object = {}
    for detection in all_detections:
        if detection.object_name not in detections_by_object:
            detections_by_object[detection.object_name] = []
        detections_by_object[detection.object_name].append(detection)
    
    # Set default x-axis label based on time_based parameter
    x_label = "Time (seconds)" if time_based else "Frame Index"
    
    # Plot each object
    for i, obj_name in enumerate(object_names):
        if i >= len(axes) or obj_name not in first_annotated_objects:
            break
        

        # Get the original activeness and frames data
        obj_data = detection_data.object_data.get(obj_name, {})
        activeness = obj_data.get("activeness", [])
        frames = obj_data.get("frame_idx", [])  # Fixed key name to match detection_parser.py
        
        if not frames or not activeness:
            print(f"No data for object: {obj_name}")
            continue
            
        # Convert to numpy arrays for easier manipulation
        frames = np.array(frames)
        activeness = np.array(activeness)
        
        # Create x-axis values (time or frames)
        x_values = frames
        if time_based:
            x_values = frames / detection_data.fps
        
        # Get detections for this object
        obj_detections = detections_by_object.get(obj_name, [])
        
        # Extract detection data
        detection_frames = np.array([d.frame_idx for d in obj_detections])
        detection_activeness = np.array([d.activeness for d in obj_detections])
        detection_energy = np.array([d.energy for d in obj_detections])
        
        detection_x = detection_frames
        if time_based:
            detection_x = detection_frames / detection_data.fps
        
        # Plot activeness
        ax1 = axes[i, 0]
        # Facecolor alpha
        facecolor_alpha = 0.2
        if obj_name in first_annotated_objects:
            ax1.set_facecolor('lightgreen')
            ax1.patch.set_alpha(facecolor_alpha)
        ax1.plot(x_values, activeness, 'b-', alpha=0.7, label='Raw Activeness')
        ax1.axhline(y=activeness_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({activeness_threshold})')
        
        # Highlight active detections if requested
        if highlight_detections:
            active_mask = detection_activeness >= activeness_threshold
            if np.any(active_mask):
                ax1.scatter(detection_x[active_mask], detection_activeness[active_mask], 
                           color='green', marker='o', s=30, alpha=0.7, label='Active Detection')
        
        ax1.set_ylabel('Activeness')
        ax1.set_title(f'{obj_name} - Activeness')
        ax1.grid(True, alpha=0.3)
        
        
        # Plot energy
        ax2 = axes[i, 1]
        if obj_name in first_annotated_objects:
            ax2.patch.set_alpha(facecolor_alpha)
            ax2.set_facecolor('lightgreen')
        ax2.plot(detection_x, detection_energy, 'g-', alpha=0.7, label='Energy')
        ax2.axhline(y=energy_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({energy_threshold})')
        
        # Highlight active detections if requested
        if highlight_detections:
            active_mask = detection_energy >= energy_threshold
            if np.any(active_mask):
                ax2.scatter(detection_x[active_mask], detection_energy[active_mask], 
                           color='green', marker='o', s=30, alpha=0.7, label='Active Detection')
        
        ax2.set_ylabel('Energy')
        ax2.set_title(f'{obj_name} - Energy')
        ax2.grid(True, alpha=0.3)
        

        # Calculate range for rectangular highlight (±2 seconds)
        if time_based:
            rect_start = first_annotated_time - 1.0
            rect_end = first_annotated_time + 1.0
            # Ensure we don't have negative time
            rect_start = max(0, rect_start)
        else:
            # Convert to frame numbers
            fps = detection_data.fps
            rect_start = int(max(0, (first_annotated_time - 1.0) * fps))
            rect_end = int((first_annotated_time + 1.0) * fps)
        
        # Rectangle width
        rect_width = rect_end - rect_start
        
        # For the activeness plot
        # Vertical line at annotation time
        ax1.axvline(x=first_annotated_time if time_based else int(first_annotated_time * detection_data.fps), 
                color='r', linestyle='--', alpha=0.5, label=f'Annotation ({first_annotated_time:.2f}s)')
        # Add rectangle spanning ±2 seconds
        ax1.axvspan(rect_start, rect_end, alpha=0.2, color='yellow', label='±2s Window')
        
        # For the energy plot
        # Vertical line at annotation time
        ax2.axvline(x=first_annotated_time if time_based else int(first_annotated_time * detection_data.fps), 
                color='r', linestyle='--', alpha=0.5, label=f'Annotation ({first_annotated_time:.2f}s)')
        # Add rectangle spanning ±2 seconds
        ax2.axvspan(rect_start, rect_end, alpha=0.2, color='yellow', label='±2s Window')
    #LEGENDS
   

    # Collect all legend elements from first subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine unique legend entries
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Add one legend to the figure
    fig.legend(unique_handles, unique_labels, loc='upper right', fontsize='small')



    # Set common labels
    fig.supxlabel(x_label)
    session = session_camera[0]
    camera = session_camera[1]
    fig.suptitle(f"{session} / {camera}: Activeness / Energy ", fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



def main():

    import argparse

    parser = argparse.ArgumentParser(description="Plot detection energy and activeness.")

    parser.add_argument("--session", type=str, required=True, help="Session name (e.g., 'imi_session1_2')")
    parser.add_argument("--camera", type=str, default="multi", help="Camera view (e.g., 'cam_side_l')")
    parser.add_argument("--energy_threshold", type=float, default=0.3, help="Energy threshold for active detections")
    parser.add_argument("--activeness_threshold", type=float, default=0.5, help="Activeness threshold for active detections")
    parser.add_argument("--output", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/detection_plots", help="Saving folder for the plot")


    # Parse arguments
    args = parser.parse_args()
    session = args.session
    camera = args.camera
    energy_threshold = args.energy_threshold
    activeness_threshold = args.activeness_threshold


    detection_root = "/nas/project_data/B1_Behavior/rush/kaan/hoi/outputs/tracking_multi"
    # Get DetectionData and AnnotationData
    detection_data = DetectionData(session_name=session, camera_view=camera,
    detection_root = detection_root)
    detections = detection_data.process_with_energy(energy_threshold=energy_threshold, activeness_threshold=activeness_threshold)

    annotation_data = AnnotationData(session_name=session)
    first_annotated_time = annotation_data.get_first_segment("HandUsed").start_time

    # Convert annotation time to frame number
    first_annotated_frame = first_annotated_time * detection_data.fps
    first_annotated_objects = annotation_data.get_first_played_object(combine=True)

    output_folder = os.path.join(args.output, session,camera)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Output folder: {output_folder}")

    plot_detection_energy(
    detection_data,
    object_names= None,
    activeness_threshold=activeness_threshold,
    energy_threshold=energy_threshold,
    first_annotated_time=first_annotated_time,
    first_annotated_objects = first_annotated_objects if len(first_annotated_objects) > 0 else [first_annotated_objects],
    figsize=(24, 16),
    time_based=False,  # Changed to False to use frame numbers
    highlight_detections=True,
    session_camera = (session, camera),
    save_path=os.path.join(output_folder, f"{session}_{camera}_activeness_energy_plot.png")
    )
    print(f"Plot saved to {output_folder}")
if __name__ == "__main__":
    
    main()
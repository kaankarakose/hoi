"""
Plots for visualizing detection results.

"""


import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING, Any

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from object_interaction_detection.evaluation.utils.detection_parser import DetectionData

def plot_detection_en123123ergy(
    detection_data: Any,  # Type will be DetectionData
    object_names: Optional[List[str]] = None,
    activeness_threshold: float = 0.5,
    energy_threshold: float = 0.3,
    first_annotated_time: float = 0.0,
    figsize: Tuple[int, int] = (60, 40),
    time_based: bool = True,
    highlight_detections: bool = True,
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
    n_objects = len(object_names)
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
        if i >= len(axes):
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
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    # Set common labels
    fig.supxlabel(x_label)
    fig.suptitle('Object Detection: Activeness vs Energy', fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
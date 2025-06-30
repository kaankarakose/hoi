import os
import sys
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from tqdm import tqdm

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
from  object_interaction_detection.dataloaders.motion_filtered_loader import MotionFilteredLoader
from  object_interaction_detection.tracking.annotation_parser import AnnotationData


def get_frame_sequence(frame_indices, start_time, duration=100):
    """
    Get a sequence of frames starting from start_time.
    
    Args:
        frame_indices: List of available frame indices (may be non-sequential)
        start_time: Starting frame index
        duration: Number of frames to get after start_time
    
    Returns:
        List of frame indices
    """
    # Convert to sorted list for easier handling
    available_frames = sorted(set(frame_indices))
    
    # Find the starting position
    start_idx = None
    for i, frame in enumerate(available_frames):
        if frame >= start_time:
            start_idx = i
            break
    
    if start_idx is None:
        return []  # No frames at or after start_time
    
    # Get up to 'duration' frames starting from start_idx
    selected_frames = available_frames[:start_idx + duration]
    
    return selected_frames


class ObjectActivenessTracker:
    """
    Tracks object activeness over time using the MotionFilteredLoader.
    
    This class collects activeness data for objects across frames and CAMERA VIEW - Slight different from 
    object_activeness_tracker.py, saves the data to JSON, and creates visualizations.
    """
    
    def __init__(self,frame_indices: List[int], 
                 annotation_data: Optional[AnnotationData] = None, 
                 session_name: Optional[str] = None,
                 annotation_root: Optional[str] = None,
                 fps = 30):
        """
        Initialize the activeness tracker.
        
        Args:
            frame_indices: List of frame indices to process
            annotation_data: Optional AnnotationData instance for visualization
            session_name: Session name to automatically load annotations
            annotation_root: Optional root directory for annotations
        """
        self.camera_views = ['cam_top', 'cam_side_r', 'cam_side_l'] # Camera views to process
        
      
        self.annotation_data = annotation_data
        self.session_name = session_name

        # Load annotations from session name if provided and annotation_data is None
        if annotation_data is None and session_name is not None:
            self.annotation_data = AnnotationData()
            if not self.annotation_data.load_from_session(session_name, annotation_root):
                logging.warning(f"Failed to load annotations for session: {session_name}")
                raise ValueError(f"Failed to load annotations for session: {session_name}")
        self.fps = fps
        self.start_time = int(self.annotation_data.annotations['HandUsed'][0].start_time * self.fps)


        #self.frame_indices = frame_indices[:self.start_time + min(frame_idx)] # frame_indices may not start from 0 and sequence may not be continuous
        self.frame_indices = get_frame_sequence(frame_indices, self.start_time, duration=120)

        # Dictionary to store activeness data for each object
        # Format: {object_name: {"activeness": [], "frame_idx": [], "object_id": id}}
        self.object_data = {}
        
    def collect_data(self, motion_loader):
        """
        Collect activeness data for all objects across frames by comparing mask sizes across camera views
        and choosing the camera view with the largest mask for each object.
        
        Args:
            motion_loader: MotionFilteredLoader instance
        """
        
        total_frames = len(self.frame_indices)
        logging.info(f"{total_frames} is processing")
        for i, frame_idx in tqdm(enumerate(self.frame_indices),desc = "Processing frames"):
            # Load features from ALL camera views for this frame
            camera_features = {}
            camera_object_maps = {}
            
            for camera_view in self.camera_views:
                features = motion_loader.load_features(camera_view, frame_idx)
                if features['motion_filtered']['success']:
                    camera_features[camera_view] = features
                    camera_object_maps[camera_view] = features['motion_filtered']['object_id_map']
            
            # Skip frame if no camera views have valid features
            if not camera_features:
                continue
            

            # Get all unique objects across all camera views
            all_objects = set()
            for obj_map in camera_object_maps.values():
                all_objects.update(obj_map.keys())
            
            # For each object, find the camera view with the largest mask
            for obj_name in all_objects:
                best_camera_view = None
                largest_mask_size = 0
                
                # Compare mask sizes across camera views for this object
                for camera_view, features in camera_features.items():
                    if obj_name in camera_object_maps[camera_view]:
                        # Get mask size for this object in this camera view
                        mask_size = self.get_mask_size(features, obj_name)                        
                        if mask_size > largest_mask_size:
                            largest_mask_size = mask_size
                            best_camera_view = camera_view
                
                # Skip if no valid mask found for this object
                if best_camera_view is None:
                    continue
                print(best_camera_view)
                print(frame_idx)
                print(obj_name)
                # Get activeness using the camera view with the largest mask
                activeness_result = motion_loader.get_activeness_by_object(
                    best_camera_view, frame_idx, obj_name
                    )
                
                # Skip if activeness calculation failed
                if not activeness_result['success']:
                    continue
                
                # Get activeness value
                activeness = activeness_result['activeness']
                
                # Initialize object data if not already tracked
                if obj_name not in self.object_data:
                    self.object_data[obj_name] = {
                        "activeness": [],
                        "frame_idx": [],
                        "camera_view": [],  # Track which camera view was used
                        "object_id": camera_object_maps[best_camera_view][obj_name]
                    }
                
                # Append data
                self.object_data[obj_name]["activeness"].append(activeness)
                self.object_data[obj_name]["frame_idx"].append(frame_idx)
                self.object_data[obj_name]["camera_view"].append(best_camera_view)
            
        logging.info(f"Collected activeness data for {len(self.object_data)} objects across {total_frames} frames")

    def get_mask_size(self, features, obj_name):
        """
        Extract mask size for a given object from features.
        You'll need to implement this based on your data structure.
        
        Args:
            features: Features dictionary for a camera view
            obj_name: Name of the object
            
        Returns:
            int: Size of the mask (e.g., number of pixels, area, etc.)
        """

        object_id = features['motion_filtered']['object_id_map'][obj_name]
        mask = features['motion_filtered']['mask']

        # Count pixels that belong to this object ID
        if hasattr(mask, 'shape'):  # numpy array
            mask_size = np.sum(mask == object_id)
        else:  # regular list/array
            mask_size = sum(1 for pixel in mask.flat if pixel == object_id) if hasattr(mask, 'flat') else \
                    sum(1 for row in mask for pixel in row if pixel == object_id)
        
        return int(mask_size)
    
    def save_to_json(self, output_path: str):
        """
        Save the collected data to a JSON file.
        
        Args:
            output_path: Path to the output JSON file
        """
        print(self.object_data)
        # Create the data structure to save
        data = {
            "objects": self.object_data,
            "metadata": {
                "num_frames": len(self.frame_indices),
                "num_objects": len(self.object_data),
                "frame_range": [min(self.frame_indices), max(self.frame_indices)]
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for obj_name, obj_data in data["objects"].items():
            for key, value in obj_data.items():
                if isinstance(value, np.ndarray):
                    obj_data[key] = value.tolist()
                elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    obj_data[key] = [v.tolist() for v in value]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"Saved activeness data to {output_path}")
            
    def visualize_basic(self, output_path: Optional[str] = None, 
                        title: Optional[str] = None, threshold: float = 0.25):
        """
        Create a basic visualization of object activeness over time.
        
        Args:
            output_path: Path to save the visualization (optional)
            title: Title for the visualization (optional)
            threshold: Activeness threshold (default: 0.25)
        """
        # Create figure and axis
        plt.figure(figsize=(12, 8))
        
        # Generate colors for each object using tab10 colormap
        cmap = plt.cm.get_cmap('tab10')
        
        # Plot activeness for each object
        for i, (obj_name, data) in enumerate(self.object_data.items()):
            color = cmap(i % 10)
            frame_indices = data["frame_idx"]
            activeness = data["activeness"]
            
            plt.plot(frame_indices, activeness, label=obj_name, color=color, linewidth=2)
        
        # Add threshold line
        plt.axhline(y=threshold, color='r', linestyle='--', 
                   label=f"Motion Threshold ({threshold})", alpha=0.7)
        
        # Add labels and legend
        plt.xlabel("Frame Index")
        plt.ylabel("Activeness")
        plt.title(title or f"Object Activeness Over Time")
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
        plt.grid(True, alpha=0.3)
        
        # Ensure y-axis starts at 0
        plt.ylim(bottom=0)
        
        # Save the figure if an output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logging.info(f"Saved activeness visualization to {output_path}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def visualize_advanced(self, output_path: Optional[str] = None, 
                           title: Optional[str] = None, 
                           threshold: float = 0.25,
                           fps: float = 30.0):
        """
        Create an advanced visualization showing object activeness over time
        with highlighting of active regions, separation of major/minor objects,
        and annotation overlays.
        
        Args:
            output_path: Path to save the visualization (optional)
            title: Title for the visualization (optional)
            threshold: Activeness threshold (default: 0.25)
            fps: Frames per second for annotation timing (default: 30.0)
        """
        # Separate objects into major and minor categories based on activity
        major_objects = {}
        minor_objects = {}
        
        for obj_name, data in self.object_data.items():
            active_frames = sum(1 for a in data["activeness"] if a >= threshold)
            if active_frames >= 10:  # At least 10 frames above threshold
                major_objects[obj_name] = data
            else:
                minor_objects[obj_name] = data
        
        # Determine if we need an additional subplot for annotations
        use_annotation_subplot = self.annotation_data is not None
        
        # Create figure with appropriate number of subplots
        if use_annotation_subplot:
            fig, axs = plt.subplots(2, 1, figsize=(14, 12), 
                                   gridspec_kw={'height_ratios': [3, 1]})
            ax1, ax_annot = axs
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                          gridspec_kw={'height_ratios': [3, 1]})
        
        # Get colormaps
        major_cmap = plt.cm.get_cmap('tab10')
        minor_cmap = plt.cm.get_cmap('Pastel1')
        # Camera markers
        camera_markers = {
            'cam_top': '^',      # triangle up
            'cam_side_r': 's',   # square  
            'cam_side_l': 'o'    # circle
        }
        camera_colors = {
            'cam_top': '#E31A1C',       # Bright red
            'cam_side_r': '#1F78B4',    # Steel blue
            'cam_side_l': '#33A02C'     # Forest green
        }
        # Plot major objects
        for i, (obj_name, data) in enumerate(major_objects.items()):
            color = major_cmap(i % 10)
            frame_indices = data["frame_idx"]
            activeness = data["activeness"]
            camera_views = data["camera_view"]
            # Plot the activeness line
            ax1.plot(frame_indices, activeness, label=obj_name, color=color, linewidth=2)
            # Scatter points with markers based on camera view
            for frame_idx, active_val, cam_view in zip(frame_indices, activeness, camera_views):
                marker = camera_markers[cam_view]
                cam_color = camera_colors[cam_view]
                ax1.scatter(frame_idx, active_val, marker=marker, 
                   facecolors=cam_color, edgecolors='black', 
                   s=80, alpha=0.8, linewidths=0.5, zorder=5)
            # Highlight active regions
            active_regions = []
            start_idx = None
            
            for j, act in enumerate(activeness):
                if act >= threshold:
                    if start_idx is None:
                        start_idx = frame_indices[j]
                elif start_idx is not None:
                    active_regions.append((start_idx, frame_indices[j-1]))
                    start_idx = None
                    
            # Add final region if needed
            if start_idx is not None:
                active_regions.append((start_idx, frame_indices[-1]))
            
            # Shade active regions
            for start, end in active_regions:
                ax1.axvspan(start, end, color=color, alpha=0.1)
        
        ax1.axhline(y=threshold, color='r', linestyle='--', 
                label=f"Motion Threshold ({threshold})", alpha=0.7)
        start_time = int(self.annotation_data.annotations['HandUsed'][0].start_time * fps)
        ax1.axvline(x=start_time, color='g', linestyle='-', linewidth=3, alpha=0.7,
                    label = "Annotation")
        # Configure main plot
        ax1.set_title(title or f"Object Activeness")
        ax1.set_ylabel("Activeness")
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(0, (start_time * 2))  # Changed to limit x-axis from 0 to 100
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
        # Create legend elements for camera markers

        camera_legend_elements = []
        # Create a descriptive label for the camera
        cam_labels = {
            'cam_top': 'Top Camera',
            'cam_side_r': 'Right Side Camera', 
            'cam_side_l': 'Left Side Camera'
        }
        for cam_view, marker in camera_markers.items():
            color = camera_colors[cam_view]
            label = cam_labels[cam_view]
            
            # Create legend element
            legend_element = Line2D([0], [0], marker=marker, color='w', 
                                markerfacecolor=color, markeredgecolor=color,
                                markersize=10, label=label, linestyle='None')
            camera_legend_elements.append(legend_element)

        # Add Camera legends
        # Create the original object legend first and save it
        original_legend = ax1.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

        # Add separate camera marker legend in upper left
        camera_legend = ax1.legend(camera_legend_elements, 
                                [elem.get_label() for elem in camera_legend_elements],
                                loc="upper left", 
                                title="Camera Views", 
                                frameon=True, 
                                fancybox=True, 
                                shadow=True,
                                fontsize=10)

        # Add the original legend back (since the second legend call replaced it)
        ax1.add_artist(original_legend)
        
        # Add annotation data if available
        if use_annotation_subplot:
            # Set the fps in the annotation data
            self.annotation_data.set_fps(fps)
            
            # Get min and max frame indices to determine plot range
            min_frame = min(min(data["frame_idx"]) for data in self.object_data.values())
            max_frame = max(max(data["frame_idx"]) for data in self.object_data.values())
            min_time = min_frame / fps
            max_time = max_frame / fps
            
            # Define colors for different hand types
            hand_colors = {
                'L': 'blue',
                'R': 'red',
                'B': 'purple'
            }
            
            # Add hand usage segments
            y_pos_hands = 0.2
            y_height_hands = 0.4
            
            # Draw hand annotations
            for segment in self.annotation_data.get_all_segments("HandUsed"):
                # Skip segments outside our frame range
                if segment.end_time < min_time or segment.start_time > max_time:
                    continue
                    
                # Get segment color based on hand type
                hand_type = segment.get_hand_used()
                color = hand_colors.get(hand_type, 'gray')
                
                # Convert start time to frame index for plotting
                start_frame = segment.start_time * fps
                
                # Add vertical line at start time
                ax_annot.axvline(x=start_frame, color=color, linestyle='-', linewidth=2, alpha=0.7)
                
                # Add text label at start time
                y_text_pos = y_pos_hands + y_height_hands/2
                ax_annot.text(start_frame + 5, y_text_pos, hand_type,
                             horizontalalignment='left', verticalalignment='center',
                             fontsize=10, fontweight='bold', color=color, 
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.3'))
            
            # Add object identity segments
            y_pos_objects = 0.7
            y_height_objects = 0.4
            
            # Draw object annotations
            for segment in self.annotation_data.get_all_segments("ObjectIdentity"):
                # Skip segments outside our frame range
                if segment.end_time < min_time or segment.start_time > max_time:
                    continue
                    
                # Convert start time to frame index for plotting
                start_frame = segment.start_time * fps
                
                # Add vertical line at start time
                ax_annot.axvline(x=start_frame, color='green', linestyle='-', linewidth=2, alpha=0.7)
                
                # Add text label - truncate if too long
                object_text = segment.metadata
                if len(object_text) > 15:
                    object_text = object_text[:12] + '...'
                
                # Add text label at start time
                y_text_pos = y_pos_objects + y_height_objects/2
                ax_annot.text(start_frame + 5, y_text_pos, object_text,
                             horizontalalignment='left', verticalalignment='center',
                             fontsize=10, fontweight='bold', color='darkgreen', 
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.3'))
            
            # Configure annotation subplot
            ax_annot.set_xlim(min_frame, max_frame)
            ax_annot.set_ylim(0, 1.2)
            ax_annot.set_title("Annotations: Hand Usage and Object Identity")
            ax_annot.set_xlabel("Frame Index")
            ax_annot.set_yticks([0.4, 0.9])
            ax_annot.set_yticklabels(["Hand Used", "Objects"])
            ax_annot.grid(True, alpha=0.3)
            
            # Add a legend for hand types
            from matplotlib.patches import Patch
            hand_legend_elements = [Patch(facecolor=color, alpha=0.4, edgecolor='black',
                                         label=hand_type) 
                                   for hand_type, color in hand_colors.items()]
            ax_annot.legend(handles=hand_legend_elements, loc='upper right')
            
        # Save the figure if an output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logging.info(f"Saved advanced visualization to {output_path}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def create_heatmap(self, output_path: Optional[str] = None, 
                      title: Optional[str] = None,
                      threshold: float = 0.25):
        """
        Create a heatmap visualization showing object activeness over time.
        
        Args:
            output_path: Path to save the visualization (optional)
            title: Title for the visualization (optional)
            threshold: Activeness threshold for highlighting (default: 0.25)
        """
        # First, normalize frame indices to create a regular grid
        all_frames = sorted(list(set().union(*[data["frame_idx"] for data in self.object_data.values()])))
        frame_to_idx = {frame: i for i, frame in enumerate(all_frames)}
        
        # Create a matrix for the heatmap
        objects = list(self.object_data.keys())
        heatmap_data = np.zeros((len(objects), len(all_frames)))
        
        # Fill the matrix with activeness values
        for i, obj_name in enumerate(objects):
            data = self.object_data[obj_name]
            for j, (frame, act) in enumerate(zip(data["frame_idx"], data["activeness"])):
                col_idx = frame_to_idx[frame]
                heatmap_data[i, col_idx] = act
        
        # Create the figure
        plt.figure(figsize=(16, 8))
        
        # Create the heatmap
        im = plt.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                      interpolation='nearest', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Activeness')
        
        # Add labels
        plt.ylabel('Object')
        plt.xlabel('Frame Index')
        plt.title(title or f"Object Activeness Heatmap")
        
        # Set tick labels
        plt.yticks(range(len(objects)), objects)
        
        # For x-axis, show actual frame numbers but only some of them to avoid crowding
        if len(all_frames) > 20:
            step = len(all_frames) // 20
            tick_positions = range(0, len(all_frames), step)
            tick_labels = [all_frames[i] for i in tick_positions]
            plt.xticks(tick_positions, tick_labels)
        else:
            plt.xticks(range(len(all_frames)), all_frames)
        
        # Add a horizontal line to separate objects based on some criteria
        active_counts = [(obj, sum(1 for a in data["activeness"] if a >= threshold)) 
                        for obj, data in self.object_data.items()]
        active_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Find where to draw separation line (after major objects)
        separator_idx = 0
        for i, (obj, count) in enumerate(active_counts):
            if count < 10:  # Less than 10 active frames
                separator_idx = i
                break
        
        if 0 < separator_idx < len(objects):
            plt.axhline(y=separator_idx - 0.5, color='white', linestyle='-', linewidth=2)
        
        # Save the figure if an output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logging.info(f"Saved heatmap visualization to {output_path}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()


import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Track object activeness across video frames")
    
    parser.add_argument("--session", type=str, required=True,
                       help="Session name to process")
    parser.add_argument("--data-root", type=str,default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data",
                       help="Root directory for data")
    parser.add_argument("--camera", type=str, default="cam_top",
                       help="Camera view to process (default: cam_top)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for visualizations and data")
    parser.add_argument("--num-frames", type=int, default=100,
                       help="Number of frames to process (default: 100)")
    parser.add_argument("--score-threshold", type=float, default=0.52)
                       
    parser.add_argument("--threshold", type=float, default=0.25,
                       help="Activeness threshold (default: 0.25)")
    parser.add_argument("--annotation-root", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/annotations",
                       help="Root directory for annotations (default: /nas/project_data/B1_Behavior/rush/kaan/hoi/annotations)")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Frames per second for annotation timing (default: 30.0)")
    
    return parser.parse_args()

def main():
    """Main function to track object activeness."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize the MotionFilteredLoader
    logging.info(f"Initializing MotionFilteredLoader for session {args.session}")
    motion_loader = MotionFilteredLoader(
        session_name=args.session,
        data_root_dir=args.data_root,
        score_threshold=args.score_threshold,
        
    )
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.session)
    os.makedirs(output_dir, exist_ok=True)
    
  # Get valid frame indices for all camera views and find intersection
    valid_frame_by_camera = []
    for camera_view in motion_loader.camera_views:
        valid_frames = motion_loader.combined_loader.hamer_loader.get_valid_frame_idx(camera_view)
        valid_frame_by_camera.append(valid_frames)
    # Find frames that are valid in ALL camera views
    if valid_frame_by_camera:
        common_valid_frames = list(set.intersection(*map(set, valid_frame_by_camera)))
    else:
        raise ValueError("No valid frames found for any camera view?")
    
    # Determine frame range to process
    min_frame = min(common_valid_frames)
    max_frame = max(common_valid_frames)
    
    # Create activeness tracker with annotation data
    tracker = ObjectActivenessTracker(
        frame_indices=common_valid_frames,
        session_name=args.session,
        annotation_root=args.annotation_root
    )
    
    # Log annotation status
    if tracker.annotation_data is not None:
        annotation_count = len(tracker.annotation_data.get_all_segments("HandUsed"))
        object_count = len(tracker.annotation_data.get_all_segments("ObjectIdentity"))
        logging.info(f"Loaded annotations: {annotation_count} hand segments and {object_count} object segments")
    else:
        logging.warning("No annotation data found or loaded")
    
    # Collect data
    tracker.collect_data(motion_loader)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output paths
    base_filename = f"{args.session}"
    json_path = os.path.join(output_dir, f"{base_filename}_activeness.json")
    basic_viz_path = os.path.join(output_dir, f"{base_filename}_visualization_basic.png")
    advanced_viz_path = os.path.join(output_dir, f"{base_filename}_visualization_advanced.png")
    heatmap_path = os.path.join(output_dir, f"{base_filename}_visualization_heatmap.png")
    
    # Save data
    tracker.save_to_json(json_path)
    
    # Create visualizations
    tracker.visualize_basic(basic_viz_path, threshold=args.threshold)
    tracker.visualize_advanced(
        output_path=advanced_viz_path, 
        threshold=args.threshold,
        fps=args.fps
    )
    tracker.create_heatmap(
        output_path=heatmap_path, 
        title=f"Object Activeness Heatmap - {args.session}",
        threshold=args.threshold
    )
    
    # Create a simplified annotation timeline if annotations are available
    if tracker.annotation_data is not None:
        timeline_path = os.path.join(output_dir, f"{base_filename}_annotation_timeline.png")
        tracker.annotation_data.visualize_timeline(save_path=timeline_path)
        logging.info(f"Saved annotation timeline to {timeline_path}")
    
    logging.info(f"Processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
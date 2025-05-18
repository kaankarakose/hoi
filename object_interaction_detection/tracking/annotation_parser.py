"""
Annotation Parser for handling human behavior annotations with timing information.

This module provides classes for parsing, storing, and retrieving annotation data
related to hand usage and object interactions.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from datetime import timedelta
import logging

@dataclass
class AnnotationSegment:
    """A single segment of an annotation with start/end times and metadata."""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    duration: float    # Duration in seconds
    metadata: str      # Annotation-specific metadata (e.g., "B" for both hands, or "AMF1,HAIRBRUSH" for objects)
    
    def contains_time(self, time_seconds: float) -> bool:
        """Check if this segment contains the given time."""
        return self.start_time <= time_seconds < self.end_time
    
    def contains_frame(self, frame_idx: int, fps: float = 30.0) -> bool:
        """Check if this segment contains the given frame."""
        time_seconds = frame_idx / fps
        return self.contains_time(time_seconds)
    
    def get_objects(self) -> List[str]:
        """For ObjectIdentity annotations, return the list of objects."""
        return [obj.strip() for obj in self.metadata.split(',')]
    
    def get_hand_used(self) -> str:
        """For HandUsed annotations, return the hand used (L/R/B)."""
        return self.metadata


class AnnotationData:
    """Parser and container for annotation data."""
    
    def __init__(self, session_name=None):
        """Initialize annotation data, optionally loading from session.
        
        Args:
            session_name: If provided, automatically load annotations for this session
        """
        self.annotations = {
            "HandUsed": [],
            "ObjectIdentity": []
        }
        self.fps = 30.0  # Default FPS
        self.session_name = session_name
        
        # Automatically load annotations if session name is provided
        if session_name:
            self.load_from_session(session_name)
    
    @staticmethod
    def _parse_time(time_str: str) -> float:
        """Parse a time string in various formats to seconds.
        
        Handles formats:
        - HH:MM:SS.mmm (timestamp format)
        - SS.mmm (seconds with milliseconds)
        - SSSSS (milliseconds as integer)
        """
        if not time_str or time_str.isspace():
            return 0.0
            
        if ':' in time_str:
            # Parse from timestamp format HH:MM:SS.mmm
            try:
                h, m, s = time_str.split(':')
                return int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                # Handle unexpected format gracefully
                return 0.0
        elif '.' in time_str:
            # Format is likely seconds.milliseconds
            try:
                return float(time_str)
            except ValueError:
                return 0.0
        else:
            # Try parsing as float, could be milliseconds or frames
            try:
                value = float(time_str)
                # If value is very large (> 10000), it might be milliseconds
                if value > 10000:
                    return value / 1000.0
                return value
            except ValueError:
                return 0.0
    
    def load_from_text(self, text: str):
        """Load annotations from text content."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            parts = re.split(r'\s+', line)
            
            if len(parts) < 10:
                continue  # Skip invalid lines
                
            # Using the user-provided column structure:
            # First column: annotation type
            # Columns 2-4: start time in different formats 
            # Columns 5-7: end time in different formats
            # Columns 8-10: duration in different formats
            # Column 11: metadata (hand type or objects)
                
            annotation_type = parts[0]
            
            # Use the timestamp format (first of each group)
            start_time = self._parse_time(parts[1])  # First timestamp
            end_time = self._parse_time(parts[4])    # Second timestamp
            duration = self._parse_time(parts[7])    # Third timestamp
            
            # The last part is the metadata (hand type or objects)
            metadata = parts[10] if len(parts) > 10 else ''
            
            segment = AnnotationSegment(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                metadata=metadata
            )
            
            if annotation_type in self.annotations:
                self.annotations[annotation_type].append(segment)
    
    def load_from_file(self, file_path: str):
        """Load annotations from a file."""
        with open(file_path, 'r') as f:
            self.load_from_text(f.read())
    
    def load_from_session(self, session_name: str, annotation_root: str = None):
        """Load annotations for a specific session.
        
        Args:
            session_name: Name of the session (e.g., 'imi_session1_2')
            annotation_root: Root directory for annotations, defaults to standard path
        """
        if annotation_root is None:
            annotation_root = "/nas/project_data/B1_Behavior/rush/kaan/hoi/annotations"
        
        # Construct the path to the annotation file
        session_dir = os.path.join(annotation_root, session_name)
        annotation_file = os.path.join(session_dir, f"{session_name}.txt")
        
        # Check if the file exists
        if not os.path.exists(annotation_file):
            logging.warning(f"Annotation file not found: {annotation_file}")
            return False
        
        # Load the annotations
        try:
            self.load_from_file(annotation_file)
            logging.info(f"Loaded annotations for session {session_name} from {annotation_file}")
            return True
        except Exception as e:
            logging.error(f"Error loading annotations for session {session_name}: {e}")
            return False
    
    def set_fps(self, fps: float):
        """Set the frames per second for frame-based queries."""
        self.fps = fps
    
    def get_hand_used_at_time(self, time_seconds: float) -> Optional[str]:
        """Get which hand was used at the given time."""
        for segment in self.annotations["HandUsed"]:
            if segment.contains_time(time_seconds):
                return segment.get_hand_used()
        return None
    
    def get_objects_at_time(self, time_seconds: float) -> List[str]:
        """Get the list of objects being used at the given time."""
        for segment in self.annotations["ObjectIdentity"]:
            if segment.contains_time(time_seconds):
                return segment.get_objects()
        return []
    
    def get_hand_used_at_frame(self, frame_idx: int) -> Optional[str]:
        """Get which hand was used at the given frame."""
        time_seconds = frame_idx / self.fps
        return self.get_hand_used_at_time(time_seconds)
    
    def get_objects_at_frame(self, frame_idx: int) -> List[str]:
        """Get the list of objects being used at the given frame."""
        time_seconds = frame_idx / self.fps
        return self.get_objects_at_time(time_seconds)
    
    def get_all_objects(self) -> List[str]:
        """Get a list of all unique objects mentioned in annotations."""
        all_objects = set()
        for segment in self.annotations["ObjectIdentity"]:
            for obj in segment.get_objects():
                all_objects.add(obj)
        return sorted(list(all_objects))
    
    def get_all_segments(self, annotation_type: str) -> List[AnnotationSegment]:
        """Get all segments for a given annotation type."""
        return self.annotations.get(annotation_type, [])
    
    def visualize_timeline(self, figsize=(12, 6), save_path=None):
        """
        Create a visualization of the annotations as a timeline.
        
        Args:
            figsize: Figure size (width, height) in inches
            save_path: If provided, save to this path instead of displaying
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors
        hand_colors = {
            'L': 'blue',
            'R': 'red',
            'B': 'purple'
        }
        
        # Find max time for x-axis
        max_time = 0
        for segments in self.annotations.values():
            for segment in segments:
                max_time = max(max_time, segment.end_time)
        
        # Plot HandUsed segments
        y_offset = 0.6
        for i, segment in enumerate(self.annotations["HandUsed"]):
            color = hand_colors.get(segment.metadata, 'gray')
            ax.barh(y_offset, segment.duration, left=segment.start_time, 
                   color=color, alpha=0.6, height=0.2)
            # Add text in the middle of the bar
            text_x = segment.start_time + segment.duration / 2
            ax.text(text_x, y_offset, segment.metadata, ha='center', va='center')
        
        # Plot ObjectIdentity segments
        y_offset = 0.2
        for i, segment in enumerate(self.annotations["ObjectIdentity"]):
            ax.barh(y_offset, segment.duration, left=segment.start_time, 
                   color='green', alpha=0.6, height=0.2)
            # Add text in the middle of the bar
            text_x = segment.start_time + segment.duration / 2
            ax.text(text_x, y_offset, segment.metadata, ha='center', va='center', 
                   fontsize=8, rotation=0)
        
        # Set ticks and labels
        ax.set_yticks([0.2, 0.6])
        ax.set_yticklabels(['Objects', 'Hands'])
        
        # Set time formatting for x-axis
        def format_time(seconds, _):
            return str(timedelta(seconds=seconds)).split('.')[0]
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_title('Annotation Timeline')
        
        # Grid and tight layout
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        return fig, ax
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert annotations to pandas DataFrames.
        
        Returns:
            Dictionary with keys as annotation types and values as DataFrames
        """
        result = {}
        
        for annotation_type, segments in self.annotations.items():
            data = []
            for segment in segments:
                data.append({
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'duration': segment.duration,
                    'metadata': segment.metadata
                })
            
            if data:
                result[annotation_type] = pd.DataFrame(data)
        
        return result


def create_annotation_overlay(frame, frame_idx, annotation_data, 
                              text_color=(255, 255, 255), 
                              bg_color=(0, 0, 0, 128)):
    """
    Create an annotation overlay for a video frame.
    
    Args:
        frame: The video frame (numpy array)
        frame_idx: Current frame index
        annotation_data: AnnotationData instance
        text_color: Text color (B,G,R)
        bg_color: Background color with alpha (B,G,R,A)
        
    Returns:
        Frame with annotation overlay
    """
    import cv2
    
    # Make a copy to avoid modifying original
    result = frame.copy()
    
    # Get annotation data for this frame
    hand_used = annotation_data.get_hand_used_at_frame(frame_idx)
    objects = annotation_data.get_objects_at_frame(frame_idx)
    
    # Create a semi-transparent overlay for text background
    h, w = frame.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Prepare text
    text_lines = []
    if hand_used:
        hand_text = "Hand: "
        if hand_used == "L":
            hand_text += "Left"
        elif hand_used == "R":
            hand_text += "Right"
        else:  # "B"
            hand_text += "Both"
        text_lines.append(hand_text)
    
    if objects:
        text_lines.append("Objects: " + ", ".join(objects))
    
    if not text_lines:
        return result  # No annotations to display
    
    # Calculate text dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    padding = 10
    
    # Calculate text box height
    line_height = 30
    box_height = len(text_lines) * line_height + 2 * padding
    
    # Draw background box
    cv2.rectangle(overlay, (0, 0), (w, box_height), bg_color, -1)
    
    # Draw text
    for i, line in enumerate(text_lines):
        y_pos = padding + (i + 0.5) * line_height
        cv2.putText(overlay, line, (padding, int(y_pos)), font, font_scale, 
                    text_color, font_thickness, cv2.LINE_AA)
    
    # Blend overlay with original frame
    alpha_overlay = overlay[:, :, 3:4] / 255.0
    alpha_frame = 1.0 - alpha_overlay
    
    for c in range(3):  # Apply alpha blending to RGB channels
        result[:, :, c] = overlay[:, :, c] * alpha_overlay[:, :, 0] + \
                         result[:, :, c] * alpha_frame[:, :, 0]
    
    return result


# Example usage
if __name__ == "__main__":
    # Example annotation text - now with correct column alignment
    example_text = """HandUsed	00:00:02.014	2.014	2014	00:00:07.775	7.775	7775	00:00:05.761	5.761	5761	B
HandUsed	00:00:07.775	7.775	7775	00:00:10.830	10.83	10830	00:00:03.055	3.055	3055	L
HandUsed	00:00:10.830	10.83	10830	00:00:40.000	40.0	40000	00:00:29.170	29.17	29170	B
HandUsed	00:00:40.000	40.0	40000	00:00:46.810	46.81	46810	00:00:06.810	6.81	6810	R
HandUsed	00:00:46.810	46.81	46810	00:00:51.333	51.333	51333	00:00:04.523	4.523	4523	B
HandUsed	00:00:54.619	54.619	54619	00:00:59.333	59.333	59333	00:00:04.714	4.714	4714	R
ObjectIdentity	00:00:02.014	2.014	2014	00:00:07.775	7.775	7775	00:00:05.761	5.761	5761	AMF1,HAIRBRUSH
ObjectIdentity	00:00:07.775	7.775	7775	00:00:10.830	10.83	10830	00:00:03.055	3.055	3055	AMF1
ObjectIdentity	00:00:10.830	10.83	10830	00:00:40.000	40.0	40000	00:00:29.170	29.17	29170	AMF1,HAIRBRUSH
ObjectIdentity	00:00:40.000	40.0	40000	00:00:46.810	46.81	46810	00:00:06.810	6.81	6810	AMF1
ObjectIdentity	00:00:46.810	46.81	46810	00:00:51.333	51.333	51333	00:00:04.523	4.523	4523	AMF1
ObjectIdentity	00:00:54.619	54.619	54619	00:00:59.333	59.333	59333	00:00:04.714	4.714	4714	HAIRBRUSH"""
    
    # Create annotation data and load example
    annotations = AnnotationData()
    annotations.load_from_text(example_text)
    
    # Print some info
    print(f"All objects: {annotations.get_all_objects()}")
    print(f"Hand used at frame 100: {annotations.get_hand_used_at_frame(100)}")
    print(f"Objects at frame 100: {annotations.get_objects_at_frame(100)}")
    
    # Create a timeline visualization
    annotations.visualize_timeline(save_path="annotation_timeline.png")
    
    # Demo of overlay on fake frame
    try:
        import cv2
        # Create a fake frame for demonstration
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add annotation overlay
        annotated_frame = create_annotation_overlay(fake_frame, 100, annotations)
        # Save demo
        cv2.imwrite("annotation_overlay_demo.png", annotated_frame)
        print("Saved annotation overlay demo image.")
    except ImportError:
        print("OpenCV not available for demo.")

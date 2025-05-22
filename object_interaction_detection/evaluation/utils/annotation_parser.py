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
    ## TODO: process metadata
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
    def get_first_played_object(self, return_all=True, combine=False):
        """Get the first object(s) that was played in the session.
        
        Args:
            return_all: If True, returns all objects from the first segment, otherwise returns only the first object
            combine: If True, combines all objects into a single string joined with commas
        
        Returns:
            - When combine=True: A single string with all objects combined
            - When return_all=True and combine=False: A list of object strings
            - When return_all=False and combine=False: The first object string
        """
        first_segment = self.get_first_segment("ObjectIdentity")
        if not first_segment:
            return "" if combine else ([] if return_all else None)
            
        objects = first_segment.get_objects()
        if not objects:
            return "" if combine else ([] if return_all else None)
        
        # Return combined string with all objects
        if combine:
            return first_segment.metadata.strip()
            
        # Return all objects as list
        if return_all:
            return objects
        
        # Otherwise return just the first one
        return objects[0]
    
    def get_all_segments(self, annotation_type: str) -> List[AnnotationSegment]:
        """Get all segments for a given annotation type."""
        return self.annotations.get(annotation_type, [])
    def get_first_segment(self, annotation_type: str) -> Optional[AnnotationSegment]:
        """Get the first segment for a given annotation type."""
        return self.annotations.get(annotation_type, [])[0] if self.annotations.get(annotation_type, []) else None  
    
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



if __name__ == "__main__":
    

    annotation_data = AnnotationData(session_name="imi_session1_4")


    all_objects = annotation_data.get_all_objects()
    first_annotated_time = annotation_data.get_first_segment("ObjectIdentity").start_time

    first_played_object = annotation_data.get_first_segment("ObjectIdentity").get_objects()[0]
    print(first_played_object)
    print(first_annotated_time)
    print(annotation_data.get_first_played_object(combine=True))
    print(annotation_data.get_first_played_object(return_all=True, combine=False))
    print(annotation_data.get_first_played_object(return_all=False, combine=True))

    # print(annotation_data.get_all_segments("HandUsed"))
    # print(annotation_data.get_all_segments("ObjectIdentity"))

    # print(annotation_data.get_first_segment("HandUsed"))
    # print(annotation_data.get_first_segment("ObjectIdentity"))
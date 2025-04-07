#!/usr/bin/env python3
"""
Visualization utilities for hand-object interaction detection.
This module provides functions to visualize hand detections, object detections,
and interactions between hands and objects.
"""

import cv2
import numpy as np


def draw_hand_detections(frame, hand_results):
    """
    Draw hand detection results on the frame.
    
    Args:
        frame (numpy.ndarray): Input image/frame
        hand_results (dict): Hand detection results
        
    Returns:
        numpy.ndarray: Frame with hand visualizations
    """
    vis_frame = frame.copy()
    
    # Draw bounding boxes and keypoints for each detected hand
    for hand_type in ['left_hand', 'right_hand']:
        hand_data = hand_results.get(hand_type)
        if hand_data is None:
            continue
            
        # Draw bounding box
        x1, y1, x2, y2 = hand_data['bbox']
        color = (0, 255, 0) if hand_type == 'left_hand' else (0, 0, 255)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw hand position
        x, y = hand_data['position']
        cv2.circle(vis_frame, (x, y), 5, color, -1)
        
        # Add text label
        confidence = hand_data['confidence']
        label = f"{hand_type.replace('_', ' ').title()}: {confidence:.2f}"
        cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return vis_frame


def draw_object_detections(frame, object_results):
    """
    Draw object detection results on the frame.
    
    Args:
        frame (numpy.ndarray): Input image/frame
        object_results (list): Object detection results
        
    Returns:
        numpy.ndarray: Frame with object visualizations
    """
    vis_frame = frame.copy()
    
    # Define colors for different object classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]
    
    for detection in object_results:
        # Get bounding box coordinates
        x1, y1, x2, y2 = detection['bbox']
        
        # Get color based on class_id (with modulo to handle any number of classes)
        color = colors[detection['class_id'] % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cx, cy = detection['center']
        cv2.circle(vis_frame, (cx, cy), 5, color, -1)
        
        # Add label with class name and confidence
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return vis_frame


def draw_interactions(frame, hand_results, object_results, interactions):
    """
    Draw interactions between hands and objects.
    
    Args:
        frame (numpy.ndarray): Input image/frame
        hand_results (dict): Hand detection results
        object_results (list): Object detection results
        interactions (list): Interaction detection results
        
    Returns:
        numpy.ndarray: Frame with interaction visualizations
    """
    vis_frame = frame.copy()
    
    # Draw lines between interacting hands and objects
    for interaction in interactions:
        hand_type = interaction['hand_type']
        object_id = interaction['object_id']
        
        hand_data = hand_results.get(hand_type)
        if hand_data is None or object_id >= len(object_results):
            continue
            
        hand_position = hand_data['position']
        obj_center = object_results[object_id]['center']
        
        # Determine line color based on interaction type
        if interaction['interaction_type'] == 'contact':
            color = (0, 255, 0)  # Green for contact
            thickness = 2
        else:  # proximity
            color = (255, 255, 0)  # Yellow for proximity
            thickness = 1
            
        # Draw line between hand and object
        cv2.line(vis_frame, hand_position, obj_center, color, thickness)
        
        # Add interaction label
        mid_x = (hand_position[0] + obj_center[0]) // 2
        mid_y = (hand_position[1] + obj_center[1]) // 2
        label = f"{interaction['interaction_type']}: {interaction['confidence']:.2f}"
        cv2.putText(vis_frame, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis_frame


def create_visualization_grid(frames, titles=None, grid_size=None):
    """
    Create a grid of visualization frames.
    
    Args:
        frames (list): List of frames to display in the grid
        titles (list, optional): List of titles for each frame
        grid_size (tuple, optional): Grid size as (rows, cols), if None, determined automatically
        
    Returns:
        numpy.ndarray: Combined grid of frames
    """
    n = len(frames)
    if n == 0:
        return None
        
    # Determine grid size if not provided
    if grid_size is None:
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
    else:
        rows, cols = grid_size
        
    # Ensure all frames have the same size and type
    h, w = frames[0].shape[:2]
    dtype = frames[0].dtype
    
    # Create empty grid
    grid = np.zeros((h * rows, w * cols, 3), dtype=dtype)
    
    # Place frames in the grid
    for i, frame in enumerate(frames):
        if i >= rows * cols:
            break
            
        r, c = i // cols, i % cols
        
        # Convert grayscale to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # Resize frame if needed
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
            
        # Add title if provided
        if titles is not None and i < len(titles):
            title_frame = frame.copy()
            cv2.putText(title_frame, titles[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2)
            frame = title_frame
            
        # Copy frame to grid
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
        
    return grid

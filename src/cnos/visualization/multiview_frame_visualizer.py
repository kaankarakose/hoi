#!/usr/bin/env python3

import os
import glob
import numpy as np
import json
import base64
import cv2
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import sys
import io
from PIL import Image

# Add the parent directory to the path to import needed modules
sys.path.append(str(Path(__file__).parent.parent))


def rle2mask(rle):
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def load_rle_mask(rle_file_path):
    """
    Load an RLE mask from a file.
    
    Args:
        rle_file_path: Path to the RLE file
    
    Returns:
        numpy array: Binary mask or None if there's an error
    """
    try:
        with open(rle_file_path, 'r') as f:
            rle_data = json.load(f)
        
        # Validate RLE data
        if not isinstance(rle_data, dict) or 'size' not in rle_data or 'counts' not in rle_data:
            print(f"Invalid RLE format in {rle_file_path}")
            return None
            
        return rle2mask(rle_data)
    except Exception as e:
        print(f"Error loading mask from {rle_file_path}: {str(e)}")
        return None


class MultiViewFrameVisualizer:
    """
    Web-based visualizer for multiview frames and masks.
    Provides a Dash web application to interactively visualize frames and masks from multiple camera views.
    """
    def __init__(self, 
                 results_root_dir,
                 frames_root_dir,
                 port=8051,
                 debug=False):
        """
        Initialize the multiview frame visualizer.
        
        Args:
            results_root_dir (str): Root directory containing results organized by
                                   session -> camera_view -> object_name -> masks, scores, etc.
            frames_root_dir (str): Root directory containing original frames.
            port (int): Port to run the Dash server on
            debug (bool): Whether to run Dash in debug mode
        """
        self.results_root_dir = results_root_dir
        self.frames_root_dir = frames_root_dir
        self.port = port
        self.debug = debug
        
        # Camera views we're interested in
        self.camera_views = ['cam_top', 'cam_side_r'] #noleft camera
         
        # Frame types for each camera view (R_frames and L_frames)
        self.frame_types = ['R_frames', 'L_frames']
        
        # Store the loaded data
        self.data = {}
        self.sessions = []
        self.objects = {}  # session -> object names
        self.frames = {}   # session -> camera_view -> frame_type -> frame paths
        
        # Load data
        self._load_data()
        
        # Generate color palette for masks
        self.colors = self._generate_color_palette(30)
        
        # Initialize the Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
    
    def _generate_color_palette(self, num_colors=30):
        """Generate a colorful palette for mask visualization."""
        np.random.seed(42)  # For reproducibility
        colors = []
        for i in range(num_colors):
            # Avoid too dark or too light colors
            color = np.random.randint(50, 220, size=3).tolist()
            colors.append(color)
        return colors
    
    def _load_data(self):
        """Load data from results directory and frames directory."""
        # Get all session directories
        session_dirs = [d for d in glob.glob(os.path.join(self.results_root_dir, "*")) 
                       if os.path.isdir(d)]
        
        print(f"Found {len(session_dirs)} sessions in {self.results_root_dir}")
        print(f"Session directories: {session_dirs}")
        
        for session_dir in session_dirs:
            session_name = os.path.basename(session_dir)
            self.sessions.append(session_name)
            self.data[session_name] = {}
            self.objects[session_name] = []
            self.frames[session_name] = {}
            
            print(f"Processing session: {session_name}")
            
            # Process each camera view
            for cam_view in self.camera_views:
                cam_view_dir = os.path.join(session_dir, cam_view)
                
                if not os.path.exists(cam_view_dir):
                    print(f"  Camera view not found in results: {cam_view}")
                    continue
                
                print(f"  Processing camera view: {cam_view}")
                self.data[session_name][cam_view] = {}
                self.frames[session_name][cam_view] = {}
                
                # Process each frame type (R_frames, L_frames)
                for frame_type in self.frame_types:
                    frame_type_dir = os.path.join(cam_view_dir, frame_type)
                    frames_dir = os.path.join(self.frames_root_dir, session_name, cam_view, frame_type)
                    
                    if not os.path.exists(frame_type_dir):
                        print(f"    Frame type not found in results: {frame_type}")
                        continue
                    
                    # Debug: Check frame directory structure
                    if not os.path.exists(frames_dir):
                        # Try alternative path formats
                        alt_frames_dir = os.path.join(self.frames_root_dir, session_name, cam_view)
                        if os.path.exists(alt_frames_dir):
                            frames_dir = alt_frames_dir
                        else:
                            print(f"    Frame type not found in frames: {frame_type}")
                            print(f"    Tried: {frames_dir} and {alt_frames_dir}")
                            continue
                    
                    print(f"    Processing frame type: {frame_type}")
                    self.data[session_name][cam_view][frame_type] = {}
                    
                    # Check different frame filename patterns
                    frame_patterns = ["*.jpg", "*.png", "frame_*.jpg", "frame_*.png"]
                    frame_files = []
                    
                    for pattern in frame_patterns:
                        found_frames = glob.glob(os.path.join(frames_dir, pattern))
                        if found_frames:
                            frame_files.extend(found_frames)
                            print(f"    Found {len(found_frames)} frames with pattern {pattern}")
                    
                    self.frames[session_name][cam_view][frame_type] = sorted(frame_files)
                    
                    print(f"    Found {len(self.frames[session_name][cam_view][frame_type])} frames")
                    
                    # Process each object
                    object_dirs = [d for d in glob.glob(os.path.join(frame_type_dir, "*")) 
                                  if os.path.isdir(d)]
                    
                    print(f"    Found {len(object_dirs)} objects")
                    print(f"    Object directories: {object_dirs}")
                    
                    for object_dir in object_dirs:
                        object_name = os.path.basename(object_dir)
                        if object_name not in self.objects[session_name]:
                            self.objects[session_name].append(object_name)
                        
                        print(f"      Processing object: {object_name}")
                        
                        # Store object directory path for later mask loading
                        self.data[session_name][cam_view][frame_type][object_name] = object_dir
            
            # Sort objects alphabetically
            self.objects[session_name].sort()
    
    def get_mask_files(self, session, cam_view, frame_type, object_name, frame_idx):
        """Get mask files for a specific frame."""
        # First, try to find the object directory in the current session
        try:
            object_dir = self.data[session][cam_view][frame_type][object_name]
        except KeyError:
            print(f"Warning: Object {object_name} not found in {session}/{cam_view}/{frame_type}")
            return []
        
        # Determine frame number from filename
        try:
            frame_path = self.frames[session][cam_view][frame_type][frame_idx]
            frame_filename = os.path.basename(frame_path)
        except (KeyError, IndexError):
            # Handle case where frame index is out of range
            print(f"Warning: Frame index {frame_idx} out of range in {session}/{cam_view}/{frame_type}")
            return []
        
        # Extract frame number from filename (handles 'frame_XXXX.jpg' or 'XXXX.jpg')
        base_name = os.path.splitext(frame_filename)[0]
        if base_name.startswith('frame_'):
            frame_number = int(base_name.split('_')[1])
        else:
            # Try to convert the whole filename to an integer
            try:
                frame_number = int(base_name)
            except ValueError:
                # If conversion fails, just use the frame index
                frame_number = frame_idx + 1
        
        # Format with leading zeros (e.g., frame_0001)
        frame_name = f"frame_{frame_number:06d}"
        
        # Look for masks directory
        masks_dir = os.path.join(object_dir, "masks", frame_name)
        print(f"Looking for masks in: {masks_dir}")
        
        # If not found, try without the frame_ prefix
    
        if masks_dir and os.path.exists(masks_dir):
            # First try the expected pattern: mask_0.rle, mask_1.rle, etc.
            mask_pattern = os.path.join(masks_dir, "mask_*.rle")
            print(f"Searching for mask files with pattern: {mask_pattern}")
            mask_files = sorted(glob.glob(mask_pattern))
            print(f"Found {len(mask_files)} mask files")
            
            # If no files found, try alternate pattern without the 'mask_' prefix
            if not mask_files:
                alt_pattern = os.path.join(masks_dir, "*.rle")
                print(f"No mask_*.rle files found, trying pattern: {alt_pattern}")
                mask_files = sorted(glob.glob(alt_pattern))
                print(f"Found {len(mask_files)} files with alternate pattern")
            
            return mask_files
      
        
        return []
    
    def get_scores(self, session, cam_view, frame_type, object_name, frame_idx):
        """Get scores for a specific frame."""
        scores = {}
        try:
            object_dir = self.data[session][cam_view][frame_type][object_name]
        except KeyError:
            # Handle case where object doesn't exist in this frame type
            print(f"Warning: Object {object_name} not found in {session}/{cam_view}/{frame_type}")
            return {}
        
        # Determine frame number from filename
        try:
            frame_path = self.frames[session][cam_view][frame_type][frame_idx]
            frame_filename = os.path.basename(frame_path)
        except (KeyError, IndexError):
            # Handle case where frame index is out of range
            print(f"Warning: Frame index {frame_idx} out of range in {session}/{cam_view}/{frame_type}")
            return {}
        
        # Extract frame number from filename (handles 'frame_XXXX.jpg' or 'XXXX.jpg')
        base_name = os.path.splitext(frame_filename)[0]
        if base_name.startswith('frame_'):
            frame_number = int(base_name.split('_')[1])
        else:
            # Try to convert the whole filename to an integer
            try:
                frame_number = int(base_name)
            except ValueError:
                # If conversion fails, just use the frame index
                frame_number = frame_idx + 1
        
        # Format with leading zeros (e.g., frame_0001)
        frame_name = f"frame_{frame_number:06d}"
        
        # Look for scores directory
        scores_dir = os.path.join(object_dir, "scores", frame_name)
        if self.debug:
            print(f"Looking for scores in: {scores_dir}")
        
        # If not found, try without the frame_ prefix
        if not os.path.exists(scores_dir):
            # Try alternative paths
            alt_scores_dir = os.path.join(object_dir, "scores", str(frame_number))
            if os.path.exists(alt_scores_dir):
                scores_dir = alt_scores_dir
                if self.debug:
                    print(f"Using alternative scores directory: {scores_dir}")
        
        # Find all score files
        if os.path.exists(scores_dir):
            # Try looking for individual score files first (scores_X.txt)
            score_pattern = os.path.join(scores_dir, "score_*.txt")
            score_files = sorted(glob.glob(score_pattern))
            
            if self.debug:
                print(f"Searching for score files with pattern: {score_pattern}")
                print(f"Found {len(score_files)} score files")
            
            if score_files:
                # Individual score files
                try:
                    for i, score_file in enumerate(score_files):
                        with open(score_file, 'r') as f:
                            score = float(f.read().strip())
                            # Extract the mask index from the filename
                            # Assuming filename format is "score_X.txt" where X is the mask index
                            mask_idx = os.path.basename(score_file).split('_')[1].split('.')[0]
                            scores[f"mask_{mask_idx}"] = score
                    return scores
                except Exception as e:
                    print(f"Error reading individual score files: {str(e)}")
    
        
        return {}
    
    def visualize_frame(self, session, cam_view, frame_type, object_name, frame_idx, threshold=0.0, alpha=0.5):
        """Visualize a frame with mask overlay, highlighting the highest scoring mask."""
        if (session not in self.frames or 
            cam_view not in self.frames[session] or
            frame_type not in self.frames[session][cam_view] or
            frame_idx >= len(self.frames[session][cam_view][frame_type])):
            return None
        
        try:
            # Load original frame
            frame_path = self.frames[session][cam_view][frame_type][frame_idx]
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"ERROR: Could not load frame from {frame_path}")
                return None
            
            # Get mask files and scores (these functions now handle their own errors)
            mask_files = self.get_mask_files(session, cam_view, frame_type, object_name, frame_idx)
            score_dict = self.get_scores(session, cam_view, frame_type, object_name, frame_idx)
            
            if self.debug:
                print(f"Found {len(mask_files)} mask files")
                print(f"Found {len(score_dict)} scores")
                print(f"Score dictionary: {score_dict}")
        except Exception as e:
            print(f"Error in visualize_frame: {str(e)}")
            return None
        
        # Create list of masks and their corresponding scores
        masks_with_scores = []
        
        # Match masks with their scores based on filename
        for mask_file in mask_files:
            # Extract mask index from filename (e.g., mask_001.rle -> 001)
            mask_basename = os.path.basename(mask_file)
            mask_idx = mask_basename.split('_')[1].split('.')[0]  # Get the index part
            mask_key = f"mask_{mask_idx}"
            
            # Find corresponding score
            score = score_dict.get(mask_key, 0.0)
            
            # Add to our list
            masks_with_scores.append((mask_file, score))
        
        # If we have no valid masks with scores, return the frame as is
        if not masks_with_scores:
            return frame
        
        # Sort by score (highest first)
        masks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create two overlays: one for regular masks, one for the highest-scoring mask
        mask_overlay = np.zeros_like(frame)
        highest_mask_overlay = np.zeros_like(frame)
        
        # Flag to track if we've drawn the highest-scoring mask
        has_highest_mask = False
        max_score = masks_with_scores[0][1] if masks_with_scores else 0.0
        
        # First, draw all masks above threshold except the highest-scoring one
        for i, (mask_file, score) in enumerate(masks_with_scores[1:], 1):  # Skip the first one (highest)
            if score < threshold:
                continue
            
            # Load mask
            mask = load_rle_mask(mask_file)
            if mask is None or mask.shape != frame.shape[:2]:
                continue
            
            # Convert to 3-channel for overlay
            color = self.colors[i % len(self.colors)]
            
            # Apply color to mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color
            
            # Add to regular overlay
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 1.0, 0)
        
        # Now, draw the highest-scoring mask if it's above threshold
        if masks_with_scores and masks_with_scores[0][1] >= threshold:
            # Load the highest-scoring mask
            high_mask_file, high_score = masks_with_scores[0]
            high_mask = load_rle_mask(high_mask_file)
            
            if high_mask is not None and high_mask.shape == frame.shape[:2]:
                # Use a special color for the highest-scoring mask (bright red)
                high_color = [0, 0, 255]  # BGR format
                
                # Apply color to mask
                high_colored_mask = np.zeros_like(frame)
                high_colored_mask[high_mask > 0] = high_color
                
                # Add to highest mask overlay
                highest_mask_overlay = cv2.addWeighted(highest_mask_overlay, 1.0, high_colored_mask, 1.0, 0)
                
                if self.debug:
                    print(f"Highlighting highest-scoring mask: {high_mask_file} with score {high_score}")
                
                has_highest_mask = True
        
        # Blend frame and overlays
        # First, add the regular masks
        result = cv2.addWeighted(frame, 1.0, mask_overlay, alpha, 0)
        
        # Then, add the highest-scoring mask with higher alpha to make it stand out
        if has_highest_mask:
            # Use a higher alpha for the highest-scoring mask to make it more visible
            highest_alpha = min(alpha + 0.2, 1.0)
            result = cv2.addWeighted(result, 1.0, highest_mask_overlay, highest_alpha, 0)
            
            # # Add text to show the highest score
            # cv2.putText(result, f"Max Score: {max_score:.3f}", (10, 70), 
            #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
        return result
    
    def _setup_layout(self):
        """Set up the Dash app layout."""
        self.app.layout = html.Div([
            html.H1("MultiView Frame & Mask Visualizer", 
                    style={'textAlign': 'center', 'marginBottom': 30}),
            
            html.Div([
                html.Div([
                    html.Label("Session"),
                    dcc.Dropdown(
                        id='session-dropdown',
                        options=[{'label': s, 'value': s} for s in self.sessions],
                        value=self.sessions[0] if self.sessions else None,
                        clearable=False
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label("Object"),
                    dcc.Dropdown(
                        id='object-dropdown',
                        clearable=False
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label("Frame"),
                    dcc.Slider(
                        id='frame-slider',
                        min=0,
                        max=100,  # This will be updated based on selection
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'width': '34%', 'display': 'inline-block', 'marginTop': '10px'}),
            ], style={'width': '90%', 'margin': 'auto', 'marginBottom': 20}),
            
            html.Div([
                html.Div([
                    html.Label("Score Threshold"),
                    dcc.Slider(
                        id='threshold-slider',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.5,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'width': '45%', 'display': 'inline-block', 'marginRight': '5%'}),
                
                html.Div([
                    html.Label("Mask Opacity"),
                    dcc.Slider(
                        id='alpha-slider',
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        value=0.7,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
            ], style={'width': '90%', 'margin': 'auto', 'marginBottom': 20}),
            
            # Frame visualizations - SMALLER VERSION
            html.Div([
                # Top Camera - Left and Right frames in columns
                html.Div([
                    html.H4("Top Camera", style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '8px'}),
                    html.Div([
                        html.Div([
                            html.H5("R Frame", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Img(id='cam-top-R_frames-image', style={'width': '100%', 'maxHeight': '180px', 'border': '1px solid #ddd', 'objectFit': 'contain'})
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%', 'verticalAlign': 'top'}),
                        html.Div([
                            html.H5("L Frame", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Img(id='cam-top-L_frames-image', style={'width': '100%', 'maxHeight': '180px', 'border': '1px solid #ddd', 'objectFit': 'contain'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    ])
                ], style={'width': '100%', 'marginBottom': '15px', 'verticalAlign': 'top'}),
                
                # Side Right Camera - Left and Right frames in columns
                html.Div([
                    html.H4("Side Right Camera", style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '8px'}),
                    html.Div([
                        html.Div([
                            html.H5("R Frame", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Img(id='cam-side-r-R_frames-image', style={'width': '100%', 'maxHeight': '180px', 'border': '1px solid #ddd', 'objectFit': 'contain'})
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%', 'verticalAlign': 'top'}),
                        html.Div([
                            html.H5("L Frame", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Img(id='cam-side-r-L_frames-image', style={'width': '100%', 'maxHeight': '180px', 'border': '1px solid #ddd', 'objectFit': 'contain'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    ])
                ], style={'width': '100%', 'marginBottom': '15px', 'verticalAlign': 'top'}),
                
                # Side Left Camera - Left and Right frames in columns
                html.Div([
                    html.H4("Side Left Camera", style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '8px'}),
                    html.Div([
                        html.Div([
                            html.H5("R Frame", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Img(id='cam-side-l-R_frames-image', style={'width': '100%', 'maxHeight': '180px', 'border': '1px solid #ddd', 'objectFit': 'contain'})
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%', 'verticalAlign': 'top'}),
                        html.Div([
                            html.H5("L Frame", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Img(id='cam-side-l-L_frames-image', style={'width': '100%', 'maxHeight': '180px', 'border': '1px solid #ddd', 'objectFit': 'contain'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    ])
                ], style={'width': '100%', 'marginBottom': '15px', 'verticalAlign': 'top'}),
            ], style={'width': '70%', 'margin': 'auto', 'marginTop': 10}),
            
            # Frame information
            html.Div([
                html.H4("Frame Information", style={'textAlign': 'center', 'marginTop': 20, 'fontSize': '16px'}),
                html.Div(id='frame-info', style={'textAlign': 'left', 'maxWidth': '100%', 'margin': 'auto', 'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px', 'fontFamily': 'Arial', 'fontSize': '14px'})
            ], style={'width': '70%', 'margin': 'auto', 'marginTop': 10})
        ])
        
    def _setup_callbacks(self):
        """Set up the Dash app callbacks."""
        @self.app.callback(
            Output('object-dropdown', 'options'),
            Output('object-dropdown', 'value'),
            Input('session-dropdown', 'value')
        )
        def update_object_dropdown(session):
            if not session or session not in self.objects:
                return [], None
            
            options = [{'label': obj, 'value': obj} for obj in self.objects[session]]
            value = self.objects[session][0] if self.objects[session] else None
            
            return options, value
        
        @self.app.callback(
            Output('frame-slider', 'max'),
            Output('frame-slider', 'value'),
            [Input('session-dropdown', 'value'),
             Input('object-dropdown', 'value')]
        )
        def update_frame_slider(session, object_name):
            if not session or session not in self.frames:
                return 0, 0
            
            # Reset to frame 0 when changing sessions or objects
            # This ensures we always start at a valid frame
            
            # Find max number of frames across all camera views and frame types
            max_frames = 0
            for cam_view in self.camera_views:
                if cam_view in self.frames[session]:
                    for frame_type in self.frame_types:
                        if frame_type in self.frames[session][cam_view]:
                            max_frames = max(max_frames, len(self.frames[session][cam_view][frame_type]))
            
            return max_frames - 1, 0  # Zero-indexed, always reset to first frame
        
        @self.app.callback(
            [Output('cam-top-R_frames-image', 'src'),
             Output('cam-top-L_frames-image', 'src'),
             Output('cam-side-r-R_frames-image', 'src'),
             Output('cam-side-r-L_frames-image', 'src'),
             Output('cam-side-l-R_frames-image', 'src'),
             Output('cam-side-l-L_frames-image', 'src'),
             Output('frame-info', 'children')],
            [Input('session-dropdown', 'value'),
             Input('object-dropdown', 'value'),
             Input('frame-slider', 'value'),
             Input('threshold-slider', 'value'),
             Input('alpha-slider', 'value')]
        )
        def update_visualizations(session, object_name, frame_idx, threshold, alpha):
            # Define empty_src at the beginning to avoid reference errors
            empty_src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
            
            if not session or not object_name or frame_idx is None:
                return empty_src, empty_src, empty_src, empty_src, empty_src, empty_src, "No data selected"
            
            images = {}
            frame_info = []
            
            for cam_view in self.camera_views:
                if cam_view not in self.frames.get(session, {}):
                    for frame_type in self.frame_types:
                        images[f"{cam_view}-{frame_type}"] = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
                    continue
                
                for frame_type in self.frame_types:
                    if (frame_type not in self.frames[session][cam_view] or
                        frame_idx >= len(self.frames[session][cam_view][frame_type])):
                        images[f"{cam_view}-{frame_type}"] = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
                        continue
                    
                    # Visualize frame with mask overlay
                    result = self.visualize_frame(session, cam_view, frame_type, object_name, frame_idx, threshold, alpha)
                
                    if result is None:
                        images[f"{cam_view}-{frame_type}"] = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
                        continue
                    
                    # Convert to RGB for PIL
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    
                    # Convert to base64 image
                    buffer = io.BytesIO()
                    Image.fromarray(result_rgb).save(buffer, format='JPEG')
                    images[f"{cam_view}-{frame_type}"] = f'data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode("utf-8")}'
            
            # Add general frame information at the top
            frame_info.insert(0, html.H3(f"Debug Information for Frame {frame_idx+1}", 
                                        style={'textAlign': 'center', 'marginBottom': '20px', 'marginTop': '10px'}))
            
            # Get mask and score information for each camera view and frame type
            for cam_view in self.camera_views:
                # Skip camera views without data
                if cam_view not in self.frames.get(session, {}):
                    continue
                
                for frame_type in self.frame_types:
                    # Skip frame types without data
                    if (frame_type not in self.frames[session][cam_view] or
                        frame_idx >= len(self.frames[session][cam_view][frame_type])):
                        continue
                    
                    # Get mask files and scores
                    mask_files = self.get_mask_files(session, cam_view, frame_type, object_name, frame_idx)
                    scores = self.get_scores(session, cam_view, frame_type, object_name, frame_idx)
                
                    # Add detailed debug information
                    frame_info.append(html.H4(f"{cam_view} - {frame_type} Debug Info:", style={'marginTop': '15px', 'marginBottom': '5px'}))
                
                    # Show frame path
                    frame_path_info = "Frame path: Not found"
                    if (cam_view in self.frames.get(session, {}) and 
                        frame_type in self.frames[session][cam_view] and 
                        frame_idx < len(self.frames[session][cam_view][frame_type])):
                        frame_path_info = f"Frame path: {self.frames[session][cam_view][frame_type][frame_idx]}"
                    frame_info.append(html.P(frame_path_info, style={'fontFamily': 'monospace', 'fontSize': '0.9em'}))
                
                    # Show detected frame number
                    try:
                        frame_path = self.frames[session][cam_view][frame_type][frame_idx]
                        frame_filename = os.path.basename(frame_path)
                        base_name = os.path.splitext(frame_filename)[0]
                        if base_name.startswith('frame_'):
                            frame_number = int(base_name.split('_')[1])
                            frame_info.append(html.P(f"Detected frame number: {frame_number} (from 'frame_XXXX' format)", 
                                                style={'fontFamily': 'monospace', 'fontSize': '0.9em'}))
                        else:
                            try:
                                frame_number = int(base_name)
                                frame_info.append(html.P(f"Detected frame number: {frame_number} (from numeric filename)", 
                                                    style={'fontFamily': 'monospace', 'fontSize': '0.9em'}))
                            except ValueError:
                                frame_info.append(html.P(f"Detected frame number: {frame_idx+1} (from index)", 
                                                    style={'fontFamily': 'monospace', 'fontSize': '0.9em'}))
                    except Exception as e:
                        frame_info.append(html.P(f"Error detecting frame number: {str(e)}", 
                                            style={'fontFamily': 'monospace', 'fontSize': '0.9em', 'color': 'red'}))
                
                # Show masks information
                if mask_files:
                    # Ensure all scores are floats before comparison
                    float_scores = []
                    if scores:
                        for s in scores:
                            try:
                                # Convert any string scores to float
                                float_scores.append(float(s))
                            except (ValueError, TypeError):
                                # Skip or add a default value for invalid scores
                                float_scores.append(0.0)
                    
                    above_threshold = sum(s >= threshold for s in float_scores) if float_scores else 0
                    frame_info.append(html.P(f"Found {len(mask_files)} masks, {above_threshold} above threshold", 
                                           style={'fontFamily': 'monospace', 'fontSize': '0.9em'}))
                    
                    # Show mask paths (up to 3)
                    frame_info.append(html.P("Mask paths (up to 3):", style={'marginBottom': '2px'}))
                    for i, mask_path in enumerate(mask_files[:3]):
                        frame_info.append(html.P(f"  {i+1}. {mask_path}", 
                                               style={'fontFamily': 'monospace', 'fontSize': '0.9em', 'marginLeft': '10px', 'marginTop': '0'}))
                    
                    # Show scores (up to 3)
                    if scores:
                        frame_info.append(html.P("Scores (up to 3):", style={'marginBottom': '2px'}))
                        for i, (score, float_score) in enumerate(zip(scores[:3], float_scores[:3])):
                            color = 'green' if float_score >= threshold else 'red'
                            # Display the original score format but use float for comparison
                            try:
                                # Try to format as float if possible
                                score_display = f"{float_score:.4f}"
                            except:
                                # Otherwise just use the original string
                                score_display = str(score)
                            
                            frame_info.append(html.P(f"  {i+1}. {score_display}", 
                                                   style={'fontFamily': 'monospace', 'fontSize': '0.9em', 'marginLeft': '10px', 'marginTop': '0', 'color': color}))
                else:
                    frame_info.append(html.P("No masks found", style={'fontFamily': 'monospace', 'fontSize': '0.9em', 'color': 'red'}))
            # Return images and frame info
            return (images.get('cam_top-R_frames', empty_src), 
                    images.get('cam_top-L_frames', empty_src),
                    images.get('cam_side_r-R_frames', empty_src), 
                    images.get('cam_side_r-L_frames', empty_src),
                    images.get('cam_side_l-R_frames', empty_src), 
                    images.get('cam_side_l-L_frames', empty_src),
                    frame_info)
    
    def run(self):
        """Run the Dash app."""
        self.app.run_server(port=self.port, debug=self.debug)


def main():
    """Main function to start the visualization app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MultiView Frame & Mask Visualizer")
    parser.add_argument("--results_dir", type=str, 
                       default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/cnos_hand_results",
                       help="Root directory of results (session -> camera_view -> R_frames/L_frames -> object_name)")
    parser.add_argument("--frames_dir", type=str,
                       default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/hand_detections",
                       help="Root directory of original frames")
    parser.add_argument("--port", type=int, default=8051, help="Port for web server")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting MultiView Frame & Mask Visualizer")
    print(f"Results directory: {args.results_dir}")
    print(f"Frames directory: {args.frames_dir}")
    print(f"Port: {args.port}")
    print(f"Debug mode: {args.debug}")
    print(f"Using new folder structure: Session -> Camera View -> R_frames/L_frames -> Object Name")
    
    visualizer = MultiViewFrameVisualizer(
        results_root_dir=args.results_dir,
        frames_root_dir=args.frames_dir,
        port=args.port,
        debug=args.debug
    )
    
    print("Running web server...")
    visualizer.run()


if __name__ == "__main__":
    main()

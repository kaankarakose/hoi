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
        numpy array: Binary mask
    """
    with open(rle_file_path, 'r') as f:
        rle_data = json.load(f)

    return rle2mask(rle_data)


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
        self.camera_views = ['cam_top', 'cam_side_r', 'cam_side_l']
        
        # Store the loaded data
        self.data = {}
        self.sessions = []
        self.objects = {}  # session -> object names
        self.frames = {}   # session -> camera_view -> frame paths
        
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
                frames_dir = os.path.join(self.frames_root_dir, session_name, cam_view)
                
                if not os.path.exists(cam_view_dir):
                    print(f"  Camera view not found in results: {cam_view}")
                    continue
                
                # Debug: Check frame directory structure
                if not os.path.exists(frames_dir):
                    # Try alternative path formats
                    alt_frames_dir = os.path.join(self.frames_root_dir, session_name)
                    if os.path.exists(alt_frames_dir):
                        frames_dir = alt_frames_dir
                    else:
                        print(f"  Camera view not found in frames: {cam_view}")
                        print(f"  Tried: {frames_dir} and {alt_frames_dir}")
                        continue
                
                print(f"  Processing camera view: {cam_view}")
                self.data[session_name][cam_view] = {}
                
                # Check different frame filename patterns
                frame_patterns = ["*.jpg", "*.png", "frame_*.jpg", "frame_*.png"]
                frame_files = []
                
                for pattern in frame_patterns:
                    found_frames = glob.glob(os.path.join(frames_dir, pattern))
                    if found_frames:
                        frame_files.extend(found_frames)
                        print(f"  Found {len(found_frames)} frames with pattern {pattern}")
                
                self.frames[session_name][cam_view] = sorted(frame_files)
                
                print(f"  Found {len(self.frames[session_name][cam_view])} frames")
                
                # Process each object
                object_dirs = [d for d in glob.glob(os.path.join(cam_view_dir, "*")) 
                              if os.path.isdir(d)]
                
                print(f"  Found {len(object_dirs)} objects")
                
                for object_dir in object_dirs:
                    object_name = os.path.basename(object_dir)
                    if object_name not in self.objects[session_name]:
                        self.objects[session_name].append(object_name)
                    
                    print(f"    Processing object: {object_name}")
                    
                    # Store object directory path for later mask loading
                    self.data[session_name][cam_view][object_name] = object_dir
            
            # Sort objects alphabetically
            self.objects[session_name].sort()
    
    def get_mask_files(self, session, cam_view, object_name, frame_idx):
        """Get mask files for a specific frame."""
        object_dir = self.data[session][cam_view][object_name]
        
        # Determine frame number from filename
        frame_path = self.frames[session][cam_view][frame_idx]
        frame_filename = os.path.basename(frame_path)
        
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
        frame_name = f"frame_{frame_number:04d}"
        
        # Look for masks directory
        masks_dir = os.path.join(object_dir, "masks", frame_name)
        
        # If not found, try without the frame_ prefix
        if not os.path.exists(masks_dir):
            # Try alternative paths
            alt_masks_dir = os.path.join(object_dir, "masks", str(frame_number))
            if os.path.exists(alt_masks_dir):
                masks_dir = alt_masks_dir
        
        # Find all mask files
        if os.path.exists(masks_dir):
            mask_files = sorted(glob.glob(os.path.join(masks_dir, "mask_*.rle")))
            return mask_files
        
        return []
    
    def get_scores(self, session, cam_view, object_name, frame_idx):
        """Get scores for a specific frame."""
        object_dir = self.data[session][cam_view][object_name]
        
        # Determine frame number from filename
        frame_path = self.frames[session][cam_view][frame_idx]
        frame_filename = os.path.basename(frame_path)
        
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
        frame_name = f"frame_{frame_number:04d}"
        
        # Look for scores directory
        scores_dir = os.path.join(object_dir, "scores", frame_name)
        
        # If not found, try without the frame_ prefix
        if not os.path.exists(scores_dir):
            # Try alternative paths
            alt_scores_dir = os.path.join(object_dir, "scores", str(frame_number))
            if os.path.exists(alt_scores_dir):
                scores_dir = alt_scores_dir
        
        # Find all score files
        scores = []
        if os.path.exists(scores_dir):
            score_files = sorted(glob.glob(os.path.join(scores_dir, "score_*.txt")))
            
            for score_file in score_files:
                try:
                    with open(score_file, 'r') as f:
                        score = float(f.read().strip())
                        scores.append(score)
                except:
                    scores.append(0.0)
        
        return scores
    
    def visualize_frame(self, session, cam_view, object_name, frame_idx, threshold=0.0, alpha=0.5):
        """Visualize a frame with mask overlay, highlighting the highest scoring mask."""
        if (session not in self.frames or 
            cam_view not in self.frames[session] or 
            frame_idx >= len(self.frames[session][cam_view])):
            return None
        
        # Get frame path
        frame_path = self.frames[session][cam_view][frame_idx]
        
        # Read frame
        frame = cv2.imread(frame_path)
        if frame is None:
            return None
        
        # Get mask files and scores
        mask_files = self.get_mask_files(session, cam_view, object_name, frame_idx)
        scores = self.get_scores(session, cam_view, object_name, frame_idx)
        
        # If we don't have scores, assume all masks are above threshold
        if not scores and mask_files:
            scores = [1.0] * len(mask_files)
        
        # Make sure we have the same number of masks and scores
        mask_files = mask_files[:len(scores)]
        
        # Find max score index
        max_score_idx = -1
        max_score = -1.0
        
        if scores:
            max_score_idx = np.argmax(scores)
            max_score = scores[max_score_idx]
        
        # Create two overlays: one for regular masks, one for the highest-scoring mask
        mask_overlay = np.zeros_like(frame)
        highest_mask_overlay = np.zeros_like(frame)
        
        # Flag to track if we've drawn the highest-scoring mask
        has_highest_mask = False
        
        # First, draw all masks above threshold except the highest-scoring one
        for i, (mask_file, score) in enumerate(zip(mask_files, scores)):
            if score < threshold:
                continue
                
            # Skip the highest-scoring mask for now
            if i == max_score_idx:
                continue
            
            # Load mask
            mask = load_rle_mask(mask_file)
            
            # Convert to 3-channel for overlay
            color = self.colors[i % len(self.colors)]
            
            # Apply color to mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color
            
            # Add to regular overlay
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 1.0, 0)
        
        # Now, draw the highest-scoring mask if it's above threshold
        if max_score_idx >= 0 and max_score >= threshold:
            # Load the highest-scoring mask
            high_mask_file = mask_files[max_score_idx]
            high_mask = load_rle_mask(high_mask_file)
            
            # Use a special color for the highest-scoring mask (bright red)
            high_color = [0, 0, 255]  # BGR format
            
            # Apply color to mask
            high_colored_mask = np.zeros_like(frame)
            high_colored_mask[high_mask > 0] = high_color
            
            # Add to highest mask overlay
            highest_mask_overlay = cv2.addWeighted(highest_mask_overlay, 1.0, high_colored_mask, 1.0, 0)
            
            has_highest_mask = True
        
        # Blend frame and overlays
        # First, add the regular masks
        result = cv2.addWeighted(frame, 1.0, mask_overlay, alpha, 0)
        
        # Then, add the highest-scoring mask with higher alpha to make it stand out
        if has_highest_mask:
            # Use a higher alpha for the highest-scoring mask to make it more visible
            highest_alpha = min(alpha + 0.2, 1.0)
            result = cv2.addWeighted(result, 1.0, highest_mask_overlay, highest_alpha, 0)
            
            # Add text to show the highest score
            cv2.putText(result, f"Max Score: {max_score:.3f}", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.3, (0, 0, 255), 3)
                    
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
            
            # Frame visualizations
            html.Div([
                html.Div([
                    html.H3("Top Camera", style={'textAlign': 'center'}),
                    html.Img(id='cam-top-image', style={'width': '100%', 'border': '1px solid #ddd'})
                ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H3("Side Right Camera", style={'textAlign': 'center'}),
                    html.Img(id='cam-side-r-image', style={'width': '100%', 'border': '1px solid #ddd'})
                ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H3("Side Left Camera", style={'textAlign': 'center'}),
                    html.Img(id='cam-side-l-image', style={'width': '100%', 'border': '1px solid #ddd'})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'width': '90%', 'margin': 'auto', 'marginTop': 20}),
            
            # Frame information
            html.Div([
                html.H3("Frame Information", style={'textAlign': 'center', 'marginTop': 30}),
                html.Div(id='frame-info', style={'textAlign': 'left', 'maxWidth': '90%', 'margin': 'auto', 'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'fontFamily': 'Arial'})
            ], style={'width': '90%', 'margin': 'auto', 'marginTop': 20})
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
            [Input('session-dropdown', 'value')]
        )
        def update_frame_slider(session):
            if not session or session not in self.frames:
                return 0, 0
            
            # Find max number of frames across all camera views
            max_frames = 0
            for cam_view in self.camera_views:
                if cam_view in self.frames[session]:
                    max_frames = max(max_frames, len(self.frames[session][cam_view]))
            
            return max_frames - 1, 0  # Zero-indexed
        
        @self.app.callback(
            [Output('cam-top-image', 'src'),
             Output('cam-side-r-image', 'src'),
             Output('cam-side-l-image', 'src'),
             Output('frame-info', 'children')],
            [Input('session-dropdown', 'value'),
             Input('object-dropdown', 'value'),
             Input('frame-slider', 'value'),
             Input('threshold-slider', 'value'),
             Input('alpha-slider', 'value')]
        )
        def update_visualizations(session, object_name, frame_idx, threshold, alpha):
            if not session or not object_name or frame_idx is None:
                empty_src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
                return empty_src, empty_src, empty_src, "No data selected"
            
            images = {}
            frame_info = []
            
            for cam_view in self.camera_views:
                if (cam_view not in self.frames.get(session, {}) or 
                    frame_idx >= len(self.frames[session][cam_view])):
                    images[cam_view] = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
                    continue
                
                # Visualize frame with mask overlay
                result = self.visualize_frame(session, cam_view, object_name, frame_idx, threshold, alpha)
                
                if result is None:
                    images[cam_view] = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
                    continue
                
                # Convert to RGB for PIL
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                
                # Convert to base64 image
                buffer = io.BytesIO()
                Image.fromarray(result_rgb).save(buffer, format='JPEG')
                images[cam_view] = f'data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode("utf-8")}'
            
            # Add general frame information at the top
            frame_info.insert(0, html.H3(f"Debug Information for Frame {frame_idx+1}", 
                                        style={'textAlign': 'center', 'marginBottom': '20px', 'marginTop': '10px'}))
            
            # Get mask and score information for each camera view
            for cam_view in self.camera_views:
                # Skip camera views without data
                if (cam_view not in self.frames.get(session, {}) or 
                    frame_idx >= len(self.frames[session][cam_view])):
                    continue
                    
                # Get mask files and scores
                mask_files = self.get_mask_files(session, cam_view, object_name, frame_idx)
                scores = self.get_scores(session, cam_view, object_name, frame_idx)
                
                # Add detailed debug information
                frame_info.append(html.H4(f"{cam_view} Debug Info:", style={'marginTop': '15px', 'marginBottom': '5px'}))
                
                # Show frame path
                frame_path_info = "Frame path: Not found"
                if cam_view in self.frames.get(session, {}) and frame_idx < len(self.frames[session][cam_view]):
                    frame_path_info = f"Frame path: {self.frames[session][cam_view][frame_idx]}"
                frame_info.append(html.P(frame_path_info, style={'fontFamily': 'monospace', 'fontSize': '0.9em'}))
                
                # Show detected frame number
                try:
                    frame_path = self.frames[session][cam_view][frame_idx]
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
                            frame_info.append(html.P(f"Could not determine frame number from: {base_name}", 
                                                   style={'fontFamily': 'monospace', 'fontSize': '0.9em', 'color': 'red'}))
                except Exception as e:
                    frame_info.append(html.P(f"Error processing frame number: {str(e)}", 
                                           style={'fontFamily': 'monospace', 'fontSize': '0.9em', 'color': 'red'}))
                
                # Show masks information
                if mask_files:
                    above_threshold = sum(s >= threshold for s in scores) if scores else 0
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
                        for i, score in enumerate(scores[:3]):
                            color = 'green' if score >= threshold else 'red'
                            frame_info.append(html.P(f"  {i+1}. {score:.4f}", 
                                                   style={'fontFamily': 'monospace', 'fontSize': '0.9em', 'marginLeft': '10px', 'marginTop': '0', 'color': color}))
                else:
                    frame_info.append(html.P("No masks found", style={'fontFamily': 'monospace', 'fontSize': '0.9em', 'color': 'red'}))
            
            return images.get('cam_top', ''), images.get('cam_side_r', ''), images.get('cam_side_l', ''), frame_info
    
    def run(self):
        """Run the Dash app."""
        self.app.run_server(port=self.port, debug=self.debug)


def main():
    """Main function to start the visualization app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MultiView Frame & Mask Visualizer")
    parser.add_argument("--results_dir", type=str, 
                       default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/cnos_results",
                       help="Root directory of results (session -> camera_view -> object_name)")
    parser.add_argument("--frames_dir", type=str,
                       default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/orginal_frames",
                       help="Root directory of original frames")
    parser.add_argument("--port", type=int, default=8051, help="Port for web server")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting MultiView Frame & Mask Visualizer")
    print(f"Results directory: {args.results_dir}")
    print(f"Frames directory: {args.frames_dir}")
    print(f"Port: {args.port}")
    print(f"Debug mode: {args.debug}")
    
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

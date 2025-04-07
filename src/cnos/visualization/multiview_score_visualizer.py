#!/usr/bin/env python3

import os
import glob
import numpy as np
import json
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import sys

# Add the parent directory to the path to import needed modules
sys.path.append(str(Path(__file__).parent.parent))

class MultiViewScoreVisualizer:
    """
    Web-based visualizer for multiview object detection scores.
    Provides a Dash web application to interactively visualize score data from multiple camera views.
    """
    def __init__(self, 
                 results_root_dir,
                 port=8050,
                 debug=False):
        """
        Initialize the multiview score visualizer.
        
        Args:
            results_root_dir (str): Root directory containing results organized by
                                   session -> camera_view -> object_name -> masks, scores, etc.
            port (int): Port to run the Dash server on
            debug (bool): Whether to run Dash in debug mode
        """
        self.results_root_dir = results_root_dir
        self.port = port
        self.debug = debug
        
        # Camera views we're interested in
        self.camera_views = ['cam_top', 'cam_side_r', 'cam_side_l']
        
        # Store the loaded data
        self.data = {}
        self.sessions = []
        self.objects = {}  # session -> object names
        
        # Load data
        self._load_data()
        
        # Initialize the Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
    
    def _load_data(self):
        """Load data from results directory."""
        # Get all session directories
        session_dirs = [d for d in glob.glob(os.path.join(self.results_root_dir, "*")) 
                       if os.path.isdir(d)]
        
        print(f"Found {len(session_dirs)} sessions in {self.results_root_dir}")
        
        for session_dir in session_dirs:
            session_name = os.path.basename(session_dir)
            self.sessions.append(session_name)
            self.data[session_name] = {}
            self.objects[session_name] = []
            
            print(f"Processing session: {session_name}")
            
            # Process each camera view
            for cam_view in self.camera_views:
                cam_view_dir = os.path.join(session_dir, cam_view)
                if not os.path.exists(cam_view_dir):
                    print(f"  Camera view not found: {cam_view}")
                    continue
                
                print(f"  Processing camera view: {cam_view}")
                self.data[session_name][cam_view] = {}
                
                # Process each object
                object_dirs = [d for d in glob.glob(os.path.join(cam_view_dir, "*")) 
                              if os.path.isdir(d)]
                
                print(f"  Found {len(object_dirs)} objects")
                
                for object_dir in object_dirs:
                    object_name = os.path.basename(object_dir)
                    if object_name not in self.objects[session_name]:
                        self.objects[session_name].append(object_name)
                    
                    print(f"    Processing object: {object_name}")
                    
                    # Collect frame score files
                    scores_data = self._collect_frame_scores(object_dir)
                    
                    # Store scores data
                    self.data[session_name][cam_view][object_name] = scores_data
                    
                    print(f"    Found {len(scores_data)} frames with scores")
            
            # Sort objects alphabetically
            self.objects[session_name].sort()
    
    def _collect_frame_scores(self, object_dir):
        """Collect scores from individual frame files."""
        scores_data = {}
        scores_dir = os.path.join(object_dir, "scores")
        
        if not os.path.exists(scores_dir):
            return scores_data
        
        # Find all frame directories
        frame_dirs = sorted(glob.glob(os.path.join(scores_dir, "frame_*")))
        
        for frame_dir in frame_dirs:
            frame_name = os.path.basename(frame_dir)
            # Extract frame number - convert from format like 'frame_0001' to int
            frame_num = int(frame_name.split('_')[1])
            
            # Find scores in this frame
            score_files = glob.glob(os.path.join(frame_dir, "score_*.txt"))
            if not score_files:
                continue
            
            # Get the highest score
            max_score = 0
            for score_file in score_files:
                with open(score_file, 'r') as f:
                    try:
                        score = float(f.read().strip())
                        max_score = max(max_score, score)
                    except:
                        continue
            
            scores_data[str(frame_num)] = max_score
        
        return scores_data
    
    def _setup_layout(self):
        """Set up the Dash app layout."""
        self.app.layout = html.Div([
            html.H1("MultiView Object Detection Score Visualizer", 
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
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                
                html.Div([
                    html.Label("Object"),
                    dcc.Dropdown(
                        id='object-dropdown',
                        clearable=False
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                
                html.Div([
                    html.Label("Score Threshold"),
                    dcc.Slider(
                        id='threshold-slider',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'width': '30%', 'display': 'inline-block'}),
            ], style={'width': '90%', 'margin': 'auto', 'marginBottom': 30}),
            
            # Add a section for the scores plot
            html.Div([
                dcc.Graph(id='scores-plot', style={'height': '600px'})
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
            Output('scores-plot', 'figure'),
            [Input('session-dropdown', 'value'),
             Input('object-dropdown', 'value'),
             Input('threshold-slider', 'value')]
        )
        def update_plot(session, object_name, threshold):
            if not session or not object_name:
                return go.Figure()
            
            # Create a subplot for each camera view
            fig = make_subplots(rows=3, cols=1, 
                               shared_xaxes=True,
                               subplot_titles=self.camera_views,
                               vertical_spacing=0.1)
            
            # Set common x-axis title
            fig.update_xaxes(title_text="Frame Number", row=3, col=1)
            
            # Track min/max frame numbers to normalize x-axis
            min_frame = float('inf')
            max_frame = 0
            
            # Colors for different camera views
            colors = {
                'cam_top': 'rgb(31, 119, 180)',      # blue
                'cam_side_r': 'rgb(255, 127, 14)',   # orange
                'cam_side_l': 'rgb(44, 160, 44)'     # green
            }
            
            # Process each camera view
            for i, cam_view in enumerate(self.camera_views):
                row = i + 1
                
                # Check if we have data for this view
                if (cam_view not in self.data.get(session, {}) or 
                    object_name not in self.data[session].get(cam_view, {})):
                    # Add empty trace with note about missing data
                    fig.add_annotation(
                        text="No data available",
                        xref="x domain", yref="y domain",
                        x=0.5, y=0.5,
                        showarrow=False,
                        row=row, col=1
                    )
                    continue
                
                # Get scores data
                scores_data = self.data[session][cam_view][object_name]
                if not scores_data:
                    # Add empty trace with note about missing data
                    fig.add_annotation(
                        text="No score data available",
                        xref="x domain", yref="y domain",
                        x=0.5, y=0.5,
                        showarrow=False,
                        row=row, col=1
                    )
                    continue
                
                # Convert to pandas for easier processing
                df = pd.DataFrame([
                    {'frame': int(k), 'score': v} 
                    for k, v in scores_data.items()
                ])
                
                if df.empty:
                    continue
                
                # Sort by frame number
                df = df.sort_values('frame')
                
                # Update min/max frames
                min_frame = min(min_frame, df['frame'].min())
                max_frame = max(max_frame, df['frame'].max())
                
                # Add trace for all scores (with lower opacity)
                fig.add_trace(
                    go.Scatter(
                        x=df['frame'],
                        y=df['score'],
                        mode='lines',
                        name=f"{cam_view} (all)",
                        line=dict(width=1, color=colors[cam_view], dash='dot'),
                        opacity=0.3,
                        showlegend=False
                    ),
                    row=row, col=1
                )
                
                # Filter by threshold
                filtered_df = df[df['score'] >= threshold]
                
                # Add trace for scores above threshold
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['frame'],
                        y=filtered_df['score'],
                        mode='lines+markers',
                        name=f"{cam_view}",
                        line=dict(width=2, color=colors[cam_view]),
                        marker=dict(size=6, color=colors[cam_view])
                    ),
                    row=row, col=1
                )
                
                # Add threshold line
                fig.add_trace(
                    go.Scatter(
                        x=[df['frame'].min(), df['frame'].max()],
                        y=[threshold, threshold],
                        mode='lines',
                        line=dict(color='red', width=1, dash='dash'),
                        name='Threshold' if i == 0 else 'Threshold (hidden)',
                        showlegend=(i == 0)  # Only show in legend once
                    ),
                    row=row, col=1
                )
                
                # Set y-axis label and range
                fig.update_yaxes(
                    title_text="Score",
                    range=[0, 1],
                    row=row, col=1
                )
            
            # Ensure all subplots have same x-axis range if we have data
            if min_frame != float('inf') and max_frame > 0:
                for i in range(len(self.camera_views)):
                    fig.update_xaxes(range=[min_frame, max_frame], row=i+1, col=1)
            
            # Update layout
            fig.update_layout(
                height=600,
                margin=dict(l=50, r=50, t=80, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                title=f"Detection Scores for {object_name} in {session}"
            )
            
            return fig
    
    def run(self):
        """Run the Dash app."""
        self.app.run_server(port=self.port, debug=self.debug)


def main():
    """Main function to start the visualization app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MultiView Object Detection Score Visualizer")
    parser.add_argument("--results_dir", type=str, 
                       default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/cnos_results",
                       help="Root directory of results (session -> camera_view -> object_name)")
    parser.add_argument("--port", type=int, default=8050, help="Port for web server")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting MultiView Object Detection Score Visualizer")
    print(f"Results directory: {args.results_dir}")
    print(f"Port: {args.port}")
    print(f"Debug mode: {args.debug}")
    
    visualizer = MultiViewScoreVisualizer(
        results_root_dir=args.results_dir,
        port=args.port,
        debug=args.debug
    )
    
    print("Running web server...")
    visualizer.run()


if __name__ == "__main__":
    main()

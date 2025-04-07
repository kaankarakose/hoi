#!/usr/bin/env python3

import os
import glob
import numpy as np
import cv2
import json
import base64
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from PIL import Image
import io
import sys
from pathlib import Path

# Add the parent directory to the path to import the MaskScoreVisualizer class
sys.path.append(str(Path(__file__).parent.parent))
from visualization.visual import MaskScoreVisualizer


class WebVisualizer:
    """
    Web-based visualizer for segmentation masks and scores.
    Provides a Dash web application to interactively visualize the data.
    """
    def __init__(self, 
                 masks_root_dir, 
                 scores_root_dir, 
                 frames_dir=None,
                 port=8050,
                 debug=False):
        """
        Initialize the web visualizer.
        
        Args:
            masks_root_dir (str): Root directory containing mask folders
            scores_root_dir (str): Root directory containing score folders
            frames_dir (str, optional): Directory containing original frames
            port (int): Port to run the Dash server on
            debug (bool): Whether to run Dash in debug mode
        """
        self.masks_root_dir = masks_root_dir
        self.scores_root_dir = scores_root_dir
        self.frames_dir = frames_dir
        self.port = port
        self.debug = debug
        
        # Create the underlying visualizer
        self.visualizer = MaskScoreVisualizer(
            masks_root_dir=masks_root_dir,
            scores_root_dir=scores_root_dir,
            frames_dir=frames_dir
        )
        
        # Initialize the Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the Dash app layout."""
        self.app.layout = html.Div([
            html.H1("Segmentation Mask & Score Visualizer", 
                    style={'textAlign': 'center', 'marginBottom': 30}),
            
            html.Div([
                html.Div([
                    html.Label("Frame Selection"),
                    dcc.Slider(
                        id='frame-slider',
                        min=0,
                        max=self.visualizer.num_frames - 1,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'marginBottom': 20}),
                
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
                ], style={'marginBottom': 20}),
                
                html.Div([
                    html.Label("Mask Transparency"),
                    dcc.Slider(
                        id='alpha-slider',
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.7,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'marginBottom': 20}),
            ], style={'width': '80%', 'margin': 'auto', 'marginBottom': 30}),
            
            html.Div([
                html.Div([
                    html.Img(id='visualization-image', style={'width': '100%'}),
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H3("Frame Information"),
                    html.Div(id='frame-info'),
                    
                    html.H3("Mask Details", style={'marginTop': 20}),
                    html.Div(id='mask-info')
                ], style={'width': '25%', 'display': 'inline-block', 'marginLeft': '5%', 'verticalAlign': 'top'})
            ], style={'width': '90%', 'margin': 'auto'}),
            
            # Add a section for score distribution
            html.Div([
                html.H2("Score Distribution", style={'marginTop': 30, 'textAlign': 'center'}),
                dcc.Graph(id='score-distribution')
            ], style={'width': '80%', 'margin': 'auto', 'marginTop': 30})
        ])
    
    def _setup_callbacks(self):
        """Set up the Dash app callbacks."""
        @self.app.callback(
            [Output('visualization-image', 'src'),
             Output('frame-info', 'children'),
             Output('mask-info', 'children')],
            [Input('frame-slider', 'value'),
             Input('threshold-slider', 'value'),
             Input('alpha-slider', 'value')]
        )
        def update_visualization(frame_idx, threshold, alpha):
            # Generate visualization
            vis_img = self.visualizer.visualize_frame(
                frame_idx, threshold, alpha, show_scores=True
            )
            
            # Convert to base64 image
            image_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            image_pil.save(buffer, format='JPEG')
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Get data for this frame
            data = self.visualizer.load_mask_data(frame_idx)
            masks = data["masks"]
            scores = data["scores"]
            frame_name = data["frame_name"]
            
            # Generate frame info
            frame_info = [
                html.P(f"Frame: {frame_name}"),
                html.P(f"Total masks: {len(masks)}"),
                html.P(f"Masks above threshold: {sum(s >= threshold for s in scores)}")
            ]
            
            # Generate mask info
            valid_masks = [(i, s) for i, s in enumerate(scores) if s >= threshold]
            valid_masks.sort(key=lambda x: x[1], reverse=True)  # Sort by score
            
            mask_info = []
            for i, (mask_idx, score) in enumerate(valid_masks[:100]):  # Show top 10
                mask_info.append(html.Div([
                    html.P(f"Mask {mask_idx}: Score = {score:.4f}", 
                           style={'marginBottom': 5, 'fontWeight': 'bold'}),
                ]))
            
            if not mask_info:
                mask_info = [html.P("No masks above threshold")]
            
            return f'data:image/jpeg;base64,{encoded_image}', frame_info, mask_info
        
        @self.app.callback(
            Output('score-distribution', 'figure'),
            [Input('threshold-slider', 'value')]
        )
        def update_score_distribution(threshold):
            # Collect frame-wise scores
            frame_scores = {}
            all_scores = []
            frame_dirs = sorted(glob.glob(os.path.join(self.scores_root_dir, "frame_*")))
            
            # Limit to first 100 frames for performance
            for frame_dir in frame_dirs[:10]:
                frame_name = os.path.basename(frame_dir)  # e.g., frame_0001
                frame_num = int(frame_name.split('_')[1])  # e.g., 1
                
                # Collect scores for this frame
                scores_for_frame = []
                score_files = glob.glob(os.path.join(frame_dir, "score_*.txt"))
                
                for score_file in score_files:
                    with open(score_file, 'r') as f:
                        try:
                            score = float(f.read().strip())
                            scores_for_frame.append(score)
                            all_scores.append(score)
                        except:
                            continue
                
                # Store scores for this frame
                if scores_for_frame:
                    frame_scores[frame_num] = scores_for_frame
            
            # Create a figure with subplots: one for histogram, one for frame-wise scores
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f"Score Distribution (first 100 frames, threshold={threshold:.2f})",
                    f"Highest Scores per Frame (above threshold={threshold:.2f})"
                ),
                vertical_spacing=0.15,
                row_heights=[0.4, 0.6],
                specs=[
                    [{}],  # First row, first column - normal y-axis only
                    [{"secondary_y": True}]  # Second row, first column - with secondary y-axis
                ]
            )
            
            # Add histogram to first subplot
            fig.add_trace(
                go.Histogram(
                    x=all_scores,
                    nbinsx=50,
                    opacity=0.8,
                    name="All Scores"
                ),
                row=1, col=1
            )
            
            # Add threshold line to histogram
            fig.add_vline(
                x=threshold, 
                line_width=2, 
                line_dash="dash", 
                line_color="red",
                row=1, col=1
            )
            
            # Add statistics as annotations
            if all_scores:
                mean_score = np.mean(all_scores)
                median_score = np.median(all_scores)
                min_score = min(all_scores)
                max_score = max(all_scores)
                
                stats_text = (f"Mean: {mean_score:.4f}<br>"
                             f"Median: {median_score:.4f}<br>"
                             f"Min: {min_score:.4f}<br>"
                             f"Max: {max_score:.4f}")
                
                fig.add_annotation(
                    x=0.15,
                    y=0.85,
                    xref="x domain",
                    yref="y domain",
                    text=stats_text,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    row=1, col=1
                )
            
            # Second subplot: Frame-wise highest scores
            # Extract data for the plot
            frame_numbers = []
            max_scores = []
            num_objects_above_threshold = []
            
            for frame_num in sorted(frame_scores.keys()):
                scores = frame_scores[frame_num]
                # Filter scores based on threshold
                filtered_scores = [s for s in scores if s >= threshold]
                
                if filtered_scores:  # Only add if there are scores above threshold
                    frame_numbers.append(frame_num)
                    max_scores.append(max(filtered_scores))
                    num_objects_above_threshold.append(len(filtered_scores))
            
            # Create bar chart for max scores per frame
            if frame_numbers:  # Only create plot if we have data
                # Add max score trace
                fig.add_trace(
                    go.Bar(
                        x=frame_numbers,
                        y=max_scores,
                        name="Highest Score",
                        marker_color='rgb(26, 118, 255)'
                    ),
                    row=2, col=1
                )
                
                # Add number of objects above threshold as a secondary y-axis
                fig.add_trace(
                    go.Scatter(
                        x=frame_numbers,
                        y=num_objects_above_threshold,
                        name="Objects Above Threshold",
                        mode="lines+markers",
                        marker=dict(color='rgb(255, 100, 100)'),
                        line=dict(width=2, dash='dot')
                    ),
                    row=2, col=1,
                    secondary_y=True
                )
                
                # Update layout for the second subplot
                fig.update_yaxes(
                    title_text="Max Score", 
                    range=[max(threshold, min(max_scores) * 0.9), max(max_scores) * 1.1],
                    row=2, col=1
                )
                fig.update_yaxes(
                    title_text="Object Count", 
                    secondary_y=True,
                    row=2, col=1
                )
                fig.update_xaxes(
                    title_text="Frame Number",
                    row=2, col=1
                )
            else:
                # If no frames have scores above threshold, show a message
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    text="No objects found above threshold",
                    showarrow=False,
                    font=dict(size=14),
                    row=2, col=1
                )
            
            # Update overall layout
            fig.update_layout(
                height=800,  # Increase height to accommodate both plots
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
    
    def run_server(self):
        """Run the Dash app server."""
        print(f"Starting server on http://localhost:{self.port}/")
        self.app.run_server(debug=self.debug, port=self.port, host='0.0.0.0')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Web-based visualization for segmentation masks and scores')
    parser.add_argument('--masks', required=True, help='Root directory containing mask folders')
    parser.add_argument('--scores', required=True, help='Root directory containing score folders')
    parser.add_argument('--frames', help='Directory containing original frames (optional)')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create web visualizer
    visualizer = WebVisualizer(
        masks_root_dir=args.masks,
        scores_root_dir=args.scores,
        frames_dir=args.frames,
        port=args.port,
        debug=args.debug
    )
    
    # Run the server
    visualizer.run_server()


if __name__ == "__main__":
    main()

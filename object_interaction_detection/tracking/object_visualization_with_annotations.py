"""
Object visualization with annotations example
This script demonstrates how to integrate the AnnotationData parser with the MotionFilteredLoader
to create visualizations that include both object activeness and annotation information.
"""

import os
import sys
import numpy as np
import cv2
import argparse
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from object_interaction_detection.dataloaders.motion_filtered_loader import MotionFilteredLoader
from object_interaction_detection.tracking.annotation_parser import AnnotationData, create_annotation_overlay


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize objects with annotation data')
    parser.add_argument('--session', type=str, required=True, help='Session name')
    parser.add_argument('--data-root', type=str, required=True, help='Root data directory')
    parser.add_argument('--camera', type=str, required=True, help='Camera view')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--annotation-root', type=str, default=None, 
                        help='Root directory for annotations (default: /nas/project_data/B1_Behavior/rush/kaan/hoi/annotations)')
    parser.add_argument('--start-frame', type=int, default=0, help='Start frame index')
    parser.add_argument('--end-frame', type=int, default=None, help='End frame index')
    parser.add_argument('--fps', type=float, default=30.0, help='Video FPS for annotation timing')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the MotionFilteredLoader
    motion_loader = MotionFilteredLoader(args.session, args.data_root)
    
    # Load the annotation data automatically for the session
    annotation_data = AnnotationData()
    annotation_data.load_from_session(args.session, args.annotation_root)
    annotation_data.set_fps(args.fps)
    
    # Determine start and end frames
    if args.end_frame is None:
        # Try to get the valid frame range from the loader
        frame_indices = motion_loader.combined_loader.get_valid_frame_indices(args.camera)
        end_frame = max(frame_indices) if frame_indices else 1000  # Default if can't determine
    else:
        end_frame = args.end_frame
    
    start_frame = args.start_frame
    
    # Create a timeline visualization of the annotations
    timeline_path = os.path.join(args.output_dir, f"{args.session}_{args.camera}_annotation_timeline.png")
    annotation_data.visualize_timeline(save_path=timeline_path)
    print(f"Saved annotation timeline to {timeline_path}")
    
    # Process each frame
    for frame_idx in range(start_frame, end_frame + 1):
        try:
            # Skip frames that don't exist
            if frame_idx not in motion_loader.combined_loader.get_valid_frame_indices(args.camera):
                continue
                
            # Get the motion-filtered object mask visualization
            visualization = motion_loader.visualize_motion_filtered_masks(
                args.camera, frame_idx, with_flow_vectors=True
            )
            
            if visualization is None:
                print(f"Warning: No visualization for frame {frame_idx}")
                continue
            
            # Get annotation data for this frame
            hand_used = annotation_data.get_hand_used_at_frame(frame_idx)
            objects = annotation_data.get_objects_at_frame(frame_idx)
            
            # Get all moving objects detected in this frame
            moving_objects = motion_loader.get_all_moving_objects(args.camera, frame_idx)
            
            # Calculate activeness scores for objects mentioned in the annotations
            activeness_scores = {}
            for obj_name in objects:
                if obj_name in moving_objects:
                    activeness = motion_loader.get_activeness(args.camera, frame_idx, obj_name)
                    activeness_scores[obj_name] = activeness.get('score', 0)
                else:
                    activeness_scores[obj_name] = 0
            
            # Add annotation information to the visualization
            annotated_viz = create_annotation_overlay(visualization, frame_idx, annotation_data)
            
            # Add additional overlay with activeness scores
            h, w = annotated_viz.shape[:2]
            score_text = "Activeness Scores:"
            
            # Sort objects by activeness score
            sorted_objects = sorted(activeness_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Add each object's score
            for i, (obj_name, score) in enumerate(sorted_objects):
                score_text += f"\n{obj_name}: {score:.3f}"
                # Highlight if it's a moving object
                if obj_name in moving_objects:
                    score_text += " (moving)"
            
            # Add the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            padding = 10
            
            # Determine text position (right side of the frame)
            text_x = w - 200
            text_y = 50
            
            # Get text dimensions
            text_lines = score_text.split('\n')
            line_height = 20
            
            # Draw semi-transparent background
            bg_height = len(text_lines) * line_height + 2 * padding
            overlay = annotated_viz.copy()
            cv2.rectangle(overlay, (text_x - padding, text_y - padding), 
                         (w - padding, text_y + bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_viz, 0.4, 0, annotated_viz)
            
            # Draw the text
            for i, line in enumerate(text_lines):
                y_pos = text_y + i * line_height
                cv2.putText(annotated_viz, line, (text_x, y_pos), font, font_scale, 
                           (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            # Save the visualization
            output_path = os.path.join(args.output_dir, f"{args.session}_{args.camera}_frame_{frame_idx:06d}.png")
            cv2.imwrite(output_path, annotated_viz)
            
            # Print progress
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}/{end_frame}")
        
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue
    
    # Create a summary of objects mentioned in the annotations
    summary = {
        "session": args.session,
        "camera": args.camera,
        "total_frames": end_frame - start_frame + 1,
        "annotations": {
            "all_objects": annotation_data.get_all_objects(),
            "hand_segments": len(annotation_data.get_all_segments("HandUsed")),
            "object_segments": len(annotation_data.get_all_segments("ObjectIdentity"))
        }
    }
    
    # Save summary
    summary_path = os.path.join(args.output_dir, f"{args.session}_{args.camera}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processing complete. Results saved to {args.output_dir}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

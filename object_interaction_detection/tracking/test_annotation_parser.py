#!/usr/bin/env python
"""
Test script for the annotation parser.
This script checks if we can correctly load and parse annotations for a session.
"""

import os
import sys
import argparse
import logging

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from object_interaction_detection.tracking.annotation_parser import AnnotationData


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test annotation parser')
    parser.add_argument('--session', type=str, default='imi_session1_2', 
                      help='Session name to test (default: imi_session1_2)')
    parser.add_argument('--annotation-root', type=str, default='/nas/project_data/B1_Behavior/rush/kaan/hoi/annotations', 
                      help='Root directory for annotations')
    parser.add_argument('--fps', type=float, default=30.0, 
                      help='Video FPS for annotation timing')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load annotations for the session
    logging.info(f"Loading annotations for session: {args.session}")
    annotations = AnnotationData()
    success = annotations.load_from_session(args.session, args.annotation_root)
    
    if not success:
        logging.error(f"Failed to load annotations for session: {args.session}")
        return
    
    # Set FPS
    annotations.set_fps(args.fps)
    
    # Get annotation summary
    logging.info("Annotation summary:")
    logging.info(f"  Hand segments: {len(annotations.get_all_segments('HandUsed'))}")
    logging.info(f"  Object segments: {len(annotations.get_all_segments('ObjectIdentity'))}")
    
    # Get all unique objects
    all_objects = annotations.get_all_objects()
    logging.info(f"All objects in annotations: {all_objects}")
    
    # Check a few frames
    test_frames = [0, 30, 60, 120, 300, 900]
    for frame_idx in test_frames:
        hand = annotations.get_hand_used_at_frame(frame_idx)
        objects = annotations.get_objects_at_frame(frame_idx)
        
        logging.info(f"Frame {frame_idx}:")
        logging.info(f"  Hand used: {hand}")
        logging.info(f"  Objects: {objects}")
    
    # Create timeline visualization
    output_dir = os.path.join(os.getcwd(), "timeline_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    timeline_path = os.path.join(output_dir, f"{args.session}_annotation_timeline.png")
    annotations.visualize_timeline(save_path=timeline_path)
    logging.info(f"Saved timeline visualization to: {timeline_path}")


if __name__ == "__main__":
    main()

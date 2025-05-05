#!/usr/bin/env python3
"""
Individual Video Processing for a Single Camera View

This script processes a single camera view using the HAMER model.
It allows specifying the exact camera path rather than a base directory with all sessions.
"""

import os
import cv2
import numpy as np
import torch
import argparse
import json
from pathlib import Path
from video_processor import VideoProcessor

def extract_session_camera_info(camera_path):
    """
    Extract session name and camera view from the camera path
    
    Args:
        camera_path: Path to the camera view directory
        
    Returns:
        Tuple of (session_name, camera_view)
    """
    parts = camera_path.strip('/').split('/')
    # The session name is usually the second-to-last part and camera view is the last
    camera_view = parts[-1]
    session_name = parts[-2] if len(parts) >= 2 else "unknown_session"
    
    return session_name, camera_view

def main():
    """Process a single camera view with the HAMER model"""
    parser = argparse.ArgumentParser(description="Process a single camera view with HAMER")
    
    parser.add_argument("--camera_path", type=str, required=True,
                        help="Path to the camera view directory with frames")
    parser.add_argument("--output_dir", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/hand_detections",
                        help="Base output directory for results")
    
    # Processing options
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame for processing")
    parser.add_argument("--end", type=int, default=None,
                        help="End frame for processing")
    parser.add_argument("--step", type=int, default=1,
                        help="Process every Nth frame")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for processing")
    
    # Visualization options
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering of hands on output frames")
    parser.add_argument("--side-view", action="store_true",
                        help="Include side view in rendering")
    
    # Save options
    parser.add_argument("--no-save-meshes", action="store_true",
                        help="Don't save hand crops and 3D pose data")
    
    args = parser.parse_args()
    
    # Validate camera path
    camera_path = args.camera_path
    if not os.path.isdir(camera_path):
        print(f"Error: Camera path not found: {camera_path}")
        return
    
    # Extract session name and camera view from path
    session_name, camera_view = extract_session_camera_info(camera_path)
    print(f"Processing session: {session_name}, camera: {camera_view}")
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        device = f"cuda:{args.gpu_id}"
    else:
        print("CUDA not available, using CPU")
        device = "cpu"
        raise ValueError("CUDA not available!")
    
    # Create processor
    processor = VideoProcessor(
        output_dir=args.output_dir,
        render_hands=not args.no_render,
        save_meshes=not args.no_save_meshes,
        side_view=args.side_view,
        device=device
    )
    
    # Process this single camera view
    parent_dir = os.path.dirname(camera_path)
    processor.process_frame_sequence(
        camera_path,
        output_dir=args.output_dir,
        start_frame=args.start,
        end_frame=args.end,
        step=args.step,
        session_name=session_name,
        camera_view=camera_view
    )
    
    print(f"Completed processing {session_name}/{camera_view}")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

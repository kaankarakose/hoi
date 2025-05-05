#!/usr/bin/env python3
"""
Process only incomplete videos using the HAMER model.
This script first checks which videos are incomplete and then only processes those.
"""

import os
import json
import argparse
import torch
from video_processor import VideoProcessor, get_session_list

def check_processing_completion(root_dir, frames_root):
    """
    Checks which session/camera pairs are fully processed by comparing bounding_boxes.json frame count
    with the number of original frames.
    
    Returns lists of complete, incomplete, and missing processing pairs.
    """
    complete = []
    incomplete = []
    missing = []
    
    for session in sorted(os.listdir(frames_root)):
        session_path = os.path.join(frames_root, session)
        if not os.path.isdir(session_path):
            continue
        for camera_view in sorted(os.listdir(session_path)):
            camera_path = os.path.join(session_path, camera_view)
            if not os.path.isdir(camera_path):
                continue
                
            bbox_file = os.path.join(root_dir, session, camera_view, 'bounding_boxes.json')
            frames_dir = os.path.join(frames_root, session, camera_view)
            
            # Count original frames
            if not os.path.isdir(frames_dir):
                num_frames = 0
            else:
                num_frames = len([
                    f for f in os.listdir(frames_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ])
                
            # Count processed frames
            if os.path.isfile(bbox_file):
                try:
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                    num_bboxes = len(bbox_data)
                except Exception as e:
                    num_bboxes = 0
                    
                if num_frames > 0 and num_frames - num_bboxes < 300:
                    complete.append((session, camera_view, num_frames))
                else:
                    incomplete.append((session, camera_view, num_bboxes, num_frames))
            else:
                missing.append((session, camera_view, num_frames))
                
    return complete, incomplete, missing

def print_completion_report(complete, incomplete, missing, root_dir):
    """Print a summary of the completion report."""
    print("\n=== Processing Completion Report ===")
    print(f"Root directory: {root_dir}\n")
    
    print(f"\nComplete (all frames processed): {len(complete)}")
    for session, camera, n in complete:
        print(f"  [OK] {session}/{camera} ({n} frames)")
        
    print(f"\nIncomplete (partial processing): {len(incomplete)}")
    for session, camera, n_bbox, n_frames in incomplete:
        print(f"  [INCOMPLETE] {session}/{camera}: bounding_boxes.json frames = {n_bbox}, original frames = {n_frames}")
        
    print(f"\nMissing (no bounding_boxes.json): {len(missing)}")
    for session, camera, n_frames in missing:
        print(f"  [MISSING] {session}/{camera} (original frames: {n_frames})")
        
    print("\nDone.")

def main():
    """Main entry point for the selective video processing pipeline"""
    parser = argparse.ArgumentParser(description="Process only incomplete image sequences for hand tracking using HAMER")
    
    # Input/output options
    parser.add_argument("--frames_dir", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/orginal_frames",
                        help="Directory containing original video frames")
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
    parser.add_argument("--print_only", action="store_true",
                        help="Only print the completion report without processing")
    
    # Visualization options
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering of hands on output frames")
    parser.add_argument("--side-view", action="store_true",
                        help="Include side view in rendering")
    
    # Save options
    parser.add_argument("--no-save-meshes", action="store_true",
                        help="Don't save hand crops and 3D pose data")
    
    args = parser.parse_args()
    
    # Check which videos are incomplete
    complete, incomplete, missing = check_processing_completion(args.output_dir, args.frames_dir)
    print_completion_report(complete, incomplete, missing, args.output_dir)
    
    # If print-only mode, exit here
    if args.print_only:
        return
    
    # Combine incomplete and missing videos to process
    to_process = [(session, camera) for session, camera, _, _ in incomplete]
    to_process.extend([(session, camera) for session, camera, _ in missing])
    
    if not to_process:
        print("No incomplete videos to process. Exiting.")
        return
    
    print(f"\nWill process {len(to_process)} incomplete/missing videos.")
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        device = f"cuda:{args.gpu_id}"
    else:
        print("CUDA not available, using CPU")
        device = "cpu"
        raise ValueError("Cuda not available!")
    
    # Create processor
    processor = VideoProcessor(
        output_dir=args.output_dir,
        render_hands=not args.no_render,
        save_meshes=not args.no_save_meshes,
        side_view=args.side_view,
        device=device
    )
    
    # Process each incomplete session/camera pair
    for session_name, camera_view in to_process:
        print(f"\nProcessing incomplete: {session_name}/{camera_view}")
        session_path = os.path.join(args.frames_dir, session_name)
        camera_path = os.path.join(session_path, camera_view)
        
        if not os.path.isdir(camera_path):
            print(f"Warning: Camera path not found: {camera_path}. Skipping.")
            continue
        
        processor.process_camera_view(
            session_path,
            camera_view,
            output_base_dir=args.output_dir,
            start_frame=args.start,
            end_frame=args.end,
            step=args.step
        )
        
        # Clear any cached GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\nAll incomplete videos have been processed.")

if __name__ == "__main__":
    main()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

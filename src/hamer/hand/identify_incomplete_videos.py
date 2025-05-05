#!/usr/bin/env python3
"""
Identify incomplete videos without requiring HAMER dependencies.
This script will list which videos need processing and can output a shell script to run them.
"""

import os
import json
import argparse

def check_processing_completion(root_dir, frames_root, special_dir=None):
    """
    Checks which session/camera pairs are fully processed by comparing bounding_boxes.json frame count
    with the number of original frames.
    
    Args:
        root_dir: Directory containing output data
        frames_root: Directory containing input frames
        special_dir: Special directory for cam_side_l frames
        
    Returns lists of complete, incomplete, and missing processing pairs.
    """
    complete = []
    incomplete = []
    missing = []
    
    # Get all sessions from frames_root
    sessions = set()
    for session in sorted(os.listdir(frames_root)):
        session_path = os.path.join(frames_root, session)
        if os.path.isdir(session_path):
            sessions.add(session)
    
    # Add any additional sessions from special_dir if provided
    if special_dir and os.path.exists(special_dir):
        for session in sorted(os.listdir(special_dir)):
            session_path = os.path.join(special_dir, session)
            if os.path.isdir(session_path):
                sessions.add(session)
    
    # Process each session
    for session in sorted(list(sessions)):
        # Check regular frames directory
        session_path = os.path.join(frames_root, session)
        if os.path.isdir(session_path):
            for camera_view in sorted(os.listdir(session_path)):
                # Skip cam_side_l if we have a special directory for it
                if camera_view == 'cam_side_l' and special_dir:
                    continue
                    
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

        # Handle special directory for cam_side_l if provided
        if special_dir:
            special_session_path = os.path.join(special_dir, session)
            if os.path.isdir(special_session_path):
                camera_view = 'cam_side_l'  # We only use special_dir for cam_side_l
                special_camera_path = os.path.join(special_session_path, camera_view)
                
                if os.path.isdir(special_camera_path):
                    bbox_file = os.path.join(root_dir, session, camera_view, 'bounding_boxes.json')
                    frames_dir = special_camera_path
                    
                    # Count original frames
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

def generate_processing_script(incomplete, missing, frames_dir, output_dir, special_dir=None, gpu_id=0):
    """Generate a bash script to process incomplete videos."""
    to_process = [(session, camera) for session, camera, _, _ in incomplete]
    to_process.extend([(session, camera) for session, camera, _ in missing])
    
    if not to_process:
        print("No incomplete videos to process.")
        return None
    
    script_content = "#!/bin/bash\n\n"
    script_content += f"# Auto-generated script to process {len(to_process)} incomplete videos\n\n"
    
    for session_name, camera_view in to_process:
        # Use special directory for cam_side_l if provided
        if camera_view == 'cam_side_l' and special_dir:
            input_dir = f"{special_dir}/{session_name}/{camera_view}"
        else:
            input_dir = f"{frames_dir}/{session_name}/{camera_view}"
            
        cmd = f"python individual_video_processor.py --camera_path \"{input_dir}\" "
        cmd += f"--output_dir \"{output_dir}\" --gpu_id {gpu_id}\n"
        script_content += cmd
    
    return script_content

def main():
    parser = argparse.ArgumentParser(description="Identify incomplete videos for HAMER processing")
    
    # Input/output options
    parser.add_argument("--frames_dir", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/orginal_frames",
                        help="Directory containing original video frames")
    parser.add_argument("--special_dir", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/torso_hands_removed",
                        help="Special directory for cam_side_l frames")
    parser.add_argument("--output_dir", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/hand_detections",
                        help="Base output directory for results")
    parser.add_argument("--generate_script", action="store_true",
                        help="Generate bash script to process incomplete videos")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for processing")
    parser.add_argument("--script_output", type=str, default="process_incomplete.sh",
                        help="Output file for the generated script")
    
    args = parser.parse_args()
    
    # Check which videos are incomplete
    complete, incomplete, missing = check_processing_completion(
        args.output_dir, args.frames_dir, args.special_dir)
    print_completion_report(complete, incomplete, missing, args.output_dir)
    
    # Generate script if requested
    if args.generate_script:
        script_content = generate_processing_script(
            incomplete, missing, args.frames_dir, args.output_dir, args.special_dir, args.gpu_id)
        if script_content:
            with open(args.script_output, "w") as f:
                f.write(script_content)
            os.chmod(args.script_output, 0o755)  # Make executable
            print(f"\nGenerated processing script: {args.script_output}")
            print("You can run it with:")
            print(f"  ./{args.script_output}")

if __name__ == "__main__":
    main()

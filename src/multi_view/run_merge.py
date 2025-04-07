#!/usr/bin/env python3
"""
Helper script to run the multi-view merging pipeline.
This demonstrates how to use the MultiViewMerger class for different sessions and objects.
"""

import argparse
import os
import glob
from pathlib import Path
import json
from merge_views import MultiViewMerger

def get_available_sessions(base_dir):
    """Get all available sessions in the base directory."""
    if not os.path.exists(base_dir):
        return []
    
    sessions = [os.path.basename(d) for d in glob.glob(os.path.join(base_dir, "*")) 
                if os.path.isdir(d)]
    return sessions

def get_available_objects(base_dir, session, camera_view="cam_side_l"):
    """Get all available objects in a session and camera view."""
    session_dir = os.path.join(base_dir, session, camera_view)
    if not os.path.exists(session_dir):
        return []
    
    objects = [os.path.basename(d) for d in glob.glob(os.path.join(session_dir, "*")) 
               if os.path.isdir(d)]
    return objects

def main():
    parser = argparse.ArgumentParser(description="Run multi-view merging for multiple sessions and objects")
    parser.add_argument("--base-dir", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/cnos_results",
                        help="Base directory containing prediction data")
    parser.add_argument("--camera-params-dir", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/camera_params",
                        help="Directory containing camera parameter files")
    parser.add_argument("--output-dir", type=str, 
                        default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/multi_view_results",
                        help="Directory to save merged results")
    parser.add_argument("--sessions", type=str, nargs="+", help="Session names to process (optional)")
    parser.add_argument("--objects", type=str, nargs="+", help="Object names to process (optional)")
    parser.add_argument("--cameras", type=str, nargs="+", 
                        default=["cam_side_l", "cam_side_r", "cam_top"],
                        help="Camera views to use")
    parser.add_argument("--strategy", type=str, default="triangulate", 
                        choices=["triangulate", "highest", "average", "union"],
                        help="Strategy to merge masks")
    parser.add_argument("--list-only", action="store_true", 
                        help="Only list available sessions and objects without processing")
    parser.add_argument("--check-camera-params", action="store_true",
                        help="Check if camera parameter files exist for the specified cameras")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if camera parameter files exist
    if args.check_camera_params or args.strategy == 'triangulate':
        print("Checking camera parameter files...")
        camera_params_exist = True
        for camera in args.cameras:
            camera_param_path = os.path.join(args.camera_params_dir, f"{camera}.json")
            if not os.path.exists(camera_param_path):
                print(f"  Warning: Camera parameter file not found: {camera_param_path}")
                camera_params_exist = False
        
        if not camera_params_exist:
            print("  Note: Missing camera parameter files will prevent 3D triangulation.")
            print("  Create camera parameter files in the camera_params_dir with intrinsic and extrinsic matrices.")
            if args.check_camera_params:
                print("  Example camera parameter file format:")
                example = {
                    "intrinsic": [
                        [800.0, 0.0, 320.0],
                        [0.0, 800.0, 240.0],
                        [0.0, 0.0, 1.0]
                    ],
                    "extrinsic": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                }
                print(json.dumps(example, indent=2))
        else:
            print("  All camera parameter files found.")
    
    # Get available sessions
    available_sessions = get_available_sessions(args.base_dir)
    if not available_sessions:
        print(f"No sessions found in {args.base_dir}")
        return
    
    print("Available sessions:")
    for session in available_sessions:
        print(f"  - {session}")
    
    # Determine sessions to process
    sessions_to_process = args.sessions if args.sessions else available_sessions
    sessions_to_process = [s for s in sessions_to_process if s in available_sessions]
    
    if not sessions_to_process:
        print("No valid sessions to process")
        return
    
    print(f"\nWill process {len(sessions_to_process)} sessions: {', '.join(sessions_to_process)}")
    
    # Process each session
    for session in sessions_to_process:
        print(f"\nProcessing session: {session}")
        
        # Get available objects for this session
        available_objects = get_available_objects(args.base_dir, session)
        if not available_objects:
            print(f"  No objects found for session {session}")
            continue
        
        print("  Available objects:")
        for obj in available_objects:
            print(f"    - {obj}")
        
        # Determine objects to process
        objects_to_process = args.objects if args.objects else available_objects
        objects_to_process = [o for o in objects_to_process if o in available_objects]
        
        if not objects_to_process:
            print(f"  No valid objects to process for session {session}")
            continue
        
        print(f"  Will process {len(objects_to_process)} objects: {', '.join(objects_to_process)}")
        
        if args.list_only:
            continue
        
        # Process each object
        for obj in objects_to_process:
            print(f"\n  Processing object: {obj}")
            
            merger = MultiViewMerger(
                session_name=session,
                object_name=obj,
                camera_views=args.cameras,
                base_dir=args.base_dir,
                camera_params_dir=args.camera_params_dir,
                output_dir=args.output_dir,
                merge_strategy=args.strategy
            )
            
            merger.process_all_frames()
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()

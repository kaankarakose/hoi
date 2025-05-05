import os
import argparse

import json

def check_processing_completion(root_dir, frames_root):
    """
    Checks which session/camera pairs are fully processed by comparing bounding_boxes.json frame count
    with the number of original frames.
    Prints a summary of complete, incomplete, and missing processing.
    """
    complete = []
    incomplete = []
    missing = []
    
    for session in sorted(os.listdir(root_dir)):
        session_path = os.path.join(root_dir, session)
        if not os.path.isdir(session_path):
            continue
        for camera_view in sorted(os.listdir(session_path)):
            camera_path = os.path.join(session_path, camera_view)
            if not os.path.isdir(camera_path):
                continue
            bbox_file = os.path.join(camera_path, 'bounding_boxes.json')
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
                #print(num_frames - num_bboxes, "fark")
                if num_frames > 0 and num_frames - num_bboxes < 300:
                    complete.append((session, camera_view, num_frames))
                else:
                    incomplete.append((session, camera_view, num_bboxes, num_frames))
            else:
                missing.append((session, camera_view, num_frames))
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

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Check which videos are fully processed (bounding_boxes.json matches original frames)")
    #parser.add_argument("root_dir", type=str,default = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/orginal_frames" ,help="Root directory containing processed session folders")
    #parser.add_argument("frames_root", type=str,default = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/hand_detections", help="Root directory containing original frames (session/camera_view)")
    #args = parser.parse_args()
    #check_processing_completion(args.root_dir, args.frames_root)
    root_dir = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/hand_detections"
    frames_root = "/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/orginal_frames"
    check_processing_completion(root_dir, frames_root)

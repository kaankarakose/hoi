#!/usr/bin/env python3
"""
Video Processing Pipeline for Hand Tracking

This module provides a pipeline for processing image sequences to track hands using the HAMER model.
It handles image input/output, frame processing, and visualization of results.
"""

import os
import cv2
import numpy as np
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Import the HAMERWrapper
from HAMERWrapper import HAMERWrapper, Hand

class VideoProcessor:
    """
    Process image sequences to detect and track hands using the HAMER model
    
    This class provides functionality to:
    1. Load and process image sequences
    2. Detect and track hands in each frame
    3. Visualize and save results
    """
    
    def __init__(self, 
                 output_dir=None, 
                 render_hands=True, 
                 save_meshes=True, 
                 side_view=False,
                 device=None):
        """
        Initialize the video processor
        
        Args:
            output_dir: Directory to save output files (None for no saving)
            render_hands: Whether to render hand meshes on frames
            save_meshes: Whether to save hand mesh data
            side_view: Whether to include a side view in rendering
            device: Device to run the model on (None for auto-detection)
        """
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
        self.render_hands = render_hands
        self.save_meshes = save_meshes
        self.side_view = side_view
        
        # Initialize the hand model
        self.hand_model = HAMERWrapper(device=device)
        

    
    def _crop_hand_with_padding(self, image, bbox, pad_factor=0.8):
        """
        Crop a hand from an image with additional padding
        
        Args:
            image: Source image
            bbox: Hand bounding box [x1, y1, x2, y2]
            pad_factor: Padding factor relative to bbox size
            
        Returns:
            Cropped image with padding, or None if invalid crop
        """
        if bbox is None or image is None:
            return None, None
            
        # Convert bbox to integers
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Check if bbox is valid
            if x1 >= x2 or y1 >= y2:
                print(f"Warning: Invalid bbox dimensions: [{x1}, {y1}, {x2}, {y2}]")
                return None, None
            
            # Calculate padding
            width = x2 - x1
            height = y2 - y1
            pad_x = int(width * pad_factor)
            pad_y = int(height * pad_factor)
            
            # Apply padding with bounds checking
            img_h, img_w = image.shape[:2]
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(img_w, x2 + pad_x)
            y2_pad = min(img_h, y2 + pad_y)
            
            # Check if padded dimensions are valid
            if x1_pad >= x2_pad or y1_pad >= y2_pad:
                print(f"Warning: Invalid padded bbox dimensions: [{x1_pad}, {y1_pad}, {x2_pad}, {y2_pad}]")
                return None, None
            
            # Crop the image
            crop = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            
            # Check if crop is empty
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"Warning: Empty crop from bbox [{x1}, {y1}, {x2}, {y2}] with padding [{x1_pad}, {y1_pad}, {x2_pad}, {y2_pad}]")
                return None, None
                
            return crop, [x1_pad, y1_pad, x2_pad, y2_pad]
            
        except Exception as e:
            print(f"Error cropping hand: {str(e)}")
            return None, None
    
    def _save_outputs(self, session_name, camera_view, frame_index, hands, image):
        """
        Save hand crops and 3D mesh data according to the desired structure:
        Session_name -> Camera_view -> R_frames & L_frames -> cropped images and 3D poses
        Also saves bounding box information to a separate JSON file for easy retrieval
        
        Args:
            session_name: Name of the current session
            camera_view: Name of the current camera view
            frame_index: Index of the current frame
            hands: Dictionary with left and right hand objects
            image: Current frame image
        """
        if self.output_dir is None:
            return
            
        # Create the session and camera directories
        session_dir = os.path.join(self.output_dir, session_name)
        camera_dir = os.path.join(session_dir, camera_view)
        os.makedirs(camera_dir, exist_ok=True)
        
        # Create directories for left and right hand frames
        left_frames_dir = os.path.join(camera_dir, "L_frames")
        right_frames_dir = os.path.join(camera_dir, "R_frames")
        os.makedirs(left_frames_dir, exist_ok=True)
        os.makedirs(right_frames_dir, exist_ok=True)
        
        # Create directory for 3D poses
        poses_dir = os.path.join(camera_dir, "3D_poses")
        os.makedirs(poses_dir, exist_ok=True)
        
        # Bounding box information to save
        bbox_info = {
            'frame_index': frame_index,
            'left_hand': None,
            'right_hand': None
        }
        left_bbox = np.array([0,0,0,0])
        right_bbox = np.array([0,0,0,0])

        # Process left hand
        if hands['left'] != []:
            left_bbox = hands['left'][0].bbox

            # # Expand bounding box for each additional hand
            # for hand in hands['left'][1:]:
            #     current_bbox = hand.bbox
            #     left_bbox = np.array([
            #             min(left_bbox[0], current_bbox[0]),  # min x1
            #             min(left_bbox[1], current_bbox[1]),  # min y1
            #             max(left_bbox[2], current_bbox[2]),  # max x2
            #             max(left_bbox[3], current_bbox[3])   # max y2
            #         ])
                            
            bbox_info['left_hand'] = left_bbox.tolist() if isinstance(left_bbox, np.ndarray) else left_bbox
                
            # Save cropped image
            left_crop, left_crop_bbox = self._crop_hand_with_padding(image, left_bbox)
            if left_crop is not None and not left_crop.size == 0:
                left_crop_path = os.path.join(left_frames_dir, f"frame_{frame_index:06d}.jpg")
                cv2.imwrite(left_crop_path, left_crop)
            
            # Save 3D pose data
            left_pose_data = {
                'vertices': hands['left'][0].vertices,
                'cam_t': hands['left'][0].cam_t,
                'left_crop_bbox' : left_crop_bbox,
                'bbox': left_bbox,
                'is_right': False
            }
            left_pose_path = os.path.join(poses_dir, f"left_hand_{frame_index:06d}.npz")
            np.savez(left_pose_path, **left_pose_data)
        
        # Process right hand
        if hands['right'] != []:
            right_bbox = hands['right'][0].bbox
            # Expand bounding box for each additional hand
            # for hand in hands['left'][1:]:
            #     current_bbox = hand.bbox
            #     right_bbox = np.array([
            #             min(right_bbox[0], current_bbox[0]),  # min x1
            #             min(right_bbox[1], current_bbox[1]),  # min y1
            #             max(right_bbox[2], current_bbox[2]),  # max x2
            #             max(right_bbox[3], current_bbox[3])   # max y2
            #         ])
            bbox_info['right_hand'] = right_bbox.tolist() if isinstance(right_bbox, np.ndarray) else right_bbox
            
            # Save cropped image
            right_crop, right_crop_bbox = self._crop_hand_with_padding(image, right_bbox)
            if right_crop is not None and not right_crop.size == 0:
                right_crop_path = os.path.join(right_frames_dir, f"frame_{frame_index:06d}.jpg")
                cv2.imwrite(right_crop_path, right_crop)
            
            # Save 3D pose data
            right_pose_data = {
                'vertices': hands['right'][0].vertices,
                'cam_t': hands['right'][0].cam_t,
                'right_crop_bbox' : right_crop_bbox,
                'bbox': right_bbox,
                'is_right': True
            }
            right_pose_path = os.path.join(poses_dir, f"right_hand_{frame_index:06d}.npz")
            np.savez(right_pose_path, **right_pose_data)
        
        # Save the bounding box information
        self._save_bbox_info(session_dir, camera_view, frame_index, bbox_info)
    
    def _save_bbox_info(self, session_dir, camera_view, frame_index, bbox_info):
        """
        Save bounding box information to a JSON file for easy retrieval
        
        Args:
            session_dir: Path to the session directory
            camera_view: Camera view name
            frame_index: Current frame index
            bbox_info: Dictionary containing bounding box information
        """
        # Path to the bounding box JSON file
        bbox_file = os.path.join(session_dir, camera_view, 'bounding_boxes.json')
        
        # Initialize or load the existing bbox data
        if os.path.exists(bbox_file):
            try:
                with open(bbox_file, 'r') as f:
                    all_bbox_data = json.load(f)
            except json.JSONDecodeError:
                # Handle case of empty or corrupted file
                all_bbox_data = {}
        else:
            all_bbox_data = {}
        
        # Add the current frame's bbox info
        frame_key = str(frame_index)
        all_bbox_data[frame_key] = bbox_info
        
        # Save the updated bbox data
        with open(bbox_file, 'w') as f:
            json.dump(all_bbox_data, f, indent=2)

    
    def process_frame_sequence(self, frames_dir, output_dir=None, start_frame=0, end_frame=None, step=1, session_name=None, camera_view=None):
        """
        Process a sequence of frames from a directory
        
        Args:
            frames_dir: Directory containing frame images
            output_dir: Directory to save processed frames (None to use default)
            start_frame: Index of first frame to process (0-indexed)
            end_frame: Index of last frame to process (None for all frames)
            step: Process every Nth frame
        
        Returns:
            Statistics dictionary
        """
        # Set output directory
        if output_dir is None and self.output_dir is not None:
            output_dir = self.output_dir
        
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get all frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        if not frame_files:
            print(f"No frames found in directory: {frames_dir}")
            return {}
        
        # Adjust end frame if needed
        total_frames = len(frame_files)
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        

    
        # Create progress bar
        frames_to_process = ((end_frame - start_frame) + step - 1) // step
        pbar = tqdm(total=frames_to_process, desc="Processing frames")
        
        # Process frames in the specified range
        for frame_idx in range(start_frame, end_frame, step):
            if frame_idx >= len(frame_files):
                break
                
            # Get frame path
            frame_path = os.path.join(frames_dir, frame_files[frame_idx])
            
            # Set output path if needed
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"processed_{frame_files[frame_idx]}")
            else:
                output_path = None
            
            # Process frame
            img = cv2.imread(frame_path)
            
            if img is None:
                print(f"Warning: Could not read frame {frame_path}")
                pbar.update(1)
                continue
            
            has_hands = self.hand_model.process_frame(img)
            if not has_hands:
                pbar.update(1)
                continue ## No hands detected then what to do?
            # Get hand detection results
            hands = self.hand_model.getHands()
            
            # Store original image before rendering
            original_img = img.copy()
            
            # Render hands if requested
            if self.render_hands and has_hands:
                img = self.hand_model.render(img, side_view=self.side_view)
            
            # Save outputs (crops and 3D poses) if requested
            if has_hands:
                    
                # Determine session and camera names if not provided
                current_session = session_name or os.path.basename(os.path.dirname(frames_dir))
                current_camera = camera_view or os.path.basename(frames_dir)
                
                # Always save if hands are detected
                if self.save_meshes:
                    self._save_outputs(current_session, current_camera, frame_idx, hands, original_img)
            
            # # Save processed frame if needed
            # if output_path is not None:
            #     cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))           
    
    
    def process_camera_view(self, session_dir, camera_view, output_base_dir=None, **kwargs):
        """
        Process all frames for a single camera view
        
        Args:
            session_dir: Directory containing camera view directories
            camera_view: Name of the camera view directory
            output_base_dir: Base directory for outputs (None to use default)
            **kwargs: Additional arguments for process_frame_sequence
            
        Returns:
            Statistics dictionary
        """
        camera_path = os.path.join(session_dir, camera_view)
        
        if not os.path.isdir(camera_path):
            print(f"Camera view directory not found: {camera_path}")
            return {}
            
        print(f"Processing camera view: {camera_view}")
        
        # Set output directory
        if output_base_dir is not None:
            output_dir = os.path.join(output_base_dir, os.path.basename(session_dir), camera_view)
        else:
            output_dir = None
            
        # Process frames with session and camera names
        session_name = os.path.basename(session_dir)
        return self.process_frame_sequence(
            camera_path, 
            output_dir, 
            session_name=session_name,
            camera_view=camera_view,
            **kwargs
        )
    
    def process_session(self, session_dir, output_base_dir=None, **kwargs):
        """
        Process all camera views in a session directory
        
        Args:
            session_dir: Directory containing camera view directories
            output_base_dir: Base directory for outputs (None to use default)
            **kwargs: Additional arguments for process_frame_sequence
            
        Returns:
            Dictionary of camera views and their processing statistics
        """
        if not os.path.isdir(session_dir):
            print(f"Session directory not found: {session_dir}")
            return {}
            
        print(f"Processing session: {os.path.basename(session_dir)}")
        
        # Get all camera views
        camera_views = [d for d in os.listdir(session_dir) 
                       if os.path.isdir(os.path.join(session_dir, d))]
        
        if not camera_views:
            print(f"No camera view directories found in: {session_dir}")
            return {}
            
        # Process each camera view
        results = {}
        for camera_view in camera_views:
            if camera_view in ['cam_side_l']: continue # Do not process cam_side_l ( cam_side_l has two person in the scene)
            # try:
            self.process_camera_view(
                session_dir, 
                camera_view, 
                output_base_dir, 
                **kwargs
            )
            
            # except Exception as e:
            #     print(f"Error processing camera view {camera_view}: {e}")
            #     results[camera_view] = {'error': str(e)}
                
    def process_multiple_sessions(self, base_dir, output_base_dir=None, **kwargs):
        """
        Process all session directories in a base directory
        
        Args:
            base_dir: Base directory containing session directories
            output_base_dir: Base directory for outputs (None to use default)
            **kwargs: Additional arguments for process_frame_sequence
            
        Returns:
            Dictionary of sessions and their processing statistics
        """
        if not os.path.isdir(base_dir):
            print(f"Base directory not found: {base_dir}")
            return {}
            
        # Get all session directories
        sessions = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d))]
        
        if not sessions:
            print(f"No session directories found in: {base_dir}")
            return {}
            
        print(f"Found {len(sessions)} sessions to process")
        
        # Process each session
        results = {}
        for session in sessions:
            try:
                session_path = os.path.join(base_dir, session)
                stats = self.process_session(
                    session_path, 
                    output_base_dir, 
                    **kwargs
                )
                results[session] = stats
            except Exception as e:
                print(f"Error processing session {session}: {e}")
                results[session] = {'error': str(e)}
                
        return results


def get_session_list(frames_dir):
    """Get list of sessions in the frames directory"""
    return [d for d in os.listdir(frames_dir) 
           if os.path.isdir(os.path.join(frames_dir, d))]

def main():
    """Main entry point for the video processing pipeline"""
    parser = argparse.ArgumentParser(description="Process image sequences for hand tracking using HAMER")
    
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
    parser.add_argument("--num_splits", type=int, default=1,
                        help="Number of parts to split the session dataset into")
    parser.add_argument("--split_id", type=int, default=0,
                        help="Which split to process (0-indexed, must be less than num_splits)")
    
    # Visualization options
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering of hands on output frames")
    parser.add_argument("--side-view", action="store_true",
                        help="Include side view in rendering")
    
    # Save options
    parser.add_argument("--no-save-meshes", action="store_true",
                        help="Don't save hand crops and 3D pose data")
    parser.add_argument("--save-rendered-frames", action="store_true",
                        help="Save frames with rendered hand visualization")
    
    args = parser.parse_args()
    
    # Validate split parameters
    if args.split_id >= args.num_splits:
        raise ValueError(f"Split ID ({args.split_id}) must be less than num_splits ({args.num_splits})")
    
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
    
    # Get list of sessions
    all_sessions_list = get_session_list(args.frames_dir)
    print(f"Found {len(all_sessions_list)} sessions in total")
    
    # Split sessions into parts if specified
    if args.num_splits > 1:
        sessions_per_split = len(all_sessions_list) // args.num_splits
        start_idx = args.split_id * sessions_per_split
        end_idx = start_idx + sessions_per_split if args.split_id < args.num_splits - 1 else len(all_sessions_list)
        sessions_list = all_sessions_list[start_idx:end_idx]
        print(f"Processing split {args.split_id + 1}/{args.num_splits}: {len(sessions_list)} sessions (from {start_idx} to {end_idx-1})")
    else:
        sessions_list = all_sessions_list
        print(f"Processing all {len(sessions_list)} sessions")
    
    # Process each session in this split
    for session_name in sessions_list:
        session_path = os.path.join(args.frames_dir, session_name)
        print(f"\nProcessing session: {session_name}")
        processor.process_session(
            session_path,
            output_base_dir=args.output_dir,
            start_frame=args.start,
            end_frame=args.end,
            step=args.step
        )
        # Clear any cached GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    

if __name__ == "__main__":
    main()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import cv2

from utils import (
    read_frame_data,
    merge_masks,
    save_rle,
    mask_to_rle,
    rle_to_mask,
    NumpyEncoder,
    get_camera_matrix,
    triangulate_mask
)

from cameraClass import CameraSystem


class MultiViewMerger:
    """
    Class to merge predictions from multiple camera views.
    """
    
    def __init__(self, session_name, object_name, camera_views=None, 
                 base_dir="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/cnos_results",
                 camera_params_dir="/nas/project_data/B1_Behavior/rush/kaan/hoi/camera_params",
                 output_dir="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/multi_view_results",
                 merge_strategy="triangulate"):
        """
        Initialize the MultiViewMerger.
        
        Args:
            session_name: Name of the session (e.g., 'imi_session1_2')
            object_name: Name of the object (e.g., 'AMF1')
            camera_views: List of camera views (e.g., ['cam_side_l', 'cam_side_r', 'cam_top'])
                          If None, will use all available camera views.
            base_dir: Base directory containing the prediction data.
            output_dir: Directory to save merged results.
            merge_strategy: Strategy to merge masks ('highest', 'average', or 'union')
        """
        self.session_name = session_name
        self.object_name = object_name
        self.base_dir = base_dir
        self.camera_params_dir = camera_params_dir
        self.output_dir = output_dir
        self.merge_strategy = merge_strategy
        
        # Find camera views if not provided
        if camera_views is None:
            session_dir = os.path.join(base_dir, session_name)
            if os.path.exists(session_dir):
                camera_views = [d for d in os.listdir(session_dir) 
                               if os.path.isdir(os.path.join(session_dir, d)) and d.startswith('cam_')]
            else:
                raise ValueError(f"Session directory not found: {session_dir}")
        
        self.camera_views = camera_views
        print(f"Using camera views: {self.camera_views}")
        
        # Prepare output directory
        self.output_session_dir = os.path.join(output_dir, session_name)
        self.output_object_dir = os.path.join(self.output_session_dir, object_name)
        os.makedirs(os.path.join(self.output_object_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(self.output_object_dir, 'scores'), exist_ok=True)
        os.makedirs(os.path.join(self.output_object_dir, '3d_points'), exist_ok=True)
        
        # Load camera parameters
        self.camera_params = self.load_camera_params()
        
    def load_camera_params(self):
        """
        Load camera intrinsic and extrinsic parameters for each camera view.
        
        Returns:
            Dictionary mapping camera view to camera parameters and CameraSystem object
        """
        camera_params = {}
        
        # First load traditional camera parameters (for backward compatibility)
        for camera_view in self.camera_views:
            # Path to camera parameters file # camera_view - cam_side_l,cam_side_r, cam_top
            params_file = os.path.join(self.camera_params_dir, f"{camera_view}.json")
            
            # Check if file exists
            if os.path.exists(params_file):
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                    
                    # Convert parameter arrays to numpy arrays
                    if 'intrinsic' in params:
                        params['intrinsic'] = np.array(params['intrinsic'])
                    if 'extrinsic' in params:
                        params['extrinsic'] = np.array(params['extrinsic'])
                    
                    camera_params[camera_view] = params
                except Exception as e:
                    print(f"Error loading camera parameters for {camera_view}: {e}")
            else:
                print(f"Warning: Camera parameters file not found for {camera_view}: {params_file}")
                print(f"3D triangulation will not be available for this camera.")
        
        # Load the unified camera system using CameraSystem
        try:
            camera_system_file = os.path.join(self.camera_params_dir, "/nas/project_data/B1_Behavior/rush/kaan/hoi/camera_params/camera.json")
            if os.path.exists(camera_system_file):
                with open(camera_system_file, 'r') as f:
                    camera_data = json.load(f)
                self.camera_system = CameraSystem.from_json(camera_data)
                print(f"Successfully loaded camera system with {len(self.camera_system.camera_poses)} camera poses")
            else:
                print(f"Warning: Camera system file not found: {camera_system_file}")
                self.camera_system = None
        except Exception as e:
            print(f"Error loading camera system: {e}")
            self.camera_system = None
        
        return camera_params
    
    def get_frame_ids(self):
        """Get all available frame IDs across all camera views."""
        all_frame_ids = set()
        
        for camera_view in self.camera_views:
            masks_dir = os.path.join(self.base_dir, self.session_name, camera_view, 
                                    self.object_name, 'masks')
            if os.path.exists(masks_dir):
                frame_dirs = [d for d in os.listdir(masks_dir) 
                             if os.path.isdir(os.path.join(masks_dir, d)) and d.startswith('frame_')]
                all_frame_ids.update(frame_dirs)
        
        # Sort frame IDs to process them in order
        return sorted(all_frame_ids)
    
    def process_frame(self, frame_id):
        """
        Process a single frame across all camera views.
        
        Args:
            frame_id: Frame ID (e.g., 'frame_0001')
            
        Returns:
            Dictionary mapping object index to merged results.
        """
        # Collect data from all camera views
        views_data = {}
        
        for camera_view in self.camera_views:
            base_path = os.path.join(self.base_dir, self.session_name, camera_view)
            frame_data = read_frame_data(base_path, frame_id, self.object_name)
            if frame_data:
                views_data[camera_view] = frame_data
        
        if not views_data:
            return {}
        
        # For each detected object (by index), collect scores and masks from all views
        result = {}
        all_indices = set()
        for view_data in views_data.values():
            all_indices.update(view_data.keys())
        
        # Prepare camera parameters for triangulation
        if self.merge_strategy == 'triangulate':
            if self.camera_system is not None:
                # Use the new CameraSystem class for triangulation
                camera_params_list = {
                    'camera_system': self.camera_system,
                    'views': self.camera_views
                }
            else:
                # Fallback to traditional camera parameters
                camera_params_list = [
                    self.camera_params[camera_view]
                    for camera_view in self.camera_views
                    if camera_view in self.camera_params
                ]
        else:
            camera_params_list = None
        
        for idx in all_indices:
            # Collect scores and masks for this index across views
            scores = []
            masks = []
            camera_views_with_data = []
            
            for camera_view, view_data in views_data.items():
                if idx in view_data:
                    score, mask = view_data[idx]
                    scores.append(score)
                    masks.append(mask)
                    camera_views_with_data.append(camera_view)
            
            if scores and masks:
                if self.merge_strategy == 'triangulate':
                    if self.camera_system is not None:
                        # Use the new CameraSystem class for triangulation
                        filtered_camera_params = {
                            'camera_system': self.camera_system,
                            'views': camera_views_with_data
                        }
                    else:
                        # Fallback to traditional camera parameters
                        filtered_camera_params = [
                            self.camera_params[view]
                            for view in camera_views_with_data
                            if view in self.camera_params
                        ]
                    
                    # Merge masks using triangulation
                    merged_result = merge_masks(masks, scores, filtered_camera_params, self.merge_strategy)
                
                result[idx] = merged_result
        
        return result
    
    def save_frame_results(self, frame_id, merged_data):
        """
        Save merged results for a frame.
        
        Args:
            frame_id: Frame ID (e.g., 'frame_0001')
            merged_data: Dictionary mapping object index to merged results.
        """
        if not merged_data:
            return
        
        # Create output directories for this frame
        frame_masks_dir = os.path.join(self.output_object_dir, 'masks', frame_id)
        frame_scores_dir = os.path.join(self.output_object_dir, 'scores', frame_id)
        frame_3d_dir = os.path.join(self.output_object_dir, '3d_points', frame_id)
        
        os.makedirs(frame_masks_dir, exist_ok=True)
        os.makedirs(frame_scores_dir, exist_ok=True)
        os.makedirs(frame_3d_dir, exist_ok=True)
        
        # Save all data to a single JSON file for each frame
        all_data = {}
        
        # Save individual results based on the merge strategy
        for idx, merged_result in merged_data.items():
            # If we have 3D points, save them
            if 'points3d' in merged_result:
                # Save 3D points as JSON
                points_path = os.path.join(frame_3d_dir, f'points3d_{idx}.json')
                with open(points_path, 'w') as f:
                    json.dump({
                        'points': merged_result['points3d'].tolist(),
                        'score': merged_result['score']
                    }, f, cls=NumpyEncoder, indent=2)
                
                # Add to all_data
                all_data[str(idx)] = {
                    'points3d': merged_result['points3d'].tolist(),
                    'score': merged_result['score']
                }
                
                # Also save as numpy array for easy loading
                np.save(os.path.join(frame_3d_dir, f'points3d_{idx}.npy'), 
                       merged_result['points3d'])
                
                # Write score to score file
                score_path = os.path.join(frame_scores_dir, f'score_{idx}.txt')
                with open(score_path, 'w') as f:
                    f.write(str(merged_result['score']))
            
            # If we have a 2D mask, save it
            elif 'mask' in merged_result:
                # Save mask
                mask_path = os.path.join(frame_masks_dir, f'mask_{idx}.rle')
                save_rle(merged_result['mask'], mask_path)
                
                # Save score
                score_path = os.path.join(frame_scores_dir, f'score_{idx}.txt')
                with open(score_path, 'w') as f:
                    f.write(str(merged_result['score']))
                
                # Add to all_data
                all_data[str(idx)] = {
                    'mask': merged_result['mask'],
                    'score': merged_result['score']
                }
        
        # Save all data JSON
        all_data_path = os.path.join(frame_3d_dir if self.merge_strategy == 'triangulate' else frame_masks_dir, 
                                    'all_data.json')
        with open(all_data_path, 'w') as f:
            json.dump(all_data, f, cls=NumpyEncoder, indent=2)
    
    def process_all_frames(self):
        """Process all frames and merge predictions."""
        frame_ids = self.get_frame_ids()
        print(f"Found {len(frame_ids)} frames to process")
        
        for frame_id in tqdm(frame_ids):
            merged_data = self.process_frame(frame_id)
            self.save_frame_results(frame_id, merged_data)
            
        print(f"Processed {len(frame_ids)} frames")
        print(f"Results saved to {self.output_object_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge predictions from multiple camera views")
    parser.add_argument("--session", type=str, required=True, help="Session name (e.g., 'imi_session1_2')")
    parser.add_argument("--object", type=str, required=True, help="Object name (e.g., 'AMF1')")
    parser.add_argument("--cameras", type=str, nargs="+", help="Camera views to use (e.g., 'cam_side_l cam_side_r cam_top')")
    parser.add_argument("--base-dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/cnos_results",
                        help="Base directory containing prediction data")
    parser.add_argument("--camera-params-dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/camera_params",
                        help="Directory containing camera parameter files")
    parser.add_argument("--output-dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/multi_view_results",
                        help="Directory to save merged results")
    parser.add_argument("--strategy", type=str, default="triangulate", 
                        choices=["triangulate"],
                        help="Strategy to merge masks")
    
    args = parser.parse_args()
    
    merger = MultiViewMerger(
        session_name=args.session,
        object_name=args.object,
        camera_views=args.cameras,
        base_dir=args.base_dir,
        camera_params_dir=args.camera_params_dir,
        output_dir=args.output_dir,
        merge_strategy=args.strategy
    )
    
    merger.process_all_frames()


if __name__ == "__main__":
    main()

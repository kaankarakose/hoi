import os
import torch
import argparse
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from multi_object_wrapper import MultiObjectInferenceWrapper  # Import the multi-object wrapper

def get_object_list(objects_dir):
    """Get list of available objects from npz files directory"""
    return [os.path.basename(f) for f in glob.glob(os.path.join(objects_dir, "*")) if os.path.isdir(f)]

def get_video_list(frames_dir):
    """Get list of video sessions"""
    return [os.path.basename(f) for f in glob.glob(os.path.join(frames_dir, "*")) if os.path.isdir(f)]

def get_camera_views(video_path):
    """Get list of camera views for a video"""
    return [os.path.basename(f) for f in glob.glob(os.path.join(video_path, "*")) if os.path.isdir(f)]

def get_frame_types(camera_path):
    """Get list of frame types (L_frames, R_frames) for a camera view"""
    return [os.path.basename(f) for f in glob.glob(os.path.join(camera_path, "*")) if os.path.isdir(f) and os.path.basename(f) in ['L_frames', 'R_frames']]

def process_video_session(video_name, video_path, objects_dir, objects_list, output_base_dir, conf_threshold=0.1, camera_filters=None):
    """Process a single video session with all camera views, frame types, and all objects"""
    print(f"Processing video session: {video_name}")
    
    # Get camera views
    camera_views = get_camera_views(video_path)
    print(f"Found camera views: {camera_views}")

    for camera_view in camera_views:
        # Skip camera views if they're filtered out
        if camera_filters and camera_view not in camera_filters:
            print(f"Skipping camera view: {camera_view}")
            continue
            
        camera_path = os.path.join(video_path, camera_view)
        print(f'{camera_path} is processing')
        # Check if the camera has L_frames and R_frames folders
        frame_types = get_frame_types(camera_path)
        print(f"Found frame types: {frame_types}")
        
        if 'L_frames' in frame_types or 'R_frames' in frame_types:
            # Process each frame type separately
            for frame_type in frame_types:
                if frame_type in ['L_frames', 'R_frames']:
                    frames_path = os.path.join(camera_path, frame_type)
                    #print(f'Processing {frame_type} in {camera_view}')
                    process_frames_folder_multi_object(video_name, camera_view, frame_type, frames_path, objects_dir, 
                                                    objects_list, output_base_dir, conf_threshold)
        else:
            raise ValueError("Camera view must have L_frames or R_frames folder")

def process_frames_folder_multi_object(video_name, camera_view, frame_type, frames_path, objects_dir, objects_list, 
                                     output_base_dir, conf_threshold, specific_objects=None):
    """Process frames in a specific frame type folder with multiple objects at once"""
    print(f"Processing frames folder: {frame_type} in {camera_view}")
    
    try:
        # Get frames for this folder
        frames = [os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))]
        print(f"Found {len(frames)} frames in {frames_path}")
        if len(frames) > 0:
            print(f"First frame example: {frames[0]}")
        
        if not frames:
            raise ValueError("No frames found in frames folder")
        
        # Filter objects if specific ones are requested
        if specific_objects:
            filtered_objects = [obj for obj in objects_list if obj in specific_objects]
            if not filtered_objects:
                print(f"Warning: None of the specified objects {specific_objects} found in objects list")
                return
            objects_to_process = filtered_objects
        else:
            objects_to_process = objects_list
        
        # Create template paths dictionary for all objects to process
        template_paths = {}
        for object_name in objects_to_process:
            template_path = os.path.join(objects_dir, object_name, 'ref_feats.npz')
            if os.path.exists(template_path):
                template_paths[object_name] = template_path
            else:
                print(f"Template not found for {object_name}: {template_path}")
        
        if not template_paths:
            print("No valid templates found")
            return
        
        # Create base output directory for this session/camera/frame_type
        base_output_dir = os.path.join(output_base_dir, video_name, camera_view, frame_type)
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Create a single wrapper for all objects
        print(f"Processing {len(template_paths)} objects: {', '.join(template_paths.keys())}")
        
        # Initialize the wrapper
        wrapper = MultiObjectInferenceWrapper(
            conf_threshold=conf_threshold,
            output_dir=base_output_dir,
            gpu_id=torch.cuda.current_device(),
            session_name=video_name,
            camera_view=camera_view,
            frame_type=frame_type
        )
        
        # Load templates for all objects at once
        wrapper.load_templates(template_paths)
        
        # Process all frames for all objects
        for frame in tqdm(frames, desc=f"Processing {video_name}/{camera_view}/{frame_type}"):
            frame_info = os.path.basename(frame).split('.')[0]
            
            # Check if all objects have been processed for this frame
            all_processed = True
            for obj_name in template_paths.keys():
                obj_mask_dir = os.path.join(base_output_dir, obj_name, 'masks')
                if not os.path.exists(obj_mask_dir):
                    all_processed = False
                    break
                    
                obj_mask_files = glob.glob(os.path.join(obj_mask_dir, f"{frame_info}*"))
                if not obj_mask_files:
                    all_processed = False
                    break
            
            if all_processed:
                # Skip if all objects have been processed for this frame
                continue
            
            try:
                # Process frame for all objects
                detections = wrapper.process_frame(frame, custom_conf_threshold=conf_threshold)
                
                # Clear CUDA cache after each frame to help manage memory
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing frame {frame_info}: {e}")
                continue
        
        # Clean up
        torch.cuda.empty_cache()

    
    except Exception as e:
        print(f"Error processing frames folder {frame_type}: {str(e)}")
        import traceback
        traceback.print_exc()




def main():
    parser = argparse.ArgumentParser(description="Run multi-object detection with CNOS across multiple videos, cameras, and objects")
    parser.add_argument("--frames_dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/hand_detections",
                        help="Directory containing original video frames")
    parser.add_argument("--objects_dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/objects_dino_vectors",
                        help="Directory containing object template npz files")
    parser.add_argument("--output_dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/multi_cnos_result",
                        help="Base output directory for results")

    parser.add_argument("--session", type=str, default=None, help="Video session to process")
    parser.add_argument("--camera_view", type=str, default=None, help="Camera view to process")
    parser.add_argument("--object_name", type=str, default=None, help="Object to process (can specify multiple with comma-separated list)")

    parser.add_argument("--conf_threshold", type=float, default=0.10,
                        help="Confidence threshold for detections")
    parser.add_argument("--num_splits", type=int, default=1,
                        help="Number of parts to split the video dataset into")
    parser.add_argument("--split_id", type=int, default=0,
                        help="Which split to process (0-indexed, must be less than num_splits)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for processing")
    parser.add_argument("--camera_filters", type=str, default="cam_top,cam_side_r,cam_side_l",
                        help="Comma-separated list of camera views to process (default: cam_top,cam_side_r)")
    args = parser.parse_args()
    
    # Validate split parameters
    if args.split_id >= args.num_splits:
        raise ValueError(f"Split ID ({args.split_id}) must be less than num_splits ({args.num_splits})")
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("CUDA not available, using CPU")
    
    # Get list of objects
    objects_list = sorted(get_object_list(args.objects_dir))
    print(f"Found {len(objects_list)} objects: {objects_list}")
    
    # Get list of videos
    all_videos_list = sorted(get_video_list(args.frames_dir))
    print(f"Found {len(all_videos_list)} videos in total")

    # Parse camera filters
    camera_filters = args.camera_filters.split(',') if args.camera_filters else None
    print(f"Using camera filters: {camera_filters}")

    # Parse specific objects if provided
    specific_objects = args.object_name.split(',') if args.object_name else None
    if specific_objects:
        print(f"Processing only specific objects: {specific_objects}")

    # Split videos into parts

    videos_per_split = len(all_videos_list) // args.num_splits  # This will be 2 (24 ÷ 12)
    start_idx = args.split_id * videos_per_split
    end_idx = start_idx + videos_per_split if args.split_id < args.num_splits - 1 else len(all_videos_list)
    videos_list = all_videos_list[start_idx:end_idx]



    
    
    for video_name in videos_list:
        video_path = os.path.join(args.frames_dir, video_name)
  
        process_video_session(video_name, video_path, args.objects_dir, objects_list, args.output_dir, 
                            args.conf_threshold, camera_filters)

        

if __name__ == "__main__":
    main()
import os
import torch
import argparse
from tqdm import tqdm
from inference_wrapperv2 import InferenceWrapper
import glob

def get_object_list(objects_dir):
    """Get list of available objects from npz files directory"""
    return [os.path.basename(f) for f in glob.glob(os.path.join(objects_dir, "*")) if os.path.isdir(f)]

def get_video_list(frames_dir):
    """Get list of video sessions"""
    return [os.path.basename(f) for f in glob.glob(os.path.join(frames_dir, "*")) if os.path.isdir(f)]

def get_camera_views(video_path):
    """Get list of camera views for a video"""
    return [os.path.basename(f) for f in glob.glob(os.path.join(video_path, "*")) if os.path.isdir(f)]

def process_video_session(video_name, video_path, objects_dir, objects_list, output_base_dir, conf_threshold=0.1):
    """Process a single video session with all camera views and all objects"""
    print(f"Processing video session: {video_name}")
    
    # Get camera views
    camera_views = get_camera_views(video_path)
    for camera_view in camera_views:

        # Process camera view
        if camera_view in ['cam_top','cam_side_r']: continue
            
        camera_path = os.path.join(video_path, camera_view)
        print(f'{camera_path} is processing')
        process_camera_view(video_name, camera_view, camera_path, objects_dir, objects_list, output_base_dir, conf_threshold)

def process_video_session_seperated(
        video_path,
        session,
        camera_view, 
        object_name,
        objects_dir,
        objects_list,
        output_dir,
        conf_threshold):
    """Process a single video session with specific camera view and object"""
    print(f"Processing video session: {session}")
 
    
    # If camera_view is specified, only process that view
    if camera_view:
        camera_path = os.path.join(video_path, camera_view)
        print(f'Processing camera view: {camera_path}')
        process_camera_view(session,
            camera_view,
            camera_path, 
            objects_dir, 
            objects_list, 
            output_dir, 
            conf_threshold,
            object_name)
    else:
        # Process all camera views if none specified
        camera_views = get_camera_views(video_path)
        for cam_view in camera_views:
            if cam_view in ['cam_side_l','cam_side_r']: continue
            camera_path = os.path.join(video_path, cam_view)
            print(f'Processing camera view: {camera_path}')
            process_camera_view(session,
                cam_view,
                camera_path, 
                objects_dir, 
                objects_list, 
                output_dir, 
                conf_threshold,
                object_name)


def process_camera_view(video_name,
 camera_view,
  camera_path, 
  objects_dir,
  objects_list,
   output_base_dir,
 conf_threshold,
 specific_object = None):
    """Process a single camera view with all objects or a specific object"""
    print(f"Processing camera view: {camera_view}")
    
    # Get frames for this camera view
    frames = [os.path.join(camera_path, f) for f in os.listdir(camera_path) if f.endswith(('.jpg', '.png'))]
    if not frames:
        print(f"No frames found in {camera_path}")
        return
        
    # Process each object
    if specific_object:
        # Process only the specified object
        if specific_object in objects_list:
            print(f"Processing only the specific object: {specific_object}")
            process_object(video_name, camera_view, specific_object, frames, objects_dir, output_base_dir, conf_threshold)
        else:
            print(f"Warning: Object '{specific_object}' not found in objects list")
    else:
        # Process all objects
        for object_name in objects_list:
            process_object(video_name, camera_view, object_name, frames, objects_dir, output_base_dir, conf_threshold)

def process_object(video_name, camera_view, object_name, frames, objects_dir, output_base_dir, conf_threshold):
    """Process a single object for all frames in a camera view"""
    print(f"Processing object: {object_name}")
    
    # Create output directory structure: Video_name -> Camera_view -> object_name
    output_dir = os.path.join(output_base_dir, video_name, camera_view, object_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wrapper with current GPU (already set at the start of the program)
    wrapper = InferenceWrapper(
        conf_threshold=conf_threshold,
        output_dir=output_dir,
        gpu_id=torch.cuda.current_device()
    )
    
    # Load object template
    template_path = os.path.join(objects_dir, object_name, 'ref_feats.npz')
    wrapper.load_templates(template_path)
    
    # Process frames
    for frame in tqdm(frames, desc=f"Processing {object_name} in {video_name}/{camera_view}"):
        detections, scores = wrapper.run(frame)
        if detections is None:
            print('NO MASK')
            continue
        # Get masks and scores
        masks = wrapper.get_masks(detections)
        scores = wrapper.get_scores(detections)
        # Extract frame info for saving
        frame_info = os.path.basename(frame).split('.')[0]
        
        # Save results

        masked_saved = wrapper.save_masks(detections, frame_info)
        bbox_saved = wrapper.save_bbox(detections, object_name, frame_info)
        scores_saved = wrapper.save_scores(scores, frame_info)
        # Clear CUDA cache
        torch.cuda.empty_cache()         


def main():
    parser = argparse.ArgumentParser(description="Run object detection with CNOS across multiple videos, cameras, and objects")
    parser.add_argument("--frames_dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/orginal_frames",
                        help="Directory containing original video frames")
    parser.add_argument("--objects_dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/objects_dino_vectors",
                        help="Directory containing object template npz files")
    parser.add_argument("--output_dir", type=str, default="/nas/project_data/B1_Behavior/rush/kaan/hoi/processed_data/cnos_results",
                        help="Base output directory for results")

    parser.add_argument("--session", type = str, default = None, help="Video session to process")
    parser.add_argument("--camera_view", type = str, default = None, help="Camera view to process")
    parser.add_argument("--object_name", type = str, default = None, help="Object to process")

    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--num_splits", type=int, default=1,
                        help="Number of parts to split the video dataset into")
    parser.add_argument("--split_id", type=int, default=0,
                        help="Which split to process (0-indexed, must be less than num_splits)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for processing")
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
    objects_list = get_object_list(args.objects_dir)
    print(f"Found {len(objects_list)} objects")
    
    # Get list of videos
    all_videos_list = get_video_list(args.frames_dir)
    print(f"Found {len(all_videos_list)} videos in total")


    # Split videos into parts
    if args.num_splits > 1:
        videos_per_split = len(all_videos_list) // args.num_splits
        start_idx = args.split_id * videos_per_split
        end_idx = start_idx + videos_per_split if args.split_id < args.num_splits - 1 else len(all_videos_list)
        videos_list = all_videos_list[start_idx:end_idx]
        print(f"Processing split {args.split_id + 1}/{args.num_splits}: {len(videos_list)} videos (from {start_idx} to {end_idx-1})")
    else:
        videos_list = all_videos_list
        print(f"Processing all {len(videos_list)} videos")

    if args.session is None:
        # Process each video in this split
        for video_name in videos_list:
            video_path = os.path.join(args.frames_dir, video_name)
            process_video_session(video_name, video_path, args.objects_dir, objects_list, args.output_dir, args.conf_threshold)
    else:
        # Process specific session, camera view, and object
        print(f"Processing specific session: {args.session}")
        if args.camera_view:
            print(f"    with camera view: {args.camera_view}")
        if args.object_name:
            print(f"    with object: {args.object_name}")
        video_path = os.path.join(args.frames_dir, args.session)
        process_video_session_seperated(
            video_path,
            args.session,
            args.camera_view, 
            args.object_name,
            args.objects_dir,
            objects_list,
            args.output_dir,
            args.conf_threshold)


if __name__ == "__main__":
    main()

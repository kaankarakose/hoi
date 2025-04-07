import os
import subprocess
from pathlib import Path
import cv2
import os

def extract_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    frame_count = 0
    
    while True:
        # Read a frame
        success, frame = video.read()
        
        # Break if no frame was read
        if not success:
            break
        
        # Save the frame
        output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(output_path, frame)
        
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Release video object
    video.release()
    
    print(f"Extraction complete. Saved {frame_count} frames to {output_dir}")


if __name__ == "__main__":
    # Example usage
    video_path = "/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/amf1.mp4"  # Replace with your video path
    output_dir = "/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/amf1_frames"    # Replace with your desired output directory
    
    extract_frames(video_path, output_dir)
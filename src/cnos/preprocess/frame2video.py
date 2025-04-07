import cv2
import os
import glob
import argparse

import re

def natural_sort(file_list):
    """Sort file names naturally (so frame_10 comes after frame_9, not after frame_1)"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    
    return sorted(file_list, key=alphanum_key)

# Then replace all instances of natsorted(image_files) with natural_sort(image_files)

def frames_to_video(input_dir, output_file, fps=30, codec='mp4v'):
    """
    Convert a directory of image frames to a video file using OpenCV.
    
    Args:
        input_dir (str): Directory containing the image frames
        output_file (str): Path for the output video file
        fps (int): Frames per second for the output video
        codec (str): FourCC codec code (e.g., 'mp4v', 'avc1', 'XVID')
    """
    # Get list of image files and sort them naturally
    # This handles frame_1.png, frame_2.png, ..., frame_10.png correctly
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    if not image_files:
        image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    # Sort files naturally (so frame_10 comes after frame_9, not after frame_1)
    image_files = natural_sort(image_files)
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        raise ValueError(f"Could not read image file: {image_files[0]}")
    
    height, width, channels = first_image.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Process each image
    total_frames = len(image_files)
    for i, image_file in enumerate(image_files):
        print(f"Processing frame {i+1}/{total_frames}: {os.path.basename(image_file)}")
        
        # Read image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read {image_file}, skipping")
            continue
            
        # Write frame to video
        video_writer.write(image)
    
    # Release resources
    video_writer.release()
    print(f"Video saved to {output_file}")
    print(f"Video properties: {width}x{height}, {fps} fps, {len(image_files)} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image frames to video")
    parser.add_argument("--input_dir", required=True, help="Directory containing image frames")
    parser.add_argument("--output_file", required=True, help="Output video file path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec code (default: mp4v)")
    
    args = parser.parse_args()
    
    frames_to_video(args.input_dir, args.output_file, args.fps, args.codec)
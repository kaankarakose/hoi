import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add necessary paths to import HAMERWrapper
project_path = Path('/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt')
hand_object_path = project_path / 'hand-object'
if str(hand_object_path) not in sys.path:
    sys.path.append(str(hand_object_path))

from hand.HAMERWrapper import HAMERWrapper


class HandImageCropper:
    """Class for cropping images based on hand detections using HAMERWrapper"""
    
    def __init__(self, device=None):
        """Initialize the cropper with HAMERWrapper
        
        Args:
            device: Device to run HAMER on (None for auto-detection)
        """
        self.hamer = HAMERWrapper(device=device)
        self.padding = 20  # Default padding around hands in pixels
    
    def set_padding(self, padding):
        """Set padding around hand bounding boxes
        
        Args:
            padding: Padding in pixels
        """
        self.padding = padding
    
    def crop_image(self, image, padding=None):
        """Crop image to include both hands and space between them
        
        Args:
            image: Input image (BGR format from cv2.imread)
            padding: Optional override for padding around hands
        
        Returns:
            cropped_image: Cropped image containing hands
            bbox: The bounding box used for cropping [x1, y1, x2, y2]
        """
        if padding is not None:
            old_padding = self.padding
            self.padding = padding
        
        # Process frame with HAMERWrapper to detect hands
        success = self.hamer.process_frame(image)
        
        if not success:
            print("No hands detected in the image.")
            if padding is not None:
                self.padding = old_padding
            return image, None
        
        # Get hand objects
        hands = self.hamer.getHands()
        left_hand = hands['left']
        right_hand = hands['right']
        
        # Determine combined bounding box
        bbox = self._get_combined_bbox(left_hand, right_hand, image.shape)
        
        if bbox is None:
            print("Could not determine valid bounding box.")
            if padding is not None:
                self.padding = old_padding
            return image, None
        
        # Crop the image using the bounding box
        x1, y1, x2, y2 = bbox
        cropped_image = image[y1:y2, x1:x2]
        
        # Restore original padding if needed
        if padding is not None:
            self.padding = old_padding
        
        return cropped_image, bbox
    
    def _get_combined_bbox(self, left_hand, right_hand, image_shape):
        """Calculate a bounding box that includes both hands and space between them
        
        Args:
            left_hand: Left hand object from HAMERWrapper
            right_hand: Right hand object from HAMERWrapper
            image_shape: Shape of the original image
        
        Returns:
            bbox: Combined bounding box [x1, y1, x2, y2] or None if invalid
        """
        height, width = image_shape[:2]
        
        # If only one hand is detected, use its bounding box with padding
        if left_hand is None and right_hand is None:
            return None
        elif left_hand is None:
            bbox = right_hand.boundingbox()
        elif right_hand is None:
            bbox = left_hand.boundingbox()
        else:
            # Both hands detected, create a combined bbox
            left_bbox = left_hand.boundingbox()
            right_bbox = right_hand.boundingbox()
            
            # Calculate combined bounding box coordinates
            bbox = [
                min(left_bbox[0], right_bbox[0]),  # x1 (left)
                min(left_bbox[1], right_bbox[1]),  # y1 (top)
                max(left_bbox[2], right_bbox[2]),  # x2 (right)
                max(left_bbox[3], right_bbox[3])   # y2 (bottom)
            ]
        
        # Add padding and ensure coordinates are within image bounds
        x1 = max(0, bbox[0] - self.padding)
        y1 = max(0, bbox[1] - self.padding)
        x2 = min(width, bbox[2] + self.padding)
        y2 = min(height, bbox[3] + self.padding)
        
        return [int(x1), int(y1), int(x2), int(y2)]


def crop_video_frames(video_path, output_dir, frame_step=1, device=None):
    """Process a video and save cropped frames
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save cropped frames
        frame_step: Process every Nth frame
        device: Device to run on
    
    Returns:
        List of saved frame paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize cropper
    cropper = HandImageCropper(device=device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frames
    frame_idx = 0
    saved_frames = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process every Nth frame
        if frame_idx % frame_step == 0:
            # Crop image
            cropped_frame, bbox = cropper.crop_image(frame)
            
            if bbox is not None:
                # Save cropped frame
                output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(output_path, cropped_frame)
                saved_frames.append(output_path)
                
                # Print progress
                print(f"Processed frame {frame_idx}/{total_frames} - Saved to {output_path}")
        
        frame_idx += 1
    
    cap.release()
    print(f"Processed {len(saved_frames)}/{total_frames//frame_step} frames")
    return saved_frames


def process_image_folder(input_dir, output_dir, padding=20, device=None):
    """Process all image frames in a folder
    
    Args:
        input_dir: Path to directory containing image frames
        output_dir: Directory to save cropped frames
        padding: Padding around hands in pixels
        device: Device to run on
    
    Returns:
        List of saved frame paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize cropper
    cropper = HandImageCropper(device=device)
    cropper.set_padding(padding)
    
    # Get all image files in the directory
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.isfile(os.path.join(input_dir, f)) 
                  and f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return []
    
    # Sort files to ensure consistent processing order
    image_files.sort()
    
    # Process each image
    saved_frames = []
    total_images = len(image_files)
    
    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not read image: {input_path}")
            continue
        
        # Crop image
        cropped_image, bbox = cropper.crop_image(image)
        
        if bbox is not None:
            # Preserve original filename but save to output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cropped_image)
            saved_frames.append(output_path)
            
            # Print progress
            print(f"Processed image {i+1}/{total_images} - Saved to {output_path}")
        else:
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
            saved_frames.append(output_path)
            print(f"No hands detected in {filename}")
    
    print(f"Processed {len(saved_frames)}/{total_images} images")
    return saved_frames


def demo():
    """Demo function showing how to use the cropper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crop images/videos based on hand detections")
    parser.add_argument("--input", "-i", required=True, help="Input image file, video file, or directory of frames")
    parser.add_argument("--output", "-o", required=True, help="Output directory or image file")
    parser.add_argument("--padding", "-p", type=int, default=20, help="Padding around hands in pixels")
    parser.add_argument("--frame-step", "-s", type=int, default=1, help="Process every Nth frame (videos only)")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    padding = args.padding
    frame_step = args.frame_step
    
    # Check if input is a directory
    if os.path.isdir(input_path):
        # Process all images in the directory
        process_image_folder(input_path, output_path, padding=padding)
    
    # Check if input is an image file
    elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        # Process single image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not read image: {input_path}")
            return
        
        cropper = HandImageCropper()
        cropper.set_padding(padding)
        cropped_image, bbox = cropper.crop_image(image)
        
        if bbox is not None:
            cv2.imwrite(output_path, cropped_image)
            print(f"Saved cropped image to {output_path}")
            print(f"Bounding box: {bbox}")
        else:
            print("No hands detected or invalid bounding box.")
    
    # Check if input is a video file
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video
        crop_video_frames(input_path, output_path, frame_step=frame_step)
    
    else:
        print(f"Unsupported input: {input_path}")


if __name__ == "__main__":
    demo()

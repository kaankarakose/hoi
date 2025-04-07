

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2
from pathlib import Path
import argparse


import numpy as np
import cv2
import json

def rle_to_mask(rle):
    """
    Convert RLE format to binary mask.
    
    Args:
        rle (dict): RLE representation with 'counts' and 'size' keys.
        
    Returns:
        numpy.ndarray: Binary mask array where 1 is the foreground.
    """
    # Get the mask shape from RLE
    height, width = rle['size']
    
    # Initialize an empty mask
    mask_flat = np.zeros(height * width, dtype=np.uint8)
    
    # Current position in the flattened mask
    current_pos = 0
    
    # Toggle between 0 and 1 (we start with 0)
    current_val = 0
    
    # Fill the mask according to the RLE counts
    for count in rle['counts']:
        # Fill with current value
        mask_flat[current_pos:current_pos + count] = current_val
        
        # Update position
        current_pos += count
        
        # Toggle between 0 and 1
        current_val = 1 - current_val
    
    # Reshape the mask to the original dimensions
    mask = mask_flat.reshape((height, width), order='F')  # Use Fortran order (column-major)
    
    return mask


def read_rle_file(file_path):
    """
    Read RLE data from a file and convert it to a grayscale mask.
    
    Args:
        file_path (str): Path to the RLE file.
    
    Returns:
        numpy.ndarray: Grayscale mask (8-bit, single channel).
    """
    # Read the RLE file
    with open(file_path, 'r') as f:
        rle_data = json.load(f)
    
    # Convert RLE to mask
    mask = rle_to_mask(rle_data)
    
    return mask


def read_rle_file_as_grayscale(file_path):
    """
    Read RLE data from a file and convert it to a grayscale mask (0 and 255).
    
    Args:
        file_path (str): Path to the RLE file.
        
    Returns:
        numpy.ndarray: Grayscale mask (8-bit, single channel) where 255 is foreground.
    """
    # Get binary mask (0s and 1s)
    binary_mask = read_rle_file(file_path)
    
    # Convert to grayscale (0s and 255s)
    grayscale_mask = binary_mask * 255
    
    return grayscale_mask





class MaskScoreVisualizer:
    """
    Visualizer for segmentation masks and their corresponding scores.
    This class manages the visualization of frame-wise segmentation masks and
    the corresponding confidence scores.
    """
    def __init__(self, 
                 masks_root_dir, 
                 scores_root_dir, 
                 frames_dir=None, 
                 output_dir=None):
        """
        Initialize the visualizer.
        
        Args:
            masks_root_dir (str): Root directory containing mask folders (frame_XXXX)
            scores_root_dir (str): Root directory containing score folders (frame_XXXX)
            frames_dir (str, optional): Directory containing original frames for overlay
            output_dir (str, optional): Directory to save visualization results
        """
        self.masks_root_dir = masks_root_dir
        self.scores_root_dir = scores_root_dir
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Scan for available frames
        self.frame_dirs = sorted(glob.glob(os.path.join(masks_root_dir, "frame_*")))
        self.num_frames = len(self.frame_dirs)
        
        if self.num_frames == 0:
            raise ValueError(f"No frame directories found in {masks_root_dir}")
            
        print(f"Found {self.num_frames} frames")
        
        # Colors for visualization
        self.colors = self._generate_color_palette()
        
    def _generate_color_palette(self, num_colors=30):
        """
        Generate a colorful palette for mask visualization.
        
        Returns:
            List of BGR color tuples
        """
        np.random.seed(42)  # For reproducibility
        colors = []
        for i in range(num_colors):
            # Avoid too dark or too light colors
            color = np.random.randint(50, 200, size=3).tolist()
            colors.append(color)
        return colors
    
    def load_mask_data(self, frame_idx):
        """
        Load masks and scores for a specific frame.
        
        Args:
            frame_idx (int): Index of the frame to load
            
        Returns:
            dict: Dictionary containing masks and scores
        """
        frame_name = f"frame_{frame_idx:04d}"
        mask_dir = os.path.join(self.masks_root_dir, frame_name)
        score_dir = os.path.join(self.scores_root_dir, frame_name)
        
        if not os.path.exists(mask_dir) or not os.path.exists(score_dir):
            raise ValueError(f"Mask or score directory not found for {frame_name}")
        
        # Load masks
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "mask_*.rle"))) ##


        #masks = [cv2.imread(read_rle_file(mask_file), cv2.IMREAD_GRAYSCALE) for mask_file in mask_files]
        masks = [read_rle_file_as_grayscale(mask_file) for mask_file in mask_files]
        # Load scores
        score_files = sorted(glob.glob(os.path.join(score_dir, "score_*.txt")))
        scores = []
        for score_file in score_files:
            with open(score_file, 'r') as f:
                score = float(f.read().strip())
                scores.append(score)
                
        # Check if number of masks and scores match
        if len(masks) != len(scores):
            print(f"Warning: Number of masks ({len(masks)}) does not match number of scores ({len(scores)})")
            
        # Create result dictionary
        result = {
            "frame_name": frame_name,
            "masks": masks,
            "scores": scores,
            "mask_files": mask_files,
            "score_files": score_files
        }
        
        return result
    
    def visualize_frame(self, frame_idx, threshold=0.0, alpha=0.7, show_scores=True):
        """
        Visualize masks and scores for a specific frame.
        
        Args:
            frame_idx (int): Index of the frame to visualize
            threshold (float): Score threshold for mask visualization
            alpha (float): Transparency factor for mask overlay
            show_scores (bool): Whether to display score values
            
        Returns:
            numpy.ndarray: Visualization image
        """
        # Load data for this frame
        data = self.load_mask_data(frame_idx)
        masks = data["masks"]
        scores = data["scores"]
        frame_name = data["frame_name"]
        
        # Load original frame if available
        if self.frames_dir:
            frame_path = os.path.join(self.frames_dir, f"{frame_name}.jpg")
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
            else:
                # Create a blank canvas
                if len(masks) > 0:
                    h, w = masks[0].shape
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Create a blank canvas
            if len(masks) > 0:
                h, w = masks[0].shape
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
        # Create a copy for visualization
        vis_img = frame.copy()
        
        # Overlay masks based on threshold
        valid_masks = 0
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score >= threshold:
                valid_masks += 1
                # Convert binary mask to BGR
                color_mask = np.zeros_like(frame)
                color_idx = i % len(self.colors)
                color = self.colors[color_idx]
                
                # Apply color to mask regions
                color_mask[mask > 0] = color
                
                # Overlay mask on visualization image
                vis_img = cv2.addWeighted(vis_img, 1.0, color_mask, alpha, 0)
                
                # Add score text if requested
                if show_scores:
                    # Find centroid of mask
                    moments = cv2.moments(mask)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        
                        # Add score text
                        score_str = f"{score:.2f}"
                        cv2.putText(vis_img, score_str, (cx, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(vis_img, f"Frame: {frame_name} | Masks: {valid_masks}/{len(masks)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_img
    
    def interactive_visualization(self):
        """
        Launch an interactive matplotlib visualization.
        This allows scrolling through frames and adjusting visualization parameters.
        """
        # Initialize figure
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Initial frame and parameters
        frame_idx = 0
        threshold = 0.0
        alpha = 0.7
        
        # Initial visualization
        vis_img = self.visualize_frame(frame_idx, threshold, alpha)
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        img_plot = ax.imshow(vis_img_rgb)
        
        # Frame slider
        ax_frame = plt.axes([0.25, 0.15, 0.65, 0.03])
        frame_slider = Slider(
            ax=ax_frame,
            label='Frame',
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=frame_idx,
            valstep=1
        )
        
        # Threshold slider
        ax_threshold = plt.axes([0.25, 0.1, 0.65, 0.03])
        threshold_slider = Slider(
            ax=ax_threshold,
            label='Score Threshold',
            valmin=0.0,
            valmax=1.0,
            valinit=threshold,
        )
        
        # Alpha slider
        ax_alpha = plt.axes([0.25, 0.05, 0.65, 0.03])
        alpha_slider = Slider(
            ax=ax_alpha,
            label='Mask Transparency',
            valmin=0.0,
            valmax=1.0,
            valinit=alpha,
        )
        
        # Update function
        def update(val):
            current_frame = int(frame_slider.val)
            current_threshold = threshold_slider.val
            current_alpha = alpha_slider.val
            
            new_vis_img = self.visualize_frame(
                current_frame, 
                current_threshold, 
                current_alpha
            )
            
            new_vis_img_rgb = cv2.cvtColor(new_vis_img, cv2.COLOR_BGR2RGB)
            img_plot.set_data(new_vis_img_rgb)
            fig.canvas.draw_idle()
        
        # Register the update function with each slider
        frame_slider.on_changed(update)
        threshold_slider.on_changed(update)
        alpha_slider.on_changed(update)
        
        plt.show()
        
    def visualize_all_frames(self, threshold=0.0, output_dir=None, show_progress=True):
        """
        Process all frames and save visualizations.
        
        Args:
            threshold (float): Score threshold for visualization
            output_dir (str, optional): Directory to save results
            show_progress (bool): Whether to show progress bar
        """
        if output_dir is None:
            if self.output_dir is None:
                raise ValueError("Output directory must be specified")
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        total_frames = self.num_frames
        for i in range(total_frames):
            if show_progress and i % 10 == 0:
                print(f"Processing frame {i+1}/{total_frames}")
            
            try:
                vis_img = self.visualize_frame(i, threshold)
                output_path = os.path.join(output_dir, f"visualization_{i:04d}.jpg")
                cv2.imwrite(output_path, vis_img)
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
        
        print(f"All visualizations saved to {output_dir}")
    
    def create_video(self, output_path, fps=15, threshold=0.0):
        """
        Create a video from all frame visualizations.
        
        Args:
            output_path (str): Path to save the output video
            fps (int): Frames per second for the video
            threshold (float): Score threshold for visualization
        """
        # Create temporary directory for frames
        temp_dir = os.path.join(self.output_dir if self.output_dir else ".", "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate all frame visualizations
        print("Generating frame visualizations...")
        for i in range(self.num_frames):
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{self.num_frames}")
            
            try:
                vis_img = self.visualize_frame(i, threshold)
                output_frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(output_frame_path, vis_img)
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
        
        # Create video from frames
        print("Creating video...")
        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.jpg")))
        if not frame_files:
            raise ValueError("No frames generated for video")
        
        # Get frame dimensions from first image
        first_frame = cv2.imread(frame_files[0])
        h, w, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Add frames to video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video_writer.write(frame)
        
        # Release resources
        video_writer.release()
        
        # Clean up temporary directory
        for frame_file in frame_files:
            os.remove(frame_file)
        os.rmdir(temp_dir)
        
        print(f"Video saved to {output_path}")


def visualize_score_distribution(scores_root_dir, output_path=None):
    """
    Visualize the distribution of scores across all frames.
    
    Args:
        scores_root_dir (str): Root directory containing score folders
        output_path (str, optional): Path to save the visualization
    """
    # Collect all scores
    all_scores = []
    frame_dirs = sorted(glob.glob(os.path.join(scores_root_dir, "frame_*")))
    
    for frame_dir in frame_dirs:
        score_files = glob.glob(os.path.join(frame_dir, "score_*.txt"))
        for score_file in score_files:
            with open(score_file, 'r') as f:
                score = float(f.read().strip())
                all_scores.append(score)
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, alpha=0.75)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Segmentation Scores')
    plt.grid(alpha=0.3)
    
    # Add summary statistics
    if all_scores:
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        stats_text = (f"Mean: {mean_score:.4f}\n"
                     f"Median: {median_score:.4f}\n"
                     f"Min: {min_score:.4f}\n"
                     f"Max: {max_score:.4f}")
        
        plt.figtext(0.15, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.8))
    
    if output_path:
        plt.savefig(output_path)
        print(f"Score distribution saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation masks and scores')
    parser.add_argument('--masks', required=True, help='Root directory containing mask folders')
    parser.add_argument('--scores', required=True, help='Root directory containing score folders')
    parser.add_argument('--frames', help='Directory containing original frames (optional)')
    parser.add_argument('--output', help='Output directory for visualizations')
    parser.add_argument('--threshold', type=float, default=0.0, help='Score threshold for visualization')
    parser.add_argument('--video', help='Create video and save to specified path')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for video')
    parser.add_argument('--stats', action='store_true', help='Show score distribution statistics')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive visualization')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = MaskScoreVisualizer(
        masks_root_dir=args.masks,
        scores_root_dir=args.scores,
        frames_dir=args.frames,
        output_dir=args.output
    )
    
    # Show score distribution if requested
    if args.stats:
        stats_output = os.path.join(args.output, "score_distribution.png") if args.output else None
        visualize_score_distribution(args.scores, stats_output)
    
    # Launch interactive visualization if requested
    if args.interactive:
        visualizer.interactive_visualization()
    
    # Create video if requested
    if args.video:
        visualizer.create_video(args.video, args.fps, args.threshold)
    
    # Process all frames if not interactive and no video requested
    if not args.interactive and not args.video and args.output:
        visualizer.visualize_all_frames(args.threshold)


if __name__ == "__main__":
    main()

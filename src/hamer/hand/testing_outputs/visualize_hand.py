#!/usr/bin/env python3
"""
Script to visualize hand meshes from NPZ files containing HAMER hand tracking data.
"""

import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# Add the parent directory to sys.path for proper imports
sys.path.append(str(Path(__file__).parent.parent))
from HAMERWrapper import HAMERWrapper

# Set correct path to HAMER data
hamer_data_path = '/nas/project_data/B1_Behavior/rush/kaan/hoi/src/hamer/hand/_DATA'
os.environ['HAMER_DATA_DIR'] = hamer_data_path

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils.renderer import Renderer

def visualize_hand_npz(npz_path, output_dir=None, background_color=(255, 255, 255), img_size=(800, 800)):
    """
    Visualize hand mesh from an NPZ file
    
    Args:
        npz_path: Path to the NPZ file containing hand data
        output_dir: Directory to save visualization (None for no saving)
        background_color: RGB color for the background
        img_size: Size of the output image (width, height)
    
    Returns:
        Rendered image of the hand mesh
    """
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the NPZ file
    print(f"Loading hand data from: {npz_path}")
    hand_data = np.load(npz_path, allow_pickle=True)
    
    # Extract hand data
    vertices = hand_data['vertices']
    cam_t = hand_data['cam_t'] 
    is_right = hand_data['is_right']
    bbox = hand_data['bbox']
    
    print(f"Hand data loaded:")
    print(f"- Vertices shape: {vertices.shape}")
    print(f"- Camera translation: {cam_t}")
    print(f"- Is right hand: {is_right}")
    print(f"- Bounding box: {bbox}")
    
    # Load HAMER model to get faces for the mesh
    # We only need the model configuration and faces for rendering
    # Use existing HAMER models from the specified path
    print(f"Using HAMER models from {hamer_data_path}")
    
    # Create a HAMERWrapper instance to get the model and model_cfg
    wrapper = HAMERWrapper()
    model = wrapper.model
    model_cfg = wrapper.model_cfg
    
    # Initialize renderer using the HAMERWrapper's renderer
    renderer = wrapper.renderer
    
    # Create background image
    background = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    bg_color = background_color[::-1]  # Convert RGB to BGR for OpenCV
    background[:] = bg_color
    
    # Convert image size to numpy array for rendering calculations
    img_size_np = np.array([img_size[0], img_size[1]])
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_size_np)
    
    # Define rendering arguments
    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )
    
    # Create list of vertices and camera translations for rendering
    all_verts = [vertices]
    all_cam_t = [cam_t]
    all_right = [1 if is_right else 0]
    
    # Render hand mesh (RGBA)
    print("Rendering hand mesh...")
    cam_view = renderer.render_rgba_multiple(
        all_verts, cam_t=all_cam_t, render_res=img_size_np, is_right=all_right, **misc_args
    )
    
    # Combine with background
    alpha = cam_view[:, :, 3:4]
    rgb = cam_view[:, :, :3]
    rendered_img = rgb * alpha + background[:, :, ::-1] / 255.0 * (1 - alpha)
    
    # Convert to uint8 for display/saving
    rendered_img = (rendered_img * 255).astype(np.uint8)
    
    # Also render a side view for better visualization
    side_view = renderer.render_rgba_multiple(
        all_verts, cam_t=all_cam_t, render_res=img_size_np, is_right=all_right, 
        cam_rot=np.array([0, 90, 0]), **misc_args
    )
    
    # Combine with background
    side_alpha = side_view[:, :, 3:4]
    side_rgb = side_view[:, :, :3]
    side_rendered_img = side_rgb * side_alpha + background[:, :, ::-1] / 255.0 * (1 - side_alpha)
    side_rendered_img = (side_rendered_img * 255).astype(np.uint8)
    
    # Create a combined view (front and side)
    combined_img = np.zeros((img_size[1], img_size[0]*2, 3), dtype=np.uint8)
    combined_img[:, :img_size[0]] = rendered_img[:, :, ::-1]  # Front view on left (BGR for OpenCV)
    combined_img[:, img_size[0]:] = side_rendered_img[:, :, ::-1]  # Side view on right (BGR for OpenCV)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_img, "Front View", (50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_img, "Side View", (img_size[0] + 50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Add hand side label
    hand_side = "Right Hand" if is_right else "Left Hand"
    cv2.putText(combined_img, hand_side, (50, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Save the rendered image if output_dir is specified
    if output_dir is not None:
        output_filename = os.path.basename(npz_path).replace('.npz', '_visualization.png')
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, combined_img)
        print(f"Visualization saved to: {output_path}")
    
    return combined_img

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize hand data from NPZ files")
    parser.add_argument("npz_path", help="Path to the NPZ file containing hand data")
    parser.add_argument("--output-dir", help="Directory to save the visualization", default="./")
    parser.add_argument("--width", type=int, help="Width of output image", default=800)
    parser.add_argument("--height", type=int, help="Height of output image", default=800)
    
    args = parser.parse_args()
    
    # Visualize hand
    visualize_hand_npz(args.npz_path, args.output_dir, img_size=(args.width, args.height))

#!/usr/bin/env python3
"""
Script to visualize hand meshes from NPZ files alongside the original image.
Uses bounding box information to place the hand properly.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json

# Add the parent directory to sys.path for proper imports
sys.path.append(str(Path(__file__).parent.parent))
from hamer.utils.renderer import Renderer

def load_mano_faces(mano_path='/nas/project_data/B1_Behavior/rush/kaan/hoi/src/hamer/hand/_DATA/data/mano/MANO_RIGHT.pkl'):
    """Load faces from MANO pickle file"""
    try:
        # Load MANO model
        with open(mano_path, 'rb') as f:
            mano_model = pickle.load(f, encoding='latin1')
        
        # Extract faces from model
        faces = mano_model['f']
        print(f"Successfully loaded faces from {mano_path}")
        return faces
    except Exception as e:
        print(f"Error loading MANO model from {mano_path}: {e}")
        print("Falling back to default faces")
        # Create a placeholder for faces (just for visualization)
        return np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

def load_config():
    """Load model configuration for the renderer"""
    # Create a simple config object with required attributes
    class Config:
        def __init__(self):
            self.EXTRA = type('', (), {})()
            self.EXTRA.FOCAL_LENGTH = 5000.0
            self.MODEL = type('', (), {})()
            self.MODEL.IMAGE_SIZE = 224.0
    
    return Config()

def get_original_frame(npz_path, frame_index=None, frame_path=None):
    """Find the original frame corresponding to the NPZ file
    
    Args:
        frame_path: Direct path to the frame image (overrides frame_index)
        
    Returns:
        Loaded image or None if not found
    """
  
    return cv2.imread(str(frame_path))

def get_bounding_box(npz_path, frame_index=None):
    """Get the bounding box for the hand from the bounding_boxes.json file"""
    # Extract frame index from NPZ filename if not provided
    if frame_index is None:
        frame_index = int(os.path.basename(npz_path).split('_')[-1].split('.')[0])
    
    # Path to the bounding boxes JSON file
    data_dir = Path(npz_path).parent.parent
    bbox_path = data_dir / "bounding_boxes.json"
    
    if not bbox_path.exists():
        print(f"Warning: Bounding boxes file not found at {bbox_path}")
        return None
    
    # Load the bounding boxes
    with open(bbox_path, 'r') as f:
        bbox_data = json.load(f)
    
    # Get the bounding box for the current frame
    frame_key = str(frame_index)
    if frame_key in bbox_data:
        is_left = "left" in os.path.basename(npz_path)
        box_key = "left_hand" if is_left else "right_hand"
        bbox = bbox_data[frame_key].get(box_key)
        return bbox
    else:
        print(f"Warning: No bounding box found for frame {frame_index}")
        return None

def visualize_with_image(npz_path, output_dir=None, frame_index=None, frame_path=None):
    """
    Visualize hand mesh alongside the original image
    
    Args:
        npz_path: Path to the NPZ file containing hand data
        output_dir: Directory to save visualization (None for no saving)
    """
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the NPZ file
    print(f"Loading hand data from: {npz_path}")
    hand_data = np.load(npz_path, allow_pickle=True)
    
    # Extract hand data
    vertices = hand_data['vertices']
    cam_t = hand_data.get('cam_t', None)
    is_right = hand_data.get('is_right', False)
    bbox = hand_data.get('bbox', None)
    
    print(f"Hand data loaded:")
    print(f"- Vertices shape: {vertices.shape}")
    if cam_t is not None:
        print(f"- Camera translation: {cam_t}")
    print(f"- Is right hand: {is_right}")
    if bbox is not None:
        print(f"- Bounding box from NPZ: {bbox}")
    
    # Get the frame index from the filename if not provided
    if frame_index is None:
        frame_index = int(os.path.basename(npz_path).split('_')[-1].split('.')[0])
        print(f"Using frame index from filename: {frame_index}")
    
    # Load the original frame
    original_frame = get_original_frame(npz_path, frame_index, frame_path)
    if original_frame is None:
        print("No original frame found. Creating blank image.")
        original_frame = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # Get the bounding box
    if bbox is None:
        bbox = get_bounding_box(npz_path, frame_index)
        if bbox is not None:
            print(f"- Bounding box from JSON: {bbox}")
        else:
            print("No bounding box found. Using default values.")
            # Create a default bounding box in the center of the image
            h, w = original_frame.shape[:2]
            bbox = [w//4, h//4, w*3//4, h*3//4]
    
    # Load MANO faces
    faces = load_mano_faces()
    
    # Setup renderer
    renderer = Renderer(load_config(), faces=faces)
    
    # Create the visualization
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Original frame with bounding box
    ax1 = fig.add_subplot(131)
    ax1.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Frame with Bounding Box")
    ax1.axis('off')
    
    # Draw bounding box
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    
    # Create crop of hand region
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        # Add padding
        pad_x = int((x2-x1) * 0.2)
        pad_y = int((y2-y1) * 0.2)
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(original_frame.shape[1], x2 + pad_x)
        y2_pad = min(original_frame.shape[0], y2 + pad_y)
        
        hand_crop = original_frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    else:
        hand_crop = original_frame.copy()
    
    # 2. Cropped hand region
    ax2 = fig.add_subplot(132)
    ax2.imshow(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
    ax2.set_title("Cropped Hand Region")
    ax2.axis('off')
    
    # 3. Rendered mesh (front view)
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot mesh using Matplotlib
    # Extract x, y, z coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Create a more compact visualization
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    max_range = max(x_range, y_range, z_range)
    
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2
    
    # Plot the mesh
    for face in faces:
        verts = vertices[face]
        ax3.plot3D(verts[:, 0], verts[:, 1], verts[:, 2], color='b')
    
    # Set equal aspect ratio
    ax3.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax3.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax3.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    
    ax3.set_title("3D Hand Mesh")
    
    # Add overall title
    hand_side = "Right Hand" if is_right else "Left Hand"
    plt.suptitle(f"3D Hand Visualization - {hand_side} (Frame {frame_index})", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output directory is specified
    if output_dir is not None:
        output_filename = os.path.basename(npz_path).replace('.npz', '_with_image.png')
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    # Try to render mesh on the original image using the renderer
    try:
        # Setup for rendering
        img_size = np.array([original_frame.shape[1], original_frame.shape[0]])
        scaled_focal_length = 5000.0 / 224.0 * max(img_size)
        
        # Setup rendering args
        misc_args = dict(
            mesh_base_color=(0.7, 0.7, 0.9),
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        
        # Render hand mesh (RGBA)
        print("Rendering hand mesh on image...")
        
        # Create lists for renderer
        all_verts = [vertices]
        all_cam_t = [cam_t] if cam_t is not None else [[0, 0, 2]]  # Default camera translation
        all_right = [1 if is_right else 0]
        
        # Render mesh
        cam_view = renderer.render_rgba_multiple(
            all_verts, cam_t=all_cam_t, render_res=img_size, is_right=all_right, **misc_args
        )
        
        # Combine with original image
        alpha = cam_view[:, :, 3:4]
        rgb = cam_view[:, :, :3]
        rendered_img = rgb * alpha + cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB) / 255.0 * (1 - alpha)
        rendered_img = (rendered_img * 255).astype(np.uint8)
        
        # Save the rendered image
        if output_dir is not None:
            rendered_output = os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '_rendered.png'))
            cv2.imwrite(rendered_output, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
            print(f"Rendered image saved to: {rendered_output}")
        
        # Show the rendered image
        plt.figure(figsize=(10, 8))
        plt.imshow(rendered_img)
        plt.axis('off')
        plt.title(f"Rendered Hand Mesh on Original Image - {hand_side} (Frame {frame_index})")
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '_overlay.png')), 
                        dpi=300, bbox_inches='tight')
    
    except Exception as e:
        print(f"Error rendering mesh on image: {e}")
    
    # Show plots
    plt.show()
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize hand data alongside original image")
    parser.add_argument("npz_path", help="Path to the NPZ file containing hand data")
    parser.add_argument("--output-dir", help="Directory to save the visualization", default="./")
    parser.add_argument("--frame", type=int, help="Specific frame number to use for visualization", default=None)
    parser.add_argument("--frame-path", help="Direct path to a specific frame image file (overrides --frame)", default=None)
    
    args = parser.parse_args()
    
    # Check which method is being used to specify the frame
    if args.frame_path is not None:
        print(f"Using specified frame path: {args.frame_path}")
    elif args.frame is not None:
        print(f"Using specified frame number: {args.frame}")
    else:
        print("No specific frame specified, will detect automatically from NPZ filename")
    
    # Visualize hand with image
    visualize_with_image(args.npz_path, args.output_dir, frame_index=args.frame, frame_path=args.frame_path)

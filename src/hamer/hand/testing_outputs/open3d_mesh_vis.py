#!/usr/bin/env python3
"""
Script to visualize hand meshes from NPZ files using Open3D.
This script doesn't depend on the HAMER model initialization.
"""

import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_hand_mesh(npz_path, output_dir=None):
    """
    Visualize hand mesh from an NPZ file using Open3D
    
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
    is_right = bool(hand_data.get('is_right', False))
    bbox = hand_data.get('bbox', None)
    
    print(f"Hand data loaded:")
    print(f"- Vertices shape: {vertices.shape}")
    if cam_t is not None:
        print(f"- Camera translation: {cam_t}")
    print(f"- Is right hand: {is_right}")
    if bbox is not None:
        print(f"- Bounding box: {bbox}")
    
    # Try to load MANO faces (triangles)
    faces = None
    possible_face_paths = [
        '/nas/project_data/B1_Behavior/rush/kaan/hoi/src/hamer/hand/_DATA/mano_faces.npy',
        Path(__file__).parent.parent / '_DATA' / 'mano_faces.npy',
        Path(__file__).parent.parent.parent / '_DATA' / 'mano_faces.npy'
    ]
    
    for path in possible_face_paths:
        try:
            if os.path.exists(str(path)):
                faces = np.load(str(path))
                print(f"Loaded mesh faces from: {path}")
                break
        except Exception as e:
            print(f"Could not load faces from {path}: {e}")
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    
    # Set vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Set faces if available
    if faces is not None:
        mesh.triangles = o3d.utility.Vector3iVector(faces)
    else:
        print("Faces not available, creating a point cloud instead")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.paint_uniform_color([0.7, 0.7, 0.9])
    
    # Set color
    if faces is not None:
        mesh.paint_uniform_color([0.7, 0.7, 0.9])
    
    # Compute normals for proper rendering
    if faces is not None:
        mesh.compute_vertex_normals()
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    
    # Save a screenshot to file if output_dir is provided
    if output_dir is not None:
        # Create a visualizer object
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        
        # Add the mesh and coordinate frame
        if faces is not None:
            vis.add_geometry(mesh)
        else:
            vis.add_geometry(pcd)
        vis.add_geometry(coordinate_frame)
        
        # Get camera view control
        ctrl = vis.get_view_control()
        
        # Set view parameters and take screenshots
        hand_side = "Right Hand" if is_right else "Left Hand"
        
        # Function to save screenshots from different viewpoints
        def save_viewpoint(elev, azim, name_suffix):
            # Set up camera view
            ctrl.set_zoom(0.8)
            ctrl.set_lookat([0, 0, 0])
            ctrl.set_up([0, 1, 0])
            ctrl.set_front([np.sin(np.radians(azim)), -np.sin(np.radians(elev)), -np.cos(np.radians(azim))])
            
            # Update and render
            vis.poll_events()
            vis.update_renderer()
            
            # Generate filename and save
            output_filename = os.path.basename(npz_path).replace('.npz', f'_{name_suffix}.png')
            output_path = os.path.join(output_dir, output_filename)
            vis.capture_screen_image(output_path, do_render=True)
            print(f"Saved {name_suffix} view to: {output_path}")
        
        # Save different viewpoints
        save_viewpoint(0, 0, "front_view")
        save_viewpoint(0, 90, "side_view")
        save_viewpoint(90, 0, "top_view")
        
        # Combine all views into one image using matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Load the saved screenshots
        front_img = plt.imread(os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '_front_view.png')))
        side_img = plt.imread(os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '_side_view.png')))
        top_img = plt.imread(os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '_top_view.png')))
        
        # Display the images
        axes[0].imshow(front_img)
        axes[0].set_title("Front View")
        axes[0].axis('off')
        
        axes[1].imshow(side_img)
        axes[1].set_title("Side View")
        axes[1].axis('off')
        
        axes[2].imshow(top_img)
        axes[2].set_title("Top View")
        axes[2].axis('off')
        
        # Add title
        plt.suptitle(f"3D Hand Mesh Visualization - {hand_side}", fontsize=16)
        
        # Save combined figure
        combined_output = os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '_visualization.png'))
        plt.savefig(combined_output, dpi=300, bbox_inches='tight')
        print(f"Combined visualization saved to: {combined_output}")
        
        # Close visualizer
        vis.destroy_window()
    
    # Interactive visualization
    print("\nStarting interactive visualization...")
    print("Controls: Left-click + drag to rotate, Right-click + drag to pan")
    print("          Scroll to zoom, press 'h' to see all keyboard controls")
    
    if faces is not None:
        o3d.visualization.draw_geometries([mesh, coordinate_frame])
    else:
        o3d.visualization.draw_geometries([pcd, coordinate_frame])
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize hand data from NPZ files using Open3D")
    parser.add_argument("npz_path", help="Path to the NPZ file containing hand data")
    parser.add_argument("--output-dir", help="Directory to save the visualization", default="./")
    
    args = parser.parse_args()
    
    # Visualize hand
    visualize_hand_mesh(args.npz_path, args.output_dir)

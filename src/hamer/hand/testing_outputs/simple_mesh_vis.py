#!/usr/bin/env python3
"""
Simple script to visualize hand meshes from NPZ files using Matplotlib.
This script doesn't depend on the HAMER model initialization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation

# MANO hand model faces (triangles) - standard topology
# Load faces from the MANO pickle file
def load_mano_faces(mano_path='/nas/project_data/B1_Behavior/rush/kaan/hoi/src/hamer/hand/_DATA/data/mano/MANO_RIGHT.pkl'):
    """Load faces from MANO pickle file"""
    import pickle
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

# Get faces from the MANO model
MANO_FACES = load_mano_faces()

def visualize_hand_mesh(npz_path, output_dir=None):
    """
    Visualize hand mesh from an NPZ file using Matplotlib
    
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
    cam_t = hand_data['cam_t'] if 'cam_t' in hand_data else None
    is_right = hand_data['is_right'] if 'is_right' in hand_data else False
    bbox = hand_data['bbox'] if 'bbox' in hand_data else None
    
    print(f"Hand data loaded:")
    print(f"- Vertices shape: {vertices.shape}")
    if cam_t is not None:
        print(f"- Camera translation: {cam_t}")
    print(f"- Is right hand: {is_right}")
    if bbox is not None:
        print(f"- Bounding box: {bbox}")
    
    # Create figure for 3D visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Create front view subplot
    ax1 = fig.add_subplot(131, projection='3d')
    plot_hand_mesh(ax1, vertices, is_right, "Front View")
    ax1.view_init(elev=0, azim=0)  # Front view
    
    # Create side view subplot
    ax2 = fig.add_subplot(132, projection='3d')
    plot_hand_mesh(ax2, vertices, is_right, "Side View")
    ax2.view_init(elev=0, azim=90)  # Side view
    
    # Create top view subplot
    ax3 = fig.add_subplot(133, projection='3d')
    plot_hand_mesh(ax3, vertices, is_right, "Top View")
    ax3.view_init(elev=90, azim=0)  # Top view
    
    # Add title
    hand_side = "Right Hand" if is_right else "Left Hand"
    plt.suptitle(f"3D Hand Mesh Visualization - {hand_side}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output directory is specified
    if output_dir is not None:
        output_filename = os.path.basename(npz_path).replace('.npz', '_visualization.png')
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    # Show figure
    plt.show()
    
    return fig

def plot_hand_mesh(ax, vertices, is_right, title):
    """
    Plot hand mesh on the given axis
    
    Args:
        ax: Matplotlib 3D axis
        vertices: 3D mesh vertices
        is_right: Boolean indicating if it's a right hand
        title: Title for the subplot
    """
    # Extract x, y, z coordinates from vertices
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Normalize the coordinates to fit well in the plot
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    max_range = max(x_range, y_range, z_range)
    
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2
    
    # Plot the mesh
    try:
        # Plot using triangulation for a solid mesh
        for face in MANO_FACES:
            verts = vertices[face]
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
            ax.plot_trisurf(x, y, z, color=(0.7, 0.7, 0.9, 0.7), edgecolor=(0.2, 0.2, 0.8, 0.5), linewidth=0.5)
    except Exception as e:
        print(f"Error plotting triangles: {e}")
        print("Falling back to scatter plot")
        # Fallback: use scatter plot
        ax.scatter(x, y, z, c='b', marker='o', s=10)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    
    # Set title
    ax.set_title(title)
    
    # Set background color to white
    ax.set_facecolor('white')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize hand data from NPZ files")
    parser.add_argument("npz_path", help="Path to the NPZ file containing hand data")
    parser.add_argument("--output-dir", help="Directory to save the visualization", default="./")
    
    args = parser.parse_args()
    
    # Visualize hand
    visualize_hand_mesh(args.npz_path, args.output_dir)

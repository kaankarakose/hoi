import os
import subprocess
from pathlib import Path
import glob

def render_cad(cad_path: str, output_dir: str, lighting_intensity: float = 0.6, radius: float = 0.3) -> None:
    """
    Python wrapper for rendering CAD models using pyrender.
    
    Args:
        cad_path (str): Path to the CAD model file (not directory)
        output_dir (str): Directory where rendered images will be saved
        lighting_intensity (float, optional): Lighting intensity for rendering. Defaults to 0.6.
        radius (float, optional): Distance to camera. Defaults to 0.3.
    """
    # Ensure paths are absolute
    cad_path = str(Path(cad_path).absolute())
    output_dir = str(Path(output_dir).absolute())
    
    # Set environment variables
    os.environ['CAD_PATH'] = cad_path
    os.environ['OUTPUT_DIR'] = output_dir
    os.environ['LIGHTING_INTENSITY'] = str(lighting_intensity)
    os.environ['RADIUS'] = str(radius)
    
    # Construct the command
    cmd = [
        'python', '-m', 'src.poses.pyrender',
        cad_path,
        './src/poses/predefined_poses/obj_poses_level2.npy',
        output_dir,
        '0',  # Fixed parameter from original script
        'False',  # Fixed parameter from original script
        str(lighting_intensity),
        str(radius)
    ]
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Rendering failed with error: {e}")

if __name__ == "__main__":
    cad_folder = '/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/ados_objects'
    object_names = [name for name in os.listdir(cad_folder) if os.path.isdir(os.path.join(cad_folder, name))]
    cad_folders = [os.path.join(cad_folder, folder) for folder in object_names]
    output_dir = '/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/rendered_objects'
    for object_name, cad_folder_path in zip(object_names, cad_folders):
        # Find the .ply file in the object folder
        ply_files = glob.glob(os.path.join(cad_folder_path, "*.ply"))
        if not ply_files:
            print(f"Warning: No .ply file found in {cad_folder_path}")
            continue
            
        cad_file = ply_files[0]  # Use the first .ply file if multiple exist
        rendered_output_dir = os.path.join(output_dir, object_name)
        if os.path.exists(rendered_output_dir): continue
        
        os.makedirs(rendered_output_dir, exist_ok=True)

        print(f"Processing {object_name}: {cad_file}")
        try:
            render_cad(cad_file, rendered_output_dir, lighting_intensity=0.6, radius=0.3)
        except Exception as e:
            print(f"Error processing {object_name}: {e}")
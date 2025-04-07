import os
import numpy as np
from PIL import Image
import trimesh
import shutil

def obj_to_ply(obj_path, out_dir, template_name):
  
    vertices = []  # [x,y,z]
    normals = []   # [nx,ny,nz]
    texcoords = [] # [u,v]
    faces = []     # [(v1,vt1,vn1), (v2,vt2,vn2), (v3,vt3,vn3)]
    
    # Parse OBJ file manually to preserve texture coordinates
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Vertex
                _, x, y, z = line.split()

                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('vn '):
                # Normal
                _, nx, ny, nz = line.split()
                normals.append([float(nx), float(ny), float(nz)])
            elif line.startswith('vt '):
                # Texture coordinate
                _, u, v = line.split()
                texcoords.append([float(u), float(v)])
            elif line.startswith('f '):
                # Face (format: v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
                face_vertices = []
                for vertex_str in line.split()[1:]:
                    v, vt, vn = map(lambda x: int(x)-1, vertex_str.split('/'))
                    face_vertices.append((v, vt, vn))
                faces.append(face_vertices)
    
    # Get the directory containing the OBJ file
    obj_dir = os.path.dirname(obj_path)
    color_path = os.path.join(obj_dir, "color.jpg")
    
    if os.path.exists(color_path):
        # Create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        
        # Copy the color image to output directory
        out_color_path = os.path.join(out_dir, f"{template_name}.jpg")
        shutil.copy2(color_path, out_color_path)
        
        # Convert to PLY format
        out_ply_path = os.path.join(out_dir, f"{template_name}.ply")
        
        # Create vertex attribute mapping to avoid duplicates
        vertex_map = {}
        unique_vertices = []
        remapped_faces = []
        
        # Process faces and create unique vertices
        for face in faces:
            face_indices = []
            for v_idx, vt_idx, vn_idx in face:
                # Create a unique key for this vertex combination
                key = (v_idx, vt_idx, vn_idx)
                if key not in vertex_map:
                    vertex_map[key] = len(unique_vertices)
                    unique_vertices.append({
                        'vertex': vertices[v_idx],
                        'normal': normals[vn_idx],
                        'texcoord': texcoords[vt_idx]
                    })
                face_indices.append(vertex_map[key])
            remapped_faces.append(face_indices)
        
        with open(out_ply_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment TextureFile {template_name}.jpg\n")
            
            # Write vertex properties
            f.write(f"element vertex {len(unique_vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("property float texture_u\n")
            f.write("property float texture_v\n")
            
            # Write face properties
            f.write(f"element face {len(remapped_faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write unique vertices with their attributes
            for v_data in unique_vertices:
                vertex = v_data['vertex']
                normal = v_data['normal']
                texcoord = v_data['texcoord']
                f.write(f"{vertex[0]:.4f} {vertex[1]:.4f} {vertex[2]:.4f} "
                        f"{normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f} "
                        f"{texcoord[0]:.4f} {texcoord[1]:.4f}\n")
            
            # Write faces with remapped indices
            for face in remapped_faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print(f"Successfully converted {obj_path} to {out_ply_path}")
        return True
        
    else:
        print(f"Error: color.jpg not found in {obj_dir}")
        return False
            

def process_templates(raw_templates_dir, out_dir):
    """
    Process all templates in the raw templates directory.
    
    Args:
        raw_templates_dir (str): Directory containing raw templates
        out_dir (str): Output directory for processed templates
    """
    for root, dirs, files in os.walk(raw_templates_dir):
        for file in files:
            if file.endswith('.obj'):
                template_name = os.path.splitext(file)[0]
                obj_path = os.path.join(root, file)
                obj_to_ply(obj_path, out_dir, template_name)

if __name__ == "__main__":
    # Example usage
    raw_templates_dir = "/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/raw_templates"
    out_dir = "/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/ados_objects"

    raw_folders = [os.path.join(raw_templates_dir, folder) for folder in os.listdir(raw_templates_dir) 
               if os.path.isdir(os.path.join(raw_templates_dir, folder))]
    object_names = [name for name in os.listdir(raw_templates_dir)]

    for object_name, raw_object_folder in zip(object_names,raw_folders):
        print(object_name)
        output_dir = os.path.join(out_dir, object_name)
        os.makedirs(output_dir, exist_ok = True)


        process_templates(raw_object_folder, output_dir)
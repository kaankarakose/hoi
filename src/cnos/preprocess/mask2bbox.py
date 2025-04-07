import numpy as np
import json
from PIL import Image

def mask_to_bbox(mask):
    # Convert mask image to grayscale numpy array
    if isinstance(mask, str):
        mask = Image.open(mask).convert('L')
    elif isinstance(mask, Image.Image):
        mask = mask.convert('L')
    else:
        raise ValueError("Mask must be a file path or PIL Image")
    
    mask_array = np.array(mask)
    
    # Find the indices of non-zero elements
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    
    # Get the bounding box coordinates
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Return the bounding box as a JSON
    bbox = {
        "xmin": int(xmin),
        "ymin": int(ymin),
        "xmax": int(xmax),
        "ymax": int(ymax)
    }
    return json.dumps(bbox)


if __name__ =="__main__":
    # import sys
    # mask_path = sys.argv[1]
    bbox = mask_to_bbox('/nas/project_data/B1_Behavior/rush/ados-objects/object_pose_mega/rtdt/cnos/cnos/output/masks/frame_0054/mask_0.png')
    print(bbox)
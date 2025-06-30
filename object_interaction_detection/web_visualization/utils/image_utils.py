"""
Utility functions for image processing and conversion
"""

import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

##
OBJECT_COLORS = {
    'AMF1': [161, 113, 146],      # Color for AMF1
    'AMF2': [131, 136, 51],       # Color for AMF2
    'AMF3': [130, 171, 146],      # Color for AMF3
    'BOX': [112, 62, 54],         # Color for BOX
    'CUP': [146, 79, 156],        # Color for CUP
    'DINOSAUR': [99, 93, 79],     # Color for DINOSAUR
    'FIRETRUCK': [94, 121, 171],  # Color for FIRETRUCK
    'HAIRBRUSH': [59, 175, 143],  # Color for HAIRBRUSH
    'PINCER': [210, 196, 209],    # Color for PINCER 
    'WRENCH': [67, 126, 87],      # Color for WRENCH
}

def image_to_base64(image_array):
    """
    Convert a numpy array image to base64 encoded string for web display
    
    Args:
        image_array: Numpy array containing image data
        
    Returns:
        String with base64 encoded image data with data URL prefix
    """
    if image_array is None:
        return ""
    
    # Convert to uint8 if not already
    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    # Handle different channel formats
    if len(image_array.shape) == 2:
        # Convert grayscale to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        # Convert RGBA to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array)
    
    # Save to bytes buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Convert to base64
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    
    return f"data:image/png;base64,{img_base64}"

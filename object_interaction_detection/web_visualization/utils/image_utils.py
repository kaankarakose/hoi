"""
Utility functions for image processing and conversion
"""

import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

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

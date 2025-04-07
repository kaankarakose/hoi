import os
import json
import numpy as np
import cv2

def rle2mask(rle) :
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def load_rle_mask(rle_file_path):
    """
    Load an RLE mask from a file.
    
    Args:
        rle_file_path: Path to the RLE file
    
    Returns:
        numpy array: Binary mask
    """
    with open(rle_file_path, 'r') as f:
        rle_data = json.load(f)
    

    return rle2mask(rle_data)


def apply_mask_to_frame(frame, mask, color=(0, 0, 255), alpha=0.5):
    """
    Apply a binary mask to a frame with the specified color and transparency.
    
    Args:
        frame: Input image (BGR format)
        mask: Binary mask (1 - mask, 0 - background)
        color: Color to apply (BGR format)
        alpha: Transparency value (0-1)
    
    Returns:
        numpy array: Frame with the mask applied
    """
    # Make a copy of the frame to avoid modifying the original
    result = frame.copy()
    
    # Create a colored mask
    colored_mask = np.zeros_like(frame)
    colored_mask[mask == 1] = color
    
    # Blend the colored mask with the frame
    cv2.addWeighted(colored_mask, alpha, result, 1 - alpha, 0, result)
    
    # Add a contour around the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    
    return result

def visualize_and_save(frame_path, mask_path, output_path):
    """
    Load a frame and a mask, visualize the mask on the frame, and save the result.
    
    Args:
        frame_path: Path to the frame image
        mask_path: Path to the RLE mask file
        output_path: Path to save the visualization
    """
    # Load the frame
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Could not load frame from {frame_path}")
    
    # Load the mask
    mask = load_rle_mask(mask_path)
    
    # Make sure the mask and frame have compatible dimensions
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Apply the mask to the frame
    result = apply_mask_to_frame(frame, mask)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    

    #Get frame name
    frame_n = os.path.basename(mask_path).split('.')[0] + '.png'
    print(frame_n)
    # Save the result
    output_name = os.path.join(output_path,frame_n)
    cv2.imwrite(output_name, result)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize RLE masks on frames')
    parser.add_argument('--mode', type=str, choices=['test', 'visualize'], default='test',
                        help='Mode: "test" to create a test mask, "visualize" to visualize an existing mask')
    parser.add_argument('--frame', type=str, help='Path to the frame image (for both modes: in test mode, this is optional; in visualize mode, this is required)')
    parser.add_argument('--mask', type=str, help='Path to the RLE mask file (for visualize mode)')
    parser.add_argument('--output', type=str, help='Path to save the visualization (for visualize mode)',default='./test_output')

    
    args = parser.parse_args()
 

        
    visualize_and_save(args.frame, args.mask, args.output)

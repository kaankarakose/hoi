import numpy as np
import json



def rle2mask(rle):
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
    return rle_data
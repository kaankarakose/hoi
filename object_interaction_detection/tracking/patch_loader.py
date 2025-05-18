# patch_loader.py
"""
This script ensures the NumPy patch is loaded before any other imports.
"""
import os
import sys

# Add the directory containing the patch to Python's path
patch_dir = os.path.dirname(os.path.abspath(__file__))
if patch_dir not in sys.path:
    sys.path.insert(0, patch_dir)

# Import the patch to apply it
import numpy_patch

# Print confirmation
print("NumPy patch loaded and applied successfully")
"""
This module patches NumPy's zeros function to handle memory allocation errors
by falling back to uint8 dtype when bool arrays fail to allocate.
"""
import numpy as np
import logging

# Store the original function
_original_np_zeros = np.zeros

def _patched_zeros(*args, **kwargs):
    """A safer version of np.zeros that handles memory errors for bool arrays"""
    try:
        # First try with the original parameters
        return _original_np_zeros(*args, **kwargs)
    except Exception as e:
        # If it's a bool array, try with uint8 instead
        if kwargs.get('dtype') == bool or (len(args) > 1 and args[1] == bool):
            try:
                # Replace bool with uint8
                if 'dtype' in kwargs:
                    kwargs['dtype'] = np.uint8
                else:
                    args = list(args)
                    if len(args) > 1:
                        args[1] = np.uint8
                    args = tuple(args)
                
                print("[MEMORY FIX] Using uint8 instead of bool for array allocation")
                return _original_np_zeros(*args, **kwargs)
            except Exception:
                # If that still fails, re-raise the original error
                raise e
        else:
            # If it's not a bool array issue, re-raise
            raise e

# Apply the patch
np.zeros = _patched_zeros
print("NumPy zeros function successfully patched to handle memory errors")
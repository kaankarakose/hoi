# numpy_patch.py - Enhanced version
import numpy as np
import gc
import logging

# Store the original function
original_zeros = np.zeros

def safe_zeros(*args, **kwargs):
    """A safer version of np.zeros that handles memory errors with multiple fallbacks"""
    # First try: Original parameters
    try:
        return original_zeros(*args, **kwargs)
    except Exception as e:
        print(f"[MEMORY WARNING] Failed to allocate array with original parameters, trying alternatives...")
        
        # Second try: Force garbage collection and retry
        gc.collect()
        try:
            print("[MEMORY FIX] Retrying after garbage collection")
            return original_zeros(*args, **kwargs)
        except Exception:
            # Third try: Convert bool to uint8 if applicable
            if kwargs.get('dtype') == bool or (len(args) > 1 and args[1] == bool):
                try:
                    if 'dtype' in kwargs:
                        kwargs['dtype'] = np.uint8
                    else:
                        args = list(args)
                        if len(args) > 1:
                            args[1] = np.uint8
                        args = tuple(args)
                    
                    print("[MEMORY FIX] Using uint8 instead of bool")
                    return original_zeros(*args, **kwargs)
                except Exception:
                    pass
            
            # Fourth try: Allocate in chunks (for 2D arrays)
            try:
                shape = args[0] if args else kwargs.get('shape')
                dtype = args[1] if len(args) > 1 else kwargs.get('dtype', np.float64)
                
                if isinstance(shape, tuple) and len(shape) == 2:
                    rows, cols = shape
                    print(f"[MEMORY FIX] Allocating in chunks ({rows}x{cols})")
                    
                    # Allocate in smaller rows
                    chunk_size = 100  # Allocate 100 rows at a time
                    result = None
                    
                    for i in range(0, rows, chunk_size):
                        end_i = min(i + chunk_size, rows)
                        chunk_rows = end_i - i
                        
                        chunk = original_zeros((chunk_rows, cols), dtype=dtype)
                        
                        if result is None:
                            result = chunk
                        else:
                            result = np.vstack((result, chunk))
                        
                        # Force garbage collection after each chunk
                        del chunk
                        gc.collect()
                    
                    return result
            except Exception as chunk_error:
                print(f"[MEMORY ERROR] Failed chunked allocation: {chunk_error}")
            
            # Fifth try: Try with a smaller temporary array
            try:
                shape = args[0] if args else kwargs.get('shape')
                dtype = args[1] if len(args) > 1 else kwargs.get('dtype', np.float64)
                
                if isinstance(shape, tuple) and len(shape) == 2:
                    rows, cols = shape
                    print(f"[MEMORY FIX] Creating minimal array and padding ({rows}x{cols})")
                    
                    # Create a minimal array
                    minimal_array = original_zeros((1, cols), dtype=dtype)
                    
                    # Pad it to full size
                    result = np.zeros_like(minimal_array, shape=(rows, cols))
                    
                    return result
            except Exception as minimal_error:
                print(f"[MEMORY ERROR] Failed minimal allocation: {minimal_error}")
            
            # If all else fails, raise the original error
            raise e

# Apply the patch
np.zeros = safe_zeros
print("[MEMORY FIX] NumPy zeros function patched with enhanced memory management")
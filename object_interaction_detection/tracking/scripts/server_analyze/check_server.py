import numpy as np
import gc
import time

array_shape = (1200, 1920)
dtype = bool
required_memory_mb = (np.prod(array_shape) * np.dtype(dtype).itemsize) / (1024 * 1024)
print(f"Trying to allocate {required_memory_mb:.2f} MiB array.")

# Attempt direct allocation
try:
    arr = np.zeros(array_shape, dtype=dtype)
    print("Direct allocation succeeded.")
    del arr
    gc.collect()
except np.core._exceptions._ArrayMemoryError as e:
    print(f"Direct allocation failed: {e}")

# Try allocating smaller chunks
chunk_shape = (400, 1920)
num_chunks = array_shape[0] // chunk_shape[0]
chunks = []
try:
    for i in range(num_chunks):
        chunk = np.zeros(chunk_shape, dtype=dtype)
        chunks.append(chunk)
    print("Small chunk allocations succeeded.")
    del chunks
    gc.collect()
except np.core._exceptions._ArrayMemoryError as e:
    print(f"Small chunk allocation failed at chunk {i}: {e}")

# Try allocating and immediately re-allocating
try:
    arr1 = np.zeros(array_shape, dtype=dtype)
    print("Initial allocation succeeded.")
    del arr1
    gc.collect()
    time.sleep(0.1) # Give time for potential defragmentation (though OS dependent)
    arr2 = np.zeros(array_shape, dtype=dtype)
    print("Immediate re-allocation succeeded.")
    del arr2
    gc.collect()
except np.core._exceptions._ArrayMemoryError as e:
    print(f"Immediate re-allocation failed: {e}")

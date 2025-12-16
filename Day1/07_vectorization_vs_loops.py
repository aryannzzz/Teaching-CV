"""
Day 1: Part 2 - NumPy Essentials
Section 3.8: Vectorization vs Loops
Demonstrates why vectorization is crucial for performance
"""
import time
import numpy as np

# Create test data
arr = np.random.rand(1000000)

# LOOP APPROACH (SLOW)
print("Testing loop vs vectorized operations...")
start = time.time()
result_loop = np.zeros_like(arr)
for i in range(len(arr)):
    result_loop[i] = arr[i] ** 2
loop_time = time.time() - start

# VECTORIZED APPROACH (FAST)
start = time.time()
result_vectorized = arr ** 2
vectorized_time = time.time() - start

print(f"\nLoop time: {loop_time:.4f}s")
print(f"Vectorized time: {vectorized_time:.4f}s")
print(f"Speedup: {loop_time / vectorized_time:.1f}x faster!")
# Typically 50-100x faster!

# Verify results are identical
print(f"Results match: {np.allclose(result_loop, result_vectorized)}")

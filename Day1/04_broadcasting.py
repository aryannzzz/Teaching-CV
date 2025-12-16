"""
Day 1: Part 2 - NumPy Essentials
Section 3.5: Broadcasting
"""
import numpy as np

# Scalar broadcasting
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
result = arr + 10  # Add 10 to every element
print("Scalar broadcasting:")
print(result)
# [[11 12 13]
#  [14 15 16]]

# 1D to 2D broadcasting
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

result = arr_2d + arr_1d  # arr_1d broadcasts across rows
print("\n1D to 2D broadcasting:")
print(result)
# [[11 22 33]
#  [14 25 36]]

# Column broadcasting
col_vector = np.array([[10],
                       [20]])
result = arr_2d + col_vector  # Broadcasts across columns
print("\nColumn broadcasting:")
print(result)
# [[11 12 13]
#  [24 25 26]]

# Image example - adjust RGB channels independently
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
channel_adjustments = np.array([1.2, 1.0, 0.8])  # Boost red, reduce blue
adjusted = np.clip(image * channel_adjustments, 0, 255).astype(np.uint8)
print(f"\nImage shape: {image.shape}, Adjusted: {adjusted.shape}")

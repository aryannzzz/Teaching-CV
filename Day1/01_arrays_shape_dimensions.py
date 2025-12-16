"""
Day 1: Part 2 - NumPy Essentials
Section 3.2: Arrays, Shape, and Dimensions
"""
import numpy as np

# Creating arrays
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Array properties
print(f"Shape: {arr_2d.shape}")      # (3, 3) - rows, columns
print(f"Dimensions: {arr_2d.ndim}")  # 2
print(f"Size: {arr_2d.size}")        # 9 total elements
print(f"Data type: {arr_2d.dtype}")  # int64

# Image array representations
# Grayscale: (height, width)
gray_image = np.zeros((480, 640), dtype=np.uint8)
print(f"Grayscale shape: {gray_image.shape}")  # (480, 640)

# RGB: (height, width, channels)
color_image = np.zeros((480, 640, 3), dtype=np.uint8)
print(f"Color shape: {color_image.shape}")  # (480, 640, 3)

# Batch of images: (batch_size, height, width, channels)
batch = np.zeros((32, 224, 224, 3), dtype=np.uint8)
print(f"Batch shape: {batch.shape}")  # (32, 224, 224, 3)

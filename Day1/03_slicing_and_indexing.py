"""
Day 1: Part 2 - NumPy Essentials
Section 3.4: Slicing and Indexing
"""
import numpy as np

arr = np.array([[10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120]])

# Basic indexing
print(arr[0, 0])      # 10 - first element
print(arr[2, 3])      # 120 - last element
print(arr[-1, -1])    # 120 - negative indexing

# Row and column selection
print(arr[0, :])      # [10 20 30 40] - first row
print(arr[:, 1])      # [20 60 100] - second column
print(arr[:, -1])     # [40 80 120] - last column

# Slicing ranges
print(arr[0:2, 1:3])  
# [[20 30]
#  [60 70]]

# Boolean indexing
mask = arr > 50
print(arr[mask])      # [60 70 80 90 100 110 120]

# Fancy indexing
rows = [0, 2]
cols = [1, 3]
print(arr[rows, cols])  # [20 120]

# For images - crop center region
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
h, w = image.shape[:2]
crop_size = 200
center_crop = image[
    h//2 - crop_size//2 : h//2 + crop_size//2,
    w//2 - crop_size//2 : w//2 + crop_size//2
]
print(f"Original: {image.shape}, Cropped: {center_crop.shape}")

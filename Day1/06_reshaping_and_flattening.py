"""
Day 1: Part 2 - NumPy Essentials
Section 3.7: Reshaping and Flattening
"""
import numpy as np

# Reshape
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr.reshape(2, 3)
print("Reshaped:")
print(reshaped)
# [[1 2 3]
#  [4 5 6]]

# Automatic dimension inference with -1
auto = arr.reshape(3, -1)  # NumPy calculates: (3, 2)
print(f"Auto reshape shape: {auto.shape}")

# Flatten to 1D
flat = reshaped.flatten()     # Returns copy
ravel = reshaped.ravel()      # Returns view (faster)
print(f"Flattened: {flat}")
print(f"Raveled: {ravel}")

# Transpose
transposed = reshaped.T
print(f"Transposed:\n{transposed}")
# [[1 4]
#  [2 5]
#  [3 6]]

# Image examples
# Flatten for ML model input
mnist_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
flat_image = mnist_image.flatten()
print(f"MNIST Image shape: {mnist_image.shape}, Flat: {flat_image.shape}")
# (28, 28) -> (784,)

# Add batch dimension
single_image = np.random.rand(224, 224, 3)
batched = single_image.reshape(1, 224, 224, 3)
# or
batched = np.expand_dims(single_image, axis=0)
print(f"Single image: {single_image.shape}, Batched: {batched.shape}")

# Rearrange dimensions (HWC to CHW for PyTorch)
hwc_image = np.random.rand(224, 224, 3)
chw_image = np.transpose(hwc_image, (2, 0, 1))
print(f"HWC: {hwc_image.shape}, CHW: {chw_image.shape}")

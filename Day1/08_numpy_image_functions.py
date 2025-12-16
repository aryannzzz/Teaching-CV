"""
Day 1: Part 2 - NumPy Essentials
Section 3.9: Useful NumPy Functions for Images
"""
import numpy as np

image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Statistical operations
print(f"Mean: {image.mean()}")
print(f"Std: {image.std()}")
print(f"Min: {image.min()}, Max: {image.max()}")
print(f"Median: {np.median(image)}")

# Per-channel statistics
print(f"Mean per channel: {image.mean(axis=(0, 1))}")  # [R, G, B] means
print(f"Std per channel: {image.std(axis=(0, 1))}")

# Clipping values
clipped = np.clip(image, 50, 200)  # Constrain to [50, 200]

# Normalization
normalized = (image - image.min()) / (image.max() - image.min())

# Standardization (zero mean, unit variance)
standardized = (image - image.mean()) / image.std()

# Concatenation
img1 = np.zeros((100, 100, 3), dtype=np.uint8)
img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

hstack = np.hstack([img1, img2])  # Horizontal: (100, 200, 3)
vstack = np.vstack([img1, img2])  # Vertical: (200, 100, 3)
dstack = np.dstack([img1, img2])  # Depth: (100, 100, 6)

# Stacking for batches
batch = np.stack([img1, img2, img1])  # (3, 100, 100, 3)

# Splitting arrays
split_images = np.split(batch, 3, axis=0)  # Split batch into individual images

# Where (conditional selection)
# Create binary mask
mask = image > 127
# Apply different operations based on condition
output = np.where(mask, 255, 0)  # White where >127, black otherwise

print(f"Image shape: {image.shape}")
print(f"Clipped range: {clipped.min()}-{clipped.max()}")
print(f"Normalized range: {normalized.min():.4f}-{normalized.max():.4f}")
print(f"HStack shape: {hstack.shape}")
print(f"VStack shape: {vstack.shape}")
print(f"Batch shape: {batch.shape}")
print(f"Split count: {len(split_images)}")

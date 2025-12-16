"""
Day 1: Part 2 - NumPy Essentials
Section 3.3: Array Creation Methods
"""
import numpy as np

# Zeros and ones
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
full = np.full((3, 3), 7)  # Fill with specific value

# Identity matrix
identity = np.eye(4)

# Random arrays
random_uniform = np.random.rand(3, 3)        # Uniform [0, 1)
random_normal = np.random.randn(3, 3)        # Normal distribution
random_int = np.random.randint(0, 255, (3, 3))  # Integers

# Ranges
arange = np.arange(0, 10, 2)           # [0 2 4 6 8]
linspace = np.linspace(0, 1, 5)        # 5 evenly spaced values

# Like operations (match shape/dtype of existing array)
arr = np.array([[1, 2], [3, 4]])
zeros_like = np.zeros_like(arr)
ones_like = np.ones_like(arr)

print("Zeros:\n", zeros)
print("Ones:\n", ones)
print("Identity:\n", identity)
print("Random uniform:\n", random_uniform)
print("Random normal:\n", random_normal)
print("Random int:\n", random_int)

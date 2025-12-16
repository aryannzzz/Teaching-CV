"""
Day 1: Part 2 - NumPy Essentials
Section 3.6: Element-wise Operations
"""
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Arithmetic
print("Arithmetic operations:")
print(f"arr + 10 = {arr + 10}")       # [11 12 13 14 15]
print(f"arr * 2 = {arr * 2}")        # [2 4 6 8 10]
print(f"arr ** 2 = {arr ** 2}")       # [1 4 9 16 25]
print(f"arr / 2 = {arr / 2}")        # [0.5 1. 1.5 2. 2.5]

# Array to array
arr2 = np.array([5, 4, 3, 2, 1])
print(f"\narr + arr2 = {arr + arr2}")     # [6 6 6 6 6]
print(f"arr * arr2 = {arr * arr2}")     # [5 8 9 8 5]

# Comparison operations
print(f"\narr > 3 = {arr > 3}")        # [False False False True True]
print(f"arr == 3 = {arr == 3}")       # [False False True False False]

# Mathematical functions
print(f"\nsqrt(arr) = {np.sqrt(arr)}")   # Square root
print(f"exp(arr) = {np.exp(arr)}")      # Exponential
print(f"log(arr) = {np.log(arr)}")      # Natural log
print(f"sin(arr) = {np.sin(arr)}")      # Sine

# Image brightness adjustment
image = np.random.randint(0, 200, (100, 100), dtype=np.uint8)
brighter = np.clip(image + 50, 0, 255).astype(np.uint8)
darker = np.clip(image - 50, 0, 255).astype(np.uint8)
contrast = np.clip(image * 1.5, 0, 255).astype(np.uint8)
print(f"\nImage brightness adjustments completed")
print(f"Original range: {image.min()}-{image.max()}")
print(f"Brighter range: {brighter.min()}-{brighter.max()}")
print(f"Darker range: {darker.min()}-{darker.max()}")

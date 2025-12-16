#!/usr/bin/env python3
"""
Day 1 Assignment Solution: Pencil Sketch Effect
Computer Vision Bootcamp

This program converts photographs into realistic pencil sketch drawings
using classical image processing techniques.

Author: CV Bootcamp
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path


def pencil_sketch(image_path, blur_kernel=21, save_intermediate=False):
    """
    Convert an image to pencil sketch effect using dodge and burn technique.
    
    Args:
        image_path (str): Path to input image
        blur_kernel (int): Gaussian blur kernel size (must be odd)
        save_intermediate (bool): If True, save intermediate steps
    
    Returns:
        tuple: (original_rgb, sketch) or (None, None) if error occurs
    """
    try:
        # Step 1: Load image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None, None
        
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Invert the grayscale image
        inverted = 255 - gray
        
        # Step 4: Apply Gaussian blur to inverted image
        # Ensure kernel size is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
            print(f"Adjusted blur kernel to {blur_kernel} (must be odd)")
        
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)
        
        # Step 5: Invert the blurred image
        inverted_blur = 255 - blurred
        
        # Step 6: Divide and scale to create sketch effect
        # Add small epsilon to avoid division by zero
        sketch = cv2.divide(gray, inverted_blur + 1e-6, scale=256.0)
        
        # Ensure values are in valid range and convert to uint8
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)
        
        # Save intermediate steps if requested
        if save_intermediate:
            base_name = Path(image_path).stem
            cv2.imwrite(f'{base_name}_step1_gray.jpg', gray)
            cv2.imwrite(f'{base_name}_step2_inverted.jpg', inverted)
            cv2.imwrite(f'{base_name}_step3_blurred.jpg', blurred)
            cv2.imwrite(f'{base_name}_step4_inverted_blur.jpg', inverted_blur)
            print(f"Saved intermediate processing steps")
        
        return image_rgb, sketch
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None


def pencil_sketch_color(image_path, blur_kernel=21):
    """
    Create a colored pencil sketch effect.
    
    Args:
        image_path (str): Path to input image
        blur_kernel (int): Gaussian blur kernel size
    
    Returns:
        tuple: (original_rgb, color_sketch) or (None, None) if error
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply sketch effect to Value channel only
        _, v_sketch = pencil_sketch_grayscale(hsv[:, :, 2], blur_kernel)
        
        if v_sketch is None:
            return None, None
        
        # Replace Value channel with sketch
        hsv[:, :, 2] = v_sketch
        
        # Reduce saturation for more realistic pencil effect
        hsv[:, :, 1] = (hsv[:, :, 1] * 0.6).astype(np.uint8)
        
        # Convert back to RGB
        color_sketch = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image_rgb, color_sketch
        
    except Exception as e:
        print(f"Error creating color sketch: {str(e)}")
        return None, None


def pencil_sketch_grayscale(gray_image, blur_kernel=21):
    """
    Apply pencil sketch effect to a grayscale image.
    Helper function for color sketch processing.
    
    Args:
        gray_image: Grayscale image as numpy array
        blur_kernel: Gaussian blur kernel size
    
    Returns:
        tuple: (original, sketch) or (None, None) if error
    """
    try:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        
        inverted = 255 - gray_image
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)
        inverted_blur = 255 - blurred
        sketch = cv2.divide(gray_image, inverted_blur + 1e-6, scale=256.0)
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)
        
        return gray_image, sketch
        
    except Exception as e:
        print(f"Error in grayscale sketch: {str(e)}")
        return None, None


def display_result(original, sketch, title="Pencil Sketch Effect", save_path=None):
    """
    Display original and sketch images side-by-side using matplotlib.
    
    Args:
        original: Original image (RGB format)
        sketch: Sketch image (grayscale or RGB)
        title: Title for the figure
        save_path: Optional path to save the comparison figure
    """
    if original is None or sketch is None:
        print("Error: Cannot display None images")
        return
    
    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Display original
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display sketch (handle both grayscale and color)
    if len(sketch.shape) == 2:  # Grayscale
        axes[1].imshow(sketch, cmap='gray')
    else:  # Color
        axes[1].imshow(sketch)
    
    axes[1].set_title('Pencil Sketch', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def save_sketch(sketch, output_path):
    """
    Save sketch image to file.
    
    Args:
        sketch: Sketch image (grayscale or color)
        output_path: Path where to save the image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR if color image
        if len(sketch.shape) == 3:
            sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, sketch_bgr)
        else:
            cv2.imwrite(output_path, sketch)
        
        print(f"Sketch saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving sketch: {str(e)}")
        return False


def process_multiple_images(image_paths, output_folder, blur_kernel=21):
    """
    Process multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        output_folder: Folder to save sketches
        blur_kernel: Gaussian blur kernel size
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(image_paths)} images...")
    print(f"Output folder: {output_folder}\n")
    
    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {image_path}")
        
        original, sketch = pencil_sketch(image_path, blur_kernel)
        
        if original is not None and sketch is not None:
            # Generate output filename
            input_name = Path(image_path).stem
            output_path = Path(output_folder) / f"sketch_{input_name}.jpg"
            
            # Save sketch
            save_sketch(sketch, str(output_path))
            
            # Display result
            display_result(original, sketch)
        else:
            print(f"  Skipped due to error\n")


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Convert images to pencil sketch effect',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python day1_assignment_solution.py input.jpg
  
  # Process with custom blur
  python day1_assignment_solution.py input.jpg --blur 25
  
  # Process and save sketch
  python day1_assignment_solution.py input.jpg --output sketch.jpg
  
  # Create color sketch
  python day1_assignment_solution.py input.jpg --color
  
  # Process multiple images
  python day1_assignment_solution.py img1.jpg img2.jpg img3.jpg --batch output_folder/
        """
    )
    
    parser.add_argument('images', nargs='+', help='Input image path(s)')
    parser.add_argument('--output', '-o', help='Output path for sketch')
    parser.add_argument('--blur', '-b', type=int, default=21, 
                       help='Blur kernel size (odd number, default: 21)')
    parser.add_argument('--color', '-c', action='store_true',
                       help='Create color pencil sketch')
    parser.add_argument('--batch', help='Batch process to output folder')
    parser.add_argument('--intermediate', '-i', action='store_true',
                       help='Save intermediate processing steps')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display images (save only)')
    
    args = parser.parse_args()
    
    # Batch processing mode
    if args.batch:
        process_multiple_images(args.images, args.batch, args.blur)
        return
    
    # Single image processing
    if len(args.images) > 1:
        print("Multiple images provided without --batch flag")
        print("Use --batch <folder> for batch processing")
        sys.exit(1)
    
    image_path = args.images[0]
    
    # Check if file exists
    if not Path(image_path).is_file():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    print(f"Processing: {image_path}")
    print(f"Blur kernel: {args.blur}x{args.blur}")
    
    # Process image
    if args.color:
        print("Mode: Color pencil sketch")
        original, sketch = pencil_sketch_color(image_path, args.blur)
    else:
        print("Mode: Grayscale pencil sketch")
        original, sketch = pencil_sketch(image_path, args.blur, args.intermediate)
    
    if original is None or sketch is None:
        print("Processing failed!")
        sys.exit(1)
    
    # Save sketch if output path provided
    if args.output:
        save_sketch(sketch, args.output)
    
    # Display result unless --no-display flag is set
    if not args.no_display:
        mode = "Color" if args.color else "Grayscale"
        display_result(original, sketch, f"{mode} Pencil Sketch Effect", 
                      args.output if args.output else None)
    
    print("\nProcessing complete!")


if __name__ == '__main__':
    # If no command line arguments, run interactive mode
    if len(sys.argv) == 1:
        print("=== Pencil Sketch Converter ===\n")
        print("Interactive Mode")
        print("-" * 40)
        
        image_path = input("Enter image path: ").strip()
        
        if not Path(image_path).is_file():
            print(f"Error: File not found: {image_path}")
            sys.exit(1)
        
        blur_input = input("Enter blur kernel size (default 21): ").strip()
        blur_kernel = int(blur_input) if blur_input else 21
        
        color_input = input("Create color sketch? (y/n, default n): ").strip().lower()
        is_color = color_input == 'y'
        
        output_path = input("Save sketch to (press Enter to skip): ").strip()
        output_path = output_path if output_path else None
        
        print("\nProcessing...")
        
        if is_color:
            original, sketch = pencil_sketch_color(image_path, blur_kernel)
        else:
            original, sketch = pencil_sketch(image_path, blur_kernel)
        
        if original is not None and sketch is not None:
            if output_path:
                save_sketch(sketch, output_path)
            
            display_result(original, sketch)
            print("\nSuccess!")
        else:
            print("\nProcessing failed!")
    else:
        # Run command-line mode
        main()

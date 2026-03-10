#!/usr/bin/env python3
"""
Animation script to convert sequence of PNG images to MP4 video
This script takes the output images from main.py and creates an animation
"""

import os
import sys
import glob
import cv2
import argparse

def create_animation(results_dir='results', output_file='results/animation.mp4', fps=2):
    """
    Convert sequence of path_t=*.png files to an MP4 video
    
    Args:
        results_dir: Directory containing the PNG files
        output_file: Output video file path
        fps: Frames per second for the video
    """
    
    # Find all path images and sort by time step
    image_pattern = os.path.join(results_dir, 'path_t=*.png')
    images = sorted(glob.glob(image_pattern), 
                   key=lambda x: int(x.split('path_t=')[1].split('.png')[0]))
    
    if not images:
        print(f"❌ No images found matching '{image_pattern}'")
        print(f"   Make sure to run 'python3 code/main.py' first")
        return False
    
    print(f"✓ Found {len(images)} images")
    print(f"✓ Image sequence: {images[0]} ... {images[-1]}")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(images[0])
    if first_frame is None:
        print(f"❌ Could not read image: {images[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    print(f"✓ Image dimensions: {width}×{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Could not create video writer for {output_file}")
        return False
    
    print(f"✓ Writing frames to video...")
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠ Warning: Could not read {img_path}, skipping")
            continue
        out.write(frame)
    
    out.release()
    print(f"✓ Animation saved to: {output_file}")
    print(f"✓ Video length: {len(images)/fps:.1f} seconds at {fps} fps")
    
    return True


def create_gif(results_dir='results', output_file='results/animation.gif', fps=2):
    """
    Convert sequence of path_t=*.png files to an animated GIF
    
    Args:
        results_dir: Directory containing the PNG files
        output_file: Output GIF file path
        fps: Frames per second (affects delay)
    """
    
    try:
        from PIL import Image
    except ImportError:
        print("❌ PIL not installed. Install with: pip install Pillow")
        return False
    
    # Find all path images and sort by time step
    image_pattern = os.path.join(results_dir, 'path_t=*.png')
    images = sorted(glob.glob(image_pattern), 
                   key=lambda x: int(x.split('path_t=')[1].split('.png')[0]))
    
    if not images:
        print(f"❌ No images found matching '{image_pattern}'")
        return False
    
    print(f"✓ Found {len(images)} images")
    
    # Load all images
    frames = []
    for img_path in images:
        try:
            frames.append(Image.open(img_path))
        except Exception as e:
            print(f"⚠ Warning: Could not read {img_path}, skipping: {e}")
    
    if not frames:
        print("❌ No images could be loaded")
        return False
    
    # Calculate delay (in milliseconds)
    delay = int(1000 / fps)
    
    # Save as GIF
    frames[0].save(output_file, 
                   save_all=True, 
                   append_images=frames[1:],
                   duration=delay, 
                   loop=0)
    
    print(f"✓ Animated GIF saved to: {output_file}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert search path PNG images to video or GIF animation'
    )
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing path_t=*.png images (default: results)')
    parser.add_argument('--output', '-o', default='results/animation.mp4',
                       help='Output file path (default: results/animation.mp4)')
    parser.add_argument('--fps', '-f', type=int, default=2,
                       help='Frames per second (default: 2)')
    parser.add_argument('--gif', action='store_true',
                       help='Create GIF instead of MP4')
    
    args = parser.parse_args()
    
    print("\n🎬 Multi-Robot Search Path Animation Creator\n")
    
    if args.gif:
        success = create_gif(args.results_dir, args.output, args.fps)
    else:
        success = create_animation(args.results_dir, args.output, args.fps)
    
    if success:
        print("\n✓ Done!\n")
        sys.exit(0)
    else:
        print("\n❌ Failed to create animation\n")
        sys.exit(1)

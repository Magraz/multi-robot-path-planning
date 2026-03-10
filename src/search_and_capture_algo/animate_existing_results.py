#!/usr/bin/env python3
"""
Create animations from existing results directories
Use pre-generated PNG sequences without re-running the solver
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from animate_results import create_animation, create_gif


def list_result_directories():
    """List all available result directories"""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        print("No results/ directory found")
        return []
    
    dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    return sorted(dirs)


def main():
    """Create animations from existing results"""
    
    print("\n" + "="*70)
    print("  Create Animation from Existing Results")
    print("="*70)
    
    # List available result directories
    available = list_result_directories()
    
    if not available:
        print("\n❌ No result directories found in results/")
        print("\nAvailable options:")
        print("  1. Run: python3 code/main.py")
        print("  2. Or place PNG files in results/ with names: path_t=0.png, path_t=1.png, etc.")
        return 1
    
    print("\nAvailable result directories:\n")
    for i, d in enumerate(available, 1):
        png_count = len(list(d.glob("path_t=*.png")))
        print(f"  {i}. {d.name:<40} ({png_count} PNG files)")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create animation from existing results')
    parser.add_argument('--dir', '-d', 
                       help='Result directory to use (or number from list above)')
    parser.add_argument('--output', '-o', 
                       help='Output file path')
    parser.add_argument('--fps', '-f', type=int, default=2,
                       help='Frames per second (default: 2)')
    parser.add_argument('--gif', action='store_true',
                       help='Create GIF instead of MP4')
    
    args = parser.parse_args()
    
    # Determine which directory to use
    if args.dir:
        # Check if it's a number
        try:
            idx = int(args.dir) - 1
            if 0 <= idx < len(available):
                results_dir = available[idx]
            else:
                print(f"\n❌ Invalid choice: {args.dir}")
                return 1
        except ValueError:
            # It's a directory name
            results_dir = Path(args.dir)
            if not results_dir.exists():
                results_dir = Path(__file__).parent / "results" / args.dir
                if not results_dir.exists():
                    print(f"\n❌ Directory not found: {args.dir}")
                    return 1
    else:
        # Default to first available
        if len(available) == 1:
            results_dir = available[0]
        else:
            print("\n⚠️  Multiple result directories found. Choose one:")
            print("\nExamples:")
            print(f"  python3 {Path(__file__).name} --dir 1")
            print(f"  python3 {Path(__file__).name} --dir {available[0].name}")
            print(f"  python3 {Path(__file__).name} --dir {str(available[0])}")
            return 1
    
    print(f"\n✓ Using results from: {results_dir.name}")
    
    # Default output
    if not args.output:
        ext = '.gif' if args.gif else '.mp4'
        args.output = str(results_dir.parent / f"{results_dir.name}{ext}")
    
    print(f"✓ Output file: {args.output}")
    print(f"✓ Speed: {args.fps} fps\n")
    
    # Create animation
    if args.gif:
        success = create_gif(str(results_dir), args.output, args.fps)
    else:
        success = create_animation(str(results_dir), args.output, args.fps)
    
    if success:
        print("\n✓ Done!\n")
        return 0
    else:
        print("\n❌ Failed to create animation\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())

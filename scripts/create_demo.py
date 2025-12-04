#!/usr/bin/env python
"""Create a demo visualization of the virtual unwrapping pipeline.

This script generates synthetic or uses real data to demonstrate:
1. Surface segmentation
2. Surface extraction and unwrapping
3. Ink detection (simulated)
4. Visualization of unwrapped text
"""

import argparse
from pathlib import Path
import numpy as np
import torch

try:
    import tifffile
except ImportError:
    tifffile = None

from src.vesuvius.models import build_model
from src.vesuvius.unwrap import (
    extract_surface_from_volume,
    visualize_unwrapped_text,
    compute_surface_statistics
)
from src.vesuvius.submission import create_submission_csv
from src.vesuvius.utils import load_config


def create_synthetic_scroll_data(output_dir: Path, size=(100, 400, 400)):
    """Create synthetic scroll data for demo purposes.
    
    Args:
        output_dir: Directory to save synthetic data
        size: Volume size (depth, height, width)
    """
    print("Creating synthetic scroll data...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    D, H, W = size
    
    # Create base volume with noise
    volume = np.random.rand(D, H, W).astype(np.float32) * 0.3
    
    # Add papyrus surface (curved sheet)
    print("  - Adding papyrus surface...")
    for h in range(H):
        for w in range(W):
            # Create curved surface
            center_depth = D // 2 + int(10 * np.sin(w / 50) * np.cos(h / 50))
            thickness = 5
            
            # Add surface with intensity gradient
            for d in range(max(0, center_depth - thickness), min(D, center_depth + thickness)):
                distance = abs(d - center_depth)
                intensity = 0.8 * (1 - distance / thickness)
                volume[d, h, w] += intensity
    
    # Create surface mask
    surface_mask = np.zeros_like(volume, dtype=np.uint8)
    for h in range(H):
        for w in range(W):
            center_depth = D // 2 + int(10 * np.sin(w / 50) * np.cos(h / 50))
            thickness = 3
            for d in range(max(0, center_depth - thickness), min(D, center_depth + thickness)):
                surface_mask[d, h, w] = 1
    
    # Create synthetic "ink" pattern (text-like)
    print("  - Adding synthetic ink pattern...")
    ink_mask_2d = np.zeros((H, W), dtype=np.uint8)
    
    # Add some text-like patterns
    # Horizontal lines (like text lines)
    for y in range(50, H - 50, 40):
        # Add wavy line
        for x in range(50, W - 50):
            y_offset = int(3 * np.sin(x / 20))
            if 0 <= y + y_offset < H:
                # Add thickness
                for dy in range(-2, 3):
                    if 0 <= y + y_offset + dy < H:
                        ink_mask_2d[y + y_offset + dy, x] = 1
    
    # Add some vertical strokes (like letters)
    for x in range(60, W - 60, 25):
        for y_start in range(50, H - 100, 40):
            height = np.random.randint(15, 25)
            for y in range(y_start, min(y_start + height, H)):
                for dx in range(-1, 2):
                    if 0 <= x + dx < W:
                        ink_mask_2d[y, x + dx] = 1
    
    # Save data
    if tifffile:
        print("  - Saving files...")
        tifffile.imwrite(str(output_dir / "synthetic_volume.tif"), 
                        (volume * 255).astype(np.uint8))
        tifffile.imwrite(str(output_dir / "synthetic_surface_mask.tif"), 
                        surface_mask)
        tifffile.imwrite(str(output_dir / "synthetic_ink_gt.tif"), 
                        ink_mask_2d)
        print(f"✓ Synthetic data saved to {output_dir}")
    else:
        np.save(output_dir / "synthetic_volume.npy", volume)
        np.save(output_dir / "synthetic_surface_mask.npy", surface_mask)
        np.save(output_dir / "synthetic_ink_gt.npy", ink_mask_2d)
        print(f"✓ Synthetic data saved to {output_dir} (as .npy)")
    
    return volume, surface_mask, ink_mask_2d


def run_demo(args):
    """Run the complete demo pipeline."""
    print("=" * 80)
    print("Vesuvius Challenge - Virtual Unwrapping Demo")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Get data
    if args.use_synthetic:
        print("\n[1/5] Creating synthetic data...")
        data_dir = output_dir / "synthetic_data"
        volume, surface_mask, ink_gt = create_synthetic_scroll_data(data_dir)
    else:
        print("\n[1/5] Loading real data...")
        if not Path(args.volume_path).exists():
            print(f"✗ Volume not found: {args.volume_path}")
            print("  Use --use-synthetic to create demo data")
            return
        
        if tifffile:
            volume = tifffile.imread(args.volume_path).astype(np.float32)
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        else:
            volume = np.load(args.volume_path)
        
        # For real data, we'd run surface segmentation
        print("  Note: Using real data requires trained surface segmentation model")
        surface_mask = None
        ink_gt = None
    
    print(f"  Volume shape: {volume.shape}")
    
    # Step 2: Extract surface
    print("\n[2/5] Extracting and unwrapping surface...")
    
    if surface_mask is not None:
        surface_image, surface_coords = extract_surface_from_volume(
            volume, surface_mask, method="center"
        )
        
        stats = compute_surface_statistics(surface_coords)
        print(f"  Surface statistics:")
        print(f"    - Coverage: {stats['coverage']:.1%}")
        print(f"    - Mean depth: {stats['mean_depth']:.1f}")
        print(f"    - Depth range: {stats['min_depth']:.0f} - {stats['max_depth']:.0f}")
        
        # Save unwrapped surface
        if tifffile:
            tifffile.imwrite(str(output_dir / "unwrapped_surface.tif"),
                           (surface_image * 255).astype(np.uint8))
    else:
        # For real data without surface mask, use max projection as demo
        surface_image = np.max(volume, axis=0)
        print("  Using max projection (demo mode)")
    
    print(f"  Unwrapped surface shape: {surface_image.shape}")
    
    # Step 3: Simulate ink detection (or use ground truth)
    print("\n[3/5] Detecting ink...")
    
    if ink_gt is not None:
        # Use ground truth for demo
        ink_mask = ink_gt
        print("  Using ground truth ink mask")
    else:
        # Simulate ink detection with threshold
        # In real pipeline, this would be model inference
        threshold = np.percentile(surface_image, 90)
        ink_mask = (surface_image > threshold).astype(np.uint8)
        print("  Using simulated ink detection (threshold-based)")
    
    print(f"  Ink coverage: {ink_mask.sum() / ink_mask.size * 100:.2f}%")
    
    # Step 4: Create visualizations
    print("\n[4/5] Creating visualizations...")
    
    viz_files = visualize_unwrapped_text(
        surface_image,
        ink_mask,
        output_dir,
        fragment_id="demo",
        create_overlay=True,
        create_text_only=True
    )
    
    print("  Created visualizations:")
    for viz_type, path in viz_files.items():
        print(f"    ✓ {viz_type}: {path.name}")
    
    # Step 5: Generate submission (demo)
    print("\n[5/5] Generating Kaggle submission format...")
    
    predictions = {"demo_fragment": ink_mask}
    submission_path = output_dir / "demo_submission.csv"
    create_submission_csv(predictions, submission_path, validate=True)
    
    # Summary
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  Visualizations:")
    for viz_type, path in viz_files.items():
        print(f"    - {path.name}")
    print(f"  Submission:")
    print(f"    - {submission_path.name}")
    
    if args.use_synthetic:
        print(f"\n  Synthetic data:")
        print(f"    - synthetic_data/synthetic_volume.tif")
        print(f"    - synthetic_data/synthetic_surface_mask.tif")
        print(f"    - synthetic_data/synthetic_ink_gt.tif")
    
    print("\nNext steps:")
    print("  1. View visualizations in output directory")
    print("  2. Check demo_submission.csv for RLE format")
    print("  3. Use real data with --volume-path for actual inference")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Create demo visualization")
    parser.add_argument("--output-dir", default="demo_output",
                       help="Output directory for demo files")
    parser.add_argument("--use-synthetic", action="store_true",
                       help="Create and use synthetic data (recommended for demo)")
    parser.add_argument("--volume-path", default=None,
                       help="Path to real volume data (TIFF file)")
    parser.add_argument("--synthetic-size", nargs=3, type=int,
                       default=[100, 400, 400],
                       help="Size of synthetic volume (depth height width)")
    
    args = parser.parse_args()
    
    try:
        run_demo(args)
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

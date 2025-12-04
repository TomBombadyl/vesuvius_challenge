#!/usr/bin/env python
"""Visualize real Vesuvius Challenge data.

This script loads actual CT scan data and ground truth labels,
extracts the papyrus surface, and creates visualizations.
"""

import argparse
from pathlib import Path
import numpy as np
import tifffile

from src.vesuvius.unwrap import (
    extract_surface_from_volume,
    visualize_unwrapped_text,
    compute_surface_statistics
)


def main():
    parser = argparse.ArgumentParser(description="Visualize real Vesuvius data")
    parser.add_argument("--volume-id", default="1407735",
                       help="Volume ID to visualize")
    parser.add_argument("--data-root", default="vesuvius_kaggle_data",
                       help="Root directory for data")
    parser.add_argument("--output-dir", default="real_data_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--method", default="center", choices=["center", "mean", "max"],
                       help="Surface extraction method")
    parser.add_argument("--ink-threshold-percentile", type=float, default=85,
                       help="Percentile for simulated ink detection")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    volume_path = data_root / "train_images" / f"{args.volume_id}.tif"
    label_path = data_root / "train_labels" / f"{args.volume_id}.tif"
    
    print("=" * 80)
    print("Vesuvius Challenge - Real Data Visualization")
    print("=" * 80)
    
    # Check files exist
    if not volume_path.exists():
        print(f"✗ Volume not found: {volume_path}")
        return 1
    
    if not label_path.exists():
        print(f"✗ Label not found: {label_path}")
        return 1
    
    # Load data
    print(f"\n[1/4] Loading data for volume {args.volume_id}...")
    print(f"  Volume: {volume_path}")
    print(f"  Label:  {label_path}")
    
    volume = tifffile.imread(str(volume_path)).astype(np.float32)
    label = tifffile.imread(str(label_path))
    
    print(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
    print(f"  Label shape:  {label.shape}, dtype: {label.dtype}")
    print(f"  Surface voxels: {(label > 0).sum():,} ({(label > 0).sum() / label.size * 100:.1f}%)")
    
    # Normalize volume
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # Extract surface
    print(f"\n[2/4] Extracting surface (method: {args.method})...")
    surface_image, surface_coords = extract_surface_from_volume(
        volume, label, method=args.method
    )
    
    print(f"  Unwrapped surface shape: {surface_image.shape}")
    
    # Compute statistics
    stats = compute_surface_statistics(surface_coords)
    print(f"  Surface statistics:")
    print(f"    - Coverage: {stats['coverage']:.1%}")
    print(f"    - Mean depth: {stats['mean_depth']:.1f} voxels")
    print(f"    - Depth range: {stats['min_depth']:.0f} - {stats['max_depth']:.0f}")
    print(f"    - Valid pixels: {stats['num_valid_pixels']:,}")
    
    # Simulate ink detection
    print(f"\n[3/4] Simulating ink detection...")
    print(f"  Note: Using threshold-based simulation (no real ink labels)")
    print(f"  Threshold: {args.ink_threshold_percentile}th percentile")
    
    # Only consider non-zero pixels for threshold
    valid_pixels = surface_image[surface_image > 0]
    if len(valid_pixels) > 0:
        threshold = np.percentile(valid_pixels, args.ink_threshold_percentile)
        ink_mask = (surface_image > threshold).astype(np.uint8)
        ink_coverage = ink_mask.sum() / ink_mask.size * 100
        print(f"  Threshold value: {threshold:.4f}")
        print(f"  Simulated ink coverage: {ink_coverage:.2f}%")
    else:
        print("  Warning: No valid surface pixels found!")
        ink_mask = np.zeros_like(surface_image, dtype=np.uint8)
    
    # Create visualizations
    print(f"\n[4/4] Creating visualizations...")
    output_dir = Path(args.output_dir)
    
    viz_files = visualize_unwrapped_text(
        surface_image,
        ink_mask,
        output_dir,
        fragment_id=args.volume_id,
        create_overlay=True,
        create_text_only=True
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for viz_type, path in viz_files.items():
        print(f"  ✓ {viz_type}: {path.name}")
    
    print("\nOpen these files to see:")
    print("  - The unwrapped ancient papyrus surface")
    print("  - Simulated ink detection")
    print("  - Overlay visualization")
    print("  - Text-only view")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

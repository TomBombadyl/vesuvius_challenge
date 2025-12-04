#!/usr/bin/env python
"""End-to-end pipeline for Vesuvius Challenge virtual unwrapping.

This script runs the complete workflow:
1. Load 3D CT volume
2. Segment papyrus surface (if needed)
3. Extract and flatten surface
4. Detect ink on flattened surface
5. Generate Kaggle submission CSV
6. Create visualizations

Usage:
    python scripts/run_full_pipeline.py \\
        --volume-path vesuvius_kaggle_data/test_images/fragment_1/ \\
        --surface-checkpoint checkpoints/surface_segmentation.pt \\
        --ink-checkpoint checkpoints/ink_detection.pt \\
        --output-dir results/fragment_1/ \\
        --visualize
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

try:
    import tifffile
except ImportError:
    tifffile = None

from src.vesuvius.data import load_volume_from_directory
from src.vesuvius.infer import sliding_window_predict
from src.vesuvius.models import build_model
from src.vesuvius.postprocess import apply_postprocessing
from src.vesuvius.submission import create_submission_csv, mask_to_rle
from src.vesuvius.unwrap import (
    extract_surface_from_volume,
    visualize_unwrapped_text,
    compute_surface_statistics,
)
from src.vesuvius.utils import load_config, configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Vesuvius unwrapping pipeline")
    
    # Input/output paths
    parser.add_argument("--volume-path", required=True, help="Path to volume directory or TIFF file")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--fragment-id", default=None, help="Fragment ID (auto-detected if not provided)")
    
    # Model checkpoints
    parser.add_argument("--surface-checkpoint", default=None, help="Surface segmentation model checkpoint")
    parser.add_argument("--ink-checkpoint", required=True, help="Ink detection model checkpoint")
    
    # Config files
    parser.add_argument("--surface-config", default="configs/experiments/exp001_3d_unet_topology.yaml",
                        help="Surface segmentation config")
    parser.add_argument("--ink-config", default="configs/ink_detection.yaml",
                        help="Ink detection config")
    
    # Pipeline options
    parser.add_argument("--skip-surface-segmentation", action="store_true",
                        help="Skip surface segmentation (use if volume is already a flattened surface)")
    parser.add_argument("--surface-method", default="center", choices=["center", "mean", "max"],
                        help="Method for extracting surface from 3D volume")
    
    # Inference options
    parser.add_argument("--threshold", type=float, default=None,
                        help="Binarization threshold (uses config default if not specified)")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    
    # Output options
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--save-intermediate", action="store_true",
                        help="Save intermediate outputs (surface mask, unwrapped surface, etc.)")
    parser.add_argument("--create-submission", action="store_true",
                        help="Create Kaggle submission CSV")
    
    return parser.parse_args()


def load_volume(volume_path: Path, logger) -> np.ndarray:
    """Load 3D volume from directory or TIFF file."""
    volume_path = Path(volume_path)
    
    if volume_path.is_file() and volume_path.suffix in ['.tif', '.tiff']:
        logger.info(f"Loading volume from TIFF: {volume_path}")
        if tifffile is None:
            raise ImportError("tifffile required to load TIFF volumes")
        volume = tifffile.imread(str(volume_path))
    elif volume_path.is_dir():
        logger.info(f"Loading volume from directory: {volume_path}")
        # Assume directory contains numbered slices
        slice_files = sorted(volume_path.glob("*.tif")) + sorted(volume_path.glob("*.tiff"))
        if not slice_files:
            raise ValueError(f"No TIFF files found in {volume_path}")
        
        if tifffile is None:
            raise ImportError("tifffile required to load TIFF volumes")
        
        slices = [tifffile.imread(str(f)) for f in slice_files]
        volume = np.stack(slices, axis=0)
    else:
        raise ValueError(f"Invalid volume path: {volume_path}")
    
    logger.info(f"Loaded volume: shape={volume.shape}, dtype={volume.dtype}")
    return volume


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] range."""
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    return volume.astype(np.float32)


def segment_surface(
    volume: np.ndarray,
    model: torch.nn.Module,
    config: Dict,
    device: torch.device,
    logger
) -> np.ndarray:
    """Segment papyrus surface from 3D volume."""
    logger.info("Running surface segmentation...")
    start = time.time()
    
    # Prepare input
    volume_tensor = torch.from_numpy(volume[None, None, ...])  # Add batch and channel dims
    
    # Run inference
    pred = sliding_window_predict(model, volume_tensor, config["inference"], device)
    
    # Binarize
    threshold = config["inference"].get("threshold", 0.5)
    surface_mask = (pred >= threshold).astype(np.uint8)
    
    elapsed = time.time() - start
    logger.info(f"Surface segmentation complete in {elapsed:.1f}s")
    logger.info(f"Surface coverage: {surface_mask.sum() / surface_mask.size * 100:.1f}%")
    
    return surface_mask


def detect_ink(
    surface_volume: np.ndarray,
    model: torch.nn.Module,
    config: Dict,
    device: torch.device,
    logger,
    use_tta: bool = False
) -> np.ndarray:
    """Detect ink on flattened surface volume."""
    logger.info("Running ink detection...")
    start = time.time()
    
    # Prepare input
    surface_tensor = torch.from_numpy(surface_volume[None, None, ...])
    
    # Update config for TTA if requested
    if use_tta:
        config["inference"]["tta"] = "flips"
    
    # Run inference
    from src.vesuvius.infer import inference_with_tta
    pred = inference_with_tta(model, surface_tensor, config["inference"], device)
    
    # Binarize
    threshold = config["inference"].get("threshold", 0.5)
    ink_mask = (pred >= threshold).astype(np.uint8)
    
    # Post-process
    post_cfg = config.get("postprocess", {})
    ink_mask = apply_postprocessing(ink_mask, post_cfg)
    
    elapsed = time.time() - start
    logger.info(f"Ink detection complete in {elapsed:.1f}s")
    logger.info(f"Ink coverage: {ink_mask.sum() / ink_mask.size * 100:.2f}%")
    
    return ink_mask


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = configure_logging(output_dir / "pipeline.log")
    logger.info("=" * 80)
    logger.info("Vesuvius Challenge - Virtual Unwrapping Pipeline")
    logger.info("=" * 80)
    
    # Determine fragment ID
    fragment_id = args.fragment_id
    if fragment_id is None:
        fragment_id = Path(args.volume_path).stem
    logger.info(f"Fragment ID: {fragment_id}")
    
    # Load volume
    volume = load_volume(Path(args.volume_path), logger)
    volume = normalize_volume(volume)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Step 1: Surface Segmentation (if needed)
    if args.skip_surface_segmentation:
        logger.info("Skipping surface segmentation (using volume as-is)")
        surface_volume = volume
        surface_mask = None
    else:
        if args.surface_checkpoint is None:
            raise ValueError("--surface-checkpoint required when not skipping surface segmentation")
        
        # Load surface segmentation model
        logger.info(f"Loading surface segmentation model from {args.surface_checkpoint}")
        surface_config = load_config(args.surface_config)
        surface_model = build_model(surface_config["model"])
        checkpoint = torch.load(args.surface_checkpoint, map_location="cpu")
        surface_model.load_state_dict(checkpoint["state_dict"])
        surface_model.to(device)
        surface_model.eval()
        
        # Segment surface
        surface_mask = segment_surface(volume, surface_model, surface_config, device, logger)
        
        # Extract and flatten surface
        logger.info(f"Extracting surface using method: {args.surface_method}")
        from src.vesuvius.unwrap import extract_surface_from_volume
        surface_image, surface_coords = extract_surface_from_volume(
            volume, surface_mask, method=args.surface_method
        )
        
        # Compute surface statistics
        stats = compute_surface_statistics(surface_coords)
        logger.info(f"Surface statistics: {stats}")
        
        # For ink detection, we need a 3D volume around the surface
        # Use the original volume as surface_volume for now
        # In a more sophisticated pipeline, extract neighborhood around surface
        surface_volume = volume
        
        if args.save_intermediate:
            if tifffile:
                tifffile.imwrite(str(output_dir / f"{fragment_id}_surface_mask.tif"), surface_mask)
                tifffile.imwrite(str(output_dir / f"{fragment_id}_surface_image.tif"), 
                                (surface_image * 255).astype(np.uint8))
            logger.info("Saved intermediate surface outputs")
    
    # Step 2: Ink Detection
    logger.info(f"Loading ink detection model from {args.ink_checkpoint}")
    ink_config = load_config(args.ink_config)
    
    # Override threshold if specified
    if args.threshold is not None:
        ink_config["inference"]["threshold"] = args.threshold
        logger.info(f"Using custom threshold: {args.threshold}")
    
    ink_model = build_model(ink_config["model"])
    checkpoint = torch.load(args.ink_checkpoint, map_location="cpu")
    ink_model.load_state_dict(checkpoint["state_dict"])
    ink_model.to(device)
    ink_model.eval()
    
    # Detect ink
    ink_mask_3d = detect_ink(surface_volume, ink_model, ink_config, device, logger, use_tta=args.tta)
    
    # Project to 2D (take max projection or slice)
    if ink_mask_3d.ndim == 3:
        ink_mask_2d = np.max(ink_mask_3d, axis=0)  # Max projection
    else:
        ink_mask_2d = ink_mask_3d
    
    logger.info(f"2D ink mask shape: {ink_mask_2d.shape}")
    
    # Save ink mask
    if tifffile:
        tifffile.imwrite(str(output_dir / f"{fragment_id}_ink_mask.tif"), ink_mask_2d)
    
    # Step 3: Create Kaggle Submission (if requested)
    if args.create_submission:
        logger.info("Creating Kaggle submission CSV...")
        predictions = {fragment_id: ink_mask_2d}
        submission_path = output_dir / "submission.csv"
        create_submission_csv(predictions, submission_path)
    
    # Step 4: Visualizations (if requested)
    if args.visualize:
        logger.info("Creating visualizations...")
        
        # Get surface image for visualization
        if surface_mask is not None:
            surface_image_vis = surface_image
        else:
            # Use max projection of volume
            surface_image_vis = np.max(volume, axis=0)
        
        viz_files = visualize_unwrapped_text(
            surface_image_vis,
            ink_mask_2d,
            output_dir,
            fragment_id=fragment_id,
            create_overlay=True,
            create_text_only=True
        )
        
        logger.info("Visualizations created:")
        for viz_type, path in viz_files.items():
            logger.info(f"  - {viz_type}: {path}")
    
    logger.info("=" * 80)
    logger.info("Pipeline complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

"""
Validate trained model on external dataset with ground truth masks.
Reuses sliding-window inference from infer.py.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import tifffile

from .infer import sliding_window_predict, inference_with_tta, gaussian_weights
from .metrics import compute_dice, compute_iou
from .models import build_model
from .utils import configure_logging, load_config


def load_external_volume_pair(
    image_path: Path,
    mask_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image and mask from external dataset.
    Applies same normalization as training pipeline.
    
    Returns:
        (image_np, mask_np): Both shape (D, H, W), normalized to [0,1]
    """
    # Load image
    image = tifffile.imread(str(image_path)).astype(np.float32)
    
    # Normalize to [0, 1] (per-volume min-max, matching training)
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)
    
    # Load mask
    mask = tifffile.imread(str(mask_path)).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)  # Binarize
    
    # Ensure both are (D, H, W) - adjust if needed for your convention
    # If loaded as (H, W, D), transpose to (D, H, W)
    if image.ndim == 3 and image.shape[0] < image.shape[2]:
        # Likely (H, W, D), transpose to (D, H, W)
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
    
    return image, mask


def validate_external_dataset(
    model,
    image_paths: list,
    mask_paths: list,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    logger,
) -> pd.DataFrame:
    """
    Run inference on external volumes and compute metrics.
    """
    model.eval()
    inference_cfg = cfg["inference"]
    thresholds = np.linspace(0.30, 0.55, 25)
    
    results = []
    
    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths), 1):
        volume_id = img_path.stem
        logger.info(f"[{idx}/{len(image_paths)}] Processing {volume_id}...")
        
        try:
            # Load external pair
            image_np, mask_np = load_external_volume_pair(img_path, mask_path)
            logger.info(f"  Image shape: {image_np.shape}, dtype: {image_np.dtype}")
            logger.info(f"  Mask shape: {mask_np.shape}, range: [{mask_np.min()}, {mask_np.max()}]")
            
            # Convert to tensor (batch + channel)
            volume_tensor = torch.from_numpy(image_np[np.newaxis, np.newaxis, ...]).float().to(device)
            
            # Run inference with TTA if configured
            pred_np = inference_with_tta(model, volume_tensor, inference_cfg, device)
            logger.info(f"  Prediction range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
            
            # Evaluate over threshold range
            gt = (mask_np > 0).astype(bool)
            
            for thresh in thresholds:
                pred = (pred_np >= thresh).astype(bool)
                
                # Compute metrics
                dice = compute_dice(pred, gt)
                iou = compute_iou(pred, gt)
                
                # Also compute per-class precision/recall
                tp = np.sum(pred & gt)
                fp = np.sum(pred & ~gt)
                fn = np.sum(~pred & gt)
                
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                
                results.append({
                    "volume_id": volume_id,
                    "threshold": float(thresh),
                    "dice": float(dice),
                    "iou": float(iou),
                    "precision": float(precision),
                    "recall": float(recall),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                })
            
            logger.info(f"  ✓ Completed {volume_id}")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {volume_id}: {e}", exc_info=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = output_dir / "external_validation_results.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info(f"Saved results to {csv_path}")
    
    # Print summary
    logger.info("\n=== EXTERNAL VALIDATION SUMMARY ===")
    logger.info("\nPer-threshold statistics (mean across volumes):")
    summary = df.groupby("threshold")[["dice", "iou", "precision", "recall"]].mean()
    logger.info(summary.to_string())
    
    # Best threshold by Dice
    best_dice_idx = summary["dice"].idxmax()
    best_dice = summary.loc[best_dice_idx, "dice"]
    logger.info(f"\nBest threshold by Dice: {best_dice_idx:.2f} (Dice={best_dice:.4f})")
    
    # Per-volume statistics
    logger.info("\nPer-volume statistics (at default threshold 0.42):")
    per_vol = df[df["threshold"] == 0.42][["volume_id", "dice", "iou"]]
    logger.info(per_vol.to_string())
    
    logger.info(f"\nMean Dice (at 0.42): {per_vol['dice'].mean():.4f} ± {per_vol['dice'].std():.4f}")
    logger.info(f"Mean IoU (at 0.42): {per_vol['iou'].mean():.4f} ± {per_vol['iou'].std():.4f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Validate on external dataset")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--image-dir", required=True, help="Directory with image TIFFs")
    parser.add_argument("--mask-dir", required=True, help="Directory with mask TIFFs")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-volumes", type=int, default=None, help="Limit to N volumes (for testing)")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "validate_external.log")
    
    logger.info("=== External Dataset Validation ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Image dir: {args.image_dir}")
    logger.info(f"Mask dir: {args.mask_dir}")
    
    # Load config and model
    cfg = load_config(args.config)
    model = build_model(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded on {device}")
    
    # Find image/mask pairs
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    
    image_files = sorted(image_dir.glob("*.tif"))
    if args.max_volumes:
        image_files = image_files[:args.max_volumes]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Build mask paths (assumes same filename pattern)
    mask_files = [mask_dir / img.name for img in image_files]
    
    # Validate
    results_df = validate_external_dataset(
        model,
        image_files,
        mask_files,
        cfg,
        device,
        output_dir,
        logger,
    )
    
    logger.info("\n✓ External validation complete!")


if __name__ == "__main__":
    main()


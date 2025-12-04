#!/usr/bin/env python
"""Optimize binarization threshold on validation data."""

import argparse
from pathlib import Path
import numpy as np
import torch
import tifffile
from src.vesuvius.models import build_model
from src.vesuvius.infer import sliding_window_predict
from src.vesuvius.metrics import evaluate_metrics
from src.vesuvius.utils import load_config, configure_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize binarization threshold")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--val-images-dir", required=True, help="Directory with validation images")
    parser.add_argument("--val-labels-dir", required=True, help="Directory with validation labels")
    parser.add_argument("--thresholds", nargs="+", type=float, 
                        default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
                        help="Thresholds to evaluate")
    parser.add_argument("--output-dir", default="threshold_optimization",
                        help="Output directory for results")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--max-volumes", type=int, default=None,
                        help="Maximum number of volumes to process")
    return parser.parse_args()


def evaluate_threshold(predictions, ground_truth, threshold, spacing):
    """Evaluate metrics for a specific threshold."""
    binary_pred = (predictions >= threshold).astype(np.uint8)
    metrics = evaluate_metrics(binary_pred, ground_truth, 
                               tolerance_mm=2.0, spacing=spacing)
    return metrics


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = configure_logging(output_dir / "optimize_threshold.log")
    logger.info("=" * 80)
    logger.info("Threshold Optimization")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Validation images: {args.val_images_dir}")
    logger.info(f"Validation labels: {args.val_labels_dir}")
    logger.info(f"Thresholds to test: {args.thresholds}")
    
    # Load model
    logger.info("\nLoading model...")
    cfg = load_config(args.config)
    model = build_model(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model loaded: {num_params/1e6:.1f}M parameters")
    logger.info(f"  Device: {device}")
    
    # Get validation volumes
    val_images = sorted(Path(args.val_images_dir).glob("*.tif"))
    if args.max_volumes:
        val_images = val_images[:args.max_volumes]
    
    logger.info(f"\nFound {len(val_images)} validation volumes")
    
    if len(val_images) == 0:
        logger.error("No validation images found!")
        return
    
    # Run inference on all validation volumes
    logger.info("\nRunning inference on validation set...")
    logger.info("=" * 80)
    
    all_predictions = []
    all_ground_truths = []
    
    for idx, img_path in enumerate(val_images, 1):
        volume_id = img_path.stem
        label_path = Path(args.val_labels_dir) / f"{volume_id}.tif"
        
        if not label_path.exists():
            logger.warning(f"[{idx}/{len(val_images)}] No label for {volume_id}, skipping")
            continue
        
        logger.info(f"[{idx}/{len(val_images)}] Processing {volume_id}...")
        
        try:
            # Load and normalize
            volume = tifffile.imread(str(img_path)).astype(np.float32)
            logger.info(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
            
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            
            # Load ground truth
            gt = tifffile.imread(str(label_path))
            logger.info(f"  GT shape: {gt.shape}, positive voxels: {gt.sum()}")
            
            # Run inference
            volume_tensor = torch.from_numpy(volume[None, None, ...])
            with torch.no_grad():
                pred = sliding_window_predict(model, volume_tensor, 
                                              cfg["inference"], device)
            
            logger.info(f"  Prediction shape: {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
            
            all_predictions.append(pred)
            all_ground_truths.append(gt)
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {volume_id}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_predictions) == 0:
        logger.error("No predictions generated! Check your data paths.")
        return
    
    logger.info(f"\n✓ Inference complete: {len(all_predictions)} volumes processed")
    
    # Evaluate each threshold
    spacing = tuple(cfg["data"].get("resample_spacing", [0.04, 0.04, 0.04]))
    results = []
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating thresholds")
    logger.info("=" * 80)
    
    for threshold in args.thresholds:
        metrics_list = []
        
        for pred, gt in zip(all_predictions, all_ground_truths):
            metrics = evaluate_threshold(pred, gt, threshold, spacing)
            metrics_list.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        
        results.append({
            'threshold': threshold,
            **avg_metrics
        })
        
        logger.info(f"Threshold {threshold:.2f}: "
                   f"Dice={avg_metrics['dice']:.4f}, "
                   f"IoU={avg_metrics['iou']:.4f}, "
                   f"Precision={avg_metrics.get('precision', 0):.4f}, "
                   f"Recall={avg_metrics.get('recall', 0):.4f}")
    
    # Find best threshold
    logger.info("\n" + "=" * 80)
    logger.info("Results")
    logger.info("=" * 80)
    
    best_result = max(results, key=lambda x: x['dice'])
    logger.info(f"\n✓ Best threshold (by Dice): {best_result['threshold']:.2f}")
    logger.info(f"  Dice: {best_result['dice']:.4f}")
    logger.info(f"  IoU: {best_result['iou']:.4f}")
    logger.info(f"  Precision: {best_result.get('precision', 0):.4f}")
    logger.info(f"  Recall: {best_result.get('recall', 0):.4f}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = output_dir / "threshold_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Results saved to {csv_path}")
    
    # Print recommendation
    logger.info("\n" + "=" * 80)
    logger.info("Recommendation")
    logger.info("=" * 80)
    logger.info(f"Update your config file with:")
    logger.info(f"  inference:")
    logger.info(f"    threshold: {best_result['threshold']:.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

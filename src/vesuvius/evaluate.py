"""
Evaluation script for validation/test predictions.

Computes Surface Dice, VOI, and Topology Score for predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tifffile

from .data import read_metadata, split_records_by_fold
from .metrics import evaluate_metrics
from .utils import configure_logging, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--predictions-dir", required=True, help="Directory containing prediction .tif files")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val",
                        help="Split to evaluate (train/val/test)")
    parser.add_argument("--output-csv", help="Optional: save results to CSV")
    return parser.parse_args()


def load_prediction(pred_path: Path) -> np.ndarray:
    """Load prediction from .tif file."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    return tifffile.imread(pred_path.as_posix()).astype(bool)


def load_ground_truth(label_path: Path) -> np.ndarray:
    """Load ground truth label, handling unlabeled pixels (value=2)."""
    if not label_path or not label_path.exists():
        return None
    label = tifffile.imread(label_path.as_posix())
    # Convert: 0=background, 1=foreground, 2=unlabeled -> 0=background, 1=foreground
    # Mask out unlabeled pixels (value=2)
    mask = label == 1
    return mask.astype(bool)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    predictions_dir = Path(args.predictions_dir)
    
    logger = configure_logging()
    logger.info("Evaluating predictions from %s", predictions_dir)
    
    # Load metadata
    if args.split == "test":
        logger.warning("Test set has no ground truth labels. Evaluation not possible.")
        return
    
    metadata = read_metadata(Path(cfg["paths"]["train_csv"]), data_root)
    
    # Filter by split if needed
    if args.split in ["train", "val"]:
        data_cfg = cfg["data"]
        train_folds = data_cfg.get("train_folds", [0, 1, 2])
        val_folds = data_cfg.get("val_folds", [3])
        train_records, val_records = split_records_by_fold(metadata, train_folds, val_folds)
        if args.split == "train":
            metadata = train_records
        else:
            metadata = val_records
    
    # Get spacing and tolerance
    spacing = tuple(cfg["data"].get("resample_spacing", [0.04, 0.04, 0.04]))
    tolerance_mm = cfg["metrics"].get("surface_dice_tolerance_mm", 2.0)
    
    results: List[Dict[str, float | str]] = []
    
    for record in metadata:
        volume_id = record.volume_id
        pred_path = predictions_dir / f"{volume_id}.tif"
        
        if not pred_path.exists():
            logger.warning("Prediction not found for %s, skipping", volume_id)
            continue
        
        pred_mask = load_prediction(pred_path)
        gt_mask = load_ground_truth(record.label_path)
        
        if gt_mask is None:
            logger.warning("Ground truth not found for %s, skipping", volume_id)
            continue
        
        # Ensure shapes match
        if pred_mask.shape != gt_mask.shape:
            logger.warning("Shape mismatch for %s: pred %s vs gt %s, skipping",
                          volume_id, pred_mask.shape, gt_mask.shape)
            continue
        
        # Compute metrics
        metrics = evaluate_metrics(pred_mask, gt_mask, tolerance_mm, spacing)
        
        result = {
            "volume_id": volume_id,
            **metrics
        }
        results.append(result)
        
        logger.info("Volume %s: SurfaceDice=%.4f, VOI=%.4f, TopoScore=%.4f",
                   volume_id, metrics["surface_dice"], metrics["voi"], metrics["topo_score"])
    
    if not results:
        logger.error("No valid predictions found for evaluation")
        return
    
    # Compute averages
    df = pd.DataFrame(results)
    avg_metrics = {
        "surface_dice": df["surface_dice"].mean(),
        "voi": df["voi"].mean(),
        "topo_score": df["topo_score"].mean(),
    }
    
    logger.info("=" * 60)
    logger.info("Average Metrics (%d volumes):", len(results))
    logger.info("  Surface Dice: %.4f", avg_metrics["surface_dice"])
    logger.info("  VOI:          %.4f", avg_metrics["voi"])
    logger.info("  TopoScore:    %.4f", avg_metrics["topo_score"])
    logger.info("=" * 60)
    
    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        df.to_csv(output_path, index=False)
        logger.info("Results saved to %s", output_path)
        
        # Also save summary
        summary_path = output_path.with_suffix(".summary.csv")
        summary_df = pd.DataFrame([avg_metrics])
        summary_df.to_csv(summary_path, index=False)
        logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()


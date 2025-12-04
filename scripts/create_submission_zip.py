#!/usr/bin/env python
"""Create Kaggle submission.zip from prediction masks.

This script takes prediction masks and creates a properly formatted
submission.zip file for the Vesuvius Challenge Surface Detection competition.

Submission format requirements:
- File: submission.zip
- Contents: One .tif file per test image
- Naming: [image_id].tif
- Dimensions: Must match source image exactly
- Data type: uint8 (same as train masks)
"""

import argparse
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tifffile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Kaggle submission.zip from predictions"
    )
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing prediction .tif files",
    )
    parser.add_argument(
        "--test-csv",
        default="vesuvius_kaggle_data/test.csv",
        help="Path to test.csv with expected image IDs",
    )
    parser.add_argument(
        "--output-path",
        default="submission.zip",
        help="Output path for submission.zip",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate submission format before creating ZIP",
    )
    return parser.parse_args()


def load_test_ids(test_csv_path: Path) -> List[str]:
    """Load expected test image IDs from test.csv."""
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")
    
    df = pd.read_csv(test_csv_path)
    
    # Try different possible column names
    for col in ["Id", "id", "volume_id", "image_id", "fragment_id"]:
        if col in df.columns:
            return df[col].astype(str).tolist()
    
    raise ValueError(f"Could not find ID column in {test_csv_path}")


def validate_prediction_file(
    pred_path: Path,
    expected_shape: tuple = None,
    expected_dtype: np.dtype = np.uint8
) -> Dict[str, any]:
    """Validate a single prediction file.
    
    Returns:
        Dict with validation results and metadata
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "shape": None,
        "dtype": None,
        "file_size_mb": pred_path.stat().st_size / (1024 * 1024),
    }
    
    try:
        # Load prediction
        pred = tifffile.imread(str(pred_path))
        result["shape"] = pred.shape
        result["dtype"] = pred.dtype
        
        # Check data type
        if pred.dtype != expected_dtype:
            result["errors"].append(
                f"Wrong dtype: {pred.dtype}, expected {expected_dtype}"
            )
            result["valid"] = False
        
        # Check shape if provided
        if expected_shape is not None and pred.shape != expected_shape:
            result["errors"].append(
                f"Wrong shape: {pred.shape}, expected {expected_shape}"
            )
            result["valid"] = False
        
        # Check values are binary (0 or 1)
        unique_vals = np.unique(pred)
        if not np.all((unique_vals == 0) | (unique_vals == 1)):
            result["warnings"].append(
                f"Non-binary values found: {unique_vals}"
            )
        
        # Check for empty predictions
        if pred.sum() == 0:
            result["warnings"].append("Prediction is empty (all zeros)")
        
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to load: {e}")
    
    return result


def create_submission_zip(
    predictions_dir: Path,
    test_ids: List[str],
    output_path: Path,
    validate: bool = True
) -> None:
    """Create submission.zip from prediction files.
    
    Args:
        predictions_dir: Directory containing .tif prediction files
        test_ids: List of expected test image IDs
        output_path: Path for output submission.zip
        validate: If True, validate each file before adding to ZIP
    """
    predictions_dir = Path(predictions_dir)
    output_path = Path(output_path)
    
    print("=" * 80)
    print("Creating Kaggle Submission ZIP")
    print("=" * 80)
    print(f"Predictions directory: {predictions_dir}")
    print(f"Expected test IDs: {len(test_ids)}")
    print(f"Output: {output_path}")
    print()
    
    # Find all prediction files
    pred_files = {}
    for test_id in test_ids:
        pred_path = predictions_dir / f"{test_id}.tif"
        if pred_path.exists():
            pred_files[test_id] = pred_path
        else:
            print(f"⚠ Warning: Missing prediction for {test_id}")
    
    print(f"Found {len(pred_files)}/{len(test_ids)} prediction files")
    
    if len(pred_files) == 0:
        raise ValueError("No prediction files found!")
    
    # Validate predictions
    if validate:
        print("\nValidating predictions...")
        all_valid = True
        
        for test_id, pred_path in pred_files.items():
            result = validate_prediction_file(pred_path)
            
            status = "✓" if result["valid"] else "✗"
            print(f"  {status} {test_id}: shape={result['shape']}, "
                  f"dtype={result['dtype']}, size={result['file_size_mb']:.1f}MB")
            
            if result["errors"]:
                for error in result["errors"]:
                    print(f"      Error: {error}")
                all_valid = False
            
            if result["warnings"]:
                for warning in result["warnings"]:
                    print(f"      Warning: {warning}")
        
        if not all_valid:
            raise ValueError("Validation failed! Fix errors before creating submission.")
        
        print("\n✓ All predictions validated")
    
    # Create ZIP file
    print(f"\nCreating {output_path}...")
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for test_id, pred_path in pred_files.items():
            # Add file to ZIP with just the filename (no directory structure)
            zf.write(pred_path, arcname=f"{test_id}.tif")
    
    # Verify ZIP was created
    if not output_path.exists():
        raise RuntimeError("Failed to create submission.zip")
    
    zip_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print("=" * 80)
    print("✓ Submission ZIP created successfully!")
    print("=" * 80)
    print(f"File: {output_path}")
    print(f"Size: {zip_size_mb:.1f} MB")
    print(f"Contains: {len(pred_files)} prediction files")
    
    # List contents
    print("\nContents:")
    with zipfile.ZipFile(output_path, "r") as zf:
        for name in sorted(zf.namelist()):
            info = zf.getinfo(name)
            print(f"  - {name} ({info.file_size / 1024:.1f} KB)")
    
    print("\nNext steps:")
    print("  1. Validate: python scripts/validate_submission.py")
    print("  2. Upload to Kaggle competition")
    print("=" * 80)


def main():
    args = parse_args()
    
    try:
        # Load test IDs
        test_ids = load_test_ids(Path(args.test_csv))
        
        # Create submission
        create_submission_zip(
            Path(args.predictions_dir),
            test_ids,
            Path(args.output_path),
            validate=args.validate
        )
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

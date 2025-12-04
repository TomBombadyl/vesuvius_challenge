#!/usr/bin/env python
"""Validate Kaggle submission.zip format.

This script checks that a submission.zip file meets all requirements
for the Vesuvius Challenge Surface Detection competition.

Requirements:
- File named submission.zip
- Contains .tif files named [image_id].tif
- Each .tif matches source image dimensions
- Data type is uint8
- All test images have predictions
"""

import argparse
import zipfile
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import tifffile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Kaggle submission.zip format"
    )
    parser.add_argument(
        "--submission-path",
        default="submission.zip",
        help="Path to submission.zip file",
    )
    parser.add_argument(
        "--test-csv",
        default="vesuvius_kaggle_data/test.csv",
        help="Path to test.csv with expected IDs",
    )
    parser.add_argument(
        "--test-images-dir",
        default="vesuvius_kaggle_data/test_images",
        help="Directory with test images (for dimension checking)",
    )
    parser.add_argument(
        "--check-dimensions",
        action="store_true",
        help="Verify dimensions match source images (slower)",
    )
    return parser.parse_args()


def load_test_ids(test_csv_path: Path) -> Set[str]:
    """Load expected test image IDs."""
    df = pd.read_csv(test_csv_path)
    
    for col in ["Id", "id", "volume_id", "image_id"]:
        if col in df.columns:
            return set(df[col].astype(str).tolist())
    
    raise ValueError(f"Could not find ID column in {test_csv_path}")


def validate_submission_zip(
    submission_path: Path,
    expected_ids: Set[str],
    test_images_dir: Path = None,
    check_dimensions: bool = False
) -> Dict:
    """Validate submission.zip format.
    
    Returns:
        Dict with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # Check file exists
    if not submission_path.exists():
        results["valid"] = False
        results["errors"].append(f"Submission file not found: {submission_path}")
        return results
    
    # Check it's a ZIP file
    if not zipfile.is_zipfile(submission_path):
        results["valid"] = False
        results["errors"].append("File is not a valid ZIP archive")
        return results
    
    # Open and inspect ZIP
    with zipfile.ZipFile(submission_path, "r") as zf:
        file_list = zf.namelist()
        
        # Check for directory structure (should be flat)
        if any("/" in name or "\\" in name for name in file_list):
            results["warnings"].append(
                "ZIP contains directory structure (should be flat)"
            )
        
        # Extract IDs from filenames
        submitted_ids = set()
        for filename in file_list:
            # Get base name without path
            base_name = Path(filename).name
            
            # Check extension
            if not base_name.endswith(".tif"):
                results["errors"].append(
                    f"Invalid file extension: {base_name} (must be .tif)"
                )
                results["valid"] = False
                continue
            
            # Extract ID
            image_id = base_name[:-4]  # Remove .tif
            submitted_ids.add(image_id)
        
        # Check for missing IDs
        missing_ids = expected_ids - submitted_ids
        if missing_ids:
            results["errors"].append(
                f"Missing predictions for {len(missing_ids)} images: {list(missing_ids)[:5]}..."
            )
            results["valid"] = False
        
        # Check for extra IDs
        extra_ids = submitted_ids - expected_ids
        if extra_ids:
            results["warnings"].append(
                f"Extra predictions for {len(extra_ids)} images: {list(extra_ids)[:5]}..."
            )
        
        # Validate each file
        print("\nValidating individual files...")
        file_stats = []
        
        for filename in file_list:
            base_name = Path(filename).name
            if not base_name.endswith(".tif"):
                continue
            
            image_id = base_name[:-4]
            
            # Read file from ZIP
            with zf.open(filename) as f:
                file_data = f.read()
            
            # Load as TIFF
            try:
                import io
                pred = tifffile.imread(io.BytesIO(file_data))
                
                # Check dtype
                if pred.dtype != np.uint8:
                    results["errors"].append(
                        f"{base_name}: Wrong dtype {pred.dtype} (must be uint8)"
                    )
                    results["valid"] = False
                
                # Check dimensions if requested
                if check_dimensions and test_images_dir:
                    source_path = test_images_dir / f"{image_id}.tif"
                    if source_path.exists():
                        source = tifffile.imread(str(source_path))
                        if pred.shape != source.shape:
                            results["errors"].append(
                                f"{base_name}: Shape mismatch {pred.shape} vs {source.shape}"
                            )
                            results["valid"] = False
                
                # Collect stats
                file_stats.append({
                    "id": image_id,
                    "shape": pred.shape,
                    "dtype": str(pred.dtype),
                    "size_kb": len(file_data) / 1024,
                    "coverage": pred.sum() / pred.size * 100,
                })
                
                status = "✓" if pred.dtype == np.uint8 else "✗"
                print(f"  {status} {base_name}: {pred.shape}, {pred.dtype}, "
                      f"{len(file_data)/1024:.1f}KB, {pred.sum()/pred.size*100:.1f}% coverage")
                
            except Exception as e:
                results["errors"].append(f"{base_name}: Failed to read - {e}")
                results["valid"] = False
        
        results["stats"] = {
            "num_files": len(file_list),
            "num_predictions": len(submitted_ids),
            "total_size_mb": submission_path.stat().st_size / (1024 * 1024),
            "file_details": file_stats,
        }
    
    return results


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Kaggle Submission Validator")
    print("=" * 80)
    print(f"Submission: {args.submission_path}")
    print(f"Test CSV: {args.test_csv}")
    
    try:
        # Load expected IDs
        expected_ids = load_test_ids(Path(args.test_csv))
        print(f"Expected test IDs: {len(expected_ids)}")
        
        # Validate submission
        results = validate_submission_zip(
            Path(args.submission_path),
            expected_ids,
            Path(args.test_images_dir) if args.check_dimensions else None,
            check_dimensions=args.check_dimensions
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("Validation Results")
        print("=" * 80)
        
        if results["valid"]:
            print("✓ SUBMISSION IS VALID")
        else:
            print("✗ SUBMISSION HAS ERRORS")
        
        if results["errors"]:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"  ✗ {error}")
        
        if results["warnings"]:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"  ⚠ {warning}")
        
        if results["stats"]:
            print(f"\nStatistics:")
            print(f"  Files in ZIP: {results['stats']['num_files']}")
            print(f"  Predictions: {results['stats']['num_predictions']}")
            print(f"  Total size: {results['stats']['total_size_mb']:.1f} MB")
        
        print("=" * 80)
        
        return 0 if results["valid"] else 1
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

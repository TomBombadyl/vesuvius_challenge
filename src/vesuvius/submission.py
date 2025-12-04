"""Kaggle submission utilities for Vesuvius Challenge.

This module provides functions to convert binary masks to run-length encoding (RLE)
and generate submission CSV files in the format required by Kaggle competitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def mask_to_rle(mask: np.ndarray, use_1_indexing: bool = True) -> str:
    """Convert 2D binary mask to run-length encoding.
    
    Args:
        mask: 2D binary array (H, W) with 0s and 1s
        use_1_indexing: If True, use 1-indexed positions (Kaggle standard)
        
    Returns:
        RLE string in format "start1 length1 start2 length2 ..."
        
    Example:
        >>> mask = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])
        >>> rle = mask_to_rle(mask)
        >>> print(rle)  # "3 2 5 2" (1-indexed)
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    
    # Flatten in row-major order (C-style: left-to-right, top-to-bottom)
    pixels = mask.flatten(order='C')
    
    # Add padding to handle edge cases
    pixels = np.concatenate([[0], pixels, [0]])
    
    # Find run starts and ends
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    
    # Runs are in pairs: [start, end, start, end, ...]
    # We want starts of 1s and their lengths
    rle_pairs = []
    for i in range(0, len(runs), 2):
        if i + 1 < len(runs):
            start = runs[i]
            end = runs[i + 1]
            length = end - start
            
            # Apply 1-indexing if requested (Kaggle standard)
            if use_1_indexing:
                start += 1
            
            rle_pairs.append(f"{start} {length}")
    
    return " ".join(rle_pairs) if rle_pairs else ""


def rle_to_mask(rle: str, shape: Tuple[int, int], use_1_indexing: bool = True) -> np.ndarray:
    """Convert run-length encoding back to 2D binary mask.
    
    Args:
        rle: RLE string in format "start1 length1 start2 length2 ..."
        shape: Output mask shape (H, W)
        use_1_indexing: If True, interpret positions as 1-indexed (Kaggle standard)
        
    Returns:
        2D binary mask (H, W)
        
    Example:
        >>> rle = "3 2 5 2"
        >>> mask = rle_to_mask(rle, (2, 4))
        >>> print(mask)
        [[0 0 1 1]
         [1 1 0 0]]
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    if not rle or rle.strip() == "":
        return mask.reshape(shape, order='C')
    
    # Parse RLE pairs
    rle_numbers = [int(x) for x in rle.split()]
    
    if len(rle_numbers) % 2 != 0:
        raise ValueError(f"RLE must have even number of values, got {len(rle_numbers)}")
    
    # Fill mask
    for i in range(0, len(rle_numbers), 2):
        start = rle_numbers[i]
        length = rle_numbers[i + 1]
        
        # Convert from 1-indexed to 0-indexed if needed
        if use_1_indexing:
            start -= 1
        
        mask[start:start + length] = 1
    
    return mask.reshape(shape, order='C')


def validate_rle_roundtrip(mask: np.ndarray, use_1_indexing: bool = True) -> bool:
    """Validate that RLE encoding/decoding is lossless.
    
    Args:
        mask: 2D binary mask to test
        use_1_indexing: Whether to use 1-indexed positions
        
    Returns:
        True if roundtrip is successful, False otherwise
    """
    rle = mask_to_rle(mask, use_1_indexing=use_1_indexing)
    reconstructed = rle_to_mask(rle, mask.shape, use_1_indexing=use_1_indexing)
    return np.array_equal(mask, reconstructed)


def create_submission_csv(
    predictions: Dict[str, np.ndarray],
    output_path: Path,
    use_1_indexing: bool = True,
    validate: bool = True
) -> None:
    """Generate Kaggle submission CSV from predictions.
    
    Args:
        predictions: Dict mapping fragment IDs to 2D binary masks
        output_path: Path to save submission.csv
        use_1_indexing: Whether to use 1-indexed RLE (Kaggle standard)
        validate: If True, validate RLE roundtrip for each mask
        
    Format:
        CSV with columns: Id, Predicted
        Each row: fragment_id, rle_string
        
    Example:
        >>> predictions = {
        ...     "fragment_1": np.array([[0, 1], [1, 0]]),
        ...     "fragment_2": np.array([[1, 1], [0, 0]])
        ... }
        >>> create_submission_csv(predictions, Path("submission.csv"))
    """
    rows = []
    
    for fragment_id, mask in predictions.items():
        # Validate mask
        if mask.ndim != 2:
            raise ValueError(f"Mask for {fragment_id} must be 2D, got shape {mask.shape}")
        
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"Mask for {fragment_id} must be binary (0s and 1s)")
        
        # Convert to RLE
        rle = mask_to_rle(mask, use_1_indexing=use_1_indexing)
        
        # Validate roundtrip if requested
        if validate:
            if not validate_rle_roundtrip(mask, use_1_indexing=use_1_indexing):
                raise ValueError(f"RLE roundtrip validation failed for {fragment_id}")
        
        rows.append({"Id": fragment_id, "Predicted": rle})
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Submission CSV saved to {output_path}")
    print(f"  - {len(rows)} fragments")
    print(f"  - Total RLE length: {sum(len(r['Predicted']) for r in rows)} chars")


def load_submission_csv(csv_path: Path, use_1_indexing: bool = True) -> Dict[str, str]:
    """Load submission CSV and return dict of fragment IDs to RLE strings.
    
    Args:
        csv_path: Path to submission.csv
        use_1_indexing: Whether RLE uses 1-indexed positions
        
    Returns:
        Dict mapping fragment IDs to RLE strings
    """
    df = pd.read_csv(csv_path)
    
    if "Id" not in df.columns or "Predicted" not in df.columns:
        raise ValueError("CSV must have 'Id' and 'Predicted' columns")
    
    return dict(zip(df["Id"], df["Predicted"]))


def visualize_rle_comparison(
    mask: np.ndarray,
    rle: str,
    shape: Tuple[int, int],
    use_1_indexing: bool = True
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Visualize original mask vs RLE-decoded mask for debugging.
    
    Args:
        mask: Original 2D binary mask
        rle: RLE string
        shape: Expected shape for decoding
        use_1_indexing: Whether RLE uses 1-indexed positions
        
    Returns:
        Tuple of (original_mask, decoded_mask, matches)
    """
    decoded = rle_to_mask(rle, shape, use_1_indexing=use_1_indexing)
    matches = np.array_equal(mask, decoded)
    
    return mask, decoded, matches


def get_rle_stats(rle: str) -> Dict[str, int]:
    """Get statistics about an RLE string.
    
    Args:
        rle: RLE string
        
    Returns:
        Dict with stats: num_runs, total_pixels, avg_run_length
    """
    # Handle NaN or non-string inputs from pandas
    if not isinstance(rle, str) or not rle or rle.strip() == "":
        return {"num_runs": 0, "total_pixels": 0, "avg_run_length": 0}
    
    rle_numbers = [int(x) for x in rle.split()]
    lengths = [rle_numbers[i + 1] for i in range(0, len(rle_numbers), 2)]
    
    return {
        "num_runs": len(lengths),
        "total_pixels": sum(lengths),
        "avg_run_length": int(np.mean(lengths)) if lengths else 0,
        "max_run_length": max(lengths) if lengths else 0,
        "min_run_length": min(lengths) if lengths else 0,
    }


__all__ = [
    "mask_to_rle",
    "rle_to_mask",
    "validate_rle_roundtrip",
    "create_submission_csv",
    "load_submission_csv",
    "visualize_rle_comparison",
    "get_rle_stats",
]

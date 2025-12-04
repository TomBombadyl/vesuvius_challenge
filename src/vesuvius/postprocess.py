from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size voxels."""
    # Make a copy to avoid modifying input
    mask = mask.copy()
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    
    # Count voxels in each component (including background at index 0)
    counts = np.bincount(labeled.ravel())
    
    # Mark components to remove (skip background at index 0)
    remove = counts < min_size
    remove[0] = False  # Never remove background
    
    # Remove small components
    remove_idx = remove[labeled]
    mask[remove_idx] = 0
    return mask


def morphological_closing(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    structure = ndimage.generate_binary_structure(3, 2)
    structure = ndimage.iterate_structure(structure, radius)
    return ndimage.binary_closing(mask, structure=structure)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask)


def apply_postprocessing(mask: np.ndarray, cfg: Dict) -> np.ndarray:
    output = mask.astype(bool)
    if cfg.get("remove_small_components_voxels"):
        output = remove_small_components(output, cfg["remove_small_components_voxels"])
    if cfg.get("fill_holes"):
        output = fill_holes(output)
    if cfg.get("closing_radius"):
        output = morphological_closing(output, cfg["closing_radius"])
    return output.astype(mask.dtype)


__all__ = ["apply_postprocessing", "remove_small_components", "morphological_closing"]


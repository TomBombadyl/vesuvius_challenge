"""Virtual unwrapping and visualization utilities for Vesuvius Challenge.

This module provides functions to extract papyrus surfaces from 3D volumes,
flatten them to 2D, and visualize detected ink on unwrapped surfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    plt = None

try:
    from PIL import Image
except ImportError:
    Image = None


def extract_surface_from_volume(
    volume: np.ndarray,
    surface_mask: np.ndarray,
    method: str = "center"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract voxels along segmented surface and flatten to 2D.
    
    Args:
        volume: 3D CT volume (D, H, W) with intensity values
        surface_mask: 3D binary mask (D, H, W) marking papyrus surface
        method: Extraction method - "center", "mean", or "max"
            - "center": Take center depth of each surface column
            - "mean": Average all surface voxels in each column
            - "max": Take maximum intensity in each column
            
    Returns:
        Tuple of (surface_image, surface_coords)
        - surface_image: 2D array (H, W) with extracted intensities
        - surface_coords: 2D array (H, W) with depth coordinates of surface
        
    Example:
        >>> volume = np.random.rand(100, 200, 200)
        >>> mask = np.zeros_like(volume)
        >>> mask[45:55, :, :] = 1  # Flat surface at depth 45-55
        >>> surface, coords = extract_surface_from_volume(volume, mask)
        >>> print(surface.shape)  # (200, 200)
    """
    if volume.shape != surface_mask.shape:
        raise ValueError(f"Volume and mask shapes must match: {volume.shape} vs {surface_mask.shape}")
    
    D, H, W = volume.shape
    surface_image = np.zeros((H, W), dtype=volume.dtype)
    surface_coords = np.zeros((H, W), dtype=np.float32)
    
    # Process each (h, w) column
    for h in range(H):
        for w in range(W):
            column_mask = surface_mask[:, h, w]
            surface_indices = np.where(column_mask > 0)[0]
            
            if len(surface_indices) == 0:
                # No surface in this column - use NaN or 0
                surface_image[h, w] = 0
                surface_coords[h, w] = -1
                continue
            
            if method == "center":
                # Take center depth
                center_idx = surface_indices[len(surface_indices) // 2]
                surface_image[h, w] = volume[center_idx, h, w]
                surface_coords[h, w] = center_idx
                
            elif method == "mean":
                # Average all surface voxels
                surface_image[h, w] = np.mean(volume[surface_indices, h, w])
                surface_coords[h, w] = np.mean(surface_indices)
                
            elif method == "max":
                # Take maximum intensity
                max_idx = surface_indices[np.argmax(volume[surface_indices, h, w])]
                surface_image[h, w] = volume[max_idx, h, w]
                surface_coords[h, w] = max_idx
                
            else:
                raise ValueError(f"Unknown extraction method: {method}")
    
    return surface_image, surface_coords


def extract_surface_neighborhood(
    volume: np.ndarray,
    surface_coords: np.ndarray,
    depth_radius: int = 5
) -> np.ndarray:
    """Extract a neighborhood around the surface for ink detection.
    
    Args:
        volume: 3D CT volume (D, H, W)
        surface_coords: 2D array (H, W) with depth coordinate of surface at each (h,w)
        depth_radius: Number of slices above and below surface to extract
        
    Returns:
        4D array (H, W, 2*depth_radius+1) with surface neighborhood
        
    This is useful for ink detection models that need context around the surface.
    """
    D, H, W = volume.shape
    neighborhood = np.zeros((H, W, 2 * depth_radius + 1), dtype=volume.dtype)
    
    for h in range(H):
        for w in range(W):
            center_d = int(surface_coords[h, w])
            
            if center_d < 0:  # No surface at this location
                continue
            
            for offset, d_idx in enumerate(range(center_d - depth_radius, center_d + depth_radius + 1)):
                if 0 <= d_idx < D:
                    neighborhood[h, w, offset] = volume[d_idx, h, w]
    
    return neighborhood


def project_ink_to_surface(
    ink_mask: np.ndarray,
    surface_coords: np.ndarray,
    volume_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Map 2D detected ink back onto 3D surface coordinates.
    
    Args:
        ink_mask: 2D binary mask (H, W) showing ink locations
        surface_coords: 2D array (H, W) with depth coordinates
        volume_shape: Shape of original 3D volume (D, H, W)
        
    Returns:
        3D binary mask (D, H, W) with ink projected onto surface
    """
    D, H, W = volume_shape
    
    if ink_mask.shape != (H, W):
        raise ValueError(f"Ink mask shape {ink_mask.shape} doesn't match surface coords {(H, W)}")
    
    projected = np.zeros(volume_shape, dtype=np.uint8)
    
    for h in range(H):
        for w in range(W):
            if ink_mask[h, w] > 0:
                d = int(surface_coords[h, w])
                if 0 <= d < D:
                    projected[d, h, w] = 1
    
    return projected


def visualize_unwrapped_text(
    surface_image: np.ndarray,
    ink_mask: np.ndarray,
    output_dir: Path,
    fragment_id: str = "fragment",
    create_overlay: bool = True,
    create_text_only: bool = True
) -> Dict[str, Path]:
    """Create visualizations of unwrapped surface and detected ink.
    
    Args:
        surface_image: 2D array (H, W) with papyrus texture/intensity
        ink_mask: 2D binary mask (H, W) showing detected ink
        output_dir: Directory to save visualizations
        fragment_id: Identifier for this fragment (used in filenames)
        create_overlay: If True, create overlay of ink on papyrus
        create_text_only: If True, create black-on-white text image
        
    Returns:
        Dict mapping visualization type to saved file path
        
    Creates:
        - {fragment_id}_surface.png - Unwrapped papyrus texture
        - {fragment_id}_ink.png - Binary ink mask
        - {fragment_id}_overlay.png - Ink highlighted on papyrus (optional)
        - {fragment_id}_text.png - Black ink on white background (optional)
    """
    if plt is None:
        raise ImportError("matplotlib required for visualization")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # 1. Save unwrapped surface
    surface_path = output_dir / f"{fragment_id}_surface.png"
    plt.figure(figsize=(12, 12))
    plt.imshow(surface_image, cmap='gray')
    plt.title(f"Unwrapped Surface - {fragment_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(surface_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files['surface'] = surface_path
    
    # 2. Save binary ink mask
    ink_path = output_dir / f"{fragment_id}_ink.png"
    plt.figure(figsize=(12, 12))
    plt.imshow(ink_mask, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Detected Ink - {fragment_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(ink_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files['ink'] = ink_path
    
    # 3. Create overlay (ink highlighted on papyrus)
    if create_overlay:
        overlay_path = output_dir / f"{fragment_id}_overlay.png"
        plt.figure(figsize=(12, 12))
        
        # Normalize surface image for display
        surface_norm = (surface_image - surface_image.min()) / (surface_image.max() - surface_image.min() + 1e-8)
        
        # Create RGB image
        rgb = np.stack([surface_norm, surface_norm, surface_norm], axis=-1)
        
        # Highlight ink in red (semi-transparent)
        ink_alpha = 0.6
        rgb[ink_mask > 0, 0] = ink_alpha * 1.0 + (1 - ink_alpha) * rgb[ink_mask > 0, 0]  # Red channel
        rgb[ink_mask > 0, 1] = ink_alpha * 0.0 + (1 - ink_alpha) * rgb[ink_mask > 0, 1]  # Green channel
        rgb[ink_mask > 0, 2] = ink_alpha * 0.0 + (1 - ink_alpha) * rgb[ink_mask > 0, 2]  # Blue channel
        
        plt.imshow(rgb)
        plt.title(f"Ink Overlay - {fragment_id}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files['overlay'] = overlay_path
    
    # 4. Create text-only image (black ink on white background)
    if create_text_only:
        text_path = output_dir / f"{fragment_id}_text.png"
        plt.figure(figsize=(12, 12))
        
        # Invert: white background, black ink
        text_image = 1 - ink_mask.astype(np.float32)
        
        plt.imshow(text_image, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Text Only - {fragment_id}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(text_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files['text'] = text_path
    
    print(f"✓ Visualizations saved to {output_dir}")
    for viz_type, path in saved_files.items():
        print(f"  - {viz_type}: {path.name}")
    
    return saved_files


def create_comparison_visualization(
    surface_image: np.ndarray,
    predicted_ink: np.ndarray,
    ground_truth_ink: Optional[np.ndarray],
    output_path: Path,
    fragment_id: str = "fragment"
) -> None:
    """Create side-by-side comparison of prediction vs ground truth.
    
    Args:
        surface_image: 2D unwrapped surface
        predicted_ink: 2D predicted ink mask
        ground_truth_ink: 2D ground truth ink mask (optional)
        output_path: Path to save comparison image
        fragment_id: Fragment identifier for title
    """
    if plt is None:
        raise ImportError("matplotlib required for visualization")
    
    if ground_truth_ink is None:
        # Just show prediction
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(surface_image, cmap='gray')
        axes[0].set_title("Unwrapped Surface")
        axes[0].axis('off')
        
        axes[1].imshow(predicted_ink, cmap='gray')
        axes[1].set_title("Predicted Ink")
        axes[1].axis('off')
    else:
        # Show surface, prediction, ground truth, and difference
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        axes[0, 0].imshow(surface_image, cmap='gray')
        axes[0, 0].set_title("Unwrapped Surface")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(predicted_ink, cmap='gray')
        axes[0, 1].set_title("Predicted Ink")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(ground_truth_ink, cmap='gray')
        axes[1, 0].set_title("Ground Truth Ink")
        axes[1, 0].axis('off')
        
        # Difference: TP (white), FP (red), FN (blue)
        diff = np.zeros((*predicted_ink.shape, 3))
        diff[..., 0] = predicted_ink  # Red channel = predictions
        diff[..., 1] = ground_truth_ink  # Green channel = ground truth
        diff[..., 2] = ground_truth_ink  # Blue channel = ground truth
        
        axes[1, 1].imshow(diff)
        axes[1, 1].set_title("Difference (White=TP, Red=FP, Cyan=FN)")
        axes[1, 1].axis('off')
    
    plt.suptitle(f"Ink Detection Results - {fragment_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison saved to {output_path}")


def compute_surface_statistics(surface_coords: np.ndarray) -> Dict[str, float]:
    """Compute statistics about the extracted surface.
    
    Args:
        surface_coords: 2D array (H, W) with depth coordinates
        
    Returns:
        Dict with statistics: mean_depth, std_depth, min_depth, max_depth, coverage
    """
    valid_coords = surface_coords[surface_coords >= 0]
    
    if len(valid_coords) == 0:
        return {
            "mean_depth": 0.0,
            "std_depth": 0.0,
            "min_depth": 0.0,
            "max_depth": 0.0,
            "coverage": 0.0,
            "num_valid_pixels": 0
        }
    
    total_pixels = surface_coords.size
    
    return {
        "mean_depth": float(np.mean(valid_coords)),
        "std_depth": float(np.std(valid_coords)),
        "min_depth": float(np.min(valid_coords)),
        "max_depth": float(np.max(valid_coords)),
        "coverage": float(len(valid_coords) / total_pixels),
        "num_valid_pixels": int(len(valid_coords))
    }


def smooth_surface_coords(
    surface_coords: np.ndarray,
    sigma: float = 2.0
) -> np.ndarray:
    """Smooth surface coordinates to reduce noise.
    
    Args:
        surface_coords: 2D array (H, W) with depth coordinates
        sigma: Gaussian smoothing sigma
        
    Returns:
        Smoothed surface coordinates
    """
    # Only smooth valid coordinates
    valid_mask = surface_coords >= 0
    smoothed = surface_coords.copy()
    
    if np.any(valid_mask):
        # Fill invalid regions with interpolation for smoothing
        filled = surface_coords.copy()
        filled[~valid_mask] = np.mean(surface_coords[valid_mask])
        
        # Apply Gaussian smoothing
        smoothed_filled = ndimage.gaussian_filter(filled, sigma=sigma)
        
        # Restore original values where valid, use smoothed elsewhere
        smoothed[valid_mask] = smoothed_filled[valid_mask]
    
    return smoothed


__all__ = [
    "extract_surface_from_volume",
    "extract_surface_neighborhood",
    "project_ink_to_surface",
    "visualize_unwrapped_text",
    "create_comparison_visualization",
    "compute_surface_statistics",
    "smooth_surface_coords",
]

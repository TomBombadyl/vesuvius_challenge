from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    surface_a = np.bitwise_xor(mask_a, ndimage.binary_erosion(mask_a, structure=structure))
    surface_b = np.bitwise_xor(mask_b, ndimage.binary_erosion(mask_b, structure=structure))

    dt = ndimage.distance_transform_edt(~surface_b, sampling=spacing)
    distances = dt[surface_a]
    return distances


def surface_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_mm: float,
                 spacing: Tuple[float, float, float]) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    if not pred_mask.any() and not gt_mask.any():
        return 1.0
    if not pred_mask.any() or not gt_mask.any():
        return 0.0
    dist_ab = _surface_distances(pred_mask, gt_mask, spacing)
    dist_ba = _surface_distances(gt_mask, pred_mask, spacing)
    tp_ab = (dist_ab <= tolerance_mm).sum()
    tp_ba = (dist_ba <= tolerance_mm).sum()
    denom = len(dist_ab) + len(dist_ba) + 1e-6
    return (tp_ab + tp_ba) / denom


def variation_of_information(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(int).ravel()
    gt = gt_mask.astype(int).ravel()
    joint_hist = np.histogram2d(pred, gt, bins=2)[0]
    joint_prob = joint_hist / np.sum(joint_hist)
    pred_prob = np.sum(joint_prob, axis=1)
    gt_prob = np.sum(joint_prob, axis=0)
    eps = 1e-9
    h_pred = -np.sum(pred_prob * np.log(pred_prob + eps))
    h_gt = -np.sum(gt_prob * np.log(gt_prob + eps))
    mi = np.sum(joint_prob * np.log(joint_prob / (pred_prob[:, None] * gt_prob[None, :] + eps) + eps))
    return (h_pred + h_gt - 2 * mi)


def topo_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    labeled_pred, num_pred = ndimage.label(pred_mask)
    labeled_gt, num_gt = ndimage.label(gt_mask)
    hole_pred = np.logical_not(pred_mask)
    hole_gt = np.logical_not(gt_mask)
    labeled_hole_pred, hole_count_pred = ndimage.label(hole_pred)
    labeled_hole_gt, hole_count_gt = ndimage.label(hole_gt)
    betti0_diff = abs(num_pred - num_gt)
    hole_diff = abs(hole_count_pred - hole_count_gt)
    return 1.0 / (1.0 + betti0_diff + 0.5 * hole_diff)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection over Union (Jaccard index)."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice similarity coefficient (F1 score)."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.sum(pred_mask & gt_mask)
    card_pred = np.sum(pred_mask)
    card_gt = np.sum(gt_mask)
    denom = card_pred + card_gt
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / denom)


def evaluate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_mm: float,
                     spacing: Tuple[float, float, float]) -> Dict[str, float]:
    return {
        "surface_dice": surface_dice(pred_mask, gt_mask, tolerance_mm, spacing),
        "voi": variation_of_information(pred_mask, gt_mask),
        "topo_score": topo_score(pred_mask, gt_mask),
    }


__all__ = ["evaluate_metrics", "surface_dice", "variation_of_information", "topo_score", "compute_iou", "compute_dice"]


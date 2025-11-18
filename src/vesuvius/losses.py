from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (1, 2, 3, 4)
    intersection = torch.sum(probs * targets, dims)
    denominator = torch.sum(probs + targets, dims)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=logits.device))
    return loss_fn(logits, targets)


def soft_erode(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    return -F.max_pool3d(-img, kernel_size, stride=1, padding=padding)


def soft_dilate(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    return F.max_pool3d(img, kernel_size, stride=1, padding=padding)


def soft_open(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return soft_dilate(soft_erode(img, kernel_size), kernel_size)


def soft_skeletonize(img: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    skeleton = torch.zeros_like(img)
    for _ in range(iterations):
        eroded = soft_erode(img)
        temp = F.relu(img - soft_open(img))
        skeleton = skeleton + F.relu(temp - skeleton)
        img = eroded
    return torch.clamp(skeleton, 0.0, 1.0)


def cldice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0, iterations: int = 3) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = torch.clamp(targets, 0.0, 1.0)
    pred_skeleton = soft_skeletonize(probs, iterations)
    target_skeleton = soft_skeletonize(targets, iterations)
    tprec = (torch.sum(pred_skeleton * targets) + smooth) / (torch.sum(pred_skeleton) + smooth)
    tsens = (torch.sum(target_skeleton * probs) + smooth) / (torch.sum(target_skeleton) + smooth)
    cldice = 2 * tprec * tsens / (tprec + tsens + 1e-6)
    return 1 - cldice


def morph_skeleton_loss(logits: torch.Tensor, targets: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_skeleton = soft_skeletonize(probs, iterations)
    target_skeleton = soft_skeletonize(targets, iterations)
    return F.l1_loss(pred_skeleton, target_skeleton)


def surface_distance_loss(logits: torch.Tensor, targets: torch.Tensor, tolerance_mm: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    batch_losses = []
    for b in range(probs.shape[0]):
        pred = probs[b, 0]
        target = targets[b, 0]
        target_np = target.detach().cpu().numpy() > 0.5
        pred_np = pred.detach().cpu().numpy() > 0.5
        dt_target = ndimage.distance_transform_edt(~target_np)
        dt_pred = ndimage.distance_transform_edt(~pred_np)
        dt_target = torch.from_numpy(dt_target).to(pred.device).float()
        dt_pred = torch.from_numpy(dt_pred).to(pred.device).float()
        loss_tp = torch.mean(torch.clamp(dt_target / tolerance_mm, max=1.0) * (1 - pred))
        loss_fp = torch.mean(torch.clamp(dt_pred / tolerance_mm, max=1.0) * target)
        batch_losses.append(loss_tp + loss_fp)
    return torch.stack(batch_losses).mean()


def soft_topology_loss(logits: torch.Tensor, targets: torch.Tensor, betti_threshold: float = 0.2) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    soft_components = soft_erode(probs, kernel_size=5)
    soft_holes = 1 - soft_dilate(1 - probs, kernel_size=5)
    target_components = soft_erode(targets, kernel_size=5)
    target_holes = 1 - soft_dilate(1 - targets, kernel_size=5)

    betti_loss = F.l1_loss(soft_components, target_components)
    hole_activation = torch.relu(betti_threshold - soft_holes).mean()
    target_activation = torch.relu(betti_threshold - target_holes).mean()
    hole_loss = torch.abs(hole_activation - target_activation)
    return betti_loss + hole_loss


@dataclass
class LossComponent:
    name: str
    weight: float
    params: Dict


class CompositeTopologyLoss(nn.Module):
    def __init__(self, components_cfg: List[Dict]):
        super().__init__()
        self.components: List[LossComponent] = []
        for comp in components_cfg:
            name = comp["name"]
            weight = comp.get("weight", 1.0)
            params = {k: v for k, v in comp.items() if k not in {"name", "weight"}}
            self.components.append(LossComponent(name=name, weight=weight, params=params))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = 0.0
        details: Dict[str, float] = {}
        for component in self.components:
            value = self._compute(component, logits, targets)
            total = total + component.weight * value
            details[component.name] = float(value.detach().cpu())
        return total, details

    def _compute(self, component: LossComponent, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        name = component.name
        params = component.params
        if name == "weighted_bce":
            return weighted_bce_loss(logits, targets, pos_weight=params.get("pos_weight", 1.0))
        if name == "soft_dice":
            return soft_dice_loss(logits, targets, smooth=params.get("smooth", 1.0))
        if name == "cldice":
            return cldice_loss(logits, targets, smooth=params.get("smooth", 1.0), iterations=params.get("iterations", 3))
        if name == "morph_skeleton":
            return morph_skeleton_loss(logits, targets, iterations=params.get("iterations", 3))
        if name == "surface_distance":
            return surface_distance_loss(logits, targets, tolerance_mm=params.get("tolerance_mm", 2.0))
        if name in {"toploss_simple", "topology"}:
            return soft_topology_loss(logits, targets, betti_threshold=params.get("betti_threshold", 0.2))
        raise ValueError(f"Unknown loss component {name}")


__all__ = ["CompositeTopologyLoss"]


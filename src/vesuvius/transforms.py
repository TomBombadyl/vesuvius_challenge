from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage


def _maybe_apply(prob: float) -> bool:
    return random.random() < prob


def _random_uniform(low: float, high: float) -> float:
    return random.uniform(low, high)


def _ensure_numpy(arr: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        np_arr = arr
    else:
        np_arr = arr.detach().cpu().numpy()
    if np_arr.ndim == 4 and np_arr.shape[0] == 1:
        np_arr = np_arr[0]
    return np_arr


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 3:
        arr = arr[None, ...]  # channel-first
    return torch.from_numpy(arr.astype(np.float32))


def random_flip(image: np.ndarray, mask: np.ndarray, axes: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    for ax in axes:
        if random.random() < 0.5:
            image = np.flip(image, axis=ax)
            mask = np.flip(mask, axis=ax)
    return image, mask


def random_rotate(image: np.ndarray, mask: np.ndarray, degrees: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    angle = random.uniform(degrees[0], degrees[1])
    axes_choices = [(0, 1), (0, 2), (1, 2)]
    axes = random.choice(axes_choices)
    image = ndimage.rotate(image, angle=angle, axes=axes, reshape=False, order=1, mode="nearest")
    mask = ndimage.rotate(mask, angle=angle, axes=axes, reshape=False, order=0, mode="nearest")
    return image, mask


def random_scale(image: np.ndarray, mask: np.ndarray, scale_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    original_shape = image.shape
    scale = random.uniform(scale_range[0], scale_range[1])
    zoom_factors = (scale, scale, scale)
    image = ndimage.zoom(image, zoom=zoom_factors, order=1)
    mask = ndimage.zoom(mask, zoom=zoom_factors, order=0)
    return _center_crop_or_pad(image, mask, original_shape)


def anisotropy_scale(image: np.ndarray, mask: np.ndarray, factors: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    original_shape = image.shape
    scales = tuple(random.uniform(factors[0], factors[1]) for _ in range(3))
    image = ndimage.zoom(image, zoom=scales, order=1)
    mask = ndimage.zoom(mask, zoom=scales, order=0)
    return _center_crop_or_pad(image, mask, original_shape)


def slice_jitter(image: np.ndarray, mask: np.ndarray, max_voxels: int) -> Tuple[np.ndarray, np.ndarray]:
    shift = random.randint(-max_voxels, max_voxels)
    image = np.roll(image, shift=shift, axis=0)
    mask = np.roll(mask, shift=shift, axis=0)
    return image, mask


def patch_dropout(image: np.ndarray, mask: np.ndarray, probability: float, min_keep_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > probability:
        return image, mask
    depth, height, width = image.shape
    keep_ratio = random.uniform(min_keep_ratio, 1.0)
    drop_depth = int(depth * (1 - keep_ratio))
    drop_height = int(height * (1 - keep_ratio))
    drop_width = int(width * (1 - keep_ratio))
    z = random.randint(0, max(0, depth - drop_depth))
    y = random.randint(0, max(0, height - drop_height))
    x = random.randint(0, max(0, width - drop_width))
    image[z:z + drop_depth, y:y + drop_height, x:x + drop_width] = 0
    mask[z:z + drop_depth, y:y + drop_height, x:x + drop_width] = 0
    return image, mask


def elastic_deform(image: np.ndarray, mask: np.ndarray, sigma: float, magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
    if sigma <= 0 or magnitude <= 0:
        return image, mask
    random_state = np.random.RandomState(None)
    shape = image.shape
    dz = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * magnitude
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * magnitude
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * magnitude
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    indices = (z_coords + dz, y_coords + dy, x_coords + dx)
    image = ndimage.map_coordinates(image, indices, order=1, mode="reflect")
    mask = ndimage.map_coordinates(mask, indices, order=0, mode="reflect")
    return image, mask


def gamma_transform(image: np.ndarray, gamma_range: Tuple[float, float]) -> np.ndarray:
    gamma = random.uniform(*gamma_range)
    image = np.clip(image, 0, None)
    max_val = np.max(image) if np.max(image) > 0 else 1.0
    image = image / max_val
    return np.power(image, gamma) * max_val


def gamma_noise(image: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    noise = np.random.gamma(shape=strength, scale=1.0, size=image.shape)
    return image * noise


def gaussian_noise(image: np.ndarray, std_range: Tuple[float, float]) -> np.ndarray:
    std = random.uniform(*std_range)
    if std == 0:
        return image
    noise = np.random.normal(0.0, std, size=image.shape)
    return image + noise


def gaussian_blur(image: np.ndarray, sigma_range: Tuple[float, float]) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    if sigma == 0:
        return image
    return ndimage.gaussian_filter(image, sigma=sigma)


def cutout(image: np.ndarray, mask: np.ndarray, holes: int, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    depth, height, width = image.shape
    min_size, max_size = size
    for _ in range(holes):
        hole_size = random.randint(min_size, max_size)
        z = random.randint(0, max(0, depth - hole_size))
        y = random.randint(0, max(0, height - hole_size))
        x = random.randint(0, max(0, width - hole_size))
        image[z:z + hole_size, y:y + hole_size, x:x + hole_size] = 0
        mask[z:z + hole_size, y:y + hole_size, x:x + hole_size] = 0
    return image, mask


def _center_crop_or_pad(image: np.ndarray, mask: np.ndarray, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Crop/pad to target shape after zoom."""
    image = _crop_or_pad_to_shape(image, target_shape)
    mask = _crop_or_pad_to_shape(mask, target_shape)
    return image, mask


def _crop_or_pad_to_shape(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    result = volume
    for axis in range(3):
        diff = result.shape[axis] - target_shape[axis]
        if diff > 0:  # crop
            start = diff // 2
            end = start + target_shape[axis]
            result = np.take(result, indices=range(start, end), axis=axis)
        elif diff < 0:  # pad
            pad_width = [(0, 0)] * 3
            pad = (-diff) // 2
            remainder = (-diff) - pad
            pad_width[axis] = (pad, pad + remainder)
            result = np.pad(result, pad_width, mode="constant")
    return result


@dataclass
class AugmentationConfig:
    spatial: Optional[Dict[str, Any]] = None
    intensity: Optional[Dict[str, Any]] = None


class AugmentationPipeline:
    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        np_image = _ensure_numpy(image)
        np_mask = _ensure_numpy(mask)

        spatial_cfg = self.config.spatial or {}
        intensity_cfg = self.config.intensity or {}

        if spatial_cfg.get("flip_axes"):
            np_image, np_mask = random_flip(np_image, np_mask, tuple(spatial_cfg["flip_axes"]))
        if spatial_cfg.get("rotate_deg"):
            np_image, np_mask = random_rotate(np_image, np_mask, tuple(spatial_cfg["rotate_deg"]))
        if spatial_cfg.get("scale"):
            np_image, np_mask = random_scale(np_image, np_mask, tuple(spatial_cfg["scale"]))
        if spatial_cfg.get("elastic", {}).get("enabled", False):
            elastic_cfg = spatial_cfg["elastic"]
            np_image, np_mask = elastic_deform(
                np_image, np_mask, elastic_cfg.get("sigma", 6.0), elastic_cfg.get("magnitude", 0.08)
            )
        if spatial_cfg.get("anisotropy_scale"):
            np_image, np_mask = anisotropy_scale(np_image, np_mask, tuple(spatial_cfg["anisotropy_scale"]))
        if spatial_cfg.get("slice_jitter_voxels"):
            np_image, np_mask = slice_jitter(np_image, np_mask, int(spatial_cfg["slice_jitter_voxels"]))
        if spatial_cfg.get("patch_dropout"):
            pd_cfg = spatial_cfg["patch_dropout"]
            np_image, np_mask = patch_dropout(
                np_image,
                np_mask,
                pd_cfg.get("probability", 0.1),
                pd_cfg.get("min_keep_ratio", 0.7),
            )
        if spatial_cfg.get("cutout"):
            cut_cfg = spatial_cfg["cutout"]
            np_image, np_mask = cutout(
                np_image,
                np_mask,
                cut_cfg.get("holes", 1),
                tuple(cut_cfg.get("size", [16, 32])),
            )

        if intensity_cfg.get("gamma"):
            np_image = gamma_transform(np_image, tuple(intensity_cfg["gamma"]))
        if intensity_cfg.get("gamma_noise"):
            np_image = gamma_noise(np_image, intensity_cfg["gamma_noise"])
        if intensity_cfg.get("gaussian_noise_std"):
            np_image = gaussian_noise(np_image, tuple(intensity_cfg["gaussian_noise_std"]))
        if intensity_cfg.get("gaussian_blur_sigma"):
            np_image = gaussian_blur(np_image, tuple(intensity_cfg["gaussian_blur_sigma"]))

        return _to_tensor(np_image), _to_tensor(np_mask)


def build_augmentation_pipeline(augmentation_cfg: Optional[Dict[str, Any]]) -> Optional[AugmentationPipeline]:
    if not augmentation_cfg:
        return None
    spatial = augmentation_cfg.get("spatial")
    intensity = augmentation_cfg.get("intensity")
    return AugmentationPipeline(AugmentationConfig(spatial=spatial, intensity=intensity))


__all__ = [
    "AugmentationPipeline",
    "AugmentationConfig",
    "build_augmentation_pipeline",
]


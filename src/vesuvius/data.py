from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .patch_sampler import ForegroundPatchSampler
from .transforms import build_augmentation_pipeline


@dataclass
class VolumeRecord:
    volume_id: str
    image_path: Path
    label_path: Optional[Path]
    spacing: Optional[Tuple[float, float, float]]
    fold: Optional[int]


def read_metadata(train_csv: Path, data_root: Path) -> List[VolumeRecord]:
    df = pd.read_csv(train_csv)
    records: List[VolumeRecord] = []
    for _, row in df.iterrows():
        volume_id = str(row.get("volume_id", row.get("id", row.get("fragment_id"))))
        img_path = row.get("image_path")
        if img_path:
            image_path = Path(img_path)
        else:
            image_path = data_root / "train_images" / f"{volume_id}.tif"
        lbl_path = row.get("label_path")
        if lbl_path:
            label_path = Path(lbl_path)
        else:
            default_label = data_root / "train_labels" / f"{volume_id}.tif"
            label_path = default_label if default_label.exists() else None
        spacing_vals = []
        spacing = None
        for key in ("spacing_z", "spacing_y", "spacing_x"):
            val = row.get(key) if key in row else None
            if val is None or pd.isna(val):
                spacing = None
                break
            spacing_vals.append(float(val))
        if len(spacing_vals) == 3:
            spacing = tuple(spacing_vals)  # type: ignore[assignment]
        fold = int(row["fold"]) if "fold" in row else None
        records.append(
            VolumeRecord(
                volume_id=volume_id,
                image_path=image_path,
                label_path=label_path,
                spacing=spacing,
                fold=fold,
            )
        )
    return records


def clip_intensity(volume: np.ndarray, clip_values: Tuple[float, float]) -> np.ndarray:
    return np.clip(volume, clip_values[0], clip_values[1])


def normalize_volume(volume: np.ndarray, mode: str = "zscore") -> np.ndarray:
    if mode == "zscore":
        mean = volume.mean()
        std = volume.std() + 1e-6
        return (volume - mean) / std
    if mode == "minmax":
        vmin, vmax = np.min(volume), np.max(volume)
        if vmax - vmin < 1e-6:
            return volume
        return (volume - vmin) / (vmax - vmin)
    if mode == "percentile":
        low, high = np.percentile(volume, (1, 99))
        volume = np.clip(volume, low, high)
        return (volume - low) / (high - low + 1e-6)
    return volume


def denoise_volume(volume: np.ndarray, method: str = "median3d", kernel_size: int = 3) -> np.ndarray:
    if method == "median3d":
        from scipy.ndimage import median_filter

        return median_filter(volume, size=kernel_size)
    return volume


def mask_unlabeled(label: np.ndarray, unlabeled_value: int) -> np.ndarray:
    label = np.copy(label)
    label[label == unlabeled_value] = 0
    return label


def crop_foreground(volume: np.ndarray, label: np.ndarray, margin: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    foreground = np.argwhere(label > 0)
    if foreground.size == 0:
        return volume, label
    z_min, y_min, x_min = foreground.min(axis=0)
    z_max, y_max, x_max = foreground.max(axis=0)
    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    z_max = min(z_max + margin, volume.shape[0] - 1)
    y_max = min(y_max + margin, volume.shape[1] - 1)
    x_max = min(x_max + margin, volume.shape[2] - 1)
    volume = volume[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
    label = label[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
    return volume, label


def resample_volume(volume: np.ndarray, current_spacing: Tuple[float, float, float],
                    target_spacing: Tuple[float, float, float], mode: str = "trilinear") -> np.ndarray:
    scale = tuple(cs / ts for cs, ts in zip(current_spacing, target_spacing))
    new_shape = tuple(int(round(volume.shape[idx] * scale[idx])) for idx in range(3))
    tensor = torch.from_numpy(volume).float()[None, None, ...]
    tensor = F.interpolate(tensor, size=new_shape, mode="trilinear" if mode == "trilinear" else "nearest",
                           align_corners=False)
    return tensor[0, 0].numpy()


class VolumeCache:
    def __init__(self, max_size: int = 50) -> None:
        """
        Cache for volume data with size limit to prevent memory accumulation.
        
        Args:
            max_size: Maximum number of volumes to cache. When exceeded, oldest entries are evicted.
        """
        self.cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self.access_order: List[str] = []  # Track access order for LRU eviction
        self.max_size = max_size

    def get(self, record: VolumeRecord, preprocess_cfg: Dict, force_reload: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        key = record.volume_id
        if not force_reload and key in self.cache:
            # Update access order (move to end)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]

        image = tifffile.imread(record.image_path.as_posix()).astype(np.float32)
        label = None
        if record.label_path and record.label_path.exists():
            label = tifffile.imread(record.label_path.as_posix()).astype(np.float32)

        data_cfg = preprocess_cfg
        if data_cfg.get("intensity_clip"):
            image = clip_intensity(image, tuple(data_cfg["intensity_clip"]))
        if data_cfg.get("denoise", {}).get("enabled", False):
            denoise_cfg = data_cfg["denoise"]
            image = denoise_volume(image, denoise_cfg.get("method", "median3d"), denoise_cfg.get("kernel_size", 3))
        if label is not None and data_cfg.get("mask_unlabeled_value") is not None:
            label = mask_unlabeled(label, data_cfg["mask_unlabeled_value"])
        if label is not None and data_cfg.get("crop_foreground_margin"):
            image, label = crop_foreground(image, label, data_cfg["crop_foreground_margin"])
        if record.spacing and data_cfg.get("resample_spacing"):
            target_spacing = tuple(data_cfg["resample_spacing"])
            image = resample_volume(image, record.spacing, target_spacing)
            if label is not None:
                label = resample_volume(label, record.spacing, target_spacing, mode="nearest")
        if data_cfg.get("normalize"):
            image = normalize_volume(image, data_cfg["normalize"])

        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
        
        self.cache[key] = (image, label)
        if key not in self.access_order:
            self.access_order.append(key)
        return image, label


class PatchDataset(Dataset):
    def __init__(
        self,
        records: List[VolumeRecord],
        preprocess_cfg: Dict,
        patch_size: Sequence[int],
        sampler_cfg: Dict,
        augmentation_cfg: Optional[Dict],
        iterations_per_epoch: int,
        mode: str = "train",
    ) -> None:
        self.records = records
        self.preprocess_cfg = preprocess_cfg
        self.patch_size = tuple(patch_size)
        self.mode = mode
        self.iterations = iterations_per_epoch
        self.cache = VolumeCache()
        self.sampler = ForegroundPatchSampler(self.patch_size, sampler_cfg)
        self.augmentation = build_augmentation_pipeline(augmentation_cfg)

    def __len__(self) -> int:
        return self.iterations

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self._select_record(idx)
        image, label = self.cache.get(record, self.preprocess_cfg)
        if label is None:
            raise ValueError("Training patch dataset requires labels.")
        patch, mask, stats = self.sampler.sample(image, label)
        patch_tensor = torch.from_numpy(patch[None, ...].astype(np.float32))
        mask_tensor = torch.from_numpy(mask[None, ...].astype(np.float32))
        if self.augmentation:
            patch_tensor, mask_tensor = self.augmentation(patch_tensor, mask_tensor)
        sample = {
            "image": patch_tensor,
            "mask": mask_tensor,
            "volume_id": record.volume_id,
            "patch_stats": stats,
        }
        return sample

    def _select_record(self, idx: int) -> VolumeRecord:
        if self.mode == "train":
            return random.choice(self.records)
        return self.records[idx % len(self.records)]


class FullVolumeDataset(Dataset):
    def __init__(
        self,
        records: List[VolumeRecord],
        preprocess_cfg: Dict,
        augmentation_cfg: Optional[Dict] = None,
    ) -> None:
        self.records = records
        self.preprocess_cfg = preprocess_cfg
        self.cache = VolumeCache()
        self.augmentation = build_augmentation_pipeline(augmentation_cfg) if augmentation_cfg else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image, label = self.cache.get(record, self.preprocess_cfg)
        image_tensor = torch.from_numpy(image[None, ...].astype(np.float32))
        label_tensor = None
        if label is not None:
            label_tensor = torch.from_numpy(label[None, ...].astype(np.float32))
        if self.augmentation:
            image_tensor, _ = self.augmentation(image_tensor, label_tensor or image_tensor)
        return {
            "image": image_tensor,
            "label": label_tensor,
            "volume_id": record.volume_id,
        }


def split_records_by_fold(records: List[VolumeRecord], train_folds: Iterable[int], val_folds: Iterable[int]) -> Tuple[List[VolumeRecord], List[VolumeRecord]]:
    train, val = [], []
    train_folds = set(int(f) for f in train_folds)
    val_folds = set(int(f) for f in val_folds)
    for record in records:
        if record.fold is None:
            train.append(record)
            continue
        if record.fold in train_folds:
            train.append(record)
        elif record.fold in val_folds:
            val.append(record)
    return train, val


__all__ = [
    "VolumeRecord",
    "read_metadata",
    "PatchDataset",
    "FullVolumeDataset",
    "split_records_by_fold",
]


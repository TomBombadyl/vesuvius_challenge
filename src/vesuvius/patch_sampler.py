from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass
class SamplerConfig:
    foreground_ratio: float = 0.5
    rejection_probability: float = 0.1
    max_retries: int = 6


def _compute_stats(patch: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    fg_ratio = float((mask > 0).mean())
    stats = {
        "fg_ratio": fg_ratio,
        "mean": float(patch.mean()),
        "std": float(patch.std()),
    }
    return stats


class ForegroundPatchSampler:
    def __init__(self, patch_size: Sequence[int], config: Dict) -> None:
        self.patch_size = tuple(int(v) for v in patch_size)
        self.config = SamplerConfig(
            foreground_ratio=config.get("foreground_ratio", 0.5),
            rejection_probability=config.get("rejection_probability", 0.1),
            max_retries=config.get("max_retries", 6),
        )

    def sample(self, volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        depth, height, width = volume.shape
        pz, py, px = self.patch_size
        padded_volume, padded_mask = self._pad_if_needed(volume, mask)
        depth, height, width = padded_volume.shape
        last_patch = None
        for attempt in range(self.config.max_retries):
            z = random.randint(0, max(0, depth - pz))
            y = random.randint(0, max(0, height - py))
            x = random.randint(0, max(0, width - px))
            patch = padded_volume[z:z + pz, y:y + py, x:x + px]
            patch_mask = padded_mask[z:z + pz, y:y + py, x:x + px]
            stats = _compute_stats(patch, patch_mask)
            last_patch = (patch, patch_mask, stats)
            if stats["fg_ratio"] >= self.config.foreground_ratio:
                break
            if random.random() > self.config.rejection_probability:
                break
        assert last_patch is not None
        return last_patch

    def _pad_if_needed(self, volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pad_z = max(0, self.patch_size[0] - volume.shape[0])
        pad_y = max(0, self.patch_size[1] - volume.shape[1])
        pad_x = max(0, self.patch_size[2] - volume.shape[2])
        pads = (
            (pad_z // 2, pad_z - pad_z // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        )
        if pad_z or pad_y or pad_x:
            volume = np.pad(volume, pads, mode="constant")
            mask = np.pad(mask, pads, mode="constant")
        return volume, mask


__all__ = ["ForegroundPatchSampler"]


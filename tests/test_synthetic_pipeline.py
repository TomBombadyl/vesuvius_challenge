import numpy as np
import pytest
import torch

from src.vesuvius.data import PatchDataset, VolumeRecord
from src.vesuvius.losses import CompositeTopologyLoss
from src.vesuvius.models import build_model

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


def create_dummy_volume(tmp_path):
    image = np.zeros((32, 64, 64), dtype=np.float32)
    image[8:24, 16:48, 16:48] = 1.0
    mask = np.zeros_like(image)
    mask[12:20, 24:40, 24:40] = 1.0
    img_path = tmp_path / "img.tif"
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(img_path, image)
    tifffile.imwrite(mask_path, mask)
    record = VolumeRecord(
        volume_id="dummy",
        image_path=img_path,
        label_path=mask_path,
        spacing=(0.05, 0.05, 0.05),
        fold=0,
    )
    return record


def test_forward_pass(tmp_path):
    if tifffile is None:
        pytest.skip("tifffile not available")
    record = create_dummy_volume(tmp_path)
    data_cfg = {
        "patch_size": [16, 32, 32],
        "sampler": {"foreground_ratio": 0.2},
        "intensity_clip": [0, 1],
        "normalize": "zscore",
    }
    dataset = PatchDataset(
        records=[record],
        preprocess_cfg=data_cfg,
        patch_size=data_cfg["patch_size"],
        sampler_cfg=data_cfg["sampler"],
        augmentation_cfg=None,
        iterations_per_epoch=2,
        mode="train",
    )
    batch = dataset[0]
    inputs = batch["image"].unsqueeze(0)
    targets = batch["mask"].unsqueeze(0)
    model = build_model(
        {
            "type": "unet3d_residual",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 8,
            "channel_multipliers": [1, 2],
        }
    )
    logits = model(inputs)["logits"]
    loss_fn = CompositeTopologyLoss(
        [
            {"name": "weighted_bce", "weight": 0.5, "pos_weight": 2.0},
            {"name": "soft_dice", "weight": 0.5},
        ]
    )
    loss, _ = loss_fn(logits, targets)
    assert torch.isfinite(loss)


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FullVolumeDataset, read_metadata, split_records_by_fold
from .models import build_model
from .postprocess import apply_postprocessing
from .utils import configure_logging, load_config, log_gpu_memory, save_config

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Vesuvius Challenge")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Split to run inference on (train/val/test). For test, uses test.csv")
    parser.add_argument("--max-volumes", type=int, default=None,
                        help="Max number of volumes to process (for quick testing)")
    return parser.parse_args()


def gaussian_weights(shape: Tuple[int, int, int], sigma: float) -> np.ndarray:
    zz, yy, xx = [np.linspace(-1, 1, s) for s in shape]
    z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
    dist = np.sqrt(z ** 2 + y ** 2 + x ** 2)
    weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    return weights.astype(np.float32)


def generate_coords(volume_shape, patch_size, overlap):
    strides = [ps - ov for ps, ov in zip(patch_size, overlap)]
    for dim in range(3):
        if strides[dim] <= 0:
            raise ValueError("Overlap must be smaller than patch size")

    z_steps = list(range(0, max(1, volume_shape[0] - patch_size[0] + 1), strides[0]))
    y_steps = list(range(0, max(1, volume_shape[1] - patch_size[1] + 1), strides[1]))
    x_steps = list(range(0, max(1, volume_shape[2] - patch_size[2] + 1), strides[2]))

    z_steps.append(volume_shape[0] - patch_size[0])
    y_steps.append(volume_shape[1] - patch_size[1])
    x_steps.append(volume_shape[2] - patch_size[2])

    for z in sorted(set(z_steps)):
        for y in sorted(set(y_steps)):
            for x in sorted(set(x_steps)):
                yield z, y, x


def apply_tta(volume: torch.Tensor, axes: Tuple[int, ...]) -> torch.Tensor:
    flipped = volume
    for axis in axes:
        flipped = torch.flip(flipped, dims=(axis + 2,))
    return flipped


def inverse_tta(prediction: torch.Tensor, axes: Tuple[int, ...]) -> torch.Tensor:
    return apply_tta(prediction, axes)


def tta_combinations(mode: str) -> List[Tuple[int, ...]]:
    if mode == "none":
        return [tuple()]
    if mode == "flips":
        return [tuple(), (0,), (1,), (2,)]
    if mode == "full_8x":
        return [(),
                (0,),
                (1,),
                (2,),
                (0, 1),
                (0, 2),
                (1, 2),
                (0, 1, 2)]
    raise ValueError(f"Unknown TTA mode {mode}")


def sliding_window_predict(model, volume: torch.Tensor, cfg: Dict, device: torch.device) -> np.ndarray:
    """Sliding-window inference with Gaussian blending.
    
    Uses patches matching training distribution [96, 160, 160] with 50% overlap.
    Memory footprint: ~20-30GB peak on A100 for 256×384×384 volumes.
    """
    model.eval()
    patch_size = tuple(cfg["patch_size"])
    overlap = tuple(cfg["overlap"])
    sigma = cfg.get("gaussian_blend_sigma", 0.125)
    weights = gaussian_weights(patch_size, sigma)
    
    # Extract spatial dimensions (remove batch and channel dimensions)
    volume_np = volume[0, 0].cpu().numpy()  # [D, H, W]
    pred_accumulator = np.zeros_like(volume_np)
    weight_accumulator = np.zeros_like(volume_np)

    with torch.no_grad():
        for z, y, x in generate_coords(volume_np.shape, patch_size, overlap):
            patch = volume[:, :, z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]].to(device)
            logits = model(patch)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            pred_accumulator[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]] += probs * weights
            weight_accumulator[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]] += weights
    
    weight_accumulator = np.maximum(weight_accumulator, 1e-6)
    return pred_accumulator / weight_accumulator


def inference_with_tta(model, volume: torch.Tensor, cfg: Dict, device: torch.device) -> np.ndarray:
    mode = cfg.get("tta", "none")
    combos = tta_combinations(mode)
    preds = []
    for axes in combos:
        flipped_volume = apply_tta(volume, axes) if axes else volume
        pred = sliding_window_predict(model, flipped_volume, cfg, device)
        pred_tensor = torch.from_numpy(pred[None, None, ...])
        unflipped = inverse_tta(pred_tensor, axes) if axes else pred_tensor
        preds.append(unflipped.numpy()[0, 0])
    return np.mean(preds, axis=0)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    inference_cfg = cfg["inference"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "infer.log")
    save_config(cfg, output_dir / "config_resolved.yaml")
    logger.info("Running inference with config %s", args.config)

    model = build_model(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Determine which CSV and split to use
    data_root = Path(cfg["paths"]["data_root"])
    if args.split == "test":
        # Use test.csv for test set
        test_csv = data_root / "test.csv"
        if not test_csv.exists():
            # Fallback: check if test_csv is specified in config
            test_csv = Path(cfg["paths"].get("test_csv", test_csv))
        if test_csv.exists():
            logger.info("Using test.csv for test set inference")
            metadata = read_metadata(test_csv, data_root)
        else:
            logger.warning("test.csv not found, falling back to train.csv")
            metadata = read_metadata(Path(cfg["paths"]["train_csv"]), data_root)
    else:
        # Use train.csv and filter by split
        metadata = read_metadata(Path(cfg["paths"]["train_csv"]), data_root)
        if args.split in ["train", "val"]:
            data_cfg = cfg["data"]
            train_folds = data_cfg.get("train_folds", [0, 1, 2])
            val_folds = data_cfg.get("val_folds", [3])
            train_records, val_records = split_records_by_fold(metadata, train_folds, val_folds)
            if args.split == "train":
                metadata = train_records
                logger.info("Using %d train volumes", len(metadata))
            else:  # val
                metadata = val_records
                logger.info("Using %d validation volumes", len(metadata))
    
    if not metadata:
        raise ValueError(f"No volumes found for split: {args.split}")
    
    # Limit to max_volumes if specified (for quick testing)
    if args.max_volumes is not None:
        metadata = metadata[:args.max_volumes]
        logger.info("Limited to %d volumes (--max-volumes)", args.max_volumes)
    
    dataset = FullVolumeDataset(metadata, cfg["data"], augmentation_cfg=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_idx, batch in enumerate(loader, 1):
        volume_id = batch["volume_id"][0]
        image = batch["image"]
        actual_depth = image.shape[2]
        image = image.to(device)
        logger.info("[%d/%d] Processing %s (depth=%d)...", batch_idx, len(loader), volume_id, actual_depth)
        pred = inference_with_tta(model, image, inference_cfg, device)
        threshold = inference_cfg.get("threshold", 0.5)
        binary = (pred >= threshold).astype(np.uint8)
        post_cfg = cfg.get("postprocess", {})
        binary = apply_postprocessing(binary, post_cfg)
        out_path = output_dir / f"{volume_id}.tif"
        if tifffile:
            tifffile.imwrite(out_path.as_posix(), binary.astype(np.uint8))
        else:
            np.save(out_path.with_suffix(".npy"), binary)
        logger.info("✓ Saved prediction for %s", volume_id)
        log_gpu_memory(logger, prefix="[GPU] ")


if __name__ == "__main__":
    main()


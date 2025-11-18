from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FullVolumeDataset, read_metadata
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
    model.eval()
    patch_size = tuple(cfg["patch_size"])
    overlap = tuple(cfg["overlap"])
    sigma = cfg.get("gaussian_blend_sigma", 0.125)
    weights = gaussian_weights(patch_size, sigma)
    volume_np = volume[0].numpy()
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
    experiment_cfg = cfg["experiment"]
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

    metadata = read_metadata(Path(cfg["paths"]["train_csv"]), Path(cfg["paths"]["data_root"]))
    dataset = FullVolumeDataset(metadata, cfg["data"], augmentation_cfg=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        volume_id = batch["volume_id"][0]
        image = batch["image"].to(device)
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
        logger.info("Saved prediction for %s", volume_id)
        log_gpu_memory(logger, prefix="[Inference] ")


if __name__ == "__main__":
    main()


"""
Kaggle Notebook Template for Vesuvius Challenge Submission

This notebook runs inference on the test set and generates submission.zip

IMPORTANT:
- No internet access in Kaggle environment
- All dependencies must be in dataset or pre-installed
- Must complete within 9 hours
- Must produce submission.zip with .tif files
"""

import os
import zipfile
from pathlib import Path
import numpy as np
import torch
import tifffile

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (adjust based on Kaggle dataset structure)
CHECKPOINT_PATH = "/kaggle/input/vesuvius-model-checkpoint/best.pt"  # Update to your checkpoint path
CONFIG_PATH = "/kaggle/input/vesuvius-model-checkpoint/config.yaml"  # Optional: if you include config
DATA_ROOT = Path("/kaggle/input/vesuvius-challenge")
TEST_CSV = DATA_ROOT / "test.csv"
TEST_IMAGES_DIR = DATA_ROOT / "test_images"
OUTPUT_DIR = Path("/kaggle/working/predictions")

# Inference settings (from config)
PATCH_SIZE = [80, 144, 144]
OVERLAP = [40, 96, 96]
THRESHOLD = 0.42
TTA_MODE = "full_8x"  # Options: "none", "flips", "full_8x"
MIN_COMPONENT_VOXELS = 600
MORPH_CLOSING_RADIUS = 3

# ============================================================================
# MODEL LOADING
# ============================================================================

print("Loading model...")
# Import your model code (you'll need to include src/vesuvius/ in your dataset)
import sys
sys.path.append("/kaggle/input/vesuvius-model-checkpoint")  # Adjust path

from src.vesuvius.models import build_model
from src.vesuvius.infer import (
    inference_with_tta,
    sliding_window_predict,
    tta_combinations,
    apply_tta,
    inverse_tta,
    generate_coords,
    gaussian_weights,
)
from src.vesuvius.postprocess import apply_postprocessing
from src.vesuvius.data import read_metadata, FullVolumeDataset
from src.vesuvius.utils import load_config

# Load config (if included in dataset)
if os.path.exists(CONFIG_PATH):
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg["model"]
    inference_cfg = cfg["inference"]
    # Override with config values if available
    PATCH_SIZE = inference_cfg.get("patch_size", PATCH_SIZE)
    OVERLAP = inference_cfg.get("overlap", OVERLAP)
    THRESHOLD = inference_cfg.get("threshold", THRESHOLD)
    TTA_MODE = inference_cfg.get("tta", TTA_MODE)
    MIN_COMPONENT_VOXELS = inference_cfg.get("min_component_voxels", MIN_COMPONENT_VOXELS)
    MORPH_CLOSING_RADIUS = inference_cfg.get("morph_closing_radius", MORPH_CLOSING_RADIUS)
else:
    # Default model config (adjust to match your training config)
    model_cfg = {
        "type": "unet3d_residual",
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 40,
        "channel_multipliers": [1, 2, 2, 4, 4],
        "blocks_per_stage": 3,
        "norm": "instance",
        "activation": "mish",
        "deep_supervision": True,
        "dropout": 0.15,
    }

# Build model
model = build_model(model_cfg)

# Load checkpoint
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ============================================================================
# DATA LOADING
# ============================================================================

print("Loading test metadata...")
metadata = read_metadata(TEST_CSV, DATA_ROOT)

# Data preprocessing config (should match training config)
data_cfg = {
    "resample_spacing": [0.035, 0.035, 0.035],
    "intensity_clip": [-1200, 800],
    "normalize": "zscore",
    "denoise": {
        "enabled": True,
        "method": "median3d",
        "kernel_size": 3,
    },
}

dataset = FullVolumeDataset(metadata, data_cfg, augmentation_cfg=None)
print(f"Found {len(dataset)} test volumes")

# ============================================================================
# INFERENCE
# ============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

inference_cfg = {
    "patch_size": PATCH_SIZE,
    "overlap": OVERLAP,
    "gaussian_blend_sigma": 0.125,
    "tta": TTA_MODE,
    "threshold": THRESHOLD,
    "min_component_voxels": MIN_COMPONENT_VOXELS,
    "morph_closing_radius": MORPH_CLOSING_RADIUS,
}

postprocess_cfg = {
    "remove_small_components_voxels": MIN_COMPONENT_VOXELS,
    "fill_holes": True,
    "closing_radius": MORPH_CLOSING_RADIUS,
}

print(f"Starting inference with TTA={TTA_MODE}, threshold={THRESHOLD}...")

for idx, record in enumerate(metadata):
    volume_id = record.volume_id
    print(f"[{idx+1}/{len(metadata)}] Processing {volume_id}...")
    
    # Load volume
    image, _ = dataset.cache.get(record, data_cfg)
    image_tensor = torch.from_numpy(image[None, None, ...].astype(np.float32))
    
    # Run inference with TTA
    with torch.no_grad():
        pred = inference_with_tta(model, image_tensor, inference_cfg, device)
    
    # Threshold and post-process
    binary = (pred >= THRESHOLD).astype(np.uint8)
    binary = apply_postprocessing(binary, postprocess_cfg)
    
    # Save prediction
    output_path = OUTPUT_DIR / f"{volume_id}.tif"
    tifffile.imwrite(output_path.as_posix(), binary.astype(np.uint8))
    print(f"  Saved: {output_path}")

print(f"Inference complete. Predictions saved to {OUTPUT_DIR}")

# ============================================================================
# CREATE SUBMISSION.ZIP
# ============================================================================

print("Creating submission.zip...")
submission_zip = Path("/kaggle/working/submission.zip")

with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for pred_file in sorted(OUTPUT_DIR.glob("*.tif")):
        zf.write(pred_file, pred_file.name)
        print(f"  Added: {pred_file.name}")

print(f"Submission created: {submission_zip}")
print(f"Total files: {len(list(OUTPUT_DIR.glob('*.tif')))}")

# Verify submission
with zipfile.ZipFile(submission_zip, "r") as zf:
    files = zf.namelist()
    print(f"Verification: submission.zip contains {len(files)} files")
    if len(files) > 0:
        print(f"  First file: {files[0]}")
        print(f"  Last file: {files[-1]}")

print("\n" + "="*60)
print("SUBMISSION READY")
print("="*60)
print(f"File: {submission_zip}")
print(f"Size: {submission_zip.stat().st_size / (1024*1024):.2f} MB")
print(f"Files: {len(files)}")
print("="*60)


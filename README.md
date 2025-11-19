# Vesuvius Challenge – Surface Detection

3D segmentation toolkit for the [Vesuvius Challenge](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection). Trains on Google Cloud A100 VM and runs inference in Kaggle notebooks.

---

## Quick Start

### Training
```bash
cd /mnt/disks/data/repos/vesuvius_challenge
source .venv/bin/activate
export PYTHONPATH=/mnt/disks/data/repos/vesuvius_challenge

# Full topology-aware training
python -m src.vesuvius.train --config configs/experiments/exp001_full.yaml --device cuda
```

### Inference
```bash
export PYTHONPATH=src
python src/vesuvius/infer.py \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint runs/exp001_3d_unet_topology/checkpoints/best.pt \
  --output-dir runs/exp001_3d_unet_topology/infer
```

---

## Features

- **Config-driven pipeline** with YAML inheritance
- **3D data processing:** Resampling, denoising, FG-aware patch sampling, sliding-window inference
- **Topology-aware losses:** Weighted BCE + Dice + clDice + skeleton + surface-distance + TopoLoss
- **Models:** Residual UNet (deep supervision) + optional Swin UNETR
- **Training:** AMP, gradient accumulation/clipping, EMA, schedulers
- **Inference:** Sliding-window with Gaussian blending, TTA, post-processing

---

## Environment

- **Hardware:** GCP `a2-ultragpu-1g` VM (A100 80GB) for training
- **Data:** `gs://vesuvius-kaggle-data` mounted via `gcsfuse` at `/home/<user>/gcs_mount`
- **Python:** 3.10+ with PyTorch + SciPy ecosystem

### Data Access Setup
```bash
# Install gcsfuse (Debian 12)
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb http://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/gcsfuse-archive-keyring.gpg >/dev/null
sudo apt-get update && sudo apt-get install -y gcsfuse

# Mount bucket
mkdir -p ~/gcs_mount
gcsfuse --implicit-dirs vesuvius-kaggle-data ~/gcs_mount
```

---

## Configuration

- **Base config:** `configs/vesuvius_baseline.yaml`
- **Experiments:** `configs/experiments/exp001_*.yaml`
- **Config inheritance:** Use `_base_` key to inherit and override
- **Resolved configs:** Saved to `runs/<exp>/config_resolved.yaml`

### Recommended Training Config
```yaml
training:
  workers: 4
  train_batch_size: 2
  train_iterations_per_epoch: 2000
  max_epochs: 32
```

---

## Project Structure

```
├── configs/
│   ├── vesuvius_baseline.yaml
│   └── experiments/
│       ├── exp001_3d_unet_topology.yaml
│       └── exp001_full.yaml
├── src/vesuvius/
│   ├── data.py          # Datasets, preprocessing
│   ├── models.py        # UNet variants
│   ├── losses.py        # Composite topology losses
│   ├── train.py         # Training loop
│   └── infer.py         # Inference pipeline
├── tests/
└── DEVLOG.md            # Development log
```

---

## Documentation

- **`DEVLOG.md`** - Development history and decisions
- **`STEPS_PER_EPOCH_GUIDE.md`** - Guide on choosing steps per epoch
- **`LOSS_COMPONENTS_EXPLAINED.md`** - Detailed loss component explanations
- **`TEST_RUN_EVALUATION.md`** - Test run analysis and recommendations

---

## Testing

```bash
export PYTHONPATH=src
pytest tests/test_synthetic_pipeline.py
```

---

## Status

✅ **System validated** - Test run completed successfully (32 epochs, checkpoints saving)  
✅ **Ready for full training** - All fixes applied, optimizations in place  
📊 **Next:** Full training run with optimized configuration

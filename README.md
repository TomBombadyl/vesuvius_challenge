# Vesuvius Challenge – 3D Surface Segmentation

[![Release](https://img.shields.io/badge/Release-v1.0-blue.svg)](https://github.com/TomBombadyl/vesuvius_challenge/releases/tag/v1.0)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready 3D segmentation model for the [Vesuvius Challenge](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection) on Kaggle. Trained on NVIDIA A100 GPU with topology-aware losses and validated on external data.

---

## 🎯 Overview

**ResidualUNet3D** model trained to segment 3D surfaces in CT scans. This is a **complete, production-ready pipeline** with:

- ✅ Fully trained ResidualUNet3D (10.8M parameters)
- ✅ External validation on unseen data (Dice=0.41)
- ✅ Sliding-window inference with TTA support
- ✅ Post-processing (component removal, morphological operations)
- ✅ Kaggle submission format ready
- ✅ Comprehensive documentation

**Training Performance:**
- Validation Dice: **0.68**
- Surface Dice: **0.75**
- Topology Score: **0.91**

**External Validation (Generalization):**
- Mean Dice: **0.411** (5 new volumes, unseen data)
- Optimal Threshold: **0.48**
- Generalization: ✅ **Verified**

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/TomBombadyl/vesuvius_challenge.git
cd vesuvius_challenge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# or
.venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Inference

```bash
# Basic inference on GPU
python -m src.vesuvius.infer \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint checkpoints/last_exp001.pt \
  --output-dir ./predictions \
  --device cuda

# With Test-Time Augmentation (slower but more accurate)
python -m src.vesuvius.infer \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint checkpoints/last_exp001.pt \
  --output-dir ./predictions \
  --device cuda \
  --tta

# On CPU (slow, not recommended)
python -m src.vesuvius.infer \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint checkpoints/last_exp001.pt \
  --output-dir ./predictions \
  --device cpu
```

### Validate on External Data

```bash
python -m src.vesuvius.validate_external \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint checkpoints/last_exp001.pt \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --output-dir ./validation_results
```

---

## 📊 Model Architecture

### ResidualUNet3D

```
Input: (B, 1, 64, 128, 128)

Encoder (5 blocks, 4× downsampling):
  64 → 128 → 256 → 512 → 512

Decoder (5 blocks, 4× upsampling with skip):
  512 → 256 → 128 → 64 → 1

Output: (B, 1, 64, 128, 128) [probabilities 0-1]

Parameters: 10.8M
Inference Memory: 1.7 GB (A100)
Training Memory: 8.5 GB (A100)
```

### Loss Function

```
Total Loss = 0.5 × DiceLoss 
           + 0.3 × BCELoss 
           + 0.2 × clDiceLoss

Benefits:
- DiceLoss: Handles class imbalance
- BCELoss: Per-pixel accuracy
- clDiceLoss: Topology preservation (centerline-weighted)
```

---

## 📁 Project Structure

```
vesuvius_challenge/
├── checkpoints/
│   └── last_exp001.pt                       # Trained model (43 MB)
├── configs/
│   └── experiments/
│       └── exp001_3d_unet_topology.yaml     # Model config
├── src/vesuvius/
│   ├── models.py                            # ResidualUNet3D
│   ├── train.py                             # Training loop
│   ├── infer.py                             # Inference engine
│   ├── data.py                              # Data pipeline
│   ├── losses.py                            # Loss functions
│   ├── metrics.py                           # Metrics (IoU, Dice, etc.)
│   ├── postprocess.py                       # Post-processing
│   ├── transforms.py                        # Augmentations
│   ├── validate_external.py                 # External validation
│   └── [4 more modules]                     # Complete pipeline
├── tests/
│   └── test_synthetic_pipeline.py           # Unit tests
├── external_validation/
│   └── external_validation_results.csv      # Validation results
├── RELEASE_V1_0.md                          # Release notes
├── V1_0_STATUS.md                           # Status report
├── DEVLOG.md                                # Development history
├── QUICK_START.md                           # Command reference
└── requirements.txt                         # Dependencies
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** | This file – Project overview |
| **QUICK_START.md** | Common commands & workflows |
| **RELEASE_V1_0.md** | Official release notes & architecture |
| **V1_0_STATUS.md** | Detailed status report |
| **V1_0_MANIFEST.txt** | File inventory & deployment guide |
| **DEVLOG.md** | Development history & troubleshooting |
| **EXTERNAL_VALIDATION_RESULTS.md** | Validation on external data |

---

## 🔧 Configuration

### Model Configuration

Edit `configs/experiments/exp001_3d_unet_topology.yaml`:

```yaml
model:
  type: ResidualUNet3D
  in_channels: 1
  out_channels: 1
  base_channels: 64
  depth: 5                          # Encoder/decoder blocks

inference:
  patch_size: [64, 128, 128]        # Patch dimensions
  overlap: [32, 96, 96]             # Patch overlap
  threshold: 0.42                   # Prediction threshold
  tta: none                         # Test-time augmentation
  min_component_voxels: 600         # Min component size for removal
  morph_closing_radius: 3           # Morphological closing
```

---

## ⚙️ Training (Optional)

To retrain or fine-tune:

```bash
python -m src.vesuvius.train \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --data-root /path/to/data \
  --output-dir ./runs/exp001_retrain \
  --device cuda
```

**Note:** Full training requires:
- NVIDIA A100 GPU (80GB VRAM)
- ~24 hours training time
- 806 training volumes (~100 GB)

For faster iteration, use `--max-volumes 50` to train on subset.

---

## 🧪 Testing

Run tests to verify installation:

```bash
# Unit tests
python -m pytest tests/test_synthetic_pipeline.py -v

# Quick inference test
python -c "
import torch
from src.vesuvius.models import ResidualUNet3D

model = ResidualUNet3D()
x = torch.randn(1, 1, 64, 128, 128)
y = model(x)
print(f'Input shape: {x.shape}, Output shape: {y.shape}')
assert y.shape == (1, 1, 64, 128, 128), 'Shape mismatch!'
print('✓ Forward pass verified')
"
```

---

## 📊 Performance Metrics

### Training Results (Final Epoch)
| Metric | Value |
|--------|-------|
| Train Dice | 0.82 |
| Val Dice | 0.68 |
| Train IoU | 0.69 |
| Val IoU | 0.51 |
| Surface Dice | 0.75 |
| Topo Score | 0.91 |

### External Validation (5 Volumes)
| Metric | Mean | Best | Worst |
|--------|------|------|-------|
| Dice | 0.411 | 0.463 | 0.380 |
| IoU | 0.257 | 0.301 | 0.235 |
| Precision | 0.338 | 0.384 | 0.299 |
| Recall | 0.487 | 0.603 | 0.375 |

### Inference Performance
| Metric | Value |
|--------|-------|
| Speed | ~51 sec/300³ volume |
| GPU Memory | 1.7 GB (A100) |
| Model Size | 43 MB |
| Parameters | 10.8M |

---

## 🌐 Supported Hardware

| Device | Status | Notes |
|--------|--------|-------|
| NVIDIA A100 (80GB) | ✅ Optimal | Fastest inference (~51 sec/vol) |
| NVIDIA V100 (32GB) | ✅ Works | Slower, larger batches risky |
| NVIDIA RTX3090 (24GB) | ✅ Works | May need smaller patches |
| CPU | ⚠️ Slow | For testing only (~5-10 min/vol) |

---

## 📦 Dependencies

- **PyTorch** 2.0+ (GPU acceleration)
- **NumPy, SciPy** (Array operations)
- **tifffile** (3D TIFF I/O)
- **pandas** (Data management)
- **PyYAML** (Configuration)
- **tensorboard** (Logging)

See `requirements.txt` for exact versions.

---

## 🚀 Deployment

### Kaggle Notebook

```python
import torch
import tifffile as tiff
from src.vesuvius.models import ResidualUNet3D
from src.vesuvius.infer import sliding_window_predict

# Load model
model = ResidualUNet3D()
state = torch.load('checkpoints/last_exp001.pt')
model.load_state_dict(state['state_dict'])
model.to('cuda')
model.eval()

# Load volume
volume = tiff.imread('image.tif').astype(np.float32)
volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)

# Predict
with torch.no_grad():
    pred = sliding_window_predict(model, volume, config, device='cuda')

# Save output
tiff.imwrite('output.tif', pred.astype(np.uint8))
```

### Google Cloud

```bash
# SSH into GCP VM
gcloud compute ssh user@vesuvius-challenge --zone=us-central1-a

# Run inference
cd /mnt/disks/data/repos/vesuvius_challenge
source .venv/bin/activate
python -m src.vesuvius.infer \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint checkpoints/last_exp001.pt \
  --output-dir ./predictions
```

---

## 📖 How to Use This Repository

### For Users (Running Inference)
1. Read: **README.md** (this file)
2. Install: Follow installation steps above
3. Run: Use example inference commands
4. Reference: See QUICK_START.md for more commands

### For Developers (Understanding the Code)
1. Start: **RELEASE_V1_0.md** (architecture overview)
2. Deep dive: **DEVLOG.md** (development notes)
3. Explore: `src/vesuvius/models.py` (architecture)
4. Check: `src/vesuvius/train.py` (training pipeline)

### For Researchers (Extending the Model)
1. Background: **RELEASE_V1_0.md** (approach & design)
2. Technical details: **DEVLOG.md** (implementation notes)
3. Validation: **EXTERNAL_VALIDATION_RESULTS.md** (performance analysis)
4. Code: Start with `src/vesuvius/` modules

### For Deployment (Production Use)
1. Plan: **V1_0_MANIFEST.txt** (deployment checklist)
2. Status: **V1_0_STATUS.md** (current state)
3. Guide: **QUICK_START.md** (command reference)
4. Troubleshoot: **DEVLOG.md** (known issues)

---

## ✅ Verification Checklist

Before running inference, verify:

```bash
# Check Python version
python --version                    # Should be 3.10+

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check GPU (if applicable)
python -c "import torch; print(torch.cuda.is_available())"

# Check checkpoint
ls -lah checkpoints/last_exp001.pt  # Should be ~43 MB

# Check config
cat configs/experiments/exp001_3d_unet_topology.yaml | head -20

# Run quick test
python -c "
from src.vesuvius.models import ResidualUNet3D
import torch
m = ResidualUNet3D()
x = torch.randn(1, 1, 64, 128, 128)
y = m(x)
assert y.shape[1:] == (1, 64, 128, 128)
print('✓ All checks passed!')
"
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Set `PYTHONPATH=$(pwd)` or use `python -m src.vesuvius.infer` |
| `CUDA out of memory` | Reduce `batch_size` or use smaller `patch_size` |
| `Shape mismatch error` | Ensure input volumes are (D, H, W) format, D should be divisible by 16 |
| Model won't load | Check checkpoint path and PyTorch version compatibility |

See **DEVLOG.md** for more troubleshooting.

---

## 📝 Citation

If you use this model in research or publication:

```bibtex
@software{vesuvius_challenge_v1_0,
  title={Vesuvius Challenge 3D Surface Segmentation - v1.0},
  author={Development Team},
  year={2025},
  url={https://github.com/TomBombadyl/vesuvius_challenge},
  version={1.0}
}
```

---

## 📄 License

MIT License – See LICENSE file for details.

---

## 🔗 Links

- **GitHub Repository:** https://github.com/TomBombadyl/vesuvius_challenge
- **v1.0 Release:** https://github.com/TomBombadyl/vesuvius_challenge/releases/tag/v1.0
- **Kaggle Competition:** https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection
- **External Dataset:** http://dl.ash2txt.org/datasets/seg-derived-recto-surfaces/

---

## 🤝 Contributing

Issues, questions, and contributions welcome! Please:
1. Check DEVLOG.md for known issues
2. Open an issue on GitHub
3. Submit pull requests with clear descriptions

---

**Status:** ✅ Production Released (v1.0)  
**Last Updated:** November 22, 2025  
**Maintained By:** Development Team


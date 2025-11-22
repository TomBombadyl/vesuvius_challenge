# External Dataset Validation

## Overview

Validate your trained ResidualUNet3D model on external paired datasets (images + ground truth masks). Reuses your existing sliding-window inference and TTA logic.

## Usage

```bash
python -m src.vesuvius.validate_external \
  --config configs/experiments/exp001_3d_unet_topology.yaml \
  --checkpoint checkpoints/last_exp001.pt \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --output-dir runs/external_validation \
  --max-volumes 10
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to experiment YAML config |
| `--checkpoint` | Path to trained model checkpoint |
| `--image-dir` | Directory containing `.tif` image volumes |
| `--mask-dir` | Directory containing `.tif` ground truth masks |
| `--output-dir` | Where to save results CSV and logs |
| `--device` | torch device (default: `cuda`) |
| `--max-volumes` | Limit to N volumes for quick testing (default: all) |

## What It Does

1. **Loads model** from checkpoint
2. **Pairs images with masks** by filename
3. **Runs inference** on each volume using sliding-window + TTA
4. **Evaluates at thresholds 0.30-0.55** computing:
   - Dice coefficient
   - IoU
   - Precision & Recall
5. **Outputs CSV** with per-volume + per-threshold metrics
6. **Logs summary** with best threshold recommendation

## Output Files

**`runs/external_validation/`**
- `external_validation_results.csv` - Full metrics table
- `validate_external.log` - Detailed execution log

**CSV Columns:**
```
volume_id, threshold, dice, iou, precision, recall, tp, fp, fn
```

## Expected Performance

### Baseline (Vesuvius Challenge training data)
- Mean Dice: ~0.65-0.72 (depending on threshold)
- Mean IoU: ~0.32 (at default threshold 0.42)

### On External Dataset
- **Good generalization:** Dice > 0.65 ✅
- **Acceptable:** Dice 0.55-0.65 ⚠️
- **Problem:** Dice < 0.55 ❌

## Example: Your Data

If your volumes are named:
```
/mnt/data/s1_z10240_y2560_x2560_0000.tif   (image)
/mnt/data/s1_z10240_y2560_x2560.tif        (mask)
```

You could either:
1. Rename to match (e.g., both `s1_z10240_y2560_x2560.tif`)
2. Modify `load_external_volume_pair()` to customize naming logic

## Normalization

By default, uses **per-volume min-max normalization** (same as training):
```python
image = (image - image.min()) / (image.max() - image.min())
```

If your images use different normalization (e.g., percentile-based), edit `load_external_volume_pair()` in `src/vesuvius/validate_external.py`.

## Next Steps

1. **Quick test (3 volumes):**
   ```bash
   python -m src.vesuvius.validate_external \
     --config configs/experiments/exp001_3d_unet_topology.yaml \
     --checkpoint checkpoints/last_exp001.pt \
     --image-dir ... --mask-dir ... \
     --output-dir runs/external_validation \
     --max-volumes 3
   ```

2. **Check results:**
   ```bash
   cat runs/external_validation/external_validation_results.csv
   ```

3. **If Dice > 0.65:** Proceed to Phase 1 (full inference on training data)
4. **If Dice < 0.55:** Debug normalization or checkpoint

---

**Reuses:** `sliding_window_predict()`, `inference_with_tta()`, `gaussian_weights()` from `infer.py`

**No retraining:** Purely evaluation on frozen checkpoint


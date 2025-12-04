# Architecture Documentation

Technical deep-dive into the Vesuvius Challenge Surface Detection solution.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Loss Functions](#loss-functions)
3. [Training Pipeline](#training-pipeline)
4. [Inference Pipeline](#inference-pipeline)
5. [Post-Processing](#post-processing)
6. [Virtual Unwrapping](#virtual-unwrapping)

---

## Model Architecture

### ResidualUNet3D

3D U-Net with residual connections for robust feature learning.

**Key Specifications:**
- **Parameters:** 10.8M (exp001) to 36.7M (ink detection variant)
- **Input:** (B, 1, D, H, W) - Batch, channels, depth, height, width
- **Output:** (B, 1, D, H, W) - Binary segmentation logits
- **Depth Levels:** 5 (4× downsampling)
- **Base Channels:** 32-40 (configurable)

**Architecture Flow:**

```
Input: (1, 1, 64, 128, 128)

Encoder:
  Block 1: 1 → 40 channels   [64, 128, 128]
  Pool → MaxPool3d(2)
  Block 2: 40 → 80 channels  [32, 64, 64]
  Pool → MaxPool3d(2)
  Block 3: 80 → 80 channels  [16, 32, 32]
  Pool → MaxPool3d(2)
  Block 4: 80 → 160 channels [8, 16, 16]
  Pool → MaxPool3d(2)
  Block 5: 160 → 160 channels [4, 8, 8]

Bottleneck:
  160 → 320 channels [4, 8, 8]

Decoder (with skip connections):
  Up 1: 320 + 160 → 160 channels [8, 16, 16]
  Up 2: 160 + 80 → 80 channels   [16, 32, 32]
  Up 3: 80 + 80 → 80 channels    [32, 64, 64]
  Up 4: 80 + 40 → 40 channels    [64, 128, 128]

Output Head: 40 → 1 channel [64, 128, 128]
```

**Residual Block:**
```
x → Conv3d(3×3×3) → InstanceNorm → Mish → Conv3d(3×3×3) → InstanceNorm → Dropout
  ↓                                                                         ↓
  → Conv3d(1×1×1) if channels change ────────────────────────────────────→ +
                                                                            ↓
                                                                          Mish
```

**Key Features:**
- **Residual connections:** Improve gradient flow
- **Instance normalization:** Better than batch norm for 3D medical imaging
- **Mish activation:** Smooth, non-monotonic activation
- **Deep supervision:** Auxiliary losses from decoder stages
- **Activation checkpointing:** Trade compute for memory

---

## Loss Functions

### Composite Topology Loss

Weighted combination of 6 loss components:

```python
Total Loss = 0.35 × WeightedBCE
           + 0.35 × SoftDice
           + 0.10 × clDice
           + 0.08 × MorphSkeleton
           + 0.07 × SurfaceDistance
           + 0.05 × TopoLossSimple
```

### 1. Weighted Binary Cross-Entropy (35%)

```python
BCE = -[w_pos × y × log(σ(p)) + (1-y) × log(1-σ(p))]
```

- **Purpose:** Pixel-wise classification
- **pos_weight:** 2.8 (compensate for class imbalance)
- **Why:** Handles sparse foreground (papyrus is ~16% of volume)

### 2. Soft Dice Loss (35%)

```python
Dice = 1 - (2 × Σ(p × y) + smooth) / (Σp + Σy + smooth)
```

- **Purpose:** Region overlap optimization
- **smooth:** 1.0 (numerical stability)
- **Why:** Directly optimizes segmentation quality metric

### 3. clDice - Centerline Dice (10%)

```python
clDice = (2 × Σ(skel(p) × y) + smooth) / (Σskel(p) + Σy + smooth)
```

- **Purpose:** Preserve thin structures and connectivity
- **skeleton_kernel:** 3×3×3
- **Why:** Papyrus sheets are thin surfaces

### 4. Morphological Skeleton Loss (8%)

```python
MorphLoss = MSE(morph_skel(p), morph_skel(y))
```

- **Purpose:** Match topological skeleton
- **iterations:** 3
- **Why:** Enforce structural similarity

### 5. Surface Distance Loss (7%)

```python
SurfDist = Hausdorff_distance(surface(p), surface(y))
```

- **Purpose:** Minimize boundary errors
- **tolerance_mm:** 2.0
- **Why:** Directly optimizes competition metric (SurfaceDice@2.0)

### 6. Simple Topology Loss (5%)

```python
TopoLoss = |Betti_0(p) - Betti_0(y)| + |Betti_1(p) - Betti_1(y)|
```

- **Purpose:** Preserve topological features
- **betti_threshold:** 0.2
- **Why:** Avoid splits and mergers (optimizes TopoScore)

---

## Training Pipeline

### Data Loading

**Patch-based training:**
- **Patch size:** [80, 144, 144] voxels
- **Stride:** [40, 112, 112] (50% overlap)
- **Sampling:** Foreground-aware (65% patches contain surface)
- **Rejection:** 5% of background patches rejected

**Augmentation:**
- Spatial: Rotation (±20°), scaling (0.85-1.2×), elastic deformation
- Intensity: Gamma (0.65-1.6), noise, blur
- 3D-specific: Anisotropy scaling, slice jitter, patch dropout

### Training Loop

```python
for epoch in range(150):
    # Training
    for batch in train_loader:
        images, masks = batch
        outputs = model(images)
        
        # Compute loss
        loss, components = composite_loss(outputs['logits'], masks)
        
        # Deep supervision (if enabled)
        if outputs['aux_logits']:
            for aux_logits in outputs['aux_logits']:
                loss += 0.5 * composite_loss(aux_logits, masks)
        
        # Backprop with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # EMA update
        if ema:
            ema.update(model)
    
    # Validation
    if epoch % val_interval == 0:
        val_metrics = validate(ema.model if ema else model, val_loader)
        
        # Save best checkpoint
        if val_metrics['surface_dice'] > best_metric:
            save_checkpoint('best.pt')
```

**Optimizer:** AdamW
- Learning rate: 3e-4
- Weight decay: 1e-2
- Betas: (0.9, 0.999)

**Scheduler:** OneCycleLR
- Max LR: 6e-4
- Warmup: 20% of training
- Annealing: Cosine

**EMA:** Exponential Moving Average
- Decay: 0.998
- Used for validation and inference

---

## Inference Pipeline

### Sliding Window Strategy

```python
def sliding_window_predict(model, volume, config):
    patch_size = config['patch_size']  # [64, 128, 128]
    overlap = config['overlap']        # [32, 96, 96] (50%)
    
    # Initialize accumulators
    pred_sum = zeros_like(volume)
    weight_sum = zeros_like(volume)
    
    # Gaussian weights for blending
    weights = gaussian_weights(patch_size, sigma=0.125)
    
    # Generate patch coordinates
    for z, y, x in generate_coords(volume.shape, patch_size, overlap):
        # Extract patch
        patch = volume[:, :, z:z+pz, y:y+py, x:x+px]
        
        # Predict
        with torch.no_grad():
            logits = model(patch)['logits']
            probs = torch.sigmoid(logits)
        
        # Accumulate with Gaussian weighting
        pred_sum[z:z+pz, y:y+py, x:x+px] += probs * weights
        weight_sum[z:z+pz, y:y+py, x:x+px] += weights
    
    # Normalize
    return pred_sum / weight_sum
```

**Key Features:**
- **Overlap:** 50% in all dimensions for smooth blending
- **Gaussian weighting:** Reduces edge artifacts
- **Memory efficient:** Processes one patch at a time
- **GPU optimized:** Mixed precision (FP16) inference

### Test-Time Augmentation (TTA)

Optional for improved accuracy:

```python
# TTA modes
'none': No augmentation (fastest)
'flips': 4× (flip each axis)
'full_8x': 8× (all flip combinations)

# Average predictions
pred_final = mean([pred_original, pred_flip_z, pred_flip_y, pred_flip_x, ...])
```

---

## Post-Processing

### Pipeline Steps

1. **Binarization**
   ```python
   binary_mask = (predictions >= threshold).astype(uint8)
   ```
   - Threshold: 0.42 (optimized on validation)

2. **Small Component Removal**
   ```python
   remove_small_components(mask, min_size=600)
   ```
   - Removes noise and isolated voxels
   - Preserves main papyrus sheets

3. **Morphological Closing**
   ```python
   binary_closing(mask, structure=sphere(radius=3))
   ```
   - Fills small gaps
   - Connects nearby surface regions

### Parameter Tuning

| Parameter | Surface Seg | Ink Detection |
|-----------|-------------|---------------|
| Threshold | 0.42 | 0.50 |
| Min component | 600 voxels | 50 voxels |
| Closing radius | 3 | 1 |

---

## Virtual Unwrapping

### Surface Extraction

Extract 2D surface from 3D volume:

```python
def extract_surface_from_volume(volume, surface_mask, method='center'):
    surface_image = zeros((H, W))
    
    for h in range(H):
        for w in range(W):
            # Find surface voxels in this column
            surface_indices = where(surface_mask[:, h, w] > 0)
            
            if method == 'center':
                # Take center depth
                idx = surface_indices[len(surface_indices) // 2]
                surface_image[h, w] = volume[idx, h, w]
            elif method == 'mean':
                # Average all surface voxels
                surface_image[h, w] = mean(volume[surface_indices, h, w])
            elif method == 'max':
                # Maximum intensity
                surface_image[h, w] = max(volume[surface_indices, h, w])
    
    return surface_image
```

### Visualization Pipeline

1. **Surface extraction:** 3D → 2D unwrapping
2. **Ink detection:** Threshold or ML-based
3. **Overlay generation:** Ink on papyrus texture
4. **Text rendering:** Black on white for readability

---

## Performance Optimizations

### Memory Management

- **Activation checkpointing:** Reduces memory by 40%
- **Mixed precision (FP16):** 2× faster inference
- **Patch-based processing:** Handles arbitrary volume sizes
- **Gradient accumulation:** Effective batch size = batch_size × accum_steps

### Speed Optimizations

- **Persistent workers:** Avoid dataloader restart overhead
- **Prefetching:** Load next batch while training
- **Pin memory:** Faster CPU→GPU transfer (disabled due to spawn context)
- **Compiled model:** torch.compile() for 15-20% speedup (optional)

### Computational Requirements

**Training:**
- GPU: NVIDIA A100 80GB
- Memory: ~8.5 GB VRAM
- Time: ~24 hours for 150 epochs
- Data: 806 volumes (~100 GB)

**Inference:**
- GPU: NVIDIA A100 80GB
- Memory: ~1.7 GB VRAM
- Speed: ~51 seconds per 300³ volume
- Batch size: 1 (limited by memory)

---

## Evaluation Metrics

### Competition Metrics

**Final Score = 0.30 × TopoScore + 0.35 × SurfaceDice@2.0 + 0.35 × VOI_score**

### 1. SurfaceDice@τ (35%)

Surface-aware Dice coefficient with tolerance τ=2.0mm:

```python
def surface_dice(pred, gt, tolerance_mm=2.0, spacing=(0.04, 0.04, 0.04)):
    # Extract surfaces
    pred_surface = extract_surface_voxels(pred)
    gt_surface = extract_surface_voxels(gt)
    
    # Compute distances
    pred_to_gt = nearest_distance(pred_surface, gt_surface, spacing)
    gt_to_pred = nearest_distance(gt_surface, pred_surface, spacing)
    
    # Count matches within tolerance
    pred_matches = (pred_to_gt <= tolerance_mm).sum()
    gt_matches = (gt_to_pred <= tolerance_mm).sum()
    
    # Symmetric Dice
    return (pred_matches + gt_matches) / (len(pred_surface) + len(gt_surface))
```

### 2. TopoScore (30%)

Topological correctness via Betti number matching:

```python
def topo_score(pred, gt):
    # Compute Betti numbers (k=0: components, k=1: tunnels, k=2: cavities)
    betti_pred = compute_betti_numbers(pred)  # [B0, B1, B2]
    betti_gt = compute_betti_numbers(gt)
    
    # Topological F1 per dimension
    f1_scores = []
    for k in [0, 1, 2]:
        if betti_pred[k] > 0 or betti_gt[k] > 0:
            tp = min(betti_pred[k], betti_gt[k])
            precision = tp / betti_pred[k] if betti_pred[k] > 0 else 0
            recall = tp / betti_gt[k] if betti_gt[k] > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
    
    # Weighted average
    return mean(f1_scores)
```

### 3. VOI Score (35%)

Variation of Information for instance consistency:

```python
def voi_score(pred, gt, alpha=0.3):
    # Connected components
    pred_labels = label_components(pred)
    gt_labels = label_components(gt)
    
    # Compute VOI split and merge
    voi_split = conditional_entropy(gt_labels, pred_labels)
    voi_merge = conditional_entropy(pred_labels, gt_labels)
    voi_total = voi_split + voi_merge
    
    # Convert to bounded score
    return 1 / (1 + alpha * voi_total)
```

---

## Data Pipeline

### Dataset Structure

```
vesuvius_kaggle_data/
├── train.csv          # Metadata with volume IDs and folds
├── test.csv           # Test volume IDs
├── train_images/      # 797 training volumes (.tif)
├── train_labels/      # 797 ground truth masks (.tif)
└── test_images/       # Test volumes (.tif)
```

### Volume Format

- **File type:** TIFF (3D stack)
- **Dimensions:** Variable (typically 300-400³ voxels)
- **Data type:** uint8 (0-255 intensity)
- **Spacing:** ~0.04mm isotropic
- **Labels:** uint8 binary (0=background, 1=papyrus surface)

### Preprocessing

1. **Normalization:** Min-max to [0, 1]
2. **Resampling:** To 0.035mm spacing (optional)
3. **Denoising:** Median filter 3×3×3 (optional)
4. **Intensity clipping:** [-1200, 800] HU (optional)

---

## Virtual Unwrapping Pipeline

### Complete Workflow

```
3D CT Volume (D×H×W)
    ↓
[Surface Segmentation Model]
    ↓
3D Surface Mask (D×H×W binary)
    ↓
[Surface Extraction]
    ↓
2D Unwrapped Surface (H×W)
    ↓
[Ink Detection Model] (optional)
    ↓
2D Ink Mask (H×W binary)
    ↓
[Visualization]
    ↓
Readable Text Images
```

### Implementation

**Surface Extraction:**
```python
# Extract center of surface at each (h,w) location
for h, w in product(range(H), range(W)):
    surface_depths = where(surface_mask[:, h, w] == 1)[0]
    if len(surface_depths) > 0:
        center_depth = surface_depths[len(surface_depths) // 2]
        unwrapped[h, w] = volume[center_depth, h, w]
```

**Visualization:**
- Surface texture: Grayscale unwrapped papyrus
- Ink overlay: Red highlighting on texture
- Text view: Black on white for readability
- Comparison: Side-by-side with ground truth

---

## Code Organization

### Module Responsibilities

**models.py:**
- Model architecture definitions
- Factory function `build_model(config)`
- Supports: ResidualUNet3D, LightUNet3D, SwinUNETR

**data.py:**
- Dataset classes (PatchDataset, FullVolumeDataset)
- Metadata reading and fold splitting
- Volume loading and preprocessing

**losses.py:**
- CompositeTopologyLoss
- Individual loss components
- Loss weighting and combination

**metrics.py:**
- SurfaceDice, TopoScore, VOI
- Evaluation utilities
- Metric computation

**infer.py:**
- Sliding window inference
- TTA implementation
- Gaussian blending

**postprocess.py:**
- Component removal
- Morphological operations
- Hole filling

**submission.py:**
- RLE encoding (for ink detection)
- ZIP creation (for surface detection)
- Validation utilities

**unwrap.py:**
- Surface extraction
- Visualization generation
- Statistics computation

---

## Configuration System

### YAML Structure

```yaml
experiment:
  name: exp001
  seed: 2025
  output_dir: runs/exp001

model:
  type: unet3d_residual
  base_channels: 40
  # ... architecture params

data:
  patch_size: [80, 144, 144]
  sampler:
    foreground_ratio: 0.65
  # ... data params

loss:
  components:
    - name: weighted_bce
      weight: 0.35
    # ... other losses

training:
  max_epochs: 150
  train_batch_size: 1
  # ... training params

inference:
  patch_size: [64, 128, 128]
  threshold: 0.42
  tta: none
```

### Config Inheritance

```yaml
# exp001_3d_unet_topology.yaml
_base_: ../vesuvius_baseline.yaml

# Override specific params
model:
  base_channels: 40  # Override from base
```

---

## Best Practices

### For Training

1. **Start with baseline config** (`vesuvius_baseline.yaml`)
2. **Use foreground-aware sampling** (65% ratio)
3. **Enable EMA** for stable validation
4. **Monitor all loss components** (not just total)
5. **Validate every epoch** to catch overfitting
6. **Save multiple checkpoints** (best, last, epoch milestones)

### For Inference

1. **Use 50% overlap** for smooth predictions
2. **Enable TTA** for final submission (slower but better)
3. **Optimize threshold** on validation set
4. **Apply minimal post-processing** to preserve topology
5. **Validate output format** before submission

### For Debugging

1. **Check shapes** at every step
2. **Visualize predictions** on validation data
3. **Monitor GPU memory** usage
4. **Log patch statistics** during training
5. **Test on small subset** before full run

---

## References

### Papers

- Parsons et al. (2023): EduceLab-Scrolls dataset
- Shit et al. (2021): clDice loss
- Stucki et al. (2024): Betti matching for topology
- Nikolov et al. (2021): Medical image segmentation

### Resources

- Competition: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection
- Vesuvius Challenge: https://scrollprize.org/
- EduceLab: https://educelab.engr.uky.edu/
- Volume Cartographer: https://github.com/educelab/volume-cartographer

---

**Last Updated:** December 4, 2025  
**Version:** 2.0

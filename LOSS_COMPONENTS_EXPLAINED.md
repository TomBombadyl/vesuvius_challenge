# Training Loss Components Explained

## Overview

Your model uses a **composite loss function** that combines multiple specialized losses to train a 3D segmentation model for detecting papyrus scroll surfaces. Each component targets different aspects of the segmentation problem.

---

## Your Current Metrics

```
Train metrics: {
  'loss': 1.689,                    # Total combined loss
  'weighted_bce': 0.909,            # Binary classification accuracy
  'soft_dice': 0.718,               # Overlap between prediction and ground truth
  'cldice': 0.687,                  # Centerline/skeleton matching
  'morph_skeleton': 0.048,          # Morphological skeleton similarity
  'surface_distance': 0.148,        # Surface boundary accuracy
  'toploss_simple': 0.416           # Topology preservation (holes, connectivity)
}
```

---

## 1. **`loss` (Total Combined Loss) = 1.689**

**What it is:**
- The **sum of all weighted loss components**
- Formula: `loss = Σ(weight_i × component_i)`
- This is what the optimizer minimizes

**How it helps:**
- ✅ **Primary training signal**: The optimizer uses this to update model weights
- ✅ **Overall progress indicator**: Lower = better model performance
- ✅ **Balances multiple objectives**: Combines pixel-level, boundary, and topology constraints

**What to watch:**
- Should **decrease over epochs** (currently 1.689 is reasonable for early training)
- If it plateaus, model may need more training or different hyperparameters
- If it increases, may indicate overfitting or learning rate too high

---

## 2. **`weighted_bce` (Weighted Binary Cross-Entropy) = 0.909**

**What it is:**
- Standard binary classification loss with **class imbalance handling**
- Formula: `BCE = -[y·log(σ(x)) + (1-y)·w·log(1-σ(x))]`
- `pos_weight` parameter increases penalty for missing positive pixels (ink)

**How it helps:**
- ✅ **Pixel-level accuracy**: Ensures each voxel is classified correctly
- ✅ **Handles class imbalance**: Ink pixels are rare (~5-20% of volume), so `pos_weight > 1` emphasizes them
- ✅ **Stable gradients**: Provides smooth, well-behaved gradients for optimization

**What to watch:**
- Should decrease over time (0.909 is moderate for early training)
- If very high (>1.5), model is struggling with basic classification
- If very low (<0.3), model may be overconfident or overfitting

**Why it's important:**
- Foundation loss - ensures model learns to distinguish ink from background
- Without this, model might ignore rare ink pixels entirely

---

## 3. **`soft_dice` (Soft Dice Loss) = 0.718**

**What it is:**
- Measures **overlap** between predicted and ground truth masks
- Formula: `Dice = 2·|A∩B| / (|A| + |B|)`, Loss = 1 - Dice
- Uses soft (probabilistic) predictions, not hard thresholds

**How it helps:**
- ✅ **Volume overlap**: Encourages model to predict correct overall shape
- ✅ **Handles class imbalance**: Less sensitive to background pixels than BCE
- ✅ **Complementary to BCE**: Focuses on shape/volume, not individual pixels

**What to watch:**
- Should decrease over time (0.718 means ~28% overlap - reasonable for early training)
- Target: <0.3 (70%+ overlap) for good segmentation
- If stays high, model is missing large portions of the surface

**Why it's important:**
- Dice is a standard medical imaging metric
- Ensures model captures the **extent** of the surface, not just individual pixels
- Works well with imbalanced data (rare ink pixels)

---

## 4. **`cldice` (Centerline Dice) = 0.687**

**What it is:**
- Measures overlap of **skeletonized/centerline** versions of prediction and target
- Uses soft morphological operations to extract "skeleton" (centerline) of surfaces
- Formula: Compares skeleton of prediction with target, and skeleton of target with prediction

**How it helps:**
- ✅ **Topology preservation**: Ensures predicted surface has correct **connectivity**
- ✅ **Thin structure detection**: Focuses on the "core" of surfaces, not just boundaries
- ✅ **Handles thickness variations**: Model can predict slightly thicker/thinner surfaces as long as centerline matches

**What to watch:**
- Should decrease over time (0.687 means ~31% centerline overlap)
- Target: <0.4 (60%+ centerline overlap) for good topology
- If much higher than `soft_dice`, model is predicting disconnected fragments

**Why it's important:**
- Papyrus scrolls are **thin surfaces** - the centerline is critical
- Prevents model from predicting disconnected blobs or holes
- Ensures **topological correctness** (surfaces should be connected)

---

## 5. **`morph_skeleton` (Morphological Skeleton Loss) = 0.048**

**What it is:**
- Direct **L1 distance** between skeletonized prediction and skeletonized target
- Uses soft morphological operations (erosion/dilation) to extract skeletons
- Formula: `L1(pred_skeleton, target_skeleton)`

**How it helps:**
- ✅ **Skeleton accuracy**: Ensures the "core" of predicted surface matches ground truth
- ✅ **Complementary to cldice**: Provides direct skeleton matching, not just overlap
- ✅ **Smooth gradients**: L1 loss provides stable optimization

**What to watch:**
- Should decrease over time (0.048 is already quite low - good sign!)
- Target: <0.05 for well-trained model
- If increases, model skeleton is diverging from target

**Why it's important:**
- Directly enforces skeleton matching (complementary to cldice)
- Lower value (0.048) suggests model is learning correct centerline structure
- Works well with thin surfaces like papyrus scrolls

---

## 6. **`surface_distance` (Surface Distance Loss) = 0.148**

**What it is:**
- Measures **distance** between predicted and target surface boundaries
- Uses `scipy.ndimage.distance_transform_edt` to compute distance maps
- Penalizes predictions that are far from true surface (within tolerance)

**How it helps:**
- ✅ **Boundary accuracy**: Ensures predicted surface is close to ground truth surface
- ✅ **Tolerance-based**: Only penalizes distances > tolerance (default: 2.0mm)
- ✅ **Handles small misalignments**: Allows slight position errors if within tolerance

**What to watch:**
- Should decrease over time (0.148 is moderate)
- Target: <0.1 for well-aligned surfaces
- **Note**: Computed every 16 steps (memory optimization), so value may be approximate

**Why it's important:**
- Surface detection requires **precise boundary localization**
- Prevents model from predicting surfaces that are "close but not quite right"
- Critical for accurate scroll reconstruction

**Memory Optimization:**
- This is the **most expensive** loss (uses scipy distance transforms)
- Computed every 16 steps to reduce memory pressure by 93.75%
- On skipped steps, uses lightweight boundary L1 approximation

---

## 7. **`toploss_simple` (Simple Topology Loss) = 0.416**

**What it is:**
- Measures **topological correctness** using Betti numbers and hole detection
- Approximates number of connected components and holes in prediction vs target
- Formula: Penalizes mismatches in connectivity and holes

**How it helps:**
- ✅ **Topology preservation**: Ensures prediction has correct number of connected components
- ✅ **Hole detection**: Prevents model from creating false holes or filling real ones
- ✅ **Connectivity**: Ensures surfaces are properly connected (not fragmented)

**What to watch:**
- Should decrease over time (0.416 is moderate for early training)
- Target: <0.2 for topologically correct surfaces
- If high, model is creating/filling holes or fragmenting surfaces

**Why it's important:**
- Papyrus scrolls should be **continuous surfaces** (no holes, no fragments)
- Prevents common segmentation errors: disconnected fragments, false holes, merged surfaces
- Ensures **topological correctness** beyond just pixel accuracy

---

## How They Work Together

### Loss Hierarchy & Weights

**From your config (`exp001_3d_unet_topology.yaml`):**

```
┌─────────────────────────────────────────┐
│  Total Loss (1.689)                    │
│  = Σ(weight × component)                │
└─────────────────────────────────────────┘
         │
         ├─→ weighted_bce (0.909) × 0.35 = 0.318  ← Pixel-level accuracy (35% weight)
         ├─→ soft_dice (0.718) × 0.35 = 0.251     ← Volume overlap (35% weight)
         ├─→ cldice (0.687) × 0.10 = 0.069        ← Centerline matching (10% weight)
         ├─→ morph_skeleton (0.048) × 0.08 = 0.004 ← Skeleton accuracy (8% weight)
         ├─→ surface_distance (0.148) × 0.07 = 0.010 ← Boundary precision (7% weight)
         └─→ toploss_simple (0.416) × 0.05 = 0.021 ← Topology preservation (5% weight)
```

**Weight Distribution:**
- **70%** to pixel/volume losses (BCE + Dice) - foundation
- **18%** to topology/connectivity losses (cldice + morph_skeleton) - structure
- **12%** to boundary/topology losses (surface_distance + toploss) - refinement

**Why these weights?**
- **BCE + Dice (70%)**: Primary training signal - model must learn basic segmentation first
- **Topology losses (18%)**: Important but secondary - refine structure after basics learned
- **Boundary/topology (12%)**: Fine-tuning - polish boundaries and fix topology errors

### Training Strategy

1. **Early Training (Epochs 1-10):**
   - `weighted_bce` and `soft_dice` dominate → Model learns basic shape
   - Other losses provide gentle guidance

2. **Mid Training (Epochs 11-20):**
   - `cldice` and `morph_skeleton` become important → Model refines centerline
   - `surface_distance` improves boundary accuracy

3. **Late Training (Epochs 21-32):**
   - `toploss_simple` becomes critical → Model fixes topology errors
   - All losses should be decreasing together

---

## Interpreting Your Current Metrics

### What's Good ✅
- **`morph_skeleton` = 0.048**: Very low! Model is learning correct skeleton structure
- **`surface_distance` = 0.148**: Moderate, should improve with more training
- **All losses are finite**: No NaN/inf issues

### What Needs Improvement ⚠️
- **`weighted_bce` = 0.909**: High - model still struggling with pixel classification
- **`soft_dice` = 0.718**: High - only ~28% overlap, needs improvement
- **`cldice` = 0.687**: High - centerline matching needs work
- **`toploss_simple` = 0.416**: Moderate - topology errors present

### Expected Trajectory 📈

As training progresses (32 epochs), you should see:

```
Epoch 1:  loss=1.689, bce=0.909, dice=0.718, cldice=0.687, ...
Epoch 10: loss=1.200, bce=0.600, dice=0.500, cldice=0.500, ...
Epoch 20: loss=0.900, bce=0.400, dice=0.350, cldice=0.400, ...
Epoch 32: loss=0.700, bce=0.300, dice=0.250, cldice=0.300, ...
```

**Target values for well-trained model:**
- `loss`: <0.8
- `weighted_bce`: <0.4
- `soft_dice`: <0.3
- `cldice`: <0.4
- `morph_skeleton`: <0.05 (already good!)
- `surface_distance`: <0.1
- `toploss_simple`: <0.2

---

## Why This Multi-Loss Approach?

### Single Loss Problems ❌
- **BCE alone**: Model might predict disconnected blobs
- **Dice alone**: Model might miss thin structures
- **Topology alone**: Model might have wrong shape

### Multi-Loss Benefits ✅
- **Comprehensive**: Covers pixel, boundary, and topology
- **Robust**: Multiple signals prevent overfitting to one metric
- **Domain-specific**: Tailored for thin surface detection (papyrus scrolls)

---

## Key Takeaways

1. **`loss`** = Total training signal (what optimizer minimizes)
2. **`weighted_bce`** = Pixel-level accuracy (foundation)
3. **`soft_dice`** = Volume overlap (shape matching)
4. **`cldice`** = Centerline matching (connectivity)
5. **`morph_skeleton`** = Skeleton accuracy (thin structure)
6. **`surface_distance`** = Boundary precision (localization)
7. **`toploss_simple`** = Topology preservation (holes, connectivity)

**All losses work together** to train a model that:
- ✅ Correctly classifies pixels (BCE)
- ✅ Matches overall shape (Dice)
- ✅ Preserves connectivity (cldice, morph_skeleton)
- ✅ Has precise boundaries (surface_distance)
- ✅ Maintains topology (toploss_simple)

---

## References

- **Dice Loss**: Standard medical imaging metric
- **clDice**: "clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation" (Shit et al., 2021)
- **Surface Distance**: Hausdorff distance variant for medical imaging
- **Topology Loss**: Betti number-based topology preservation


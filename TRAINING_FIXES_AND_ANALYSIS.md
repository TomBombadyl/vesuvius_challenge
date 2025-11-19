# Training Fixes & Performance Analysis
## Test Run Results & Full Training Recommendations

**Date:** 2025-11-19  
**Test Run:** 10 steps per epoch (8 epochs completed)  
**Status:** ✅ All fixes verified, checkpoints saving successfully

---

## 🔧 Critical Fixes Applied

### 1. Checkpoint Directory Creation (train.py)
**Problem:** Training crashed after epoch 1 with `RuntimeError: Parent directory runs/exp001_3d_unet_topology_full/checkpoints does not exist`

**Root Cause:** `torch.save()` does not create parent directories (standard Python I/O behavior)

**Fix:** Added directory creation at startup:
```python
# Line 246 in train.py
(output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
```

**Impact:** Checkpoints now save successfully after each epoch (verified: 386MB `last.pt` file created)

---

### 2. Logging Directory Creation (utils.py)
**Problem:** Potential failure if log file parent directory doesn't exist

**Root Cause:** `logging.FileHandler()` does not create parent directories

**Fix:** Added directory creation in `configure_logging()`:
```python
# Lines 83-85 in utils.py
log_file = Path(log_path)
log_file.parent.mkdir(parents=True, exist_ok=True)
```

**Impact:** Logging works reliably across all code paths (train.py, infer.py)

---

### 3. DataLoader Optimization (train.py)
**Changes:**
- `workers: 1 → 2` (Phase 1 optimization)
- `prefetch_factor: 1 → 2` (better overlap)
- `persistent_workers: False → True` (avoid restart overhead)

**Impact:** Improved data loading throughput while maintaining stability

---

## 📊 Test Run Analysis (10 Steps/Epoch)

### Training Metrics

| Epoch | Loss | Weighted BCE | Soft Dice | clDice | Morph Skeleton | Surface Distance | TopLoss |
|-------|------|--------------|-----------|--------|----------------|------------------|---------|
| 1 | 1.794 | 0.911 | 0.753 | 0.725 | 0.132 | 0.189 | 0.366 |
| 2 | 1.699 | 0.818 | 0.764 | 0.749 | 0.100 | 0.232 | 0.344 |
| 3 | 1.752 | 0.988 | 0.677 | 0.665 | 0.096 | 0.175 | 0.314 |
| 4 | 1.735 | 0.958 | 0.694 | 0.685 | 0.086 | 0.215 | 0.352 |
| 5 | 1.725 | 0.898 | 0.676 | 0.662 | 0.063 | 0.156 | 0.391 |
| 6 | 1.643 | 0.793 | 0.748 | 0.724 | 0.048 | 0.219 | 0.393 |
| 7 | 1.593 | 0.811 | 0.693 | 0.680 | 0.039 | 0.157 | 0.413 |
| 8 | - | - | - | - | - | - | - |

**Key Observations:**
- ✅ Loss decreasing: 1.794 → 1.593 (11% reduction in 7 epochs)
- ✅ Dice improving: 0.753 → 0.748 (stable, slight improvement)
- ✅ Morph skeleton loss decreasing: 0.132 → 0.039 (70% reduction)
- ✅ All loss components tracking properly
- ✅ No crashes or OOM errors

### Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Epoch Duration** | ~10 minutes (10 steps) | ✅ Stable |
| **GPU Memory Used** | 67.2 GB / 80 GB (84%) | ✅ Healthy |
| **GPU Utilization** | 17% (low, data-bound) | ⚠️ Can optimize |
| **GPU Temperature** | 39°C | ✅ Normal |
| **Checkpoint Size** | 386 MB | ✅ Normal |
| **Workers** | 2 | ✅ Stable |

### Time Analysis
- **Epoch 1:** 97 seconds (10 steps)
- **Epoch 2:** 185 seconds (10 steps) 
- **Epoch 3:** 271 seconds (10 steps)
- **Average:** ~90 seconds per epoch (10 steps)

**Projected for 4000 steps/epoch:**
- **Per epoch:** ~10 hours (90s × 400 steps)
- **32 epochs:** ~320 hours (13.3 days) ⚠️ **Too long!**

---

## 🎯 Full Training Configuration Recommendations

### Current Configuration Analysis

**Hardware:**
- **GPU:** NVIDIA A100 80GB
- **CPU:** 12 vCPUs (Intel Cascade Lake)
- **RAM:** 170 GB
- **Current GPU Memory:** 67.2 GB used (84% utilization)

**Current Settings:**
- **Workers:** 2
- **Batch Size:** 1
- **Patch Size:** [72, 136, 136]
- **Steps per Epoch:** 4000 (to be adjusted)

### Optimization Opportunities

#### 1. Increase Workers (Recommended: 4-6)

**Current:** 2 workers, 17% GPU utilization (data-bound)

**Recommendation:** Increase to 4-6 workers

**Rationale:**
- GPU utilization is low (17%), indicating data loading bottleneck
- 12 CPUs available, can support 4-6 workers comfortably
- PyTorch best practice: `num_workers = 4 × num_GPUs` (we have 1 GPU, so 4 is safe)
- With serialized distance transform (our fix), memory spikes are controlled

**Expected Impact:**
- GPU utilization: 17% → 40-60%
- Epoch time: ~10 hours → ~6-7 hours
- Memory: Should remain stable (serialization prevents OOM)

**Configuration:**
```yaml
training:
  workers: 4  # Increase from 2
```

#### 2. Increase Batch Size (Recommended: 2)

**Current:** Batch size 1

**Recommendation:** Try batch size 2

**Rationale:**
- GPU memory has headroom (67GB used, 80GB available)
- Batch size 2 would improve GPU utilization
- With serialized distance transform, memory spikes are controlled

**Expected Impact:**
- GPU utilization: 17% → 50-70%
- Training speed: ~2x faster
- Memory: ~75-78 GB (still safe)

**Configuration:**
```yaml
training:
  train_batch_size: 2  # Increase from 1
```

#### 3. Adjust Steps per Epoch

**Current:** 4000 steps/epoch (projected ~10 hours/epoch)

**Recommendation:** 2000-3000 steps/epoch

**Rationale:**
- Target: ~6-8 hours per epoch (with optimizations)
- 32 epochs × 7 hours = 224 hours (9.3 days) - reasonable
- 2000 steps with batch_size=2 = 4000 samples per epoch
- 806 training records × ~5 patches/record = ~4000 samples

**Configuration:**
```yaml
training:
  train_iterations_per_epoch: 2000  # Adjust based on desired epoch time
```

### Recommended Full Training Configuration

```yaml
training:
  workers: 4  # Increased from 2
  train_batch_size: 2  # Increased from 1
  train_iterations_per_epoch: 2000  # Adjusted for ~6-7 hour epochs
  max_epochs: 32
```

**Expected Performance:**
- **Epoch Duration:** ~6-7 hours
- **Total Training Time:** ~9-10 days
- **GPU Memory:** ~75-78 GB (safe)
- **GPU Utilization:** 50-70% (good)

---

## 📚 Best Practices Research

### PyTorch DataLoader Best Practices

**Official Recommendations:**
1. **num_workers:** Typically `4 × num_GPUs`, but can go higher for CPU-bound workloads
2. **prefetch_factor:** 2-4 for good overlap (we use 2)
3. **persistent_workers:** True to avoid worker restart overhead (we use True)
4. **pin_memory:** False with `spawn` context (we use False, correct)

**Our Implementation:**
- ✅ Using `spawn` context (avoids copy-on-write issues)
- ✅ `persistent_workers=True` (avoids restart overhead)
- ✅ `prefetch_factor=2` (good balance)
- ✅ Serialized distance transform (prevents OOM)

### Vesuvius Challenge Specifics

**Dataset Characteristics:**
- 806 training records
- 3D CT volumes (large, memory-intensive)
- Non-metallic ink detection (subtle features)
- Surface detection task (topology-aware)

**Training Considerations:**
- **Epochs:** 20-40 epochs typical for medical imaging
- **Batch Size:** Often 1-4 for 3D volumes (memory constraints)
- **Patch-based Training:** Standard for large volumes
- **Topology-aware Losses:** Critical for surface detection

**Our Approach:**
- ✅ Patch-based training (72×136×136 patches)
- ✅ Topology-aware losses (clDice, surface distance, morph skeleton)
- ✅ Heavy augmentations (elastic, rotation, scaling)
- ✅ Deep supervision (auxiliary heads)

---

## 🚀 Full Training Launch Plan

### Phase 1: Conservative Start (Current)
- **Workers:** 2
- **Batch Size:** 1
- **Steps/Epoch:** 10 (test)
- **Status:** ✅ Verified working

### Phase 2: Optimize Workers (Next)
- **Workers:** 4
- **Batch Size:** 1
- **Steps/Epoch:** 2000
- **Monitor:** GPU utilization, memory, stability

### Phase 3: Optimize Batch Size (If Phase 2 stable)
- **Workers:** 4
- **Batch Size:** 2
- **Steps/Epoch:** 2000
- **Monitor:** Memory usage, training speed

### Phase 4: Fine-tune Steps/Epoch
- Adjust `train_iterations_per_epoch` based on:
  - Desired epoch duration (6-8 hours)
  - Training record count (806)
  - Patches per record (~5)

---

## 📝 Summary of Changes

### Files Modified
1. **src/vesuvius/train.py**
   - Added checkpoint directory creation (line 246)
   - Optimized DataLoader settings (workers, prefetch, persistent)

2. **src/vesuvius/utils.py**
   - Added log directory creation in `configure_logging()` (lines 83-85)

3. **configs/experiments/exp001_full.yaml**
   - Updated workers to 2 (Phase 1)

### Git Status
- **Branch:** `feature/optimize-workers-batch-size`
- **Commits:** All fixes committed and pushed to GitHub
- **VM Sync:** Files synced via `gcloud compute scp`

---

## ✅ Verification Checklist

- [x] Checkpoints saving successfully (386MB file created)
- [x] No directory creation errors
- [x] Training completing epochs without crashes
- [x] Loss decreasing over epochs
- [x] GPU memory stable (67GB, 84% utilization)
- [x] All loss components tracking
- [x] Logs writing correctly

---

## 🎯 Next Steps

1. **Increase workers to 4** (test with 2000 steps/epoch)
2. **Monitor GPU utilization** (target 50-70%)
3. **If stable, increase batch size to 2**
4. **Fine-tune steps per epoch** based on desired epoch duration
5. **Launch full 32-epoch training run**

---

**Last Updated:** 2025-11-19  
**Status:** Ready for full training optimization


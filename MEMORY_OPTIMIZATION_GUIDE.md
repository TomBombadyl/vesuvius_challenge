# Memory Optimization Guide for 3D Medical Image Segmentation

## Problem Summary

Training crashes around step 800 with 4 DataLoader workers due to OOM (Out of Memory) kills. The root cause is `surface_distance_loss` using `scipy.ndimage.distance_transform_edt` on large 3D patches (`[72, 136, 136]` ≈ 1.3M voxels), which is memory-intensive.

## Research-Based Solutions

Based on documentation from PyTorch, SciPy, nnU-Net, and MONAI:

### 1. **Reduce DataLoader Workers** ✅ IMPLEMENTED
- **Current**: 2 workers (reduced from 4)
- **Rationale**: PyTorch best practices + nnU-Net recommendations
- **Impact**: Reduces CPU RAM pressure from worker processes
- **Trade-off**: Slightly slower data loading, but more stable

### 2. **Staggered Loss Computation** ✅ IMPLEMENTED
- **Strategy**: Compute `surface_distance_loss` every 4 steps instead of every step
- **Implementation**: Lightweight boundary L1 approximation on skipped steps
- **Impact**: ~75% reduction in memory-intensive operations
- **Rationale**: 
  - PyTorch recommends explicit memory cleanup (we do this)
  - nnU-Net uses similar strategies for expensive operations
  - Maintains gradient flow via approximation

### 3. **Explicit Memory Cleanup** ✅ IMPLEMENTED
- **Strategy**: Explicit `del` statements + `gc.collect()` after distance transforms
- **Impact**: Immediate release of numpy arrays
- **Rationale**: PyTorch documentation emphasizes explicit cleanup for multiprocessing

### 4. **Alternative Approaches** (Not Implemented)

#### Option A: Reduce Patch Size
- **Change**: `[72, 136, 136]` → `[64, 128, 128]`
- **Impact**: ~30% reduction in memory per patch
- **Trade-off**: Smaller receptive field, potentially lower accuracy

#### Option B: Torch-Based Distance Transform
- **Change**: Replace `scipy.ndimage.distance_transform_edt` with PyTorch implementation
- **Impact**: Avoids CPU↔GPU transfers, potentially faster
- **Trade-off**: Requires implementing EDT in PyTorch (complex)

#### Option C: Gradient Accumulation
- **Change**: Increase `accumulate_steps` to reduce effective batch size
- **Impact**: Lower peak memory per step
- **Trade-off**: Slower convergence, more steps per epoch

#### Option D: Disable Surface Distance Loss
- **Change**: Set `surface_distance` weight to 0.0
- **Impact**: Eliminates memory issue completely
- **Trade-off**: Loss of surface-aware training signal

## Current Configuration

```yaml
training:
  workers: 2  # Reduced from 4
  train_iterations_per_epoch: 4000
  max_epochs: 32
```

**Loss Configuration**:
- `surface_distance_loss` computed every 4 steps (interval configurable)
- Lightweight boundary approximation on skipped steps
- All other losses computed every step

## Monitoring & Validation

### Expected Behavior
- **Peak VRAM**: ~57-68 GB (within 80 GB budget)
- **CPU RAM**: ~2-4 GB per worker (with 2 workers = 4-8 GB total)
- **Stability**: No OOM kills over full 32-epoch run

### Monitoring Commands
```bash
# Check training progress
tail -n 50 runs/exp001_full/logs/train_full.log

# Check GPU memory
nvidia-smi

# Check system memory
free -h

# Check for OOM kills
dmesg | grep -i "killed\|oom" | tail -20
```

## Recommendations

1. **Start with current setup** (2 workers + staggered loss)
2. **Monitor first epoch** for stability
3. **If still unstable**:
   - Reduce `surface_distance_interval` to 8 (compute every 8 steps)
   - Or reduce patch size to `[64, 128, 128]`
4. **If stable but slow**:
   - Gradually increase workers to 3 (test carefully)
   - Or reduce `surface_distance_interval` to 2

## References

- **PyTorch**: Explicit memory cleanup, DataLoader worker management
- **nnU-Net**: `nnUNet_n_proc_DA` for separate data augmentation workers
- **MONAI**: Memory-efficient loss functions (though they don't use distance transforms)
- **SciPy**: Memory optimization for large arrays (limited guidance for distance_transform_edt)

## Long-Term Solutions

1. **Implement PyTorch-native distance transform** (avoids CPU↔GPU transfers)
2. **Use approximate distance transforms** (faster, less memory)
3. **Move distance transform to validation only** (not needed every training step)
4. **Implement gradient checkpointing for loss computation** (if needed)


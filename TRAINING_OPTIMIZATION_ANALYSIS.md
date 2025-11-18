# Training Performance Analysis & Optimization Plan

## Current Status

**Hardware:**
- GPU: NVIDIA A100 80GB
- CPU: 12 vCPUs
- RAM: 170GB
- Boot Disk: 200GB

**Current Configuration:**
- `batch_size: 1` ⚠️ **CRITICAL BOTTLENECK**
- `workers: 1` ⚠️ **UNDERUTILIZING 12 CPUs**
- `prefetch_factor: 1` ⚠️ **TOO CONSERVATIVE**
- `patch_size: [72, 136, 136]` (1.3M voxels per patch)
- `persistent_workers: False`

**Performance:**
- **Step time:** ~18 seconds per step
- **Epoch time:** ~20 hours (4000 steps)
- **Total training:** ~26.7 days (32 epochs)
- **GPU utilization:** 0% between batches (data loading bottleneck)

## Root Cause Analysis

### 1. Batch Size = 1 is the Primary Bottleneck

**Current:**
- Processing 1 patch at a time
- GPU waits ~15-17 seconds for next batch
- GPU utilization: 0% during data loading

**Impact:**
- With batch_size=1, we're using <5% of GPU compute capacity
- Each forward/backward pass takes ~1-2 seconds, but we wait 15+ seconds for data

### 2. Single Worker Underutilizes CPU

**Current:**
- 1 worker loading data sequentially
- 11 CPUs idle
- Data loading is the bottleneck (not GPU compute)

**Impact:**
- With 12 CPUs, we should use 4-6 workers (PyTorch best practice: `num_workers = num_CPUs / 2`)
- Each worker can load data in parallel, reducing wait time

### 3. Conservative Prefetching

**Current:**
- `prefetch_factor=1` means each worker only prefetches 1 batch ahead
- No overlap between data loading and GPU computation

**Impact:**
- GPU sits idle while waiting for next batch
- Should prefetch 2-4 batches per worker for better overlap

## Optimization Recommendations

### Phase 1: Increase Batch Size (Highest Impact)

**Goal:** Utilize more GPU memory and reduce data loading overhead per step

**Strategy:**
1. Start with `batch_size=2` (test if fits in VRAM)
2. If stable, try `batch_size=4`
3. Monitor VRAM usage (currently using 77GB/80GB, so we have ~3GB headroom)

**Expected Impact:**
- **2x speedup** with batch_size=2 (process 2 patches per step)
- **4x speedup** with batch_size=4 (process 4 patches per step)
- Reduces data loading overhead per sample

**VRAM Calculation:**
- Current: 77GB used with batch_size=1
- Patch size: 72×136×136 × 4 bytes (float32) = ~5.3MB per patch
- With batch_size=2: +5.3MB (negligible)
- With batch_size=4: +15.9MB (still within limits)

### Phase 2: Increase Workers (Medium Impact)

**Goal:** Parallelize data loading across multiple CPUs

**Strategy:**
1. Start with `workers=4` (1/3 of CPUs, conservative)
2. If stable, increase to `workers=6` (1/2 of CPUs, optimal)
3. Monitor CPU RAM usage (each worker uses ~7GB)

**Expected Impact:**
- **2-3x speedup** in data loading
- Reduces GPU idle time
- Better CPU utilization

**Memory Calculation:**
- Each worker: ~7GB RSS (from previous observations)
- 4 workers: ~28GB (well within 170GB RAM)
- 6 workers: ~42GB (still safe)

**PyTorch Best Practice:**
- Rule of thumb: `num_workers = num_CPUs / 2` = 6 workers
- With `spawn` context, each worker is independent (no copy-on-write issues)

### Phase 3: Optimize Prefetching (Low-Medium Impact)

**Goal:** Overlap data loading with GPU computation

**Strategy:**
1. Increase `prefetch_factor=2` (2 batches per worker ahead)
2. Consider `persistent_workers=True` if memory allows (avoids worker restart overhead)

**Expected Impact:**
- **10-20% speedup** from better overlap
- Reduces GPU idle time between batches

### Phase 4: Consider Gradient Accumulation (If Batch Size Limited)

**Goal:** Simulate larger effective batch size without increasing memory

**Strategy:**
- If batch_size=4 still doesn't fit, use `accumulate_steps=2` with batch_size=2
- Effective batch size = 2 × 2 = 4

## Recommended Configuration Changes

### Option A: Conservative (Safe, 2-3x speedup)

```yaml
training:
  train_batch_size: 2  # Double batch size
  workers: 4  # Use 1/3 of CPUs
  accumulate_steps: 1
```

**Expected:** ~10 hours per epoch (2x speedup)

### Option B: Moderate (Balanced, 4-6x speedup)

```yaml
training:
  train_batch_size: 4  # Quadruple batch size
  workers: 6  # Use 1/2 of CPUs (PyTorch best practice)
  accumulate_steps: 1
```

**Expected:** ~5-7 hours per epoch (4x speedup)

### Option C: Aggressive (Maximum, 6-8x speedup)

```yaml
training:
  train_batch_size: 4
  workers: 8  # Use 2/3 of CPUs
  accumulate_steps: 1
```

**Expected:** ~3-4 hours per epoch (6x speedup)

## Implementation Plan

### Step 1: Test Batch Size Increase (Low Risk)

1. Stop current training (or let it finish epoch 1)
2. Update `exp001_full.yaml`:
   ```yaml
   training:
     train_batch_size: 2  # Change from 1
     workers: 1  # Keep at 1 initially to isolate batch size effect
   ```
3. Run a short test (50-100 steps)
4. Monitor:
   - VRAM usage (should stay <80GB)
   - Step time (should decrease)
   - GPU utilization (should increase)

### Step 2: Test Worker Increase (Medium Risk)

1. If batch_size=2 works, update:
   ```yaml
   training:
     train_batch_size: 2
     workers: 4  # Increase from 1
   ```
2. Run a short test (50-100 steps)
3. Monitor:
   - CPU RAM usage (should stay <100GB)
   - Step time (should decrease further)
   - No OOM kills

### Step 3: Optimize Prefetching (Low Risk)

1. Update `src/vesuvius/train.py`:
   ```python
   prefetch_factor=2,  # Increase from 1
   persistent_workers=True,  # Enable if memory allows
   ```
2. Test and monitor memory

### Step 4: Full Optimization (If All Tests Pass)

1. Final config:
   ```yaml
   training:
     train_batch_size: 4
     workers: 6
     accumulate_steps: 1
   ```
2. Update DataLoader:
   ```python
   prefetch_factor=2,
   persistent_workers=True,
   ```

## Expected Results

**Current:** 20 hours/epoch → 26.7 days total

**After Optimization (Option B):**
- **5-7 hours/epoch** → **6.7-9.3 days total**
- **3-4x speedup** overall

**After Optimization (Option C):**
- **3-4 hours/epoch** → **4-5.3 days total**
- **5-6x speedup** overall

## Risk Assessment

**Low Risk:**
- Increasing batch_size to 2 (we have VRAM headroom)
- Increasing prefetch_factor to 2

**Medium Risk:**
- Increasing workers to 4-6 (need to monitor CPU RAM)
- Increasing batch_size to 4 (need to verify VRAM)

**High Risk:**
- Increasing workers to 8+ (may cause CPU RAM pressure)
- Changing multiple things at once (harder to debug)

## Monitoring Commands

```bash
# Check VRAM usage
nvidia-smi

# Check CPU RAM usage
free -h
ps aux | grep python | awk '{sum+=$6} END {print sum/1024/1024 " GB"}'

# Check step time
grep 'Train step' runs/exp001_full/logs/train_full.log | tail -n 5

# Check for OOM errors
grep -i 'killed\|oom\|out of memory' runs/exp001_full/logs/train_full.log
```

## Conclusion

**The training is slow because we're severely underutilizing the hardware:**
- Batch size of 1 wastes GPU compute
- Single worker wastes 11 CPUs
- Conservative prefetching wastes overlap opportunities

**With proper optimization, we can achieve 4-6x speedup, reducing training time from 26.7 days to 4-7 days.**

The key is to **test incrementally** and monitor memory usage at each step.


# Root Cause Analysis: Why 2 Workers Crashed

## The Problem

**Symptom:** Training crashes with 2 workers, but works with 1 worker

**Root Cause:** `surface_distance_loss` uses `scipy.ndimage.distance_transform_edt` which is **extremely memory-intensive** on large 3D patches.

## Memory Breakdown

### Per Distance Transform Operation
- **Input patch:** 72×136×136 = 1.33M voxels = 1.27 MB (bool)
- **Output distance map:** 72×136×136 = 1.33M voxels = 10.16 MB (float64)
- **Total per transform:** ~11.43 MB
- **Per batch item:** 2 transforms (target + pred) = ~22.86 MB

### With Multiple Workers
- **1 worker:** ~23 MB peak during surface_distance computation
- **2 workers (simultaneous):** ~46 MB peak
- **4 workers (simultaneous):** ~92 MB peak

### The Real Issue: Synchronized Computation

**Critical Problem:** All workers hit the expensive computation at the **same step** because `step_count` is shared across the model.

```python
# In CompositeTopologyLoss.forward()
self.step_count += 1  # This is per-model, not per-worker!
if self.step_count % self.surface_distance_interval != 0:
    # All workers skip together
else:
    # All workers compute together = MEMORY SPIKE
```

**What happens:**
1. Step 16: All workers compute `surface_distance_loss` simultaneously
2. Each worker: ~23 MB for distance transforms
3. 2 workers = 46 MB + base memory (7GB each) = **~14GB+ peak**
4. OOM killer terminates workers

## Current Mitigations (Not Enough)

### ✅ Already Implemented:
1. **Staggered computation:** Every 16 steps (93.75% reduction)
2. **Explicit cleanup:** `del` + `gc.collect()`
3. **VolumeCache limit:** max_size=50 (prevents unbounded growth)
4. **Spawn context:** Avoids copy-on-write issues

### ❌ Still Problematic:
- **Synchronized execution:** All workers compute at same step
- **Memory spike:** Even with staggering, simultaneous computation causes OOM

## Solutions (Ranked by Impact)

### Solution 1: Serialize Distance Transform Computation ⭐ BEST
**Strategy:** Use a lock to ensure only ONE worker computes distance transform at a time

**Implementation:**
```python
import threading

# Global lock for distance transform computation
_distance_transform_lock = threading.Lock()

def surface_distance_loss(logits, targets, tolerance_mm=2.0):
    probs = torch.sigmoid(logits)
    batch_losses = []
    for b in range(probs.shape[0]):
        pred = probs[b, 0]
        target = targets[b, 0]
        
        # SERIALIZE: Only one worker computes at a time
        with _distance_transform_lock:
            target_np = target.detach().cpu().numpy() > 0.5
            pred_np = pred.detach().cpu().numpy() > 0.5
            
            dt_target = ndimage.distance_transform_edt(~target_np)
            dt_pred = ndimage.distance_transform_edt(~pred_np)
            
            dt_target_t = torch.from_numpy(dt_target).to(pred.device, non_blocking=True).float()
            dt_pred_t = torch.from_numpy(dt_pred).to(pred.device, non_blocking=True).float()
            
            del target_np, pred_np, dt_target, dt_pred
            gc.collect()
        
        # Loss computation (outside lock, fast)
        loss_tp = torch.mean(torch.clamp(dt_target_t / tolerance_mm, max=1.0) * (1 - pred))
        loss_fp = torch.mean(torch.clamp(dt_pred_t / tolerance_mm, max=1.0) * target)
        batch_losses.append(loss_tp + loss_fp)
        
        del dt_target_t, dt_pred_t
    
    return torch.stack(batch_losses).mean()
```

**Impact:**
- ✅ Prevents simultaneous memory spikes
- ✅ Allows multiple workers safely
- ✅ Minimal performance impact (lock held briefly)
- ✅ Maintains all loss computation

**Risk:** Low - simple synchronization

---

### Solution 2: Increase Staggering Interval ⭐ EASY
**Strategy:** Compute surface_distance_loss less frequently (every 32 or 64 steps)

**Implementation:**
```python
# In CompositeTopologyLoss.__init__()
self.surface_distance_interval = 32  # or 64
```

**Impact:**
- ✅ Reduces frequency of memory spikes
- ✅ Still allows simultaneous computation (but less often)
- ✅ Easy to implement

**Risk:** Low - but may reduce loss signal quality

---

### Solution 3: Disable Surface Distance Loss in Training ⭐ NUCLEAR
**Strategy:** Only compute surface_distance_loss during validation

**Implementation:**
```python
# In CompositeTopologyLoss.forward()
if component.name == "surface_distance":
    if mode == "train":
        # Use approximation always in training
        value = boundary_approximation(...)
    else:
        # Full computation only in validation
        value = self._compute(component, logits, targets)
```

**Impact:**
- ✅ Eliminates memory issue completely
- ✅ Validation still uses full loss
- ❌ Loses training signal

**Risk:** Medium - may hurt model quality

---

### Solution 4: Use PyTorch-Native Distance Transform ⭐ LONG-TERM
**Strategy:** Replace scipy with PyTorch implementation (avoids CPU↔GPU transfers)

**Implementation:**
- Use `kornia` or implement EDT in PyTorch
- More complex, but avoids numpy/scipy overhead

**Impact:**
- ✅ Faster (GPU-native)
- ✅ Less memory (no CPU↔GPU transfers)
- ❌ Requires implementation work

**Risk:** Medium - requires significant development

---

### Solution 5: Reduce Patch Size for Distance Transform Only ⭐ COMPROMISE
**Strategy:** Downsample patches before distance transform

**Implementation:**
```python
# Downsample to 36×68×68 before distance transform
target_small = F.avg_pool3d(target.unsqueeze(0), kernel_size=2, stride=2)
pred_small = F.avg_pool3d(pred.unsqueeze(0), kernel_size=2, stride=2)
# Compute distance transform on smaller patch
# Upsample result back
```

**Impact:**
- ✅ 8x memory reduction (2³ = 8)
- ✅ Approximate but still useful
- ❌ Less precise

**Risk:** Low - but reduces precision

---

## Recommended Approach

### Phase 1: Serialize Computation (Solution 1) ⭐
**Why:** Safest, maintains all functionality, allows multiple workers

**Steps:**
1. Add threading lock to `surface_distance_loss`
2. Test with 2 workers
3. If stable, increase to 4-6 workers

### Phase 2: Increase Batch Size
**Why:** Once workers are stable, increase batch size for speedup

**Steps:**
1. With serialized distance transform, test batch_size=2
2. Then batch_size=4

### Phase 3: Optimize Further (Optional)
**Why:** Additional speedups if needed

**Options:**
- Increase staggering interval to 32
- Or implement PyTorch-native distance transform

## Expected Results

**With Serialized Distance Transform + 4 Workers:**
- ✅ No OOM crashes
- ✅ 4x data loading speedup
- ✅ Can safely increase batch_size

**With Batch Size 4 + 4 Workers:**
- ✅ 4x throughput (batch size)
- ✅ 4x data loading (workers)
- ✅ **16x overall speedup potential**

## Implementation Priority

1. **HIGH:** Serialize distance transform (Solution 1)
2. **HIGH:** Test with 2 workers first
3. **MEDIUM:** Increase to 4-6 workers if stable
4. **MEDIUM:** Increase batch size to 2-4
5. **LOW:** Further optimizations (staggering, PyTorch-native)

## Conclusion

**The crash wasn't just about workers - it was about synchronized memory spikes from distance transforms.**

**Fix:** Serialize the expensive computation so only one worker does it at a time.

**Result:** Can safely use 4-6 workers + larger batch sizes = **massive speedup**


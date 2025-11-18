# Safe Optimization Plan - Post Serialization Fix

## Current Status ✅

**Training Stability:**
- **Step 549** after ~3 hours
- **Memory:** 77GB/80GB (96% - stable)
- **GPU utilization:** 0% (waiting for data - bottleneck confirmed)
- **No crashes:** Training is stable with 1 worker

**Performance:**
- **Rate:** ~183 steps/hour (~3 steps/minute)
- **At this rate:** ~20 hours per epoch (4000 steps)
- **Total:** ~26.7 days for 32 epochs

## Root Cause Fixed ✅

**Serialization implemented:**
- Distance transform computation now serialized with threading lock
- Prevents simultaneous memory spikes across workers/batches
- Safe to test with multiple workers now

## Safe Testing Plan

### Phase 1: Test 2 Workers (Low Risk) ⭐ START HERE

**Why:** Minimal risk, validates serialization fix works

**Changes:**
```yaml
# configs/experiments/exp001_full.yaml
training:
  workers: 2  # Increase from 1
  train_batch_size: 1  # Keep at 1 initially
```

**Expected:**
- ✅ No OOM crashes (serialization prevents memory spikes)
- ✅ ~2x faster data loading
- ✅ Step time: ~9-10 seconds (vs ~18 seconds)
- ✅ Epoch time: ~10 hours (vs ~20 hours)

**Test Duration:** Run for 100-200 steps, monitor memory

**Success Criteria:**
- No OOM kills
- Memory stable <80GB
- Step time decreases

---

### Phase 2: Test Batch Size 2 (Low Risk)

**Why:** Small increase, validates VRAM headroom

**Changes:**
```yaml
training:
  workers: 2  # Keep from Phase 1
  train_batch_size: 2  # Increase from 1
```

**Expected:**
- ✅ VRAM: ~77.5GB (minimal increase)
- ✅ ~2x throughput (2 patches per step)
- ✅ Step time: ~5-6 seconds
- ✅ Epoch time: ~5-6 hours

**Test Duration:** Run for 100-200 steps

**Success Criteria:**
- VRAM stays <80GB
- No crashes
- Throughput increases

---

### Phase 3: Increase Workers to 4 (Medium Risk)

**Why:** Better CPU utilization, faster data loading

**Changes:**
```yaml
training:
  workers: 4  # Increase from 2
  train_batch_size: 2  # Keep from Phase 2
```

**Expected:**
- ✅ CPU RAM: ~28GB (4 workers × 7GB each)
- ✅ ~2x faster data loading (vs 2 workers)
- ✅ Step time: ~3-4 seconds
- ✅ Epoch time: ~3-4 hours

**Test Duration:** Run for 200-300 steps

**Success Criteria:**
- CPU RAM <100GB
- No OOM kills
- Stable training

---

### Phase 4: Increase Batch Size to 4 (Medium Risk)

**Why:** Maximum throughput, full GPU utilization

**Changes:**
```yaml
training:
  workers: 4  # Keep from Phase 3
  train_batch_size: 4  # Increase from 2
```

**Expected:**
- ✅ VRAM: ~78GB (still <80GB)
- ✅ ~2x throughput (vs batch_size=2)
- ✅ Step time: ~2-3 seconds
- ✅ Epoch time: ~2-3 hours
- ✅ **Total training: 2.7-4 days (vs 26.7 days)**

**Test Duration:** Run for 300-500 steps

**Success Criteria:**
- VRAM <80GB
- Stable training
- GPU utilization >50%

---

### Phase 5: Optimize Further (Optional)

**If Phase 4 is stable, consider:**

1. **Increase workers to 6:**
   ```yaml
   workers: 6  # Use 50% of CPUs (PyTorch best practice)
   ```

2. **Enable persistent workers:**
   ```python
   persistent_workers=True,  # Avoid worker restart overhead
   prefetch_factor=2,  # Better overlap
   ```

3. **Reduce surface_distance interval:**
   ```python
   # In losses.py, if memory allows
   self.surface_distance_interval = 8  # More frequent computation
   ```

## Expected Final Performance

**Target Configuration:**
- `workers: 4-6`
- `batch_size: 4`
- Serialized distance transforms

**Expected Results:**
- **Step time:** ~2-3 seconds (vs ~18 seconds)
- **Epoch time:** ~2-3 hours (vs ~20 hours)
- **Total training:** ~2.7-4 days (vs 26.7 days)
- **Speedup:** **6-10x overall**

## Monitoring Commands

```bash
# Check current step
gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command "cd /mnt/disks/data/repos/vesuvius_challenge && grep 'Train step' runs/exp001_full/logs/train_full.log | tail -n 1"

# Check memory
gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command "nvidia-smi && echo '' && free -h"

# Check for errors
gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command "cd /mnt/disks/data/repos/vesuvius_challenge && grep -i 'error\|killed\|oom\|exception' runs/exp001_full/logs/train_full.log | tail -n 10"

# Check CPU RAM usage
gcloud compute ssh dylant@vesuvius-challenge --project vesuvius-challenge-478512 --zone us-central1-a --command "ps aux | grep 'pt_data_worker' | awk '{sum+=\$6} END {print sum/1024/1024 \" GB\"}'"
```

## Risk Assessment

**Low Risk:**
- ✅ Phase 1: 2 workers (serialization protects us)
- ✅ Phase 2: batch_size=2 (minimal VRAM increase)

**Medium Risk:**
- ⚠️ Phase 3: 4 workers (need to monitor CPU RAM)
- ⚠️ Phase 4: batch_size=4 (need to monitor VRAM)

**Mitigation:**
- Test incrementally
- Monitor at each phase
- Have rollback plan (current config works)

## Next Steps

1. **Wait for current training to finish epoch 1** (or stop it if you want to test now)
2. **Apply Phase 1 changes** (workers=2)
3. **Test for 100-200 steps**
4. **If stable, proceed to Phase 2**
5. **Continue incrementally**

## Summary

**Current:** 1 worker, batch_size=1 → 26.7 days
**Target:** 4 workers, batch_size=4 → 2.7-4 days
**Speedup:** 6-10x

**Key:** Serialization fix makes this safe to test. Start with Phase 1 and proceed incrementally.


# Steps Per Epoch: Best Practices Guide

## Understanding the Fundamentals

### What is an Epoch?
An **epoch** is one complete pass through your training dataset. The number of **steps per epoch** determines how many batches the model processes during that pass.

### Basic Formula

```
Steps per Epoch = Total Training Samples / Batch Size
```

**Example:**
- 1,000 training samples
- Batch size of 32
- Steps per epoch = 1,000 / 32 = 31.25 → **31 steps** (round down if drop_last=True)

---

## Two Main Approaches

### 1. **Full Dataset Approach** (Traditional)

**Principle:** Process the entire dataset once per epoch.

**Calculation:**
```python
steps_per_epoch = len(dataset) // batch_size
```

**Pros:**
- ✅ Model sees all data each epoch
- ✅ Consistent learning signal
- ✅ Standard practice for most tasks
- ✅ Easy to track progress (epoch = full dataset)

**Cons:**
- ❌ Can be slow for large datasets
- ❌ Fixed time per epoch (hard to adjust)

**When to Use:**
- Small to medium datasets (< 100K samples)
- When you want standard epoch semantics
- When dataset size is fixed and known

---

### 2. **Fixed Steps Approach** (Time/Resource Constrained)

**Principle:** Set a specific number of steps per epoch regardless of dataset size.

**Calculation:**
```python
steps_per_epoch = desired_steps  # e.g., 2000
```

**Pros:**
- ✅ Control training time per epoch
- ✅ Consistent epoch duration
- ✅ Easy to plan training schedule
- ✅ Works well with patch-based/iterative datasets

**Cons:**
- ❌ May not see all data each epoch
- ❌ Need to ensure enough steps for convergence
- ❌ Less intuitive (epoch ≠ full dataset)

**When to Use:**
- Large datasets or infinite/iterative datasets
- **Patch-based training** (like yours!)
- Time-constrained training (e.g., Kaggle 9-hour limit)
- When you want consistent epoch duration

---

## Your Specific Case: Patch-Based Training

### Your Current Setup

**Dataset Characteristics:**
- **806 training records** (3D CT volumes)
- **Patch-based sampling** (random patches from volumes)
- **Iterative dataset** (generates patches on-the-fly)

**Current Code Logic** (from `train.py` lines 69-72):
```python
train_iters = cfg["training"].get("train_iterations_per_epoch")
if train_iters is None:
    patches_per_volume = cfg["training"].get("patches_per_volume", 16)
    train_iters = max(1, len(train_records) * patches_per_volume)
```

**Default Calculation:**
- 806 records × 16 patches/record = **12,896 steps/epoch**
- With batch_size=1: **12,896 steps**
- With batch_size=2: **6,448 steps**

### Why Fixed Steps Makes Sense for You

1. **Infinite Dataset:** Your `PatchDataset` samples patches randomly, so there's no fixed "dataset size"
2. **Time Control:** You can set epoch duration (e.g., 6-8 hours)
3. **Flexibility:** Easy to adjust without recalculating dataset size
4. **Standard Practice:** Patch-based training often uses fixed steps

---

## Best Practices for Choosing Steps Per Epoch

### 1. **Consider Your Dataset Type**

| Dataset Type | Recommended Approach | Reasoning |
|--------------|---------------------|-----------|
| **Fixed-size dataset** (images, tabular) | Full dataset | Standard, intuitive |
| **Patch-based** (3D volumes, large images) | Fixed steps | Infinite samples, time control |
| **Streaming/iterative** | Fixed steps | No fixed size |
| **Very large dataset** (>1M samples) | Fixed steps | Time constraints |

**Your case:** ✅ Patch-based → **Fixed steps recommended**

---

### 2. **Consider Training Time Constraints**

**Target Epoch Duration:**
- **Short epochs (1-2 hours):** Good for debugging, quick iterations
- **Medium epochs (4-8 hours):** Good balance, allows monitoring
- **Long epochs (12+ hours):** Risk of losing progress if crash

**Your Constraints:**
- VM can run continuously (no 9-hour limit)
- Want to monitor progress regularly
- **Recommendation:** 6-8 hour epochs

**Calculation:**
```
Steps per Epoch = (Desired Epoch Time in seconds) / (Time per Step in seconds)
```

**Example:**
- Target: 7 hours = 25,200 seconds
- Time per step: ~12 seconds (with optimizations)
- Steps = 25,200 / 12 = **2,100 steps**

---

### 3. **Consider Dataset Coverage**

**Question:** How many unique samples should the model see per epoch?

**For Patch-Based Training:**
```
Samples per Epoch = Steps per Epoch × Batch Size
```

**Your Case:**
- 806 training records
- ~5-10 patches per record (varies by volume size)
- Total unique patches: ~4,000-8,000

**Recommendations:**
- **Minimum:** 2,000-3,000 samples/epoch (50-75% coverage)
- **Good:** 4,000-6,000 samples/epoch (full coverage)
- **Excessive:** >10,000 samples/epoch (diminishing returns)

**With batch_size=2:**
- 2,000 steps = 4,000 samples ✅ Good coverage
- 3,000 steps = 6,000 samples ✅ Full coverage

---

### 4. **Consider Learning Rate Schedule**

**If using epoch-based LR schedule:**
- More steps/epoch = slower LR decay
- Fewer steps/epoch = faster LR decay

**If using step-based LR schedule:**
- Steps per epoch doesn't affect LR directly
- Total steps = steps_per_epoch × num_epochs

**Your Case:**
- Check your scheduler type in config
- If epoch-based: More steps = slower decay (may need more epochs)
- If step-based: Steps per epoch is less critical

---

### 5. **Consider Validation Frequency**

**Question:** How often do you want to validate?

**Options:**
- Every epoch (standard)
- Every N epochs (e.g., every 5 epochs)
- Every N steps (e.g., every 10,000 steps)

**Your Case:**
- Currently: Validation every epoch (if val_records exist)
- With 2,000 steps/epoch: Validates every ~7 hours
- **Recommendation:** Keep every epoch (good balance)

---

## Practical Recommendations for Your Project

### Option 1: Conservative (Recommended for First Full Run)

```yaml
training:
  train_batch_size: 2
  train_iterations_per_epoch: 2000  # 4,000 samples/epoch
  max_epochs: 32
```

**Rationale:**
- 2,000 steps × 2 batch_size = 4,000 samples/epoch
- Covers ~100% of unique patches (806 records × ~5 patches)
- ~6-7 hours per epoch (with 4 workers)
- 32 epochs = ~9-10 days total

---

### Option 2: Aggressive (Faster Training)

```yaml
training:
  train_batch_size: 2
  train_iterations_per_epoch: 3000  # 6,000 samples/epoch
  max_epochs: 32
```

**Rationale:**
- 3,000 steps × 2 batch_size = 6,000 samples/epoch
- More samples per epoch (better coverage)
- ~9-10 hours per epoch
- 32 epochs = ~12-13 days total

---

### Option 3: Time-Constrained (Faster Iterations)

```yaml
training:
  train_batch_size: 2
  train_iterations_per_epoch: 1500  # 3,000 samples/epoch
  max_epochs: 40  # More epochs to compensate
```

**Rationale:**
- 1,500 steps × 2 batch_size = 3,000 samples/epoch
- ~4-5 hours per epoch (faster iterations)
- 40 epochs = ~8-9 days total
- More frequent checkpoints

---

## Decision Framework

### Step 1: Determine Your Constraints
- [ ] Maximum epoch duration? (e.g., 8 hours)
- [ ] Total training time budget? (e.g., 10 days)
- [ ] Desired dataset coverage per epoch? (e.g., 100%)

### Step 2: Calculate Time Per Step
Run a test with your configuration:
```python
# Measure time for 100 steps
start = time.time()
for i, batch in enumerate(train_loader):
    if i >= 100:
        break
time_per_step = (time.time() - start) / 100
```

### Step 3: Calculate Steps Per Epoch
```
steps_per_epoch = (desired_epoch_time_seconds) / time_per_step
```

### Step 4: Verify Dataset Coverage
```
samples_per_epoch = steps_per_epoch × batch_size
coverage = samples_per_epoch / estimated_unique_patches
```

**Target:** 50-100% coverage per epoch

### Step 5: Adjust if Needed
- Too many steps? → Reduce steps or increase batch size
- Too few steps? → Increase steps or decrease batch size
- Not enough coverage? → Increase steps
- Too long per epoch? → Reduce steps

---

## Common Pitfalls to Avoid

### ❌ Too Few Steps
- **Problem:** Model doesn't see enough data per epoch
- **Symptom:** Slow convergence, poor performance
- **Fix:** Increase steps or batch size

### ❌ Too Many Steps
- **Problem:** Very long epochs, slow iteration
- **Symptom:** Hard to monitor, risk of losing progress
- **Fix:** Reduce steps or increase batch size

### ❌ Ignoring Batch Size
- **Problem:** Steps calculated without considering batch size
- **Symptom:** Inconsistent sample counts
- **Fix:** Always consider: `samples = steps × batch_size`

### ❌ Not Testing First
- **Problem:** Set steps without measuring actual time
- **Symptom:** Epochs take much longer than expected
- **Fix:** Always test with small number first (e.g., 10-50 steps)

---

## Your Code's Current Behavior

### Default Calculation (if not specified):
```python
# From train.py lines 69-72
if train_iters is None:
    patches_per_volume = cfg["training"].get("patches_per_volume", 16)
    train_iters = max(1, len(train_records) * patches_per_volume)
```

**Current Default:**
- 806 records × 16 patches = **12,896 steps**
- With batch_size=1: **12,896 steps**
- **Problem:** This is very long! (~20+ hours per epoch)

### Recommended Override:
```yaml
training:
  train_iterations_per_epoch: 2000  # Override default
```

**Why:**
- More reasonable epoch duration (~6-7 hours)
- Still good dataset coverage (4,000 samples with batch_size=2)
- Easier to monitor and manage

---

## Summary: Quick Decision Guide

**For Your Vesuvius Challenge Project:**

1. **Start with:** 2,000 steps/epoch (batch_size=2)
   - 4,000 samples/epoch
   - ~6-7 hours per epoch
   - Good coverage

2. **Monitor:**
   - Epoch duration (target: 6-8 hours)
   - GPU utilization (target: 50-70%)
   - Loss convergence

3. **Adjust if:**
   - Epochs too long → Reduce to 1,500 steps
   - Epochs too short → Increase to 2,500-3,000 steps
   - Poor convergence → Increase steps or batch size

4. **Remember:**
   - Steps × Batch Size = Samples per Epoch
   - More steps = longer epochs but better coverage
   - Balance time constraints with dataset coverage

---

**Key Takeaway:** For patch-based training like yours, **fixed steps per epoch** is the right approach. Choose a number that balances:
- ✅ Reasonable epoch duration (6-8 hours)
- ✅ Good dataset coverage (50-100% of unique patches)
- ✅ Total training time budget (9-10 days for 32 epochs)

